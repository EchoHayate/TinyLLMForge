#!/usr/bin/env bash
# Run the CPU-offload correctness harness on the GPU box.
#
# This is the first run in this line of work where a token is produced by the offload path
# rather than by a stand-in: KV in CPU DRAM, selection on the GPU from incrementally
# maintained summaries, attention arithmetic on the CPU. The gates before this one measured
# how fast the mechanism could be; this one measures whether it is the same mechanism.
#
# Same env convention as tools/run_e2e_sparse_remote.sh: the host user-site has a flash_attn
# built against a newer libstdc++ and importing transformers.models.qwen3 pulls it in and
# dies, so the user-site is shadowed by a symlink farm that omits the broken packages.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
REMOTE_ROOT="/data00/home/sitian/tllm/cpu-offload-correctness/"
TARGET_MODEL="${TARGET_MODEL:-/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-cpuoffload-$(date +%Y%m%d-%H%M%S)}"
SEQ_LEN="${SEQ_LEN:-8192}"
VARIANT="${VARIANT:-distractor}"
GRANULARITY="${GRANULARITY:-32}"
K_FRAC="${K_FRAC:-0.051}"
DENSE_LAYERS="${DENSE_LAYERS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
VERIFY_EVERY="${VERIFY_EVERY:-1}"
# RANDOM_MODEL=1 builds a Qwen3 with Qwen3-8B's per-layer shape and meaningless weights. Use it
# when the checkpoint is not on the box: every check in this harness compares two paths through
# the same weights, so the weights' meaning is irrelevant here (fidelity is a separate gate).
RANDOM_MODEL="${RANDOM_MODEL:-0}"
RANDOM_LAYERS="${RANDOM_LAYERS:-36}"
GPU="${GPU:-2}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi

LOCAL_OUT="experiments/cpu_offload_correctness/${RUN_TAG}"
mkdir -p "${LOCAL_OUT}"
CTRL_PATH="/tmp/ssh-cpuoffload-$(echo "${REMOTE_HOST}" | tr -c "A-Za-z0-9" _)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto
     -o "ControlPath=${CTRL_PATH}" -o ControlPersist=1800 "${REMOTE_HOST}")

if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 3
fi

echo ">>> staging ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source/tools'"
# --no-xattrs: macOS extended attributes make tar exit 1 under pipefail, which used to leave
# an empty output directory and a confusing "the run produced nothing" message
tar --no-xattrs -C . -cf - \
  tools/cpu_offload_correctness.py \
  tools/e2e_sparse_attention.py \
  tools/needle_haystack_variants.py \
  | "${SSH[@]}" "tar -C '${REMOTE_DIR}/source' -xf -"

SITEPATCH="${REMOTE_DIR}/sitepatch"
"${SSH[@]}" \
  "REMOTE_USER_SITE='${REMOTE_USER_SITE}' SITEPATCH='${SITEPATCH}' REMOTE_SITE_EXCLUDE='${REMOTE_SITE_EXCLUDE}' bash -s" \
  <<'REMOTE_SITEPATCH'
set -euo pipefail
mkdir -p "${SITEPATCH}"
for entry in "${REMOTE_USER_SITE}"/*; do
  base="$(basename "${entry}")"
  drop=0
  for prefix in ${REMOTE_SITE_EXCLUDE}; do
    case "${base}" in
      "${prefix}"*) drop=1 ;;
    esac
  done
  if [[ "${drop}" == 0 ]]; then
    ln -s "${entry}" "${SITEPATCH}/${base}"
  fi
done
REMOTE_SITEPATCH

REMOTE_ENV="PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false"
REMOTE_ENV+=" PYTHONPATH='${SITEPATCH}:${REMOTE_DIR}/source'"
REMOTE_ENV+=" LD_LIBRARY_PATH='${REMOTE_LD_LIBRARY_PATH}'"

GIT_REV="$(git rev-parse HEAD)"
GIT_DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
{
  echo "run_tag=${RUN_TAG}"
  echo "model=${TARGET_MODEL}"
  echo "random_model=${RANDOM_MODEL}"
  echo "random_layers=${RANDOM_LAYERS}"
  echo "seq_len=${SEQ_LEN}"
  echo "variant=${VARIANT}"
  echo "granularity=${GRANULARITY}"
  echo "k_frac=${K_FRAC}"
  echo "dense_layers=${DENSE_LAYERS}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "verify_every=${VERIFY_EVERY}"
  echo "source_revision=${GIT_REV}"
  echo "source_dirty_files=${GIT_DIRTY}"
  echo "gpu=${GPU}"
} > "${LOCAL_OUT}/provenance.txt"

"${SSH[@]}" "uptime; who | wc -l; echo ---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" \
  > "${LOCAL_OUT}/contamination.txt" 2>&1 || true
echo ">>> host state:"; head -1 "${LOCAL_OUT}/contamination.txt"

echo ">>> cpu offload correctness: gran=${GRANULARITY} k_frac=${K_FRAC}"
set +e
if [[ "${RANDOM_MODEL}" == "1" ]]; then
  MODEL_ARGS="--random-model --random-layers ${RANDOM_LAYERS}"
else
  MODEL_ARGS="--model '${TARGET_MODEL}'"
fi

"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} ${REMOTE_ENV} \
  '${REMOTE_PYTHON}' tools/cpu_offload_correctness.py ${MODEL_ARGS} \
  --seq-len ${SEQ_LEN} --variant ${VARIANT} --granularity ${GRANULARITY} \
  --k-frac ${K_FRAC} --dense-layers '${DENSE_LAYERS}' \
  --max-new-tokens ${MAX_NEW_TOKENS} --verify-every ${VERIFY_EVERY} \
  --out-json '${REMOTE_DIR}/correctness.json' 2>&1 | tee '${REMOTE_DIR}/correctness.txt'"
RC=$?
set -e
# the harness exits non-zero when a correctness check fails, and that output is the point
"${SSH[@]}" "cat '${REMOTE_DIR}/correctness.txt'" > "${LOCAL_OUT}/correctness.txt" || true
"${SSH[@]}" "cat '${REMOTE_DIR}/correctness.json'" > "${LOCAL_OUT}/correctness.json" 2>/dev/null || true

echo ">>> done: ${LOCAL_OUT} (harness rc=${RC})"
exit "${RC}"
