#!/usr/bin/env bash
# Run the end-to-end sparse decode gate on the GPU box.
#
# Unlike the selector fidelity run, nothing large has to come back: the whole output is
# one JSON per variant. The model still has to be where the weights are, so the run is
# remote and only the verdict travels.
#
# Same env convention as tools/run_selector_fidelity_remote.sh: the host's user-site has
# a flash_attn built against a newer libstdc++, and merely importing
# transformers.models.qwen3 pulls it in and dies, so the user-site is shadowed by a
# symlink farm that omits the broken packages.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
REMOTE_ROOT="/data00/home/sitian/tllm/e2e-sparse-runs/"
TARGET_MODEL="${TARGET_MODEL:-/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-e2e-sparse-$(date +%Y%m%d-%H%M%S)}"
SEQ_LEN="${SEQ_LEN:-8192}"
VARIANTS="${VARIANTS:-natural distractor}"
GRANULARITIES="${GRANULARITIES:-32 256}"
K_FRACS="${K_FRACS:-0.02 0.051 0.11 0.25}"
ARMS="${ARMS:-quest_shared_heads quest_per_head recency sink_recency random}"
DENSE_LAYERS="${DENSE_LAYERS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
GPU="${GPU:-2}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi

LOCAL_OUT="experiments/e2e_sparse_attention/${RUN_TAG}"
mkdir -p "${LOCAL_OUT}"
CTRL_PATH="/tmp/ssh-e2esparse-$(echo "${REMOTE_HOST}" | tr -c "A-Za-z0-9" _)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto
     -o "ControlPath=${CTRL_PATH}" -o ControlPersist=1800 "${REMOTE_HOST}")

if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 3
fi

echo ">>> staging ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source/tools'"
tar -C . -cf - \
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
  echo "seq_len=${SEQ_LEN}"
  echo "variants=${VARIANTS}"
  echo "granularities=${GRANULARITIES}"
  echo "k_fracs=${K_FRACS}"
  echo "arms=${ARMS}"
  echo "dense_layers=${DENSE_LAYERS}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "source_revision=${GIT_REV}"
  echo "source_dirty_files=${GIT_DIRTY}"
  echo "gpu=${GPU}"
} > "${LOCAL_OUT}/provenance.txt"

"${SSH[@]}" "uptime; echo ---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" \
  > "${LOCAL_OUT}/contamination.txt" 2>&1 || true
echo ">>> host state:"; head -1 "${LOCAL_OUT}/contamination.txt"

for variant in ${VARIANTS}; do
  echo ">>> e2e sparse decode: ${variant}"
  "${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} ${REMOTE_ENV} \
    '${REMOTE_PYTHON}' tools/e2e_sparse_attention.py --model '${TARGET_MODEL}' \
    --seq-len ${SEQ_LEN} --variant ${variant} \
    --granularities ${GRANULARITIES} --k-fracs ${K_FRACS} --arms ${ARMS} \
    --dense-layers '${DENSE_LAYERS}' --max-new-tokens ${MAX_NEW_TOKENS} \
    --out-json '${REMOTE_DIR}/e2e-${variant}.json' 2>&1 | tee '${REMOTE_DIR}/e2e-${variant}.txt'"
  "${SSH[@]}" "cat '${REMOTE_DIR}/e2e-${variant}.json'" > "${LOCAL_OUT}/e2e-${variant}.json"
  "${SSH[@]}" "cat '${REMOTE_DIR}/e2e-${variant}.txt'" > "${LOCAL_OUT}/e2e-${variant}.txt"
done

echo ">>> done: ${LOCAL_OUT}"
