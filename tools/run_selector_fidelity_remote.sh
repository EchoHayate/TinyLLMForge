#!/usr/bin/env bash
# Run the selector fidelity gate on the GPU box.
#
# The measurement needs real post-RoPE Q/K from a real model, so the dump has to happen
# where the weights are. The Q/K tensors are large (~100 MB per variant) and useless
# once reduced, so the fidelity pass runs remotely too and only JSON comes back.
#
# Provenance is recorded on purpose: the last time we trusted a benchmark without
# recording who else was on the machine, the numbers turned out to be a lower bound
# rather than a measurement.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
REMOTE_ROOT="/data00/home/sitian/tllm/selector-fidelity-runs/"
TARGET_MODEL="${TARGET_MODEL:-/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-selector-fidelity-$(date +%Y%m%d-%H%M%S)}"
SEQ_LEN="${SEQ_LEN:-8192}"
LAYERS="${LAYERS:-auto6}"
VARIANTS="${VARIANTS:-repetitive natural distractor}"
GRANULARITIES="${GRANULARITIES:-256 64 32}"
K_FRACS="${K_FRACS:-0.02 0.05 0.11 0.25 0.50}"
GPU="${GPU:-2}"
# The host's user-site has a flash_attn built against a newer libstdc++ than the system
# one, and merely importing transformers.models.qwen3 pulls it in and dies. The working
# convention on this box (see tools/run_kv8_quest_quality_remote.sh) is to shadow the
# user-site with a symlink farm that omits the broken packages, and to point
# LD_LIBRARY_PATH at the miniforge libs.
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi

LOCAL_OUT="experiments/selector_fidelity/${RUN_TAG}"
mkdir -p "${LOCAL_OUT}"
# One multiplexed connection: the jump proxy intermittently refuses new sessions,
# and a mid-run refusal under `set -e` would abandon a half-finished remote dir.
CTRL_PATH="/tmp/ssh-selfid-$(echo "${REMOTE_HOST}" | tr -c "A-Za-z0-9" _)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto
     -o "ControlPath=${CTRL_PATH}" -o ControlPersist=900 "${REMOTE_HOST}")

if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 3
fi

echo ">>> staging ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source/tools' '${REMOTE_DIR}/dump'"
tar -C . -cf - \
  tools/dump_needle_qk.py \
  tools/needle_haystack_variants.py \
  tools/kv_selector_fidelity.py \
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
  echo "layers=${LAYERS}"
  echo "variants=${VARIANTS}"
  echo "granularities=${GRANULARITIES}"
  echo "k_fracs=${K_FRACS}"
  echo "source_revision=${GIT_REV}"
  echo "source_dirty_files=${GIT_DIRTY}"
  echo "gpu=${GPU}"
} > "${LOCAL_OUT}/provenance.txt"

"${SSH[@]}" "uptime; echo ---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" \
  > "${LOCAL_OUT}/contamination.txt" 2>&1 || true
echo ">>> host state:"; head -1 "${LOCAL_OUT}/contamination.txt"

echo ">>> dumping post-RoPE Q/K for: ${VARIANTS}"
"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} ${REMOTE_ENV} \
  '${REMOTE_PYTHON}' tools/dump_needle_qk.py --model '${TARGET_MODEL}' --seq-len ${SEQ_LEN} \
  --layers ${LAYERS} --variants ${VARIANTS} --out-dir '${REMOTE_DIR}/dump' 2>&1 | tee '${REMOTE_DIR}/dump.log'"

for variant in ${VARIANTS}; do
  echo ">>> fidelity: ${variant}"
  "${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && ${REMOTE_ENV} \
    '${REMOTE_PYTHON}' tools/kv_selector_fidelity.py \
    --dump '${REMOTE_DIR}/dump/qk-${variant}.npz' --meta '${REMOTE_DIR}/dump/qk-${variant}.meta.json' \
    --granularities ${GRANULARITIES} --k-fracs ${K_FRACS} \
    --out-json '${REMOTE_DIR}/fidelity-${variant}.json' 2>&1 | tee '${REMOTE_DIR}/fidelity-${variant}.txt'"
done

echo ">>> pulling results"
"${SSH[@]}" "cat '${REMOTE_DIR}/dump.log'" > "${LOCAL_OUT}/dump.log"
for variant in ${VARIANTS}; do
  "${SSH[@]}" "cat '${REMOTE_DIR}/fidelity-${variant}.json'" > "${LOCAL_OUT}/fidelity-${variant}.json"
  "${SSH[@]}" "cat '${REMOTE_DIR}/fidelity-${variant}.txt'" > "${LOCAL_OUT}/fidelity-${variant}.txt"
  "${SSH[@]}" "cat '${REMOTE_DIR}/dump/qk-${variant}.meta.json'" > "${LOCAL_OUT}/qk-${variant}.meta.json"
done
"${SSH[@]}" "cd '${REMOTE_DIR}/dump' && sha256sum *.npz *.meta.json" \
  > "${LOCAL_OUT}/dump_sha256.txt" 2>&1 || true

echo ">>> done: ${LOCAL_OUT}"
