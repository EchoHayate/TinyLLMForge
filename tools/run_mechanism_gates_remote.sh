#!/usr/bin/env bash
# Run the three remaining gates on the GPU box in one pass:
#   1. async handshake  - how much of the 1.355 ms per-layer coordination is recoverable
#   2. selector cost    - what Quest scoring itself costs at gran=32, on CPU and on GPU
#   3. pipeline         - whether cross-microbatch overlap turns headroom into a real saving,
#                         plus how many concurrent sequences the CPU can actually serve
#
# The pipeline arm links the CPU kernel as a shared library built from the same translation
# unit as the standalone benchmark, so the scheduling result is measured against the same
# 2.051 ms/step kernel and not a re-implementation.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
REMOTE_ROOT="/data00/home/sitian/tllm/mechanism-gates/"
RUN_TAG="${RUN_TAG:-mechanism-$(date +%Y%m%d-%H%M%S)}"
LAYERS="${LAYERS:-36}"
SEQ="${SEQ:-8192}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-912}"
CPU_THREADS="${CPU_THREADS:-64}"
WEIGHT_GB="${WEIGHT_GB:-16.0}"
BATCHES="${BATCHES:-1 2 4 8}"
GPU="${GPU:-2}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi

LOCAL_OUT="experiments/mechanism_gates/${RUN_TAG}"
mkdir -p "${LOCAL_OUT}"
CTRL_PATH="/tmp/ssh-mech-$(echo "${REMOTE_HOST}" | tr -c "A-Za-z0-9" _)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto
     -o "ControlPath=${CTRL_PATH}" -o ControlPersist=1800 "${REMOTE_HOST}")

if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 3
fi

echo ">>> staging ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source/tools'"
# --no-xattrs: bsdtar exits 1 on macOS provenance xattr warnings, and pipefail then kills
# the run during staging with an empty output directory as the only symptom
tar --no-xattrs --exclude '__pycache__' -C . -cf - \
  tools/async_handshake_gate.py tools/selector_cost_gate.py \
  tools/microbatch_pipeline_prototype.py \
  tools/cpu_sparse_attention_lib.c tools/cpu_sparse_attention_bench.c \
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

{
  echo "run_tag=${RUN_TAG}"
  echo "layers=${LAYERS} seq=${SEQ} tokens_per_step=${TOKENS_PER_STEP}"
  echo "cpu_threads=${CPU_THREADS} weight_gb=${WEIGHT_GB} batches=${BATCHES}"
  echo "gpu=${GPU}"
  echo "source_revision=$(git rev-parse HEAD)"
  echo "source_dirty_files=$(git status --porcelain | wc -l | tr -d ' ')"
} > "${LOCAL_OUT}/provenance.txt"

"${SSH[@]}" "uptime; echo ---; who | wc -l; echo ---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" \
  > "${LOCAL_OUT}/contamination.txt" 2>&1 || true
echo ">>> host state:"; head -1 "${LOCAL_OUT}/contamination.txt"

echo ">>> building the CPU kernel as a shared library"
"${SSH[@]}" "cd '${REMOTE_DIR}/source/tools' && gcc -O3 -march=native -fopenmp -shared -fPIC \
  -o libcpu_sparse_attn.so cpu_sparse_attention_lib.c -lm && echo build ok"

echo ">>> [1/3] async handshake gate"
"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} \
  ${REMOTE_ENV} OMP_NUM_THREADS=1 '${REMOTE_PYTHON}' tools/async_handshake_gate.py \
  --layers ${LAYERS} --out-json '${REMOTE_DIR}/handshake.json' 2>&1 \
  | tee '${REMOTE_DIR}/handshake.txt'"

echo ">>> [2/3] selector cost gate"
"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} \
  ${REMOTE_ENV} numactl --interleave=all '${REMOTE_PYTHON}' tools/selector_cost_gate.py \
  --layers ${LAYERS} --seq ${SEQ} --cpu-threads 1 8 ${CPU_THREADS} \
  --out-json '${REMOTE_DIR}/selector.json' 2>&1 | tee '${REMOTE_DIR}/selector.txt'"

echo ">>> [3/3] cross-microbatch pipeline prototype"
"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} \
  ${REMOTE_ENV} OMP_NUM_THREADS=${CPU_THREADS} OMP_PROC_BIND=spread OMP_PLACES=cores \
  numactl --interleave=all '${REMOTE_PYTHON}' tools/microbatch_pipeline_prototype.py \
  --so '${REMOTE_DIR}/source/tools/libcpu_sparse_attn.so' \
  --layers ${LAYERS} --seq ${SEQ} --tokens-per-step ${TOKENS_PER_STEP} \
  --cpu-threads ${CPU_THREADS} --weight-gb ${WEIGHT_GB} --batches ${BATCHES} \
  --out-json '${REMOTE_DIR}/pipeline.json' 2>&1 | tee '${REMOTE_DIR}/pipeline.txt'"

for f in handshake selector pipeline; do
  "${SSH[@]}" "cat '${REMOTE_DIR}/${f}.json'" > "${LOCAL_OUT}/${f}.json"
  "${SSH[@]}" "cat '${REMOTE_DIR}/${f}.txt'" > "${LOCAL_OUT}/${f}.txt"
done

echo ">>> done: ${LOCAL_OUT}"
