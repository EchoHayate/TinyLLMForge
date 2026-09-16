#!/usr/bin/env bash
# Overlap gate: run the GPU decode loop with and without the CPU attention benchmark
# hammering the cores, and measure the per-layer Q/output round trip.
#
# The two previous gates each measured one side in isolation. This one measures the
# interference between them, which is the part that no amount of kernel tuning fixes:
#   - a busy CPU can starve the CUDA launch thread
#   - 2 x n_layers small transfers per step cost latency, not bandwidth
#
# Needs both a GPU and the compiled C benchmark on the same host, so everything runs
# remote. Same sitepatch convention as the other GPU runners (the host's user-site
# flash_attn is built against a newer libstdc++ and importing qwen3 dies without it).
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/tllm/env/bin/python}"
REMOTE_ROOT="/data00/home/sitian/tllm/overlap-runs/"
TARGET_MODEL="${TARGET_MODEL:-/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-overlap-$(date +%Y%m%d-%H%M%S)}"
SEQ_LEN="${SEQ_LEN:-8192}"
STEPS="${STEPS:-30}"
# trajectory-level budget from the e2e gate (11.1% of 8k, rounded to a unit multiple),
# i.e. the expensive end of what the offload would actually have to compute
CPU_TOKENS="${CPU_TOKENS:-912}"
CPU_THREADS="${CPU_THREADS:-64}"
# measured on this host by the CPU attention gate at tokens=896, 64 threads
CPU_ATTN_MS="${CPU_ATTN_MS:-2.051}"
GPU="${GPU:-2}"
REMOTE_USER_SITE="${REMOTE_USER_SITE:-/data00/home/sitian/.local/lib/python3.11/site-packages}"
REMOTE_SITE_EXCLUDE="${REMOTE_SITE_EXCLUDE:-flash_attn torchvision}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-/data00/home/sitian/tllm/miniforge/lib}"

REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi

LOCAL_OUT="experiments/cpu_gpu_overlap/${RUN_TAG}"
mkdir -p "${LOCAL_OUT}"
CTRL_PATH="/tmp/ssh-overlap-$(echo "${REMOTE_HOST}" | tr -c "A-Za-z0-9" _)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto
     -o "ControlPath=${CTRL_PATH}" -o ControlPersist=1800 "${REMOTE_HOST}")

if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 3
fi

echo ">>> staging ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}/source/tools'"
# --no-xattrs: bsdtar exits 1 on the com.apple.provenance xattr warnings, and with
# pipefail that silently kills the run during staging
tar --no-xattrs --exclude '__pycache__' -C . -cf - \
  tools/cpu_gpu_overlap_gate.py tools/engine_step_cpu_interference.py \
  tools/cpu_sparse_attention_bench.c tinyvllm \
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
# the GPU driver process must not fight the OpenMP pool for the launch thread by accident;
# torch's own intra-op pool is what would do that, so it is pinned to one thread
REMOTE_ENV+=" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"

{
  echo "run_tag=${RUN_TAG}"
  echo "model=${TARGET_MODEL}"
  echo "seq_len=${SEQ_LEN} steps=${STEPS}"
  echo "cpu_tokens_per_step=${CPU_TOKENS} cpu_threads=${CPU_THREADS}"
  echo "cpu_attn_ms_reference=${CPU_ATTN_MS}"
  echo "gpu=${GPU}"
  echo "source_revision=$(git rev-parse HEAD)"
  echo "source_dirty_files=$(git status --porcelain | wc -l | tr -d ' ')"
} > "${LOCAL_OUT}/provenance.txt"

"${SSH[@]}" "uptime; echo ---; who | wc -l; echo ---; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader" \
  > "${LOCAL_OUT}/contamination.txt" 2>&1 || true
echo ">>> host state:"; head -1 "${LOCAL_OUT}/contamination.txt"

echo ">>> building the CPU load generator"
"${SSH[@]}" "cd '${REMOTE_DIR}/source/tools' && gcc -O3 -march=native -fopenmp \
  -o cpu_sparse_attention_bench cpu_sparse_attention_bench.c -lm && echo build ok"

echo ">>> running the overlap gate"
"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} ${REMOTE_ENV} \
  '${REMOTE_PYTHON}' tools/cpu_gpu_overlap_gate.py --model '${TARGET_MODEL}' \
  --seq-len ${SEQ_LEN} --steps ${STEPS} \
  --cpu-bench-binary '${REMOTE_DIR}/source/tools/cpu_sparse_attention_bench' \
  --cpu-threads ${CPU_THREADS} --cpu-tokens-per-step ${CPU_TOKENS} \
  --cpu-attn-ms ${CPU_ATTN_MS} \
  --out-json '${REMOTE_DIR}/overlap.json' 2>&1 | tee '${REMOTE_DIR}/overlap.txt'"

"${SSH[@]}" "cat '${REMOTE_DIR}/overlap.json'" > "${LOCAL_OUT}/overlap.json"
"${SSH[@]}" "cat '${REMOTE_DIR}/overlap.txt'" > "${LOCAL_OUT}/overlap.txt"

# The HF loop above is launch-bound (~34 ms/step), so its absolute inflation cannot be
# charged against the 12.113 ms engine window. Measure the same interference on the engine
# with CUDA graphs on, which is the system the cost model actually describes.
echo ">>> engine step under CPU load"
"${SSH[@]}" "set -o pipefail; cd '${REMOTE_DIR}/source' && CUDA_VISIBLE_DEVICES=${GPU} ${REMOTE_ENV} \
  '${REMOTE_PYTHON}' tools/engine_step_cpu_interference.py --model '${TARGET_MODEL}' \
  --seq-len ${SEQ_LEN} --cpu-threads ${CPU_THREADS} \
  --cpu-tokens-per-step ${CPU_TOKENS} \
  --cpu-bench-binary '${REMOTE_DIR}/source/tools/cpu_sparse_attention_bench' \
  --out-json '${REMOTE_DIR}/engine-interference.json' 2>&1 \
  | tee '${REMOTE_DIR}/engine-interference.txt'" || echo "!!! engine phase failed, see log"
"${SSH[@]}" "cat '${REMOTE_DIR}/engine-interference.json' 2>/dev/null" > "${LOCAL_OUT}/engine-interference.json" || true
"${SSH[@]}" "cat '${REMOTE_DIR}/engine-interference.txt' 2>/dev/null" > "${LOCAL_OUT}/engine-interference.txt" || true

echo ">>> done: ${LOCAL_OUT}"
