#!/usr/bin/env bash
# Measure CPU-side selected-token attention on the server, across thread counts and the
# two fidelity budgets the end-to-end gate produced (answer-level ~3.3%, trajectory-level
# ~11.1% of an 8k context).
#
# The number that matters is step_ms versus the GPU weight-read window c0 = 12.113 ms. If
# the CPU cannot finish one decode step's attention inside that window it cannot be hidden
# and the whole offload plan is dead regardless of how good the selector is.
#
# NUMA interleave is used because the gather gate measured interleave beating single-node
# pinning for exactly this access pattern.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_ROOT="/data00/home/sitian/tllm/cpu-attn-runs/"
RUN_TAG="${RUN_TAG:-cpu-attn-$(date +%Y%m%d-%H%M%S)}"
LAYERS="${LAYERS:-36}"
SEQ="${SEQ:-8192}"
KV_HEADS="${KV_HEADS:-8}"
GROUP_SIZE="${GROUP_SIZE:-4}"
DIM="${DIM:-128}"
GRANULARITY="${GRANULARITY:-32}"
TOKEN_BUDGETS="${TOKEN_BUDGETS:-272 448 912 2048 8192}"
THREAD_COUNTS="${THREAD_COUNTS:-16 32 64 128}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"

REMOTE_DIR="${REMOTE_ROOT}${RUN_TAG}"
if [[ "${REMOTE_DIR}" != "${REMOTE_ROOT}"* || "${REMOTE_DIR}" == "${REMOTE_ROOT}" ]]; then
  echo "unsafe REMOTE_DIR: ${REMOTE_DIR}" >&2
  exit 2
fi

LOCAL_OUT="experiments/cpu_sparse_attention/${RUN_TAG}"
mkdir -p "${LOCAL_OUT}"
CTRL_PATH="/tmp/ssh-cpuattn-$(echo "${REMOTE_HOST}" | tr -c "A-Za-z0-9" _)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto
     -o "ControlPath=${CTRL_PATH}" -o ControlPersist=1800 "${REMOTE_HOST}")

if "${SSH[@]}" "test -e '${REMOTE_DIR}'"; then
  echo "immutable remote tag already exists: ${REMOTE_DIR}" >&2
  exit 3
fi

echo ">>> staging ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '${REMOTE_DIR}'"
"${SSH[@]}" "cat > '${REMOTE_DIR}/cpu_sparse_attention_bench.c'" < tools/cpu_sparse_attention_bench.c

{
  echo "run_tag=${RUN_TAG}"
  echo "layers=${LAYERS} seq=${SEQ} kv_heads=${KV_HEADS} group_size=${GROUP_SIZE} dim=${DIM}"
  echo "granularity=${GRANULARITY}"
  echo "token_budgets=${TOKEN_BUDGETS}"
  echo "thread_counts=${THREAD_COUNTS}"
  echo "iters=${ITERS} warmup=${WARMUP}"
  echo "source_revision=$(git rev-parse HEAD)"
  echo "source_dirty_files=$(git status --porcelain | wc -l | tr -d ' ')"
} > "${LOCAL_OUT}/provenance.txt"

"${SSH[@]}" "uptime; echo ---; who | wc -l; echo ---; lscpu | grep -E 'Model name|Socket|NUMA node\(s\)|Thread|Core'; echo ---; grep -o -m1 -E 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' '; echo; echo ---; numactl --hardware | head -8" \
  > "${LOCAL_OUT}/contamination.txt" 2>&1 || true
echo ">>> host state:"; head -1 "${LOCAL_OUT}/contamination.txt"

echo ">>> building"
"${SSH[@]}" "cd '${REMOTE_DIR}' && gcc -O3 -march=native -fopenmp -o cpu_sparse_attention_bench cpu_sparse_attention_bench.c -lm && ./cpu_sparse_attention_bench --self-test" \
  | tee "${LOCAL_OUT}/self_test.txt"

: > "${LOCAL_OUT}/results.txt"
for tokens in ${TOKEN_BUDGETS}; do
  for threads in ${THREAD_COUNTS}; do
    echo ">>> tokens=${tokens} threads=${threads}"
    "${SSH[@]}" "cd '${REMOTE_DIR}' && OMP_PROC_BIND=spread OMP_PLACES=cores \
      numactl --interleave=all ./cpu_sparse_attention_bench \
      --layers ${LAYERS} --seq ${SEQ} --kv-heads ${KV_HEADS} --group-size ${GROUP_SIZE} \
      --dim ${DIM} --granularity ${GRANULARITY} --tokens-per-step ${tokens} \
      --threads ${threads} --iters ${ITERS} --warmup ${WARMUP}" \
      | tee -a "${LOCAL_OUT}/results.txt"
    echo "---" >> "${LOCAL_OUT}/results.txt"
  done
done

echo ">>> done: ${LOCAL_OUT}"
