#!/usr/bin/env bash
# Remote blockwise/chunked prefill smoke runner.
#
# Intended usage from the remote TinyLLMForge checkout:
#   tools/smoke_blockwise_prefill_remote.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model tools/smoke_blockwise_prefill_remote.sh
#   RUN_REAL_SMOKE=0 tools/smoke_blockwise_prefill_remote.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_REMOTE_PYTHON="/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_REMOTE_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_REMOTE_PYTHON}"
  else
    PYTHON_BIN="python3"
  fi
fi

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export TINYVLLM_DIST_PORT="${TINYVLLM_DIST_PORT:-34567}"
export MASTER_PORT="${MASTER_PORT:-${TINYVLLM_DIST_PORT}}"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_MATH_SMOKE="${RUN_MATH_SMOKE:-1}"
RUN_REAL_SMOKE="${RUN_REAL_SMOKE:-1}"

MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
SMOKE_TAG="${SMOKE_TAG:-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-profile_out}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
MATH_OUT="${MATH_OUT:-${OUT_DIR}/blockwise_prefill_attn_online_softmax_smoke_${SMOKE_TAG}.json}"
REAL_OUT="${REAL_OUT:-${OUT_DIR}/kv_offload_blockwise_prefill_real_longctx_smoke_${SMOKE_TAG}.json}"
MATH_LOG="${MATH_LOG:-${LOG_DIR}/blockwise_prefill_attn_online_softmax_smoke_${SMOKE_TAG}.log}"
REAL_LOG="${REAL_LOG:-${LOG_DIR}/kv_offload_blockwise_prefill_real_longctx_smoke_${SMOKE_TAG}.log}"

BLOCKWISE_PREFILL_PREFIX_TOKENS="${BLOCKWISE_PREFILL_PREFIX_TOKENS:-2176}"
BLOCKWISE_PREFILL_CHUNK_TOKENS="${BLOCKWISE_PREFILL_CHUNK_TOKENS:-128}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_PREFILL_TOKENS_PER_STEP="${MAX_NUM_PREFILL_TOKENS_PER_STEP:-256}"
KV_OFFLOAD_GPU_BLOCKS="${KV_OFFLOAD_GPU_BLOCKS:-2}"
KV_OFFLOAD_LOGICAL_BLOCKS="${KV_OFFLOAD_LOGICAL_BLOCKS:-8}"
KV_OFFLOAD_BLOCKWISE_BLOCKS="${KV_OFFLOAD_BLOCKWISE_BLOCKS:-1}"
PROMPT_REPEAT="${PROMPT_REPEAT:-64}"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

if [[ -z "${PROMPT:-}" ]]; then
  PROMPT="$("${PYTHON_BIN}" - "${PROMPT_REPEAT}" <<'PY'
import sys
repeat = int(sys.argv[1])
sentence = (
    "TinyLLMForge blockwise prefill remote smoke verifies a long context prompt "
    "with repeated stable tokens and deterministic decoding. "
)
print(sentence * repeat, end="")
PY
)"
fi

run_preflight() {
  "${PYTHON_BIN}" -m py_compile \
    tinyvllm/config.py \
    tinyvllm/engine/model_runner.py \
    tinyvllm/engine/scheduler.py \
    tinyvllm/layers/attention.py \
    tinyvllm/utils/context.py \
    tools/profile_ngram_commit.py \
    tools/test_chunked_prefill.py
  "${PYTHON_BIN}" tools/test_chunked_prefill.py
  "${PYTHON_BIN}" tools/test_ngram_speculative.py
}

run_with_log() {
  local log_path="$1"
  shift
  if ! "$@" >"${log_path}" 2>&1; then
    echo "[smoke] command failed; log=${log_path}" >&2
    "${PYTHON_BIN}" - "${log_path}" <<'PY' >&2
import sys

path = sys.argv[1]
try:
    with open(path, errors="replace") as f:
        lines = f.readlines()
except OSError as exc:
    print(f"failed to read log {path}: {exc}")
    raise SystemExit(0)

print(f"--- last {min(80, len(lines))} log lines: {path} ---")
for line in lines[-80:]:
    print(line, end="")
print("--- end log ---")
PY
    return 1
  fi
}

run_math_smoke() {
  run_with_log "${MATH_LOG}" "${PYTHON_BIN}" tools/profile_ngram_commit.py \
    --blockwise-prefill-attn-smoke \
    --blockwise-prefill-prefix-tokens "${BLOCKWISE_PREFILL_PREFIX_TOKENS}" \
    --blockwise-prefill-chunk-tokens "${BLOCKWISE_PREFILL_CHUNK_TOKENS}" \
    --out-json "${MATH_OUT}"
}

run_real_smoke() {
  run_with_log "${REAL_LOG}" "${PYTHON_BIN}" tools/profile_ngram_commit.py \
    --mode baseline-only \
    --model "${MODEL_PATH}" \
    --prompt "${PROMPT}" \
    --max-output-len "${MAX_OUTPUT_LEN}" \
    --temperature 0.0 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-prefill-tokens-per-step "${MAX_NUM_PREFILL_TOKENS_PER_STEP}" \
    --kv-offload-mvp0 \
    --kv-offload-gpu-blocks "${KV_OFFLOAD_GPU_BLOCKS}" \
    --kv-offload-logical-blocks "${KV_OFFLOAD_LOGICAL_BLOCKS}" \
    --kv-offload-blockwise-prefill \
    --kv-offload-blockwise-decode \
    --kv-offload-blockwise-blocks "${KV_OFFLOAD_BLOCKWISE_BLOCKS}" \
    --out-json "${REAL_OUT}"
}

print_summary() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import json
import sys

failed = []
for path in sys.argv[1:]:
    with open(path) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    print(path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("gate_pass") is not True:
        failed.append(path)
if failed:
    raise SystemExit("gate_pass failed for: " + ", ".join(failed))
PY
}

summary_paths=()

echo "[smoke] repo=${REPO_ROOT}"
echo "[smoke] python=${PYTHON_BIN}"
echo "[smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[smoke] TINYVLLM_DIST_PORT=${TINYVLLM_DIST_PORT} MASTER_PORT=${MASTER_PORT}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  echo "[smoke] running preflight checks"
  run_preflight
fi

if [[ "${RUN_MATH_SMOKE}" == "1" ]]; then
  echo "[smoke] running blockwise prefill math smoke -> ${MATH_OUT}"
  echo "[smoke] math log -> ${MATH_LOG}"
  run_math_smoke
  summary_paths+=("${MATH_OUT}")
fi

if [[ "${RUN_REAL_SMOKE}" == "1" ]]; then
  echo "[smoke] running real-model long-context smoke -> ${REAL_OUT}"
  echo "[smoke] real log -> ${REAL_LOG}"
  run_real_smoke
  summary_paths+=("${REAL_OUT}")
fi

if (( ${#summary_paths[@]} > 0 )); then
  echo "[smoke] summaries"
  print_summary "${summary_paths[@]}"
fi

echo "[smoke] done"
