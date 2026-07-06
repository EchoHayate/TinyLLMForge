#!/usr/bin/env bash
# Remote blockwise/chunked prefill smoke runner.
#
# Intended usage from the remote TinyLLMForge checkout:
#   tools/smoke_blockwise_prefill_remote.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model tools/smoke_blockwise_prefill_remote.sh
#   RUN_REAL_SMOKE=0 tools/smoke_blockwise_prefill_remote.sh
#   RUN_GPU_BLOCKS_MATRIX=1 tools/smoke_blockwise_prefill_remote.sh
#   RUN_MULTI_PROMPT_SMOKE=1 tools/smoke_blockwise_prefill_remote.sh

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
RUN_GPU_BLOCKS_MATRIX="${RUN_GPU_BLOCKS_MATRIX:-0}"
RUN_MULTI_PROMPT_SMOKE="${RUN_MULTI_PROMPT_SMOKE:-0}"

MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
SMOKE_TAG="${SMOKE_TAG:-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-profile_out}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
MATH_OUT="${MATH_OUT:-${OUT_DIR}/blockwise_prefill_attn_online_softmax_smoke_${SMOKE_TAG}.json}"
REAL_OUT="${REAL_OUT:-${OUT_DIR}/kv_offload_blockwise_prefill_real_longctx_smoke_${SMOKE_TAG}.json}"
MULTI_PROMPT_OUT="${MULTI_PROMPT_OUT:-${OUT_DIR}/kv_offload_blockwise_prefill_real_multiprompt_smoke_${SMOKE_TAG}.json}"
MATH_LOG="${MATH_LOG:-${LOG_DIR}/blockwise_prefill_attn_online_softmax_smoke_${SMOKE_TAG}.log}"
REAL_LOG="${REAL_LOG:-${LOG_DIR}/kv_offload_blockwise_prefill_real_longctx_smoke_${SMOKE_TAG}.log}"
MULTI_PROMPT_LOG="${MULTI_PROMPT_LOG:-${LOG_DIR}/kv_offload_blockwise_prefill_real_multiprompt_smoke_${SMOKE_TAG}.log}"

BLOCKWISE_PREFILL_PREFIX_TOKENS="${BLOCKWISE_PREFILL_PREFIX_TOKENS:-2176}"
BLOCKWISE_PREFILL_CHUNK_TOKENS="${BLOCKWISE_PREFILL_CHUNK_TOKENS:-128}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MULTI_PROMPT_MAX_NUM_SEQS="${MULTI_PROMPT_MAX_NUM_SEQS:-2}"
MULTI_PROMPT_COUNT="${MULTI_PROMPT_COUNT:-2}"
MAX_NUM_PREFILL_TOKENS_PER_STEP="${MAX_NUM_PREFILL_TOKENS_PER_STEP:-256}"
KV_OFFLOAD_GPU_BLOCKS="${KV_OFFLOAD_GPU_BLOCKS:-2}"
KV_OFFLOAD_GPU_BLOCKS_MATRIX="${KV_OFFLOAD_GPU_BLOCKS_MATRIX:-1 2 4}"
KV_OFFLOAD_LOGICAL_BLOCKS="${KV_OFFLOAD_LOGICAL_BLOCKS:-8}"
KV_OFFLOAD_BLOCKWISE_BLOCKS="${KV_OFFLOAD_BLOCKWISE_BLOCKS:-1}"
PROMPT_REPEAT="${PROMPT_REPEAT:-64}"
MULTI_PROMPT_REPEAT="${MULTI_PROMPT_REPEAT:-40}"
MATRIX_REQUIRE_PASS="${MATRIX_REQUIRE_PASS:-1}"

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

build_prompt_args() {
  local prompt_count="${1:-1}"
  local prompt_repeat="${2:-${PROMPT_REPEAT}}"
  local prompt_prefix="${3:-${PROMPT:-}}"
  "${PYTHON_BIN}" - "${prompt_count}" "${prompt_repeat}" "${prompt_prefix}" <<'PY'
import sys

count = int(sys.argv[1])
repeat = int(sys.argv[2])
prefix = sys.argv[3]
base = (
    "TinyLLMForge blockwise prefill remote smoke verifies a long context prompt "
    "with repeated stable tokens and deterministic decoding. "
)
for idx in range(count):
    prompt = prefix if prefix else f"Prompt {idx}: " + base * repeat
    if prefix and count > 1:
        prompt = f"Prompt {idx}: " + prompt
    print("--prompt")
    print(prompt)
PY
}

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
  local gpu_blocks="${1:-${KV_OFFLOAD_GPU_BLOCKS}}"
  local out_path="${2:-${REAL_OUT}}"
  local log_path="${3:-${REAL_LOG}}"
  local prompt_count="${4:-1}"
  local max_num_seqs="${5:-${MAX_NUM_SEQS}}"
  local prompt_repeat="${6:-${PROMPT_REPEAT}}"
  local prompt_prefix="${7:-${PROMPT:-}}"
  local prompt_args=()
  while IFS= read -r line; do
    prompt_args+=("${line}")
  done < <(build_prompt_args "${prompt_count}" "${prompt_repeat}" "${prompt_prefix}")
  local cmd=(
    "${PYTHON_BIN}" tools/profile_ngram_commit.py
    --mode baseline-only \
    --model "${MODEL_PATH}" \
    "${prompt_args[@]}" \
    --max-output-len "${MAX_OUTPUT_LEN}" \
    --temperature 0.0 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-prefill-tokens-per-step "${MAX_NUM_PREFILL_TOKENS_PER_STEP}" \
    --kv-offload-mvp0 \
    --kv-offload-gpu-blocks "${gpu_blocks}" \
    --kv-offload-logical-blocks "${KV_OFFLOAD_LOGICAL_BLOCKS}" \
    --kv-offload-blockwise-prefill \
    --kv-offload-blockwise-decode \
    --kv-offload-blockwise-blocks "${KV_OFFLOAD_BLOCKWISE_BLOCKS}" \
    --out-json "${out_path}"
  )
  run_with_log "${log_path}" "${cmd[@]}"
}

run_multi_prompt_smoke() {
  echo "[smoke] running real-model multi-prompt smoke -> ${MULTI_PROMPT_OUT}"
  echo "[smoke] multi-prompt log -> ${MULTI_PROMPT_LOG}"
  run_real_smoke \
    "${KV_OFFLOAD_GPU_BLOCKS}" \
    "${MULTI_PROMPT_OUT}" \
    "${MULTI_PROMPT_LOG}" \
    "${MULTI_PROMPT_COUNT}" \
    "${MULTI_PROMPT_MAX_NUM_SEQS}" \
    "${MULTI_PROMPT_REPEAT}" \
    ""
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

print_matrix_summary() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import json
import os
import sys

failed = []
headers = [
    "path",
    "gate_pass",
    "elapsed_s",
    "h2d_copies",
    "d2h_copies",
    "evictions",
    "resident_blocks",
]
print("\t".join(headers))
for path in sys.argv[1:]:
    if not os.path.exists(path):
        print("\t".join([path, "missing", "", "", "", "", ""]))
        failed.append(path)
        continue
    with open(path) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    stats = data.get("kv_offload", summary.get("kv_offload_stats", {}))
    row = [
        path,
        str(summary.get("gate_pass")),
        str(summary.get("elapsed_s", "")),
        str(stats.get("h2d_copies", summary.get("h2d_copies", ""))),
        str(stats.get("d2h_copies", summary.get("d2h_copies", ""))),
        str(stats.get("evictions", summary.get("evictions", ""))),
        str(stats.get("resident_blocks", summary.get("resident_blocks", ""))),
    ]
    print("\t".join(row))
    if summary.get("gate_pass") is not True:
        failed.append(path)
if failed and os.environ.get("MATRIX_REQUIRE_PASS", "1") == "1":
    raise SystemExit("matrix gate_pass failed for: " + ", ".join(failed))
PY
}

run_real_smoke_matrix() {
  local matrix_paths=()
  local failed_runs=()
  local gpu_blocks
  for gpu_blocks in ${KV_OFFLOAD_GPU_BLOCKS_MATRIX}; do
    local matrix_out="${REAL_OUT%.json}_gpu${gpu_blocks}.json"
    local matrix_log="${REAL_LOG%.log}_gpu${gpu_blocks}.log"
    echo "[smoke] running real-model long-context matrix gpu_blocks=${gpu_blocks} -> ${matrix_out}"
    echo "[smoke] real matrix log -> ${matrix_log}"
    if run_real_smoke "${gpu_blocks}" "${matrix_out}" "${matrix_log}"; then
      matrix_paths+=("${matrix_out}")
    else
      failed_runs+=("gpu${gpu_blocks}")
      matrix_paths+=("${matrix_out}")
    fi
  done
  echo "[smoke] matrix summary"
  print_matrix_summary "${matrix_paths[@]}"
  if (( ${#failed_runs[@]} > 0 )) && [[ "${MATRIX_REQUIRE_PASS}" == "1" ]]; then
    echo "[smoke] failed matrix commands: ${failed_runs[*]}" >&2
    return 1
  fi
}

summary_paths=()

echo "[smoke] repo=${REPO_ROOT}"
echo "[smoke] python=${PYTHON_BIN}"
echo "[smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[smoke] TINYVLLM_DIST_PORT=${TINYVLLM_DIST_PORT} MASTER_PORT=${MASTER_PORT}"
echo "[smoke] RUN_GPU_BLOCKS_MATRIX=${RUN_GPU_BLOCKS_MATRIX} KV_OFFLOAD_GPU_BLOCKS_MATRIX=${KV_OFFLOAD_GPU_BLOCKS_MATRIX}"
echo "[smoke] RUN_MULTI_PROMPT_SMOKE=${RUN_MULTI_PROMPT_SMOKE} MULTI_PROMPT_COUNT=${MULTI_PROMPT_COUNT} MULTI_PROMPT_MAX_NUM_SEQS=${MULTI_PROMPT_MAX_NUM_SEQS}"

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

if [[ "${RUN_REAL_SMOKE}" == "1" && "${RUN_GPU_BLOCKS_MATRIX}" == "1" ]]; then
  run_real_smoke_matrix
elif [[ "${RUN_REAL_SMOKE}" == "1" ]]; then
  echo "[smoke] running real-model long-context smoke -> ${REAL_OUT}"
  echo "[smoke] real log -> ${REAL_LOG}"
  run_real_smoke "${KV_OFFLOAD_GPU_BLOCKS}" "${REAL_OUT}" "${REAL_LOG}"
  summary_paths+=("${REAL_OUT}")
fi

if [[ "${RUN_MULTI_PROMPT_SMOKE}" == "1" ]]; then
  run_multi_prompt_smoke
  summary_paths+=("${MULTI_PROMPT_OUT}")
fi

if (( ${#summary_paths[@]} > 0 )); then
  echo "[smoke] summaries"
  print_summary "${summary_paths[@]}"
fi

echo "[smoke] done"
