#!/usr/bin/env bash
set -euo pipefail

KRB5CCNAME="${KRB5CCNAME:-FILE:/Users/bytedance/krb5cc_sitian}"
export KRB5CCNAME

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
CHECKPOINT="${CHECKPOINT:-/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
TAG="${TAG:-qwen35-mtp-ownership-${RANDOM}-$$}"
DIST_PORT="${TINYVLLM_DIST_PORT:-$((30000 + (RANDOM % 20000)))}"
REMOTE_RUN_ROOT="/data00/home/sitian/sitian-workspace01/tllm/qwen35-mtp-runs/${TAG}"
REMOTE_ARTIFACT="${REMOTE_RUN_ROOT}/artifacts/qwen35_mtp_model_runner_ownership_gate.json"
LOCAL_RUN_BASE="${LOCAL_RUN_BASE:-artifacts/qwen35-mtp-runs}"
LOCAL_RUN_ROOT="${LOCAL_RUN_BASE}/${TAG}"
LOCAL_ARTIFACT="${LOCAL_RUN_ROOT}/qwen35_mtp_model_runner_ownership_gate.json"

SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ControlMaster=no
  -o ControlPath=none
)
SSH=(ssh "${SSH_OPTIONS[@]}")
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none"

retry() {
  local max_attempts="$1"
  shift
  local attempt=1
  until "$@"; do
    if (( attempt >= max_attempts )); then
      return 1
    fi
    sleep "${attempt}"
    attempt=$((attempt + 1))
  done
}

retry 3 "${SSH[@]}" "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_RUN_ROOT}' && cp -a '${REMOTE_BASE}/.' '${REMOTE_RUN_ROOT}/'"

SOURCE_FILES=(
  tinyvllm/config.py
  tinyvllm/speculative/__init__.py
  tinyvllm/speculative/adapter.py
  tinyvllm/speculative/batch_runtime.py
  tinyvllm/engine/speculative_proposal_executor.py
  tinyvllm/engine/speculative_model_runner.py
  tinyvllm/engine/speculative_runtime.py
  tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py
  tinyvllm/engine/qwen35_mtp_registration.py
  tinyvllm/engine/model_runner.py
  tinyvllm/engine/llm_engine.py
  tinyvllm/engine/proposal_kv_cache.py
  tinyvllm/engine/qwen35_mtp_executor.py
  tinyvllm/engine/qwen35_mtp_graph.py
  tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py
  tinyvllm/engine/qwen35_mtp_graph_scratch.py
  tinyvllm/utils/context.py
  tinyvllm/layers/qwen35_full_attention.py
  tinyvllm/models/qwen35_checkpoint.py
  tinyvllm/models/qwen35_components.py
  tinyvllm/models/qwen35_mtp_checkpoint.py
  tinyvllm/models/qwen35_mtp.py
  tools/qwen35_mtp_real_checkpoint_gate.py
  tools/qwen35_mtp_model_runner_ownership_gate.py
)

retry 3 rsync -a --relative \
  -e "${RSYNC_SSH}" \
  "${SOURCE_FILES[@]}" \
  "${REMOTE_HOST}:${REMOTE_RUN_ROOT}/"

"${SSH[@]}" "${REMOTE_HOST}" \
  "cd '${REMOTE_RUN_ROOT}' && \
   mkdir -p '${REMOTE_RUN_ROOT}/artifacts' && \
   CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' \
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH='${REMOTE_RUN_ROOT}' \
   TINYVLLM_DIST_PORT='${DIST_PORT}' \
   '${REMOTE_PYTHON}' tools/qwen35_mtp_model_runner_ownership_gate.py \
     --checkpoint '${CHECKPOINT}' \
     --output '${REMOTE_ARTIFACT}'"

mkdir -p "${LOCAL_RUN_ROOT}"
retry 3 rsync -a \
  -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ARTIFACT}" "${LOCAL_RUN_ROOT}/"

python3 - "${LOCAL_ARTIFACT}" <<'PY'
import json
from pathlib import Path
import sys

from tools.qwen35_mtp_model_runner_ownership_gate import (
    REQUIRED_BATCH_SIZES,
    REQUIRED_Q_VALUES,
    validate_ownership_gate_report,
)

artifact = Path(sys.argv[1])
report = json.loads(artifact.read_text(encoding="utf-8"))
validate_ownership_gate_report(
    report,
    required_q_values=REQUIRED_Q_VALUES,
    required_batch_sizes=REQUIRED_BATCH_SIZES,
)
if report.get("status") != "PASS":
    raise SystemExit("ownership gate status is not PASS")
if report.get("promotion_classification") != "NOT_PROMOTABLE":
    raise SystemExit("ownership gate promotion boundary changed")
print("status=PASS")
PY

printf 'remote_run_root=%s\n' "${REMOTE_RUN_ROOT}"
printf 'local_artifact=%s\n' "${LOCAL_ARTIFACT}"
