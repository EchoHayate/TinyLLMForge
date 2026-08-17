#!/usr/bin/env bash
set -euo pipefail

export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203
MONITOR_OUTPUT=/Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-monitor-20260811-r606
MASTER_PID=

if ! ssh -S "$CONTROL_PATH" -O check sitian@10.232.195.203 \
  >/dev/null 2>&1; then
  rm -f "$CONTROL_PATH"
  ssh -MN -S "$CONTROL_PATH" \
    -o ControlMaster=yes \
    -o ControlPersist=7d \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    sitian@10.232.195.203 &
  MASTER_PID=$!
  for _ in {1..30}; do
    if ssh -S "$CONTROL_PATH" -O check sitian@10.232.195.203 \
      >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
      wait "$MASTER_PID"
    fi
    sleep 1
  done
  ssh -S "$CONTROL_PATH" -O check sitian@10.232.195.203 \
    >/dev/null
fi

cleanup_master() {
  if [[ -n "$MASTER_PID" ]]; then
    ssh -S "$CONTROL_PATH" -O exit sitian@10.232.195.203 \
      >/dev/null 2>&1 || true
    wait "$MASTER_PID" 2>/dev/null || true
  fi
}
trap cleanup_master EXIT

set +e
/opt/homebrew/opt/python@3.12/bin/python3.12 \
  /Users/bytedance/dev/TinyLLMForge-adaptive-ngram/tools/run_qwen35_tp4_strict_p1_monitor.py \
  --monitor-tag qwen35-tp4-strict-p1-monitor-20260811-r606 \
  --run-tag qwen35-tp4-strict-p1-canonical-20260811-r607 \
  --control-path "$CONTROL_PATH" \
  --prerequisites /Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-readiness-20260806-r551/correctness_prerequisites.json \
  --local-model-manifest /Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/qwen35_hybrid_state/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json \
  --remote-model-dir /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model \
  --remote-model-manifest /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json \
  --interval-s 60 \
  --required-ready-samples 2 \
  --max-samples 10080 \
  --resume-existing
monitor_exit=$?
set -e

if [[ -f "$MONITOR_OUTPUT/monitor_result.json" ]] \
  || [[ -f "$MONITOR_OUTPUT/monitor_failure.json" ]]; then
  exit 0
fi
exit "$monitor_exit"
