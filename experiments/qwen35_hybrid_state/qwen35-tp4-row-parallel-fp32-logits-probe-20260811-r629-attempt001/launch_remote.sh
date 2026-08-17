#!/usr/bin/env bash

set -euo pipefail

REMOTE="$1"
SOURCE="$2"
PYTHON="/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"

trap 'status=$?; printf "%s\n" "$status" > "$REMOTE/exit_status.txt"' EXIT

nvidia-smi \
  --query-gpu=index,uuid,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  > "$REMOTE/resource_guard.raw.csv"
"$PYTHON" \
  "$REMOTE/resource_guard.py" \
  "$REMOTE/resource_guard.raw.csv" \
  "$REMOTE/resource_guard.json" \
  > "$REMOTE/resource_guard.stdout" \
  2> "$REMOTE/resource_guard.stderr"

export CUDA_VISIBLE_DEVICES=2,4,5,6
export PYTHONPATH="$REMOTE/tools:$SOURCE"
export PYTHONDONTWRITEBYTECODE=1

timeout --signal=TERM --kill-after=60s 1800s \
  "$PYTHON" \
  "$REMOTE/remote_probe_runner.py" \
  "$SOURCE" \
  "$REMOTE/result.json" \
  "$REMOTE/base_probe_runner.py" \
  > "$REMOTE/remote_execution.stdout" \
  2> "$REMOTE/remote_execution.stderr"
