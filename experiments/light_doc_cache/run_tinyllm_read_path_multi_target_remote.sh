#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE="${REMOTE:-sitian@10.232.195.203}"
CONTROL_PATH="${CONTROL_PATH:-/tmp/ssh-sitian-10.232.195.203}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
REMOTE_PY="${REMOTE_PY:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL="${MODEL:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
GPU="${GPU:-auto}"
TARGET_LIMIT="${TARGET_LIMIT:-0}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-$REMOTE_REPO/profile_out/light_doc_cache_multi_target_$TAG}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-$SCRIPT_DIR/read_path_multi_target_$TAG}"
POLICY_FILE="${POLICY_FILE:-experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json}"
TARGET_FILE="${TARGET_FILE:-experiments/light_doc_cache/read_path_multi_target_prompts_v1.json}"

SSH=(ssh)
RSYNC_RSH="ssh"
if [[ -S "$CONTROL_PATH" ]]; then
  SSH+=( -o "ControlPath=$CONTROL_PATH" )
  RSYNC_RSH="ssh -o ControlPath=$CONTROL_PATH"
fi

SYNC_PATHS=(
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py
  experiments/light_doc_cache/make_multi_target_read_path_report.py
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py
  experiments/light_doc_cache/read_path_multi_target_prompts_v1.json
  "$POLICY_FILE"
  tinyvllm/engine/light_doc_cache_runtime.py
)

while IFS= read -r policy_path; do
  [[ -n "$policy_path" ]] && SYNC_PATHS+=( "$policy_path" )
done < <(
  python3 - "$REPO_ROOT/$POLICY_FILE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("default_policy_dir", "base_safe_policy_dir"):
    directory = payload.get(key)
    if directory:
        print(f"{directory}/policy_rows.csv")
PY
)

"${SSH[@]}" "$REMOTE" "mkdir -p '$REMOTE_REPO'"
(
  cd "$REPO_ROOT"
  rsync -av --relative -e "$RSYNC_RSH" "${SYNC_PATHS[@]}" "$REMOTE:$REMOTE_REPO/"
)

"${SSH[@]}" "$REMOTE" \
  "cd '$REMOTE_REPO' && PYTHONDONTWRITEBYTECODE=1 '$REMOTE_PY' -m py_compile \
   experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
   experiments/light_doc_cache/make_multi_target_read_path_report.py \
   experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
   experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py \
   tinyvllm/engine/light_doc_cache_runtime.py"

if [[ "$GPU" == "auto" ]]; then
  GPU="$(
    "${SSH[@]}" "$REMOTE" \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
     | sort -t, -k2,2n | head -1 | cut -d, -f1 | tr -d ' '"
  )"
fi

PORT="$(
  "${SSH[@]}" "$REMOTE" \
    "$REMOTE_PY -c 'import socket; s=socket.socket(); s.bind((\"\",0)); print(s.getsockname()[1]); s.close()'"
)"

"${SSH[@]}" "$REMOTE" \
  "cd '$REMOTE_REPO' && mkdir -p '$REMOTE_OUTPUT' && \
   CUDA_VISIBLE_DEVICES='$GPU' \
   TINYVLLM_DIST_PORT='$PORT' \
   MASTER_PORT='$PORT' \
   PYTHONPATH='$REMOTE_REPO' \
   '$REMOTE_PY' experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
     --model '$MODEL' \
     --policy-file '$POLICY_FILE' \
     --target-file '$TARGET_FILE' \
     --calibration-prompt 'Light Doc Cache TinyLLM calibration prompt.' \
     --calibration-prompt-extra 'Light Doc Cache second calibration prompt for trained recovery.' \
     --calibration-prompt-extra 'Light Doc Cache third calibration prompt for Qwen KV recovery.' \
     --source-count 2 \
     --max-model-len 512 \
     --gpu-memory-utilization 0.30 \
     --target-limit '$TARGET_LIMIT' \
     --output-dir '$REMOTE_OUTPUT'"

mkdir -p "$LOCAL_OUTPUT"
rsync -av -e "$RSYNC_RSH" "$REMOTE:$REMOTE_OUTPUT/" "$LOCAL_OUTPUT/"

test -f "$LOCAL_OUTPUT/multi_target_summary.json"
cat "$LOCAL_OUTPUT/multi_target_report.md"
