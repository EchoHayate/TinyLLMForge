#!/usr/bin/env bash
set +e
export REMOTE_SOURCE='/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/source'
export REMOTE_ARTIFACTS='/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/artifacts'
export MODEL_PATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'
export SELECTED_GPU_CSV='7,5,3,2'
export DIST_PORT_BASE='5275'
export MASTER_PORT_BASE='5375'
export REMOTE_PYTHON='/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'
export MIN_FREE_MEMORY_MIB='49152'
export MAX_GPU_UTILIZATION='10'
export MAX_POST_CELL_DRIFT_MIB='4096'
export POST_SETTLE_ATTEMPTS='12'
export POST_SETTLE_INTERVAL_SECONDS='5'
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/remote_campaign.py'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py     --authority '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/source'     > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/verify.remote.json' 2>> '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.log'
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.status.tmp'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-performance-runs/opaque-af94c09becd3fad5b33ff61f/campaign.status'
exit "${campaign_status}"
