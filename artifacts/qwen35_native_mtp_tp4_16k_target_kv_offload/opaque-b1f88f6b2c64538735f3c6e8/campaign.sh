#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/source'
export CUDA_VISIBLE_DEVICES='7,5,3,2'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py   --model '/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'   --gpu-indices '7,5,3,2'   --dist-port-base '7985'   --master-port-base '8085'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/artifacts/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/source'     > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/verify.remote.json'     2>> '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.log'
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE     > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.status.tmp'
else
  printf '%s\n' FAILED     > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-16k-target-kv-offload-runs/opaque-b1f88f6b2c64538735f3c6e8/campaign.status'
exit "${campaign_status}"
