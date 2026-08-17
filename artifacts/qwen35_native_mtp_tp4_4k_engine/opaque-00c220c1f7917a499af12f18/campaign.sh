#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/source'
export CUDA_VISIBLE_DEVICES='7,5,3,2'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/qwen35_native_mtp_tp4_4k_engine_gate.py   --model '/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'   --gpu-indices '7,5,3,2'   --tp1-result '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/tp1-result.json'   --dist-port-base '4556'   --master-port-base '4656'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/artifacts/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_native_mtp_tp4_4k_engine_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/source'     > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/verify.remote.json'     2>> '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.log'
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.status.tmp'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/campaign.status'
exit "${campaign_status}"
