#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/source'
export CUDA_VISIBLE_DEVICES='3,0,6,4'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/qwen35_generic_speculative_tp4_32k_gate.py   --model '/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'   --gpu-indices '3,0,6,4'   --dist-port-base '9641'   --master-port-base '9741'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/artifacts/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_generic_speculative_tp4_32k_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/source'     --out '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/verify.remote.json'     >> '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.log' 2>&1
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.status.tmp'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-758ce5286988bca040f8c65c/campaign.status'
exit "${campaign_status}"
