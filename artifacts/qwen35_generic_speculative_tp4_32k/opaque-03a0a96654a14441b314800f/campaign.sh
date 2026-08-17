#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/source'
export CUDA_VISIBLE_DEVICES='7,5,3,2'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/qwen35_generic_speculative_tp4_32k_gate.py   --model '/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'   --gpu-indices '7,5,3,2'   --dist-port-base '3804'   --master-port-base '3904'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/artifacts/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_generic_speculative_tp4_32k_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/source'     --out '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/verify.remote.json'     >> '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.log' 2>&1
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.status.tmp'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-32k-runs/opaque-03a0a96654a14441b314800f/campaign.status'
exit "${campaign_status}"
