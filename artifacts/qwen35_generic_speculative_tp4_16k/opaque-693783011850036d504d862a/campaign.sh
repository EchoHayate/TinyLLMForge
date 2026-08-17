#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/source'
export CUDA_VISIBLE_DEVICES='2,7,6,5'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/qwen35_generic_speculative_tp4_16k_gate.py   --model '/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'   --gpu-indices '2,7,6,5'   --dist-port-base '4368'   --master-port-base '4468'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/artifacts/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_generic_speculative_tp4_16k_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/source'     --out '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/verify.remote.json'     >> '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.log' 2>&1
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.status.tmp'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp4-16k-runs/opaque-693783011850036d504d862a/campaign.status'
exit "${campaign_status}"
