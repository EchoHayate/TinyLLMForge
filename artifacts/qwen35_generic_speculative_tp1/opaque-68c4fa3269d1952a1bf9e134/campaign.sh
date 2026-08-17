#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/source'
export CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/source'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python' tools/qwen35_generic_speculative_tp1_gate.py   --model '/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model'   --gpu-index '0'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/artifacts/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_qwen35_generic_speculative_tp1_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/artifacts/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/source'     > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/verify.remote.txt' 2>&1
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.status.tmp'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.status.tmp'
fi
mv   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.status.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/qwen35-generic-speculative-tp1-runs/opaque-68c4fa3269d1952a1bf9e134/campaign.status'
exit "${campaign_status}"
