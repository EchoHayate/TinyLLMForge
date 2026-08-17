#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge'
export CUDA_VISIBLE_DEVICES='7'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/blockwise_speculative_verifier_gate.py run   --model '/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B'   --out '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/result.json'   > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/remote.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_blockwise_speculative_verifier_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/result.json'     '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge'     --output '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/verify.remote.json'     >> '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/remote.log' 2>&1
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/campaign.status'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/campaign.status'
fi
exit "${campaign_status}"
