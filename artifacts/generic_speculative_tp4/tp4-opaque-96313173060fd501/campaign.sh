#!/usr/bin/env bash
set +e
cd '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge'
export CUDA_VISIBLE_DEVICES='7,5,2,1'
export PYTHONPATH='/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge'
'/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'   tools/generic_speculative_tp4_gate.py   --model '/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B'   --gpu-indices '7,5,2,1'   --dist-port-base '36893'   --master-port-base '36993'   --output-dir '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/authority'   > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/remote.log' 2>&1
campaign_status=$?
if (( campaign_status == 0 )); then
  '/data00/home/sitian/sitian-workspace01/tllm/env/bin/python'     tools/verify_generic_speculative_tp4_gate.py     '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/authority'     --source-root '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge'     --out '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/verify.remote.json'     >> '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/remote.log' 2>&1
  campaign_status=$?
fi
printf '%s\n' "${campaign_status}"   > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/campaign.exit_code.tmp'
mv   '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/campaign.exit_code.tmp'   '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/campaign.exit_code'
if (( campaign_status == 0 )); then
  printf '%s\n' COMPLETE > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/campaign.status'
else
  printf '%s\n' FAILED > '/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/artifacts/generic_speculative_tp4/tp4-opaque-96313173060fd501/campaign.status'
fi
exit "${campaign_status}"
