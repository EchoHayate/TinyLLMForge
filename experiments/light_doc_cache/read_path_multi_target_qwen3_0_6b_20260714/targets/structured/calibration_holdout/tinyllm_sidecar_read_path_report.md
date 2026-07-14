# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `36`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `calibrated_multi_correlated`
- Recovery bank file: `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/profile_out/light_doc_cache_multi_target_20260714_final_evidence/calibration/multi_source_recovery_bank.json`
- Logical stored KV bytes: `3,400,704`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `13.832`
- Max abs logit diff: `5.25`
- Mean abs logit diff: `0.921632`
- Argmax match: `True`
- Original argmax: `785`
- Restored argmax: `785`
