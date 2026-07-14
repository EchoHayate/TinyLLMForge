# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `52`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `calibrated_multi_correlated`
- Recovery bank file: `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/profile_out/light_doc_cache_multi_target_20260714_final_evidence/calibration/multi_source_recovery_bank.json`
- Logical stored KV bytes: `4,912,128`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `12.6161`
- Max abs logit diff: `5.5625`
- Mean abs logit diff: `0.950152`
- Argmax match: `True`
- Original argmax: `4226`
- Restored argmax: `4226`
