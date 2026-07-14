# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `229`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `calibrated_multi_correlated`
- Recovery bank file: `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/profile_out/light_doc_cache_multi_target_20260714_final_evidence/calibration/multi_source_recovery_bank.json`
- Logical stored KV bytes: `21,612,032`
- Logical byte saving fraction: `17.71%`
- Missing-token MSE: `15.8519`
- Max abs logit diff: `8.375`
- Mean abs logit diff: `1.42944`
- Argmax match: `True`
- Original argmax: `16141`
- Restored argmax: `16141`
