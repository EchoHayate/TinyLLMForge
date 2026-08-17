# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `14`
- KV cache shape: `[2, 28, 805, 256, 8, 128]`
- Recovery mode: `calibrated_multi_correlated`
- Recovery bank file: `/tmp/light_doc_cache_tinyllm_calibrated_kv_qwen3_0_6b_20260713_holdout/multi_source_recovery_bank.json`
- Logical stored KV bytes: `1,322,496`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `11.7772`
- Max abs logit diff: `3.10938`
- Mean abs logit diff: `0.455011`
- Argmax match: `True`
- Original argmax: `785`
- Restored argmax: `785`
