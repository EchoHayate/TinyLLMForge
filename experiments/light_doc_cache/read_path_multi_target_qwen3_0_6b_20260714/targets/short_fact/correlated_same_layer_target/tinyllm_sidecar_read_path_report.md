# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `31`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `correlated`
- Correlated source map: `same_layer`
- Logical stored KV bytes: `2,948,608`
- Logical byte saving fraction: `17.07%`
- Missing-token MSE: `6.38993`
- Max abs logit diff: `3.40625`
- Mean abs logit diff: `0.579231`
- Argmax match: `False`
- Original argmax: `785`
- Restored argmax: `16141`
