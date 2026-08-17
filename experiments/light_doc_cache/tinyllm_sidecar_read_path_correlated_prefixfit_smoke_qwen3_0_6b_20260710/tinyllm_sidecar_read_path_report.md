# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `13`
- KV cache shape: `[2, 28, 503, 256, 8, 128]`
- Recovery mode: `correlated`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing-token MSE: `13.2058`
- Max abs logit diff: `6.9375`
- Mean abs logit diff: `1.10957`
- Argmax match: `False`
- Original argmax: `1815`
- Restored argmax: `13173`
