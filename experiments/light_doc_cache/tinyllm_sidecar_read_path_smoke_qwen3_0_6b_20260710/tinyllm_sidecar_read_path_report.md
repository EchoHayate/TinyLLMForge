# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `13`
- KV cache shape: `[2, 28, 805, 256, 8, 128]`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing-token MSE: `38.586`
- Max abs logit diff: `17.5`
- Mean abs logit diff: `3.13622`
- Argmax match: `False`
- Original argmax: `1815`
- Restored argmax: `50927`
