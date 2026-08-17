# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `13`
- KV cache shape: `[2, 28, 503, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing-token MSE: `13.7399`
- Max abs logit diff: `5.5625`
- Mean abs logit diff: `0.787285`
- Argmax match: `False`
- Original argmax: `1815`
- Restored argmax: `3491`
