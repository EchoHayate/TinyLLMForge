# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `36`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `3,400,704`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `19.1593`
- Max abs logit diff: `3.20312`
- Mean abs logit diff: `0.523985`
- Argmax match: `True`
- Original argmax: `785`
- Restored argmax: `785`
