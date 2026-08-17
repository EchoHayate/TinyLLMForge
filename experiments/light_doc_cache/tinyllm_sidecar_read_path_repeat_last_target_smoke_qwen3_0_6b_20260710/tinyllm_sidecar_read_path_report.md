# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `14`
- KV cache shape: `[2, 28, 748, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `1,322,496`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `15.2284`
- Max abs logit diff: `4.0625`
- Mean abs logit diff: `0.598329`
- Argmax match: `True`
- Original argmax: `785`
- Restored argmax: `785`
