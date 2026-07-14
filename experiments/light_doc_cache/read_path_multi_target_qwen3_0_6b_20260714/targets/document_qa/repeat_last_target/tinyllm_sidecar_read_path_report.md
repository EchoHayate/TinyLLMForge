# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `212`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `20,026,368`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `10.1103`
- Max abs logit diff: `4.5625`
- Mean abs logit diff: `0.755026`
- Argmax match: `False`
- Original argmax: `785`
- Restored argmax: `32`
