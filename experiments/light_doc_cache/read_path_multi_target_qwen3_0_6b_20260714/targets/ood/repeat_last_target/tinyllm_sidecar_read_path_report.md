# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `52`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `4,912,128`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `11.331`
- Max abs logit diff: `4.1875`
- Mean abs logit diff: `0.577385`
- Argmax match: `True`
- Original argmax: `4226`
- Restored argmax: `4226`
