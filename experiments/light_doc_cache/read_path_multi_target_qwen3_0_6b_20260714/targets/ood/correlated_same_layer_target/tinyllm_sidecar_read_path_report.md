# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `52`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `correlated`
- Correlated source map: `same_layer`
- Logical stored KV bytes: `4,912,128`
- Logical byte saving fraction: `17.63%`
- Missing-token MSE: `7.20804`
- Max abs logit diff: `3.6875`
- Mean abs logit diff: `0.566026`
- Argmax match: `False`
- Original argmax: `4226`
- Restored argmax: `4396`
