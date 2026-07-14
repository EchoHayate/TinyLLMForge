# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `79`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `correlated`
- Correlated source map: `same_layer`
- Logical stored KV bytes: `7,482,880`
- Logical byte saving fraction: `17.41%`
- Missing-token MSE: `6.42021`
- Max abs logit diff: `1.97656`
- Mean abs logit diff: `0.292868`
- Argmax match: `True`
- Original argmax: `594`
- Restored argmax: `594`
