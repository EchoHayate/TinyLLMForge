# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `31`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `2,948,608`
- Logical byte saving fraction: `17.07%`
- Missing-token MSE: `12.4205`
- Max abs logit diff: `4.8125`
- Mean abs logit diff: `0.903192`
- Argmax match: `False`
- Original argmax: `785`
- Restored argmax: `334`
