# TinyLLM Sidecar Read-Path Smoke

Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `229`
- KV cache shape: `[2, 28, 773, 256, 8, 128]`
- Recovery mode: `repeat_last`
- Logical stored KV bytes: `21,612,032`
- Logical byte saving fraction: `17.71%`
- Missing-token MSE: `11.8554`
- Max abs logit diff: `5.1875`
- Mean abs logit diff: `0.779818`
- Argmax match: `False`
- Original argmax: `16141`
- Restored argmax: `785`
