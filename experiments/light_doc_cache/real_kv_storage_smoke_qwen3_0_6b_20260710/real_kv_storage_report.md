# Real KV Storage Smoke

Boundary: real HF past_key_values storage/recovery smoke; not wired into TinyLLM runtime hot path.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Input tokens: `12`
- KV cache shape: `[2, 28, 1, 16, 8, 128]`
- Recovery mode: `linear_tail`
- Missing-token MSE: `23.1936`
- Missing-token max abs error: `302`
- Full tensor bytes: `1,835,008`
- Stored tensor bytes: `1,133,568`
- Saved tensor bytes: `701,440`
- Byte saving fraction: `38.23%`
