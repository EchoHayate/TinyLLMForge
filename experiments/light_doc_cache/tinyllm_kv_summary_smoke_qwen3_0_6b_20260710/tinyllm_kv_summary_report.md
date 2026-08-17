# TinyLLM KV Summary Smoke

Boundary: reads TinyLLM ModelRunner.kv_cache allocation and planned Light Doc Cache accounting only; no runtime compression is applied.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `10`
- KV cache shape: `[2, 28, 805, 256, 8, 128]`
- Allocated KV cache bytes: `23,634,903,040`
- Logical full KV bytes for plan seq_len: `1,146,880`
- Planned recovered KV bytes: `202,240`
- Planned stored KV bytes: `944,640`
- Planned byte saving fraction: `17.63%`
- Planned compression ratio: `1.2141x`
