# TinyLLM Sidecar Storage Smoke

Boundary: materializes a compressed sidecar from ModelRunner.kv_cache and restores into a temporary tensor; no attention hot-path read is changed.

- Model: `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Prompt tokens: `11`
- KV cache shape: `[2, 28, 805, 256, 8, 128]`
- Recovery mode: `linear_tail`
- Sidecar full tensor bytes: `23,634,903,040`
- Sidecar stored tensor bytes: `1,059,328`
- Sidecar saved tensor bytes: `23,633,843,712`
- Sidecar allocated-capacity byte saving fraction: `100.00%`
- Logical full KV bytes for plan seq_len: `1,261,568`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.03%`
- Missing-token MSE: `28.4588`
- Missing-token max abs error: `274`
