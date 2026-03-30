# Token-Level INT8 KV Cache Quantization

## 目标描述
针对用户的高级优化需求，我们在 `tinyvllm` 中引入 **Token 级 KV Cache INT8 量化**。通过将占用极高显存的 KV Cache 压缩一半（FP16 -> INT8），在相同的 VRAM 资源下实现上下文长度或并发数的翻倍。

## 提议的更改方案

### `tinyvllm/config.py`
- 新增 `kv_quantization: str = None` 参数，如果设置为 `"int8"`，则启用 KV 缓存量化。

### `tinyvllm/engine/model_runner.py`
- 修改 `allocate_kv_cache`：
  - 如果 `kv_quantization == "int8"`，计算 `block_bytes` 时改用 1-byte (INT8)，极大拉高 `num_kvcache_blocks`。
  - 初始化 `self.kv_cache` 时将其 `dtype` 设为 `torch.int8`。
  - 额外分配一个缩放因子缓存池 `self.kv_cache_scales`（类型 `torch.float32`），形状为 `[2, num_layers, num_blocks, block_size, num_kv_heads]`（精确到每个 Head 的独立 Scale，保持精度）。
  - 在分配循环中，同步将 `module.k_cache_scale` 和 `module.v_cache_scale` 挂载到每层的注意力模块上。

### `tinyvllm/layers/attention.py`
- 编写一个新的 Triton Kernel `store_kvcache_int8_kernel`（或利用现有时增加分支）：
  - 由于量化粒度是 Per-Token-Per-Head，Kernel 需要按照 `(num_tokens, num_kv_heads)` 的 2D Grid 执行。
  - Kernel 内部计算当前 token 当前 head 维度的最大绝对值。
  - 计算 `Scale = max_abs / 127.0`。
  - 将 FP16 K/V 张量除以 Scale，转化为 `int8` 并存入 `k_cache_ptr` / `v_cache_ptr`。
  - 将求出的 Scale 存入 `k_scale_ptr` / `v_scale_ptr` 的对应 Slot。
- 修改 `store_kvcache` 和 `Attention` 的 `forward`，判断如果传入的 cache 是 int8，则走量化存入逻辑，并将 Scale Tensor 传给 `flash_attn`。

### `tinyvllm/kernels/flash_attention.py`
- 在 `_flash_attn_fwd_kernel` 中增加 `k_scale_ptr` / `v_scale_ptr` 参数。
- 读取 K 和 V 数据后（此时读取到的是 INT8），同步从内存读取对应 Token + Head 的 Scale 因子。
- 在执行点乘（dot）前：`k = k.to(tl.float16) * scale_k`，强制转换为运算精度并复原。
- `flash_decoding_fwd` 同理增加反量化逻辑。

## 验证计划
- 编写测试，固定随机数种子，跑一段长本文 Prompt。分别比较 `kv_quantization="int8"` 和 `None` 时，最终 Logits 的均方误差 (MSE) 和生成的连贯性。
