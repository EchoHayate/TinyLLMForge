# 子项目 A 验收报告：`tinyvllm` 原生透明前缀裁剪与多槽位 BF16 LoRA 运行时

## 1. 交付目标与架构概览

将实验验证成熟的 **`7.31 MiB` BF16 权重缓存（Parametric Prefix Cache）** 原生集成进 `tinyvllm` 推理引擎主干（含 `Paged KV Cache`、`FlashAttention` 与 `CUDA Graph` 捕获/回放），对上层调用方提供零侵入透明体验：

1. **入口透明最长前缀匹配与裁剪（`PrefixLoRARegistry`）**：
   - `LLMEngine.register_promoted_prefix_lora(...)` 注册高频长前缀（如 `3,090` tokens 的 System + Few-shot 示范）及其短前缀替代（`40` tokens）和目标槽位 `slot_id`。
   - 调用方照常向 `LLMEngine.add_request(...)` 传入原始 `3,400+` token 长提示词（无需改业务代码）；引擎入口 O(L) 最长前缀匹配命中后，在分配 Paged KV Block 前自动将其裁剪为 `~350` token 短输入并绑定 `Sequence.lora_slot_id`。
2. **多槽位 BF16 权重池与无分支混合 Batch 旁路（`QKVLoRASlotPool`）**：
   - 在全部 `36` 层 `QKVParallelLinear` 上预分配槽位池 `[max_slots + 1, ...]`，其中 **槽位 `0` 恒定全零（代表裸底座直通）**。
   - 混合 Batch（同时包含裸底座请求、不同租户/配方的 LoRA 请求）在同一次 Prefill / Mixed / Decode 前向中，通过 `index_select + torch.bmm` 一次性无 Python 分支完成按 token 路由叠加。
3. **CUDA Graph 原生兼容**：
   - `ModelRunner.capture_cudagraph` 为 `[1, 2, 4, 8, 16, 32, 48, 64]` 全档位静态绑定 `graph_vars["lora_token_slot_ids"]` 缓冲区；Decode 回放前仅需原地 `.fill_(0)` 与写入当前 Batch 的 `seq.lora_slot_id`，零重捕获、零图外分支。

---

## 2. 本地单测与远端 A100 真实引擎端到端验收

- **本地单测**：`tools/test_prefix_lora_runtime.py`（4/4 通过）与 `tools/test_profile_prefix_cache.py`（通过）。
- **远端真实引擎验收配置**：单张空闲 `NVIDIA A100 80GB PCIe`（`free_mib = 80,979`），启动原生 `tinyvllm.LLMEngine`（`enforce_eager=False`，捕获 8 档 CUDA Graph：`[1, 2, 4, 8, 16, 32, 48, 64]`，`prefix_lora_max_slots=4, prefix_lora_rank=8`），同时挂载两块已封存的真实适配器：
  - `slot=1`：`lora_diff_kl_v2`（Gate G8 差异聚焦配方，`36` 层常驻 **`7,667,712 B` = `7.3125 MiB`**）并绑定 `3,090 -> 40` token 自动前缀重写规则；
  - `slot=2`：`lora_csd_v2`（Gate G9 上下文自蒸馏零退化配方，`36` 层常驻 **`7,667,712 B` = `7.3125 MiB`**）。

### 2.1 异构 6 请求混合 Batch 等价性与透明裁剪核验

在同一轮混合 Batch 中并发调度 6 条分属 3 种模式的真实任务，并与各自独立单请求 CUDA Graph 运行结果逐字符比对：

| 任务 ID | 接入模式 | 原始输入 Tokens | 引擎重写后 Tokens | 解析后 `lora_slot_id` | 初始分配 KV Blocks (`block_size=256`) | 与单请求 CUDA Graph 输出逐字符一致 | 首轮 (`initial`) 8 用例沙箱通过数 |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: |
| `WR03` | 裸底座直通 (`base`) | `365` | `365` | `0`（全零哨兵） | `2` | **True** | `8 / 8` |
| `WR04` | 裸底座直通 (`base`) | `229` | `229` | `0`（全零哨兵） | `1` | **True** | `8 / 8` |
| `WT05` | **传长 Prompt 自动识别裁剪** (`text_auto_promoted`) | **`3,426`** | **`376`** | **`1` (`lora_diff_kl`)** | **`2`**（原需 `14` 块） | **True** | **`8 / 8`** |
| `WT06` | **传长 Prompt 自动识别裁剪** (`text_auto_promoted`) | **`3,396`** | **`346`** | **`1` (`lora_diff_kl`)** | **`2`**（原需 `14` 块） | **True** | **`8 / 8`** |
| `WR11` | 显式指定槽位 2 (`explicit_slot2`) | `308` | `308` | `2` (`lora_csd`) | `2` | **True** | `8 / 8` |
| `WT01` | 显式指定槽位 2 (`explicit_slot2`) | `381` | `381` | `2` (`lora_csd`) | `2` | **True** | `3 / 8`（与 G9 首轮一致，二轮修复） |

- **混合 Batch 隔离与数值一致性**：`6 / 6` 条异构请求在单次混合 Batch + CUDA Graph 回放下的解码输出与逐条单跑 **100% 逐字符完全一致**（槽位 `0` 裸底座零污染，槽位 `1` 与槽位 `2` 互不串扰）。

### 2.2 真 `Paged KV Cache + FlashAttention + CUDA Graph` 并发吞吐与块占用实测（`Batch = 16`）

在同一个 `tinyvllm.LLMEngine` 内，用 16 道真实评测题（`WT01..WT12` + `WR01..WR04`，`max_tokens=128`，预热后取中位数）正面对比 **`tinyvllm` 原生块共享 Warm Automatic Prefix Caching (APC)** 与 **透明晋升 BF16 LoRA（含每步 CUDA Graph 内 `bmm` 旁路开销）**：

| 引擎服务形态 (`Batch=16`, FlashAttention + CUDA Graph) | 峰值占用 Paged KV Blocks | 峰值占用 Paged KV 显存 | 中位端到端耗时 | 实际输出吞吐 (`output tok/s`) |
| :--- | ---: | ---: | ---: | ---: |
| **原生预热块共享 Prefix Cache (`Warm APC`)** | `45` 块 | `1,620.0 MiB` | `7,423.1 ms` | `267.41 tok/s` |
| **透明前缀晋升 `7.31 MiB` BF16 LoRA (`Promoted Slot=1`)** | **`33` 块 (`-26.7%`)** | **`1,188.0 MiB` (`-432.0 MiB`)** | **`7,019.7 ms`** | **`279.93 tok/s` (`1.047x`)** |

- 即使在 `tinyvllm` 原生 Paged Block 物理共享（16 条请求共享同一份 12 块前缀 KV）且开启 FlashAttention 与 CUDA Graph 的最强基线下，透明晋升 BF16 LoRA 依然**净省下全部 12 个共享前缀块（单租户省 `432 MiB`，`K` 租户即省 `432 * K MiB`）**，并且算上旁路 `torch.bmm` 开销后端到端解码吞吐仍**反超原生 Warm APC `+4.68%`（`1.047x`）**，彻底免除了 Decode 阶段扫描 3,090 个前缀 token 的显存带宽开销。
