# `tinyvllm` 原生前缀晋升多槽位 LoRA 运行时设计规范（子项目 A）

日期：2026-11-26，Asia/Shanghai。

状态：**设计定稿（承接 `experiments/lora_necessity/` 已完成的双冠军 `8/24` 训练配方与 `Batch=64` `1,095.6 decode tok/s` 系统验证，将该能力作为一等公民特性原生集成进 `tinyvllm/` 推理引擎主干）。**

---

## 1. 目标与核心约束

将“高频共享长 System / Few-shot 前缀 $\rightarrow$ `7.31 MiB` BF16 外挂 LoRA 权重 Cache”正式接入 `tinyvllm`（Paged KV Cache + FlashAttention + `QKVParallelLinear` + CUDA Graph）运行时，满足四条硬指标：

1. **默认零侵入（Zero-Overhead Default）**：当 `Config.prefix_lora_max_slots == 0`（默认）时，不分配任何槽位显存、不改变任何原有 `QKVParallelLinear` 算子路径或 CUDA Graph 行为，所有既有单元测试 100% 保持通过。
2. **入口透明前缀裁剪（Transparent Prefix Promotion）**：`PrefixLoRARegistry` 支持注册 `(long_prefix_token_ids, replacement_short_prefix_token_ids, lora_slot_id)`；调用方无需修改业务层长 Prompt，`LLMEngine.add_request` 精确匹配前缀后自动将 `~3,500` token 请求裁剪为 `~450` token 短请求并绑定 `Sequence.lora_slot_id`。
3. **多租户混合 Batch 并发（Mixed-Batch Multi-LoRA）**：同一 Prefill 或 Decode Batch 内允许同时混跑裸底座请求（`slot = 0` 全零哨兵槽）与多个不同专精领域的 `7.31 MiB` BF16 LoRA 请求（`slot = 1..K`），输出与单序列独立执行逐位/数值等价。
4. **100% 原生兼容 CUDA Graph（Zero-Recapture Graph Replay）**：通过将 `lora_token_slot_ids` 纳入 `ModelRunner.graph_vars` 静态缓冲区并以 `slot = 0` 全零张量作为默认填充，不同请求动态切换任意已装载 `lora_slot_id` 时**无需重录任何 CUDA Graph**，直接复用同一张已捕获图高速回放。

---

## 2. 模块架构与接口边界

### 2.1 新增独立包：`tinyvllm/lora/`

- `tinyvllm/lora/__init__.py` 与 `tinyvllm/lora/prefix_lora_manager.py`：
  1. **`PrefixLoRARegistry`**：
     - `register_promoted_prefix(slot_id: int, long_prefix_ids: Sequence[int], short_prefix_ids: Sequence[int], *, name: str = "")`：校验 `1 <= slot_id <= max_slots`、`len(long_prefix_ids) > len(short_prefix_ids)` 且前缀无歧义冲突，建立基于前缀长度分组的精确元组哈希索引；
     - `match_and_rewrite(prompt_token_ids: list[int], explicit_slot_id: int = 0) -> tuple[list[int], int]`：若 `explicit_slot_id > 0` 直接保留原输入与显式槽位；否则按最长匹配检查 `prompt_token_ids` 头部是否精确命中已注册 `long_prefix_ids`，命中则返回 `list(short_prefix_ids) + prompt_token_ids[len(long_prefix_ids):]` 与对应 `slot_id`，未命中原样返回 `(prompt_token_ids, 0)`。
  2. **`QKVLoRASlotPool`（挂载于每层 `QKVParallelLinear`）**：
     - 预分配连续 GPU BF16 缓冲区（共 `max_slots + 1` 个槽位，其中 **索引 `0` 恒定全零且禁止覆写**）：
       - `lora_a_q`: `[max_slots + 1, rank, hidden_size]`（BF16）
       - `lora_b_q`: `[max_slots + 1, q_shard_size, rank]`（BF16，按当前 `tp_rank` 切片并已预乘缩放因子 `alpha / rank`）
       - `lora_a_v`: `[max_slots + 1, rank, hidden_size]`（BF16）
       - `lora_b_v`: `[max_slots + 1, kv_shard_size, rank]`（BF16，按当前 `tp_rank` 切片并已预乘缩放因子 `alpha / rank`）
     - 旁路前向计算 `apply_qkv_lora(x_flat: torch.Tensor, base_qkv: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor`：
       - 输入展平隐状态 `x_flat`（`[N, hidden_size]`）与 `slot_ids`（`[N]`，`int64`）；
       - 通过 `torch.index_select` 取出每个 token 对应的 `[N, rank, hidden_size]` 与 `[N, out_shard, rank]`，经两次 `torch.bmm`（FP32 累加转回 `base_qkv.dtype`）得到 `delta_q`（`[N, q_shard_size]`）与 `delta_v`（`[N, kv_shard_size]`），原地/非破坏性叠加至 `base_qkv[:, :q_shard_size]` 与 `base_qkv[:, q_shard_size + kv_shard_size:]`（中间 `K` 段保持不变，避免由小幅数值变动破坏 BF16 精度，且规避原地合并的 7-bit 尾数吞噬）。
  3. **`install_qkv_lora_pools(model, max_slots: int, rank: int)` 与 `load_adapter_into_slot(model, slot_id: int, adapter_tensors: dict, *, alpha: float = 16.0, rank: int = 8)`**：
     - 支持直接加载 `experiments/lora_necessity/` 产出的 FP32 或 `7.31 MiB` BF16 字典（含 `model.layers.{i}.self_attn.{q_proj,v_proj}.{adapter_a,adapter_b}`），按 `tp_rank / tp_size` 自动切分写入对应层的 `slot_id`，支持 `clear_lora_slot(model, slot_id)` 热卸载归零。

### 2.2 现有引擎 5 处极薄钩子

1. `tinyvllm/config.py`：新增 `prefix_lora_max_slots: int = 0`、`prefix_lora_rank: int = 8`（校验非负整数）。
2. `tinyvllm/sampling_params.py` 与 `tinyvllm/engine/sequence.py`：
   - `SamplingParams` 新增 `lora_slot_id: int = 0`；
   - `Sequence.__init__` 保存 `self.lora_slot_id = int(getattr(sampling_params, "lora_slot_id", 0))`，并在序列化状态中包含该整数。
3. `tinyvllm/utils/context.py`：
   - `Context` 新增 `lora_token_slot_ids: torch.Tensor | None = None`，`set_context` 支持传入该字段，`reset_context` 自动复位为 `None`。
4. `tinyvllm/layers/linear.py`（`QKVParallelLinear`）：
   - `forward(self, x: torch.Tensor)`：先执行原有 `out = super().forward(x)`；仅当 `getattr(self, "qkv_lora_pool", None) is not None` 且 `get_context().lora_token_slot_ids is not None` 时，调用 `self.qkv_lora_pool.apply(x, out, get_context().lora_token_slot_ids)`。
5. `tinyvllm/engine/model_runner.py` 与 `tinyvllm/engine/llm_engine.py`：
   - 当 `config.prefix_lora_max_slots > 0` 时，`ModelRunner` 初始化后自动对 `self.model` 调用 `install_qkv_lora_pools`；
   - `prepare_prefill` 构造展平 `lora_slot_ids = [seq.lora_slot_id for seq in seqs for _ in range(chunk_end - chunk_start)]`（若存在 mixed-batch decode step 同样逐 token 对齐）送入 `set_context(..., lora_token_slot_ids=...)`；
   - `prepare_decode` 构造 `[seq.lora_slot_id for seq in seqs]` 送入 `set_context(..., lora_token_slot_ids=...)`；
   - `capture_cudagraph`（当 `prefix_lora_max_slots > 0` 时）分配 `lora_token_slot_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")` 加入 `graph_vars` 并传入捕获期 `set_context`，`run_model` 图回放前将 `context.lora_token_slot_ids` 拷入 `graph_vars["lora_token_slot_ids"][:bs]`；
   - `LLMEngine` 初始化 `self.prefix_lora_registry = PrefixLoRARegistry(config.prefix_lora_max_slots)`，暴露 `register_promoted_prefix_lora(...)` 高层接口，并在 `add_request` 编码 token 后自动执行 `match_and_rewrite`。

---

## 3. 错误处理与测试方案

1. **防御性校验**：越界 `slot_id`、尝试修改保留槽位 `0`、不匹配的层数或 `rank`、重复冲突的长前缀注册均立即抛出 `ValueError`。
2. **本地单元测试（`tools/test_prefix_lora_runtime.py`）**：
   - 验证 `PrefixLoRARegistry` 最长前缀精确重写、未命中透传、显式槽位覆盖及异常拦截；
   - 在 CPU/PyTorch 上构造 `QKVParallelLinear` 与多序列混合展平输入（`slot = [0, 1, 2, 0]`），验证混合 Batch 输出与各序列单独挂载原版 `LoRAProjection` 输出在 `1e-2`（BF16）/`1e-5`（FP32）内严格一致，且 `slot = 0` 行与未挂载基线 **逐比特完全一致（`0.0` 误差）**；
   - 回归执行原有 `tools/test_chunked_prefill.py` 与 `tools/test_profile_prefix_cache.py`，确保零回归。
3. **远端单卡 A100 真 `tinyvllm` 引擎端到端验证**：
   - 启动真实 `tinyvllm.LLMEngine`（加载 `Qwen3-8B` BF16，开启 Paged KV Cache + FlashAttention + **CUDA Graph**，`prefix_lora_max_slots=4`）；
   - 将已训好的 `lora_diff_kl`（`slot=1`）与 `lora_csd`（`slot=2`）同时热装载进同一引擎实例并注册 `3,050` token 监督前缀；
   - 验证直接输入原始 `3,500` token 长请求自动触发裁剪 + CUDA Graph 推理，混合并发吞吐、KV 块占用与生成正确性全部实测落盘。
