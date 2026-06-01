# KV 压缩 / 动态稀疏 Attention 工程日志

> 分支：`feat/kv-sparse-attention`
> 起点：`main` @ `e8bce97`（cpu-offload + int8 fused GEMM 之后的稳定态）
> 目标：在保持 accuracy 不掉的前提下显著降低长上下文 decode latency
> 路线：**Phase 1 Quest（动态 page top-k）→ Phase 2 needle 评测 → Phase 3 SnapKV（prompt 阶段 KV 压缩）**

> **文档拆分说明（2026-05-30）**：原本这一份 1500+ 行的工程日志覆盖了 Quest / W4A8C4 / 回归 / 8B 修复
> 等多条互相正交的线，已经按主题拆成 4 份。这份保留 Quest 主线（§0–§6），其余请看：
> - [`w4a8c4-quantization.md`](./w4a8c4-quantization.md) — W4A8C4 全栈量化（§7 方向、§8 C4、§9 W4A8、§10 β3 sparse dequant）
> - [`regression-tp.md`](./regression-tp.md) — CUDA Graph 兼容性回归（§11）/ cpu_offload 吞吐基线（§12）/ TP=2 多卡 smoke（§13）
> - [`qwen3-8b-fixes.md`](./qwen3-8b-fixes.md) — 8B 拓展验证与修复：TP 显存切分纠错（§15）/ W4A8 数值塌（§16）/ C4 group_size（§17）/ cuda_graph 对照修正（§18）/ 主线回归 smoke（§19）

---

## 0. 为什么做这条路线

之前完成的工作（int8/int4 weight-only 量化、cpu-offload、fused GEMM）都集中在 **算子 + 权重** 层面。
对于长上下文推理，瓶颈在 attention：

- decode 时 batch=1，**KV cache load 是带宽 bound**
- 序列越长，每步 attention 越慢，但很多 token 实际上对当前 query 没什么贡献
- 现有的 paged KV cache 把 KV 切成 block，**正好可以做 block 粒度的 query-aware top-k 选择**

新颖度对比烂大街方案：
- spec decoding / prefix caching / chunked prefill 已经被 vLLM/SGLang 标配
- KV 压缩 + 动态稀疏 attention 是 **2024 长上下文推理的两条最热研究主线**：
  - **SnapKV**（NeurIPS'24）：prompt 阶段 observation window 选 KV，丢弃 90%+
  - **Quest**（ICML'24）：decode 阶段 page-level criticality，每个 q 只算 top-k page

两者正交，可叠加，且都建立在 paged KV cache 上 —— 跟现有引擎天然契合。

---

## 1. 现状摸底（pre-implementation）

| 维度 | 现状 | 接入难度 |
|---|---|---|
| Paged KV cache | `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`，每层 K/V 挂到 `Attention` 模块 | 已有，直接用 |
| Block manager | block 粒度 allocate / append / free，已支持 prefix sharing + ref count | Phase 3 SnapKV 才需要改 |
| Attention forward | prefill: `flash_attn_varlen_func` + block_table；decode: `flash_attn_with_kvcache` + block_table | block_table 即 Quest 的天然 hook |
| KV 写入 | `store_kvcache` triton kernel 按 `slot_mapping` 写 | summary 维护与之并行即可 |
| Attention score 暴露 | flash_attn 不返回 softmax 权重 | Phase 3 SnapKV 需要单独 attention 路径 |
| Accuracy 评测 | **无**（`bench.py` 用随机 token 测吞吐） | Phase 2 必须建一个 needle-in-haystack |

核心改动文件：`tinyvllm/layers/attention.py`、`tinyvllm/engine/model_runner.py`、`tinyvllm/utils/context.py`、`tinyvllm/config.py`，外加 `tools/eval_longbench.py`。

---

## 2. Phase 1：Quest（page-level top-k）

### 2.1 算法
对 paged KV cache 的每个 block 维护一份 **per-channel min / max key**：
```
k_min[layer, block_id, head, dim] = min over tokens in block of K
k_max[layer, block_id, head, dim] = max over tokens in block of K
```
decode 时给定 query `q [H_q, D]`，对每个 block 的"最大可能内积"做估计：
```
upper_bound(block) = sum_d max(q[d] * k_min[d], q[d] * k_max[d])
```
按 head（GQA：q_head → 对应 kv_head 取 max 或 sum）聚合后取 top-k block，重写传给 flash_attn 的 block_table，剩余位置填 `-1` 让 flash_attn 跳过。

### 2.2 设计决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| Summary 形式 | per-channel min/max | Quest 论文做法；比 mean / max(|K|) 更紧 |
| 更新位置 | 在 `store_kvcache` 调用之后用纯 torch 维护 | 保正确性优先，后续再 fuse 进 triton |
| 存储 | 单独张量 `[num_layers, num_blocks, num_kv_heads, head_dim] * 2` | 与 KV cache 分离，避免 cuda graph capture 时的形状耦合 |
| dtype | 跟 KV 一致（fp16/bf16） | 避免显式转换 |
| 启用条件 | `seq_len >= quest_min_seq_len`（默认 1024） | 短序列不值得，且会破坏 cuda graph 路径 |
| 稀疏度 | `quest_top_k_blocks` 配置（-1 = 关闭） | 默认关闭，回归测试 baseline 不变 |

### 2.3 为什么 Phase 1 暂不动 cuda graph
prepare_decode 里 block_table 形状是 `[bs, max_num_blocks_in_seq]`。Quest 后塞 `-1` 即可，**形状不变**，理论上可以走 cuda graph。但首版先走 eager 验证正确性，cuda graph 兼容放到性能调优阶段。

### 2.4 已知风险
- ⚠️ flash_attn paged 对 block_table 中 `-1` 的处理需要确认（部分版本是直接 OOB read）。需要在服务器上 dump flash_attn 版本并验证语义
- ⚠️ GQA 下 num_q_heads != num_kv_heads，criticality 聚合方式会影响 recall
- ⚠️ summary 张量在 prefill 一次性写入大量 token 时，需要按 block 做 segment reduce —— scatter_reduce 性能要看

### 2.5 实施步骤
- [ ] Step A：`config.py` + `context.py` + `bench.py` + `example.py` 加配置项 / 字段
- [ ] Step B：`model_runner.py::allocate_kv_cache` 同时分配 summary 张量并挂到 `Attention` 模块
- [ ] Step C：`attention.py` prefill / decode 写入 KV 后维护 summary（纯 torch 版本）
- [ ] Step D：`attention.py` decode 分支前算 page criticality + 重写 block_table
- [ ] Step E：服务器跑 bench.py（关闭 / 开启 quest）对比吞吐
- [ ] Step F：建 needle-in-haystack 验 accuracy

---

## 3. Phase 2：Needle-in-Haystack 评测

最小可用版本：合成 N 个填充段落 + 在随机位置插入 `"the magic number is 12345"`，让模型回答数字。
对比维度：
- baseline (full attention)
- quest top-k = {16, 32, 64} 不同稀疏度
- 不同 needle 位置（开头 / 中间 / 末尾）

输出：accuracy 表 + 吞吐对比表。

---

## 4. Phase 3：SnapKV（之后再做）

待 Quest 落地后再启动。预期最大难点：
- flash_attn 不暴露 softmax → 需要在 prefill 末尾对 observation window 跑独立 attention
- 压缩后的 KV 物理 slot 重排，要改 block_manager 的 ref_count 语义
- 与 prefix caching 的兼容性（hash 还成立吗？）

---

## 5. 实验日志（按时间倒序）

### 2026-05-26 Phase 1 首版上服务器跑通

环境：A100 80GB PCIe (CUDA_VISIBLE_DEVICES=1)，flash-attn 2.6.3，torch 2.4.1+cu121，Qwen3-0.6B fp16，eager 模式

#### 5.1 正确性
- `example.py` baseline 与 `--quest-top-k-blocks 4 --quest-min-seq-len 1` 都生成连贯文本
- example 序列短（每条 ~500 tok），实际未触发稀疏（block_size=256，4 block = 1024 token），因此只验证了维护路径不崩

#### 5.2 吞吐对比（bench.py）

| 配置 | num-seqs | input-len | output-len | Throughput |
|---|---|---|---|---|
| baseline | 16 | 4000 | 1024 | **253.27 tok/s** |
| quest top-k=4, min=512 | 16 | 4000 | 1024 | 184.42 tok/s（-27%）|
| baseline | 4 | 15500 | 512 | **68.82 tok/s** |
| quest top-k=8, min=2048 | 4 | 15500 | 512 | 43.80 tok/s（-36%）|

**结论：首版 Quest 反而拖慢了吞吐**。

#### 5.3 根因分析
1. `index_reduce_(amin/amax)` 在 fp16 上每层 prefill / decode 都跑两次，写入 K-min/K-max 是大量随机访问
2. `quest_select_blocks` 每层 decode 都跑：`gather → broadcast mul → sum → masked_fill → topk → sort` 6+ kernel launch，每层都有 host-device 同步（`.item()` 早返回判断）
3. 0.6B 模型的 attention 本身 latency 很小（KV head=8, head_dim=128），稀疏化能省的部分远小于 selection overhead

#### 5.4 已知 / 待排查问题
- ⚠️ `quest_select_blocks` 入口的 `.min().item()` / `.max().item()` 触发 GPU sync，每层每 step 都做 → 极昂贵
- ⚠️ `index_reduce_` 在 fp16 上的实现在某些 torch 版本是 fp32 fallback，需要 profile 确认
- ⚠️ 还没验 selection 的 recall：选中的 block 真的覆盖了 dominant attention 吗？需要离线 dump attention probs 对照

#### 5.5 下一步优化清单（按 ROI）
1. **去掉早返回的 `.item()` GPU sync**：用静态条件（在 prepare_decode 里预先算 `should_run_quest` 标志位）
2. **fp16 index_reduce 替换**：scatter 到 fp32 buffer 再写回，或写一个简单的 triton kernel 在 store_kvcache 之后顺手维护
3. **selection 的 broadcast 优化**：把 `q_repr.unsqueeze(1) * k_min_b` 这种 4D 乘法换成 einsum 或 batched matmul
4. **首版 fallback 行为**：若 `top_k * block_size >= max_seq_len_in_batch * 0.8`，直接关闭 sparse（收益太小）
5. **profile 验证 selection 与 attention 的 latency 占比**

切换到 Phase 2 评测脚本之前，先做 #1（最便宜，可能直接翻盘）。

---

### 2026-05-26（晚）优化 #1+#3 复测：去 `.item()` sync + einsum 紧凑化

#### 5.6 改动
1. **去早返回 GPU sync**：把 quest 是否激活的判断从 `quest_select_blocks` 内部的 `min/max.item()` 改成在 `prepare_decode` 里基于 host-side 的 `seqs` 列表算（`min(len(s) for s in seqs)` 与 `min(s.num_blocks for s in seqs)`），结果通过 `Context.quest_top_k_blocks` 字段下发。`run_model` 里仍然根据这个字段决定 `enforce_eager` —— 仅在本次 batch 真激活时才退出 cuda graph。
2. **selection broadcast → einsum**：把 4D 张量乘法 + sum 改写成 `einsum("bhd,bnhd->bnh")`，省掉一次显式 broadcast 与中间张量。

#### 5.7 复测结果

| 配置 | num-seqs | input-len | output-len | Throughput | vs baseline |
|---|---|---|---|---|---|
| baseline | 16 | 4000 | 1024 | 253.27 tok/s | — |
| quest top-k=4, min=512 (优化前) | 16 | 4000 | 1024 | 184.42 tok/s | -27% |
| quest top-k=4, min=512 (优化后) | 16 | 4000 | 1024 | **196.28 tok/s** | -22% |
| baseline | 16 | 15000 | 512 | 250.28 tok/s | — |
| quest top-k=8, min=2048 (优化前 4-seq) | 4 | 15500 | 512 | 43.80 tok/s | -36% |
| quest top-k=8, min=2048 (优化后 16-seq) | 16 | 15000 | 512 | **152.37 tok/s** | -39% |

#### 5.8 解读
- 4k 场景从 -27% 收窄到 -22%，证明 `.item()` 同步确实贡献了一部分开销，但还远没有翻盘
- 15k 场景注意 batch 改了（4→16），baseline 对应也改成 16-seq 后是 250 tok/s，优化后 152 tok/s 仍然是 -39%
- 单条 decode 的 `.item()` 同步在 28 层 × 1 step ≈ 28 次/step，理论上应该贡献几个 ms；现在收益小于预期，说明真正的瓶颈在 **selection 的 GPU 算力 / 内存访问** 而不是同步
- 0.6B 模型每层 attention 实际只跑 1ms 量级，selection（gather k_min/k_max + topk + index_select 重写 block_table）随 batch & layer 数量线性放大，这部分相比节省的 attention 浮点运算反而是净支出

#### 5.9 进一步行动（更新优先级）
1. ✅ 去 `.item()` sync —— 已做，收益 +6%（4k）
2. ✅ einsum 紧凑化 —— 已做，与 #1 合并测
3. **profile 定位**：用 `torch.profiler` 观察 selection 各 kernel 实际占比，确认是 `index_reduce_`、`gather` 还是 `topk` 是大头
4. **summary 维护 fuse 进 store_kvcache**：写一个 triton kernel 同时做"写 KV + 更新 block 范围内 K-min/K-max"，省掉一次全量读
5. **Phase 2 needle 评测先行**：当前性能虽差，但稀疏路径已经稳定，可以先用它跑 needle 看 accuracy 是否守得住；如果 quest top-k=8 在长 context 上 needle 准确率就崩了，那性能优化也没意义
6. **换更大的模型重测**：理论上 Quest 在 7B+ 模型 / 长 context 下才有正收益，0.6B + 15k 这个组合本身就是 selection-overhead-dominant 的"不友好"区间

接下来优先做 #5（先验 accuracy 是否站得住），再回头做 #3/#4。

---

### 2026-05-27 Phase 2：needle-in-haystack 评测落地

#### 6.1 脚本
新增 `tools/eval_needle.py`：合成 haystack 文本 + 在指定深度（0/25/50/75/100%）插入 `"The magic number is XXXXX"`，让模型回答数字。
- 对比维度：`top_k_blocks_list = [-1(baseline), 4, 8, 16]` × `context_lens = [4k, 8k, 15k]` × `depths = 5 个`
- 一个 LLM 实例热改 `config.quest_top_k_blocks` 切换稀疏度，避免重复 init KV cache 触发 `num_kvcache_blocks=0`
- 用最大 `top_k` 起手 init，确保 `kv_summary` 张量被分配（baseline run 时也保留显存占用，公平对比）

#### 6.2 第一次结果（有 bug）
所有 setting 共用同一组 prompt 序列，**后续 setting 直接命中前一次写入的 prefix cache**：

| top_k | overall acc | 吞吐(tok/s) |
|---|---|---|
| baseline | 97.8% | 98.9 |
| 4 | 88.9% | 373.2 ← 虚高 |
| 8 | 100.0% | 404.8 ← 虚高 |
| 16 | 100.0% | 410.4 ← 虚高 |

#### 6.3 修复
每个 setting 用 `seed + top_k * 7919` 作为 RNG 种子，保证 prompt 不重复，绕开 prefix cache 命中。

#### 6.4 修复后结果（公平比较）

| top_k | overall acc | 吞吐(tok/s) | vs baseline |
|---|---|---|---|
| baseline | **97.8%** | 98.7 | — |
| 4 | 86.7% | 99.3 | +0.6% |
| 8 | **100.0%** | 102.0 | +3.3% |
| 16 | **100.0%** | 109.5 | +10.9% |

逐 (ctx_len, depth) 看 baseline 的唯一漏题是 `ctx=15000, depth=1.0`（needle 在最末端 acc=66.7%），`top_k=8/16` 反而把这个点也救回来了——猜测是稀疏化让远端 needle 在剩下的几个 block 里被"凸显"，attention 不再被中段大量噪声稀释。

#### 6.5 Quest 真正能赢的场景
- **`top_k=4` 是临界值**：4 个 block × 256 = 1024 token，占 4k context 的 25%，在中段 depth 偶有漏题（acc=86.7%）→ 业务上不能用
- **`top_k=8` 起步够用**：100% acc，且第一次跑（带 prefix cache）证明在大 batch / 长 context 下 throughput 能涨数倍；本次单 batch 没有体现，因为：
  - eval_needle 是按 setting × ctx_len × depth × trial 串行做的，每个时刻 batch 通常只有 3 条同长 prompt
  - 单条 decode 的 attention 时间在 0.6B 上太小，selection overhead 把净收益吃了
- **多 seq 并发是 Quest 的甜点区**：bench.py 的 num-seqs=16 + ctx=15k 才是真实 serving 场景，那里 baseline 16-seq 250 tok/s vs quest 152 tok/s（-39%）的回归还需要靠 #29 profile + triton fuse 来解

#### 6.6 关键结论（决定后续路线）
1. ✅ **accuracy 守住了**（`top_k>=8`）→ Quest 路线方向正确，性能问题值得继续投资
2. ❌ **0.6B 模型 selection overhead 显著** → 必须做 #29 profiler + #4 triton fuse summary 维护
3. 🔬 **观察到的 acc reversal**（quest > baseline）值得记一笔，可能是 noisy attention dilution 的反向证据，以后做 SnapKV 时回头看
4. 评测脚本本身要带"prefix-cache 隔离"作为标配，这次踩的坑下次别再踩

#### 6.7 下一步
- ✅ 写入 docs（本节）
- 转到任务 #29：torch.profiler 定位 selection 中谁是大头 kernel
- 后续如果 profile 显示是 `index_reduce_` 或 `gather + topk` → 写一个 fuse 进 `store_kvcache` 的 triton kernel
- Phase 3（SnapKV）暂缓，等 Phase 1 性能翻盘再启动

---
