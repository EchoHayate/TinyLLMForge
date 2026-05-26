# KV 压缩 / 动态稀疏 Attention 工程日志

> 分支：`feat/kv-sparse-attention`
> 起点：`main` @ `e8bce97`（cpu-offload + int8 fused GEMM 之后的稳定态）
> 目标：在保持 accuracy 不掉的前提下显著降低长上下文 decode latency
> 路线：**Phase 1 Quest（动态 page top-k）→ Phase 2 needle 评测 → Phase 3 SnapKV（prompt 阶段 KV 压缩）**

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

### 2026-05-26 起点
- 建分支 `feat/kv-sparse-attention` 自 `main`
- 完成 codebase 调研，确定 5 文件改动面
- 写本文档作为留痕模板，下一步进入 Step A
