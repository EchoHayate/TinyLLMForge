# 端到端稀疏 decode 门：选择器保住了答案，但没有保住轨迹

日期：2026-09-17
分支：`feat/kv-sparse-attention`
产物：`tools/e2e_sparse_attention.py`、`tools/test_e2e_sparse_attention.py`、
`tools/run_e2e_sparse_remote.sh`、`experiments/e2e_sparse_attention/e2e-sparse-margin-20260917-003529/`

## 为什么要有这一步

上一份报告（`2026-09-16-selector-fidelity-gate.md`）在选择器层面给出了一个很锐的结论：
32-token 粒度下，仓库里在跑的 shared-head Quest 排名在 k/L≈5.1% 就能把答案 token 全部留住。
但那是**必要条件，不是充分条件**，它有三个说不清的地方：

1. coverage 只在 prompt 最后一个位置量过，而 decode 每步 query 都在动、选中集合每步都重选；
2. 只量了 6 个采样层，真实运行要稀疏化全部 36 层；
3. 留住答案 token，不等于留住了指令、格式、以及答案周围的句法框架。

所以这一轮把回路闭上：**prefill 走 dense，decode 的每一层每一步都做 query-aware 选择**，
然后和同一个 prompt 的 dense 续写逐 token 对比。坏臂（recency / sink_recency / random）
在**完全相同的预算**下跑，因为所有臂都能过的门不是门。

一个必须先说清的工程事实：**当前引擎跑不了 gran=32**。
`tinyvllm/config.py` 里 `assert self.kvcache_block_size % 256 == 0`（flash-attn paged-KV 的约束），
所以这个 harness 用「把选中的 32-token 单元 gather 成连续 buffer 再算 attention」的方式实现 32 粒度。
这不是权宜之计：CPU offload 路径本来就必须这么做（CPU 把选中单元 gather 到 staging buffer 再算），
所以这里量的就是我们打算造的那个东西。

## 怎么量的

- 模型：Qwen3-8B（真权重，远端 A100 80GB PCIe，`CUDA_VISIBLE_DEVICES=2`）
- prompt：8192 token，三种 needle 构造（repetitive 控制组 / natural 词汇分散 / distractor 多个同形干扰项）
- 选择数学：复刻引擎的 `quest_score_kernel`——把 per-channel min/max 上界在**所有 kv head 上求和**成一个共享排名；
  GQA 组内 query 取 amax；强制保留第 0 个和最后一个单元
- 所有臂（包括坏臂）都强制保留最后一个单元：它装着当前 token 自己的 key，丢掉它是 kernel bug 而不是选择策略
- layer 0 保持 dense（上一轮量出它没有可用的稀疏结构）
- prefill 只跑一次，KV 快照 clone 给每个臂，所以臂之间的差异只来自 decode 选择
- 每步记录 dense 自己的 top-2 logit 间距，用来**归因** token 分歧，而不是靠猜

单测 22 个通过，其中包含「门必须能判 NON_DISCRIMINATIVE / INVALID_PROMPT」的用例。

## 结果

三个 variant 的判决都是 **DISCRIMINATIVE**：所有坏臂在所有格子里都答错（token 一致率 0.03–0.30），
仓库在跑的 shared-head 选择器答对。

### 粒度是决定性旋钮，端到端复现了选择器层面的预测

「shipped 选择器答对所需的实际读取比例」（touched_frac，含强制单元与末尾残缺单元，不是名义 k/L）：

| variant | gran=32 答对 | gran=256 答对 | 倍数 |
|---|---|---|---|
| repetitive | 2.15% | 25.2% | 11.7× |
| natural | 3.32% | 25.2% | 7.6× |
| distractor | 2.15% | 25.2% | 11.7× |

gran=256 在 k/L≤5% 的格子里**全部答错**，而且此时强制的 sink+末尾两个单元本身就已经吃掉 3.1% 预算——
粗块把预算浪费在强制单元上。**引擎当前的 256 块配置，在这条轴上离最优操作点差了约 8–12×。**

### 答案对了，轨迹没对，这是两个不同的门

| variant | 答对所需 touched | token 逐位相同所需 touched |
|---|---|---|
| repetitive | 2.15% | 3.32% |
| natural | 3.32% | 25% 也做不到 |
| distractor | 2.15% | 11.1% |

用 dense 自己的 top-2 logit 间距做归因，分歧位置的性质很清楚：

```
natural     good-arm 首次分歧 index=13，dense 在该步 top-2 间距 = 0.125 logits（该 prompt 间距中位数 8.88）
distractor  good-arm 首次分歧 index=16，dense 在该步 top-2 间距 = 0.125 logits（中位数 9.88）
repetitive  good-arm 首次分歧 index=11，dense 在该步 top-2 间距 = 2.25 logits
```

也就是说：**多数分歧发生在 dense 自己几乎无所谓的近平局步上**，答案本身（前 11–16 个 token）是逐位一致的。
但有两个反例必须记下来：

- distractor 在 gran=32、k/L=0.02 时首次分歧在 index=14，dense 该步间距 **8.0 logits**，是真分歧
  （答案出现在更早的位置，所以仍判答对）——低预算下确实会改变模型的真实判断；
- distractor 在 gran=32 时，k/L=0.11 逐位全同（agreement=1.00），而预算更大的 k/L=0.25 反而在 index=24 分歧
  （agreement=0.73）。**预算变大不保证更接近 dense**——选中集合是离散跳变的，整条轨迹可以翻。

结论：**不要再说「无损」**。可以说的是：32 粒度下 query-aware 稀疏 decode 在读 ~3% 上下文时保住答案，
但不复现 dense 的逐 token 轨迹，且轨迹一致性对预算非单调。

### 和 CPU 带宽门对账（这一轮把结论收紧了）

CPU gather 门（Xeon 8336C，interleave=all，128 KiB 粒度 RAND ≈110 GB/s）给出的窗口是：
GPU 读权重的 `c0≈12.1 ms` 里，CPU 大约能搬 ~900 个 bf16 token/step。

| 保真度要求 | 每步需要的 token（8k 上下文） | 相对 ~900 的余量 |
|---|---|---|
| 答案正确即可 | 3.32% × 8192 ≈ 272 | ≈ 3.3× |
| 逐 token 复现 dense（能做到时） | 11.1% × 8192 ≈ 910 | ≈ 1.0×，没有余量 |

上一轮基于 attention mass 估的「约 2× 余量」现在被拆成两个数：
**如果产品接受答案级保真，可行域是舒服的；如果要求轨迹一致，8k 上下文下就是贴边，没有余量。**
这个区别以前被 mass 指标糊掉了。

### per-head 与 shared-head：继续留 shared

- 端到端两者互有胜负：distractor gran=32 / k=0.02 只有 shared 过；repetitive gran=256 / k=0.11 只有 per-head 过；
- per-head 在这个 harness 里 wall-clock 慢 2.1–2.3×（3.97s vs 1.87s / 32 步），因为每个 kv head 要各自 gather 一套；
- 结论不变：**不要把仓库实现「修」成论文式 per-head**。它更贵，端到端没有系统性优势。

## 这一轮改变了什么决定

1. **32-token 选择粒度从「建议」升级为「必须」**：256 块要 25% 上下文，32 单元只要 2–3%，端到端确认。
   引擎侧要么放开 `kvcache_block_size % 256 == 0`，要么走 gather-to-staging 路径（CPU offload 天然是后者）。
2. **验收标准要写成两级**：答案级保真（~3%）和轨迹级保真（~11%，且不保证）。
   后面所有 CPU offload 的时间预算必须声明自己针对哪一级。
3. **token 一致率要配 dense top-2 间距一起看**，否则会把模型自己的近平局记成选择器的错。
4. layer 0 保持 dense 的决定在端到端下继续有效（没有为它单独消耗预算）。

## 还没证的东西（不要提前当成结论）

- 只有单 needle、depth=0.5、8k 上下文、一个模型、一次 seed。多 needle / 多深度 / 32k 还没扫。
- 32 粒度的选择器**开销**没算进任何时间预算：本 harness 每步每层重算 min/max，引擎里是增量维护，
  但 32 粒度意味着 8× 的单元数、8× 的 top-k 规模，这笔钱还没量。
- CPU 侧还没有真的算过一次 attention：能不能藏进 12.1 ms 窗口，是下一道门（overlap 微基准），
  在那之前「省显存且不变慢」仍然只是一个尚未被证伪的假设。

## 复现

```bash
RUN_TAG=e2e-sparse-$(date +%Y%m%d-%H%M%S) SEQ_LEN=8192 \
VARIANTS="natural distractor repetitive" GRANULARITIES="32 256" \
K_FRACS="0.02 0.035 0.051 0.11 0.25" \
ARMS="quest_shared_heads quest_per_head recency sink_recency random" \
MAX_NEW_TOKENS=32 bash tools/run_e2e_sparse_remote.sh
```

宿主污染已记录：load average ≈98–106，30 个用户在线。本轮只量正确性与 token 一致率，
唯一受污染影响的数字是 wall-clock（per-head 的 2.1× 慢只能当量级参考，不能当性能结论）。
