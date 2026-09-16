# 选择器保真度门：query-aware 选择器到底保住了什么

日期：2026-09-16
作者：吴思天
工具：`tools/kv_selector_fidelity.py`、`tools/dump_needle_qk.py`、`tools/needle_haystack_variants.py`
原始数据：`experiments/selector_fidelity/selector-fidelity-20260916-2248/`
模型：Qwen3-8B（真实权重，post-RoPE 的真 Q/K，非合成）

## 为什么必须先建这个门

CPU offload + query-aware attention 的带宽门已经过了：32-token 块（128 KiB）散乱 gather 在
Xeon 8336C 上能拿到 110 GB/s，`interleave=all`，8 物理核就够。预算换算出来是每步只碰
k≈900 个 token（k/L≈11%，bf16）。

**也就是说要扔掉 89% 的上下文。** 而此前唯一的质量门（`tools/eval_needle.py`）给每个臂都打
100.0% 满分，包括我们预期很差的臂。**一个不可能失败的门，不能给任何决定背书。**

所以这个门的设计目标不是"证明方案可行"，而是**能失败**：把已知坏臂（random / 只留最近 /
均匀抽样）和真选择器放在同一把尺子下，如果坏臂也一样好，判决直接是 `NON_DISCRIMINATIVE`，
数字作废。

## 方法

度量在选择器层面，不经过模型输出，因此不会被"任务本身好做"救回来：

| 指标 | 含义 |
|---|---|
| recovered mass | 被选中 token 集合内的真实 softmax 概率之和 |
| **needle coverage** | **真正携带答案的 token 有没有被保住** |
| recall@32 | 真 top-32 token 的留存比例 |

七个臂，其中四个是故意做坏的对照：

- `oracle_kvhead`：按真实概率上限选块，块选择器的理论上界
- `quest_per_head`：per-channel min/max 上界，**按 kv head 各自选**（Quest 论文写法）
- `quest_shared_heads`：**仓库真实实现**——`quest_score_kernel` 把上界在所有 kv head 和通道上
  求和，得到**所有 head 共享的一个排名**
- 坏臂：`recency`（只留最近）、`sink_recency`（只留首块+尾部）、`uniform_stride`、`random`

三种 haystack：`repetitive`（现有构造，作为对照）、`natural`（词汇多样填充）、
`distractor`（4 个同形式诱饵 needle，问题指名要哪一个）。

单元测试 16 个，其中两个专门测**门能判 NON_DISCRIMINATIVE**（合成张量只用于验证估计器本身
算的是它声称的东西，测量路径里没有任何合成注意力）。

## 结果一：真正瞎的不是 prompt，是度量

我原本的假设是"needle 任务太退化所以门失效"。**这个假设只对了一半，而且不是主要原因。**

```
gran=32, layer=7, k/L=50%：
  sink_recency（只留首块 + 尾部）  recovered mass = 0.974
  同一个臂                          answer coverage = 0.000
```

**只留注意力 sink 和最近块，就能保住 97% 的注意力质量，同时把答案 100% 丢掉。** 三个 variant、
所有 layer≥7 都是这个形状。原因是 attention sink：质量绝大部分压在首块和近处，答案 token 只
占极小一份概率，但那一份决定输出。

直接后果，也是本轮最有价值的一条：

- **以"保留了多少 attention mass"作为 KV 稀疏化的验收指标是错的**——这是文献和工程里非常常见
  的做法，而它可以在 97% 保真的同时答错。
- `oracle_kvhead`（按质量选的"理论最优"）在 coverage 上**反而不如**真选择器（0.475–0.825 vs
  1.000）。**质量最优 ≠ 任务最优**，这不是措辞问题，是可测的 30–50pp 差距。

对照证据：90 个 cell 里，mass 判据只有 6–20 个 cell 有分辨力，coverage 判据有 44–53 个。
所以判据已改为 coverage 优先，mass 降级为诊断项。

## 结果二：仓库自己的实现比论文写法更强（反直觉）

layers≥7 上，coverage 达到 1.000 所需的最小 k/L：

| 选择粒度 | 仓库实现（共享排名） | per-head（论文写法） |
|---|---|---|
| 256 token（= 当前 KV block） | 25.0%（2048 token） | **从未达到** |
| 64 token | 4.7 – 10.9% | 50%（4096 token） |
| **32 token** | **5.1%（416 token）** | 50%（4096 token） |

`quest_shared_heads` 在 mass 上略输，在 coverage 上**全面领先**。解释：共享排名等价于一次跨
head 的"投票/并集"，答案 token 只要对某些 head 分数高就会被保住；per-head 选择让每个 head
各自把它丢掉。三个 variant、五个 layer 一致，不是单点噪声。

**工程含义**：不要"按论文修正"把仓库选择器改成 per-head——那会让检索变差，而且更贵。

## 结果三：粒度是决定性旋钮，而 32 token 恰好免费

gran=32、layers≥7、`natural` variant 的 coverage：

| k/L | k tokens | 仓库实现 | per-head | oracle(按质量) | 最好的坏臂 |
|---|---|---|---|---|---|
| 2.0% | 160 | 0.600 | 0.500 | 0.400 | 0.050 |
| **5.1%** | **416** | **1.000** | 0.775 | 0.625 | 0.050 |
| 10.9% | 896 | 1.000 | 0.825 | 0.750 | 0.150 |
| 25.0% | 2048 | 1.000 | 0.900 | 0.825 | 0.225 |

当前引擎的 KV block 是 **256 token**，在这个粒度上要 25–50% 预算才能全保住答案；降到 32 token
只要 **5.1%**。而 32 token × 8 层 KV = 128 KiB，正是带宽门里 `RAND/SEQ = 0.98–1.00` 的那一档
——**换成细粒度选择在带宽上不花钱**。

两道门第一次可以对账：

```
带宽允许（12.1 ms GPU 读权重窗口内）：k ≈ 900 token（bf16）/ ≈ 2900（int8+VNNI）
检索需要（gran=32, layers≥7）：      k ≈ 416 token
→ 交集存在，余量约 2×（bf16）
```

## 结果四：第 0 层不检索

layer 0 上所有臂的 coverage 都是 0.000（k/L=50% 时也只有 0.125）。答案 token 在第一层根本没有
被特殊对待。**首层必须保持 dense**（或只用 sink+recency），不能纳入稀疏选择。这是一个具体的
实现约束，不是观察。

## 局限（这些没解决之前不能加大结论）

1. **选择器层面的度量，不是端到端准确率。** coverage=1.0 只证明答案 token 没被扔掉，不证明
   稀疏 attention 之后模型答对——数值分布变了仍可能出错。端到端仍需单独验。
2. **单 needle、单深度（0.5）、单模型、单上下文长度（8192）、单 seed。** 深度扫描、多 needle、
   多模型都没做。needle 位置在 4077–4094，正好在中段，尚未验证 sink 附近或极深处的行为。
3. **`distractor` 的模型正确性一开始被我判成 False，那是 `--max-new-tokens=12` 截断造成的假象**
   （补跑 32 token 后 `correct=True`，答案 60494 完整输出）。默认值已改为 32。**这类"看起来像
   结论其实是仪器 bug"的情况，是这轮唯一一次差点写进结论的错误。**
4. **`repetitive` 对照没有按预期退化。** 在 coverage 判据下它同样有分辨力（坏臂 0.000）。原始
   门禁 100% 满分的主因是端到端 5 位数字答案太容易，而不是 haystack 太简单。
5. 机器仍是共享的（load average≈103，31 users），但本门是确定性计算，污染只影响耗时不影响数值。

## 下一步

1. **深度扫描 + 多 needle**：把 depth ∈ {0.05, 0.25, 0.5, 0.75, 0.95} 和 2–4 个同时有效的
   needle 加进去，检查 coverage=1.0 的 k/L 门槛是否稳定在 5%。这是把"416 token 够用"变成
   可验收数字的前提。
2. **端到端稀疏 attention 复核**：用 gran=32、共享排名、k/L=5–11% 真的跑一遍 decode，看输出是否
   仍正确——把选择器保真度和最终准确率之间的缝补上。
3. **CPU 侧 attention 原型**：先做无 selector 的 CPU full attention 正确性臂（对齐 GPU 结果），
   再接 selector；NUMA 用 `interleave=all`；int8 分支用 VNNI。
4. **`a` 项标定**：CPU 协调会推高每序列固定开销，而它决定天花板是 8.7× 还是 49.7×。
