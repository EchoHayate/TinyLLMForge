# latent 方向重选（2026-09-14）：先修分母，再挑赛道

本文档做两件事：

1. 报告一个**会改变前几轮结论量级的分母问题**——那个 ~40 ms 的"上下文无关常数"，大部分不是硬件物理，
   是 `tinyvllm` 多序列 decode 强制走 eager 的开销。
2. 在此基础上，用一套统一定价重新挑 latent 方向，而不是继续在 KV 字节这一条上加注。

---

## 一、分母修正：40 ms 常数里约 27 ms 是引擎开销

### 代码证据

`tinyvllm/engine/model_runner.py:12557`

```python
# FlashAttention decode replay is only correctness-validated for one
# sequence. Multi-sequence captured graphs can corrupt rows after the
# first one, so keep the batch-1 graph fast path and fail closed to
# eager execution for larger decode batches.
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

`model_runner.py:12842` 的分发条件里含 `multi_sequence_decode`，命中即 `_run_eager_logits`。
而 `model_runner.py:13360` 明明为 `bs = 1,2,4,8,16,32,...` 都捕获了 decode graph——**这些图对 B≥2
永远不会被 replay**。这是有意为之（正确性），不是 bug，但它意味着：

> 我们前几轮所有 B≥2 的测量，无论命令行写的是 eager 还是 graph，跑的都是同一条 eager 路径。

这正好解释了此前那个"graph 路径 B=1 只有 13–18 ms、B=2 直接跳到 43–50 ms"的断崖：不是 CUDA graph
在多序列下变慢，是多序列根本没用图。

### 这 40 ms 里有多少是物理

Qwen3-8B 权重 bf16 约 16.4 GB，A100-80G HBM 带宽 1.935 TB/s（实测效率按 80–85% 折）：

```
16.4 GB / 1.935 TB/s = 8.5 ms   (理论下限)
16.4 GB / 1.5-1.6 TB/s = 10.2-10.9 ms   (现实下限)
```

实测 B=1 走图：**12.98 ms**（L=2048）/ 17.6 ms（L=32768），与上面这条 roofline 同一量级。
而 B≥2 的 eager 常数是 **39.8 ms**。

```
39.8 ms (实测 eager 常数)  -  ~13 ms (roofline + 图路径实测)  =  ~27 ms 引擎开销
```

### 对既有结论的影响：方向不变，量级要重算

`c1 ≈ 0.0837 us/token` 这个斜率是带宽性质的，换引擎应大体保留；被高估的是常数。把常数换成
生产级引擎可达的 ~13 ms，同一批实测数据的含义就变了：

| | 当前引擎（c0=39.8 ms） | 生产级引擎（c0≈13 ms 假设） |
|---|---|---|
| L*B=262144 时 KV 项 | 21.9 ms / 62 ms = **35%** | 21.9 ms / 35 ms = **63%** |
| KV 压 4× 的单步收益 | -26% | **约 -47%** |
| 常见负载 L*B=65536 | -9% | 约 -24% |

也就是说，**"延迟轴已死"这个判断是 engine-conditioned 的，不是硬件事实。** 在一个不会为每个多序列
decode 步付 27 ms Python 开销的引擎上，KV 字节要值钱得多；容量轴那个 ~2000 seq/s 的天花板同样是
被这个常数压出来的，常数降下来天花板会抬。

⚠️ 反向的诚实提醒：eager 路径是 CPU-bound 的，GPU 上的 KV 读取可能与 Python 开销部分重叠，
因此实测斜率 `c1` 可能**低估**了真实 KV 成本。两个方向都指向同一件事：这台机器上"KV 字节不值钱"
这个结论，是引擎伪影，不能作为选方向的依据。

### 一行实验就能验证

`config.multi_sequence_cuda_graphs`（默认 `False`）会切到另一条动态多序列图路径
（`multi_sequence_cuda_graph_batch_allowlist = (2, 4, 8)`）。把它打开、把 allowlist 扩到
sweep 用到的 batch，重跑一遍 GATE A，就能知道常数能降到多少。**这应当是下一个动作，早于任何新方向的
选择**——因为它决定了 KV 字节到底值 3.5× 还是更多。

---

## 二、统一定价：为什么该换赛道，而不是在 KV 字节上加注

把所有 latent 相关想法按"它压缩的是什么"分三类，用我们自己的实测数给每类定价：

| 压缩对象 | 单位收益 | 有没有天花板 | 实测依据 |
|---|---|---|---|
| **每 token 的 KV 字节**（MLA / CARE / KV 量化 / 驱逐） | 只影响单步中的 KV 项 | **有，且很硬**：当前引擎 3.5× | GATE A + wall sweep |
| **上下文 token 数量**（Cartridges / CARL / gist / C2C） | 同时省 prefill、KV 容量、每步 KV 读取 | 无独立天花板，随压缩比线性 | 由 `c1` 与 prefill 成本推得 |
| **生成步数**（latent reasoning / 投机 / 早退） | 每省一步省一整个 `c0`：当前 **39.8 ms**，生产级 ~13 ms | 无天花板，线性 | GATE A 的 `c0` |

**这就是重选的核心判据：在一个固定开销主导的系统里，压"步数"和压"token 数量"都比压"每步字节"值钱，
而且不受那个 3.5× 天花板约束。** 我们花几轮测出来的最有用的东西不是 3.5×，而是
"**每一个 decode 步都要先付 ~40 ms 过路费**"这个事实——它直接给"少走几步"和"少喂几个 token"这两类
方向标了高价。

---

## 三、候选方向（含文献锚点）与可证伪的 GATE 0

### 候选 1：Cartridges / CARL 式离线自学习上下文表示
- 锚点：Cartridges（ICLR 2026）、CARL: Cartridge Adaptation through Reinforcement Learning（OpenReview `kXSgL8z9LN`）
- 压缩对象：**token 数量**。把一份被反复查询的长语料离线训练成一个小的 KV/前缀表示，在线时不再喂原文。
- 为什么现在更值得看：它省的是 prefill + 常驻 KV + 每步 KV 读取三份成本，不撞 3.5× 天花板。
- 前提假设（必须先验）：**同一份长上下文被重复查询**。若你的目标负载是一次性长 prompt，离线训练摊不平，
  方向直接不成立。
- GATE 0（半天，纯测量不训练）：统计目标负载里"同一长上下文被复用的次数分布"。复用次数中位数 < 5 就否掉。

### 候选 2：latent agent memory（隐状态跨 turn 复用）
- 锚点：A Survey of Agent Memory in the Second Half（OpenReview `XycbogUAeJ`）中的 latent-state memory 一类；
  Cache-to-cache: Direct semantic communication between LLMs（ICLR 2026）
- 压缩对象：**token 数量**（跨 turn 不再重放历史文本，而是复用隐状态/KV）。
- 风险：这是当前 agent memory 里最热但最少硬结论的一支；综述把它列为方向而非成熟方法。
  评测口径难定，容易做成"看起来能跑但说不清赢在哪"。
- GATE 0：先量"多轮 agent 轨迹里，历史文本重放占 prefill 的比例"。若 prefix cache 已经把这部分吃掉，
  latent 记忆的增量收益就很小——**这一步很可能直接否掉它**。

### 候选 3：latent reasoning / 减少 decode 步数
- 锚点：Efficient Inference for Large Reasoning Models: A Survey；Thinking Without Words: A Survey of
  Latent Chain-of-Thought Reasoning
- 压缩对象：**步数**，单位价格最高（每步 39.8 ms / 生产级 13 ms）。
- ⚠️ **必须先自我质疑**：我们上一条被关掉的线（latent action speculation）就属于"减少步数"家族。
  区别在于：投机执行需要 verify，收益被接受率吃掉，这是它死掉的原因；latent reasoning 是让模型
  在连续空间里少生成 token，不需 verify，但**需要训练、且改变输出分布与可解释性**。
- 对你的定位的现实考量：这是模型/训练方向，不是 infra 方向。以 AI Infra 为目标的项目里，
  它的落地成本和话语权都偏低。
- GATE 0：在目标任务上统计"输出 token 数的分布与可压缩冗余"，并明确"精度不许掉多少"。

---

## 四、建议顺序

1. **先修分母**（半小时级）：打开 `multi_sequence_cuda_graphs`、扩 batch allowlist，重跑 GATE A，
   拿到真实的 `c0`。在此之前不要给任何方向下定价结论。
2. **再做候选 1 的 GATE 0**（半天，纯统计）：目标负载的长上下文复用次数分布。这是唯一一个能同时
   否掉候选 1 和候选 2 的廉价测量。
3. 若复用次数够高 → 走 Cartridges 线（省 token 数量，不撞天花板，且 infra 属性强：涉及离线编译、
   缓存管理、命中调度）。
4. 若复用次数低 → 回到 KV 字节线，但**必须带着修正后的 `c0`** 重新定价；若修正后 KV 项占比升到 60%+，
   那条线的价值会比现在看起来大得多，GATE B（低秩 latent vs 量化/驱逐，同字节比质量）值得建。

## 五、这几轮真正的产出（用于复盘时引用）

- Stage 0 成本模型被证伪：`c0` 差 3 倍、`c1` 差 0.55 倍、函数形式含曲率与纯 batch 项。
- KV 字节的兑换率被量化：当前引擎上界 3.5×，天花板 ~2000 seq/s。
- **发现该兑换率本身受引擎伪影污染**（多序列 decode 不走图，每步多付 ~27 ms）——这是本文档最重要的一条，
  也是"先修分母再选方向"的直接理由。

---

## 附录（2026-09-14 晚）：分母已实测修正，结论按新分母重排

本文档正文里"27 ms 是引擎开销"当时只是代码推断。现在已经跑完实测，见
`2026-09-14-gatea-rerun-multi-sequence-graph.md`。摘要：

- 打开 `multi_sequence_cuda_graphs` 后还发现两层坑：捕获**每次都失败**（capture 区内触发
  torch.compile 重编译，dynamo 读 CUDA RNG state，非法），以及捕获预算把 B=2 判成
  `single_capture_budget`。都不体现在 step 时间上，是靠给每一步打 dispatch 标签抓出来的。
  引擎修复见 commit `3c3b6b6a`。
- 修好后 B≥2 全部 24/24 步真实 replay 捕获图。常数从 **40.44 ms 降到 11.74 ms**，
  Stage 0 假设的 13.05 ms 基本被证实（比值 0.90）；`c1` 实测 0.161 us/token 对假设 0.151（比值 1.06）。
- 正确性已验证：同 prompt greedy 解码，eager 与 graph 路径 token id **完全一致**
  （0.6B L=1024 B=4，以及 8B L=8192 B=8）。

因此本文档正文的方向定价要按下面改：

| 结论（正文） | 现状 |
|---|---|
| 延迟轴已死，KV 字节不值钱 | **撤回**。L=8192,B=32 时 KV 项 42.1 ms / 63.6 ms = 66%；L=32768,B=8 时 74% |
| 容量轴收益上界 ~3.5× | **作废**。两次 wall sweep 都跑在 eager 路径（每步多背 ~28 ms），人为压平了吞吐曲线 |
| 优先做 Cartridges（压 token 数）而非 KV 字节 | **降级为待定**。引擎侧证据现在明显偏向 KV 字节这条线 |
| latent 需要打败的基线 | **加强**。KV 字节现在值 42 ms，`kv_quant_bits=8/4` 能用极低成本吃掉大部分，低秩 latent 必须先赢过量化 |

下一步顺序不变但内容换了：先把 wall sweep 在 graph 路径上重跑（需要把 allowlist 和
static/reserved 捕获预算扩到 B=128/144，且还不确定这么宽的 page table 捕获能不能装下），
再用 M2 常数（`c0 = 11.79 ms`, `a = 0.238 ms/seq`, `c1 = 0.161 us/token`）重算容量算术，
最后才在 KV 字节线和 token 数线之间做选择。
