# 受控外挂 LoRA 后续四项优化攻关路线图（Roadmap）

日期：2026-11-24，Asia/Shanghai。

状态：**四大优化攻关方向已全部实测跑通并归档闭环（Gate G8、Gate G9、Gate G10、Gate G11 全线完成）。**

---

## 0. 当前已锁定的实测基线（出发点）

在 `Qwen3-8B` + `confirmatory_v2_r1`（8 道监督题 `V01..V08`、12 道同机制迁移题 `WT01..WT12`、12 道异机制保留题 `WR01..WR12`）上，前期实验已确立以下定量锚点（详见 `experiments/lora_necessity/20261124-lora-restraint-ablation-results.md`）：

- **长提示词参考基准（`text`）**：`transfer = 4/12`、`retention = 4/12`、**总分 `8/24`**；但单次输入膨胀至 `~3,500` token（前缀 KV Cache 约 `428.9 MiB`），48 次生成耗时 `6,686.32s`。
- **无约束深训外挂 LoRA（`lora_conv`，96 步 `KL=0`）**：`transfer = 4/12`（追平 `text`），但 `retention = 2/12`（误伤 `WR03`、`WR11`），总分 `6/24`。
- **当前最优外挂基线（`lora_kl_64`，Gate G7-R1：64 步 + `R01..R16` 全词表自蒸馏锚定 `lambda_kl=2.0`）**：
  - 推理形态：**`14.62 MiB` 纯标准 `rank-8` `(A, B)` LoRA（零外置路由器、零推理代码侵入）**；
  - 质量表现：`transfer = 3/12`、**`retention = 4/12`（100% 零遗忘，未见异类 `D01..D04` Top-1 选词一致率 `99.15%`）**、**总分 `7/24`**；
  - 系统收益：48 次生成耗时 `752.17s`（**比 `text` 快 `8.89×`**），训练+锚定准备共 `195.03s`（**`0.79` 道题即回本**）。
- **已证伪/触顶边界**：
  - 外置单向量读题隐状态余弦硬开关（Gate G6-A）因多约束题型风格分量占比过高（跨机制余弦 `0.581` vs 同机制 `0.701`），未见同类题召回仅 `4/12`，不继续深挖纯单向量硬阈值；
  - 仅在原有 8 条标准答案上继续强压 CE Loss 至 `0.018`（Gate G7-R2，96 步 `KL=1.0`）不会提升迁移（仍为 `3/12`），反而因死记硬背局部变量名使保留回落至 `3/12`。

---

## 1. 待逐项攻克的四大优化方向

### 方向一（已完成突破 ✅）：长提示词 Teacher → 短提示词 Student 真·上下文自蒸馏（Context Self-Distillation）与无标注扩增

- **实测达成状态（Gate G9，`20261125-context-self-distillation-results.md`）**：
  - 已实现 `experiments/lora_necessity/context_self_distillation_ablation.py`（正反双路 100% 全词表 KL 自蒸馏 + 28 道无答案题干 `use_cache=True` 自动生成并经沙箱筛出 `17` 条双公开用例全过真经验 + 差异行 `4×` 权重）；
  - 未见异类 `D01..D04` 选词一致率达 **`100.00%`（`118 / 118` token，KL 低至 `0.00057`）**；
  - 24 题评估实现 **`transfer = 4/12`、`retention = 4/12`、总分 `8/24`（在全部两个子集上 1:1 打平长提示词 `text`）**，且在全部 24 道题上相对裸底座实现 **`right_to_wrong = 0`（零改坏严格帕累托改进）**，推理快 **`8.87×`**。

- **现有代码瓶颈**：
  - `experiments/lora_necessity/confirmatory_v2_training.py` 与 `confirmatory_v2_contract.py:263`（`confirmatory_training_jobs`）目前对人工编写的 `reference_source` 做硬标签 `0/1` Cross-Entropy，强迫 `rank-8` 权重去拟合特定临时变量名（如 `live, st, en`），且训练提示词含 `Failure: [...]` 见证行而首轮测试提示词不含该行（存在训练—测试输入微小错位）。
  - 同时，受限于人工编写 `reference_source` 的数量，每个机制只有 2 道监督题（共 8 题），触及少样本泛化天花板。
- **具体改法**：
  1. **Teacher–Student 全词表软分布对齐**：以冻结底座拼接长提示词示例 `p_base(· | [Examples + Task])` 作为 Teacher，以挂载 LoRA 但仅输入短题干 `p_LoRA(· | [Task])` 作为 Student，在 Teacher 自身贪婪生成的修复代码轨迹上最小化全词表前向 KL 散度，并同时保留无关背景集 `B` 的 `KL=2.0` 零空间护栏：
     `L_total = KL(p_base(· | Examples, x_pos) || p_LoRA(· | x_pos)) + lambda_bg * KL(p_base(· | x_bg) || p_LoRA(· | x_bg))`
  2. **无标注同域裸题干扩增**：无需任何人工标准答案或私有用例，仅收集/构造 32–64 道同机制无答案裸题干，离线让 Teacher 生成软分布轨迹喂给 LoRA，把有效训练样本量放大 `4~8×`，而外挂体积仍锁死 `14.62 MiB`。
  3. **消除格式错位**：训练期 Student 输入统一使用与线上首轮完全一致的无 `Failure` 短题干格式。
- **验收指标**：在保持 `retention >= 4/12`（零净退化）的前提下，将 `transfer` 从 `3/12` 推升至 `>= 4/12`（总分 `>= 8/24`，正面追平或反超长提示词 `text`）。

---

### 方向二（已完成突破 ✅）：错代码 vs 修复代码差异 Token 聚焦加权（Diff-Token Focused Loss）

- **实测达成状态（Gate G8，`20261124-diff-focused-lora-results.md`）**：
  - 已实现 `experiments/lora_necessity/diff_focused_lora_ablation.py`（修复改动行 `4.0×`、骨架行 `1.0×`、无意义 `ValueError` 文案缩写 `0.25×` + 首轮无 `Failure:` 格式对齐 + `KL=2.0` 零空间锚定）；
  - **同机制迁移跃升至 `5 / 12`（零回退反超长提示词 `text` 的 `4 / 12`！）**，保留集 `3 / 12`，**24 题总分升至 `8 / 24`（正式追平长提示词 `text`，同时快 `8.87×`、`0.81` 道题回本）**。

- **现有代码瓶颈**：
  - `experiments/lora_necessity/training_worker.py:34`（`answer_loss`）对目标函数全部 ~230 个答案 token 等权求平均。然而代码修复任务中约 `80%~85%` 的 token 是原样照抄 `initial_source` 的函数签名、输入校验和返回骨架，真正承载算法修复规律的关键改动仅占 `15%~20%`，导致低秩容量被大量“照抄梯度”稀释。
- **具体改法**：
  1. 在训练准备阶段，基于 `difflib.SequenceMatcher` 在 token 序列或代码行级别对齐 `initial_source` 与目标修复代码，自动标记新增/修改的 **Diff Token 集合**。
  2. 将监督损失改为加权交叉熵 / 加权 KL：对未改动的骨架 token 赋予基础权重 `1.0`，对承载核心修复逻辑的 Diff Token 赋予聚焦权重 `w_diff in [3.0, 5.0]`（归一化总权重保持学习率尺度一致），让 `rank-8` 子空间优先编码“如何修错”而非“如何复述原题”。
- **验收指标**：在相同步数与相同 `lambda_kl` 下，对比等权训练在开发集 `C01..C12` 及迁移集上的错误模式修复率提升，且语法合法率保持 `100%`。

---

### 方向三（已完成定型 ✅）：等字节预算下扩展至 MLP 知识记忆层（Attention + MLP 混合挂载）

- **实测达成状态（Gate G10，`20261125-hybrid-mlp-and-kv-serving-results.md`）**：
  - `13.31 MiB` FP32（`6.66 MiB` BF16，`rank=4` 覆盖 `q/v + mlp.down_proj`）逐题 100% 复现 `5/12` 迁移 + `3/12` 保留 = `8/24`；确立纯 `q_proj/v_proj` 已吃满容量且算子数少 25%，锁定为最优算子拓扑。

- **现有代码瓶颈**：
  - `experiments/lora_necessity/training_worker.py:75`（`_targets`）仅挂载 `self_attn.q_proj` 与 `self_attn.v_proj`。在删去 `3,050` token 上下文示例后，注意力层已无外部示例可供检索；而 Transformer 中承担“无上下文内化规则与键值记忆”的核心模块是 FFN/MLP 层（`gate_proj`、`up_proj`、`down_proj`）。
- **具体改法**：
  1. 在**严格锁死总外挂体积不膨胀（仍控制在 `<= 15 MiB`）**的前提下，调整 rank 分配（例如 `rank=4` 覆盖 `q_proj, v_proj, up_proj, down_proj` 或 `o_proj + down_proj`），对比纯 `q/v` 与 `Attention + MLP` 混合挂载在零上下文规律内化上的差异。
  2. 沿用 `kl_anchored_lora_ablation.py` 的背景集全词表 KL 锚定，验证 MLP 挂载后是否同样能被完美约束在无关任务的零空间内（`D01..D04` Top-1 一致率 `>= 98%`）。
- **验收指标**：等参数字节数下对比纯 `q/v` 在同机制迁移题上的得分净增与保留集稳定性。

---

### 方向四（已完成系统实测闭环 ✅）：`7.31 MiB` BF16 压缩 + 真 SDPA / Warm Prefix KV Cache 高并发吞吐与容量基准

- **实测达成状态（Gate G11，`20261125-hybrid-mlp-and-kv-serving-results.md`）**：
  - 在 A100 80GB 开启硬件 `SDPA` 与 **100% 免费预热 Prefix KV Cache** 同台压测下：`B=32` 时权重 Cache 将吐字延迟从 `88.39 ms` 压至 **`58.85 ms`（快 `1.50×`）**、额外 KV 显存从 `23.1 GiB` 压至 **`3.4 GiB`（省 `85.2%`）**；`B=64` 时冷/热 Prefix KV Cache 双双 **`CUDA_OOM` 崩溃**，而权重 Cache 稳跑 **`1,095.6 decode tok/s`（总吞吐较热缓存上限翻 `2.65×`）**，单卡多专精驻留上限从 `17~96` 个跃升至 **`5,666` 个**；同时实测揭示原地 BF16 权重相加存在 7-bit 尾数吞噬现象，锁定“激活空间动态旁路 `7.31 MiB` BF16 LoRA”为零损高吞吐标准部署形态。

- **现有代码瓶颈**：
  1. **训练前采样过慢**：`kl_anchored_lora_ablation.py:276`（`prepare_kl_anchor_trajectories`）使用 `use_cache=False` 逐词重算，16+4 道短题耗时 `79.9s`，占了 `195s` 总固定开销的 `41%`；
  2. **推理旁路仍有 `~12%` 开销且占 `FP32` 双倍体积**：`training_worker.py:21-31`（`LoRAProjection`）以 FP32 存储并在每层做 `BF16 -> FP32 -> BF16` 旁路矩阵乘，使 `lora_kl_64`（`752.2s`）比纯 `base`（`669.6s`）慢约 `12.3%`；
  3. **尚未在开启真实 KV Cache 的引擎并发条件下量化系统红利**：当前实验为严格隔离统一采用 `use_cache=False` 单请求串行评测，尚未在 `TinyLLMForge` 真实 KV Cache 引擎上实测多并发吞吐曲线。
- **具体改法**：
  1. **离线压制提速至 `< 40 秒`**：锚定轨迹提取阶段启用只读 KV Cache + Mini-batch 并行解码，将总准备+训练时间从 `195s` 压进 `~35-45s`（回本题数从 `0.79` 道进一步压至 **`< 0.2` 道题**）。
  2. **`7.31 MiB` BF16 存储 + 单租户可逆原地权重合并 / 多租户融合算子**：支持将导出适配器存为紧凑 BF16（`7.31 MiB`，相对 `428.9 MiB` 前缀 KV Cache 压缩 **`58.7×`**，相对 32K 规范压缩 **`615×`**），单专精场景支持 `W_merged = W_base + (alpha/r) * B @ A` 可逆合并实现 **`0.0%` 推理延迟开销、`0` 额外显存**。
  3. **`TinyLLMForge` 冷热前缀自动晋升（Tiered Prefix-to-LoRA Promotion）原型与吞吐基准**：在开启真实 KV Cache 的推理服务下，端到端对比“驻留 `3,050 ~ 32,000` token Prefix KV Cache”与“晋升替换为 `7.31 MiB` 受控 LoRA”在 **TTFT、Decode TPOT、最大并发 Batch 容量与显存池占用** 上的实测曲线（填补 vLLM / SGLang 缺失的中间闭环）。

---

## 2. 推荐推进顺序与防作弊纪律

1. **推荐顺序**：
   - 先攻算法上限：**方向二（Diff-Token 加权，改动最小、立竿见影） → 方向一（Teacher–Student 上下文自蒸馏 + 无标注题干扩增） → 方向三（等预算 MLP 挂载消融）**；
   - 再收系统王牌：**方向四（40 秒极速压制 + `7.31 MiB` 零开销合并 + 真 KV Cache 高并发吞吐基准）**。
2. **实验纪律（沿用现有硬规矩）**：
   - 超参与损失形式优先在独立开发集 `recalibration_v2_r1`（`C01..C12`）与未见异类探针 `D01..D04` 上调优验证；
   - 在 `confirmatory_v2_r1` 上对比时如实标注后验机制演进属性；一旦锁定最终配方，可构造第 3 套全新盲测集（如医疗/法律/代码规范专精集）做一次性封存确认；
   - 继续保持零 `#` 注释、零单字母标识符、生成阶段零私有用例泄漏、公开文档与提交零内部环境敏感信息。
