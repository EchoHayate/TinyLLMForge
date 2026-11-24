# 受控外挂 LoRA 路线 A：读题隐状态背景降噪按需开关设计规范（Gate G6）

日期：2026-11-23，Asia/Shanghai。

状态：**设计草案（承接已完成确认性对照的 Gate G4-B 与收敛消融 Gate G5，针对少样本全局 LoRA 的可塑性—稳定性权衡开展机制改进验证；所有治理标志保持 `false`，不替代 Gate G4-B 盲测确认结论）。**

---

## 1. 问题定义与实证起点

在 `confirmatory_v2_r1`（8 道监督题 `V01..V08`、12 道同机制迁移题 `WT01..WT12`、12 道异机制保留题 `WR01..WR12`）上，Gate G4-B 与 Gate G5 确立了明确的算法与系统事实：

1. **系统侧压缩与提速显著**：外挂 `rank-8` LoRA（仅 `14.62 MiB`）将单次平均输入从长提示词 `text` 的 `~3,500` token 压回 `~450` token，48 次生成墙钟从 `6,686.32s` 降至 `726.39s`（**快 `9.20×`**，95.59s 训练固定开销仅需 `0.39` 道题即回本），峰值 allocated 显存减少 `4.73 GiB`。
2. **算法侧暴露出“训浅不涨、训深误伤保留”的跷跷板**：
   - 16 步欠拟合 LoRA（`lora`，`lr=5e-5`，末轮平均 loss `0.7812`）：保留题 `4/12`（零退化），但迁移题仅 `2/12`（`70.8%` 首轮输出与 `base` 逐字符相同）；
   - 96 步收敛 LoRA（`lora_conv`，`lr=2e-4` 线性衰减至 `2e-5`，末轮平均 loss `0.0130`）：在零上下文示例下将同机制迁移提升至 **`4/12`（正面追平 `text`）**，但因每一层无差别叠加静态 $\Delta W \cdot x$，将异机制保留题从 `base` 的 `4/12` 拉低至 `2/12`（`WR03`、`WR11` 发生 `right_to_wrong` 退化，总分 `6/24` vs `text` 的 `8/24`）。

本设计执行用户选定的递进攻关策略中的 **路线 A（推理期极轻量“按需开关”，若路线 A 在前置探路门未达标再转向训练期约束路线 B 或组合路线 C）**：在不改动 96 步收敛 LoRA 训练目标的前提下，为适配器附加一个百 KB 级的“读题隐状态背景降噪指纹开关”，对同机制题放行 LoRA、对异机制题自动切回原装裸底座 `base`。

---

## 2. 严格数据边界与输入文件哈希钉死

为彻底杜绝测试集调参、标签泄漏或编号前缀作弊，全部输入文件均通过 SHA256 严格钉死，且按职责物理隔离：

| 角色 | 数据源与题号范围 | 题数 | 允许用途 | 冻结文件 SHA256 |
| --- | --- | ---: | --- | --- |
| **正样本锚点集 $P$** | `confirmatory_v2_r1/supervision.json`（`V01..V08`，含 `bounds/multiplicity/order/reset` 各 2 题） | 8 | 训练 96 步 LoRA + 提取正样本目标机制指纹 | `1b2173eb0cd45dec646485e2e4b6eedb03d3e44aad43caff8aab8111f05213cc` |
| **通用背景参照集 $B$** | `data/prompts_v1.json` 中 `R01..R16`（含 `signed_rounding/matrix_axis/escape_scan/validation` 各 4 题，与目标机制及 `WR` 机制均完全不相交） | 16 | 计算 Python 题公共背景均值 $\mu_B$ 并通过留一法自动标定阈值 $\tau$ | `3668d996bb0e05d600859fd12780b9f582a65aa2ad6b1881fd0bcce9832b5a05` |
| **独立开发正探针 $D_{\text{pos}}$** | `recalibration_v2_r1/prompts.json`（`C01..C12`，含 `bounds/multiplicity/order/reset` 各 3 题） | 12 | **仅用于 Gate G6-A 盲测开关召回率**（严禁参与 $\mu_B$ 或 $\tau$ 拟合） | `0b8b037017edda84c28afdf0b7a6dd547d6bfd50582a60287c733f58f6ac1e38` |
| **独立开发负探针 $D_{\text{neg}}$** | `data/prompts_v1.json` 中 `D01..D04`（含 `dev_sign/dev_stride/dev_nested_lookup/dev_count_predicate` 4 个未见异类机制） | 4 | **仅用于 Gate G6-A 盲测未见异类机制关断率**（严禁参与 $\mu_B$ 或 $\tau$ 拟合） | 同 `prompts_v1.json`（由 `structure_v1.json` `3905663be1d4cd4eb022f5d0995dcd651fafdc6514ecff0288957cec9cb0762e` 钉死划分） |
| **端到端评估集 $E$** | `confirmatory_v2_r1/prompts.json` + `oracles.json`（`WT01..WT12` + `WR01..WR12`） | 24 | **仅在 Gate G6-A 一次性达标锁死后，在 Gate G6-B 执行单次端到端评估** | `f78ea75b5a3010f5d5d5038f29d7f22037aad3c020d47ba5705b1048244ce37b` / `8d251cdc6d2f7163d6e928965bd46a04d0b418904cd89eb6cfa83dcb3f1d3a04` |

硬性防作弊约束：
1. 路由器在打包与推理全流程中**严禁读取 `family`、`split`、`mechanics`、`reference_source`、`initial_witness` 或任何私有用例 `H01..H06`**，仅允许读取与线上单题推理完全一致的 4 个公开字段：`entry_point`、`spec`、`initial_source`、`public_cases`。
2. **去编号脱敏**：送入路由器提取隐状态前，必须将题目的 `"id"` 字段统一重写为固定常量字符串 `"TASK"`，彻底阻断通过 `"V"` / `"WT"` / `"WR"` / `"C"` / `"R"` 字符前缀区分数据集的捷径。

---

## 3. 方案 A1 数学定义与运行机制

### 3.1 去编号提示词与题干隐状态提取（`extract_router_embedding`）

对任意待测或锚点题目 $x$：
1. 构造脱敏公开任务对象 `sanitized_task = {"id": "TASK", "entry_point": x["entry_point"], "spec": x["spec"], "initial_source": x["initial_source"], "public_cases": x["public_cases"]}`，调用与 `base` 首轮完全一致的 `_render_user_prompt(sanitized_task, examples_list=None, witness_obj=None)` 生成单轮对话消息。
2. 使用冻结 `Qwen3-8B` 分词器执行 `apply_chat_template(..., tokenize=True, add_generation_prompt=True)` 得到完整 `input_ids`，同时定位题干起始标记 `"Current task:\n"` 之后、生成引导标记之前的 **纯题干 token 区间 `[span_start, span_end)`**（排除所有题目共用的顶部固定指令前缀，避免指令均值稀释题目语义）。
3. 在**未挂载任何 LoRA 的原装冻结 BF16 `Qwen3-8B` 基座**上，于 `torch.no_grad()` 下执行 1 次只读 Prefill：`model(input_ids=..., attention_mask=..., use_cache=False, output_hidden_states=True, return_dict=True)`。
4. 取语义抽象最稳定的中后段 Transformer 层集合 $L = \{18, 19, \dots, 30\}$（共 13 层，对应 36 层网络的 `50%..83%` 深度）：
   - 对每一层 $l \in L$，将 `hidden_states[l][0, span_start:span_end]` 转为 `float32` 并在 token 维求均值得到 $\bar{h}^{(l)}(x) \in \mathbb{R}^{4096}$，再做层内单位化 $u^{(l)}(x) = \bar{h}^{(l)}(x) / \|\bar{h}^{(l)}(x)\|_2$；
   - 对 13 层单位向量求均值并再次单位化，得到该题的原始 4096 维单位语义向量：
     $$e(x) = \frac{\sum_{l \in L} u^{(l)}(x)}{\left\|\sum_{l \in L} u^{(l)}(x)\right\|_2} \in \mathbb{R}^{4096}$$

### 3.2 背景均值降噪与多锚点最近邻打分（`fit_whitened_prototype_router`）

1. **扣背景底噪**：在通用背景参照集 $B = \{\text{R01}\dots\text{R16}\}$ 上计算公共均值向量 $\mu_B = \frac{1}{|B|} \sum_{b \in B} e(b)$。对任意向量 $e(x)$ 定义去底噪单位化映射：
   $$\tilde{e}(x; \mu) = \frac{e(x) - \mu}{\|e(x) - \mu\|_2}$$
2. **8 点独立正锚点打分**：由于正样本集 $P = \{\text{V01}\dots\text{V08}\}$ 覆盖 4 类不同目标机制（每类 2 题），不将 8 个向量平均成单点，而是保留全部 8 个去底噪锚点 $\{\tilde{e}(p; \mu_B)\}_{p \in P}$，以**最大余弦相似度**衡量新题是否命中任一已学目标机制：
   $$s(x) = \max_{p \in P} \langle \tilde{e}(x; \mu_B), \tilde{e}(p; \mu_B) \rangle$$
3. **无人工调参的留一法（LOO）自动定阈值 $\tau$**：
   - 正样本留一分（`8` 个）：对每个 $p_i \in P$，将其自身从锚点集中剔除（剩余 7 个锚点中仍保留同机制的另 1 道伙伴题，与测试期未见同机制题的最近邻尺度严格一致）：
     $$s_{\text{pos}, i} = \max_{p \in P \setminus \{p_i\}} \langle \tilde{e}(p_i; \mu_B), \tilde{e}(p; \mu_B) \rangle$$
   - 背景负样本留一分（`16` 个）：对每个 $b_j \in B$，将其自身从背景均值计算中剔除（令 $\mu_{B \setminus \{b_j\}} = \frac{1}{|B|-1}\sum_{b \in B \setminus \{b_j\}} e(b)$），防止样本自减产生虚假负偏置：
     $$s_{\text{neg}, j} = \max_{p \in P} \langle \tilde{e}(b_j; \mu_{B \setminus \{b_j\}}), \tilde{e}(p; \mu_{B \setminus \{b_j\}}) \rangle$$
   - 在全部相邻排序分数的中点候选集上，选取最大化平衡准确率 $\text{BalAcc}(\tau) = \frac{1}{2}\left(\frac{1}{8}\sum_{i=1}^8 \mathbb{I}[s_{\text{pos}, i} > \tau] + \frac{1}{16}\sum_{j=1}^{16} \mathbb{I}[s_{\text{neg}, j} \le \tau]\right)$ 的最优区间；若存在并列最优区间，取宽度最大的间隔中点（最大间隔准则）作为唯一冻结阈值 $\tau$。
4. **序列化体积与完整性**：指纹路由包仅需存储 `8` 个正锚点向量（`8 × 4096 × 4 B = 128 KiB`）、`1` 个背景均值向量 $\mu_B$（`16 KiB`）及标量阈值 $\tau$ 与 SHA256 元数据，总计约 **`148 KiB`**（仅为 `14.62 MiB` LoRA 权重的 **`~1.0%`**）。

### 3.3 推理期硬切换执行流

在 Gate G6-B 评估每道题 `tid` 时：
1. 在原装裸底座上执行 1 次 `extract_router_embedding`（约 `0.10` 秒）计算 $s(\text{tid})$，判定 `gate_open = (s(tid) > tau)`；
2. 若 `gate_open == True`：在 `mounted_adapter(model, loaded_adapter)` 上下文内执行 `initial` 与 `feedback` 两轮贪婪生成（完整复用 96 步收敛 LoRA）；
3. 若 `gate_open == False`：直接在未挂载适配器的原装裸底座 `model` 上执行 `initial` 与 `feedback` 两轮贪婪生成，物理保证该题零权重扰动、零保留退化。

---

## 4. 模块划分与代码卫生约束

严格遵守“不修改任何已提交确认性模块”与仓库代码规范（`experiments/lora_necessity/` 下**零 `#` 注释、零单字母 AST 标识符**、纯标准库可导入契约与远端 PyTorch 执行分离）：

1. `experiments/lora_necessity/gated_lora_router.py`（新增）
   - 脱敏渲染与 `"Current task:\n"` token 区间定位；
   - 纯 Python / NumPy / PyTorch 兼容的去底噪余弦打分、留一法最大间隔阈值求解、路由指纹包序列化与 SHA256 校验；
   - 裸底座隐状态提取函数 `extract_router_embedding`（仅在远端 GPU 进程按需导入 `torch`）。
2. `experiments/lora_necessity/test_gated_lora_router.py`（新增）
   - 验证去编号脱敏彻底抹除 `V/WT/WR/C/D/R` ID 差异、严禁读取 `family/split` 字段、留一法阈值无自泄漏、合成正交簇上 `BalAcc == 1.0` 及边界回退逻辑。
3. `experiments/lora_necessity/confirmatory_v2_gated_ablation.py` 与 `test_confirmatory_v2_gated_ablation.py`（新增）
   - **Gate G6-A 探路包构造与收集器**：单次远端只读提取 `P(8) + B(16) + D_pos(12) + D_neg(4)` 共 `40` 道题的路由向量（零接触 `WT/WR`），计算 LOO 指标与独立开发集 `C01..C12` / `D01..D04` 盲测指标并校验早停门禁；
   - **Gate G6-B 端到端评估包构造、双阶段执行与交叉对账器**：在 G6-A 达标后，单次执行 96 步 LoRA 训练 + 指纹打包 + 24 题实时门控双轮生成（Phase 1，零 `oracles.json` 泄漏）+ 远端 CPU `bwrap`+`seccomp` 全量判分（Phase 2），并自动与已归档 Gate G4-B `base` 及 Gate G5 `lora_conv` 做逐题一致性对账。

---

## 5. 两阶段验证关卡、早停止损线与成功判据

### 5.1 Gate G6-A：独立开发集 1 分钟盲测探路门（零接触 `WT/WR`）

在远端单卡 A100 上仅对 `P(8) + B(16) + C01..C12(12) + D01..D04(4)` 共 **40 道题各跑 1 次裸底座 Prefill**（预计 GPU 墙钟 `< 60` 秒，零代码生成、零接触 `WT01..WR12`）：
- **必须同时满足以下 3 条硬门禁方可进入 Gate G6-B**：
  1. 校准集留一法平衡准确率 $\text{BalAcc}_{\text{LOO}}(P, B) \ge 0.80$；
  2. 独立未见同机制开发集 `C01..C12` 开关打开率（召回率）$\ge 9 / 12$（`75.0%`）；
  3. 独立未见异类开发集 `D01..D04` 开关误开率 $\le 1 / 4$（`25.0%`，即至少关断 `3 / 4`）且背景集留一误开率 $\le 4 / 16$（`25.0%`）。
- **早停止损规则**：若上述任一条未达标，立即判定 **路线 A（纯隐状态相似度硬开关）区分度不足而终止**，不跑 Gate G6-B，直接按既定路线图转入 **路线 B（训练期锚定正则/幅度约束 LoRA）** 设计。

### 5.2 Gate G6-B：24 题端到端实测门与成功标准

在 Gate G6-A 达标且获得单次远端执行授权后，锁定同一 $(\mu_B, P, \tau)$ 对 `WT01..WT12` + `WR01..WR12` 执行完整端到端训练、实时门控推理与隔离沙箱评分：
- **协议与确定性自检**：`48/48` 候选完整落盘、`0 UNKNOWN`、基座前后 SHA256 不变；所有 `gate_open == False` 题目的首轮生成须与 Gate G4-B 归档 `base` 逐字符一致，所有 `gate_open == True` 题目的首轮生成须与 Gate G5 归档 `lora_conv` 逐字符一致（双重验证门控切换零副作用）。
- **攻关成功硬指标（对比长提示词 `text` 与原 `lora_conv`）**：
  - 同机制迁移（`WT01..WT12`）：保持 **`4 / 12`**（相对 `lora_conv` 零损失，追平 `text`）；
  - 异机制保留（`WR01..WR12`）：从 `lora_conv` 的 `2 / 12` 恢复至 **`4 / 12`**（救回被误伤的保留题，相对 `base` 净退化为 `0`）；
  - 24 题总通过数：达到 **`>= 8 / 24`**（正面追平或超越长提示词 `text` 的 `8 / 24`），同时保持相对 `text` **`>= 8.5×` 的端到端推理速度优势** 与 **`< 15 MiB` 的总外挂体积**。
- **治理与报告诚实性**：所有输出报告与归档清单保持 `benchmark_eligible = false`、`formal_evaluation_ready = false`，明确标注为针对 `confirmatory_v2_r1` 可塑性—稳定性硬伤的后验受控机制攻关实验；若未来需宣称全新盲测确证，须在全新第 3 套独立题集上复现。
