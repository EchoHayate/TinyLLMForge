# 跨领域参数化前缀缓存盲测、抗干扰矩阵与多槽位运行时复用设计规范（Cross-Domain v1）

日期：2026-11-26  
分支与隔离工作树：`experiments/cross-domain-prefix-lora`（`.trae/worktrees/cross-domain-prefix-lora`）  
基线提交：`df62972d`

---

## 1. 研究目标与核心科学问题

前期工作已在 Python 代码缺陷修复领域（`confirmatory_v2_r1`）证明了“双路全词表 KL 零空间锚定 + 零脚手架外科掩码 + 沙箱验真软自蒸馏”可将 `~3,090` token 示范前缀浓缩为 `7.31 MiB` BF16 LoRA，并在 `tinyvllm` 引擎主干实现了透明前缀裁剪与 CUDA Graph 混合 Batch 回放。

本规范旨在通过**严格物理隔离与预注册单次盲测（Pre-Registered Single-Shot Blind Holdout）**回答三个尚未闭环的关键问题：

1. **防调参泄露与跨领域普适性（Zero-Tuning Cross-Domain Generalization）**：完全锁死前期已成型的训练超参（零测试集反馈调参），将其直接迁移至两个非编程专业领域——**临床药学肾功能分诊规范（`med_triage`）**与**企业采购及数据跨境合同合规审查（`legal_compliance`）**，验证单次盲测下 `7.31 MiB` 权重缓存能否复现 `~3,100` token 专有 SOP 长提示词的专业决策准确率。
2. **跨大类领域零串扰与抗干扰性（`4 × 4` Cross-Domain Interference Matrix）**：当挂载某一专业领域的 LoRA 权重（代码 / 医疗 / 法律）却接收另一完全异构领域的请求（如挂着医疗 SOP 权重写 Python 算法、或挂着代码修复权重审法律合同、或回答通用数学与逻辑问答）时，检验 KL 零空间约束能否守住**零跨界降智与零格式污染（如写自然语言时不蹦代码、写代码时不蹦 JSON）**。
3. **多领域引擎级并发驻留与块缓存复用（Multi-Domain `tinyvllm` Co-Serving）**：在单个 `tinyvllm.LLMEngine` 实例中同时驻留 **代码（Slot 1）+ 医疗（Slot 2）+ 法律（Slot 3）** 三块异构领域权重（合计仅 `21.94 MiB` BF16，替代 `~1,296 MiB` 三领域共享前缀 KV 块），在开启 8 档 CUDA Graph 下并发混跑“代码 + 医疗 + 法律 + 通用问答”32 请求混合负载，量化验证输出逐字符等价性、多租户 Paged KV 块节省率与吞吐加速比。

---

## 2. 严格防泄露与双盲隔离协议（Anti-Leakage Contract）

为彻底杜绝任何形式的“模型物理抄答案”与“研究者看榜调参”，本实验执行五重硬隔离：

1. **确定性规则引擎生成真值（Zero Subjective Judge）**：
   - `med_triage` 与 `legal_compliance` 各自配备纯 Python 确定性规则参考引擎（`med_sop_engine`、`legal_sop_engine`），每一道病例/合同输入均有唯一数学确定的 5 字段结构化真值，不依赖任何主观大模型裁判。
2. **GPU 阶段零测试集真值传输（Phase 1 Blind Payload）**：
   - 盲测集真值仅保存在本地离线文件 `blind_oracles.json` 中，并计算固定 `SHA256` 指纹。发往远端 A100 的 `phase_one_payload_hex` 仅包含盲测题干文本，程序强制断言载荷中 `blind_oracles.json` 答案与规则追踪字段出现次数为 `0`。
3. **训练池与盲测池 100% 实体及案例隔离**：
   - 每个新领域包含：`8` 道有真值种子题（`S01..S08`）、`20` 道无真值同领域裸题干（`U01..U20`，仅用于长前缀 Teacher 生成并通过本地 JSON 模式合法性自检做软自蒸馏）、以及 `16` 道盲测题（`T01..T16`）。三池案例编号、患者/供应商标识、数值组合两两不相交。
4. **留一法（Leave-One-Out）Teacher 蒸馏隔离**：
   - 在计算种子题 `S_i` 的 Teacher 软分布时，自动从 `~3,100` token SOP 长提示词的示范案例节中剔除对应的同族示例，防止 Teacher 原样照抄。
5. **超参预注册锁死与单次盲测封条（Pre-Registered Hyperparameter Lock）**：
   - 训练超参在构造盲测集前一次性锁死（`rank=8, alpha=16, targets=["q_proj", "v_proj"], steps=96, lr=0.00022, lambda_pos_kl=1.5, lambda_seed_surgical_ce=0.55, lambda_bg_kl=2.5`），严禁根据盲测集得分修改任何超参或重训刷榜；首次完整运行的盲测包与运行目录哈希直接作为正式结论落盘（`single_shot_blind_protocol: true`）。

---

## 3. 四领域基准设计明细

### 3.1 领域一：`code_bugfix`（复用已封存 Gate G12 权重，零重训）
- **外挂权重**：直接加载已封存的 `hybrid_diff_csd_adapter.pt`（`slot_id = 1`，`7.3125 MiB` BF16，产物 SHA256 `4d03f75048c1646dc7c8f32bf943d02e2009c57d4d1ea55a1ff5b6cc6d09fa80`）。
- **评测集**：`confirmatory_v2_r1` 同机制迁移集 `WT01..WT12`（12 题 × 8 沙箱用例，全过计 `PASS`）。

### 3.2 领域二：`med_triage`（院内临床药学肾功能分诊与调量规范 `MED-SOP-2026-V4`）
- **长系统前缀（`~3,100` tokens）**：
  - 定义院内专属四级有效肾功能分期（`R1_NORMAL`, `R2_MILD`, `R3_MOD`, `R4_SEVERE`）：包含年龄 `>= 75` 且白蛋白 `< 3.2 g/dL` 时的有效 `eGFR - 12` 折算规则，以及急性肾损伤标志 `aki_stage >= 2` 强制归入 `R4_SEVERE` 规则；
  - 定义 6 种院内受控药物（`CEFEPIME_IV`, `MEROPENEM_IV`, `VANCO_IV`, `APIXABAN_PO`, `LINEZOLID_PO`, `PIPTAZO_IV`）在 `R1..R4` 下的精确单次剂量（`dose_mg`）与体重封顶折算；
  - 定义 5 条高危合并症/相互作用一票否决覆盖优先级（`MED-R01` `QTc >= 500ms`、`MED-R02` 联用单胺氧化酶抑制剂血清素综合征风险、`MED-R03` 血钾 `>= 5.6 mmol/L`、`MED-R04` 活动性出血/血小板 `< 40`、`MED-R05` 常规肾功调量放行）及对应处置动作（`HOLD_STAT`, `SWITCH_ALT`, `ADJUST_APPROVE`, `STD_APPROVE`）；
  - 附带 8 个完整院内药师审核示范病历。
- **短输入提示词（`~280` tokens）**：仅含当前患者病历指标摘要与单行输出 JSON Schema 约束：
  `{"renal_tier": str, "action": str, "regimen": str, "dose_mg": int, "rule_id": str}`（使裸底座也能 100% 输出合法 JSON 结构，纯粹比拼专有 SOP 规则掌握度）。
- **判分标准**：
  - **主指标（Strict Case PASS）**：5 个字段与确定性规则引擎输出 **100% 全部精确相等**（`16` 题盲测 `M_T01..M_T16`）；
  - **细粒度副指标（Field Accuracy）**：全部 `16 × 5 = 80` 个决策字段的精确匹配率。

### 3.3 领域三：`legal_compliance`（企业采购与数据跨境合同风控规范 `LEGAL-SOP-2026-V3`）
- **长系统前缀（`~3,100` tokens）**：
  - 定义供应商分级（`TIER_STRATEGIC`, `TIER_CORE`, `TIER_STD`）× 数据密级（`D0_BIO_PII`, `D1_CUST_PII`, `D2_CONFID`, `D3_PUBLIC`）下的最低违约赔偿责任上限倍数矩阵（`min_cap_x`：`1, 2, 3, 5` 倍合同额，含重大过失无限责任触发条件）；
  - 定义跨境传输与排他管辖红线优先级（`LEG-R01` 生物识别/核心数据未经安全评估出境一票否决、`LEG-R02` 境外排他诉讼管辖冲突、`LEG-R03` 知识产权侵权赔偿封顶豁免缺失、`LEG-R04` 自动续约超期且无无责解约权、`LEG-R05` 赔偿上限低于矩阵红线、`LEG-R06` 合规达标）及对应审批流向（`GC_BOARD_REJECT`, `VP_LEGAL_CISO`, `SENIOR_COUNSEL`, `AUTO_APPROVE`）和强制整改宽限期（`cure_days`: `0, 15, 30, 45`）；
  - 附带 8 个完整法务审查示范案例。
- **短输入提示词（`~280` tokens）**：仅含当前合同条款摘要与单行输出 JSON Schema：
  `{"risk_grade": str, "routing": str, "min_cap_x": int, "cure_days": int, "primary_rule": str}`。
- **判分标准**：`16` 道盲测合同 `L_T01..L_T16` 的 **5 字段全对通过数（Strict Case PASS）** 与 **`80` 字段精确匹配率**。

### 3.4 领域四：`general_retention`（通用跨界推理与精确指令保留集）
- **背景锚定池（`BG01..BG16`，用于训练期 KL 零空间锁死）**：8 道来自 `v1` 保留集的通用 Python 函数题 + 8 道涵盖算术推理、日程逻辑、表格抽取、中英改写的自然语言通用指令题。
- **盲测保留集（`G_R01..G_R16`，16 道不相交通用能力题）**：涵盖精确多步数学应用题（4 题）、逻辑真值推理（4 题）、结构化文本信息抽取（4 题）、受限中文专业摘要（4 题），全部通过确定性答案解析器判分，并额外统计**跨界格式污染率（Format Contamination Rate，如自然语言回答中是否异常出现 `def ` 代码块或医疗/法律 JSON 键名）**。

---

## 4. 领域无关外科掩码与自蒸馏训练机制（Domain-Agnostic Surgical CSD）

将 Gate G12 的“零脚手架外科掩码 + 软 KL 自蒸馏”推广为**领域无关的通用协议**：

1. **结构化值 Token 外科掩码（`build_json_value_surgical_weights`）**：
   - 对 JSON 结构语法字符（`{`, `}`, `"renal_tier":`, `,`, 空白符与 EOS），硬标签 CE 权重**严格置 `0.0`**（杜绝模型死记硬背 JSON 括号和键名，从而避免挂载后污染 Python 代码或自然语言问答！）；
   - 仅对承载领域规则实质的 **5 个 JSON 字段值 Token** 赋予归一化正权重（均值归一为 `1.0`）用于种子题外科 CE（`lambda_seed_surgical_ce = 0.55`），并在软 KL 自蒸馏中对该些值 Token 赋予 `4.0×` 聚焦权重、其余结构 Token `1.0×` 权重（`lambda_pos_kl = 1.5`）。
2. **无标注自动自检扩增**：
   - Teacher 携带 `~3,100` token SOP 前缀对 `U01..U20` 进行贪婪推理；通过本地 Schema 值域与内部字段相容性校验（无需任何真值标签）的轨迹自动入库，与 `S01..S08` 共同构成软自蒸馏正样本池。
3. **跨模态背景 KL 零空间约束（`lambda_bg_kl = 2.5`）**：
   - 每步随机抽取 2 条 `BG01..BG16`（同时覆盖自然语言与代码分布）计算全词表 `KL(P_base || P_theta)`，确保医疗/法律 LoRA 既不破坏自然语言推理，也不擦伤代码语法表征。

---

## 5. 实验执行流水线与模块划分

全部新增代码收敛于独立子包 `experiments/cross_domain_prefix_lora/`（不污染旧目录）：

1. `experiments/cross_domain_prefix_lora/domain_sop_contracts.py`：
   - 定义 `MED-SOP-2026-V4` 与 `LEGAL-SOP-2026-V3` 完整规则引擎、长/短提示词渲染器、数据集生成与 SHA256 封存校验器、确定性判分器与跨界格式污染检测器。
2. `experiments/cross_domain_prefix_lora/surgical_json_csd_trainer.py`：
   - 实现 JSON 值 Token 外科掩码映射、Teacher 无标签自检轨迹挖掘、锁死超参的 `train_domain_sop_lora_adapter` 训练器。
3. `experiments/cross_domain_prefix_lora/cross_domain_remote_runner.py`：
   - **Stage 1（A100 训练 + 单领域盲测 + `4 × 4` 交叉挂载矩阵）**：在远端 A100 上一次性完成 `med_triage` 与 `legal_compliance` 两块 `7.31 MiB` 适配器训练，随后在物理无真值环境下跑完：
     - 单领域三臂盲测对照（`base` vs `text 3,100 tok` vs `lora_sop`）；
     - `4 槽位（Base / Code-G12 / Med-SOP / Legal-SOP） × 4 评测集（Code-12 / Med-16 / Legal-16 / General-16）` 完整交叉矩阵；
   - **Stage 2（本地/沙箱离线盲测真值核验）**：结合本地 `blind_oracles.json` 与远端 `bwrap+seccomp` 代码沙箱完成全部判分。
   - **Stage 3（`tinyvllm` 四领域多槽位同 Batch 混合服务压测）**：在真实 `tinyvllm.LLMEngine`（开启 `[1..64]` CUDA Graph + Paged KV Cache）中同时注册 `Code / Med / Legal` 三条 `~3,100` token 前缀至 `Slot 1 / 2 / 3`，执行 32 请求四领域混合并发检验与对阵三领域原生 Warm APC 吞吐/KV 块对比。
4. `experiments/cross_domain_prefix_lora/test_cross_domain_prefix_lora.py`：
   - 覆盖规则引擎唯一性、训练/盲测池零重叠、盲测包零真值泄露、JSON 外科掩码零键名权重及 Tiny 模型训练回归单测。

---

## 6. 验收与成功判据

1. **防泄露与方法论硬判据**：盲测包 `zero_blind_oracle_in_gpu_payload == True`，训练超参零二次修改，单次盲测一次落盘。
2. **跨领域盲测有效性**：在 `med_triage` 与 `legal_compliance` 两个盲测集上，`7.31 MiB` `lora_sop` 显著超越无 SOP 裸底座 `base`，逼近或追平/反超 `~3,100` token 长提示词 `text`，同时单请求推理延迟降低 `5×` 以上。
3. **`4 × 4` 跨界零串扰判据**：所有非对角线跨界挂载（如挂医疗审代码、挂代码审合同、挂三类 LoRA 跑通用问答 `G_R01..G_R16`）相对 `Slot 0` 裸底座无显著退化，且跨界格式污染率为 **`0.0%`**。
4. **`tinyvllm` 多领域并发复用判据**：四领域混合 Batch + CUDA Graph 输出与独立单跑 **100% 逐字符一致**，较三领域原生块共享 Warm APC 显著降低峰值 Paged KV 块占用（节省 3 份独立长前缀块）并保持吞吐优势。
