# v2 同信息 LoRA 四臂确认性对照实验设计（Gate G4）

日期：2026-11-22，Asia/Shanghai。

状态：**设计草案（承接已达标 `DEVELOPMENT_DIAGNOSTIC_USABLE` 的新校准 Gate G3，准备冻结全新 v2 监督/迁移/保留数据集并执行真实 Qwen3-8B LoRA 训练与四臂对照）。**

## 1. 研究问题与 Gate G3 实证动机

Gate G3（`source_run_id=b0dc5a77fa55f0b6a2a169f68a529673`）在多约束有状态 Python 修复任务上确立了三项关键事实：
1. Qwen3-8B 在该难度层级上既未满分饱和也未崩盘（首轮 `base=6/12`、`text=4/12`，二轮均为 `6/12`，留有 `50%` 真实提升空间）；
2. 将监督示例直接塞进提示词（`text`）带来了 **`4.41×` 输入 token 膨胀** 与 **`5.37×` GPU 生成墙钟膨胀**（`2,031.4s` vs `378.2s`），且首轮因长上下文干扰在两道题上出现退化；
3. 粗粒度 `visible_tests` 二轮状态反馈相对隐去结果的 `retry` 净增为 **`0`**。

据此，本确认性实验在同一冻结多约束复杂度下直接回答核心问题：
- **主问题（同信息参数增强：`lora_text` vs `text`）**：在获取逐字节相同可信多约束监督示例的前提下，将这批示例同时参数化进 rank-8 LoRA 是否在未见同机制迁移题（`transfer`）上带来净修复收益，同时不在未见异机制保留题（`retention`）上造成退化？
- **核心实用与成本问题（上下文压缩：`lora` vs `text` 及 `lora` vs `base`）**：把同一批监督示例训练成外挂 LoRA 后，推理时不拼长文本示例（`lora`）能否消除 `text` 的 **`~5×` 推理耗时膨胀与首轮长上下文干扰**，并精确测算单次 LoRA 训练开销需要多少次推理即可摊平？

## 2. 全新冻结数据集结构（`confirmatory_v2_r1`）

严格遵守 `2026-11-17-lora-development-recalibration-design.md` 第 7–8 节规定：**绝不复用 `C01..C12` 开发校准题，绝不复用 v1 任何 `S/D/T/R` 旧题**。全新构造并通过动态与分词双门禁核验后一次性冻结 `confirmatory_v2_r1`（共 **32 道全新多约束纯 Python 单函数修复题**，每题至少 3 条耦合状态/边界义务、2 个公开用例 `P01..P02` + 6 个私有用例 `H01..H06`）：

1. **`supervision_v2`（`V01..V08`，8 题）**：
   - 覆盖校准过的 4 类目标机制（`bounds`、`reset`、`multiplicity`、`order`），每类 2 题；
   - 同时提供逐字节一致的 `text` / `lora_text` 上下文示例块与 `lora` / `lora_text` 的 8 条答案监督训练样本（`answer-only cross-entropy`）。
2. **`transfer_v2`（`WT01..WT12`，12 题）**：
   - 同属上述 4 类目标机制（每类 3 题），函数名、领域场景与输入结构与 `supervision_v2` 及 `C01..C12` 完全不重叠，检验同机制知识向新题的迁移效果。
3. **`retention_v2`（`WR01..WR12`，12 题）**：
   - 属于与监督集完全不相交的 4 类控制机制（`dispatch` 状态分派、`window` 滑窗统计、`balance` 配对栈平衡、`dedup` 保序归并，每类 3 题），检验挂载 LoRA 后是否发生灾难性遗忘或跨机制负迁移。

## 3. 训练配置与四臂推理协议

完全沿用 `pilot_protocol.configuration()["training"]` 与已验证通过的 `training_worker.py` / `adapter_artifact.py` / `observed_generation.py`：
- **基座模型**：`Qwen3-8B`（BF16 冻结，`attn_implementation="eager"`，`use_cache=False`，贪婪解码，单次输入 `<= 4096` token、输出 `<= 256` token，1× 空闲 A100 80GB）。
- **LoRA 训练配置（仅使用 `V01..V08` 8 条样本，零接触 `WT/WR`）**：
  - `rank=8`，`alpha=16`，`dropout=0`，目标层为全部 36 层 `self_attn.q_proj` 与 `self_attn.v_proj`（FP32 适配器参数，Kaiming 初始化 `A`、全零初始化 `B`）；
  - 优化器 `AdamW(lr=5e-5, weight_decay=0)`，`batch_size=1`，`gradient_clip=1.0`，`seed=42`，`epochs=2`（共 **16 步** 精确 answer-only loss 更新）；
  - 训练前后完整校验基座张量 SHA256 零变动，保存 `adapter.pt` 并通过 `weights_only=True` 往返核验。
- **四臂配对推理与两轮评估（24 题 × 4 臂 = 96 个 episode，共 192 次贪婪生成）**：
  - 对每道评估题（`WT01..WT12` + `WR01..WR12`，共 24 题），四臂（`base`、`text`、`lora`、`lora_text`）按题号轮换配平顺序执行：
    - 第 1 轮（`initial`）：首轮贪婪生成（直接产出 **首轮单次生成准确率与推理耗时对比**）；
    - 公开核验：在 `bwrap`+`seccomp` 子进程中跑 `P01`/`P02`；
    - 第 2 轮（`feedback`）：基于 `visible_tests` 反馈贪婪生成最终候选（产出 **二轮反馈后准确率对比**）。
  - **两阶段物理防泄漏**：全部 192 次生成与 LoRA 训练在 Phase 1 完成后拉回本地封存，Phase 2 再将全部 `192` 份候选 × `8` 用例（`<= 48` 批 × 32 项）送入远端 CPU `bwrap`+`seccomp` 沙箱统一评分。

## 4. 判定准则与成本核算

1. **主要比较与辅助比较（同时保留首轮 `initial` 与最终轮 `feedback` 双指标）**：
   - 同信息主比较：`lora_text` 对 `text` 在 `WT01..WT12` 上的配对 `wrong_to_right - right_to_wrong` 净增，以及在 `WR01..WR12` 上的 `right_to_wrong` 退化数；
   - 上下文压缩比较：`lora` 对 `text` 及 `lora` 对 `base` 在 `WT01..WT12` / `WR01..WR12` 上的配对净增、单题推理延迟比值（`wall_ns`）与显存峰值差值；
   - 成本摊销点计算：用实测 `lora_train_wall_ns + adapter_save_load_ns` 除以单题 `(text_wall_ns - lora_wall_ns)`，给出外挂 LoRA 追平长提示词推理耗时的精确盈亏平衡题数（break-even episodes）。
2. **结论诚实性原则**：
   - 不预设 LoRA 必须获胜：无论结果是 **正收益（`PROMISING_FOR_REPLICATION`）**、**无质量增益但显著压缩推理耗时**、**无收益（`NO_GAIN`）** 还是 **保留退化（`OBSERVED_REGRESSION`）**，只要协议、隔离、完整性与 `192` 次生成 / `1536` 项评分全部核验无误，即为本次研究的可靠实测结论。

## 5. 执行关卡顺序

1. **Gate G4-A（本地实现 + 动态/分词前置核验）**：构造 `confirmatory_v2_r1` 32 题数据集及契约模块，跑通本地单元测试，单次远端 `bwrap`+`seccomp` 动态验证全部参考解/见证解，并单次冷进程核验 Qwen3-8B 分词长度（训练 8 行 + 评估提示与参考行全部 `<= 4096 / 256`）。
2. **Gate G4-B（单次 LoRA 训练 + 192 次四臂生成 + 隔离最终评分）**：在 1× 空闲 A100 80GB 上单次完成 16 步 LoRA 训练与 192 次四臂生成，随后完成 `bwrap`+`seccomp` 最终评分与独立离线全量复核，产出最终研究结论报告。
