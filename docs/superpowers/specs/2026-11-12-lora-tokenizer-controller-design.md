# LoRA tokenizer：有序审计控制器

日期：2026-11-12。范围：已批准 tokenizer 准入总规格的本地控制器子阶段。

## 1. 决策与边界

承接 `2026-11-12-lora-tokenizer-admission-design.md` 第 4–6 节以及已封存的
owned session。继续指令授权本地实现，不授权连接、元数据发现、真实 tokenizer
加载、训练、程序执行或发布。既有文件全部保留，本阶段仅增加文件。

比较三种实现：把报告拼装塞进 session 会侵入已封存组件；仅统计成功行会丢失
停止后的分母且无法验证回执；采用独立回执校验模块与顺序控制器，保留纯函数
测试和运行时所有权边界。选择第三种，不增加通用流水线框架或 CLI。

## 2. 输入与信任

控制端既有 `build_inventory` 构造全部 324 行，通过 `encoded` 得到规范字节，
并冻结该字节的 SHA256。执行接口只接收这个最小清单及获准摘要，不接收整个
task_sources、oracle 或候选执行器。不新增清单生成逻辑，不重复现有数据 pins。
获准摘要由受信控制端提供；自行重算摘要不是独立授权或来源认证。

校验规范 JSON 字节、摘要、严格字段、版本、四个数据 pins、参考投影摘要、
全部行摘要、连续 ordinal、类别顺序 8/136/44/136、类别对应 arm/answer/sample
形态、合法消息角色和安全 ID。参考投影从 references 行重建为
`{"id": task_id, "reference_source": answer}`，核对既有 `REFERENCE_SHA256`。
清单未通过时抛出固定错误，不尝试 session，也不凭空给坏清单补造 324 行。

## 3. 两个模块

`tokenizer_audit_report.py` 提供：

- `validate_inventory(raw, approved_sha256)`：纯字节/结构校验，返回独立对象。
- `validate_record(row, record, run_id, profile_sha256, previous_chain,
  identity_sha256=None)`：逐条验证，返回独立记录；可在每次计数后立即调用。
- `build_report(inventory_raw, approved_inventory_sha256, profile_raw,
  approved_profile_sha256, records, finalization, failure=None, abort_failed=False)`：
  从原始受信输入和私有回执重新核验，再创建 `{"public": ..., "private": ...}`。

`tokenizer_audit.py` 提供：

- `run_audit(root, profile_raw, approved_profile_sha256, inventory_raw,
  approved_inventory_sha256)`：内部调用现有 `open_session`，无 tokenizer、
  session factory 或身份注入参数。测试可在模块边界 patch，生产无该入口。

导入上述模块不得枚举真实元数据、访问快照、加载库或消耗一次性 session。

## 4. 回执与停止

在运行前验证整个输入和 profile（`validate_profile`，不是 `inspect_runtime`）。
打开一次 session，依次提交完整 messages/answer。每次调用后立刻检查回执的
精确字段、kind、run/profile、ordinal、消息/回答摘要、状态、长度算术和链。
初值 `digest([])`；每步为
`digest({"previous": previous_chain, "record": record_without_chain_sha256})`。
identity 摘要必须始终一致，不接受中立计数或模拟结果升格。

每条回执验证提示文本摘要、IDs 的严格非负整数类型、计数及剩余容量；
目标存在时验证 full/prompt/target/tail 拼接、target_start、回答摘要、
长度与状态一致。输入超限时 target 必须为空；无答案行不能带 target。
最终身份还用于核对 measurement_state 摘要、EOS 唯一且位于 target 末尾。
这里只重建结构与链，不用另一种 tokenizer 重编码，更不是恶意进程隔离。

第一次 `INPUT_LIMIT` / `OUTPUT_LIMIT` 后不再 measure，但调用一次 finish，
要求终结回执保留相同 stop_reason。所有行 OK 时也必须 finish。
OPEN/MEASURE/RECEIPT/FINALIZE 异常均无重试；有 session 时尽力 abort 一次。
取消类 BaseException 在释放尝试后原样重抛，不能变成正常报告。
普通错误不输出异常文本；公开失败仅含固定 phase、ordinal、reason。
abort 失败单独标记，不能冒充进程或资源清理成功。

`failure` 严格为 `{"phase": ..., "ordinal": ...}`：

- OPEN：ordinal=null，records 为空，finalization=null。
- MEASURE / RECEIPT：ordinal=len(records)，小于 324，之前记录全部 OK，
  finalization=null；该行分别标记 MEASUREMENT_FAILED / RECEIPT_INVALID。
- FINALIZE：ordinal=null，之前记录覆盖全部行或截止首个 limit；
  finalization=null。已完成计数仍保留，不升格为成功终结。

无 failure 时要求唯一匹配的终结回执：kind/run/profile/count/chain/stop
严格匹配，identity 内容与每条 identity/state 摘要相符；
runtime/snapshot 与 profile、run、固定文件 pins 对齐；五个 flags 严格 false。
缺失、重放、重复、乱序、跨 run/profile、终结数量或链变化均拒绝。
终结本身不是外部身份认证，回执校验不能认证任意调用者伪造的进程证据。

## 5. 报告与私有边界

固定 kind 为 `LORA_TOKENIZER_AUDIT_PENDING_CLEANUP`，永不产生
`LORA_REAL_TOKENIZER_AUDIT`。保留：

- 完整 324 行有序公开投影，只含 ordinal/category/task_id/arm/row_sha256、
  status、input_tokens/target_tokens/full_sequence_tokens（无测量时 null）。
- 分类 scheduled/checked/not_checked/over_limit，checked 只计通过校验的回执；
  基础设施失败不算模型答错。未访问行一律 `NOT_RUN_AFTER_STOP`。
- 首个 stop_reason/stop_ordinal；finalization 失败不会覆盖更早的 limit。
  `failure` 另外记录终结失败，`session_finalized` 明确真假。
- `all_rows_within_budget` 仅在 324 条合规记录且终结验证通过时 true。
  `checked_inventory_within_budget` 固定 false，因为独立清理尚未验证。
  `cleanup_status="CLEANUP_UNVERIFIED"`，五个 flags 固定 false。
- run/profile/inventory 摘要；不把身份、原始错误或任意字典展开到公开摘要。

private 保存独立完整清单、已校验 records 和成功验证的 finalization。
输出两侧互不共享可变对象。私有 prompt/answer/IDs 不进入公开摘要。
此接口只返回内存对象；文件 0700/0600、传输、实现源码哈希封装、资源计量、
进程退出和清理证据由后续包装层负责，不在这里假装已落盘或已受限。
公开字段白名单与严格输入结构共同避免把任意附加字段带到日志。

## 6. 验收

先纯报告 TDD，再顺序控制器 TDD；每个源码/测试对独立审查，最后总审。
实际 frozen 清单用于验证分母和内容；tokenizer/session 仅用合成边界。
验证每个类别的首/尾停止、加载/测量/回执/终结失败、取消、abort 失败、
严格整型、目标边界、链/身份漂移、输入篡改、私有泄露和导入纯净。
至少一次经真实 owned-session 代码路径的合成加载集成测试，不能只断言 mock
被调用。全套现有 144 项回归保持；不改旧源、测试、数据、文档或封存证据。

自查：无远程授权扩张；324 行不等于生成次数；完整分母与 fail-stop 相容；
规范摘要是受信控制端的批准绑定而非自证；报告验证与进程清理分开；
LoRA 对 text 的净收益和 RSI 均尚未被本阶段证明。

## 7. 本地实施结果

本地控制器已完成，两个组件审查及全组件终审均为 Spec PASS / Quality
APPROVED，无未关闭缺陷。主控重新运行 191 项测试通过（原 144 项全部保留，
新增 47 项），耗时 15.308 秒；冷导入、源文件指纹、完整新增 diff 重建和
404 个既有保护文件检查通过。详见本阶段计划、进度账本和
`/tmp/tlf-tokenizer-controller-20261112-2a6nok/final-verification.json`；
以该独占回执存在且哈希一致为封存条件。

这些结果只验收本地实现与合成测试。真实会话代码路径的合成集成覆盖到首个
合成超限；324 行成功路径由合成回执会话覆盖，不是 324 行真实 tokenizer
测试。没有真实运行时准入、独立进程清理或 LoRA 收益结论。
