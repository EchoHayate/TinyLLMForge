# 新校准：执行回执与内容重建

日期：2026-11-20，Asia/Shanghai。

状态：**本地回执校验通过；没有新增真实执行或 LoRA 效果结果。**

## 已接通

新增 `recalibration_execution_evidence.py`：

- 严格核对 READY 的批次、输入包、源码、路径及预期 runtime 摘要。
- DONE 绑定原始 READY 和前四个完整帧的摘要，核对实际输入声明、
  backend PID、退出码、回收、输入完整性与时间关系。
- 从 backend 原始字节重建既有 480 项内容报告，不接收远端汇总 PASS。
- backend 与 SSH 分别判断。backend 有完整有效结果但 SSH 失败时，
  保留结果而停止后续；缺少有效 DONE 时保留本批派发后未知。
- 正常的参考/见证答案不匹配不截断固定排程；运行或回执失败会停止。
  不能用 SSH exit0、32 行结果或一个 completed 字段替代完整核验。

## 验证

人工构造的原始进程/JSONL/五帧回执覆盖完整 15 批与 480 项，
参考 192、见证 36，以及篡改、缺帧、超时、未回收、FD 漂移、
参考/见证不匹配和不得继续派发的情形。没有运行存储的题目源码。

| 检查 | 实际结果 |
| --- | --- |
| 实现前 RED | 29 个预期的未实现断言失败，exit1 |
| 聚焦 GREEN | 29 通过，exit0 |
| evidence/wire/content 窄回归 | 87 通过，含聚焦29，exit0 |
| 原文件保全 | 190 份摘要不变 |
| 独立规格与代码审阅 | Russell 通过，未发现 P0/P1/P2 |

没有错误或跳过；没有重复旧66/157等套件。父代理导入闭包只含九个纯项目
模块，未导入远端 backend、worker、tokenizer 或模型。
独立审阅核对保存的测试日志、差异和摘要，没有重跑测试。

实现 SHA256：
`5d5e58433065650ed950fee326638fce6619e4372b74b7c631bd9727cbf3aac7`。

测试 SHA256：
`52a636271bf8c9a76ab433d30a0a5fcb53f9a3984244d01450ba356fc35feb05`。

私有记录根：
`/private/tmp/lora-recalibration-evidence-20261120-dtPPVL/`。
原始日志为 `red.log/green.log/regression.log`，即时退出码为相应 `.exit`；
审阅为 `fresh-review.md`，父代理核验为 `implementation-verification.json`。

## 不能推出什么

当前 package/config 摘要是调用者提供的声明，尚未绑定完整启动代码、
真实预检、实际控制器和一次性 claim。`receipt_binding_consistent=true`
只说明所给回执一致；`collection_complete=true` 只说明所给15批记录完整。
即使人工回执全部正确，以下仍为 false：

`binding_complete`、`trusted_collection_accepted`、`dynamic_checks_passed`、
`launch_ready`、`run_authorized`、`tokenizer_length_validated` 和研究标志。

不能把 preparation-only 记录或完整模拟矩阵改名成真实结果。
后续需完成完整启动绑定、远端监督与本地主控，再取得独立审阅的一次性
真实执行范围及当前预检材料。缺少历史私有目录不等于当前 SSH 认证失败。

这不是任务最终结论。真实 LoRA 16 步工程资格已有历史结果，但旧 base/text
两组各4/4的实验不能区分收益；尚无同信息 `lora_text` 对 `text` 的效果比较。
继续的目标是可靠的实验结论，包括可能无收益或不划算，而不是追求 LoRA
必须获胜。无 SSH/网络、模型运行、训练、提交或推送。
