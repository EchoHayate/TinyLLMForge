# LoRA 校准题库：合成长度报告与会话协调

日期：2026-11-20，Asia/Shanghai。

本地实现、合成测试、独立任务审阅及最终两组件集成审阅已完成，
零项活动 Critical／Important／Minor。
本文件不报告真实 tokenizer 长度、模型正确率或 LoRA 收益。

## 本轮完成的工作

- `recalibration_tokenizer_report.py`：先由固定数据和原始合成回执重建
  120 行清单，再核对每条测量、摘要链、计数、身份连续性与结束回执。
- `recalibration_tokenizer_audit.py`：用明确传入的本地模拟会话，
  顺序调用 measure／finish／abort，遇到问题即停止。
- 对应两个新测试文件。旧版 324 行入口及旧计数核心未修改。

报告保持 24 行首轮、48 行参考组合、48 行固定续答的完整分母。
公开部分只包含身份、摘要、计数和状态；消息、答案、token IDs 和模板
仅在私有回执里保留。不能只给一个 PASS 就绕过原始数据重建。

固定源码检查依赖不满足时，不打开会话，返回计划 120／检查 0。
正常输入或答案超限时，停止后续测量并 finish；
测量／回执异常时 abort；finish 缺失或校验失败也 abort。
已接收回执与传入消息均使用独立副本，回调不能通过修改旧对象改变
已接受的历史。取消信号在尝试中止后继续向上传递，不吞成普通结果。

## 真实验证与合成判断的区别

测试构造的 token IDs、环境版本、快照摘要和结束记录是模拟数据，
不是本机加载 tokenizer 后的测量。合成 finalization 中的
`tokenizer_loaded=True` 只是沿用回执 schema 的模拟字段。

- `synthetic_all_rows_within_budget` 只表示给定合成记录自洽且完整结束，
  不是实际长度通过。预算仍为输入 4096、答案含唯一 EOS 256，
  `enable_thinking=False`；没有改写或截断答案。
- `tokenizer_length_validated`、真实来源／动态完成、运行／投递授权，
  以及 benchmark／formal evaluation／net benefit／通用隔离标志均 false。
- 失败报告的已检查前缀仍是 provisional：没有成功 finalization，
  不能将某行 `OK` 或 `checked=120` 解读为已结束的有效通过。
- `abort_failed=False` 不是独立证明资源释放；没有伪造 cleanup 状态。
- `session_factory` 是调用者传入的执行能力，不是认证或沙箱机制。
  本阶段只传入 FakeSession，没有默认真实 opener、根目录或真实会话导入。

## 测试与独立核验

最终实测命令均在仓库根目录，以已有 Python/unittest 运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_report -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_audit -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_tokenizer_audit_report test_tokenizer_runtime_profile test_recalibration_data_contract test_recalibration_validation test_recalibration_rendering test_recalibration_tokenizer_inventory test_recalibration_tokenizer_report test_recalibration_tokenizer_audit -v
```

- 报告：32 项 focused 通过，77 项当时的窄回归通过。
- 协调器：24 项 focused 通过；最终八模块集成 157 项通过。
- 157 项已包含上述 32＋24 项，不与 focused 或旧回归重复累加。
- 最终测试退出码均为 0，无失败、错误或跳过。
- RED 保留：报告 7 项初始／31 项扩展断言失败；
  协调器 8 项初始／24 项扩展断言失败，均非导入错误。

报告候选代码存在一处内部 helper 参数类型宽松：可选 identity/state
摘要接受内容相同的字符串子类。已先复现两个失败子用例，
再用严格 `_hex` 检查修复。正常报告路径本已传递内建字符串；
这里是直接 helper 调用的严格类型补齐，不是发现真实来源校验被绕过。

主控另做独立检查：

- 完整、输入超限、答案超限三种报告的分类算术和 360 次行身份比较。
- 五种模拟会话历史：完整、两种超限、第二条回执失效、最后结束回执失效。
- 结束失败即使已有 120 条记录，也不能得到合成预算通过。
- 检查进程没有导入 transformers、torch 或 tokenizer_session。
- 本轮明确基线中的 178 份既有文件字节不变。

以上独立检查仍使用人工合成回执，不是第二份真实执行证据。

## 证据与身份

证据根目录：`/private/tmp/lora-recalibration-tokenizer-20261120-6fCWPY`。

| 项目 | SHA256 |
| --- | --- |
| 报告组件 | `71d3e9b465127f0c75b91aa1949dc54a1ab695811e4ebfc7075a7b2679058cf6` |
| 报告测试 | `e23bbbb0a90ee2efcd51a7afac1f2501f250acf61aa495e8138379fc392ff5e7` |
| 协调组件 | `23a7bb8a1a273be006234e58811ae3227505730ada78b3b372a44c633e885e1c` |
| 协调测试 | `202c781c1c5108eb1a765cfee9ab873defaf01d99e72bff99ede057486839230` |

实施报告和原始日志为 `task-1-*`、`task-2-*`；实际四文件差异为
`final-components.diff`。独立算术及调用历史凭据分别为
`parent-report-verification.json`、`parent-coordinator-verification.json`。
初始失败和修复前通过记录均保留，不替代最终源码对应的验证。
实施中一次代理接口限流后按已有文件接续，没有重做完成任务。

报告独立审阅：Spec Compliant／Quality Approved，零项发现；
审阅 SHA256 为 `79a340b12fe78118e0e04c90972015ab8f47c4ef14e984bcd14302d05275e747`。
协调器独立审阅和最终报告／协调器集成审阅均通过，零项发现；
审阅 SHA256 为 `9f68dadd0c20ab2403682dbddd05a99c9bd1457e9966ef18f315cbe742363ed5`。
审阅范围不扩张为整个适配器或分支的真实运行准入。

主控最终核验为 `parent-final-verification.json`，绑定四份最终源码／测试、
原始测试日志、独立审阅、父级检查及本结果文档摘要。
分支保持 `experiments/feedback-active-verification`，
HEAD 保持 `c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`；
tracked/index 无差异，新增文件未提交。

## 剩余边界

本轮无 SSH、网络、真实题目执行、tokenizer／模型加载、生成或训练，
无安装、遥测、提交／推送、分支操作或清理旧文件。
没有新增 LoRA 效果数据，也未回写封存数据的 pending/false。

已完成的六个新版本地模块仍是静态／合成组件，不是可直接投递的真实
执行链。接下来需核对具体启动包、真实来源与会话绑定，再分别获取并
核验实际固定源码结果和 tokenizer 长度结果；这不自动授权真实运行。
上述验证仍不等于新题库的模型或 LoRA 效果对照。
