# 新版 LoRA 校准题库：数据契约完成

日期：2026-11-19，Asia/Shanghai。

**第一块本地适配已完成：新版数据可以通过独立的固定身份加载和静态一致性检查。**
这是输入契约，不是 480 次实际题目执行，也不是 120 行真实 tokenizer 长度通过。

## 本次交付

- `recalibration_data_contract.py`：先核对新版五文件及原监督文件的字节摘要，
  再检查题目、入口签名、两参考、三错误源码、八用例及指定错误见证。
- `test_recalibration_data_contract.py`：21 项本地测试，覆盖固定文件变化、
  跨文件错配、bool/int 区别、见证错误、JSON/AST 边界和返回对象独立性。
- `load_dataset` 只接受固定文件；`validate_task` 可检查内存中的单题结构，
  但不证明其原始身份、语义正确性或执行权限。

两接口都不执行存储的函数。返回对象是普通数据快照，不能作为不可伪造的
运行凭证；运行、基准准入、正式评估、收益和通用隔离标志保持 false。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 实现前 RED | 8 个预期的缺少实现断言失败，退出码 1 |
| 最终新模块测试 | 21 项通过，退出码 0，无跳过 |
| 新模块及旧契约/渲染窄回归 | 30 项通过，退出码 0，无跳过 |
| 独立 Spec Compliance | PASS |
| 独立 Task Quality | PASS，0 Critical / Important / Minor |
| 既有文件保护 | 明确清单中的 167 个文件 SHA256 均未改变 |
| 实际题目函数执行 / tokenizer / 模型 / 训练 | 均为 0 |

30 项包含上面的 21 项，不能相加成 51 个不同测试。
独立审阅读取代码和原始测试证据，没有再次执行同一测试套件。

实施中发现并通过失败测试修正了三个适配问题：
C10 原有 metadata 包含 concern 字段；描述字段需拒绝非文本；
原始文件映射需拒绝 str 子类键。没有为适配而修改题库。

最终窄回归命令：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
python3 -m unittest test_validation_contract test_prompt_rendering \
test_recalibration_data_contract -v
```

源码 SHA256：
`52695a9e2fb7e90b04f4fc7c95dcdc8b0310bbb0c269e6779890b9dff989ed02`。
测试 SHA256：
`cb74252172abb3b24daf93cc45b750c8ad38c5988849699efc21b1f04758df1d`。

## 证据与剩余工作

证据根：`/private/tmp/lora-recalibration-data-contract-20261119-xRMkGn`。
其中 `task-1-report.md`、`task-1-review.md`、`task-1.diff` 保存实施、审阅和
实际改动；原始 RED/GREEN 日志及退出码分别保留。
主控也重新核对源码摘要、回归原始输出和 167 文件保护清单。

接下来仍需新增固定源码排程/回执判定，以及消息渲染、120 行清单和分词
报告/会话适配。这些组件尚未由本子计划实现；完成后仍需具体启动包审阅。
原题库封存不回写，当前结果不说明参考答案实际正确或 LoRA 有额外收益。

本轮没有远程连接、网络传输、依赖安装、遥测上传、旧 session 恢复、
新分支/worktree、提交/推送或清理既有文件。
