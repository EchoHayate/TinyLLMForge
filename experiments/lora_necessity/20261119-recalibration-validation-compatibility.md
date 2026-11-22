# 新版题库验证入口兼容性检查

日期：2026-11-19，Asia/Shanghai。

本次仅检查本地文件、JSON 结构和源码接口，没有执行题目函数、调用 tokenizer、
连接远端、加载模型或训练。本文不是动态通过报告，也不是远程启动许可。

## 工作区与封存

当前目录为 `/Users/bytedance/dev/TinyLLMForge`，分支为
`experiments/feedback-active-verification`，HEAD 为
`c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`。
检查时 tracked/index 无差异；所有既有 untracked 文件保留。

`data/recalibration_v2_r1/` 的五个文件重新计算 SHA256 后均与交接封存一致：

| 文件 | SHA256 |
| --- | --- |
| manifest.json | `2d8bc46cb387ca80948aea6a9c9a42264e66560666bc515dbc3cf79fde1e9690` |
| prompts.json | `0b8b037017edda84c28afdf0b7a6dd547d6bfd50582a60287c733f58f6ac1e38` |
| oracles.json | `a07803a677db263e40e71423d0f1c423c093e7a26f99939274fdcd91ddef4be2` |
| structure.json | `b5ede6dd62e380530fb1ce158a0e2aefb2bb6690f0a2dcd3ee2b3d3bea729e6e` |
| review.md | `012f49362d11547b7d4cf11baf3341210a8e9dc102a425b33f94a45b49a8b2fc` |

没有回写这些文件中的 pending/false，也没有改动原 v1/v2 封存。

## 接口检查结果

以下路径均相对于 `experiments/lora_necessity/`。

| 组件 | 当前约束 | 对新版的结论 |
| --- | --- | --- |
| `validation_contract.py` | v1 文件 pins、四种源码角色、固定 1416 项排程 | 不能直接生成新版五种源码的 480 项排程 |
| `candidate_execution_protocol.py` | 仅 D01–D04、T01–T16、R01–R16；仅 P01/P02；round=1 | 不是通用固定源码接口，不能直接提交 C 题或 H 用例 |
| `candidate_execution_admission.py` | `admit_request` 先调用上述 `prepare_request` | 外层 AST 检查也继承旧身份限制，不能绕过协议直接宣称新版已准入 |
| `validation_task_worker.py` | 单次只接收 request_id/source/entry_point/args；源码至多 4096 字节，请求至多 8192 字节 | 数据尺寸兼容；仅此不证明隔离或运行可用，不能在宿主直接调用 |
| `validation_remote.py` | 固定源码最小投影；每包 1–32 项；运行环境必须与批准值一致 | 可作为复用候选，但本轮未核验远端现状或批准新的执行包 |
| `prompt_rendering.py` | v1 pins、固定 136 个 episode、旧反馈修订指令 | 不能直接渲染新版 C 题及 results_withheld 分支 |
| `tokenizer_inventory.py` / `tokenizer_audit_report.py` | v1 pins、旧参考摘要、固定 324 行和类别顺序；记录检查也依赖旧行结构 | 不能只替换 inventory 或改数量，必须另行适配清单和报告校验 |
| `tokenizer_session.py` | `measure(messages, answer)` 本身不绑定题号或行数；仍检查运行环境、五文件快照、身份、摘要链和结束状态 | 适合复用底层会话；不是新版清单、启动包或结果报告已经可用 |

旧 v1 的动态 1416 项和 tokenizer 324 行已有完成报告，本轮没有重放。
两份历史结果只用于辨认已完成能力，不证明新版或当前远端可用。

## 本次静态数据检查

标准库检查脚本仅解析 JSON、比较字段和计算序列化字节数，退出码为 0。
它没有 import/exec 任何存储的题目源码，也没有调用执行入口。

- C01–C12 的 prompts/oracles 顺序一致。
- 每题两份参考、三份错误源码、P01/P02 与 H01–H06 八例齐全。
- prompts 的公开输入/期望与 oracles 一致；initial_source 与 initial 错误源码一致。
- 36 个具名错误见证均指向已有用例，预测值在数据上区别于正确期望。
  这不是预测已被实际运行证实。
- 总计 60 份源码、480 个源码×用例位置，其中参考检查为 192 个位置。
- 按旧固定源码 worker 的四字段投影、`request-0000` 至 `request-0479`
  和 canonical ASCII JSON 计算：最大源码 UTF-8 为 735 字节，
  最大请求为 976 字节，低于当前 4096/8192 字节限额。
  这没有核验完整批次包大小，也不是 token 长度检查。

## 尚未完成

需要独立版本的固定源码排程/结果判定，以及新版提示清单/长度报告适配。
不能重命名 C/H 身份来冒充旧任务，不能放宽旧版本的准入规则。
具体实现方案和隔离启动包仍需审阅；本次没有增加执行许可。

动态核验应逐项证明 192 个参考位置符合正确期望、36 个错误见证符合预测
并区别于正确结果，同时保留全部 480 位置的执行状态。
长度核验需要真实 Qwen3-8B tokenizer、完整八例和两种续答；
4096 输入、256 输出含 EOS 的预算不变。
固定续答样例不能证明所有未来模型输出构成的续答都合格。
