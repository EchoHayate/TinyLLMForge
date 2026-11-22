# 候选源码与公开反馈契约

日期：2026-11-16。继续已批准的两轮代码修复协议，不启动候选执行。

## 选择与边界

采用独立纯函数组件，而不是扩展宿主固定程序 worker，或直接接入 Docker。
前者可在不执行任何候选的条件下验证协议；宿主 worker 没有候选隔离，
当前 Docker 查询也不足以准入。保留现有生成器、提示、训练产物与数据。
本阶段不安装、不遥测、不提交、不推送、不改服务，不重新运行历史任务。

## 接口

`candidate_valid(prompts_raw, task_id, arm, candidate) -> bool`：
从固定 prompts 获取函数签名。候选必须是 UTF-8、至多 65536 字节，
整个模块只有一个普通函数，名称及参数 AST 与初始函数一致；
不得有装饰器、返回注解、类型参数、导入或额外/嵌套函数及类。
只做 AST 与编译检查，绝不调用函数、exec 或 eval，不清理 Markdown。
空串、非法语法或格式返回 false；调用者身份/类型错误抛 ValueError。
超长或非法编码抛 ValueError，与现有 second_round 输入上限一致。
静态检查不是安全沙箱，允许的函数体仍然可能包含危险调用。

`visible_requests(prompts_raw, task_id, arm, candidate, *, attempt_id) -> list`：
attempt_id 是控制端提供的 32 位小写十六进制本次尝试标识。
无效源码返回空列表；合法源码按公开用例原顺序生成请求。
每项固定 version=1、round=1，包含 attempt_id/task_id/arm/case_id、
entry_point/source/args。不含 expected、reference、隐藏评分数据。
只允许 D 任务的 base/text，以及 T/R 任务的四组。
调用端负责 attempt_id 唯一性、单次执行、时限和隔离；摘要不提供认证。

`visible_feedback(..., receipts_raw, *, attempt_id) -> dict`：
重新构造请求，解析最多 65536 字节的严格 JSON 收据列表，要求数量及顺序
完全一致。每项绑定完整请求 canonical SHA256，拒绝多字段、缺失、重复键、
跨任务/组/候选/尝试的错配。受信控制端收据状态只能是 completed、
program_error、timeout、output_limit；completed 必须带 outcome，其他
状态不得带 outcome。outcome 只接受 return/value 或 raises/ValueError；
通过 canonical JSON 与公开 expected 比较，区分 bool/int。
只输出 second_round 已接受的 case id/status，不返回原始输出或异常文本。
基础设施失败、缺失收据、非法编码是协议错误，不伪造 mismatch/timeout。
无效源码只接受空收据列表并返回 invalid_source，不能假称执行过测试。

## 验收

本地合成收据覆盖正确、错误、异常、超时与错配，接入原 second_round；
检查所有 136 个合法 episode 的原始函数可解析并能构造公开请求。
这不是任务重跑：不执行初始函数、生成候选或评分程序。
不读取隐藏评分文件；不训练或加载真实模型。四个 readiness flags 仍为 false。
真实下一步仍是合格隔离执行器、4 个开发任务 base/text 两轮，然后才是
32 个迁移/保留任务；本组件不宣称完成任一远程评测。
