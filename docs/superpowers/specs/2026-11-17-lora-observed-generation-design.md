# 生成调用的外层观察与失败记录

日期：2026-11-17。按用户明确要求连续实现、修复和验证，不逐步等待确认。

## 范围与选择

保留现有 `generation_core.py` 不变。选择新增一个小型调用观察封装，
而不是重跑训练来证明生成，也不把任务执行、模型加载和评分塞进生成组件。
它供下一次真实底座的非任务生成检查使用，然后才能成为开发流程的组成部分。
不改变 greedy、256-token 上限、4096 输入上限、禁用 cache 或两轮协议。

## 接口

新增 `observed_generation.py`：

`generate_observed(model, tokenizer, messages, *, expected_tokenizer_sha256,
call_id, artifact_dir)`

`call_id` 为 32 位小写十六进制。artifact_dir 必须已经存在。
模型和 tokenizer 由可信调用方提供；封装不加载模型、不连接远程、不选任务，
不读取监督或隐藏评分文件、不训练、不挂载 adapter，也不执行输出文本。
本地仅使用合成模型、合成消息或可信测试替身。

只调用一次原有 `generate_candidate`。对消息做独立快照，并在启动前以
0600/独占方式保存 claim：kind、call_id、messages_sha256、
expected_tokenizer_sha256。claim 是尝试已准入的证据，不证明远端硬杀前
一定执行过一次 forward。既有 claim 禁止重试或覆盖。

使用模型根节点的 forward 前后 hooks 分别记录观察到的入口和返回事件，
不统计内部子模块，不修改参数、输入或输出。它们不是失败时普遍精确的
forward 函数体完成数或 kernel 数：前置 hook 排在既有本地前置 hooks 后，
后置 hook 排在既有本地后置 hooks 前，但全局后置 hooks 仍先于本地后置
hooks。既有前置 hook 抛错可产生 0/0；函数体已返回、全局后置 hook 抛错
可产生 1/0。不得据此认定函数体没有执行、没有返回或没有成本。
结束时移除自己添加的 hooks，保留原有 hooks。安装 hooks 失败也保存错误
并移除已安装的部分。

## 结果与失败

成功返回 `generation`、`observation` 和独立的 `record_cleanup`；
其中前两项分别独占保存为 generation.json 和 observation.json：

- generation 是原组件完整原样记录；
- observation 包含 kind、call_id、status=completed、messages_sha256、
  forward_started、forward_completed、elapsed_ns、failure=null；
- 两个 forward 计数必须等于组件的 forward_calls，且它等于 output_tokens；
- 检查消息/tokenizer 身份、文本 SHA256 和 model_calls=1 与结果一致。

任何 core/hook/绑定异常，以及记录发布前的创建、写入、关闭或 link 异常
都中止，不产生模型反馈、不截断输入、不再生成。
尽力保存 failed observation，包含可观察的两个 forward 计数、已过墙钟、
有限长异常类名/信息，输出 token 数未知时不写零。失败后抛出
GenerationAttemptError，并通过异常链保留原始原因。
错误记录保存也失败时必须显式抛出，不假装已有完整失败回执。

### 发布边界与临时文件清理

记录先写入并关闭 0600 临时文件，再以不覆盖既有文件的 hard link 发布。
成功 link 是单个记录的发布边界；只有 `status=completed` 的
observation.json 成功发布才是整个尝试的成功提交点。generation.json
单独发布不构成成功提交。发布后的临时名字 unlink 是独立的尽力清理，
不改变已发布的终态；它失败不能使已提交成功变成异常，也不能把已发布的
failed observation 说成未保存。发布前的主要异常也不能被 unlink 异常覆盖。
不重写终态、不重试生成，也不增加提交后的必须成功的日志写入。

返回值的 `record_cleanup`，或失败异常的同名属性，固定包含
`failed_count` 和 `failures` 列表。每个清理失败条目包含 `record`、
`temporary`（仅临时文件名）、`published`（该次 link 是否成功）及
有限长 `error.type/message`。正常值是 0 和空列表。这计数的是本次调用
未解决的临时名字清理失败，而非之后某时刻经扫描核实的残留文件数；
通常每次失败留下一个 0600 临时名字。清理不重试，异常不会用 warning
抛出（包括 `-W error` 模式），也不会把临时清理失败混入模型/core 失败。
hook 移除失败仍然是提交前的致命错误，不适用此非致命清理规则。

这些诊断是调用方可见的独立结果，不注入原样 generation 或已发布的
observation，不声称落盘记录包含其自身发布后的清理结果。若调用方需要
归档，应把它们作为非致命维护信息；维护/恢复读取者可枚举
`.generation.json.*` 和 `.observation.json.*` 统计实际残留，不能把
临时文件当作终态，不能用残留否定已发布的完成记录，不能因此重新生成。
observation 的 elapsed_ns 截止其构建时刻，不包括其自身写入/清理，
也不是全链路耗时。此提交边界不提供掉电 fsync 持久性保证。

在进程硬杀或断连时可能只留下 claim 或 generation.json。外层监督器必须把
无已发布终态 observation 的尝试标为不完整、实际 forward/token 数未知；
有效的 failed observation 表示失败，completed observation 配合匹配的
claim/generation 才表示完成。任何这些情况均不得据此重试或报告零成本。
本封装不声称能处理 SIGKILL，也不替代远端父进程截止时间。

## 验证

先写缺失组件 RED，再实现 GREEN。测试正常合成返回、forward 中途错误、
forward 已返回但解码失败、安装第二个 hook 失败、原有 hook 保留、
消息别名隔离、结果身份/计数错误、重复 claim、非法 ID、记录写入失败。
每条失败路径验证只调用一次或启动前未调用，且移除自己安装的 hooks。
独立审查 P1 修复先复现 RED：分别在 generation/completed observation/
failed observation 发布后注入临时 unlink 失败；覆盖发布前 link 失败、
write/link 与 unlink 同时失败，保证主要原因不变、残留清理计数完整。
验证 generation 已发布但 completed observation 发布失败仍是失败；
验证全局后置 hook 抛错时函数体已返回但观察计数为 1/0，以及既有前置
hook 抛错时为 0/0；失败仍不填写未知 token/model-call 总量。
独立审查后才能用于新的真实底座非任务生成检查；本组件通过不等于开发题、
LoRA 效果或 RSI 已通过。
