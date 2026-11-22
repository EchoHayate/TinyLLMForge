# 四题开发流程协调器

日期：2026-11-17。范围为 D01–D04 × base/text，两轮，各八个 episode，
最多十六次生成。不使用适配器、不训练、不打开迁移/保留输出。
本设计不授权远程启动；运行包和真实执行隔离仍需单独核验。

## 范围与结构

在已完成的提示、观察式生成、公开反馈、最终准入/执行和纯评分组件外，
增加一个小型串行协调器，而不是再建通用推理引擎或修改旧协议。
不复制题库，不让生成器访问最终评分计划，不将模型自己的判断作为答案。

先在 `development_schedule.py` 实现纯八身份排程和hash-only最终用例清单；
再在 `development_coordinator.py` 实现运行时协调，避免把冻结材料和
进程失败处理塞进同一个函数。排程的精确返回字段、ID算法和测试在
`../plans/2026-11-17-lora-development-schedule.md` 中固定。

运行时依赖通过三个可信调用方提供的
回调传入：generate、execute_public、execute_final。真实远程驱动将它们
固定绑定至已核验的 observer、首轮和最终准入执行器。合成测试可以提供
固定返回记录的回调，但这种测试不是实际模型或候选执行。

协调器只证明其调用次序、身份关联、反馈投影和归约逻辑。回调的真正身份、
模型完整性、worker 隔离、原始执行流和 SSH 传输真实性由固定运行包及
独立事后验证负责，不能由回调自报或一个 SHA 字符串代替。

## 固定顺序

调用原 `episode_jobs`，只筛选 D01–D04/base/text，核对八个唯一身份。
保留原始 messages 和 examples_sha256，不重新写提示。
顺序在任何输出产生前固定为：

1. D01 base、D01 text。
2. D02 text、D02 base。
3. D03 base、D03 text。
4. D04 text、D04 base。

每个 episode 连续完成两轮和最终评分，再进入下一个。没有跨题记忆、
自适应重排、已通过后的提前跳过、best-of-two 或生成重试。
新 run_id 和各 call_id/attempt_id 必须在持久化 claim 中绑定；
ID 由 run/task/arm/round/用途的规范 JSON digest 确定，不依赖输出表现。
最终逐例 attempt_id 采用最终评分器中已经冻结的算法。

启动前读取并核验 source/manifest/supervision/prompts 四个原冻结 hash；
只在可信协调侧持有私有源数据。提前固定 D-only 完整用例身份与数量、
运行顺序、调用上限和 source/runtime/model/tokenizer 绑定。
不能读完某题输出后再挑用例或替换任务。

## 每个 episode

1. 持久化 episode 身份和第一轮消息摘要，创建独占第一轮生成目录。
2. 调用 generate 一次，绑定 call_id、expected tokenizer、原始 messages。
   验证成功 claim/observation/generation 与返回值和保存字节一致：
   call ID、messages digest、tokenizer identity、候选原始 UTF-8/hash、
   一次 model call、输入/输出 token 限制、root forward 观察计数。
   用 observer 原 `_validate` 复用其既定检查，补足协调器身份关联。
3. 用原 `candidate_valid` 判定第一轮源码格式。普通 invalid_source 时
   不执行候选，以空 requests/receipts 调用原 `visible_feedback`。
   格式有效时使用原 `visible_requests` 的两条 P01/P02 请求，每条只经
   execute_public 一次，按顺序保留 admission、audit、receipt 和原始流。
4. 只将原 `visible_feedback` 产生的固定反馈传给 `second_round`。不放入
   原始 stdout、异常消息、私有 expected、终轮成绩或准入错误说明。
5. 第二轮始终再调用 generate 一次，即使第一轮公开用例全通过。完整
   第二轮提示仍由原生成器实际测量，超过4096输入 token 即中止；
   不截断示例、裁剪候选、缩短输出或重试。
6. 固定第二轮 generation.json 原始字节，交给 `build_final_plan`；
   候选必须精确来自此记录。语法无效则零执行并保留完整应测分母。
   有效则逐条 execute_final，不传任何 expected 值。全体最终回执交给
   `score_final`，成绩只存可信报告，不进入任何下一次模型消息。

每个生成目录和每个执行目录彼此独立，避免 claim.json/observation.json
同名冲突。调用前记录“准备调用”，调用成功后记录完整关联；不得把准备
调用或目录存在当成实际 model forward、实际候选启动或完整执行。

## 错误与持续执行

基础设施、准入、安全或身份失败立即中止整次运行。保留失败事件、可读取
的实际 claim/observation、原始错误和此前已完成结果；其余八个固定
episode 分母中的位置记 missing。不从新目录重新提交同一逻辑尝试。

普通第一轮 invalid_source 按规定仍走第二轮；普通 mismatch、合法
ValueError outcome 和已有执行器的 program_error/timeout/output_limit
不是基础设施错误。继续固定两轮流程，不额外增加一次纠错。
所有合法完成的开发 episode 可按原顺序收集，校准不通过只禁止后续
迁移/保留阶段，不把观察到的第一轮问题藏掉。

生成失败的调用不能返回假空候选。区分协调器回调已进入、生成 claim
已保存、实际 root forward 开始/完成、成功生成、输入/输出 token 已知。
失败 token 数未知则为 null，不填零。第二轮输入上限拒绝仍保留首轮
成本和第二轮真实失败证据，不能把它计为普通功能失败。

已提交生成记录后的临时文件清理失败，不把成功生成改写成失败并重试；
保留 record_cleanup。是否满足整次运行的清理检查单独报告。协调器
自己的终态发布遵循相同原则：发布成功后的维护错误另列，不撤销成功。
不增加用户已取消的聚合磁盘/内存预算或断线自动清理要求。

## 开发校准

功能结果与校准判断是不同字段。首轮无效而终轮通过，可以同时记录
`final_status=pass` 和 `first_round_invalid=true`，不能二选一覆盖历史。

校准采用以下保守顺序，在生成任何开发输出之前固定：

1. 已确认的协议/身份/隔离错误：INVALID_PROTOCOL；不作效果推断。
2. 缺失 episode、失败生成、未知必要成本或 timeout：
   INCONCLUSIVE_INCOMPLETE。
3. 首轮或终轮 invalid_source，任一轮 program_error/output_limit，
   或 base/text 两组均4/4通过，或两组均0/4通过：
   INCONCLUSIVE_CALIBRATION。
4. 完整且没有上述问题：DEVELOPMENT_CHECKS_SATISFIED。

第4种也不是难度充分、LoRA 有效、泛化提高或部署准入；不自动启动32题。
出现两组之一全对/全错，不等同于“两组同时全对/全错”。
保留各原因列表，不能因优先级只输出一个标签而删掉其他已知问题。
首轮无效但第二轮恢复仍触发第3项，这是保留解析预警的保守选择。

## 记录和成本

保存根运行 claim、八个计划身份、逐调用/逐用例关联、各 episode 结果、
最终开发报告和失败时的 partial report。独占发布，不覆盖旧记录。
首次失败不能被后续保存或清理异常覆盖；额外持久化错误另列。

输出应至少区分：scheduled episodes=8、scheduled generation calls=16、
completed episodes/calls、已核验 generation claims、观察到的 forward、
每轮语法状态、公开反馈机会数、实际公开/最终执行数、最终应测/已测/
通过/缺失用例、base/text各四题的功能结果、校准原因和缺失位置。

原生成 timing、observer elapsed、每次执行 audit elapsed、协调器外层
wall 分层报告；嵌套时间不重复相加当总时间。模型加载、共享准备和整体
外层成本由远程驱动另记。内存未知保持 null。不能从这四题的过程检查
给出 LoRA 速度、经济或任务收益结论；所有原 FLAGS 仍为 false。

## 验证范围

本地仅合成 callbacks 和标准 unittest，不加载真实底座、不解释候选代码。
覆盖八身份轮换、两轮固定调用、公开全通过仍第二轮、首轮无效恢复、
最终 invalid 分母、只公开反馈流向模型、私有 expected 不进入 callback、
各阶段失败不重试、身份/回执错绑、独占重复拒绝、清理/发布错误语义、
partial missing 及校准优先级。回调伪造出的成功不能标成真实模型实验。

通过组件测试和独立审查后，再准备新的远程运行包；不得复用已消费的
非任务 smoke 或准入诊断入口来运行这八个 episode。
