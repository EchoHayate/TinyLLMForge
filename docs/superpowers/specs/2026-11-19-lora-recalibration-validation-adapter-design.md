# LoRA 校准题库：新版验证适配设计

日期：2026-11-19，Asia/Shanghai。

状态：用户已认可“新增适配、复用底层”的方案；本文件待书面审阅。
尚未编写本设计对应的实现，也未执行新版题目或真实 tokenizer。
本文件不授权远程连接、私有材料传输、模型生成或训练。

## 1. 目标与范围

为已封存的 `recalibration_v2_r1` 增加两组独立版本的本地组件：

1. 固定源码验证：构造 480 个源码×用例位置，核对执行回执并判定数据是否一致。
2. 真实长度检查适配：渲染和冻结 120 行清单，调用既有 tokenizer 会话接口，
   重建完整计数报告。

本阶段实现和测试只使用本地 JSON/AST、静态投影和合成回执，不运行存储的
题目函数，不启动真实 tokenizer。通过后再审阅具体隔离启动包与一次性范围。
实际远程检查仍是最终目标，不能用本地组件通过冒充已完成实际检查。

不实现新版模型生成 driver、LoRA 训练或效果对照；不改变题目、答案、评分、
八个监督示例或 token 预算；不重放旧实验，不恢复旧 session。
不安装依赖、创建分支、提交/推送或清理既有文件。

## 2. 方案选择与不变部分

选择新增独立适配层，复用 `validation_remote.py` 的固定源码后端和
`tokenizer_session.py` 的受信会话作为后续执行候选。

- 不直接修改旧 v1 的 pins、题号、1416/324 固定分母或消息协议：
  这会扩大旧入口契约并影响历史证据。
- 不另写隔离器或 tokenizer 计数核心：本轮缺的是数据和报告接口，
  不是重新实现已存在的底层能力。
- 不把 C 题改名成 D/T/R，也不把 H 用例伪装成 P01/P02。
  `candidate_execution_protocol.py` 与 AST admission 不作为本次固定源码入口。

此前静态兼容性记录：
`experiments/lora_necessity/20261119-recalibration-validation-compatibility.md`。
远端主机、解释器、隔离能力和已安装环境在本轮均未实时核验；
历史成功不能替代新执行包的核验。

## 3. 冻结输入与模块边界

题库根目录为 `experiments/lora_necessity/data/recalibration_v2_r1/`：

| 文件 | SHA256 |
| --- | --- |
| manifest.json | `2d8bc46cb387ca80948aea6a9c9a42264e66560666bc515dbc3cf79fde1e9690` |
| prompts.json | `0b8b037017edda84c28afdf0b7a6dd547d6bfd50582a60287c733f58f6ac1e38` |
| oracles.json | `a07803a677db263e40e71423d0f1c423c093e7a26f99939274fdcd91ddef4be2` |
| structure.json | `b5ede6dd62e380530fb1ce158a0e2aefb2bb6690f0a2dcd3ee2b3d3bea729e6e` |
| review.md | `012f49362d11547b7d4cf11baf3341210a8e9dc102a425b33f94a45b49a8b2fc` |

八例只来自 `experiments/lora_necessity/data/supervision_v1.json`，
SHA256 为 `4dafb0ac759697e9b80c5bf097b363082a17a34e6b78fb3d3b91d37a76beb78a`。
不修改以上文件，也不更新其历史 pending/false 字段。

新增模块均位于 `experiments/lora_necessity/`：

| 模块 | 单一职责 |
| --- | --- |
| `recalibration_data_contract.py` | 校验固定字节、严格 JSON、跨文件绑定、C01–C12 和源码/用例/见证身份 |
| `recalibration_validation.py` | 480 项排程、最小 worker 投影、批次绑定、纯回执重建与数据判定 |
| `recalibration_rendering.py` | 只接收公开任务和固定八例投影，渲染 base/text 及两种续答 |
| `recalibration_tokenizer_inventory.py` | 绑定通过的数据报告、构造 120 行私有清单并校验身份/顺序 |
| `recalibration_tokenizer_report.py` | 独立的新版行校验、测量/结束回执校验和纯报告重建 |
| `recalibration_tokenizer_audit.py` | 顺序协调既有会话的 measure/finish/abort，不负责网络或进程启动 |

每个模块配对应的本地测试；不为复用而修改旧模块或调用旧固定分母的报告入口。
公用 canonical JSON、摘要及严格解析可复用已有无副作用函数。
模块导入不得加载 tokenizer、模型、远程执行器或运行题目源码。

JSON 拒绝重复键、非有限数、错误类型和额外字段。固定输入先核对原始字节
摘要，再核对结构，不能在解析重排后才验证原始身份。
检查源码只允许 AST/签名检查，不 import/exec 存储的源码。

## 4. 固定源码排程和数据判定

### 4.1 完整身份

固定顺序为 C01–C12；每题依次：
`reference_a`、`reference_b`、`initial`、`variant_1`、`variant_2`；
每份源码依次 P01、P02、H01–H06。ordinal 为 0–479。

每行绑定题库摘要、task ID、source role、case ID、入口、源码/输入/期望摘要、
适用时的见证预测，以及该行 canonical 摘要。
排程用独立 kind `LORA_RECALIBRATION_VALIDATION_PLAN` 和 version=1；
其中 data version 为 `recalibration_v2_r1`，不能与旧 v1 排程混用。

核对 prompts 与 oracles 的公开用例、入口和 initial_source；
每题严格有两参考、三错误源码、八例，且 36 个见证均有效并区别于正确期望。
initial 的指定见证必须是公开用例。

### 4.2 最小投影与批次

worker 仅收到 `request_id/source/entry_point/args`；
`request_id` 固定为 `request-0000` 至 `request-0479`。
worker 不收到正确答案、见证预测、参考/错误角色或全量 oracle。
每次隔离调用只能看到当前一项投影，而不是整份题库。

按排程连续分成 15 批、每批 32 项，不打乱、不重试、不去重。
本地批次包装绑定独立 run ID、批序、ordinal 区间、排程/包/worker 摘要；
worker request_id 在本次 run 内唯一，跨 run 依靠外层身份区分。
封包时重新检查源码 4096 字节、请求 8192 字节、worker 16384 字节和
整体包 131072 字节的现有上限，不能只凭单项大小推断整包可用。

纯组件只产生计划字节并核对留存回执，不调用 SSH/subprocess。
现有后端的单项 2 秒与批次 90 秒边界不在本设计中放宽；
远程认证、隔离探针、传输期限、排他启动记录和完整进程生命周期
由后续执行包设计核验。不能省略这些检查就把后端直接投入新版运行。

### 4.3 结果与停止

回执必须与批次 READY/done、输入包和预定行对应，并验证退出/回收、完整输入、
输出格式、顺序、无重复、无多余记录和停止位置。整个结果由本地从原始
受信采集记录重建，不接受调用者只提交一个 PASS 布尔值。
JSON 结果采用严格类型与 canonical 字节比较，不能让 Python 的
`True == 1` 或类似宽松比较消除答案差异。

正常返回但与正确答案不同，是语义观察，不是基础设施失败：

- 192 个参考位置必须全部符合正确期望。
- 36 个错误源码的具名见证必须符合预写错误预测，且区别于正确期望。
- 其他错误源码位置记录实际结果，不要求也全部失败或符合未给出的预测。
- 两参考不一致、参考不符或见证预测不符时，数据检查判失败；
  若进程与证据正常，继续剩余固定位置以完整展示差异，不换题重跑。
- 连接、身份、隔离、回执绑定或进程执行异常时停止；后续位置明确未执行，
  缺失不能算通过。无法证明是否执行的已投递位置单列为未知，不自动重投。

报告保留完整 480 个位置，以及参考、见证、各题完成度和首个停止原因。
`matrix_complete` 仅表示所有位置有完整有效结果；
`data_checks_satisfied` 还要求上述参考和见证条件全部满足。
两者不等价，也不代表模型正确率、任意代码隔离能力或 LoRA 收益。
合成回执测试只验证上述判定逻辑，不提供真实执行来源证明。

## 5. 消息渲染与 120 行长度清单

### 5.1 公开渲染

首轮沿用旧 `INSTRUCTION` 的准确文本及 canonical ASCII JSON 格式。
当前任务只投影 `entry_point/id/initial_source/public_cases/spec`。
text 的八例按 S01–S08 排序，逐字段沿用旧 `_examples` 的投影：
上述公开字段，加原 `initial_witness` 的 public_case_id/predicted_initial，
以及原 `reference_source`。冻结完整八例投影摘要，不删减或替换。
该投影的 canonical ASCII JSON 为 7514 字节，SHA256 为
`e1634ad790ed551484a91fe55db36fc2f74b9615d2e5f273989552ea4f6cb290`；
这只是嵌入提示的八例 JSON 大小，不是完整消息大小或 token 数。
base 不含八例；C 题参考答案、H 用例、推导和错误变体不进入提示。

两种续答共用同一首轮消息和完整 initial_source assistant 消息。
共同修订指令固定为：

`Return the final complete function, even if you believe the first candidate is correct.`

末条 user content 为共同指令、换行、canonical JSON 结果对象。
feedback 使用本次完整通过的动态报告中 initial 的 P01/P02 实际结果，
投影为 `visible_tests`、case ID 和 pass/mismatch 状态；
不包含隐藏用例、实际私有值或错误预测。retry 使用
`{"kind":"results_withheld"}`，不含执行结果，也不使用旧的
“Public-test feedback follows”。

实际构造可运行的固定续答清单，要求完整 480 项报告的数据检查通过，
并绑定原始证据摘要；不能只凭手写状态或见证预测构造“真实反馈”。
合成回执可用于本地测试，但测试输出不是可投递的真实清单。
若数据检查未通过，长度阶段标为依赖未满足，不把合成清单送入真实审计。

### 5.2 行数和顺序

| 类别 | 行数 | 测量内容 |
| --- | ---: | --- |
| first | 24 | 12 题 × base/text，只有首轮输入 |
| references | 48 | 12 题 × base/text × reference_a/reference_b，首轮输入及完整答案含 EOS |
| sampled_second | 48 | 12 题 × base/text × feedback/retry，完整续答输入 |
| 合计 | 120 | 不合并相同文本的行，不增加训练消息或模型生成 |

类别依上表顺序；每类内部 C01–C12，奇数题 base/text、偶数题 text/base。
references 的每个 arm 内 reference_a 在前、reference_b 在后。
sampled_second 在 `(题序号-1) mod 4` 为 0/1 时 base 先 feedback 后 retry、
text 相反；为 2/3 时交换，以沿用上位设计的分叉顺序。
这里的 120 是清单行数，不是 tokenizer 内部方法调用次数。

各行绑定 ordinal/category/task/arm，适用时的 reference ID 或 branch，
messages/answer 摘要、样例 initial 源码摘要、公开反馈投影摘要和行摘要。
完整清单另绑定题库、八例、动态证据和渲染实现摘要。
清单 kind 为 `LORA_RECALIBRATION_TOKENIZER_INVENTORY`，version=1，
data version 为 `recalibration_v2_r1`；严格拒绝旧 324 行格式。

references 中的答案只作为受信计数器的独立 answer 字符串，不放入原任务
user 消息；计数器可为计算完整序列添加 assistant 答案，但不执行它。
不上传全量 oracle。48 行参考组合不意味着有 48 份不同参考源码。
私有文本和 token IDs 只进入私有记录；公开报告只给身份、计数和状态。
具体私有投影传输仍需后续启动范围明确批准。

## 6. 分词会话与报告

复用 `tokenizer_session.open_session` 及其 measure/finish/abort；
运行 profile、快照、guard 和计数核心不变。保留
`tokenizer_runtime_profile.py` 的 Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`、五文件 pins、
Python 3.12.13、Transformers 4.57.1 及完整已安装版本清单约束。
不自动下载、换 tokenizer、加载权重或启用 CUDA。

`enable_thinking=False`，输入最多 4096 tokens，答案含唯一 EOS 最多
256 tokens。保持两路模板 token IDs 一致、完整消息/答案字节不变、
输入前缀边界准确、EOS 后模板尾部单独记账。禁止截断或改写答案凑通过。

新版 report 校验自己的 120 行 schema，不调用旧 `_row` 或固定 `_ORDER`。
同时保留现有证据要求：run/profile/输入身份、每行摘要链、身份连续性、
实际 token IDs 与计数、测量状态、快照/运行环境、最终结束记录。
本地必须能由私有原始清单、profile、120 条测量和 finalization 重建报告，
不能只信任远端返回的汇总。

第一处超限或测量/身份/回执错误即停止，余下行标记未检查。
正常超限需 finish 保留状态；异常需 abort 并保留失败和释放结果。
未完成的结束记录不能按成功处理，不自动重开会话。
报告分别列计划、已检查、未检查、超限、停止原因和会话结束状态。

只有完整 120 行有效且未超限，并完成可信采集和本地重建后，才报告本次
固定清单长度通过。旧清理要求不自动恢复；不得伪造 cleanup 状态，
实际启动范围按既有用户约束单独审阅。
固定续答使用题库 initial_source，而非尚未生成的模型答案；
`all_possible_second_rounds_validated=false` 始终保留。
真实模型续答日后仍需逐次做输入长度守卫。

## 7. 本地验收与后续执行边界

本地测试必须覆盖：

- 固定 pins、完整分母、重复/缺失/乱序/越界身份、初始源码和公开用例跨文件绑定。
- 480 行和 15 批投影确定性；正确答案和角色不进入 worker 请求；
  未投递、投递结果未知和已观察结果不混为一类。
- 参考 192/192、见证 36/36 的正反例；错误源码正常错误答案不触发执行故障；
  bool/int 混淆、参考分歧、预测错误与回执异常均有明确判定。
- 八例字节保持；base/text 信息边界；共同续答指令及 results_withheld；
  feedback 只来自绑定的公开动态结果，不泄漏 H 用例和答案。
- 24/48/48 行顺序、消息与答案绑定；旧 inventory/报告拒绝；
  4096/4097 输入及 256/257 含 EOS 边界、身份漂移、摘要链错误、
  停止后追加记录、缺失 finalization 与 abort 失败。
- fake session 的生命周期及无 tokenizer/模型/SSH/源码执行副作用；
  合成证据不得当作真实执行通过。
- 旧 v1 相关回归保持原语义；所有题库封存字节和旧源码保持不变。

实施结束需独立审查新增组件及测试，记录实际运行的测试命令与结果；
不把本规格中的测试清单写成已经通过。

实际检查前还需单独审阅精确启动包：本轮新身份和根目录、主机/UID、
解释器与环境、backend/worker 摘要、隔离预检、480 项包摘要和传输边界、
120 行私有清单与 tokenizer profile、期限、停止和原始回执收集。
纯组件验收不免除启动适配或再次审阅；不重用旧 claim/run 或自动重试。
本设计不恢复已取消的总内存/磁盘预算和自动清理前提。

本阶段结束只表示新版验证组件准备就绪；只有后续真实回执核验才能分别
更新新的动态结果报告和长度结果报告。原始封存文件不回写。
`benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`、
通用 `isolated` 标志保持 false，不自动推进训练或模型对照。

## 8. 本次设计自查

2026-11-19，主控完成文档自查；不是独立代码审阅或实际执行验收。

- 已重新核对五份题库文件和原监督文件摘要，全部与本文一致。
- 已核对 S01–S08、7514 字节八例投影及其摘要；
  480/192/36 和 24+48+48=120 的计数关系成立。
- 固定源码执行与真实模型生成分开；正常语义差异与执行/证据异常分开。
- 真实 feedback 依赖本次动态回执，未把手推或合成测试当实际观察。
- 本地实现、书面审阅和未来远程启动分开；具体启动包不在本次许可内。
- 未发现占位项或相互冲突的预算；原封存和 tracked/index 均未改动。

尚待用户审阅本文件；获认可后进入实施计划。当前没有新增实现或测试结果。
