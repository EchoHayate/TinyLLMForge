# LoRA 新校准：固定源码真实执行与证据绑定

日期：2026-11-20，Asia/Shanghai。

状态：承接用户对分阶段方案的“继续”；本书面规格待审阅。
本阶段只设计第一段固定源码检查，不代表新启动包已完成或真实运行获准。
实现、测试、实时预检、实际 480 项执行分别验收，不相互替代。

## 1. 目标和不做的事

将已有 `recalibration_validation` 的 480 项排程和纯报告接到一个新的、
单次使用的受信控制器。保留原始输入、真实进程观察和回执，使本地能够
重建“执行了哪些项、结果是什么、哪些未知”，而不是接收一个远端 PASS。

固定 12 题，每题两参考、三错误源码、八用例；共 480 项、15 批、
每批 32 项。参考分母 192，具名错误见证分母 36。首批 ordinal 0–31
只覆盖 C01 的前 32 项，不把它误记为一整题或另加一次试跑。

不做 tokenizer 120 行检查、模型生成、LoRA 训练/加载或效果对照；
不运行旧 D01、旧 1416 项或旧 324 行；不执行任意生成程序。
不更改冻结数据、旧源码、六个已完成新版纯组件、评分或 token 预算。
不安装、遥测、切分支/worktree、提交/push、清理旧文件或恢复旧 session。
本地只允许 JSON/AST、纯组件及人工合成回执；题库源码不得在宿主执行。

上位依据：

- `docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`
- `experiments/lora_necessity/20261120-recalibration-launch-readiness.md`

## 2. 已选方案和模块职责

采用“新启动外层 + 原后端/纯重建”。不改旧版分母/入口，不翻转
synthetic-only 标志，也不重写 bwrap、seccomp 或 tokenizer 核心。
这是上一轮三种方案中的第一种；其余两种扩大历史兼容或重复实现范围。

新增模块限于以下四种职责，路径均在 `experiments/lora_necessity/`：

| 拟新增模块 | 职责 |
| --- | --- |
| `recalibration_execution_package.py` | 固定输入、依赖、排程和 15 个精确请求包；校验启动描述，不启动进程 |
| `recalibration_execution_remote.py` | 固定可信 bootstrap/监督程序，验证输入，管理一个 backend 子进程并返回原始字节和结束证据 |
| `recalibration_execution_controller.py` | 本地独占 claim、有界 SSH 传输、串行批次、停止与持久化；不执行题目、不替代评分 |
| `recalibration_execution_evidence.py` | 从保存字节重建传输/批次绑定和原报告；不连接、不启动、不信任汇总 PASS |

每个模块配聚焦测试。生产入口不接受任意 `session_factory`、
`transport_callback` 或一个 `real=True` 来获得真实来源结论。
测试可以替换内部进程/时钟边界，但模拟产物明确为合成证据，
不作为生产启动凭据或真实结果报告。

复用的后端和 worker 当前 SHA256：

- `validation_remote.py`：
  `9be98748dfd2bdcdd132c6ed4c04c3dd67f11d49d6049fcbe7960bde622690da`
- `validation_task_worker.py`：
  `d2c8095b4d1e854a2f497b084226a92b47aa8b083d59c8ab272b088544fedca4`

复用 `build_batches`、`build_report` 和严格 JSON/摘要函数；
不得从本地 report 的恒 false 字段推断本批没执行，也不得改写这些字段。
旧对象表示内容重建结果，新外层另行表示来源绑定和本次执行结论。

## 3. 输入冻结和许可边界

### 3.1 本地材料

六份固定数据仍逐字节匹配 `recalibration_data_contract.FILE_PINS`。
本地保存完整数据和原始 manifest，派发侧只导出四字段请求：
`request_id/source/entry_point/args`。正确期望、见证预测、角色、推导、
全量 oracle 和完整数据报告不进入远端包。

新 `stage_id` 和 15 个不同 `run_id` 都是 32 位小写 hex；
stage ID 与所有 batch ID 也不得相同。不能沿用任何旧消费身份。
整个本地 package 绑定：

- kind/version/data_version、stage ID、固定顺序的 15 个 batch ID；
- 六文件、plan、backend、worker 和每批原始 bundle 的摘要、字节数；
- 本次控制器、监督程序、package/verifier 及实际本地依赖闭包的源码清单；
- 精确 SSH argv 前缀、远端解释器、预期主机/UID、规范绝对路径；
- 原始实时预检记录摘要、严格 `expected_runtime`、期限和输出边界；
- 本地私有证据根、远端新 stage 根、每批独占路径和精确启动命令摘要。

本地依赖清单至少包含 `pilot_protocol`、`validation_contract`、
`validation_report`、`validation_real_report`、`recalibration_data_contract`、
`recalibration_validation` 及实际加载的其他本项目模块；实现阶段从真实
导入闭包核验，不只哈希四个新文件。worker 作为字节材料处理，不在本地导入。
完整本地清单不等于远端源码清单：远端不需要评分和数据加载模块。

所有 JSON 拒绝重复键、非有限数、额外字段、错误类型和 bool/int 混淆；
字节摘要先于解析验证。摘要字段使用 64 位小写 hex。
未具备上述实测字段时只产生非可执行准备记录，不产生“可运行 manifest”。

### 3.2 实时预检不是历史报告复制

实时预检需另外审阅其确切命令和人工探针，且不执行题库。
该步骤必须观察主机/UID、解释器与 bwrap、seccomp、命名空间、
禁止提权、只读挂载、宿主路径不可见、socket/fork/clone3 拒绝，
以及超时、输出上限和直接子进程回收。
只对假文件及人工固定程序探测，不读取真实凭据、不做压力或漏洞测试。

`expected_runtime` 与当前 `validation_remote.run_bundle` 输出完全同形，
包括 hostname、uid、python、python_root、python_sha256、bwrap_sha256、
seccomp_sha256、argv_prefix、no_new_privs 和 host_namespaces。
它不是 tokenizer 的版本 profile，不用历史版本或测试 fixture 填充。

其中 `argv_prefix` 包含 seccomp FD 参数。预检与实际监督启动必须采用
相同 FD 布局；不能为消除误差而删除、归一化或忽略该字段。
命名空间、FD 或其他 runtime 字段漂移时拒绝执行，先解释实际差异，
不自动接受新的期望值。预检通过不等于任意代码安全。

原路径缺失的历史 SSH/profile 材料不作为当前可用证据。
精确连接路线需重新取得并审阅；保留严格主机密钥校验、原身份认证、
禁止凭据委派/agent 转发/连接复用/自动重试的边界。
不猜 argv，不修改全局 SSH 配置、known_hosts 或凭据。

完成本地实现不授权该预检；完成预检也不自动执行题目。
实际派发前，主控核验精确 package 摘要与独立审阅的一次性范围相同。
摘要参数是完整性校验，不是软件可自行产生的用户授权。

## 4. 一次批次的可信链路

### 4.1 本地 claim 和私有投递

本地新根 0700、记录 0600；排他创建，拒绝符号链接、路径逃逸、
不匹配的 owner 和已有 stage/batch claim。远端仅在已审阅父目录下
建立对应独占 stage/batch 路径，不触碰旧根。
第一次成功建立 stage claim 后即消费整段运行身份；
失败或崩溃不能通过换目录、换 ID 自动重试。

每批先重新核对源码/数据/包字节和剩余期限，持久化 batch claim，
再调用一次 SSH。claim 保存时间、stage/batch/package/command 绑定，
并在持久化成功前禁止任何投递。保存失败立即停止，不能继续下一批。
生产只使用已固定的真实进程入口；凭据和完整内部 argv 留在私有记录。

远端可信 bootstrap 经严格 SSH 认证后先核对主机/UID、解释器隔离参数、
本次配置和独占 batch 身份，发出 `READY`，控制端核对后才发送任务 payload。
bootstrap 是已审阅的可信程序，不包含题目源码或评分数据。
任务 payload 通过 stdin 发送，不拼进 shell 命令或普通日志。

### 4.2 远端监督与原后端

可信监督程序先收齐有限长度 payload，拒绝截断、多余字节和摘要不符；
只在这些检查全部通过后启动 backend 子进程。
远端源码包只包含精确 backend 字节、当前批 worker/request bundle，
以及监督所需的固定配置；不让整个本地仓库进入远端解释器搜索路径。

backend 仍执行原 `run_bundle(bundle_raw, expected_runtime=...)`，
mode 为 `matrix`，保留其严格运行时比较及逐项 bwrap 启动。
backend/worker 字节在远端执行前再次哈希。worker 的存储题目源码只在
对应隔离子进程中解释；可信监督层和 backend 层均不直接调用这些函数。

新增监督层拥有 backend 的 PID/进程组并采集其 stdout/stderr。
它返回实际启动结果、完整输入状态、退出码、回收状态、耗时及原始输出摘要；
异常时停止自身创建的 backend 进程组并做有界回收，不操作其他任务。
backend 每行既有回收记录与监督层 backend 回收是不同层，均须保留。
监督层不得把“kill 已发送”写成“已回收”。

### 4.3 有界线格式

采用本版本独立的五帧顺序：
`READY / BACKEND_STDOUT / BACKEND_STDERR / DIAGNOSTIC / DONE`。
固定 magic 为 ASCII `LRV01`，头格式为网络序 `!5sBQ`；
channel 为 1–5，长度按 unsigned 64-bit 编码。
拒绝乱序、重复、未知通道、截断和最后帧之后的任何字节。
不调用旧 tokenizer bootstrap 或其固定清单/report 验证入口。

BACKEND_STDOUT/BACKEND_STDERR 携带原始 bytes，不二次 JSON-hex 嵌套；
据此在本地重建旧 `_process_result` 所需的 hex 字段，原 bytes 另存。
READY/DONE 是严格 JSON。READY 绑定 stage/batch/run、package/config、
主机/UID和独占根；DONE 再绑定 READY、payload、全部前帧摘要及监督观察。
DONE 只能在 backend 退出/有界回收尝试之后产生，不能先发成功再等待子进程。
失败 DONE 保留失败状态，不能要求必须成功才允许留存回执。

单帧上限：READY/DONE 各 4096 bytes；BACKEND_STDOUT 8 MiB；
BACKEND_STDERR 128 KiB；DIAGNOSTIC 64 KiB。
同时保持本地 wire stdout 总量最多 8 MiB、SSH stderr 最多 128 KiB，
包含所有头和帧；单帧最大值不保证能同时占满。
总量先到上限同样停止并保留证据，不截去诊断后假装完整成功。
所有私有 stdout/stderr 保留原始字节，公开报告不含任务源码或内部身份。

## 5. 期限和顺序停止

| 层 | 固定边界 |
| --- | --- |
| 单项 worker | 沿用墙钟 2 秒、CPU 2 秒及既有文件/描述符/输出限制 |
| 原 backend | 每批 90 秒，不放宽 |
| 新远端监督工作 | 从 READY 发出前的固定时点起最多 95 秒，含输入等待、backend 启动/执行；有界回收另至多 2 秒 |
| SSH 传输工作 | 每批最多 110 秒；本地有界回收另至多 2 秒 |
| 整段 stage | 从 stage claim 后的固定单调时钟起，最多 1800 秒；末尾保留 5 秒收尾窗口 |

stage 最后 5 秒不再启动或继续工作，只作有界退出/证据收尾。
每批与远端 grant 使用实际剩余预算，不重置 stage 期限；
READY 后若不足远端所需 95+2 秒及传输收尾余量，就不发送 payload。
本地/远端不比较绝对 monotonic 时间戳，只传剩余时长。
远端以发 READY 前的自身时钟为锚点；grant 只能缩短原 95 秒期限，
收到 grant 或 payload 不刷新该锚点，网络传输时间也消耗这个窗口。
backend 的外层采集最多使用当时剩余工作时长；原 backend 内部 90 秒
上限不变，外层剩余不足时提前停止而非放宽。原采集器至多 2 秒的回收
留在工作窗口之后，计入上述 97 秒需求。
控制端发送 grant 时为远端 97 秒之外另留 5 秒传输余量，不足则不投递。
READY 等待也计入 SSH 110 秒；远端 bootstrap 在自己的进程启动后
至多 10 秒内发 READY，之后输入等待受同一个 95 秒锚点约束。

这些是进程安全边界，不是恢复用户取消的总研究时间预算、
聚合内存/磁盘配额或自动清理要求。期限超出、时钟倒退/失效、
输出溢出、取消、回收不确定或持久化失败均停止。
取消先尽力保存和有界回收，再传播取消信号，不吞成普通成功返回。

15 批严格串行。只有当前批原始证据验证完成、backend 与 SSH 的结束
证据有效、该批 32 项正常完成，才进入下一批。
参考/见证的正常语义不匹配不阻断后续批次：完整展示固定数据差异。
程序错误、异常、超时、身份或回执错误则立即停止后续批次。
不在失败后补跑，也不重复首批作为“正式”执行。

## 6. 派发未知、原报告和真实外层

保留三套相互独立的信息：

1. 派发：未发送、已尝试发送、远端已收齐、backend 已启动；
   字节发送到本地管道不等于远端收齐，claim 本身也不等于任务启动。
2. 内容：480 项的原始观察、参考匹配、见证匹配及缺失；
   由原 `build_report` 重建，不由远端声称的计数填充。
3. 生命周期：backend 退出/回收、SSH 退出/回收、远端状态是否已知。

映射到原 report 的 `records` 只使用已尝试派发的连续批前缀：
如果没有可验证的 backend 采集记录，最后一批使用 `transport=None`，
保守保留 `unknown_after_dispatch`，其余为 `not_dispatched`。
新外层可用额外确凿证据说明“任务未启动”，但不回写原 report 的状态，
不删除失败记录，也不把未知项改成模型答错或正确。

完整 DONE 且 backend 退出/回收、输入、帧和 ready/row/done 校验成立，
才能确认批次结束。SSH exit 0、某行 `completed`、接收到了全部 32 行
都不能单独取代监督结束证据。断线后禁止自动补连接查询或重投；
若要取回已有远端证据，单独审阅只读范围，不能先假定没有执行。

新外层报告的 kind 为 `LORA_RECALIBRATION_EXECUTION_EVIDENCE`、version=1。
包含 package/stage/源码/预检/原始记录摘要、原 `content_report`、
每批派发和生命周期、首个停止原因，以及三个分开的结果：

- `binding_complete`：提供的原始材料与包/声明绑定自洽；
  纯离线验证能够得到此值，不能仅凭它宣称真实执行。
- `trusted_collection_accepted`：只有主控核对受信真实控制器的实际启动、
  本地原始记录及独立回执审阅后才接受；不是可由调用者传入的布尔开关，
  离线验证器和合成测试不自行设为 true。
- `dynamic_checks_passed`：在真实采集已接受的前提下，内容报告
  matrix_complete/data_checks_satisfied 全为 true。

控制器持久化的原始外层报告与主控最终接受记录分开保存：
前者的后两项在验收前保持 false；接受记录绑定其摘要和审阅依据，
不覆盖封存报告。真实研究结论从这份独立接受记录读取。
普通 hash/claim 并非第三方密码学执行证明，信任边界仍是当前受信主控、
严格认证的远端和已审阅程序；受信记录若被任意伪造，纯解析器无法识破。

失败的真实采集也可被接受为真实的失败证据，但不能通过动态数据检查。
即使动态检查通过，`tokenizer_length_validated`、benchmark/formal
evaluation/net benefit/通用 isolated 仍 false。原报告的恒 false 字段
不改写；“完整运行”和“数据检查满足”分别报告。

## 7. 本地验收和实施顺序

本地验收只用人工固定小程序、人工进程回执和假传输；
不执行存储题目或真实 SSH，不导入 tokenizer/model，不安装测试依赖。
实现顺序：package → evidence verifier → remote/controller → 集成审阅。
先写失败测试，再实现；不重复六个已完成组件的开发。
沿用现有标准库测试方式，不运行会联网的 bootstrap/flush。

新增测试必须覆盖：

- 15 包/480 项确定性、全量包上限、错误 role/expected/oracle 不进入远端；
  任一数据/源码/命令/runtime/payload 变化、重复 run ID 都拒绝。
- 包不完整、原始预检缺失或仅有旧报告时无法进入真实启动准备完成状态；
  单一 `approved=True`、`real=True` 或摘要自洽不构成授权/来源验收。
- stage/batch 排他 claim、输入前 READY 绑定、少字节/多字节、
  额外字段、重复/乱序/超量/截断帧、DONE 摘要/退出码/回收矛盾。
- 真实外层的原始 backend 字节能够重建已有报告；
  SSH 成功但 backend 失败、backend 完成但 DONE 缺失均不晋级成功。
- 满 480 项但参考或见证不符时正常结束且数据失败；
  基础设施失败立即停止，未派发/未知/已观察分母完整。
- 上一批收齐但保存失败时零后续派发；取消/次生保存错误保留；
  缩短剩余期限、READY 拖延、时钟异常、FD/runtime 漂移均不放宽。
- 假路径、符号链接、旧 claim、仅改 ID 试图重投均拒绝；
  错误报告不覆盖旧证据，不伪造 cleanup 或远端退出。
- 合成 fixture 最多验证 `binding_complete`，不生成真实接受记录；
  原组件/data/旧封存字节不变，接口导入不启动子进程或连接。

本地进程/传输测试通过后，独立审查实际差异与完整包设计。
远端探针和实际题目执行仍须各自核验精确命令、源码、身份、原始证据与
一次性范围；任何实现测试结果都不能自动消费真实运行许可。
本阶段只交付第一段能力，不同时新增 tokenizer 的真实清单和启动器。

## 8. 当前交付状态

本轮只有此设计和顺序 ledger 追加，未创建上述四个实现文件或测试文件。
书面审阅后才能转入实施计划；当前没有可运行 manifest、实时预检或实际
480 项结果。没有把缺失的历史材料补写成新观察，也没有改动旧报告。

设计自查覆盖：固定规模、数据/源码 pins、原接口兼容、双层进程结束、
未知项保留、真实/合成来源边界、期限嵌套和最终许可范围。
已明确远端期限从 READY 前起算，不能在网络传输后重新获得完整预算；
wire 总量与各帧上限同时生效，不将各帧最大值相加后声称仍在总限内。
具体实测字段缺失是禁止启动的条件，不是可用默认值或实现占位许可。
