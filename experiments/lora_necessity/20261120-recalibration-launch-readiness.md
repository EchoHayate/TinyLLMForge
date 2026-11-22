# 新版 LoRA 校准：真实启动准备核对

日期：2026-11-20，Asia/Shanghai。

状态：**本地接口核对完成；具体真实启动包尚未就绪。**
本文是准备检查与方案建议，不是已批准的启动规格、实现计划或运行许可。
没有新增实际题目结果、tokenizer 长度或 LoRA 效果数据。

## 1. 当前工作区与已完成部分

工作区 `/Users/bytedance/dev/TinyLLMForge`，分支
`experiments/feedback-active-verification`，HEAD
`c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`。
检查时 tracked/index 无差异，既有 untracked 保留，未恢复旧 session。

按 `/private/tmp/lora-recalibration-tokenizer-20261120-6fCWPY/` 下的
`baseline-files.json` 与 `parent-final-verification.json` 重新计算摘要：
178 份既有文件、29 个最终凭据绑定均一致。29 个绑定中包含顺序 ledger；
本轮随后只向 ledger 追加本阶段记录，该项新摘要不再属于旧封存。
没有重跑此前完成的 157 项合成套件，也没有重新派发已完成任务。

六个新版组件的本地职责已完成，不需要再实现一次。
本轮读取 15 个相关模块并用 AST/摘要核对接口，没有导入它们，
更没有执行题库中的源码字符串。

## 2. 可复用接口与不能直接复用的入口

以下源码名均相对于 `experiments/lora_necessity/`。

| 接口 | 本轮从源码确认的事实 | 后续用途与缺口 |
| --- | --- | --- |
| `recalibration_validation.build_batches` | 冻结 480 项、15 个不同 batch run ID、每批 32 项、规范编码与完整包大小检查 | 可复用排程；不负责真实派发或来源证明 |
| `validation_remote.run_bundle` | 支持 `matrix`，1–32 项；在执行前比较整个 `expected_runtime`；每项至多 2 秒、每批 90 秒 | 可作为固定源码后端；仍需当前隔离探针和受信控制端 |
| `validation_remote.collect_process` | 有界 stdin/stdout/stderr、超时、进程组终止与直接子进程回收记录 | 可复用采集机制；本地 SSH 子进程退出不等于远端工作进程已退出 |
| `recalibration_validation.build_report` | 从原始批次回执重建 480 项及参考 192／见证 36；真实来源和动态完成标志仍 false | 可复用内容判断；必须另行绑定真实启动与采集来源 |
| `recalibration_rendering` | 公开任务与固定八例，feedback/retry 分支分开 | 可复用文本渲染；真实 feedback 必须来自本次已验证动态回执 |
| 新版 inventory/report/audit | 明确 synthetic-only；factory 是执行能力而非来源认证；真实标志恒 false | 保持现状，不改名或翻转布尔值来冒充真实路径 |
| `tokenizer_session.open_session` | 不绑定旧行数；校验 profile、只读快照、冷运行环境、身份、摘要链与结束记录 | 可复用真实计数会话；需要新版受信协调入口 |
| `tokenizer_runtime_profile` | Python 3.12.13、Transformers 4.57.1、固定 revision 与五文件 pins；实际完整包版本表必须相等 | 不能只填两项版本，也不能以测试 fixture 充当实测版本表 |
| `tokenizer_host_remote` | `SOURCE_NAMES` 固定 11 个旧模块；bootstrap 导入旧 `validate_inventory` 和 `run_audit` | 旧 bootstrap 不支持新版，不能只换 inventory JSON |
| `tokenizer_host_launch` | `build_launch` 经旧 bundle 校验；`execute_once` 绑定旧 envelope/report；READY 后才发 payload | 传输机制可参考，整套旧启动入口不能直接投入新版 |

两个关键边界：

1. `tokenizer_host_remote.install_task_denial` 在 bootstrap 中拒绝
   fork/clone/thread；固定源码后端则需要启动逐项 bwrap 子进程。
   因此不能把两个阶段塞进已启用 tokenizer 限制的同一进程。
   tokenizer 还要求冷进程、单次 RuntimeGuard，不能借用先前会话。
2. RuntimeGuard 的离线环境变量和模型后端屏蔽不是通用 OS 网络沙箱。
   tokenizer 只计数可信文本；题目源码仅在固定源码隔离进程执行。
   本次未验证两种进程的实时能力，不宣称通用 `isolated=true`。

## 3. 历史环境与实际可用材料

仓库结果文档仍存在：

- `20261115-full-matrix-results.md`：旧 1416 项真实动态检查。
- `20261115-full-tokenizer-audit-results.md`：旧 324 行真实长度检查。
- `20261116-remote-runtime-checkpoint-results.md`：历史解释器与环境观察。
- `20261116-real-lora-qualification-results.md`：真实 16 步 LoRA 工程验收。
- `20261117-development-remote-results.md`：base/text 各 4/4，校准饱和；
  当次没有挂 LoRA，不能作为 LoRA 效果结果。

这些是历史记录，不是 11 月 20 日的远端实测。本轮按记录的原路径检查，
以下五个私有目录均不存在：

- `tlf-remote-validation-20261115-WdUFIzYw`
- `tlf-tail-validation-20261115-wh6z70so`
- `tlf-tokenizer-full-20261115-8JzI7WOn`
- `tlf-ssh-route-diagnosis-20261116-zn2m844f`
- `tlf-candidate-executor-20261117-gb1gZX`

又在已知 macOS 临时父目录、`/private/tmp` 和
`/Users/bytedance/Desktop/local` 下至多两层按相同目录名前缀查找，
没有找到对应目录；没有搜索或读取旧 session 日志。
此结论仅限上述原路径和有限查找，不能声称这些材料在所有备份中均不存在，
也不推翻旧文档记载的历史成功。

因此本轮没有取到可重新核验的旧 SSH argv、完整版本 profile 或旧私有
控制器包，不能把历史成功升级为当前可启动。缺材料也不等于已观察到 SSH
失败。本轮没有网络连接、票据查询、`ssh -G`、远端探针或资源查询。

固定源码阶段需要当前 hostname/UID、解释器路径与摘要、bwrap/seccomp
摘要、命名空间和隔离探针。tokenizer 阶段需要当前完整版本表、五文件
摘要与来源路径，以及新 run 的只读快照。两类 runtime 对象不是同一 schema，
不得把 tokenizer profile 直接当作固定源码的 `expected_runtime`。
这些是需要后续实测并冻结的字段，不以历史值或合成值补齐。

## 4. 方案比较

1. **建议：增加新版真实启动与证据外层，保留旧组件不变。**
   内容重建继续复用新版纯函数；真实外层另行验证受信控制端的原始
   claim、输入、进程与输出绑定。另建真实清单及会话入口，复用 renderer、
   snapshot 和 session，不修改现有 synthetic-only 返回对象。
   成本是需要少量新版衔接实现；优点是不重写隔离和计数核心、不破坏旧证据。
2. 修改旧启动器的来源列表、旧分母和 synthetic 标志。
   改动看似少，但会混淆历史入口与证据含义；不建议。
3. 另写完整隔离器和 tokenizer 核心。
   重复已有能力，扩大验证范围；当前缺口不是这些核心，暂不做。

这里只提出推荐方案，尚未编写或批准新版 launch 代码。

## 5. 建议的分阶段启动包

### A. 实时预检与一次性运行身份

先审阅独立预检范围，再实测身份、已有运行时和人工隔离探针；
预检不投递题库、不加载 tokenizer/模型、不安装依赖。
认证必须保留严格主机密钥校验和禁止凭据委派/转发的既有边界，
不从不完整旧材料猜 argv，也不改全局 SSH 配置。

当前不能填写精确可执行 manifest：新身份、目标根、控制器源码、
完整依赖摘要、预检原始证据与批准摘要均未冻结。
未来 manifest 需绑定这些字段，而不是接受调用者传入 `approved=true`。

### B. 固定源码 480 项

- 单独控制器按冻结顺序串行 15 批，每项一个隔离进程；
  第一批是 ordinal 0–31，不等于完整 C01（每题 40 项）。
- scope/stage/batch 独占 claim，在发送前留存；每批绑定新 run ID、
  backend/worker、精确 bundle 和运行时摘要。未知派发不得自动重投。
- 只发送 `request_id/source/entry_point/args`；期望值、见证预测、源码角色
  与全量 oracle 留在本地。正确性比较在本地完成。
- 保留单项 2 秒、backend 批次 90 秒；建议 SSH 每批最多 110 秒，
  动态阶段整体 1800 秒。后两项是待审阅的启动建议，不是已生效配置；
  整体期限不得因下一批开始而刷新，不增加总内存/磁盘配额或自动清理前提。
- 保存精确输入、原始 stdout/stderr、退出/回收和阶段终止记录；
  从这些原始字节重建报告，不能只接收远端汇总 PASS。
  本地进程回收与远端结束分别记账，无法证明远端状态时保留 UNKNOWN 并停止。
- 正常语义不匹配继续其余固定位置，最后报告真实差异；
  连接/身份/隔离/执行/证据异常立即停止，未投递和投递未知分开。

只有真实来源绑定成立、480 项完整、192 个参考和 36 个见证检查满足，
才准许构造下一阶段真实清单。`build_report` 原返回对象保持原样；
真实结论由独立证据外层产生，摘要不是远程第三方证明。

### C. 真实 tokenizer 120 行

- 在本地由 B 的原始证据重建动态结果，再用 renderer 构造实际清单；
  不把现有 synthetic envelope 改标签后投递，不上传全量 oracle 或动态
  私有回执。远端只需已绑定的 120 行消息/答案计数投影、profile 和
  固定源码依赖包；48 行参考组合只携带本来获准单独计数的答案。
- 动态证据此前不存在，因此完整清单摘要不能在 B 执行前假造。
  B 后生成精确私有清单/manifest，再审阅其字节绑定和传输范围，
  不是凭本次文档提前放行 C。
- 新冷进程、新 profile/run、新只读 snapshot；复用真实
  `open_session(root, raw, approved_sha256)`，无 GPU、模型权重或生成。
- 新 bootstrap 必须支持新版来源列表与清单入口；旧 11 模块包不直接复用。
  本地真实报告要从同一动态输入、清单、profile、120 条原始测量和
  finalization 重建，不信任远端 report 或 callback 布尔值。
- 保留输入 4096、答案含唯一 EOS 256、`enable_thinking=False`；
  不改写、不截断，不增加训练消息行。建议复用已有可冻结的 1200 秒
  工作期限，不按旧 324 行耗时线性推算新版实际耗时。
- 超限停止并 finish；异常 abort；缺失/无效结束回执不能算长度通过。
  进程退出证据和会话结束证据分别保留，不自动重新打开会话。

## 6. 收尾和下一项

本轮只新增本准备记录并追加 ledger，不改源码、测试、数据或历史报告。
轻量核验检查旧文件摘要、本文路径/接口/关键摘要与源码一致，以及
分支/HEAD、tracked/index 状态；不运行真实入口或重复完整测试。

接下来最小工作是把上述推荐方案细化为新版真实证据外层与启动适配的
书面规格，解决精确输入、来源、分阶段许可、停止与可重建回执契约。
尚未进入实现或真实启动，历史材料不足的字段继续明确待实测，不伪造。
真实启动前仍须独立审阅最终包及一次性范围。

这两段完成也只是新题库可用于校准的前提，不是模型答题或 LoRA 收益。
不恢复 CED 测时，不训练，不进行新版 base/text 或 lora_text 对照；
benchmark/formal evaluation/net benefit/通用 isolated 继续为 false。
