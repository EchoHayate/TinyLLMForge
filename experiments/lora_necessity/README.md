# LoRA necessity pilot

检验“相同可信纠错信息下，参数更新是否比文本记忆更有用”。
本目录与 SHINE 复现、旧无权重选择诊断分开，不接入 TinyLLMForge 主运行时。

设计已获用户批准；第一阶段纯 CPU 合同与合成计分器已完成。
第二阶段的静态数据修订与独立审查已完成：44 项具体题目中的三项结构问题
已替换并经复审关闭，数据分层与证据绑定也已复核。
2026-11-15 已取得 **1416/1416 项源码×用例的正常远程回执**：
44 个任务、354 项参考结果全部符合期望，132 项指定错误见证全部按预期暴露。
先前因 SSH 认证失败而缺失的 328 项，已在用户批准的独立尾部阶段补齐，
已完成的 1088 项没有重跑。题库动态检查完成，**不是训练或 LoRA 效果验收**。
全量结果见 `20261115-full-matrix-results.md`。
详见 `DATA_AUDIT.md`；`data/` 是离线材料，不是主运行时数据入口。
CPU 阶段状态见 `../../.superpowers/sdd/lora-necessity-20261110-progress.md`；
当前数据阶段见 `../../.superpowers/sdd/lora-necessity-data-20261111-progress.md`。
本次结构修订见 `../../.superpowers/sdd/lora-data-structure-revision-20261111-progress.md`。

动态核验控制端、两轮提示构造与 token 计数接口已完成，通过 48 项本地测试、
分项独立审查及独立总审。这一阶段仍只处理模拟观察和 tokenizer 替身，
未执行题目或训练。
进度与审查证据见 `../../.superpowers/sdd/lora-validation-20261111-progress.md`。

2026-11-15 已完成固定清单的真实 tokenizer 长度检查：**324/324 行，
零超限**，最大输入 2432/4096 tokens、最大答案含 EOS 104/256 tokens。
完整报告已重建核对，详见 `20261115-full-tokenizer-audit-results.md`。
这不代表题目函数通过动态验证，也不是训练或 LoRA 效果结果；
固定样例第二轮不能覆盖未来所有真实生成输入。

同日已完成本地动态核验路径预检：固定 Python 探针可运行，仓库/假凭据
读取、文件写入、本机网络连接、fork 和其他程序启动均被拒绝，超时检查通过。
用户随后指定改走远程，本地执行方案已停用。远程 worker、有限输出采集和
本地实际报告现已接通；最终人工探针及独立审查通过后，20:43 +08:00
单次执行 D01：**32/32 项正常完成，参考实现 8/8 符合期望，
3/3 个指定错误见证按预期暴露**。远端批次耗时 3.480 秒，
含 SSH 往返采集为 4.821 秒；没有训练或模型生成。
详见 `20261115-remote-d01-results.md`；早先路径预检保留于
`20261115-dynamic-validation-preflight.md`。
这是首题数据核验，不是 32 个模型成功样本；局部隔离探针也不准入任意生成代码。

首次扩批按每批最多 32 项执行。33 个新批次正常完成，第 34 个新批次
在 SSH 跳板机认证时失败，未收到执行回执；之后的批次没有启动。
已有 274 项参考实现观察、101 项指定错误见证，33 个完整任务满足数据检查。
原始回执、逐批报告和全分母汇总已在本地重新构建核对；失败和缺失项均保留。
随后仅查身份的连接成功，不等于剩余题目已执行，也未触发自动重试。
当时状态详见 `20261115-partial-matrix-results.md`，原始失败报告保持不变。
随后获准仅重投认证失败的 32 项并首次投递其余 296 项，11 个尾部批次全部完成；
跨阶段汇总保留旧失败，全部 44 个任务通过三项预设数据检查。
尾部 SSH 采集累计 55.817 秒，远端批次累计 38.720 秒，没有模型加载或训练。
后续仍需训练/生成组件验收，才能公平比较 `lora_text` 与 `text`。

## 已定义的实验

- 8 个监督示例、4 个协议开发任务、16 个迁移任务、16 个保留任务。
- `base`、`text`、`lora`、`lora_text` 四组；开发任务只运行前两组。
- 主要比较 `lora_text` 对 `text`，两组在推理时看到相同的监督示例。
- 一个候选梯度配置：rank 8 q/v LoRA，冻结基础模型，共 16 个更新步。
- 每任务每组最多两次模型调用，拟定总上限 272 次。
- 迁移至少净多通过 2 题、没有保留任务从 text 正确变为 lora_text 错误，
  才值得独立复验；这不是统计确认或生产验收。

这些是协议配置，不是已经发生的训练、生成、运行或研究结果。
完整设计：`../../docs/superpowers/specs/2026-11-10-lora-necessity-pilot-design.md`。

## 第一阶段 API

`pilot_protocol.py` 仅依赖 Python 3.12 标准库：

- `configuration()`：返回独立的固定配置副本。
- `build_manifest(tasks)`：验证 44 项元数据的分组、家族、唯一 ID/模板/摘要，
  并复制形成清单；不读取文件内容，不执行题目。
- `validate_manifest(manifest)`：验证固定配置和元数据，返回规范 JSON 的摘要。
- `score_synthetic(manifest, records, checks)`：只接受显式合成的最终结果记录，
  验证固定配对和预算，返回按组/家族的计数、配对变化及成本分量。

任务元数据的字段为 `id`、`split`、`family`、`template_id`、
`content_sha256`。唯一模板名不证明函数结构不同，摘要也不证明内容可信。
测试使用的摘要是合成标识摘要，不是真实训练数据。

记录只允许 `kind="synthetic"`，包含 `task_id`、`arm`、`status`、
`model_calls`、`input_tokens`、`output_tokens`、`episode_ns`。
状态为 `pass`、`fail`、`invalid` 或 `timeout`；记录缺失仍保留原分母。
episode token 计数是最多两次调用的合计，不能验证实际每次调用的 token 上限。
缺少成本保留为未知，已观察到的部分和完整合计分别报告。

`checks` 是测试者输入的合成状态，不是参数、隔离或清理的真实凭证。
结果顶层始终为 `status="SYNTHETIC_ONLY"`；分支结果单独放在
`simulated_decision`，包括：

- `PROMISING_FOR_REPLICATION`
- `NO_GAIN_IN_THIS_PILOT`
- `OBSERVED_REGRESSION`
- `INCONCLUSIVE_INCOMPLETE`
- `INCONCLUSIVE_CALIBRATION`
- `INVALID_PROTOCOL`

判定先检查协议失败、记录/成本不完整和开发校准，再检查保留任务退化与
迁移收益。`OBSERVED_REGRESSION` 专指保留集合至少一项 text 通过而
lora_text 未通过。没有这种保留任务退化时，迁移净增小于 2（包括负值）
归为 `NO_GAIN_IN_THIS_PILOT`，但配对计数保留全部退化。

例如，一个完整合成案例中，迁移 text 为 8/16、lora_text 为 0/16，
保留任务没有改坏，则模拟判定为 `NO_GAIN_IN_THIS_PILOT`，同时报告迁移
`right_to_wrong=8`、`net=-8`。这个标签不表示两组一样好。

畸形字段、重复记录、超预算等输入错误抛出 `ValueError`，不当作普通模型失败。
无论模拟结果为何，`benchmark_eligible`、`formal_evaluation_ready`、
`net_benefit_established`、`isolated` 都是 false。不能把合成分支名当真实效果。

## 验证

在仓库根目录运行：

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v
```

自动化测试覆盖协议、人工安排的结果与本地子进程传输，不加载真实模型/
分词器，不执行题目候选，不连接 GPU、网络或 SSH，不安装依赖。
上面的真实远端 tokenizer 检查是单独的有界运行，不是 unittest。
测试数量不是有效研究样本量。

## 尚未交付

题库静态修订、固定清单的真实 tokenizer 长度检查和 1416 项动态核验已完成。
这些检查不证明任务之间的统计独立性或模型错误独立性。尚缺：

1. 训练 worker：loss token 屏蔽、优化器白名单、基础参数不变与 LoRA 更新证据。
2. 隔离候选执行器：执行生成程序需要真正的资源与凭据隔离，不能调用旧宿主
   固定程序执行器代替。
3. 真实输入/适配器/生成/工具/训练/退出收据绑定和完整流程成本。
4. 经独立审查与新准入的有界模型对照实验；目前没有训练或模型效果结论。

上述组件需要分别实现和验收，不能给 `kind` 换个值就将合成计分器当作真实
运行验收器。当前也不计算整条流水线耗时或训练摊销；训练成本缺席，不是零。
达到开发筛选门槛后仍需独立复验，不能自动进入错误反馈闭环或引擎集成。

## 动态核验控制逻辑

`validation_contract.py` 绑定当前冻结题库，形成 1416 项源码×用例清单，
并分离工作请求与控制端答案。`validation_report.py` 只处理明确的模拟观察。
控制逻辑测试不执行题目；模拟报告不能证明参考实现正确、隔离有效或模型获益。
`validation_task_worker.py` 与 `validation_remote.py` 仅在远程隔离进程中执行
已冻结、静态审查的小函数；`validation_real_report.py` 在本地绑定原始回执并
比较未上传的答案。`validation_matrix.py` 核对完整分母及任务检查，
`validation_tail.py` 保留原阶段失败并归并获准接续的观察。
完整冻结题库已通过动态数据检查，生成候选后端仍未准入。
此前非规范 JSON 的摘要口径问题已修复：后端与报告端均对原始 bundle 字节
计算 SHA256，人工排版回归已通过；旧 D01 报告保留当时的历史说明。

## 提示与 token 预算控制逻辑

`prompt_rendering.py` 从冻结公开资料构造两轮输入及八份训练消息，
保留 text/lora_text 的完整相同示例。`token_budget.py` 用注入的 tokenizer
核对模板、token ID、终止 token 与逐次输入上限。独立的 `tokenizer_session.py`
和 `tokenizer_audit.py` 路径已使用真实 Qwen3-8B tokenizer 完成固定 324 行
长度核验。44 份参考答案仅用于独立长度审计，不进入推理提示。
生成 worker 调用前的预算守卫、训练容量和隔离执行仍需分别验收。
