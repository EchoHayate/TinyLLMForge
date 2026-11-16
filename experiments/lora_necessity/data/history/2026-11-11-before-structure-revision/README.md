# LoRA necessity pilot

检验“相同可信纠错信息下，参数更新是否比文本记忆更有用”。
本目录与 SHINE 复现、旧无权重选择诊断分开，不接入 TinyLLMForge 主运行时。

设计已获用户批准；第一阶段纯 CPU 合同与合成计分器已完成。
第二阶段已编写 44 项具体题目草稿，但结构审阅仍有阻断问题，
参考源码与测试未执行，不能用于模型训练或效果验收。
详见 `DATA_AUDIT.md`；`data/` 是离线材料，不是主运行时数据入口。
CPU 阶段状态见 `../../.superpowers/sdd/lora-necessity-20261110-progress.md`；
当前数据阶段见 `../../.superpowers/sdd/lora-necessity-data-20261111-progress.md`。

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

测试只检查纯函数和人工安排的结果，不加载模型/分词器，不执行函数候选，
不连接 GPU、网络或 SSH，不安装依赖。测试数量不是有效研究样本量。

## 尚未交付

1. 44 项草稿已完成独立静态审查，但结构级划分未验收：
   S01/T04、S06/T09、S08/T16 的修复结构过近，替换提案待确认。
   354 个用例仍未动态执行，也未做实际 tokenizer 长度验证。
2. 训练 worker：loss token 屏蔽、优化器白名单、基础参数不变与 LoRA 更新证据。
3. 隔离候选执行器：执行生成程序需要真正的资源与凭据隔离，不能调用旧宿主
   固定程序执行器代替。
4. 真实输入/适配器/生成/工具/训练/退出收据绑定和完整流程成本。
5. 经独立审查与新准入的单次有界运行；目前没有训练或模型效果结论。

上述组件需要分别实现和验收，不能给 `kind` 换个值就将合成计分器当作真实
运行验收器。当前也不计算整条流水线耗时或训练摊销；训练成本缺席，不是零。
达到开发筛选门槛后仍需独立复验，不能自动进入错误反馈闭环或引擎集成。
