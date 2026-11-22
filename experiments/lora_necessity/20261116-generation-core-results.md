# 固定 LoRA 对照：候选生成组件

日期：2026-11-16。

## 本轮推进

在真实 Qwen3-8B 的 16 步 LoRA 训练链路已经完成的基础上，新增
`generation_core.py` 与 `test_generation_core.py`，落实后续对照所需的
**固定参数单次候选生成**。本轮组件测试、随机 tiny-Qwen API 兼容性、
完整回归与旧文件保全核对均通过。

这轮没有连接远程、加载真实预训练权重、重新训练真实模型、生成真实任务答案或
执行模型生成代码。因此没有新增 LoRA 对文本记忆的效果数据。
真实训练结果仍见 `20261116-real-lora-qualification-results.md`。

## 怎么实现

`generate_candidate(model, tokenizer, messages, *, expected_tokenizer_sha256)`
接受调用方已经准备好的冻结模型和消息：

- 复用现有 tokenizer 测量路径，绑定词表/模板摘要，禁止 thinking 和截断。
  输入最多 4096 token；第二轮完整的 user/assistant/user 消息也走同一检查。
- 要求模型 eval、参数冻结且无已有梯度；在 `no_grad` 中做固定 greedy
  forward。每一步显式 `use_cache=False`，从本次 prompt 与已输出 token
  构造完整前缀，不继承前一个调用的 token 列表或 KV。
- 最多输出 256 个 token；EOS 立即停止并计入输出数量。检查末位置 logits
  有限、形状合法、所选 ID 属于词表。错误直接抛出，不另选答案或自动重试。
- 只去掉终止 EOS 后解码，不 strip、不移除 Markdown、不修复代码。
  保存原始文本及其 SHA256、完整输出 IDs、输入摘要、停止原因和 token 数。
- 区分一次逻辑模型调用与内部 forward 次数；分别记录准备、生成、解码和
  整体墙钟。CPU 显存字段为 None；CUDA 分支包含驻留模型，尚未硬件验证。

该函数不读取题库、加载或挂载 adapter，不决定 arm，不评分，也不执行候选。
底座/adapter 字节完整性、模型独占、任务绑定、运行时限和失败时的已消费调用
由外层运行器负责。`no_grad` 本身不是参数完整性证明或代码隔离。

## 验证结果

### 合成行为测试

24 项测试先因组件不存在而明确失败，再全部通过，GREEN 耗时 36.215 秒。
覆盖 4096/4097 输入边界、256 输出上限、立即/末 token EOS、greedy 平分、
前缀增长和跨调用重置、第二轮消息、原文保留、UTF-8/字节上限、词表漂移、
非法 logits、模型/解码异常、无梯度、调用数及冷启动导入。

独立审查在绑定的两文件 diff 上给出 **Spec PASS / Quality APPROVED**，
没有待修复的 P0/P1/P2 项；审查没有重复运行测试。

### 随机 tiny-Qwen 兼容性

使用本地既有 PyTorch 2.12.0、Transformers 5.9.0，在 CPU 上新建随机
一层 Qwen3，BF16 参数、eager attention、eval/frozen；没有加载 checkpoint。
输入为不含真实任务内容的 56-token 合成 prompt。

- 新组件生成 256 token，按 length 上限停止。
- 相同模型/输入及显式 greedy、no-cache 配置下，全部输出 token IDs
  与 Transformers 自带生成路径**逐项完全一致**。
- 前后参数和缓冲区摘要一致，没有累积梯度；新组件也没有改变 RNG 状态。
- 新组件一次逻辑调用对应 256 次 forward；CPU 显存字段为 None。
- 兼容性子进程正常退出，含导入耗时约 8.254 秒。

此兼容性检查包含新组件与参考路径各一次合成生成，不把参考路径的计算
当作免费，也不计为真实任务调用。

这是本地随机模型的 API/算法交叉检查，不是远端 Qwen3-8B 生成验证。
远端此前训练使用 PyTorch 2.5.1+cu124、Transformers 4.57.1；
本地通过不能替代这组远端版本的兼容性验收。

## 成本边界

本组件有意采用每步重算完整前缀的参考实现，方便排除跨题 cache 和隐含
generation 配置，不是新的推理优化。它会重复计算前缀，尤其不能把这种路径
下的文本示例成本直接推广为生产引擎成本；需要缓存的生产路径须另行验收。
当前随机小模型耗时也不能外推为 8B 模型耗时或 LoRA 部署成本。

## 完成记录

- 完整目录回归正常退出：566 项运行，564 通过、2 跳过，211.831 秒。
  命令为 `python3.12 -B -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -q`，
  设置 `PYTHONDONTWRITEBYTECODE=1`；实际解释器与退出观察保存在私有回执。
  既有回归中的小张量合成训练测试仍正常执行，不是重训真实 8B 模型。
- 独立整体审查检查了兼容性脚本、原始回执、源码绑定和报告边界，
  没有新增待修复项；其要求的回归退出码、保全核对和最终文档绑定分别收尾，
  不将审查时的条件批准冒充当时已完成。
- 61 个既有源码/数据文件、56 个此前真实训练私有文件（含 adapter）保持不变；
  历史封存的 250 个私有文件、51 个工作区文件及 seal 摘要也复核一致。
  这些是字节保全检查，不是重新运行任务或把评分答案输入模型。
- 分支仍为 `experiments/feedback-active-verification`，
  HEAD `c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`；tracked/index 无改动。
  本轮组件、测试和文档均为新增未提交文件，没有 commit/push。

原始 RED/GREEN、完整回归日志、CPU 兼容性结果、两阶段审查和最终保全摘要
保存在私有证据目录，不随仓库发布；公开摘要不替代全部原始运行证据。

## 下一关

仍按原协议先用 4 道开发题、base/text 两组验收完整两轮流程，再考虑
32 道迁移/保留题的四组对照；不按 LoRA 效果选题。

还缺两类实证：

1. 候选解析、独立受限执行、可见反馈与第二次生成的完整回执；
   不可直接复用只针对已审阅固定源码的宿主执行器。
2. 远端固定模型及 adapter 上的生成兼容性、输入上限、失败/成本记录与
   开发题协议验收。模型生成与评分输出必须分开保管，不能回流训练。

本轮没有这些结果，不能报告校准通过、纠错收益或安全执行准入。
所有 `benchmark_eligible`、`formal_evaluation_ready`、
`net_benefit_established`、`isolated` 标志仍为 false。
