# 真实 Qwen3-8B LoRA 训练链路结果

日期：2026-11-16。

## 结论

已在远端一张 A100 80GB 上完成一次真实预训练 Qwen3-8B 的固定 LoRA
训练、保存、重载和卸载验收。16 步更新全部返回有限 loss/梯度范数，
三项 prompt-only forward 一致性检查通过，进程正常退出，adapter 已取回并
核对字节数与 SHA256。没有重试这次训练，没有生成答案或执行模型生成的代码。

这是**真实模型工程链路通过**，不是 LoRA 有效性、推理加速或 RSI 成立的证明。
之前的组件测试和远端环境检查已经推进到真实模型执行；旧日期报告保留其当时
状态，本报告不改写历史记录。

## 做了什么

- 新增 `training_roundtrip.py`，串联既有 `training_inputs.py`、
  `training_worker.py` 和 `adapter_artifact.py`；这三个既有模块没有修改。
- 固定底座 BF16、LoRA FP32，36 层的 q/v 投影，共 72 处；
  rank 8、alpha 16、dropout 0、seed 42。
- AdamW，学习率 `5e-5`、weight decay 0、梯度裁剪 1；
  只用八条监督样例，固定顺序两轮，共 16 步、676 个监督 token（含两轮）。
- 使用 eager attention，禁用 KV cache、autocast、量化和 TF32。
  底座冻结；这是有反向传播的监督 LoRA 训练，不是纯 forward 自更新。
- 远端 Python 3.12.13、PyTorch 2.5.1+cu124、Transformers 4.57.1。
  加载前重新核对五个权重分片及配置/tokenizer 文件，加载后检查文件 stat
  身份。哈希绑定既有运行时观察，不构成独立的上游权重来源认证。
- 启动前完成独立代码审查及远端随机 tiny-Qwen CPU 兼容性检查。
  后者是合成诊断，与这次真实预训练模型运行分开记账。

## 实际检查

五次额外 forward 都只使用第一条监督样例的 prompt，不含答案，不采样、不生成：

| 检查 | 结果 | 能证明的范围 |
| --- | --- | --- |
| 初始零增量 adapter vs 原底座 | logits 精确相同 | 初始挂载不改变这条输入的输出 |
| 保存重载后 vs 保存前最终 adapter | logits 精确相同 | 此输入上保存重载一致 |
| 卸载 adapter 后 vs 原底座 | logits 精确相同 | 此输入上恢复底座一致 |

五次 probe 的 logits 均有限；完整性检查也核对底座状态。
这些检查不是全部任务上的等价性证明，也没有把“最终 adapter 相比底座更正确”
作为已验证事实。初始与最终 adapter 的张量摘要不同，只证明参数确实更新。

两轮在线训练 loss 的逐样例算术均值分别为 **0.738967**、**0.638501**。
每个 loss 在相应更新步骤计算，不是冻结 checkpoint 上的统一评估，也不是
token 加权的验证集 loss。不能用这个下降宣称纠错、迁移或保留能力改善。

## 成本与产物

| 项目 | 实测 |
| --- | ---: |
| 16 步更新循环 | 5.413 秒 |
| 训练封装阶段（含准备及完整性检查） | 68.644 秒 |
| 初始 / 最终 / 重载 adapter 挂载检查 | 62.791 / 62.896 / 62.911 秒 |
| 保存 / 加载 artifact | 0.256 / 0.119 秒 |
| 整个 roundtrip | 258.430 秒 |
| 模型加载（roundtrip 外） | 16.065 秒 |
| checkpoint 哈希（roundtrip 外的单独计时项） | 18.150 秒 |
| 训练输入准备（roundtrip 外） | 12.101 秒 |
| 远端 worker 总墙钟 | 315.714 秒 |
| 本地训练连接总墙钟 | 317.782 秒 |
| CUDA peak allocated / reserved | 18.456 / 20.037 GiB |
| 单份 adapter 张量 | 15,335,424 bytes |
| optimizer 张量 | 30,671,424 bytes |
| 保存文件（含初始/最终 adapter 和 receipt） | 30,767,794 bytes |

训练更新循环包含在训练封装阶段，后者包含在 roundtrip；挂载阶段包含 probe
和完整性检查，不能再把它们重复相加。完整性核验在本次墙钟中占比很大，
这些挂载耗时不能当成纯 forward 时延。显存峰值包含驻留底座，不是 LoRA
独占显存。连接墙钟不包含此前准备、审查、兼容性诊断和后续 artifact 下载，
也不是完整业务成本；其余外层步骤未全部单独拆分。

取回文件与远端 receipt 的 SHA256 一致：
`699aa7f6e990d8580a84281c7f3053bfed36bab352ec4ff67aa80655f6099168`。
权重、连接参数及原始记录只保存在私有证据目录，不进入 Git。

## 验证与保全

- 封装的 12 项合成测试先 RED 后 GREEN；独立审查通过。
- 本阶段此前完整回归：542 项运行，540 通过、2 跳过，173.277 秒。
  收尾未修改实现，也没有为报告重新运行这一整套测试。
- 启动脚本 R1 的四类问题及原失败证据保留；修复后 R2 独立审查通过。
  私有启动诊断 11 项、bootstrap 检查 3 项通过。
- 收尾从原始 stdout 重建 28 条阶段事件、16 个固定顺序步骤、唯一末尾 DONE，
  核对 run/manifest/result 绑定、零退出码及三项检查；下载退出码及哈希一致。
- 59 个既有源码/数据文件、2 个封装审查文件和 7 个启动审查文件的哈希一致；
  旧封存的 250 个私有文件、51 个工作区文件及之前运行时证据的 12 个文件
  均复核通过。这里只做保全哈希，没有打开评分答案做评估。
- 分支仍为 `experiments/feedback-active-verification`，
  HEAD `c214a1bcb14a6c2c658ab6f9df2bc3bebca50939`，tracked/index 无改动。
  本阶段新增文件尚未提交或推送；没有重跑旧 1416 项验证。

私有证据索引：`train.stdout`、`train-result.json`、`collect-result.json`、
`reconciliation.json`、审查记录和 `full-suite.log`。公开摘要不替代原始证据，
这些私有文件不随仓库发布。

## 还没回答的问题

`benchmark_eligible`、`formal_evaluation_ready`、`net_benefit_established`、
`isolated` 均保持 false；本次模型生成数、候选执行数均为 0。

下一研究关卡仍是固定 adapter 的 `lora_text` 与 `text` 对照，两组获得相同
八条监督信息，分别检查纠错收益、迁移、原能力退化及完整成本。候选生成、
安全执行、结果与成本绑定还要先验收；本报告不授权直接执行评分或打开保留答案。
只有这一步才能判断“写入权重”是否比“同样的信息放在文本里”值得。
