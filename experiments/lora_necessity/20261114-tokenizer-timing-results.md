# Tokenizer phase timing and vocabulary digest reuse — November 14, 2026

## 结果

**真实 tokenizer 短输入诊断正常返回；词表摘要复用使本次输入/答案计数
耗时分别减少 27.3% / 34.8%，完整计数记录的摘要不变。**

这是 tokenizer 预处理路径的改善，不是模型 forward、LoRA 效果或 RSI
能力改善。没有重跑 324 行 inventory，也没有加载模型权重、生成或训练。

## 实际瓶颈

使用原先固定主机、Python 3.12.13、Transformers 4.57.1，以及同一个
五文件只读 tokenizer snapshot。两次诊断均重新验证完整运行时版本和
snapshot 摘要，使用新诊断身份，未复用已消耗的完整审计启动身份。

公开输入为 `Reply with OK.`，参考答案为 `OK`。每次诊断依次测量
prompt-only、带答案、额外一遍带 cProfile 的答案测量，然后完成会话。
两次均收到完整 READY/RESULT/DONE，SSH 返回码为 0。

| 未开启 cProfile 的测量 | 修改前 | 修改后 | 耗时减少 |
| --- | ---: | ---: | ---: |
| 输入计数 | 3.075 秒 | 2.235 秒 | 27.3% |
| 输入与答案计数 | 4.386 秒 | 2.862 秒 | 34.8% |

每版本、每种未开启 cProfile 的测量各只有一个样本；不是多次统计实验，
不保证其他输入、主机负载或完整批次得到相同比例。

- tokenizer 加载：修改前 2.406 秒，修改后 2.217 秒。加载实现未修改，
  不将这部分差异归因于摘要优化。
- 原版输入/答案测量的状态检查累计用时为 3.038 / 4.362 秒，
  占相应总耗时的绝大部分。
- 独立 cProfile 答案样本中，JSON 编码累计耗时从 2.409 秒降至
  0.201 秒。重复词表读取约 1.11 / 1.13 秒、backend 序列化约
  0.37 / 0.38 秒仍在；精确类型检查和字典比较也仍有成本。
  不将带 profiler 的时间与未开启 profiler 的时间混用。
- 新证据证明当前短输入路径存在上述开销；它不能追溯证明上一轮
  539 秒无报告的审计究竟停在哪个阶段或哪一行。

## 最小修改

- `tokenizer_measurement.py` 增加私有单项词表摘要缓存，每次测量独立持有。
- `tokenizer_session.py` 为会话观察持有单独缓存，释放 tokenizer 时释放。
- 每次检查仍获取完整词表，执行原有类型检查；在摘要边界再复制并精确
  检查该副本，然后比较、计算摘要和保存同一份副本。
- 相同内容不再重复排序并序列化为 canonical JSON；内容不同则计算新
  摘要，由原有状态对比拒绝。无全局缓存，不缓存 token 输出。
- 每行词表读取次数仍为 8 / 12。模板一致性、消息不变性、token ID、
  完整答案前缀、EOS、4096 输入/256 target 限制及失败关闭机制不变。

独立审查发现初版缓存可能漏掉属性 getter 在早期校验后将 `1` 改成
`True` 或 `1.0` 的情况，因为 Python 字典相等比较认为这些值相等。
六个反例先复现失败，再通过摘要边界的独立副本与精确类型检查修复；
复审通过。该初版候选保留在私密 `optimized/` 中，**从未远端执行**。

## 正确性验证

- 最终针对性测试 **78/78 通过**。
- 最终全量测试 **362 项：360 通过，2 项 Linux 专用测试跳过**，
  用时 28.620 秒。
- 64 组合成输入覆盖首轮/完整第二轮及输入、答案边界；
  修改前后完整记录逐字节相同。
- 两次真实诊断均计得输入 **16 tokens**、答案含 EOS **2 tokens**；
  三条测量的完整记录摘要分别一致，均完成三条记录的会话最终检查。
- 真实 prompt 记录 SHA256：
  `d199e672e85af8c4452366b128407098937e4b7f10508c2770829d4f96315173`。
- 真实 answer 记录 SHA256：
  `ecd4309efb3929039f0071a11977a26e546ae80e258c25e871bb420cf5cb0a02`。

以上不是全量预算验收。`benchmark_accepted=false`；
没有 LoRA 对照结果或净收益结论。剩余重复词表读取和 backend 序列化
仍然昂贵，不能因本次提速就宣称 324 行一定能在旧期限内完成。

## 证据与范围

私密根目录：
`/private/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-timing-20261114-b3DpnuLS`。

- 基线诊断：`bbea229bd6514ce99eb254429e0f1e3f`，18.916 秒。
- 复审后诊断：`82e16e4245ff4ba5a2712982bde33da4`，14.864 秒。
- `before/`、`baseline.json`：本轮修改前源码与 ledger。
- `manifest.json`、`scope.json`、`attempt/`、`validated-result.json`：
  基线输入身份、实际传输与返回。
- `optimized-reviewed/`：修订后冻结源码、独立启动身份和返回。
- `digest-red.log`、`session-digest-red.log`：旧版重复摘要计算的反例；
  首份日志包含一个随后修复的测试 import mock 误用，不能作为通过证据。
- `property-mutation-red.log`、`review-green.log`、`review-summary.json`：
  审查反例、修复与独立复审。
- `full-suite-reviewed.log`、`local-differential-reviewed.json`、
  `comparison.json`：最终回归与差分。

旧审计封存的 81 个证据文件摘要仍匹配。分支仍为
`experiments/feedback-active-verification`，HEAD 仍为
`93b4e8e0dc5174db0426a5da9e29cc360ed331a7`。没有切换分支、提交、推送、
依赖安装或远端清理；没有新增库模块黑名单。
