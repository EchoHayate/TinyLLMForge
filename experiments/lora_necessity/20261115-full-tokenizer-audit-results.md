# Full tokenizer budget audit — November 15, 2026

## 结果

**固定清单的真实 tokenizer 长度检查完成：324/324 行，零超限、零计数错误，
会话正常结束，完整报告已在本地重建核对。**

总用时 **709.736 秒（约 11 分 50 秒）**，SSH 返回码 0。
最大输入 **2432 / 4096 tokens**，最大答案含 EOS **104 / 256 tokens**。
这证明当前固定输入清单满足长度预算，不证明题目答案正确、LoRA 有效，
也不证明尚未生成的所有真实第二轮输入都能满足预算。

| 类别 | 完成/计划 | 最大输入 tokens | 最大 target tokens | 超限 |
| --- | ---: | ---: | ---: | ---: |
| 训练消息长度检查 | 8/8 | 277 | 61 | 0 |
| 首轮推理提示长度检查 | 136/136 | 2319 | 不适用 | 0 |
| 参考答案长度检查 | 44/44 | 358 | 104 | 0 |
| 固定样例第二轮提示长度检查 | 136/136 | 2432 | 不适用 | 0 |

表中的“训练消息”只是计数，**没有进行训练**。第二轮使用之前冻结的初始
源码与合成反馈样例，不是实际模型第一轮输出，因此仍保留
`all_possible_second_rounds_validated=false`。

## 怎么完成的

- 使用 11 月 14 日已经实测并复审通过的词表摘要复用实现；
  本轮没有进一步修改 tokenizer 加载或计数逻辑。
- 固定 inventory 字节完全不变：
  `b59126f7fce3d26e98375f0e94628e60a949cbea2df2b33e46b3d41ccd2683a4`。
- 同一远端主机和解释器、完整固定版本清单、同一 Qwen3-8B 五文件 pin；
  为本轮创建独立 profile/run ID 和只读 snapshot。
- 仍只通过本地文件加载 fast tokenizer，关闭在线访问与模型后端；
  不加载模型权重、不生成、不训练。
- 使用新身份执行一次完整审计，收到 READY、RESULT、STDOUT、STDERR、DONE。
  完整发送 2,239,333 字节输入，收到 5,189,172 字节帧数据。
- 校验主机、配置、输入摘要与 DONE 绑定后，用原始 inventory/profile、
  324 条私密记录及会话 finalization 调用 `build_report`。
  重建结果与收到的完整报告逐字节一致；再次核对所有行状态为 `OK`。

## 期限调整，不是放宽 token 预算

之前默认工作期限是 540 秒。短输入诊断提示这批计数可能需要约 13 分钟，
因此本轮显式采用 **1200 秒硬截止**，并将它纳入冻结配置摘要。
实际运行超过旧期限，但在新期限内完成；这不是对上一轮超时阶段的追溯证明。

- `tokenizer_host_launch.py` 支持 manifest-bound `work_timeout_seconds`。
- `tokenizer_launch_window.py` 与远端 grant 校验使用同一整数期限，
  范围 1–1200 秒；默认及旧配置回退值仍是 540。
- 工作期限从连接开始固定，接收 READY 或输入不会刷新期限。
- 原有 60 秒本地清理记账不变，不表示远端自动清理已获保证。
- 4096 输入、256 target、错误/超限即停止的规则完全不变。
- 独立审查发现初版将较短整数期限的窗口仍记为 540 秒；已先复现失败，
  再修正整数期限传递并通过复审。初版冻结候选未远端执行。

## 验证

- 最终全量测试 **367 项：365 通过，2 项 Linux 专用测试跳过**，
  用时 29.392 秒。
- 独立复审运行 33 项相关测试，全部通过。
- 实际报告：`all_rows_within_budget=true`、`session_finalized=true`、
  `failure=null`、`stop_reason=null`。
- 旧通用运行准入字段仍为 false，清理字段仍为 `CLEANUP_UNVERIFIED`。
  不伪造或修改这些字段。用户已免除自动清理要求，本轮不再以清理作为
  报告这批真实长度结果的阻塞条件，也不额外连接远端做清理。
- 不将传输层始终为 false 的 `audit_pass` 改写成研究验收；
  本轮长度结论来自已重建的 324 条测量和最终状态。

## 接下来真正缺的东西

1. **题目正确性动态核验**：当前 44 项题目及 354 个用例尚无本任务的
   真实执行验收；现有 1416 项源码×用例控制清单与计分逻辑仍是模拟验证。
   token 数正确不代表参考实现或纠错信息正确。
2. **真实运行组件**：训练 worker 的答案 token loss mask、基础参数冻结、
   LoRA 更新证据，以及隔离候选执行器、真实收据和成本绑定尚需实现/验收。
3. **公平效果对照**：再比较看到相同可信示例的 `text` 与 `lora_text`，
   同时记录改善、改坏和额外训练/推理成本。本轮没有这类结果。

未来真实生成每次调用前仍需检查实际输入长度，不能用本次固定第二轮样例
替代运行时预算守卫。本次不自动启动训练、生成或题目函数执行。

## 证据与范围

- Run ID：`cadc2426b5c2499f8dbf30ba0dcfc8a1`。
- Manifest SHA256：
  `e9d41022a9c490ccb4b7a19abc9d9e08d3d74495f5e863f0af6f1dc3aabdbb4e`。
- 私密根目录：
  `/private/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-full-20261115-8JzI7WOn`。
- `candidate-reviewed/`、`scope-reviewed.json`、`attempt-claim.json`：
  最终冻结输入、源码、期限和单次启动记录。
- `attempt/`：实际传输、完整帧、执行结果；只在私密目录保存原始内容。
- `validated-report.json`、`public-summary.json`、`category-statistics.json`：
  重建报告及不含提示正文/答案源码的统计。
- `timeout-red.log`、`short-budget-red.log`、`short-budget-green.log`、
  `full-suite-reviewed.log`、`review-summary.json`：回归和审查。

没有复用旧运行身份或自动重试，没有依赖安装、分支切换、提交或推送。
当前分支仍为 `experiments/feedback-active-verification`，
HEAD 为 `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`。
