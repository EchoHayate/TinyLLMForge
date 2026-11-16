# Standard tokenizer runtime — November 14, 2026

## 结果

**模块名误拦截的实现已移除，本地回归通过；真实 324 行计数仍未取得结果。**

本轮恢复正常 Python 导入，不再逐个核验或放行 Transformers 内部模块。
真实远端运行通过主机身份确认并接收全部输入，但在约 539 秒后结束，
只收到 READY，没有 RESULT/DONE。不能声称真实 tokenizer 已验收，
也不能把“没有返回计数”写成“远端一行也没处理”。

## 改了什么

- `tokenizer_runtime_guard.py` 删除 import finder、模块名黑名单、依赖发现包装
  和拒绝 loader。导入 `modeling_*`、`generation` 或 `safetensors` 不再被
  自动等同于执行模型。
- 保留一次性会话、主线程、冷启动和固定环境检查；继续关闭 Transformers 模型
  后端和在线访问设置，并检查 `CUDA_VISIBLE_DEVICES=""`。
- `RuntimeGuard` 现在是运行环境与生命周期检查，不是恶意代码沙箱。它不承诺
  拦截任意模型 API、网络请求或文件访问。
- 应用仍只调用本地 `AutoTokenizer.from_pretrained`，保留
  `use_fast=True`、`local_files_only=True`、`trust_remote_code=False`；
  五个文件 pin、完整版本清单、观察状态、token 计数和报告格式不变。
- 未新增模型加载、生成或训练调用，没有依赖安装、分支切换、提交或推送。

## 本地验证

- 新合成用例先在旧实现上失败，覆盖配置工具的正常依赖导入、
  不干预 Python 导入机制和不把模块缓存视为模型执行。
- 最终 guard 测试 12 项，会话测试 57 项。全量 **357 项，355 通过，
  2 项 Linux 专用测试跳过**；最后一次全量运行 27.951 秒。
- 相比旧全量 362 项，数量变化来自替换已撤销的黑名单/包装器行为测试，
  不是删除 token 预算或输入身份检查。
- 独立审查发现一个 P2 测试缓存隔离问题：已先复现，再修复并确认原缓存恢复。
  最终复审通过；取消和加载失败的用例也改为断言实际 guard 已关闭，
  而非断言它不在已经弃用的 import finder 列表中。

## 一次真实远端运行

| 项目 | 实际结果 |
| --- | --- |
| Run ID | `c2e6510e1dc54b76825d0e362c929665` |
| 新 SSH 尝试 | 1 |
| 主机/UID/config READY | 匹配 |
| 上传 payload | 2,237,789 字节，完整发送 |
| 帧回传 | 457 字节，仅 READY |
| 总耗时 | 539.023 秒 |
| SSH 返回码 | 255 |
| Collector 分类 | `PROTOCOL_OR_PROCESS_ERROR` |
| 完整计数报告 | 无 |
| 已确认的实际计数行数 | 未知 |

SSH 日志记录了 `exit-signal` 请求，未提供可确认的信号编号；结束时间距
540 秒期限约 0.98 秒，**与远端期限信号一致，但不是信号编号的直接证据**。
不能把该结果说成认证失败，也不能确认它究竟停在加载还是第几行计数。
未自动重试、未另开连接取日志、未清理远端目录。

## 后续应聚焦的实际开销

在当前真实 `session.measure` 路径上，用已有测试 tokenizer 和一个
150,000 项合成词表进行本地计时，没有加载真实模型或真实 tokenizer：

| 行类型 | 每行 `get_vocab()` 调用 | 本地单次耗时 |
| --- | ---: | ---: |
| 只计提示词 | 8 | 0.185 秒 |
| 同时计参考答案 | 12 | 0.280 秒 |

按 272 行无答案、52 行有答案的 inventory，若全部正常完成，仅逐行测量就会
触发 **2,800 次完整词表读取**，尚不含启动/结束检查。该路径还会反复计算
词表摘要和序列化 tokenizer backend。

这是已复现的重复工作，不是本次远端慢的唯一根因证明；合成 backend 很小，
且本地硬件不同，不能用上述时间推算远端完成时间。本轮未修改这条测量路径。
下一步应给实际计数路径做耗时定位，并考虑将不变的 tokenizer 身份检查移到
合适的会话边界，保留消息、token ID、模板一致性和长度预算检查，
而不是继续逐个审计库工具或直接反复重跑 324 行。

## 证据

私密根目录：
`/private/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-standard-20261114-86m9cmt6`。

- `before/`、`baseline.json`：变更前源码与进度记录。
- `guard-red-v2.log`、`session-loader-red-v2.log`：修复前预期失败。
- `review-cache-red.log`、`review-cache-green.log`：缓存问题复现与修复。
- `full-suite-prelaunch.log`、`review-summary.json`：最终测试与审查摘要。
- `candidate/`、`authorization.json`、`attempt-claim.json`：冻结输入和本轮范围。
- `attempt/transport.json`、`attempt/wire.stdout`、`attempt/transport.stderr`：
  实际返回；原始日志仅私密保留。
- `transport-classification.json`：不含样本的失败分类。
- `synthetic-measurement-cost.json`：本地合成开销测量，不是远端进度。
- `verification.json`：最终源码、文档和证据摘要。

本轮不产生任何 LoRA 优势、训练有效性或 RSI 结论。
