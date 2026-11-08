# Online fixed-development diagnostic

`development_protocol.py` 把四题开发材料接到真实五策略协议；
`development_online.py` 管理一次在线实验及真实 CPU 核验；
`development_model_worker.py` 是独立的远程模型进程。
原合成计分器不变，也不把合成收据改标签后当成真实数据。

## 数据如何流动

本地先固定 24 个配对格子和 120 条策略记录。模型只收到规格、初始程序、
中性 A/B/C 候选、可见核验输入、反馈以及已发生的公开交互。
候选身份、家族、真假反馈标签和评分答案不传到远程。

`active_verify` 可以直接选，也可以发一次 `VERIFY`；本地此时才运行该题
初始程序的两个固定输入，并返回期望值和真实输出。`always_verify` 则先
运行同样的工具，再做一次模型选择。`rethink` 有两次选择，但没有新证据。
不运行模型生成的代码，不微调权重，不跨题保留 KV 或对话。

每个模型收据保留请求 ID 和公开请求摘要。ID 绑定运行、题目格子、策略和
轮次；离线回放会重建提示，拒绝把另一个题目或策略的输出移用过来。
摘要只检查一致性，不是运行来源的密码学认证。

## 本地接口

从仓库根目录把 `experiments/feedback_adaptation` 加入 `PYTHONPATH` 后：

```python
from development_protocol import build_manifest, manifest_digest, score
from development_online import ModelProcess, run_campaign
```

- `build_manifest(run_id)`：32 位小写十六进制运行 ID；固定四题、单排列、
  两个初始状态和三个反馈条件，策略顺序按格子轮换。
- `ModelProcess(argv, output_dir, timeout_seconds=600, expected_ready=...)`：
  延迟启动一个受信任的模型子进程，用 JSONL 交互。`argv` 只能来自操作者，
  绝不能由模型输出构造。`expected_ready` 应与冻结的模型、源码、机器身份、
  依赖版本和有效生成配置逐项相符。
- `run_campaign(output_dir, manifest, model, audit)`：输出叶目录必须全新。
  `audit` 是固定 CPU `audit` 运行的 `manifest.json` 和实际 `stdout.json`，
  分别放在 `manifest` 与 `observation` 键下。函数重新校验审计准入；
  模型必须提供 `start()`、一次请求调用、`finish()` 和 `close()`。
- `score(manifest, frozen_digest, records, audit)`：固定分母离线计分，
  检查模型请求与工具结果的绑定，不创造新的模型或程序执行证据。

`ModelProcess` 本身不是 GPU 准入器。远程运行需另行提供受信任启动清单、
只读模型目录、全部文件摘要、指定 GPU/机器身份和独占目录；本仓库不保存
个人服务器地址或认证配置。模型 worker 的 `launch.json` 字段见
`docs/superpowers/plans/2026-11-08-feedback-online-development.md`。
运行前应在实际依赖环境中做无 CUDA 的分词和有效生成配置预检。

## 预算与失败

最多 144 次模型调用、48 次工具调用、96 次固定核验用例执行。
模型每次输入上限 4096、输出上限 32 token；必须包含正常 EOS 才接受选择。
文字解释、空输出、截断或第二次 VERIFY 都不会获得重试机会。
工具失败消耗唯一核验机会，但允许剩余的最终选择。

传输/模型服务故障停止整批；已经发生的原始字节与部分轨迹保留，
未执行格子继续留在分母里。标准输出结束与标准错误结束分别处理，
清理时也尝试有界收集终止尾部，不把丢失内容填成成功。
清理错误会单列并使整批标记为 aborted。

受控远程启动还需独占启动锁、595 秒 TERM 加 5 秒 kill-after 看门狗。
本地 600 秒是启动/完成验收预算，不是覆盖所有文件写入的 OS 级硬时限。
关闭本地 SSH 不等于证明远程已退出，运行后仍需检查远程退出收据与目标卡。

## 读结果

`records.json` 在计分前封存；`journal.jsonl` 保留请求、原始响应、接受事件
及结束顺序。`probe-*` 是本轮实际在线执行的固定程序收据。
这些材料含本机信息和私有评分内容，不应直接上传或放进模型提示。

结果分开报告选择成功、部署回退、修复与误纠正、失败/缺失、核验次数和成本。
模型生成、模型调用、跨进程往返、工具和 episode 墙钟不是同一个口径；
加载、预检、审计等共享成本另列。CPU/GPU 实际消耗和总进程峰值内存未测时
仍为未知。有限开发集的质量/成本差值不自动成为正式净收益结论。

运行 `python3.12 -m unittest discover -s experiments/feedback_adaptation -p 'test_*.py' -q`
可验证本地协议和替身传输；测试中存在真实固定 CPU 程序，但不存在真实模型
生成。保持 `benchmark_eligible=false`、`formal_evaluation_ready=false` 和
`isolated=false`，不把开发诊断或进程分离说成安全沙箱。
