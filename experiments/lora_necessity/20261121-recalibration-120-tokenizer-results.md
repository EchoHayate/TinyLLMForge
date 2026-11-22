# 新校准：120 行真实 Qwen3-8B Tokenizer 长度核验通过结果

日期：2026-11-21，Asia/Shanghai。

状态：**120 / 120 行真实冷进程 Qwen3-8B 分词长度核验一次全部通过（零超限、零异常、会话正常结束并在本地重建核对一致）；新校准题库正式具备进入 72 次生成配对校准实验的数据与长度前提。**

## 1. 实测统计结果

按已审阅冻结包（`run_id=81b42b40fc9ffdfb506e0198bf7d8b96`，`package_sha256=3a08217bad0876993f119ccfc26dcc545fa023d372d6107ccff14db04590346c`）单次执行，总耗时 `280.219` 秒（约 4 分 40 秒），SSH `returncode=0`，`stderr=0` 字节，`binding_complete=true`，`secondary_errors=[]`，`session_finalized=true`：

| 类别 | 完成 / 计划 | 输入 token 范围（上限 4096） | 目标答案 token 范围（含唯一 EOS，上限 256） | 超限数 |
| --- | ---: | ---: | ---: | ---: |
| `first`（首轮 `base`/`text` 提示） | **24 / 24** | `394 – 2537` | 不适用 | **0** |
| `references`（24 份参考实现 × 2 记忆条件） | **48 / 48** | `394 – 2537` | `38 – 145` | **0** |
| `sampled_second`（二轮 `feedback`/`retry` 提示） | **48 / 48** | `485 – 2734` | 不适用 | **0** |
| **合计** | **120 / 120** | **最大 `2734`（余量 1362）** | **最大 `145`（余量 111）** | **0** |

## 2. 私有证据摘要

- 运行目录：`/private/tmp/lora-recal-120-tok-runs-20261121-1SgvyU/81b42b40fc9ffdfb506e0198bf7d8b96`
  - `wire.bin`（`2,073,501` bytes）：`0a46c80b7860b53cac355ea2936c5d2758ddd67aef1c6165b3962bd92aef7815`
  - `ssh-stderr.bin`（`0` bytes）：`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - `tokenizer-report-full.json`（`6,963,720` bytes）：`b36cb3f5ce7cfa4fbd0ddcd9ec96743bb1788962398aac7c185a188d6d766513`
  - `tokenizer-controller-summary.json`（`50,216` bytes）：`9d4886880dea9a2f0a8d2ed0cd9c7c066815dece4676b882a7855ce1b5751c45`
- 调用方捕获目录：`/private/tmp/lora-recal-120-tok-capture-20261121-GfJGXj`
  - `tokenizer-caller-return.json`：`dcff75565be76984a8a0468cdddefa053475d18daad47e1f5633c84fd3953f9d`

## 3. 结论边界

本次结果证明新版 12 道校准题在 `4096` 输入 / `256` 输出预算下具备充裕的长度空间（最大输入 `2734`，最大参考答案 `145`）；真实运行生成时每次调用前仍继续执行实时输入长度守卫（保留 `all_possible_second_rounds_validated=false`）。本次未加载预训练模型权重或 GPU，下一步进入 Gate G3（Qwen3-8B 的 72 次 `base`/`text` × `initial`/`feedback`/`retry` 配对校准诊断）。
