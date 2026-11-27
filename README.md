# TinyLLMForge

TinyLLMForge 是一个用 PyTorch 和 Triton 写的 LLM 推理引擎，起步参考
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。
主要在 Qwen3-0.6B / 8B 上做实验：从分页 KV cache 和多卡推理，到量化、
稀疏 attention，再到连续多步 decode。Qwen3.5 的混合状态与 MTP 也有单独的实验路径。

## Features

- **Runtime** — Paged KV cache、自动前缀复用、chunked prefill、混合调度、tensor parallel。
- **Decoding** — CUDA Graph、Exact Greedy Decode Burst、n-gram / SAM / learned draft。
- **Attention** — Quest 动态选块、Triton fused scoring、blockwise attention。
- **Memory** — 权重与 KV cache 量化、A8 / SmoothQuant、CPU offload、KV staging、多槽位 Prefix-to-LoRA 参数化前缀卸载。

## Spotlight

### 参数化前缀缓存（Prefix-to-LoRA）与信息论边界

自动前缀复用（APC）省掉了重复 prefill，但 decode 每生成一个新 token 仍需从 HBM 扫描数千 token 的共享前缀 KV；高并发下容易打满显存带宽甚至触发 OOM，多业务长前缀并存也会迅速耗尽块池。

这里把高频复用的 `3K–4.5K` token 规则/示范前缀离线蒸馏进 `7.31–32.62 MiB` 的 BF16 LoRA 槽位：`LLMEngine` 在入口通过最长前缀或双锚点模板匹配剥离静态长前缀（保留动态日期、会话头等可变字段），`QKVParallelLinear` 与 `mlp.down_proj` 在同一套静态 CUDA Graph 内按 token 槽位并行叠加旁路更新（兼容 `rank=8` 与 `rank=16` 同图混跑）。训练侧用无关轨迹全词表 KL 散度把参数更新锁死在通用能力的零空间内，并对语法脚手架清零硬标签损失，`4×4` 跨领域交叉挂载下 `136/136` 次异构请求实现 `0.0%` 格式污染。

这套机制有清晰的信息论与系统边界：
- **有损缓存区（`≤` 原长提示词）**：对单段固定上下文做纯 KL 蒸馏受数据处理不等式约束，本质是有损参数化近似，理论精度上限不超过原长提示词。它的工程价值不在 `2×–3×` 温和压缩区（该区间免训练 `INT8 KV` 即可无损覆盖），而在 **`10×–60×` 极致显存压缩与缩短序列加速区**——在 `10×` 压缩档位下，免训练静态稀疏驱逐（`SnapKV-10%`）在多分支合同规范上跌至 `2.5%`（`1/40`），而结合约 `75` token 高熵数字残差表的密集编译 LoRA 保持了 `17×` 于后者的严格通过率与 `3.8×` 推理提速。
- **跨入自进化的条件（`>` 单次长提示词）**：只有引入 **`Pass@K` 沙箱验真筛选**（向权重注入外部裁判提供的淘汰比特）或 **超上下文窗口规模的持续历史经验累积**（注入单次窗口装不下、塞进去反而引发注意力稀释的历史经验比特），参数化权重才能突破单次长提示词上限，从“省显存的有损 Cache”变成“持续自进化记忆”。

[运行时实现](tinyvllm/lora/prefix_lora_manager.py) ·
[代码域消融](experiments/lora_necessity/20261126-hybrid-diff-csd-results.md) ·
[跨领域与帕累托报告](experiments/cross_domain_prefix_lora/20261126-six-vulnerabilities-resolution-report.md)

### Decode Burst

CUDA Graph 减少了 kernel launch 开销，但普通 decode 仍要在每个 token 后回到
CPU。Decode Burst 把这个往返挪到一组 token 的末尾：GPU 上产生的 token 直接作为
下一步输入，连续执行 K 次，再统一回传结果、提交 KV 和序列状态。

没有 draft model，也没有猜测或接受率：**每个 token 仍然执行一次目标模型 forward**。
实现中包含固定 K8 和按上下文选择 K8 / K16 的策略。它省下的是同步、调度和
元数据处理，代价是 CPU 要等一组 token 算完才能看到结果。
[实现](tinyvllm/engine/exact_greedy_decode_burst.py) ·
[实验记录](docs/superpowers/audits/2026-08-24-context-gated-elastic-exact-burst-audit.md)

### KV8 + Quest

Quest 只读取选中的 KV blocks，但选块本身也要花时间。固定 top-k 在小 batch 下
反而会拖慢 KV8 decode，所以这里加了一个 batch 级判断：预计省下的反量化工作量
够大才选块，否则走 KV8 full attention。评分和 block summary 更新也做了 Triton fusion。
[实验记录](docs/kv-sparse-attention.md)

### W4A8 的长上下文修复

W4A8 + SmoothQuant 最初能生成文字，但长文检索会复读、丢信息。逐层消融后，
在这组 Qwen3-8B 实验里，保留尾部 4 层的浮点激活比保留首部 4 层更有效，
needle 召回达到 96.7%。相关配置和中间失败结果都留在
[量化日志](docs/qwen3-8b-fixes.md)里。

## Benchmarks

以下是几组已记录的对照，不是同一次测试，也不是把所有开关打开后的总收益。

| 实验 | 工作负载 | 对照与结果 |
| :--- | :--- | :--- |
| **CUDA Graph** | Qwen3-0.6B · A100 · TP1 · B=8 | eager **259.55** → graph **2342.10 tok/s** |
| **Decode Burst** | Qwen3-0.6B · A100 · TP1 · B=1 · context 256 / 2048 | 固定 K8 → 自适应 K8/K16：TPOT 中位数 **−2.85%**，P95 **−2.13%** |
| **Quest + fused score** | Qwen3-8B · 16K needle | 相对 full attention，吞吐 **+4.5%–6.8%**，召回 **100%** |
| **KV8 + 自适应 Quest** | Qwen3-8B · A100 · TP1 · eager · context 8192 · B=8–19 | 相对 KV8 full，decode step 延迟 **−26.7%–43.5%** |
| **W4A8 + SmoothQuant** | Qwen3-8B · 长上下文 needle | 尾部 4 层保留浮点激活，召回 **96.7%** |
| **Prefix-to-LoRA (SDPA 并发)** | Qwen3-8B · A100 · prefix 3050 + query 450 · B=32 / 64 | B=32 额外显存 **23.1 → 3.4 GiB**，吞吐 **+50.6%**；B=64 Warm KV **OOM** → LoRA **1095.6 decode tok/s** |
| **Prefix-to-LoRA (4 领域多槽位)** | Qwen3-8B · tinyvllm CUDA Graph · 32 并发 (代码/医/法/通用) | 相对 3 领域 Warm APC，峰值 Paged KV 块 **−39.8%**（省 **1.30 GiB**），吞吐 **+4.66%**，跨界污染 **0/136** |
| **Prefix-to-LoRA (公开 BBH)** | Qwen3-0.6B / 8B · N=120 × 3 seeds · 12-shot CoT 蒸馏 | 0.6B：**9.2%** (0-shot) → **50.0%±0.8%** (`p<2e-11`，超长提示词 **48.3%**）；8B：**13.3%** → **62.5%±3.0%** (`p<5e-15`) |
| **KV 压缩帕累托对照** | Qwen3-8B · N=40 合同规则 · ~10× 显存压缩档位 | 同为 ~50 MiB 驻留与 ~14.5s 耗时，`SnapKV-10%` **2.5% (1/40)** → `LoRA+微残差` **42.5% (17/40)** |

<details>
<summary>测试条件、代价和没通过的实验</summary>

- **CUDA Graph**：上表输入长度 256、输出长度 64。另一组 B=16、输入 1024、
  输出 256 的对照是 517.19 → 4421.90 tok/s。都是 decode 吞吐，不包含模型初始化。
  [记录：§18](docs/qwen3-8b-fixes.md)
- **Decode Burst**：表中是 K8/K16 对固定 K8 的增量收益，不是相对普通 decode 的收益。
  验证范围为单请求 greedy、非流式、`ignore_eos=True`；K16 最大主机可见间隔约 37 ms，
  当前默认关闭。[记录](docs/superpowers/audits/2026-08-24-context-gated-elastic-exact-burst-audit.md)
- **Quest fusion**：score-only 微基准的 9–22× 加速只属于评分 kernel，
  不能当作端到端加速。[记录：§22](docs/qwen3-8b-fixes.md)
- **KV8 自适应策略**：B=4/6 回退到 KV8 full，仍有约 1%–1.5% 的额外延迟；
  配套固定 needle 测试中，两条路径均为 25/25。[记录](docs/kv-sparse-attention.md)
- **Prefix-to-LoRA 边界与未通过分支**：① 推理前零参数隐藏层路由探针在异类题上关断率 100%，
  但短题干上同机制题召回率仅 25.0%（3/12），已放弃并改走训练期 KL 零空间约束；
  ② 将 `~1e-4` 量级的 $\Delta W$ 原地合并回 BF16 底座权重会被 7-bit 尾数吞噬（代码集 `8/24 → 6/24`），
  因此运行时保留独立 BF16 `(A, B)` 走旁路计算；③ 开启 Warm APC 时离线蒸馏一块 LoRA 需约 2 分钟，
  低并发单请求需万次级复用才摊平训练耗时，一次性冷启动长文档仍应直接走 KV cache。
  [记录](experiments/lora_necessity/20261125-hybrid-mlp-and-kv-serving-results.md) ·
  [六大边界闭环](experiments/cross_domain_prefix_lora/20261126-six-vulnerabilities-resolution-report.md)
- **Cohort Burst**：多请求版本在正式负载下出现输出不一致和尾延迟回归，未通过。
  [复盘](artifacts/slo_cohort_burst/20260914-canonical-960fde15-r1/final_bundle/report.md)
- **推测解码**：提案、验证、接受前缀提交和拒绝后缀回滚已有实现；adaptive n-gram、
  SAM 和 router 的历史实验没有通过各自门槛，learned drafter 也不能据此宣称整体完成。
  [历史结果](docs/experiment-history.md) ·
  [阶段审计](docs/superpowers/audits/2026-08-16-phase1-completion-audit.md)

历史结果绑定当时的源码和配置，不代表当前 HEAD 已重新跑过整套 GPU 验证。

</details>

## Notes

| 代码 | 记录 |
| :--- | :--- |
| [Engine / Scheduler / KV](tinyvllm/engine/) | [APC 与前缀复用](experiments/prefix_cache/README.md) |
| [Multi-slot Prefix LoRA](tinyvllm/lora/) | [跨领域与帕累托边界报告](experiments/cross_domain_prefix_lora/20261126-six-vulnerabilities-resolution-report.md) |
| [Attention / Triton kernels](tinyvllm/layers/) | [KV 稀疏 Attention](docs/kv-sparse-attention.md) |
| [Models](tinyvllm/models/) | [Qwen3-8B 工程日志](docs/qwen3-8b-fixes.md) |
| [Speculative decoding](tinyvllm/speculative/) | [历史实验汇总](docs/experiment-history.md) |
| [Profiler / tests / verifiers](tools/) | [Light Doc Cache](experiments/light_doc_cache/README.md) |

设计和审计在 [`docs/superpowers/`](docs/superpowers/)，部分实验的原始数据不随仓库分发。
