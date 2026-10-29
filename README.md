# TinyLLMForge

**一个从零搭建、持续优化的个人 LLM 推理引擎项目。**

以 Python、PyTorch 和 Triton 为基础，围绕 Qwen3-0.6B / Qwen3-8B，
从请求调度、分页 KV cache 和模型执行做起，逐步扩展到 CUDA Graph、
量化、稀疏 Attention、KV offload 和推测解码，并探索 Qwen3.5 混合状态模型。

核心关注点是：**推理时间花在哪里、显存被什么占用，以及一个优化究竟在什么条件下有效。**

## 做了什么

- **搭建推理引擎**：实现请求调度、prefill / decode、分页 KV cache、
  自动前缀复用（APC）、采样和张量并行。
- **优化执行与访存**：引入 CUDA Graph、Exact Greedy Decode Burst、
  Triton kernel fusion、Quest 动态选块，以及权重、激活和 KV cache 量化。
- **探索更复杂的运行时**：实现 chunked prefill、混合调度、CPU/GPU KV
  staging 与 blockwise attention，研究多种 draft source 的推测解码。
- **建立实验验证链路**：配套 profiler、质量评测、固定对照实验和独立 verifier，
  同时记录性能收益、正确性问题和没有通过门槛的方向。

## Highlights

以下数字来自各自的历史对照实验，不是当前 HEAD 的统一跑分，也不能相乘作为总加速比。

### 1. CUDA Graph：减少小模型 decode 的启动开销

把重复的 GPU 执行序列捕获并重放，减少逐算子 launch 开销。
在 **Qwen3-0.6B / A100** 的历史对照中，单卡 decode 吞吐从
**259.55 → 2342.10 tok/s**；更大 workload 下从 **517.19 → 4421.90 tok/s**，
约为 eager 的 **8.5–9×**。

亮点不仅是数字，也在于明确了这个小模型场景的主要瓶颈是启动开销，
而不是简单地增加 GPU 数量。
详见 [CUDA Graph 对照与纠错，§18](docs/qwen3-8b-fixes.md)。

### 2. Decode Burst：跨多个 token 摊薄同步与调度开销

普通 decode 每生成一个 token，就回传 CPU、更新调度和元数据，再启动下一步。
Exact Greedy Decode Burst 让 GPU 上的输出直接成为下一步输入，连续执行 K 步，
最后一次性回传 token 并提交 KV / 序列状态，减少逐 token 的主机往返。
**它不是推测解码**：不需要 draft model，每个 token 仍执行一次目标模型 forward。

在 **Qwen3-0.6B / A100 / TP1 / 单请求 greedy** 的历史实验中，
自适应 K8/K16 相对固定 K8，在符合条件的 256 / 2048-token 上下文中，
**平均每 token 耗时（TPOT）的中位数降低 2.85%，P95 降低 2.13%**，
并通过输出一致性验证。代价是 token 批量可见，K16 最长主机可见间隔约 **37 ms**；
该结果仅覆盖非流式、`ignore_eos=True` 的固定范围，当前仍默认关闭。

后续多请求 SLO-aware Cohort Burst 暴露输出不一致与尾延迟回归，不能作为成功加速。
详见 [单请求 Decode Burst 审计](docs/superpowers/audits/2026-08-24-context-gated-elastic-exact-burst-audit.md)
与 [Cohort Burst 实验结论](artifacts/slo_cohort_burst/20260914-canonical-960fde15-r1/final_bundle/report.md)。

### 3. Quest + Triton：从算子融合到端到端收益

将 Quest 选块评分中的 gather、乘法和归约融合为一个 Triton kernel，
减少中间张量与 kernel launch。**评分算子微基准加速约 9–22×**；
在 Qwen3-8B 的 16K needle 评测中，相对 full attention，
Quest + fused score 在 **100% 召回**下取得 **4.5%–6.8% 吞吐提升**。

这是两种不同口径：前者衡量 kernel 优化，后者衡量整条稀疏 Attention 路径。
详见 [Quest kernel fusion，§22](docs/qwen3-8b-fixes.md)。

### 4. KV8 + 自适应 Quest：不是一直稀疏，而是在划算时稀疏

针对固定开启 Quest 在小 batch 下反而变慢的问题，引入 batch 级启用策略：
只有预计能省下足够多的 KV8 反量化 block，才执行稀疏选块，否则回退到 KV8 full。

在 **Qwen3-8B / A100 80GB / TP1 / eager / 8192 context** 的固定实验中，
**B=8–19 的 decode step 延迟相对 KV8 full 降低约 26.7%–43.5%**；
B=4/6 回退，额外延迟约 1%–1.5%。配套固定 needle 测试中，
KV8 full 与自适应策略均为 **25/25**。

这个结果的价值在于把 selector 的固定成本纳入决策，而不是假定稀疏一定更快。
详见 [KV8 + Quest 自适应摊销实验](docs/kv-sparse-attention.md)。

### 5. 量化质量修复：找出真正敏感的层

不只实现 W4A8、KV4/KV8 和 SmoothQuant，也追踪量化后长上下文复读、
检索召回下降的原因。通过层级消融，发现该 Qwen3-8B 配置中
**尾部层比首部层更值得保留浮点激活**。

W4A8 + SmoothQuant 配合跳过尾部 4 层 A8，在记录的长上下文 needle
实验中达到 **96.7% 召回**。这里的亮点是定位误差来源、做精度分配，
而不是把“能生成 token”当作量化成功。
详见 [量化修复与层级消融](docs/qwen3-8b-fixes.md)。

### 6. 推测解码与 KV 生命周期：不止是生成几个 draft token

围绕 n-gram、SAM、独立 learned drafter 和 Qwen3.5 MTP，
构建提案、验证、接受前缀提交、拒绝后缀回滚，以及 Proposal-KV 的释放与驻留管理；
把 draft source 和目标模型的 KV 状态管理分开。

这部分值得保留的是运行时结构与正确性验证能力，而不是一个通用加速承诺。
历史 adaptive n-gram、SAM 和 router 实验也保留了未通过门槛的结果，
用于区分“提案接受率高”“KV 状态正确”和“端到端确实更快”。
实现见 [`tinyvllm/speculative/`](tinyvllm/speculative/)；
结果与边界见 [历史实验记录](docs/experiment-history.md)和
[阶段审计](docs/superpowers/audits/2026-08-16-phase1-completion-audit.md)。

## 实验记录

- [KV 稀疏 Attention](docs/kv-sparse-attention.md)：Quest 与 KV8 自适应策略。
- [Qwen3-8B 工程日志](docs/qwen3-8b-fixes.md)：量化、kernel、offload 与性能诊断。
- [APC / Prefix Cache](experiments/prefix_cache/README.md)：跨请求前缀复用与 TTFT 验证。
- [Light Doc Cache](experiments/light_doc_cache/README.md)：KV 恢复与压缩研究。
- [历史实验汇总](docs/experiment-history.md)：原首页的性能数字、失败结论与证据路径。

项目起步参考 `GeeeekExplorer/nano-vllm`，之后围绕上述推理优化方向持续扩展。
