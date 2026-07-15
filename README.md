# TinyLLMForge
build a tiny LLM engine from scratch

- [] apply APC(automatic prefix caching)

## 实验收益概览

从 2026-05 以来，本项目围绕 Qwen3-0.6B / Qwen3-8B 做了多轮推理引擎实验，当前比较明确的收益如下：

- **CUDA Graph decode**：0.6B decode 场景下主要收益来自减少 launch overhead，实测约 **8–9×** 吞吐提升。
- **Quest fused score kernel**：Quest score-only 微基准约 **9–22×** 加速；端到端 needle 16K 里，在 100% 召回下约 **+4.5%~6.8% TPS**。
- **W4A8 + SmoothQuant + skip-last**：把 8B 长上下文量化路径从不可用/低召回救到约 **96%+ needle 召回**，关键结论是尾部层比首部层更值得保留 fp activation。
- **KV8 + Quest full-stack**：KV cache 8bit 量化节省约 **50% KV 显存**；叠加 Quest sparse-dequant 后，在高召回配置下约 **+33% TPS**，高吞吐折中档约 **+46% TPS**。
- **n-gram speculative decoding**：在重复 prompt / 高 acceptance / 模拟 128MiB H2D upload 的长上下文场景中，wall-clock 约 **2.04×**，模拟 upload 成本约 **-54%**；该收益依赖场景，暂不作为通用默认路径。
- **KV offload / blockwise attention**：已打通 GPU staging、dirty writeback、H2D reload、prefetch/eviction 与 blockwise attention 正确性；局部 `gpu_blocks=4` matrix 中 H2D / eviction 计数约 **-33%**，但端到端 tok/s 收益仍需更严格 benchmark 证明。
- **DFlash profiler-only**：已完成 hidden-to-draft / draft-model-stub ABI、batch schema 与 contract 验证；当前不接 runtime，因此对现有推理速度 **0% 直接收益**，主要价值是降低未来接真实 draft model 的风险。

## Adaptive n-gram speculative decoding canonical gate

2026-07-15 在 Qwen3-0.6B 上完成了严格的 greedy、单序列 canonical gate：4 类固定 prompt × 5 个隔离 policy（normal greedy、fixed K1/K2/K4、adaptive K∈{1,2,4}）× 7 次重复，共 140 个独立进程。复现入口：

```bash
RUN_TAG=qwen3-06b-canonical-20260715-025426 \
LOCAL_OUT=$PWD/experiments/adaptive_ngram/qwen3-06b-canonical-20260715-025426 \
CUDA_VISIBLE_DEVICES=3 \
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
tools/run_adaptive_ngram_gate_remote.sh canonical
```

Canonical 五件套位于：
`experiments/adaptive_ngram/qwen3-06b-canonical-20260715-025426/`。

| Policy | aggregate median tok/s |
|---|---:|
| normal greedy | 33.815941 |
| fixed K1 | 32.979915 |
| fixed K2 | 32.925531 |
| fixed K4 | 37.962906 |
| adaptive K∈{1,2,4} | 37.574839 |

结果与边界：

- 140/140 行完整，所有进程返回 0；同 prompt 的候选输出与 normal greedy token 序列完全一致，trajectory replay 和 adaptive exercise 均通过。
- Adaptive 相对 normal greedy 为 **+11.12%**，但相对最佳 fixed K4 为 **-1.022%**。
- Adaptive 的聚合 token acceptance 为 **91.23%**；相对 fixed K4，median wasted draft tokens 从 `24` 降到 `15`（**-37.5%**），median zero-accept verify cost 从 `224.830 ms` 降到 `180.475 ms`（**-19.73%**）。
- 固定 gate 要求 adaptive 至少比最佳 fixed 快 2%；或者落后不超过 1%，同时 waste 至少降低 20%、zero-accept cost 至少降低 15%。后两项通过，但 `-1.022% < -1%`，因此最终结论严格为 **NO_GO**，没有在观察结果后放宽阈值。
- 该结论只覆盖记录的 Qwen3-0.6B、greedy、单序列、固定 prompt bank 和 profiler-owned 路径；不证明 ragged/batched target verification、生产 batch throughput、queueing tail latency、memory-capacity reduction，亦不能外推到其他模型。
- 保留 adaptive policy、correctness verifier 和 gate 基础设施用于后续研究；当前只可在已验证的高重复单序列 regime 中考虑 fixed K4。下一优先方向是更高质量的 draft source，而不是继续在同一 gate 上调 adaptive 阈值。

## Prompt + Dynamic SAM drafter gate

2026-07-15 完成了 profiler-owned token suffix automaton drafter、match-aware
`K∈{0,4,8,16}`、严格五文件 gate、resume、动态端口和隔离远端 runner。
最初 smoke 因 multi-token verifier 的 KV materialization 与逐 token decode
不等价而判为 `INCOMPLETE`。修复方式是在 acceptance 判定不变的前提下，使用
正常 decode kernel 重物化 `accepted_tokens[:-1]` 的 KV。修复后 25-row smoke
exactness、trace 和 policy exercise 均通过，最终严格结论为 **NO_GO**。

常用命令：

```bash
tools/run_sam_drafter_gate_remote.sh preflight
CUDA_VISIBLE_DEVICES=5 tools/run_sam_drafter_gate_remote.sh smoke
CUDA_VISIBLE_DEVICES=5 tools/run_sam_drafter_gate_remote.sh canonical
RESUME=1 RUN_TAG="${RUN_TAG}" \
  CUDA_VISIBLE_DEVICES=5 tools/run_sam_drafter_gate_remote.sh canonical
python3 tools/sam_drafter_gate.py verify \
  --out-dir experiments/sam_drafter/qwen3-06b-sam-remat-smoke-20260715-162323
```

修复后严格 smoke 证据位于：
`experiments/sam_drafter/qwen3-06b-sam-remat-smoke-20260715-162323/`。
它包含 5 prompts × 5 policies × 1 repetition = 25 个独立进程，五件套为
`manifest.json`、`raw_rows.json`、`event_rows.json`、`summary.json` 和
`report.md`。

验证结果：

- 25/25 rows、进程、动态端口和 artifact 可独立重建；SAM trace
  reconciliation 通过。
- Match-aware SAM 覆盖了 `K=0/4/8/16`、prompt/generated continuation、
  zero accept 和 fully accepted multi-token proposal。
- `runtime_mutation=false`、`profiler_owned=true`；没有修改
  `LLMEngine.step()`、scheduler、`Sequence` 或公开生成 API。
- exact-output correctness 已通过。远端两-prompt A/B 和完整 smoke 均证明
  accepted KV 重物化修复了稳定分叉。
- aggregate median tok/s：baseline `32.5173`、n-gram K4 `25.9687`、
  adaptive n-gram `25.7055`、fixed SAM K16 `28.3399`、match-aware SAM
  `28.6274`。
- match-aware SAM 相对 baseline 为 **-10.72%**；相对 n-gram K4 为
  **+8.44%**。它减少 verify attempts `25%`，但 drafted waste 相对 K4
  增加约 `218%`。
- 除 transition-heavy `+5.95%` 外，natural `-10.72%`、structured
  `-14.02%`、repeated `-7.40%`、prompt-copy `-26.75%` 均触发固定的
  per-class regression gate。
- SAM 两个 policy 合计需要 588 次 accepted-KV decode 重物化；仅
  `sam_match_aware` 的重物化累计约 `8.76s`。正确性 fallback 抵消了 draft
  source 的潜在吞吐收益。

因此无需消耗 175 个模型进程去确认一个已经大幅越过阈值的负结果；保留
25-row 五类 prompt smoke 作为严格 `NO_GO` 证据。下一方向不是继续调 SAM
match-aware 阈值，而是让 batch verifier 原生生成与正常 decode 等价的 KV，
或等待兼容 learned drafter/checkpoint 后重新设计 gate。

当前结论只覆盖 Qwen3-0.6B、greedy、单序列、profiler-owned 路径；不证明
ragged/batched correctness、production batch throughput、queue tail latency、
non-greedy correctness或 memory reduction。

## 参考资料
```
https://github.com/GeeeekExplorer/nano-vllm
https://space.bilibili.com/362867186?spm_id_from=333.788.upinfo.detail.click
https://zhuanlan.zhihu.com/p/1932035278089987994
https://zhuanlan.zhihu.com/p/1932473745584394614
https://zhuanlan.zhihu.com/p/1925484783229698084
