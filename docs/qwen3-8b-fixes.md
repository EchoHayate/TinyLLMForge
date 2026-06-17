# Qwen3-8B 上验证与一系列纠错（2026-05-29 ~ 2026-05-30）

> 主线（§13 / `regression-tp.md`）的 8 条 config 在 0.6B 上跑通后，搬到 8B 暴露的"质量塌"
> + "度量错"+ "事实错"+ "对照不公"，集中在这一卷。每节是一个独立 commit / 一组 commit。

## 14. 7B（Qwen3-8B）上验证 W4A8C4 + TP 显存切分（2026-05-29）

### 14.1 动机与选模型

§13 在 0.6B 上看到 W4 / KV4 叠加时输出复读，怀疑是模型太小撑不住量化误差，
真正能否用要在中等模型上验。这一波目标：
**(a) 把 W4A8C4（W4A8 + KV4 全栈量化叠加）跑到 7B 量级看质量；(b) 顺带验"TP 是否在大模型上真切显存"**。

选模型：仓库代码是 `Qwen3ForCausalLM`，必须用 Qwen3 架构。modelscope 上 Qwen3 系列**没有 7B**
（只有 1.7B / 4B / 8B），改用 **Qwen3-8B**（fp16 ~16GB，最接近 7B 量级，架构原生兼容）。

### 14.2 跑法

扩展 `tools/tp_smoke.py`：
- 加 `w4a8c4` 配置（`int4 weight + act_quant_bits=8 + kv_quant_bits=4`），上波 §13 漏了这条核心
- 加 `--prompt-source english`：8 条固定英文 prompt + 贪心采样，跨配置可对比
- 收集前 3 条 `text_samples` 而不是只首条

跑了 4 条 TP=1（baseline / c4_only / w4a8 / w4a8c4）+ 4 条 TP=2（同上）共 8 跑次。

### 14.3 结果

A100 + Qwen3-8B + `gpu_memory_utilization=0.7` + `num_seqs=8, max_input=256, max_output=64`：

| config | tp | decode_tps | peak_gb | 质量（首条 sample 节选） |
|---|---|---|---|---|
| baseline | 1 | **187.40** | 53.13 | ✅ "Paris. The capital of France is Paris..." |
| c4_only | 1 | 120.74 | 53.17 | ⚠️ "Paris. The capital of Paris is... Hmm" |
| w4a8_g128 | 1 | 34.99 | 51.73 | ❌ `\  \     \    \    \  ...` |
| w4a8c4 | 1 | 33.67 | 51.74 | ❌ `\  \    \  \  \  \   \  ...` |
| baseline | 2 | 155.36 | 53.51 | ✅ "Paris. The capital of France is Paris..." |
| c4_only | 2 | 95.84 | 53.52 | ⚠️ "Italy is Rome. The capital of Germany is...?" |
| w4a8_g128 | 2 | 63.85 | 52.86 | ❌ `                 \                  ` |
| w4a8c4 | 2 | 60.28 | 52.87 | ❌ `         \          \      \  ...` |

### 14.4 三个明确结论

**(1) W4A8 实现在 8B 上不能用 — 不是 TP 的锅**

W4A8 / W4A8C4 在 TP=1 单卡也塌（一模一样的反斜杠），坐实**根因是 W4A8 量化路径本身**，
不是 TP 引入的回归。0.6B 上 §10 早就观察到复读，本以为是模型太小，现在 8B 上**塌得更彻底**
（连复读都不复读，直接吐 backslash），说明 W4A8 实现存在数值问题，至少包括以下可能：
- act 量化的 dynamic scale 在 layer 之间累积误差
- W4 dequant + A8 matmul 的 fused kernel 数值不稳
- group_size=128 对 8B 隐层维度（`hidden_size=4096`，每行 32 组）颗粒度可能仍不够

但**单 KV4（c4_only）质量勉强在线**：TP=1 出"Paris. Hmm" 略迷糊但逻辑通；TP=2 出"Italy is Rome"
（事实错但格式对）。说明 **C4 路径自己是 OK 的，问题集中在 W4A8**。

**(2) TP 在 8B 上仍然没真切显存**
> 注：本条结论已被 §15 纠错。`peak_mem_gb` 是 weight + KV cache 的并集，KV cache 自动吃满
> 剩余显存把 weight 切下来的空间又加了回来。weight 自己其实切了，详见 §15。

| config | tp=1 peak | tp=2 peak | 切分了吗 |
|---|---|---|---|
| baseline | 53.13 | 53.51 | ❌ 反而略多 |
| c4_only | 53.17 | 53.52 | ❌ 反而略多 |
| w4a8_g128 | 51.73 | 52.86 | ❌ 反而略多 |
| w4a8c4 | 51.74 | 52.87 | ❌ 反而略多 |

延续 §13 的现象。根因是 KV cache 容量按 `gpu_memory_utilization * 总显存 - 已用` 反算，
TP=2 时**每张卡都独立按"剩余显存的 70%"分 KV cache**，把 weight 切分省下来的空间又吃回去。
这不是 TP 实现的 bug，是配置策略的问题——要看到真切分效果，得把 `gpu_memory_utilization`
按 tp_size 自动缩，或者直接显式指定 `kv_cache_ratio`。这一波**只验通路兼容**，节省策略的修
留给后续如果真要用 TP 跑大模型再做。

**(3) TP=2 decode TPS 比 TP=1 慢（除了 W4A8 路径）**

| config | tp=1 tps | tp=2 tps | 比值 |
|---|---|---|---|
| baseline | 187.40 | 155.36 | 0.83× |
| c4_only | 120.74 | 95.84 | 0.79× |
| w4a8_g128 | 34.99 | 63.85 | **1.83×** |
| w4a8c4 | 33.67 | 60.28 | **1.79×** |

baseline / C4 路径 TP=2 比 TP=1 慢约 17–21%（NCCL 通信开销 > 计算并行收益，符合 8B + bs=8 这个量级）。
W4A8 路径反而 TP=2 比 TP=1 快接近 2×：怀疑是 weight dequant 是计算瓶颈，TP 把 dequant 也并行了；
但既然 W4A8 输出垃圾，这个 TPS 不能当真实可用吞吐采纳。

### 14.5 已知不做

- ❌ **修 W4A8 量化在 8B 上的塌**：根因可能在 dynamic act scale / dequant kernel 数值，
  追这个要先在 0.6B 上重做 ppl 漂移定位 → 找到坏层 → 重做 group_size 扫描或换 per-channel scale，
  工程量大；本仓库主线是稀疏注意力 (Quest/C4)，W4A8 本来就是 §9 加的辅料
  → 后续 §16 给了"轻量修法"，§20 给了 SmoothQuant 完整方案
- ❌ **TP 节省策略修复**（按 tp_size 缩 gpu_memory_utilization）：偏离主线，0.6B/8B 单卡都跑得动
  → 后续 §15 纠错，根本不需要修
- ❌ **跑 Qwen2.5-7B**：modelscope 上有，但要补 Qwen2 架构 modeling，工作量超出 smoke 验证范畴
- ❌ **加 perplexity 评测**：贪心 + 固定 prompt 的 text_sample 已经能定性判断"塌"，
  跑 ppl 要拉 wikitext，磁盘只剩 96GB，性价比低

### 14.6 文件留痕

- `tools/tp_smoke.py`：加 `w4a8c4` 配置 + `--prompt-source english` + 多条 text_samples
- `tp_smoke_out/tp_smoke_8b_tp1.json`：TP=1 baseline
- `tp_smoke_out/tp_smoke_8b_tp1_quant.json`：TP=1 c4_only / w4a8 / w4a8c4 对照
- `tp_smoke_out/tp_smoke_8b_tp2.json`：TP=2 baseline + 量化叠加 4 条
- 远端模型：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B`（modelscope 拉，~16GB，不入仓库）

## 15. TP 显存切分纠错：拆分 weight / KV cache 度量后的真相（2026-05-29）

### 15.1 上一波的错误结论

§13 / §14 都断言"TP 没真切显存"，证据是 TP=1 vs TP=2 的 `peak_mem_gb` 几乎一样（都 ~53GB）。
当时给出的解释是"KV cache 按 `gpu_memory_utilization * 总显存` 反算时每张卡都拿到独立的 70%，
把 weight 切下来的省了又吃回去"。

### 15.2 重新审视：peak 是错的度量

`tools/tp_smoke.py` 只读 `torch.cuda.max_memory_allocated()`，这个值天然包含 **weight + KV cache + 临时 buffer**。
KV cache 在 `allocate_kv_cache()` 里按"剩余显存的 util 比例"自动填满 → **peak 总会逼近上限**，
不论 TP 是否真切了 weight。**用 peak 判断 TP 切分根本不可能得到正确结论**。

### 15.3 修：把 weight 单独度量出来

`tinyvllm/engine/model_runner.py::allocate_kv_cache` 开头加一行：

```python
# 记一次 weight-only 占用（KV cache 还没分），方便外部观察 TP 是否真切到了 weight
self.weight_mem_bytes = torch.cuda.memory_allocated()
```

`tools/tp_smoke.py` 把它和 `peak - weight`（≈ KV cache）一起写进 result：

```python
wmb = getattr(llm.model_runner, "weight_mem_bytes", 0)
result["weight_mem_gb"] = round(wmb / (1024**3), 3)
result["kv_cache_mem_gb"] = round((torch.cuda.max_memory_allocated() - wmb) / (1024**3), 3)
```

### 15.4 真相：TP 一直在切 weight

A100 + Qwen3-8B + `gpu_memory_utilization=0.7`：

| config | tp | **weight_gb** | kv_gb | peak_gb |
|---|---|---|---|---|
| baseline | 1 | **15.288** | 37.847 | 53.135 |
| baseline | 2 | **7.660** | 45.845 | 53.506 |

`7.66 ≈ 15.29 / 2`，**weight 完美切半**。peak 看起来一样，是因为 KV cache 把多出来的 ~8GB
自动填满了——这本来就是 `gpu_memory_utilization` 的设计意图（把卡用满，越多 KV 越能容纳并发）。

### 15.5 重新审视 §13 / §14 的"已知不做"条目

§14.5 列了一条："修 TP 节省策略（按 tp_size 缩 gpu_memory_utilization）"。
**这条根本不需要修**。`gpu_memory_utilization` 的语义是"每张卡用满到多少"，
TP=2 时每卡分到自己的 weight + 额外 KV cache 空间，这是正常用法。
真要省整体显存，用户应该自己显式设小 `gpu_memory_utilization`，或显式给 `num_kvcache_blocks`。

### 15.6 结论

- ✅ TP 实现是对的，weight 真切了
- ✅ `gpu_memory_utilization` 行为是对的，KV cache 自动吃满空间是设计意图
- ❌ 上波的"TP 不省显存"是**度量错误**，不是 TP 实现 bug
- 度量层修复后，后续所有 mem 相关分析必须分 weight / kv / peak 三档看，不能只看 peak

### 15.7 文件留痕

- `tinyvllm/engine/model_runner.py`：`allocate_kv_cache` 开头记 `self.weight_mem_bytes`
- `tools/tp_smoke.py`：result 加 `weight_mem_gb` / `kv_cache_mem_gb`，summary 表头同步
- `tp_smoke_out/tp_smoke_8b_tp{1,2}_split.json`：纠错后的 TP=1/2 baseline 度量结果

## 16. W4A8 / W4A8C4 数值塌：activation outlier 软裁剪 + W4 g32（2026-05-30）

承接 §14.4：W4A8 / W4A8C4 在 8B 上吐 `\\\\\\\` 全塌，0.6B 复读。本节在不引入 SmoothQuant
的前提下，给一个"轻量修法 + 诚实标注边界"的修复。

### 16.1 病因定位

W4A8 path = `dequant_int4(W) @ fake_quant_act_int8(x)`。两个嫌疑点：

1. W4 的 group_size=128 在 8B（hidden=4096）上太粗，每组覆盖 128 列，outlier 把组 scale 拉爆；
2. 激活 fake-quant 用纯 amax 当 scale。transformer 激活有约 1% outlier channel，absmax 比典型
   channel 高 10–100×，单 outlier 让整 token 的 scale 拉爆，典型 channel 的小值全 round 到 0。
   叠加 W4 量化误差后 logits 直接乱掉。

实验佐证（TP=1，单卡 0.6B）：
- 把 `fake_quantize_act_int8` 的 scale 改成 `min(absmax, p999*1.5) / 127`：0.6B 从复读修到能续句；
- 8B 上单改激活 fake-quant 还不够；再叠 W4 g32 才能让 W4-only 跑出 "Paris / London / 阶乘"。

### 16.2 修法

#### 16.2.1 激活 fake-quant 软 outlier 裁剪

`tinyvllm/layers/quantization.py::fake_quantize_act_int8`：

```python
abs_x = x_fp.abs()
abs_max = abs_x.amax(dim=-1, keepdim=True).clamp_min(1e-8)
p999 = _percentile_along_last(abs_x, 0.999).clamp_min(1e-8)
clip_val = torch.minimum(abs_max, p999 * 1.5)   # 软 outlier 保护
s = clip_val / 127.0
```

每 token 仅允许 0.1% channel 当 outlier，其余被 clip 到 1.5×p999。无校准开销，纯 runtime。

试过 p99 × 1.2，太激进，把"次 outlier"的真信号也 clip 没了，反而更差，回退 p999 × 1.5。

#### 16.2.2 W4 group_size=32（8B 必须）

`tools/tp_smoke.py` 加 `w4_g32` / `w4a8_g32` / `w4a8c4_g32` 三条 config，
weight 分组从 128 缩到 32。代价：scales 体积 ×4，weight 显存从 5.79 → 6.42 GB（+0.62 GB / 11%）。

### 16.3 8B TP=1 结果（`tp_smoke_8b_tp1_quantfix.json`）

| config | weight_gb | text 质量 |
|---|---|---|
| w4_g128 | 5.794 | ❌ `C C " "═══` 全塌 |
| **w4_g32** | 6.417 | ✅ "Paris / London / 阶乘"，连贯 |
| w4a8_g128 | 5.794 | ❌ `ffff...` 全塌 |
| w4a8_g32 | 6.417 | ⚠️ 不塌但短语复读：`is is is is...`、`small village near the mountains, ...` 循环 |
| w4a8c4_g128 | 5.794 | ❌ `ffff...` 全塌 |
| w4a8c4_g32 | 6.417 | ⚠️ 同 w4a8_g32：不塌但复读 |

decode_tps 全部 32–36 tok/s，没退化。

### 16.4 三个明确结论

- ✅ **W4 single quant 修好了**：W4 g32 在 8B 上能正常输出。
- ⚠️ **W4A8 / W4A8C4 仍偏弱**：从"全塌"修到"不塌但复读"，达到 §14.4 的最低目标
  ("能跑出像样输出"，但显然不接近 fp16 质量)。
- ❌ **不要期待轻量修法替代 SmoothQuant**：transformer 激活 outlier 是结构问题，
  per-token p999 软裁剪只能压一部分。要真正修好 W4A8 必须把 outlier 从激活迁移到权重
  （SmoothQuant：`x' = x / s`，`W' = diag(s) W`），需要离线校准一遍激活 scale，本期不做。
  → 已在 §20 落地为完整 SmoothQuant 管线。

### 16.5 已知不做

- SmoothQuant / GPTQ / AWQ：需要离线校准管线，超出 toy infra 工程范围。
  → 已在 §20 完成 SmoothQuant 落地。
- W4A8 上线产品质量：当前 W4A8 仅作为"全栈量化能跑通"的演示路径，不作为推荐配置。
- KV cache 4-bit 在 8B 上的精度（D 项）：见后续 §17。

### 16.6 文件留痕

- `tinyvllm/layers/quantization.py`：`fake_quantize_act_int8` p999 软裁剪 + docstring 标注边界
- `tools/test_weight_quant.py`：`test_fake_quant_act` 改为 MSE-based 验证（单值上界放宽到 amax）
- `tools/tp_smoke.py`：加 `w4_g32` / `w4a8_g32` / `w4a8c4_g32` 三条 config
- `tp_smoke_out/tp_smoke_8b_tp1_quantfix.json`：8B 量化修复后 smoke 结果

## 17. C4 在 8B 上事实错误：根因是 group_size=head_dim（2026-05-30）

承接 §14.4：C4 (g=128) 在 8B 上跑 "The capital of France is" 续出 "Paris. The capital of
Paris is... Hmm" — 格式对、事实错、还自言自语。本节定位并修复。

### 17.1 病因

Qwen3-8B `head_dim = 128`，C4 默认 `kv_quant_group_size = 128`。
**group_size == head_dim 意味着每个 KV head 只共享 1 个 scale**，等于 per-head（不是 per-group）量化。
KV 中常见现象：head 内不同维度的能量分布差异大（attention/ROPE 后某些维度专门承载位置信息，
absmax 比其他维度大若干倍），单 scale 拍下去会把"小 amplitude 维度"全 round 到 ±1/±2，
信息丢失到事实开始漂。

§13 在 0.6B 上没暴露这条，是因为 0.6B 的 `head_dim` 也是 128，但模型表征能力弱，"漂事实"被
"模型本来就不会说事实"掩盖了；8B 上事实信号明确，漂就显出来了。

### 17.2 修法 + 验证

不动默认值（保 §14 的对照可重现），通过 `kv_quant_group_size` 直接传更小的值即可。
扩展 `tools/tp_smoke.py` 加 `c4_g64` / `c4_g32` 两条 config，TP=1 / Qwen3-8B / 英文 prompt：

| config | decode_tps | weight_gb | kv_gb | text_sample |
|---|---|---|---|---|
| c4_only (g=128) | 118.44 | 15.288 | 37.88 | ⚠️ "Paris. The capital of France is Paris. The capital of Paris is... Hmm" |
| **c4_g64** | 122.40 | 15.288 | 37.88 | ✅ "Paris. The capital of Germany is Berlin. The capital of Italy is Rome..." |
| **c4_g32** | 115.19 | 15.288 | 37.88 | ✅ "Paris. The capital of Italy is Rome. The capital of Spain is Madrid..." |

### 17.3 三个明确结论

- ✅ **g=64 / g=32 都修好事实漂**：能正确续出 Berlin / Rome / Madrid。
- ✅ **TPS 与显存几乎不变**：g 缩小后 scale 张量占的字节数随之增长，但相对 KV cache 是 1‰ 量级，
  KV mem 三档 37.88 GB 完全一致。
- ✅ **g=64 是 sweet spot**：TPS 反而最高（122 vs g=128 的 118），可能是 g=64 的 triton kernel
  调度更友好（HALF=32，align 32 lane）。

### 17.4 推荐用法

`kv_quant_group_size` **必须严格小于 `head_dim`**，等于的话退化成 per-head 量化、事实漂。
对 head_dim=128 的模型（Qwen3-8B、Llama2-7B 等）推荐 64；对 head_dim=64 的小模型推荐 32。

未改默认（§14 对照可复现），调用方自己传。

### 17.5 已知不做

- 改默认值：会影响 §13 / §14 的对照实验复现。后续如果要做 release 默认值再改。
- 加 runtime warn `group_size >= head_dim`：可加但与"用户应理解参数语义"矛盾，留给文档。
- per-token 而非 per-channel KV scale：C4 已经是 per-(slot, head, group) scale，
  再切 token 维度会让 scale 张量翻倍，性价比低。

### 17.6 文件留痕

- `tools/tp_smoke.py`：加 `c4_g64` / `c4_g32` 两条 config
- `tp_smoke_out/tp_smoke_8b_tp1_c4sweep.json`：C4 group_size 扫描结果

## 18. F 项纠错：cuda_graph_baseline 1798 tps 不是异常，是对照不公平（2026-05-30）

§13.5 把 TP=2 0.6B 的 `cuda_graph_baseline decode_tps=1798` 标成"⚠️ 不正常"，理由是
"§11 单卡同样配置只有 ~250 tps"。重做对照后：**这是错误的怀疑，单卡 + cuda graph 在
smoke 配置下本来就能上 2300 tps。**

### 18.1 重做对照（0.6B，A100，smoke 配置 n=8, in=256, out=64）

| tp | config | decode_tps | text_sample（首条） |
|---|---|---|---|
| 1 | baseline | 259.55 | "Paris. Italy is Rome. Spain is Madrid. China is Beijing. Japan..." |
| 1 | cuda_graph_baseline | **2342.10** | "Paris. Italy is Rome. Spain is Madrid. China is Beijing. Japan..." |
| 2 | baseline | 209.22 | 同上 |
| 2 | cuda_graph_baseline | **1896.06** | 同上 |

更大 workload（n=16, in=1024, out=256）单卡：

| config | decode_tps |
|---|---|
| baseline | 517.19 |
| cuda_graph_baseline | **4421.90** |

### 18.2 三个明确结论

- ✅ **graph 输出与 eager 完全一致**：TP=1 / TP=2 上贪心采样三条 text_sample 字字对齐，
  完全没有 §13.5 担心的"NCCL all-reduce 被吞 → garbage 但快"。
- ✅ **8–9× 倍速是真实 launch overhead 收益**：0.6B 单层 attention/FFN 在 A100 上几十微秒，
  Python 侧 schedule + per-op kernel launch + per-step 同步成本占总 step 时间的 80%+。
  graph 把这些全消掉，吞吐基本只受 kernel 本身限制。
- ❌ **§13.5 标的"⚠️"是错误的怀疑**：之前拿 `bench.py n=16 ctx=4000` 的 ~250 tps 跟 smoke
  `n=8 in=256 out=64` 的 1798 tps 对比，前者 per-step 计算重 + ctx 长 KV gather 成本高，
  根本不是同一个 workload。同 workload 单卡也能上 2300 tps，1798 在 TP=2（多通信成本）下完全合理。

### 18.3 为什么之前会误判

§11 / §13 同期上线，§13 写"异常观察"时没有跑单卡同 workload 对照，只凭"§11 我记得是 250"
做了横向对比。教训：**cross-section 比数字时必须跑同 workload，不能跨脚本（bench.py vs
tp_smoke.py）凭印象比**。

### 18.4 已知不做

- 改 §13.5 / §13.6 表格：保历史轨迹完整，本节作为修正注解；§13.6 的"⚠️ TP × cuda graph"
  请阅读到本节为止。
- 重测 §11 ablation 行：§11 是 bench.py 的 long-ctx 数据，本身没问题，不需要重做。

### 18.5 文件留痕

- `tp_smoke_out/tp_smoke_06b_tp1_cgraph.json`：单卡 0.6B baseline / cuda_graph 对照
- `tp_smoke_out/tp_smoke_06b_tp2_cgraph.json`：双卡 0.6B baseline / cuda_graph 对照
- `tp_smoke_out/tp_smoke_06b_tp1_cgraph_big.json`：单卡 0.6B 大 workload 对照（n=16, in=1024, out=256）

## 19. 主线回归 smoke：四个修复 commit 没打坏旧路径（2026-05-30）

§16 / §17 / §18 一波改了 4 个文件 + 4 个 commit。这一节做"全量主线回归"，确认数值修复没把
baseline / quest / cpu_offload / cuda_graph 这些不动的路径冲坏。

### 19.1 跑法

跑 4 跑次的笛卡尔积：{0.6B, 8B} × {TP=1, TP=2}，每跑 7–8 条核心 config：
`baseline / quest / c4_g64 / w4_g32 / w4a8_g32 / w4a8c4_g32 / cpu_offload / cuda_graph_baseline`
（8B 不跑 cpu_offload，单卡放得下，没意义）。

### 19.2 结果摘要

| 跑次 | init+gen ok | 输出连贯 / 复读不崩 | weight_mem 切分 |
|---|---|---|---|
| 0.6B TP=1 | 8/8 | 5 / 3 | 1.140 GB |
| 0.6B TP=2 | 8/8 | 5 / 3 | **0.592 ≈ 1.140/2** ✅ |
| 8B TP=1 | 7/7 | 5 / 2 | 15.288 GB |
| 8B TP=2 | 7/7 | 5 / 2 | **7.660 ≈ 15.288/2** ✅ |

### 19.3 各路径状态（vs 历史）

| 路径 | 状态 | 备注 |
|---|---|---|
| baseline | ✅ 全 ok | 0.6B 256/8 tps、8B 194 tps，与 §13/§14 一致 |
| quest | ✅ 全 ok | TPS 略低于 baseline，符合 quest 选块开销 |
| c4_g64 | 8B ✅ / 0.6B ⚠️ | 8B 上 §17 的 Berlin/Rome/Madrid 复现；0.6B 上塌（与 §13 一致，小模型 + KV4 本来就掉） |
| w4_g32 | ✅ 全 ok | §16 的 Paris/London/阶乘复现 |
| w4a8_g32 / w4a8c4_g32 | ⚠️ 全部"复读不崩" | §16.3 / §16.4 标定的"达 §14.4 最低目标"状态复现 |
| cpu_offload | ✅ 全 ok | 0.6B 上 weight_mem 0.378 GB（offload 后剩驻留权重） |
| cuda_graph_baseline | ✅ 全 ok | TP=1 / TP=2 与 baseline 输出字字对齐，§18 结论复现 |

### 19.4 三个明确结论

- ✅ **四个修复 commit (E / A+B+C / D / F) 全部无回归**：旧路径 TPS / 显存 / 文本质量与历史报告一致
- ✅ **TP weight 切分仍然严格 ≈ 1/tp_size**：8B `15.288 → 7.660`、0.6B `1.140 → 0.592`，§15 修正持续生效
- ⚠️ **量化叠加在 0.6B 上的质量问题继续存在**：c4_g64 / w4a8_g32 / w4a8c4_g32 在 0.6B 上塌或复读，
  这是模型表征能力 vs 量化误差的耦合问题，不在本期修复范围；8B 上 W4 g32 和 c4_g64 都是
  "可用质量"，验证了"中等模型 + 合适参数"的量化路径是可行的

### 19.5 已知不做

- **0.6B 上修 c4_g64 复读**：0.6B 模型本身在 KV4 量化下表征不够，§13.4 早就观察到，
  不是新引入的回归
- **W4A8 真正修好**：仍需 SmoothQuant，§16.5 已标定 → 已在 §20 完成
- **加自动 regression diff 脚本**：smoke 输出已经 JSON 化，未来真要做 CI 再加；当前手动 4 跑次足够

### 19.6 文件留痕

- `tp_smoke_out/tp_smoke_06b_tp1_regression.json`
- `tp_smoke_out/tp_smoke_06b_tp2_regression.json`
- `tp_smoke_out/tp_smoke_8b_tp1_regression.json`
- `tp_smoke_out/tp_smoke_8b_tp2_regression.json`

---

## 20. SmoothQuant 落地修 W4A8 复读（2026-05-31）

### 20.1 问题（§16.5 历史欠账）

§16 已经把 W4A8 从"完全乱码"救到"复读但不崩"，但 8B 上仍能稳定看到 prompt 复读
(`The capital of France is the same as the capital of France is...`)。本质是 per-token
int8 fake-quant + p999 软裁剪只能压住部分 outlier，对 channel-wise outlier 无能为力。

### 20.2 方案：SmoothQuant per-input-channel 等价变换

```
y = x · W^T = (x / s) · (W * s)^T
s[i] = max(|X[:,i]|)^α / max(|W[:,i]|)^(1-α)
```

激活离群值迁到权重侧，激活分布变平滑，per-token A8 量化误差大幅下降。

### 20.3 落地内容

| 文件 | 改动 |
|---|---|
| `tools/calibrate_smoothquant.py` 新建 | TP=1 校准、prompt-bank、per-Linear absmax 聚合、`s` clamp + finite 兜底、bundle 落盘 |
| `tinyvllm/utils/loader.py` | `_apply_smoothquant_scales`：注入 weight × s + register `smooth_scale` buffer，RowParallel 按 rank 切片 |
| `tinyvllm/layers/linear.py` | `_linear_forward` 顶部 `x = x / smooth_scale`（在 A8 假量化之前） |
| `tinyvllm/config.py` | `smoothquant_scale_path` / `smoothquant_alpha` 字段 + 校验 |
| `tinyvllm/engine/model_runner.py` | `load_model` 透传路径 |
| `tools/tp_smoke.py` | `w4a8_sq_g128` / `w4a8_sq_g32` 配置 + `--smoothquant-scale-path` CLI |
| `tools/test_smoothquant_cpu.py` 新建 | 7 项 CPU 单测（数值等价 / NaN / 形状 / TP / buffer 存活 / outlier 抑制证明 / 校准管线） |

### 20.4 α 扫优结果（Qwen3-8B 短 prompt）

| α | 输出质量 |
|---|---|
| 无 SQ | 单字 garbage / 严重复读 |
| 0.3 ~ 0.7 | 短语级复读 |
| **0.85** | **连贯输出，接近 fp baseline** |

最终采用 α = 0.85。

### 20.5 TP=2 多卡验证

`/tmp/sq_scales_qwen3_8b_a0.85.pt`（TP=1 校准产物）→ TP=2 加载时 RowParallel
按 `tp_rank` narrow 切片，kernel 跑通：

| 配置 | TPS | 显存 | 质量 |
|---|---|---|---|
| W4A8+SQ TP=1 | 33.59 | 9.8 GB | 连贯 |
| W4A8+SQ TP=2 | **60.57** (1.8× scaling) | 5.0 GB / rank（精确切半） | 输出与 TP=1 一致 |

### 20.6 单测全绿

`tools/test_smoothquant_cpu.py` 7 项全绿，关键指标：
**SQ 开后 outlier 激活的 A8 量化 MSE 降到原来的 6.5%** —— 证实 SQ 核心动机。

---

## 21. needle 长上下文 SQ 回归 + W4A8 真实瓶颈定位（2026-06-01）

### 21.1 问题

短 prompt α=0.85 看着接近 baseline，但长上下文是 W4A8 真正的考点。
跑 needle 16K 召回对照（4 ctx_len × 5 depth × 2 trials = 30 sample/setting）：

| 配置 | 召回率 | TPS |
|---|---|---|
| FP16 baseline | **100.0%** | 51.41 |
| W4 g32（无 A8） | **100.0%** | 34.19 |
| W4A8 g32 + SQ α=0.85 | 40.0% | 23.53 |
| W4A8 g32 NO-SQ | 6.7% | 24.02 |

### 21.2 关键洞察

- **SQ 把长文召回从 6.7% 救回 40%（6×）**，方向正确
- **W4 单独完全保住 100% 召回**，证明 4-bit 权重不是塌方原因
- 真正的瓶颈是 **A8 在长文 attention 累积下的噪声放大**

### 21.3 W4A8C4+SQ 全栈融合

四组短 prompt 对照：

| 配置 | 输出质量 |
|---|---|
| W4A8 + SQ（KV fp16） | 大体连贯，"capital of France"短语级复读 |
| **W4 + SQ + C4（无 A8）** | **4/4 prompt 全连贯，最稳的 4-bit 全栈折中** |
| W4A8C4 + SQ 全栈 | 部分塌（climate 末尾死循环、capital 退化） |
| W4A8C4 NO-SQ | 单字复读全崩 |

**结论**：A8 × KV-C4 噪声叠加是塌方主因，SQ 单独救不回。
4-bit 全栈生产路径推荐 **W4 g32 + SQ + C4**（不开 A8）。

### 21.4 A8 长文塌方根因诊断（2026-06-01）

写诊断脚本对同一 needle prompt 抓 fp16 / W4 / W4A8+SQ 三种配置的 raw token 输出：

| 样本 | fp16 | W4 only | W4A8+SQ |
|---|---|---|---|
| ctx=4096, d=0.5 | ✓ "The magic number is 71429..." | ✓ 同左 | ✗ "Answer with only magic number..." 复读 |
| ctx=8192, d=0.0 | ✓ | ✓ | ✗ "Answer with only the digits..." 复读 |
| ctx=8192, d=0.5 | ✓ | ✓ | ✓（但末尾开始复读） |
| ctx=15000, d=0.5 | ✓ | ✓ | ✓ |

**根因画像**：
1. W4 没问题，**纯 A8 路径是塌方源**
2. 失败模式不是答错数字，**是复读 prompt 末尾片段（question / instruction）**
3. 这是 **logits 噪声 × softmax 末尾放大** 的典型特征：
   - SQ 把 outlier 移到 weight 后，weight 极值变大
   - W4 g32 还能吃下，但 **A8 per-token 动态量化**被这些极值撑开 → 大部分 act 通道压到 1-2 个 bin
   - 长文 attention 累积后，末尾 query 的 logits 信号被噪声淹没 → 贪心解码陷入"复读问题本身"
4. 短文不复读、长文才复读 — 因为长文 KV 累积更多噪声 token

### 21.5 后续方向（备选，未实施）

| 方向 | 思路 | 工程量 | 预期 |
|---|---|---|---|
| **方向 1** | **首尾若干层（前 2 + 后 2）关 A8，中间层用 A8** | 小 | 40% → 70%+ |
| 方向 2 | SQ α 逐层自适应（per-layer α-grid） | 中 | 40% → 60% |
| 方向 3 | A8 改 per-channel 静态 scale（用 SQ calib act_max 固定，不再 per-token 动态） | 中（要重做校准） | 更稳，需验证 |

倾向方向 1：outlier 在首尾层最严重，跳过这几层成本最小、命中可能最高。

### 21.6 文件留痕

- `/tmp/sq_scales_qwen3_8b_a0.85.pt`（远端）
- `needle_sq_results/needle_fp16.json` / `needle_w4_g32.json` / `needle_w4a8_sq_a085_g32.json` / `needle_w4a8_nosq_g32.json`
- 诊断脚本：`/tmp/needle_diag.py`（远端）

---

## 22. Quest summary triton fuse 性能优化（2026-06-01）

### 22.1 问题

`quest_select_blocks` 的现状是 PyTorch 拆 4 步：

```python
k_min_b = k_min_layer[safe_idx]   # gather → [B, max_blocks, kv_h, d]  HBM 物化
k_max_b = k_max_layer[safe_idx]   # 同上
qm = max(einsum(q, k_min_b), einsum(q, k_max_b))  # 双 einsum，fp16 reduce
criticality = qm.sum(dim=-1).to(fp32)
```

每个 program 4 次 kernel launch + 2 个 4D HBM 中间 tensor，decode 路径每层每步都跑，
瓶颈相当突出。

### 22.2 方案：单个 fused triton kernel

`quest_score_kernel`：grid=(B, max_blocks)，每个 program：
1. 读 1 个 `block_id`（int32）
2. 读 KVHD = `num_kv_heads * head_dim` 个 q_repr（fp16）
3. 读 KVHD 个 k_min + k_max
4. 在 register 里算 `max(q*kmin, q*kmax)`，**fp32 reduce**
5. 写 1 个 fp32 score

省掉两个 [B, N, kv_h, d] 4D 中间 tensor + 3 次 launch。

### 22.3 实现

`tinyvllm/layers/attention.py:224` 加 `quest_score_kernel` + `_next_pow2` helper，
`quest_select_blocks` 内部把"gather + 双 einsum + max + sum"段替换成单次 kernel 调用，
topk / sort / sparse_bt 这一段不动。

### 22.4 结果

**微基准（score-only，独立测）**：

| 配置 (B × max_blocks) | ref | fused | speedup |
|---|---|---|---|
| 4 × 60 | 232 us | 24 us | **9.71×** |
| 16 × 200 | 225 us | 23 us | 9.82× |
| 64 × 200 | 536 us | 24 us | **22.58×** |

**端到端 needle 16K（Qwen3-8B fp16）**：

| 配置 | 召回率 | TPS |
|---|---|---|
| baseline（无 quest） | 100% | 51.83 |
| **quest top-k=8 + fused** | **100%** | 54.19 (+4.5%) |
| **quest top-k=16 + fused** | **100%** | 55.36 (+6.8%) |

### 22.5 关键性质

- **正确性**：topk 选块集合与原版完全一致（fp32 reduce 比原 fp16 reduce 更稳）
- **不破质量**：needle 16K 全 100% 召回
- **加速能在 e2e 看到**：top-k=16 反超 baseline，说明节省的 attention 时间已经超过 score 开销
- **CUDA Graph 兼容**：fused kernel 是普通 elementwise，capture 安全

### 22.6 文件留痕

- `tinyvllm/layers/attention.py`（quest_score_kernel + 改造 quest_select_blocks）
- `needle_quest_results/needle_quest_fused.json`
- 微基准脚本：`/tmp/test_quest_fused.py` / `/tmp/test_quest_fused_v2.py`

---

## 23. 方向1：首尾层关 A8 修复 W4A8+SQ 长文塌方（2026-06-01）

### 23.1 背景

§21 把 W4A8+SQ α=0.85 长文召回 6.7% → 40%，但仍远不及 W4-only 的 100%。
根因定位为 A8 per-token 量化在首尾层 outlier 极端 channel 上撑爆量化范围，
softmax 把噪声放大成"长文末尾指令复读"。

### 23.2 方案

让前 N 层 + 后 N 层保 fp 激活、中间层走 A8。SQ 仍全程开启（不动 weight），
仅在 forward 里跳过 A8 fake-quant。

### 23.3 实现

新增 config 字段 + loader helper：

- `tinyvllm/config.py`: `act_quant_skip_first` / `act_quant_skip_last`，值域校验
- `tinyvllm/utils/loader.py`: `disable_act_quant_in_layers(model, skip_first, skip_last)`
  正则抽 `model.layers.{idx}.` 拿层数，对首/尾 N 层 LinearBase 把 `act_quant_bits` 设回 0
- `tinyvllm/engine/model_runner.py`: 透传两个新参数到 `load_model`
- `tools/eval_needle.py`: CLI `--act-quant-skip-first/last`

注入时机：SQ 之后、`finalize_quantization` 之前（A8 是 forward 时按实例属性读，
位置无强约束，但放这里和 SQ 同窗口最自然）。

### 23.4 验证（Qwen3-8B, W4A8 g32 + SQ α=0.85, skip_first=2 / skip_last=2）

needle 16K eval（ctx ∈ {4096, 8192, 15000}, depth ∈ {0, 0.25, 0.5, 0.75, 1.0}, n=2）：

| 配置 | overall_acc | TPS |
|---|---|---|
| §21 W4A8+SQ α=0.85（无 skip） | ~40% | — |
| **W4A8+SQ α=0.85, skip 2/2（baseline / full attn）** | **93.3%** | 23.87 |
| 同上 + Quest top-k=8 | 76.7% | 25.54 |
| 同上 + Quest top-k=16 | 73.3% | 25.98 |
| 同上 + Quest top-k=4 | 60.0% | 25.50 |

### 23.5 结论

- **目标超额完成**：40% → 93.3%（预期 70%+），方向1是 W4A8+SQ 长文场景
  的稳态修复
- 仅 4 层（共 36 层）退到 fp 激活，A8 收益基本保留
- Quest 叠加 skip2/2 在 top-k=8/16 仍能达 73-77%（之前 W4A8+SQ 完全
  不可用）；说明 outlier 修复后 Quest 又重新可用

### 23.6 文件留痕

- `needle_sq_results/needle_w4a8_sq_a085_g32_skip2.json`
- 配置：`/tmp/sq_scales_qwen3_8b_a0.85.pt` + `--act-quant-skip-first 2 --act-quant-skip-last 2`


---

## 24. Quest update_block_summary triton fuse + 短序列 auto-fallback（2026-06-01）

### 24.1 背景

§22 把 Quest **read 端**（select_blocks 的 score 计算）fuse 进 triton 拿了 8-22× 微基准、
端到端 +6.8% TPS。**write 端**（`update_block_summary` 维护 per-block min/max）
原来还是两次 `index_reduce_(amin/amax)`：每层 prefill+decode 都跑一次，是 §22 的天然对偶。

`docs/kv-sparse-attention.md §5.5 #2` 早就标了"换 triton kernel"，本节落地。

同时落 `§5.5 #4`：top_k * block_size >= max_seq * 0.8 时短序列 selection overhead
反向拖慢 → 直接降级 full attention。

### 24.2 设计

#### 24.2.1 fp16 atomic 不通用 → kv_summary 改 fp32

triton 的 `atomic_min/atomic_max` 在 fp16 上不可移植；同时 §22 的 `quest_score_kernel`
读端已经 `.to(fp32)` 做 reduce，把 buffer 直接改成 fp32 存：内存 ×2 但每 layer 只一份
（`[2, L, num_blocks, kv_h, head_dim]`，对 8B / 32 layer / 80GB KV pool 来说是 <1GB），
正确性更稳。

#### 24.2.2 一个 kernel 替原两次 index_reduce_

```python
@triton.jit
def update_block_summary_kernel(key_ptr, slot_ptr, k_min_ptr, k_max_ptr, ...):
    pid = tl.program_id(0)            # 一个 token 一个 program
    slot = tl.load(slot_ptr + pid)
    block_id = slot // BLOCK_SIZE
    k = tl.load(key_ptr + pid * KVHD + offs).to(fp32)
    base = block_id * KVHD + offs
    tl.atomic_min(k_min_ptr + base, k, mask=mask)
    tl.atomic_max(k_max_ptr + base, k, mask=mask)
```

- 一次 HBM 读 `k`，一次 launch；两次 atomic 同一 buffer 行
- 原 PyTorch 路径：`block_ids` 临时 tensor + 两次 `index_reduce_` （内部各 一次 launch + 全 buffer 扫）
- atomic_min/atomic_max 是确定性归约，结果与 ref index_reduce_ **完全相等**（fp32 reduce）

### 24.3 短序列 auto-fallback

`prepare_decode` host 端早判加一条：

```python
cover = cfg_top_k * self.block_size
short_seq_skip = cover >= max_seq_len_host * 0.8
```

满足时把 `quest_active_top_k` 设回 -1，整个 batch 这一步走 full attention，
跳过 score / topk / gather。`kv-sparse-attention.md §5.5 #4` 落地。

### 24.4 验证

#### 24.4.1 等价性（CUDA + Qwen3 KV 形状）

`/tmp/test_update_summary_fuse.py` 三 case 全 0 误差：

```
[case 1] decode 风格：N=8 nb=128 bs=256 kvh=8 hd=128 → diff_min/max=0.0000
[case 2] prefill 长序列：N=4096 nb=64 → diff=0.0000
[case 3] 累计写两次同 slot：diff=0.0000
ALL PASS
```

含 inf 模式一致性（未命中 block 仍保持 ±inf 初值）。

#### 24.4.2 needle 16K 不破坏质量

Qwen3-8B fp16 needle eval（ctx ∈ {4096, 8192, 15000}, 5 depth, n=2）：

| 配置 | acc | TPS |
|---|---|---|
| baseline | **100%** | 56.49 |
| top-k=8 | **100%** | 58.20 |
| top-k=16 | **100%** | 58.52 |

对比 §22.4（仅 score fuse 没有 summary fuse）：
| 配置 | §22 | §24 | Δ |
|---|---|---|---|
| baseline | 51.83 | 56.49 | +9.0% |
| top-k=8 | 54.19 | 58.20 | +7.4% |
| top-k=16 | 55.36 | 58.52 | +5.7% |

baseline 也涨说明每层 prefill summary 维护开销节省真实存在；top-k 路径同时享受
read+write 双 fuse。

### 24.5 文件留痕

- `tinyvllm/layers/attention.py`：新增 `update_block_summary_kernel`，重写 `update_block_summary`
- `tinyvllm/engine/model_runner.py`：`kv_summary` dtype 改 fp32；`prepare_decode` 加 `short_seq_skip` 早判
- `needle_quest_results/needle_quest_summary_fuse.json`
- 等价性测试：`/tmp/test_update_summary_fuse.py`


---

## 25. W4A8C4 全栈 4-bit + SQ + skip2/2 长文复测（2026-06-02）

### 25.1 动机

§23 把 W4A8+SQ+skip2/2 长文召回救到 93.3%。问题：再叠加 KV4（C4）做成
**全栈 4-8-4 量化**（weight 4bit + act 8bit + KV 4bit），首尾层关 A8 的修复
是否仍然 hold？§21 当时记录 W4A8C4+SQ（无 skip）仍部分塌方，本节验证 skip
之后能否翻盘到 70%+。

### 25.2 配置

在 §23 基础上仅加 `--kv-quant-bits 4 --kv-quant-group-size 32`：

```
int4 g32 + A8 + SQ(α=0.85) + skip_first=2/skip_last=2 + KV4 g32
```

needle 16K（ctx ∈ {4096, 8192, 15000}, 5 depth, n=2，full attention 无 Quest）。

### 25.3 结果

| 配置 | overall_acc | TPS |
|---|---|---|
| W4A8+SQ+skip2/2（§23，不开 C4） | 93.3% | 23.87 |
| **W4A8C4+SQ+skip2/2（全栈 4-8-4）** | **66.7%** | 12.45 |

per-(ctx, depth) 失败点分布（acc<100% 的格子）：

```
ctx=4096  depth=0.50 → 0%    depth=0.75 → 50%
ctx=8192  depth=0.25 → 0%    depth=0.50 → 50%   depth=1.00 → 0%
ctx=15000 depth=0.25 → 50%   depth=1.00 → 50%
```

### 25.4 结论

- **C4 叠加掉 ~27 个点**（93.3% → 66.7%），**未达 70% 稳态线**
- 失败点**散布在中段 depth（0.25/0.5/0.75）和末尾**，不是单一长度塌方
  → 典型 **KV4 量化噪声导致 attention 定位漂移**，与 A8 复读塌方（集中在末尾
  指令复读）是不同失败模式
- TPS 几乎腰斩（23.87 → 12.45）：C4 decode 路径每步把命中块全 dequant 成 fp16，
  长 ctx 下瞬态 buffer 开销大（§17 / `w4a8c4-quantization.md §10` 已知问题，
  需 Quest 叠加 sparse-dequant 才能翻盘）
- **判定：全栈 4-8-4 在长上下文上"能跑但不稳"**。要坐实稳态可用，KV 量化需要
  升级：(a) KV4 改非对称 + per-channel scale，或 (b) KV 只压到 8bit（C8）而非 4bit

### 25.5 下一步备选（未实施）

| 方向 | 思路 | 预期 |
|---|---|---|
| C8 替 C4 | KV cache 量化降到 8bit，噪声远小于 4bit | 召回回到 85%+，显存省一半（仍优于 fp16） |
| KV4 非对称 + per-channel | 当前是对称 group scale，换非对称带 zero-point | 召回 +10~15% |
| C4 + Quest sparse-dequant | 只 dequant top-k 块，省瞬态 buffer + 修 TPS | TPS 翻盘，召回看 top-k |

### 25.6 文件留痕

- `needle_sq_results/needle_w4a8c4_sq_a085_skip2.json`
- 配置：§23 全部参数 + `--kv-quant-bits 4 --kv-quant-group-size 32`


---

## 26. C8 替 C4：全栈量化稳态落地（2026-06-02）

### 26.1 动机

§25 判定 KV4 是全栈 4-8-4 的真瓶颈（93.3% → 66.7%）。按 §25.5 备选里"C8 替 C4"
最划算：KV 只压到 8bit（噪声比 4bit 小一个量级），显存仍省一半。本节实现 C8
路径并复测。

### 26.2 实现

C8 的 store / dequant 比 C4 简单（int8 不需要 nibble pack / 符号扩展）：

- `tinyvllm/layers/attention.py`:
  - `store_kvcache_q8_kernel` / `store_kvcache_q8`：对称 group 量化，
    `scale = max(|x|)/127`，int8 直存（不 pack）
  - `dequant_kv_blocks_q8`：gather 命中块 int8 → 按 group scale 展开 fp
  - forward 三处（write / prefill / decode）把 `kv_quant_bits == 4` 改成
    `in (4, 8)`，内部用 `_dequant = dequant_kv_blocks if bits==4 else dequant_kv_blocks_q8`
    分发；C8 与 C4、Quest 叠加路径全部复用
- `tinyvllm/config.py`: C8 加 `kv_quant_symmetric` 校验
- `tinyvllm/engine/model_runner.py`: §（已有）C8 分配 int8 cache + group scale 已就位

注：内存分配层（`allocate_kv_cache` 的 `kvq_bits==8` 分支）此前已写好但 forward
从未接，本节把 forward 接上才真正可用。

### 26.3 等价性验证

`/tmp/test_c8_roundtrip.py` 三组 group_size round-trip：

```
gs=32  max_abs_err=2.0e-02  mean_abs_err=4.5e-03
gs=64  max_abs_err=2.0e-02  mean_abs_err=5.0e-03
gs=128 max_abs_err=2.0e-02  mean_abs_err=5.5e-03
PASS
```

mean abs err ~0.005，比 C4 小约一个量级（符合 int8 vs int4 量化步长比）。

### 26.4 needle 16K 结果

Qwen3-8B, int4 g32 + A8 + SQ(α=0.85) + skip2/2 + **KV8 g32**（full attention）：

| 配置 | KV | overall_acc | TPS |
|---|---|---|---|
| W4A8+SQ+skip2/2（§23） | fp16 | 93.3% | 23.87 |
| **W4A8C8+SQ+skip2/2（本节）** | **8bit** | **90.0%** | 17.14 |
| W4A8C4+SQ+skip2/2（§25） | 4bit | 66.7% | 12.45 |

C8 仅掉 3.3 个点（vs fp16 KV），失败点只剩 `ctx=8192 depth=0.0/1.0`（边界位置），
中段全 100%。

### 26.5 结论

- **全栈量化稳态可用方案确定：W4 + A8(skip 首尾2层) + SQ(α=0.85) + KV8 g32**
  - 召回 90%（接近 fp16 KV 的 93.3%），KV 显存省一半
- KV 量化档位的明确权衡：
  - **KV4**：显存省 75%，但召回塌到 66.7%（attention 定位漂移）→ 需非对称/per-channel 救
  - **KV8**：显存省 50%，召回 90% → 当前**最优稳态点**
- TPS 17.14 仍低于 fp16 KV 的 23.87：C8 decode 同样每步全块 dequant，瓶颈是瞬态
  buffer + dequant 开销，不是带宽——叠 Quest sparse-dequant 可修（§25.5 同理）

### 26.6 文件留痕

- `tinyvllm/layers/attention.py`：`store_kvcache_q8` / `dequant_kv_blocks_q8` + forward 分发
- `tinyvllm/config.py`：C8 对称校验
- `needle_sq_results/needle_w4a8c8_sq_a085_skip2.json`
- 等价性测试：`/tmp/test_c8_roundtrip.py`


---

## 27. C8 + Quest sparse-dequant：修 TPS 同时保召回（2026-06-02）

### 27.1 动机

§26 拿到 C8 稳态召回 90%，但 TPS 17.14 仍明显低于 fp16 KV 的 23.87 —— 瓶颈是
decode 每步把**全部命中块** dequant 成 fp16 的瞬态 buffer + dequant 开销（不是
KV 带宽）。§26.5 指出叠 Quest sparse-dequant 可修：Quest 只选 top-k 块 →
sparse-dequant 只对这 k 块反量化，瞬态 buffer 从 `B*max_blocks` 缩到 `B*top_k`。

该路径 §26 实现 C8 时已自动复用（decode `quest_active` 分支），本节纯实验验证。

### 27.2 结果

Qwen3-8B, W4A8C8+SQ(α=0.85)+skip2/2, needle 16K：

| 配置 | 召回率 | TPS |
|---|---|---|
| C8 baseline（全块 dequant） | 90.0% | 17.12 |
| C8 + Quest top-k=8 | 73.3% | 25.09 |
| **C8 + Quest top-k=16** | **90.0%** | **23.26 (+36%)** |

### 27.3 结论

- **top-k=16 是甜点**：召回与 baseline 完全持平（90%），TPS +36%，**追平 fp16 KV
  全 attention 的 23.87** —— 即"KV 显存省一半 + 召回 90% + 不掉速"三者兼得
- top-k=8 太激进：sparse-dequant 漏块导致召回掉到 73.3%（与 §23 纯 W4A8 的
  top-k=8=76.7% 一致，说明掉点来自 Quest 选块而非 C8）
- 机理印证：C8 的 TPS 瓶颈确实是 dequant 瞬态 buffer 而非带宽 —— 一旦只 dequant
  16 块，开销立刻回落到全 attention 水平

### 27.4 全栈量化最终推荐配置

**W4(g32) + A8(skip 首尾2层) + SQ(α=0.85) + KV8(g32) + Quest(top-k=16)**

- 召回 90%（长上下文 needle 16K）
- weight 4bit + KV 8bit 显存双省
- TPS 23.26，不输 fp16 KV 全 attention

### 27.5 文件留痕

- `needle_sq_results/needle_w4a8c8_sq_quest.json`
- 路径：§26 的 C8 forward + 既有 Quest sparse-dequant（无新代码）

---

## 28. 研究：A8 skip 层数消融 —— 推翻"首尾对称 skip"的经验配置（2026-06-02）

### 28.1 问题

§23 的 `skip first=2/last=2` 是拍脑袋定的（首尾对称），只验证了它有效（40%→93.3%），
从未回答：(1) outlier 集中在首层还是尾层？(2) skip 几层性价比最高？(3) 2/2 是否
既不饱和又不过度？把"经验配置"做成"有依据的最优配置"。

### 28.2 先验：per-layer 激活 outlier 诊断

新增 `tools/diag_layer_outlier.py`：forward_pre_hook 收集每个 decoder layer 所有
LinearBase 输入激活的 per-channel absmax / p99-median 比 / kurtosis。Qwen3-8B 实测：

```
layer |   amax | p99/med | kurtosis     观察
  0   |   8.5  |  15.5   |   9260       L0 反而很干净
  1-3 |  48-79 | 20-236  |  4-6万       首部中等
  6   | 5952.0 |  14.7   | 24.5万       ★ 极端 outlier 层
 8-11 |  12-17 |  7-8    |  <900        中段最干净
 31   |  116   |         |              尾部开始爬
 ...单调递增...
 35   | 1328.0 |  26.4   |  1.6万       ★ 尾部最强
```

两个先验：**L6 是孤立的 amax 怪物层；尾部 L31-35 amax 单调递增**。

### 28.3 实验矩阵（W4A8 g32 + SQ α=0.85, needle 16K）

扩展 loader 支持 `act_quant_skip_layers`（显式层列表），跑 6 组对照：

| 配置 | 关 A8 的层 | 召回率 | TPS |
|---|---|---|---|
| A 对照 | 无 | 40.0% | 24.55 |
| C 只关 L6 | {6} | **26.7%** ↓ | 24.60 |
| D L6+尾 | {6, 35} | 76.7% | 24.75 |
| E 精准 | {6, 33, 34, 35} | 93.3% | 25.22 |
| **F 尾4层** | {32, 33, 34, 35} | **96.7%** 🏆 | 25.23 |
| G 首4层 | {0, 1, 2, 3} | 70.0% | 25.19 |
| （旧）skip2/2 | {0, 1, 34, 35} | 93.3% | 23.87 |

### 28.4 发现（多处反直觉）

1. **只关 L6 反而更差（40% → 26.7%）**：L6 amax=5952 虽最极端，但单独关它会打破
   SQ 已建立的层间数值平衡，有害。→ **outlier amax 不是决定召回的直接指标**，
   "关掉最毒的层"这一直觉是错的。
2. **尾部 >> 首部**：F（只关尾 4 层）拿到全场最高 96.7%，G（只关首 4 层）只有 70%。
   尾部层更靠近 logits，A8 噪声直接污染贪心解码；首部噪声还会被后续层 re-normalize
   稀释。印证 diag 里 L31-35 单调递增的观测。
3. **首部 skip 是浪费**：旧 skip2/2 把 2 个名额花在首部（L0 还很干净）。同样 4 层
   预算，F（尾4层）召回 93.3% → 96.7%，且配置更规整。

### 28.5 结论 & 配置更新

- **最优策略：只关尾部 N 层，而非首尾对称。** Qwen3-8B 上 **skip last=4（关 L32-35）
  = 96.7%** 是新最优点。
- 旧推荐（§23/§27 的 skip first=2/last=2）**应更新为 skip last=4**。
- 方法论留痕：**单层 outlier 指标（amax/kurtosis）不能直接指导 skip 选择**，要靠
  端到端召回消融；位置（靠近 logits 的尾部）比 outlier 绝对强度更重要。

### 28.6 文件留痕

- `tools/diag_layer_outlier.py`（per-layer outlier 诊断）
- `tinyvllm/utils/loader.py`：`disable_act_quant_in_layers` 加 `skip_layers` 显式列表
- `tinyvllm/config.py` / `model_runner.py` / `tools/eval_needle.py`：透传 `act_quant_skip_layers`
- `needle_sq_results/needle_ablate_*.json`（6 组）
- 诊断输出：`/tmp/outlier.json`

---

## 29. skip-last 拐点扫描 + 全栈最优端到端确认（2026-06-02）

### 29.1 skip-last 档位扫描

补全 §28 只测了 last=4 的空白（纯 W4A8 g32 + SQ α=0.85, needle 16K）：

| skip last | 召回率 | TPS |
|---|---|---|
| 2 | 83.3% | 24.92 |
| 3 | 83.3% | 25.03 |
| **4** | **96.7%** | 25.23 |
| 5 | 96.7% | 25.44 |
| 6 | 86.7% | 25.66 |

**拐点清晰：last=4 是甜点**。last=2/3 不够（83.3%），last=4/5 达峰（96.7%），
last=6 反而回落（86.7%）—— 关太多层后损失的 A8 量化收益开始反噬召回。
**last=4 用最少层数达到峰值召回**。

### 29.2 全栈最优端到端确认

把 skip last=4 整合进全栈 C8+Quest，对比 §27 旧版（skip first=2/last=2）：

配置：**W4 g32 + A8(skip last=4) + SQ(α=0.85) + KV8 g32 + Quest**

| 档 | 旧 skip2/2（§27） | 新 skip last=4 |
|---|---|---|
| C8 baseline（全块 dequant） | 90.0% / 17.14 | **96.7%** / 17.11 |
| C8 + Quest top-k=8 | 73.3% / 25.09 | **90.0%** / 25.09 |
| C8 + Quest top-k=16 | 90.0% / 23.26 | **93.3%** / 23.24 |

**三档全面提升**：
- baseline 90% → **96.7%**
- top-k=8 大涨 73.3% → **90.0%**（尾部 skip 让 Quest 在激进稀疏下也稳）
- top-k=16 90% → **93.3%**，TPS 持平（追平 fp16 KV 全 attention）

### 29.3 最终推荐配置（更新 §27.4）

**W4(g32) + A8(skip last=4) + SQ(α=0.85) + KV8(g32) + Quest(top-k=16)**

- 召回 93.3%（长上下文 needle 16K），不开 Quest 的 baseline 96.7%
- weight 4bit + KV 8bit 显存双省
- TPS 23.24，不输 fp16 KV 全 attention
- 相比 §27 的首尾对称 skip2/2，召回每档 +3~17 点，零额外成本

### 29.4 文件留痕

- `needle_sq_results/needle_sweep_last{2,3,5,6}.json`
- `needle_sq_results/needle_fullstack_skiplast4.json`

## 30. SQ α 逐层自适应（2026-06-02）

### 30.1 问题

当前最优配置仍使用全局 `SQ α=0.85`。但 §28 的诊断已经说明 Qwen3-8B 的 outlier
并不均匀：L6 是极端怪物层，尾部 L31-L35 递增，干净层和 outlier 层共用同一个 α
可能不是最优。全局 α 偏高会把所有层的激活离群值都更激进地迁移到权重侧，可能增加干净层
W4 重建误差；全局 α 偏低又压不住 L6/尾部 outlier。

### 30.2 改动

在 `tools/calibrate_smoothquant.py` 增加 `--alpha-mode layer-adaptive`：

- hook 收集方式不变，仍记录每个 `LinearBase` 的 per-input-channel activation absmax。
- 对每个 decoder layer 取该层所有 LinearBase 的最大 `log1p(act_absmax_max)` 作为 outlier score。
- 将 score min-max 归一化后映射到 `[--alpha-min, --alpha-max]`：

```text
alpha(layer) = alpha_min + (alpha_max - alpha_min) * norm(score)^gamma
```

默认参数：`alpha_min=0.85, alpha_max=0.90, gamma=1.0`。`log1p` 用来压缩 L6 这类极端值，
避免单个怪物层把所有中尾层都挤到 alpha_min。校准产物额外保存
`alpha_mode / alpha_by_layer / alpha_by_module`，loader 无需改动，因为落盘的仍然是最终 per-channel scale。

### 30.3 待跑实验

先生成逐层 α 的 SQ scale：

```bash
python tools/calibrate_smoothquant.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --output /tmp/sq_scales_qwen3_8b_layer_adaptive.pt \
  --alpha-mode layer-adaptive \
  --alpha-min 0.85 --alpha-max 0.90 --alpha-gamma 1.0 \
  --num-prompts 96 --max-model-len 2048 --gpu-memory-utilization 0.7
```

再用当前最佳策略验证 needle 16K：

```bash
python tools/eval_needle.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --out-json needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4.json \
  --context-lens 4096 8192 15000 --depths 0.0 0.25 0.5 0.75 1.0 --num-trials 2 \
  --top-k-blocks-list -1 16 \
  --quantization int4 --quant-group-size 32 --act-quant-bits 8 \
  --smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt \
  --act-quant-skip-last 4 \
  --kv-quant-bits 8 --kv-quant-group-size 32 --gpu-memory-utilization 0.7
```

对照目标是 §29.2 的 **93.3% / 23.24 TPS**。如果逐层 α 能提高 top-k=16 或至少保持召回，
下一步再扫 `alpha_min/max/gamma`；如果下降，则说明当前瓶颈主要由 A8 skip / Quest 选块决定，
SQ α 全局 0.85 已接近最优。

### 30.4 第一轮实验：激进下界失败

先试了较激进的下界 `alpha_min=0.65, alpha_max=0.90, gamma=1.0`：

| setting | 召回率 | TPS |
|---|---:|---:|
| baseline | 63.3% | 17.20 |
| Quest top-k=16 | 60.0% | 23.45 |

结论：**不能把干净层 α 大幅降到 0.65**。虽然这会减少干净层的 W4 重建压力，但 A8 仍然需要
足够强的 SQ 抑制中间层 outlier；过低 α 会让非 skip 层重新暴露 A8 激活量化误差。

### 30.5 第二/三轮实验：保守下界恢复并提升 Quest

进一步做了两档保守下界：

| adaptive α | baseline | Quest top-k=16 | 结论 |
|---|---:|---:|---|
| `[0.82, 0.90]` | 86.7% / 17.18 | 76.7% / 23.41 | 下界仍偏低，召回掉点 |
| **`[0.85, 0.90]`** | **96.7% / 17.17** | **96.7% / 23.39** | 保持 baseline 峰值，同时 top-k=16 +3.4 点 |

`[0.85, 0.90]` 的 per-layer α 分布基本围绕全局 0.85，只对极端层小幅抬高：
L2=0.900、L6=0.886、L16=0.882、L34=0.891、L35=0.898，尾部 L29-L33 约 0.864~0.868。

对比 §29.2 原推荐（全局 α=0.85）：

| setting | 全局 α=0.85 | 逐层 α=[0.85,0.90] |
|---|---:|---:|
| C8 baseline | 96.7% / 17.11 | 96.7% / 17.17 |
| C8 + Quest top-k=16 | 93.3% / 23.24 | **96.7% / 23.39** |

结论：逐层 α 只有在**不低于当前全局最优 α=0.85**时才有价值；它不是“干净层降 α”，而是
“以 0.85 为地板，对极端 outlier 层小幅加压”。当前新推荐：
**W4(g32) + A8(skip last=4) + SQ(layer-adaptive α=0.85~0.90) + KV8(g32) + Quest(top-k=16)**。

文件留痕：
- `/tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4.json`

## 31. Quest top-k 复测与甜点扫描（2026-06-02）

### 31.1 动机

§30 的 n=2 结果显示 layer-adaptive SQ 能让 Quest top-k=16 从 93.3% 提到 96.7%。
但 n=2 总样本只有 30 条，单条波动就是 3.3 个点；因此继续用 n=3（45 条）复测，
并扫描 `top_k ∈ {8,12,16,24}`，看是否存在比 16 更好的速度/质量甜点。

### 31.2 并行实验踩坑：dist 端口硬编码

两组 eval 并行跑时，`ModelRunner` 固定使用 `tcp://localhost:2333` 初始化 NCCL，第二个进程会报：

```text
RuntimeError: EADDRINUSE, message: address already in use
```

已修复为读取 `TINYVLLM_DIST_PORT` / `MASTER_PORT`，默认仍是 2333。这样多组单卡实验可以并行跑：

```bash
CUDA_VISIBLE_DEVICES=1 TINYVLLM_DIST_PORT=2333 python tools/eval_needle.py ...
CUDA_VISIBLE_DEVICES=3 TINYVLLM_DIST_PORT=2345 python tools/eval_needle.py ...
```

### 31.3 n=3 top-k 扫描结果

配置固定为：**W4 g32 + A8(skip last=4) + KV8 g32 + Quest**。

| SQ scale | baseline | top-k=8 | top-k=12 | top-k=16 | top-k=24 |
|---|---:|---:|---:|---:|---:|
| 全局 α=0.85 | **97.8% / 18.40** | 91.1% / **27.18** | 88.9% / 26.16 | 95.6% / 24.96 | 95.6% / 18.66 |
| 逐层 α=[0.85,0.90] | 93.3% / 18.35 | 88.9% / **27.28** | **97.8% / 26.29** | 93.3% / 25.09 | 93.3% / 18.79 |

观察：

- `top-k=8` 最快（~27 tok/s），但召回明显掉到 89~91%，太激进。
- `top-k=24` 没有更稳，TPS 退回到 ~18.7，接近 baseline，性价比差。
- 全局 α 下 `top-k=16` 更稳：95.6% / 24.96。
- 逐层 α 下 `top-k=12` 成为甜点：97.8% / 26.29，质量追平/超过 baseline，同时速度比 top-k=16 更快。

### 31.4 失败样本初步分析

失败 bucket 主要集中在 `ctx=4096` 的边界位置（depth=0 / 0.5 / 1.0），长上下文 15K 反而较稳。
这提示当前 needle eval 的短上下文边界/插入位置对结果影响很大，并不完全是“上下文越长越难”。

注意：当前 `tools/eval_needle.py` 为避免 prefix cache 影响吞吐，每个 top-k setting 使用不同 seed，
所以不同 top-k 的失败样本不是同一组 magic number，不能严格逐条判断“Quest 是否漏掉 baseline 命中的同一条”。
下一轮如果要做根因归因，需要加一个“固定 prompt 集合、只看质量、不看 TPS”的模式。

### 31.5 当前推荐更新

如果追求最好的端到端速度/质量平衡，推荐改为：

**W4(g32) + A8(skip last=4) + SQ(layer-adaptive α=0.85~0.90) + KV8(g32) + Quest(top-k=12)**

理由：n=3 下 `top-k=12` 达到 **97.8% / 26.29 tok/s**，比之前 top-k=16 的 ~23~25 tok/s 更快，
且质量不低于 baseline。由于 n=3 仍有随机波动，下一步需要跑固定 prompt 的质量归因 + n=5 稳定复测。

文件留痕：

- `needle_sq_results/needle_sq_layer_adaptive_floor085_topk_sweep_n3.json`
- `needle_sq_results/needle_sq_global_a085_topk_sweep_n3.json`

## 32. fixed-prompt Quest 归因模式（2026-06-03）

### 32.1 问题

§31 的 top-k 扫描为了避免 prefix cache 让后续 setting 的 TPS 虚高，每个 `top_k` 都会加不同
seed offset。因此不同 top-k 看到的是不同 magic number/prompt 集合，只能比较总体趋势，不能逐条回答：

> top-k=12 命中而 top-k=8 失败，是 Quest 真的漏块，还是刚好抽到了不同样本？

### 32.2 改动

`tools/eval_needle.py` 增加 `--fixed-prompts`：

- 默认关闭，保持原行为：每个 top-k 使用不同 seed，避免 prefix cache 影响吞吐统计。
- 开启后，所有 top-k 共用同一批 `(ctx_len, depth, trial, magic, prompt)`，用于质量归因；每个 setting
  开始前会清空跨 setting prefix-cache 元数据，避免后续 setting 复用前一个 setting 的 KV block。
- 新增 `build_eval_batch(tokenizer, args, top_k)` 统一生成 prompt/metas，测试覆盖两种行为：
  - `--fixed-prompts`：baseline/top-k prompt 完全一致。
  - 默认模式：不同 top-k 使用不同 magic，保持旧吞吐评测语义。
- 新增 `clear_prefix_cache(llm)`：清空 `BlockManager.hash_to_block_id`，并只重置 `ref_count==0` 的空闲
  block 的 `hash/token_ids`，避免误伤正在运行的序列。

使用方式：

```bash
python tools/eval_needle.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --out-json needle_sq_results/needle_sq_layer_adaptive_fixed_prompts_topk.json \
  --context-lens 4096 8192 15000 --depths 0.0 0.25 0.5 0.75 1.0 --num-trials 3 \
  --top-k-blocks-list -1 8 12 16 24 --fixed-prompts \
  --quantization int4 --quant-group-size 32 --act-quant-bits 8 \
  --smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt \
  --act-quant-skip-last 4 --kv-quant-bits 8 --kv-quant-group-size 32 --gpu-memory-utilization 0.7
```

### 32.3 prefix cache 污染确认

未清 cache 的 fixed-prompt 首轮结果（baseline 先跑）：

| setting | acc | tok/s | 失败样本 |
|---|---:|---:|---|
| baseline | 93.3% | 18.26 | `(4096,0.0,t2)`, `(4096,0.5,t0)`, `(4096,0.5,t1)` |
| top-k=8 | 100.0% | 95.19 | 无 |
| top-k=12 | 100.0% | 84.45 | 无 |
| top-k=16 | 97.8% | 73.47 | `(4096,0.0,t2)` |
| top-k=24 | 97.8% | 37.08 | `(4096,0.0,t2)` |

这里后续 setting 的 TPS 高到 70~95 tok/s，明显不是正常 prefill+decode 口径，而是命中了前一个 setting
留下的 prefix cache。因此这组只能说明“同 prompt 归因功能可用”，不能直接用来比较 top-k 质量/速度。

反序验证（`--top-k-blocks-list 12 -1 --fixed-prompts`，top-k=12 先跑）：

| setting | acc | tok/s | 失败样本 |
|---|---:|---:|---|
| top-k=12 | 95.6% | 25.14 | `(4096,0.5,t0)`, `(4096,0.5,t1)` |
| baseline | 97.8% | 37.09 | `(4096,0.0,t2)` |

结论：同进程 fixed-prompt 如果不清 prefix cache，结果会依赖 setting 顺序；尤其吞吐必然虚高，质量归因也会变得难解释。
根因在 `BlockManager.deallocate()` 释放 block 时保留 `hash_to_block_id` / `token_ids` 以支持 prefix reuse，
而 `prepare_prefill()` 会从 `seq.num_cached_tokens` 后开始送 token。正常吞吐评测用不同 seed 绕开这个问题；
fixed-prompt 归因必须显式清 cache 或拆成独立进程。

### 32.4 验证状态

本地和远端均运行 `tools/test_eval_needle_fixed_prompts.py`，已验证：

```text
eval_needle fixed-prompt tests passed
```

覆盖内容：

- `--fixed-prompts` 下 baseline/top-k 生成完全相同 prompt/magic。
- 默认模式仍为不同 top-k 使用不同 seed offset。
- `clear_prefix_cache(llm)` 会清空 cache 索引，并只重置空闲 block 元数据。

文件留痕：

- `needle_sq_results/needle_sq_layer_adaptive_fixed_prompts_topk.json`（未清 cache，baseline 先跑）
- `needle_sq_results/needle_sq_layer_adaptive_fixed_prompts_topk12_first.json`（未清 cache，top-k=12 先跑）
- `tools/test_eval_needle_fixed_prompts.py`

## 33. cache-cleared fixed-prompt Quest 归因复测（2026-06-03）

### 33.1 环境修复

远端实际可用的评测环境是：

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
torch=2.4.1+cu121, transformers=5.8.1, flash_attn=2.6.3
```

误用 `/data00/home/sitian/miniconda3/envs/py311/bin/python` 时会先缺 `Qwen3Config`，升级 transformers 后又缺
`flash_attn`；而该 env 的 PyTorch 是 cu126，本机 nvcc 是 11.7，源码编译 flash-attn 会 CUDA mismatch。
后续长评测统一使用 `tllm/env/bin/python`。

### 33.2 n=3 cache-cleared fixed-prompt：baseline vs top-k=12

命令要点：`--top-k-blocks-list -1 12 --fixed-prompts --num-trials 3`，且每个 setting 前清 prefix cache。

| setting | acc | tok/s | 失败样本 |
|---|---:|---:|---|
| baseline | 93.3% | 18.24 | `(4096,0.0,t2)`, `(4096,0.5,t0)`, `(4096,0.5,t1)` |
| top-k=12 | 95.6% | 24.81 | `(4096,0.5,t0)`, `(4096,0.5,t1)` |

逐样本 diff：`top-k=12` 没有新增失败，反而修复了 baseline 在 `(4096,0.0,t2, magic=15306)` 的失败。
这说明 §31 中 top-k=12 的甜点不是 prefix-cache 假象；清 cache 后速度仍有 **+36.0%**（24.81 vs 18.24 tok/s）。

### 33.3 n=5 稳定复测

继续把 trial 从 3 提到 5（总样本 75 条）：

| setting | acc | tok/s | 失败数 |
|---|---:|---:|---:|
| baseline | 94.7% | 19.19 | 4/75 |
| top-k=12 | 96.0% | 26.80 | 3/75 |

失败样本：

| 样本 | baseline | top-k=12 | 现象 |
|---|---:|---:|---|
| `(4096,0.0,t2,15306)` | fail | hit | top-k=12 修复 |
| `(4096,0.5,t2,76150)` | fail | fail | 共同失败 |
| `(4096,0.5,t3,28254)` | fail | fail | 共同失败 |
| `(15000,0.0,t0,74694)` | fail | hit | top-k=12 修复 |
| `(15000,0.25,t1,84384)` | hit | fail | top-k=12 新增失败，输出截断成 `84` |

结论：

- `top-k=12` 在固定 prompt、清 cache 的公平口径下仍然比 baseline 快，且 n=5 质量没有下降：
  **96.0% / 26.80 tok/s** vs baseline **94.7% / 19.19 tok/s**。
- Quest 的主要风险不是“系统性漏 needle block”，因为 75 条里只新增 1 条 Quest 独有失败；更多失败来自
  `ctx=4096 depth=0.5` 这类模型/提示边界本身不稳的样本。
- 当前推荐保持不变：
  **W4(g32) + A8(skip last=4) + SQ(layer-adaptive α=0.85~0.90) + KV8(g32) + Quest(top-k=12)**。

### 33.4 top-k=8/16 同 prompt 补跑

为了判断 top-k=12 的优势是不是偶然，又用同一批 n=5 prompt 补跑 `top_k ∈ {8,16}`：

| setting | acc | tok/s | 失败数 | 备注 |
|---|---:|---:|---:|---|
| baseline | 94.7% | 19.19 | 4/75 | 对照，来自 §33.3 |
| top-k=8 | 94.7% | **28.11** | 4/75 | 最快，但新增 `(8192,0.75,t4)` 误答 `222225` |
| **top-k=12** | **96.0%** | 26.80 | **3/75** | 质量/速度综合最优 |
| top-k=16 | 94.7% | 25.59 | 4/75 | 更慢，且没有修复 baseline 的 2 个边界失败 |

关键 hit matrix：

| 样本 | baseline | top-k=8 | top-k=12 | top-k=16 |
|---|---:|---:|---:|---:|
| `(4096,0.0,t2,15306)` | fail | hit | hit | fail |
| `(4096,0.5,t2,76150)` | fail | fail | fail | fail |
| `(4096,0.5,t3,28254)` | fail | fail | fail | fail |
| `(8192,0.75,t4,22225)` | hit | fail | hit | hit |
| `(15000,0.0,t0,74694)` | fail | fail | hit | fail |
| `(15000,0.25,t1,84384)` | hit | hit | fail | hit |

解释：

- `top-k=8` 的速度最高，但开始出现 Quest 独有错误；n=5 下它没有比 baseline 提质，只是提速。
- `top-k=16` 不是“更大更稳”：它的失败集合几乎回到 baseline，且速度低于 top-k=12。
- `top-k=12` 恰好避开了 top-k=8 的过稀疏漏召回，也不像 top-k=16 那样回落到 baseline 失败模式；
  这支持 §31 的“top-k=12 是甜点”结论。

文件留痕：

- `needle_sq_results/needle_sq_layer_adaptive_fixed_prompts_cacheclear_topk12.json`
- `needle_sq_results/needle_sq_layer_adaptive_fixed_prompts_cacheclear_topk12_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_fixed_prompts_cacheclear_topk8_16_n5.json`

## 34. Needle prompt 边界归因：分隔符不是免费午餐（2026-06-03）

### 34.1 问题

§33 的固定 prompt hit matrix 显示共同失败主要集中在 `ctx=4096 depth=0.5`。进一步检查 prompt token 边界发现，
当前 `build_prompt()` 是直接把 needle token 插入 haystack token 中间，可能产生单词粘连：

```text
... The sun is yellow. HereThe magic number is 76150. Remember it.  we go. ...
```

这说明 `ctx=4096 depth=0.5` 的共同失败不一定是 Quest 漏块，也可能是 needle eval 构造本身在 token 边界上不自然。

### 34.2 最小 probe：只看 `ctx=4096 depth=0.5`

对同一组 n=5 magic，比较三种 needle 插入格式：

| style | 模板 | baseline | top-k=12 |
|---|---|---:|---:|
| original | `The magic number is X. Remember it. ` | 3/5 | 3/5 |
| space | ` The magic number is X. Remember it. ` | 3/5 | 3/5 |
| newline | `\n\nThe magic number is X. Remember it.\n\n` | **5/5** | **5/5** |

结论：对 `4096/0.5` 这个失败桶，空行分隔确实能消除 `HereThe` 这类粘连，并把共同失败样本全部修复。

### 34.3 全量 n=5 反证：全局 newline 会伤害长文召回

但把 `--needle-style newline` 扩展到完整 n=5 eval 后，结果明显变差：

| needle style | setting | acc | tok/s |
|---|---|---:|---:|
| original | baseline | 94.7% | 19.19 |
| original | top-k=12 | **96.0%** | 26.80 |
| newline | baseline | 74.7% | 19.21 |
| newline | top-k=12 | 74.7% | 26.79 |

newline 失败集中在 `depth=0.0/0.25/0.5`，输出常见模式是复读 filler / question，而不是数字：

```text
Answer with only the digits, nothing else. Answer with only the digits, nothing else. ...
There and back again. The grass is green. ...
```

解释：空行 delimiter 局部解决了中间插入的单词粘连，但全局改变了 prompt 分布；在长重复 haystack 中，
多余换行反而更容易触发模型复读或忽略 needle。因此不能直接把历史 needle eval 默认改成 newline。

### 34.4 代码改动与当前建议

`tools/eval_needle.py` 增加 `--needle-style {original,newline}`，默认仍是 `original`，用于后续做 prompt 构造消融，
不影响历史结果可比性。测试覆盖 newline 模式确实会用空行包住 needle。

当前建议：

- 主评测继续使用默认 `original`，保持与 §29~§33 可比。
- 如果要研究“模型真实召回能力”而不是“当前 needle benchmark 口径”，再单独跑 `--needle-style newline` 或更合理的
  delimiter 设计；但不能和历史 original 结果混在一张表里。
- `ctx=4096 depth=0.5` 的共同失败应归因于 **prompt 构造边界 + 模型脆弱性**，不是 Quest 的系统性漏块。

文件留痕：

- `needle_sq_results/needle_boundary_variant_probe.json`
- `needle_sq_results/needle_sq_layer_adaptive_newline_fixed_prompts_topk12_n5.json`

## 35. Speculative Decoding v0：n-gram draft 离线上限估计（2026-06-03）

### 35.1 问题

前面几轮主线都在优化单步 decode 的每 token 成本（W4/A8/SQ/KV8/Quest）。另一个推理引擎方向是减少目标模型
decode 次数：先用便宜 draft 产生多个候选 token，再用目标模型一次性验证。完整 speculative decoding 要改 KV cache、
scheduler 和 verification path，风险较高；因此先做一个 **n-gram draft 离线 profiler**，回答一个更小的问题：

> 当前任务/输出分布里，基于 prompt/history 的 n-gram draft 到底有多少可接受 token？

如果离线 replay 的接受率很低，说明 n-gram draft 不值得进入在线路径；如果接受率较高，再继续做真正的 batch verification。

### 35.2 v0 实现

新增 `tinyvllm/speculative/ngram.py`，只做纯 token 序列逻辑：

- `propose_ngram_draft(history, ngram_size, max_draft_tokens)`：拿当前 history 末尾 n-gram，在历史中找最近一次相同 n-gram，
  把该位置之后的 token 作为 draft。
- `replay_ngram_acceptance(tokens, prompt_len, ngram_size, max_draft_tokens)`：对已经生成好的 token 流做离线 replay，
  统计 draft token 中有多少和真实后续 token 前缀一致。
- `summarize_replay_stats()`：输出 JSON-friendly 指标。

新增 `tools/profile_ngram_spec.py`，先正常调用 `LLM.generate()`，再对 prompt+output token 做 replay，避免在第一版就触碰
online KV cache / scheduler。

### 35.3 Qwen3-8B smoke 结果

配置沿用当前推荐量化栈：

```text
Qwen3-8B + W4(g32) + A8(skip last=4) + SQ(layer-adaptive floor=0.85) + KV8(g32)
ngram_size=3, max_draft_tokens=4, max_output_len=64, temperature=0.0
```

远端 A100 smoke（3 条内置 prompt，共 192 个 decode position）：

| metric | value |
|---|---:|
| positions | 192 |
| draft_events | 127 |
| drafted_tokens | 508 |
| accepted_tokens | 384 |
| acceptance_rate | **75.6%** |
| avg_draft_len | 4.0 |

分 prompt 看差异很大：

| prompt | output 现象 | acceptance_rate |
|---|---|---:|
| 简单事实问答 | 量化栈仍出现 `Wait, no.` 复读 | 97.4% |
| 代码补全 | 正常结构化输出 | 57.1% |
| 重复 haystack 后续 | 继续复读 Q/A 模板 | 57.3% |

### 35.4 结论

- n-gram draft 在“复读/模板化输出”上非常容易命中，所以总接受率 **75.6%** 不能直接等价为线上收益；它部分反映了当前
  W4A8+SQ 栈仍会在某些短 prompt 上复读。
- 更有参考价值的是非纯复读样本仍有约 **57%** 接受率，说明 prompt/history n-gram 作为 cheap draft 有一定潜力。
- v0 暂不接 online path。下一步若继续做，需要实现“多 token target verification + KV 回滚/提交”或先做更轻的
  batch verification 原型；否则离线 profiler 已足够作为研究工具。

文件留痕：

- `tinyvllm/speculative/ngram.py`
- `tinyvllm/speculative/__init__.py`
- `tools/profile_ngram_spec.py`
- `tools/test_ngram_speculative.py`

## 36. Chunked Prefill v0：先做安全拆块，不做 mixed batch（2026-06-03）

### 36.1 问题

当前 scheduler 是 **prefill 优先且一次性 prefill 完整 prompt**：只要 `waiting` 队列里有新请求，`schedule()` 就先把它完整
prefill 完，再进入 decode。这对长 prompt 很不友好：一个 8k/15k prompt 会把 decode 暂停很久。

完整的 serving 方案应该把 decode token 和 prefill chunk 混在同一个 batch 里，但这会同时牵动 attention context、KV cache、
CUDA graph 和采样路径。v0 先做保守版本：**仍然每步只跑 pure prefill 或 pure decode，但把 prefill 拆成小 chunk**，把单次
prefill 对调度器的阻塞时间上限压下来。

### 36.2 实现边界

新增两个配置：

```python
max_num_prefill_tokens_per_step: int = 0   # 0 = 关闭，保持旧行为
chunked_prefill_decode_first: bool = True  # 有 running decode 时优先 decode
```

核心状态拆分：

- `waiting`：还没分配 KV block。
- `prefilling`：已分配 KV block，但 prompt KV 还没全部算完。
- `running`：prompt prefill 已完成，可进入 decode。

每条 `Sequence` 新增：

- `num_computed_tokens`：已经真实写入 KV cache 的 prompt token 数。
- `prefill_chunk_start/end/final`：当前 prefill chunk 的边界，以及是否需要采样首个输出 token。

调度语义：

- 中间 chunk：`is_prefill=True, do_sample=False`，只写 KV，不采样、不 append token。
- final chunk：`is_prefill=True, do_sample=True`，取 chunk 最后一个位置 logits，采样首个输出 token，随后进入 `running`。
- decode：保持旧路径。

### 36.3 最关键的安全修复：delayed prefix-cache commit

不能在分配 KV block 时立刻把未来完整 block 发布到 `hash_to_block_id`。否则后续相同 prefix 的请求可能复用到“CPU 元数据已存在、
GPU KV 还没写入”的 block。

因此 v0 把 `BlockManager.allocate()` 加了 `publish_hashes` 参数：

- 默认 `publish_hashes=True`，旧非 chunked 路径不变。
- chunked prefill 用 `publish_hashes=False`，只分配 block，不发布 hash。
- prefix-cache 命中的已计算 block 即使在 `publish_hashes=False` 下也必须恢复 `hash/token_ids`，否则后续块的 hash chain 会断。
- 每个 chunk postprocess 后调用 `commit_prefill(seq, old_end, new_end)`，只发布已经计算完成的完整 block。

测试覆盖：第一块 chunk 完成后只发布第一块 hash，第二块 hash 必须等第二个 chunk 完成后才出现在 `hash_to_block_id`。

### 36.4 smoke 结果

本地非 GPU 测试：

```text
chunked prefill tests passed
ngram speculative tests passed
eval_needle fixed-prompt tests passed
```

远端 A100 + Qwen3-0.6B 贪心 smoke：同一长 prompt，默认 prefill vs `max_num_prefill_tokens_per_step=16` 输出 token 完全一致。

```text
match: True
chunk_text: " The answer is blue. So, the answer is blue.\nThe answer is blue"
```

### 36.5 结论与下一步

- v0 已经验证 chunked prefill 的基本正确性：中间 chunk 不会误 append；final chunk 能采样；未计算 block 不会污染 prefix cache；
  小模型 smoke 与默认 prefill 贪心输出一致。
- 当前还不是完整 serving 级 chunked prefill：没有 mixed prefill+decode batch，`decode_first=True` 可能让新长 prompt 在已有长 decode
  期间等待更久。它解决的是“单次 prefill step 太长”的安全拆块问题。
- 下一步如果继续推进，应做 latency profiler：构造“一个长 prompt 进入时已有多条 running decode”的场景，对比 default、
  chunked prefill、未来 mixed batch 的 decode step 间隔。

文件留痕：

- `tinyvllm/config.py`
- `tinyvllm/engine/sequence.py`
- `tinyvllm/engine/block_manager.py`
- `tinyvllm/engine/scheduler.py`
- `tinyvllm/engine/model_runner.py`
- `tinyvllm/engine/llm_engine.py`
- `tools/test_chunked_prefill.py`
- `docs/superpowers/plans/2026-06-03-chunked-prefill-v0.md`

## 37. Chunked Prefill latency profiler：v0 拆块的收益与副作用（2026-06-03）

### 37.1 问题

§36 只验证了 chunked prefill 的正确性，还没回答性能问题：当已有 decode 请求正在跑时，新进来一个长 prompt，
default prefill、chunked prefill-first、chunked decode-first 分别会怎样影响 decode 间隔和首 token 延迟？

新增 `tools/profile_chunked_prefill.py`，不用改 engine，只手动驱动 `LLM.add_request()` / `LLM.step()` 并记录每一步：

- `kind`: prefill / decode
- `tokens`: 本 step 实际处理 token 数
- `dt_ms`: step latency（CUDA sync 后计时）
- `outputs`: 本 step 完成的请求数

统计输出包括 prefill/decode p50/p95/max、`first_output_ms`、以及相邻 decode step 之间的最大间隔。

### 37.2 测试 workload

远端 A100 + Qwen3-0.6B：

```text
num_decode_seqs=2
decode_prompt_tokens=64
long_prompt_tokens=512
max_output_len=8
inject_long_after_decode_steps=2
max_model_len=1024
max_num_batched_tokens=1024
enforce_eager=True
```

解释：先让 2 条短请求进入 decode，跑 2 个 decode step 后插入一条 512-token 长 prompt，观察长 prefill 对 decode 的阻塞。
profiler 内部先跑 batch=2 和 batch=1 warmup，避免 torch.compile/cold-shape 把单步延迟污染到几十秒。

### 37.3 结果

| mode | prefill steps | prefill p95 | decode p50 | max decode gap | first_output_ms | total_ms |
|---|---:|---:|---:|---:|---:|---:|
| default | 2 | 37.08 ms | 32.81 ms | **68.49 ms** | 303.35 ms | **367.66 ms** |
| chunked prefill-first (`chunk=128`) | 6 | 38.20 ms | 33.73 ms | 170.72 ms | 445.64 ms | 510.67 ms |
| chunked decode-first (`chunk=128`) | 6 | 38.06 ms | 30.58 ms | 180.16 ms | **256.59 ms** | 867.33 ms |

几个现象：

- 对 0.6B + 512-token prompt，这里的 default 一次性 prefill 本身只有 ~35 ms；把它拆成 4 个 128-token chunk 后，单个 chunk
  并没有明显更快，反而多了 step overhead。
- prefill-first chunked 会连续跑 4 个长 prompt chunk，再恢复 decode，所以最大 decode gap 从 68 ms 增到 171 ms；这是 v0
  “只拆块但不 mixed batch”的直接副作用。
- decode-first 会优先把已有 running decode 请求跑完，第一批短请求更快完成（first output 256 ms），但长 prompt 被推迟，
  总 wall time 变长到 867 ms；这说明 decode-first 是偏向在线 decode latency 的策略，不是整体吞吐最优。

### 37.4 结论

- 在当前 0.6B 小模型 / 512-token prompt 上，**chunked prefill v0 不是吞吐优化**；它更像是一个安全机制和后续 mixed batch 的地基。
- 只做 pure prefill/pure decode 的 v0 无法真正解决“长 prefill 与 decode 公平混排”：prefill-first 会连续 chunk 阻塞 decode，
  decode-first 会让新长 prompt 等已有 decode 完成。
- 下一步真正值得做的是 **mixed prefill+decode batch** 或更轻量的 scheduler policy：每跑 N 个 prefill chunk 插入一次 decode，
  而不是让 prefill chunk 或 decode 任一方独占调度器。

文件留痕：

- `tools/profile_chunked_prefill.py`
- `tools/test_profile_chunked_prefill.py`
- `docs/superpowers/plans/2026-06-03-chunked-prefill-latency-profiler.md`
- 远端结果：`/tmp/chunked_prefill_latency_default.json`
- 远端结果：`/tmp/chunked_prefill_latency_chunked.json`
- 远端结果：`/tmp/chunked_prefill_latency_decode_first.json`

## 38. Chunked Prefill 公平调度：每 N 个 chunk 让出一次 decode（2026-06-03）

### 38.1 问题

§37 证明了 v0 的两个极端策略都有明显缺点：

- `prefill-first`：新长 prompt 会连续跑完所有 chunk，已有 decode 被连续阻塞。
- `decode-first`：已有 decode 很快完成，但新长 prompt 被推迟，总 wall time 变长。

在不做 mixed prefill+decode kernel 的前提下，一个更小的折中是：**允许 prefill 连续跑 N 个 chunk，但达到阈值后，
如果已有 running decode，就强制让出 1 个 decode batch**。

### 38.2 实现

新增配置：

```python
chunked_prefill_max_consecutive_chunks: int = 0
```

语义：

- `0`：关闭，保持 §36 的行为。
- `>0`：chunked prefill 开启且 `chunked_prefill_decode_first=False` 时生效。
- Scheduler 维护 `_consecutive_prefill_chunks`：
  - 每调度一个 prefill chunk 就 `+1`。
  - 每调度 decode 就清零。
  - 如果计数达到阈值且 `running` 非空，则本 step 先调度 decode。

测试覆盖：`chunked_prefill_max_consecutive_chunks=2` 时，两个 prefill chunk 后必须让出一个 decode batch，然后再恢复 prefill。

profiler 增加：

```text
--max-consecutive-prefill-chunks N
```

### 38.3 远端 smoke：balanced policy

沿用 §37 的 Qwen3-0.6B workload，新增一条：

```text
mode=chunked
max_num_prefill_tokens_per_step=128
chunked_prefill_max_consecutive_chunks=1
enforce_eager=True
```

| mode | prefill steps | decode steps | max decode gap | first_output_ms | total_ms |
|---|---:|---:|---:|---:|---:|
| default | 2 | 9 | **68.49 ms** | 303.35 ms | **367.66 ms** |
| chunked prefill-first | 6 | 9 | 170.72 ms | 445.64 ms | 510.67 ms |
| chunked decode-first | 6 | 21 | 180.16 ms | **256.59 ms** | 867.33 ms |
| **balanced N=1** | 6 | 12 | 74.50 ms | 443.82 ms | 597.66 ms |

balanced 的 step 序列符合预期：prefill / decode 交替穿插，插入长 prompt 后不再连续 4 个 prefill chunk 独占调度器。

### 38.4 结论

- `N=1` 明显修复了 prefill-first 的 decode gap：从 **170.72 ms 降到 74.50 ms**，接近 default 的 68.49 ms。
- 代价是 total wall time 比 default 高，也比 prefill-first 高；这是 pure batch scheduler 下“公平性换吞吐”的预期结果。
- 这条策略比 decode-first 更均衡：不会让长 prompt 一直等到短 decode 全结束，也不会让 prefill chunk 连续霸占调度器。
- 当前建议：保留默认 `chunked_prefill_max_consecutive_chunks=0`，实验/serving 场景可试 `N=1~2`；真正高性能仍需要 mixed batch。

文件留痕：

- `tinyvllm/config.py`
- `tinyvllm/engine/scheduler.py`
- `tools/test_chunked_prefill.py`
- `tools/profile_chunked_prefill.py`
- `docs/superpowers/plans/2026-06-03-chunked-prefill-fair-scheduler.md`
- 远端结果：`/tmp/chunked_prefill_latency_balanced.json`

## 39. Mixed Prefill+Decode v0：复用 varlen prefill 路径的保守 mixed batch（2026-06-03）

### 39.1 问题

§38 的 fair scheduler 仍是 pure batch：一个 step 不是 prefill chunk，就是 decode batch。它能把最大 decode gap 从
170.72 ms 拉回到 74.50 ms，但代价是 total wall time 变长，因为 prefill 和 decode 仍然互相让路，不能同一步推进。

下一步尝试一个保守 mixed v0：**不写新 kernel，只把 decode seq 包装成 query length = 1 的 varlen prefill row，
和一个 prefill chunk 放进同一次 prefill forward**。

### 39.2 实现

新增配置：

```python
chunked_prefill_mixed_batch: bool = False
```

启用条件：

- `max_num_prefill_tokens_per_step > 0`
- `chunked_prefill_decode_first=False`
- `chunked_prefill_mixed_batch=True`
- 当前存在 running decode，且可以调度到一个 prefill chunk

核心路径：

- Scheduler 为 mixed batch 中每条 seq 打临时标记：
  - `step_is_decode=True`：这条 seq 本 step 是 decode token。
  - `step_is_decode=False`：这条 seq 本 step 是 prefill chunk。
  - `step_do_sample=False`：中间 prefill chunk 的 logits 会被丢弃，不 append token。
- `ModelRunner.prepare_mixed()` 复用 prefill varlen attention：
  - prefill row 使用 `[prefill_chunk_start, prefill_chunk_end)`。
  - decode row 使用 `input_id=seq.last_token`、`position=len(seq)`、`seqlen_q=1`、`seqlen_k=len(seq)`。
  - 只要存在 decode row，就会带 `block_tables`，让 prefill attention 从 KV cache 读完整上下文。
- `Scheduler._postprocess_mixed()` 按 seq role 分流：
  - 中间 prefill chunk：commit 已计算完整 block，丢弃 sampled token，回 `prefilling`。
  - final prefill chunk：commit 后 append sampled token，进入 `running` 或结束。
  - decode seq：append sampled token，进入 `running` 或结束。

### 39.3 本地验证

```text
python3 tools/test_chunked_prefill.py
chunked prefill tests passed

python3 tools/test_profile_chunked_prefill.py
chunked prefill profiler tests passed

python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/scheduler.py tinyvllm/engine/sequence.py tinyvllm/config.py
python3 -m py_compile tools/profile_chunked_prefill.py
```

测试新增覆盖：

- mixed scheduler 必须同时返回 prefill chunk 和 running decode，batch kind 为 `mixed`。
- mixed intermediate prefill chunk 只 commit KV，不 append token；decode seq 同步 append token。
- mixed final prefill chunk 与多条 decode row 按 `seqs` 顺序消费 sampled token。
- mixed fallback 成普通 prefill 时仍计入 `chunked_prefill_max_consecutive_chunks`，避免让路策略失效。
- TP worker pickle/unpickle 后保留 `step_is_decode` / `step_do_sample`。
- profiler summary 把 `mixed` 计入 decode progress，用于更合理地计算 decode gap。

### 39.4 远端 smoke：mixed v0

远端 A100 + Qwen3-0.6B，同一 profiler workload，在 GPU 3 上重跑 default / chunked / decode-first / balanced / mixed。

mixed 命令：

```text
mode=mixed
max_num_prefill_tokens_per_step=128
enforce_eager=True
```

| mode | prefill steps | mixed steps | decode steps | max decode gap | first_output_ms | total_ms |
|---|---:|---:|---:|---:|---:|---:|
| default | 2 | 0 | 33 | 65.75 ms | 1027.30 ms | **1089.63 ms** |
| chunked prefill-first | 12 | 0 | 33 | 288.12 ms | 1378.22 ms | 1443.17 ms |
| chunked decode-first | 12 | 0 | 155 | 290.08 ms | **948.94 ms** | 4896.86 ms |
| balanced N=1 | 12 | 0 | 42 | 74.62 ms | 1387.08 ms | 1716.79 ms |
| **mixed v0** | 1 | 11 | 31 | **39.00 ms** | 1010.05 ms | 1336.65 ms |

解释：

- mixed v0 的 decode gap 最好：39.00 ms，说明 mixed step 确实让 decode progress 不再被 prefill chunk 阻断。
- total wall time 比 prefill-first 和 balanced 好，但仍慢于 default。原因是 v0 每个 mixed step 仍走 prefill varlen path，并且中间 prefill chunk
  仍会产出/采样一个最终丢弃的 logits row。
- first_output 接近 default，但略慢；decode-first 仍最早，但代价是 total wall time 极差。
- 注意 profiler 里初始 4 条 decode prompt 也会被 chunked/mixed admission 影响：mixed v0 会把后续短 prompt prefill 与已有 decode 混起来，
  因此它不是单纯“长 prompt 插入后才 mixed”的 serving 策略。

### 39.5 当前边界

- v0 已通过远端 GPU smoke，但还不是最终 serving 策略。
- mixed row 走 prefill varlen path，暂不叠加 Quest/C4 评估。
- 中间 prefill chunk 仍会产生一个 logits row 并采样一次；postprocess 会丢弃它。后续若要极致优化，可改 LMHead/采样器只返回需要采样的 row。

文件留痕：

- `tinyvllm/config.py`
- `tinyvllm/engine/sequence.py`
- `tinyvllm/engine/scheduler.py`
- `tinyvllm/engine/model_runner.py`
- `tinyvllm/engine/llm_engine.py`
- `tools/test_chunked_prefill.py`
- `tools/profile_chunked_prefill.py`
- `tools/test_profile_chunked_prefill.py`
- `docs/superpowers/plans/2026-06-03-mixed-prefill-decode-v0.md`
- 远端结果：`/tmp/chunked_prefill_latency_default_rerun.json`
- 远端结果：`/tmp/chunked_prefill_latency_chunked_rerun.json`
- 远端结果：`/tmp/chunked_prefill_latency_decode_first_rerun.json`
- 远端结果：`/tmp/chunked_prefill_latency_balanced_rerun.json`
- 远端结果：`/tmp/chunked_prefill_latency_mixed.json`

## 40. Mixed Prefill+Decode sample mask：跳过中间 prefill chunk 采样（2026-06-04）

### 40.1 问题

§39 的 mixed v0 为了让同一个 batch 内 decode row 能采样，把 `do_sample=True` 传给整个 mixed batch。
这样中间 prefill chunk 虽然最终不会 append token，但仍会在 LMHead 输出后被 sampler 采样一次，然后在
`_postprocess_mixed()` 里丢弃。

这个开销理论上主要来自 `[1, vocab]` 的 argmax/softmax/Gumbel-Max；如果 vocab 采样是 mixed v0 的主要瓶颈，
跳过这部分应该降低 mixed step latency。

### 40.2 实现

把 mixed batch 的采样协议从“每个 logits row 都返回一个 token”改成“只为 `step_do_sample=True` 的 row 返回 token”：

- `ModelRunner._select_sample_rows()`：
  - 非 mixed：保持旧行为，采样所有 row。
  - mixed：按 `seq.step_do_sample` 过滤 logits rows 和 sample seqs。
- `Scheduler._postprocess_mixed()`：
  - 中间 prefill chunk 不再消费 dummy sampled token。
  - decode row / final prefill chunk 仍按 `seqs` 中需要采样的顺序消费 token。

测试把 intermediate mixed case 从 `[dummy, decode]` 改成只传 `[decode]`，防止协议回退。

### 40.3 验证

本地：

```text
python3 tools/test_chunked_prefill.py
chunked prefill tests passed

python3 tools/test_profile_chunked_prefill.py
chunked prefill profiler tests passed

python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/scheduler.py tinyvllm/engine/sequence.py tinyvllm/config.py tools/profile_chunked_prefill.py
```

远端 A100 + Qwen3-0.6B，复用 §39 mixed workload：

| mode | prefill steps | mixed steps | decode steps | max decode gap | first_output_ms | total_ms |
|---|---:|---:|---:|---:|---:|---:|
| mixed v0 (§39) | 1 | 11 | 31 | 39.00 ms | 1010.05 ms | 1336.65 ms |
| mixed + sample mask | 1 | 11 | 31 | **38.93 ms** | 1023.33 ms | 1359.66 ms |

### 40.4 结论

- sample mask 正确性成立：中间 prefill chunk 不再需要 dummy sampled token。
- latency 没有变好，total 反而在本次 smoke 中从 1336.65 ms 到 1359.66 ms；差异很可能被 profiler 抖动和 prefill forward 主开销覆盖。
- 结论：mixed v0 的主要瓶颈不是“多采样一个 prefill row”，而是仍走 varlen prefill path / LMHead 仍产出中间 prefill logits。
- 下一步若继续优化，应考虑：
  - LMHead 支持按 row mask 只计算需要采样的 hidden states；或
  - 更严格的 admission policy：只在长 prompt 已经进入 `prefilling` 后 mixed，避免初始短 prompt prefill 也被 chunked/mixed 化；或
  - 真正的 decode+prefill mixed attention kernel，而不是把 decode 包装成 prefill row。

文件留痕：

- `tinyvllm/engine/model_runner.py`
- `tinyvllm/engine/scheduler.py`
- `tools/test_chunked_prefill.py`
- 远端结果：`/tmp/chunked_prefill_latency_mixed_sample_mask.json`

## 41. Chunked Prefill short prompt batching：短 prompt 不再逐条 admission（2026-06-04）

### 41.1 问题

§39/§40 的 profiler 暴露了一个非 mixed-kernel 本身的问题：chunked prefill 开启后，Scheduler 的 prefill admission
一次只取一条 waiting seq。即使初始 4 条 decode prompt 都只有 64 token、且 `max_num_prefill_tokens_per_step=128`，也会被拆成多次
prefill admission；mixed 模式下还会把后续短 prompt prefill 与已有 decode 混在一起。

这会放大 chunked/mixed 的调度开销，并让 profiler 不再只测“长 prompt 插入后”的影响。

### 41.2 实现

在 chunked prefill 路径中保守加入 short prompt batching：

- 如果当前调度的是 `prefilling` 队列中的长 prompt，保持单 chunk 行为。
- 如果从 `waiting` 取出的第一条 seq 在本 step 就是 final chunk，则继续从 waiting 队列追加更多短 prompt：
  - `len(candidate) <= max_num_prefill_tokens_per_step`
  - 不超过 `max_num_seqs`
  - 不超过 `max_num_batched_tokens`
  - `BlockManager.can_allocate(candidate)` 成立
- 所有被追加的 short prompt 都走 final prefill：`do_sample=True`，postprocess 中正常 append 首个输出 token。
- 长 prompt / 中间 chunk 仍保持一次只调度一个 chunk，避免把不同 `do_sample` 语义混进普通 prefill 3-tuple。
- mixed 模式额外保留至少 1 个 `max_num_seqs` slot 给 decode，避免 short prompt batching 把已有 running decode 挤出 mixed batch。

新增测试：两条 4-token prompt 在 `max_num_prefill_tokens_per_step=4` 下应同一 step prefill，并分别消费 sampled token。
回归测试：`chunked_prefill_mixed_batch=True` 且已有 running decode 时，即使 waiting 中有 `max_num_seqs` 条短 prompt，
本轮也必须返回 mixed batch，并保留 decode row。

### 41.3 验证与远端 smoke

本地：

```text
python3 tools/test_chunked_prefill.py
chunked prefill tests passed

python3 tools/test_profile_chunked_prefill.py
chunked prefill profiler tests passed

python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/scheduler.py tinyvllm/engine/sequence.py tinyvllm/config.py tools/profile_chunked_prefill.py
```

远端 A100 + Qwen3-0.6B，复用 §39 workload：

| mode | prefill steps | mixed steps | decode steps | max decode gap | first_output_ms | total_ms |
|---|---:|---:|---:|---:|---:|---:|
| chunked prefill-first (§39 rerun) | 12 | 0 | 33 | 288.12 ms | 1378.22 ms | 1443.17 ms |
| chunked + short batching | 9 | 0 | 33 | 291.93 ms | 1265.50 ms | 1330.37 ms |
| mixed + sample mask (§40) | 1 | 11 | 31 | 38.93 ms | 1023.33 ms | 1359.66 ms |
| mixed + short batching | 1 | 8 | 33 | 36.06 ms | **1004.02 ms** | **1300.95 ms** |
| mixed + short batching + decode slot reserve | 1 | 8 | 33 | **36.46 ms** | 1013.51 ms | 1316.84 ms |

### 41.4 结论

- short prompt batching 修复了 chunked 模式的 admission 低效：初始短 prompts 回到一个 prefill step，prefill steps 从 12 降到 9。
- 对 prefill-first，total 从 1443.17 ms 降到 1330.37 ms，first output 也提前；但长 prompt 的连续 prefill chunk 仍会造成约 292 ms decode gap。
- 对 mixed，mixed steps 从 11 降到 8；加入 decode slot reserve 后仍保持较低 decode gap（36.46 ms），total 为 1316.84 ms。
- decode slot reserve 是正确性/公平性修复：防止短 prompt batching 占满 batch，让已有 running decode 被挤出 mixed step。
- 这说明 mixed v0 的一个主要可修收益不是 sample mask，而是 admission policy：不要把本可同批完成的短 prompt 拆成多步。

文件留痕：

- `tinyvllm/engine/scheduler.py`
- `tools/test_chunked_prefill.py`
- 远端结果：`/tmp/chunked_prefill_latency_chunked_short_batch.json`
- 远端结果：`/tmp/chunked_prefill_latency_mixed_short_batch.json`
- 远端结果：`/tmp/chunked_prefill_latency_mixed_short_batch_reserve.json`

## 42. Mixed Prefill+Decode logits indices：LMHead 只计算需要采样的 row（2026-06-04）

### 42.1 问题

§40 的 sample mask 只是在 LMHead 输出之后过滤 logits row：中间 prefill chunk 的 sampled token 不再生成，
但 LMHead 仍然会先对这个 row 做一次 `[hidden, vocab]` 投影。对 mixed batch 来说，intermediate prefill chunk
的 logits 最终一定被丢弃，因此这部分 vocab 维度计算是纯开销。

目标：把 row selection 前移到 LMHead 输入侧，让 prefill/mixed forward 只对真正需要采样的 hidden row 计算 logits。

### 42.2 实现

新增 `Context.logits_indices`：

```python
logits_indices: torch.Tensor | None = None
```

语义：在 `context.is_prefill=True` 时，`logits_indices` 指向 flattened hidden states 中需要进入 LMHead 的 row。

具体改动：

- `ModelRunner.prepare_mixed()`：按 mixed batch 中每条 seq 的 query row 起点 `q_start` 收集 logits row。
  - `step_do_sample=True`：加入 `q_start + seqlen_q - 1`。
  - `step_do_sample=False`：跳过，代表 intermediate prefill chunk 不需要 logits。
- `set_context(..., logits_indices=...)`：把 indices 透传到全局 context。
- `ParallelLMHead.forward()`：prefill 路径优先使用 `context.logits_indices`；若为 `None`，回退旧逻辑
  `context.cu_seqlens_q[1:] - 1`，保持普通 prefill 兼容。
- `ModelRunner._select_sample_rows()`：mixed 下不再切 logits tensor，只返回需要采样的 seq 列表；因为 LMHead 已经只输出这些 row。

### 42.3 验证

本地新增 CPU 单测：构造 6 行 hidden、`cu_seqlens_q=[0,2,5,6]`、`logits_indices=[1,5]`，确认
LMHead 输出形状从旧的 3 row 变成 2 row，并且数值等于 `hidden[[1, 5]] @ W^T`。

远端 A100 + Qwen3-0.6B，复用 §41 mixed workload：

| mode | mixed steps | max decode gap | total_ms |
|---|---:|---:|---:|
| mixed + short batching + decode slot reserve (§41) | 8 | **36.46 ms** | **1316.84 ms** |
| mixed + logits_indices | 8 | 139.45 ms | 1515.69 ms |

### 42.4 结论

- 正确性成立：LMHead 不再为 `step_do_sample=False` 的 intermediate prefill chunk 计算 logits，采样协议也保持一致。
- 本次远端 latency 没有改善：`total_ms` 从 1316.84 ms 上升到 1515.69 ms，`max_gap` 从 36.46 ms 上升到 139.45 ms。
- 画像解释：本次 workload 中长 prompt chunk 更早进入第一个 mixed step，p95/max 被大 chunk 的 attention 计算拉高；
  `logits_indices` 只能减少 LMHead vocab 投影，对 chunk attention 主开销无能为力，因此 wall time 受 admission 时序和 chunk 位置影响更大。
- 结论：保留 `logits_indices` 作为正确的低开销协议优化，但它不是 mixed v0 当前总 latency 的主瓶颈。继续优化应优先看
  admission policy / chunk size 自适应 / 真正的 mixed attention kernel。

文件留痕：

- `tinyvllm/utils/context.py`
- `tinyvllm/layers/embed_head.py`
- `tinyvllm/engine/model_runner.py`
- `tools/test_chunked_prefill.py`
- 远端结果：`/tmp/chunked_prefill_latency_mixed_logits_indices.json`

## 43. W4A8+SQ needle 回归复核：skip last=4 是当前稳态配置（2026-06-05）

### 43.1 背景

§23 曾记录 `W4A8+SQ α=0.85 + skip first=2/last=2` 在 Qwen3-8B needle 16K 上达到 93.3%。
后续 §27/§29 已经指出首部 skip 浪费，尾部 outlier 更关键，`skip_last=4` 是更稳配置。

本轮在同步最新代码到远端后重新跑 needle，目标是确认当前代码路径下推荐配置是否仍然成立。

### 43.2 远端环境

- 机器：`sitian@10.232.195.203`
- 代码目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`
- Python：`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`（torch 2.4.1+cu121）
- GPU：`CUDA_VISIBLE_DEVICES=7`
- 模型：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B`
- SQ scale：`/tmp/sq_scales_qwen3_8b_a0.85.pt`

### 43.3 复跑结果

先复跑旧配置 `skip first=2 / skip last=2`：

```text
--quantization int4 --quant-group-size 32
--act-quant-bits 8
--smoothquant-scale-path /tmp/sq_scales_qwen3_8b_a0.85.pt
--act-quant-skip-first 2 --act-quant-skip-last 2
```

完整 baseline-only needle（ctx=4096/8192/15000，depth=0/0.25/0.5/0.75/1.0，n=2）：

| config | overall_acc | throughput |
|---|---:|---:|
| W4A8+SQ α=0.85 + skip2/2 | 83.3% | 25.60 tok/s |

失败仍是典型长文末尾 instruction 复读：`Answer with only the digits...`，不是数字算错。

再跑当前推荐配置 `skip_last=4`：

```text
--quantization int4 --quant-group-size 32
--act-quant-bits 8
--smoothquant-scale-path /tmp/sq_scales_qwen3_8b_a0.85.pt
--act-quant-skip-last 4
```

远端确认跳过层：

```text
[act-quant-skip] disabled A8 on 16 LinearBase modules (layers=[32, 33, 34, 35], total=36)
```

结果：

| config | overall_acc | throughput |
|---|---:|---:|
| **W4A8+SQ α=0.85 + skip_last=4** | **100.0%** | 25.52 tok/s |

所有 ctx/depth bucket 均为 100%。

### 43.4 结论

- `skip2/2` 仍比无 skip 的 40% 明显好，但当前复跑只到 83.3%，不再作为推荐配置。
- `skip_last=4` 当前复跑达到 100%，并且 TPS 与 `skip2/2` 基本持平，说明把 A8 skip 预算集中在尾部 outlier 层更稳。
- 后续 W4A8+SQ 长上下文默认应优先使用：

```text
--quantization int4 --quant-group-size 32
--act-quant-bits 8
--smoothquant-scale-path /tmp/sq_scales_qwen3_8b_a0.85.pt
--act-quant-skip-last 4
```

### 43.5 full-stack 复跑：layer-adaptive SQ + KV8 + Quest top-k=12

继续复跑 §31 的速度/质量平衡配置：

```text
--quantization int4 --quant-group-size 32
--act-quant-bits 8 --act-quant-skip-last 4
--smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt
--kv-quant-bits 8 --kv-quant-group-size 32
--top-k-blocks-list -1 12
```

结果（同样是 30 sample，n=2）：

| setting | overall_acc | throughput |
|---|---:|---:|
| C8 baseline/full attention | **100.0%** | 17.23 tok/s |
| C8 + Quest top-k=12 | 96.7% | **24.53 tok/s** |

对比 §43.3 的 fp16 KV full-attn `skip_last=4`：100.0% / 25.52 tok/s。
KV8 baseline 主要收益是 KV 显存，不是速度；叠加 Quest top-k=12 后在保持 96.7% 召回的同时，
相对 KV8 full attention 提升约 42% TPS。

当前 full-stack 推荐仍是：

```text
W4(g32) + A8(skip last=4) + SQ(layer-adaptive α=0.85~0.90) + KV8(g32) + Quest(top-k=12)
```

### 43.6 fixed-prompt n=5 公平口径复跑

为了排除不同 setting 使用不同随机 prompt 带来的质量归因偏差，继续复跑 fixed-prompt + cache-clear 口径：

```text
--fixed-prompts
--num-trials 5
--top-k-blocks-list -1 12
--smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt
--act-quant-skip-last 4
--kv-quant-bits 8 --kv-quant-group-size 32
```

结果（75 sample / setting）：

| setting | overall_acc | throughput | failures |
|---|---:|---:|---:|
| C8 baseline/full attention | 96.0% | 19.45 tok/s | 3/75 |
| C8 + Quest top-k=12 | 93.3% | **27.10 tok/s** | 5/75 |

失败集合：

- baseline 的 3 个失败都集中在 `(ctx=4096, depth=0.5)`，输出为 question/instruction 复读。
- top-k=12 继承这 3 个失败，并新增 2 个：
  - `(15000, 0.25, trial=0, magic=35043)` 输出截成 `354`；
  - `(15000, 0.75, trial=4, magic=82255)` 输出成 `822255`。

对比历史 fixed-prompt n=5：

| run | baseline | top-k=12 |
|---|---:|---:|
| §33 历史 | 94.7% / 19.19 | **96.0% / 26.80** |
| 本轮复跑 | **96.0% / 19.45** | 93.3% / **27.10** |

结论：top-k=12 的速度收益仍稳定（约 +39% TPS），但本轮 fixed-prompt 质量比历史低 2/75 个样本，
不再声称 top-k=12 “不降质量”；更准确表述是：**top-k=12 是高吞吐折中档，baseline/full attention 是最高召回档**。

### 43.7 fixed-prompt n=5 top-k=8/12/16 补扫

为了判断 §43.6 中 top-k=12 掉到 93.3% 后，当前甜点是否仍是 12，继续用同一 fixed-prompt 口径补跑
`top_k ∈ {8,16}`。

| setting | overall_acc | throughput | failures |
|---|---:|---:|---:|
| C8 baseline/full attention | 96.0% | 19.45 tok/s | 3/75 |
| C8 + Quest top-k=8 | 94.7% | **28.39 tok/s** | 4/75 |
| C8 + Quest top-k=12 | 93.3% | 27.10 tok/s | 5/75 |
| C8 + Quest top-k=16 | **96.0%** | 25.83 tok/s | 3/75 |

失败集合上，baseline/top-k=16 只有同一组 `(4096, depth=0.5, trial=1/2/3)` question 复读失败；
top-k=8 额外新增 `(15000, depth=0.0, trial=0, magic=74694)`；top-k=12 额外新增两条 15K 数字截断/重复错误。

本轮 fixed-prompt n=5 下，推荐从 §31 的 top-k=12 调整为：

- **最高召回**：C8 baseline/full attention（96.0%，19.45 tok/s）或 Quest top-k=16（96.0%，25.83 tok/s）。
- **最高吞吐折中**：Quest top-k=8（94.7%，28.39 tok/s）。
- **top-k=12** 本轮不再是最优点：速度低于 top-k=8，召回低于 top-k=16。

文件留痕：

- `needle_sq_results/needle_w4a8_sq_a085_g32_skip2_rerun_full_baseline.json`
- `needle_sq_results/needle_w4a8_sq_a085_g32_skiplast4_rerun_baseline.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_topk12_rerun_n2.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_fixed_prompts_topk12_rerun_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_fixed_prompts_topk8_16_rerun_n5.json`

## 44. TP=2 W4A8+SQ 稳态配置验证（2026-06-05）

继续验证 §43 推荐的 `skip_last=4` 在 tensor parallel 下是否存在 scale slicing / shape / 通信问题。

### 44.1 TP smoke：W4A8+SQ g32 + skip_last=4

命令口径：

```text
CUDA_VISIBLE_DEVICES=5,7
tools/tp_smoke.py
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
  --tp-size 2
  --filter w4a8_sq_g32
  --smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt
  --act-quant-skip-last 4
  --prompt-source english
  --out-file tp_smoke_results/tp2_w4a8_sq_g32_skiplast4.json
```

结果：

| config | TP | init | gen | decode_tps | weight_mem | kv_cache_mem | peak_mem |
|---|---:|---|---|---:|---:|---:|---:|
| W4A8+SQ g32 + skip_last=4 | 2 | ok | ok | 62.27 | 3.219 GB | 49.006 GB | 52.225 GB |

远端日志确认每个 rank 都成功加载 SQ scale 并跳过尾部 4 层 A8：

```text
[smoothquant] applied scales to 144 modules (skipped 0 without matching key)
[act-quant-skip] disabled A8 on 16 LinearBase modules (layers=[32, 33, 34, 35], total=36)
```

### 44.2 full-stack smoke：W4A8+SQ + KV8 + Quest top-k=16

继续用同两张 A100 跑完整组合：

```text
W4(g32) + A8(skip_last=4) + SQ(layer-adaptive floor085) + KV8(g32) + Quest(top-k=16), TP=2
```

结果：

| config | TP | gen_ok | elapsed |
|---|---:|---|---:|
| W4A8+SQ g32 + KV8 g32 + Quest top-k=16 | 2 | true | 4.86 s |

该 smoke 只验证初始化、TP 分片、SQ scale 注入、KV8 cache、Quest summary 与一次 generate 路径均可跑通；
不是 needle 质量结论。样例中第一个 prompt 仍有逗号复读，后续若要给 TP=2 质量背书，需要把
`eval_needle.py` 的 `tensor_parallel_size=1` 参数化后再跑 fixed-prompt needle。

### 44.3 工具修复

本轮第一次 smoke 用 `--out-file tp_smoke_results/...json` 时，主流程已经跑通但汇总写文件失败：

```text
FileNotFoundError: .../tp_smoke_out/tp_smoke_results/tp2_w4a8_sq_g32_skiplast4.json
```

根因是 `tp_smoke.py` 会把 `--out-file` 拼到固定 `tp_smoke_out/` 下，但没有创建 `out_file` 中的子目录。
已补 `prepare_summary_path()`，在写 summary 前创建 parent directory，并加入 CPU helper 测试覆盖。

文件留痕：

- `tp_smoke_out/tp_smoke_results/tp2_w4a8_sq_g32_skiplast4.json`
- `tp_smoke_out/tp_smoke_results/tp2_fullstack_w4a8_sq_kv8_quest16.json`

### 44.4 TP=2 fixed-prompt needle 质量复核（2026-06-08）

为把 §44.2 的“通路跑通”推进到 needle 质量背书，先将 `tools/eval_needle.py` 的
`tensor_parallel_size=1` 硬编码改成可配置：

```text
--tp-size 2
```

默认仍为 TP=1；本轮新增 CPU 测试覆盖 `tp_size=2 -> tensor_parallel_size=2` 的参数透传。

远端 fixed-prompt n=5 口径：

```text
CUDA_VISIBLE_DEVICES=3,5
--tp-size 2
--fixed-prompts --num-trials 5
--top-k-blocks-list -1 16
--quantization int4 --quant-group-size 32
--act-quant-bits 8 --act-quant-skip-last 4
--smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt
--kv-quant-bits 8 --kv-quant-group-size 32
```

随后复用同一 fixed-prompt seed 单独补跑 `--top-k-blocks-list 4/5/6/8/12`，
用于确认 TP=2 下最高吞吐档、历史折中档，以及更激进稀疏档的质量边界。

结果（75 sample / setting）：

| setting | TP | overall_acc | throughput | failures |
|---|---:|---:|---:|---:|
| C8 baseline/full attention | 2 | 94.7% | 27.31 tok/s | 4/75 |
| C8 + Quest top-k=4 | 2 | 80.0% | 35.94 tok/s | 15/75 |
| C8 + Quest top-k=5 | 2 | 89.3% | 35.54 tok/s | 8/75 |
| C8 + Quest top-k=6 | 2 | 94.7% | **35.54 tok/s** | 4/75 |
| C8 + Quest top-k=8 | 2 | 94.7% | **35.12 tok/s** | 4/75 |
| C8 + Quest top-k=12 | 2 | 94.7% | 34.10 tok/s | 4/75 |
| C8 + Quest top-k=16 | 2 | 94.7% | **33.21 tok/s** | 4/75 |

`top-k=6/8/12/16` 的失败集合与 full attention 完全相同，均集中在 `(ctx=4096, depth=0.5)` 的 4 个 trial：

- `trial=0, magic=86465`
- `trial=1, magic=38631`
- `trial=2, magic=76150`
- `trial=4, magic=46941`

失败输出仍是 question/instruction 复读，没有新增 15K 长上下文数字截断/重复错误。

`top-k=5` 与 `top-k=4` 仍会额外掉质量，边界很清楚：

- `top-k=5` 掉到 89.3%，比 baseline/top-k=6 多 4 个 `(4096, depth=0.25)` haystack 复读失败；
- `top-k=4` 掉到 80.0%，额外失败进一步扩大到：
  - `(4096, depth=0.25)`：5/5 全错，输出为 haystack 复读；
  - `(8192, depth=0.25)`：5/5 全错，出现错误 magic 数字或截断数字；
  - `(8192, depth=0.75)`：新增 1 条错误数字。

与 TP=1 fixed-prompt n=5 的 top-k=16 结果（§43.7：96.0% / 25.83 tok/s，3/75 failures）相比，
TP=2 本轮多 1 个 `(4096, depth=0.5)` 复读失败，但 top-k=6/8/12/16 与 full attention 完全对齐；
因此当前可给出的 TP=2 结论是：**scale slicing / KV8 / Quest TP 通路稳定，top-k=6/8/12/16 都不引入额外 needle 质量损失；top-k=6 是当前最高吞吐且不掉质量档（相对 TP=2 full attention +30.1%），top-k=8 是更稳妥的推荐吞吐档（+28.6%），top-k=12 为中间档（+24.9%），top-k=16 是更保守档（+21.6%）；top-k≤5 过稀疏，不推荐**。

文件留痕：

- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_tp2_fixed_prompts_topk16_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_tp2_fixed_prompts_topk8_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_tp2_fixed_prompts_topk12_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_tp2_fixed_prompts_topk4_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_tp2_fixed_prompts_topk6_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_tp2_fixed_prompts_topk5_n5.json`

## 45. 潜空间 KV 压缩：从 uniform KV-Cartridge 到 Attention Matching（2026-06-08）

前面 §36~§44 的主线都是在既有 KV cache 上做**读写侧优化**：chunked prefill 降低 prefill 阻塞、
KV8 降显存、Quest 在 decode 阶段只读 top-k block。用户提出进一步尝试类似 Cartridges 的
“潜空间压缩”方向，但优先级明确为**秒级可用性**：不接受对每段上下文做数小时端到端梯度下降。

本轮先落两级原型：

1. **KV-Cartridge v0 / uniform read-side compaction**：不训练、不改写 KV cache，只在 decode 时把
   `block_table` 压成少量代表性 block（保首尾，中间均匀抽样），验证“简单压缩历史 block”是否足够；
2. **Fast KV Compaction via Attention Matching 核心算法**：实现论文
   *Fast KV Compaction via Attention Matching*（arXiv:2602.16284）里的快速
   `AM-HighestAttnKeys` 单 layer/head 数学路径，先验证 `beta + C_v least-squares` 是否真的能显著降低
   attention-output matching 误差。

### 45.1 KV-Cartridge v0 实现与 fixed-prompt 曲线

实现提交：

```text
c1bd1c5 实验：加入 KV-Cartridge read-side 压缩原型
```

关键改动：

- `tinyvllm/engine/kv_cartridge.py`：新增 uniform block 选择、batch 启用判定、compact `block_table/context_lens` helper；
- `tinyvllm/engine/model_runner.py`：decode 准备阶段在 `set_context()` 前可替换 compact block table；
- `tinyvllm/config.py`：新增 `kv_cartridge_blocks` / `kv_cartridge_min_seq_len` / `kv_cartridge_mode`；
- `tools/eval_needle.py`：新增 `--kv-cartridge-blocks` 等评测参数；
- `tools/test_kv_cartridge.py`、`tools/test_eval_needle_fixed_prompts.py`：覆盖 helper 与参数透传。

远端 fixed-prompt n=5，沿用当前全栈稳态配置：

```text
Qwen3-8B
W4 g32 + A8 skip_last=4 + layer-adaptive SQ floor0.85
KV8 g32
--fixed-prompts --num-trials 5
ctx=4096/8192/15000, depth=0/0.25/0.5/0.75/1.0
```

结果（75 sample / setting）：

| setting | overall_acc | throughput | failures |
|---|---:|---:|---:|
| KV-Cartridge uniform b=4 | 33.3% | 30.00 tok/s | 50/75 |
| KV-Cartridge uniform b=6 | 26.7% | 29.32 tok/s | 55/75 |
| KV-Cartridge uniform b=8 | 42.7% | 28.77 tok/s | 43/75 |
| KV-Cartridge uniform b=12 | 56.0% | 27.66 tok/s | 33/75 |
| KV-Cartridge uniform b=16 | 69.3% | 26.33 tok/s | 23/75 |
| KV-Cartridge uniform b=32 | **96.0%** | 19.48 tok/s | 3/75 |

对比 §43.7 的 TP=1 推荐点：

- `C8 + Quest top-k=16`：96.0% / 25.83 tok/s / 3 failures；
- `KV-Cartridge uniform b=32`：96.0% / 19.48 tok/s / 3 failures。

结论：**uniform read-side compaction 只能作为 sanity baseline**。它在 b≤16 时质量明显不够；b=32 能回到
Quest top-k=16 的质量，但吞吐反而更低，说明“均匀保块”没有利用 query/KV 内容，无法替代 Quest 或真正的
latent compaction。

文件留痕：

- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_kv8_kvcartridge_b4_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_kv8_kvcartridge_b6_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_kv8_kvcartridge_b8_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_kv8_kvcartridge_b12_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_kv8_kvcartridge_b16_n5.json`
- `needle_sq_results/needle_sq_layer_adaptive_floor085_skiplast4_kv8_kvcartridge_b32_n5.json`

### 45.2 Attention Matching 核心算法实现

论文与代码参考：

- 论文：*Fast KV Compaction via Attention Matching*，<https://arxiv.org/pdf/2602.16284>
- 代码：<https://github.com/adamzweiger/compaction>

实现提交：

```text
9bb4677 实现：加入 Attention Matching KV 压缩核心算法
```

本轮先实现单 `(layer, KV-head)` 的 `AM-HighestAttnKeys` 核心，不急着接真实 decode cache：

1. 给定原始 `K,V ∈ R^{T×d}` 与 reference queries `Q ∈ R^{n×d}`；
2. 计算 `softmax(QK^T/sqrt(d))`，按 RMS/mean/max attention score 选择 top `budget` 个 key；
3. 固定 compact keys `C_k=K[S]`，用 box-NNLS 拟合 attention bias `beta`，让 compact keys 的
   unnormalized attention mass 接近原始 KV mass；
4. 固定 `C_k,beta`，用 least squares 拟合 compact values `C_v`，最小化
   `softmax(QC_k^T/sqrt(d)+beta) C_v` 与原始 `softmax(QK^T/sqrt(d)) V` 的输出误差。

关键文件：

- `tinyvllm/engine/attention_matching.py`
  - `highest_attention_key_indices()`：AM-HighestAttnKeys 的 key 选择；
  - `fit_attention_bias()`：attention-mass matching；
  - `fit_compacted_values()`：attention-output matching；
  - `attention_matching_highest_keys()`：端到端返回 `C_k/beta/C_v/indices`；
- `tools/test_attention_matching.py`：覆盖 key selection、mass matching、value LSQ 优于 direct value、输出 shape/dtype。

远端 A100 合成单 head 验证：

```text
T=1024, d=128, queries=128, device=cuda
```

| budget | ratio | direct subset MSE | AM fitted MSE | improvement |
|---:|---:|---:|---:|---:|
| 16 | 1.6% | 1.677880e-01 | 1.291376e-03 | 129.93× |
| 32 | 3.1% | 9.969880e-02 | 9.998721e-04 | 99.71× |
| 64 | 6.2% | 5.242080e-02 | 5.916372e-04 | 88.60× |
| 128 | 12.5% | 2.262743e-02 | 3.463814e-04 | 65.33× |

这说明论文核心 insight 在我们的张量口径上成立：**同样选择高 attention keys，直接使用原始 `V[S]` 误差很大；
加入 `beta + C_v least-squares` 后 attention output matching 误差下降约 65~130 倍。**

验证命令：

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_attention_matching.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_kv_cartridge.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_eval_needle_fixed_prompts.py
```

输出：

```text
attention matching tests passed
KV-Cartridge helper tests passed
eval_needle fixed-prompt tests passed
```

### 45.3 后续执行门槛与 todo

当前不要直接做 OMP-fast。原因：OMP 是论文更强版本，但工程复杂度更高，且真实 decode 路径尚未接入
`C_k/beta/C_v`。最高性价比路线是：

1. **先把 `AM-HighestAttnKeys` 接入真实模型 decode 路径**：
   - 支持 compact prefix `C_k/beta/C_v`；
   - 先只走 eager decode，不碰 CUDA graph；
   - 先用 context-prefill queries，后续再补 repeat-prefill / self-study；
   - 先跑 budget `b=16/32/64/128` 的 fixed-prompt needle 曲线。
2. **判定是否进入 OMP-fast**：
   - 如果 `AM-HighestAttnKeys b=16/32` 能接近 Quest top-k=16 的质量（当前参考：96.0% / 3 failures），
     再继续做 OMP-fast；
   - 如果 b=16/32 质量明显不够，但 b=64/128 才接近，则先优化 query 采样/分层预算，不急着上 OMP；
   - 如果 AM 在真实 decode 上无法明显优于 uniform KV-Cartridge，则 OMP-fast 暂缓，优先查 query 分布与
     `beta` 在 FlashAttention 路径中的注入方式。

后续推荐实验表：

| method | budget | 目标 |
|---|---:|---|
| full attention / KV8 | - | 质量上界 |
| Quest | top-k=16 | 当前 TP=1 参考点：96.0% / 25.83 tok/s |
| KV-Cartridge uniform | b=32 | 简单 read-side baseline：96.0% / 19.48 tok/s |
| AM-HighestAttnKeys | b=16/32 | 若接近 Quest 质量，进入 OMP-fast |
| AM-HighestAttnKeys | b=64/128 | 判断 AM 是否只是需要更大 budget |
| OMP-fast | TBD | 仅在 AM-HighestAttnKeys 达到门槛后实现 |

### 45.4 AM compact tensors 接入 decode 路径（2026-06-09）

实现提交：

```text
7a04c82 实现：接入 Attention Matching decode 路径
```

本轮按“秒级可用性优先”只做 eager decode 验证路径，不碰 CUDA graph：

1. `tinyvllm/engine/attention_matching.py` 新增 `attention_matching_decode()`：
   - 输入 dense fp KV cache `[B, T, kv_h, hd]` 与 decode query `[B, q_h, hd]`；
   - 按 GQA group 在每个 `(batch, kv_head)` 上运行 `AM-HighestAttnKeys`；
   - 输出路径显式使用 `C_k / beta / C_v`；
   - 当 `budget >= seq_len` 时退化为 full attention，方便单测校验。
2. `tinyvllm/layers/attention.py` 接入两条 decode 分支：
   - fp16 KV：按 `block_table` gather 成 dense cache 后进入 AM；
   - KV8：先 dequant 命中 blocks，再 reshape 成 dense cache 后进入 AM；
   - KV4 暂不支持，避免一次性引入 int4 dequant + AM attribution 混淆。
3. `tinyvllm/engine/model_runner.py`：
   - AM 开启时跳过 CUDA graph capture；
   - decode 阶段按 `am_compact_min_seq_len` 与 `am_compact_blocks` 判定是否启用；
   - AM 与 Quest / KV-Cartridge 互斥，保证 needle 曲线可归因。
4. `tools/eval_needle.py` 新增 AM CLI：
   - `--am-compact-blocks`；
   - `--am-compact-min-seq-len`；
   - `--am-compact-score-method`；
   - `--am-compact-beta-bound`；
   - `--am-compact-ridge-lambda`。

本地可执行验证（当前 macOS `/usr/bin/python3` 无 torch，AM torch 单测需远端 A100 补跑）：

```text
python3 tools/test_eval_needle_fixed_prompts.py  -> eval_needle fixed-prompt tests passed
python3 tools/test_chunked_prefill.py            -> chunked prefill tests passed
python3 -m py_compile ...                        -> passed
git diff --check                                 -> passed
python3 tools/test_attention_matching.py         -> blocked locally: ModuleNotFoundError: No module named 'torch'
```

远端 A100 状态：Kerberos 凭据仍过期，`ssh sitian@10.232.195.203` 返回
`Connection closed by UNKNOWN port 65535`，`kinit -R` 返回
`Matching credential (krbtgt/BYTEDANCE.COM@BYTEDANCE.COM) not found`。待凭据恢复后补跑：

```bash
python tools/test_attention_matching.py
python tools/eval_needle.py --model <Qwen3-8B> --kv-quant-bits 8 --am-compact-blocks 16  --num-trials 5 --fixed-prompts
python tools/eval_needle.py --model <Qwen3-8B> --kv-quant-bits 8 --am-compact-blocks 32  --num-trials 5 --fixed-prompts
python tools/eval_needle.py --model <Qwen3-8B> --kv-quant-bits 8 --am-compact-blocks 64  --num-trials 5 --fixed-prompts
python tools/eval_needle.py --model <Qwen3-8B> --kv-quant-bits 8 --am-compact-blocks 128 --num-trials 5 --fixed-prompts
```

### 45.5 AM-HighestAttnKeys 真实模型 fixed-prompt needle 曲线（2026-06-09）

远端环境与口径：

```text
机器：sitian@10.232.195.203 / A100 80GB PCIe
Python：/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
模型：/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
配置：W4(g32) + A8(skip_last=4) + SQ(layer-adaptive floor0.85) + KV8(g32)
评测：--fixed-prompts --num-trials 5
ctx=4096/8192/15000, depth=0/0.25/0.5/0.75/1.0，共 75 sample / setting
```

命令要点：

```bash
python tools/eval_needle.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --fixed-prompts --num-trials 5 \
  --quantization int4 --quant-group-size 32 \
  --act-quant-bits 8 --act-quant-skip-last 4 \
  --smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt \
  --kv-quant-bits 8 --kv-quant-group-size 32 \
  [--top-k-blocks-list -1 16 | --kv-cartridge-blocks 32 | --am-compact-blocks B]
```

结果：

| setting | overall_acc | throughput | failures | 备注 |
|---|---:|---:|---:|---|
| C8 baseline/full attention | 96.0% | 19.34 tok/s | 3/75 | 质量上界 |
| C8 + Quest top-k=16 | 96.0% | 25.73 tok/s | 3/75 | 当前推荐召回档 |
| KV-Cartridge uniform b=32 | 96.0% | 19.38 tok/s | 3/75 | uniform sanity baseline |
| AM-HighestAttnKeys b=16 | **96.0%** | 2.16 tok/s | 3/75 | 与 Quest/full 同失败集合 |
| AM-HighestAttnKeys b=32 | **96.0%** | 2.12 tok/s | 3/75 | 与 Quest/full 同失败集合 |
| AM-HighestAttnKeys b=64 | **96.0%** | 2.06 tok/s | 3/75 | 与 Quest/full 同失败集合 |
| AM-HighestAttnKeys b=128 | 93.3% | 1.96 tok/s | 5/75 | `gpu_memory_utilization=0.6` 重跑；新增 2 个 `4096/depth=0.0` haystack 复读失败 |

失败集合：

- baseline / Quest top-k=16 / KV-Cartridge b=32 / AM b=16/32/64 的失败完全一致：
  - `(ctx=4096, depth=0.5, trial=0, magic=86465)`；
  - `(ctx=4096, depth=0.5, trial=1, magic=38631)`；
  - `(ctx=4096, depth=0.5, trial=2, magic=76150)`；
  - 输出均为 question/instruction 复读，未抽到数字。
- AM b=128 除上述 3 个外，新增：
  - `(ctx=4096, depth=0.0, trial=2, magic=15306)`；
  - `(ctx=4096, depth=0.0, trial=4, magic=77013)`；
  - 输出为 haystack 复读。

b=128 说明：默认 `gpu_memory_utilization=0.9` 首跑在一张已有其它进程占用的 A100 上因 KV8 full dequant buffer OOM；随后用
`--gpu-memory-utilization 0.6` 在空闲卡重跑完成。因此 b=128 的吞吐与其它 setting 不完全同显存口径，但质量结论仍可作为参考。

结论：

1. **质量门槛已达到**：AM b=16/32 已达到 Quest top-k=16 的 96.0% / 3 failures 参考门槛，且失败集合完全一致；按 §45.3 的规则，
   可以进入 OMP-fast 方向。
2. **当前 eager v0 不能作为可用吞吐方案**：AM b=16/32/64 只有约 2 tok/s，显著慢于 Quest top-k=16 的 25.73 tok/s。
   这是预期工程代价：当前每层、每步、每个 KV head 都即时 gather/dequant dense KV，并在线做 top attention key、beta NNLS、C_v least-squares。
3. **下一步做 OMP-fast 时必须同时改执行形态**：不要只把选择器从 HighestAttnKeys 换成 OMP；需要避免每个 decode step 重复完整拟合，至少应考虑
   reference query 采样、按 block/层缓存 compact tensors、低频 refresh，或只在长上下文阶段触发，否则 OMP-fast 即使质量更好也会被 Python eager overhead 吞掉。

文件留痕：

- `needle_sq_results/needle_am_compare_full_quest16_n5.json`
- `needle_sq_results/needle_am_compare_kvcartridge_b32_n5.json`
- `needle_sq_results/needle_am_compare_am_b16_n5.json`
- `needle_sq_results/needle_am_compare_am_b32_n5.json`
- `needle_sq_results/needle_am_compare_am_b64_n5.json`
- `needle_sq_results/needle_am_compare_am_b128_n5_retry_gmem06.json`

### 45.6 AM-OMP selector 接入（2026-06-11）

承接 §45.5：AM-HighestAttnKeys b=16/32 已达到 Quest top-k=16 的质量门槛，因此进入 OMP-fast 方向。
本轮先做最小质量验证入口，而不是吞吐优化版：只把 Attention Matching 的 key selector 从
`highest` 扩展为可选 `omp`，beta box fitting 与 `C_v` least-squares 仍复用 §45.2 的实现。

实现范围：

- `tinyvllm/engine/attention_matching.py`：新增 `omp_attention_key_indices()` 与
  `attention_matching_compact_keys(selector=...)`；
- OMP 不是全 token 暴力搜索，而是先用 HighestAttnKeys 取一个小候选池（默认 `max(2*b, b+4)`，
  可通过 `--am-omp-candidate-pool-size` 覆盖），
  再在候选池内逐步 greedy 拟合，避免长上下文 fixed-prompt smoke 直接不可用；
- `Config.am_compact_selector`：支持 `highest` / `omp`；
- `tools/eval_needle.py --am-compact-selector`：真实 needle 评测可直接切换 selector。

本轮刻意不做：

- compact tensor cache；
- decode step 低频 refresh；
- block/layer 级 selector 结果复用；
- OMP GPU kernel 化。

原因是先分离质量问题和执行形态问题：若 OMP 在 fixed-prompt needle 上不能优于 HighestAttnKeys，
则不值得继续优化工程路径；若 OMP 质量更稳，再进入 OMP-fast execution-shape 优化。

本地验证：

```bash
python3 -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python3 tools/test_attention_matching.py
```

远端验证（`sitian@10.232.195.203`，`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`）：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

最小模型链路 smoke（Qwen3-0.6B，`ctx=512/depth=0.5/num_trials=1/max_output=1/b=2`）：

| selector | acc | throughput | 输出文件 |
|---|---:|---:|---|
| highest | 0.0% | 20.39 tok/s | `needle_sq_results/needle_am_selector_highest_b2_smoke.json` |
| omp (`candidate_pool=4`) | 0.0% | 19.25 tok/s | `needle_sq_results/needle_am_selector_omp_smoke.json` |

这个 smoke 只证明 CLI/config/decode 链路走通，不作为质量结论；正式质量结论仍需按 §45.5 的 8B fixed-prompt 口径跑 b=16/32。

追加 8B 单样本对照（Qwen3-8B，`ctx=4096/depth=0.5/num_trials=1/max_output=4`，同一 magic=60494）：

| selector | b | candidate_pool | acc | throughput | raw 输出 |
|---|---:|---:|---:|---:|---|
| HighestAttnKeys | 16 | - | 0.0% | 1.2248 tok/s | ` The magic number is` |
| HighestAttnKeys | 32 | - | 0.0% | 1.2576 tok/s | ` The magic number is` |
| OMP | 16 | 20 | 0.0% | 0.0173 tok/s | ` The magic number is` |
| OMP | 32 | 36 | 0.0% | 0.0051 tok/s | ` The magic number is` |

输出文件：

- `needle_sq_results/needle_am_highest_b16_ctx4096_d05_n1_smoke.json`
- `needle_sq_results/needle_am_highest_b32_ctx4096_d05_n1_smoke.json`
- `needle_sq_results/needle_am_omp_b16_ctx4096_d05_n1_smoke.json`
- `needle_sq_results/needle_am_omp_b32_ctx4096_d05_n1_smoke.json`

结论：当前 eager OMP 质量链路已接通，但执行形态不可用。即使只在 HighestAttnKeys 候选池内做 greedy OMP，
`ctx=4096` 单样本吞吐也从 Highest 的约 `1.2 tok/s` 降到 OMP b=16 的 `0.017 tok/s`、b=32 的 `0.005 tok/s`。
因此不应继续跑默认 full grid 的 OMP n=1/n=5；下一步必须先做 OMP-fast execution-shape（缓存 compact tensors、低频 refresh、减少每步/每层重拟合），
否则正式 fixed-prompt 曲线没有工程意义。

### 45.7 AM-OMP decode-step cache / low-frequency refresh 原型（2026-06-11）

为验证 §45.6 的 execution-shape 判断，本轮加了最小 cache 原型：每个 Attention layer 持有一个
`AttentionMatchingDecodeCache`，首次 decode 或超过 `am_compact_cache_refresh_interval` 时拟合 compact KV；
刷新间隔内复用 `C_k/beta/C_v`，只对当前 query 做 compact attention 输出。

新增配置 / CLI：

- `Config.am_compact_cache_refresh_interval`；
- `tools/eval_needle.py --am-compact-cache-refresh-interval`。

远端验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

8B 单样本 cache 对照（同 §45.6：`ctx=4096/depth=0.5/num_trials=1/max_output=4`）：

| selector | b | candidate_pool | cache_refresh | acc | throughput | raw 输出 |
|---|---:|---:|---:|---:|---:|---|
| OMP | 16 | 20 | 0 | 0.0% | 0.0173 tok/s | ` The magic number is` |
| OMP | 16 | 20 | 8 | 0.0% | 0.0476 tok/s | ` The magic is the` |
| OMP | 32 | 36 | 0 | 0.0% | 0.0051 tok/s | ` The magic number is` |
| OMP | 32 | 36 | 8 | 0.0% | 0.0147 tok/s | ` The magic is the` |

输出文件：

- `needle_sq_results/needle_am_omp_b16_ctx4096_d05_n1_cache8_smoke.json`
- `needle_sq_results/needle_am_omp_b32_ctx4096_d05_n1_cache8_smoke.json`

结论：decode-step cache 能把 OMP b=16/b=32 都提约 `2.8×`，证明“每 step 重拟合”确实是主要瓶颈之一；
但绝对吞吐仍远低于 HighestAttnKeys（约 `1.2 tok/s`）。下一步需要把 cache 从“按 decode step 复用”升级为
“prefill 后/长上下文阶段一次构建，按 layer/block/kv-head 持久复用”，并减少首次拟合里的 `lstsq/solve` 次数，否则 OMP 仍不具备跑 full grid 的工程意义。

### 45.8 AM-OMP prefill-build persistent compact cache 原型（2026-06-11）

承接 §45.7，本轮把 compact KV 构建前移到 prefill：当 `am_compact_cache_refresh_interval > 0` 且无 prefix-cache block table 时，
每层在 prefill 结束后用 prefill 阶段的 Q/K/V 参考样本构建 `C_k/beta/C_v`，decode 阶段按 block-table signature 复用。

新增实现：

- `build_attention_matching_prefill_cache()`：从 batched prefill Q/K/V 构建持久 compact cache；
- `am_prefill_cache_ref_query_stride` / `--am-prefill-cache-ref-query-stride`：控制 prefill reference query 采样密度；
- prefill 构建只在真实 KV cache 已分配时触发，避免 warmup 空 cache 误访问。

远端验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

8B 单样本 prefill-cache 对照（`ctx=4096/depth=0.5/num_trials=1`）：

| selector | b | candidate_pool | ref_stride | max_output | acc | throughput | raw 输出 |
|---|---:|---:|---:|---:|---:|---:|---|
| OMP | 16 | 20 | 16 | 4 | 0.0% | 0.0226 tok/s | ` The magic is the` |
| OMP | 16 | 20 | 256 | 4 | 0.0% | 0.0225 tok/s | ` The magic is the` |
| OMP | 16 | 20 | 256 | 16 | 0.0% | 0.0899 tok/s | ` The magic is the magic is ...` |
| OMP | 32 | 36 | 256 | 16 | 0.0% | 0.0276 tok/s | ` The magic is the magic is ...` |

输出文件：

- `needle_sq_results/needle_am_omp_b16_ctx4096_d05_n1_prefillcache_smoke.json`
- `needle_sq_results/needle_am_omp_b16_ctx4096_d05_n1_prefillcache_s256_smoke.json`
- `needle_sq_results/needle_am_omp_b16_ctx4096_d05_n1_prefillcache_s256_o16_smoke.json`
- `needle_sq_results/needle_am_omp_b32_ctx4096_d05_n1_prefillcache_s256_o16_smoke.json`

结论：prefill-build persistent cache 证明了“decode 阶段复用 compact KV”可以随输出长度摊薄成本：b=16 在 `max_output=16`
达到 `0.0899 tok/s`，高于 §45.7 的 decode-step cache8（`0.0476 tok/s`，max_output=4）和无 cache（`0.0173 tok/s`）。
但质量明显不行，输出变成 `The magic is the ...` 复读；说明仅用稀疏 prefill reference query 构建全局 compact KV，会损失 decode-step query 自适应性。
下一步若继续 OMP-fast，应该做“prefill 阶段构建多个 query-cluster / block-local compact cache”，decode 按当前 query 选择最近 cluster，
而不是单一全局 compact cache。

### 45.9 AM-OMP multi-cluster compact KV bank 原型（2026-06-12）

承接 §45.8，本轮实现 query-cluster compact bank：prefill 阶段不再只为每个 `(seq, kv_head)` 构建一份
`C_k/beta/C_v`，而是先对 reference queries 做轻量 deterministic k-means，再为每个 query cluster 分别拟合 compact KV；
decode 阶段用当前 query 到 centroid 的距离选择最近 cluster。

新增实现：

- `AttentionMatchingCacheEntry.compacts/centroids`：一个 cache entry 可持有多份 compact KV；
- `AttentionMatchingDecodeCache.get(..., query=...)`：cache hit 时按当前 query 路由 cluster；
- `build_attention_matching_prefill_cache(..., num_clusters=...)` 与 `attention_matching_decode(..., num_clusters=...)`；
- `Config.am_compact_num_clusters` / `tools/eval_needle.py --am-compact-num-clusters`；
- `Config.am_compact_route_top_k` / `--am-compact-route-top-k`：可把最近的多个 cluster 做 beta-shift ensemble，
  避免硬路由只选一个 compact bank；
- 单测覆盖 multi-cluster bank 构建、cluster routing、top-k ensemble、cache hit/miss 计数。

远端验证（`sitian@10.232.195.203`，`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`）：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

链路 smoke：

| model | ctx | selector | b | candidate_pool | clusters | max_output | acc | throughput | 输出文件 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | 512 | OMP | 2 | 4 | 1 | 2 | 0.0% | 0.78 tok/s | `needle_sq_results/needle_am_omp_clusters1_smoke.json` |
| Qwen3-0.6B | 512 | OMP | 2 | 4 | 2 | 2 | 0.0% | 0.35 tok/s | `needle_sq_results/needle_am_omp_clusters2_smoke.json` |
| Qwen3-0.6B | 512 | OMP | 2 | 4 | 2 | 8 | 0.0% | 0.72 tok/s | `needle_sq_results/needle_am_omp_clusters2_out8_smoke.json` |
| Qwen3-8B | 512 | OMP | 2 | 4 | 2 | 1 | 0.0% | 0.14 tok/s | `needle_sq_results/needle_8b_am_omp_clusters2_smoke.json` |

8B `ctx=4096/depth=0.5/num_trials=1/max_output=16/b=16/ref_stride=256` cluster 对比（同一 magic=60494）：

| clusters | route_top_k | acc | throughput | raw 输出 |
|---:|---:|---:|---:|---|
| 1 | 1 | 0.0% | 0.0960 tok/s | `The magic is the magic\nThe magic number is the magic...` |
| 2 | 1 | 0.0% | 0.0473 tok/s | `The grass magicass\n\nThe grass The grass...` |
| 4 | 1 | 0.0% | 0.0253 tok/s | `The answer:  is the answer is the answer...` |
| 4 | 2 | 0.0% | 0.0243 tok/s | `The answer: The answer: The answer...` |

输出文件：

- `needle_sq_results/needle_8b_am_omp_b16_ctx4096_c1_o16.json`
- `needle_sq_results/needle_8b_am_omp_b16_ctx4096_c2_o16.json`
- `needle_sq_results/needle_8b_am_omp_b16_ctx4096_c4_o16.json`
- `needle_sq_results/needle_8b_am_omp_b16_ctx4096_c4_rtop2_o16.json`

结论：multi-cluster bank 的端到端链路已打通，并且 8B 最小 smoke 可运行；但当前实现仍是 Python eager + 每个 cluster
独立 OMP/least-squares 拟合，`ctx=4096/b=16` 下吞吐基本随 cluster 数线性下降。质量上 hard routing 和 top-k ensemble
都没有召回 magic number，只是把单 bank 的 `The magic is...` 复读变成其他局部模式复读。说明仅按 prefill query cluster
做全局 compact KV 仍不足以恢复 needle 召回；下一步不应继续扩大 cluster 数，而应转向 block/span-local compact bank 或
query-conditioned 候选池复用，让中间 needle span 有机会进入 compact basis，同时把 `beta+C_v` fitting 做 layer-batched 化。

### 45.10 AM-OMP span-local compact bank 原型（2026-06-12）

承接 §45.9 的 negative result，本轮不再继续增加 query cluster，而是做 key span-local bank：把历史 KV 按连续 token span
切分，每个 span 单独拟合一份 compact KV，decode 时通过 `route_top_k` 把多个 span compact concat 成一个局部 ensemble。
这样至少能保证中间 span 也有自己的 compact basis，不会被全局 Highest/OMP 直接挤掉。

新增实现：

- `_build_span_local_compact_bank()`：按连续 key span 构建 compact bank，并把 span-local indices 还原成全局位置；
- `build_attention_matching_prefill_cache(..., num_key_spans=...)` / `attention_matching_decode(..., num_key_spans=...)`；
- `Config.am_compact_num_key_spans` / `tools/eval_needle.py --am-compact-num-key-spans`；
- 单测覆盖 span-local bank 构建、每个 span 的 indices 覆盖、decode ensemble 命中。

远端验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

8B `ctx=4096/depth=0.5/num_trials=1/max_output=16/ref_stride=256/spans=4/route_top_k=4` smoke（同一 magic=60494）：

| selector | per-span b | candidate_pool | total compact keys | acc | throughput | raw 输出 |
|---|---:|---:|---:|---:|---:|---|
| OMP | 4 | 8 | 16 | 0.0% | 0.1602 tok/s | `The grass is must\n Spin...` |
| OMP | 8 | 12 | 32 | 0.0% | 0.0665 tok/s | `The grass is a\n\n. The grass is the grass...` |

输出文件：

- `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_o16.json`
- `needle_sq_results/needle_8b_am_omp_b8_ctx4096_s4_rtop4_o16.json`

结论：span-local bank 相比 query-cluster c4 的吞吐更好（例如 total compact keys=16 时 `0.1602 tok/s`，高于 c1 全局 b16 的
`0.0960 tok/s`），因为每个 OMP 只在较短 span 内拟合；但质量仍未召回 magic number，输出变成 haystack 模式复读。
这说明“覆盖 needle 所在 span”仍不够，compact value fitting 在只用通用 prefill query refs 时没有学到“回答数字”的 decode query。
下一步应转向 query-conditioned refresh：prefill 只缓存 span/candidate basis，decode 首步按当前 query 轻量重拟合 beta/C_v，
或至少对候选 span 做 current-query top-k refinement，而不是完全复用 prefill fitted `C_v`。

### 45.11 AM-OMP query-conditioned decode refit（2026-06-12）

承接 §45.10，本轮验证“prefill 只缓存 selected basis，decode 用当前 query 重拟合 `beta/C_v`”是否能恢复质量。
实现上不重新跑 OMP selector，而是复用 cache 里的 `indices`：cache hit 后用当前 decode query 对这些 selected keys 重新调用
`fit_attention_bias()` 与 `fit_compacted_values()`，再做 compact attention 输出。

新增实现：

- `_refit_compact_for_queries()`：给定 cached `indices` 和当前 query，重拟合 `beta/C_v`；
- `attention_matching_decode(..., decode_refit=True)`；
- `Config.am_compact_decode_refit` / `tools/eval_needle.py --am-compact-decode-refit`；
- 单测 `test_decode_refit_recomputes_cached_values_for_current_query()` 校验 cache hit 后输出等于手动 refit 结果。

远端验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

8B `ctx=4096/depth={0.25,0.5,0.75}/num_trials=1/max_output=16/ref_stride=256` 对比：

| mode | b | spans | route_top_k | decode_refit | acc | throughput | raw 输出 |
|---|---:|---:|---:|---|---:|---:|---|
| span-local OMP | 4 | 4 | 4 | off | 0.0% | 0.1602 tok/s | `The grass is must...` |
| span-local OMP | 4 | 4 | 4 | on | 100.0% | 0.2326 tok/s | `The magic number is XXXXX. Remember it.` |
| global OMP | 16 | 1 | 1 | on | 100.0% | 0.1385 tok/s | `The magic number is XXXXX. Remember it.` |

输出文件：

- `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_o16.json`
- `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_depth3_o16.json`
- `needle_sq_results/needle_8b_am_omp_b16_ctx4096_refit_depth3_o16.json`

三深度明细（span-local b=4/spans=4/route_top_k=4/refit）：

| depth | magic | answer | hit |
|---:|---:|---:|---|
| 0.25 | 60494 | 60494 | yes |
| 0.50 | 65125 | 65125 | yes |
| 0.75 | 15306 | 15306 | yes |

结论：decode-time query-conditioned refit 是当前 AM-OMP-fast 最关键的质量修复。此前 prefill-fitted `C_v` 导致复读；只要保留
selected basis、在 decode 用当前 query 重拟合 `beta/C_v`，8B fixed-prompt 4096 三个深度都能正确输出 magic number。
吞吐上 span-local b=4/spans=4/refit 达到 `0.2326 tok/s`，比 global b16/refit 的 `0.1385 tok/s` 更好，也高于无 refit 的
span-local b=4 单样本 `0.1602 tok/s`（批量三条时摊薄 warmup/构建成本）。下一步应该把 refit 的 least-squares 路径批量化，
并跑 `ctx=8192/15000` 与 `b/spans` sweep，确认长上下文下 selected basis 是否仍覆盖 needle。

### 45.12 AM-OMP refit 长上下文 smoke（2026-06-12）

承接 §45.11，本轮固定当前最有希望的执行形态：`span-local OMP, per-span b=4, spans=4, route_top_k=4,
decode_refit=on`，向 8K/15K 上下文扩展。除 15K 为避免一次 batch 过大设置 `max_num_seqs=1` 外，其他配置沿用
§45.11：`depth={0.25,0.5,0.75}, num_trials=1, max_output=16`。

远端验证前仍先跑：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

长上下文结果：

| ctx | depths | acc | throughput | raw 输出形态 | 输出文件 |
|---:|---|---:|---:|---|---|
| 4096 | 0.25/0.5/0.75 | 100.0% | 0.2326 tok/s | `The magic number is XXXXX. Remember it.` | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_depth3_o16.json` |
| 8192 | 0.25/0.5/0.75 | 100.0% | 0.1565 tok/s | `The magic number is XXXXX. Remember it.` | `needle_sq_results/needle_8b_am_omp_b4_ctx8192_s4_rtop4_refit_depth3_o16.json` |
| 15000 | 0.25/0.5/0.75 | 100.0% | 0.2801 tok/s | `The magic number is XXXXX...` | `needle_sq_results/needle_8b_am_omp_b4_ctx15000_s4_rtop4_refit_depth3_o16.json` |

15K 三深度明细：

| depth | magic | answer | hit | raw 摘要 |
|---:|---:|---:|---|---|
| 0.25 | 60494 | 60494 | yes | `The magic number is 60494. The magic number is ...` |
| 0.50 | 65125 | 65125 | yes | `The magic number is 65125. Remember it...` |
| 0.75 | 15306 | 15306 | yes | `The magic number is 15306. Remember it.` |

结论：span-local selected basis + decode query-conditioned refit 在 4K/8K/15K 三个上下文长度的单样本三深度 smoke 上都能召回
needle。15K 的吞吐高于 8K 主要受调度/输出长度/`max_num_seqs` 差异影响，不应做横向性能结论；质量结论更重要：
selected basis 在长上下文下仍覆盖 needle span。下一步应做两类工作：

1. 质量 sweep：`ctx={4096,8192,15000}, depths=5, n>=2`，对比 `spans=4/8`、`per-span b=2/4/8`；
2. 性能优化：把 decode refit 中每层/每 head 的 `fit_attention_bias + fit_compacted_values` 批量化，否则 full-grid 吞吐仍会被 Python eager + small solve 限制。

### 45.13 AM-OMP refit 性能拆解与 fast small-solve（2026-06-12）

§45.12 的质量已经说明 refit 方向正确，但吞吐仍很低。本轮先拆最明显的热点：`decode_refit=on` 时虽然复用 selected
indices，不再重跑 OMP selector，但 `fit_compacted_values()` 为了重拟合 `C_v` 会先算一次 full attention target
`attention_output(queries, keys, values)`；随后还要做每层/每 KV head 的小矩阵 solve。因此速度慢不是 query routing，而是
“每步 full target + 小 solve + Python eager loop”。

新增轻量实验入口：

- `Config.am_compact_decode_refit_mode` / `--am-compact-decode-refit-mode {full,direct,beta}`；
- `full`：原质量路径，用 full target 重拟合 `beta/C_v`；
- `direct`：只重拟合 beta，`C_v = V[selected]`，不算 full target；
- `beta`：只重拟合 beta，复用 prefill fitted `C_v`。

同时做了一个不改变 `full` 语义的低秩 solve 优化：当 `num_queries < compact_keys` 时，

```text
(A^T A + λI)^-1 A^T y = A^T (A A^T + λI)^-1 y
```

把 decode refit 中常见的 `16x16` ridge solve 改成 `4x4` solve；`solve_box_nnls()` 里的 underdetermined 小矩阵也避免
SVD-based `lstsq`，直接走低秩 solve + clamp。

远端验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

8B `ctx=4096/depth={0.25,0.5,0.75}/b=4/spans=4/route_top_k=4/max_output=16` 对比：

| refit_mode | acc | throughput | raw 输出结论 |
|---|---:|---:|---|
| full（旧 solve） | 100.0% | 0.2326 tok/s | 正确输出 magic number |
| direct | 0.0% | 0.2626 tok/s | 无数字 / 模式复读 |
| beta | 0.0% | 0.2499 tok/s | 胡乱 token / 模式复读 |
| full（fast small-solve） | 100.0% | 0.2467 tok/s | 正确输出 magic number |

输出文件：

- `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_direct_depth3_o16.json`
- `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_beta_depth3_o16.json`
- `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_full_fastsolve_depth3_o16.json`

结论：

- 不能简单省掉 `C_v` refit：`direct/beta` 虽然略快，但质量直接掉到 0%。
- fast small-solve 保住质量，吞吐从 `0.2326` 到 `0.2467 tok/s`，只有约 `6%` 提升，说明主要瓶颈不是 solve 维度，
  而是每 step 每层每 head 仍在算 full attention target 与 Python eager 循环。
- 下一步真正有意义的性能优化应是结构性改造：
  1. 把所有 KV head 的 `full target + beta/C_v refit` 做 layer-batched；
  2. 尽量复用一次 dense/flash attention target，而不是在 AM Python loop 内按 KV head 重算；
  3. 进一步把 refit interval 降频（例如每 2/4 token refit 一次，中间只更新 beta 或直接复用），验证质量/速度拐点。

### 45.14 AM-OMP decode refit interval 降频实验（2026-06-12）

承接 §45.13 的第三点，本轮把 `decode_refit` 结果缓存起来，新增：

- `Config.am_compact_decode_refit_interval`；
- `Context.am_compact_decode_refit_interval`；
- `--am-compact-decode-refit-interval N`；
- `AttentionMatchingDecodeCache.refit_entries/get_refit/put_refit`；
- 单测 `test_decode_refit_interval_reuses_refitted_compact_values()`，验证 interval 内第二次 decode 会命中 refitted compact。

验证入口仍先跑：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

远端结果：`attention matching tests passed`。

8B smoke 固定 §45.13 的配置：`ctx=4096/depth={0.25,0.5,0.75}/num_trials=1/b=4/spans=4/route_top_k=4/max_output=16`，只改
`decode_refit_interval`。

| decode_refit_interval | acc | throughput | time | raw 输出形态 | 输出文件 |
|---:|---:|---:|---:|---|---|
| 1（fast small-solve baseline） | 100.0% | 0.2467 tok/s | - | 正确输出 magic number | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_full_fastsolve_depth3_o16.json` |
| 2 | 0.0% | 0.2542 tok/s | 188.9s | `The magic number is ...` 后接无数字/异常 token | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_i2_depth3_o16.json` |
| 4 | 0.0% | 0.2515 tok/s | 190.9s | `The magic number:` 后接非答案 token | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_i4_depth3_o16.json` |
| 8 | 0.0% | 0.2658 tok/s | 180.6s | `The magic number:` 后接非答案 token/复读 | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_i8_depth3_o16.json` |

三组 interval 的 magic 固定为 `60494/65125/15306`，均未提取到正确 answer。典型 raw：

- `interval=2`: `The magic number is  the $__$$__$...`
- `interval=4`: `The magic number: number is Wh m.sup.2number...`
- `interval=8`: `The magic numberachtsachtsies...`

结论：不能靠“每 N token 复用一次 refitted C_v”来保质量。needle 任务中生成数字的每个 decode token 对 query-conditioned
`C_v` 都很敏感；即使 `interval=2`，第二个 token 起复用上一 token 的 `C_v/beta` 就会破坏输出链路。吞吐也只从 `0.2467`
小幅到 `0.25~0.27 tok/s`，收益不足以抵消质量坍塌。

因此 `decode_refit_interval` 只能保留为实验开关，默认必须是 `1`。下一步性能优化不应再沿 refit 降频做，而应转向：

1. layer 内 KV-head batched refit，减少 Python eager loop；
2. 一次性计算/复用 dense full-attention target，避免每 KV head 重算 target；
3. 进一步把 `fit_attention_bias` 和 `fit_compacted_values` 合并成 batch solve，保留每 token full refit 的质量语义。

### 45.15 AM-OMP layer 内 batch refit 原型（2026-06-12）

承接 §45.14，继续保留 `decode_refit_interval=1` 的质量语义，不再降频；改为把同一层内可同形状的
`(batch row, KV head)` refit miss 收集起来批量求解。实现点：

- 新增 `_refit_compacts_for_query_groups_batched()`：输入 `keys/values=[N,L,D]`、`queries=[N,R,D]`、`selected=[N,M]`，
  批量求 `beta` 和 `C_v`；
- `attention_matching_decode()` 先保留原 compact cache/route 逻辑，只把需要 full refit 的项放到 `pending_refits`，最后按
  `(L,R,M,D,dtype,device,mode)` 分桶 batch solve；
- 不改外部接口，不改默认配置；不能 batch 的形状保留 scalar fallback；
- 新增 `test_decode_refit_batches_multiple_rows_and_kv_heads()`，覆盖 `B=2, KV heads=2, GQA group=2`，并和逐 head 手工
  `fit_attention_bias + fit_compacted_values` 对齐。

远端验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

8B smoke 仍固定 `ctx=4096/depth={0.25,0.5,0.75}/num_trials=1/b=4/spans=4/route_top_k=4/max_output=16`：

| 实现 | acc | throughput | time | raw 输出形态 | 输出文件 |
|---|---:|---:|---:|---|---|
| full refit + fast small-solve | 100.0% | 0.2467 tok/s | - | 正确输出 magic number | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_full_fastsolve_depth3_o16.json` |
| full refit + layer 内 batch refit | 100.0% | 0.2730 tok/s | 175.8s | `The magic number is XXXXX. Remember it.` | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batch_depth3_o16.json` |

三深度明细：

| depth | magic | answer | hit | raw 摘要 |
|---:|---:|---:|---|---|
| 0.25 | 60494 | 60494 | yes | `The magic number is 60494. Remember it.` |
| 0.50 | 65125 | 65125 | yes | `The magic number is 65125. Remember it.` |
| 0.75 | 15306 | 15306 | yes | `The magic number is 15306. Remember it.` |

结论：batch refit 保住了 §45.13 的 100% 质量，同时吞吐从 `0.2467` 提到 `0.2730 tok/s`，约 `+10.7%`。提升仍有限，说明
当前瓶颈还包括 full attention target 本身、KV cache dense gather/dequant、以及 attention layer 之间的 Python 调度。下一步应继续做：

1. 进一步把 batch refit 的 target 计算从每 KV head 的 `[L,D]` attention 改成 layer 级一次性 dense/flash target；
2. 如果保持 Python eager，至少把 `attention_output(q_group, compact.keys, compact.values, beta)` 也改成 batched output，减少最后一段小 matmul loop；
3. 跑 `ctx=8192/15000` batch refit smoke，确认长上下文质量不回退。

### 45.16 AM-OMP batched compact output 与长上下文复测（2026-06-14）

承接 §45.15，本轮继续做一个小的 eager loop 收敛：在 pending refit 已经按 bucket 完成 batch solve 后，把最后的 compact attention
输出也改成 batch matmul：

- 新增 `_attention_output_batched(queries, keys, values, beta)`，形状为 `queries=[N,R,D]`、`keys/values=[N,M,D]`、
  `beta=[N,M]`；
- `_process_pending_decode_refits()` 对同一 bucket 的 refitted compact 一次性计算输出，再逐项写回 `out`；
- scalar fallback 与 refit cache hit 路径保持不变，外部接口不变。

远端验证仍先跑完整编译和 AM 单测：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

4K 对比：

| 实现 | acc | throughput | time | 输出文件 |
|---|---:|---:|---:|---|
| batch refit（§45.15） | 100.0% | 0.2730 tok/s | 175.8s | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batch_depth3_o16.json` |
| batch refit + batched compact output | 100.0% | 0.2702 tok/s | 177.7s | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batchout_depth3_o16.json` |

这次 batch output 没带来稳定增益，说明最后的 compact 输出小 matmul 不是主要热点；0.2702 vs 0.2730 更像运行噪声。

随后复测长上下文，固定 `b=4/spans=4/route_top_k=4/decode_refit=on/interval=1/max_output=16/depth={0.25,0.5,0.75}`：

| ctx | acc | throughput | time | raw 输出形态 | 输出文件 |
|---:|---:|---:|---:|---|---|
| 4096 | 100.0% | 0.2702 tok/s | 177.7s | 正确输出 magic number | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batchout_depth3_o16.json` |
| 8192 | 100.0% | 0.1739 tok/s | 275.9s | 正确输出 magic number | `needle_sq_results/needle_8b_am_omp_b4_ctx8192_s4_rtop4_refit_batchout_depth3_o16.json` |
| 15000 | 100.0% | 0.3440 tok/s | 139.5s | 正确输出 magic number | `needle_sq_results/needle_8b_am_omp_b4_ctx15000_s4_rtop4_refit_batchout_depth3_o16.json` |

长上下文三深度明细：

| ctx | depth | magic | answer | hit | raw 摘要 |
|---:|---:|---:|---:|---|---|
| 8192 | 0.25 | 60494 | 60494 | yes | `The magic number is 60494. Remember it.` |
| 8192 | 0.50 | 65125 | 65125 | yes | `The magic number is 65125. Remember it.` |
| 8192 | 0.75 | 15306 | 15306 | yes | `The magic number is 15306. Remember it.` |
| 15000 | 0.25 | 60494 | 60494 | yes | `The magic number is 60494. The magic number is ...` |
| 15000 | 0.50 | 65125 | 65125 | yes | `The magic number is 65125. Remember it.` |
| 15000 | 0.75 | 15306 | 15306 | yes | `The magic number is 15306. Remember it.` |

结论：batch refit 路径在 4K/8K/15K 均保持 100% needle recall；batched compact output 本身没有明显速度收益。下一步真正的性能突破仍应
集中在更大的热点：

1. layer 级 full attention target 复用，避免每个 KV head 重算 `attention_output(q_group, k_seq, v_seq)`；
2. 减少 decode 前的 dense gather/dequant 成本；
3. 如果继续保持 AM-OMP selected basis，考虑把 refit 的 `selected gather + beta/C_v solve + compact output` 融成一个更少 Python dispatch 的层内 kernel/批处理块。

### 45.17 AM-OMP batched refit tensor-return 负结果（2026-06-14）

在 §45.16 之后，又尝试把 `_refit_compacts_for_query_groups_batched()` 从“返回 `list[AttentionMatchedKV]` 后再 stack 输出”改成直接返回
batched tensors，目标是减少 interval=1 时无用的逐项 compact 构造。实现上质量语义不变，但 4K smoke 结果反而变慢：

| 实现 | acc | throughput | time | 输出文件 |
|---|---:|---:|---:|---|
| batch refit + batched compact output（§45.16） | 100.0% | 0.2702 tok/s | 177.7s | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batchout_depth3_o16.json` |
| batched refit direct tensor return | 100.0% | 0.2575 tok/s | 186.4s | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batched_tensor_depth3_o16.json` |

三深度仍全部命中：`60494/65125/15306`。由于吞吐从 `0.2702` 降到 `0.2575 tok/s`，该实验已回滚到 §45.16 的实现，并重新验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

结论：继续微调 Python 对象构造已经没有稳定收益，甚至会因不同 tensor 生命周期/stack 路径导致波动。后续不应再在这个局部继续抠，应该转向：

1. KV cache dense gather/dequant 路径；
2. layer 级 target 计算复用；
3. 更大粒度的 fused/batched kernel。

### 45.18 AM-OMP decode CPU sync 清理（2026-06-14）

本轮排查发现两个不改变数学语义、但可能造成 decode 热路径 GPU→CPU 同步的位置：

1. `decode_refit_interval=1` 时仍无条件构造 `_refit_cache_key()`，其中会执行 `compact.indices.detach().cpu().tolist()`；但 interval=1 时
   `cache_refit_enabled=False`，这个 key 实际不会用于 get/put。
2. 每个 attention layer 都调用 `_am_cache_signatures(block_tables)`，内部是 `block_tables.detach().to("cpu").tolist()`；同一个 decode step 的
   `block_tables` 对所有层相同，应该在 `prepare_decode()` host 侧一次性构造。

改动：

- `attention_matching_decode()` 只在 `cache_refit_enabled=True` 时构造 `refit_key`；默认质量路径 `decode_refit_interval=1` 不再做
  `compact.indices.cpu().tolist()`；
- `Context` 增加 `am_compact_cache_signatures`；
- `ModelRunner.prepare_decode()` 直接从 host `block_table_rows` 生成 signatures；
- `Attention.forward()` 优先使用 `context.am_compact_cache_signatures`，缺失时才 fallback 到 `_am_cache_signatures(block_tables)`。

验证：

```bash
python -m py_compile tinyvllm/engine/attention_matching.py tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
python tools/test_attention_matching.py
```

结果：`attention matching tests passed`。

4K smoke 结果：

| 实现 | acc | throughput | time | 输出文件 |
|---|---:|---:|---:|---|
| §45.16 batch refit + batched compact output | 100.0% | 0.2702 tok/s | 177.7s | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_batchout_depth3_o16.json` |
| CPU sync 清理后 | 100.0% | 0.2644 tok/s | 181.5s | `needle_sq_results/needle_8b_am_omp_b4_ctx4096_s4_rtop4_refit_cpu_sync_depth3_o16.json` |

三深度仍全部命中：`60494/65125/15306`。吞吐没有稳定提升，`0.2644` 相比 `0.2702` 更像运行噪声范围内的轻微回落。
该改动仍保留：它删除了 interval=1 下确定无用的 GPU→CPU key 构造，并避免每层重复从 GPU block table 生成 signatures；即使 smoke
吞吐未显著改善，也能减少潜在 sync 点，代码语义更合理。

结论：CPU sync 清理不是当前 4K 单样本 smoke 的主瓶颈。至此，Python/eager 层面的安全小优化基本已经验证完。后续若继续提升，应直接进入
KV cache dense gather/dequant 或 block-table aware fused refit 方向。

### 45.19 20 tok/s 目标与 baseline 上限评估（2026-06-14）

用户明确提出速度至少需要提升到 `20+ tok/s`，即相对当前 AM-OMP `0.26~0.27 tok/s` 提升约 100 倍。先测不启用 AM 的 baseline
FlashAttention 路径，作为实际可达到的上限参考。配置保持 `depth={0.25,0.5,0.75}/num_trials=1/max_output=16`，只关闭 AM：

```bash
python tools/eval_needle.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --context-lens <ctx> --depths 0.25 0.5 0.75 --num-trials 1 \
  --max-output-len 16 --top-k-blocks-list -1 --gpu-memory-utilization 0.7
```

baseline 结果：

| ctx | acc | throughput | time | 输出文件 |
|---:|---:|---:|---:|---|
| 4096 | 100.0% | 31.0608 tok/s | 1.5s | `needle_sq_results/needle_8b_baseline_ctx4096_depth3_o16.json` |
| 8192 | 100.0% | 19.9901 tok/s | 2.4s | `needle_sq_results/needle_8b_baseline_ctx8192_depth3_o16.json` |
| 15000 | 100.0% | 8.5881 tok/s | 5.6s | `needle_sq_results/needle_8b_baseline_ctx15000_depth3_o16.json` |

对比当前 AM-OMP 最好路径：

| ctx | AM-OMP throughput | baseline throughput | 差距 |
|---:|---:|---:|---:|
| 4096 | 0.2644~0.2702 tok/s | 31.0608 tok/s | 约 115x |
| 8192 | 0.1739 tok/s | 19.9901 tok/s | 约 115x |
| 15000 | 0.3440 tok/s | 8.5881 tok/s | 约 25x |

结论：`20+ tok/s` 不是硬件/模型层面不可能，4K baseline 已经达到 `31 tok/s`，8K baseline 约 `20 tok/s`；但它不是 AM-OMP
路径，而是正常 FlashAttention。当前 AM-OMP/full decode refit 之所以慢两个数量级，是因为为了保质量，它每步每层仍要：

1. gather/dequant dense KV；
2. 计算 full QK scores 来拟合 beta 的 full attention mass；
3. 计算 full target `softmax(QK) @ V` 来拟合 `C_v`；
4. 做小 solve 和 compact output。

也就是说，AM-OMP/full-refit 在数学上已经接近“用 Python/eager 重新实现一遍 full attention target + 额外拟合”，不可能靠局部小优化追到
FlashAttention 的 `20+ tok/s`。要达到用户要求，有两条实际路线：

1. **产品可用路线**：对 8B/15K 这类质量已由 baseline 满足的场景，直接关闭 AM，走 FlashAttention baseline；这是目前唯一已验证能达到
   4K `31 tok/s`、8K `20 tok/s` 的路径。
2. **研究路线**：如果必须保留 AM-OMP selected basis，则需要重写为 block-table aware fused kernel，把 `full score/mass/target + selected gather + solve`
   尽量融合，至少避免 dense gather 和 Python eager。这个不是继续改 Python 函数能完成的 100x 提速，而是新的 CUDA/Triton 工程。

短期工程建议：保留当前 AM-OMP 作为质量/研究路径；默认推理不要开启 AM。后续若继续 AM 性能，应先写 profiler/benchmark，把 dense gather、full QK、PV target、solve
分段计时，再决定是否投入 fused kernel。

### 45.20 AM layer-wise 开关（2026-06-14）

用户提出“不是每一次 forward 阶段都压缩”的方向。由于 `decode_refit_interval>1` 已验证会让 needle recall 掉到 `0%`，不能降低每个启用 AM 层内部的 refit 频率；本轮改为降低**启用 AM 的层数**：未启用 AM 的层直接走原 baseline FlashAttention decode 路径，启用 AM 的层保持原 AM-OMP/full-refit 质量路径。

实现：

- `Config` 新增：
  - `am_compact_skip_first_layers`
  - `am_compact_skip_last_layers`
  - `am_compact_enable_layers`
  - `am_compact_layer_stride`
- `Context` 透传上述字段；
- `ModelRunner.allocate_kv_cache()` 给每个 `Attention` 注入 `layer_idx/num_hidden_layers`；
- `Attention.forward()` 通过 `_am_compact_layer_enabled()` 判断当前层是否启用 AM；
- `tools/eval_needle.py` 增加对应 CLI flags。

推荐 sweep 先从少量层开始找 100% recall 边界：

```bash
# 每隔 2 层开 AM：约 1/2 层数走 AM
python tools/eval_needle.py ... \
  --am-compact-blocks 4 --am-compact-selector omp \
  --am-compact-num-key-spans 4 --am-compact-route-top-k 4 \
  --am-compact-decode-refit --am-compact-layer-stride 2

# 跳过首尾，仅中间层走 AM
python tools/eval_needle.py ... \
  --am-compact-blocks 4 --am-compact-selector omp \
  --am-compact-num-key-spans 4 --am-compact-route-top-k 4 \
  --am-compact-decode-refit \
  --am-compact-skip-first-layers 8 --am-compact-skip-last-layers 8

# 显式只开若干层；该参数会覆盖 skip/stride
python tools/eval_needle.py ... \
  --am-compact-blocks 4 --am-compact-selector omp \
  --am-compact-num-key-spans 4 --am-compact-route-top-k 4 \
  --am-compact-decode-refit \
  --am-compact-enable-layers 8 12 16 20 24
```

验证：

```bash
python3 -m py_compile tinyvllm/config.py tinyvllm/utils/context.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/eval_needle.py tools/test_attention_matching.py
```

本机 `python3 tools/test_attention_matching.py` 因当前 Python 环境缺少 `torch` 未能运行；语法编译检查已通过。

## 46. Hidden-state reinjection / latent scratchpad smoke（2026-06-17）

用户提出的新方向是：如果 agent 模型主要是“自己操作”，那么内部通信单元不一定要是 token；token 是给人读的离散表示，hidden state 才更接近模型自己的连续表征。目标不是继续优化 AM 的 KV 压缩，而是验证 **hidden state 能否作为 latent scratchpad / agent 内部通信格式**，从而减少 token 化 CoT 的开销。

### 46.1 相关工作定位

这个想法不是空穴来风，和几类已有研究方向一致：

- **Coconut / Chain of Continuous Thought**：把部分 reasoning step 放在连续 latent space 中，而不是全部展开成自然语言 token。
- **Gist / soft prompt / prefix tuning 类方法**：用少量连续向量承载上下文压缩或任务状态，但通常需要训练。
- **Latent agent communication / hidden-state communication**：多 agent 不一定通过自然语言对话，可以传递连续状态；优点是高带宽、低冗余，缺点是不可解释且通常需要对齐训练。
- **本轮实验差异**：我们先不训练，只做 smoke test：把模型最后一层 hidden state 通过 projector 映射回 embedding 空间，再作为下一步 `input_embeds` 喂回模型，看是否立即数值坍塌。

### 46.2 代码改动

核心链路已经接通：

| 文件 | 改动 |
|---|---|
| `tinyvllm/models/qwen3.py:190` | `Qwen3Model.forward()` 增加 `input_embeds` 参数；`input_embeds is not None` 时绕过 `embed_tokens(input_ids)`，直接使用连续向量。 |
| `tinyvllm/models/qwen3.py:227` | `Qwen3ForCausalLM.forward()` 透传 `input_embeds`。 |
| `tinyvllm/engine/model_runner.py:555` | `run_model()` 增加 `input_embeds` 与 `return_hidden`；latent path / return hidden path 强制 eager，避免 CUDA graph capture 不兼容。 |
| `tools/eval_latent_reinjection.py:91` | projector 工厂：`identity`、`rmsnorm`、`linear`、`mlp`。当前均未训练。 |
| `tools/eval_latent_reinjection.py:252` | 主实验链路：prompt prefill → 取最后 hidden → 连续执行 K 个 latent `input_embeds` step → 再 greedy token decode。 |
| `tools/eval_latent_reinjection.py:320` | CLI 支持 `--task needle/arithmetic/tool_action`、`--latent-steps-list`、`--projectors`。 |

关键路径：

```text
prompt token prefill
  -> logits, hidden_states = run_model(..., return_hidden=True)
  -> hidden = last prompt hidden
  -> repeat K times:
       latent_embed = projector(hidden)
       append dummy token only for position/KV bookkeeping
       run_model(..., input_embeds=latent_embed, return_hidden=True)
       hidden = new hidden
  -> normal token decode
```

注意：latent step 仍然会写 KV cache，只是不经过 tokenizer / embedding lookup，也不产生人类可读 token。这里的 dummy token 只用于 `Sequence` 长度、position、block table 管理，并不会作为 embedding 使用。

### 46.3 Smoke 实验口径

远端环境：

```text
机器：sitian@10.232.195.203 / A100 80GB
Python：/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
模型：/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
GPU：CUDA_VISIBLE_DEVICES=1（GPU0 当时被占用，需避开）
```

任务：

1. `needle`：长上下文中插入 magic number，问最后数字。
2. `arithmetic`：简单多步整数运算。
3. `tool_action`：agent 场景里从 `LS/READ/GREP` 中选择下一步工具。

代表命令：

```bash
python tools/eval_latent_reinjection.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --task needle \
  --context-lens 4096 --depths 0.25 0.5 0.75 --num-trials 1 \
  --latent-steps-list 0 1 2 4 8 16 \
  --projectors identity rmsnorm \
  --max-model-len 4096 --gpu-memory-utilization 0.7 \
  --out-json needle_sq_results/latent_smoke_needle.json
```

### 46.4 结果

#### Needle 4K

| projector | K latent steps | accuracy | 结论 |
|---|---:|---:|---|
| RMSNorm | 0/1/2/4/8/16 | 100% | latent reinjection 不会立刻破坏已在 KV cache 中的检索信息。 |
| identity | 0/1/2 | 100% | 短 K 仍稳定。 |
| identity | 4 | 66.7% | 未归一化 hidden 直接回灌开始不稳。 |

文件留痕：

- `needle_sq_results/latent_smoke_needle.json`

#### Arithmetic

| projector / K | accuracy |
|---|---:|
| 全部配置 | 0% |

文件留痕：

- `needle_sq_results/latent_smoke_arithmetic.json`

#### Tool action

| projector / K | accuracy |
|---|---:|
| 全部配置 | 0% |

文件留痕：

- `needle_sq_results/latent_smoke_tool_action.json`

### 46.5 解释

当前结果支持两个判断：

1. **hidden-state reinjection 的数值通路成立**：`RMSNorm projector` 下，K=16 的 latent step 仍没有让 needle 检索崩掉，说明连续 hidden state 经 `input_embeds` 回灌后能维持模型基本动态，不是一步 NaN / 乱码。
2. **无训练 latent step 还不会“思考”**：needle 成功更可能是 prompt prefill 阶段已经把 magic number 写入 KV cache，latent step 只是没有破坏它；arithmetic/tool_action 0% 说明 untrained projector 不会自动学会把 latent step 当 CoT reasoning 使用。

换句话说：

```text
当前 smoke 证明的是：hidden state 可以作为稳定的内部传递格式。
还没证明的是：hidden state 可以在无训练情况下替代 token CoT 完成新 reasoning。
```

### 46.6 结论

- 用户的核心判断“token 是给人读的，hidden state 更像 AI 自己的语言”在架构上是合理的。
- 但模型原始训练目标仍是“hidden -> logits -> token”，它没有被训练成“hidden -> hidden reasoning step -> answer”的闭环。
- 因此下一步重点不是继续堆 K，而是训练/蒸馏 projector 或 latent transition，让 latent step 对齐 token CoT 的中间状态。

### 46.7 下一步计划

| 代号 | 实验 | 目标 | 判定标准 |
|---|---|---|---|
| A2 | harder latent stability | 扩大 K、ctx、任务难度，测 RMSNorm/MLP projector 是否长期稳定 | needle 不掉点；输出不复读/不崩。 |
| B1 | teacher CoT distillation | 用 teacher token CoT 监督 trainable projector / latent transition | arithmetic/tool_action 从 0% 明显上升。 |
| C1 | agent action toy | 构造小型 agent 状态转移任务，latent step 输出 tool/action | hidden 通信能否替代短自然语言 thought。 |
| D1 | latent token cache policy | 把 latent step 作为特殊 KV entry 管理，避免污染 prefix cache | 多请求/复用场景下结果稳定。 |

优先级建议：先做 **B1 teacher CoT distillation**。原因是 smoke 已说明通路稳定，瓶颈不是数值崩塌，而是没有训练信号；继续无训练扩 K 大概率只是在“保持/破坏 KV”之间摆动，不会凭空出现 reasoning。
