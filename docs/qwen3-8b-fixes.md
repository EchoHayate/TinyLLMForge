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


