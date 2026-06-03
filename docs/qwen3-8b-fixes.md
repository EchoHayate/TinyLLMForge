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
