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

## 47. B1 teacher CoT distillation 最小原型（2026-06-17）

### 47.1 目标

§46 的 smoke 已经证明 hidden-state reinjection 的数值链路可用，但 arithmetic / tool_action 在无训练 projector 下仍是 0%。因此本轮不再继续无训练堆 K，而是实现一个最小 B1 原型：

```text
prompt last hidden  -> trainable projector -> teacher-CoT hidden
```

核心假设：如果一个 latent token 要替代一段 token CoT，它应该对齐“prompt + teacher CoT prefix”之后的 hidden state，而不是只把 prompt hidden 原样回灌。

### 47.2 实现

新增：`tools/train_latent_projector.py`

关键设计：

1. **LLM 冻结，只收集 hidden targets**：
   - 对每条 synthetic case 收集 `prompt` 的最后 hidden，作为 source；
   - 再收集 `prompt + teacher_prefix` 的最后 hidden，作为 target；
   - 训练只对 projector 做 MSE / cosine loss，不反传进 LLM。
2. **projector 结构**：
   - `TrainableRMSLinearProjector = RMSNorm + Linear(hidden, hidden)`；
   - linear 用 identity 初始化，避免一开始偏离 §46 中表现最稳的 RMSNorm 路径太远。
3. **teacher prefix 构造**：
   - arithmetic：显式写出 `a+b`、乘法、减法步骤，最后以 `Final integer:` 结束；latent token 之后的可见 decode 应该直接输出整数。
   - tool_action：显式写出 “The best next tool is X.\nTool:”；latent token 之后的可见 decode 应该输出工具名。
4. **评测方式**：
   - 训练后直接复用 `tools/eval_latent_reinjection.py` 的 `generate_with_latent_steps()`；
   - 用训练后的 projector 做 1 个 latent step，再正常 greedy decode。

### 47.3 为什么不直接做端到端反传

当前 TinyLLMForge 推理路径是 inference-first：

- `ModelRunner.run_model()` 带 `@torch.inference_mode()`；
- latent eval 的 `generate_with_latent_steps()` 也带 `@torch.inference_mode()`；
- decode attention / KV cache 写入路径是推理式 direct-drive，不是训练图设计。

所以第一版不改推理内核，不做“loss -> model -> input_embeds -> projector”的端到端梯度，而是先做更小的 offline hidden-target distillation。这样可以先回答：**仅靠 teacher hidden 对齐，latent token 是否能从 0% 拉起来**。

### 47.4 运行命令

远端 8B arithmetic smoke 建议先跑小规模：

```bash
CUDA_VISIBLE_DEVICES=1 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --task arithmetic \
  --train-cases 64 --eval-cases 16 \
  --context-len 512 --depth 0.5 \
  --epochs 80 --batch-size 8 --lr 1e-4 \
  --latent-steps 1 --max-new-tokens 16 \
  --max-model-len 1024 --gpu-memory-utilization 0.7 \
  --checkpoint needle_sq_results/latent_projector_arithmetic.pt \
  --out-json needle_sq_results/latent_projector_arithmetic.json
```

如果 arithmetic 有提升，再跑 tool_action：

```bash
CUDA_VISIBLE_DEVICES=1 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --task tool_action \
  --train-cases 64 --eval-cases 16 \
  --context-len 512 --depth 0.5 \
  --epochs 80 --batch-size 8 --lr 1e-4 \
  --latent-steps 1 --max-new-tokens 8 \
  --max-model-len 1024 --gpu-memory-utilization 0.7 \
  --checkpoint needle_sq_results/latent_projector_tool_action.pt \
  --out-json needle_sq_results/latent_projector_tool_action.json
```

### 47.5 当前验证状态

本地和远端语法检查均通过：

```bash
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

第一轮远端 smoke 先暴露了一个 prompt 构造问题：如果像 §46 的 arithmetic/tool_action 那样把任务 core 插入 haystack 中间，decode 时 prompt 末尾其实是 haystack filler，模型会继续输出 `The sky is blue...`，评测没有意义。因此 B1 脚本已改为 **prefix-only filler**：haystack 只放在任务前面，保证最后一个 token 仍是 `Reasoning:`。

修正后跑 8B 小规模 arithmetic smoke：

```text
train_cases=8, eval_cases=4, context_len=512, epochs=3, latent_steps=1, max_new_tokens=64
```

结果：

| 指标 | 结果 |
|---|---:|
| train loss | 0.1418 → 0.0708 → 0.0598 |
| strict last-int accuracy | 25.0% |
| contains-expected accuracy | 100.0% |
| throughput | 5.82 tok/s |

四条 eval 都生成了完整算式，并包含正确 final answer：

| expected | 生成摘要 |
|---:|---|
| 116 | `58 + 14 = 72. 72 * 3 = 216. 216 - 100 = 116. The answer is 116...` |
| 85 | `23 + 10 = 33. 33 * 6 = 198. 198 - 113 = 85. Answer: 85...` |
| 423 | `84 + 51 = 135. 135 * 4 = 540. 540 - 117 = 423. The answer is 423...` |
| 63 | `33 + 58 = 91. 91 * 2 = 182. 182 - 119 = 63. The answer is 63...` |

文件留痕：

- `needle_sq_results/latent_projector_arithmetic_smoke.json`：旧 prompt 构造，0%，输出 haystack continuation。
- `needle_sq_results/latent_projector_arithmetic_prefix_smoke.json`：prefix-only prompt，16 token 输出已有算式但未完整到答案。
- `needle_sq_results/latent_projector_arithmetic_prefix_o64_final_smoke.json`：64 token 输出包含正确答案；旧 strict last-int metric 为 25%。

解释：这是一个正向信号，但还不是最终目标。projector 已经学会把 latent token 推向 teacher reasoning manifold，所以可见输出变成了正确 CoT；但它还没有把 CoT 压缩成“不可见 latent 思考后直接输出 final answer”。后续 metric 需要同时记录：

- `contains_expected`：latent 是否携带了正确任务信息；
- `strict answer-only`：是否真的完成隐藏 CoT 压缩。


### 47.6 判定标准

§46 的无训练 baseline 是：

| task | untrained latent projector |
|---|---:|
| arithmetic | 0% |
| tool_action | 0% |

B1 第一轮只要求证明趋势，不要求一次到位：

| 结果 | 解释 |
|---|---|
| eval accuracy 仍为 0% | hidden-target MSE 不足以让 latent token 替代 CoT；下一步需要端到端 logits distillation 或多 latent step target。 |
| accuracy 明显 >0% | 说明 teacher hidden 对齐能把 latent scratchpad 从“稳定但不会思考”推进到“带有任务信息”。 |
| train loss 下降但 eval 仍 0% | projector 学到了 hidden target 的几何相似性，但该 hidden 作为 `input_embeds` 回灌后未形成正确可见 token，需要把 loss 改到“latent step 后 logits 对齐 answer token”。 |

### 47.7 下一步

1. 远端先跑 arithmetic 小规模 smoke，和 §46 的 0% 对比。
2. 如果仍 0%，实现 B1.2：logits-level distillation，即对齐 latent step 后的 logits 到 teacher answer 首 token。
3. 如果 arithmetic 有提升，再扩大到 `train_cases=256/1024`，并测试 `latent_steps=2/4`。
4. tool_action 作为 agent 场景补验：如果 tool_action 能提升，说明 hidden-state 通信更接近 agent 内部决策格式。

### 47.8 B1.2 logits-level distillation 负结果（2026-06-17）

承接 §47.7，本轮把训练目标从纯 hidden target 扩展到 answer-token logits，希望把“可见 CoT”压成“latent 内部思考后直接输出 final answer”。

实现改动：`tools/train_latent_projector.py`

- `DistillCase.answer_token_id`：记录 expected answer 的首个 token id；注意这里必须用 `tokenizer.encode(str(expected))`，不能用带前导空格的 `" " + expected`，否则模型会先学输出空格/自然语言 continuation。
- `--hidden-loss-weight` / `--logit-loss-weight`：支持 hidden MSE/cosine 与 LMHead CE 的加权组合。
- eval 同时记录：
  - `accuracy`：answer-only 命中，即输出开头就是 expected；
  - `contains_accuracy`：输出中是否包含 expected。

#### 47.8.1 direct LMHead logits loss

先不反传穿过 transformer，只让 projector 输出在 LMHead 上直接预测 answer 首 token：

```text
logits = lm_head(projector(prompt_hidden))
loss = CE(logits, first_answer_token)
```

远端 8B smoke：

```text
train_cases=8, eval_cases=4, epochs=20, train_device=cuda
```

结果：

| setting | hidden_loss_weight | logit_loss_weight | answer-only acc | contains acc | 现象 |
|---|---:|---:|---:|---:|---|
| logits 主导 | 0.1 | 1.0 | 0% | 0% | 输出 `the answer is 100/127/...`，答案错误 |
| hidden+logits 混合 | 1.0 | 0.1 | 0% | 0% | 输出回到泛化 reasoning 开头，但不含正确答案 |

输出文件：

- `needle_sq_results/latent_projector_arithmetic_b12_logits_smoke.json`：前导空格 target，0%。
- `needle_sq_results/latent_projector_arithmetic_b12_logits_nospace_smoke.json`：无前导空格 target，0%。
- `needle_sq_results/latent_projector_arithmetic_b12_h1_l01_smoke.json`：hidden/logit 混合，0%。

结论：**直接 LMHead logits loss 不等价于 latent-step logits loss**。它训练的是“projector 输出作为最终 hidden 时能预测 answer”，但部署时 projector 输出会作为 `input_embeds` 再过一遍 transformer；这个空间错位会破坏答案，甚至比 §47.5 的 hidden-only 更差。

#### 47.8.2 true one-step logits loss 尝试

为了让训练目标匹配部署路径，进一步尝试：

```text
prompt prefill KV 固定
projector(prompt_hidden) -> input_embeds
frozen transformer decode 1 step
CE(next-token logits, first_answer_token)
```

这个方向理论上更正确，但当前 TinyLLMForge 推理内核不是训练图设计，实际被两个问题卡住：

1. `prepare_decode()` 复用的 host/GPU staging buffer 可能是在 inference mode 中创建的；离开 inference mode 后再 in-place copy 会触发：

```text
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed
```

2. 把 `prepare_decode()` 包回 inference mode 后，继续反传 frozen transformer，又遇到 decoder 内部 in-place hidden/residual 更新导致 autograd version mismatch：

```text
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

因此 `--soft-step-epochs` 只作为实验入口保留；当前不能作为可用训练路径。要做真正的 B1.2，需要为 latent one-step training 单独实现一个 autograd-safe forward：不复用 inference staging buffer、不写 KV cache，或者复制一条 HuggingFace/torch 原生 forward 路径做 teacher forcing。

### 47.9 当前结论

- §47.5 的 hidden-only distillation 是目前最有效的正向信号：`contains_expected=100%`，说明 latent token 已经携带了正确 reasoning 信息，但仍会把 CoT 显式吐出来。
- §47.8 的 direct logits loss 没有把输出压成 answer-only，反而破坏了正确 CoT。
- 真正的 “latent 内部思考后直接输出 final answer” 不能只靠 LMHead loss；需要训练目标穿过 **同一条 latent input_embeds -> transformer -> logits** 路径。

下一步建议改为 **B1.3 autograd-safe latent training path**：

1. 不走 TinyLLMForge KV-cache inference decode；单独构造一个小 batch 的 `inputs_embeds` teacher-forcing forward。
2. 冻结 LLM，只训练 projector。
3. loss 直接对齐 answer-only token 序列，而不是只对齐首 token。
4. 先在 Qwen3-0.6B 上验证 answer-only，再迁移到 8B。

### 47.10 B1.3 HF teacher-forcing 路径（2026-06-17）

已按 §47.9 落地一条 autograd-safe 训练路径：

```text
prompt -> HF forward(output_hidden_states=True) -> prompt_last_hidden
prompt embeddings + projector(prompt_last_hidden) + answer_prefix embeddings
  -> frozen HF CausalLM(inputs_embeds=..., attention_mask=..., use_cache=False)
  -> CE(answer-only token sequence)
```

实现位置：`tools/train_latent_projector.py`

关键点：

- `--hf-teacher-forcing`：切到 HuggingFace 原生 forward，不走 TinyLLMForge KV-cache inference decode。
- `collect_source_hidden_hf()`：批量收集 prompt 最后一个非 padding token 的 final hidden。
- `_build_hf_teacher_batch()`：构造 `inputs_embeds = prompt_embeds + latent_embed + answer_prefix_embeds`，并把 causal logits 对齐到完整 answer token 序列。
- `train_projector_hf_teacher_forcing()`：冻结 LLM，只更新 `TrainableRMSLinearProjector`。
- `generate_hf_with_latent()`：评测时同样先插入 1 个 latent embed，再 greedy decode 可见 answer。

第一轮 Qwen3-0.6B smoke：

```text
train_cases=64, eval_cases=16, epochs=50, batch_size=8, task=arithmetic
loss: 6.14 -> 1.91
eval: answer-only acc=0%, contains acc=0%
现象：输出类似 "1414424"、"6424"、"14724532"，没有形成稳定 answer-only。
```

这个负结果不是 autograd 链路失败：loss 能下降，说明梯度确实穿过了 `inputs_embeds -> frozen transformer -> logits -> projector`。更可能的问题是训练/评测口径仍有错位：

1. **padding side 错位**：部分 decoder-only tokenizer 默认 left padding，而训练 source hidden 是 batched/padded，eval source hidden 是 unpadded；如果直接用 `attention_mask.sum()-1` 取最后 hidden，left padding 时会取错 token。
2. **answer 序列没有终止监督**：只监督数字 token 会让模型在 greedy decode 时继续补数字，导致 `1414424` 这类连续数字串；这不是严格的 answer-only 格式。

已修正：

- 新增 `_last_nonpad_indices()`，用 rightmost non-padding index，兼容 left/right padding。
- HF teacher-forcing mode 强制 `tokenizer.padding_side = "right"`，与手写 `inputs_embeds` padding 保持一致。
- `_answer_token_ids()` 默认把 `eos_token_id` 追加到 answer target，训练目标变成 `answer + EOS`，避免 answer 后继续生成数字。
- 新增 `_patch_torch_custom_op_string_annotations()`，兼容远端 torch 2.4.x 与较新 Transformers custom-op 字符串 annotation 的 schema inference 冲突。

修正后已完成远端 sanity smoke：

```text
Qwen3-0.6B, train_cases=2, context_len=64, epochs=1, batch_size=1, skip_eval
结果：模型加载、HF inputs_embeds teacher-forcing forward、loss.backward()、checkpoint/json 保存均成功。
```

后续复跑建议先用同一组参数验证修正是否消除数字串；若仍为 0%，再扩大到 Qwen3-8B 或增加 train_cases。当前优先判定标准仍是 `answer-only acc`，`contains acc` 只作为 latent 是否携带任务信息的辅助指标。

### 47.11 B1.3 修正版复跑结果（2026-06-18）

本轮继续验证 §47.10 的三个修正：right padding 对齐、rightmost non-padding hidden、answer+EOS 终止监督。

#### 47.11.1 Qwen3-0.6B arithmetic

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic, train_cases=64, eval_cases=16,
context_len=512, epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| teacher-forcing loss | 8.0815 -> 1.5030 |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 117.4s |

代表输出从上一轮的长数字串变成了较短 answer-like 数字，但仍是错误答案：`114`、`1146`、`144`、`1068` 等。说明 `answer+EOS` 终止监督确实减少了无限数字续写，但 **0.6B + 单 latent token + 线性 projector 仍没有学会 arithmetic reasoning 压缩**。

落盘文件：

- `needle_sq_results/latent_projector_arithmetic_b13_hf_fixed.json`
- `needle_sq_results/latent_projector_arithmetic_b13_hf_fixed.pt`

#### 47.11.2 Qwen3-0.6B tool_action

命令参数：

```text
model=Qwen3-0.6B, task=tool_action, train_cases=64, eval_cases=16,
context_len=512, epochs=50, batch_size=8, lr=1e-4, max_new_tokens=8
```

结果：

| 指标 | 结果 |
|---|---:|
| teacher-forcing loss | 11.8468 -> 0.0044 |
| answer-only acc | 100.0% |
| contains acc | 100.0% |
| elapsed | 116.3s |

全部 16 条 eval 都直接输出正确工具名：`READ` / `LS` / `GREP`，没有 visible CoT。这是当前最强正向结果：对于 agent action selection 这类低熵决策任务，**一个 latent token 可以承载内部决策并直接输出 action**。

落盘文件：

- `needle_sq_results/latent_projector_tool_action_b13_hf_fixed.json`
- `needle_sq_results/latent_projector_tool_action_b13_hf_fixed.pt`

#### 47.11.3 Qwen3-8B arithmetic smoke

命令参数：

```text
model=Qwen3-8B, task=arithmetic, train_cases=32, eval_cases=8,
context_len=512, epochs=20, batch_size=2, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| teacher-forcing loss | 4.6471 -> 1.1375 |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 125.0s |

代表输出：`124`、`127`、`177`、`125`、`174`，仍是 answer-like 但错误的整数。结论和 0.6B arithmetic 一致：单 latent token 的 answer-only 训练能学到“输出整数的格式”，但没有可靠承载多步算术的中间计算。

落盘文件：

- `needle_sq_results/latent_projector_arithmetic_b13_hf_8b_smoke.json`
- `needle_sq_results/latent_projector_arithmetic_b13_hf_8b_smoke.pt`

#### 47.11.4 当前判断

| 任务 | B1.3 修正版结果 | 解释 |
|---|---:|---|
| tool_action | 100% answer-only | hidden-state latent 通信对 agent 低熵动作选择是可行的。 |
| arithmetic | 0% answer-only | 单 latent token + 线性 projector 不足以压缩多步符号计算。 |

这说明用户的核心方向仍然成立，但需要区分任务类型：

1. **agent action / routing / tool choice**：可以优先推进，latent hidden state 比自然语言 CoT 更接近模型内部决策格式。
2. **多步 arithmetic / symbolic reasoning**：需要更强 latent scratchpad，例如多 latent token、非线性 projector/transition、多阶段 teacher CoT 对齐，或者先蒸馏到 visible CoT 再逐步减少可见 token。

下一步建议不再把 arithmetic 当唯一判据，而是拆两条线：

- **C1 agent-action latent policy**：扩大 tool/action 任务分布，加入更复杂 observation 和参数化 action，验证 hidden-state action channel 的泛化。
- **B1.4 multi-latent arithmetic**：把 `inputs_embeds = prompt + K 个 latent + answer_prefix`，训练 K=2/4/8 的 latent scratchpad，而不是强迫 1 个 latent token 承载完整计算。

### 47.12 C1 / B1.4 双线推进（2026-06-18）

承接 §47.11，本轮把 latent scratchpad 拆成两条路线同时验证：

1. **C1 agent-action latent policy**：扩大 tool/action 数据分布，测试 agent 内部 hidden channel 是否能泛化到更多 observation 模板。
2. **B1.4 multi-latent arithmetic**：把 HF teacher-forcing 从 1 个 latent token 扩展到 K 个 latent token，测试 K=2/4/8 是否改善 arithmetic。

#### 47.12.1 代码改动

文件：`tools/train_latent_projector.py`

- `build_tool_action_distill_cases()`：从固定 6 条 scenario 扩展为随机模板生成：
  - `READ`：已知文件路径，需要读取单文件内容；
  - `LS`：已知目录路径，需要列目录；
  - `GREP`：未知文件路径，需要跨文件搜索符号/字符串/日志模式；
  - 随机加入无关干扰句，避免模型只记固定表述。
- `_project_latent_sequence()`：新增 K-step latent 生成：

```text
hidden_0 = prompt_last_hidden
for step in 1..K:
    latent_step = projector(hidden_{step-1})
    hidden_step = latent_step
```

- `_build_hf_teacher_batch()`：训练输入从

```text
prompt_embeds + latent + answer_prefix
```

扩展为：

```text
prompt_embeds + latent_1 + ... + latent_K + answer_prefix
```

loss 对齐从最后一个 latent 位置开始：`logits[prompt_len + K - 1] -> answer_token_0`。

- `generate_hf_with_latent()` / `evaluate_hf_teacher_forcing()`：评测同样使用 `--latent-steps K`。

#### 47.12.2 C1 expanded tool_action 结果

命令参数：

```text
model=Qwen3-0.6B, task=tool_action, train_cases=128, eval_cases=48,
context_len=512, latent_steps=1, epochs=40, batch_size=8, lr=1e-4
```

结果：

| 指标 | 结果 |
|---|---:|
| teacher-forcing loss | 9.8890 -> 0.0018 |
| answer-only acc | 100.0% |
| contains acc | 100.0% |
| elapsed | 190.9s |

代表输出：`GREP`、`LS`、`READ`。48 条 eval 全部 answer-only 命中。少数 raw 会在正确首 token 后继续短句，例如 `GREP\nThe answer is: ...`，但 `answer_only` metric 仍命中；后续若要严格动作协议，可以把 eval 的 `max_new_tokens` 降到 1 或加入更强 EOS/stop 约束。

落盘文件：

- `needle_sq_results/latent_projector_c1_tool_action_expanded_k1.json`
- `needle_sq_results/latent_projector_c1_tool_action_expanded_k1.pt`

结论：C1 方向继续成立。对 agent tool/action 低熵决策，1 个 latent token 足够承载内部状态并输出动作，即使 observation 模板扩展后仍能泛化。

#### 47.12.3 B1.4 multi-latent arithmetic sweep

命令参数统一为：

```text
model=Qwen3-0.6B, task=arithmetic, train_cases=64, eval_cases=16,
context_len=512, epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

只改变 `latent_steps`：

| latent_steps | teacher-forcing loss | answer-only acc | contains acc | 代表输出 |
|---:|---:|---:|---:|---|
| 2 | 7.6872 -> 1.4314 | 0.0% | 0.0% | `628`, `692`, `277`, `621` |
| 4 | 8.3581 -> 1.4520 | 0.0% | 0.0% | `101`, `213`, `114`, `1014` |
| 8 | 8.0733 -> 1.5883 | 0.0% | 0.0% | `144`, `147` |

落盘文件：

- `needle_sq_results/latent_projector_b14_arithmetic_k2.json`
- `needle_sq_results/latent_projector_b14_arithmetic_k4.json`
- `needle_sq_results/latent_projector_b14_arithmetic_k8.json`
- 对应 `.pt` checkpoint 同名保存。

结论：仅把同一个线性 projector recurrently unroll 成 K 个 latent token，没有改善 arithmetic。它仍然学到“输出整数”的格式，但没有学到可泛化的多步计算过程。K 越大也没有单调收益，说明瓶颈不只是 latent token 数量，而是 latent transition 的训练目标/结构不够强。

#### 47.12.4 更新后的判断

| 路线 | 当前状态 | 下一步建议 |
|---|---|---|
| C1 agent-action latent policy | 正向，expanded tool_action 仍 100% | 扩 action schema：不仅输出工具名，还输出结构化 action 参数，如 `READ path=...` / `GREP pattern=...`。 |
| B1.4 multi-latent arithmetic | 负向，K=2/4/8 仍 0% | 不再只 unroll 同一个 projector；需要引入每步独立 latent transition、teacher CoT 分段监督或 curriculum。 |

目前最有价值的推进方向是 C1：把 hidden-state channel 用在 agent 内部动作决策上，而不是先死磕多步算术。Arithmetic 可作为长期 reasoning benchmark，但短期应该换更强训练形式：

```text
prompt -> latent_1 对齐 teacher step1 hidden
       -> latent_2 对齐 teacher step2 hidden
       -> ...
       -> answer
```

也就是 **multi-latent + per-step teacher CoT hidden supervision**，而不是只在最后 answer token 上做 CE。

### 47.13 C1.1 / B1.5 继续实验（2026-06-18）

承接 §47.12，本轮继续做两件事：

1. **C1.1 structured latent action**：从只输出工具名扩展到输出结构化 action：

```text
READ path=...
GREP pattern=...
LS path=...
```

2. **B1.5 per-step CoT hidden supervision**：不只在最终 answer token 上做 CE，同时让每个 latent_i 对齐 teacher CoT 第 i 步的 hidden state。

#### 47.13.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `tool_action_structured` task：
  - prompt 明确给出 action schema：`LS path=<directory>` / `READ path=<file>` / `GREP pattern=<query>`；
  - synthetic observation 随机生成 file path / directory path / grep pattern；
  - expected answer 是完整结构化动作，例如 `READ path=internal/auth/token.py`。
- structured action 评测：
  - `_extract_structured_action()` / `_normalize_structured_action()` 支持从 raw output 抽取并规范化结构化动作；
  - `answer_only` 要求输出第一行就是完整结构化 action。
- arithmetic case 新增 `meta["teacher_steps"]`：

```text
step1:  (a + b) = x;
step2:  x * c = y;
step3:  y - d = answer.
step4:  Final integer:
```

- 新增 `collect_step_teacher_hiddens_hf()`：批量收集 `prompt + teacher_steps[:i]` 的最后 hidden，作为 latent_i 的监督目标。
- `train_projector_hf_teacher_forcing()` 新增 `--step-hidden-loss-weight`：

```text
loss = CE(answer tokens) + weight * (MSE/RMSNorm + 0.1 * cosine_loss)(latent_i, teacher_step_i_hidden)
```

#### 47.13.2 C1.1 structured action 结果

命令参数：

```text
model=Qwen3-0.6B, task=tool_action_structured,
train_cases=128, eval_cases=48, context_len=512,
latent_steps=1, epochs=40, batch_size=8, lr=1e-4, max_new_tokens=12
```

结果：

| 指标 | 结果 |
|---|---:|
| CE loss | 4.6886 -> 0.0030 |
| answer-only acc | 95.8% |
| contains acc | 95.8% |
| elapsed | 202.1s |

代表输出：

```text
GREP pattern=UserService
GREP pattern=compute_score
GREP pattern=timeout_ms
READ path=internal/auth/token.py
LS path=needle_sq_results
```

结论：C1.1 明显正向。模型不仅能通过 1 个 latent token 输出工具名，还能携带一个参数槽位（path/pattern）。这比 §47.12 的 `READ/GREP/LS` 更接近真实 agent action channel。

落盘文件：

- `needle_sq_results/latent_projector_c11_structured_action_k1.json`
- `needle_sq_results/latent_projector_c11_structured_action_k1.pt`

#### 47.13.3 B1.5 per-step CoT hidden supervision 结果

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic,
train_cases=64, eval_cases=16, context_len=512,
latent_steps=4, step_hidden_loss_weight=0.5,
epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| total loss | 8.5416 -> 1.8777 |
| CE loss | 8.3481 -> 1.5715 |
| step hidden loss | 0.3874 -> 0.6125 |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 240.2s |

代表输出：`208`、`143`、`102` 等，仍是 answer-like 但错误的整数。

落盘文件：

- `needle_sq_results/latent_projector_b15_arithmetic_k4_step_hidden_w05.json`
- `needle_sq_results/latent_projector_b15_arithmetic_k4_step_hidden_w05.pt`

#### 47.13.4 结论

| 路线 | 结果 | 解释 |
|---|---:|---|
| C1.1 structured action | 95.8% | hidden-state action channel 能承载工具名 + 一个参数槽位。 |
| B1.5 per-step arithmetic | 0% | 简单 latent-output-to-teacher-hidden 对齐仍不够；step hidden loss 甚至没有稳定下降。 |

B1.5 的负结果说明：把 projector 输出直接对齐 teacher step hidden 不是充分条件。原因可能是：

1. teacher step hidden 是 transformer **输出空间**，但 latent token 作为下一步输入是 **embedding/input space**；二者仍有空间错位。
2. 当前只有一个共享线性 projector recurrently unroll，表达力不足。
3. hidden supervision 和 answer CE 的目标可能冲突：step hidden loss 没稳定下降，说明 latent 输入空间不容易直接拟合 teacher 输出 hidden。

下一步建议：

- C1.2：继续 structured action，加入两个参数槽位，例如 `GREP pattern=... path=...`，或 action JSON。
- B1.6：不要直接用 teacher output hidden 监督 latent input；改成 **每个 latent_i 过 frozen transformer 后的 output hidden** 对齐 teacher step hidden：

```text
latent_i as input_embeds
  -> frozen transformer
  -> output_hidden_i 对齐 teacher_step_i_hidden
```

这样对齐的是同一个空间：transformer output hidden vs teacher output hidden，而不是 input latent vs output hidden。

### 47.14 C1.2 / B1.6 继续实验（2026-06-18）

承接 §47.13，本轮继续验证两条路线：

1. **C1.2 structured action 双参数槽位**：在 `GREP` action 中加入第二个参数 `path`，目标 schema 变为：

```text
READ path=<file>
LS path=<directory>
GREP pattern=<query> path=<directory>
```

2. **B1.6 transformer output-hidden supervision**：不再把 latent input embedding 直接对齐 teacher step hidden，而是在 HF teacher-forcing forward 中取 frozen transformer 最后一层在 latent token 位置的 output hidden，再与 teacher step hidden 对齐：

```text
prompt -> latent_i input_embeds -> frozen transformer -> output_hidden_i
output_hidden_i ~= teacher_step_i_hidden
loss = CE(answer tokens) + weight * hidden_loss(output_hidden_i, teacher_step_i_hidden)
```

#### 47.14.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `tool_action_structured2` task：
  - `READ` / `LS` 仍为单 `path` 参数；
  - `GREP` 扩展为 `pattern + path` 两个槽位，例如 `GREP pattern=UserService path=services/api`；
  - structured action 解析/归一化支持 `GREP pattern=... path=...`。
- 修复 `_build_hf_teacher_batch()` 的 latent position 返回，供 output-hidden supervision 精确定位 latent token 的 transformer 输出位置。
- `train_projector_hf_teacher_forcing()` 新增 `--step-hidden-mode {input,output}`：
  - `input`：旧 B1.5 行为，直接对齐 latent input；
  - `output`：B1.6 行为，对齐 frozen transformer 的 latent output hidden。

本地检查：

```text
python3 -m py_compile tools/train_latent_projector.py
```

远端也同步后执行了同样的 `py_compile`。

#### 47.14.2 C1.2 双参数 structured action 结果

命令参数：

```text
model=Qwen3-0.6B, task=tool_action_structured2,
train_cases=128, eval_cases=48, context_len=512,
latent_steps=1, epochs=40, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| CE loss | 3.9997 -> 0.0025 |
| answer-only acc | 93.8% |
| contains acc | 93.8% |
| elapsed | 204.1s |

代表输出：

```text
GREP pattern=UserService path=.
READ path=README.md
GREP pattern=timeout_ms path=configs
GREP pattern=handler_fn path=.
LS path=internal/auth
```

主要错误样例：

```text
expected: GREP pattern=ERROR path=.
output:   GREP pattern=ERROR path=Observation

expected: GREP pattern=latent_steps path=/data/project
output:   GREP pattern=latent_steps path=data/project

expected: GREP pattern=latent_steps path=.
output:   GREP pattern=latent_steps path=directory
```

落盘文件：

- `needle_sq_results/latent_projector_c12_structured2_action_k1.json`
- `needle_sq_results/latent_projector_c12_structured2_action_k1.pt`

结论：C1.2 仍然正向。相比 C1.1 的一个参数槽位，双参数 `GREP pattern + path` 让任务难度上升，准确率从 95.8% 降到 93.8%，但仍远高于随机且输出基本保持 schema。失败主要集中在 `path=.`、绝对路径 slash、以及从 prompt 中误拷贝普通词作为 path。

#### 47.14.3 B1.6 output-hidden arithmetic 结果

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic,
train_cases=64, eval_cases=16, context_len=512,
latent_steps=4, step_hidden_loss_weight=0.5, step_hidden_mode=output,
epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| total loss | 8.5325 -> 1.6473 |
| CE loss | 8.3047 -> 1.4959 |
| step output-hidden loss | 0.4565 -> 0.3029 |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 239.6s |

代表输出：

```text
101
101
101
101
272
```

落盘文件：

- `needle_sq_results/latent_projector_b16_arithmetic_k4_step_output_hidden_w05.json`
- `needle_sq_results/latent_projector_b16_arithmetic_k4_step_output_hidden_w05.pt`

#### 47.14.4 更新后的判断

| 路线 | 结果 | 判断 |
|---|---:|---|
| C1.2 双参数 structured action | 93.8% | 继续成立；hidden-state channel 可以承载 agent action schema + 参数槽位。 |
| B1.6 output-hidden arithmetic | 0.0% | 仍为负；空间对齐改善了 hidden loss，但没有形成可泛化 arithmetic reasoning。 |

B1.6 相比 B1.5 有一个局部改善：step hidden loss 从 0.4565 降到 0.3029，说明 **output-hidden 对齐确实比 input-hidden 对齐更合理**。但它仍没有解决 arithmetic 泛化，模型最后退化为少数固定整数（主要是 `101`）。

当前结论：

1. hidden-state latent channel 用于 **低熵 agent action / tool policy** 是可行方向；从工具名到一个参数、再到两个参数槽位都能训练出较高泛化准确率。
2. arithmetic 这类多步离散计算不能靠“共享线性 projector recurrent unroll + hidden MSE + answer CE”解决；即使 output-hidden supervision 也不够。
3. 下一步如果继续 B 线，应换结构而不是继续调同一个 projector：例如 per-step 独立 transition、latent token type embedding、curriculum、或把 latent policy 训练成显式 state machine。

### 47.15 B1.7 stepwise latent transition + curriculum（2026-06-18）

承接 §47.14 的判断，本轮不再继续调共享线性 projector，而是换结构做一个最小 B1.7：

```text
prompt_hidden
  -> latent_1 = transition_1(prompt_hidden + step_embed_1)
  -> latent_2 = transition_2(output_or_hidden_1 + step_embed_2)
  -> latent_3 = transition_3(output_or_hidden_2 + step_embed_3)
  -> latent_4 = transition_4(output_or_hidden_3 + step_embed_4)
  -> answer-only CE
```

实际实现仍沿用当前 teacher-forcing pipeline 的 recurrent latent 生成路径，但把共享 projector 换成：

1. **per-step transition**：每个 latent step 一个独立 `RMSNorm + Linear` transition。
2. **latent step embedding**：每一步都有一个可学习的 step embedding，加到该步 transition 输入上。
3. **linear curriculum**：训练早期只用 1 个 latent step，然后逐步扩到 2/3/4 个 latent steps。
4. **output-hidden supervision**：继续使用 B1.6 的 `step_hidden_mode=output`，即对齐 transformer output hidden，而不是 latent input。

#### 47.15.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `StepwiseRMSLinearProjector`：

```text
step_embed[step_idx] + hidden -> per-step RMSNorm+Linear transition
```

- 新增 `build_trainable_projector(hidden_size, kind, max_steps)`：
  - `--projector-kind shared`：旧共享 projector；
  - `--projector-kind stepwise`：B1.7 per-step projector。
- `_project_latent_sequence()` 支持 `projector.project_step(hidden, step_idx)`。
- `train_projector_hf_teacher_forcing()` 新增 `--latent-step-curriculum {none,linear}`：
  - `none`：全程使用 `--latent-steps`；
  - `linear`：按 epoch 线性从 1 step 增长到目标 `latent_steps`。
- 新增 `--max-latent-steps`，用于 stepwise projector 预分配 transition/step embedding。

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
```

#### 47.15.2 B1.7 arithmetic smoke 结果

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic,
train_cases=64, eval_cases=16, context_len=512,
latent_steps=4,
projector_kind=stepwise, max_latent_steps=4,
latent_step_curriculum=linear,
step_hidden_loss_weight=0.5, step_hidden_mode=output,
epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| total loss | 8.2645 -> 1.5780 |
| CE loss | 8.1214 -> 1.4186 |
| step output-hidden loss | 0.2866 -> 0.3188 |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 197.2s |

curriculum 过程：

```text
epoch 01-12: latent_steps=1
epoch 13-25: latent_steps=2
epoch 26-37: latent_steps=3
epoch 38-50: latent_steps=4
```

代表输出：

```text
146
101
417
201
123
421
147
697
343
275
427
```

落盘文件：

- `needle_sq_results/latent_projector_b17_arithmetic_k4_stepwise_output_w05_curriculum.json`
- `needle_sq_results/latent_projector_b17_arithmetic_k4_stepwise_output_w05_curriculum.pt`

#### 47.15.3 结论

B1.7 仍是负结果：accuracy / contains accuracy 都是 0%。但和 B1.6 的退化模式不同：

| 实验 | 输出退化形态 |
|---|---|
| B1.6 shared projector + output-hidden | 几乎全部坍缩到 `101`，只有少数 `272`。 |
| B1.7 stepwise + step embedding + curriculum | 输出更分散，如 `146/417/201/421/697/...`，但仍不是正确计算。 |

这说明 stepwise transition 和 curriculum 缓解了“单一固定答案坍缩”，但没有解决核心问题：当前 latent transition 没有被训练成可执行的算术状态机。

更新后的判断：

1. **结构替换是必要但不充分**：per-step transition 比共享 projector 有更强表达力，但没有自动产生可泛化算法。
2. **hidden/CE 目标仍太弱**：teacher hidden MSE + answer CE 只能学到 answer-like 数字分布，无法稳定学到 `(a+b)*c-d` 的中间变量更新。
3. 下一步如果继续 B 线，应该从“拟合 teacher hidden”转为更显式的 latent state machine：
   - latent slot 显式监督中间变量 `sum/product/final`；
   - 或把每步 transition 拆成 `read operands -> update state -> emit answer`；
   - 或先用小 MLP/linear probe 证明 latent_i 中能读出中间变量，再接回 LLM decode。

### 47.16 B1.8 arithmetic latent state probe（2026-06-18）

承接 §47.15，本轮开始把 B 线转向“显式 latent state machine / 中间变量监督”。先做最小可验证版本：不直接改变生成格式，而是在训练时给 latent output hidden 加一个辅助 probe，让每个 latent step 可读出 arithmetic 中间变量。

目标监督：

```text
latent_1 output_hidden -> sum     = a + b
latent_2 output_hidden -> product = sum * c
latent_3 output_hidden -> final   = product - d
latent_4 output_hidden -> final   = product - d  # answer-ready state
```

这里的 probe 只用于训练期 auxiliary loss，不参与最终生成；推理仍是：prompt -> latent tokens -> answer-only decode。

#### 47.16.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `ArithmeticStateProbe`：每个 latent step 一个 linear head，把 latent output hidden 分类到整数值域。
- 新增 `_build_arithmetic_state_targets()`：从 synthetic arithmetic case 的 `a/b/c/d` 构造 `sum/product/final/final` targets。
- `train_projector_hf_teacher_forcing()` 新增 state auxiliary loss：

```text
loss = answer_CE
     + step_hidden_loss_weight * output_hidden_MSE
     + state_supervision_weight * CE(probe(output_hidden_i), intermediate_value_i)
```

- 新增 CLI：

```text
--state-supervision-weight
--state-min-value
--state-max-value
```

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
```

#### 47.16.2 B1.8 smoke 结果

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic,
train_cases=64, eval_cases=16, context_len=512,
latent_steps=4,
projector_kind=stepwise, max_latent_steps=4,
latent_step_curriculum=linear,
step_hidden_loss_weight=0.2, step_hidden_mode=output,
state_supervision_weight=0.5, state_min_value=-256, state_max_value=2048,
epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| total loss | 13.0521 -> 3.2985 |
| CE loss | 8.3586 -> 1.3883 |
| step output-hidden loss | 0.3025 -> 0.6658 |
| state probe loss | 9.2534 -> 3.5552 |
| state probe acc | 0.0% -> 10.5% |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 198.9s |

代表输出：

```text
103
103
127
123
127
621
201
1066
680
830
290
204
273
240
```

落盘文件：

- `needle_sq_results/latent_projector_b18_arithmetic_k4_state_probe_w05.json`
- `needle_sq_results/latent_projector_b18_arithmetic_k4_state_probe_w05.pt`

#### 47.16.3 结论

B1.8 仍然没有让 answer decode 泛化，accuracy / contains accuracy 都是 0%。但它给出了一个更明确的诊断：

1. state probe loss 明显下降，说明 latent output hidden 中确实开始携带一部分中间变量信息。
2. 但 state probe acc 只有 10.5%，远不足以支持稳定算术状态更新。
3. answer CE 下降到 1.3883，但输出仍是 answer-like 错误整数，说明最终 decode 仍主要学到数值分布，而不是计算规则。

这说明“简单 probe 辅助监督”还不等于真正的 latent state machine。下一步 B 线如果继续，应该进一步显式化状态：

```text
方案 B1.9：latent state vector 直接监督中间变量 embedding
  - 为 sum/product/final 构造可学习 numeric embedding table
  - latent_i 不只被 probe 读出，还直接拉近到 numeric embedding(value_i)
  - 再用 answer CE 检查能否把 final state decode 成答案

或：
方案 B2：把 arithmetic 拆成真实 state-machine trainer
  - transition_1 专门预测 sum
  - transition_2 接收 sum-state + c，预测 product
  - transition_3 接收 product-state + d，预测 final
  - 最后再接 LLM decode
```

当前判断：B 线还不能证明“hidden latent 自然会算术推理”，但现在已经能定位瓶颈：不是 token decode，而是 latent state 没有被训练成可读、可更新、可组合的中间变量状态。

### 47.17 B1.9 numeric state embedding 直接监督 latent state（2026-06-18）

承接 §47.16，本轮把中间变量监督从“旁路 probe 读取 latent”改成“直接塑形 latent output hidden”：为整数值域建立 trainable numeric embedding table，并把每个 latent step 的 output hidden 拉近到对应中间变量的 embedding。

目标监督仍是：

```text
latent_1 output_hidden -> embedding(sum)     where sum     = a + b
latent_2 output_hidden -> embedding(product) where product = sum * c
latent_3 output_hidden -> embedding(final)   where final   = product - d
latent_4 output_hidden -> embedding(final)   # answer-ready state
```

#### 47.17.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `NumericStateEmbedding`：整数值域 `state_min_value..state_max_value` 对应一个可学习 embedding table。
- 新增 `_numeric_state_embedding_loss()`：

```text
state_embedding_loss = CE(cosine(output_hidden_i, numeric_embedding_table) / temperature, value_i)
                     + state_embedding_mse_weight * MSE(RMSNorm(output_hidden_i), RMSNorm(embedding(value_i)))
```

- `train_projector_hf_teacher_forcing()` 新增 numeric state embedding auxiliary loss：

```text
loss = answer_CE
     + step_hidden_loss_weight * output_hidden_MSE
     + state_embedding_loss_weight * state_embedding_loss
```

- 新增 CLI：

```text
--state-embedding-loss-weight
--state-embedding-mse-weight
```

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

#### 47.17.2 B1.9 smoke 结果

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic,
train_cases=64, eval_cases=16, context_len=512,
latent_steps=4,
projector_kind=stepwise, max_latent_steps=4,
latent_step_curriculum=linear,
step_hidden_loss_weight=0.2, step_hidden_mode=output,
state_embedding_loss_weight=0.5, state_embedding_mse_weight=0.1,
state_min_value=-256, state_max_value=2048,
epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

结果：

| 指标 | 结果 |
|---|---:|
| total loss | 12.1663 -> 4.0014 |
| CE loss | 8.1135 -> 1.3454 |
| step output-hidden loss | 0.2930 -> 0.8245 |
| state embedding loss | 7.9898 -> 4.9829 |
| state embedding CE | 7.7891 -> 4.7971 |
| state embedding MSE | 2.0072 -> 1.8583 |
| state embedding acc | 0.0% -> 2.7% |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| elapsed | 198.9s |

代表输出：

```text
142
103
427
103
127
101
103
297
103
101
103
103
595
427
201
472
```

落盘文件：

- `needle_sq_results/latent_projector_b19_arithmetic_k4_state_embed_w05.json`
- `needle_sq_results/latent_projector_b19_arithmetic_k4_state_embed_w05.pt`

#### 47.17.3 结论

B1.9 仍是负结果：accuracy / contains accuracy 都是 0%。并且 numeric embedding 直接监督没有比 B1.8 probe 监督更强：

1. state embedding loss 明显下降，但 state embedding acc 只有 2.7%，低于 B1.8 probe acc 的 10.5%。
2. answer CE 继续下降到 1.3454，但输出仍是 `101/103/127/427/...` 这类 answer-like 错误整数，说明 decode 侧仍在拟合答案分布而不是执行算术。
3. 直接拉近到 numeric embedding 并没有自动形成可组合状态机；当前 transition 仍缺少“从 operand 更新 state”的显式结构。

下一步如果继续 B 线，不应再只增加 auxiliary loss，而应转向 B2：显式 state-machine trainer，把 transition 拆成 `read operands -> update sum -> update product -> update final -> decode`，并单独验证每个 transition 的数值更新准确率。

### 47.18 B2 arithmetic state-machine trainer（2026-06-18）

承接 §47.17，本轮不再继续给 `stepwise` projector 堆旁路辅助损失，而是新增一个显式 arithmetic state-machine projector：latent step 本身按 phase 组织为 `SUM -> PRODUCT -> FINAL`，并在 projector 内部挂 transition heads 来监督和度量每一步数值状态更新。

目标：

```text
prompt hidden
  -> transition_1(SUM phase)     -> latent_sum     -> value = a + b
  -> transition_2(PRODUCT phase) -> latent_product -> value = (a + b) * c
  -> transition_3(FINAL phase)   -> latent_final   -> value = (a + b) * c - d
  -> answer-only decode
```

#### 47.18.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `ArithmeticStateMachineProjector`，作为新的 `--projector-kind arithmetic_state_machine`：
  - 每步有独立 transition；
  - 每步加 `step_embed`；
  - 每步还加 phase embedding：`SUM / PRODUCT / FINAL`；
  - projector 内部自带 `value_heads` 和 `phase_heads`，用于 transition trainer 和指标统计。
- 新增 `ArithmeticTransitionTargets` / `_build_arithmetic_transition_targets()`：从 `a/b/c/d` 构造每步 `(phase, value)` target。
- 新增 `_arithmetic_state_machine_loss_and_metrics()`：

```text
state_machine_loss = value_weight * CE(value_head(latent_i), value_i)
                   + phase_weight * CE(phase_head(latent_i), phase_i)
```

- 新增 `evaluate_arithmetic_state_machine_hf()`，在 eval set 上单独报告：
  - value accuracy
  - phase accuracy
  - transition accuracy
  - sum/product/final accuracy
  - full sequence accuracy
- 新增 CLI：

```text
--projector-kind arithmetic_state_machine
--arithmetic-state-machine-weight
--arithmetic-state-machine-value-weight
--arithmetic-state-machine-phase-weight
--arithmetic-state-machine-source {input,output}
--arithmetic-state-machine-no-repeat-final
```

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

#### 47.18.2 B2 smoke 结果

命令参数：

```text
model=Qwen3-0.6B, task=arithmetic,
train_cases=64, eval_cases=16, context_len=512,
latent_steps=3,
projector_kind=arithmetic_state_machine, max_latent_steps=3,
latent_step_curriculum=linear,
arithmetic_state_machine_weight=1.0,
arithmetic_state_machine_value_weight=1.0,
arithmetic_state_machine_phase_weight=0.2,
arithmetic_state_machine_source=output,
state_min_value=-256, state_max_value=2048,
epochs=50, batch_size=8, lr=1e-4, max_new_tokens=16
```

训练结果：

| 指标 | 结果 |
|---|---:|
| total loss | 17.9845 -> 4.1787 |
| CE loss | 8.5761 -> 1.5582 |
| state-machine loss | 9.4030 -> 2.6222 |
| value CE | 9.1767 -> 2.6220 |
| phase CE | 1.1315 -> 0.0014 |
| train transition acc | 0.0% -> 26.0% |
| train sequence acc | 0.0% -> 3.1% |
| train sum acc | 0.0% -> 14.1% |
| train product acc | 0.0% -> 43.8% |
| train final acc | 0.0% -> 20.3% |
| answer-only acc | 0.0% |
| contains acc | 0.0% |
| eval transition acc | 0.0% |
| eval sequence acc | 0.0% |
| elapsed | 126.5s |

代表输出：

```text
104
103
622
145
111
427
653
214
627
1136
1166
624
417
347
417
652
```

落盘文件：

- `needle_sq_results/latent_projector_b20_arithmetic_k3_state_machine_w1.json`
- `needle_sq_results/latent_projector_b20_arithmetic_k3_state_machine_w1.pt`

#### 47.18.3 结论

B2 比 B1.9 更有诊断价值，但仍是负结果：

1. **phase 学会了，value 没学会泛化**：eval phase acc = 100%，但 eval value / transition / sequence acc 都是 0%。说明模型能区分当前位置应该是 `SUM/PRODUCT/FINAL`，但没有学到数值更新规则。
2. **训练集 transition acc 有上升**：train transition acc 到 26.0%，product step 到 43.8%，说明 explicit state-machine trainer 至少能把一部分训练样本的中间状态写进 latent。
3. **泛化为 0**：eval transition acc 仍是 0%，answer-only acc 也仍是 0%。这说明当前 B2 仍主要是在拟合训练样本映射，而不是学习可组合 arithmetic update。
4. **answer CE 下降仍不等于计算**：CE 到 1.5582，但输出仍是 answer-like 错误整数。

更新后的判断：B2 证实了问题已经从“decode 失败”进一步定位到“value transition 没有泛化”。下一步如果继续，应该先脱离 LLM decode，做一个纯 latent arithmetic transition micro-benchmark：给定显式数值 code / operand code，要求 transition 模块在 held-out 数值组合上预测 sum/product/final；只有这个 micro-benchmark 过了，再接回 frozen LLM decode。

### 47.19 B2.1 pure latent arithmetic transition micro-benchmark（2026-06-18）

承接 §47.18，本轮先完全脱离 HF/LLM decode，只测试 latent transition 模块本身是否能在 held-out `(a,b,c,d)` 数值组合上学习：

```text
explicit operand codes(a,b,c,d)
  -> tuple encoder -> initial latent
  -> transition_1 -> predict sum     = a + b
  -> transition_2 -> predict product = (a + b) * c
  -> transition_3 -> predict final   = (a + b) * c - d
```

这个实验的目的不是生成答案文本，而是直接验证“latent state machine 是否具备数值更新泛化能力”。

#### 47.19.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `PureArithmeticTuple`：纯数值 tuple，不依赖 tokenizer / prompt / LLM。
- 新增 `PureArithmeticTupleEncoder`：把显式 operand code `(a,b,c,d)` 编码成初始 latent。
- 新增 `PureLatentArithmeticTransitionModel`：`tuple_encoder + ArithmeticStateMachineProjector`。
- 新增 `build_pure_arithmetic_tuples()`：生成唯一 held-out tuple 组合。
- 新增 `_build_pure_arithmetic_transition_targets()`：构造 `sum/product/final` 三步 value targets。
- 新增 `train_pure_latent_arithmetic_transition()` / `evaluate_pure_latent_arithmetic_transition()`。
- 新增 `run_pure_latent_arithmetic_benchmark()`，通过 `--pure-latent-arithmetic` 直接绕开 HF/TinyLLM 路径。
- `--model` 从 argparse required 改成运行时校验：pure micro-benchmark 不需要 model；非 pure 模式仍要求 `--model`。

新增 CLI：

```text
--pure-latent-arithmetic
--pure-hidden-size
--pure-arith-a-min / --pure-arith-a-max
--pure-arith-b-min / --pure-arith-b-max
--pure-arith-c-min / --pure-arith-c-max
--pure-arith-d-min / --pure-arith-d-max
--pure-arith-value-weight
--pure-arith-phase-weight
```

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

#### 47.19.2 B2.1 micro-benchmark 结果

命令参数：

```text
pure_latent_arithmetic=true,
train_cases=4096, eval_cases=1024,
latent_steps=3,
epochs=80, batch_size=256, lr=3e-4,
pure_hidden_size=128,
state_min_value=-256, state_max_value=2048,
pure_arith_value_weight=1.0,
pure_arith_phase_weight=0.0
```

训练集结果：

| 指标 | 结果 |
|---|---:|
| loss / value CE | 7.6035 -> 1.3839 |
| transition acc | 0.2% -> 32.2% |
| sequence acc | 0.0% -> 4.4% |
| sum acc | 0.5% -> 23.1% |
| product acc | 0.1% -> 37.6% |
| final acc | 0.1% -> 35.9% |

held-out eval 结果：

| 指标 | 结果 |
|---|---:|
| loss / value CE | 6.4072 |
| value acc | 3.4% |
| transition acc | 1.5% |
| sequence acc | 0.0% |
| sum acc | 2.3% |
| product acc | 1.7% |
| final acc | 0.4% |
| value MAE | 45.65 |
| elapsed | 69.9s |

落盘文件：

- `needle_sq_results/latent_projector_b21_pure_arithmetic_transition.json`
- `needle_sq_results/latent_projector_b21_pure_arithmetic_transition.pt`

#### 47.19.3 结论

B2.1 结果说明：即使去掉 LLM decode，只保留显式 operand code + latent transition，当前 MLP/RMSLinear state machine 仍没有学到可泛化的算术更新规则。

关键观察：

1. **训练集能拟合一部分，但远未学透**：80 epoch 后训练 transition acc 只有 32.2%，sequence acc 只有 4.4%。
2. **held-out 泛化几乎为零**：eval transition acc 1.5%，sequence acc 0.0%，final acc 0.4%。
3. **这排除了 LLM decode 作为主瓶颈**：纯 latent micro-benchmark 都不过，说明问题在数值表示与 transition 算法本身。
4. **当前 embedding classification 不适合学习组合算术**：随机/可学习离散 value embedding + MLP transition 更像查表/插值，不自然表示加法和乘法。

更新后的判断：下一步不应继续接 frozen LLM，也不应继续扩大 hidden size 或 epoch。应先把 pure micro-benchmark 改成更算法化的数值表示，例如：

```text
B2.2: structured numeric representation
  - 输入显式 scalar / normalized scalar / digit representation，而不是纯 learned value id embedding
  - transition head 同时做 regression + digit classification
  - 先要求 held-out sum/product/final sequence acc 明显非零，再回接 latent decode
```

### 47.20 B2.2 structured numeric representation（2026-06-22）

承接 §47.19，本轮把 pure latent micro-benchmark 的数值表示从纯 learned value id embedding 改为更结构化的表示，并新增 regression + digit classification head。

核心变化：

```text
operand value
  -> normalized scalar feature
  -> decimal digit one-hot feature
  -> structured tuple encoder
  -> latent transitions
  -> value classification head      # 可选，保持 B2.1 可比
  -> scalar regression head         # 新增
  -> decimal digit classification   # 新增
```

#### 47.20.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `PureStructuredArithmeticTupleEncoder`：每个 operand 用 normalized scalar + decimal digit one-hot 表示，再按 `a/b/c/d` slot 聚合成 initial latent。
- `PureLatentArithmeticTransitionModel` 新增：
  - `numeric_representation={id_embedding, structured}`；
  - 每个 latent step 的 `regression_heads`；
  - 每个 latent step 的 `digit_heads`。
- 新增 `_pure_structured_numeric_loss_and_metrics()`：

```text
structured_loss = regression_weight * SmoothL1(regressed_scalar, normalized_target_value)
                + digit_weight * CE(predicted_digits, target_decimal_digits)
```

- 新增 structured metrics：
  - digit accuracy；
  - digit exact value accuracy；
  - digit sequence accuracy；
  - regression MAE；
  - regression rounded exact accuracy。
- 新增 CLI：

```text
--pure-numeric-representation {id_embedding,structured}
--pure-arith-regression-weight
--pure-arith-digit-weight
```

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

#### 47.20.2 B2.2 structured-only 结果

命令参数：

```text
pure_latent_arithmetic=true,
pure_numeric_representation=structured,
train_cases=4096, eval_cases=1024,
latent_steps=3,
epochs=80, batch_size=256, lr=3e-4,
pure_hidden_size=128,
pure_arith_value_weight=0.0,
pure_arith_phase_weight=0.0,
pure_arith_regression_weight=1.0,
pure_arith_digit_weight=1.0
```

训练集结果：

| 指标 | 结果 |
|---|---:|
| structured loss | 1.8927 -> 0.9052 |
| regression loss | 0.0562 -> 0.0024 |
| digit CE | 1.8365 -> 0.9028 |
| digit accuracy | 35.2% -> 66.3% |
| digit exact value acc | 0.4% -> 14.2% |
| digit sequence acc | 0.0% -> 0.1% |
| regression MAE | 283.8 -> 61.7 |
| regression rounded acc | 0.1% -> 0.6% |

held-out eval：

| 指标 | 结果 |
|---|---:|
| structured loss | 0.9669 |
| digit accuracy | 61.7% |
| digit exact value acc | 9.1% |
| digit sequence acc | 0.0% |
| regression MAE | 64.4 |
| regression rounded acc | 0.5% |
| elapsed | 61.5s |

落盘文件：

- `needle_sq_results/latent_projector_b22_pure_arithmetic_structured.json`
- `needle_sq_results/latent_projector_b22_pure_arithmetic_structured.pt`

#### 47.20.3 B2.2 structured + value CE 结果

命令参数和上面相同，但打开 value CE：

```text
pure_arith_value_weight=1.0,
pure_arith_regression_weight=1.0,
pure_arith_digit_weight=1.0
```

训练集结果：

| 指标 | 结果 |
|---|---:|
| total training loss | 9.5638 -> 4.1109 |
| value CE | 7.6557 -> 3.0637 |
| structured loss | 1.9080 -> 1.0472 |
| value-head transition acc | 0.1% -> 10.7% |
| digit exact value acc | 0.4% -> 8.2% |
| sequence acc | 0.0% -> 0.0% |
| regression MAE | 292.5 -> 81.5 |

held-out eval：

| 指标 | 结果 |
|---|---:|
| value-head transition acc | 3.0% |
| sequence acc | 0.0% |
| sum acc | 5.3% |
| product acc | 3.4% |
| final acc | 0.2% |
| digit exact value acc | 5.6% |
| digit sequence acc | 0.0% |
| value-head MAE | 18.6 |
| regression MAE | 83.3 |
| regression rounded acc | 0.5% |
| elapsed | 61.2s |

落盘文件：

- `needle_sq_results/latent_projector_b22_pure_arithmetic_structured_value.json`
- `needle_sq_results/latent_projector_b22_pure_arithmetic_structured_value.pt`

#### 47.20.4 结论

B2.2 比 B2.1 有局部改善，但仍没有解决组合泛化：

1. **structured digit 表示明显优于纯 value-id embedding 的部分指标**：structured-only eval digit exact value acc 达到 9.1%，高于 B2.1 的 eval transition acc 1.5%。说明 digit representation 的确比随机 learned id 更有结构。
2. **sequence 仍为 0**：无论 structured-only 还是 structured + value CE，held-out sequence acc 都是 0.0%。这意味着三步 `sum/product/final` 组合更新仍没有成立。
3. **regression 学到近似但不能精确**：regression MAE 从数百降到约 64/83，但 rounded exact acc 只有约 0.5%，不足以作为离散算术状态。
4. **value CE 和 structured loss 会互相牵制**：打开 value CE 后 value-head eval transition acc 到 3.0%，但 digit exact value acc 从 9.1% 降到 5.6%。说明当前多头目标没有自然对齐成同一套可组合数值表示。

更新后的判断：structured numeric representation 是正确方向，但“单个 latent 向量 + MLP transition + 多头监督”仍不够算法化。下一步如果继续，应把数值状态显式拆成 digit-wise state，并让 transition 按算术规则处理 carry / multiplication decomposition，而不是让一个 dense latent 自己发现十进制进位规则。

### 47.21 B2.3 digit-wise state + carry/multiply/borrow decomposition（2026-06-22）

承接 §47.20，本轮不再让 dense latent 自己发现十进制进位规则，而是先实现一个纯 digit-wise arithmetic state-machine harness：把数值状态拆成 LSD-first 十进制 digit slots，并显式执行：

```text
add:      a_i + b_i + carry_i -> sum_i, carry_{i+1}
multiply: sum_i * c + carry_i -> product_i, carry_{i+1}
subtract: lhs_i - rhs_i - borrow_i -> final_i, borrow_{i+1}
sign:     product >= d ? positive : negative
```

这里 B2.3 先做 deterministic algorithmic baseline，目标是验证 digit-wise state / carry trace / metrics 本身是否正确；不是继续训练 dense MLP。

#### 47.21.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `PureDigitArithmeticTrace`：保存 `sum_digits/product_digits/final_digits/final_sign/add_carry/mul_carry/sub_borrow/overflow_mask`。
- 新增真实十进制 LSD-first helper：
  - `_positive_values_to_lsd_digits()`；
  - `_lsd_digits_to_positive_values()`；
  - `_infer_pure_digit_num_digits()`。
- 新增 direct target trace：`_build_direct_digit_arithmetic_trace()`。
- 新增 algorithmic digit-wise trace：`_build_algorithmic_digit_arithmetic_trace()`。
- 新增 `_pure_digit_trace_metrics()`，统计：
  - sum/product/final value accuracy；
  - sequence value accuracy；
  - digit accuracy；
  - add carry / multiply carry / subtract borrow accuracy；
  - final sign accuracy；
  - MAE / overflow cases。
- 新增 `run_pure_digit_arithmetic_benchmark()`，通过 `--pure-digit-arithmetic` 直接运行 digit-wise deterministic baseline。
- 新增 CLI：

```text
--pure-digit-arithmetic
--pure-digit-mode deterministic
--pure-digit-num-digits  # 0 表示自动推断
```

本地和远端检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

说明：本地 smoke 运行仍被 macOS `/usr/bin/python3` 缺少 torch 阻塞；远端 smoke 正常。

#### 47.21.2 B2.3 deterministic digit-wise 结果

命令参数：

```text
pure_digit_arithmetic=true,
pure_digit_mode=deterministic,
train_cases=4096,
eval_cases=1024,
batch_size=256,
pure_digit_num_digits=0  # auto -> 4 digits
```

held-out eval：

| 指标 | 结果 |
|---|---:|
| num digits | 4 |
| valid cases | 1024 |
| overflow cases | 0 |
| sum value acc | 100.0% |
| product value acc | 100.0% |
| final value acc | 100.0% |
| sequence value acc | 100.0% |
| sum digit acc | 100.0% |
| product digit acc | 100.0% |
| final digit acc | 100.0% |
| add carry acc | 100.0% |
| multiply carry acc | 100.0% |
| subtract borrow acc | 100.0% |
| final sign acc | 100.0% |
| sum/product/final MAE | 0.0 / 0.0 / 0.0 |
| elapsed | 0.067s |

落盘文件：

- `needle_sq_results/latent_projector_b23_pure_digit_arithmetic.json`
- `needle_sq_results/latent_projector_b23_pure_digit_arithmetic.pt`

#### 47.21.3 结论

B2.3 证明：一旦把状态显式拆成 digit slots，并显式建模 carry / multiplication carry / borrow，held-out `(a,b,c,d)` 上可以稳定达到 100% sequence accuracy。

这和 B2.1/B2.2 的对比很明确：

| 路线 | held-out sequence acc | 判断 |
|---|---:|---|
| B2.1 learned value-id latent | 0.0% | dense latent + value CE 不会自然学出组合算术。 |
| B2.2 structured scalar/digit heads | 0.0% | digit 表示有帮助，但 dense transition 仍不会自动学 carry。 |
| B2.3 explicit digit-wise state machine | 100.0% | 算法结构本身是关键。 |

更新后的判断：后续如果要把这条线接回“hidden-state agent reasoning”，不应再尝试让单个 dense hidden 自发学习算术，而应把 latent state 设计成结构化 slots：

```text
latent_state = {digits[], sign, carry, borrow, phase, operands}
transition = local digit operator + carry propagation
```

下一步可以做 B2.4：trainable digit-wise operator。即保持 B2.3 的 digit slots / carry trace 结构，但把每个局部算子 `digit_i, carry_i -> digit_out, carry_out` 改成小模型训练，检查它是否能在 held-out digit combinations 上泛化到 100%，再考虑接回 LLM decode。

### 47.22 B2.4 trainable local digit operator（2026-06-22）

承接 §47.21，本轮保持 B2.3 的显式 digit slots / carry trace / LSD-first 结构不变，但把局部十进制规则从 deterministic code 改成可训练小模型：

```text
add:      x_digit, y_digit, carry_in -> digit_out, carry_out
multiply: x_digit, c,       carry_in -> digit_out, carry_out
subtract: x_digit, y_digit, borrow_in -> digit_out, borrow_out
```

核心问题：如果只训练一部分 local digit combinations，小 MLP operator 能否在 held-out digit combinations 上泛化到 100%，并继续让完整 `(a,b,c,d)` tuple sequence 达到 100%。

#### 47.22.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `DigitOperatorExample`：保存单个局部 digit transition 的 `op/x/y/carry_in/digit_out/carry_out`。
- 新增 `TrainableDigitLocalOperator`：one-hot digit/carry/scalar feature + MLP，输出 `digit_head` 和 `carry_head`。
- 新增 `TrainableDigitWiseOperator`：分别持有 add / multiply / subtract 三个 local operator。
- 新增 `enumerate_digit_operator_examples()`：枚举完整 local truth table。
- 新增 `split_digit_operator_examples()`：按 stable hash 把 `(op,x,y,carry_in)` 组合拆成 train / held-out。
- 新增 `_build_trainable_digit_arithmetic_trace()`：用 trainable local operator 执行完整 B2.3 tuple trace。
- 新增 `train_pure_digit_operator()` / `evaluate_trainable_digit_operator()` / `evaluate_trainable_digit_tuple_arithmetic()`。
- `run_pure_digit_arithmetic_benchmark()` 新增 `--pure-digit-mode trainable` 分支。
- 修正 `evaluate_trainable_digit_operator()`：digit-combination table 很小，最终 operator eval 改为一次性评估，避免 per-op metrics 被 batch total 加权后失真；overall transition metrics 不受这个显示问题影响。

新增 CLI：

```text
--pure-digit-mode {deterministic,trainable,lookup}
--pure-digit-operator-hidden-size
--pure-digit-operator-depth
--pure-digit-combo-heldout-frac
--pure-digit-combo-split-seed
--pure-digit-carry-loss-weight
```

本地检查：

```text
python3 -m py_compile tools/train_latent_projector.py
```

#### 47.22.2 B2.4 trainable local operator 结果

命令参数：

```text
pure_digit_arithmetic=true,
pure_digit_mode=trainable,
train_cases=4096,
eval_cases=1024,
num_digits=4,
epochs=500,
batch_size=128,
lr=1e-3,
weight_decay=0,
pure_digit_operator_hidden_size=128,
pure_digit_operator_depth=3,
pure_digit_combo_heldout_frac=0.2,
pure_digit_combo_split_seed=0,
pure_digit_carry_loss_weight=1.0,
pure_arith_c_min=2,
pure_arith_c_max=9
```

local digit-combination split：

| split | examples |
|---|---:|
| train | 904 |
| held-out | 216 |

训练过程最后一轮：

```text
digit_op_epoch=500 train_trans=100.0% heldout_trans=66.7% tuple_seq=67.6%
```

最终 operator-level 结果（per-op 指标用 checkpoint 重新按完整 split 一次性计算）：

| 指标 | train local combos | held-out local combos |
|---|---:|---:|
| digit accuracy | 100.0% | 68.1% |
| carry accuracy | 100.0% | 95.4% |
| transition accuracy | 100.0% | 66.7% |
| add transition accuracy | 100.0% | 93.2% |
| multiply transition accuracy | 100.0% | 50.0% |
| subtract transition accuracy | 100.0% | 97.2% |

完整 tuple-level held-out eval：

| 指标 | 结果 |
|---|---:|
| valid cases | 1024 |
| overflow cases | 0 |
| sum value acc | 97.9% |
| product value acc | 71.7% |
| final value acc | 67.6% |
| sequence value acc | 67.6% |
| sum digit acc | 99.5% |
| product digit acc | 91.7% |
| final digit acc | 88.3% |
| add carry acc | 100.0% |
| multiply carry acc | 99.6% |
| subtract borrow acc | 98.4% |
| final sign acc | 99.7% |
| product MAE | 250.24 |
| final MAE | 378.20 |
| elapsed | 519.5s |

落盘文件（远端）：

- `needle_sq_results/latent_projector_b24_trainable_digit_operator.json`
- `needle_sq_results/latent_projector_b24_trainable_digit_operator.pt`

#### 47.22.3 结论

B2.4 是负结果：**trainable local digit operator 没有在 held-out digit combinations 上泛化到 100%**。

关键观察：

1. **训练 local combos 被完全记住**：train transition accuracy 达到 100.0%，说明容量足够拟合局部 truth table。
2. **held-out local combos 只到 66.7%**：carry/head 大多能学会，held-out carry accuracy 95.4%，但 digit_out 泛化只有 68.1%。
3. **乘法是主要瓶颈**：held-out multiply transition accuracy 只有 50.0%，远低于 add 的 93.2% 和 subtract 的 97.2%。这说明 `x_digit * c + carry_in -> digit_out` 不是普通 MLP 从缺失组合中自然外推出来的规则。
4. **tuple sequence 受局部错误累积限制**：held-out tuple sequence accuracy 为 67.6%，和 held-out local transition 66.7% 接近；product/final 的 MAE 很大，说明少数乘法 digit 错误会被位权放大。
5. **B2.3 的 100% 来自算法结构，不只是 digit slot 表示**：显式 slots / carry trace 是必要条件，但如果 local transition 仍是无结构 MLP，并且 local truth table 有 held-out holes，就不能保证组合泛化。

更新后的判断：如果目标是 agent hidden state 中的可靠算术，下一步不应只把 local operator 做得更大；更合理的是把 primitive digit operator 本身也算法化或加硬约束：

```text
latent_state = structured digit slots + carry trace
local_operator = constrained modular arithmetic / complete primitive table / differentiable circuit
```

也就是说，B2.4 把结论从“需要 digit-wise state”推进到“还需要算法化的 local transition”。

### 47.23 B2.5 primitive lookup local digit operator（2026-06-22）

承接 §47.22，本轮不再训练普通 MLP local operator，而是把同一个 local digit operator API 换成完整 primitive lookup table：

```text
op, x_digit, y_or_multiplier, carry_in
  -> lookup table
  -> digit_out, carry_out
```

这不是为了证明“手写算术当然能算对”，而是为了做 B2.4 的对照：

```text
B2.4: structured digit slots + learned MLP local transition + held-out local combos -> 失败
B2.5: structured digit slots + complete primitive local transition table          -> 是否恢复 100%
```

#### 47.23.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `PrimitiveLookupDigitWiseOperator`：
  - 和 `TrainableDigitWiseOperator` 暴露同样的 `forward_op(op, x, y, carry_in)` 接口；
  - 内部注册 add / multiply / subtract 的完整 lookup table；
  - 输出 one-hot-like logits，便于复用现有 operator-level CE / accuracy metrics；
  - 不需要训练。
- `run_pure_digit_arithmetic_benchmark()` 新增 `--pure-digit-mode lookup` 分支：
  - 复用 B2.4 的 local-combo split；
  - 同时报告 train local combos、held-out local combos、完整 tuple train/eval。
- 远端和本地检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
```

#### 47.23.2 B2.5 primitive lookup 结果

命令参数：

```text
pure_digit_arithmetic=true,
pure_digit_mode=lookup,
train_cases=4096,
eval_cases=1024,
batch_size=256,
train_device=cpu,
pure_digit_num_digits=0  # auto -> 4 digits
pure_digit_combo_heldout_frac=0.2,
pure_digit_combo_split_seed=0,
pure_arith_c_min=2,
pure_arith_c_max=9
```

local digit-combination split 与 B2.4 相同：

| split | examples |
|---|---:|
| train | 904 |
| held-out | 216 |

最终 operator-level 结果：

| 指标 | train local combos | held-out local combos |
|---|---:|---:|
| digit accuracy | 100.0% | 100.0% |
| carry accuracy | 100.0% | 100.0% |
| transition accuracy | 100.0% | 100.0% |
| add transition accuracy | 100.0% | 100.0% |
| multiply transition accuracy | 100.0% | 100.0% |
| subtract transition accuracy | 100.0% | 100.0% |

完整 tuple-level held-out eval：

| 指标 | 结果 |
|---|---:|
| valid cases | 1024 |
| overflow cases | 0 |
| sum value acc | 100.0% |
| product value acc | 100.0% |
| final value acc | 100.0% |
| sequence value acc | 100.0% |
| sum digit acc | 100.0% |
| product digit acc | 100.0% |
| final digit acc | 100.0% |
| add carry acc | 100.0% |
| multiply carry acc | 100.0% |
| subtract borrow acc | 100.0% |
| final sign acc | 100.0% |
| sum/product/final MAE | 0.0 / 0.0 / 0.0 |
| elapsed | 0.246s |

落盘文件（远端）：

- `needle_sq_results/latent_projector_b25_primitive_lookup_digit_operator.json`
- `needle_sq_results/latent_projector_b25_primitive_lookup_digit_operator.pt`

#### 47.23.3 结论

B2.5 是正向对照：**一旦 local primitive 完整覆盖 / 算法化，同一套 digit slots + carry trace 可以恢复 100% held-out tuple sequence accuracy**。

与 B2.4 的差异很集中：

| 路线 | local transition | held-out local transition | held-out tuple sequence |
|---|---|---:|---:|
| B2.4 | trainable MLP，held-out combos 缺失 | 66.7% | 67.6% |
| B2.5 | complete primitive lookup table | 100.0% | 100.0% |

这说明 B2.4 的失败不是 digit slots / carry trace 的失败，而是 **无结构 MLP local transition 对缺失 primitive combos 的外推失败**。

更新后的判断：如果继续 hidden-state reasoning，下一步不应再问“MLP 能不能自己发现 digit rule”，而应测试更接近真实可用系统的结构：

```text
structured latent state
  + primitive/constrained local transition
  + LLM decode bridge
```

也就是把 B2.5 的 primitive operator 接回 hidden-state / answer decode 路径，验证结构化 latent state 是否能成为 LLM 内部 reasoning substrate，而不是继续训练无约束 MLP 去猜算术表。

### 47.24 B2.6 oracle structured state -> LLM decode bridge（2026-06-22）

承接 §47.23，本轮把算术 transition 固定为已经验证过的 oracle structured state，不再测试 digit operator，而是单独测试最后一段 bridge：

```text
prompt
  + oracle structured arithmetic state
  -> small decode bridge -> latent input embeddings
  -> frozen Qwen3-0.6B
  -> answer tokens
```

目标是把失败点从“算术 transition”进一步拆出来：即使 final structured state 已经包含正确答案 digit/sign/carry trace，LLM 是否能通过一个小 bridge 读出这个状态并稳定输出答案。

#### 47.24.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `OracleStructuredDecodeBridge`：把结构化 feature 映射成 `latent_steps × hidden_size` 的 LLM input embeddings。
- 新增 `_oracle_structured_decode_features()`：为 arithmetic case 构造 oracle state feature，包含：
  - operands scalar；
  - sum/product/final digit one-hot；
  - final sign；
  - add carry / multiply carry / subtract borrow trace；
  - overflow bit。
- 新增 `_build_oracle_decode_bridge_batch()`：拼接 `prompt_embeds + bridge_latents + answer_prefix`，做 answer token teacher forcing。
- 新增 `train_oracle_decode_bridge()`：冻结 HF 模型，只训练 bridge。
- 新增 `generate_hf_with_oracle_decode_bridge()` / `evaluate_oracle_decode_bridge()`：用 bridge latent 做 greedy decode。
- 新增 `run_oracle_decode_bridge()` 和 CLI：

```text
--oracle-decode-bridge
--decode-bridge-hidden-size
--decode-bridge-depth
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
git diff --check
```

#### 47.24.2 B2.6 smoke：64 train / 16 eval

命令参数：

```text
model=Qwen3-0.6B,
oracle_decode_bridge=true,
task=arithmetic,
train_cases=64,
eval_cases=16,
context_len=512,
epochs=80,
batch_size=8,
lr=5e-4,
latent_steps=4,
decode_bridge_hidden_size=1024,
decode_bridge_depth=2,
hf_dtype=bfloat16
```

训练 teacher-forcing 指标：

| 指标 | 结果 |
|---|---:|
| loss | 5.0503 -> 0.0007 |
| token accuracy | 6.2% -> 100.0% |

held-out greedy decode：

| 指标 | 结果 |
|---|---:|
| eval cases | 16 |
| answer-only accuracy | 6.2% |
| contains accuracy | 6.2% |
| elapsed | 141.9s |

代表错误输出：

```text
expected=116  answer=176
expected=85   answer=676
expected=423  answer=441
expected=1221 answer=112
expected=74   answer=74  # only hit
```

落盘文件（远端）：

- `needle_sq_results/latent_projector_b26_oracle_decode_bridge.json`
- `needle_sq_results/latent_projector_b26_oracle_decode_bridge.pt`
- `needle_sq_results/latent_projector_b26_oracle_decode_bridge.log`

#### 47.24.3 B2.6 medium：256 train / 64 eval

命令参数差异：

```text
train_cases=256,
eval_cases=64,
context_len=256,
epochs=12,
batch_size=32
```

结果：

| 指标 | 结果 |
|---|---:|
| train loss | 4.7282 -> 0.8156 |
| train token accuracy | 8.6% -> 72.3% |
| eval answer-only accuracy | 7.8% |
| eval contains accuracy | 7.8% |
| elapsed | 46.7s |

落盘文件（远端）：

- `needle_sq_results/latent_projector_b26_oracle_decode_bridge_n256.json`
- `needle_sq_results/latent_projector_b26_oracle_decode_bridge_n256.pt`

#### 47.24.4 结论

B2.6 当前是负结果：**oracle structured state 已经包含正确答案，但小 bridge + frozen LLM decode 仍不能在 held-out arithmetic cases 上稳定读出答案**。

关键观察：

1. **bridge 可以过拟合训练 token**：64-case smoke 中 teacher-forcing token accuracy 到 100%，说明 bridge 有能力把训练样本状态映射到 frozen LLM 可接受的 answer-token 前缀。
2. **held-out decode 泛化很弱**：64/16 设置 eval 只有 6.2%，256/64 设置 eval 只有 7.8%。输出通常是“像答案的整数”，但数字不对。
3. **这不是 arithmetic transition 的问题**：B2.5 已经证明 structured state + primitive transition 是 100%；B2.6 的失败点在 `structured state -> LLM embedding/decode` 这段 bridge。
4. **直接 MLP bridge 仍像查表**：即使输入是显式 digit/sign/carry feature，直接映射到 LLM embedding 后，frozen LLM 不会自然把这些 latent embeddings 当成“可读数字状态”。

更新后的判断：要把 structured state 接回 LLM，不应只训练一个自由 MLP 把 state 投进 embedding 空间。下一步更合理的是给 decode bridge 加更强的语义约束，例如：

```text
B2.7 candidate:
  structured state -> canonical textual state tokens -> LLM decode
  structured state -> numeric token embedding composition
  structured state -> supervised hidden target of "Final integer: <answer>"
  bridge latent -> auxiliary decoder/probe must reconstruct digits before LLM decode
```

也就是说，当前 B 线已经定位出三个层次：

```text
1. dense hidden transition 学算术：失败
2. structured slots + primitive transition：成功
3. structured state -> frozen LLM decode bridge：当前失败
```

后续如果继续，应集中解决第 3 段 bridge 的可读性/语义对齐，而不是再训练算术 transition。

### 47.25 B2.7 textual oracle decode sanity checks（2026-06-22）

承接 §47.24，B2.6 说明自由 MLP bridge 不能把 structured state 稳定投到 frozen LLM 可读 embedding。B2.7 先退一步，不训练 bridge，而是把 oracle state 写成 canonical text，让 frozen Qwen3-0.6B 直接读文本，做两个 sanity check：

```text
B2.7a: prompt + "Oracle computed final integer: <answer>\nFinal integer:"
B2.7b: prompt + final_sign/final_digits_lsd textual schema + "Final integer:"
```

目标是区分：

```text
1. LLM 是否能从文本中复读/抽取 final answer；
2. LLM 是否能读懂我们当前的 LSD-first digit schema。
```

#### 47.25.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `_textual_oracle_suffix()`：构造两种 textual oracle suffix：
  - `final_answer`：直接给最终整数；
  - `digit_state`：给 `sum_digits_lsd/product_digits_lsd/final_sign/final_digits_lsd`，并提示 LSD-first 需要 reverse/drop leading zeros。
- 新增 `generate_hf_textual_oracle()` / `evaluate_textual_oracle()`。
- 新增 `run_textual_oracle_eval()` 和 CLI：

```text
--textual-oracle-eval
--textual-oracle-mode {final_answer,digit_state}
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
git diff --check
```

#### 47.25.2 B2.7a textual final-answer oracle

命令参数：

```text
model=Qwen3-0.6B,
textual_oracle_eval=true,
textual_oracle_mode=final_answer,
task=arithmetic,
eval_cases=128,
context_len=256,
max_new_tokens=16,
hf_dtype=bfloat16
```

结果：

| 指标 | 结果 |
|---|---:|
| eval cases | 128 |
| answer-only accuracy | 100.0% |
| contains accuracy | 100.0% |
| elapsed | 71.25s |

代表输出：

```text
Oracle computed final integer: 116
Final integer: 116
Answer: 116
```

落盘文件（远端）：

- `needle_sq_results/latent_projector_b27_textual_final_answer_oracle.json`
- `needle_sq_results/latent_projector_b27_textual_final_answer_oracle.pt`

#### 47.25.3 B2.7b textual digit-state oracle

命令参数：

```text
model=Qwen3-0.6B,
textual_oracle_eval=true,
textual_oracle_mode=digit_state,
task=arithmetic,
eval_cases=128,
context_len=256,
max_new_tokens=16,
hf_dtype=bfloat16,
num_digits=4
```

结果：

| 指标 | 结果 |
|---|---:|
| eval cases | 128 |
| answer-only accuracy | 2.3% |
| contains accuracy | 2.3% |
| elapsed | 70.28s |

典型错误非常稳定：模型经常直接输出 LSD 顺序或补零后的数字，而不是 reverse/drop leading zeros 后的整数。

```text
expected=116  final_digits_lsd=6,1,1,0  answer=6110
expected=85   final_digits_lsd=5,8,0,0  answer=5800
expected=423  final_digits_lsd=3,2,4,0  answer=3240
expected=145  final_digits_lsd=5,4,1,0  answer=5410
expected=1221 final_digits_lsd=1,2,2,1  answer=1221  # palindrome-like hit
```

落盘文件（远端）：

- `needle_sq_results/latent_projector_b27_textual_digit_state_oracle.json`
- `needle_sq_results/latent_projector_b27_textual_digit_state_oracle.pt`

#### 47.25.4 结论

B2.7 结果非常清晰：

1. **final-answer textual oracle 是 100%**：如果直接把最终整数写进文本，frozen LLM 可以稳定复读/抽取。说明基本 prompt、decode、eval pipeline 没问题。
2. **当前 LSD digit schema 几乎不可读**：即使明确说明 `least-significant digit first; reverse it and drop leading zeros`，模型仍大多输出 LSD 顺序加零，例如 `116 -> 6110`。这说明当前 structured state schema 对 LLM 不是自然语义。
3. **B2.6 bridge 失败有合理解释**：自由 MLP bridge 试图把 LSD digit slots 投到 embedding 空间；但 B2.7b 证明就算把这些 slots 写成显式文本，LLM 也不稳定会读，更不用说从连续 embedding 中读。

更新后的判断：下一步不要直接做 B2.7c 的“LSD canonical token embedding composition”，因为离散文本 LSD schema 本身已经失败。更合理的下一步是先换成 LLM 更自然的语义表示：

```text
B2.8 candidate:
  structured state -> canonical MSD decimal text, e.g. final_digits_msd=1,1,6
  structured state -> explicit equation text, e.g. final_abs = 116, sign = positive
  structured state -> natural language summary, e.g. The computed final integer is 116.
```

只有当离散 textual schema 本身能被 LLM 稳定读懂，再去测试 token embedding composition / hidden target alignment 才有意义。

### 47.26 B2.8 natural textual schemas for structured state（2026-06-23）

承接 §47.25，B2.7b 证明 LSD-first digit schema 对 frozen LLM 很不自然。因此本轮不做 embedding composition，而是先测试三种更自然的 textual schema：

```text
B2.8a MSD digits:       final_digits_msd=1,1,6
B2.8b final_abs KV:     final_abs=116, final_sign=positive
B2.8c natural summary:  The computed final integer is 116.
```

核心问题：如果 structured state 写成更接近 LLM 预训练分布的文本，LLM 能否稳定读出答案。

#### 47.26.1 代码改动

文件：`tools/train_latent_projector.py`

- 扩展 `_textual_oracle_suffix()`，新增三种 mode：
  - `msd_digits`：`final_digits_msd=<digits>`，MSD-first，不需要 reverse；
  - `final_abs`：`final_sign=<sign>, final_abs=<int>`；
  - `natural_summary`：`The computed final integer is <int>.`。
- 扩展 CLI：

```text
--textual-oracle-mode {final_answer,digit_state,msd_digits,final_abs,natural_summary}
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
git diff --check
```

#### 47.26.2 B2.8 结果

共同参数：

```text
model=Qwen3-0.6B,
textual_oracle_eval=true,
task=arithmetic,
eval_cases=128,
context_len=256,
max_new_tokens=16,
hf_dtype=bfloat16,
num_digits=4
```

| schema | answer-only acc | contains acc | elapsed | 判断 |
|---|---:|---:|---:|---|
| B2.7b `final_digits_lsd=6,1,1,0` | 2.3% | 2.3% | 70.28s | LLM 基本不会 reverse/drop zeros。 |
| B2.8a `final_digits_msd=1,1,6` | 66.4% | 66.4% | 61.85s | 明显改善，但 digit-list 拼接仍不稳定。 |
| B2.8b `final_abs=116, final_sign=positive` | 100.0% | 100.0% | 67.01s | KV integer field 可稳定读取。 |
| B2.8c natural summary | 100.0% | 100.0% | 67.78s | 接近 B2.7a final-answer oracle。 |

B2.8a 典型错误：

```text
expected=423 final_digits_msd=4,2,3 answer=1234
expected=63  final_digits_msd=6,3   answer=123
expected=601 final_digits_msd=6,0,1 answer=61
expected=306 final_digits_msd=3,0,6 answer=36
expected=499 final_digits_msd=4,9,9 answer=9999
```

B2.8b / B2.8c 典型输出：

```text
final_abs=116, final_sign=positive -> 116
The computed final integer is 116. -> 116
```

落盘文件（远端）：

- `needle_sq_results/latent_projector_b28_textual_msd_digits_oracle.json`
- `needle_sq_results/latent_projector_b28_textual_msd_digits_oracle.pt`
- `needle_sq_results/latent_projector_b28_textual_final_abs_oracle.json`
- `needle_sq_results/latent_projector_b28_textual_final_abs_oracle.pt`
- `needle_sq_results/latent_projector_b28_textual_natural_summary_oracle.json`
- `needle_sq_results/latent_projector_b28_textual_natural_summary_oracle.pt`

#### 47.26.3 结论

B2.8 给出明确分层：

1. **LLM 能读 canonical integer field**：`final_abs=<int>, final_sign=<sign>` 达到 100%。这说明只要 schema 对 LLM 是自然的 key-value integer，decode 没问题。
2. **LLM 能读自然语言 summary**：`The computed final integer is <int>.` 也是 100%，和 B2.7a 一致。
3. **digit list 仍不是天然稳定接口**：MSD-first 从 LSD 的 2.3% 提升到 66.4%，但仍远不到 100%。错误说明模型有时会忽略给定 digits，退回输出 `123/1234`、漏掉 0，或重复 digit。
4. **B2.6 的 bridge 目标应该换 schema**：如果连续 bridge 对齐的是 LSD/MSD digit slots，LLM 本身都不稳定会读；如果对齐的是 canonical integer field / natural summary hidden target，成功概率更高。

更新后的判断：下一步可以做 B2.9，不再把 bridge latent 对齐到 raw digit slots，而是对齐到 LLM 已证明能读的 canonical textual state：

```text
B2.9 candidate:
  structured state -> hidden target of "final_abs=<int>, final_sign=<sign>"
  structured state -> token embedding composition of canonical KV text
  structured state -> bridge latent, plus auxiliary decoder reconstructs final_abs/sign
```

这一步的目标是验证：离散 canonical text 100% 可读后，它的 token embeddings / hidden targets 是否也能作为连续 bridge 的监督目标。

### 47.27 B2.9 canonical text token embeddings oracle（2026-06-23）

承接 §47.26，本轮不训练 bridge，先验证 `inputs_embeds` 路径本身是否可靠：把 B2.8 中已经证明可读的 canonical text tokenized 后，直接查 LLM embedding table，再拼到 prompt embeddings 后面。

流程：

```text
canonical_text_ids = tokenizer.encode("final_abs=116, final_sign=positive\nFinal integer:")
canonical_embeds = model.get_input_embeddings()(canonical_text_ids)
inputs_embeds = concat(prompt_embeds, canonical_embeds)
frozen LLM greedy decode answer
```

这一步回答两个问题：

1. 离散 canonical text 可读，换成它的原生 token embeddings 后是否仍可读；
2. 如果可读，这就是后续 trainable bridge 的上界 / target manifold。

#### 47.27.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `generate_hf_embedding_oracle()`：用 `prompt_embeds + suffix_token_embeds` 做 generation。
- 新增 `evaluate_embedding_oracle()`。
- 新增 `run_embedding_oracle_eval()` 和 CLI：

```text
--embedding-oracle-eval
--embedding-oracle-mode {final_abs,natural_summary,msd_digits}
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
git diff --check
```

#### 47.27.2 B2.9 结果

共同参数：

```text
model=Qwen3-0.6B,
embedding_oracle_eval=true,
task=arithmetic,
eval_cases=128,
context_len=256,
max_new_tokens=16,
hf_dtype=bfloat16,
input path=inputs_embeds
```

| mode | 对应 B2.8 textual acc | B2.9 embedding acc | contains acc | elapsed | 判断 |
|---|---:|---:|---:|---:|---|
| `final_abs` | 100.0% | 100.0% | 100.0% | 62.39s | canonical KV token embeddings 可读。 |
| `natural_summary` | 100.0% | 100.0% | 100.0% | 64.88s | natural summary token embeddings 可读。 |
| `msd_digits` | 66.4% | 56.2% | 56.2% | 66.78s | digit-list schema 仍不稳定。 |

B2.9 `final_abs` 代表输出：

```text
embedding("final_sign=positive\nfinal_abs=116\n...Final integer:") -> 116
embedding("final_sign=positive\nfinal_abs=1221\n...Final integer:") -> 1221
```

B2.9 `msd_digits` 仍有和 B2.8a 类似的问题：

```text
expected=423  answer=123
expected=601  answer=61
expected=306  answer=36
expected=1161 answer=116
```

落盘文件（远端）：

- `needle_sq_results/latent_projector_b29_embedding_final_abs_oracle.json`
- `needle_sq_results/latent_projector_b29_embedding_final_abs_oracle.pt`
- `needle_sq_results/latent_projector_b29_embedding_natural_summary_oracle.json`
- `needle_sq_results/latent_projector_b29_embedding_natural_summary_oracle.pt`
- `needle_sq_results/latent_projector_b29_embedding_msd_digits_oracle.json`
- `needle_sq_results/latent_projector_b29_embedding_msd_digits_oracle.pt`

#### 47.27.3 结论

B2.9 是正结果：**canonical KV text 的 token embeddings 可以 100% 被 frozen LLM decode 读取**。

这排除了一个重要疑点：`inputs_embeds` 拼接、position、attention mask、dtype 路径没有根本问题。B2.6 失败不是因为 embeddings 输入机制不可用，而是因为自由 MLP bridge 没有落到 LLM 已知的 canonical text embedding manifold 上。

更新后的判断：下一步应该做 B2.10：训练 bridge 对齐 canonical KV token embeddings，而不是直接用 answer CE。

```text
B2.10:
  structured state features
    -> bridge(latent token embeddings)
    ~ MSE/cosine to embedding("final_abs=<int>, final_sign=<sign>\nFinal integer:")
    -> frozen LLM decode
```

关键评估：

```text
1. bridge embedding MSE/cosine 是否收敛；
2. auxiliary probe 能否从 bridge latent 重构 final_abs/sign；
3. greedy decode 是否接近 B2.9 oracle 上界 100%。
```

### 47.28 B2.10 canonical embedding bridge（2026-06-23）

承接 §47.27，本轮训练一个 bridge，把结构化 arithmetic state features 直接映射成 canonical text 的 token embedding 序列，而不是用 answer CE 端到端训练：

```text
structured arithmetic state features
  -> bridge(latent token embeddings)
  ~ MSE + cosine to embedding(canonical final_abs text)
  -> frozen Qwen3-0.6B greedy decode
```

这一步检验：B2.6 的失败是否只是 bridge 没有落到 LLM 熟悉的 embedding manifold；如果显式监督到 canonical token embeddings，frozen LLM 是否能重新读出答案。

#### 47.28.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 canonical suffix embedding target 构造：`_canonical_suffix_ids()`、`_max_canonical_suffix_len()`。
- 新增 MSE/cosine 对齐训练：`train_canonical_embedding_bridge()`。
- 新增 nearest-token retrieval 评估：`evaluate_canonical_embedding_retrieval()`。
- 新增 bridge latent decode：`generate_hf_with_canonical_embedding_bridge()`、`evaluate_canonical_embedding_bridge_decode()`。
- 新增 `run_canonical_embedding_bridge()` 和 CLI：

```text
--canonical-embedding-bridge
--canonical-embedding-mode {final_abs,natural_summary,msd_digits}
--canonical-embedding-cosine-weight 1.0
--canonical-embedding-target-tokens 0
--bridge-supervision-weight 1.0
```

本轮还补了两个入口问题：

1. parser/main 未挂 `--canonical-embedding-bridge`；
2. `args.canonical-embedding_target_tokens` 拼写会运行时失败，修为 `args.canonical_embedding_target_tokens`。

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help >/dev/null
```

#### 47.28.2 B2.10 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
canonical_embedding_bridge=true
task=arithmetic
train_cases=256
eval_cases=64
canonical_embedding_mode=final_abs
bridge_supervision_weight=1.0
epochs=500
batch_size=16
lr=1e-3
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=1
```

输出文件：

- `needle_sq_results/latent_projector_b210_canonical_embedding_bridge.json`
- `needle_sq_results/latent_projector_b210_canonical_embedding_bridge.pt`

#### 47.28.3 B2.10 结果

训练收敛：

| epoch | loss | MSE | cosine distance |
|---:|---:|---:|---:|
| 1 | 0.232238 | 0.030543 | 0.201695 |
| 50 | 0.001938 | 0.000286 | 0.001652 |
| 100 | 0.001159 | 0.000158 | 0.001000 |
| 200 | 0.000852 | 0.000116 | 0.000737 |
| 300 | 0.000706 | 0.000082 | 0.000623 |
| 400 | 0.000371 | 0.000034 | 0.000337 |
| 500 | 0.000365 | 0.000029 | 0.000336 |

Embedding retrieval：

| split | tokens | nearest-token acc | MSE | cosine distance |
|---|---:|---:|---:|---:|
| train | 7944 | 100.00% | 0.0000348 | 0.0002926 |
| eval | 1984 | 99.40% | 0.0000401 | 0.0023991 |

Frozen LLM decode：

| setting | accuracy | contains accuracy | misses |
|---|---:|---:|---:|
| B2.6 oracle structured-state decode bridge | 6.25% | 6.25% | 60 / 64 |
| B2.9 embedding oracle upper bound (`final_abs`) | 100.00% | 100.00% | 0 / 128 |
| B2.10 canonical embedding bridge (`final_abs`) | 90.625% | 90.625% | 6 / 64 |

B2.10 的 6 个 eval miss：

```text
expected=85    answer=58
expected=63    answer=36
expected=1221  answer=1216
expected=1179  answer=1732
expected=74    answer=87
expected=1101  answer=1116
```

代表性正确输出：

```text
expected=116 answer=116 raw=' 116\nAnswer: 116\nAnswer:\n11'
expected=423 answer=423 raw=' 423\nAnswer: 423\nAnswer:\n42'
expected=794 answer=794 raw=' 794\nAnswer: 794\nAnswer:\n79'
```

#### 47.28.4 结论

B2.10 是明显正结果：**把 structured state bridge 显式监督到 canonical token embedding manifold 后，frozen LLM decode 从 B2.6 的 6.25% 提升到 90.625%**。

这说明：

1. B2.6 的主要失败不是 `inputs_embeds` 路径，也不是 frozen LLM 不能读 state，而是自由 MLP bridge 没有稳定落到可读 embedding manifold。
2. MSE/cosine 到 canonical token embeddings 能泛化到 eval：eval nearest-token retrieval 达到 99.40%。
3. 但 decode 仍低于 B2.9 oracle 的 100%，说明即使 nearest-token 几乎正确，连续 embedding 的小偏差仍会在部分边界 case 上触发 digit substitution / transposition。
4. 下一步不应回到 answer CE，而应继续收紧 bridge 的 manifold 约束，例如：nearest-token CE、embedding normalization / tying、先 decode bridge latent 到 nearest canonical tokens 再输入 LLM、或训练 discrete bottleneck。

### 47.29 B2.11 nearest-token CE + discrete bottleneck（2026-06-23）

承接 §47.28，本轮做两个收紧约束：

```text
B2.10 loss: MSE + cosine
B2.11 loss: MSE + cosine + λ_ce * nearest-token CE

B2.10 decode: prompt_embeds + continuous bridge latents
B2.11 decode:
  continuous: prompt_embeds + bridge latents
  nearest:    prompt_embeds + embedding(argmax cosine(bridge latent, embedding table))
```

目标是区分两个失败源：

1. 如果 `nearest` 明显优于 `continuous`，说明主要是连续 embedding 小偏差导致 LLM decode 不稳；
2. 如果 `nearest` 仍不提升，说明 bridge 在 eval 上已经预测到了错误 canonical token，问题在 structured features -> token identity 的泛化。

#### 47.29.1 代码改动

文件：`tools/train_latent_projector.py`

- 新增 `_canonical_embedding_token_ce_loss()`：对 normalized bridge latent 与 normalized embedding table 做 cosine logits，再对 canonical target token ids 做 CE。
- `train_canonical_embedding_bridge()` 增加：

```text
--canonical-embedding-token-ce-weight
--canonical-embedding-token-ce-temperature
```

- `generate_hf_with_canonical_embedding_bridge()` 增加 discrete bottleneck：

```text
--canonical-embedding-decode-mode {continuous,nearest,both}
```

其中 `nearest` 会先把每个 latent slot snap 到最近 tokenizer embedding，再喂给 frozen LLM。

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help >/dev/null
git diff --check
```

#### 47.29.2 B2.11 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
canonical_embedding_bridge=true
task=arithmetic
train_cases=256
eval_cases=64
canonical_embedding_mode=final_abs
bridge_supervision_weight=1.0
canonical_embedding_cosine_weight=1.0
canonical_embedding_token_ce_weight=0.1
canonical_embedding_token_ce_temperature=0.05
canonical_embedding_decode_mode=both
epochs=500
batch_size=16
lr=1e-3
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=1
```

输出文件：

- `needle_sq_results/latent_projector_b211_canonical_embedding_bridge_ce_nearest.json`
- `needle_sq_results/latent_projector_b211_canonical_embedding_bridge_ce_nearest.pt`

#### 47.29.3 B2.11 结果

训练收敛：

| epoch | loss | MSE | cosine distance | token CE | train token CE acc |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.466205 | 0.052134 | 0.225997 | 1.880732 | 80.84% |
| 50 | 0.006670 | 0.000406 | 0.002315 | 0.039480 | 100.00% |
| 100 | 0.006520 | 0.000290 | 0.002295 | 0.039345 | 100.00% |
| 500 | 0.005092 | 0.000037 | 0.001153 | 0.039023 | 100.00% |

Embedding retrieval：

| split | tokens | nearest-token acc | MSE | cosine distance |
|---|---:|---:|---:|---:|
| train | 7944 | 100.00% | 0.0000348 | 0.0010868 |
| eval | 1984 | 99.50% | 0.0000400 | 0.0033303 |

Frozen LLM decode：

| setting | accuracy | contains accuracy | misses |
|---|---:|---:|---:|
| B2.10 continuous | 90.625% | 90.625% | 6 / 64 |
| B2.11 continuous + CE | 90.625% | 90.625% | 6 / 64 |
| B2.11 nearest + CE | 90.625% | 90.625% | 6 / 64 |

B2.11 continuous miss：

```text
expected=85    answer=88
expected=63    answer=16
expected=1221  answer=1214
expected=1179  answer=1730
expected=74    answer=87
expected=1101  answer=1010
```

B2.11 nearest miss：

```text
expected=85    answer=88
expected=63    answer=16
expected=1221  answer=1216
expected=1179  answer=1749
expected=74    answer=77
expected=1101  answer=1110
```

#### 47.29.4 结论

B2.11 是负结果，但信息量很高：**nearest-token CE 和 discrete bottleneck 没有把 90.625% 推到 100%**。

具体判断：

1. 训练集 token CE acc 很快到 100%，说明训练集 canonical token identity 可以被 bridge 完全拟合。
2. eval nearest-token retrieval 只从 B2.10 的 99.40% 小幅到 99.50%，decode 不变。
3. `nearest` decode 没有优于 `continuous`，因此 B2.10 剩余错误不是单纯连续 embedding jitter；错误更像是 bridge 在少数 eval 样本上把数字 token identity 预测成了另一个合法数字 token。
4. 这说明 canonical text embedding bridge 仍在做“从结构化 features 到文本 token identity 的插值/分类”，它不是算法式符号组合；对 held-out 数值组合仍会出现 digit substitution。

下一步不应该继续只调 CE 权重。更合理的是 B2.12：把输出 token 化问题拆开，显式预测结构化 decimal digits / sign，再用 deterministic renderer 生成 canonical text token ids，最后喂 frozen LLM。这相当于把 bridge 的可学习部分限制在 slots 上，而不是直接学习整段 canonical text embedding。

### 47.30 B2.12 explicit digit slots + deterministic renderer（2026-06-23）

承接 §47.29，本轮先做 oracle upper-bound sanity check：不让 MLP 学整段 canonical token identity，而是从结构化 arithmetic state 得到明确的 decimal digit slots / sign，然后用 deterministic renderer 拼出 canonical text，再喂 frozen LLM。

```text
structured arithmetic state
  -> final_sign + final_digits_msd/final_digits_lsd
  -> deterministic renderer: final_abs=<digits>, final_sign=<sign>
  -> tokenizer / embedding table
  -> frozen Qwen3-0.6B decode
```

这一步回答：如果 slot 是正确的，deterministic renderer 是否能消除 B2.10/B2.11 的 90.625% 上限。

#### 47.30.1 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `_final_digit_slots_from_case()`：把 arithmetic final value 拆成 explicit digit slots：

```text
final_sign
final_digits_lsd
final_digits_msd
rendered_abs
```

- `_digit_slot_renderer_suffix()`：从 digit slots deterministic render 出 canonical suffix：

```text
Rendered structured digit slots:
final_sign=positive
final_abs=116
The final_abs field was deterministically rendered from explicit MSD digit slots.
Use final_abs as the magnitude and apply final_sign.
Final integer:
```

- `generate_hf_digit_slot_renderer()` / `evaluate_digit_slot_renderer()`。
- `run_digit_slot_renderer_eval()` 和 CLI：

```text
--digit-slot-renderer-eval
--digit-slot-renderer-input-mode {input_ids,inputs_embeds,both}
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help >/dev/null
git diff --check
```

#### 47.30.2 B2.12 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
digit_slot_renderer_eval=true
digit_slot_renderer_input_mode=both
task=arithmetic
eval_cases=128
num_digits=4
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=1
```

输出文件：

- `needle_sq_results/latent_projector_b212_digit_slot_renderer.json`
- `needle_sq_results/latent_projector_b212_digit_slot_renderer.pt`

#### 47.30.3 B2.12 结果

| input mode | eval cases | accuracy | contains accuracy | misses |
|---|---:|---:|---:|---:|
| `input_ids` | 128 | 100.0% | 100.0% | 0 |
| `inputs_embeds` | 128 | 100.0% | 100.0% | 0 |

代表性 slot：

```text
expected=116
final_value=116
final_sign=positive
final_digits_lsd=[6,1,1,0]
final_digits_msd=[0,1,1,6]
rendered_abs=116
```

#### 47.30.4 结论

B2.12 是正结果：**explicit digit slots + deterministic renderer 可以在 `input_ids` 和 `inputs_embeds` 两条路径都达到 100%**。

这说明 B2.10/B2.11 的 90.625% 缺口不在 frozen LLM，也不在 tokenizer/embedding path，而在 “MLP bridge 直接学习 canonical text token identity” 这个接口。只要把 token identity 交给 deterministic renderer，LLM decode 立即回到 100%。

更新后的判断：下一步应该做 B2.13，不再训练 bridge 输出整段 text embedding，而是训练/约束 bridge 输出 explicit slots：

```text
structured state features
  -> slot predictor: sign + fixed-width decimal digits
  -> deterministic renderer
  -> frozen LLM
```

评估重点：

1. slot predictor 的 per-digit / exact-value 泛化；
2. renderer 后 frozen LLM decode 是否跟随 slot exact accuracy；
3. 如果 trainable slot predictor 仍在 held-out 数值组合失败，则需要回到 B2.5-style primitive lookup / algorithmic digit operators，而不是更大的 MLP。

### 47.31 B2.13 trainable slot predictor + deterministic renderer（2026-06-23）

承接 §47.30，本轮把 B2.12 的 oracle slots 换成 trainable slot predictor：

```text
oracle structured state features
  -> DigitSlotPredictor(MLP)
  -> final_sign + fixed-width final_digits_msd
  -> deterministic renderer(final_abs=<digits>, final_sign=<sign>)
  -> frozen Qwen3-0.6B decode
```

这一步检验：即使不让模型直接学习整段 text embedding/token identity，只让普通 MLP 学 explicit slots，它是否能在 held-out arithmetic cases 上稳定泛化。

#### 47.31.1 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `DigitSlotPredictor`：从 `_oracle_structured_decode_features()` 输出 sign logits 和 fixed-width MSD digit logits。
- `_digit_slot_targets()`：构造 `sign` 与 `final_digits_msd` 监督。
- `train_digit_slot_predictor()`：slot CE 训练。
- `evaluate_digit_slot_predictor()`：统计 sign accuracy、per-digit accuracy、exact slot accuracy。
- `generate_hf_with_digit_slot_predictor()` / `evaluate_digit_slot_predictor_decode()`：用预测 slots deterministic render 后喂 frozen LLM。
- `run_digit_slot_predictor()` 和 CLI：

```text
--digit-slot-predictor
--digit-slot-predictor-hidden-size 256
--digit-slot-predictor-depth 2
--digit-slot-digit-loss-weight 1.0
--digit-slot-decode-input-mode {input_ids,inputs_embeds,both}
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help >/dev/null
git diff --check
```

#### 47.31.2 B2.13 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
digit_slot_predictor=true
task=arithmetic
train_cases=256
eval_cases=64
num_digits=4
feature_dim=200
digit_slot_predictor_hidden_size=256
digit_slot_predictor_depth=2
epochs=500
batch_size=16
lr=1e-3
weight_decay=0.0
digit_slot_decode_input_mode=both
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=1
```

输出文件：

- `needle_sq_results/latent_projector_b213_digit_slot_predictor.json`
- `needle_sq_results/latent_projector_b213_digit_slot_predictor.pt`

#### 47.31.3 B2.13 结果

训练收敛：

| epoch | loss | sign acc | per-digit acc | exact slot acc |
|---:|---:|---:|---:|---:|
| 1 | 2.348805 | 100.0% | 31.35% | 0.0% |
| 50 | 0.001882 | 100.0% | 100.0% | 100.0% |
| 500 | 0.000003 | 100.0% | 100.0% | 100.0% |

Slot predictor：

| split | sign acc | per-digit acc | exact slot acc | misses |
|---|---:|---:|---:|---:|
| train | 100.0% | 100.0% | 100.0% | 0 / 256 |
| eval | 100.0% | 99.22% | 96.875% | 2 / 64 |

Eval slot miss：

```text
expected=626 target_digits=[0,6,2,6] pred_digits=[0,6,4,6] pred_abs=646
expected=857 target_digits=[0,8,5,7] pred_digits=[0,8,6,7] pred_abs=867
```

Renderer + frozen LLM decode：

| input mode | slot exact | decode acc | contains acc | misses |
|---|---:|---:|---:|---:|
| `input_ids` | 96.875% | 96.875% | 96.875% | 2 / 64 |
| `inputs_embeds` | 96.875% | 96.875% | 96.875% | 2 / 64 |

Decode miss 完全跟随 slot miss：

```text
expected=626 pred_abs=646 answer=646
expected=857 pred_abs=867 answer=867
```

#### 47.31.4 结论

B2.13 是关键负结果：**即使把输出接口从 text embedding/token identity 收窄成 explicit digit slots，普通 MLP slot predictor 仍然不能在 held-out arithmetic cases 上稳定到 100%**。

但它也确认了一个重要分解：

1. deterministic renderer + frozen LLM 没有额外损失；decode accuracy 精确等于 slot exact accuracy。
2. 剩余错误完全来自 trainable slot predictor 的 digit substitution。
3. 这与 B2.4 的结论一致：普通 MLP 可以记住训练组合，但不会稳定学习算法式 digit/carry 规则。

更新后的判断：这条 B 线的下一步不应该继续加大 MLP，而应该转向 B2.14：

```text
structured state / operands
  -> algorithmic digit operators or primitive lookup
  -> explicit slots
  -> deterministic renderer
  -> frozen LLM
```

也就是把 hidden-state reasoning 里的 transition 做成可组合的局部算法算子，而不是让通用 MLP 从样本中“归纳”十进制规则。

### 47.32 B2.14 primitive digit operators + deterministic renderer（2026-06-23）

承接 §47.31，本轮把 B2.13 的 trainable MLP slot predictor 换成 B2.5-style primitive digit operators：

```text
operands / structured state
  -> primitive lookup digit operators: add / mul / sub with carry/borrow
  -> explicit final_sign + final_digits_msd/final_digits_lsd
  -> deterministic renderer(final_abs=<digits>, final_sign=<sign>)
  -> frozen Qwen3-0.6B decode
```

这一步验证完整端到端路径：如果 transition 是算法式局部算子，而不是通用 MLP，slot 和 decode 是否都回到 100%。

#### 47.32.1 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `_primitive_digit_operator_slots_from_case()`：调用 `PrimitiveLookupDigitWiseOperator`，经 `_build_trainable_digit_arithmetic_trace()` 生成 final slots。
- `generate_hf_with_primitive_digit_renderer()`：把 primitive slots deterministic render 后喂给 frozen LLM。
- `evaluate_primitive_digit_renderer()`：统计 slot exact、decode、contains、overflow。
- `run_primitive_digit_renderer_eval()` 和 CLI：

```text
--primitive-digit-renderer-eval
--primitive-digit-renderer-input-mode {input_ids,inputs_embeds,both}
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help >/dev/null
git diff --check
```

#### 47.32.2 B2.14 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
primitive_digit_renderer_eval=true
primitive_digit_renderer_input_mode=both
task=arithmetic
eval_cases=128
num_digits=4
operator=PrimitiveLookupDigitWiseOperator
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=1
```

输出文件：

- `needle_sq_results/latent_projector_b214_primitive_digit_renderer.json`
- `needle_sq_results/latent_projector_b214_primitive_digit_renderer.pt`

#### 47.32.3 B2.14 结果

| input mode | eval cases | slot exact | decode acc | contains acc | overflow | misses |
|---|---:|---:|---:|---:|---:|---:|
| `input_ids` | 128 | 100.0% | 100.0% | 100.0% | 0 | 0 |
| `inputs_embeds` | 128 | 100.0% | 100.0% | 100.0% | 0 | 0 |

代表性 slot：

```text
expected=116
final_sign=positive
final_digits_lsd=[6,1,1,0]
final_digits_msd=[0,1,1,6]
rendered_abs=116
overflow=False
```

#### 47.32.4 结论

B2.14 是正结果：**primitive lookup digit operators + deterministic renderer + frozen LLM 完整路径达到 100%**。

这把 B2.10-B2.14 的失败边界切清楚了：

| variant | transition / state-to-output | eval decode |
|---|---|---:|
| B2.10 | MLP -> canonical embeddings | 90.625% |
| B2.11 | MLP + nearest-token CE/discrete bottleneck | 90.625% |
| B2.12 | oracle slots + deterministic renderer | 100.0% |
| B2.13 | trainable MLP slot predictor + renderer | 96.875% |
| B2.14 | primitive algorithmic digit operators + renderer | 100.0% |

最终判断：这条 hidden-state reasoning 的关键不是“连续 latent 能不能喂给 LLM”，而是 latent transition 是否具备算法式组合结构。普通 MLP 在小样本/held-out 数值组合下会记忆或插值，不能稳定学会十进制规则；而 explicit carry/borrow slot + primitive local operator 可以稳定泛化。

后续如果继续推进，应把研究方向从“训练一个大 MLP transition”改成：

```text
hidden state slots
  + typed local operators
  + deterministic / constrained renderer
```

也就是让模型在 hidden state 内部调用受约束的小算子，而不是让它自由生成下一步 hidden state。

### 47.33 C1 arithmetic operator VM token-saving benchmark（2026-06-23）

B 线已经证明 arithmetic hidden-state transition 的正确形态不是 free MLP，而是 typed local operator + slots + constrained renderer。本轮转向 C 线：不再继续做机制验证，而是直接量化这种结构化路径在推理侧能省多少 decode token / latency / KV growth。

#### 47.33.1 设计

对同一批 arithmetic prompts 比较两条路径：

```text
baseline:
  prompt -> Qwen3-8B greedy decode visible CoT -> final answer

operator_vm:
  prompt -> oracle operands/parser -> primitive digit operators
         -> deterministic slot renderer -> Qwen3-8B decode final answer only
```

这里 operator VM 复用 B2.14 的 `PrimitiveLookupDigitWiseOperator` 和 deterministic renderer；C1 只测试 inference acceleration，不把 parser 误差混进来。

#### 47.33.2 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `_generate_hf_greedy_timed()`：HF greedy decode，统计 generated token 数和 latency，并在读到 expected final answer 后停止。
- `evaluate_operator_vm_benchmark()`：逐 case 统计 baseline 与 operator VM 的 accuracy、generated tokens、latency、prefill/KV token 数、operator overhead。
- `run_operator_vm_benchmark()` 和 CLI：

```text
--operator-vm-benchmark
--operator-vm-input-mode {input_ids,inputs_embeds}
--operator-vm-baseline-max-new-tokens N
--operator-vm-final-max-new-tokens N
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help | grep operator-vm
```

#### 47.33.3 C1 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
task=arithmetic
eval_cases=32
context_len=512
baseline_max_new_tokens=128
operator_vm_final_max_new_tokens=8
operator_vm_input_mode=inputs_embeds
num_digits=4
hf_dtype=bfloat16
GPU=A100, CUDA_VISIBLE_DEVICES=7
```

输出文件：

- `needle_sq_results/latent_projector_c1_operator_vm_benchmark.json`
- `needle_sq_results/latent_projector_c1_operator_vm_benchmark.pt`

#### 47.33.4 C1 结果

核心 JSON：

```json
{
  "baseline": {
    "accuracy": 1.0,
    "avg_generated_tokens": 83.46875,
    "avg_latency_s": 6.597350515425205,
    "avg_total_kv_tokens": 594.6875
  },
  "operator_vm": {
    "accuracy": 1.0,
    "avg_generated_tokens": 4.0,
    "avg_latency_s": 0.33683500438928604,
    "operator_overhead_s": 0.011409908533096313,
    "avg_total_kv_tokens": 562.21875
  },
  "speedup": {
    "latency": 19.586297235902872,
    "generated_tokens": 20.8671875,
    "total_kv_tokens": 1.0577510977711078
  },
  "token_saving": {
    "generated_token_reduction": 0.9520778734556345,
    "total_kv_token_reduction": 0.05459800315291641
  }
}
```

表格：

| path | accuracy | avg generated tokens | avg latency | avg total KV tokens |
|---|---:|---:|---:|---:|
| baseline CoT | 100.0% | 83.47 | 6.597s | 594.69 |
| operator VM | 100.0% | 4.00 | 0.337s | 562.22 |

相对收益：

| metric | result |
|---|---:|
| generated token reduction | 95.21% |
| generated-token speedup | 20.87× |
| latency speedup | 19.59× |
| LLM-only latency speedup | 20.27× |
| total KV token reduction | 5.46% |
| average operator overhead | 0.0114s |

代表性 case：

```text
expected=116
baseline: generated_tokens=90, latency=7.161s, answer=116
operator_vm: generated_tokens=4, latency=0.452s, answer=116
rendered_abs=116, final_digits_msd=[0,1,1,6]
```

#### 47.33.5 结论

C1 是明确正结果：在相同 100% accuracy 下，operator VM 把 visible CoT 的平均 decode token 从 83.47 降到 4.00，decode-token 降幅 95.2%，端到端 latency 提升 19.6×。

需要注意，当前 operator VM 把 deterministic rendered suffix 作为 prompt-side context 注入，因此总 KV token 只下降 5.46%；主要收益来自**不再逐 token decode 可见推理链**。这说明 C 线的直接价值是 latency / decode budget / decode KV growth，而下一步若要继续扩大 KV savings，应把 slot state 更紧凑地注入为 hidden slots 或 typed latent operators，而不是渲染成 47 个文本 token。

### 47.34 C2.1 minimal embedding suffix upper bound（2026-06-23）

C1 的瓶颈已经很明确：visible CoT decode 被省掉了，但 deterministic renderer 仍然把 slot state 渲染成约 47 个 prompt-side text tokens。C2.1 先做最便宜的 upper bound：不训练 bridge，只把 final_abs / sign / signed answer 这类 canonical token embeddings 作为极短 suffix 注入，验证 frozen Qwen3-8B 是否仍能 100% 输出最终答案。

#### 47.34.1 设计

同一批 arithmetic prompts 比较：

```text
baseline:
  prompt -> Qwen3-8B greedy visible CoT -> final answer

C1 text VM:
  prompt -> primitive digit operators
         -> 47-token deterministic textual renderer
         -> Qwen3-8B final answer

C2.1 minimal embedding suffix:
  prompt -> primitive digit operators
         -> short canonical token embeddings, e.g.
            "\nfinal_abs=116\nFinal integer:"
         -> Qwen3-8B final answer
```

测试的 minimal suffix modes：

| mode | suffix template | avg injected tokens |
|---|---|---:|
| `final_abs` | `\nfinal_abs=<abs>\nFinal integer:` | 11.0 |
| `sign_final_abs` | `\nfinal_sign=<sign>\nfinal_abs=<abs>\nFinal integer:` | 16.0 |
| `signed_answer` | `\nanswer=<signed>\nFinal integer:` | 10.0 |
| `digits_msd` | `\nsign=<sign>\ndigits_msd=<digits>\nFinal integer:` | 17.0 |

#### 47.34.2 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `_minimal_embedding_suffix()`：把 primitive operator 的 final slots 压缩成短 canonical suffix。
- `evaluate_minimal_embedding_suffix_benchmark()`：同时评测 baseline、C1 text VM、多个 C2.1 minimal suffix modes。
- `run_minimal_embedding_suffix_benchmark()` 和 CLI：

```text
--minimal-embedding-suffix-benchmark
--minimal-embedding-suffix-modes {final_abs,sign_final_abs,signed_answer,digits_msd} ...
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help | grep -E "minimal-embedding|operator-vm"
```

#### 47.34.3 C2.1 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
task=arithmetic
eval_cases=32
context_len=512
baseline_max_new_tokens=128
operator_vm_final_max_new_tokens=8
operator_vm_input_mode=inputs_embeds
minimal_embedding_suffix_modes=final_abs sign_final_abs signed_answer digits_msd
num_digits=4
hf_dtype=bfloat16
GPU=A100, CUDA_VISIBLE_DEVICES=7
```

输出文件：

- `needle_sq_results/latent_projector_c21_minimal_embedding_suffix.json`
- `needle_sq_results/latent_projector_c21_minimal_embedding_suffix.pt`

#### 47.34.4 C2.1 结果

核心 JSON：

```json
{
  "baseline": {
    "accuracy": 1.0,
    "avg_generated_tokens": 83.46875,
    "avg_total_kv_tokens": 594.6875
  },
  "operator_vm_text": {
    "accuracy": 1.0,
    "avg_generated_tokens": 4.0,
    "avg_injected_tokens": 47.0,
    "avg_total_kv_tokens": 562.21875
  },
  "minimal_embedding_suffix": {
    "final_abs": {
      "accuracy": 1.0,
      "avg_generated_tokens": 4.0,
      "avg_injected_tokens": 11.0,
      "avg_total_kv_tokens": 526.21875
    },
    "sign_final_abs": {
      "accuracy": 1.0,
      "avg_generated_tokens": 4.0,
      "avg_injected_tokens": 16.0,
      "avg_total_kv_tokens": 531.21875
    },
    "signed_answer": {
      "accuracy": 1.0,
      "avg_generated_tokens": 4.0,
      "avg_injected_tokens": 10.0,
      "avg_total_kv_tokens": 525.21875
    },
    "digits_msd": {
      "accuracy": 0.375,
      "avg_generated_tokens": 5.125,
      "avg_injected_tokens": 17.0,
      "avg_total_kv_tokens": 533.34375
    }
  }
}
```

表格：

| path | accuracy | injected tokens | generated tokens | total KV tokens |
|---|---:|---:|---:|---:|
| baseline CoT | 100.0% | 0.0 | 83.47 | 594.69 |
| C1 text VM | 100.0% | 47.0 | 4.00 | 562.22 |
| C2.1 `final_abs` | 100.0% | 11.0 | 4.00 | 526.22 |
| C2.1 `sign_final_abs` | 100.0% | 16.0 | 4.00 | 531.22 |
| C2.1 `signed_answer` | 100.0% | 10.0 | 4.00 | 525.22 |
| C2.1 `digits_msd` | 37.5% | 17.0 | 5.13 | 533.34 |

相对 baseline：

| mode | generated token reduction | total KV reduction | total KV speedup |
|---|---:|---:|---:|
| C1 text VM | 95.21% | 5.46% | 1.058× |
| C2.1 `final_abs` | 95.21% | 11.51% | 1.130× |
| C2.1 `sign_final_abs` | 95.21% | 10.67% | 1.119× |
| C2.1 `signed_answer` | 95.21% | 11.68% | 1.132× |
| C2.1 `digits_msd` | 93.86% | 10.32% | 1.115× |

相对 C1 text VM：

| mode | injected token reduction | total KV reduction |
|---|---:|---:|
| `final_abs` | 76.6% | 6.40% |
| `sign_final_abs` | 66.0% | 5.51% |
| `signed_answer` | 78.7% | 6.58% |
| `digits_msd` | 63.8% | 5.14% |

代表性 case：

```text
expected=116
C1 text suffix: injected=47, answer=116
C2.1 final_abs suffix: "\nfinal_abs=116\nFinal integer:", injected=11, answer=116
C2.1 sign_final_abs suffix: injected=16, answer=116
C2.1 signed_answer suffix: injected=10, answer=116
```

`digits_msd` 的失败具有诊断意义：直接给 `digits_msd=0085` / `digits_msd=0423` 这类 slot string，frozen LLM 不稳定理解“去 leading zero 后拼接”，会输出错误数字（例如 `85 -> 100`、`423 -> 1083`）。这说明仅给原始 digit slots 还不够；LLM 需要 final_abs 这种已经 deterministic rendered 的 compact value，或者后续需要 constrained renderer/slot-aware bridge。

#### 47.34.5 结论

C2.1 是正结果：**C1 的 47-token textual renderer 可以压缩到 10-11 个 canonical embedding tokens，并保持 100% accuracy**。

这把 C 线下一步边界切得更清楚：

1. 如果 bridge 直接产出 `final_abs` 或 `signed_answer` 风格的 compact latent record，frozen Qwen3-8B 可以稳定读出最终答案。
2. 如果只暴露 raw digit slots（`digits_msd`），frozen LLM 不会自动稳定执行 leading-zero stripping / digit concatenation；这部分仍需要 deterministic/constrained renderer 或显式训练 slot-to-latent bridge。
3. C2.2 因此应该训练的是：

```text
(final_sign, final_digits_msd)
  -> compact final_abs / signed-answer latent embeddings
  -> frozen LLM final answer
```

而不是让 MLP 重新学习 arithmetic transition。C2.2 的目标是压缩 renderer，不是学习算术。

### 47.35 C2.2 learned compact renderer bridge（2026-06-23）

C2.1 证明了一个 oracle upper bound：如果直接注入 canonical suffix token embeddings，`signed_answer` 只需要约 10 个 injected tokens 就能保持 100% accuracy。C2.2 继续验证：能否从 explicit slots 学出这个 compact renderer bridge。

#### 47.35.1 设计

C2.2 明确不是 arithmetic learner；arithmetic transition 仍由 primitive digit operators 完成，bridge 只做 renderer compression：

```text
final_sign + final_digits_msd
  -> learned compact renderer bridge
  -> latent embeddings
  -> frozen Qwen3-8B final answer only
```

分两组：

```text
C2.2-a variable-length canonical bridge:
  slots -> same length as "\nanswer=<signed_answer>\nFinal integer:"
  objective = MSE/cosine/token-CE to canonical token embeddings

C2.2-b fixed-K compressed bridge:
  slots -> K latent embeddings, K ∈ {8,4,2,1}
  objective = answer CE through frozen Qwen3-8B
```

成功标准沿用前面定义：

```text
accuracy = 100%
avg injected slots < 10
total KV < C2.1 signed_answer 的 525.22
```

#### 47.35.2 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `CompactRendererBridge`：从 one-hot `final_sign + final_digits_msd` 输出 latent embeddings。
- `_compact_renderer_slot_features()`：构造 slot feature，维度为 `2 + num_digits * 10 = 42`。
- `train_compact_renderer_bridge()`：支持 `alignment` 与 `answer_ce` 两类目标。
- `evaluate_compact_renderer_bridge_decode()`：把 bridge 输出的 latent embeddings 注入 frozen LLM，统计 decode/latency/KV。
- `run_compact_renderer_bridge()` 和 CLI：

```text
--compact-renderer-bridge
--compact-renderer-run-variable
--compact-renderer-fixed-ks 8 4 2 1
--compact-renderer-hidden-size 1024
--compact-renderer-token-ce-weight 0.1
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help | grep compact-renderer
git diff --check
```

#### 47.35.3 C2.2-a variable bridge 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
train_cases=128
eval_cases=32
epochs=300
batch_size=16
lr=5e-4
objective=alignment
compact_renderer_token_ce_weight=0.1
max_suffix_tokens=11
feature_dim=42
hidden_size=4096
GPU=A100, CUDA_VISIBLE_DEVICES=7
```

输出文件：

- `needle_sq_results/latent_projector_c22_compact_renderer_bridge_variable.json`
- `needle_sq_results/latent_projector_c22_compact_renderer_bridge_variable.pt`

训练收敛：

| epoch | loss | token CE acc |
|---:|---:|---:|
| 1 | 1.0281 | 61.6% |
| 30 | 0.0108 | 99.9% |
| 60 | 0.0033 | 100.0% |
| 300 | 0.0009 | 100.0% |

Eval：

| path | accuracy | avg injected latent slots | avg generated tokens | avg total KV units |
|---|---:|---:|---:|---:|
| C2.1 `signed_answer` reference | 100.0% | 10.0 token embeddings | 4.00 | 525.22 |
| C2.2-a variable bridge | 81.25% | 10.0 latent slots | 4.19 | 525.41 |

代表性成功：

```text
expected=116 -> answer=116
expected=423 -> answer=423
expected=343 -> answer=343
```

代表性失败：

```text
expected=85   -> answer=87
expected=63   -> answer=67
expected=1221 -> answer=2115
expected=1179 -> answer=1729
expected=1101 -> answer=1015
```

诊断：训练集 token identity 已经接近/达到 100%，但 held-out eval 上仍出现 digit substitution。这说明当前 slot-to-embedding MLP 会学到 suffix token 的局部模式，但没有稳定泛化到所有 digit combination；这和 B2.13 的 slot predictor 负结果一致，只是错误从 slot prediction 转移到了 renderer embedding prediction。

#### 47.35.4 C2.2-b fixed-K 运行参数与结果

```text
train_cases=64
eval_cases=32
fixed_K={8,4,2,1}
fixed_epochs=5
batch_size=4
objective=answer CE through frozen Qwen3-8B
```

输出文件：

- `needle_sq_results/latent_projector_c22_compact_renderer_bridge_fixed.json`
- `needle_sq_results/latent_projector_c22_compact_renderer_bridge_fixed.pt`

结果：

| K | eval accuracy | avg injected latent slots | avg generated tokens | avg total KV units |
|---:|---:|---:|---:|---:|
| 8 | 3.125% | 8.0 | 4.84 | 524.06 |
| 4 | 3.125% | 4.0 | 6.41 | 521.63 |
| 2 | 0.0% | 2.0 | 5.00 | 518.22 |
| 1 | 3.125% | 1.0 | 5.03 | 517.25 |

代表性 K=8 输出：

```text
expected=116 -> answer=104
expected=85  -> answer=135
expected=423 -> answer=435
expected=108 -> answer=108  # 少量命中
```

固定 K 的 CE 训练在训练 token 上有下降（例如 K=8 answer CE acc 从 36.8% 到 61.6%），但远未形成可泛化的 compact latent protocol。

#### 47.35.5 结论

C2.2 当前是负结果：**learned compact renderer bridge 没有超过 C2.1 `signed_answer` upper bound**。

更具体地说：

1. C2.2-a 能把训练集 canonical token identity 学到 100%，但 eval 只有 81.25%，说明普通 MLP 从 `final_sign + final_digits_msd` 到 canonical embedding 仍会发生 held-out digit substitution。
2. C2.2-b fixed-K 虽然满足 injected slots < 10、total KV < 525.22，但 accuracy 只有 0–3.125%，完全不满足成功标准。
3. 因此 C2.1 的 100% 不是“任意 learned latent bridge 都能做到”，而是 canonical token embeddings 本身提供了强离散先验。

更新后的判断：C2.2 不应该继续简单加大 MLP 或延长 CE 训练。下一步更合理的是引入**离散/约束 renderer bridge**：

```text
final_sign + final_digits_msd
  -> deterministic final_abs / signed_answer token ids
  -> canonical token embedding lookup 或 nearest-token constrained bridge
  -> frozen LLM final answer
```

也就是保持 C2.1 的离散 token identity 先验，同时把文本长度继续压缩；如果要做 learned bridge，也应带 nearest-token / vocabulary bottleneck，而不是 free continuous latent slots。

### 47.36 C2.3 deterministic discrete lookup renderer（2026-06-23）

C2.2 的负结果说明：free continuous MLP bridge 即使在训练集学到 canonical suffix token identity，也不能稳定泛化到 held-out digit combination。C2.3 因此不再加大 MLP，而是把 C2.1 `signed_answer` 的离散 token identity 先验封装成可复用 renderer module。

#### 47.36.1 设计

```text
primitive digit operators
  -> final_sign + final_digits_msd
  -> deterministic signed_answer string
  -> tokenizer.encode("\nanswer=<signed_answer>\nFinal integer:")
  -> embedding lookup
  -> frozen Qwen3-8B final answer only
```

关键点：C2.3 不是 learned arithmetic model，也不是 learned continuous bridge；它是 deterministic discrete renderer。这样保留 C2.1 已验证的 canonical token identity，同时把 renderer 从一次性 benchmark 逻辑封装成独立模块。

#### 47.36.2 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `DiscreteLookupRenderer`：提供 `render_suffix()`、`render_token_ids()`、`render_embeddings()`，执行 slot -> canonical suffix -> token IDs -> embedding lookup。
- `evaluate_discrete_lookup_renderer()`：逐 case 统计 accuracy、slot exact、token identity、injected embedding tokens、generated tokens、KV tokens、operator/renderer/LLM latency。
- `run_discrete_lookup_renderer()` 和 CLI：

```text
--discrete-lookup-renderer
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help | grep discrete-lookup-renderer
```

#### 47.36.3 C2.3 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
task=arithmetic
eval_cases=32
context_len=512
operator_vm_final_max_new_tokens=8
pure_digit_num_digits=4
hf_dtype=bfloat16
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=7
```

输出文件：

- `needle_sq_results/latent_projector_c23_discrete_lookup_renderer.json`
- `needle_sq_results/latent_projector_c23_discrete_lookup_renderer.pt`

#### 47.36.4 C2.3 结果

核心 JSON：

```json
{
  "discrete_lookup_renderer": {
    "accuracy": 1.0,
    "contains_accuracy": 1.0,
    "avg_generated_tokens": 4.0,
    "avg_latency_s": 0.20503485202789307,
    "avg_prefill_tokens": 521.21875,
    "avg_total_kv_tokens": 525.21875,
    "avg_llm_latency_s": 0.187911756336689,
    "operator_overhead_s": 0.0075135305523872375,
    "avg_injected_tokens": 10.0,
    "avg_renderer_overhead_s": 0.009609565138816833,
    "avg_non_llm_overhead_s": 0.01712309569120407,
    "slot_exact_accuracy": 1.0,
    "token_identity_accuracy": 1.0
  },
  "c21_signed_answer_equivalence": {
    "canonical_suffix_mode": "signed_answer",
    "uses_tokenizer_encode_plus_embedding_lookup": true,
    "token_identity_accuracy": 1.0,
    "avg_injected_tokens": 10.0
  }
}
```

对比 C2.1/C2.2：

| path | accuracy | token identity | injected tokens / slots | generated tokens | total KV tokens / units |
|---|---:|---:|---:|---:|---:|
| C2.1 `signed_answer` canonical embeddings | 100.0% | 100.0% | 10.0 tokens | 4.00 | 525.22 |
| C2.2-a variable learned bridge | 81.25% | eval 不稳定 | 10.0 latent slots | 4.19 | 525.41 |
| C2.2-b fixed K=8 learned bridge | 3.125% | N/A | 8.0 latent slots | 4.84 | 524.06 |
| C2.3 discrete lookup renderer | 100.0% | 100.0% | 10.0 tokens | 4.00 | 525.22 |

代表性 case：

```text
expected=116
suffix="\nanswer=116\nFinal integer:"
suffix_token_ids=[198, 9217, 28, 16, 16, 21, 198, 19357, 7546, 25]
slot_hit=True
token_identity_hit=True
answer=116
```

逐 case 结果中 32/32 都满足：

```text
slot_hit=True
token_identity_hit=True
answer == expected
```

#### 47.36.5 结论

C2.3 是正结果：**把 C2.1 的 `signed_answer` canonical embedding path 封装为 deterministic discrete lookup renderer 后，仍保持 100% accuracy、100% token identity、平均 10 injected tokens、平均 total KV 525.22**。

这验证了 C2.2 之后的判断：问题不在 arithmetic operator，而在 renderer bridge 的表示约束。free continuous MLP 会丢失离散 token identity；deterministic / constrained discrete renderer 可以稳定保留它。

因此 C 线当前最稳结构是：

```text
hidden/typed slots
  + primitive local operators
  + deterministic discrete renderer / vocabulary-constrained bridge
```

如果继续压缩到 `<10` KV slots，下一步不应回到 free MLP，而应做 nearest-token / vocabulary bottleneck / typed latent operator 这类保留离散 identity 的受约束 bridge。

### 47.37 C2.4 vocabulary-bottleneck renderer bridge（2026-06-24）

C2.3 证明 deterministic discrete lookup renderer 可以保持 100% token identity。C2.4 继续测试更弱但更接近 learned bridge 的约束：不直接输出 free continuous embeddings，而是让 bridge 在一个小 vocabulary bottleneck 中预测 canonical suffix token ID，再做 embedding lookup。

#### 47.37.1 设计

```text
primitive digit operators
  -> final_sign + final_digits_msd
  -> learned vocabulary-bottleneck bridge
       token_logits[position, candidate_token]
       length_logits
  -> argmax token IDs
  -> embedding lookup
  -> frozen Qwen3-8B final answer only
```

candidate vocabulary 只包含 canonical renderer 需要的 17 个 token：数字 `0..9`、换行、`answer`、`=`、`Final`、` integer`、冒号，以及负号相关 token。

和 C2.2 的区别：C2.4 的 bridge 输出离散 token distribution，不直接输出任意 hidden vectors；因此如果 token ID 预测正确，后续 embedding lookup 与 C2.3 完全一致。

#### 47.37.2 代码改动

文件：`tools/train_latent_projector.py`

新增：

- `_compact_renderer_slot_features_from_slots()`：支持从 primitive operator 输出 slots 构造 feature。
- `VocabBottleneckRendererBridge`：从 `final_sign + final_digits_msd` 输出 per-position candidate-token logits 和 length logits。
- `_vocab_renderer_candidate_token_ids()`：构造 17-token constrained renderer vocabulary。
- `train_vocab_bottleneck_renderer_bridge()`：token CE + length CE 训练。
- `evaluate_vocab_bottleneck_renderer_decode()`：argmax token IDs 后 embedding lookup，再让 frozen Qwen3-8B decode final answer。
- `run_vocab_bottleneck_renderer_bridge()` 和 CLI：

```text
--vocab-bottleneck-renderer-bridge
--vocab-renderer-hidden-size 256
--vocab-renderer-depth 2
--vocab-renderer-length-ce-weight 1.0
```

检查：

```text
python3 -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/train_latent_projector.py
remote: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/train_latent_projector.py --help | grep vocab-bottleneck-renderer
```

#### 47.37.3 C2.4 运行参数

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B
task=arithmetic
train_cases=512
eval_cases=32
context_len=512
epochs=500
batch_size=32
lr=1e-3
vocab_renderer_hidden_size=256
vocab_renderer_depth=2
operator_vm_final_max_new_tokens=8
pure_digit_num_digits=4
hf_dtype=bfloat16
hf_device=cuda
GPU=A100, CUDA_VISIBLE_DEVICES=7
```

输出文件：

- `needle_sq_results/latent_projector_c24_vocab_bottleneck_renderer.json`
- `needle_sq_results/latent_projector_c24_vocab_bottleneck_renderer.pt`

#### 47.37.4 C2.4 结果

训练集收敛到 100% token identity：

```json
{
  "epoch": 500,
  "loss": 0.0000042443,
  "token_ce": 0.0000040992,
  "length_ce": 0.0000001455,
  "token_accuracy": 1.0,
  "length_accuracy": 1.0,
  "sequence_token_accuracy": 1.0,
  "tokens": 5119,
  "seqs": 512
}
```

Eval 核心 JSON：

```json
{
  "vocab_bottleneck_renderer": {
    "accuracy": 0.875,
    "contains_accuracy": 0.875,
    "avg_generated_tokens": 4.125,
    "avg_latency_s": 0.20081572979688644,
    "avg_prefill_tokens": 521.21875,
    "avg_total_kv_tokens": 525.34375,
    "avg_llm_latency_s": 0.19534886628389359,
    "operator_overhead_s": 0.004716977477073669,
    "avg_injected_tokens": 10.0,
    "avg_bridge_overhead_s": 0.0007498860359191895,
    "slot_exact_accuracy": 1.0,
    "length_accuracy": 1.0,
    "token_identity_accuracy": 0.875,
    "token_accuracy": 0.98125
  }
}
```

对比：

| path | eval accuracy | token identity | token accuracy | injected tokens / slots | total KV |
|---|---:|---:|---:|---:|---:|
| C2.1 `signed_answer` canonical embeddings | 100.0% | 100.0% | 100.0% | 10.0 | 525.22 |
| C2.2-a free continuous variable bridge | 81.25% | 不稳定 | N/A | 10.0 | 525.41 |
| C2.3 deterministic discrete lookup | 100.0% | 100.0% | 100.0% | 10.0 | 525.22 |
| C2.4 vocabulary bottleneck learned bridge | 87.5% | 87.5% | 98.125% | 10.0 | 525.34 |

失败 case 都是 digit substitution；长度预测和 primitive slots 都正确：

```text
expected=63   -> predicted_suffix="\nanswer=66\nFinal integer:"   -> answer=66
expected=1221 -> predicted_suffix="\nanswer=1219\nFinal integer:" -> answer=1219
expected=1179 -> predicted_suffix="\nanswer=1029\nFinal integer:" -> answer=1029
expected=74   -> predicted_suffix="\nanswer=77\nFinal integer:"   -> answer=77
```

代表性 `1221`：

```text
target_token_ids    = [198, 9217, 28, 16, 17, 17, 16, 198, 19357, 7546, 25]
predicted_token_ids = [198, 9217, 28, 16, 17, 16, 24, 198, 19357, 7546, 25]
target_suffix       = "\nanswer=1221\nFinal integer:"
predicted_suffix    = "\nanswer=1219\nFinal integer:"
slot_hit=True
length_hit=True
token_identity_hit=False
```

#### 47.37.5 结论

C2.4 是部分正结果但没有达到 C2.3/C2.1：vocabulary bottleneck 明显优于 C2.2 free continuous bridge（87.5% vs 81.25%，并且 token accuracy 到 98.125%），但仍不能在 held-out digit combinations 上保持 100% token identity。

关键诊断：

1. 训练集 512 cases 已经 100% sequence token accuracy，但 eval 仍有 digit substitution。
2. 所有失败 case 的 `slot_hit=True`、`length_hit=True`，错误只发生在 learned slot -> token-id renderer。
3. 因此“离散 bottleneck”本身还不够；只要 digit copying / leading-zero stripping 仍由普通 MLP 学习，就仍会出现 combinatorial generalization error。

更新后的 C 线判断：

```text
free continuous bridge          -> 失败：81.25%
learned vocabulary bottleneck   -> 改善但失败：87.5%
deterministic discrete renderer -> 成功：100%
```

下一步如果继续，方向应进一步收紧为 **compositional deterministic renderer** 或 **typed symbolic token operator**，而不是“MLP 预测 token ID”。也就是把 digit copying、leading-zero stripping、sign handling 这些 renderer 子步骤显式算法化，再研究是否能把最终 KV slots 压到 `<10`。

### 47.38 Hidden-state reasoning 方向封档结论（2026-06-24）

截至 C2.4，B/C 两条实验线对“用 hidden state 替代 token 做长上下文推理 / agent 内部通信”的判断已经足够清楚，因此暂时封住该方向，不再继续堆更大的 free MLP / bridge。

#### 47.38.1 当前证据链

实验结果可以压缩成三类：

```text
free continuous hidden transition / bridge  -> 不可靠
learned discrete/vocab bottleneck bridge    -> 有改善但仍不可靠
operator VM + deterministic discrete render -> 可靠
```

关键节点：

| 实验 | 形式 | 结果 | 诊断 |
|---|---|---:|---|
| B2.x free hidden transition | hidden -> MLP -> hidden | 负结果 | 算法/组合状态转移不稳 |
| B2.14 primitive digit operator | typed digit slots + local operator | 正结果 | 算法结构显式化后可靠 |
| C1 operator VM | CoT decode -> operator VM | 100% accuracy，约 20× decode-token/latency speedup | 推理链可由 typed VM 替代 |
| C2.1 canonical embedding suffix | 10-token `signed_answer` embedding | 100% accuracy | frozen LLM 能稳定读取 compact token record |
| C2.2 free continuous renderer bridge | slots -> continuous embeddings | 81.25% / fixed-K 0–3.125% | train token identity 也不能保证 held-out 泛化 |
| C2.3 deterministic discrete lookup renderer | slots -> token IDs -> embedding lookup | 100% accuracy，100% token identity | 离散 identity 保留后可靠 |
| C2.4 vocabulary bottleneck bridge | slots -> candidate token logits -> IDs | 87.5% accuracy，98.125% token accuracy | 离散 bottleneck 改善但仍有 digit substitution |

#### 47.38.2 为什么 continuous hidden state 不适合作为可靠长程推理介质

核心问题不是 hidden state 没有信息，而是它不具备 token/symbolic state 作为通信协议所需的稳定性质。

1. **没有稳定 identity**

   Token ID 有稳定身份，例如 `16 -> "1"`；而一个 4096 维 hidden vector 在不同上下文、不同层、不同位置中没有全局稳定含义。若不引入码本/离散化，无法可靠表达 `digit=1`、`carry=1`、`sign=positive` 这类可复用状态。

2. **连续小误差会变成离散大错**

   在 embedding 空间里，MLP 输出只要稍微偏移，nearest token 就可能从 `3` 变成 `6`、从 `1` 变成 `9`。C2.2/C2.4 的失败正是这种 digit substitution：

   ```text
   63   -> 66
   74   -> 77
   1221 -> 1219
   1179 -> 1029
   ```

   对语义任务这可能是小误差；对算法推理这是灾难性错误。

3. **组合泛化弱**

   算术、程序执行、变量绑定、leading-zero stripping、digit copy 都是组合规则。普通 MLP 容易在训练组合上达到 100%，但 held-out digit combination 仍失败，说明它学到的是统计映射，不是稳定算法。

4. **预训练目标不把 hidden state 训练成外部 ABI**

   LLM 的 hidden state 是 next-token prediction 的内部中间量，不是为 `read_state/write_state/compose_state/execute_operator` 这种外部可读写协议训练的。把 hidden state 直接持久化、传输、再注入，本质是在强行把激活当 API。

5. **长链推理需要可校验状态**

   Token / typed slots 可以 parse、compare、validate、rollback；continuous hidden state 很难判断“此刻是否真的表示 216”。长链越长，漂移和不可校验性越严重。

#### 47.38.3 对原始假设的修正

原始强假设：

```text
Agent 应该用 hidden states 替代 tokens，作为原生内部推理语言。
```

当前实验不支持这个强假设。更合理的改写是：

```text
Agent 不应只依赖 verbose natural-language CoT；
但也不应把无约束 continuous hidden states 当作可靠推理介质。

更稳的结构是：
typed discrete/state slots
+ deterministic/typed local operators
+ compact token/embedding renderer
+ LLM controller / semantic interface
```

也就是说，应该替代的是 **verbose CoT**，而不是替代 token 的离散 identity。token 仍然是当前 LLM 体系里最成熟、最方便的离散通信协议。

#### 47.38.4 封档判断

暂时封住以下方向：

```text
hidden state as native long-context reasoning medium
free continuous latent transition
free MLP renderer bridge
MLP-only slot -> token-id bridge
```

保留但不立即推进的方向：

```text
typed symbolic token operator
compositional deterministic renderer
vocabulary/grammar constrained rendering
LLM fine-tune 后的原生 latent protocol
```

后续主线切回实际推理加速：

```text
decode token / latency reduction
prefill/KV efficiency
speculative decoding
attention/KV cache optimization
batching/scheduler/runtime profiling
```

当前 C 线可沉淀为一个负/正混合结论：**continuous hidden state 可以承载语义，但不可靠承载算法状态；可靠推理加速应优先使用 typed slots + deterministic operators + compact discrete renderer。**

### 47.39 Mixed prefill admission policy：短 prompt 不进入 mixed batch（2026-06-24）

hidden-state reasoning 方向封档后，主线切回实际 runtime 推理加速。本节先做一个小步、可回归的 scheduler 改动：mixed prefill+decode 已能降低长 prompt 插入时的 decode gap，但不应把短 prompt 也强行 mixed/chunked 化；短 prompt 更适合等 running decode 空闲后一次性 prefill。

#### 47.39.1 问题

当前 `chunked_prefill_mixed_batch=True` 时，只要存在 running decode，scheduler 会尝试把 waiting prompt 的 prefill chunk 和 decode 合到同一个 varlen prefill forward：

```text
running decode + waiting prefill -> mixed batch
```

这对长 prompt 有意义，因为可以避免长 prefill 阻塞 decode；但对短 prompt 不一定划算：短 prompt 本来一次 prefill 就能完成，把它混进 decode step 会制造一个很重的 mixed step，反而拉高 first-output latency / decode gap。

#### 47.39.2 代码改动

文件：

- `tinyvllm/config.py`
- `tinyvllm/engine/scheduler.py`
- `tools/profile_chunked_prefill.py`
- `tools/test_chunked_prefill.py`

新增配置：

```python
chunked_prefill_mixed_min_prompt_tokens: int = 0
```

含义：

```text
0：保持旧行为，任何 waiting prompt 都可进入 mixed batch。
>0：mixed 只接纳 remaining prompt tokens >= 阈值的新 waiting prompt；短 prompt 会被 defer，直到 running decode 空闲后再 prefill。
```

scheduler 逻辑：

```text
if mixed enabled and running:
  if prefilling exists:
    allow mixed drain
  elif waiting[0].remaining_prompt_tokens >= threshold:
    allow mixed
  else:
    schedule decode only
```

注意：阈值只控制**新 waiting prompt 接入**；已经进入 `prefilling` 的长 prompt 会继续允许 mixed drain，避免中途卡住。

Profiler 新增参数：

```text
--mixed-min-prompt-tokens N
--short-insert-prompt-tokens N
--inject-short-after-decode-steps N
```

新增单测：

- `test_mixed_min_prompt_tokens_defers_short_waiting_prompt_to_decode()`
- `test_mixed_min_prompt_tokens_still_admits_long_waiting_prompt()`

#### 47.39.3 校验

本地：

```text
python3 -m py_compile tinyvllm/config.py tinyvllm/engine/scheduler.py tools/profile_chunked_prefill.py tools/test_chunked_prefill.py
python3 tools/test_chunked_prefill.py
python3 tools/test_profile_chunked_prefill.py
git diff --check
```

远端：

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tinyvllm/config.py tinyvllm/engine/scheduler.py tools/profile_chunked_prefill.py tools/test_chunked_prefill.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_profile_chunked_prefill.py
```

均通过。

#### 47.39.4 Benchmark

使用 Qwen3-0.6B，模拟 4 个 running decode 请求，decode 2 step 后插入一个 64-token short prompt。比较：

```text
baseline mixed:
  mixed_min_prompt_tokens=0

admission policy:
  mixed_min_prompt_tokens=128
```

公共参数：

```text
model=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
mode=mixed
num_decode_seqs=4
decode_prompt_tokens=64
short_insert_prompt_tokens=64
inject_short_after_decode_steps=2
inject_long_after_decode_steps=100000
max_num_prefill_tokens_per_step=128
max_output_len=32
max_model_len=2048
max_num_batched_tokens=2048
max_num_seqs=16
enforce_eager=True
GPU=A100, CUDA_VISIBLE_DEVICES=7
```

输出文件：

- `profile_out/chunked_prefill_mixed_short_no_admission.json`
- `profile_out/chunked_prefill_mixed_short_min128.json`

结果：

| mode | mixed steps | prefill steps | total ms | first output ms | max decode gap ms | decode p50 ms | decode p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed, no admission | 1 | 1 | 2619.05 | 2325.76 | 373.92 | 45.48 | 137.68 |
| mixed, min_prompt_tokens=128 | 0 | 2 | 3107.64 | 1386.50 | 183.60 | 40.41 | 82.13 |

逐步行为变化：

```text
no admission:
  step 3: mixed, tokens=68, dt=373.92ms

min_prompt_tokens=128:
  step 3..31: decode only
  step 32: short prompt prefill, tokens=64, dt=57.69ms
```

#### 47.39.5 结论

这是一个明确的 latency-policy tradeoff：

1. `mixed_min_prompt_tokens=128` 成功避免了短 prompt 进入 mixed batch。
2. first-output latency 从 `2325.76ms` 降到 `1386.50ms`，改善约 `40.4%`。
3. max decode gap 从 `373.92ms` 降到 `183.60ms`，改善约 `50.9%`。
4. decode p95 从 `137.68ms` 降到 `82.13ms`，改善约 `40.3%`。
5. 总 wall time 从 `2619.05ms` 增加到 `3107.64ms`，因为短 prompt 被延后到 running decode 完成后才 prefill/decode。

因此该策略适合 latency-sensitive / serving fairness 场景：优先保护已有 running decode 的 first-token / decode-gap；如果目标是最小整体 makespan，则可以保持阈值为 `0`。

推荐默认仍保持 `0` 以兼容旧行为；serving 侧可按 workload 打开，例如：

```text
chunked_prefill_mixed_min_prompt_tokens = max_num_prefill_tokens_per_step
```

下一步推理加速建议继续沿 scheduler/runtime profiling 做：长 prompt 插入场景下比较 `mixed_min_prompt_tokens=0/128/256/512`，并和 decode-first / fair chunked policy 放在同一张 latency-vs-makespan 曲线上。

### 47.39.6 Mixed token-budget reserve：decode query 也计入 `max_num_batched_tokens`（2026-07-08）

本轮继续沿 scheduler/runtime profiling 做一个小步修正：mixed prefill+decode batch 复用 varlen prefill path，decode row 在 `prepare_mixed()` 中也是 query length = 1。旧 scheduler 只用 `max_num_batched_tokens` 限制 prefill chunk tokens，再追加 decode rows；因此在 tight token budget 下可能实际送入 `prefill_tokens + decode_rows > max_num_batched_tokens` 的 mixed batch。

改动：

- `Scheduler._schedule_chunked_prefill()` 增加内部参数 `max_prefill_tokens`，默认仍等于 `max_num_batched_tokens`，保持非 mixed 行为不变。
- `Scheduler._schedule_mixed_prefill_decode()` 在拉取 prefill chunk 时先为至少 1 个 decode query token 预留 token budget。
- mixed 追加 decode rows 时继续检查 `prefill_tokens + len(decode_seqs) < max_num_batched_tokens`，避免多条 decode row 把总 query tokens 撑过预算。

新增回归测试：

- `test_mixed_prefill_reserves_token_budget_for_decode_queries()`：`max_num_batched_tokens=12`、3 个 4-token short prompt + 1 条 running decode 时，mixed batch 只接纳 2 个 short prompt + 1 个 decode row，保留第 3 个 waiting prompt。
- `test_mixed_decode_rows_respect_remaining_token_budget()`：`max_num_batched_tokens=9`、2 个 4-token short prompt + 2 条 running decode 时，只追加 1 条 decode row，保证 `prefill_tokens + decode_rows <= max_num_batched_tokens`。

本地验证：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_profile_chunked_prefill.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/scheduler.py tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/sequence.py tinyvllm/config.py tools/profile_chunked_prefill.py tools/test_chunked_prefill.py
git diff --check
```

均通过。

远程验证：

```text
rsync tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py -> sitian@10.232.195.203:/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_profile_chunked_prefill.py
```

均通过。

GPU profiler smoke 这次没有形成可用性能数字：`tools/profile_chunked_prefill.py --mode mixed ... --max-num-batched-tokens 129` 在远程启动后 log 保持 0 字节，进程进入 D-state：

```text
STAT=D
WCHAN=os_acquire_rwlock_write
COMMAND=/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_chunked_prefill.py ...
```

同时 SSH 偶发 `Connection closed by UNKNOWN port 65535`，`klist`、目标 `nc`、`ssh ... 'echo remote-ok'` 一度正常，说明当前 GPU profile 阻塞更像远程 GPU/driver/内核等待或链路状态问题，不应作为代码回归结论。该进程 `kill -9` 后仍短时间处于 D-state，需等内核态返回或由远程环境回收。

结论：

1. 这是一个 correctness/fairness-oriented 的 scheduler budget 修正，不声称 throughput 提升。
2. 它让 mixed batch 的实际 query token 数与 `max_num_batched_tokens` 语义一致，避免 tight-budget serving 下意外超配。
3. 当前已用本地和远程纯 Python 测试覆盖行为；GPU benchmark 需等远程 GPU/driver 状态恢复后重跑。

### 47.39.7 Mixed first-chunk budget clamp：首个 prefill chunk 也给 decode 留预算（2026-07-08）

继续检查 §47.39.6 后发现一个相邻边界：当 `max_num_prefill_tokens_per_step > mixed 剩余 token budget` 时，首个 prefill chunk 会先按 `max_num_prefill_tokens_per_step` 切块，随后因为没有剩余 budget 追加 decode row，整个 step 退化成普通 prefill。这与 mixed batch 的目标不一致：既然已有 running decode，tight budget 下也应缩小 prefill chunk，至少保留 1 个 decode query row。

改动：

- `_schedule_chunked_prefill(max_prefill_tokens=...)` 会把 `max_prefill_tokens` 传给首个 waiting seq 和已有 `prefilling` seq。
- `_schedule_one_prefill_chunk(seq, max_chunk_tokens=...)` 在 `max_num_prefill_tokens_per_step`、`max_chunk_tokens`、剩余 prompt tokens 三者中取最小值。
- short prompt batching 的后续 seq 也按剩余 `max_prefill_tokens - num_batched_tokens` 计算，避免后续 chunk 挤占 decode 预算。

新增回归测试：

- `test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget()`：`max_num_batched_tokens=5`、`max_num_prefill_tokens_per_step=8`、已有 1 条 running decode 和 12-token waiting prompt 时，scheduler 应返回 mixed batch，prefill chunk 从 8 缩到 4，并保留 1 个 decode row。

验证：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_profile_chunked_prefill.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py tools/profile_chunked_prefill.py
git diff --check
```

本地通过；同步到远程后，远程：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_profile_chunked_prefill.py
```

也通过。

### 47.40 S1 online n-gram speculative dry-run profiler（2026-06-24）

上一节的 mixed prefill admission policy 属于 scheduler policy，收益依赖 workload 形状；更扎实的推理加速主线切到 speculative decoding。当前路线拆成四步：

```text
S1 online ngram speculative dry-run
S2 KV-safe target verification
S3 accepted-token KV commit
S4 full online speculative decoding benchmark
```

本节完成 S1：只观察真实 TinyLLM decode loop 中的 token 流，在线提出 n-gram draft 并和后续真实 token 比较，但**不改变输出、不额外写 KV、不做 rollback/commit**。

#### 47.40.1 代码改动

文件：

- `tinyvllm/speculative/ngram.py`
- `tools/profile_ngram_online.py`
- `tools/test_ngram_speculative.py`

新增 online dry-run 状态：

```python
@dataclass
class NGramOnlineDryRunState:
    pending_tokens: list[int]
    active_match_start: int = -1
    active_drafted_tokens: int = 0

@dataclass
class NGramOnlineDryRunTotals:
    decode_positions: int = 0
    draft_events: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_events: int = 0
    completed_drafts: int = 0
    no_draft_positions: int = 0
```

核心函数：

```python
def ngram_online_dry_run_step(
    history_before_decode: list[int],
    actual_next_token: int,
    state: NGramOnlineDryRunState,
    totals: NGramOnlineDryRunTotals,
    ngram_size: int,
    max_draft_tokens: int,
) -> dict[str, int | bool | list[int]]:
```

行为：

1. 若当前没有 pending draft，则从 `history_before_decode` 的末尾 n-gram 找历史最近匹配，并提出最多 `max_draft_tokens` 个 draft token。
2. 用真实 decode 产生的 `actual_next_token` 检查 pending draft 的第一个 token。
3. 命中则累计 accepted 并弹出 pending 前缀；不命中则累计 rejected 并清空 pending。
4. 只更新 CPU 侧统计对象，不触碰 scheduler / model runner / KV cache。

#### 47.40.2 profiler 设计

`tools/profile_ngram_online.py` 直接驱动真实 `LLM.step()`：

```text
before = waiting + running 的 token_ids 快照
llm.step()
sampled_rows = scheduled seq 中 token_ids 比 before 增长的行
对每个 sampled row 调 ngram_online_dry_run_step()
```

这里没有依赖 mixed batch 的 `step_is_decode` 标记，因为 mixed postprocess 后会重置该标记；更稳的判据是 scheduled `Sequence.token_ids` 是否相对 step 前增长。这样普通 decode、mixed decode，以及 final prefill 中采样首 token 的情况都能被统一观察到。

输出 JSON 包括：

- `summary`：总体 `decode_positions / draft_events / drafted_tokens / accepted_tokens / acceptance_rate / draft_coverage / theoretical_decode_step_reduction`。
- `per_sequence`：每条 seq 的同类统计。
- `events`：前 N 个 proposed/accepted/rejected 事件，便于人工检查 draft 行为。
- `step_records`：每步 batch_kind、耗时、sampled rows、完成输出数。

#### 47.40.3 本地校验

本地语法和纯 token helper 测试已通过：

```text
python3 -m py_compile tinyvllm/speculative/ngram.py tools/profile_ngram_online.py tools/test_ngram_speculative.py
python3 tools/test_ngram_speculative.py
python3 tools/profile_ngram_online.py --help
```

结果：

```text
ngram speculative tests passed
```

补充了两个 online dry-run 单测：

- `test_online_dry_run_accepts_pending_prefix_across_steps()`：验证一个 2-token draft 可以跨两个真实 decode step 累计 accepted 并 completed。
- `test_online_dry_run_rejects_and_clears_pending_tokens()`：验证首 token 不匹配时会 rejected 并清空 pending。

本机没有 TinyLLM 运行依赖（当前 `/usr/bin/python3` 缺少 `transformers`），实际模型 profiling 未在本机完成：

```text
ModuleNotFoundError: No module named 'transformers'
```

#### 47.40.4 远端 S1 smoke 结果

远端环境：

```text
机器：sitian@10.232.195.203 / A100
代码目录：/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
Python：/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
GPU：CUDA_VISIBLE_DEVICES=7
模型：Qwen3-0.6B fp16
```

先同步 S1 文件到远端，并做远端语法/单测校验：

```text
python -m py_compile tinyvllm/speculative/ngram.py tinyvllm/engine/llm_engine.py tools/profile_ngram_online.py tools/test_ngram_speculative.py
python tools/test_ngram_speculative.py
```

结果：

```text
ngram speculative tests passed
```

运行命令：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_online.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_online_s1_06b.json
```

总体统计（3 条内置 prompt，每条 64 output token）：

| metric | value |
|---|---:|
| decode_positions | 192 |
| draft_events | 28 |
| drafted_tokens | 112 |
| accepted_tokens | 92 |
| rejected_events | 5 |
| completed_drafts | 21 |
| no_draft_positions | 95 |
| acceptance_rate | **82.1%** |
| avg_draft_len | 4.0 |
| draft_coverage | **14.6%** |
| theoretical_decode_step_reduction | 32.4% |
| decode_steps | 64 |
| elapsed_s | 41.56s |

分 sequence 看差异很明显：

| seq | draft_events | accepted_tokens | acceptance_rate | draft_coverage | theoretical reduction |
|---:|---:|---:|---:|---:|---:|
| 4 | 13 | 47 | 90.4% | 20.3% | 42.3% |
| 5 | 3 | 4 | 33.3% | 4.7% | 5.9% |
| 6 | 12 | 41 | 85.4% | 18.8% | 39.0% |

注意：这次输出里的 `seq_id` 是 4/5/6，导致旧版 profiler 的 `prompt_tokens` 映射为空；原因是不能假设 `seq_id == prompt index`。已修正为 `prompt_lens_by_seq_id[seq.seq_id]` 映射，并同步到远端。该问题只影响 per-sequence 展示字段，不影响总体 speculative 统计。

#### 47.40.5 8B 对齐命令

若要和 §35 的 8B 量化离线 replay 对齐，则使用：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_online.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --quantization int4 \
  --quant-group-size 32 \
  --act-quant-bits 8 \
  --act-quant-skip-last 4 \
  --smoothquant-scale-path /tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt \
  --kv-quant-bits 8 \
  --kv-quant-group-size 32 \
  --out-json profile_out/ngram_online_s1_8b_w4a8sqkv8.json
```

#### 47.40.6 当前判断

S1 的代码路径已经具备三个安全边界：

1. **输出不变**：仍然只使用原始 `LLM.step()` 产生的 token。
2. **KV 不变**：没有 speculative token 写入、回滚或提交。
3. **scheduler 不变**：只读取 `last_scheduled_seqs` 和 step 前后的 token_ids 差异。

远端 online stats 给出两个信号：

1. 一旦 n-gram 命中，接受率很高：总体 `acceptance_rate=82.1%`，两个重复/模板化样本超过 85%。
2. 但触发覆盖率不高：总体 `draft_coverage=14.6%`，普通代码解释样本只有 4.7%。

因此结论不是“直接做完整 speculative decoding”，而是：**进入 S2 是值得的，但范围必须收窄为 KV-safe target verification 原型**。S2 只验证：在不提交 speculative KV 的前提下，target 能否一次验证 draft tokens，并把 accepted prefix 与原逐 token decode 对齐。若 S2 验证成本/状态管理可控，再进入 S3 accepted-token KV commit；否则停在 S1 profiler。

### 47.41 S2 KV-safe target verification 原型（2026-06-24）

承接 §47.40，本节实现 S2 的最小安全原型：**不提交 speculative KV，只用 target model 对 n-gram draft 做一次性验证，并检查 target accepted prefix 是否与逐 token decode replay 对齐**。

#### 47.41.1 设计边界

S2 不做以下事情：

```text
不把 draft token append 到真实 Sequence
不把 speculative KV 接到 live block_table
不发布 prefix-cache hash
不改变 scheduler 输出
不跳过真实 decode step
```

S2 只做：

```text
history + draft[:-1] 作为临时 verifier prefill 输入
一次 target forward 取 h[-1], d0, ..., d{n-2} 的 logits
greedy argmax 得到 target continuation
count_accepted_prefix(draft, target continuation)
和原逐 token 输出 replay 的 accepted prefix 比较
```

也就是验证这个等价关系：

```text
target_verify_accept(draft | history)
== replay_accept(draft, actual_future)
```

在 greedy decode 下，如果 verifier 的上下文和正常逐 token decode 对齐，二者在可比较范围内应完全一致。

#### 47.41.2 代码改动

文件：

- `tinyvllm/engine/block_manager.py`
- `tinyvllm/speculative/ngram.py`
- `tools/profile_ngram_verify.py`
- `tools/test_ngram_speculative.py`

`BlockManager` 新增 scratch block 分配：

```python
def allocate_ephemeral(self, seq: Sequence):
    """Allocate scratch KV blocks without prefix-cache lookup or publication."""
```

关键点：

1. 不查 `hash_to_block_id`，避免 verifier 误复用 live prefix cache 后拿不到中间 logits。
2. 不写 `hash_to_block_id`，避免 scratch KV 被 prefix cache 发现。
3. verifier 结束后 `deallocate(seq)`，scratch KV 即不可达；底层显存内容可残留，但没有 block_table/hash 引用。

`ngram.py` 新增：

```python
@dataclass
class NGramTargetVerifyStats:
    verify_events: int = 0
    verified_tokens: int = 0
    target_accepted_tokens: int = 0
    replay_accepted_tokens: int = 0
    mismatched_events: int = 0
    truncated_future_events: int = 0

def count_accepted_prefix(draft_tokens, target_tokens) -> int
```

`tools/profile_ngram_verify.py` 流程：

1. 先正常 `LLM.generate()` 得到逐 token baseline 输出。
2. replay prompt+output token stream，复用 S1 的 online dry-run state；只有 `event["proposed"]` 时触发 verifier。
3. verifier 构造临时 `Sequence(history + draft[:-1])`，用 `allocate_ephemeral()` 分配 scratch KV。
4. 调 `prepare_prefill()` 后显式设置 `get_context().logits_indices = [len(history)-1, ..., len(history)+len(draft)-2]`，因为 `ParallelLMHead` 默认 prefill 只保留最后一个 logit。
5. 一次 target forward 得到 draft 每个位置的 greedy target token。
6. 比较 `target_accepted` 与 `replay_accepted`。

#### 47.41.3 本地校验

```text
python3 -m py_compile tinyvllm/engine/block_manager.py tinyvllm/speculative/ngram.py tools/profile_ngram_verify.py tools/test_ngram_speculative.py
python3 tools/test_ngram_speculative.py
python3 tools/profile_ngram_verify.py --help
git diff --check
```

结果：

```text
ngram speculative tests passed
```

本地仍因缺少 `transformers` 不跑模型；模型 smoke 在远端跑。

#### 47.41.4 远端 smoke

远端先同步 S2 文件并校验：

```text
python -m py_compile tinyvllm/engine/block_manager.py tinyvllm/speculative/ngram.py tools/profile_ngram_verify.py tools/test_ngram_speculative.py
python tools/test_ngram_speculative.py
```

结果通过。

完整 64-token smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_verify.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-verifications 128 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_verify_s2_06b.json
```

结果：

| metric | value |
|---|---:|
| decode_positions | 192 |
| draft_events / verify_events | 28 |
| drafted / verified tokens | 112 |
| replay accepted tokens | 92 |
| target accepted tokens | 98 |
| mismatched_events | **0** |
| mismatch_rate | **0.0%** |
| truncated_future_events | 2 |
| replay_acceptance_rate | 82.1% |
| target_acceptance_rate | 87.5% |
| generation_elapsed_s | 40.60s |
| verify_elapsed_s | 1.34s |

分 prompt：

| prompt | verify_events | replay accepted | target accepted | mismatch_rate |
|---:|---:|---:|---:|---:|
| 0 | 13 | 47 | 50 | 0.0% |
| 1 | 3 | 4 | 4 | 0.0% |
| 2 | 12 | 41 | 44 | 0.0% |

`target_accepted_tokens > replay_accepted_tokens` 的原因是两个 proposal 出现在输出尾部，baseline future token 不足 4 个，因此 replay 只能比较已生成范围；target verifier 仍能继续验证完整 draft，所以记入 `truncated_future_events=2`。在可比较范围内 `mismatched_events=0`。

#### 47.41.5 结论

S2 原型验证了核心正确性：

```text
n-gram draft proposal
-> target one-pass verification
-> accepted prefix
```

在 greedy decode 下与逐 token baseline replay 完全对齐（可比较事件 mismatch=0）。这说明 target verification 的 logits 对齐和 scratch KV 隔离路径是可行的。

但当前 verifier 是离线 full-prefill 原型，不是最终高效实现：每个 verify event 都用 `history + draft[:-1]` 重新 prefill，验证的是正确性/接口边界，不是速度。下一步 S3 才应处理：

```text
使用 live KV prefix + draft token verification
只提交 accepted token 的 KV
拒绝时丢弃 speculative KV / 或写入目标 token KV
维护 block_table、num_tokens、last_token、prefix-cache hash 的一致性
```

进入 S3 的前置条件已经满足：S2 的 accepted-prefix 语义成立，且 scratch verifier 不污染 live KV/cache。

### 47.42 S3 accepted-token KV commit 小范围原型（2026-06-25）

承接 §47.41，S3 不直接进入完整 online speculative decoding，而是先做一个窄口径 smoke：同一 `LLM` 实例中跑两个相同 greedy request，一个保持 baseline，另一个最多触发一次 n-gram verify+commit；最终要求 candidate 输出 token 与 baseline 完全一致。

#### 47.42.1 设计边界

S3 只验证 accepted-token commit 的状态一致性：

```text
Sequence.token_ids
Sequence.last_token
Sequence.num_tokens / num_completion_tokens
Sequence.block_table
BlockManager.blocks[*].hash / token_ids
BlockManager.hash_to_block_id prefix-cache 链
reserved block 的释放
```

S3 仍不做完整加速调度：

```text
不改 scheduler policy
不批量处理多个 speculative events
不测吞吐提升
不进入非 greedy sampling
不把 unaccepted token 接入 live Sequence
```

#### 47.42.2 代码改动

文件：

- `tinyvllm/engine/block_manager.py`
- `tools/profile_ngram_commit.py`
- `tools/test_chunked_prefill.py`

`BlockManager` 新增三类接口：

```python
def reserve_append_blocks(self, seq: Sequence, num_new_tokens: int) -> list[int]
def release_reserved_blocks(self, block_ids: list[int])
def commit_accepted_tokens(self, seq: Sequence, accepted_tokens: list[int], reserved_block_ids: list[int])
```

commit 顺序是：

```text
1. 先按 draft 长度 reserve 可能需要的新 block，但不加入 live seq.block_table
2. verifier 用 live block_table + reserved blocks 的 proxy block_table 写 KV
3. 根据 accepted_count 只把需要的新 block extend 到 seq.block_table
4. 逐个 seq.append_token(token_id)，更新 token_ids / last_token / num_tokens
5. publish_full_blocks(seq)，只给已经完整 materialized 的 block 发布 prefix-cache hash
6. release_reserved_blocks(unused_blocks)
```

`tools/profile_ngram_commit.py` 的 verifier 使用 decode-style prefill context：

```text
input_tokens = [seq.last_token] + draft_tokens
slot_positions = [history_len - 1, ..., history_len + len(draft_tokens) - 1]
block_tables = [live block_table + reserved_blocks]
logits_indices = [0, ..., len(draft_tokens) - 1]
```

这样每一行 logits 分别对应：

```text
h[-1] -> target d0
d0    -> target d1
...
```

接受前缀后只提交 `draft_tokens[:accepted]`；如果 accepted 里包含 EOS 或超过 `max_tokens`，先裁剪再 commit。

#### 47.42.3 本地校验

新增 block-manager 级测试覆盖三种边界：

1. 接受部分 draft：append sequence，保留需要的新 block，释放 unused reserved block。
2. 接受 0 个 token：不改 sequence，释放全部 reserved blocks。
3. 接受 token 跨多个完整 block：发布多段 prefix-cache hash 链。

本地命令：

```bash
python3 tools/test_chunked_prefill.py
python3 tools/test_ngram_speculative.py
python3 -m py_compile tools/profile_ngram_commit.py tinyvllm/engine/block_manager.py
```

结果：

```text
chunked prefill tests passed
ngram speculative tests passed
```

#### 47.42.4 远端 smoke 结果

远端 smoke 命令：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_commit_s3_06b.json
```

结果：

| metric | value |
|---|---:|
| exit_code | 0 |
| outputs_match | true |
| committed | true |
| accepted_count | 2 |
| commit_attempts | 1 |
| zero_accept_events | 0 |
| no_draft_steps | 7 |
| baseline_output_tokens | 64 |
| candidate_output_tokens | 64 |
| elapsed_s | 64.41 |
| gate_pass | true |

commit event：

| field | value |
|---|---:|
| step | 8 |
| history_len | 22 |
| draft_len | 4 |
| target accepted prefix | 2 |
| reserved_blocks | 0 |
| num_tokens_after | 24 |
| last_token_after | 21619 |

`draft_tokens=[13440, 21619, 13, 4710]`，target verifier 给出 `target_tokens=[13440, 21619, 1, 330]`，因此接受前两个 token 并只提交这两个 token。candidate 最终 64 个 output token 与 baseline 完全一致。

这次修正了两个 S3 smoke 问题：

1. commit 尝试必须发生在下一次 `llm.step()` 之前；如果放在 step 之后，会错过 online proposal 的时机，表现为 `committed=false`。
2. verifier 使用 live KV prefix 时，`cu_seqlens_q=[0, query_len]`，但 `cu_seqlens_k` 必须是 `[0, history_len + len(draft_tokens)]`；否则 target attention 看不到完整 prefix，表现为 `commit_attempts>0` 但 `zero_accept_events` 全部为 0 接受。

当前 S3 remote smoke gate 已通过：

```text
summary.outputs_match == true
summary.committed == true
summary.accepted_count > 0
summary.gate_pass == true
```

#### 47.42.5 当前判断

S3 的 CPU-side metadata commit 边界已经有本地测试覆盖，核心一致性约束成立：

```text
accepted tokens append 后，Sequence 与 block_table 长度匹配
完整 block 才发布 prefix-cache hash
未使用 reserved block 被释放
zero-accept 不改变 live Sequence
```

S3 当前可以进入 S4，但 S4 仍应保持窄口径：先做完整 online speculative decoding benchmark 的 greedy 单 batch 版本，再扩到多 prompt / 量化 / 8B。

### 47.43 S4 完整 online speculative decoding 窄口径 benchmark（2026-06-26）

承接 §47.42，S4 先不扩范围，只在 greedy、单 prompt、单 batch、Qwen3-0.6B fp16 上跑完整 online n-gram speculative decoding benchmark。实现仍复用 `tools/profile_ngram_commit.py`，但通过 `--max-commit-events 0` 允许 candidate 在整个生成过程中持续触发 verify+commit。

#### 47.43.1 实现边界

S4 相比 S3 的变化：

```text
S3: 最多 1 次 accepted-token commit，只验证 commit path 正确性
S4: 允许无限次 commit，直到 candidate finished，用 baseline request 校验最终输出
```

仍保持以下限制：

```text
greedy only: temperature=0.0
single prompt pair: baseline + candidate
single GPU / tensor_parallel_size=1
不扩多 prompt
不扩量化
不扩 8B
```

`tools/profile_ngram_commit.py` 新增：

```text
--max-commit-events N
  N=1: 默认 S3 smoke
  N=0: S4 full online speculative benchmark，直到完成前不限制 commit 次数
```

summary 新增 benchmark 字段：

```text
commit_events
drafted_tokens
acceptance_rate
candidate_autoregressive_steps_avoided
candidate_step_reduction
```

#### 47.43.2 远端命令

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-commit-events 0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_spec_s4_06b.json
```

#### 47.43.3 远端结果

| metric | value |
|---|---:|
| exit_code | 0 |
| outputs_match | true |
| baseline_output_tokens | 64 |
| candidate_output_tokens | 64 |
| gate_pass | true |
| commit_events | 10 |
| commit_attempts | 10 |
| zero_accept_events | 0 |
| no_draft_steps | 15 |
| drafted_tokens | 40 |
| accepted_count | 38 |
| acceptance_rate | 95.0% |
| candidate_autoregressive_steps_avoided | 38 |
| candidate_step_reduction | 59.4% |
| elapsed_s | 52.48 |

首个 commit event：

```text
step=8, history_len=22
 draft_tokens  = [13440, 21619, 13, 4710]
 target_tokens = [13440, 21619, 1, 330]
 accepted      = [13440, 21619]
 accepted_count= 2
```

最后一个 commit event：

```text
step=25, history_len=73
 draft_tokens  = [13440, 21619, 1, 646]
 target_tokens = [13440, 21619, 1, 646]
 accepted_count= 4
```

#### 47.43.4 当前判断

S4 窄口径 benchmark 通过：candidate 在多次 online verify+commit 后，最终 64 个 output token 与 baseline 完全一致。这说明从 S1/S2/S3 串起来的最小闭环已经成立：

```text
online n-gram proposal
-> target one-pass verification over live KV prefix
-> accepted-token KV/metadata commit
-> subsequent normal decode continues from committed state
-> final output matches baseline
```

但这还不是最终性能结论。当前结果只能说明窄口径正确性和理论 decode step reduction；elapsed_s 仍包含 baseline+candidate 同跑、Python 侧 profiler、额外 target verification forward 等开销，不能直接当作吞吐提升。

下一步建议仍按范围递增：

1. 多 prompt greedy benchmark：复用 S1 的 3 个 prompt，统计 per-prompt commit/acceptance/output match。
2. 只跑 candidate-only 与 baseline-only 分离计时，避免同一 engine 里双请求互相影响。
3. 再扩 8B fp16 / W4A8+KV8 量化组合。

### 47.44 S4 多 prompt greedy benchmark（2026-06-26）

承接 §47.43，本轮把 S4 从单 prompt 扩到多 prompt greedy benchmark，但仍不进入量化/8B，也不做 baseline-only / candidate-only 分离计时。

#### 47.44.1 实现改动

`tools/profile_ngram_commit.py` 的 `--prompt` 改为可重复传入：

```text
--prompt PROMPT_A --prompt PROMPT_B --prompt PROMPT_C
```

每个 prompt 在同一个 `LLM` 实例中创建一对请求：

```text
prompt_i -> baseline_seq_i + candidate_seq_i
```

profiler 对每个 candidate 独立维护：

```text
commit_events
commit_attempts
zero_accept_events
drafted_tokens
accepted_count
outputs_match
```

`--max-commit-events` 语义改为 **per candidate**：

```text
--max-commit-events 1: 每个 candidate 最多 commit 1 次
--max-commit-events 0: 每个 candidate 不限制 commit 次数
```

整体 gate 仍然是：

```text
所有 prompt outputs_match == true
总 committed == true
总 accepted_count > 0
```

#### 47.44.2 远端命令

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt 'Repeat the following phrase five times: alpha beta gamma alpha beta gamma.' \
  --prompt 'Write a short Python function and then explain each line briefly.' \
  --prompt 'The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. What color is the sky?' \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-commit-events 0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_spec_s4_06b_multiprompt.json
```

#### 47.44.3 远端结果

总体：

| metric | value |
|---|---:|
| exit_code | 0 |
| num_prompts | 3 |
| outputs_match | true |
| baseline_output_tokens | 192 |
| candidate_output_tokens | 192 |
| gate_pass | true |
| commit_events | 14 |
| commit_attempts | 15 |
| zero_accept_events | 1 |
| no_draft_steps | 132 |
| drafted_tokens | 60 |
| accepted_count | 43 |
| acceptance_rate | 71.7% |
| candidate_autoregressive_steps_avoided | 43 |
| candidate_step_reduction | 22.4% |
| elapsed_s | 40.31 |

分 prompt：

| prompt | outputs_match | commit_events | attempts | zero_accept | drafted | accepted | acceptance | step_reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | true | 10 | 10 | 0 | 40 | 38 | 95.0% | 59.4% |
| 1 | true | 3 | 3 | 0 | 12 | 4 | 33.3% | 6.2% |
| 2 | true | 1 | 2 | 1 | 8 | 1 | 12.5% | 1.6% |

#### 47.44.4 当前判断

多 prompt greedy S4 gate 通过，说明完整 online verify+commit 在更混合的 greedy 文本场景下仍保持输出一致：三个 prompt 的 candidate output token 均与 baseline 完全一致。

但覆盖率和收益高度依赖 prompt：重复模板 prompt 的 accepted/token reduction 明显，普通代码解释 prompt 和短重复事实 prompt 的收益很低。因此下一步不应直接扩 8B/量化，而应先做 **baseline-only / candidate-only 分离计时**：

```text
baseline-only: 正常 TinyLLM generate/step
candidate-only: 单请求 online speculative generate/step
```

目标是区分：理论 step reduction 是否能抵消 target verification forward + Python profiler overhead。

### 47.45 S4 baseline-only / candidate-only 分离计时（2026-06-26）

承接 §47.44，本轮不扩 8B/量化，而是把同一批 3 个 greedy prompt 拆成两次独立运行：

```text
baseline-only: 只跑正常 TinyLLM decode
candidate-only: 只跑 online speculative decode
```

这样避免 `baseline + candidate` 同一 engine 中互相影响调度和计时。

#### 47.45.1 实现改动

`tools/profile_ngram_commit.py` 新增模式：

```text
--mode paired          # 默认，baseline+candidate 同 engine，用于输出一致性 gate
--mode baseline-only   # 只跑 baseline，输出 token_ids/text 和 wall-clock
--mode candidate-only  # 只跑 speculative candidate，输出 token_ids/text、commit 统计和 wall-clock
```

candidate-only 仍使用同一条 S4 speculative 路径：

```text
online n-gram draft
-> target verify over live KV prefix
-> accepted-token commit
-> normal decode continue
```

输出一致性通过读取两份 JSON 后离线比较 `per_prompt[*].token_ids`。

#### 47.45.2 远端命令

Baseline-only：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode baseline-only \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt 'Repeat the following phrase five times: alpha beta gamma alpha beta gamma.' \
  --prompt 'Write a short Python function and then explain each line briefly.' \
  --prompt 'The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. What color is the sky?' \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-commit-events 0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_spec_s4_06b_baseline_only.json
```

Candidate-only：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode candidate-only \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt 'Repeat the following phrase five times: alpha beta gamma alpha beta gamma.' \
  --prompt 'Write a short Python function and then explain each line briefly.' \
  --prompt 'The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. The grass is green. The sky is blue. The sun is yellow. Here we go. What color is the sky?' \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 3 \
  --max-draft-tokens 4 \
  --max-commit-events 0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_spec_s4_06b_candidate_only.json
```

#### 47.45.3 远端结果

总体：

| mode | output tokens | elapsed_s | tok/s | gate |
|---|---:|---:|---:|---:|
| baseline-only | 192 | 40.67 | 4.72 | true |
| candidate-only | 192 | 71.52 | 2.68 | true |

Candidate speculative 统计：

| metric | value |
|---|---:|
| commit_events | 13 |
| commit_attempts | 17 |
| zero_accept_events | 4 |
| no_draft_steps | 130 |
| drafted_tokens | 68 |
| accepted_count | 42 |
| acceptance_rate | 61.8% |
| candidate_step_reduction | 21.9% |

离线 token 对齐：

| prompt | outputs_match | candidate commits | candidate accepted | step_reduction |
|---:|---:|---:|---:|---:|
| 0 | true | 10 | 38 | 59.4% |
| 1 | true | 3 | 4 | 6.2% |
| 2 | true | 0 | 0 | 0.0% |

总输出一致性：

```text
all_outputs_match = true
```

计时对比：

```text
candidate_elapsed / baseline_elapsed = 1.76x
wallclock_speedup = 0.57x
```

#### 47.45.4 当前判断

分离计时说明：当前 Python-level online speculative prototype **正确但不加速**。虽然 candidate-only 接受了 42 个 token，理论 step reduction 为 21.9%，但额外 target verification forward、Python 调度/统计开销、以及当前逐 event 验证方式超过了减少 decode step 的收益。

因此下一步不应扩 8B/量化；应先做候选路径降开销：

1. **减少 verifier 调用开销**：只在更高置信 n-gram 命中时触发，或提高最小 draft 长度/历史重复阈值。
2. **批量 verify 多个 candidate**：把同一步多个 speculative events 合并到一次 target forward。
3. **candidate-only 工程化路径**：从 profiler 迁到 engine/scheduler 内部，减少 Python 侧循环和 JSON 统计开销。
4. 再复测 baseline-only / candidate-only；只有 candidate-only 接近或超过 baseline，才值得扩 8B/量化。

### 47.46 S4 n-gram 触发策略 sweep（2026-06-26）

承接 §47.45，本轮先不改 engine，而是在 candidate-only 分离计时框架下 sweep 更保守的触发策略，目标是验证“减少低置信 verifier 调用”能否抵消一部分 speculative overhead。

固定条件：

```text
model=Qwen3-0.6B fp16
temperature=0.0
prompts=3
max_output_len=64
mode=candidate-only
baseline-only elapsed_s=40.67, tok/s=4.72
```

#### 47.46.1 远端命令矩阵

| tag | ngram_size | max_draft_tokens |
|---|---:|---:|
| n3d4 | 3 | 4 |
| n4d4 | 4 | 4 |
| n5d4 | 5 | 4 |
| n3d8 | 3 | 8 |

所有 candidate 输出均与 baseline-only 的 `token_ids` 离线逐 prompt 对齐：

```text
n3d4 all_outputs_match=true
n4d4 all_outputs_match=true
n5d4 all_outputs_match=true
n3d8 all_outputs_match=true
```

#### 47.46.2 结果

| tag | elapsed_s | tok/s | vs baseline elapsed | wallclock speedup | attempts | zero_accept | accepted | acceptance | step_reduction | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 40.67 | 4.72 | 1.00x | 1.00x | - | - | - | - | - | true |
| n3d4 | 71.52 | 2.68 | 1.76x | 0.57x | 17 | 4 | 42 | 61.8% | 21.9% | true |
| n4d4 | 59.61 | 3.22 | 1.47x | 0.68x | 13 | 2 | 38 | 73.1% | 19.8% | true |
| n5d4 | 37.44 | 5.13 | 0.92x | 1.09x | 11 | 2 | 35 | 79.5% | 18.2% | true |
| n3d8 | 64.06 | 3.00 | 1.58x | 0.63x | 13 | 4 | 46 | 44.2% | 24.0% | true |

分 prompt 看，`n5d4` 的收益几乎全部来自重复模板 prompt：

| prompt | commit_events | accepted | acceptance | step_reduction |
|---:|---:|---:|---:|---:|
| 0 | 9 | 35 | 87.5% | 54.7% |
| 1 | 0 | 0 | 0.0% | 0.0% |
| 2 | 0 | 0 | 0.0% | 0.0% |

#### 47.46.3 当前判断

触发策略比单纯扩大 draft 更重要：

1. `n3d8` 虽然 accepted token 最多（46），但 verifier 输入更长且 acceptance 下降，整体仍慢于 baseline。
2. `n4d4` 比 `n3d4` 少触发、少 zero-accept，耗时下降，但仍慢于 baseline。
3. `n5d4` 是当前唯一超过 baseline 的配置：触发更保守，减少无效 verifier 调用，在这批 prompt 上达到 `1.09x` wall-clock speedup。

但这个结论仍需谨慎：`n5d4` 的加速来自 prompt 0 的强重复模式，prompt 1/2 基本没有收益。因此下一步不是扩模型，而是继续做触发策略工程化：

```text
默认从 ngram_size=5, max_draft_tokens=4 作为 candidate 策略起点
加入 min_acceptance/命中质量统计，避免普通 prompt 上频繁 verifier
在 candidate-only 路径复测更多 prompt，确认 n5d4 不是样本偶然
```

### 47.47 KV offload blockwise decode：decode 不再要求全部 visible blocks 一次性 resident（2026-06-30）

承接 KV offload MVP-1 的剩余限制，本轮把 exact blockwise/streaming attention 的 online-softmax merge 接进真实 decode attention 路径。目标不是性能优化，而是先验证：decode 阶段可以按 window staging KV blocks，并在窗口级 attention 后用 online softmax 合并结果，从而不再要求所有 visible logical blocks 一次性 resident 在 GPU staging slots 中。

#### 47.47.1 实现边界

当前实现仍是受控 correctness path：

```text
默认关闭，需要显式打开 --kv-offload-blockwise-decode
仅 kv_offload_mvp0=True 时可用
仅 fp16/bf16 KV；kv_quant_bits 必须为 0
仅 decode path；prefill 仍走原路径
不支持 Quest / KV-Cartridge / AM compact / KV4 / KV8
不改 scheduler / BlockManager / attention kernel
blockwise decode MVP 使用 PyTorch gather + online-softmax，性能不是最终形态
```

新增配置 / CLI：

```bash
--kv-offload-blockwise-decode
--kv-offload-blockwise-blocks N
--max-num-prefill-tokens-per-step N
```

#### 47.47.2 关键实现

- `tinyvllm/config.py`
  - 新增 `kv_offload_blockwise_decode`。
  - 新增 `kv_offload_blockwise_blocks`。
- `tinyvllm/utils/context.py`
  - 在 attention context 中传入 `kv_offload_manager`、logical block tables、context lens、write blocks、blockwise window size。
- `tinyvllm/engine/model_runner.py`
  - `prepare_decode()` 在 blockwise 模式下只预先 stage 当前 write blocks。
  - 不再要求 decode visible logical blocks 一次性全部 resident。
  - logical block rows 传给 attention，由 attention layer 按 window 触发 staging。
- `tinyvllm/layers/attention.py`
  - decode 时若 `context.kv_offload_blockwise_decode=True`：
    - 当前 token KV 先写入 staging slot；
    - 标记 write block dirty，避免窗口 staging 时丢当前层 KV；
    - 按 logical block window 调 `KVOffloadMVP0.ensure_resident()`；
    - 从当前 layer 的 physical slot gather window K/V；
    - 用 online softmax merge 每个 window 的 attention 结果。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/utils/context.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/layers/attention.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
```

远程已同步并通过静态检查。

#### 47.47.3 blockwise attention 数学回归

输出：`profile_out/blockwise_attn_online_softmax_regression_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 16 |
| streamed_tokens | 2048 |
| max_abs_error | 2.98e-08 |
| relative_error | 2.31e-07 |

结论：online-softmax blockwise merge 与一次性 full attention 的数值误差在 smoke 范围内可接受，作为真实 decode path 的数学基线通过。

#### 47.47.4 真实模型单请求 blockwise decode smoke

Qwen3-0.6B，单条约 385-token prompt，`kv_offload_gpu_blocks=2`，`kv_offload_blockwise_blocks=1`。

输出：`profile_out/kv_offload_blockwise_decode_real_smoke_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 2 |
| decode_steps | 1 |
| prefetch_plans | 58 |
| prefetch_read_blocks | 56 |
| prefetch_write_blocks | 3 |
| d2h_copies | 3 |
| h2d_copies | 0 |

这里没有 H2D reload 是预期的：单请求 2 个 logical blocks 正好等于 staging slots，未触发驱逐。

#### 47.47.5 真实模型两请求 blockwise decode thrash

两条约 385-token prompt，`max_num_seqs=1`，`kv_offload_gpu_blocks=2`，`kv_offload_blockwise_blocks=1`。两个 request 交替 decode，跨 request 触发 staging eviction/reload。

输出：`profile_out/kv_offload_blockwise_decode_real_two_request_thrash_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 4 |
| decode_steps | 2 |
| evictions | 6 |
| h2d_copies | 4 |
| d2h_copies | 6 |
| h2d_batches | 4 |
| d2h_batches | 4 |
| prefetch_plans | 116 |
| prefetch_read_blocks | 112 |
| prefetch_write_blocks | 6 |

#### 47.47.6 当前结论与剩余限制

现在真实 decode attention 路径已经可以：

```text
按 window stage KV blocks
局部 attention
online softmax merge
跨 request 触发 eviction/reload
```

也就是说，decode 阶段已经不再要求“全部 visible logical blocks 一次性 resident”。在当前中间态下，可以先采用更窄的端到端约束：

```text
prefill 时 gpu_blocks >= prompt blocks
decode 时允许 visible blocks > staging slots
```

当前剩余限制仍在 prefill：prefill 仍走原路径。如果单条 prompt 本身超过 staging slots，prefill 阶段仍可能要求 prefix blocks 一次性可见。完整解除端到端限制的下一步是做 blockwise/chunked prefill attention。

### 47.48 KV offload blockwise/chunked prefill correctness path（2026-06-30）

承接 §47.47，本轮开始解除 prefill 的剩余限制：在 chunked prefill 场景下，非首个 prefill chunk 的 query 需要 attend 到历史 prefix KV。旧路径会在 `prepare_prefill()` 阶段把所有 prefix logical blocks 一次性翻译成 physical staging slots，因此当 `prefix blocks > kv_offload_gpu_blocks` 时仍会失败。

本轮新增一个受控 correctness path：prefill chunk 只在 `prepare_prefill()` 里 stage 当前 chunk 的 write blocks；prefix read blocks 保持 logical rows 传给 attention layer，由 attention 按 block window 触发 staging，并用 online softmax 合并 prefix window attention 与当前 chunk local causal attention。

#### 47.48.1 实现边界

```text
默认关闭，需要显式打开 --kv-offload-blockwise-prefill
依赖 kv_offload_mvp0=True
依赖 chunked prefill：max_num_prefill_tokens_per_step > 0
仅 fp16/bf16 KV；kv_quant_bits 必须为 0
仍不支持 mixed prefill+decode + KV offload
不支持 Quest / KV-Cartridge / AM compact / KV4 / KV8
Correctness-first PyTorch gather + online-softmax；不是性能版 kernel
```

当前 staging slot 需求变为：

```text
当前 prefix window logical blocks + 当前 prefill chunk write blocks <= kv_offload_gpu_blocks
```

也就是说，本轮解除的是 prefix read blocks 一次性 resident 限制；当前 chunk 的 write blocks 仍必须可保护在 staging 中，避免后续 layer 使用 stale slot mapping。

#### 47.48.2 关键实现

- `tinyvllm/config.py`
  - 新增 `kv_offload_blockwise_prefill`。
  - 约束 `kv_offload_blockwise_prefill` 必须依赖 `kv_offload_mvp0` 和 chunked prefill。
  - 显式禁止 `kv_offload_mvp0 + chunked_prefill_mixed_batch`。
- `tinyvllm/utils/context.py`
  - 新增 `kv_offload_blockwise_prefill`。
  - 新增 `kv_offload_prefill_chunk_starts` / `kv_offload_prefill_chunk_ends`。
  - 复用 logical block rows、context lens、write blocks、window blocks。
- `tinyvllm/engine/model_runner.py`
  - `KVOffloadMVP0.ensure_resident()` 新增 `protected_logical_blocks`，用于保护当前 write blocks 不被 read-window staging 驱逐。
  - `prepare_prefill()` 在 blockwise prefill 模式下：
    - 只 stage 当前 chunk write blocks；
    - 对 partial write block 使用 `require_valid=True`，避免覆盖已写入的 prefix token KV；
    - 不再对 prefix read blocks 做全量 `translate_block_rows()`；
    - 把 visible logical rows 和 chunk 边界传入 context。
- `tinyvllm/layers/attention.py`
  - 新增 `_blockwise_online_prefill_attention()`：
    - prefix `[0, chunk_start)` 按 logical block window 扫描；
    - 每个 window 调 `ensure_resident(..., protected_logical_blocks=write_blocks)`；
    - 从当前 physical slot gather prefix K/V；
    - 当前 chunk `[chunk_start, chunk_end)` 直接使用本 forward 的 `k/v`，并加 local causal mask；
    - prefix windows 与 local causal chunk 通过 online softmax exact merge。
  - blockwise decode 也同步在 window staging 时保护当前 write blocks，降低 slot stale 风险。
- `tools/profile_ngram_commit.py`
  - 新增 CLI：

```bash
--kv-offload-blockwise-prefill
--blockwise-prefill-attn-smoke
--blockwise-prefill-prefix-tokens N
--blockwise-prefill-chunk-tokens N
```

#### 47.48.3 本地检查

本地无 torch，不能运行 synthetic attention smoke；已完成可在本地执行的静态和纯 Python 回归：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/utils/context.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/layers/attention.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
git -C /Users/bytedance/dev/TinyLLMForge diff --check
```

结果：

```text
py_compile passed
chunked prefill tests passed
ngram speculative tests passed
diff --check passed
```

#### 47.48.4 远程 smoke 结果

2026-07-02 已把本轮 7 个改动文件同步到远程运行目录 `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`，并在远程 torch/CUDA 环境完成验证。

远程基础回归：

```text
py_compile passed
chunked prefill tests passed
ngram speculative tests passed
```

数学 smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --blockwise-prefill-attn-smoke \
  --blockwise-prefill-prefix-tokens 1024 \
  --blockwise-prefill-chunk-tokens 128 \
  --blockwise-attn-window-tokens 128 \
  --out-json profile_out/blockwise_prefill_attn_online_softmax_smoke_20260630.json
```

输出：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 18 |
| streamed_tokens | 2240 |
| max_abs_error | 2.5331974029541016e-07 |
| relative_error | 1.0576243517384856e-06 |

真实模型长 prompt smoke：

```bash
CUDA_VISIBLE_DEVICES=7 TINYVLLM_DIST_PORT=34567 MASTER_PORT=34567 \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --mode baseline-only \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --prompt '<约 1381 tokens，超过 2 个 KV blocks 且不超过 8 个 logical blocks 的长 prompt>' \
  --max-output-len 1 \
  --temperature 0.0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 1 \
  --max-num-prefill-tokens-per-step 256 \
  --kv-offload-mvp0 \
  --kv-offload-gpu-blocks 2 \
  --kv-offload-logical-blocks 8 \
  --kv-offload-blockwise-prefill \
  --kv-offload-blockwise-decode \
  --kv-offload-blockwise-blocks 1 \
  --out-json profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260630.json
```

输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 1 |
| decode_steps | 0 |
| elapsed_s | 36.93855146318674 |
| prefill chunks | 6 |
| chunk token sizes | 256, 256, 256, 256, 256, 101 |
| h2d_copies | 391 |
| d2h_copies | 6 |
| evictions | 395 |
| prefetch_plans | 426 |
| prefetch_read_blocks | 420 |
| prefetch_write_blocks | 6 |
| logical_blocks | 8 |
| gpu_blocks | 2 |
| resident_blocks | 2 |

执行备注：一次无效尝试使用了约 4141-token prompt，超过 `max_model_len=4096` 且超过 `kv_offload_logical_blocks=8` 可覆盖容量，触发 scheduler 空 decode assert；不是 blockwise prefill correctness 失败。另一次复跑遇到端口占用，后续显式设置 `TINYVLLM_DIST_PORT=34567 MASTER_PORT=34567` 后通过。

#### 47.48.5 当前判断

代码路径已经从“prefill 必须一次性 stage 全部 prefix blocks”改成“chunked prefill 的 prefix read blocks 由 attention layer 按 window staging”。这使端到端目标进入可验证状态：

```text
prefill: prefix read blocks > staging slots 可按 window 扫描
prefill: current chunk write blocks 仍需留在 staging slots 中
说明：需要通过 max_num_prefill_tokens_per_step 控制当前 chunk 不跨太多 write blocks
```

远程 torch/GPU smoke 已确认：

1. blockwise prefill online-softmax 与 full prefill attention 数值对齐；
2. 单条 prompt 的 prefix blocks 超过 staging slots 时，prefill 不再触发 full-resident 限制；
3. decode 继续沿用 §47.47 的 blockwise path。

#### 47.48.6 后续改进机会与当前落地状态

1. 已落地：在 `Scheduler.add()` 增加 admission 前置校验，提前拒绝 `prompt_tokens + max_tokens > max_model_len`，以及最大 KV footprint 超过 logical KV blocks 的请求，避免无效 prompt 后续退化成 scheduler 空 decode assert。
2. 已落地：在 `tools/test_chunked_prefill.py` 增加 max_model_len、prompt 超 logical KV 容量、decode KV 容量边界测试，并补充 `num_kvcache_blocks=1/2/4`、`max_num_prefill_tokens_per_step=1/2/4`、短 prompt batch 受 `max_num_seqs/max_num_batched_tokens` 限制的本地系统边界测试。
   - 2026-07-03 本地验证：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 tools/test_chunked_prefill.py` 通过；`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tools/test_chunked_prefill.py tinyvllm/engine/scheduler.py` 通过；`git diff --check` 通过。
3. 已落地：新增 `tools/smoke_blockwise_prefill_remote.sh`，默认封装远程 Python、`CUDA_VISIBLE_DEVICES=7`、`TINYVLLM_DIST_PORT=34567`、`MASTER_PORT=34567`、`PYTHONPATH`、数学 smoke 和真实模型长 prompt smoke；远程运行目录下直接执行 `tools/smoke_blockwise_prefill_remote.sh` 即可，常用覆盖如 `RUN_REAL_SMOKE=0`、`CUDA_VISIBLE_DEVICES=0`、`MODEL_PATH=/path/to/model`。
   - 2026-07-03 已在远程用 `SMOKE_TAG=20260703_script tools/smoke_blockwise_prefill_remote.sh` 完整验证入口：preflight 通过，数学 smoke `gate_pass=true`，真实模型长 prompt smoke `gate_pass=true`。
   - 输出：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260703_script.json`、`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260703_script.json`；对应 log 写到同名 `.log`，避免长 prompt JSON 全量刷终端。
   - 2026-07-06 已扩展轻量 GPU blocks matrix 模式：`RUN_GPU_BLOCKS_MATRIX=1`，默认 `KV_OFFLOAD_GPU_BLOCKS_MATRIX="1 2 4"`，每组输出带 `_gpuN` 后缀的 JSON/log，并汇总 `gate_pass`、`elapsed_s`、`h2d_copies`、`d2h_copies`、`evictions`、`resident_blocks`。
4. 远程/GPU 单 prompt 矩阵已验证：`SMOKE_TAG=20260706_gpu_matrix RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 tools/smoke_blockwise_prefill_remote.sh`。
   - `gpu_blocks=1`：预期容量失败，log 中报 `blockwise prefill window plus current write blocks exceed GPU staging slots: required=2, gpu_blocks=1`；这是当前实现边界，不是 correctness mismatch。
   - `gpu_blocks=2`：`gate_pass=true`，`elapsed_s=60.20214011892676`，`h2d_copies=391`，`d2h_copies=6`，`evictions=395`，`resident_blocks=2`。
   - `gpu_blocks=4`：`gate_pass=true`，`elapsed_s=55.02894039079547`，`h2d_copies=249`，`d2h_copies=6`，`evictions=251`，`resident_blocks=4`。
   - 2026-07-06 已补多 prompt batch smoke：`SMOKE_TAG=20260706_multiprompt_len2048 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=0 RUN_MULTI_PROMPT_SMOKE=1 MAX_MODEL_LEN=2048 GPU_MEMORY_UTILIZATION=0.85 KV_OFFLOAD_LOGICAL_BLOCKS=8 MULTI_PROMPT_REPEAT=24 tools/smoke_blockwise_prefill_remote.sh`，`num_prompts=2`、`gate_pass=true`、`elapsed_s=32.339650828391314`、`output_tokens=2`、`h2d_copies=786`、`d2h_copies=12`、`evictions=792`、`resident_blocks=2`。
5. 部分落地：`KVOffloadMVP0.ensure_resident()` 已把同一轮多个 dirty evictions 的 D2H writeback 改成 deferred batched D2H，并在 clean eviction 复用 slot 前等待 pending D2H event。
   - 新增 `tools/test_kv_offload.py`，覆盖 dirty evictions batch 成 1 个 D2H batch/span，以及 clean eviction 等待 pending D2H。
   - 本地 `py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py` 与 `git diff --check` 通过。
   - 远程 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程 migration smoke：`profile_out/kv_offload_batched_dirty_evict_migration_20260708_r2.json`，`gate_pass=true`、`h2d_copies=2`、`d2h_copies=4`、`h2d_batches=1`、`d2h_batches=2`、`d2h_batch_spans=2`、`copy_waits=6`。
   - 远程 thrash smoke：`profile_out/kv_offload_batched_dirty_evict_thrash_20260708_r2.json`，`gate_pass=true`、`h2d_copies=8`、`d2h_copies=6`、`h2d_batches=4`、`d2h_batches=2`、`d2h_batch_spans=2`、`prefetch_plans=4`。
6. 部分落地：copy event wait coalescing。batched H2D/D2H 为多个 logical blocks 记录同一个 CUDA event 时，`wait_for_blocks()` 和 deferred eviction wait 现在按 event 去重，只 wait/统计一次。
   - 新增 `test_wait_for_blocks_coalesces_identical_h2d_events()` 和 `test_deferred_eviction_waits_once_per_identical_d2h_event()`。
   - 远程 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程 migration smoke：`profile_out/kv_offload_wait_coalesce_migration_20260708.json`，`gate_pass=true`、`copy_waits=4`，低于上一轮同口径 `6`。
   - 远程 thrash smoke：`profile_out/kv_offload_wait_coalesce_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`，低于上一轮同口径 `19`；`h2d_batches=4`、`d2h_batches=2`、`prefetch_plans=4`。
7. 部分落地：`ensure_resident(..., wait=True)` 已等待的 H2D blocks 会从 `pending_wait_blocks` 中移除，避免后续 `wait_for_pending()` 重复等待。
   - 新增 `test_ensure_resident_wait_clears_pending_h2d_waits()`。
   - 远程 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程 migration smoke：`profile_out/kv_offload_pending_wait_clear_migration_20260708.json`，`gate_pass=true`、`copy_waits=4`。
   - 远程 thrash smoke：`profile_out/kv_offload_pending_wait_clear_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`、`h2d_batches=4`、`d2h_batches=2`、`prefetch_plans=4`。
8. 部分落地：`lru_cost` victim policy 已对 `logical_block in pending_wait_blocks` 增加 pending H2D penalty，避免在有其他候选时驱逐刚发起 H2D、尚未被 forward 等待/消费的 block。
   - 新增 `test_evict_policy_avoids_pending_h2d_block_when_possible()`。
   - 远程 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程 thrash smoke：`profile_out/kv_offload_pending_h2d_penalty_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`、`h2d_copies=8`、`d2h_copies=6`、`evictions=11`、`prefetch_plans=4`。
9. 部分落地：blockwise decode/prefill read window 不再每轮 `wait_for_pending()` drain 全局 pending H2D，而是调用 `wait_for_blocks(list(unique_blocks), clear_pending=True)` 只等待当前窗口实际读取的 logical blocks，减少无关 copy 被提前同步的机会。
   - `KVOffloadMVP0.wait_for_blocks()` 新增 `clear_pending=False`；`wait_for_pending()` 仍保留 drain-all 语义。
   - 新增 `test_wait_for_blocks_clear_pending_api_without_cuda()` 和 `test_wait_for_blocks_can_clear_only_requested_pending_h2d_waits()`。
   - TDD RED：远程临时回退旧签名后，新增测试按预期失败：`TypeError: KVOffloadMVP0.wait_for_blocks() got an unexpected keyword argument 'clear_pending'`。
   - 本地 `py_compile tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/test_kv_offload.py` 与 `git diff --check` 通过。
   - 远程 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_window_wait_clear_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_window_wait_clear_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 远程真实模型 smoke：`SMOKE_TAG=20260708_window_wait_clear RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34687 MASTER_PORT=34687 tools/smoke_blockwise_prefill_remote.sh` 通过；`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260708_window_wait_clear.json`，`gate_pass=true`、`elapsed_s=29.39703532680869`、`output_tokens=1`。
10. 部分落地：blockwise decode/prefill read window 保留 first-seen logical block 顺序，不再从 `set` 直接转 `list`，让 H2D request / LRU touch / victim 决策顺序更稳定。
   - 新增 `_unique_blocks_in_order()` 和 `tools/test_blockwise_attention_planning.py`。
   - TDD RED：旧 set 路径下远程测试失败于 `assert manager.ensure_calls == [[2, 0, 1]]`；实现后远程测试通过。
   - 远程 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_ordered_windows_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_ordered_windows_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
11. 部分落地：decode/prefill read window staging 已收敛到 `_stage_blockwise_read_window()`，统一 first-seen 去重、capacity guard、`prefetch_*` 统计、`ensure_resident()` 和 `wait_for_blocks(..., clear_pending=True)`，为后续合并连续 prefetch plan 做单入口准备。
   - 新增 `test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks()`。
   - TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_stage_blockwise_read_window'`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_stage_helper_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_stage_helper_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
12. 部分落地：blockwise logical row 现在在 decode/prefill 函数入口通过 `_normalize_logical_block_rows()` 一次性过滤/转 int，并返回 `max_blocks`，避免每个 decode window、每个 row 重复做 host-side list comprehension。
   - 新增 `test_normalize_logical_block_rows_filters_once_and_reports_max_blocks()`。
   - TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_normalize_logical_block_rows'`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_normalize_rows_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_normalize_rows_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
13. 部分落地：`_stage_blockwise_read_window()` API 改为接收 `future_extra_blocks` / `protected_extra_blocks` / `capacity_extra_blocks`，由 helper 单入口完成 first-seen unique list、unique set、future/protected/capacity set 构造，decode/prefill 调用点不再重复 unique planning。
   - TDD RED：远程旧 helper 签名下按预期失败：`TypeError: _stage_blockwise_read_window() got an unexpected keyword argument 'future_extra_blocks'`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_unique_once_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_unique_once_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
14. 部分落地：blockwise decode read-window mask 复用 positions template。旧路径每个 window 新建 `torch.arange(max_window_tokens)`；现在入口预建 `position_template = torch.arange(block_size * window_blocks)`，每个 window 切片复用。
   - 新增 `_decode_window_mask()` 和 `test_decode_window_mask_reuses_position_template()`。
   - TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_decode_window_mask'`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_mask_template_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_mask_template_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
15. 部分落地：blockwise prefill local causal mask 复用 q/k position templates。旧路径每个 row 新建 `torch.arange(q_len)`；现在入口按最大 chunk 长度预建 templates，每个 row 切片复用。
   - 新增 `_local_causal_mask()` 和 `test_local_causal_mask_reuses_position_templates()`。
   - TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_local_causal_mask'`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_local_mask_template_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_local_mask_template_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
16. 部分落地：blockwise prefill prefix window 现在使用 `_merge_attention_window(..., mask=None)` 表示 all-valid prefix attention，跳过每个 prefix window 的全 True mask 分配和 masked fill；local causal chunk 仍传 `_local_causal_mask()`。
   - 新增 `test_merge_attention_window_accepts_none_mask_as_all_valid()`。
   - TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_merge_attention_window'`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_merge_helper_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_merge_helper_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
17. 部分落地：`_merge_attention_window(..., mask=None)` 的 all-valid 分支不再创建 `torch.ones(scores.shape[:-1])` valid mask，直接 merge；只保留 `running_m=-inf` 的 old_weight 防护。
   - 新增 `test_merge_attention_window_none_mask_does_not_allocate_valid_mask()`。
   - TDD RED：旧实现按预期失败：`AssertionError: torch.ones called`。
   - 远程 `tools/test_blockwise_attention_planning.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_no_valid_alloc_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_no_valid_alloc_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
18. 已验证：上述 blockwise planning 小优化栈已补跑脚本级集成 smoke：`SMOKE_TAG=20260708_blockwise_planning_stack RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34721 MASTER_PORT=34721 tools/smoke_blockwise_prefill_remote.sh`。
   - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260708_blockwise_planning_stack.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260708_blockwise_planning_stack.json`，`gate_pass=true`、`elapsed_s=41.34747215360403`、`output_tokens=1`。
   - 单次 wall-clock 受远程 GPU/负载影响，只证明集成路径无回归，不作为严格性能结论。
19. 部分落地：KV offload write-block staging 已抽成 `_stage_kv_offload_write_blocks()`，blockwise prefill/decode 共用同一套 valid/fresh write-block 分组逻辑，并跳过空分组的 `ensure_resident()` 调用。
   - 新增 `tools/test_kv_write_staging.py`，覆盖重复 block 去重、valid/fresh 拆分、空 valid/fresh 分组不触发空 staging。
   - TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_stage_kv_offload_write_blocks'`。
   - 远程 `tools/test_kv_write_staging.py` 与 GPU4 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_write_staging_20260709_final.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_write_staging_20260709_final.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：prefill/decode write staging 行为已经统一，重复 block 统计按 unique block 计数，并减少空 staging 调用；这是统一 KV access planner 的低风险中间步骤。
20. 已落地：`KVOffloadMVP0.ensure_resident([])` 增加空输入 fast path，去重后直接返回 `{}`，不再触发空 D2H/H2D enqueue、空 wait 或后续 mapping 构造。
   - 新增 `test_ensure_resident_empty_blocks_is_noop_without_copy_hooks()`，用无 CUDA fake manager 覆盖空输入 no-op 行为。
   - TDD RED：远程旧实现按预期失败：`AssertionError`，因为空输入仍调用了 enqueue hook。
   - 远程 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_empty_ensure_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_empty_ensure_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：空 block staging 现在是根部 no-op，减少下游 copy/wait helper 的无效调用；不改变任何非空 staging 语义。
21. 部分落地：新增 `KVOffloadMVP0.map_block_rows()` 作为只映射已 resident block rows 的 helper；`translate_block_rows()` 仍保留 stage+translate 语义并复用该 mapping helper。full-attention decode 在 read/write staging 已完成后改用 `map_block_rows()` 构造 physical block table，避免同一 forward 内再跑一次 translate/staging 入口。
   - 新增 `test_map_block_rows_uses_existing_resident_slots_without_staging()`，用无 CUDA fake manager 验证 map-only 不触发 D2H/H2D enqueue 或 wait。
   - TDD RED：远程旧实现按预期失败：`AttributeError: '_NoopKVOffload' object has no attribute 'map_block_rows'`。
   - 远程 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_map_rows_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_map_rows_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：已 stage 的 full-attention decode block-table 构造可以复用 existing mapping，减少一次无必要的 staging/translation 入口；其它调用方继续用 `translate_block_rows()` 保持原语义。
22. 部分落地：新增 `KVOffloadMVP0.map_slots_for_positions()` 作为只映射已 resident slot positions 的 helper；`translate_slots_for_positions()` 仍保留 ensure+translate 语义并复用该 helper。
   - 新增 `test_map_slots_for_positions_uses_existing_resident_slots_without_staging()`，用无 CUDA fake manager 验证 map-only slot 计算不触发 D2H/H2D enqueue 或 wait。
   - TDD RED：远程旧实现按预期失败：`AttributeError: '_NoopKVOffload' object has no attribute 'map_slots_for_positions'`。
   - 远程 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_map_slots_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_map_slots_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：KV manager 现在同时有 block-row 与 slot-position 两个 map-only helper，为后续把已 stage 的 prefill/decode slot mapping 统一到 planner helper 做准备；现有 `translate_slots_for_positions()` 外部语义不变。
23. 部分落地：新增 `_stage_kv_offload_write_positions()`，把 blockwise prefill write positions -> first-write offsets -> write staging -> slot mapping 收敛到单一 helper；`prepare_prefill()` 的 blockwise prefill 分支不再手写 slot list。
   - 新增 `test_stage_kv_offload_write_positions_stages_once_then_maps_slots()`，验证 helper 只产生一次 write staging plan，并复用 `map_slots_for_positions()` 输出 slots。
   - TDD RED：远程旧实现按预期失败：`ImportError: cannot import name '_stage_kv_offload_write_positions'`。
   - 远程 `tools/test_kv_write_staging.py` 与 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_write_positions_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_write_positions_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：blockwise prefill write staging、write-block 统计和 slot mapping 已经进入单一 helper，减少后续优化 write/read planner 时需要维护的手写分支。
24. 部分落地：新增 `_stage_kv_offload_full_decode_blocks()`，把 full-attention decode 的 future block 集合、valid read blocks 过滤、`prefetch_*` 统计以及 read/write 两次 `ensure_resident()` 收敛到单一 helper；`prepare_decode()` 非 blockwise 分支只保留 helper 调用和 map-only block-table 构造。
   - 新增 `test_stage_kv_offload_full_decode_blocks_matches_existing_plan_shape()`，锁住旧逻辑的 read/write 统计与两次 `ensure_resident()` 调用形状。
   - TDD RED：远程旧实现按预期失败：`ImportError: cannot import name '_stage_kv_offload_full_decode_blocks'`。
   - 远程 `tools/test_kv_write_staging.py` 与 `tools/test_kv_offload.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_full_decode_helper_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_full_decode_helper_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：full-attention decode staging 的 read/write planning 进入单一 helper，行为与旧路径一致，但后续合并连续 prefetch plan / 降低 clean eviction 抖动时只需改一个入口。
25. 已落地：`KVOffloadMVP0.ensure_resident()` 增加 resident no-copy fast path：当请求块全部已 resident、没有 H2D、没有 D2H、没有 deferred wait 时，直接返回 mapping，不再调用空 `_enqueue_d2h_pairs()`、空 `_enqueue_h2d_pairs()` 或空 `wait_for_blocks()`。
   - 新增 `test_ensure_resident_already_resident_blocks_skips_empty_copy_hooks()`，覆盖已 resident + `wait=True` 仍应跳过空 copy/wait hooks。
   - TDD RED：远程旧实现按预期失败：`AssertionError`，因为已 resident 路径仍调用了空 enqueue hook。
   - 远程 `tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_resident_fastpath_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_resident_fastpath_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：重复 staging 命中已 resident blocks 时少走一层空 copy/wait 调度；非 resident、dirty eviction、H2D reload 路径不变。
26. 已验证：上述 KV planner/helper 小优化栈已补跑脚本级集成 smoke：`SMOKE_TAG=20260709_kv_planner_helpers RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34731 MASTER_PORT=34731 tools/smoke_blockwise_prefill_remote.sh`。
   - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_kv_planner_helpers.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
   - 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_kv_planner_helpers.json`，`gate_pass=true`、`elapsed_s=30.003381814807653`、`output_tokens=1`。
   - 单次 wall-clock 受远程 GPU/负载影响，只证明集成路径无回归，不作为严格性能结论。
27. 已落地：`KVOffloadMVP0.wait_for_blocks([])` 增加空输入 fast path，空 block set 直接返回，不再获取 `torch.cuda.current_stream()`。
   - 新增 `test_wait_for_blocks_empty_is_noop_without_cuda_stream()`，monkeypatch `torch.cuda.current_stream` 抛错，确保空 wait 不触碰 CUDA stream。
   - TDD RED：远程旧实现按预期失败：`AssertionError: current_stream called`。
   - 远程 `tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_empty_wait_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_empty_wait_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：空 wait list 不再进入 CUDA stream 查询路径；非空 H2D wait / clear_pending 语义不变。
28. 已落地：`KVOffloadMVP0.wait_for_blocks(blocks)` 增加 no-event fast path；如果请求 blocks 没有任何 pending H2D event，则只在 `clear_pending=True` 时清理 pending set，并直接返回，不再获取 `torch.cuda.current_stream()`。
   - 新增 `test_wait_for_blocks_without_events_clears_pending_without_cuda_stream()`，monkeypatch `torch.cuda.current_stream` 抛错，确保无 event 的 wait 只清理 pending、不触碰 CUDA stream。
   - TDD RED：远程旧实现按预期失败：`AssertionError: current_stream called`。
   - 远程 `tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_no_event_wait_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_no_event_wait_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：read window / staging 命中已 resident 或无 pending event 的 blocks 时，少走一次 CUDA stream 查询；有真实 H2D event 的 wait 语义和 event 去重逻辑保持不变。
29. 已落地：`ensure_resident()` 在 clean block 被 fresh write block 驱逐、且没有 dirty D2H、没有 CPU-valid H2D、没有 deferred wait 时，不再调用空 `_enqueue_d2h_pairs()`、空 `_enqueue_h2d_pairs()` 或空 `wait_for_blocks()`。
   - 新增 `test_ensure_resident_clean_fresh_eviction_skips_empty_copy_hooks()`，构造 `gpu_blocks=1`、clean resident block 被 fresh block 替换的无 CUDA 场景。
   - TDD RED：远程旧实现按预期失败：`AssertionError`，因为 clean fresh eviction 路径仍调用了空 D2H enqueue hook。
   - 远程 `tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
   - 远程数学 smoke：`profile_out/blockwise_decode_clean_fresh_noop_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`；`profile_out/blockwise_prefill_clean_fresh_noop_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
   - 结论：fresh write block 覆盖 clean resident block 的常见路径少走空 enqueue/wait 调度；dirty eviction、H2D reload、pending D2H wait 路径不变。
30. 已验证：上述 wait/enqueue fast-path 小优化栈已补跑脚本级集成 smoke：`SMOKE_TAG=20260709_kv_wait_enqueue_fastpaths RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34741 MASTER_PORT=34741 tools/smoke_blockwise_prefill_remote.sh`。
   - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_kv_wait_enqueue_fastpaths.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
   - 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_kv_wait_enqueue_fastpaths.json`，`gate_pass=true`、`elapsed_s=30.008230686187744`、`output_tokens=1`。
   - 单次 wall-clock 受远程 GPU/负载影响，只证明集成路径无回归，不作为严格性能结论。
31. 已验证：wait/enqueue fast-path 栈后的 GPU blocks matrix：`SMOKE_TAG=20260709_fastpath_gpu_matrix RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=1 RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34751 MASTER_PORT=34751 tools/smoke_blockwise_prefill_remote.sh`。
   - `gpu_blocks=1`：预期容量失败，log 中报 `blockwise prefill window plus current write blocks exceed GPU staging slots: required=2, gpu_blocks=1`。
   - `gpu_blocks=2`：`gate_pass=true`、`elapsed_s=29.56653244420886`、`h2d_copies=391`、`d2h_copies=6`、`evictions=395`、`resident_blocks=2`。
   - `gpu_blocks=4`：`gate_pass=true`、`elapsed_s=28.152612544596195`、`h2d_copies=249`、`d2h_copies=6`、`evictions=251`、`resident_blocks=4`。
   - 结论：主要开销仍来自 staging slot 容量导致的反复 H2D/evict，下一步更值得做减少跨 window 的重复 staging / 合并连续 prefetch，而不是继续只消除 Python 空调用。
32. 部分落地：blockwise prefill read-window caller 增加 next-window future hint。当前 prefix window stage 时，把紧邻下一个 prefix window 加入 `future_logical_blocks`，但不加入 protected/capacity 集合；因此不改变容量边界，只影响 `lru_cost` eviction 评分，减少稍宽 staging slots 下即将复用 block 被提前 evict 的概率。
   - TDD RED：新增 `tools/test_blockwise_attention_planning.py::test_blockwise_prefill_read_windows_hint_next_prefix_blocks`，远程旧实现按预期 `AssertionError`，因为 caller 没有传下一个 prefix window。
   - 远程目标测试通过：`tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`。
   - 集成 smoke：`SMOKE_TAG=20260709_prefill_next_window_hint RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34761 MASTER_PORT=34761 tools/smoke_blockwise_prefill_remote.sh`。
     - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_prefill_next_window_hint.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
     - 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_prefill_next_window_hint.json`，`gate_pass=true`、`elapsed_s=30.453897550702095`、`output_tokens=1`。
   - GPU blocks matrix：`SMOKE_TAG=20260709_prefill_next_window_hint_matrix RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=1 RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34762 MASTER_PORT=34762 tools/smoke_blockwise_prefill_remote.sh`。
     - `gpu_blocks=1`：预期容量失败，仍报 `required=2, gpu_blocks=1`。
     - `gpu_blocks=2`：`gate_pass=true`、`elapsed_s=29.297932274639606`、`h2d_copies=391`、`d2h_copies=6`、`evictions=395`、`resident_blocks=2`。
     - `gpu_blocks=4`：`gate_pass=true`、`elapsed_s=28.453177761286497`、`h2d_copies=193`、`d2h_copies=6`、`evictions=195`、`resident_blocks=4`。
   - 结论：`gpu_blocks=2` copy pressure 基本不变；`gpu_blocks=4` 相比上一轮 matrix 的 H2D 249→193、evict 251→195，说明 future hint 能减少重复 cross-window staging。单次 wall-clock 仍受远程环境影响，不单独作为性能结论。
33. 待做：继续合并连续 prefetch plan、降低 clean eviction 抖动，尤其把 next-window hint 扩展为可控多窗口 lookahead；后续再考虑 Triton/FlashAttention 风格 window kernel。
34. 部分落地：write-block staging 已抽成 `_stage_kv_offload_write_blocks()` 并同时复用于 blockwise prefill/decode；下一步继续抽象统一的 KV block access planner，让 prefill/decode 共享 `plan_read_blocks()`、`stage_blocks()`、`evict_blocks()`、`commit_write_blocks()` 语义。
35. 已落地：DFlash feasibility spike 已写入 `docs/dflash-feasibility.md`，只做接口/接入点预研，不直接实现完整 DFlash；已完成 Phase 1，把 n-gram target verify/commit 包装为通用 `verify_and_commit_block()` 并保留 n-gram 行为不变。远程 Qwen3-0.6B candidate-only smoke 已验证 `commit_event.draft_source="ngram"`。Phase 2 plumbing 已验证：新增 `--draft-source {ngram,dflash-toy,dflash-toy-ngram-or-repeat}`、`--allow-zero-accept`、deterministic `repeat_recent_tokens` toy block draft model，以及记录 zero-accept attempts 的 `verify_events`；远程 `dflash-toy` smoke `gate_pass=true`、`zero_accept_events=3`、`verify_events[].draft_source="dflash-toy"`。
   - 2026-07-07 accepted-friendly toy smoke 已通过：`--draft-source dflash-toy-ngram-or-repeat` 输出 `profile_out/dflash_phase2_toy_hybrid_candidate_smoke_20260707.json`，`gate_pass=true`、`commit_events=1`、`accepted_count=2`、`acceptance_rate=1.0`、`draft_metadata.toy_strategy="ngram_or_repeat"`、`draft_metadata.selected_strategy="ngram"`、`match_start=7`、`ngram_size=3`。
   - 2026-07-07 profiler-only hidden debug 已通过：`--debug-target-hidden` 输出 `profile_out/dflash_phase2_hidden_debug_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`target_hidden_debug.shape=[3, 1024]`、`target_hidden_debug.dtype="torch.bfloat16"`、`target_hidden_debug.device="cuda:0"`。该路径只在 profiler verify hook 调 `run_model(..., return_hidden=True)`，不改 `LLMEngine.step()` 或核心 runtime。下一步若继续 DFlash，应先做真实 draft model stub / hidden-to-draft adapter 的 profiler-only 实验。
   - 2026-07-07 hidden-to-draft adapter stub 已通过：`--debug-hidden-to-draft-stub --debug-hidden-to-draft-top-k 2` 输出 `profile_out/dflash_phase3_hidden_to_draft_stub_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_stub.adapter="target_hidden_topk_stub"`、`shape=[3, 1024]`、`dtype="torch.bfloat16"`、`device="cuda:0"`、`top_k=2`、`rows=2`；preview 中 row 0 top tokens `[13440, 8287]`，row 1 top tokens `[21619, 13440]`。stub 只记录 target hidden metadata 和 verify logits top-k preview，不采样、不替换 draft tokens、不改 acceptance rule/runtime。
   - 2026-07-07 hidden-to-draft adapter interface schema/timing 已通过：输出 `profile_out/dflash_phase3_adapter_interface_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_stub.interface_version=1`、`runtime_mutation=false`、`input_schema.hidden_states.shape=[3, 1024]`、`input_schema.logits.shape=[2, 151936]`、`output.draft_token_ids=[13440, 21619]`、`timing_ms.adapter_total_ms=142.16194674372673`、`logits_to_cpu_ms=8.412063121795654`、`topk_ms=133.64192843437195`。当前 timing 是 Python profiler preview 成本，不代表未来优化 adapter latency；价值是固定 profiler-only ABI 和计时字段。
   - 2026-07-07 `--hidden-to-draft-adapter {topk-stub,linear-stub}` 和 `linear-stub` skeleton 已通过：第一次 `TINYVLLM_DIST_PORT=34574 MASTER_PORT=34574` 失败于 `EADDRINUSE`，换 `34674` 重跑成功；输出 `profile_out/dflash_phase3_linear_stub_interface_smoke_20260707.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_adapter="linear-stub"`、`hidden_to_draft_stub.adapter="target_hidden_linear_stub"`、`input_schema.adapter="linear-stub"`、`output_schema.projection="deterministic_placeholder"`、`output.draft_token_ids=[13440, 21619]`、`timing_ms.linear_projection_ms=0.0002868473529815674`。`linear-stub` 当前只改变 profiler JSON 和 no-op projection timing，不参与 acceptance/runtime。
   - 2026-07-07 `linear-stub` deterministic hidden projection skeleton 已通过：把 no-op projection 换成固定 seed/pseudo weight 的 hidden rows -> 小 vocab candidate set 投影，仍只写 profiler JSON，不参与 draft proposal、acceptance、commit 或 runtime。第一次 `TINYVLLM_DIST_PORT=34676 MASTER_PORT=34676` 失败于 `EADDRINUSE`，换 `35676` 重跑成功；输出 `profile_out/dflash_phase3_linear_projection_stub_smoke_20260707_r2.json`，`gate_pass=true`、`accepted_count=2`、`hidden_to_draft_stub.adapter="target_hidden_linear_stub"`、`output_schema.projection="deterministic_hidden_linear_stub"`、`projection_metadata.seed=17`、`candidate_token_ids=[0,1,2,3,4,5,6,7]`、`hidden_dim=1024`、`candidate_count=8`、`output.draft_token_ids=[0,0,7]`、`rows=3`、`output.num_rows=3`、`input_schema.hidden_states.shape=[3,1024]`、`input_schema.logits.shape=[3,151936]`、`timing_ms.hidden_to_cpu_ms=0.17523393034934998`、`linear_projection_ms=3.5200342535972595`、`adapter_total_ms=12.286126613616943`。已修复 hidden rows 与 logits rows 数量不一致时 `rows`/`output.num_rows` 误按 logits rows 计数的问题。
   - 2026-07-07 `topk-stub` vs `linear-stub` 3x remote compare 已通过：固定端口序列第二个进程仍遇到 `EADDRINUSE`，改为远程动态探测空闲端口后 6 个 JSON 均成功（`profile_out/dflash_phase3_adapter_compare_{topk-stub,linear-stub}_r{1,2,3}_20260707_cmp2.json`）。`topk-stub` 三次均 `gate_pass=true`、`output.rows=2`、`draft_token_ids=[13440,21619]`、`projection="logits_topk"`、`adapter_total_ms=124.903983±1.449532`、`topk_ms=116.005514±1.719655`；`linear-stub` 三次均 `gate_pass=true`、`output.rows=3`、`draft_token_ids=[0,0,7]`、`projection="deterministic_hidden_linear_stub"`、`adapter_total_ms=12.578132±0.081115`、`hidden_to_cpu_ms=0.191076±0.019228`、`linear_projection_ms=3.593331±0.057019`。ABI 字段足够继续 profiler-only 真实 draft model stub，但真实接入前建议显式化 `hidden_rows`/`logit_rows`/`projected_rows` 并拆出 `draft_model_forward_ms`、`candidate_select_ms`。
   - 2026-07-07 已继续补齐 ABI/timing 字段：`input_schema` 新增 `hidden_rows`、`logit_rows`、`projected_rows`，`output_schema`/`output` 新增 `projected_rows`，`timing_ms` 新增 `draft_model_forward_ms` 与 `candidate_select_ms`；同时修正 `linear-stub` 下 `input_schema.logits.shape` 始终按真实 logits rows 记录，不再被 hidden projected rows 覆盖。本地测试已覆盖 hidden rows=3、logit rows=2、projected rows=3 的不一致场景；远程 `profile_out/dflash_phase3_adapter_abi_fields_smoke_20260707.json` 验证 `gate_pass=true`、`accepted_count=2`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`input_schema.logits.shape=[2,151936]`、`output.projected_rows=3`、`candidate_select_ms=3.570154309272766`、`draft_model_forward_ms=0.0`。
   - 2026-07-07 已新增 profiler-only `draft-model-stub` adapter：CLI 支持 `--hidden-to-draft-adapter draft-model-stub`，输出 `target_hidden_draft_model_stub`、`projection="deterministic_draft_model_stub"`、`output.candidate_token_ids`、`output.candidate_logits`、`draft_model_metadata`，并真实占用 `draft_model_forward_ms`；仍保持 `runtime_mutation=false`，不影响 draft proposal/acceptance/runtime。远程第一次用 GPU7 失败于 `assert auto_num_blocks > 0`，原因是 GPU7 显存占用约 69GiB/80GiB；改用 GPU3 后 `profile_out/dflash_phase3_draft_model_stub_smoke_20260707_gpu3.json` 通过，`gate_pass=true`、`accepted_count=2`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=3.5831667482852936`、`candidate_select_ms=0.021237879991531372`。commit 的 `draft_tokens`/`accepted_tokens` 仍是 `[13440,21619]`，证明 stub 未参与 acceptance/runtime。
   - 2026-07-07 已补 3-way adapter compare：GPU4 上对 `topk-stub`、`linear-stub`、`draft-model-stub` 各跑 3 次，输出 `profile_out/dflash_phase3_adapter_3way_compare_{topk-stub,linear-stub,draft-model-stub}_r{1,2,3}_20260707_cmp3.json`。9 次均 `gate_pass=true`、`accepted_count=2`，commit `draft_tokens`/`accepted_tokens` 均为 `[13440,21619]`。`draft-model-stub` 三次 candidate ids/logits、`draft_token_ids=[7,7,7]`、metadata 完全稳定；timing 均值/标准差：`adapter_total_ms=13.104054±0.257300`、`draft_model_forward_ms=3.721040±0.024405`、`candidate_select_ms=0.021578±0.001956`、`hidden_to_cpu_ms=0.349854±0.030066`。对照：`topk-stub adapter_total_ms=131.355740±6.488973`，`linear-stub adapter_total_ms=12.964208±0.502438`。下一步若继续，建议把 deterministic pseudo forward 边界抽成独立函数/类，再替换真实 draft model forward，仍保持 profiler-only `runtime_mutation=false`。
   - 2026-07-08 已把 deterministic pseudo forward 抽成 `run_draft_model_stub(hidden_rows, candidate_token_ids, top_k)`，输出 `candidate_token_ids`、`candidate_logits`、`draft_token_ids`、`draft_scores`、`preview`、`metadata`、`timing_ms`；`summarize_hidden_to_draft_stub(..., adapter="draft-model-stub")` 复用该 helper，方便后续替换真实 draft model forward。远程 `profile_out/dflash_phase3_draft_model_stub_boundary_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=3.5160109400749207`、`candidate_select_ms=0.015269964933395386`，commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。
   - 2026-07-08 已把 draft model boundary 包成 dataclass/config shell：新增 `DraftModelStubConfig(seed, stub_version)`、`DraftModelResult(...).to_dict()`，`run_draft_model_stub(..., config=None) -> DraftModelResult`，并显式校验 empty candidates 与 ragged hidden rows；这是未来真实 draft model forward 的 profiler-only 返回/错误边界，不接入 `LLMEngine.step()`。本地 `tools/test_ngram_speculative.py`、`py_compile`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `tools/test_ngram_speculative.py` 与 Qwen3 smoke `profile_out/dflash_phase3_draft_model_dataclass_shell_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`input_schema.adapter="draft-model-stub"`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_metadata.stub_version=1`、`draft_model_forward_ms=4.899017512798309`、`candidate_select_ms=0.01652538776397705`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。
   - 2026-07-08 已继续补输入侧 contract：新增 `DraftModelInput.from_rows(...).to_dict()/schema()`，显式记录 `hidden_rows`、`candidate_token_ids`、`top_k`、`source_shape`、`source_dtype`、`source_device`；`run_draft_model_stub()` 保持旧签名兼容，也可直接接收 `DraftModelInput`。本地 `tools/test_ngram_speculative.py`、`py_compile`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `profile_out/dflash_phase3_draft_model_input_contract_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`draft_model_metadata.input_schema.source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`、`top_k=2`、`output.draft_token_ids=[7,7,7]`、第一行 candidate ids `[7,1]`、`draft_model_forward_ms=5.126491189002991`、`candidate_select_ms=0.02197548747062683`；commit `draft_tokens`/`accepted_tokens` 仍为 `[13440,21619]`。下一步若继续 DFlash，建议做多 prompt/batch shape smoke 或把 profiler-only draft model schema 抽成小模块，仍不要接 runtime。
   - 2026-07-08 已完成多 prompt / batch shape smoke：远程 `profile_out/dflash_phase3_draft_model_batch_shape_smoke_20260708.json` 使用 2 个 prompt、`--max-num-seqs 2`、`draft-model-stub`、`top_k=2`，结果 `gate_pass=true`、`num_prompts=2`、`commit_events=2`、`accepted_count=4`。每个 prompt 各 `commit_events=1`、`verify_events=1`、`accepted_count=2`；两个 event 的 `draft_model_metadata.input_schema` 都独立记录 `hidden_rows=3`、`hidden_dim=1024`、`candidate_count=8`、`top_k=2`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`。prompt0 commit tokens `[13440,21619]`、draft model ids `[7,7,7]`、第一行 candidate `[7,1]`；prompt1 commit tokens `[6303,6176]`、draft model ids `[1,2,2]`、第一行 candidate `[1,2]`。结论：batch 场景下 event 级 DraftModelInput schema 未发生 prompt 间串写，仍保持 `runtime_mutation=false`。下一步若继续 DFlash，建议把 profiler-only draft model schema 抽成小模块或补更多形状覆盖，仍不要接 runtime。
   - 2026-07-08 已把 profiler-only draft model schema 抽成 `tools/draft_model_schema.py` 小模块，包含 `DraftModelInput`、`DraftModelResult`、`DraftModelStubConfig`；`tools/profile_ngram_commit.py` 只导入这些 dataclass，测试直接加载该模块并确认 profiler 返回同一类对象。本地 `tools/test_ngram_speculative.py`、`py_compile tools/draft_model_schema.py tools/profile_ngram_commit.py tools/test_ngram_speculative.py`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `profile_out/dflash_phase3_draft_model_schema_module_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`，commit `draft_tokens`/`accepted_tokens=[13440,21619]` 不变。下一步若继续，建议补更多形状覆盖或做真实 draft model 接入前置检查（vocab/tokenizer/hidden_dim contract），仍不要接 runtime。
   - 2026-07-08 已补真实 draft model 接入前置 contract 检查：`tools/draft_model_schema.py` 新增 `DraftModelContract` 与 `validate_draft_model_contract()`，支持 profiler-only 校验 `expected_hidden_dim`、`target_vocab_size`、`draft_vocab_size`、`tokenizer_family` / `draft_tokenizer_family`，并记录 candidate token id min/max；本地测试覆盖 hidden_dim mismatch、candidate id 超 target vocab、tokenizer family mismatch。`run_draft_model_stub()` 默认记录宽松 `contract` metadata，也可传显式 contract；远程第一次 smoke 因 `EADDRINUSE` 失败，换动态高端口后 `profile_out/dflash_phase3_draft_model_contract_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`contract.compatible=true`、`actual_hidden_dim=1024`、`candidate_id_min=0`、`candidate_id_max=7`，commit `draft_tokens`/`accepted_tokens=[13440,21619]` 不变。下一步若继续，建议补更多形状覆盖，仍不要接 runtime/真实 checkpoint。

#### 47.48.7 尝试记录 / 排坑记录（2026-07-06）

1. 远程 SSH / Kerberos：
   - 裸 `ssh sitian@10.232.195.203` 在 TRAE 进程里曾报 `Connection closed by UNKNOWN port 65535`，根因不是目标机 22 端口不可达，而是当前进程看不到 macOS API Kerberos cache。
   - `nc -vz 10.232.195.203 22` 成功，说明目标端口可达；`klist` 在 TRAE 进程里报 `Cache not found: API:11111111-...`，而用户外部 Terminal 里 `klist` 显示 `sitian@BYTEDANCE.COM` 可用。
   - `jump-proxy-hl` 不允许普通 session，`ssh sitian@jump-proxy-hl 'echo jump-ok'` 会报 `unknown channel type: session`；正确验证方式是直连目标 `ssh sitian@10.232.195.203 'echo remote-ok'` 或用 `-W` 作为 ProxyCommand。
   - 已把 `~/.ssh/config` 的 `jump-proxy-hl` / `jump-proxy-hla` / `jump-proxy-lf` 用户显式改为 `sitian`，避免 ProxyCommand 默认用本机用户 `bytedance`。
   - 稳定方案：使用 file cache `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` 建 ControlMaster，例如 `ssh -MNf -o ControlMaster=yes -o ControlPath=/tmp/ssh-sitian-10.232.195.203-new -o ControlPersist=2h sitian@10.232.195.203`；后续 rsync/ssh 使用 `ssh -S /tmp/ssh-sitian-10.232.195.203-new ...`。
   - 老 socket `/tmp/ssh-sitian-10.232.195.203` 偶发 `Connection closed by UNKNOWN port 65535`；遇到时重新建新 ControlMaster 即可。
2. rsync 同步：
   - 第一次多源 rsync 没用 `--relative`，曾把 `qwen3-8b-fixes.md`、`test_chunked_prefill.py` 临时同步到远程 repo 根目录；已删除这两个误同步副本。
   - 正确同步方式：`rsync -av -e 'ssh -S /tmp/ssh-sitian-10.232.195.203-new -o BatchMode=yes' --relative <files> sitian@10.232.195.203:/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/`。
3. GPU blocks matrix：
   - `gpu_blocks=1` 失败是当前实现边界：`required=2, gpu_blocks=1`，因为 blockwise prefill 需要 `prefix window blocks + 当前 chunk write blocks <= gpu staging slots`；不要把它当 correctness mismatch。
   - `gpu_blocks=2/4` 已通过；`gpu_blocks=4` 的 H2D/eviction 次数比 2 少，符合 staging slots 增多的预期。
4. Multi-prompt batch smoke：
   - 默认 `MAX_MODEL_LEN=4096`、`GPU_MEMORY_UTILIZATION=0.7`、`MULTI_PROMPT_COUNT=2` 初次失败在模型初始化阶段：`allocate_kv_cache()` 里 `assert auto_num_blocks > 0`，不是 attention correctness 失败。
   - 可用参数是降低 `MAX_MODEL_LEN=2048`、提高 `GPU_MEMORY_UTILIZATION=0.85`、缩短 `MULTI_PROMPT_REPEAT=24`，最终 `num_prompts=2` 通过。
5. 本地验证注意：
   - 本地 `python3 -m py_compile` 若写 macOS 系统 pyc cache 失败，可用 `PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache`。
   - 当前未跟踪的 `.agents/`、`.codex/`、`needle_sq_results/` 是无关本地/实验产物，之前提交均未纳入。
