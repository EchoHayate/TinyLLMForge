# 多路径回归：cuda graph / cpu_offload / TP=2

> 本卷收纳"在不修任何核心算法的前提下，把所有路径在新维度（cuda graph / cpu_offload bench / TP=2）上跑一遍 smoke + 修补回归 bug"的工程日志。
> 算法实现见 `kv-sparse-attention.md`（Quest）、`w4a8c4-quantization.md`（C4 / W4A8 / β3）。

## 11. CUDA Graph 兼容性回归

### 11.1 动机
前几波改动（β3 sparse dequant / cpu_offload 加 prefetch stream / W4A8 fake-quant / Quest 选块）都只在
`enforce_eager=True` 下做 smoke。打开 cuda graph 后哪些路径会在 capture 阶段崩、哪些 capture 通过、
哪些 capture 通过但 replay 数值崩——之前没有系统性回归。

### 11.2 跑法
新建 `tools/cuda_graph_smoke.py`：每条路径在独立子进程里 init `LLM(enforce_eager=False, ...)` + `generate`，
捕获 init / generate 阶段的异常，输出汇总表。

```
CUDA_VISIBLE_DEVICES=3 python tools/cuda_graph_smoke.py \
  --model /path/to/Qwen3-0.6B \
  --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.55 \
  --out-json cuda_graph_smoke_final.json
```

覆盖配置：`baseline / quest / c4_only / c4_quest / w4_g128 / w4a8_g32 / cpu_offload`。

### 11.3 第一次跑：cpu_offload 在 capture 阶段崩

| label | init | gen | init_s | err |
|---|---|---|---:|---|
| baseline | ok | ok | 55.4 | |
| quest | ok | ok | 54.6 | |
| c4_only | ok | ok | 30.9 | （`config.kv_quant_bits==4` 时 capture 已被 init 阶段 skip） |
| c4_quest | ok | ok | 32.2 | （同上） |
| w4_g128 | ok | ok | 55.0 | |
| w4a8_g32 | ok | ok | 53.7 | |
| **cpu_offload** | **FAIL** | FAIL | 33.4 | `RuntimeError: CUDA error: operation failed due to a previous error during capture` |

cpu_offload 在 capture 阶段崩的根因：cpu_offload 用独立 stream 异步把下一层权重 H2D，
然后用 `event.wait(compute_stream)` 同步。cuda graph 的 capture mode 只允许"一个 stream 上的 op 序列"被录制，
跨 stream 的 H2D + event sync 会触发"previous error during capture"。

### 11.4 修复：init/run_model 双侧加 cpu_offload skip

`tinyvllm/engine/model_runner.py`：

```python
# init 阶段（构造 ModelRunner 末尾）
skip_cudagraph = (
    self.enforce_eager
    or config.kv_quant_bits == 4   # 已有
    or config.cpu_offload          # 新增
)
if not skip_cudagraph:
    self.capture_cudagraph()

# run_model 阶段
offload_active = self.config.cpu_offload
if (is_prefill or self.enforce_eager or input_ids.size(0) > 512
        or quest_active or c4_active or offload_active):
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    # graph replay 路径
    ...
```

两处都改是必要的：init 阶段 skip 了 capture 后 `self.graphs` 不存在，
run_model 还走 replay 分支会 AttributeError。

### 11.5 修复后全量回归：7/7 PASS

| label | init | gen | init_s | gen_s | 备注 |
|---|---|---|---:|---:|---|
| baseline | ok | ok | 58.1 | 28.3 | 走 cuda graph |
| quest | ok | ok | 54.0 | 26.9 | capture 通过；runtime 触发 quest_active 时退 eager |
| c4_only | ok | ok | 32.3 | 28.6 | init 阶段 skip capture，全程 eager |
| c4_quest | ok | ok | 31.4 | 28.6 | 同上 |
| w4_g128 | ok | ok | 54.9 | 27.9 | 走 cuda graph |
| w4a8_g32 | ok | ok | 53.5 | 26.9 | 走 cuda graph（act fake-quant 是 elementwise，可 capture） |
| cpu_offload | ok | ok | 30.9 | 27.5 | init 阶段 skip capture（修复后） |

观察：
- **init 时间分两档**：能 capture 的 ~54s（含 ~25s 录 graph），skip 的 ~31s
- **gen 时间几乎一样**（27-28s）：因为 prompt 短 + max_tokens=16，graph 收益被 prefill / sampler 稀释；
  这跟 §10 / profiler 的结论一致——0.6B 上 attention 不是瓶颈，cuda graph 只省 launch overhead，
  在小 batch 短 prompt 上看不出明显差异。这是预期，不是回归
- **生成的文本**：6 条用 temperature=0 跑 `"The quick brown fox"` 都拿到 `"jumps over the lazy dog. ..."`；
  w4 / w4a8 是量化版（temperature=0 仍带轻微数值漂移），文本继续走"trees / story is about a fox"，
  读起来连贯，没崩成乱码

### 11.6 已知不修
- ❌ **让 c4 路径也接 cuda graph**：sparse dequant 每步要 alloc 一个 size 依赖于 batch 实际 block 数的临时 buffer，
  这是动态 alloc，capture 不进。要修的话得把 dequant buffer 改成"pre-allocated 固定 max_blocks 大小，每步只读一部分"，
  代价是显存 worst-case 始终占用，§10.6 已留过这个 known-no
- ❌ **让 cpu_offload 也接 cuda graph**：cuda graph 的 capture 不支持 cross-stream 异步 H2D；
  workaround 是在 capture 前同步 wait 完所有层权重——那就退化为同步 offload，prefetch 收益消失，得不偿失

### 11.7 文件留痕
- `tools/cuda_graph_smoke.py`（新建，~150 行）：spawn 子进程跑 7 条路径的 cuda graph 兼容性回归
- `tinyvllm/engine/model_runner.py`：init 阶段 skip 条件改成 `enforce_eager or kv_quant==4 or cpu_offload`，
  run_model 也加 `offload_active`
- `cuda_graph_smoke_final.json`（修复后全量结果，7/7 PASS）

## 12. cpu_offload 吞吐基线测量（2026-05-29）

### 12.1 动机

§11 只证明了 cpu_offload 在 cuda graph 路径下不崩（init 阶段 skip capture），但全程 eager 跑下来
decode TPS 损失多少、显存到底省没省，还没量化。这是回答"0.6B 上 cpu_offload 实用吗"必须有的数据。

### 12.2 跑法

新建 `tools/bench_offload.py`：spawn 子进程跑每个 `(mode, ctx)` 组合，避免 cpu_offload 的独立 stream /
异步 H2D buffer 污染下一轮，也保证 `torch.cuda.max_memory_allocated()` 干净。

每个子进程：
1. warmup 一次后 `reset_peak_memory_stats()`，确保峰值测的是稳态
2. 手动 `add_request` + `step`，用 `step()` 返回的 `num_tokens` 符号区分 prefill (>0) / decode (<0)
3. 每步 `torch.cuda.synchronize()` 后计时
4. 结束时输出 `prefill_tps` / `decode_tps` / `peak_mem_gb`

### 12.3 结果

A100 + Qwen3-0.6B：

| mode | ctx | num_seqs | prefill_tps | decode_tps | peak_mem_gb |
|---|---|---|---|---|---|
| baseline | 4096 | 16 | 77552.56 | **266.26** | 1.943 |
| offload | 4096 | 16 | 63211.61 | **61.10** | 2.201 |
| baseline | 15000 | 4 | 60075.22 | **512.19** | 14.111 |
| offload | 15000 | 4 | 49660.81 | **74.10** | 14.137 |

decode_tps 相对损失：
- ctx=4k：61.10 / 266.26 = **0.23×（-77%，慢 4.36×）**
- ctx=15k：74.10 / 512.19 = **0.14×（-86%，慢 6.9×）**

### 12.4 反常观察：peak_mem 几乎不省

| ctx | baseline peak_mem_gb | offload peak_mem_gb | 节省 |
|---|---|---|---|
| 4096 | 1.943 | 2.201 | **-0.26（反而多）** |
| 15000 | 14.111 | 14.137 | -0.026 |

这看起来很反直觉——cpu_offload 不就是为了省显存吗？拆开看：

- **0.6B 模型权重总共 ~1.2 GB**（fp16），其中保留在 GPU 的 embedding/lm_head/最后两层 + 当前层 +
  prefetch 下一层 ≈ 0.6 GB；理论上能省 ~0.6 GB 权重
- **但 cpu_offload 自己还引入了** pinned host buffer + prefetch GPU buffer + 独立 stream 的 workspace，
  这部分加起来抵消了节省
- **ctx=4k 时甚至更高**：因为 KV cache 容量按 `gpu_memory_utilization * 总显存 - 当前已用` 反算，
  offload 模式下 init 时 GPU 已用更少 → 自动分配更多 KV blocks → 总占用反而上去

也就是说：在 0.6B 这种"权重本来就放得下"的场景，cpu_offload **省的那点权重显存又被它自己的 buffer
和扩大的 KV 池吃回去了**，净节省 ≈ 0。

### 12.5 结论：0.6B 上 cpu_offload 是纯负担

| 场景 | 结论 |
|---|---|
| 模型权重 < GPU 显存（如本仓库 0.6B） | ❌ 不要开。decode 慢 4–7×，显存几乎不省 |
| 模型权重远超 GPU 显存（如 70B 单卡） | cpu_offload 才是它的设计场景 |

cpu_offload 在本仓库保留意义：作为"装不下大模型"的后备路径 + cuda graph 兼容性回归用例，
不作为 0.6B 推理的常规配置推荐。

### 12.6 已知不做

- ❌ **优化 0.6B 上 cpu_offload 性能**：根因是 0.6B 权重 H2D 时间和单步 decode 计算时间几乎同量级，
  prefetch 重叠空间很小；继续优化是把"装得下的模型卸载到 CPU"这条路本身做得更好，性价比低
- ❌ **改用 zero-copy 共享 host 内存**：A100 走 PCIe，host-pinned 已经是极限；要进一步省得换硬件
- ❌ **测 1B+ 模型上的 cpu_offload**：本仓库聚焦 Qwen3-0.6B 配套优化，不扩到更大模型

### 12.7 文件留痕

- `tools/bench_offload.py`（新建，~210 行）：spawn 子进程跑 (mode × ctx) 组合的吞吐 + peak mem 测量
- `bench_offload_out/{baseline,offload}_ctx{4096,15000}.json`：每条配置的详细数据（A100 远端，不入仓库）
- `bench_offload_out/summary.json`：4 行汇总

## 13. TP=2 多卡路径兼容性回归（2026-05-29）

### 13.1 动机

仓库历史上多数优化（Quest / C4 / W4 / W4A8 / cpu_offload / cuda_graph）都是单卡上写的，
TP 路径作为后写入的一条横切，跟它们正交但没有交叉验证过。这一波目标：
**把 §6–§12 攒下来的 8 条配置，在 TP=2 上各跑一遍 smoke，证明它们都不崩、能出 token**。

### 13.2 跑法

新建 `tools/tp_smoke.py`：spawn 子进程串行跑每条 config（端口 2333 + shm name `tinyvllm` 都硬编码，
不能并发）。每条记 init_ok / gen_ok / decode_tps / peak_mem_gb / text_sample / error。

跑前清理 `/dev/shm/tinyvllm`，避免上一轮 worker 异常退出残留文件污染下一条。
prompt 用 16 条随机 token id（避免 tokenizer 不一致），warmup 一次后清峰值。

### 13.3 审计修出来的 4 个 TP bug

跑之前先走 Explore 审计了一遍 TP 实现，发现 4 处实际跑起来肯定崩的位置，先修后跑：

**(1) `tinyvllm/layers/linear.py` — QKVParallelLinear 双切**

```python
# 修复前：output_size 已经是切完的，再传给 ColumnParallelLinear 又会被切一次
output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
# 修复后：传 total（未切分）的，让 ColumnParallel 自己切
output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * head_size
```
症状：`narrow start (1024) + length (512) exceeds dimension size (1024)`，weight loader 越界。

**(2) `tinyvllm/layers/embed_head.py` — ParallelLMHead 拼接维度错**

```python
# 修复前：沿 batch 维拼接 → [N*tp, vocab/tp]，sampler 维度对不上
logits = torch.cat(all_logits, 0)
# 修复后：沿 vocab 维拼接 → [N, vocab]
logits = torch.cat(all_logits, 1)
```
症状：`size of tensor a (8) must match the size of tensor b (4) at non-singleton dimension 0`。

**(3) `tinyvllm/engine/sequence.py` — `__setstate__` 解包错**

```python
# 修复前：对 state[-1]（一个 token_ids 或 last_token）解包 4 个变量 → ValueError
self.num_tokens, ..., self.block_table = state[-1]
# 修复后：state[:4] 解包前 4 个；num_cached_blocks 是 @property，反推回 num_cached_tokens
self.num_tokens, self.num_prompt_tokens, num_cached_blocks, self.block_table = state[:4]
self.num_cached_tokens = num_cached_blocks * self.block_size
```

**(4) `tinyvllm/engine/model_runner.py` — `call()` 传参未展开**

```python
# 修复前：args（tuple）作为单个参数写进去，worker 端 *args 解包后多了一层 tuple
self.write_shm(method_name, args)
# 修复后：展开
self.write_shm(method_name, *args)
```
症状：`ModelRunner.run() missing 1 required positional argument: 'is_prefill'`。

### 13.4 结果

A100 + Qwen3-0.6B + TP=2，8/8 init_ok + gen_ok：

| config | decode_tps | peak_mem_gb (rank0) |
|---|---|---|
| baseline | 206.07 | 53.359 |
| quest | 188.24 | 53.359 |
| c4_only | 134.62 | 53.371 |
| c4_quest | 131.80 | 53.369 |
| w4_g128 | 124.75 | 53.283 |
| w4a8_g128 | 98.09 | 53.076 |
| cpu_offload | 147.67 | 53.358 |
| cuda_graph_baseline | **1798.18** | 53.368 |

### 13.5 异常观察

**(1) peak_mem ~53GB / rank**：单卡 0.6B baseline 才 ~2GB，这里两卡各 53GB（合计 106GB）。
说明 KV cache 容量按 `gpu_memory_utilization=0.7` 反算时每张卡都拿到了"独立的 70%"，
没有按 TP 切；而 weight 也没有真切到一半（gpu_memory_utilization 默认抽到的剩余空间被吃满了）。
这一波**只验兼容性**，"TP 是否真省显存"留给后续真做大模型时再追。
> 注：§15（`qwen3-8b-fixes.md`）已纠错——TP 一直在切 weight，是 peak 这个度量本身把 KV cache 算进来了。

**(2) cuda_graph_baseline decode_tps=1798 异常高**：单卡 §11 同样配置只有 ~250 tps。
怀疑是 NCCL × cuda graph capture 路径把 all-reduce 吞掉了 / 或者 timer 被异步队列吃掉，
不应当作真实吞吐采纳。这里只读 init_ok + gen_ok 两个布尔位，TPS 数字打个 ⚠️ 标记。
> 注：§18（`qwen3-8b-fixes.md`）已纠错——这个数字在重做对照后是合理的，不是异常。

**(3) c4_quest / w4_g128 文本明显复读**（`here here here here ...`）：
这是 0.6B 模型在 W4 / KV4 叠加下的精度坍塌，不是 TP 的锅（单卡 §10 就观察到过类似现象），
通过 `ignore_eos=True` 强制跑满 32 token 才显得明显。smoke 不评估生成质量，gen_ok 只看是否非空。

### 13.6 结论

| 维度 | 状态 |
|---|---|
| 8 条历史路径在 TP=2 上是否都能 init + 出 token | ✅ 全部 PASS |
| TP 真正省显存 | ❌ 没省（rank0 = 53GB），需要后续在 7B+ 模型上验真 TP 切分（已在 §15 纠错） |
| TP × cuda graph | ⚠️ 不崩，但 TPS 数据可疑，不作为吞吐结论（已在 §18 纠错） |
| TP × cpu_offload | ✅ 不崩 |
| TP × KV4 / W4 / W4A8 | ✅ 不崩，质量是另说 |

### 13.7 已知不做

- ❌ 在 0.6B 上对比 TP=1 vs TP=2 吞吐：0.6B 单卡就放得下，TP 通信开销只会让吞吐变差，没意义
- ❌ 调 TP × cuda graph 的真实 TPS：要先解决 NCCL × graph capture 的同步语义，工程量大且偏离主线
- ❌ 修 c4_quest / w4 路径的复读：这是模型量化精度问题，不是 TP 引入的回归

### 13.8 文件留痕

- `tools/tp_smoke.py`（新建，~240 行）：spawn 子进程跑 8 条 config 的 TP=2 兼容性回归
- `tinyvllm/layers/linear.py`：QKVParallelLinear 双切修复（output_size 传 total）
- `tinyvllm/layers/embed_head.py`：ParallelLMHead cat 维度修复（dim=0 → dim=1）
- `tinyvllm/engine/sequence.py`：`__setstate__` 解包修复 + num_cached_blocks @property 绕过
- `tinyvllm/engine/model_runner.py`：`call()` 传参 `*args` 展开
- `tp_smoke_out/tp_smoke.json`：8 条 config 的详细结果（A100 远端，不入仓库）
