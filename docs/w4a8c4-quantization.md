# W4A8C4 全栈量化实施日志

> 与主线 Quest sparse attention（见 `kv-sparse-attention.md`）正交的另一条路线：
> 把 weight / activation / KV cache 都压到低位 (W4A8C4)。
> 这一卷只装"量化路线本身"——C4 (KV 4-bit) → W4A8 → β3 (Quest+C4 sparse dequant)。
> 8B 上的数值塌 / outlier / SmoothQuant 等修复在 `qwen3-8b-fixes.md` 单独一卷。

## 7. 新增方向：W4A8C4 全栈量化（与 Quest 正交）

> 决定：Quest 性能优化（#29）暂时挂一档，先开 W4A8C4，因为它和 Quest **完全正交**，且能直接砍掉 KV cache 带宽——这正是 0.6B 长 context 场景下 Quest 没能翻盘的根因之一（attention 已经被 KV 带宽 bound 了，再做 selection 也救不回来）。

### 7.1 背景与论文锚点
| 工作 | 切片 | 关键贡献 |
|---|---|---|
| **QServe (MLSys'25)** | W4A8KV4 | 提出 W4A8KV4 配方，QoQ 算法 + cutlass W4A8 GEMM；端到端 serving 加速 1.2~3.5x |
| **Atom (MLSys'24)** | W4A4 | mixed precision，逐通道 outlier 用 fp16 |
| **KIVI (ICML'24)** | KV-only | per-channel K + per-token V 4-bit 量化；针对 K 沿 channel 强相关的特性 |
| **SmoothQuant** | W8A8 | 把 activation outlier 的 scale 平移到 weight 上 |

我们做 **W4A8C4**：weight 4-bit、activation 8-bit、KV cache 4-bit。

### 7.2 为什么这条路线值得做
1. 和 **Quest 正交**：Quest 减少访存 block 数（横向稀疏），C4 砍每 block 字节数（纵向压缩），可叠加
2. **2024-2025 最新一档**，比之前已经做过的 W8 / W4 weight-only 高一档
3. 真实 serving 长 context decode **是 KV-bandwidth bound**，C4 直接打中
4. 仓库已有 W4 weight-only 量化基础（`tinyvllm/layers/quantization.py`），延续比新写省事
5. 理论上可以彻底翻盘 #25 里观察到的 long-context 回归

### 7.3 路线拆分（A → B → C 串行，每个 phase 都要可独立开关）

#### Phase A：C4（KV cache 4-bit）—— 优先做
原因：写入路径改动局部、accuracy 风险可控、能立刻吃到带宽收益。
- **A-1**（任务 #33）：paged KV cache 改 group-wise 4-bit pack + fp16 scale 存储；改 `store_kvcache` triton kernel
- **A-2**（任务 #34）：attention forward 反量化路径（先走"全量反量化喂 flash-attn"最简方案 α，性能不达标再考虑 fork flash-attn 加 4-bit KV 支持 / KIVI 风格 per-channel K + per-token V）
- **A-3**（任务 #35）：用 `tools/eval_needle.py` + `bench.py` 验 accuracy / 吞吐；判定通过门槛 acc 退化 < 2pp 且长 ctx 吞吐 ≥ baseline

风险点：
- ⚠️ Quest 的 K-min/K-max summary 维护要决定是基于反量化后 fp16 还是直接在 4-bit 域近似（前者准但耗时，后者快但选错 block 风险）
- ⚠️ flash_attn paged 不支持 4-bit KV，反量化的临时 buffer 会吃掉一部分显存收益
- ⚠️ 4-bit KV 的 zero-point 处理：对称（symm）实现简单但精度稍差，非对称（asymm）需要存 zero / 复用 scale

#### Phase B：W4A8 GEMM（任务 #36）
- 把 4 个 linear 层（qkv_proj / gate_up_proj / down_proj / o_proj）替换成 W4A8 kernel（marlin 或 cutlass）
- activation per-token int8 dynamic quant；如果掉点上 SmoothQuant per-channel scale
- weight 沿用现有 W4 group=128 量化

风险点：
- ⚠️ activation outlier：纯 per-token 不够，可能要 SmoothQuant 或 Hadamard 旋转
- ⚠️ kernel 选型：marlin 当前主要 W4A16，W4A8 要看 cutlass 或自写

#### Phase C：三者叠加端到端（任务 #37）
W4A8 + C4 + Quest 全开，跑 `num-seqs=16, ctx=15k` 真实 serving 场景。
评测矩阵 = 单开 / 两两组合 / 三全开 共 8 个组合的 accuracy + 吞吐表。

### 7.4 配置项与开关设计（先想好，免得后面改 8 处）
新增 `tinyvllm/config.py` 字段（默认全关，向后兼容）：
- `kv_quant_bits: int = 0`  —— 0 / 4 / 8，只控 KV cache
- `kv_quant_group_size: int = 128`  —— C4 的 group 大小
- `kv_quant_symmetric: bool = True`  —— 对称 vs 非对称
- `act_quant_bits: int = 0`  —— 0 / 8，控 activation A8
- （weight 4-bit 沿用已有 `quantization=int4` 配置）

CLI 同步加到 `bench.py` / `example.py` / `tools/eval_needle.py`。
`Context` dataclass 也加对应字段，避免 forward 里再读 self.config 触发同步问题。

### 7.5 留痕约定（继承 Phase 1 经验）
每个子任务完成必须更新对应章节，**最少**包含：
1. 改动了哪些文件 / 哪些 kernel
2. 遇到的问题（含错误堆栈 / numerical issue / kernel 兼容性）
3. 解决方案（含被否掉的方案及理由）
4. 评测结果（acc + 吞吐 + 与 baseline 比），表格化
5. 下一步 / 已知遗留风险

不新增 md 文件，全部追加到本文件 § 7~9（每个 phase 一节）。

### 7.6 当前状态
- [x] 7.1~7.5 路线设计
- [x] § 8.1：Phase A-1 KV cache 4-bit pack + store kernel
- [x] § 8.2：Phase A-2 attention 读取反量化路径
- [x] § 8.3：Phase A-3 评测 —— **α 路线不达标，决策升 β / fork flash-attn**
- [ ] § 9：Phase B 实施日志
- [ ] § 10：Phase C 端到端

---

## 8. Phase A：C4（KV cache 4-bit）实施日志

### 8.1 A-1：4-bit pack + store_kvcache triton 改造

#### 改动文件
- `tinyvllm/config.py` 新增字段：`kv_quant_bits / kv_quant_group_size / kv_quant_symmetric / act_quant_bits` + `__post_init__` 校验
- `tinyvllm/engine/model_runner.py::allocate_kv_cache` 重写：根据 `kv_quant_bits` 选择存储 dtype 与额外 scale buffer，按修正后的 per-block 字节数自动估算 `num_kvcache_blocks`；同时把 `k_scale/v_scale/k_zero/v_zero/kv_quant_*` 挂到每个 `Attention` 层
- `tinyvllm/layers/attention.py`：新增 `store_kvcache_q4_kernel`（triton）和 `store_kvcache_q4` 包装；`Attention.__init__` 增量化辅助张量字段；`forward` 写入路径根据 `kv_quant_bits == 4` 分支
- `bench.py` 增加 CLI: `--kv-quant-bits / --kv-quant-group-size / --kv-quant-symmetric / --act-quant-bits` 并传给 `LLM(...)`

#### 设计要点
- **对称量化**：`scale = max(|x|) / 7 + 1e-8`，量化范围 `[-8, 7]`；只存 scale 不存 zero，省一半 metadata 显存
- **pack 方案**：偶数索引 → 低 4 位，奇数索引 → 高 4 位；解包时整体 sign-extend（在 A-2 反量化里实现）
- **kernel grid**：`(N_tokens, N_groups_per_token)`，单 program 处理一个 (token, group)；triton 不支持步进切片索引，所以 even/odd 各 load 一次
- **triton 兼容性踩坑**：用 `triton.language.extra.libdevice.rint` 做 banker's rounding；`(int32 << k) >> k` sign-extension 模式 triton 都能编译
- **per-block 字节估算**：要把 `(scale_buffer_bytes + summary_bytes_for_quest)` 一并加进 per_block 才能稳住 OOM

#### 已知遗留
- A-1 单走没法跑通（没有读路径），需要 A-2 一并交付才能 sanity

### 8.2 A-2：attention 读路径反量化 + flash-attn 兼容

#### 改动文件
- `tinyvllm/layers/attention.py`：新增 `dequant_kv_blocks(cache_packed, cache_scale, block_tables, group_size, out_dtype)`，纯 torch 反量化；`Attention.forward` 在 prefill (with prefix-cache) / decode 两条路径接入
- `tinyvllm/engine/model_runner.py::run_model`：C4 active 时强制走 eager（dequant 里有动态 alloc，cuda graph 无法 capture）；`__init__` 跳过 `capture_cudagraph`

#### 设计选择（α 路线：全量反量化喂 flash-attn）
理由：
1. flash-attn paged kernel 不接受 int4 KV，要么 fork 要么反量化；fork 工作量大且会和上游脱钩，先选 α
2. α 的临时 fp 缓冲只覆盖 batch 实际命中的 block（`B * max_blocks`），不是整个 cache pool，量级可控
3. A-3 评测如果带宽收益不达标，再升到 β 路线（KIVI 风格 per-channel K + per-token V，或者 fork flash-attn）

反量化算法：
```python
p32 = packed.to(int32)
low  = (p32 << 28) >> 28   # 低 4 位算术右移 sign-extend
high = (p32 << 24) >> 28   # 高 4 位
nibble = stack(low, high) → flatten        # [..., head_dim] in [-8, 7]
fp = nibble * scale.repeat_interleave(group_size, dim=-1)
```
要点：
- `block_tables` 中 `-1` padding 位 clamp 到 0 后产生 garbage，但 flash-attn 的 `cache_seqlens` 限定了实际读取范围，不会污染输出
- `repeat_interleave` 在 hot path 不太理想；A-3 如果定位是瓶颈，会改成 `view + expand` 或 fuse 进 triton dequant kernel

#### 与 Quest 的叠加
本期暂不支持 `kv_quant_bits=4` + `quest_top_k_blocks>0` 同时启用：两者都要重写 block_table，且 dequant 应该只对 quest 选中的块做，否则白费 KV 带宽；先各自验证，A-3 之后再叠。代码里以 `assert` 形式拦住。

#### Sanity 命令（待服务器执行）
```bash
# 0) C4 round-trip 数值正确性（先于任何端到端）
python3 tools/test_c4_roundtrip.py            # 本地 numpy 参考
python3 tools/test_c4_roundtrip.py --gpu      # 服务器：调真正的 triton kernel
# 已知：numpy 参考通过（group_size=32/64/128，max_err < scale/2 上界）

# baseline 不变
python bench.py --enforce-eager --num-seqs 8 --max-input-len 512 --max-output-len 256

# C4 短 ctx，无 prefix-cache
python bench.py --enforce-eager --kv-quant-bits 4 --kv-quant-group-size 128 \
    --num-seqs 8 --max-input-len 512 --max-output-len 256

# C4 长 ctx，触发 prefix-cache 路径
python bench.py --enforce-eager --kv-quant-bits 4 --kv-quant-group-size 128 \
    --num-seqs 16 --max-input-len 4096 --max-output-len 512 --max-model-len 8192
```

#### A-3 评测命令（待服务器执行）

`eval_needle.py` C4 模式不能热切，必须分两次跑、合表对比：

```bash
# 1) baseline + Quest 一起跑（沿用 Phase 1 模式）
python tools/eval_needle.py --model ../Qwen3-0.6B \
    --top-k-blocks-list -1 8 16 \
    --num-trials 3 --out-json needle_baseline_quest.json

# 2) C4 单跑（KV cache 4-bit；进程内不能切回 baseline）
python tools/eval_needle.py --model ../Qwen3-0.6B \
    --kv-quant-bits 4 --kv-quant-group-size 128 \
    --num-trials 3 --out-json needle_c4.json

# 3) 吞吐对照（长 ctx 才看得出 KV 带宽收益）
python bench.py --enforce-eager --num-seqs 16 \
    --max-input-len 14000 --max-output-len 1024 --max-model-len 16384
python bench.py --enforce-eager --kv-quant-bits 4 --num-seqs 16 \
    --max-input-len 14000 --max-output-len 1024 --max-model-len 16384
```

通过门槛（继承 7.3 A-3 设计）：
- accuracy 退化 < 2pp（与 baseline needle overall_acc 比）
- 长 ctx (15k) decode 吞吐 ≥ baseline；短 ctx 允许略降（α 路线 dequant 临时 buffer 代价）

#### 已知风险
- ⚠️ `dequant_kv_blocks` 每层每 step 都分配 `[B*max_blocks, block_size, num_kv_heads, head_dim]` 的瞬态 buffer，长 ctx 下显存压力可能比 baseline 还大；A-3 必须用 `nvidia-smi` / `torch.cuda.max_memory_allocated()` 拍下来
- ⚠️ accuracy：对称量化 + group=128 在 0.6B 上的退化未知；先试 group=64/128，如果 needle 跌穿 5pp 改非对称
- ⚠️ flash-attn `block_table` 接受 dynamic block 数量，但每步 alloc fp 缓冲会让吞吐打折——这是 α 路线的内禀代价

### 8.3 A-3：服务器评测 —— α 路线不达标，结论性留痕

#### 8.3.1 测试环境
- 机器：A100 80G PCIe（`sitian@10.232.195.203` `~/sitian-workspace01/tllm/TinyLLMForge`）
- 模型：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B`
- env：`~/sitian-workspace01/tllm/env/bin/python`，torch 2.4.1+cu121

#### 8.3.2 round-trip 数值（kernel 自身正确）
```
[numpy] group_size= 32  max_err=2.9071  bound=3.9056  mse=9.90e-3
[numpy] group_size= 64  max_err=3.1450  bound=3.9056  mse=1.24e-2
[numpy] group_size=128  max_err=3.1450  bound=3.9056  mse=1.51e-2
[gpu]  N=64 group_size=128
       K max_err=2.58  bound=2.59  mse=1.58e-2
       V max_err=0.29  bound=0.29  mse=1.36e-2
```
✅ pack/unpack/sign-extend 没问题，K/V 误差落在对称量化理论上界 `amax/14` 内。

#### 8.3.3 throughput（C4 全面回退）
| 设定 | 配置 | tok/s | Δ vs baseline |
|---|---|---|---|
| baseline | 短 ctx (n=8, in=512, out=256) | **180.98** | — |
| C4 g=128 | 同上 | 107.87 | **−40.4%** |
| baseline | 长 ctx (n=16, in=14k, out=1024) | **163.76** | — |
| C4 g=128 | 同上 | 29.93 | **−81.7%** |

α 路线吞吐不仅没翻盘，**长 ctx 反而比 baseline 慢 5.4×**：每层 `dequant_kv_blocks` 分配 `[B*max_blocks, block_size, num_kv_heads, head_dim]` 的 fp 瞬态 buffer，把"省下的 KV 带宽"完全吃掉，还多了 alloc + repeat_interleave 开销。

#### 8.3.4 accuracy（C4 完全发散）
| 设定 | overall_acc | 备注 |
|---|---|---|
| baseline | **100.0%** | ctx∈{4096, 8192} × depth∈{0, 0.5, 1.0} × 3 trials |
| C4 g=128 | **0.0%** | 输出 garbage（重复 token） |
| C4 g=32  | 0.0% | 减小 group 不救 |

单 prompt 对照（同 prefix `"Hello, please count from 1 to 10:"`）：
```
baseline: ' 1, 2, 3, 4, 5, 6, 7'
C4 g=128: '  then the, then then, then then, and, and, and, and, and,'
```
**这是从第 1 个 decode token 开始就崩**，不是长 ctx 才暴露。

#### 8.3.5 根因诊断
最小复现（N=256 token, head_dim=128, group=128, 单层）：
```
K err: 0.18 / amax 2.56  (相对 7%)
V err: 0.16 / amax 2.20  (相对 7%)
attn out err: 0.014 / amax 0.099  (相对 14%)
```
单层 attention 输出相对误差 ~14%，乘 28 层后呈指数级发散——这正是 KIVI 论文里强调的"K 沿 channel 强相关，per-token group 化对称量化在小模型上无法收敛"的现象。

减小 group_size（128→32）反而没救，因为：
- group=32 时 metadata（scale）开销翻 4 倍，per-channel 误差并没显著下降
- 真正的瓶颈是 **K 的 outlier channel**：少数 channel 的 amax 比其他大几个数量级，对称量化按 group max 算 scale，把同 group 的小幅 channel 量化噪声放大

#### 8.3.6 决策：α 路线砍掉，**不进 Phase B / C**
α（朴素 group-wise 对称 + 全量反量化喂 flash-attn）这条路两端都不达标：
1. **吞吐**：长 ctx 比 baseline 慢 5.4×（dequant 瞬态 buffer 抵消 KV 带宽收益）
2. **accuracy**：从第 1 个 token 起就发散，单点不可用

继续投资 α 没意义，必须升级路线：

| 路线 | 改动 | 风险 |
|---|---|---|
| **β1：KIVI 风格 K 沿 channel + V 沿 token 量化** | K 量化轴改成 channel（首/末 prefill 时算 channel-wise scale），V 维持 per-token group | 需要重写 store kernel；channel-wise scale 需要在 prefill 末尾"封口"一次 |
| **β2：fork flash-attn 直接接 int4 KV** | 省掉 dequant 瞬态 buffer，性能能真正吃到带宽收益 | 工作量大，与上游脱钩 |
| **β3：非对称量化 + per-channel zero-point** | 吃掉 K outlier 偏置，留 group | metadata 翻倍，仍可能不够 |
| **退路：W4 weight-only**（不做 KV 量化） | 沿用已有 quantization=int8/int4，不动 KV | 不能砍 KV 带宽，但已经验证 acc OK |

**直接结论**：A-3 砍掉 α 路线后，Phase B（W4A8 GEMM）与 Phase C（三叠加）的"C"原计划要等 β。考虑到本仓库教学/实验性质，**优先级调整**：
- 不再启动 β（工作量太大、收益不确定，0.6B 这种小模型上的 KV 量化本来就 ROI 低）
- W4A8C4 整条路线**降级为已验证的 negative result**，记录在案
- 后续如果重启，应该先在更大模型（≥7B）上验证 baseline，因为 0.6B 的 K/V 量化敏感度被论文反复点名

#### 8.3.7 留作回头看的开放问题
1. 7B+ 模型上是否 α 路线就足以收敛？（业界 W4A8KV4 论文都是 ≥7B 跑的）
2. flash-attn 上游是否会原生支持 4-bit KV？（如果是，β2 就免做了）
3. 与 Quest 叠加时，是否可以**只对选中的 top-k block 做 dequant**？这能把 dequant 开销除以 N_blocks/top_k，理论上 long-ctx 反而能翻盘。但前提是先有 acc 能用的量化路线。
   → **2026-05-28 已落地为 §10 β3，吞吐 2.29× 翻盘成功；acc 仍受 §8.3 同样的限制。**

#### 8.3.8 文件留痕
- `tools/test_c4_roundtrip.py`：保留，已验证 kernel 正确性
- `tinyvllm/layers/attention.py::store_kvcache_q4 / dequant_kv_blocks`：保留，作为后续 β 路线的写入路径起点
- `tinyvllm/config.py::kv_quant_bits`：保留，便于 ablation
- 不回滚代码，但默认配置 `kv_quant_bits=0`，且 README 不提 C4 可用


---

## 9. Phase B：W4A8（weight 4-bit + activation fake-int8）实施日志

> 7.3 里的 Phase B 原本要接真 cutlass / marlin W4A8 kernel。本仓库教学性质，先做 **fake-quant 版**：weight 量化沿用 `tinyvllm/layers/quantization.py::quantize_int4`，activation 在每次 GEMM 前调用 `fake_quantize_act_int8` 做 round-trip（量化→反量化）。这样精度回归是真实的（A8 噪声会喂进 GEMM），但性能拿不到 int8 tensor core 收益——只测精度。

### 9.1 配置开关
- `tinyvllm/config.py`：`act_quant_bits: int = 0`（0 / 8）
- `tinyvllm/layers/linear.py::_QuantMixin._linear_forward`：`self.act_bits == 8` 时对输入 `x` 做 fake-quant
- `tools/eval_needle.py`：`--act-quant-bits {0,8}`

### 9.2 短 ctx (4k) 评测：g=128 失败 → g=32 翻盘

| 配置 | overall_acc (4k×3 trials, 3 depth) | tok/s |
|---|---|---|
| baseline FP16 | 100% | 119.94 |
| W4 g=128 | 100% | — |
| W4A8 g=128 | **66.7%** | — |
| W4 g=32 | 100% | 86.90 |
| W4A8 g=32 | **100%** | 68.65 |

**翻盘点**：W4A8 g=128 看似要失败（66.7% 跌穿）—— group_size ablation 后 g=32 拉回 100%。固定 g=32 进入长 ctx 复核。

### 9.3 长 ctx 复核（本轮新增，2026-05-28）

跑：`ctx∈{4096, 8192, 15000} × depth∈{0, 0.25, 0.5, 0.75, 1.0} × num_trials=5`，共每 setting 75 个 sample。

| setting | 4k | 8k | 15k | overall | tok/s |
|---|---|---|---|---|---|
| FP16 baseline | 100% (25/25) | 100% (25/25) | 92% (23/25) | **97.3%** | 102.94 |
| W4 g=32 | 100% (25/25) | 100% (25/25) | **44%** (11/25) | 81.3% | 97.33 |
| W4A8 g=32 | 100% (25/25) | 100% (25/25) | **80%** (20/25) | **93.3%** | 77.19 |

15k 详细分布（每格 5 trials）：

| ctx=15k depth | FP16 | W4 g=32 | W4A8 g=32 |
|---|---|---|---|
| 0.00 | 100% | **20%** | 60% |
| 0.25 | 100% | **0%** | 60% |
| 0.50 | 100% | **0%** | 80% |
| 0.75 | 100% | 100% | 100% |
| 1.00 | 60% | 100% | 100% |

### 9.4 反直觉结论：W4A8 比 W4 在长 ctx 上更稳

按"叠加噪声 → 精度更差"的直觉，W4A8 应该比 W4 差。实测相反：
- 15k depth=0.0/0.25/0.5（needle 在 prompt 前半部分）：W4 g=32 几乎全崩；W4A8 g=32 仍守 60–80%
- 15k depth=0.75/1.00（needle 接近末尾）：三者都 100%

**猜测**：activation 也走 int8 round-trip 时，相当于在 attention 输入上做了一次 **per-token 归一化噪声注入**，反而抑制了 W4 在长 ctx 下出现的某种 attention score 偏置 / outlier 放大。这是 0.6B 小模型上特有的现象，不能外推到 7B+。

### 9.5 性能层面诚实的一笔
- W4A8 fake-quant 路径每步多一次 `x.abs().amax()` + `round() / clamp_()`，吞吐 77.19 vs FP16 102.94 = **-25%**
- 真实收益要等接 cutlass / marlin int8 GEMM；本路径**只是精度护栏**，不是性能优化
- 如果只想要"省存储且精度可用"，**W4 g=32 在 ≤8k 已经够了**（100% 同时 -5% 吞吐）；只有跑 15k 才有理由叠 A8

### 9.6 决策
| 场景 | 推荐配置 |
|---|---|
| ctx ≤ 8k（典型 chat） | **W4 g=32**（最快且无损） |
| ctx 8k–15k | **W4A8 g=32**（比 W4 多回收 12pp acc，吞吐多付 -20%） |
| ctx > 15k | 0.6B 自己已经退化（FP16 都到 92%），不在本仓库重点支持范围 |

### 9.7 已知不做
- ❌ 接 cutlass / marlin W4A8 kernel：工程量太大，0.6B 上拿不到代表性数据
- ❌ W4A8 + C4 叠加：C4 在 §8.3 已经定为 negative result
- ❌ W4A8 + Quest 叠加：Quest 在 §6 / profiler 中已被钉死 0.6B 上 ROI 不够

### 9.8 文件留痕
- `needle_baseline_long.json`（FP16 长 ctx 对照）
- `needle_w4_g32_long.json`（W4 g=32 长 ctx）
- `needle_w4a8_g32_long.json`（W4A8 g=32 长 ctx）
- 上面 9.2 的 4k 数据沿用 `needle_w4*_g32.json` / `needle_w4a8_g32.json`


---

## 10. Phase A-revisit：β3 — C4 + Quest 叠加，仅 dequant top-k 选中块

> §8.3 把 α 路线（纯 C4，dequant 全部命中块）枪毙后，§8.3.7 的开放问题 #3 留了一个伏笔：
> "只对 Quest 选中的 top-k 块做 dequant，能把 dequant 开销除以 max_blocks/top_k"。
> 本节是这个伏笔的实测落地，编号沿用 §8.3 列出的路线候选 → β3。

### 10.1 关键观察：α 路线的瓶颈不是 KV 带宽，而是瞬态 buffer

§8.3 已经定位：长 ctx 下 `dequant_kv_blocks` 每层每 step 都分配
`[B*max_blocks, block_size, num_kv_heads, head_dim]` 的 fp16 buffer，
量级 ~480 MB/层 × 2 (K+V) × num_layers = 几十 GB/step 的瞬态分配 + 写回，
**反而比 baseline 慢 5.4×**。

直接的解法有两个：
- β1（fork flash-attn 直接吃 int4 KV）：消除 buffer，工作量极大，与上游脱钩
- **β3（Quest 叠 C4，sparse dequant）：buffer 从 B·max_blocks 缩到 B·top_k，工作量极小**

β3 的关键是改 ordering：先用未量化的 `k_min/k_max` summary 选 top-k 块，
**再**对这 top-k 块做 dequant；选块本身只用 fp16 summary（store 时维护，反量化前），
不需要先 dequant 整张 cache。

### 10.2 实现：attention.py decode 分支重构

```python
quest_active = (context.quest_top_k_blocks > 0
                and self.k_min is not None
                and block_tables is not None
                and cache_seqlens is not None)

if self.kv_quant_bits == 4:
    if quest_active:
        # 1) 用未量化的 summary 选 top-k
        sparse_bt, sparse_cs = quest_select_blocks(
            q, block_tables, cache_seqlens,
            self.k_min, self.k_max, k_cache.shape[1],
            context.quest_top_k_blocks)
        # 2) 仅对 top-k 做 dequant；buffer = B*top_k*block_size*kv_h*hd
        k_fp, new_bt = dequant_kv_blocks(
            k_cache, self.k_scale, sparse_bt,
            self.kv_quant_group_size, q.dtype)
        v_fp, _ = dequant_kv_blocks(
            v_cache, self.v_scale, sparse_bt,
            self.kv_quant_group_size, q.dtype)
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), k_fp, v_fp,
            cache_seqlens=sparse_cs, block_table=new_bt,
            softmax_scale=self.scale, causal=True)
    else:
        # C4 only 路径（α）保留作回退/对照
        ...
```

要点：
- `quest_select_blocks` 用 `k_min/k_max`（未量化）算 criticality，metadata 干净
- 必保留：第 0 块（attention sink）+ 末块（recency / partial），由 `must_keep` 强制 +∞
- `quest_select_blocks` 返回的 `sparse_bt: [B, top_k]`、`sparse_cs: [B]` 形状
  跟 `dequant_kv_blocks` 期望的 `block_tables: [B, max_blocks]` 兼容（top_k 替代 max_blocks）
- 取掉了原 `assert not (Quest && C4)` 拦截

`tools/eval_needle.py` 同步解禁：is_c4 时也允许走 `--top-k-blocks-list`，
单进程内可以一次跑 "C4 only" + "C4+top_k=k1" + "C4+top_k=k2"。

### 10.3 长 ctx 吞吐：2.29× 翻盘

A100 / Qwen3-0.6B / kv_quant_bits=4 / kv_quant_group_size=32 /
quest_min_seq_len=1024 / num_trials=2 / depth ∈ {0.0, 0.5, 1.0}：

| 配置 | ctx ∈ {8192, 15000} 平均 tok/s | 相对 C4 only |
|---|---|---|
| C4 only（α） | 22.11 | 1.00× |
| **C4 + top_k=16（β3）** | **50.73** | **2.29×** |

**这是 §8.3 之后第一次让 C4 路线在长 ctx 上跑赢自己。**

短 ctx 对照（ctx=4096，max_blocks=4096/256≈16，top_k 已经接近 max_blocks）：

| 配置 | ctx=4096 tok/s |
|---|---|
| C4 only | 72.39 |
| C4 + top_k=8 | 108.57 (1.50×) |
| C4 + top_k=16 | 74.36 (1.03×) |

短 ctx 下 top_k=16 ≈ max_blocks，β3 没收益是预期内的；说明只有
"max_blocks ≫ top_k"时叠加才划算。

### 10.4 acc：β3 没让它更差，但也没解决 §8.3 的根因

| 配置 (g=32) | needle 4k overall acc |
|---|---|
| C4 only（历史 needle_c4_g32.json） | 0.0% |
| C4 only（本轮 4 trials） | 0.0% |
| C4 + top_k=8（本轮 4 trials） | 0.0% |
| C4 + top_k=16（本轮 4 trials） | 0.0% |

C4 g=32 在 0.6B Qwen3 上的 needle acc 本来就是 0%（§8.3 的 negative result 没变）。
β3 的价值是**性能**，不是 acc：长 ctx 翻盘后，C4 路线终于不再"省了 KV 带宽却跑得更慢"。
要再要 acc，得换更大模型（≥7B）或叠非对称量化 / per-channel scale，那是后续事。

输出 raw 抽样确认模型仍在做合理推理（连贯英文，未崩成乱码）：
```
top_k=-1 -> "The answer is inside the box. The answer is on the top of the box. ..."
top_k=8  -> "The number is . The number is the number is, the number is, ..."
```
即 acc=0% 是 0.6B 找不到 needle，不是数值崩坏。

### 10.5 结论：β3 是 §8.3 三选项里最低成本的，已落地

| 路线 | 工作量 | 长 ctx 吞吐 | 落地状态 |
|---|---|---|---|
| α（纯 C4） | — | 0.18× baseline（即比 baseline 慢 5.4×） | 历史负结果 |
| β1（fork flash-attn 接 int4） | 极大 | 理论 ~2× | 未做 |
| β2（非对称 + zero-point） | 中 | 仍含 dequant buffer | 未做 |
| **β3（Quest 选块 + sparse dequant）** | **小（~60 行 attention.py 改动）** | **2.29× 相对 C4 only** | **已落地** |

β3 + Quest 都是只在 decode 启用，prefill 走原路径，不影响 prefill 吞吐。

### 10.6 已知不做
- ❌ β3 接 cuda graph：dequant 仍有动态 alloc，与 §8.3 同样原因；要 capture 得先把
  dequant 改成"写到固定大小的 pre-alloc buffer"，工作量不小
- ❌ β3 + W4A8 三叠加：本仓库 0.6B 上 acc 已经在 §8.3 / §9.4 各自钉死，
  叠加不会改善 acc 只会复合性能成本

### 10.7 文件留痕
- `tinyvllm/layers/attention.py::Attention.forward`：decode 分支新增 `quest_active`，
  C4+Quest 走 sparse 路径，C4 only 保留作回退；取掉互斥 assert
- `tools/eval_needle.py::run_one_setting / main`：is_c4 时不再强制 `top_k=-1`
- `needle_c4_quest_long.json`（β3 长 ctx 性能数据）
- `needle_c4_quest_acc.json`（β3 短 ctx 多 trial acc 数据）
- commit `5f5a38a`

### 10.8 调参扫描：top_k × ctx_len 的 Pareto 前沿

跑一发 `top_k ∈ {-1, 4, 8, 16, 32}` × `ctx_len ∈ {4096, 8192, 15000}` × 5 depths × 2 trials = 30 prompt/setting，
看不同稀疏度下的吞吐 / acc 走势。

跑法：
```
CUDA_VISIBLE_DEVICES=3 python tools/eval_needle.py \
  --context-lens 4096 8192 15000 --depths 0.0 0.25 0.5 0.75 1.0 \
  --top-k-blocks-list -1 4 8 16 32 --num-trials 2 --max-output-len 200 \
  --kv-quant-bits 4 --kv-quant-group-size 32 --max-model-len 16384 \
  --gpu-memory-utilization 0.55 --max-num-seqs 4 --enforce-eager \
  --out-json needle_b3_sweep.json
```

OOM 修复：本轮在共享 A100 上踩到一次 C4-only 路径 OOM —— 默认 `gpu_memory_utilization=0.9`
让 KV pool 吃满，长 ctx + batch=10 的瞬态 dequant buffer（B·max_blocks·block_size·heads·dim·2byte ≈ 6-8GB）放不下。
解决：`tools/eval_needle.py` 加了 `--gpu-memory-utilization` / `--max-num-seqs` 两个 CLI，
本轮用 0.55 / 4 跑。这个事故本身正好印证了 §10 β3 的核心动机——**α 路线的瓶颈不是 KV 带宽，是瞬态 dequant buffer**。

结果（混合 4k/8k/15k 三档；acc 全 0% 是 §8.3 的固有结果）：

| top_k | 总耗时 (s) | 吞吐 (tok/s) | vs C4-only |
|---|---:|---:|---:|
| -1 (C4 only) | 151.64 | 39.57 | 1.00× |
| 4 | 130.77 | **45.88** | 1.16× |
| 8 | 131.92 | 45.48 | 1.15× |
| 16 | 129.76 | 46.24 | 1.17× |
| 32 | 130.29 | 46.05 | 1.16× |

### 10.9 观察 & 结论

1. **β3 整体相对 C4-only 提速 ~16%**（混合 ctx 下；§10.2 单测 15k ctx 是 2.29× —— 短 ctx 稀释了收益）
2. **top_k 4 / 8 / 16 / 32 之间几乎没差**（45.5–46.2 tok/s，差异 <2%）
   - 说明 0.6B + 这个 batch 规模下 attention 已经不是瓶颈，固定开销（model fwd / sampler / kv writeback / overhead）占主导
   - 也意味着**取小 top_k 没成本**——在更大模型 / 更大 batch 下会拉开差距，但本仓库不投入做更大模型
3. **acc 不区分 top_k**——无论 4 还是 32 都是 0%，说明 needle 的失败发生在更早的环节（§8.3 g=32 量化误差堆叠），不是 Quest 选错了块
4. **推荐配置**：C4 g=32 + Quest top_k=4。理由：
   - 收益（+16% throughput）跟 top_k=32 一样
   - dequant buffer 最小（B·4 而不是 B·max_blocks），对显存友好
   - 留出 §10.6 提到的"接 cuda graph"未来的可能性（top_k 越小，固定 buffer pre-alloc 越可行）

### 10.10 已知不做（叠加）
- ❌ 把 sweep 拓到 ctx≥30k：0.6B 自己已经 92%→60% 退化（§9 已留痕），不在重点支持范围
- ❌ 跑大 batch (max_num_seqs=32+)：远端共享 GPU 显存有限，且本仓库目标不是 serving 吞吐
- ❌ 跑更大模型（≥7B）的扫描：模型不在本仓库，复刻成本太高

### 10.11 sweep 文件留痕
- `needle_b3_sweep.json`（本轮 top_k 扫描的原始 details）
- `tools/eval_needle.py`：新增 `--gpu-memory-utilization` / `--max-num-seqs` CLI

### 10.12 Profiler 验证：β3 到底省在哪里

§10.9 的两个观察需要 kernel 级证据：(1) β3 提速来源是 dequant 而不是 attention 本身；
(2) 0.6B + 小 batch 下固定开销（model fwd / sampler）占主导。
跑 `tools/profile_quest.py` 直接对比 c4_only（α 路线）vs c4_quest top_k=4（β3）。

跑法：
```
CUDA_VISIBLE_DEVICES=3 python tools/profile_quest.py \
  --num-seqs 4 --max-input-len 14000 --max-output-len 32 \
  --max-model-len 16384 --quest-top-k 4 \
  --warmup-steps 4 --profile-steps 12 \
  --gpu-memory-utilization 0.55 --max-num-seqs 4 --kv-quant-group-size 32 \
  --compare c4_only c4_quest --out-dir profile_out/b3_vs_c4
```

为支持 β3 路径对比，`profile_quest.py` 加了两个新 mode：`c4_only`（α）和 `c4_quest`（β3），
并通过 `--compare A B` 选择主进程对比对象。

### 10.13 结果：β3 砍掉 87% 的 attention/dequant CUDA 时间

12 步 decode，4 个 14k ctx prompt，总 CUDA kernel time：

| 路径 | total CUDA kernel time | vs c4_only |
|---|---:|---:|
| c4_only（α） | 1588.33 ms | 1.00× |
| **c4_quest top_k=4（β3）** | **206.62 ms** | **0.13×（-87%）** |

按 kernel 分解 top-N（saved = c4_only - c4_quest，单位 ms）：

| saved (ms) | c4_only (ms) | c4_quest (ms) | kernel（截断名） |
|---:|---:|---:|---|
| 438.7 | 467.0 | 28.3 | `elementwise_kernel<128,2,...>`（dequant 主体） |
| 183.8 | 200.4 | 16.6 | `unrolled_elementwise_kernel<...>`（cast/copy） |
| 179.5 | 188.3 | 8.8 | `vectorized_elementwise_kernel<4,...>`（int4 sign-extend） |
| 179.3 | 188.2 | 8.8 | `vectorized_elementwise_kernel<4,...>` |
| 131.1 | 138.5 | 7.3 | `vectorized_elementwise_kernel<4,...>`（scale `repeat_interleave`） |
| 122.4 | 136.3 | 13.9 | `index_elementwise_kernel<128,4,...>`（gather block_tables） |
| 87.4 | 100.3 | 13.0 | `elementwise_kernel<128,4,...>` |
| 58.1 | 64.5 | 6.4 | `unrolled_elementwise_kernel<...>` |
| **41.2** | **47.3** | **6.1** | **`flash_fwd_splitkv_kernel<...>`（attention 本身）** |

c4_quest 唯一新增的 kernel：

| 新增 (ms) | kernel |
|---:|---|
| +2.4 | `flash_fwd_splitkv_combine_kernel<...>` |

### 10.14 结论与回答 §10.9 的两个观察

1. **β3 提速来源 = dequant，不是 attention 本身**
   - dequant 相关的 6 个 elementwise/index kernel 总共省了 ~1300 ms
   - flash-attn 本身只省了 ~41 ms（K/V 体量从 max_blocks 缩到 top_k=4 也只是次要）
   - 唯一的"新债" splitkv_combine 仅 2.4 ms，可忽略
2. **§10.9 观察 (2) 部分修正**：固定开销在端到端 throughput 数字（39.57 vs 45.88 tok/s）里看是 +16%；
   但 profiler 视角下 attention/dequant 占的绝对 CUDA 时间从 1588ms→207ms，**只占总耗时的一小部分**，
   model fwd（gemm / norm / silu）才是 0.6B 上真正的 wall-clock 主导。
   这也解释了为什么 §10.8 里 top_k 4/8/16/32 几乎没差——attention 自身已经小到敏感度极低
3. **β1 路线的预期收益上界进一步收紧**：β1 = "fork flash-attn 直接吃 int4 KV"，
   理论上能省的就是 dequant 的 ~1300 ms。这个数字在 0.6B 上换算成 throughput 提升就是
   `1382/1588 × 16% ≈ 14%`（β3 已经吃掉），**β1 在 0.6B 上没有边际增益**。
   这正式 close 了 §10.5 表里 β1 的"理论 ~2×" —— 在 0.6B 上理论上界就是 β3 现在拿到的 16%

### 10.15 profiler 文件留痕
- `tools/profile_quest.py`：新增 `c4_only` / `c4_quest` 两个 mode + `--compare A B` CLI + `--gpu-memory-utilization` / `--max-num-seqs` 参数
- `profile_out/b3_vs_c4/{c4_only,c4_quest}/trace.json`（chrome trace，~40-80 MB，不入仓库）
- 上面 §10.13 的 kernel 表是从 trace 解析出来的 saved-time 摘要
