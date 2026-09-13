# Latent KV capacity, Stage 1a GATE A: the Stage 0 decode model is falsified

Run: `experiments/kvcapacity_step_scaling/step-scaling-measure-20260913-191122`
Model: Qwen3-8B on one A100 80GB, `tinyvllm` serving path.
Grid (fully covered, 18/18 cells measured on both paths):

```
8192  : B=1,2,4,8,16,32
16384 : B=1,2,4,8,16
32768 : B=1,2,4,8
40448 : B=1,2,4
```

## Verdict

**FAIL on the eager path, FAIL on the graph path.** The failure is decisive, not a harness
artifact: coverage 18/18, window stability 18/18 within 5%, sample dispersion 18/18 under
25% of the median.

## What the Stage 0 model assumed

```
step_ms(L,B) = c0 + c1 * L * B      c0 = 13.05 ms   c1 = 0.151 us/token
```

## What the serving path actually does (eager, fit on all batches)

```
M1  step_ms = 39.789 + 0.0837e-3 * L*B      R^2 = 0.9543   [FAIL, threshold 0.98]
M2  + 0.1756 ms/seq pure batch term         R^2 = 0.9694   residual -33.0%
M3  + 1.363e-10 * (L*B)^2 curvature         R^2 = 0.9671   quadratic = 15.0% of the step
                                                            at L*B=262144  [FAIL, tol 10%]
```

Equal-product consistency, which M1 requires: 4/5 groups agree within 10%; `L*B=32768`
spreads 12.7% (45.500 / 43.554 / 40.029 ms), i.e. the same resident-token count costs
measurably different amounts depending on how it is split between L and B.

## Three separate ways Stage 0 was wrong

1. **The constant is ~3x off.** Measured `c0 = 39.789 ms` against `13.05 ms` assumed
   (ratio 3.049). The graph path shows why: batch 1 runs a CUDA graph fast path at
   15-18 ms, batch 2 jumps to 43-50 ms (ratio 2.58-2.68 at L=32768/40448). Stage 0's
   `c0` was fit on batch-1 data and therefore describes an execution path that the
   capacity argument itself never uses, since capacity gains only matter at B >= 2.
2. **The slope is ~0.55x off.** Measured `c1 = 0.0837 us/token` against `0.151`
   (ratio 0.554), eager; `0.1054` (ratio 0.698) on the graph path fit at B >= 2.
   Resident KV is *cheaper* per token than Stage 0 assumed.
3. **The functional form is wrong, not just the constants.** There is real curvature
   (15% of the step at the largest cell, eager; 19.8% graph) and a real pure-batch term
   (8.7% eager, 22.1% graph). An affine function of `L*B` alone cannot absorb either.

## Consequence for the latent KV capacity line

- The Stage 0 break-even artifact (`experiments/kvcapacity_stage0/2026-09-13/gate.json`)
  is computed on falsified constants and must not be cited.
- **Do not proceed to GATE B (`phi_probe` / head slicing) on the old numbers.**
- Two directions are honest from here:
  - Refit the cost model as M2 + curvature on the measured multi-batch regime, then re-run
    the Stage 0 break-even gate and see whether the capacity argument survives the larger
    constant. Note the direction of the two dominant errors: a 3x larger `c0` shrinks the
    fraction of the step that KV residency explains, which makes KV compression *less*
    valuable per unit of compression, and the smaller `c1` cuts the same way.
  - Or reframe: at these context lengths the step is dominated by a ~40 ms
    context-independent constant, and resident KV explains roughly 20 ms of a 60 ms step
    only at `L*B = 262144`. Compressing KV cannot beat that constant.

The second reading is the more likely one and should be checked first, because it decides
whether GATE B is worth building at all.

---

# 附录：高并发 sweep（2026-09-13 20:20）——容量轴没有死

GATE A 只杀掉了延迟轴。容量轴的唯一真闸门是：**并发继续加上去，吞吐还涨不涨？**
如果吞吐在撞到 KV 墙之前就饱和，那么"压缩 KV 以装下更多序列"这件事本身无收益，
GATE B 不值得建。

Run: `experiments/kvcapacity_step_scaling/step-scaling-sweep-20260913-202001`
Grid（固定 L，把 B 推到 KV 预算边缘）：`2048: 1..128`、`8192: 1..40`，eager 与 graph 双路径。
工具：`tools/kvcapacity_batch_sweep_analysis.py`（不是 GATE A 判定器；sweep 不是预注册网格，
不能给出 GATE A 的 PASS/FAIL）。

## 读数：CAPACITY OPEN（两条路径一致）

| L | 最大 B | 该点单步 | 吞吐 | 全程扩展效率 | 每序列成本 a |
|---|---|---|---|---|---|
| 2048 | 128 | 67.07 ms | 1908 seq/s | 0.58 of proportional | 0.215 ms/seq (eager) |
| 8192 | 40 | 71.69 ms | 558 seq/s | 0.55 of proportional | 0.774 ms/seq (eager) |

吞吐在最大并发处**仍在上升**，峰值就落在最大 B 上，没有出现拐头。

## 两个污染源必须先剔掉，否则读数是假的

第一次 sweep 的 graph 路径读出 CAPACITY WEAK，纯属假象：

- **B=1 是 CUDA graph 快路径**（2048 下 12.980 ms 对 B=2 的 43.142 ms）。跨这条边界算边际吞吐
  会得到负值，把后面每一步都染成"崩塌"。分析工具现在沿用 GATE A 的 1.30 阈值检出该边界，
  从 B=2 起读，B=1 单独报告。剔除后 graph 路径同样是 CAPACITY OPEN，扩展效率 0.65。
- **L=2048 B=96 是污染格**（stdev 111.623 ms 对 median 58.402 ms，离散度 191%）。已按
  GATE A 的 25% 阈值排除在拟合之外。

两者都写成回归测试（`tools/test_kvcapacity_batch_sweep_contaminants.py`），用的是本次跑出的真实数值。

## 每序列成本的结构，与 GATE A 自洽

`a(L) ≈ c1 * L + a_pure`：

```
L=2048: 0.215 ms/seq   ≈ 0.0837us * 2048  (0.171) + ~0.04
L=8192: 0.774 ms/seq   ≈ 0.0837us * 8192  (0.686) + ~0.09
```

即高并发下的主导项仍是 KV 常驻，而不是与 KV 无关的固定 per-seq 开销——这对容量论证是利好。
曲率没有定论：eager 在 L=8192 B=40 处 B² 项占 13.8%，graph 在同点只占 0.8%，两条路径不一致，
这个量级还不能当成事实。

## 对 GATE B 的意义：天花板约 3.4×，而且是实测而非外推

L=8192 当前的墙在 B=40（KV 预算 ~52 GiB）。**L=2048 的那条曲线在物理上就是"8192 被压 4× 后"
的类比**：同样的 KV 字节数，序列数 4 倍。它实测给出 B=128 时 1908 seq/s、单步 67 ms，
对比 8192 现在的 558 seq/s，即 **约 3.4× 吞吐**，代价是单步从 71.7 ms 降到 67.1 ms（不升反降）。

必须承认这个类比高估：MLA 式压缩省的是 KV 字节，但会加回上投影计算，且逻辑上下文仍是 8192，
注意力的计算量不会跟着字节一起降。所以 3.4× 是上界，不是预期值。

## 结论与下一步

- 延迟轴：死。最好情况 -26%（最大负载）/ -9%（常见负载），不值得。
- 容量轴：**活**。吞吐在 KV 墙处仍在涨，主导项是 KV 常驻，压缩换并发有约 3.4× 的上界。
- 因此 GATE B（`phi_probe` / head slicing）值得建，但必须换目标函数：
  **考核吞吐（seq/s，固定 KV 字节预算），不再考核单步延迟**。Stage 0 那套按单步延迟算 break-even
  的框架应当废弃重写，`c0=13.05ms / c1=0.151us` 两个常数一并作废。
- 建 GATE B 前还欠一件事：把 sweep 推到真正的 KV 墙（2048 的墙约在 B≈176，本次网格只到 128
  就停了，是设计停的不是被拒绝停的），确认吞吐在墙上仍未拐头。

---

# 附录二：推到真正的 KV 墙（2026-09-13 22:52）——容量收益有上界，约 3.5×

Run: `experiments/kvcapacity_step_scaling/step-scaling-sweep-wall-20260913-225202`
Grid：`2048: 32,64,96,128,144,160,176,192`、`8192: 16,32,40,44,48`，eager + graph 双路径。

## 墙在哪里，是引擎自己说的

| L | 最大能同时跑的 B | 该点常驻 token | KV 字节 | 更大的 B 发生了什么 |
|---|---|---|---|---|
| 2048 | **144** | 294912 | 43.5 GiB | B=160 被引擎拆成 148 + 12 两批，目标并发一步都没跑成 |
| 8192 | **40** | 327680 | 45.0 GiB | B=44 / 48 同样跑不成 |

`max_num_seqs` 由 worker 设为 `maxB+4`，所以这不是配置卡的，是 KV 预算卡的。墙在约
295k–328k 常驻 token、43–45 GiB。

## 吞吐在墙前进入平台期，约 2000 seq/s

| L=2048 | B=32 | 64 | 96 | 128 | 144 |
|---|---|---|---|---|---|
| eager 单步 | 48.65 | 51.30 | 58.33 | 63.51 | 73.86 ms |
| eager 吞吐 | 658 | 1247 | 1646 | **2015** | 1950 seq/s |
| graph 吞吐 | 690 | 1253 | 1581 | 1883 | **1987** seq/s |

两条路径对"144 到底是峰还是回落"意见相反（eager 认为 128 是峰、144 掉 3.3%；graph 认为 144
仍在涨），差异在 3% 量级。**诚实的读法是：吞吐在墙前进入约 1950–2015 seq/s 的平台期，
而不是"仍在线性上涨"。** eager 路径读数因此从上一轮的 CAPACITY OPEN 收紧为
**CAPACITY BOUNDED**；L=8192 在其墙（B=40，562 seq/s）处仍在上升，未见平台。

L=2048 eager 在 B=144 处 B² 项占 48.8%，但 graph 在同点只占 14.4%，且这一项几乎全部由最后
那一个点撑起来——不能当成"确认存在二次项"，只能说墙附近的形状不再是线性。

## 收益上界：约 3.5×，现在是两端都实测的

L=8192 的墙给 562 seq/s（eager）/ 553 seq/s（graph）。同样 KV 字节预算下把序列数放大 4 倍的
物理类比（L=2048）给出的平台是 1950–2015 seq/s。

```
2015 / 562 = 3.59        1987 / 553 = 3.59
```

**约 3.5–3.6×，且单步不升反降（71.2 ms → 63.5 ms）。** 这个数字上一轮是"实测一端 + 外推一端"，
现在两端都落在实测点上。仍需保留的高估来源不变：MLA 式压缩省字节但加回上投影计算，
逻辑上下文仍是 8192，注意力计算量不随字节下降——所以 3.5× 是上界。

## 顺手修掉的两个工具缺陷

- **读数会掩盖内部峰值**：原逻辑只要有任一 context 仍在上升就报 CAPACITY OPEN，会把 L=2048
  的平台/回落藏掉。新增 `CAPACITY BOUNDED` 类别，并把 `CAPACITY DEAD` 的定义收紧为"最佳吞吐
  相比最小 batch 没有超过 10% 的增益"——先涨后落属于 bounded，不属于 dead，两者导向不同决策。
- **artifact 的引擎身份字段全是 null**：`LLM` 在这个构建里不直接暴露 `config`，原查找在第一个
  AttributeError 就放弃了，于是 `max_num_seqs` / `num_kvcache_blocks` / KV 容量全部记成 null。
  一份看起来完整、实际什么都没记的 provenance 比没有更糟。已改为穿过 wrapper 逐层解析并加了
  测试；**本次 wall run 的 artifact 仍缺这些字段**，下一轮才会带上。

## 容量结论定稿

- 延迟轴：死。
- 容量轴：**活，但有界**。墙在 43–45 GiB / ~300k 常驻 token；吞吐在墙前进入 ~2000 seq/s 平台；
  用 KV 字节换并发的收益上界约 **3.5×**。
- GATE B 可以建，目标函数必须是**固定 KV 字节预算下的吞吐（seq/s）**，并且要对着 3.5× 这个上界
  记分：任何压缩方案若在精度可接受的前提下拿不到该上界的显著一部分，就不值得继续。
- Stage 0 那套按单步延迟算 break-even 的框架整体作废（`c0=13.05ms` / `c1=0.151us` 两个常数一并作废）。
