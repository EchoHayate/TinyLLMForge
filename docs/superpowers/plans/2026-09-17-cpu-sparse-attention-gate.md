# CPU 侧稀疏 attention 门：算得动，但前提是别写出一个笨 kernel

日期：2026-09-17
分支：`feat/kv-sparse-attention`
产物：`tools/cpu_sparse_attention_bench.c`、`tools/run_cpu_sparse_attention_remote.sh`、
`experiments/cpu_sparse_attention/cpu-attn-20260917-005409/`

## 这道门要回答什么

gather 带宽门（2026-09-16）只量了「这台 CPU 能多快**搬**散列的 KV」。它没碰算术、没碰 softmax，
也没碰一个真实 offload kernel 必须做的事：读 K 算分、每 token 每 kv head 存 4 个分数、
再读一遍 V 做加权和。估算通常就死在这些「多出来的几趟」上。

所以这道门量的是**整步 attention**，而不是它前面的 memcpy：

- 形状：一条 decode 序列，KV 常驻 host DRAM，`K[layer][token][kv_head][dim]`，bf16（引擎存的就是 bf16）
- Qwen3-8B 形状：36 层、8 kv head、32 q head（group=4）、dim=128
- 每步每层按 32-token 单元选 tokens（query-aware 选择本身的开销不在这里量）
- 不用 staging buffer：真 kernel 会在 cache line 还热的时候直接转换并乘，
  单独走一趟 gather 会把 KV 读两遍、把成本算高
- 判据：`step_ms` 对比 GPU 读权重窗口 `c0 = 12.113 ms`。装不进这个窗口，offload 就藏不住

自测：AVX-512 路径对 fp64 参考实现比对，`max_abs_err = 4.9e-08`（服务器上），
且 `dim` 必须是 16 的倍数——self test 特意用 dim=32，否则测的是服务器上根本不会跑的标量分支。

## 先说一个方法论级别的坑：第一版 kernel 让 CPU 显得很废

第一版把 q-head 索引写在 channel 循环里面、把 bf16 转换交给编译器，结果：

```
896 tokens (10.9% of 8k), 32 threads:  step_ms = 16.98   eff_bandwidth = 7.8 GB/s
```

这个数会直接得出「trajectory 级预算装不进 12.1 ms 窗口，CPU offload 死了」的结论。
把内层换成「一次转换 16 个 bf16 lane、每个 q head 一个累加寄存器、K/V 各只流一遍」之后：

```
896 tokens (10.9% of 8k), 32 threads:  step_ms =  2.60   eff_bandwidth = 50.9 GB/s
```

**6.5×**。同一台机器、同一个访存模式、同一个时刻。
如果不做这一步，就会用一个 strawman kernel 去否掉一条本来可行的路。
以后凡是「CPU 算不动」的结论，必须先给出 roofline 距离，否则不算证据。

## 结果

`numactl --interleave=all`、`OMP_PROC_BIND=spread`、20 次迭代取均值，单位 ms：

| 每步 token 数 | 占 8k 上下文 | 16 线程 | 32 线程 | 64 线程 | 128 线程 |
|---|---|---|---|---|---|
| 256 | 3.1%（答案级保真） | 1.19 | 0.61 | **0.42** | 12.21 |
| 448 | 5.5% | 2.07 | 1.16 | **0.80** | 12.81 |
| 896 | 10.9%（轨迹级保真） | 4.80 | 2.60 | **2.05** | 13.59 |
| 2048 | 25% | 12.45 | 7.12 | **4.74** | 17.11 |
| 8192 | 100%（dense，参照） | 56.01 | 31.07 | **20.10** | 31.52 |

### 1. 两级保真预算都装得进窗口，而且余量很大

- 答案级（256 token/step）：64 线程 **0.42 ms**，占 12.1 ms 窗口的 **3.5%**
- 轨迹级（896 token/step）：64 线程 **2.05 ms**，占窗口的 **17%**
- 即使只给 16 线程，两者也分别只要 1.19 / 4.80 ms，仍然装得进

### 2. dense CPU attention 装不进窗口——稀疏正是可行性的来源

100% 上下文要 20.1 ms（最好情况），比窗口超 **1.66×**。
所以这条路不是「CPU 顺手就能算 attention」，而是
**query-aware 选择把一个装不进窗口的负载变成了占窗口 3.5%–17% 的负载**。
这是目前为止对「为什么 offload 必须配 query-aware」最直接的一条实测证据。

### 3. 用满 SMT 会直接崩掉，别把线程数当越多越好

128 线程（= 64 物理核的超线程满配，且宿主 load average 110）在**所有**预算上都退化到 12–17 ms，
方差极大（256 token 那格：best 6.44 / worst 17.92）。64 线程是拐点。
注意这和 gather 门的结论**不同**：那边 128 线程还在涨（纯访存并行度受益），
这里因为要真算，抢的是核而不是内存并行度。**不能把 gather 门的线程结论直接搬过来。**

### 4. 不要用 gather 峰值带宽给 offload 做预算

| 口径 | 有效带宽 |
|---|---|
| 纯 scattered gather（128 KiB 粒度，RAND） | ~110 GB/s |
| 本 attention kernel 峰值（64 线程） | ~64 GB/s |

多出来的那几趟（存分数、softmax、输出累加）吃掉约 40%。
之前计划里「12.1 ms 窗口内约 900 token/step」是从 gather 峰值推的估计，
现在可以直接从实测反推 ceiling：dense 8192 token 要 20.1 ms，
所以窗口内的**实测**上限约 `8192 × 12.1 / 20.1 ≈ 4900 token/step`。

于是可行域重新对账（8k 上下文，每 token 全层 KV = 144 KiB，与 `KV_BYTES_PER_TOKEN=147456` 一致）：

| 保真度 | 检索需要 | 实测 CPU 上限 | 余量 |
|---|---|---|---|
| 答案级 | 256 token/step | ~4900 | **19×** |
| 轨迹级 | 896 token/step | ~4900 | **5.5×** |

上一轮报告写的「轨迹级贴边、没有余量」是基于那个 ~900 的估计，**现在被实测推翻**：
真正的贴边点在 25% 预算附近（2048 token → 4.74 ms，还有 2.6× 余量）而不是 11%。
两个方向的错误都出现在同一个地方——**用没量过的常数替代测量**。

## 这一轮改变了什么决定

1. **CPU 侧算力不再是主要风险**：8k 上下文、36 层、bf16、64 物理核，轨迹级预算只占窗口 17%。
   风险转移到 overlap 与同步（下一道门），以及长上下文（32k/128k）下的线性放大。
2. **线程配置写死 64（物理核），禁止 SMT 满配**，并且要 `interleave=all`。
3. **budget 计算必须用 attention kernel 的实测带宽（~64 GB/s），不能用 gather 峰值（110 GB/s）。**
4. **kernel 质量本身是实验变量**：报任何「CPU 不行」的结论前，先报 roofline 距离。

## 还没算进任何预算的东西（重要）

- **选择器自己的开销**：min/max summary 维护 + top-k。32 粒度意味着 8× 单元数（8k → 256 单元/层），
  这笔钱在 GPU 侧还是 CPU 侧、算多少，都还没量。
- **PCIe 往返**：Q 下行（36 层 × 32 head × 128 × 2B = 288 KiB/step）、attention 输出上行（同量级），
  以及每层一次同步的延迟。延迟而非带宽才是这里的风险。
- **真 overlap**：本次是 CPU 独占测量，GPU 空转。真实场景下 CUDA launch 线程要占核、
  H2D/D2H 要占 DRAM 带宽，`c0` 窗口未必能真的被填满。
- **长上下文线性放大**：32k 上下文、轨迹级 11% ≈ 3600 token/step → 按当前 64 线程斜率约 8.2 ms，
  仍在窗口内但余量只剩 1.5×；128k 会直接爆窗口，必须靠「选中 token 数不随 L 线性增长」的证据支撑。
- 宿主污染：load average 110–114、29–30 个用户在线。**所有数字都是下界**（真机独占只会更快），
  但 128 线程那一列的崩塌里有多少是宿主抢核、多少是 SMT 本身，本轮分不开。

## 复现

```bash
RUN_TAG=cpu-attn-$(date +%Y%m%d-%H%M%S) \
TOKEN_BUDGETS="272 448 912 2048 8192" THREAD_COUNTS="16 32 64 128" ITERS=20 \
bash tools/run_cpu_sparse_attention_remote.sh
```
