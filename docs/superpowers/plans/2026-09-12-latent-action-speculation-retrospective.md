# 复盘：latent action speculation 从提出到关闭

状态：线已关闭。
时间跨度：2026-09-10 至 2026-09-12。
GPU 消耗：单卡 A100 80GB PCIe，累计约数小时；**训练 0 次**。
最终结论：**假设被证伪，且证伪它的不是任何一个 drafter，而是"actor 一轮到底要多少 GPU 时间"这个分母的测法。**

本文是这条线的收口文档，汇总六份阶段文档的结论、每一次转向的原因、以及可复用的方法论教训。阶段文档（英文）：

```text
docs/superpowers/plans/2026-09-10-latent-action-speculation-stage1a.md
docs/superpowers/plans/2026-09-10-latent-action-speculation-stage1a-bis.md
docs/superpowers/plans/2026-09-11-latent-action-speculation-stage1b-step0.md
docs/superpowers/plans/2026-09-11-latent-action-speculation-stage1b-step0b.md
docs/superpowers/plans/2026-09-11-latent-action-speculation-stage1b-step1.md
docs/superpowers/plans/2026-09-11-latent-action-speculation-stage1b-erratum.md
docs/superpowers/plans/2026-09-12-latent-action-speculation-stage1b-step2.md
```

---

## 0. 最初的假设

出发点是一个直觉：LLM/Agent 之间传递上下文目前全靠文字 token，而文字是有损且昂贵的中间表示；如果改用 hidden states / 连续向量 / 离散 latent code，也许能在 Agent 推理里省下大量 token 和时间。

在 Agent 场景，这个直觉被具体化成 **action-level speculation**：

```text
Agent 的瓶颈从 token decode 转移到  think -> act -> observe  的串行 loop。
如果一个廉价 drafter 能提前猜出下一个 action，就可以在 actor 还在想的时候
把工具先跑起来；猜对则提交（commit-on-match），猜错则回滚。
```

关键设计选择（这一条从头到尾没有动摇，也是唯一没被推翻的部分）：

- latent 只允许出现在 **draft 侧**。
- 提交契约仍然是 **文本 action 的规范化 digest 逐字节相等**。
- 保证的是 **trajectory equivalence**（轨迹等价），不是 token 分布无损。
- 有副作用的 action **fail-closed**，不投机。

这样 latent 的不可解释性只影响"猜得准不准"，不影响正确性。

## 1. Stage 0：先写成本模型，不写 kernel

第一个决定是**先不实现，先算账**。共享 serving 下 drafter 占用的 GPU 不是免费的，它同时抬高利用率、压缩排队余量，猜错还要付回滚。

```text
rho_spec = rho * (1 + tau)
W_spec   = D * (1 + tau) / (1 - rho_spec)
T_hit    = max(W_spec, alpha * W_spec + T)
T_miss   = W_spec + T + R
T_spec   = p * T_hit + (1 - p) * T_miss

稳定性边界： rho * (1 + tau) < 1
```

其中 `D` = actor 产出一个 action 的 GPU 需求，`tau` = drafter 税（相对 D），
`p` = 命中率，`T` = 工具耗时，`R` = 回滚代价，`rho` = 基线利用率。

产物是 `tinyvllm/agentspec/`：与 `tinyvllm/speculative/` 完全隔离，不含 torch/transformers/numpy 依赖，48 个测试。它给出的是**最小命中率** `min_p`：低于它，无论 drafter 多便宜都亏。

> 这个决定在事后看是整条线最正确的一步。它让后面三个提案都在纸上被杀掉，没有一次进入训练。

## 2. Stage 1a：第一次测 drafter 税，harness 说了第一次谎

用 HF eager 逐 token 循环测 Qwen3-8B actor / Qwen3-0.6B drafter：

```text
context   actor_s   tau_text   tau_code   tau_code_ckv
   1024    1.1962     0.7275     0.0239         0.0231
   4096    1.4772     0.6137     0.0509         0.0176
  16384    3.4196     0.3554     0.1344         0.0085
```

看起来结论很漂亮：文本 drafter 太贵，code head + 压缩上下文几乎免费。

但 0.6B 与 8B 的 decode step 比值落在 0.60~0.78 —— 一个 13 倍参数差的模型对，单步只快 30%，说明测的不是模型，是 **Python 循环的 overhead**。`D` 被 inflate 了，`tau` 被同步压小。作废重测。

## 3. Stage 1a-bis：在真实 serving path 上重测 D

改用 `tinyvllm.LLM`，逐 scheduler step 驱动引擎，分离 prefill / decode，同时跑 CUDA graph 与 eager。过程中发现引擎的 prefill step 本身就吐第一个 token，于是：

```text
D          = actor_prefill(L)   + (A - 1) * actor_step
G_code_ckv = drafter_prefill(B) + head          # B = 512 压缩预算
```

CUDA graph 结果：

```text
context      D_s  a_pre_s  a_step_ms  d_step_ms  ratio  tau_text  tau_code  tau_ckv
   1024   0.4907   0.0814     13.202      3.752  0.284    0.3078    0.0708   0.0691
   4096   0.7487   0.3196     13.843      4.259  0.308    0.2397    0.0634   0.0453
  16384   2.0385   1.5573     15.521      5.513  0.355    0.2244    0.1406   0.0166
```

decode step 比值降到 0.28~0.36，符合物理直觉。判决：

- **GO**，范围锁定 `code_drafter_ckv`（压缩上下文 + 一个 code head）。
- 文本 drafter 淘汰。
- Stage 1b 入门条件：`p >= 0.491 @1024 / 0.485 @4096 / 0.478 @16384`。

看起来成本侧已经解决，剩下唯一未知是命中率。**这个判断是错的，但要到 erratum 才被发现。**

## 4. Stage 1b step 0：先用真实 trace 验证可预测性

在训练任何东西之前，先问语料本身允不允许。两份公开语料，纯 CPU：

```text
Salesforce/APIGen-MT-5k   (cc-by-nc-4.0)   4977 traces / 21955 actions
nebius/SWE-agent-trajectories (cc-by-4.0)  6669 traces / 174815 actions
```

免训练 predictor 的 exact action match（eligible 子集）：

```text
predictor                        apigen    swe_agent
global_top1                      0.0125       0.0836
repeat_last                      0.0095       0.0943
prefix_mode                      0.0047       0.1936
bigram_prev                      0.2065       0.1111
trigram_prev2                    0.2078       0.1129
oracle_in_codebook_4096          0.5813       0.2052
tool 粒度 trigram                0.6446       0.5885
```

结论：

- **NO-GO**：固定 4096-entry discrete code head 做 exact action 预测。SWE 语料的 oracle 上界只有 0.2052，连"码本里存在这个 action"都做不到。
- tool identity 相对容易（0.59~0.64），**arguments 才是瓶颈**。
- 参数可从上下文 copy 的比例 0.6342 / 0.5289，所以提出 tool classifier + argument copy pointer。

## 5. 转向：用户否掉 pointer head

反馈是"还要训练这个指针头，这也太麻烦了吧"。这个否决是对的，而且指出了一个我当时没意识到的错误：

> Stage 1a-bis 定价的是 **"输出 token 数很少"**，不是 **"必须是一个 head"**。head 只是 k=1 的极限情形。真正的约束是一个 **output token budget**，而没人量过这个 budget 有多大、真实 action 写成文本有多长。

于是把 drafter 成本改写成 `compressed_prefill + k * drafter_decode_step`，反解成本模型求允许的 `k`，再用 Qwen3 tokenizer 量真实 action 长度。

```text
break-even k（压缩 512-token 上下文, p=0.75, rollback 0.5s, rho=0.6）
context   tool.2s   tool1s   tool5s   tool20s
1024            0       11       27        27
4096            0       11       42        42
16384           0       10       67       102
```

结论分成两个区间：

- **快工具（约 1s）**：确实必须 k≈1，非 compact head 不可；但收益天花板只有 1.21x，margin 比一次 prefill 抖动还薄，不值得为它做训练。
- **慢工具（>=5s）**：明文 prompted drafter 保留了完美 head 收益的 80~92%，**不训练就能测**。

预注册（在跑任何 GPU 之前写死）：

```text
regime     tool_seconds >= 5.0, rho=0.6, rollback=0.5
drafter    prompted Qwen3-0.6B, greedy, 无训练
SWE        cap k=11   required p: 0.5304 @1024 / 0.4032 @4096 / 0.2064 @16384
APIGen     cap k=27   required p: 0.7488 @1024 / 0.5765 @4096 / 0.3308 @16384
coverage   >= 0.60
STOP LINE  prompted drafter @4096 低于 0.4032 即终止，不升级为训练
```

## 6. Stage 1b step 1：prompted drafter 被证伪，0/30

A100 上跑 prompt selection（rows 0-256）与打分（rows 256-1256）分离，三种 prompt style，两种 variant。

```text
512-token 上下文，Qwen3-0.6B
run                       cap  coverage   p_spec    p_eff   tool_acc
swe_agent_tail_v2          11    0.4430   0.0316   0.0140     0.0745
apigen_tail_tools_v3       27    0.3610   0.0526   0.0190     0.1440
```

三条独立轴全部试过，全部失败：

| 变量 | 最好的 p_spec | 最松门槛 | 比值 |
|---|---|---|---|
| 0.6B / 512 ctx | 0.0526 | 0.2064 | 0.16x |
| **8B actor / 同样 512 ctx** | 0.1702 | 0.2064 | 0.51x |
| 0.6B / 3584 ctx | 0.0859 | 0.2064 | 0.16x |

- 不是模型太小：8B 在 SWE 上反而更差（0.0282 vs 0.0316）。
- 不是上下文太短：APIGen 涨 1.6 倍，SWE 反而崩到 0.0042。

**不依赖成本的那一刀**：把 drafter 税设为 0（一个完全免费的 drafter），profitability guard 仍要求 `p >= 0.2896 @1024 / 0.2108 @4096 / 0.0909 @16384`。全场最好成绩 0.0859，连免费地板都没摸到。

**最刺眼的一条**：prompted 0.6B 被 bigram 计数器打了 2.4x / 3.5x，多花几个数量级算力；连 step 0 认为"容易"的 tool identity 都输（0.07~0.25 vs 0.59~0.64）。

当时留下一格幸存者：16384 ctx + 5s 工具，免费 drafter 只需 `p>=0.0909`，trigram 的 0.2078/0.1129 过线。

## 7. Erratum：分母是假的，幸存者也是假的

追问"这条压缩降成本的线还有没有空间"时，回去读了 worker 的注释，它自己写着：

> 每次重复都用全新随机 prompt，因为引擎带 prefix caching，复用 prompt 会让 prefill 看起来免费。

这对**测单次 prefill 成本**是正确的，对**测一轮 agent 的 demand** 是偷换。而 `block_manager.py` 实现了 block-hash prefix caching，且 deallocate **保留** hash 映射与 token_ids 直到 block 被物理回收 —— append-only 的 agent 回来时只 prefill 最新那段观察。

```text
context      D 冷      D 热      actor prefill
   1024   0.4907   0.4544      0.081 -> 0.045
   4096   0.7487   0.4741      0.320 -> 0.045   (-86%)
  16384   2.0385   0.5262      1.557 -> 0.045   (-97%)
```

后果有三层：

**① tau 的趋势整个反过来。** 压缩上下文从未在买"递减的税"，它只是被一个不该存在的大分母除了一下。

```text
context   tau_ckv 冷   tau_ckv 热
   1024       0.0691       0.0746
   4096       0.0453       0.0715
  16384       0.0166       0.0644
```

**② 幸存者那一格死了。** 热分母下 16384 的免费地板从 0.0909 抬到 0.2754，trigram 的 0.2078/0.1129 双双失败。

**③ 顺手把"压缩本身值多少"也算了。** decode step 拟合 `13.05ms + 0.151us/token * L`，KV attention 占一个 step 的比例：

```text
context     1024    4096   16384   65536   131072
KV attn     1.2%    4.5%   15.9%   43.1%    60.3%
```

一个**完美**压缩器（KV attention 归零）能省的 wall clock：16384 → **1.39%**，65536 → 5.33%，131072 → 10.12%。而 prefill 那半边已被 prefix cache 无损吃掉 97%。

**容量论证也不成立**：Qwen3-8B bf16 的 KV 是 0.141 MiB/token，单卡 A100 util 0.85 扣掉权重与 workspace 剩约 47.6 GiB。

```text
context   GiB/agent   显存上限   算力上限@rho0.6   谁先卡
  16384        2.25   21.2 个            6.3 个   算力
  65536        9.00    5.3 个            4.6 个   算力
 131072       18.00    2.6 个            3.4 个   显存
```

要到 128k 显存才开始卡。而且卡了也轮不到有损压缩：把闲置 KV 甩到 host，PCIe4 约 25GB/s，16384 是 **97ms 恢复 vs 1557ms 冷重算，无损且快 16 倍**。有损压缩要打败的是 97ms，不是 1557ms。

erratum 同时写下唯一能翻盘的条件：**如果真实 agent 每轮追加约 4000 token，热 demand 就仍然很大，投机还能成立。** 这个数从来没人测过。

## 8. Stage 1b step 2：翻盘条件不成立

先厘清三个之前被合并的量：

- `assistant`：actor 自己生成的 token —— **已驻留**，decode 本身写了 KV，不需重算。
- `observation`：环境追加的新文本。
- `prefill`：引擎真正要重算的 = `observation + wrapper + ((ctx_before + assistant) mod block)`。因为 block 粒度是 256，上一轮尾部残块不能复用。

语料重新下载，sha256 与 step 0 逐字节一致。实测：

```text
SWE-agent  2000 traces / 53440 turns
  observation  mean 432.7  p50 136  p90 1246  p99 2084  max 22931
  prefill      mean 560.3  p50 295  p90 1378  p99 2162  max 23023
  ctx_before   mean 12675  p50 10853  p90 25547  max 37125

APIGen-MT  5000 traces (全量) / 41127 turns
  observation  mean 216.6  p50  46  p90  583  p99 1344
  prefill      mean 346.2  p50 239  p90  718  p99 1494
  ctx_before   mean  5184  p50 4799  p90 6833
```

**Agent 确实攒出了长上下文，但它是一小口一小口攒的。** 需要均值约 4000，实测中位 295 / 239，连 p99 都到不了。

```text
corpus      pre_p50  pre_p99   D_warm   min_p@warm   最好免训练 p   gate
swe_agent       295     2162   0.5055       0.2835        0.1129   FAIL
apigen          239     1494   0.4543       0.3057        0.2078   FAIL

过线轮次占比：swe 热 0.0001 (4/53440) 冷 0.4472
              apigen 热 0.0003 (12/41127) 冷 0.6823
```

再给运维一个能直接查的数 —— **prefix cache 要 miss 到多少投机才划算**：

```text
corpus      D 阈值      需要的 miss 率
swe_agent   1.5715s              89%
apigen      0.7625s              79%
```

也就是说，action speculation 只在"五轮里四轮命中不了缓存"的机器上划算。**那不是投机机会，那是缓存在抖 —— 而抖的正解是无损 offload。投机能赚钱的区间，恰好是应该用别的手段修掉的区间。**

稳健性：`block_size` 是唯一自由参数，且往有利于投机的方向推，扫到 1024（引擎实际值 4 倍）时过线比例仍 ≤0.001。

## 9. 最终判决

| 结论 | 状态 |
|---|---|
| action-level speculation + 任何免训练 drafter | **关闭**。翻盘条件事先写好，中位数差 10 倍 |
| 压缩 drafter 上下文"随长度趋零"的成本论证 | **撤回**。它除的是冷 prefill |
| 训练 pointer head 救线 | **不做**。8B 诊断证明短板不是容量 |
| latent / discrete code 作为 action 表示 | **无任何测量支持** |
| 64k~128k 超长上下文 KV 压缩（容量轴） | 仍开着，很窄，对手是无损 offload，且与 latent 脱钩 |
| `tinyvllm/agentspec/` 成本闸门 | **保留为资产** |

## 10. 可复用的教训

**① 分母比分子重要。** 这条线不是死在任何一个 drafter 上，是死在 `D` 的测法上。测 `tau = drafter成本 / D` 的时候，所有注意力都放在分子（drafter 有多便宜），而分母被一个"技术上正确、语义上错误"的 harness 设置放大了 4 倍。**任何比值型指标，先审分母。**

**② "便宜且无用"是最贵的失败模式。** `code_drafter_ckv` 的 tau=0.0166 看起来完美，但它从未被要求预测任何东西。一个成本极低、命中率 0.03 的 drafter，比一个成本高、命中率 0.6 的 drafter 糟糕得多。**成本和效果必须在同一次实验里被测，不能一个 Stage 测成本、下一个 Stage 测效果。**

**③ 无损基线必须先定价。** prefix caching 和 host offload 都是无损、已标配、且比方案本身更强，但整条线从头到尾没跟它们比过一次。**新方案的对手不是"什么都不做"，是"现有的免费无损方案"。**

**④ 预注册阈值省下了大量争论。** step 0b 在跑 GPU 之前写死了 required p、coverage floor 和 STOP LINE。于是 step 1 出结果时，没有"再调调 prompt 试试"的空间 —— 0/30 就是 0/30。**先写判决标准，再跑实验。**

**⑤ 免训练基线是真对手，不是陪衬。** 一个 bigram 计数器打赢了 prompted 0.6B（2.4x/3.5x）和 8B。**在提议训练任何东西之前，先让最蠢的方法跑满。**

**⑥ harness 会说谎，而且不止一次。** Stage 1a 是 eager 循环 overhead，Stage 1a-bis 是冷 prefill。两次都不是建模错误，是**测量方法学**错误，而且两次都让结论朝有利方向偏。**当一个结果对自己太友好时，先怀疑 harness。**

**⑦ 一个能证伪的成本闸门，比一个能跑的 kernel 值钱。** 三个提案（固定码本 head / pointer head / prompted drafter）全部在纸面或纯 CPU 阶段被杀掉，训练 0 次，GPU 累计数小时。`tinyvllm/agentspec/` 的 48 个测试是这条线唯一的净资产。

## 11. 附：所有 artifact

```text
tinyvllm/agentspec/{action,latent_adapter,cost_model,router}.py
tools/agentspec_breakeven_gate.py, test_agentspec_breakeven_gate.py
tools/agentspec_drafter_tax_worker.py, run_agentspec_drafter_tax_remote.sh
tools/agentspec_engine_demand_worker.py, run_agentspec_engine_demand_remote.sh,
      agentspec_engine_demand_verdict.py
tools/agentspec_trace_normalize.py, agentspec_trace_match_baseline.py,
      agentspec_trace_copyability.py
tools/agentspec_output_token_budget.py, agentspec_action_token_length.py
tools/agentspec_prompted_drafter_{evalset,match_worker,verdict,debug}.py,
      run_agentspec_prompted_drafter_remote.sh
tools/agentspec_context_growth.py, agentspec_context_growth_verdict.py

experiments/agentspec_drafter_tax/       Stage 1a
experiments/agentspec_engine_demand/     Stage 1a-bis
experiments/agentspec_trace_baseline/    Stage 1b step 0 / 0b
experiments/agentspec_output_budget/     Stage 1b step 0b
experiments/agentspec_prompted_drafter/  Stage 1b step 1
experiments/agentspec_context_growth/    Stage 1b step 2
```

语料一律不提交（CC BY-NC 4.0 / CC BY 4.0），只落 token 统计与派生成本；每个 artifact 带 `payload_sha256`，输入带 `input_sha256`。

commit 序列：

```text
6d8c413  Stage 0 analytic gate
430b622  Stage 1a harness
3ca6553  Stage 1a measurement
3cfcbab  Stage 1a-bis serving-path D
9d8cca3  Stage 1b step 0 falsification
1269e58  Stage 1b step 0b token budget
aeb6cbd  Stage 1b step 1 prompted drafter falsified
127c19c  erratum: cold-prefill artifact retracted
e45668e  Stage 1b step 2 context growth, line closed
```
