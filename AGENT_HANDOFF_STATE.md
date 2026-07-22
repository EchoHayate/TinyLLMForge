# Agent Handoff State

> 目的：上下文中断后，新的 agent 先读这个文件，避免重新猜工作区、远程环境和当前任务状态。

## 2026-07-22 Heuristic-equivalent exact-width CUDA Graph diagnostic GO

### 最终结论

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- HEAD before this handoff update：`ca994fb`
- Source tree SHA256：
  `7997be699eb85046d399959b268ada0ab92cd14c06fbca1547d3833d1876db8e`
- Authoritative canonical：
  `experiments/cuda_graph/qwen3-06b-heuristic-exact-width-canonical-20260722-055943/`
- Independent verifier：
  `experiments/cuda_graph/qwen3-06b-heuristic-exact-width-canonical-20260722-055943/independent-verification/summary.json`
- Gate A：`EXACT_REPLAY_CORRECT`
- Gate B：`LEGACY_COMPATIBLE`
- Policy：`POLICY_EXACT`
- Rounded negative control：`ROUNDED_REPLAY_CORRUPT`
- Structural failures：`0`

这次 fresh source-bound canonical 已完成 diagnostic `GO`：FlashAttention
2.6.3 等价 split heuristic 与 exact page-table-width 共同进入 graph
identity 后，exact-width CUDA Graph replay 在完整矩阵上与 candidate eager
一致，并且 candidate eager 与现有 legacy auto-eager 语义兼容。

这仍然**不是 production 性能提升结论**。Production
`batch > 1` eager guard 未修改，README 未更新；当前引擎线上路径仍不会使用
该 multi-sequence graph candidate。下一步若继续，必须先做独立 production
设计，加入 graph allowlist/budget、fallback/telemetry、初始化与显存上限，
然后跑真实吞吐、ITL、request-rate、graph-hit 和 memory gate。

### Canonical 覆盖与独立审计

冻结合同：

```text
Gate A                         189 processes
Gate B                         126 processes
total                          315 processes
batches                        2, 3, 4, 5, 8, 9, 16
trajectories                   uniform-short, ragged-context,
                               duplicate-and-distinct
repetitions                    3
warmup / measured steps        2 / 16
raw rows                       5040
layer observation rows         5040
KV observation rows            5040
unique dynamic ports           630
logit tolerance                rtol=1e-3, atol=1e-2
```

Fresh independent verifier：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  verify-only \
  --run-tag qwen3-06b-heuristic-exact-width-canonical-20260722-055943 \
  --verifier-python \
  /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

结果：

```text
verify.exitcode                0
classification                 EXACT_REPLAY_CORRECT
legacy_compatibility           LEGACY_COMPATIBLE
policy_integrity               POLICY_EXACT
rounded_classification         ROUNDED_REPLAY_CORRUPT
corrupt exact cases            0
corrupt rounded cases          63 / 63
incompatible legacy pairs      0
policy failures                0
structural failures            0
```

额外独立审计已逐项确认：

1. `315` 个 case ID 全部唯一且与 manifest/process rows 完全一致；
2. `630` 个 `TINYVLLM_DIST_PORT` / `MASTER_PORT` 全局唯一，且每个
   process 的两端口不同；
3. `189 / 126 / 63` 的 Gate A、Gate B process、compatibility pair
   计数一致；
4. raw/layer/KV rows 均为 `5040`；
5. `sha256sums.txt` 中 `1083` 个 artifact 全部重新计算匹配；
6. manifest、staging source evidence、promoted source evidence 的 source
   tree SHA256 一致；
7. 所有 process row 均为 `PASS`；
8. production eager guard 精确保留一处：
   `multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1`；
9. `README.md` 无 tracked diff。

本地 fresh regression：

```text
tools/test_multi_sequence_cuda_graph_gate.py   PASS
tools/test_context_modes.py                    PASS
tools/test_model_runner_spec_verify.py         PASS
changed Python files py_compile                PASS
git diff --check                               PASS
```

### 共享 GPU 容量失败的根因

Canonical 初次在
`b5__uniform-short__compat__r2__legacy_eager_auto__auto-s0` 停止，后续
resume 又在
`b8__uniform-short__compat__r2__candidate_eager_heuristic__fa2-263-exact-width`
停止。两次 traceback 都只发生在 LLM 初始化：

```text
ModelRunner.allocate_kv_cache()
assert auto_num_blocks > 0
```

Diagnostic 固定 `gpu_memory_utilization=0.55`。GPU 0 总显存
`81920 MiB`，55% budget 为 `45056 MiB`；失败时外部共享进程已占用约
`46800-47000 MiB`。两次都有另一个短时 int4 校验进程额外占用约
`16 GiB`，因此模型加载前就已超过 KV budget。

没有修改 production 或 diagnostic source，也没有 kill 任何共享进程。
等待外部任务自然退出、GPU used 回落到约 `30619 MiB` 后，用相同 run tag、
source snapshot、配置和 case 原样 `--resume`：

- 原 b5 失败 case 转为 `PASS`；
- 原 b8 失败 case 转为 `PASS`；
- 最终 `315/315` 全部完成；
- 没有 token、logit、layer、KV、policy、hash 或 verifier failure。

因此根因已确认是共享 GPU 的瞬时容量竞争，不是 heuristic/exact-width
candidate correctness 回归。

### 书面设计、执行入口与边界

设计：

```text
docs/superpowers/specs/2026-07-22-heuristic-equivalent-exact-width-cuda-graph-recovery-design.md
```

批准计划：

```text
docs/superpowers/plans/2026-07-22-heuristic-equivalent-exact-width-cuda-graph-recovery.md
```

Canonical resume 命令：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  heuristic-exact-width-canonical \
  --run-tag qwen3-06b-heuristic-exact-width-canonical-20260722-055943 \
  --resume \
  --verifier-python \
  /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

后续决策：

1. Diagnostic candidate 已具备进入 production **设计阶段**的证据；
2. 不得直接删除 production eager guard；
3. rounded-width replay 已被完整负控证明 corruption，永久保持
   diagnostic-only / production-disabled；
4. 下一阶段必须先批准书面 production spec，再实现 bounded exact-width
   graph cache 和 eager fallback；
5. 只有 production source-bound 性能 gate 证明吞吐/ITL/request-rate
   改善且初始化与显存预算通过后，才能宣称推理引擎真正变快或更省。

## 2026-07-22 Fixed-split multi-sequence CUDA Graph recovery hard checkpoint

### 最终结论

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- Source base commit：`a482f0ee44c43ac46aa6cc256a6f14649714414e`
- Source tree SHA256：
  `afcb22ada96e9341717d93ee51e4b5d1e87332c4d79744f9650cbfc4799fb842`
- Authoritative canonical：
  `experiments/cuda_graph/qwen3-06b-fixed-split-canonical-20260721-174406/`
- Independent verifier：
  `experiments/cuda_graph/qwen3-06b-fixed-split-canonical-20260721-174406/independent-verification/summary.json`
- Gate A：`EXACT_REPLAY_CORRECT`
- Rounded diagnostic：`ROUNDED_REPLAY_CORRUPT`
- Gate B：`LEGACY_INCOMPATIBLE`
- Hard checkpoint 未通过：**Tasks 8-10 未授权，production batch>1 eager
  guard 必须保持，README 不更新。**

该结果证明：当 candidate eager、capture 和 replay 都固定使用
`flash_attn_num_splits=16` 时，exact-key multi-sequence graph replay 在冻结的
189-process matrix 上与 fixed16 eager 一致。它不证明 fixed16 与现有
auto-split eager 语义兼容，也不证明 production 吞吐、延迟或显存有提升。

### 书面设计与执行入口

设计：

```text
docs/superpowers/specs/2026-07-21-fixed-split-multi-sequence-cuda-graph-recovery-design.md
```

批准计划：

```text
docs/superpowers/plans/2026-07-21-fixed-split-multi-sequence-cuda-graph-recovery.md
```

Canonical 初始命令：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  fixed-split-canonical \
  --run-tag qwen3-06b-fixed-split-canonical-20260721-174406 \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

同一 source-bound run 的恢复命令：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  fixed-split-canonical \
  --run-tag qwen3-06b-fixed-split-canonical-20260721-174406 \
  --resume \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

执行期间 GPU 0 受到共享服务的瞬态显存竞争。仅当 traceback 同时包含
`allocate_kv_cache` 和 `auto_num_blocks > 0` 时，外部 supervisor 才等待
GPU 0 连续三次 `<=31000 MiB` 后恢复同一 run；没有吞掉任何算法、tensor、
token、KV 或 verifier 失败。

### 环境与冻结合同

- Remote：`sitian@10.232.195.203`
- Host：`n232-195-203`
- GPU：`NVIDIA A100 80GB PCIe`
- Driver：`535.261.03`
- CUDA runtime：`12.1`
- Python：`3.11.15`
- PyTorch：`2.4.1+cu121`
- FlashAttention：`2.6.3`
- Transformers：`5.8.1`
- Model：
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Gate A：`7 batches × 3 trajectories × 3 modes × 3 repetitions = 189`
- Gate B：`7 batches × 3 trajectories × 2 policies × 3 repetitions = 126`
- Batch：`2, 3, 4, 5, 8, 9, 16`
- Trajectory：`uniform-short`、`ragged-context`、
  `duplicate-and-distinct`
- Warmup / measured decode steps：`2 / 16`
- Auto split：`0`
- Fixed split：`16`
- Logit tolerance：`rtol=1e-3`、`atol=1e-2`
- Greedy token arrays：exact comparison

### Canonical 完整性审计

除 verifier 自身外，另一次只读审计逐项验证并重算：

- `315` 个 unique case IDs；
- `630` 个 unique dynamic ports；
- `189` 个 same-policy process rows；
- `126` 个 compatibility process rows；
- `63` 个完整 compatibility pairs；
- `5040` 个 raw rows；
- `5040` 个 layer rows；
- `5040` 个 KV rows；
- 每个 layer row 都有完整 `28/28` layers 且 finite；
- source/environment/prompt identity 一致；
- fixed graph 只与 fixed16 eager 比较；
- fixed16 eager 只与 auto eager 做 compatibility 比较；
- `sha256sums.txt` 中 `1083` 个文件 hash 全部从磁盘重算一致；
- independent verifier structural failures：`0`。

关键 hash：

```text
manifest.json
8950dc3fdd8ec9b9cf7e137dd717a61113212f9639c8be1dc02a00c6a92360b7

process_rows.jsonl
169694a82c6529e992453b1dea8bfea02026a00955e91f9544f9c0ff5884cc51

independent-verification/summary.json
9168f4273d4e672ef772344f42266d08d2b18b4d837e4b92ddf3275bd30121d1
```

### Gate A 与 rounded 结果

Gate A 的 `63` 个 exact graph cases 全部通过 fixed16 eager 对照：

```text
classification=EXACT_REPLAY_CORRECT
corrupt_exact_case_ids=[]
```

这支持原 multi-sequence corruption 与 capture/replay 期间
FlashAttention auto split identity 不稳定有关；显式 fixed16 后
exact-key replay correctness 恢复，但本 gate 不证明它是唯一根因。

Rounded graph 的 `63/63` cases 全部 corruption：

```text
rounded_classification=ROUNDED_REPLAY_CORRUPT
```

首个 independent divergence：

```json
{
  "case_id": "b16__duplicate-and-distinct__rounded_graph_fixed16__fixed16-s16__r0",
  "evidence": "logits",
  "kind": "close_failure",
  "row_id": 0,
  "step_id": 0
}
```

因此 rounded replay 仍只能作为 diagnostic negative control，绝不能进入
production。

### Gate B 失败模式

Gate B 的 `54/63` pairs 完全兼容；`9/63` pairs 不兼容，且模式跨三次
repetition 完全一致：

```text
b8__ragged-context__compat__r0/r1/r2
b9__ragged-context__compat__r0/r1/r2
b16__ragged-context__compat__r0/r1/r2
```

只读 raw tensor 复算得到：

- `9/63` pairs 的 logits 超出冻结 tolerance；
- 同 `9/63` pairs 的 intermediate layer tensors 超出 tolerance；
- `6/63` pairs 发生 greedy argmax/token 改变：
  batch `8/9`、ragged-context、全部三次 repetition；
- batch `16` ragged-context 的三次 repetition 虽然 greedy token 暂时相同，
  但 logits 最大绝对差达到 `5.75`，仍不满足兼容合同；
- uniform-short、duplicate-and-distinct，以及 ragged-context batch
  `2/3/4/5` 的所有 repetitions 都是 tensor-exact compatible。

batch `8/9` 的首个 token divergence 均为 measured step `2`、row `6`：

```text
auto token=19357
fixed16 token=840
```

该失败不是“容差略紧”这么简单：固定 split 已在可复现 workload 上改变
greedy 输出，因此不能用放宽 tolerance、忽略 logits mismatch 或只看
producer PASS 的方式解除 production guard。

### Producer 与 independent verifier 边界

顶层 producer `summary.json` 自报：

```text
EXACT_REPLAY_CORRECT / ROUNDED_REPLAY_CORRECT / LEGACY_COMPATIBLE
```

它只汇总 process-level producer 状态，不能替代 tensor comparison。独立
verifier 从 hashed raw logits/layers/KV/token artifacts 重算后给出：

```text
EXACT_REPLAY_CORRECT / ROUNDED_REPLAY_CORRUPT / LEGACY_INCOMPATIBLE
```

因此 authoritative 结论只能使用
`independent-verification/summary.json`，不能使用 producer
`summary.json`。

### Production 决策与后续边界

- `tinyvllm/engine/model_runner.py` 中
  `multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1`
  的 eager guard 保持不变。
- 未修改 `tinyvllm/config.py`、production graph capture/dispatch、
  `tinyvllm/engine/llm_engine.py`、arrival-load evidence 或 README。
- 没有运行 production performance gate，因此当前没有新的 production
  性能提升数字；fixed16 exact replay correctness 不能等价为性能收益。
- Tasks 8-10 不得继续，除非另起书面 root-cause/design，解决
  auto-vs-fixed16 的 ragged batch semantic drift，并重新跑 fresh
  source-bound Gate A/Gate B canonical。
- 不能通过放宽 tolerance、删除 ragged-context、缩小 batch matrix、
  忽略 token mismatch 或把 eager fallback 改成 fixed16 来绕过 Gate B。

## 2026-07-21 Multi-sequence CUDA Graph canonical hard gate

### 当前结论

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- Final source commit：`9e1610a`
- Authoritative canonical：
  `experiments/cuda_graph/qwen3-06b-cuda-graph-canonical-20260721-r3/`
- Producer `summary.json` 自报：
  `EXACT_REPLAY_CORRECT / ROUNDED_REPLAY_CORRECT`
- **独立 verifier 才是 hard gate，最终分类为：
  `EXACT_REPLAY_CORRUPT / ROUNDED_REPLAY_CORRUPT`**
- Canonical 189/189 model processes 全部 producer PASS，结构失败为 0；
  corruption 是证据重算发现的数值 correctness failure，不是进程、存储、
  source identity 或 artifact 完整性失败
- Stop condition 已触发：**不得实现或启用 production multi-sequence
  exact-key CUDA Graph dispatch**
- `tinyvllm/engine/model_runner.py` 的 production batch>1 eager guard 必须保持；
  README 不更新，不能宣称 multi-sequence CUDA Graph 性能提升

### 书面设计与实现

设计与计划：

```text
docs/superpowers/specs/2026-07-21-multi-sequence-cuda-graph-correctness-and-batching-gate-design.md
docs/superpowers/plans/2026-07-21-multi-sequence-cuda-graph-correctness-and-batching-gate.md
```

提交：

```text
8c8a75f docs: design multi-sequence cuda graph gate
875b8d2 docs: plan multi-sequence cuda graph gate
b8b5ddd test: freeze multi-sequence cuda graph gate
6feaef8 feat: add isolated cuda graph diagnostic
783c49b feat: verify cuda graph diagnostic evidence
6a38ce7 feat: orchestrate remote cuda graph diagnostic
a0f4131 fix: bound remote cuda graph artifacts
9e1610a fix: retry remote diagnostic port collisions
```

核心工具：

```text
tools/multi_sequence_cuda_graph_contract.py
tools/diagnose_multi_sequence_cuda_graph.py
tools/verify_multi_sequence_cuda_graph_diagnostic.py
tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
tools/test_multi_sequence_cuda_graph_gate.py
```

### Final source-bound canonical

执行命令：

```bash
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  diagnostic-canonical \
  --run-tag qwen3-06b-cuda-graph-canonical-20260721-r3 \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

环境与 source identity：

```text
host                 sitian@10.232.195.203
remote Python        /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model                /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
GPU                  NVIDIA A100 80GB PCIe
PyTorch              2.4.1+cu121
CUDA runtime         12.1
source tree SHA256   547fde844c73221c503657610e49e97c5255c2e2c986f3afc698abf99588a91f
```

冻结矩阵：

```text
batch sizes          2, 3, 4, 5, 8, 9, 16
trajectories         uniform-short, ragged-context, duplicate-and-distinct
modes                eager, exact_graph, rounded_graph
repetitions          0, 1, 2
process count        7 x 3 x 3 x 3 = 189
warmup steps         2
measured steps       16
```

结构审计：

```text
case directories                 189
producer PASS case results       189
non-zero model process exits       0
process_rows.jsonl rows          189
raw_rows.jsonl rows            3,024
layer_observations rows        3,024
kv_observations rows           3,024
structural verifier failures       0
remote run root after finish    CLEANED
```

Remote artifact cleanup 在运行期间始终只保留 0 或 1 个当前 case 目录；
普通占用约 4.1 MB，大 batch 观察峰值约 177 MB，避免了旧 canonical r1
因远端 `/tmp` 累积约 9 GB 后 `ENOSPC` 的失败。

### Authoritative independent verifier result

权威 artifacts：

```text
experiments/cuda_graph/qwen3-06b-cuda-graph-canonical-20260721-r3/independent-verification/summary.json
experiments/cuda_graph/qwen3-06b-cuda-graph-canonical-20260721-r3/independent-verification/report.md
experiments/cuda_graph/qwen3-06b-cuda-graph-canonical-20260721-r3/independent-verification/verify.exitcode
```

Fresh 手工复跑：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  --run-dir \
  experiments/cuda_graph/qwen3-06b-cuda-graph-canonical-20260721-r3 \
  --kind diagnostic
```

复跑 exit code 为 `1`，且稳定重现：

```text
classification             EXACT_REPLAY_CORRUPT
rounded classification     ROUNDED_REPLAY_CORRUPT
case count                 189
structural failures        0
exact corrupt cases        15
rounded corrupt cases      63
```

Exact corruption 的模式是确定且跨 repetition 重现的：

```text
trajectory                 ragged-context only
batch sizes                4, 5, 8, 9, 16
repetitions                0, 1, 2
per-batch failures         3/3 repetitions
```

Exact corrupt case ids：

```text
b4__ragged-context__exact_graph__r0
b5__ragged-context__exact_graph__r0
b8__ragged-context__exact_graph__r0
b9__ragged-context__exact_graph__r0
b16__ragged-context__exact_graph__r0
b4__ragged-context__exact_graph__r1
b5__ragged-context__exact_graph__r1
b8__ragged-context__exact_graph__r1
b9__ragged-context__exact_graph__r1
b16__ragged-context__exact_graph__r1
b4__ragged-context__exact_graph__r2
b5__ragged-context__exact_graph__r2
b8__ragged-context__exact_graph__r2
b9__ragged-context__exact_graph__r2
b16__ragged-context__exact_graph__r2
```

Rounded replay 63/63 cases 全部 corrupt，因此 padded/rounded replay 也不得
进入 production。Verifier 的全局排序首个 divergence 是：

```json
{
  "case_id": "b16__duplicate-and-distinct__rounded_graph__r0",
  "evidence": "logits",
  "kind": "close_failure",
  "row_id": 0,
  "step_id": 0
}
```

注意：该 `first_divergence` 是所有 exact + rounded divergence 按 case id
排序后的第一个，不代表 exact corruption 只发生在该 rounded case。Exact
admission 应直接依据 `corrupt_exact_case_ids` 和顶层
`EXACT_REPLAY_CORRUPT`。

### Runner hardening history

1. Canonical r1 在 119/189 后因远端为所有 case 保留 artifacts，最终
   `/tmp` 满并报 `OSError: [Errno 28] No space left on device`。
2. `a0f4131` 改为每个 case 下载并原子替换本地目录后，只删除自己唯一的
   remote case path；fresh smoke r2 18/18 验证 bounded cleanup。
3. Canonical r2 在 135/189 遇到 `MASTER_PORT` 的瞬态
   `EADDRINUSE`。已有 bounded retry 只读取 `launcher_stderr.txt`，但真实
   traceback 在 diagnostic `output/stderr.txt`，因此被误判为不可重试。
4. `9e1610a` 通过 TDD 聚合两份 stderr，使现有 bounded retry 可识别
   collision；本地完整 gate 与 remote immutable-source preflight PASS。
5. 因 source hash 改变，r2 没有 resume 或混入新 source，而是 fresh 跑
   canonical r3。

### Exact replay root-cause probe

Canonical NO-GO 后对 capture/replay 输入链做了只读审计，并在远端唯一
临时 source copy 上做最小 probe；production checkout、production dispatch
和 batch>1 eager guard 均未修改。Probe artifacts：

```text
experiments/cuda_graph/qwen3-06b-cuda-graph-root-cause-split-probe-20260721/
```

静态与动态审计没有发现 graph input metadata 漏拷贝：
`input_ids`、`positions`、`slot_mapping`、`context_lens` 和
`block_tables` 在 replay 前均先清零，再覆盖所有 active rows。根因证据
指向 FlashAttention 2.6.3 decode 的 split selection：

```text
flash_attn_with_kvcache(..., num_splits=0)
```

其中 `num_splits=0` 使用自动 heuristic，CUDA Graph capture 会固化 capture
时选择的 kernel/split 路径，而 eager 会针对当步 ragged shape/length
重新选择。Canonical verifier 因而可能在比较两条不同 reduction 路径，
并把数值差异分类为 replay corruption。`num_splits=1` 并不能修复 batch 4；
`num_splits=16` 则提供了稳定的固定策略，且与已有 speculative verifier
的固定 split 先例一致。

最关键的同策略对照为：

```text
comparison      fixed-split16 CUDA Graph replay vs fixed-split16 eager
trajectory      ragged-context
batches         4, 5, 8, 9, 16
repetition      0
process exits   all 0
result          all_equal=true
```

`graph-vs-eager-s16-comparison.json` 对每个 batch 独立确认：

- logits：逐元素相等，`max_abs=0`；
- 所有 layer hook outputs：逐元素相等，`max_abs=0`；
- KV `keys_before`、`values_before`、`keys_after`、`values_after`：
  全部逐元素相等，`max_abs=0`。

这排除了“固定 split 下 graph replay 本身仍会破坏 ragged rows”的假设，
并把当前最强根因定位为 auto split 在 graph capture/replay 与 eager
动态执行之间的策略不一致。早先
`fixed-split16-matrix-comparison.json` 在 batch 8/9/16 仍显示差异，是因为
它把 fixed-split16 graph 与 canonical auto-split eager 比较；这证明不同
split 算法路径可能产生显著数值差异，不是同策略 graph corruption。

该 probe **不改变 canonical hard gate**，也不证明 production 修复完成：

1. 只覆盖历史 exact-corrupt 的 5 个 batch、单 trajectory、单 repetition；
2. 未证明 fixed split 对原 auto-split eager 满足 canonical tolerance；
3. 未跑完整 189-case matrix，也未验证 rounded graph；
4. 未测 fixed split 的吞吐、延迟或显存收益；
5. artifacts 是 root-cause evidence，不是 production admission artifact。

### Production safety state

`tinyvllm/engine/model_runner.py` 当前仍 fail closed：

```python
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

该条件进入 eager path；只有 batch 1 保留 CUDA Graph replay。不要删除、
绕过或放宽该 guard。Producer summary 的 `CORRECT` 是非权威 proxy，已经
被独立 verifier 推翻，不能作为 production admission 依据。

### Prompt-to-artifact completion audit

1. **批准的 189-process canonical matrix**：
   189 个 case directory、189 个 producer PASS、189 个 process rows；
   7 batches × 3 trajectories × 3 modes × 3 repetitions 全覆盖。
2. **Remote-only GPU/model execution**：
   manifest/environment 绑定指定 `sitian` host、A100、Qwen3-0.6B 和
   remote Python；未修改远端 checkout。
3. **Immutable source identity**：
   `source_evidence.json`、`source_snapshot.tar.gz`、manifest 和
   environment 绑定同一 source tree SHA256。
4. **Unique dynamic ports and transient collision handling**：
   runner 全局保留已用端口，只有 `EADDRINUSE` 可 bounded retry；
   `9e1610a` 确保 diagnostic stderr 会进入 retry classifier。
5. **Bounded remote storage**：
   逐 case 下载后清理唯一 remote path；canonical 完成后 remote root
   为 `CLEANED`。
6. **Independent verifier hard gate**：
   artifact 已落盘，fresh 手工复跑 exit `1`，稳定返回
   `EXACT_REPLAY_CORRUPT`；不是只依赖 runner stdout。
7. **Exact replay admission**：
   15 个 exact cases 在 ragged batch 4/5/8/9/16 上跨 3 repetitions
   corrupt，因此 admission 条件不满足。
8. **Rounded replay admission**：
   63/63 rounded cases corrupt，不能启用 padded replay。
9. **Production guard**：
   `model_runner.py` batch>1 decode 仍强制 eager，未做 production
   dispatch 变更。
10. **Documentation/claim boundary**：
    本 handoff 记录 authoritative NO-GO；README 保持不变，不宣称性能
    提升。

结论：本轮 correctness gate 已完成，但产品结论是 **NO-GO**。当前没有
multi-sequence CUDA Graph 性能提升，因为该路径未获准进入 production；
已验证的收益是避免错误优化上线，并保留 batch-1 graph fast path。

### 后续方向

不要直接继续 production exact-key dispatch。若未来要修复
multi-sequence replay，必须先做新的书面 root-cause/design：

1. 针对 ragged exact failures 对比 eager/exact 的 per-row logits、
   per-layer outputs 与 active KV slot，定位首次 exact divergence，而不是
   仅看全局 rounded `first_divergence`；
2. 优先审计 capture/replay 的 row-specific context metadata、slot mapping、
   cu-seqlens/sequence lengths 和 graph input zero/copy 顺序；
3. 任何修复必须 TDD，并 fresh source-bound 跑完整 189-process canonical；
4. 只有独立 verifier 返回 `EXACT_REPLAY_CORRECT` 才能重新讨论 exact-key
   production dispatch；rounded replay 需要独立的
   `ROUNDED_REPLAY_CORRECT`，不能由 exact 结果推断。
5. 若采用固定 `num_splits` 修复，fresh canonical 的 eager baseline 与
   graph path 必须使用书面冻结且可审计的同一 split policy；不得再把
   auto-split eager 与 fixed-split graph 混作 replay correctness 对照。
6. 任何 source、split policy、verifier 或 matrix 变化都必须生成新的
   source-bound artifacts，并由独立 verifier 重算；只有 full canonical
   hard gate 通过后，production batch>1 eager guard 才有资格重新评审。

## 2026-07-20 P4 SAM backlog-adaptive mixed-prefill canonical

### 当前结论

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- 最终 source commit：
  `6c416252d41b4ec8d9783fc9eb20a9fd7e2c22a2`
- authoritative canonical：
  `qwen3-06b-sam-p4-canonical-v2-20260720-142635`
- 独立 verifier 最终分类：**NO_GO**，不是 `INCOMPLETE`
- `correctness_failures=[]`、`structural_failures=[]`，但 P4 没有吞吐或
  显存收益，并违反 decode-tail / fairness guards
- P4 必须保持 disabled-by-default，不能 promotion，不能宣称
  engine-wide speedup；`README.md` 按书面计划保持不变

### 目标与实现

本轮目标是把 production arrival-load 下的 backlog-adaptive
mixed-prefill 做成 fail-closed 的 `P4` 候选，并用 source-bound
`preflight → smoke → calibration → canonical → local independent verify`
链路判断它是否真的让引擎更快或更省。

P4 稳定名称为 `sam_backlog_adaptive_mixed_prefill`：

- `waiting_depth >= 8` 连续两次后进入 `active`
- `waiting_depth <= 2` 连续两次后停止新 admission
- runnable decode 存在时最多连续两个 mixed step，然后强制一次
  decode-only yield
- mixed admission 在确认可保留 decode row 前不分配 prefill KV、不调用
  `may_append()`、不改变队列所有权
- adaptive 默认关闭，且与 always-on mixed-prefill、KV offload 互斥

书面设计与计划：

```text
docs/superpowers/specs/2026-07-20-sam-adaptive-mixed-prefill-design.md
docs/superpowers/plans/2026-07-20-sam-adaptive-mixed-prefill-implementation.md
```

实现提交：

```text
b9e4d0b feat(scheduler): add adaptive mixed configuration
e08c52f feat(scheduler): add adaptive mixed state machine
122d418 fix(scheduler): make mixed admission transactional
0752016 feat(scheduler): route backlog adaptive mixed prefill
9316292 feat(gate): add source-bound P4 policy matrix
a9e8734 feat(verifier): reconstruct P4 scheduler state
fc9e7c6 test(gate): bind P4 remote evidence chain
6c41625 fix(eval): model adaptive draining normalization
```

### Final source-bound evidence

```text
host                 sitian@10.232.195.203
python               /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model                /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
GPU                  NVIDIA A100 80GB PCIe
source tree SHA256   9634f2362f5cbbcc7177dfbead6894878c257a149739b20c5395ee2a559521c4
environment SHA256   2899270bf7c81f4ac940468bab240f9732ab20bc42b640e5c6affe15fe45c485
workload SHA256      09e7a5a9839ec7397ca50259020e654e3a914b2ef55ae7d5783a91a0fd1094db
remote exitcode      0
independent exitcode 0
classification       NO_GO
```

最终链路 run tags：

```text
preflight    qwen3-06b-sam-p4-preflight-v2-20260720-140858
smoke        qwen3-06b-sam-p4-smoke-v2-20260720-140921
calibration  qwen3-06b-sam-p4-calibration-v2-20260720-141438
canonical    qwen3-06b-sam-p4-canonical-v2-20260720-142635
```

Calibration：

```text
lambda_ref_rps                    0.5625
maximum stable throughput rps     0.6451410340387169
```

Canonical 结构审计：

```text
matrix                            P0/P3/P4 × 6 scenarios × 3 repetitions
case rows                         54/54, all PASS
unique case ids                   54/54
request timeline rows             4,392
scheduler trace rows              138,990
memory trace rows                 138,990
process port pairs                54/54 unique
individual dynamic port values    108/108 unique
artifact SHA-256                  13/13 matched
exact/correct lifecycle failures  0
structural failures               0
```

Authoritative artifacts：

```text
experiments/arrival_load/qwen3-06b-sam-p4-canonical-v2-20260720-142635/case_rows.jsonl
experiments/arrival_load/qwen3-06b-sam-p4-canonical-v2-20260720-142635/run_manifest.json
experiments/arrival_load/qwen3-06b-sam-p4-canonical-v2-20260720-142635/summary.json
experiments/arrival_load/qwen3-06b-sam-p4-canonical-v2-20260720-142635/independent-verify/summary.json
experiments/arrival_load/qwen3-06b-sam-p4-canonical-v2-20260720-142635/independent-verify/verify.exitcode
```

### P4 performance result

以下均为同 scenario、同 repetition 的 `P4 / P0` 比值，由独立 verifier
重算：

| Metric | Median ratio | Worst ratio | Interpretation |
|---|---:|---:|---|
| request throughput | 0.999911 | 0.954690 | 无 aggregate throughput 收益，最坏低 4.53% |
| p95 TTFT | 0.728248 | 2.164649 | aggregate TTFT 改善，但有严重离群 |
| p99 TTFT | 0.852532 | 1.071182 | aggregate 改善 |
| p95 ITL | 1.225369 | 12.327660 | median 已回退 22.54%，最坏 12.33x |
| p99 ITL | 1.064474 | 5.889306 | 最坏违反 10% guard |
| p99 E2E | 0.902166 | 1.072265 | aggregate 改善 |
| maximum decode gap | 1.038452 | 7.641892 | 最坏违反 10% guard |
| peak CUDA reserved | 1.000535 | 1.000817 | 无显存节省 |
| peak KV bytes | 3.142857 | 5.142857 | KV footprint 显著增加 |

P4 guard failures：

```text
p99_itl_ns regression exceeds 10%
maximum_decode_gap_ns regression exceeds 10%
service bucket p95 E2E regression exceeds 10%
```

因此 `benefit_path=null`。P4 的确降低 aggregate prefill/TTFT/E2E，但它
没有吞吐或 memory benefit，并把部分 decode 请求推入严重 tail/fairness
离群，不能推广。

### Tail regression attribution

最严重问题集中在 `mixed_service_fairness r0`：

```text
throughput ratio             0.954690
p95 ITL ratio               12.327660
p99 ITL ratio                5.889306
maximum decode gap ratio     7.641892
p95 TTFT ratio               2.164649
```

同一 case 的 service-bucket p95 E2E 离群：

```text
medium__short   38.662x   0.234 s -> 9.035 s
long__short     28.908x   0.359 s -> 10.382 s
short__long     16.291x   0.765 s -> 12.455 s
long__long      14.096x   0.902 s -> 12.720 s
medium__long     8.241x   1.729 s -> 14.245 s
```

`mixed_service_fairness r2` 仍有 `medium__short=1.610x` 和
`long__short=1.115x` bucket 回退，说明问题不只是一项 metric 的计算噪声。
此外 `long_prompt_pressure` 三个 repetition 的 p95 ITL 均约
`3.77x–3.92x`，表明 mixed prefill 在长 prompt 压力下持续挤压 decode
cadence。`burst` 虽有约 `1.99x–2.40x` throughput ratio 和显著 TTFT
收益，但其三个 repetition 的 p95 ITL 仍为 `1.08x–1.28x`，不足以覆盖
全 workload 的 tail guard。

Trace 显示 P4 确实被执行而非 dormant：例如
`mixed_service_fairness r0` 有 35 个 mixed prefill+decode step、16 个
decode yield、76 个 decode fallback，最大 waiting depth 15。当前证据
指向 controller 的 activation/admission 粒度和固定“两 mixed 后一
yield”不足以保护 heterogeneous decode service，而不是状态机未触发。

### Verifier fix history

首个 canonical
`qwen3-06b-sam-p4-canonical-20260720-104058` 的远端模型矩阵完成，但旧
verifier 报：

```text
ValueError: adaptive state transition mismatch
```

根因是 verifier 漏建模 scheduler 在同一 scheduling decision 内把
`draining + empty prefilling` 规范化为 `inactive` 的合法行为。通过 TDD
新增 synthetic regression 并修复独立重建逻辑，提交 `6c41625`。由于
verifier 也受 source identity 绑定，旧 canonical 没有被复用；修复后
完整重跑 final-source v2 五阶段链路。

### Prompt-to-artifact completion audit

1. **Disabled-by-default/fail-closed config**：
   `tinyvllm/config.py` 与 `tools/test_chunked_prefill.py` 覆盖五个字段、
   阈值关系和 mixed/KV-offload 互斥。
2. **Exact controller state machine**：
   `tinyvllm/engine/scheduler.py` 与 dependency-light tests 覆盖
   inactive/active/draining、enter/exit hysteresis、mixed-step cap 和
   decode yield。
3. **Transactional admission**：
   scheduler tests 证明不能保证 decode row 时，不分配 prefill KV、不
   `may_append()`、不转移 queue ownership。
4. **P3/P4 shared execution path**：
   P3 remains diagnostic；P4 是唯一可决定 top-level classification 的
   candidate。
5. **Exact canonical matrix**：
   `run_manifest.json.expected_case_ids` 与 54 个 unique `case_rows`
   一致，覆盖 6 scenarios × 3 policies × 3 repetitions。
6. **Source/environment/workload binding**：
   manifest、preflight、smoke identity 和 independent verifier 使用同一
   三个 SHA-256；final source 是 clean commit `6c41625`。
7. **Remote safety contract**：
   只在指定 `sitian` host/runtime/model 运行；54 个 model process 使用
   108 个互异动态端口；runner tests 禁止 rsync、远端 checkout mutation、
   shared `/tmp` 清理和杀 unrelated processes。
8. **Independent correctness/structure**：
   verifier exit 0，54 cases 全 PASS，
   `correctness_failures=[]`、`structural_failures=[]`。
9. **Independent performance classification**：
   verifier 从 timeline/scheduler/memory traces 重算 P4/P0 paired metrics，
   得到 `NO_GO`、`benefit_path=null` 和三项 tail guards。
10. **Claim and documentation boundary**：
    本节记录负面结果；按计划只在 `GO` 时更新 README，因此 README 未改；
    raw experiment artifacts 保持 untracked。

本轮实现、source-bound 远程验证和审计均已完成，但产品性能目标没有达成：
P4 是经过正确性验证的实验策略，不是可推广的性能优化。

### 下一优化方向

不要继续只调 enter/exit threshold 来包住这个 workload，也不要把 burst
局部吞吐收益外推成 engine-wide speedup。下一轮应另建书面 spec 和 gate，
优先验证 **decode-SLO-aware mixed admission**：

1. admission 前读取 runnable decode 的 token age / recent decode gap；
2. 只要任何 decode row 接近固定 gap budget，就立即 suppress mixed
   admission，而不是等“两 mixed 后一 yield”；
3. 对 short-output service bucket 设置更强保护，并把 long-prompt prefill
   chunk budget 从固定 128 改为受 decode slack 约束；
4. 先用现有 canonical workload 做 deterministic trace replay / counterfactual
   gate，再运行全新 source-bound remote chain；
5. promotion 仍要求 correctness/structure 全过、所有 tail guards ≤1.10，
   且至少出现一个可重复的 throughput、latency 或 memory benefit path。

若该方向仍没有吞吐/显存 benefit，应停止 scheduler-level mixed-prefill
微调，转向更可能产生结构性收益的 kernel/CUDA Graph overhead 或
quantization，并分别建立独立 gate。

## 2026-07-17 Speculation profitability router controlled canonical

### 当前结论

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- controlled canonical：
  `qwen3-06b-router-controlled-canonical-20260717-154410`
- 最终分类：**NO_GO**，不是 `INCOMPLETE`
- 不得宣称引擎整体已提升，也不得执行 real-source gate 来制造产品 `GO`
- 下一主方向：停止 native-verifier/router micro-optimization，切换到
  production batching、kernel/CUDA Graph overhead 或 quantization，
  每条方向都必须另建书面设计与 gate

### 远端环境与证据

```text
host              sitian@10.232.195.203
python            /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model             /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
GPU               5 / NVIDIA A100 80GB PCIe
remote run root   /data00/home/sitian/sitian-workspace01/tllm/speculation-router-runs/qwen3-06b-router-controlled-canonical-20260717-154410
local compact     experiments/speculation_router/qwen3-06b-router-controlled-canonical-20260717-154410
```

完整 `raw/` 与 `logs/` 各约 8.6 GB，总 artifacts 约 18.3 GB，保留远端。
本地只保留 required aggregate artifacts、source snapshot 和独立 verifier
输出；13/13 aggregate artifact SHA-256 与 `artifact_hashes.json` 一致。

### Source identity

```text
base commit       63953089f30d0e9506461a3eb1e44bc9df8d778e
source dirty      false
source tree       a67f8a574e43c88758b517e75f588d94ff647390e33219544f5b45426b5ffcc1
source patch      e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
patch bytes       0
```

### Router semantics

- `K<=1`、finished、output budget exhausted -> baseline
- `K>=2 && native_compatible` -> native multi-token verifier
- incompatible -> fail closed，只有显式 opt-in 才 baseline fallback
- controlled drafts 标记为 `controlled_target_derived`，不能进入
  real-source stage，也不能产生产品 `GO`

### Canonical coverage 与独立审计

```text
cases                         18/18
policies per case             5/5
case rows                     90/90
native events                 35
router rows                   18
unique dynamic port pairs     90/90
duplicate port pairs          0
remote exitcode               0
independent verifier exitcode 0
classification                NO_GO
```

覆盖 K1 fallback、K2/4/8/16 zero/one/partial/full acceptance、EOS、
output-budget、block-boundary 和 multiblock continuation。

独立 verifier 不是复用远端 summary：它从 canonical
`source_snapshot.tar.gz` 解出当时 verifier，重新读取并 canonical-hash
全部 raw JSON，再重算 classification。结果保存在：

```text
experiments/speculation_router/qwen3-06b-router-controlled-canonical-20260717-154410/independent-verify/verify.stdout
experiments/speculation_router/qwen3-06b-router-controlled-canonical-20260717-154410/independent-verify/verify.exitcode
```

最终 flags：

```text
exactness_pass              false
replay_elimination_pass     false
router_isolation_pass       false
performance_direction_pass false
```

NO_GO reasons：

```text
k8-eos-boundary/always_native continuation mismatch
k8-eos-boundary/always_native output token mismatch
k8-eos-boundary/oracle continuation mismatch
k8-eos-boundary/oracle output token mismatch
k8-eos-boundary/routed_native continuation mismatch
k8-eos-boundary/routed_native output token mismatch
```

### Performance decomposition

`routed_native / baseline` elapsed ratio：

```text
K1 fallback median  0.979660
K2 median           1.071625  range 1.055554..1.286760
K4 median           1.058623  range 0.940476..1.165044
K8 median           0.902935  range 0.791938..1.272164
K16 median          1.025948  range 0.751078..1.355439
```

K16 full-accept ratio `0.751078`，即该受控 case elapsed 约低 `24.9%`；
K8 budget-boundary ratio `0.791938`。这些局部收益不能覆盖短 draft、
zero/one acceptance 回归，更不能覆盖 K8 EOS exactness failure，因此不能
外推成引擎整体提升。

### Retry / recovery history

- 首轮 transport 使用 rsync 时出现
  `Connection closed by UNKNOWN port 65535`，改为 tar 原子上传与 8 MiB
  SSH 分块下载。
- preflight 与 launch heredoc 曾被 `ssh -n` 丢弃；分别补 RED tests，
  改用 stdin-capable `SSH_STREAM`。
- canonical 执行中 ControlMaster channel 多次 broken pipe，但远端模型
  进程已 run-local 后台脱离，重建 master 后确认矩阵持续推进，未重跑。
- canonical 完成后合法 0-byte `source.patch` 触发 `.partial` 不存在；
  通过 TDD 修复，并新增 `download-only`，commit：
  `cc91930 fix: recover published router artifacts safely`。
- 一次误用 `RESUME=1` 证明该模式会重新执行而非仅下载；原 canonical
  保存在 run-local `artifacts.previous`，核验 exitcode 0、472 files、
  summary `NO_GO` 后原子恢复；失败 resume 保存在
  `artifacts.failed-resume`。

### Implementation commits

```text
68152c0 feat: add fixed speculation profitability router
4decb20 feat: route short speculative drafts to baseline
ca73ee2 refactor: share source-auditable gate evidence
8b519b2 feat: add controlled speculation router gate
a06bb85 feat: execute routed verifier envelope cases
ded02a8 feat: add source-attributed speculation performance gate
0fb06ae feat: add auditable speculation router remote gate
ebff32f fix: harden speculation router evidence recovery
9c91d7f fix: harden speculation gate remote transport
29932f9 fix: preserve remote preflight stdin
6395308 fix: preserve remote gate launch stdin
cc91930 fix: recover published router artifacts safely
```

### Task 11 conditional gate

Task 11 **跳过且 fail closed**：

- controlled classification 已是 `NO_GO`；
- workspace 没有经单独设计批准的 `draft_source.json` /
  `prompt_bank.json`；
- 没有 source-attributed、non-target-derived、non-debug、checkpoint /
  tokenizer identity 完整的 compatible drafter source。

不得用 debug stub、target-derived draft 或改名的 negative control 运行
real-source smoke。

### Prompt-to-artifact completion checklist

- [x] K<=1 baseline isolation -> `tinyvllm/speculative/router.py`；
  `router_rows.json`；`tools/test_speculation_router.py`
- [x] K>=2 native dispatch -> `event_rows.json`；oracle/always-native/
  routed-native rows；`tools/test_native_verifier_oracle.py`
- [x] zero replay/copy/rematerialization gate -> `event_rows.json` +
  `classify_controlled_gate()`；canonical 结果为 fail，不伪装为 pass
- [x] K 2/4/8/16 acceptance matrix -> `manifest.json.case_matrix` +
  `case_rows.json`，18 cases × 5 policies
- [x] EOS/budget/block/continuation -> `k8-eos-boundary`、
  `k8-budget-boundary`、K4 boundary、K16 multiblock rows + oracle fields
- [x] controlled threshold -> canonical snapshot verifier 独立重算
  `summary.json`，classification `NO_GO`
- [x] source identity -> `source_evidence.json`、`source_preflight.json`、
  `source_snapshot.tar.gz`，13/13 aggregate hashes matched
- [x] dynamic unique ports -> `case_rows.json[].process` independent audit，
  90/90 unique pairs
- [x] real-source-only GO boundary -> controlled target-derived stage cannot
  produce GO；classifier/input-validation tests；Task 11 fail closed
- [x] remote result -> remote full raw verifier exit 0；local
  `summary.json` / `report.md` / `independent-verify/verify.stdout`
- [x] limitations and next direction -> `README.md` 与本节

### Fresh reproduction / verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_speculation_router_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_run_speculation_router_gate_remote.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
python3 -m py_compile \
  tinyvllm/speculative/router.py \
  tools/profile_ngram_commit.py \
  tools/source_audit.py \
  tools/speculation_router_gate.py \
  tools/native_verifier_oracle.py
bash -n tools/run_speculation_router_gate_remote.sh
git diff --check
```

完整 raw verifier 需在远端 run root 上执行；本地 compact artifact 不包含
8.6 GB `raw/`，因此不得把本地缺 raw 的失败误报成 canonical 证据失败。

## 2026-07-17 K1 source-auditable canonical gate

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- 远端：`sitian@10.232.195.203`
- Python：`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- 模型：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- GPU：`CUDA_VISIBLE_DEVICES=7`
- canonical artifact：
  `experiments/adaptive_ngram/20260717-k1-sam-canonical/`
- smoke artifact：
  `experiments/adaptive_ngram/20260717-k1-sam-smoke-r2/`

书面设计与执行计划：

- `docs/superpowers/specs/2026-07-17-adaptive-ngram-sam-source-evidence-design.md`
- `docs/superpowers/plans/2026-07-17-adaptive-ngram-sam-source-evidence.md`

SAM/source evidence commits：

- `efde993 test: add adaptive gate source evidence`
- `f3ccaed feat: verify adaptive gate source artifacts`
- `925e707 feat: stage auditable adaptive gate source`
- `14f2b0b fix: preserve immutable gate source preflight`
- `5963086 fix: detach long adaptive gate runs`
- `198cede fix: allow generated adaptive gate artifacts`

canonical source identity：

```text
base commit       198cede8a3b0d201588ceb547208ada111aa77b7
source dirty      true
source tree       149517ad81bd5c9be96bc03a041a430e9186537134cd502012bafa19fc5bcda6
source patch      2c23549c6e8e875cab0d1b6dc9b79031a22bacdf1da3aeb3ef20a825cbb13392
```

远端 source preflight：

- source snapshot hash verification：`returncode=0`
- remote `tools/test_ngram_speculative.py`：`returncode=0`
- source tree、patch、preflight 已嵌入 manifest，完整 artifact verifier 会从
  base commit + binary patch 独立重建 source。

canonical 完整性审计：

```text
rows                         140/140
unique run keys              140
rows per policy              28
unique port values           280/280
process returncode=0         140/140
profiler gate pass           140/140
exact output mismatches      0
source tree identities       1
correctness pass             true
trajectory replay pass       true
adaptive exercise pass       true
full artifact verifier       pass
```

canonical 最终结论：

```text
decision                     NO_GO
baseline median tok/s        32.914809
fixed K1 median tok/s        32.896915
fixed K1 vs baseline         -0.0544%
adaptive median tok/s        29.087900
adaptive vs baseline         -11.6267%
adaptive vs best fixed       -11.5786%
natural prose ratio          0.958180
transition-heavy ratio       0.845165
```

NO_GO reasons：

```text
adaptive_vs_baseline_gate_failed
adaptive_vs_fixed_gate_failed
natural_or_transition_regression
```

K1 fast path 的边界：

- `tools/profile_ngram_commit.py` 与 `tools/test_ngram_speculative.py` 仍为
  **未提交修改**。
- canonical correctness 通过，但 K1 吞吐中位数没有改善：
  `-0.0544%`，因此不得提交 K1，也不得宣称性能改善。
- 不要删除 canonical/smoke artifacts；它们是本轮 source-auditable 证据。

canonical recovery 记录：

- 首轮确实生成 140 rows，但 repetition 6 有 7 个进程在模型初始化阶段失败：
  `ModelRunner.allocate_kv_cache()` 中 `assert auto_num_blocks > 0`。
- 失败横跨 baseline/fixed/adaptive，且 GPU 7 后续恢复到约 80 GiB free；
  归因为运行期间的瞬时可用显存不足，不是 K1 语义分叉。
- 首轮 snapshot 和失败 run-key 清单保存在：
  `experiments/adaptive_ngram/20260717-k1-sam-canonical/recovery-before-resume-20260717-120956/`
- recovery 使用同一 remote immutable source、manifest、source evidence 和
  preflight，先备份首轮 rows/events/summary/report，再只剔除 7 个失败
  run keys，并以 `--resume` 补跑；最终 remote exitcode 为 `0`。
- 为避免以后手工剔除，gate resume 已改为：只把
  `returncode=0 && profiler_gate_pass=true` 的 row 当作 completed；失败 row
  原位替换、旧失败 events 清除、旧端口仍保留在去重集合。

canonical 关键文件 SHA-256：

```text
manifest.json         8605868bc709304fc188c3c6132977909a90d433d95d5a4203ba25debf9fd03d
raw_rows.json         ee551544348bc667757d9a81f66bfca65331d52938fd0525149f28483e1e759f
event_rows.json       24c089fd7ecdd432c1542a4442f65706865627ce3125f51e19bd1266beaaae79
summary.json          016e62110c2c12ab458d99fb7bbf4c19fcab5587f1e118afc285b8004013cab8
report.md             9aa3b4963528d0e2dc7f2770c2ee3fb18868d4e85803aa830a4c0896edd703d2
source_evidence.json  46b449d85ddfce18db30a0ad0cbfea06be7744ddf6221e34b193eea7e077273b
source.patch          2c23549c6e8e875cab0d1b6dc9b79031a22bacdf1da3aeb3ef20a825cbb13392
source_preflight.json 757ae207108b9b470b0dc5c284aa6fafd0ba7b6e38ebd94e3d110b2f7c8c3d23
```

复验命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/adaptive_ngram_gate.py verify \
  --out-dir experiments/adaptive_ngram/20260717-k1-sam-canonical
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_adaptive_ngram_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
python3 -m py_compile \
  tools/adaptive_ngram_gate.py \
  tools/test_adaptive_ngram_gate.py
bash -n tools/run_adaptive_ngram_gate_remote.sh
git diff --check
```

下一步：

1. 保留本轮 NO_GO，不继续微调 adaptive n-gram 阈值来追逐该 prompt bank。
2. 若继续 speculative 主线，优先评估更高质量 learned drafter 或减少
   target verification/KV materialization 固定开销的结构性方案。
3. 新方案必须另建 source-auditable smoke/canonical gate；不能复用本轮
   NO_GO 作为性能改善证据。

## 2026-07-16 APC prefix-hit-aware admission

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- commits：
  - `36788ad fix(apc): account for live prefix hits in admission`
  - `b148f08 fix(apc): budget normal prefill by cache misses`
  - `9aefb47 fix(apc): batch warm chunked prefills by misses`
  - `8d993d0 fix(apc): evict stale hashes on block reuse`
  - `662b286 fix(apc): preserve duplicate prefix mappings`
  - `14309ba feat(apc): gate warm batch admission`
  - `6632818 fix(apc): isolate cold batch cache state`
  - `be7e697 test(apc): reject invalid batch isolation artifacts`
- 修改：
  - `BlockManager.can_allocate()` 现在只从请求的 free-block 需求中扣除
    完整、token 校验通过且仍在 `used_block_ids` 中的 live prefix blocks。
  - idle cached blocks 仍位于 `free_block_ids`，重新激活会真实消耗 free block，
    因此 admission 不会把 idle hit 错算成额外容量。
  - 匹配遵守 sampleable suffix cap，并在 hash 命中后比较 token ids，避免
    collision 或最后 sample token 被错误复用。
  - `BlockManager.estimate_admission()` 只读返回
    `(reusable_tokens, required_free_blocks)`，normal/chunked scheduler 可用
    同一次 prefix scan 同时做 token budget 与 block capacity admission。
  - normal prefill 的 batch token budget 现在按
    `len(seq) - reusable_tokens` 计算，不再把已命中的完整 prefix 重复计入。
  - chunked prefill 的额外 final-prompt batching 也按 uncached suffix
    判断是否可在一个 chunk 内完成；delayed hash publication 仍保证本批新 KV
    不会形成 same-batch dependency。
  - idle cached block 被 cold miss 覆盖重用前，会条件删除仍指向该 block 的旧
    hash mapping，避免 `hash_to_block_id` 随 churn 积累 stale entries。
    cache-hit reactivation 仍会立即恢复 hash/token metadata 与 mapping。
  - duplicate/collision-safe secondary index 保留同一 hash 的所有物理 blocks；
    primary block 被覆盖时可 O(hash duplicate count) 回退到等价 token block，
    不会因 same-batch duplicate publication 丢失仍有效的 KV。
- TDD 证据：
  - 旧实现下 live-prefix admission 测试精确失败于
    `assert block_manager.can_allocate(warm) is True`。
  - 修复后 live-hit 场景可在 3-block prompt、仅 1 个 free block 时接入，
    实际 `allocate()` 命中 8 cached tokens 并恰好消耗最后 1 个 free block。
  - idle-hit 场景保守拒绝：3-block prompt、2 个 idle cached blocks、仅
    2 个 free blocks 时仍需要 3 个 free-block activations。
  - normal budget 旧实现下，9-token warm prompt 命中 8 tokens 后仍按 9
    tokens 拒绝，最终错误跌入空 decode `assert scheduled_seqs`；修复后合法的
    10-token budget 可一次接入 3 条各只需计算 1 token 的 warm prompts。
  - cold control 保持不变：10-token budget 仍只接入两条 5-token cold prompts。
  - chunked 旧实现下，4-token cold prompt 后的 9-token/8-hit warm prompt
    被 `len(candidate) > chunk_size` 拒绝；修复后两者以 4+1 actual prefill
    tokens 合批，warm prompt 从 token 8 开始计算。
  - stale-index 旧实现下，单 block 发布 prefix A、释放后被 cold prefix B
    覆盖时，`hash(A) -> block_id` 仍残留；修复后覆盖时删除 A，B commit 后
    字典只保留 `hash(B) -> block_id`。
  - duplicate-prefix 旧实现下，两个同 hash/equal-token blocks 中 primary 被
    overwrite 后，整个 prefix mapping 消失；修复后 primary 回退到另一份
    仍有效的 KV。hash collision 下 primary token 不匹配时也会在同 hash
    候选集合中找到 token-equal block。

APC canonical gate 已扩展 warm-batch evidence：

- 新 artifact：`batch_performance_rows.json`。
- 1024/2048 shared-prefix、8-request cold/warm/cache-cleared batch。
- 每条请求共享同一已 seed prefix、使用不同 suffix；warm 必须全部在一个
  model batch 中接入。
- raw rows 记录：
  - full-batch elapsed time（不是单请求 TTFT）；
  - model batch count；
  - total/per-request cached/query tokens；
  - exact token/text 与 full-logit correctness。
- batch `GO` 条件：
  - warm median model batches = 1；
  - 每请求 cached/query accounting 精确；
  - warm full-batch median elapsed 比 cold 至少改善 15%。
- remote runner 会从 `batch_performance_rows.json` 重建 batch summaries 和
  最终 decision，不信任 summary 自报；source manifest 也覆盖 runner 与两份
  CPU tests。
- cold/cache-cleared batch 现在在一次 profiler run 内的相邻 model batches
  之间调用 `clear_reusable_cache()`；否则前一 model batch 刚发布的共享 prefix
  KV 会把后一 model batch 意外预热，使 cold 路径低估 admission 成本。
- warm batch 不执行该隔离清理；它必须保留 producer seed 的 reusable KV，
  并由 scheduler 自然证明 8 条请求可一次接入。
- batch 间 cache clear 的 host wall time 计入 `host_instrumentation_ms`，从
  measured full-batch elapsed 中扣除，避免把 profiler 隔离操作算作模型时间。
- raw row 新增 `cache_isolation_between_batches`；artifact audit 强制
  cold/cache-cleared 为 `true`、warm 为 `false`，因此旧的污染 artifact
  不能通过 canonical 重算。
- `be7e697` 增加显式负例：cold 错标为未隔离、或 warm 错标为隔离时，
  `audit_batch_artifact_payloads()` 都必须拒绝 raw→summary 不一致。

官方设计对照后的边界：

- vLLM APC 与 TensorRT-LLM KV reuse 都把可回收 cache block 的 eviction/
  reuse bookkeeping 视为核心；当前 stale-index 修复属于同一基础卫生层。
- SGLang 的 RadixAttention/cache-aware scheduling 是更大的结构与策略改动。
  当前不要顺手实现 radix tree 或 waiting-queue bypass；queue bypass 会改变
  FIFO、公平性与尾延迟，必须先单独定义 throughput/TTFT/starvation gate。
- 提交后 fresh local verification：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_prefix_cache.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_eval_needle_fixed_prompts.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_oracle.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_native_verifier_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_model_runner_spec_verify.py
python3 -m py_compile \
  tools/profile_prefix_cache.py \
  tools/test_profile_prefix_cache.py \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py
bash -n tools/run_prefix_cache_gate_remote.sh
git diff --check
```

以上命令已在 `6632818` 提交前 fresh 运行并全部通过。尚未提交、仍等待远程
GPU gate 的 K1 改动继续隔离在：

- `tools/profile_ngram_commit.py`
- `tools/test_ngram_speculative.py`

远程状态刷新：

- `klist` 中 `sitian@BYTEDANCE.COM` 的
  `host/jump-proxy-hl.byted.org`、`host/10.232.195.203` ticket 均过期。
- `/tmp/ssh-sitian-10.232.195.203` ControlMaster socket 不存在。
- `ssh -o BatchMode=yes sitian@10.232.195.203 ...` 仍失败：
  `Connection closed by UNKNOWN port 65535`。
- `kinit -R` 无法无交互续期：
  `Matching credential (krbtgt/BYTEDANCE.COM@BYTEDANCE.COM) not found`；
  当前 cache 没有可续期 TGT，需要用户侧重新完成 Kerberos 登录。

认证恢复后的固定顺序：

1. APC smoke：
   `TAG=20260716-smoke REPETITIONS=1 WARMUP_REPETITIONS=1 tools/run_prefix_cache_gate_remote.sh`
2. smoke artifact audit 全部通过后再跑 APC canonical：
   `TAG=20260716 REPETITIONS=7 WARMUP_REPETITIONS=2 tools/run_prefix_cache_gate_remote.sh`
3. 继续执行 K1 canonical gate；在此之前不得提交 K1 fast path 或声称性能达标。
4. 没有 canonical APC GPU `summary.json` 前，不修改根 README 的最终
   GO/NO_GO checkbox，也不填写最终 APC 指标。
5. smoke/canonical 现在必须同时存在并通过审计：
   `performance_rows.json` 与 `batch_performance_rows.json`。

## 2026-07-15 SAM drafter gate 最终状态

- 工作目录：`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`
- 分支：`feat/adaptive-ngram-speculation`
- correctness 修复 commit：`b71e7ce`；后续还有文档/证据提交，以最终
  `git rev-parse HEAD` 为准。
- 远端：`sitian@10.232.195.203`
- Python：`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- 模型：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- 有效 smoke 使用 GPU 5；GPU 7 在一次末尾进程中因可用显存不足出现
  `auto_num_blocks > 0` assertion。
- 固定 seed：`20260715`
- 有效 smoke：25 rows，5 prompts × 5 policies × 1 repetition。
- 修复后本地五文件：
  `experiments/sam_drafter/qwen3-06b-sam-remat-smoke-20260715-162323/`
- 修复后远端目录：
  `/data00/home/sitian/sitian-workspace01/tllm/sam-drafter-gates/qwen3-06b-sam-remat-smoke-20260715-162323`

严格 verifier 结果：

```text
decision=NO_GO
observed_rows=25
correctness_pass=true
trace_reconciliation_pass=true
policy_exercise_pass=true
```

诊断证据：

1. 两次独立 `baseline-only` 重跑在 `natural_prose` 和
   `structured_code_like` 上 token-for-token 完全一致，排除跨进程 greedy
   随机分岔。
2. 同一共享 verifier 使用 fixed n-gram `K=1` 时，两类 prompt 均与 baseline
   完全一致。
3. `K>1` 时会走 batch tail prefill/KV materialization，旧实现随后稳定分叉。
4. SAM index 的最终 token count 实际正确；早期 `INCOMPLETE` 中的
   `index_integrity_token_count_mismatch` 是 event 去重保留旧值，已由
   commit `3f61619` 和回归测试修复。
5. 修复在 acceptance 判定后使用正常 decode kernel 重物化
   `accepted_tokens[:-1]` 的 KV；两-prompt A/B 和完整 25-row smoke exactness
   均通过。
6. 该 correctness fallback 很昂贵：`sam_fixed_k16` 300 次 decode calls、
   `sam_match_aware` 288 次 decode calls；两者重物化累计约 `18.03s`。

五文件 SHA-256：

```text
manifest.json   0aaf027ac3045648fd9fa92a5738edb03b27b746dd390a9c5b2d67f08662eecc
raw_rows.json   fdf0fdf94807a5bffe11065b2b90a130935af3ca42d0bc0be7663aa1f2d455e8
event_rows.json ee125f0b4258c9050905c2c00c61c22fb56ea461b6bb68bce85fcd68ae272d5c
summary.json    913d195d75aa0f448a19d1ec50469c5a3c24b619501a93b85dbe9e33dbaaf483
report.md       a4b0bb681383c4696e82378906e89c8dceeb4e4394449a817d39e6ce30f9feb6
```

关键性能结果：

```text
baseline median tok/s          32.5173
ngram fixed K4                 25.9687
ngram adaptive                 25.7055
SAM fixed K16                  28.3399
SAM match-aware                28.6274
SAM vs baseline                -10.72%
SAM vs ngram K4                +8.44%
verify-attempt reduction       25.00%
draft-waste reduction          -218.18%
```

最终 `NO_GO` 原因：

```text
sam_vs_baseline_gate_failed
critical_prompt_regression:natural_prose
critical_prompt_regression:structured_code_like
critical_prompt_regression:repeated_long_context
critical_prompt_regression:prompt_copy_retrieval
```

复现与验证：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
bash -n tools/run_sam_drafter_gate_remote.sh
python3 tools/sam_drafter_gate.py verify \
  --out-dir experiments/sam_drafter/qwen3-06b-sam-remat-smoke-20260715-162323
```

下一 TODO：

1. 不要继续调 SAM match-aware 阈值或 prompt bank；当前严格结果是 NO_GO。
2. 若继续 speculative 主线，优先让 batch target verifier 原生写出与正常
   decode 等价的 KV，避免逐 token重物化。
3. 或等待兼容的 learned drafter/checkpoint，再重新设计独立 gate。
4. 175-row canonical 未运行：25-row smoke 已在所有关键 prompt class 上大幅
   失败 baseline gate，继续运行不具成本效益。

## 必须优先使用远程环境

- 远程机器：`sitian@10.232.195.203`
- 远程项目目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`
- 远程 Python：`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- Python 版本：`Python 3.11.15`
- Torch：`2.4.1+cu121`
- CUDA：远程 Python 下 `torch.cuda.is_available() == True`
- 默认 GPU：`CUDA_VISIBLE_DEVICES=7`
- Conda/Miniforge：`/data00/home/sitian/sitian-workspace01/tllm/miniforge/bin/conda`
- Conda envs 中包含：
  - `/data00/home/sitian/sitian-workspace01/tllm/env`
  - `/data00/home/sitian/sitian-workspace01/tllm/miniforge`（base）
  - `/data00/home/sitian/miniconda3`
  - `/data00/home/sitian/miniconda3/envs/py311`

本地桌面项目入口是 `/Users/bytedance/Desktop/TinyLLMForge`（用户口径：Tiny LLM Forge）。这个路径是后续 agent 应优先读取/进入的本地项目路径；涉及 GPU、torch CUDA、profile、needle、Qwen 模型时仍必须通过 SSH 在远程跑，不要用本地 `/opt/homebrew/bin/python3.12` 代替。

## 本地路径

- 本地桌面项目入口：`/Users/bytedance/Desktop/TinyLLMForge`（用户口径：Tiny LLM Forge）
- 实际 symlink 指向：`/Users/bytedance/dev/TinyLLMForge`
- 后续 agent 本地读文件时优先用桌面路径：`/Users/bytedance/Desktop/TinyLLMForge/AGENT_HANDOFF_STATE.md`
- 本地分支：`feat/kv-sparse-attention`（本地 git 仓库）
- 远程项目目录当前看起来不是 git worktree（`git status` 报 not a git repository），更像同步后的运行目录。

## 常用远程命令模板

```bash
ssh sitian@10.232.195.203 'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && CUDA_VISIBLE_DEVICES=7 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python <script> <args>'
```

验证远程 Python/CUDA：

```bash
ssh sitian@10.232.195.203 'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && /data00/home/sitian/sitian-workspace01/tllm/env/bin/python - <<"PY"
import torch, sys
print(sys.executable)
print(torch.__version__, torch.cuda.is_available())
PY'
```

## 模型和重要产物路径

- Qwen3-8B：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-8B`
- Qwen3-0.6B：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B`
- SQ scale 常用：
  - `/tmp/sq_scales_qwen3_8b_a0.85.pt`
  - `/tmp/sq_scales_qwen3_8b_layer_adaptive_floor085.pt`
- 远程 profile 输出目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/profile_out`
- 远程 needle 输出目录：`/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/needle_sq_results`

## 当前任务主线

最近主线是 TinyLLMForge 的 Qwen3/long-context runtime 实验，尤其：

1. n-gram speculative decoding 原型：S1 online dry-run、S2 target verification、S3 accepted-token KV commit、S4 online benchmark。
2. mixed prefill admission policy：短 prompt 不进入 mixed prefill+decode batch，保护 decode latency。
3. hidden-state / latent projector 相关实验已经有大量脚本和结果，但当前更像封档/实验记录，不是 runtime 主线。

本地最近相关文件：

- `tinyvllm/speculative/ngram.py`
- `tinyvllm/engine/block_manager.py`
- `tinyvllm/engine/llm_engine.py`
- `tinyvllm/engine/scheduler.py`
- `tools/profile_ngram_online.py`
- `tools/profile_ngram_verify.py`
- `tools/profile_ngram_commit.py`
- `tools/test_ngram_speculative.py`
- `tools/test_chunked_prefill.py`
- `tools/profile_chunked_prefill.py`
- `tools/train_latent_projector.py`
- `docs/qwen3-8b-fixes.md`

## 当前已做的本地修复

在本地 `/Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py` 已修两个边界：

1. `_target_verify_and_commit()` 中 `commit_accepted_tokens()` 接管/释放 reserved blocks 后，把本地 `reserved_blocks` 清空，避免异常路径 double release。
2. `run_paired_profile()` 和 `run_candidate_only_profile()` 中，commit 后如果 `llm.is_finished()`，直接 break，避免所有请求已结束后继续 `llm.step()` 触发 scheduler 空队列 assert。

本地已通过：

```bash
git -C /Users/bytedance/dev/TinyLLMForge diff --check
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
```

注意：这些是本地 CPU/语法检查。GPU/profile/模型相关验证必须在远程跑。

## 建议下一步

1. 后续 agent 先读本文件：`/Users/bytedance/Desktop/TinyLLMForge/AGENT_HANDOFF_STATE.md`。
2. 如果要继续 S4 n-gram speculative，先确认本地改动是否已同步到远程运行目录。
3. 在远程用 Qwen3-0.6B 先跑最小 candidate-only / baseline-only smoke，再跑 8B。
4. 不要把 `.agents/`、`.codex/`、`.pt` checkpoint 这类本地配置/产物默认提交。
5. 然后读取 `docs/qwen3-8b-fixes.md` 最新章节。

## 2026-06-29 远程 S4 smoke 结果

已把本地 `tools/profile_ngram_commit.py` 的边界修复同步到远程运行目录，并在远程通过：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py --help >/dev/null
```

Qwen3-0.6B / `max_output_len=32` / `n5d4` smoke：

- candidate-only JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_smoke_20260629.json`
- candidate-only log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_smoke_20260629.log`
- baseline-only JSON：`profile_out/ngram_spec_s4_06b_baseline_n5d4_smoke_20260629.json`
- baseline-only log：`profile_out/ngram_spec_s4_06b_baseline_n5d4_smoke_20260629.log`

结果摘要：

| mode | gate_pass | output_tokens | elapsed_s | tok/s | decode_steps | commit_events | accepted_count | drafted_tokens | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate-only | true | 32 | 55.705 | 0.574 | 22 | 3 | 9 | 16 | 0.5625 |
| baseline-only | true | 32 | 57.078 | 0.561 | 31 | - | - | - | - |

纯 decode step 汇总：candidate 22 step / 22.875s，baseline 31 step / 24.044s。candidate 确实少跑 9 个 decode step，但单 step 平均更重；这个 smoke 只说明修复后正确性 gate 通过，性能结论还需要更长输出、多 prompt、热身后再判断。

## 2026-06-29 warmup 后的 S4 smoke 结果

结论：之前 55s 主要不是 decode 慢，而是冷启动/首次 shape 编译计入了测量。典型冷启动：prefill 约 32-33s，首个 decode 约 22s；warmup 后测量段恢复到约 1.1s。

已给 `tools/profile_ngram_commit.py` 增加参数：

```bash
--warmup-output-len N
```

它会在正式计时前先用同一个 LLM 跑一个 untimed warmup request，避免把 CUDA / kernel / symbolic shape setup 计入 profile 时间。远程已同步并通过 `py_compile` / `--help`。

Qwen3-0.6B / `max_output_len=32` / `warmup_output_len=4` / `n5d4`：

| mode | gate_pass | warmup_s | measured_elapsed_s | tok/s | decode_steps | commit_events | accepted_count | drafted_tokens | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 55.045 | 1.171 | 27.33 | 31 | - | - | - | - |
| candidate-only | true | 56.438 | 1.117 | 28.65 | 22 | 3 | 9 | 16 | 0.5625 |

纯 decode 段：baseline 31 step / 1127ms，candidate 22 step / 833ms。candidate 少 9 个 decode step，warmup 后测得约 1.05x wall-clock speedup；样本太短，只能作为正确性/趋势 smoke。

注意：不要在同一默认 `TINYVLLM_DIST_PORT=2333` 下并发跑多个 TinyLLM 进程，否则会报 `EADDRINUSE`。如必须并发，显式设置不同 `TINYVLLM_DIST_PORT`，但同一 GPU 并发会污染性能结果。

## 2026-06-29 target verify 分段计时

已给 `tools/profile_ngram_commit.py` 的 `_target_verify_and_commit()` 增加分段计时，并写入每个 commit event 的 `timing_ms`，summary 里聚合为 `verify_timing_ms`。字段包括：

- `reserve_blocks_ms`
- `verify_prepare_ms`
- `target_forward_ms`
- `accept_sample_ms`
- `commit_metadata_ms`
- `finish_check_ms`
- `verify_commit_total_ms`

远程已同步并通过 `py_compile` / `--help`。计时 smoke 输出：

- JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_timing_smoke_20260629.json`
- log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_timing_smoke_20260629.log`

Qwen3-0.6B / `max_output_len=32` / `warmup_output_len=4` / `n5d4` / candidate-only：

| metric | value |
|---|---:|
| gate_pass | true |
| elapsed_s | 1.062 |
| output_tokens_per_s | 30.13 |
| decode_steps | 22 |
| commit_attempts | 4 |
| commit_events | 3 |
| accepted_count | 9 |
| drafted_tokens | 16 |
| acceptance_rate | 0.5625 |

聚合 verify timing（包含 4 次 attempt，其中 1 次 zero-accept）：

| timing | ms |
|---|---:|
| reserve_blocks_ms | 0.025 |
| verify_prepare_ms | 1.848 |
| target_forward_ms | 242.415 |
| accept_sample_ms | 1.831 |
| commit_metadata_ms | 0.064 |
| finish_check_ms | 0.031 |
| verify_commit_total_ms | 246.250 |

接受事件每次约 40ms target forward；metadata commit 基本可忽略。当前主要额外成本在 target forward verify，而不是 block reservation / commit metadata。

## 2026-06-29 长上下文下验证 KV 读取摊薄

按“重点看 target verify forward 是否能在长 context/cache pressure 下摊薄 KV 读取”的口径，跑了 3593-token prompt（`alpha beta gamma ...` 重复 512 次），`max_model_len=8192`，`max_output_len=64`，`warmup_output_len=4`，Qwen3-0.6B，`n5d4`。

输出文件：

- baseline JSON：`profile_out/ngram_spec_s4_06b_baseline_n5d4_long3593_20260629.json`
- baseline log：`profile_out/ngram_spec_s4_06b_baseline_n5d4_long3593_20260629.log`
- candidate JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_long3593_20260629.json`
- candidate log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_long3593_20260629.log`

结果：

| mode | gate_pass | output_tokens | elapsed_s | tok/s | decode_steps | accepted | drafted | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 64 | 2.531 | 25.29 | 63 | - | - | - |
| candidate-only | true | 64 | 1.266 | 50.54 | 16 | 47 | 52 | 0.9038 |

分解：

| mode | prefill_ms | decode_steps | decode_ms | avg_decode_ms | verify_target_forward_ms | verify_total_ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline-only | 179.5 | 63 | 2350.8 | 37.3 | - | - |
| candidate-only | 69.2 | 16 | 643.9 | 40.2 | 531.7 | 542.1 |

关键结论：

- candidate 把 decode step 从 63 降到 16，接受了 47 个 token。
- target verify forward 总计 531.7ms，验证 52 个 draft token，平均约 10.2ms/draft token；baseline decode 平均约 37.3ms/token。
- 在这种长上下文、重复 prompt、高 acceptance 的场景下，target verify 基本表现为“一次约 40ms verify 4 个 token”，确实摊薄了长 prefix KV 读取/attention 成本。
- 这还不是 CPU→GPU KV offload/upload 压力测试；它验证的是 GPU KV 长上下文读取摊薄。若要验证“内存 upload”收益，需要再构造 KV offload/page migration 场景。

## 2026-06-29 模拟 CPU→GPU KV upload pressure

当前 TinyLLMForge 没有真正的 KV CPU offload/page migration；已有 `cpu_offload` 是 decoder layer weight offload，不是 KV offload。因此先给 `tools/profile_ngram_commit.py` 增加了一个可控模拟参数：

```bash
--simulate-kv-upload-mb 128
```

含义：每次 baseline decode step、candidate normal decode step、candidate target verify 前，额外做一次 pinned CPU → GPU copy，用来模拟每轮需要从 CPU/host memory upload KV pages 的成本。脚本会先做一次 upload warmup，避免首次 pin/copy 初始化污染计时。远程已同步并通过 `py_compile` / `--help`。

测试口径：Qwen3-0.6B，3593-token repeated prompt，`max_model_len=8192`，`max_output_len=64`，`warmup_output_len=4`，`n5d4`，`simulate_kv_upload_mb=128`。

输出：

- baseline JSON：`profile_out/ngram_spec_s4_06b_baseline_n5d4_upload128_long3593_20260629.json`
- baseline log：`profile_out/ngram_spec_s4_06b_baseline_n5d4_upload128_long3593_20260629.log`
- candidate JSON：`profile_out/ngram_spec_s4_06b_candidate_n5d4_upload128_long3593_20260629.json`
- candidate log：`profile_out/ngram_spec_s4_06b_candidate_n5d4_upload128_long3593_20260629.log`

结果：

| mode | gate_pass | output_tokens | elapsed_s | tok/s | decode_steps | commit_events | accepted | drafted | acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 64 | 2.721 | 23.52 | 63 | - | - | - | - |
| candidate-only | true | 64 | 1.334 | 47.97 | 16 | 13 | 47 | 52 | 0.9038 |

upload 模拟成本分解：

| mode | normal decode upload ms | verify upload ms | total simulated upload ms |
|---|---:|---:|---:|
| baseline-only | 334.5 | - | 334.5 |
| candidate-only | 84.6 | 68.9 | 153.5 |

关键结论：

- baseline 需要 63 次 decode upload；candidate 只需要 16 次 normal decode upload + 13 次 verify upload。
- 在每轮模拟 128MiB H2D upload 时，candidate 的 simulated upload 总成本从 334.5ms 降到 153.5ms，约减少 54%。
- 总体 wall-clock：baseline 2.721s，candidate 1.334s，约 2.04x。
- 这支持“draft + target verify 可以摊薄 KV upload/page migration 成本”的方向，但注意这仍是 copy 模拟；真正 KV offload 需要实现按 block/page 的 CPU resident KV + prefetch/upload + GPU staging cache。

后续继续推进时，已给 `tools/profile_ngram_commit.py` 的 JSON `summary` 补充模拟 upload 聚合字段，避免每次手动从 `step_records` 和 `verify_timing_ms` 里加总：

- `simulate_kv_upload_mb`
- `normal_decode_simulated_kv_upload_ms`
- `verify_simulated_kv_upload_ms`
- `total_simulated_kv_upload_ms`
- `normal_decode_simulated_kv_upload_events`
- `verify_simulated_kv_upload_events`
- `total_simulated_kv_upload_events`
- `normal_decode_simulated_kv_upload_mib`
- `verify_simulated_kv_upload_mib`
- `total_simulated_kv_upload_mib`

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
```

远程已同步并通过：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py --help >/dev/null
```

## 2026-06-30 KV offload MVP-0

按“先做真实 KV offload 的最小闭环，不直接上完整 page migration”的口径，已增加默认关闭的 MVP-0：

- `Config.kv_offload_mvp0`
- `Config.kv_offload_gpu_blocks`
- `Config.kv_offload_logical_blocks`
- `tools/profile_ngram_commit.py` 参数：
  - `--kv-offload-mvp0`
  - `--kv-offload-gpu-blocks N`
  - `--kv-offload-logical-blocks N`

实现范围：

- 仅支持 fp16/bf16 KV（`kv_quant_bits == 0`）。
- 仅支持 full attention；启用时会拒绝 Quest / KV-Cartridge / AM compact。
- 强制走 eager decode / 跳过 cuda graph。
- `seq.block_table` 保持 logical block id。
- `ModelRunner` 内新增 `KVOffloadMVP0`：
  - GPU KV cache 按 `kv_offload_gpu_blocks` 分配为 staging slots。
  - CPU pinned KV backing store 按 `kv_offload_logical_blocks` 分配。
  - `prepare_prefill()` / `prepare_decode()` 在上传 metadata 前做 logical block id -> physical GPU slot id 翻译。
  - 每次 forward 后把写过的 logical block 从 GPU slot writeback 到 CPU pinned store。
  - `tools/profile_ngram_commit.py` 的 target verify 手工 context 路径也接入了同一套 remap/writeback。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
```

远程已同步并通过：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile \
  tinyvllm/config.py tinyvllm/engine/model_runner.py tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
```

远程 Qwen3-0.6B smoke：

- baseline JSON：`profile_out/ngram_spec_s4_06b_baseline_kvoffload_mvp0_smoke_20260630.json`
- candidate JSON：`profile_out/ngram_spec_s4_06b_candidate_kvoffload_mvp0_smoke_20260630.json`

baseline-only 参数：`max_output_len=8`，`max_model_len=2048`，`kv_offload_gpu_blocks=16`，`kv_offload_logical_blocks=32`。

| mode | gate_pass | output_tokens | decode_steps | kv d2h copies | kv d2h MB | kv h2d copies |
|---|---:|---:|---:|---:|---:|---:|
| baseline-only | true | 8 | 7 | 8 | 234.9 | 0 |

candidate-only 参数：`max_output_len=16`，`ngram_size=3`，`max_draft_tokens=4`，`max_commit_events=0`，`max_model_len=2048`，`kv_offload_gpu_blocks=16`，`kv_offload_logical_blocks=32`。

| mode | gate_pass | output_tokens | decode_steps | commit_events | accepted | drafted | kv d2h copies | kv d2h MB | kv h2d copies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate-only | true | 16 | 13 | 1 | 2 | 4 | 15 | 440.4 | 0 |

注意：这只是 MVP-0 正确性闭环。由于 smoke prompt 只占 1 个 KV block，未触发 eviction/H2D reload，所以 `h2d_copies=0` 是预期现象。下一步若要验证 page migration，需要构造“active full-attention blocks <= gpu staging slots，但跨请求/跨阶段会驱逐再读回”的用例，或进入 MVP-1 做 prefetch/eviction policy。

## 2026-06-30 KV offload MVP-0 page migration smoke

为了先验证真实 page migration 语义，而不是直接改复杂 scheduler，已给 `tools/profile_ngram_commit.py` 增加 synthetic smoke：

```bash
--kv-offload-migration-smoke
```

这个 smoke 不加载模型，直接构造一个小型 `KVOffloadMVP0`：

1. GPU staging slots = 2，logical blocks = 4。
2. 写 logical block 0/1 并 D2H writeback 到 CPU pinned backing store。
3. 访问 logical block 2/3，驱逐 0/1，并写回 2/3。
4. 再访问 logical block 0/1，触发从 CPU pinned backing store H2D reload。
5. 校验 reload 后 GPU slot 内容和原始写入内容一致。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
```

远程已同步并通过静态检查。远程 smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --kv-offload-migration-smoke \
  --out-json profile_out/kv_offload_mvp0_page_migration_smoke_20260630.json
```

结果：

| metric | value |
|---|---:|
| gate_pass | true |
| evictions | 4 |
| h2d_copies | 2 |
| d2h_copies | 4 |
| h2d_bytes | 64 |
| d2h_bytes | 128 |
| resident_blocks | 2 |
| dirty_blocks | 0 |

输出文件：`profile_out/kv_offload_mvp0_page_migration_smoke_20260630.json`

结论：MVP-0 的核心 page migration 语义已打通：GPU staging slot 可驱逐，dirty logical block 可写回 CPU pinned store，之后可 H2D reload，并且 reload 内容校验通过。下一步如果继续做 MVP-1，应把这个同步迁移原语升级为：prefetch plan、LRU/成本感知 eviction、异步 copy stream/event、批量 block copy，以及 profiler 中的真实模型长上下文 thrash 场景。

## 2026-06-30 KV offload MVP-1

在 MVP-0 基础上做了保守的 MVP-1 增强，仍保持默认关闭，不改 scheduler / attention kernel / BlockManager 语义：

- `Config` 新增：
  - `kv_offload_async_copy=True`
  - `kv_offload_batch_copy=True`
  - `kv_offload_writeback_on_evict=False`
  - `kv_offload_evict_policy="lru_cost"`
- `KVOffloadMVP0` 增强：
  - 独立 CUDA copy stream + event。
  - H2D/D2H 不再每块全局 `torch.cuda.synchronize()`；forward 前只等待 required H2D events。
  - 连续 `(logical_block, gpu_slot)` span 合并成批量 copy。
  - LRU + dirty/future reuse penalty 的成本感知 victim 选择。
  - stats 增加：`copy_waits`、`h2d_batches`、`d2h_batches`、`h2d_batch_spans`、`d2h_batch_spans`、`evict_clean`、`evict_dirty`、`prefetch_plans`、`prefetch_read_blocks`、`prefetch_write_blocks`。
- `ModelRunner.prepare_prefill()` / `prepare_decode()` 增加显式 prefetch plan 统计，仍把 attention metadata 翻译成 physical GPU slot id。
- `tools/profile_ngram_commit.py` 的 target verify 手工 context 路径同步接入 `_kv_offload_before_forward()` 和 async writeback。
- 新增 synthetic thrash smoke：
  - `--kv-offload-thrash-smoke`
  - `--thrash-gpu-blocks`
  - `--thrash-logical-blocks`
  - `--thrash-window-blocks`
  - `--thrash-rounds`
  - 以及调试开关 `--kv-offload-no-async-copy`、`--kv-offload-no-batch-copy`、`--kv-offload-writeback-on-evict`、`--kv-offload-evict-policy {lru,lru_cost}`。

本地已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/config.py \
  /Users/bytedance/dev/TinyLLMForge/tinyvllm/engine/model_runner.py \
  /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/test_chunked_prefill.py
```

远程已同步并通过同样的静态检查和纯 Python 回归。

### MVP-1 migration 回归 smoke

输出：`profile_out/kv_offload_mvp1_migration_regression_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| evictions | 4 |
| h2d_copies | 2 |
| d2h_copies | 4 |
| h2d_batches | 1 |
| d2h_batches | 2 |
| h2d_batch_spans | 1 |
| d2h_batch_spans | 2 |
| copy_waits | 6 |

### MVP-1 synthetic thrash smoke

命令口径：`thrash_gpu_blocks=3`，`thrash_logical_blocks=8`，`thrash_window_blocks=2`，`thrash_rounds=24`。

输出：`profile_out/kv_offload_mvp1_thrash_smoke_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| evictions | 53 |
| h2d_copies | 48 |
| d2h_copies | 8 |
| h2d_batches | 24 |
| d2h_batches | 3 |
| h2d_batch_spans | 32 |
| d2h_batch_spans | 3 |
| prefetch_plans | 24 |
| prefetch_read_blocks | 48 |

### 真实模型 two-request thrash smoke

短 prompt 两请求，`max_num_seqs=1`，`kv_offload_gpu_blocks=1`，用于验证真实 LLM 路径中两个 request 交替 decode 触发 staging reload。

输出：`profile_out/kv_offload_mvp1_real_two_request_thrash_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 8 |
| evictions | 3 |
| h2d_copies | 2 |
| d2h_copies | 8 |
| h2d_batches | 2 |
| d2h_batches | 8 |
| prefetch_plans | 8 |

### 真实模型长上下文 thrash smoke

两条约 385-token prompt，`max_num_seqs=1`，`kv_offload_gpu_blocks=2`，`kv_offload_logical_blocks=16`。每条 request 当前 full-attention 可见 blocks <= staging slots，但两个 request 交替时会驱逐/重载。

输出：`profile_out/kv_offload_mvp1_real_longctx_thrash_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| output_tokens | 4 |
| evictions | 6 |
| h2d_copies | 4 |
| d2h_copies | 6 |
| h2d_batches | 2 |
| d2h_batches | 4 |
| prefetch_plans | 4 |
| prefetch_read_blocks | 4 |
| prefetch_write_blocks | 6 |

结论：MVP-1 的 prefetch plan / cost-aware LRU / async stream-event / batch copy / synthetic 和真实模型 thrash 场景都已打通。仍然保留 full attention 限制：单次 forward 的可见 logical blocks 必须 `<= kv_offload_gpu_blocks`；要支持单条超长上下文超过 staging slots，需要下一阶段做 streaming/blockwise attention，而不是仅靠 page migration。

### 2026-07-08 KV offload dirty eviction D2H batching

本轮继续性能主线，把 `ensure_resident()` 同一轮加载多个新 logical blocks 时触发的 dirty evictions 从逐 slot D2H 改成 deferred batched D2H：

- `_evict_slot(..., defer_dirty_writeback=True)` 只选择 victim 并统计 eviction，不立即 copy/wait/清映射。
- `ensure_resident()` 收集本轮被驱逐的 dirty `(logical_block, slot)`，统一调用一次 `_enqueue_d2h_pairs()`；连续 logical/slot pair 仍由 `_coalesce_copy_pairs()` 合并 span。
- clean eviction 如果已有 pending D2H event，也会在复用 slot 前等待，避免复用还在写回的 GPU slot。

新增 `tools/test_kv_offload.py`：

- `test_dirty_evictions_are_batched_when_loading_multiple_blocks()`：2 个 dirty resident blocks 被同一轮 `[2,3]` 加载驱逐时，要求 `d2h_copies=2`、`d2h_batches=1`、`d2h_batch_spans=1`。
- `test_deferred_clean_eviction_waits_for_pending_d2h_event()`：先手动 `writeback_dirty([0,1])`，再驱逐 clean blocks，要求 `copy_waits` 至少增加 2，证明复用 slot 前等待 pending D2H event。

验证：

- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4：`CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_kv_offload.py` 通过。
- 远程 migration smoke：`profile_out/kv_offload_batched_dirty_evict_migration_20260708_r2.json`，`gate_pass=true`、`h2d_copies=2`、`d2h_copies=4`、`h2d_batches=1`、`d2h_batches=2`、`d2h_batch_spans=2`、`copy_waits=6`。
- 远程 thrash smoke：`profile_out/kv_offload_batched_dirty_evict_thrash_20260708_r2.json`，`gate_pass=true`、`h2d_copies=8`、`d2h_copies=6`、`h2d_batches=4`、`d2h_batches=2`、`d2h_batch_spans=2`、`prefetch_plans=4`。

### 2026-07-08 KV offload copy event wait coalescing

继续减少 copy/wait 调度开销：batched H2D/D2H 会给多个 logical blocks 记录同一个 CUDA event，旧 `wait_for_blocks()` 和 deferred eviction wait 会对同一个 event 重复 `wait_event()` 并重复累计 `copy_waits`。

- `wait_for_blocks()` 现在按 `id(event)` 去重，同一个 H2D event 只 wait/统计一次。
- `ensure_resident()` 的 deferred D2H wait 同样按 event 去重；clean eviction 仍会在复用 slot 前等待 pending D2H，但同一 batched D2H event 只等待一次。
- 新增测试：
  - `test_wait_for_blocks_coalesces_identical_h2d_events()`
  - `test_deferred_eviction_waits_once_per_identical_d2h_event()`
- 本地：`py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4：`tools/test_kv_offload.py` 通过。
- 远程 migration smoke：`profile_out/kv_offload_wait_coalesce_migration_20260708.json`，`gate_pass=true`、`copy_waits=4`（上一轮同口径 dirty eviction batching 后为 6）。
- 远程 thrash smoke：`profile_out/kv_offload_wait_coalesce_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`（上一轮同口径为 19）、`h2d_batches=4`、`d2h_batches=2`、`prefetch_plans=4`。

### 2026-07-08 KV offload pending wait cleanup

继续消除重复 wait：`ensure_resident(..., wait=True)` 已同步等待本轮 H2D blocks，但旧逻辑仍把这些 logical blocks 留在 `pending_wait_blocks`，后续 `wait_for_pending()` 会再次等待同一批 H2D event。

- `ensure_resident(wait=True)` 现在在 `wait_for_blocks()` 后从 `pending_wait_blocks` 移除已等待的 H2D blocks。
- 新增 `test_ensure_resident_wait_clears_pending_h2d_waits()`：要求 wait=True 后 `{0,1}` 不再残留 pending，并且后续 `wait_for_pending()` 不增加 `copy_waits`。
- 本地 `py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4 `tools/test_kv_offload.py` 通过。
- 远程 migration smoke：`profile_out/kv_offload_pending_wait_clear_migration_20260708.json`，`gate_pass=true`、`copy_waits=4`。
- 远程 thrash smoke：`profile_out/kv_offload_pending_wait_clear_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`、`h2d_batches=4`、`d2h_batches=2`、`prefetch_plans=4`。

### 2026-07-08 KV offload pending H2D anti-thrash

继续降低 clean eviction 抖动：batched H2D 发起后，logical block 会留在 `pending_wait_blocks`。如果该 block 还没被 forward 等待/消费，就被 victim policy 选中驱逐，会浪费一次 in-flight H2D copy。`lru_cost` 的 victim score 现在对 `logical_block in pending_wait_blocks` 增加 `pending_h2d_penalty = block_nbytes * 6.0`，在有其他候选时避免驱逐刚发起 H2D 的 block。

- 新增 `test_evict_policy_avoids_pending_h2d_block_when_possible()`，手动构造 pending H2D block 是最老 slot 的场景，要求仍保留 pending block、优先驱逐非 pending clean block。
- 本地 `py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4 `tools/test_kv_offload.py` 通过。
- 远程 thrash smoke：`profile_out/kv_offload_pending_h2d_penalty_thrash_20260708.json`，`gate_pass=true`、`copy_waits=10`、`h2d_copies=8`、`d2h_copies=6`、`evictions=11`、`prefetch_plans=4`。

### 2026-07-08 Blockwise attention window-specific H2D waits

继续降低 blockwise decode/prefill 的窗口级同步面：旧 `_blockwise_online_decode_attention()` 和 `_blockwise_online_prefill_attention()` 每个 read window 都调用 `manager.wait_for_pending()`，会 drain 全局 `pending_wait_blocks`。如果 prepare 阶段或其他窗口已经发起了尚未被当前窗口消费的 H2D，这会提前等待无关 copy，削弱 copy/compute overlap。

- `KVOffloadMVP0.wait_for_blocks()` 新增 `clear_pending=False` 参数；调用方可只等待指定 logical blocks，并只从 `pending_wait_blocks` 移除这些 blocks。
- `wait_for_pending()` 仍保留 drain-all 语义，内部复用 `wait_for_blocks(..., clear_pending=True)`。
- blockwise decode/prefill read window 现在改成 `manager.wait_for_blocks(list(unique_blocks), clear_pending=True)`，只等待当前窗口真正要读的 H2D。
- 新增测试：
  - `test_wait_for_blocks_clear_pending_api_without_cuda()`：用 fake stream 覆盖 API 语义，证明只清理请求的 pending blocks。
  - `test_wait_for_blocks_can_clear_only_requested_pending_h2d_waits()`：CUDA 场景下要求 `[0]` 被等待/清理后，未请求的 pending block `[1]` 仍保留。
- TDD RED：远程临时回退旧 `wait_for_blocks(self, logical_blocks)` 签名后，新增测试按预期失败：`TypeError: KVOffloadMVP0.wait_for_blocks() got an unexpected keyword argument 'clear_pending'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/test_kv_offload.py`、`git diff --check` 通过。
- 远程 GPU4：`CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_window_wait_clear_20260708.json`，`gate_pass=true`、`chunks=8`、`streamed_tokens=1024`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_window_wait_clear_20260708.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
- 远程真实模型 smoke：`SMOKE_TAG=20260708_window_wait_clear RUN_PREFLIGHT=0 CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34687 MASTER_PORT=34687 tools/smoke_blockwise_prefill_remote.sh` 通过；真实长 prompt 输出 `profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260708_window_wait_clear.json`，`gate_pass=true`、`elapsed_s=29.39703532680869`、`output_tokens=1`。

### 2026-07-08 Blockwise attention ordered read windows

继续降低 blockwise window planning 的不确定性：旧逻辑先把 window blocks 放进 `set`，再 `list(unique_blocks)` 传给 `ensure_resident()` / `wait_for_blocks()`。这会丢掉跨 batch row 的 first-seen 顺序，使 H2D request / LRU touch / victim 决策顺序依赖 set 迭代，后续更难稳定优化 prefetch plan。

- 新增 `_unique_blocks_in_order()`，对 read window 逻辑块做 first-seen 去重，保留窗口内访问顺序。
- blockwise decode/prefill read window 都改成用 `unique_block_list` 调 `ensure_resident()` 和 `wait_for_blocks()`；`unique_blocks` 只保留给容量检查和 future/protected set。
- 新增 `tools/test_blockwise_attention_planning.py`，用 fake manager 验证 decode window `[2,0] + [1,2]` 的 staging/wait 顺序必须是 `[2,0,1]`。
- TDD RED：旧 set 路径下远程测试失败于 `assert manager.ensure_calls == [[2, 0, 1]]`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_ordered_windows_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_ordered_windows_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise read-window staging helper

继续为后续 prefetch plan 合并做结构整理：decode/prefill read window 原本各自重复执行 first-seen 去重、容量检查、`prefetch_*` stats、`ensure_resident()`、`wait_for_blocks(..., clear_pending=True)`。这些语义现在收敛到 `_stage_blockwise_read_window()`，保持行为不变，但让后续实现跨窗口/跨 row planner 时只需要改一个入口。

- `_stage_blockwise_read_window()` 返回 first-seen `unique_block_list`，统一负责：
  - capacity guard；
  - `prefetch_plans` / `prefetch_read_blocks` 统计；
  - `ensure_resident(require_valid=True)`；
  - 只等待并清理当前窗口 read blocks 的 pending H2D。
- decode/prefill blockwise attention 都改为复用 helper。
- 新增 `test_stage_blockwise_read_window_updates_stats_and_waits_only_window_blocks()`。
- TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_stage_blockwise_read_window'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_stage_helper_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_stage_helper_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise logical row normalization

继续减少 blockwise attention host-side Python planning 开销：decode 路径旧逻辑在每个 read window、每个 row 内重复执行 `[int(block) for block in row if int(block) >= 0]`；prefill 路径也在每个 row 单独重复过滤/转 int。现在新增 `_normalize_logical_block_rows()`，在函数入口一次性完成 logical row 规范化并返回 `max_blocks`。

- decode：入口处一次性生成 `block_rows, max_blocks`，窗口循环直接切片，不再每个窗口重新过滤同一行。
- prefill：入口处一次性生成 `block_rows`，row 循环直接复用。
- 新增 `test_normalize_logical_block_rows_filters_once_and_reports_max_blocks()`。
- TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_normalize_logical_block_rows'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_normalize_rows_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_normalize_rows_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise read-window unique-once staging

继续减少 window planning 重复工作：`_stage_blockwise_read_window()` 之前的调用点会先 `_unique_blocks_in_order(...)` / `set(...)` 做容量和 future/protected set，helper 内部又再做一次 first-seen 去重。现在 helper API 改为接收 `future_extra_blocks` / `protected_extra_blocks` / `capacity_extra_blocks`，由 helper 单入口完成 unique list、unique set、future/protected/capacity set 构造。

- decode/prefill 调用点不再重复构造 `unique_blocks` 和 `protected`。
- helper 内部仍返回 first-seen `unique_block_list`，并保持 stats、`ensure_resident()`、`wait_for_blocks(..., clear_pending=True)` 语义。
- TDD RED：远程旧 helper 签名下按预期失败：`TypeError: _stage_blockwise_read_window() got an unexpected keyword argument 'future_extra_blocks'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_unique_once_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_unique_once_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise decode mask template reuse

继续减少 decode window 内的小张量分配：旧 `_blockwise_online_decode_attention()` 在每个 read window 都新建 `torch.arange(max_window_tokens)` 作为 positions mask。现在在 decode 函数入口预建 `position_template = torch.arange(block_size * window_blocks)`，每个 window 只切片复用。

- 新增 `_decode_window_mask()`，统一从 positions template 和 `window_lens` 构造 decode read-window mask。
- 新增 `test_decode_window_mask_reuses_position_template()`。
- TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_decode_window_mask'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_mask_template_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_mask_template_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise prefill local causal mask template reuse

继续减少 prefill row 内小张量分配：旧 `_blockwise_online_prefill_attention()` 在每个 row 的 local causal attention 都新建 `torch.arange(q_len)` 和 `torch.arange(q_len)`。现在在函数入口按最大 chunk 长度预建 query/key position templates，每个 row 只切片复用。

- 新增 `_local_causal_mask()`，统一从预建 q/k position templates 构造 local causal mask。
- 新增 `test_local_causal_mask_reuses_position_templates()`。
- TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_local_causal_mask'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_local_mask_template_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_local_mask_template_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise prefill all-valid prefix merge

继续减少 prefix window 内无效小张量/掩码开销：historical prefix window 对当前 chunk 所有 query 都可见，旧代码仍每个 prefix window 创建全 `True` mask，并在 merge 中执行 `masked_fill`。现在抽出 `_merge_attention_window()`，支持 `mask=None` 表示 all-valid prefix window，跳过全 True mask 分配和 masked fill。

- prefix window 调 `_merge_attention_window(..., mask=None)`。
- local causal chunk 继续传 `_local_causal_mask(...)`，保持 causal 语义。
- 新增 `test_merge_attention_window_accepts_none_mask_as_all_valid()`。
- TDD RED：远程测试在 helper 缺失时按预期 `ImportError: cannot import name '_merge_attention_window'`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_merge_helper_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_merge_helper_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise prefix merge no-valid allocation

继续收紧 all-valid prefix merge 分支：上一轮 `mask=None` 虽然跳过了全 True mask 和 masked fill，但仍会创建 `torch.ones(scores.shape[:-1])` 作为 `valid`。现在 all-valid 分支不再创建 bool valid tensor，直接执行 merge；只保留对历史 `running_m=-inf` 的 old_weight 防护。

- 新增 `test_merge_attention_window_none_mask_does_not_allocate_valid_mask()`，monkeypatch `torch.ones` 抛错，确保 `mask=None` 分支不分配 valid mask。
- TDD RED：旧实现按预期失败：`AssertionError: torch.ones called`。
- 本地：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check` 通过。
- 远程：`tools/test_blockwise_attention_planning.py` 通过，`tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_no_valid_alloc_20260708.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`、`relative_error=1.853052561279985e-07`。
  - `profile_out/blockwise_prefill_no_valid_alloc_20260708.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。

### 2026-07-08 Blockwise planning stack integrated smoke

在上述 window wait、ordered planning、staging helper、row normalize、mask template、all-valid merge 优化全部推送后，补跑一次远程脚本级集成验证：

```bash
CUDA_VISIBLE_DEVICES=4 \
TINYVLLM_DIST_PORT=34721 MASTER_PORT=34721 \
RUN_PREFLIGHT=0 SMOKE_TAG=20260708_blockwise_planning_stack \
tools/smoke_blockwise_prefill_remote.sh
```

- 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260708_blockwise_planning_stack.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
- 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260708_blockwise_planning_stack.json`，`gate_pass=true`、`elapsed_s=41.34747215360403`、`output_tokens=1`。
- 结论：当前 blockwise planning 小优化栈在数学路径与真实模型 blockwise prefill 集成路径均未回归；`elapsed_s` 仍受远程 GPU/负载波动影响，不把单次 wall-clock 当作性能结论。

### 2026-07-09 KV offload write-block staging helper

继续把 prefill/decode 两条路径里重复的 write-block staging 逻辑收敛到 `_stage_kv_offload_write_blocks()`：先按出现顺序去重，再按首写 offset 拆成 valid write blocks 与 fresh write blocks，只对非空分组调用 `ensure_resident()`，避免空列表也进入 copy/staging 计划。

- 新增 `_unique_ints_in_order()` 与 `_stage_kv_offload_write_blocks(manager, write_blocks, first_write_offset_by_block, future_blocks)`。
- `prepare_prefill()` blockwise prefill 路径复用该 helper；`prepare_decode()` blockwise decode 路径也复用同一 helper。
- 新增 `tools/test_kv_write_staging.py`，覆盖 valid/fresh 分组、重复 block 去重，以及空分组不调用 `ensure_resident()`。
- TDD RED：远程测试在 helper 缺失时按预期失败：`ImportError: cannot import name '_stage_kv_offload_write_blocks' from 'tinyvllm.engine.model_runner'`。
- 远程 GREEN：`PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_kv_write_staging.py` 通过，输出 `kv write staging tests passed`。
- 远程回归：GPU4 上 `tools/test_kv_offload.py` 通过，输出 `kv offload tests passed`。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_write_staging_20260709_final.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_write_staging_20260709_final.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：write-block staging 的 prefill/decode 行为已统一，重复 block 统计按 unique block 计数，且不再为空 valid/fresh 分组制造无意义 `ensure_resident()` 调用；这是统一 KV access planner 的一个低风险中间步骤。

### 2026-07-09 KV offload empty ensure_resident fast path

继续收紧空 staging 调用：`KVOffloadMVP0.ensure_resident([])` 现在在去重后直接返回 `{}`，不再触发空 D2H/H2D enqueue、空 wait 或后续 mapping 构造。

- 新增 `test_ensure_resident_empty_blocks_is_noop_without_copy_hooks()`，用无 CUDA fake manager 覆盖空输入应返回空 mapping，且不调用 `_enqueue_d2h_pairs()`、`_enqueue_h2d_pairs()`、`wait_for_blocks()`。
- TDD RED：远程旧实现按预期失败：`AssertionError`，因为空输入仍调用了 enqueue hook。
- 远程 GREEN：`tools/test_kv_offload.py` 通过，输出 `kv offload tests passed`。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_empty_ensure_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_empty_ensure_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：空 block staging 现在是根部 no-op，减少下游 copy/wait helper 的无效调用；这是小优化，不改变任何非空 staging 语义。

### 2026-07-09 KV offload block-row map-only helper

继续减少重复 staging：`KVOffloadMVP0.translate_block_rows()` 仍负责 stage+translate；新增 `map_block_rows()` 只把已 resident logical block rows 映射为 physical slot rows，不再触发 `ensure_resident()`。full-attention decode 分支在已经完成 read/write staging 后，改用 `map_block_rows()` 构造 physical block table，避免同一 forward 内再跑一次 translate/staging 计划。

- 新增 `test_map_block_rows_uses_existing_resident_slots_without_staging()`，用无 CUDA fake manager 验证 map-only 不调用 `_enqueue_d2h_pairs()`、`_enqueue_h2d_pairs()`、`wait_for_blocks()`。
- TDD RED：远程旧实现按预期失败：`AttributeError: '_NoopKVOffload' object has no attribute 'map_block_rows'`。
- 远程 GREEN：`tools/test_kv_offload.py` 通过，输出 `kv offload tests passed`。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_map_rows_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_map_rows_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：full-attention decode 的 block-table 构造现在可以复用已 resident mapping，减少一次无必要的 staging/translation 入口调用；非 decode 调用方仍可继续使用 `translate_block_rows()` 保持原 stage+translate 语义。

### 2026-07-09 KV offload slot-position map-only helper

继续把 stage+translate 与 map-only 语义拆开：`KVOffloadMVP0.translate_slots_for_positions()` 仍负责 ensure+translate；新增 `map_slots_for_positions()` 只基于已 resident mapping 计算 physical slot ids，方便已完成 write staging 的路径复用同一映射逻辑而不再次进入 `ensure_resident()`。

- 新增 `test_map_slots_for_positions_uses_existing_resident_slots_without_staging()`，用无 CUDA fake manager 验证 map-only slot 计算不调用 `_enqueue_d2h_pairs()`、`_enqueue_h2d_pairs()`、`wait_for_blocks()`。
- TDD RED：远程旧实现按预期失败：`AttributeError: '_NoopKVOffload' object has no attribute 'map_slots_for_positions'`。
- 远程 GREEN：`tools/test_kv_offload.py` 通过，输出 `kv offload tests passed`。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_map_slots_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_map_slots_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：KV manager 现在同时有 block-row 与 slot-position 两个 map-only helper，为后续把已 stage 的 prefill/decode slot mapping 统一到 planner helper 做准备；现有 `translate_slots_for_positions()` 外部语义不变。

### 2026-07-09 KV offload write-position staging helper

继续把 blockwise prefill 热路径里的手写逻辑收敛：新增 `_stage_kv_offload_write_positions()`，负责从 write positions 计算 first-write offsets、调用 `_stage_kv_offload_write_blocks()`、再通过 `map_slots_for_positions()` 生成 physical slot ids。`prepare_prefill()` 的 blockwise prefill 分支不再手写 slot list。

- 新增 `test_stage_kv_offload_write_positions_stages_once_then_maps_slots()`，验证 helper 只产生一次 write staging plan，并复用 `map_slots_for_positions()` 输出 slots。
- TDD RED：远程旧实现按预期失败：`ImportError: cannot import name '_stage_kv_offload_write_positions'`。
- 远程 GREEN：`tools/test_kv_write_staging.py` 与 `tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_write_positions_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_write_positions_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：blockwise prefill write staging、write-block 统计和 slot mapping 已经进入单一 helper，减少后续优化 write/read planner 时需要维护的手写分支。

### 2026-07-09 KV offload full-decode staging helper

继续收敛 full-attention decode 的 read/write staging：新增 `_stage_kv_offload_full_decode_blocks()`，把 future block 集合、valid read blocks 过滤、`prefetch_*` 统计以及 read/write 两次 `ensure_resident()` 统一到单一 helper；`prepare_decode()` 非 blockwise 分支只保留 helper 调用和 map-only block-table 构造。

- 新增 `test_stage_kv_offload_full_decode_blocks_matches_existing_plan_shape()`，锁住旧逻辑的 read/write 统计与两次 `ensure_resident()` 调用形状。
- TDD RED：远程旧实现按预期失败：`ImportError: cannot import name '_stage_kv_offload_full_decode_blocks'`。
- GREEN 过程中修正测试 fake 缺失 `prefetch_read_blocks` 字段后，远程 `tools/test_kv_write_staging.py` 与 `tools/test_kv_offload.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_full_decode_helper_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_full_decode_helper_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：full-attention decode staging 的 read/write planning 进入单一 helper，行为与旧路径一致，但后续合并连续 prefetch plan / 降低 clean eviction 抖动时只需改一个入口。

### 2026-07-09 KV offload resident ensure_resident fast path

继续降低热路径调度开销：`KVOffloadMVP0.ensure_resident()` 在所有请求块已经 resident、没有 H2D、没有 D2H、没有 deferred wait 时，直接返回 mapping，不再调用空 `_enqueue_d2h_pairs()`、空 `_enqueue_h2d_pairs()` 或空 `wait_for_blocks()`。

- 新增 `test_ensure_resident_already_resident_blocks_skips_empty_copy_hooks()`，覆盖已 resident + `wait=True` 仍应跳过空 copy/wait hooks。
- TDD RED：远程旧实现按预期失败：`AssertionError`，因为已 resident 路径仍调用了空 enqueue hook；过程中补齐 fake manager 的 LRU 字段 `clock/slot_last_used`。
- 远程 GREEN：`tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_resident_fastpath_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_resident_fastpath_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：重复 staging 命中已 resident blocks 时少走一层空 copy/wait 调度；非 resident、dirty eviction、H2D reload 路径不变。

### 2026-07-09 KV planner helper stack integrated smoke

在 block-row/slot map-only helper、write-position helper、full-decode staging helper、resident fast path 全部推送后，补跑脚本级远程集成 smoke：

```bash
CUDA_VISIBLE_DEVICES=4 \
TINYVLLM_DIST_PORT=34731 MASTER_PORT=34731 \
RUN_PREFLIGHT=0 SMOKE_TAG=20260709_kv_planner_helpers \
tools/smoke_blockwise_prefill_remote.sh
```

- 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_kv_planner_helpers.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
- 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_kv_planner_helpers.json`，`gate_pass=true`、`elapsed_s=30.003381814807653`、`output_tokens=1`。
- 结论：这一组 KV planner/helper 小优化在脚本级数学路径与真实模型 blockwise prefill 集成路径均未回归；`elapsed_s` 只作为本次环境下 smoke 观测，不作为严格性能结论。

### 2026-07-09 KV offload empty wait_for_blocks fast path

继续减少空 wait 调度：`KVOffloadMVP0.wait_for_blocks([])` 现在在计算空 block set 后直接返回，不再获取 `torch.cuda.current_stream()`。这覆盖 `ensure_resident(wait=True)` 但本轮无 H2D blocks、以及上层偶发空 wait list 的情况。

- 新增 `test_wait_for_blocks_empty_is_noop_without_cuda_stream()`，monkeypatch `torch.cuda.current_stream` 抛错，确保空 wait 不触碰 CUDA stream。
- TDD RED：远程旧实现按预期失败：`AssertionError: current_stream called`。
- 远程 GREEN：`tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_empty_wait_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_empty_wait_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：空 wait list 不再进入 CUDA stream 查询路径；非空 H2D wait / clear_pending 语义不变。

### 2026-07-09 KV offload no-event wait_for_blocks fast path

继续减少 blockwise read window 的空 wait 开销：`KVOffloadMVP0.wait_for_blocks(blocks)` 现在先筛出实际存在 H2D event 的 blocks；如果请求 blocks 没有任何 pending H2D event，则只在 `clear_pending=True` 时清理 pending set，并直接返回，不再获取 `torch.cuda.current_stream()`。

- 新增 `test_wait_for_blocks_without_events_clears_pending_without_cuda_stream()`，monkeypatch `torch.cuda.current_stream` 抛错，确保无 event 的 wait 只清理 pending、不触碰 CUDA stream。
- TDD RED：远程旧实现按预期失败：`AssertionError: current_stream called`。
- 远程 GREEN：`tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_no_event_wait_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_no_event_wait_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：read window / staging 命中已 resident 或无 pending event 的 blocks 时，少走一次 CUDA stream 查询；有真实 H2D event 的 wait 语义和 event 去重逻辑保持不变。

### 2026-07-09 KV offload clean-fresh eviction no-op enqueue skip

继续减少 no-op copy hook：`ensure_resident()` 在 clean block 被 fresh write block 驱逐、且没有 dirty D2H、没有 CPU-valid H2D、没有 deferred wait 时，现在不会调用空 `_enqueue_d2h_pairs()`、空 `_enqueue_h2d_pairs()` 或空 `wait_for_blocks()`。

- 新增 `test_ensure_resident_clean_fresh_eviction_skips_empty_copy_hooks()`，构造 `gpu_blocks=1`、clean resident block 被 fresh block 替换的无 CUDA 场景，要求只更新 mapping/eviction stats，不调用空 copy/wait hooks。
- TDD RED：远程旧实现按预期失败：`AssertionError`，因为 clean fresh eviction 路径仍调用了空 D2H enqueue hook。
- 远程 GREEN：`tools/test_kv_offload.py` 与 `tools/test_kv_write_staging.py` 通过。
- 远程数学 smoke：
  - `profile_out/blockwise_decode_clean_fresh_noop_20260709.json`，`gate_pass=true`、`chunks=8`、`max_abs_error=2.9802322387695312e-08`。
  - `profile_out/blockwise_prefill_clean_fresh_noop_20260709.json`，`gate_pass=true`、`chunks=36`、`max_abs_error=2.4586915969848633e-07`。
- 结论：fresh write block 覆盖 clean resident block 的常见路径少走空 enqueue/wait 调度；dirty eviction、H2D reload、pending D2H wait 路径不变。

### 2026-07-09 KV wait/enqueue fast-path stack integrated smoke

在 empty wait、no-event wait、clean-fresh no-op enqueue 三个 copy/wait 调度优化推送后，补跑脚本级远程集成 smoke：

```bash
CUDA_VISIBLE_DEVICES=4 \
TINYVLLM_DIST_PORT=34741 MASTER_PORT=34741 \
RUN_PREFLIGHT=0 SMOKE_TAG=20260709_kv_wait_enqueue_fastpaths \
tools/smoke_blockwise_prefill_remote.sh
```

- 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_kv_wait_enqueue_fastpaths.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
- 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_kv_wait_enqueue_fastpaths.json`，`gate_pass=true`、`elapsed_s=30.008230686187744`、`output_tokens=1`。
- 结论：copy/wait fast-path 小优化栈在脚本级数学路径与真实模型 blockwise prefill 集成路径均未回归；`elapsed_s` 仅作本次 smoke 环境观测，不作为严格性能结论。

### 2026-07-09 Fast-path stack GPU blocks matrix

为观察 wait/enqueue fast-path 栈在不同 GPU staging 容量下的真实模型行为，补跑 GPU blocks matrix：

```bash
CUDA_VISIBLE_DEVICES=4 \
TINYVLLM_DIST_PORT=34751 MASTER_PORT=34751 \
RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=1 \
RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 \
SMOKE_TAG=20260709_fastpath_gpu_matrix \
tools/smoke_blockwise_prefill_remote.sh
```

结果：

| gpu_blocks | gate_pass | elapsed_s | h2d_copies | d2h_copies | evictions | resident_blocks | note |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | expected fail | - | - | - | - | - | `blockwise prefill window plus current write blocks exceed GPU staging slots: required=2, gpu_blocks=1` |
| 2 | true | 29.56653244420886 | 391 | 6 | 395 | 2 | pass |
| 4 | true | 28.152612544596195 | 249 | 6 | 251 | 4 | pass |

结论：`gpu_blocks=1` 仍是当前 blockwise prefill 的容量边界，不是 correctness mismatch；`gpu_blocks=4` 相比 `2` 明显减少 H2D/eviction 计数（391→249、395→251），说明当前主要开销仍来自 staging slot 容量导致的反复 H2D/evict，下一步更值得做的是减少跨 window 的重复 staging / 合并连续 prefetch，而不是继续只消除 Python 空调用。

## 2026-07-09 Blockwise prefill next-window future hint

基于上一轮 GPU blocks matrix，先做一个低风险 read-window planner 改动：`_blockwise_online_prefill_attention()` 在 stage 当前 prefix window 时，把紧邻的下一个 prefix window 加进 `future_logical_blocks` hint。该 hint 不进入 protected/capacity 集合，因此不会放宽 correctness 边界，也不会要求额外 staging slots；只影响 `lru_cost` eviction 评分，让 manager 更倾向保留下个窗口即将读取的 clean resident block，降低 `gpu_blocks` 稍宽时的重复 H2D/evict。

TDD/验证：

- RED：新增 `tools/test_blockwise_attention_planning.py::test_blockwise_prefill_read_windows_hint_next_prefix_blocks`，远程旧实现按预期 `AssertionError`，因为 prefill read-window caller 只把当前窗口传入 future hint。
- GREEN：`tinyvllm/layers/attention.py` 只在 prefill path 增加 `next_window` future hint；远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34761 MASTER_PORT=34761 RUN_PREFLIGHT=0 SMOKE_TAG=20260709_prefill_next_window_hint tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_prefill_next_window_hint.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
  - 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_prefill_next_window_hint.json`，`gate_pass=true`、`elapsed_s=30.453897550702095`、`output_tokens=1`。
- GPU blocks matrix：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34762 MASTER_PORT=34762 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=1 RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 SMOKE_TAG=20260709_prefill_next_window_hint_matrix tools/smoke_blockwise_prefill_remote.sh`。

| gpu_blocks | gate_pass | elapsed_s | h2d_copies | d2h_copies | evictions | resident_blocks | note |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | expected fail | - | - | - | - | - | `blockwise prefill window plus current write blocks exceed GPU staging slots: required=2, gpu_blocks=1` |
| 2 | true | 29.297932274639606 | 391 | 6 | 395 | 2 | unchanged copy pressure vs previous matrix |
| 4 | true | 28.453177761286497 | 193 | 6 | 195 | 4 | H2D 249→193、evict 251→195 |

结论：next-window future hint 对 `gpu_blocks=2` 无改善，因为只有一个 spare/none spare slot 时当前 window + write block 已基本决定 eviction；对 `gpu_blocks=4` 明显减少重复 staging（H2D 约 -22.5%、eviction 约 -22.3%）。wall-clock 单次波动不作严格结论，但 copy/evict counter 改善稳定指向后续方向：继续做多窗口 lookahead / 连续 prefetch plan 合并，同时保持 protected/capacity 语义保守。

## 2026-07-09 Blockwise prefill capacity-bounded multi-window future hint

继续沿上一个 next-window hint 做低风险扩展：新增 `_blockwise_prefill_future_hint_blocks()`，按 `gpu_blocks - 当前窗口 unique blocks - write_blocks` 计算 spare future budget，在不改变 protected/capacity 集合的前提下，把可容纳的后续 prefix blocks 加进 `future_logical_blocks`。这仍只是 eviction scoring hint，不会要求额外 resident blocks，也不改变 `gpu_blocks=1` 容量边界。

TDD/验证：

- RED：新增 `test_blockwise_prefill_read_windows_hint_capacity_bounded_future_prefix_blocks`，远程旧实现按预期 `AssertionError`，因为只 hint 下一个 prefix window。
- GREEN：新增 helper 直测 `test_blockwise_prefill_future_hint_blocks_fill_only_spare_capacity`，并更新 caller 语义测试；远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34771 MASTER_PORT=34771 RUN_PREFLIGHT=0 SMOKE_TAG=20260709_prefill_multi_window_hint tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_prefill_multi_window_hint.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
  - 真实模型长 prompt smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_prefill_multi_window_hint.json`，`gate_pass=true`、`elapsed_s=30.21120012179017`、`output_tokens=1`。
- GPU blocks matrix：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34772 MASTER_PORT=34772 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 RUN_REAL_SMOKE=1 RUN_GPU_BLOCKS_MATRIX=1 MATRIX_REQUIRE_PASS=0 SMOKE_TAG=20260709_prefill_multi_window_hint_matrix tools/smoke_blockwise_prefill_remote.sh`。

| gpu_blocks | gate_pass | elapsed_s | h2d_copies | d2h_copies | evictions | resident_blocks | note |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | expected fail | - | - | - | - | - | `blockwise prefill window plus current write blocks exceed GPU staging slots: required=2, gpu_blocks=1` |
| 2 | true | 29.661339823156595 | 391 | 6 | 395 | 2 | unchanged copy pressure |
| 4 | true | 29.4657124876976 | 166 | 6 | 168 | 4 | H2D 193→166 vs next-window, 249→166 vs baseline matrix |

结论：capacity-bounded multi-window hint 继续降低 `gpu_blocks=4` 的重复 staging：相对 next-window hint H2D 约 -14.0%、eviction 约 -13.8%；相对 wait/enqueue fast-path matrix H2D 约 -33.3%、eviction 约 -33.1%。`gpu_blocks=2` 仍无改善，符合容量没有 spare lookahead 的预期。后续若继续做 copy/evict，优先考虑把这种 future hint 与更显式的 read-window planner 合并，而不是扩大 protected 集合。

## 2026-07-09 Decode read-window capacity-bounded future hint

把 prefill 的 capacity-bounded hint helper 泛化为 `_blockwise_read_window_future_hint_blocks()`，并接入 `_blockwise_online_decode_attention()`。decode 每个 read window 现在也会把后续可容纳的 logical blocks 加进 `future_logical_blocks`，但仍只等待当前窗口 blocks，且不扩大 protected/capacity 集合。

TDD/验证：

- RED：新增 `test_blockwise_decode_read_windows_hint_capacity_bounded_future_blocks`，远程旧实现按预期 `AssertionError`，因为 decode caller 只把当前窗口/写块作为 future hint。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34781 MASTER_PORT=34781 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260709_decode_read_window_hint tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_decode_read_window_hint.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=51.44479962438345`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：decode path 已覆盖到真实模型多步 decode，无 correctness 回归。当前 `gpu_blocks=2` 的 decode KV counters 仍很高，说明 read-window hint 本身不解决低 staging 容量下的反复 reload；下一步更值得做的是把 read-window planner 显式化，复用 future hint 并减少 per-layer/per-step 重复 plan 开销，或在更宽 `gpu_blocks` 下做 decode matrix 观察。

## 2026-07-09 Blockwise GQA no-repeat window attention

继续做 window 内部内存/算子开销优化：blockwise decode/prefill 旧路径在每个 window 上用 `_repeat_kv_for_gqa()` 把 `[num_kv_heads]` K/V materialize 到 `[num_heads]`，Qwen3 这类 GQA 模型会放大 window K/V 临时张量。新增 grouped GQA score/value helpers：

- `_gqa_scores_decode()` / `_gqa_weighted_values_decode()`
- `_gqa_scores_prefill()` / `_gqa_weighted_values_prefill()`

这些 helper 通过 reshape 成 `(num_kv_heads, group_size)` 直接算 grouped attention，不再 materialize repeated K/V heads；`num_kv_heads == num_heads` 时仍走原 einsum fast path。

TDD/验证：

- RED：新增 `test_blockwise_decode_gqa_does_not_materialize_repeated_kv_heads` 和 `test_blockwise_prefill_gqa_does_not_materialize_repeated_kv_heads`，远程旧实现按预期触发 `AssertionError("repeat_kv_for_gqa called")`。
- 新增等价性测试 `test_gqa_grouped_helpers_match_repeated_kv_reference`，确认 grouped helpers 与旧 `repeat_interleave + einsum` 数值一致。
- 远程目标测试通过：`tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`。
- 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34791 MASTER_PORT=34791 RUN_PREFLIGHT=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260709_gqa_no_repeat_window tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_gqa_no_repeat_window.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
  - 真实模型长 prompt + decode smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_gqa_no_repeat_window.json`，`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=52.595071755349636`。
  - KV counters 与上一轮 decode smoke 基本一致：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`，符合该优化不改变 KV offload staging 计划、只减少 window 内 repeated K/V materialization 的预期。
- GPU blocks matrix：第一次 `SMOKE_TAG=20260709_gqa_no_repeat_matrix` 遇到 `EADDRINUSE`，换端口重跑 `SMOKE_TAG=20260709_gqa_no_repeat_matrix_r2` 通过。
  - `gpu_blocks=1`：预期容量失败，仍报 `required=2, gpu_blocks=1`。
  - `gpu_blocks=2`：`gate_pass=true`、`elapsed_s=29.652997620403767`、`h2d_copies=391`、`d2h_copies=6`、`evictions=395`、`resident_blocks=2`。
  - `gpu_blocks=4`：`gate_pass=true`、`elapsed_s=29.608206927776337`、`h2d_copies=166`、`d2h_copies=6`、`evictions=168`、`resident_blocks=4`。

结论：GQA no-repeat 没有改变 copy/evict counters，这是预期；收益点是减少每个 blockwise window 的 K/V head-expanded临时张量和对应 `repeat_interleave`。wall-clock 单次不作为严格性能结论，但 correctness、decode 覆盖和 matrix counters 都未回归。后续更大的收益仍在 read-window planner/减少 H2D reload，或进一步把 grouped GQA helper 下沉为更高效 kernel。

## 2026-07-09 Blockwise prefill prefix dense buffer no-zero-fill

继续减少 window 内临时张量开销：`_blockwise_online_prefill_attention()` 的 prefix read window 会按 `window_len` 完整从 staged KV cache copy K/V，因此原来的 `q.new_zeros(...)` 会先清零再被全量覆盖。已改为 `q.new_empty(...)`，只影响 prefix historical KV window；decode batch window 仍保留 `new_zeros`，因为 batch padding/不同 `window_lens` 需要安全填充。

TDD/验证：

- RED：新增 `test_blockwise_prefill_prefix_windows_do_not_zero_fill_dense_buffers`，远程旧实现按预期触发 `AssertionError("new_zeros called for fully-copied prefix window")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34811 MASTER_PORT=34811 RUN_PREFLIGHT=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260709_prefill_empty_buffers tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_prefill_empty_buffers.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
  - 真实模型长 prompt + decode smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_prefill_empty_buffers.json`，`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=51.531816590577364`。

结论：这是一个低风险内存带宽/allocator 微优化，不改变 KV offload staging counters；价值是 prefix window 每次少做 K/V dense buffer 清零。后续类似优化可以继续检查“完全覆盖写入”的临时张量，但不要动 decode padded dense buffer。

## 2026-07-09 Decode read-window plan cache across layers

继续减少 host-side 重复规划：blockwise decode 的 logical block rows、context lens、window size、write blocks 在一次 forward 的所有 decoder layers 中相同；旧实现每层都重算 `window_rows/window_lens/needed_blocks/future_hint_blocks`。新增 `_build_blockwise_decode_window_plan()`，并把计划缓存到 `Context.kv_offload_decode_window_plan_cache`，同一次 `set_context()` 生命周期内后续层复用该 plan；`reset_context()` 会自然清空。

TDD/验证：

- RED：新增 `test_blockwise_decode_reuses_cached_read_window_plan_across_layers`，第二次调用同一 context 时 monkeypatch `_blockwise_read_window_future_hint_blocks`，远程旧实现按预期触发 `AssertionError("decode read-window plan recomputed")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34821 MASTER_PORT=34821 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260709_decode_plan_cache tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_decode_plan_cache.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=52.10690261051059`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：该优化不改变 staging/copy 行为，主要减少每层重复 Python/list planning，为后续把 prefill/decode read-window planner 统一成显式 planner 做准备。真正降低 H2D reload 仍需要继续优化 staging capacity/reuse 策略。

## 2026-07-09 Prefill prefix-window plan cache across layers

继续减少 host-side 重复规划：blockwise prefill 的 logical block rows、chunk starts/ends、window size、write blocks 在一次 forward 的所有 decoder layers 中相同；旧实现每层都重算 prefix `window/window_len/future_hint_blocks`。新增 `_build_blockwise_prefill_window_plan()`，并把计划缓存到 `Context.kv_offload_prefill_window_plan_cache`，同一次 `set_context()` 生命周期内后续层复用该 plan；`reset_context()` 会自然清空。

TDD/验证：

- RED：新增 `test_blockwise_prefill_reuses_cached_prefix_window_plan_across_layers`，第二次调用同一 context 时 monkeypatch `_blockwise_prefill_future_hint_blocks`，远程旧实现按预期触发 `AssertionError("prefill prefix-window plan recomputed")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34831 MASTER_PORT=34831 RUN_PREFLIGHT=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260709_prefill_plan_cache tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_prefill_plan_cache.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
  - 真实模型长 prompt + decode smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_prefill_plan_cache.json`，`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=75.38668528944254`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：该优化不改变 staging/copy 行为，主要减少每层重复 prefix-window Python/list planning，并让 prefill/decode 都具备 per-forward read-window plan cache。后续更值得继续做的是把两个 plan builder 收敛为统一 planner，并继续降低低 staging 容量下的 H2D reload/clean eviction。

## 2026-07-09 Prefill local position template cache across layers

继续减少同一次 forward 多层重复 GPU 小张量创建：blockwise prefill 的 local causal mask 每层都会按同一个 `max_chunk_tokens` 在同一 device 上重建 `torch.arange(...).view(...)` 的 q/k position templates。新增 `Context.kv_offload_prefill_position_template_cache`，缓存 `(max_chunk_tokens, device, q_pos_template, k_pos_template)`；同一个 `set_context()` 生命周期内后续层复用，shape/device 变化时自动重建。

TDD/验证：

- RED：新增 `test_blockwise_prefill_reuses_cached_local_position_templates_across_layers`，第二次调用同一 context 时 monkeypatch `torch.arange`，远程旧实现按预期触发 `AssertionError("prefill local position templates recomputed")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34841 MASTER_PORT=34841 RUN_PREFLIGHT=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260709_prefill_position_template_cache tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 数学 smoke：`profile_out/blockwise_prefill_attn_online_softmax_smoke_20260709_prefill_position_template_cache.json`，`gate_pass=true`、`chunks=36`、`streamed_tokens=4544`、`max_abs_error=2.4586915969848633e-07`、`relative_error=1.4178931585709836e-06`。
  - 真实模型长 prompt + decode smoke：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260709_prefill_position_template_cache.json`，`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=67.73630218580365`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：这是低风险 host/GPU 小张量创建优化，不改变 attention 数值或 KV staging/copy 计划；收益点是每层少重建 local causal mask 的 position templates。后续可以把 decode `position_template` 也纳入同类 per-forward cache，或继续推进统一 read-window planner。

## 2026-07-10 Decode position template cache across layers

继续减少同一次 forward 多层重复 GPU 小张量创建：blockwise decode 的 window mask 每层都会按同一个 `block_size * window_blocks` 在同一 device 上重建 `torch.arange(...).view(1,1,-1)`。新增 `Context.kv_offload_decode_position_template_cache`，缓存 `(position_template_tokens, device, position_template)`；同一个 `set_context()` 生命周期内后续 decoder layers 复用，shape/device 变化时自动重建。

TDD/验证：

- RED：新增 `test_blockwise_decode_reuses_cached_position_template_across_layers`，第二次调用同一 context 时 monkeypatch `torch.arange`，远程旧实现按预期触发 `AssertionError("decode position template recomputed")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34851 MASTER_PORT=34851 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_decode_position_template_cache tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_decode_position_template_cache.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=52.79811315238476`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：该优化不改变 staging/copy 行为，主要减少每层重复 decode mask position template 创建；KV counters 与上一轮 decode smoke 一致，符合预期。下一步更高收益仍在降低低 staging 容量下的 H2D reload/eviction，或做统一 read-window planner 后再跑稳定 benchmark。

## 2026-07-10 Decode full-window dense buffer no-zero-fill

继续减少 window 内临时张量开销：blockwise decode 的 dense K/V buffer 旧路径始终用 `q.new_zeros(...)` 分配；当当前 window 对所有 batch row 都是完整覆盖写入时，清零会被随后的 `k_cache/v_cache` copy 全量覆盖。现在只在 `all(window_len == max_window_tokens)` 的完整窗口使用 `q.new_empty(...)`，partial/padded window 继续保守使用 `new_zeros(...)`。

TDD/验证：

- RED：新增 `test_blockwise_decode_full_windows_do_not_zero_fill_dense_buffers`，完整 decode window 下 monkeypatch `q.new_zeros`，远程旧实现按预期触发 `AssertionError("new_zeros called for fully-copied decode window")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34861 MASTER_PORT=34861 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_decode_empty_buffers tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_decode_empty_buffers.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=51.456428956240416`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：这是低风险内存带宽/allocator 微优化，不改变 KV staging/copy 计划；收益点是完整 decode window 每层少做 K/V dense buffer 清零。更大收益仍需继续处理低 `gpu_blocks=2` 下的 repeated reload。

## 2026-07-10 Decode window mask cache across layers

继续减少同一次 forward 多层重复 GPU 小张量创建：blockwise decode 的每个 read window 旧路径每层都会通过 `_decode_window_mask()` 重新构造 `torch.tensor(window_lens)`、比较 position template，并重新计算 `valid = mask.any(dim=-1)`。新增 `Context.kv_offload_decode_window_mask_cache`，按 `(tuple(window_lens), max_window_tokens, device)` 缓存 `(mask, valid)`；同一个 `set_context()` 生命周期内后续 decoder layers 复用。

TDD/验证：

- RED：新增 `test_blockwise_decode_reuses_cached_window_masks_across_layers`，第二次调用同一 context 时 monkeypatch `_decode_window_mask`，远程旧实现按预期触发 `AssertionError("decode window mask recomputed")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34871 MASTER_PORT=34871 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_decode_mask_cache tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_decode_mask_cache.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=66.21972536295652`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：该优化不改变 staging/copy 行为，主要减少每层重复 window mask/valid 构造；本次 wall-clock 偏慢且 H2D 时间波动大，不作为性能结论。更大收益仍需处理低 `gpu_blocks=2` 的 repeated reload，或把这些 per-forward cache 之后的剩余 Python planner 合并。

## 2026-07-10 Decode full-window mask skip

继续减少完整 decode read window 的无效 mask 路径：当 `all(window_len == max_window_tokens)` 时，当前 window 对所有 batch row 都全有效，不需要 `_decode_window_mask()`、`masked_fill(~mask)`、`valid = mask.any(...)` 和对应 `torch.where`。现在 full-window decode 直接按无 mask scores 做 softmax/merge；partial/padded window 仍走原 mask/cache 路径。

TDD/验证：

- RED：新增 `test_blockwise_decode_full_windows_skip_mask_construction`，full decode window 下 monkeypatch `_decode_window_mask`，远程旧实现按预期触发 `AssertionError("full decode window should not build mask")`。
- GREEN：远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34881 MASTER_PORT=34881 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_decode_full_window_mask_skip tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_decode_full_window_mask_skip.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=50.60374375060201`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=811`、`d2h_copies=9`、`evictions=815`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1626`。

结论：这是低风险 compute/mask 微优化，不改变 KV staging/copy 计划；full decode window 少走 mask 构造、masked fill 和 valid merge 分支。H2D/evict 计数不变，后续更大收益仍在减少 low-staging repeated reload。

## 2026-07-10 Decode alternating read-window order

针对 `gpu_blocks=2` 低 staging 容量下 decode 层间反复从同一个方向扫描 read windows、导致下一层无法复用上一层结束时仍 resident 的 tail window，本轮让奇数层反向遍历已缓存的 decode read-window plan，并为反向顺序单独计算 capacity-bounded reverse future hint。偶数层保持原正向顺序；`layer_idx < 0` 的兼容路径也保持正向，避免未注入 layer metadata 时意外改变行为。

TDD/验证：

- RED1：新增 `test_blockwise_decode_odd_layers_stage_read_windows_from_tail`，旧实现按预期报 `_blockwise_online_decode_attention() got an unexpected keyword argument 'layer_idx'`。
- RED2：新增 `test_blockwise_decode_odd_layers_hint_reverse_future_blocks`，中间实现只反向遍历、不反向 hint 时能抓到 future hint 方向错误；同时修复了 `layer_idx=-1` 被 Python `%` 误判为奇数的问题。
- GREEN：本地 `PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py` 与 `git diff --check` 通过；远程 `tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`、`tools/test_kv_offload.py` 均通过。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34731 MASTER_PORT=34731 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_decode_alternating_window_order tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_decode_alternating_window_order.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=52.29836422204971`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=728`、`d2h_copies=9`、`evictions=732`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1543`。
- Decode matrix：第一次 `MASTER_PORT=34732` 整轮失败于 `EADDRINUSE`，无 JSON 生成，不作为回归；换 `TINYVLLM_DIST_PORT=34873 MASTER_PORT=34873 KV_OFFLOAD_GPU_BLOCKS_MATRIX="2 4" SMOKE_TAG=20260710_decode_alternating_window_order_matrix2` 后通过。
  - `gpu_blocks=2`：`gate_pass=true`、`elapsed_s=52.78420425578952`、`h2d_copies=728`、`d2h_copies=9`、`evictions=732`、`resident_blocks=2`。
  - `gpu_blocks=4`：`gate_pass=true`、`elapsed_s=52.421591091901064`、`h2d_copies=336`、`d2h_copies=9`、`evictions=338`、`resident_blocks=4`。

结论：这次终于改变了 decode staging 行为，不只是 mask/小张量微优化。相对上一轮 `20260710_decode_full_window_mask_skip` 的 `gpu_blocks=2` counters，H2D `811 -> 728`、evictions `815 -> 732`、copy waits `1626 -> 1543`，约减少 10% 重复 reload/eviction；wall-clock 仍按远程单次 smoke 噪声处理，不单独宣称稳定 tok/s 提升。下一步更值得继续做的是把相邻 window 的 H2D/D2H batch 合并、或让 planner 直接输出跨层 alternating order + future hints 的统一结构，进一步减少 `prefetch_plans=933` 和 copy waits。

## 2026-07-10 Skip stale H2D waits

继续降低 decode KV offload 的同步等待开销：`wait_for_blocks(..., clear_pending=True)` 现在只等待仍在 `pending_wait_blocks` 的 blocks。旧逻辑会对传入 block 里任何有 `h2d_done` event 的 block 调 `current_stream.wait_event()`，即使该 block 的 pending 状态已经在前序窗口清掉；在 blockwise decode 反复 staging/reuse 时会造成 stale event 重复 wait 和 `copy_waits` 计数偏高。

TDD/验证：

- RED：新增 `test_wait_for_blocks_clear_pending_skips_non_pending_stale_events_without_cuda_stream`，构造 `h2d_done={0,1}` 但 `pending_wait_blocks={1}`，调用 `wait_for_blocks([0], clear_pending=True)`；旧实现会触发 `torch.cuda.current_stream()`，按预期失败。
- GREEN：`wait_for_blocks()` 在 `clear_pending=True` 时先做 `blocks &= self.pending_wait_blocks`，空集合直接返回，不触碰 CUDA stream。
- 本地验证通过：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py tools/test_kv_offload.py tools/test_blockwise_attention_planning.py`、`git diff --check`。
- 远程目标测试通过：`tools/test_kv_offload.py`、`tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34901 MASTER_PORT=34901 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_skip_stale_h2d_waits tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_skip_stale_h2d_waits.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=62.47536563873291`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=728`、`d2h_copies=9`、`evictions=732`、`prefetch_plans=933`、`prefetch_read_blocks=924`、`prefetch_write_blocks=9`、`copy_waits=1460`。
- 负分支记录：尝试过“当前 window 后立即 prefetch 下一 read window 不 wait”的实现，`SMOKE_TAG=20260710_decode_next_window_prefetch` correctness 通过，但 counters 显示 `h2d_copies=728`、`evictions=732`、`copy_waits=1543` 不优于 alternating-order baseline，且 `prefetch_plans` 从 `933` 增到 `1017`；该实现已回滚，不提交。

结论：相对 `20260710_decode_alternating_window_order`，本轮不改变 H2D/eviction 行为，但把 `copy_waits` 从 `1543 -> 1460`，约减少 5.4% wait_event 次数；这是 wait-path 去重收益，不应解读为稳定 wall-clock 提升。下一步更有价值的是继续减少 `prefetch_plans=933`，例如把相邻窗口/跨层 planner 合并为一次计划，而不是额外增加 prefetch 调用。

## 2026-07-10 Skip resident read-window staging

继续减少 blockwise decode 的 planner/staging 开销：`_stage_blockwise_read_window()` 现在遇到“当前 read window 的所有 logical blocks 已经 resident，且不在 `pending_wait_blocks`”时直接走 fast-path，不再增加 `prefetch_plans/prefetch_read_blocks`，也不再调用 `ensure_resident()` 和 `wait_for_blocks()`。为避免 fast-path 让 resident blocks 的 LRU recency 变旧，仍会对每个 unique resident block 调 `_touch(slot)`。

TDD/验证：

- RED：新增 `test_stage_blockwise_read_window_skips_resident_non_pending_window`，旧实现会把 already-resident window 仍计入 `prefetch_plans` 并调用 `ensure_resident()/wait_for_blocks()`，按预期失败。
- GREEN：新增 resident fast-path，并在测试中验证返回 unique blocks、不增加 prefetch counters、不调用 staging/wait hooks，同时更新 slot touch 顺序。
- 中间负分支：第一次 fast-path 没有 `_touch()`，真实 smoke correctness 通过但 H2D `728 -> 729`、evictions `732 -> 733`，说明 resident blocks 没刷新 LRU 会轻微恶化 eviction；已补 `_touch()` 后复测。
- 本地验证通过：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check`。
- 远程目标测试通过：`tools/test_blockwise_attention_planning.py`、`tools/test_kv_offload.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34921 MASTER_PORT=34921 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_skip_resident_read_stage_touch tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_skip_resident_read_stage_touch.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=52.374693654477596`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=728`、`d2h_copies=9`、`evictions=732`、`prefetch_plans=737`、`prefetch_read_blocks=728`、`prefetch_write_blocks=9`、`copy_waits=1460`。

结论：相对 `20260710_skip_stale_h2d_waits`，H2D/eviction/copy_waits 不劣化，同时 `prefetch_plans 933 -> 737`、`prefetch_read_blocks 924 -> 728`，约减少 21% read-window staging plan/hook 次数。这是 Python/planner 和无效 wait hook 降噪，不单独宣称稳定 wall-clock 提升。下一步若继续，应重点看剩余 737 个真实 staging windows 是否可由跨层 planner 合并，或把 `prefetch_read_blocks` 进一步降到接近 H2D copies 下界。

## 2026-07-10 Clear shared H2D pending blocks

修正 batched H2D event 的 pending 清理语义：当 `wait_for_blocks(..., clear_pending=True)` 等待了某个 H2D event 后，所有仍在 `pending_wait_blocks` 且共享同一个 `h2d_done` event 的 logical blocks 都可以一起清掉。旧逻辑只清 requested blocks；在 `_enqueue_h2d_pairs()` 把多个 logical blocks 合并到同一个 CUDA event 时，未 requested 的同批 block 会继续留在 pending，后续 resident fast-path 会误认为它们仍未等待，阻止 skip resident staging。

TDD/验证：

- RED：新增 `test_wait_for_blocks_clear_pending_clears_all_blocks_sharing_waited_event_without_cuda`，构造 `h2d_done[0] is h2d_done[1]` 且 `pending_wait_blocks={0,1,2}`，调用 `wait_for_blocks([0], clear_pending=True)`；旧实现只清 `0`，按预期失败。
- GREEN：`wait_for_blocks()` 记录本次实际等待的 event id，并在 `clear_pending=True` 时额外清掉所有共享这些 event id 的 pending blocks。
- 更新 CUDA 测试语义：原 `test_wait_for_blocks_can_clear_only_requested_pending_h2d_waits` 改为 `test_wait_for_blocks_clears_pending_blocks_that_share_h2d_event`，显式断言同批 H2D event 下 `{0,1}` 都被清掉。
- 本地验证通过：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py`、`git diff --check`。
- 远程目标测试通过：`tools/test_kv_offload.py`、`tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`。
- Decode 集成 smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34931 MASTER_PORT=34931 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_clear_shared_h2d_pending tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_clear_shared_h2d_pending.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=53.252644926309586`。
  - KV counters：`gpu_blocks=2`、`h2d_copies=728`、`d2h_copies=9`、`evictions=732`、`prefetch_plans=737`、`prefetch_read_blocks=728`、`prefetch_write_blocks=9`、`copy_waits=1460`。

结论：当前单 prompt decode smoke counters 与 `20260710_skip_resident_read_stage_touch` 一致，说明这个修复不是该 smoke 的直接收益来源；它是 batched H2D / coalesced event 场景的语义修复，避免共享 event 的 sibling blocks 长期卡在 pending，从而为后续真正合并 H2D spans、跨窗口批量 staging 铺路。

## 2026-07-10 Assign contiguous H2D slots for missing block batches

批量 reload 多个 missing logical blocks 时，`ensure_resident()` 现在先选出一组 victim slots，再把 missing logical blocks 和 selected slots 分别排序后配对。这样在 LRU victim 顺序与物理 slot 顺序相反时，仍可把连续 logical blocks 放进连续 GPU slots，让 `_coalesce_copy_pairs()` 形成更长 H2D span；外部语义不变，dirty D2H、clean eviction、LRU `_touch()` 仍按同一批 selected slots 执行。

TDD/验证：

- RED：新增 `test_ensure_resident_assigns_contiguous_missing_blocks_to_contiguous_slots_for_coalesced_h2d`，构造 `slot_last_used=[3,2,1,0]` 的反向 LRU 场景；旧实现会把 logical `4,5,6,7` 映射到 slots `3,2,1,0`，`_coalesce_copy_pairs()` 无法合成 `(4,0,4)`，按预期失败。
- GREEN：`ensure_resident()` 对同一批 missing blocks 做 sorted logical -> sorted slot 配对，使测试中的 H2D pairs 变为 `[(4,0),(5,1),(6,2),(7,3)]`，可 coalesce 成单 span。
- 本地验证通过：`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py`、`git diff --check`。本地 `tools/test_kv_offload.py` 不能跑，因为当前本机 Python 缺 `torch`。
- 远程目标测试通过：`tools/test_kv_offload.py`、`tools/test_blockwise_attention_planning.py`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`。
- 默认 decode smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34841 MASTER_PORT=34841 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 SMOKE_TAG=20260710_contiguous_h2d_slots tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_contiguous_h2d_slots.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=51.20729864016175`。
  - KV counters 与上一轮一致：`gpu_blocks=2`、`kv_offload_blockwise_blocks=1`、`h2d_copies=728`、`d2h_copies=9`、`evictions=732`、`prefetch_plans=737`、`prefetch_read_blocks=728`、`copy_waits=1460`、`h2d_batches=728`、`h2d_batch_spans=728`。该配置每次 H2D 基本只有单 block，因此不能体现 slot coalescing 收益。
- 多块窗口 decode smoke：`CUDA_VISIBLE_DEVICES=4 TINYVLLM_DIST_PORT=34843 MASTER_PORT=34843 RUN_PREFLIGHT=0 RUN_MATH_SMOKE=0 MAX_OUTPUT_LEN=4 KV_OFFLOAD_GPU_BLOCKS=4 KV_OFFLOAD_BLOCKWISE_BLOCKS=2 SMOKE_TAG=20260710_contiguous_h2d_slots_w2g4 tools/smoke_blockwise_prefill_remote.sh` 通过。
  - 输出：`profile_out/kv_offload_blockwise_prefill_real_longctx_smoke_20260710_contiguous_h2d_slots_w2g4.json`
  - summary：`gate_pass=true`、`output_tokens=4`、`decode_steps=3`、`elapsed_s=48.30962808430195`。
  - KV counters：`gpu_blocks=4`、`kv_offload_blockwise_blocks=2`、`h2d_copies=337`、`d2h_copies=9`、`evictions=339`、`prefetch_plans=276`、`prefetch_read_blocks=534`、`copy_waits=606`、`h2d_batches=267`、`h2d_batch_spans=295`。这里已经出现真实 batch/span 合并：H2D blocks 337 个被压到 267 次 enqueue / 295 个 spans。

结论：这轮不是默认 `blockwise_blocks=1/gpu_blocks=2` 的直接收益，而是为更大 staging / 多块窗口 / batch-copy 场景减少 H2D span fragmentation。后续更值得继续做的是把 read-window planner 主动输出可合并的连续 logical windows，或把 `_enqueue_h2d_pairs()` 从“按调用批次合并”提升到跨相邻窗口的显式 batch staging。

## 2026-06-30 Streaming/blockwise attention 数学 smoke

为了进入“单条超长上下文超过 staging slots”的下一阶段，先没有直接改 production attention kernel，而是在 `tools/profile_ngram_commit.py` 增加了 exact blockwise decode attention 的 online-softmax smoke：

```bash
--blockwise-attn-smoke
--blockwise-attn-batch N
--blockwise-attn-heads N
--blockwise-attn-kv-heads N
--blockwise-attn-head-dim N
--blockwise-attn-tokens N
--blockwise-attn-window-tokens N
```

实现口径：

- 构造 synthetic decode `q/k/v`。
- full attention 一次性计算作为 reference。
- blockwise 路径按 `window_tokens` 分块扫描 K/V。
- 使用 online softmax merge：维护 running max、running denominator、running output。
- 支持 GQA：`num_heads` 可大于 `num_kv_heads`，通过 KV head repeat 做对齐。
- 校验 blockwise 输出与 full attention 输出误差。

本地通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/bytedance/dev/TinyLLMForge \
  python3 /Users/bytedance/dev/TinyLLMForge/tools/profile_ngram_commit.py --help >/dev/null
```

注意：本地无 torch，实际 smoke 在远程跑。

远程 smoke：

```bash
cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --blockwise-attn-smoke \
  --blockwise-attn-tokens 2048 \
  --blockwise-attn-window-tokens 128 \
  --blockwise-attn-batch 2 \
  --blockwise-attn-heads 4 \
  --blockwise-attn-kv-heads 2 \
  --blockwise-attn-head-dim 32 \
  --out-json profile_out/blockwise_attn_online_softmax_smoke_20260630.json
```

结果：

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 16 |
| streamed_tokens | 2048 |
| max_abs_error | 2.98e-08 |
| relative_error | 2.31e-07 |
| context_lens | [2048, 1984] |

输出文件：`profile_out/blockwise_attn_online_softmax_smoke_20260630.json`

结论：exact blockwise/streaming decode attention 的数学闭环已验证；下一步若接真实 KV offload，需要把这个 online-softmax merge 放到真实 decode attention 路径，并让每个 block/window 触发 KV staging/prefetch，而不是要求一次性把所有 visible blocks staged 到 GPU。

## 2026-06-30 真实 decode attention 接入 blockwise KV offload

已把 blockwise online-softmax merge 接进真实 decode attention 路径，仍然默认关闭、仅覆盖受控范围：

- 仅 `kv_offload_mvp0=True` 时可用。
- 仅 fp16/bf16 KV（`kv_quant_bits == 0`）。
- 仅 decode path；prefill 仍走原路径。
- 不支持 Quest / KV-Cartridge / AM compact / KV4 / KV8。
- 不改 scheduler / BlockManager / attention kernel；blockwise decode MVP 使用 PyTorch gather + online-softmax，正确性优先，性能不是最终形态。

新增配置 / CLI：

```bash
--kv-offload-blockwise-decode
--kv-offload-blockwise-blocks N
--max-num-prefill-tokens-per-step N
```

关键实现：

- `tinyvllm/config.py`
  - `kv_offload_blockwise_decode`
  - `kv_offload_blockwise_blocks`
- `tinyvllm/utils/context.py`
  - 在 context 中传入 `kv_offload_manager`、logical block tables、context lens、write blocks、blockwise window size。
- `tinyvllm/engine/model_runner.py`
  - `prepare_decode()` 在 blockwise 模式下只预先 stage 当前 write blocks。
  - 不再要求 decode visible logical blocks 一次性全部 resident。
  - logical block rows 传给 attention，由 attention layer 按 window 触发 staging。
- `tinyvllm/layers/attention.py`
  - decode 时若 `context.kv_offload_blockwise_decode=True`：
    - 当前 token KV 先写入 staging slot。
    - 标记 write block dirty，避免窗口 staging 时丢当前层 KV。
    - 按 logical block window 调 `KVOffloadMVP0.ensure_resident()`。
    - 从当前 layer 的 physical slot gather window K/V。
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

### blockwise attention 数学回归

输出：`profile_out/blockwise_attn_online_softmax_regression_20260630.json`

| metric | value |
|---|---:|
| gate_pass | true |
| chunks | 16 |
| streamed_tokens | 2048 |
| max_abs_error | 2.98e-08 |
| relative_error | 2.31e-07 |

### 真实模型单请求 blockwise decode smoke

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

### 真实模型两请求 blockwise decode thrash

两条约 385-token prompt，`max_num_seqs=1`，`kv_offload_gpu_blocks=2`，`kv_offload_blockwise_blocks=1`，两个 request 交替触发 staging eviction/reload。

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

当前限制 / 下一步：

- decode attention 已不再需要“一次性 stage 全部 visible logical blocks”；它按 window stage。
- 已新增 blockwise/chunked prefill correctness path：`--kv-offload-blockwise-prefill` 下，chunked prefill 的 prefix read blocks 不再在 `prepare_prefill()` 一次性全量 resident，而是由 attention layer 按 window stage，并用 online softmax merge。
- 当前 blockwise prefill 仍要求 `prefix window blocks + 当前 chunk write blocks <= kv_offload_gpu_blocks`，因此需要通过 `--max-num-prefill-tokens-per-step` 控制当前 chunk 写入 blocks 数。
- 本地已通过 `py_compile`、`tools/test_chunked_prefill.py`、`tools/test_ngram_speculative.py`、`git diff --check`；本地无 torch，数学 smoke 和真实模型长 prompt smoke 待远程 GPU 运行。
- 当前 blockwise prefill/decode 都是 PyTorch correctness path，性能版需要 Triton/FlashAttention 风格的 window kernel 或能返回 logsumexp 的 flash 分块接口。

## 2026-07-02 Blockwise/chunked prefill 远程 smoke 已完成

本地 7 个改动文件已同步到远程运行目录 `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`，未同步 `.agents/`、`.codex/`、`needle_sq_results/`。

远程环境确认：

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
torch 2.4.1+cu121
torch.cuda.is_available() == True
torch.cuda.device_count() == 8
```

远程已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -m py_compile \
  tinyvllm/config.py tinyvllm/engine/model_runner.py tinyvllm/layers/attention.py \
  tinyvllm/utils/context.py tools/profile_ngram_commit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_ngram_speculative.py
```

结果：`chunked prefill tests passed`、`ngram speculative tests passed`。

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

结果：

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

结果：

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

注意：一次无效尝试使用了约 4141-token prompt，超过 `max_model_len=4096` 且超过 `kv_offload_logical_blocks=8` 可覆盖容量，触发 scheduler 空 decode assert；这不是 blockwise prefill correctness 失败。另一次复跑遇到端口占用，后续显式设置 `TINYVLLM_DIST_PORT=34567 MASTER_PORT=34567` 后通过。

后续改进机会与当前落地状态：

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
5. 待做：性能路径优化，包括合并 H2D/D2H copy、合并连续 prefetch plan、降低 clean eviction 抖动，以及后续引入 Triton/FlashAttention 风格 window kernel。
6. 部分落地：write-block staging 已抽成 `_stage_kv_offload_write_blocks()` 并同时复用于 blockwise prefill/decode；下一步继续抽象统一的 KV block access planner，让 prefill/decode 共享 `plan_read_blocks()`、`stage_blocks()`、`evict_blocks()`、`commit_write_blocks()` 语义。
7. 已落地：DFlash feasibility spike 已写入 `docs/dflash-feasibility.md`，只做接口/接入点预研，不直接实现完整 DFlash；已完成 Phase 1，把 n-gram target verify/commit 包装为通用 `verify_and_commit_block()` 并保留 n-gram 行为不变。远程 Qwen3-0.6B candidate-only smoke 已验证 `commit_event.draft_source="ngram"`。Phase 2 plumbing 已验证：新增 `--draft-source {ngram,dflash-toy,dflash-toy-ngram-or-repeat}`、`--allow-zero-accept`、deterministic `repeat_recent_tokens` toy block draft model，以及记录 zero-accept attempts 的 `verify_events`；远程 `dflash-toy` smoke `gate_pass=true`、`zero_accept_events=3`、`verify_events[].draft_source="dflash-toy"`。
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
   - 2026-07-09 fresh rerun：当前 `feat/kv-sparse-attention` 分支上重新跑多 prompt / batch shape smoke。第一次两个短自然语言 prompt 因 `ngram` 无可用 draft 失败（`verify_events=[]`、`committed=false`），确认是 prompt/draft-source 选择问题；改回 accepted-friendly 重复 prompt 与 `--draft-source dflash-toy-ngram-or-repeat`，动态端口 `27641` 通过，输出 `profile_out/dflash_phase3_draft_model_batch_shape_smoke_20260709_fresh_r2.json`，`gate_pass=true`、`num_prompts=2`、`commit_events=2`、`accepted_count=4`、`acceptance_rate=1.0`。独立 JSON 断言 `SCHEMA_VERIFY_OK`：每个 prompt 各 `commit_events=1`、`verify_events=1`、`accepted_count=2`；两个 event 均 `runtime_mutation=false`、`adapter="target_hidden_draft_model_stub"`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`。prompt0 `accepted_tokens=[13440,21619]`、`output.draft_token_ids=[7,7,7]`、first candidates `[7,1]`、`block_table_after=[0]`；prompt1 `accepted_tokens=[6303,6176]`、`output.draft_token_ids=[1,2,2]`、first candidates `[1,2]`、`block_table_after=[1]`。结论：当前分支 batch 场景 event 级 DraftModelInput schema 未发生 prompt 间串写，仍保持 profiler-only。
   - 2026-07-09 继续扩展 3 prompts / `--max-num-seqs 3` 覆盖：动态端口 `50061` 输出 `profile_out/dflash_phase3_draft_model_batch_shape_smoke_20260709_3prompt.json`，`gate_pass=true`、`num_prompts=3`、`commit_events=3`、`accepted_count=6`、`acceptance_rate=1.0`。独立 JSON 断言 `SCHEMA_VERIFY_3PROMPT_OK`：三个 prompt 均各 `commit_events=1`、`verify_events=1`、`accepted_count=2`；所有 event 均 `runtime_mutation=false`、`adapter="target_hidden_draft_model_stub"`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`。prompt0 `accepted_tokens=[13440,21619]`、`output.draft_token_ids=[7,7,7]`、first candidates `[7,1]`、`block_table_after=[0]`；prompt1 `accepted_tokens=[6303,6176]`、`output.draft_token_ids=[1,2,2]`、first candidates `[1,2]`、`block_table_after=[1]`；prompt2 `accepted_tokens=[5562,11958]`、`output.draft_token_ids=[7,7,7]`、first candidates `[7,1]`、`block_table_after=[2]`。结论：3 序列 batch 下 event-level DraftModelInput row/source schema 仍未串写，且仍只保持 profiler-only。
   - 2026-07-10 继续扩展 4 prompts / `--max-num-seqs 4` 覆盖：GPU4、动态端口 `38723` 输出 `profile_out/dflash_phase3_draft_model_batch_shape_smoke_20260710_4prompt.json`，`gate_pass=true`、`num_prompts=4`、`commit_events=4`、`commit_attempts=4`、`accepted_count=8`、`acceptance_rate=1.0`、`output_tokens=16`、`decode_steps=1`。独立 JSON 断言 `SCHEMA_VERIFY_4PROMPT_OK`：四个 prompt 均各有独立 event，所有 event 均 `runtime_mutation=false`、`adapter="target_hidden_draft_model_stub"`、`input_schema.adapter="draft-model-stub"`、`hidden_rows=3`、`logit_rows=2`、`projected_rows=3`、`draft_model_metadata.input_schema={hidden_rows=3, hidden_dim=1024, candidate_count=8, top_k=2, source_shape=[3,1024], source_dtype="torch.bfloat16", source_device="cuda:0"}`。prompt0 `accepted_tokens=[13440,21619]`、source `[7,7]`、`block_table_after=[0]`；prompt1 `accepted_tokens=[6303,6176]`、source `[1,2]`、`block_table_after=[1]`；prompt2 `accepted_tokens=[5562,11958]`、source `[7,7]`、`block_table_after=[2]`；prompt3 `accepted_tokens=[17788,6774]`、source `[1,1]`、`block_table_after=[3]`。结论：4 序列 batch 下 accepted/source/block-table 证据彼此独立，event-level DraftModelInput row/source schema 仍未串写，且仍只保持 profiler-only。
   - 2026-07-08 已把 profiler-only draft model schema 抽成 `tools/draft_model_schema.py` 小模块，包含 `DraftModelInput`、`DraftModelResult`、`DraftModelStubConfig`；`tools/profile_ngram_commit.py` 只导入这些 dataclass，测试直接加载该模块并确认 profiler 返回同一类对象。本地 `tools/test_ngram_speculative.py`、`py_compile tools/draft_model_schema.py tools/profile_ngram_commit.py tools/test_ngram_speculative.py`、`tools/test_chunked_prefill.py`、`git diff --check` 均通过；远程 `profile_out/dflash_phase3_draft_model_schema_module_smoke_20260708.json` 通过，`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`source_shape=[3,1024]`、`source_dtype="torch.bfloat16"`、`source_device="cuda:0"`，commit `draft_tokens`/`accepted_tokens=[13440,21619]` 不变。下一步若继续，建议补更多形状覆盖或做真实 draft model 接入前置检查（vocab/tokenizer/hidden_dim contract），仍不要接 runtime。
   - 2026-07-08 已补真实 draft model 接入前置 contract 检查：`tools/draft_model_schema.py` 新增 `DraftModelContract` 与 `validate_draft_model_contract()`，支持 profiler-only 校验 `expected_hidden_dim`、`target_vocab_size`、`draft_vocab_size`、`tokenizer_family` / `draft_tokenizer_family`，并记录 candidate token id min/max；本地测试覆盖 hidden_dim mismatch、candidate id 超 target vocab、tokenizer family mismatch。`run_draft_model_stub()` 默认记录宽松 `contract` metadata，也可传显式 contract；远程第一次 smoke 因 `EADDRINUSE` 失败，换动态高端口后 `profile_out/dflash_phase3_draft_model_contract_smoke_20260708.json` 通过：`gate_pass=true`、`accepted_count=2`、`runtime_mutation=false`、`contract.compatible=true`、`actual_hidden_dim=1024`、`candidate_id_min=0`、`candidate_id_max=7`，commit `draft_tokens`/`accepted_tokens=[13440,21619]` 不变。下一步若继续，建议补更多形状覆盖，仍不要接 runtime/真实 checkpoint。

2026-07-06 尝试记录 / 排坑记录：

2026-07-08 mixed scheduler token-budget reserve 已落地：

1. 改动：`Scheduler._schedule_chunked_prefill()` 增加内部 `max_prefill_tokens` 参数，`_schedule_mixed_prefill_decode()` 给 decode query row 预留并持续检查 `max_num_batched_tokens`，避免 mixed batch 实际 `prefill_tokens + decode_rows` 超过 token budget。
2. 新增测试：`test_mixed_prefill_reserves_token_budget_for_decode_queries()`、`test_mixed_decode_rows_respect_remaining_token_budget()`，覆盖 tight-budget 下 short prompt batching 与多 decode rows。
3. 本地验证通过：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py`、`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python3 tools/test_profile_chunked_prefill.py`、`PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache python3 -m py_compile tinyvllm/engine/scheduler.py tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/sequence.py tinyvllm/config.py tools/profile_chunked_prefill.py tools/test_chunked_prefill.py`、`git diff --check`。
4. 远程纯测试通过：同步 `tinyvllm/engine/scheduler.py`、`tools/test_chunked_prefill.py` 后，远程 `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/test_chunked_prefill.py` 与 `tools/test_profile_chunked_prefill.py` 均通过。
5. GPU profiler smoke 暂未形成性能数字：远程 `tools/profile_chunked_prefill.py --mode mixed ... --max-num-batched-tokens 129` 启动后 log 为 0 字节，进程进入 D-state（`STAT=D`、`WCHAN=os_acquire_rwlock_write`），`kill -9` 后仍需等待内核态返回；同时 SSH 偶发 `Connection closed by UNKNOWN port 65535`。这次应记录为远程 GPU/driver/链路环境阻塞，不作为代码回归。
6. 文档：`docs/qwen3-8b-fixes.md` 已追加 `47.39.6 Mixed token-budget reserve`。

2026-07-08 mixed first-chunk budget clamp 已落地：

1. 问题：`max_num_prefill_tokens_per_step > mixed 剩余 token budget` 时，首个 prefill chunk 会过大，导致 decode row 无法追加，mixed 退化成普通 prefill。
2. 改动：`_schedule_chunked_prefill(max_prefill_tokens=...)` 把 token budget 传给首个 waiting seq、已有 `prefilling` seq 和后续 short prompt batching；`_schedule_one_prefill_chunk(..., max_chunk_tokens=...)` 按 `max_num_prefill_tokens_per_step`、`max_chunk_tokens`、剩余 prompt tokens 三者取最小值。
3. 新增测试：`test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget()`，覆盖 `max_num_batched_tokens=5`、`max_num_prefill_tokens_per_step=8` 下 prefill chunk 缩到 4 并保留 decode row。
4. 本地与远程纯测试均通过：`tools/test_chunked_prefill.py`、`tools/test_profile_chunked_prefill.py`；本地 `py_compile` 与 `git diff --check` 通过。
5. 文档：`docs/qwen3-8b-fixes.md` 已追加 `47.39.7 Mixed first-chunk budget clamp`。

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

## 远程 S4 smoke 示例

```bash
ssh sitian@10.232.195.203 'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge && CUDA_VISIBLE_DEVICES=7 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python tools/profile_ngram_commit.py \
  --model /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B \
  --max-output-len 64 \
  --temperature 0.0 \
  --ngram-size 5 \
  --max-draft-tokens 4 \
  --max-commit-events 0 \
  --mode candidate-only \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 16 \
  --out-json profile_out/ngram_spec_s4_06b_candidate_n5d4_verify.json'
```

2026-07-14 继续推进：Light Doc Cache multi-target gate 已完成，最终结论
`NO_GO`。

实现 commits：

- `1cfab07` Add Light Doc Cache target matrix
- `67848ec` Add Light Doc Cache multi-target gate
- `e0417ce` Extract fixed Light Doc Cache calibration bank
- `d41a3fb` Add Light Doc Cache multi-target driver
- `f8538f3` Add Light Doc Cache multi-target remote smoke
- `b085ba0` Calibrate Light Doc Cache target lengths

实现和验证要点：

- 固定一个 calibration bank，8 个 target 全部复用同一 bank。
- 每个 target 比较：
  - `repeat_last_target`
  - `correlated_same_layer_target`
  - `calibration_holdout`
- 远端 tokenizer 实际 bucket：
  - short `31/36`
  - medium `79/51/52`
  - long `212/202/229`
- 2-target smoke 首次发现三模式 logits 完全相同；根因是现有 sidecar
  read-path 把预分配 KV 的物理 block 轴当连续序列 token 轴。复用同一 LLM
  后 `Sequence.block_table` 不再从 block 0 开始，恢复数据写错物理槽。
- 已在 default-off smoke 边界修复：按 `seq.block_table` pack sequence KV，
  restore 后 scatter 回原物理 blocks，再做临时 KV pointer swap。没有改
  scheduler、attention kernel、KV allocation lifetime 或 slot mapping。

最终 canonical artifact：

- 本地：
  `experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714/`
- 远端：
  `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge/profile_out/light_doc_cache_multi_target_20260714_final_evidence`
- Remote: `sitian@10.232.195.203`
- Hostname: `n232-195-203`
- GPU: `4`
- Dynamic port: `60495`
- Bank SHA256:
  `92ab5801523c85faa5e315cc229381818f960cc69940a4de6579688bd8e1fcc0`

独立审计：

```text
MULTI_TARGET_AUDIT_OK NO_GO
BANK_SHA256 92ab5801523c85faa5e315cc229381818f960cc69940a4de6579688bd8e1fcc0
ROWS 24 ATTEMPTS 24
BUCKET_COUNTS {'medium': 3, 'long': 3, 'short': 2}
```

聚合结果：

- `repeat_last_target`
  - mean logit diff `0.637132`
  - argmax `5/8`
  - mean missing MSE `12.2817`
- `correlated_same_layer_target`
  - mean logit diff `0.642408`
  - argmax `4/8`
  - mean missing MSE `6.28873`
- `calibration_holdout`
  - mean logit diff `1.31780`
  - argmax `5/8`
  - mean missing MSE `13.8190`

Gate：

- 8/8 paired targets completed：pass
- holdout argmax rate not lower：pass
- holdout wins >=5：fail，实际 `0/8`
- aggregate mean logit diff improves >=5%：fail，实际回退 `105.13%`
- worst relative regression <=25%：fail，实际 `322.98%`
- no correlated argmax match regressed：fail，`repetitive` 回退
- actual token buckets valid：pass

最终决策：`NO_GO`。

后续边界和方向：

- 不要在同一 8-target gate 上继续调这个 holdout selector，避免 target
  leakage / overfit。
- 不进入 attention hot path 或 physical KV allocation。
- 当前约 `17.50%` 只是 logical byte accounting，不是 GPU 显存、吞吐或
  task-quality 证明。
- 下一主分支改做 APC/shared-prefix benchmark 或 adaptive speculative
  decoding；Light Doc Cache selector 暂停。

## 2026-07-15 Adaptive n-gram speculative decoding canonical gate

### 状态与环境

- Branch：`feat/adaptive-ngram-speculation`
- Canonical source commit：`08b122daedc8ab531a5d301f0b5a82b5cb1997e5`
- Source dirty：`false`
- Run tag：`qwen3-06b-canonical-20260715-025426`
- Remote：`sitian@10.232.195.203`
- Remote Python：`/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- Model：`/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- GPU：`3`
- Isolated remote directory：
  `/data00/home/sitian/sitian-workspace01/tllm/adaptive-ngram-gates/qwen3-06b-canonical-20260715-025426`
- Local canonical artifacts：
  `experiments/adaptive_ngram/qwen3-06b-canonical-20260715-025426/`

实现范围严格保持为 profiler-owned、greedy、单序列。Adaptive state 只选择下一次 n-gram proposal cap `K∈{1,2,4}`；没有把 adaptive 状态接入 scheduler、`Sequence`、`LLM.generate()`、target logits、accepted-prefix rule、EOS rule 或 normal decode fallback。为保证 speculative accepted-token block/hash 生命周期正确，仅修复了 `block_manager.py` 的既有边界，并由 block-boundary regression 覆盖。

### 运行命令

Canonical 初始运行在远端完成 43/140 后因本地 Kerberos 票据过期中断；恢复系统 API cache 后，使用同一 run tag、同一 isolated remote directory 和 `RESUME=1` 续跑。Resume 只接受已有唯一合法 row，不平均重复或 partial row。

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram

RUN_TAG=qwen3-06b-canonical-20260715-025426 \
LOCAL_OUT=/Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/adaptive_ngram/qwen3-06b-canonical-20260715-025426 \
CUDA_VISIBLE_DEVICES=3 \
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B \
RESUME=1 \
SSH_CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
tools/run_adaptive_ngram_gate_remote.sh canonical
```

等价的 canonical 新跑入口为：

```bash
RUN_TAG="qwen3-06b-canonical-$(date +%Y%m%d-%H%M%S)" \
  tools/run_adaptive_ngram_gate_remote.sh canonical
```

Runner 上传当前最小 source snapshot 到隔离目录，远端 preflight 后逐进程运行，每个模型进程使用不同的动态 `TINYVLLM_DIST_PORT` 和 `MASTER_PORT`，最后下载 artifacts 并执行本地 verifier。

### Smoke 与 canonical 覆盖

- 最终 one-repetition smoke：
  `qwen3-06b-smoke-r4-20260715-022440`
- Smoke：20/20 rows、1 repetition、source commit `08b122d`、`source_dirty=false`
- Smoke verifier：correctness、trajectory replay、adaptive exercise 全部通过；决策为 provisional `NO_GO`
- Canonical：140/140 unique rows
  - 4 prompt classes
  - 5 isolated policies
  - 7 repetitions
  - 每个 policy 28 rows
  - 每个 prompt 35 rows
  - 每次 repetition 20 rows
  - 140 个 process JSON 对应 140 个独立模型进程
  - 280 个互异动态端口（每进程各一个 distributed port 和 master port）
  - 0 process failures

逐-run stdout/stderr 和 process JSON 保留在远端 isolated directory；Git 中 canonical 目录只持久化规范要求的五件套：

```text
manifest.json
raw_rows.json
event_rows.json
summary.json
report.md
```

### Verifier 与 artifact 完整性

Runner 的远端 verifier 和下载后的本地 verifier 均成功。第二层独立审计输出：

```text
ADAPTIVE_NGRAM_CANONICAL_AUDIT_OK NO_GO
SOURCE_COMMIT 08b122daedc8ab531a5d301f0b5a82b5cb1997e5
ROWS 140
POLICY_COUNTS {'adaptive': 28, 'baseline': 28, 'fixed_k1': 28, 'fixed_k2': 28, 'fixed_k4': 28}
PROMPT_COUNTS {'natural_prose': 35, 'repeated_long_context': 35, 'structured_mixed': 35, 'transition_heavy': 35}
REPETITION_COUNTS {0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}
```

远端与下载后本地五件套 SHA256 逐文件一致：

```text
manifest.json   f350289adaf4cfa71c523beda72291d0bb44b589623142778f3c0edc065353e4
raw_rows.json   cad13ba325dfd9b0bc48e9f4355ddac71aae25917033a58b46086ed09b158ac0
event_rows.json b975ccdadf5eb1c88fcfd8b5bc7ef0075cbec11c7f45480dea5b3ef9d15834ea
summary.json    391f3c311b0f2c59a3e3ea07c3c9da3b850491b1844c4318e455e4c550f0b42e
report.md       e3066e313c516f414762715518a8acad01996207d3e09615ac7772cfb3e795a5
```

### Canonical 性能结果

| Policy | aggregate median tok/s | drafted tokens | accepted tokens | token acceptance |
|---|---:|---:|---:|---:|
| normal greedy | 33.815941 | 0 | 0 | n/a |
| fixed K1 | 32.979915 | 742 | 700 | 94.34% |
| fixed K2 | 32.925531 | 1008 | 931 | 92.36% |
| fixed K4 | 37.962906 | 1288 | 1120 | 86.96% |
| adaptive | 37.574839 | 1197 | 1092 | 91.23% |

| Prompt | normal greedy | fixed K1 | fixed K2 | fixed K4 | adaptive |
|---|---:|---:|---:|---:|---:|
| natural prose | 33.956405 | 33.435853 | 33.271282 | 31.579000 | 32.969817 |
| structured mixed | 33.719898 | 33.486046 | 33.077964 | 32.191697 | 33.160628 |
| repeated long context | 33.935897 | 33.026123 | 32.911202 | 46.905002 | 46.608547 |
| transition heavy | 33.996681 | 32.474497 | 32.921986 | 41.950054 | 39.137848 |

聚合 waste / zero-accept：

| Metric | fixed K4 | adaptive | adaptive relative change |
|---|---:|---:|---:|
| median wasted draft tokens | 24 | 15 | -37.50% |
| median zero-accept verify cost | 224.830 ms | 180.475 ms | -19.73% |

Adaptive event exercise：

- Selected K counts：`K1=21`、`K2=70`、`K4=259`
- Transition counts：
  `full_accept_streak=154`、`hold=28`、`promote=140`、
  `weak_acceptance=7`、`zero_accept=21`
- 所有 adaptive event 可由 event record 独立 replay；selected K 只出现 `1/2/4`

### 固定 gate 判定

固定阈值没有在观察结果后修改：

| Gate | Threshold | Observed | Result |
|---|---:|---:|---|
| adaptive vs normal greedy | `>= +5%` | `+11.1158%` | PASS |
| adaptive vs best fixed direct | `>= +2%` | `-1.0222%` | FAIL |
| near-best fixed fallback | `>= -1%` | `-1.0222%` | FAIL |
| waste reduction vs K4 | `>= 20%` | `37.50%` | PASS |
| zero-accept cost reduction vs K4 | `>= 15%` | `19.7282%` | PASS |
| natural prompt / baseline | `>= 0.95` | `0.970945` | PASS |
| transition-heavy / baseline | `>= 0.95` | `1.151226` | PASS |
| exact output correctness | required | pass, no failures | PASS |
| trajectory replay | required | pass, no failures | PASS |
| adaptive exercise | required | levels/reasons complete | PASS |

最终决策严格为：

```text
NO_GO
reason: adaptive_vs_fixed_gate_failed
```

虽然 adaptive 相对 normal greedy 明显加速，并同时改善 K4 waste 和 zero-accept cost，但它相对最佳 fixed K4 的 `-1.0222%` 刚刚越过预先固定的 `-1%` fallback 下限，因此不能判 GO，也不能事后把阈值放宽到包住结果。

### 结果证明与边界

该 canonical gate 证明：

1. 在记录的 Qwen3-0.6B、greedy、单序列 prompt bank 上，adaptive 与 fixed speculative 输出都和 normal greedy token 序列一致。
2. 两阶段 verifier 与 accepted-token block/hash lifecycle 在该覆盖内通过 correctness 和 trajectory replay。
3. Adaptive 状态机确实覆盖 `K=1/2/4` 以及 promotion、weak demotion、zero-accept demotion 和 hold。
4. Adaptive 相比 normal greedy 有 `+11.12%` aggregate median tok/s，且比 K4 少 waste / zero-accept verify cost。
5. 在预先固定的综合 gate 下，当前 adaptive policy 没有战胜最佳 fixed policy，因此不能成为默认策略。

该 gate不证明：

- ragged 或 batched target verification correctness；
- production batch throughput、scheduler interaction 或 queueing-tail latency；
- GPU memory-capacity reduction；
- sampling/non-greedy correctness；
- 其他模型、prompt 分布、GPU 或并发负载上的迁移收益。

### Prompt-to-artifact completion audit

1. **Exact adaptive transition rules**：`tools/test_ngram_speculative.py` 覆盖初始化、promotion、弱接受降级、zero-accept 直降、no-match 不更新和 JSON replay；canonical `event_rows.json` trajectory replay pass。
2. **Unchanged verify/commit semantics**：变更列表未修改 `Sequence`、scheduler、`LLM.generate()` 或 `LLMEngine.step()`；candidate 与同 prompt baseline token SHA/列表完全一致。`block_manager.py` 仅修复 accepted-token block/hash lifecycle，并由 `tools/test_chunked_prefill.py` block-boundary regression 覆盖。
3. **Single sequence**：140/140 raw rows 均 `max_num_seqs=1`，每个 subprocess 只传一个 literal prompt。
4. **Five isolated policies**：baseline/fixed K1/fixed K2/fixed K4/adaptive 各 28 个独立 process rows，共 140 unique run keys。
5. **Four prompt classes**：manifest 固定 natural、mixed、high-repeat、transition-heavy 四类；summary 有四类 per-prompt median。
6. **Seven repetitions**：repetition `0..6` 各 20 rows。
7. **Dynamic distinct ports**：140 个进程记录 280 个互异 `TINYVLLM_DIST_PORT`/`MASTER_PORT`；只有 `EADDRINUSE` 允许受控重试。
8. **Actual model identity**：manifest 记录实际路径 `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`，remote validation 读取 `config.json` 并确认 `model_type=qwen3`、0.6B identity。
9. **Mandatory correctness**：`correctness_pass=true`、`correctness_failures=[]`、`trajectory_replay_pass=true`、`trajectory_failures=[]`。
10. **Fixed thresholds**：`summary.json` 保留设计时固定阈值；独立审计复算得到同一 `NO_GO`，未根据结果调整。
11. **Canonical artifact set**：Git 中目录精确包含 manifest/raw rows/event rows/summary/report 五件套。
12. **Remote and post-download verification**：runner 远端 verify、下载后本地 verify、`/tmp/audit_adaptive_ngram_canonical.py` 独立审计均通过；五件套远端/本地 SHA256 一致。
13. **README and handoff**：`README.md` 记录复现入口、关键数字、NO_GO 原因和边界；本节记录完整环境、命令、证据链和后续方向。
14. **Claim boundaries**：`report.md`、`README.md` 和本节均明确限制为 greedy single-sequence profiler-owned Qwen3-0.6B，不外推 batch/queue/memory/其他模型。

结论：14 项均有直接 artifact 或代码/测试证据，不依赖单一 green proxy。Canonical adaptive gate 已完成，但产品策略结论是 `NO_GO`。

### 下一方向

保留 adaptive policy、两阶段 verifier、block-boundary regression、deterministic matrix、resume、artifact verifier 和报告生成器。不要在同一 4-prompt gate 上继续调 EMA/阈值以追赶 `0.0222` 个百分点，避免 benchmark overfit。

短期只在已验证的高重复、greedy、单序列 regime 中把 fixed K4 作为可选 profiler policy，不能设为通用默认。下一主线优先做更高质量 draft source（例如真实小 draft model / 更强 retrieval draft）的 profiler-owned gate；进入 runtime 前仍须重新完成 exact-output、ragged batch、scheduler/load 和 memory-capacity gate。

## 2026-07-20 P5 decode-SLO-aware mixed admission smoke

### 当前状态

- Branch：`feat/adaptive-ngram-speculation`
- Source commit：`0fa05fd5a2e9534d0eeb17bb5c2108687e4488df`
- Source dirty：`false`
- Source tree SHA256：
  `bcbba4261e76cf3559c45d67e27208c38104cda560caa32a46d440ada8a31d3f`
- Fresh preflight：
  `experiments/arrival_load/qwen3-06b-p5-preflight-20260720-233159/`
- Fresh smoke：
  `experiments/arrival_load/qwen3-06b-p5-smoke-20260720-233216/`
- Remote：
  `sitian@10.232.195.203`
- Remote Python：
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- Model：
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`

### 本轮修复

提交 `0fa05fd` 修复了 smoke independent verifier：

1. smoke 顶层 hash contract 不再错误要求四个 authoritative cost
   calibration artifacts；
2. verifier 会从 `provisional_cost_calibration/` 独立重算 capacity、
   shape manifest、raw rows 和 summary；
3. provisional summary SHA、source/environment/engine identity、P5
   coefficients 和 smoke manifest marker 必须一致；
4. `actual_prefill_tokens` / `scheduled_decode_seq_ids` 按 scheduler
   producer 的真实语义验证：仅 SLO mixed admission 决策记录这些局部
   字段，普通 no-running prefill 和 decode-only branch 保持 `0` / `[]`；
5. verifier 从 scheduler trace 独立重建完整 `p5_smoke` summary。

严格 TDD：

- RED 复现：
  `ValueError: missing artifact: cost_calibration_capacity.json`
- 第二条 RED 复现普通 no-running prefill 被误判：
  `ValueError: P5 actual prefill mismatch`
- GREEN 后完整本地 suite：
  - `python3 tools/test_arrival_load_cost_calibration.py`
  - `python3 tools/test_arrival_load_gate.py`
  - `python3 tools/test_arrival_load_verify.py`
  - `python3 tools/test_run_arrival_load_gate_remote.py`
  - `python3 tools/test_chunked_prefill.py`
  - relevant `python3 -m py_compile`
  - `git diff --check`

以上全部 PASS。

### Fresh source-bound 远程结果

Fresh preflight 在远端重新运行 cost/gate/driver/verifier/chunked-prefill
tests，全部 PASS。

Fresh smoke：

- remote exit code：`0`
- independent verifier exit code：`0`
- provisional capacity：`2075` KV blocks
- required calibration shapes：`40`
- completed calibration shapes：`40/40`
- raw calibration rows：`280`
- failed attempts：`0`
- lifecycle complete：`true`
- exact outputs：`true`
- case count：`2`

Recorded summary 与 independent verifier 重建结果完全一致：

```json
{
  "classification": "INCOMPLETE",
  "lifecycle_complete": true,
  "exact_outputs": true,
  "case_count": 2,
  "p5_smoke": {
    "classification": "INCOMPLETE",
    "demand_activation_count": 1,
    "largest_chunk_admission_count": 0,
    "smaller_chunk_admission_count": 0,
    "distinct_selected_chunk_tokens": 0,
    "slo_suppression_count": 150,
    "draining_decision_count": 0
  }
}
```

### 为什么没有 mixed admission

Fresh provisional envelope：

- cost intercept：`59,859,958 ns`
- cost per prefill token：`613,182 ns`
- P5 target gap：`64,000,000 ns`
- reserve：`8,000,000 ns`
- minimum chunk：`16` tokens

最大理论 remaining slack（decode age 为 0）：

```text
64,000,000 - 8,000,000 = 56,000,000 ns
```

最小 16-token chunk 的预测成本：

```text
59,859,958 + 16 * 613,182 = 69,670,870 ns
```

因此：

```text
69,670,870 ns > 56,000,000 ns
```

当前固定 SLO 合同下不存在任何可行 token ladder 项。150 次
`cost_suppressed` 是 fail-closed 的正确结果，不是 smoke workload
偶然没有覆盖 mixed path，也不能通过增加 waiting depth 修复。

### 决策与边界

- 当前 P5 smoke 结论严格为 `INCOMPLETE`，不是 `SMOKE_ONLY`。
- 不得把该 smoke 用作 authoritative cost calibration predecessor。
- 不继续 workload calibration 或 canonical 54 cases。
- 不能宣称任何吞吐、延迟或显存性能提升。
- README 不更新。
- P4 仍为 `NO_GO`；P5 目前证明的是实现与证据链能正确 fail-closed，
  不是性能收益。

若继续 P5，必须先做新的书面设计决策，而不是调 smoke workload：

1. 修改 target gap / reserve / minimum chunk 的产品 SLO contract；或
2. 降低真实 mixed-step intercept，使最小 chunk 在 56ms budget 内可行；
   或
3. 明确放弃当前 P5 参数组合并转向其他优化。

任一 source 或固定合同变更后，都必须重新 fresh preflight → smoke；
只有 independent verifier 给出 `SMOKE_ONLY` 才能进入后续 authoritative
cost/workload/canonical 链。

## 2026-07-22 Exact CUDA Graph 8-Case Fallback Gate

已完成并提交 production exact-width multi-sequence CUDA Graph 的八类
terminal budget/fault fallback evidence 链：

- `6125a13`：批准设计；
- `1244058`：批准 implementation plan；
- `0afea3a`：冻结 `budget_fallback_rows.jsonl` 合同；
- `41966ba`：独立 verifier 重建 8/8 raw evidence；
- `e86ae1e`：八个隔离 worker 的命令、编排、聚合与 arrival binding；
- `3faa761`：harness-only fault precondition/runtime wrapper、真实 worker、
  atomic artifacts 与 structured incomplete。

生产源码没有 fault switch。以下字符串在
`tinyvllm/config.py`、`tinyvllm/engine/model_runner.py`、
`tinyvllm/engine/exact_cuda_graph_cache.py` 中均无匹配：

```text
budget_fallback_reason
fault_injection
TINYVLLM_CUDA_GRAPH_FAULT
inject_exact_cuda_graph
```

本地 fresh 验证：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_model_runner_spec_verify.py
```

结果：

```text
multi-sequence cuda graph gate tests passed
model runner spec_verify tests passed
```

Source-bound remote preflight：

```text
run tag:
qwen3-06b-exact-cuda-graph-fallback-preflight-20260722-180854
base commit:
3faa761aaa2501603cd0bec4533bfbb3044dae61
dirty:
false
tree SHA:
b66074f3dd178894cb3b999513105af36b345b2dc771be641671283abeb643bf
classification:
preflight exit 0
```

Artifact：

```text
experiments/cuda_graph/
qwen3-06b-exact-cuda-graph-fallback-preflight-20260722-180854/
```

Correctness/fault smoke 已实际调用 gate，但在任何 model worker 启动前被
GPU isolation gate 阻塞：

```text
run tag:
qwen3-06b-exact-cuda-graph-fallback-correctness-smoke-20260722-181215
classification:
INCOMPLETE
failure_reason:
unrelated_gpu_occupancy
stage:
before_worker
```

Artifact：

```text
experiments/cuda_graph/
qwen3-06b-exact-cuda-graph-fallback-correctness-smoke-20260722-181215/
incomplete.json
```

GPU 0 当时有八个 unrelated 长期服务进程：

```text
330288   root      /opt/tiger/qwen3vl-encoder/main_model.py
1302034  root      /opt/tiger/manhattan_runtime/logs/manhattan_worker.py
1302036  root      /opt/tiger/manhattan_runtime/logs/manhattan_worker.py
2811316  sunchao+  search.bert_qrec.0709 main_model.py
2812359  sunchao+  search.bert_qrec.0709 main_model.py
3472365  sunchao+  search.bert_qrec.0709 main_model.py
3473815  sunchao+  search.bert_qrec.0709 main_model.py
4039206  wangmin+  search.bert_qrec.7.16 main_model.py
```

没有 kill、没有切 GPU、没有修改远端 checkout。下一步只能在 GPU 0
无 unrelated occupancy 时，使用新 run tag 重跑
`correctness-smoke`。只有 independent verifier 得到：

```text
classification = NON_AUTHORITATIVE_SMOKE
budget_fallback_required = 8
budget_fallback_verified = 8
```

才能进入 canonical correctness；canonical correctness 独立 `GO` 后
才能进入 arrival canonical。当前仍不能宣称吞吐、延迟或显存性能提升，
也不能更新 README。

### 2026-07-22 Fault Evidence Completion Audit

在远端 GPU 0 持续被无关服务占用期间，完成了不依赖 GPU 的定向
completion audit。审计范围为 `7db219a..HEAD`，覆盖：

- `tools/multi_sequence_cuda_graph_contract.py`
- `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`
- `tools/verify_multi_sequence_cuda_graph_production.py`
- `tools/test_multi_sequence_cuda_graph_gate.py`

确认：

1. fault injection 只存在于 gate harness，production config、cache 和
   `ModelRunner` 没有 fault switch；
2. 八个 fault worker 使用隔离进程和独立动态端口；
3. fault worker 的 `case_summaries` 必须为空，不能进入 throughput、
   latency、memory、initialization 或 graph-hit performance matrix；
4. runtime method mutation 在 `_run_budget_fallback_phase()` 的
   `finally` 中恢复，恢复状态在 worker artifact 构造前写回；
5. arrival modes 不重跑 fault worker，只能绑定 source-matched、
   independently verified canonical correctness evidence。

审计发现并修复了一个 verifier 证据绑定缺口：此前
`_validate_budget_fallback_rows()` 没有使用 `correctness_rows.jsonl`，
因此只验证 `budget_fallback_rows.jsonl` 内部自洽。现在每个 fault
case 必须恰好存在一条同 `case_id` 的 correctness row，并交叉绑定：

- candidate/reference output token IDs；
- `logits_close` / `logits_allclose`；
- candidate/reference live-KV SHA。

TDD RED 证据：

```text
test_production_verifier_rejects_budget_fallback_tampering
AssertionError:
result["classification"] was not NO_GO
```

修复后 fresh GREEN：

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_model_runner_spec_verify.py
```

结果：

```text
multi-sequence cuda graph gate tests passed
model runner spec_verify tests passed
```

2026-07-22 18:17–18:22 CST 连续十次只读轮询 GPU 0，每 30 秒一次，
始终为相同八个 unrelated compute processes，约占 13.3 GiB。未启动
新 model worker。Task 6 仍是环境阻塞的 `INCOMPLETE`，本次 verifier
加固尚未取得新的 remote source-bound preflight/smoke artifact。
