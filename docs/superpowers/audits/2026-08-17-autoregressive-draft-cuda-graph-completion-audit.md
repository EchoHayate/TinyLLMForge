# Autoregressive Draft Exact-Shape CUDA Graph Completion Audit

**Date:** 2026-08-17

**Repository:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

**Runtime correctness classification:** `ESTABLISHED_IN_RECORDED_TP4_B4_Q4_SCOPE`

**Controlled performance classification:** `INCONCLUSIVE_ENVIRONMENT_PARTIAL_4_OF_8`

**Promotion:** `NOT_PROMOTABLE`

## Executive Decision

The current tree implements and has executed the approved default-off
exact-shape CUDA Graph path for the independent Qwen3 learned drafter:

```text
tensor parallel size: 4
batch size:           4
proposal length:      4
sampling:             greedy
Proposal-KV:          dense direct allocation
Proposal-KV offload:  disabled
shape policy:         exact only; no padding or rounding
```

Real TP4 execution now establishes all of the following in the recorded
scope:

- CUDA Graph capture and replay completed on all four ranks;
- production-default capture budgets admitted the real graph without an
  override;
- eager and graph target tokens were exact;
- eager and graph proposal rows were exact;
- eager and graph accepted-prefix counts were exact;
- transaction digests were identical;
- active Proposal-KV transactions returned to zero;
- accepted-prefix commit and rejected-suffix rollback remained authoritative;
- graph reset and CUDA synchronization completed before process-group
  destruction; and
- GPU 0 through GPU 3 returned to zero allocated MiB after successful
  diagnostic workers.

The initial two-second single-capture ceiling was too small for the real
TP4/B4/Q4 graph. A high-budget diagnostic measured approximately
`2.741-2.750 s` per rank and proved that retained graph memory was small:
`8,520,704 bytes` reserved and `53,408 bytes` static per rank. The production
default was therefore changed only from two seconds to four seconds:

```text
single capture ceiling: 4 s
total capture ceiling:  5 s
reserved budget:        512 MiB
static budget:          64 MiB
```

A production-default graph worker then completed with no override:

```text
captures per rank:       1
replays per rank:        1
quarantines per rank:    0
fallback_pre_replay:     0
capture time:            approximately 3.094-3.098 s
transaction digest:      d102ac0a8da78766a7e53285bb32c59d1f4026c97555a761628ff2afc4942989
active transactions:     0
accepted / proposed:     51 / 70
acceptance rate:         0.7285714285714285
```

A same-source single eager worker produced exact output, proposal,
accepted-prefix, transaction, and acceptance parity. That one-pair pilot
showed graph throughput approximately `+2.76%`, but it is not a controlled
performance conclusion because it contains one sample and substantial
bootstrap/capture variance.

The required source-bound campaign completed both warmup pairs and four of
eight measured position-balanced pairs. All four completed pairs preserved
exact correctness and all-rank replay. Before pair 4 could finish, an
unrelated root-owned `VLLM::EngineCore` appeared on physical GPU 3 and
consumed approximately `73.3 GiB`. The gate's rank 3 exited and the remaining
ranks stopped making progress. Only the gate's own process group was
terminated; the external process was not signaled or modified.

The retained four-pair aggregate is:

```text
order counts:                    eager_graph=2, graph_eager=2
exact parity:                    4 of 4
all-rank capture/replay:         4 of 4
median eager throughput:         0.989071982502008 tok/s
median graph throughput:         0.9749348881219722 tok/s
mean paired throughput delta:   -0.002993426456363135 tok/s
median eager TPOT:               1.9743923907000003 s
median graph TPOT:               2.0375849077333337 s
observed capture range:          1.758702684-3.101219179 s
```

Because the contract requires eight measured pairs and a paired bootstrap
confidence interval, the partial aggregate cannot be classified as `GO` or
`NO_GO_PERFORMANCE`.

Strict result:

```text
LOCAL_EXACT_GRAPH_IMPLEMENTATION=ESTABLISHED
REAL_TP4_CUDA_GRAPH_CAPTURE_REPLAY=ESTABLISHED
REAL_EAGER_GRAPH_CORRECTNESS_PARITY=ESTABLISHED
PRODUCTION_DEFAULT_CAPTURE_BUDGET=ESTABLISHED
GRAPH_PROCESS_GROUP_TEARDOWN_ORDER=ESTABLISHED
CONTROLLED_EIGHT_PAIR_PERFORMANCE=INCONCLUSIVE_ENVIRONMENT_PARTIAL_4_OF_8
FINAL_PROMOTION=NOT_PROMOTABLE
```

## Runtime and Lifecycle Changes

### Capture budget

`tinyvllm/config.py` keeps graph mode default-off and changes only:

```text
autoregressive_draft_cuda_graph_max_single_capture_ns:
  2_000_000_000 -> 4_000_000_000
```

The total capture, reserved-memory, static-memory, topology, batch, proposal,
sampling, allocator, and offload constraints remain unchanged.

### Capture convergence and rollback

`tinyvllm/engine/autoregressive_draft_graph.py` now:

- requires the capture backend to implement `release(entry)`;
- converges TP ranks before capture and after capture completion;
- rolls back scratch state on local or converged capture failure;
- records bounded quarantine error details;
- releases retained graph resources when an identity is quarantined;
- releases newly captured resources on rollback, identity, or budget failure;
- releases a replay entry after replay-started failure;
- exports retained static/reserved/capture-time resources; and
- provides idempotent graph-runner teardown.

### CUDA Graph release order

`tinyvllm/engine/qwen3_draft_cuda_graph_backend.py` releases an entry as:

```text
CUDAGraph.reset()
torch.cuda.synchronize()
```

The executor exposes idempotent `close()`, and `ModelRunner.exit()` invokes it
before shared-memory close, `dist.barrier()`, and process-group destruction.

The pure TP4 collective diagnostic captured and replayed all-reduce and
broadcast on all four ranks, reset the graph, synchronized CUDA, and then
destroyed the process group with exit code zero. This resolves the prior
`ProcessGroup abort timed out` lifecycle failure.

### Sparse logical context and dense physical slots

The Qwen3 graph backend and capture scratch owner no longer require logical
committed-entry count to equal dense context-token count. Proposal-KV logical
identities may be sparse while the allocator supplies dense physical source
slots. Exact sequence ownership and readable-lease validation remain
required.

## Evidence Inventory

### Pure collective lifecycle

```text
artifacts/autoregressive_draft_cuda_graph/
  20260817-tp4-collective-lifecycle-green/result/summary.json
```

Recorded behavior:

- all-reduce result `10.0` on every rank;
- broadcast result `7.0` on every rank;
- graph reset before process-group destruction; and
- four clean rank exits.

### High-budget root-cause diagnostic

```text
artifacts/autoregressive_draft_cuda_graph/
  20260817-lifecycle-green-v5-budget-diagnostic-tp4-b4-q4/
    diagnostics/high-budget-graph/result.json
```

This diagnostic established that the original quarantine was caused by the
two-second single-capture ceiling rather than static or reserved-memory
budgets.

### Production-default graph and eager pilot

```text
artifacts/autoregressive_draft_cuda_graph/
  20260817-lifecycle-green-v6-production-default-tp4-b4-q4/
    diagnostics/production-default-graph/result.json
    diagnostics/production-default-eager/result.json
```

The two workers use the same frozen source, checkpoints, prompts, TP4/B4/Q4
shape, proposal policy, and output length. Their exact correctness fields are
identical.

### Partial source-bound paired gate

```text
artifacts/autoregressive_draft_cuda_graph/
  20260817-production-default-paired-gate-tp4-b4-q4/
    INTERRUPTED.md
    partial-summary.json
    partial-remote/provenance.json
    partial-remote/environment.json
    partial-remote/source_manifest_seed.json
    partial-remote/workers/
```

Bound provenance:

```text
source commit:
  8e370881a769bbac3c70e2cd714d815e51c46fc8

source patch SHA256:
  0773c6a723967c4af6ecb7e4ab7a45f1c2b48d0e891b4eae8bef31c13d066c56

source tree SHA256:
  c0992a8c69fe7e25646288f0e27da2723cba33bf814fb272180652118c414c88

target model fingerprint:
  3c35f724f7046d02814381df890003817d8332050dc8d3ddb849df6999d1f3a0

draft model fingerprint:
  f0f36088b9a833762a704f27998eb199ecba8dba1e81f34562feea2a5e13ad6e

tokenizer fingerprint:
  e5d9359c3484afd3a8a9170f0a6b008cb54d35c9317f41d8ec549ae6b2fedf7f
```

The interrupted run did not produce the final canonical `result.json`,
archived verifier receipt, local verifier receipt, or final checksum manifest.
Those absences are expected consequences of stopping at four measured pairs.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Default-off graph mode | config and validation tests | `ACHIEVED` |
| Only TP4/B4/Q4/greedy/dense-direct/no-offload | admission tests and real workers | `ACHIEVED` |
| Exact identities without padding or rounding | graph policy tests | `ACHIEVED` |
| Capture after successful eager observations | graph state-machine tests and real counters | `ACHIEVED` |
| Private capture scratch | scratch ownership tests | `ACHIEVED` |
| Exact proposal lifecycle | executor/Proposal-KV tests and real digest parity | `ACHIEVED` |
| TP capture failure convergence | failure-injection tests | `ACHIEVED` |
| Release before process-group destruction | collective diagnostic and integration tests | `ACHIEVED` |
| Real TP4 capture/replay | v5, v6, and four measured graph workers | `ACHIEVED` |
| Production-default capture/replay | v6 result | `ACHIEVED` |
| Real eager/graph token parity | eager/graph pilot and four measured pairs | `ACHIEVED` |
| Real transaction parity and zero leaks | eager/graph pilot and four measured pairs | `ACHIEVED` |
| Controlled 2-warmup/8-measured performance | interrupted after four measured pairs | `INCONCLUSIVE_ENVIRONMENT` |
| Positive paired bootstrap lower bound | final eight-pair payload absent | `NOT_ESTABLISHED` |
| Dual verifier and final manifest | final payload absent | `NOT_ESTABLISHED` |

## Verification

Fresh local verification on 2026-08-17 used an isolated uv-managed Python
3.11 environment with PyTorch 2.7.1, Transformers 4.57.6, pytest 8.4.2, and
NumPy:

```text
expanded exact graph/runtime/Proposal-KV/speculative suite:
  780 passed in 27.87s
```

The suite covers exact graph configuration, state machine, backend, scratch,
executor, registration, ModelRunner teardown, TP1/TP4 contracts, Proposal-KV
allocation/cache/lifecycle/residency, Qwen3 draft storage, snapshot transport,
performance tooling, and source-bound verifier behavior.

## What This Proves

- The narrow production-default TP4/B4/Q4 graph can capture and replay.
- Eager and graph execution preserve exact target/proposal/acceptance and
  transaction semantics in the recorded scope.
- Four-second single-capture admission is sufficient for all retained real
  captures, whose observed maximum is approximately `3.101 s`.
- Graph resources can be reset and synchronized before NCCL/process-group
  teardown.
- Retained graph memory is small relative to model memory.
- The runtime fails closed on unsupported families and capture/replay errors.

## What This Does Not Prove

- a favorable controlled eight-pair throughput result;
- a positive paired-bootstrap confidence lower bound;
- non-regressing median TPOT in a completed campaign;
- performance amortization over multiple requests after one retained capture;
- benefit outside TP4/B4/Q4 short-context greedy dense-direct execution;
- correctness or performance with Proposal-KV offload, shape padding, dynamic
  batch, dynamic Q, or another draft architecture; or
- Phase 1 promotion readiness.

## Next Action

After physical GPU 3 is again clean:

1. use a new run tag and fresh source-bound bundle;
2. rerun two warmup pairs and eight measured position-balanced pairs;
3. require exact correctness and all-rank replay for all eight measured pairs;
4. require the archived verifier, current verifier, and checksum manifest to
   pass; and
5. classify the completed result as `GO`, `NO_GO_PERFORMANCE`, or
   `NO_GO_CORRECTNESS`.

Until that campaign completes:

```text
AUTOREGRESSIVE_DRAFT_EXACT_CUDA_GRAPH_RUNTIME=ESTABLISHED
AUTOREGRESSIVE_DRAFT_EXACT_CUDA_GRAPH_CORRECTNESS=ESTABLISHED
AUTOREGRESSIVE_DRAFT_EXACT_CUDA_GRAPH_PERFORMANCE=INCONCLUSIVE_ENVIRONMENT_PARTIAL_4_OF_8
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
