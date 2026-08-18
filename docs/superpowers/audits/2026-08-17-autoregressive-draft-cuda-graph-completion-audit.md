# Autoregressive Draft Exact-Shape CUDA Graph Completion Audit

**Date:** 2026-08-17

**Final reconciliation:** 2026-08-18

**Repository:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

**Runtime correctness classification:** `ESTABLISHED_IN_RECORDED_TP4_B4_Q4_SCOPE`

**Controlled performance classification:** `NO_GO_PERFORMANCE`

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

The fresh source-bound schema-v2 campaign
`20260817-steady-state-schema-v2-tp4-b4-q4-r3` completed two warmup pairs and
all eight measured position-balanced pairs. It used source commit
`09a23b74968139f3fb2eacb082b8bf9c79f94727`, four clean GPUs
`[1, 2, 4, 5]`, and one in-process warmup plus one measured batch in every
fresh eager or graph worker.

All eight measured pairs preserved exact target-token, logical
proposal-token, accepted-prefix, transaction-digest, and acceptance parity.
Every graph rank retained one exact TP4/B4/Q4 graph entry, increased replay
count after warmup, kept capture and resource counters unchanged during the
measured batch, and reported zero quarantine or pre-replay fallback.

The completed eight-pair aggregate is:

```text
order counts:                       eager_graph=4, graph_eager=4
exact parity:                       8 of 8
all-rank steady-state replay:       8 of 8
median eager throughput:            25.71424705405486 tok/s
median graph throughput:            30.211924132596863 tok/s
mean paired throughput delta:       +4.683516416145697 tok/s
paired bootstrap 95% CI:            [-4.083510972867552, +12.876712331393636] tok/s
median eager TPOT:                  94.181602 ms
median graph TPOT:                  72.495060 ms
median graph-minus-eager E2E:       -530.401256 ms
median graph-minus-eager proposal:  -350.421230 ms
peak eager reserved bytes:          76376178688
peak graph reserved bytes:          76395053056
accepted / proposed per worker:     51 / 70
acceptance rate:                    0.7285714285714285
```

Median throughput and median TPOT both favor graph replay, but the paired
bootstrap throughput-delta confidence interval crosses zero. The exact gate
requires `ci_low > 0`; therefore the completed controlled result is
`NO_GO_PERFORMANCE`, not `GO`.

Strict result:

```text
LOCAL_EXACT_GRAPH_IMPLEMENTATION=ESTABLISHED
REAL_TP4_CUDA_GRAPH_CAPTURE_REPLAY=ESTABLISHED
REAL_EAGER_GRAPH_CORRECTNESS_PARITY=ESTABLISHED
PRODUCTION_DEFAULT_CAPTURE_BUDGET=ESTABLISHED
GRAPH_PROCESS_GROUP_TEARDOWN_ORDER=ESTABLISHED
CONTROLLED_EIGHT_PAIR_PERFORMANCE=NO_GO_PERFORMANCE
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

## Historical Next Action Before `r3` (Superseded)

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

## 2026-08-17 Steady-State Measurement Reconciliation

The earlier four-pair partial campaign remains valid evidence for real TP4
capture/replay, eager/graph token parity, transaction parity, and graph
lifecycle cleanup. It is not valid steady-state performance evidence.

Root cause:

```text
each eager or graph pair member ran in a fresh worker process
worker warmup_runs=0
the measured graph batch therefore included:
  first successful eager observations
  first CUDA Graph capture
  subsequent replay
```

Across the four retained measured pairs, graph `backend_submit`, setup, and
selection-collective detail were lower than eager, while the graph
proposal-forward residual increased by approximately 1.91-3.25 seconds. The
retained per-rank capture range was approximately 1.76-3.10 seconds. The
alignment shows that the old measured result charged capture startup to graph
throughput; it does not establish a steady-state replay regression.

The corrected source contract is schema version 2:

```text
configuration.in_process_warmup_runs=1
worker --warmup-runs 1 --measured-runs 1
```

Every fresh eager and graph worker now runs one unmeasured batch and one
measured batch on the same engine. For each graph rank, canonical validation
requires:

```text
warmup capture_attempts=1
warmup captures=1
warmup replays>=1
measured capture_attempts == warmup capture_attempts
measured captures == warmup captures
measured replays > warmup replays
warmup and measured quarantines=0
warmup and measured fallback_pre_replay=0
ready_entry_count=1
static_bytes unchanged
reserved_bytes unchanged
total_capture_ns unchanged
```

Only the measured batch contributes E2E, throughput, TTFT, TPOT, and
proposal-forward timings. Pair-level warmups remain excluded from the
aggregate and do not substitute for the same-engine in-process warmup.

Fresh local verification:

```text
focused steady-state gate/runtime suite:
  103 passed in 2.60s

focused Python compileall:
  PASS

git diff --check:
  PASS
```

### Pre-`r3` Reconciled Prompt-to-Artifact Checklist (Superseded)

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Same-engine eager and graph warmup | worker command contract and gate tests | `ACHIEVED_LOCAL` |
| Capture excluded from measured graph timing | schema-v2 cumulative counter/resource transition checks | `ACHIEVED_LOCAL` |
| Measured graph replay occurs on every rank | measured replay count must increase from warmup | `ACHIEVED_LOCAL` |
| No measured recapture | capture attempts, captures, and capture duration must remain unchanged | `ACHIEVED_LOCAL` |
| No measured quarantine or fallback | warmup/measured failure counters must remain zero | `ACHIEVED_LOCAL` |
| Retained graph identity/resources remain stable | entry count and retained bytes must remain unchanged | `ACHIEVED_LOCAL` |
| Exact TP4/B4/Q4 and correctness semantics unchanged | production runtime files are untouched by this correction | `ACHIEVED_LOCAL` |
| Real steady-state two-warmup/eight-pair A/B | fresh schema-v2 remote payload absent | `NOT_ESTABLISHED` |
| Favorable or unfavorable controlled performance classification | fresh schema-v2 remote payload absent | `NOT_ESTABLISHED` |

The authoritative performance boundary is therefore:

```text
SCHEMA_V1_PARTIAL_CORRECTNESS_AND_LIFECYCLE=ESTABLISHED
SCHEMA_V1_PARTIAL_STEADY_STATE_PERFORMANCE=INVALID_COLD_CAPTURE_CONTAMINATED
SCHEMA_V2_STEADY_STATE_GATE_LOCAL_CONTRACT=ESTABLISHED
SCHEMA_V2_REAL_STEADY_STATE_PERFORMANCE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

The next valid performance result must use a new source-bound schema-v2 run
tag and four clean GPUs. The interrupted schema-v1 tag must not be resumed or
reclassified.

## 2026-08-18 Final Schema-v2 Completion Reconciliation

### Authoritative evidence

```text
run tag:
  20260817-steady-state-schema-v2-tp4-b4-q4-r3

local evidence root:
  artifacts/autoregressive_draft_cuda_graph/
    20260817-steady-state-schema-v2-tp4-b4-q4-r3/remote/

source commit:
  09a23b74968139f3fb2eacb082b8bf9c79f94727

source patch SHA256:
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

source tree SHA256:
  96c7961a8027e56db8cfab17eb49811b6f7e8465be9c7432fae2f6105bd9c9da

payload SHA256:
  3a8120e76fe414f2bc12fa1363f1452651bd96173eb8d98eeffd0bb068cebaa4
```

The source patch is the SHA256 of an empty patch because the source commit was
clean when the bundle was created. The source manifest verifies 130 source
files. The manifest checksum also covers all 20 raw worker JSON files, the
canonical payload, provenance, environment, source bundle, and remote
verifier receipt.

### Final prompt-to-artifact checklist

| Explicit requirement | Concrete artifact or fresh check | Verdict |
| --- | --- | --- |
| Use only the authoritative TinyLLMForge checkout | repository path resolves from `/Users/bytedance/Desktop/TinyLLMForge` to `/Users/bytedance/dev/TinyLLMForge` | `ACHIEVED` |
| Commit and push the Kerberos TTL fail-fast guard | commits `284eac6`, `9d87e41`, and `5b8ed5f`; origin branch matched locally before the run | `ACHIEVED` |
| Reject insufficient Kerberos lifetime before local/SSH side effects | `preflight.json` records READY with 32,315 seconds remaining and 5,400 seconds required | `ACHIEVED` |
| Never resume or rewrite interrupted `r1` or failed `r2` | fresh never-before-used `r3` local and remote roots | `ACHIEVED` |
| Fix the canonical schema without padding logical evidence | commit `09a23b7`; bounded logical rows preserve the real `4/4/4/4/2` active-row pattern | `ACHIEVED` |
| Preserve exact TP4/B4/Q4 graph execution identity | canonical configuration is TP4/B4/Q4; source-bound graph identity and Qwen3 backend reject non-TP4/B4/Q4 capture/replay; every graph rank retained exactly one ready entry | `ACHIEVED` |
| Prompt length 256, output length 16, greedy, dense direct, offload disabled | `result.json.configuration` | `ACHIEVED` |
| Two pair-level warmups | `result.json.warmups`, count 2 | `ACHIEVED` |
| Eight measured pairs | `result.json.pairs`, count 8 | `ACHIEVED` |
| Balanced order | four `eager_graph` and four `graph_eager` rows | `ACHIEVED` |
| One in-process warmup and one measured run per worker | schema-v2 configuration and all 20 raw worker payloads | `ACHIEVED` |
| Measured graph work is replay-only with respect to capture | warmup/measured capture attempts, captures, retained bytes, entry count, and capture duration are unchanged on every rank; replay count increases | `ACHIEVED` |
| Exact target-token parity | all eight measured eager/graph pairs | `ACHIEVED` |
| Exact logical proposal-token parity | all eight measured eager/graph pairs, including bounded ragged terminal rows | `ACHIEVED` |
| Accepted-prefix parity | all eight measured eager/graph pairs | `ACHIEVED` |
| Proposal-KV accepted-prefix commit and rejected-suffix rollback authority | transaction digest parity and zero active transactions in every measured pair | `ACHIEVED` |
| TP failure convergence and fail-closed replay boundary remain covered | source-bound runtime plus focused local regression suite | `ACHIEVED` |
| Zero quarantine and pre-replay fallback | every measured graph rank | `ACHIEVED` |
| Acceptance evidence | every measured worker reports 51 accepted of 70 proposed, rate 0.7285714285714285 | `ACHIEVED` |
| Phase timing evidence | per-pair E2E, TTFT, TPOT, proposal-forward, and proposal-detail rows in `result.json` | `ACHIEVED` |
| Memory evidence | per-rank measured peaks plus aggregate eager/graph reserved peaks | `ACHIEVED` |
| Remote verifier | `verify.remote.json` | `ACHIEVED` |
| Downloaded local verifier | `verify.local.json` | `ACHIEVED` |
| Fresh independent verifier | `/private/tmp/r3-verify-fresh.json` matched both archived receipts exactly | `ACHIEVED` |
| Final checksum manifest | fresh `shasum -a 256 -c manifest.sha256` passed every listed file | `ACHIEVED` |
| Controlled final classification | canonical and all three verifier receipts report `NO_GO_PERFORMANCE` | `ACHIEVED` |
| Versionable audit and handoff are committed and pushed | final documentation commit plus local/origin branch-head equality after push | `ACHIEVED` |

### Final controlled result

The gate's `GO` rule requires all three performance conditions:

```text
median graph throughput > median eager throughput
median graph TPOT <= median eager TPOT
paired throughput-delta bootstrap CI lower bound > 0
```

The first two conditions passed. The third did not:

```text
median eager throughput:       25.71424705405486 tok/s
median graph throughput:       30.211924132596863 tok/s
median eager TPOT:             94.181602 ms
median graph TPOT:             72.495060 ms
mean paired throughput delta:  +4.683516416145697 tok/s
bootstrap 95% CI:              [-4.083510972867552, +12.876712331393636] tok/s
```

Proposal-forward timing improved in every measured pair, with a median
graph-minus-eager delta of approximately `-350.421 ms`. End-to-end timing was
much noisier: six pairs favored graph and two regressed enough to make the
paired throughput interval cross zero. The evidence therefore supports a
real proposal-forward optimization but not a statistically stable
request-level throughput promotion.

### Final classification

```text
KERBEROS_TTL_FAIL_FAST=ESTABLISHED
SCHEMA_V2_CANONICAL_LOGICAL_EVIDENCE=ESTABLISHED
REAL_TP4_B4_Q4_GRAPH_CAPTURE_REPLAY=ESTABLISHED
REAL_EAGER_GRAPH_CORRECTNESS_PARITY=ESTABLISHED
PROPOSAL_KV_TRANSACTION_PARITY=ESTABLISHED
STEADY_STATE_CAPTURE_RESOURCE_STABILITY=ESTABLISHED
DUAL_AND_FRESH_VERIFIER=ESTABLISHED
CHECKSUM_MANIFEST=ESTABLISHED
CONTROLLED_EIGHT_PAIR_PERFORMANCE=NO_GO_PERFORMANCE
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

The next optimization should target the request-level variance outside
`proposal_forward`, especially TTFT/E2E scheduling and synchronization
dispersion. Any future performance claim requires another new tag and the
same complete source-bound gate; `r3` is immutable final evidence for this
implementation.
