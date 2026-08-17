# Phase 1 Prompt-to-Artifact Coverage Audit

Date: 2026-08-15

Repository: `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`

Decision: `PHASE_1=NOT_ACHIEVED`

Promotion: `NOT_PROMOTABLE`

## Prefix KV Sharing / Refcount / CPU-Backing Re-Audit

The current ordinary Prefix KV and Qwen3.5 hybrid-prefix paths were re-audited
against fresh tests and production source.

Fresh offline CPU Torch regression:

```text
ordinary prefix sharing/refcount/CPU-backing:
  tools/test_prefix_kv_offload_integration.py
  tools/test_chunked_prefill.py
  selection:
    prefix or reusable or hash_collision or ref_count or deduplic or intern
  19 passed, 84 deselected in 3.93s

ordinary integration duplicate full-file confirmation:
  3 passed in 0.20s

Qwen3.5 hybrid prefix cache/acquisition:
  tools/test_qwen35_hybrid_prefix_cache.py
  tools/test_qwen35_hybrid_prefix_acquisition.py
  35 passed in 3.77s

fresh disjoint local matrix:
  54 passed
```

Fresh production-source assertions cover:

```text
tinyvllm/engine/block_manager.py:
  Block.ref_count ownership
  generation increment on reset
  exact hash-plus-token reservation
  multi-owner acquire/release
  stale generation/hash/token rejection

tinyvllm/engine/model_runner.py:
  logical block/generation binding
  stale-generation metadata invalidation
  CPU-valid backing requirement
  coalesced batched H2D scheduling and counters
  cached-prefix require_valid read path

tinyvllm/engine/qwen35_hybrid_prefix_cache.py:
  byte-equality guarded tensor interning
  refcount and visible_refcount lifetime
  rollback-safe publication
  logical minus physical deduplicated_bytes accounting
  stale block-identity invalidation

tinyvllm/engine/qwen35_hybrid_prefix_acquisition.py:
  owner_count=len(sequences)
  reservation identity-bound restore
  attach only after successful restore
  hybrid lease generation publication
  failure resource release

PREFIX_SOURCE_CONTRACT_ASSERTIONS=PASS
PREFIX_PY_COMPILE=PASS
```

The ordinary CPU-backing integration still constructs `KVOffloadMVP0` through
`__new__`, uses list-backed state, and replaces H2D/D2H enqueue operations
with Python pair recorders. The hybrid tests use local CPU Torch tensors.
These results establish ownership, identity, deduplication, restore
transaction, and scheduling contracts. They do not execute a loaded model,
real CUDA copy, pinned-memory transfer, asynchronous stream/event, or
cross-request GPU runtime authority.

```text
ORDINARY_PREFIX_HASH_TOKEN_REUSE_LOCAL_CONTRACT=ESTABLISHED
ORDINARY_PREFIX_MULTI_OWNER_REFCOUNT_LOCAL_CONTRACT=ESTABLISHED
QWEN35_HYBRID_TENSOR_DEDUP_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_RESIDENCY_SCHEDULING_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_REAL_CUDA_COPY=NOT_ESTABLISHED
LOADED_CROSS_REQUEST_PREFIX_HIT_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_REFCOUNT_LIFETIME_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_DEDUP_BYTE_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_CPU_RESTORE_AUTHORITY=NOT_ESTABLISHED
GENERIC_HYBRID_STATE_PREFIX_DEDUP=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## TP4 Batch-4 Instability Telemetry and Same-Policy Priming Authority

Four source-bound campaigns cover both policy orders without and with
same-policy priming:

```text
unprimed target then learned:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-instability-telemetry-gpu3467-r3-20260815

unprimed learned then target:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-instability-telemetry-reverse-gpu3467-r4-20260815

primed target then learned:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-instability-telemetry-primed-target-learned-gpu3467-r5-20260815

primed learned then target:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-instability-telemetry-primed-learned-target-gpu3467-r6-20260815
```

All four bundles report:

```text
timing status:                  PASS
timing classification:          UNSTABLE
telemetry status:               PASS
telemetry classification:       RUNTIME_VARIANCE_SUSPECTED
exact greedy parity:            true
remote timing verifier:         PASS
remote telemetry verifier:      PASS
local timing verifier:          PASS
local telemetry verifier:       PASS
manifest:                       PASS
sampler stderr:                 0 bytes
```

The primed runner executes `2` warmups and `1` discarded measured request
batch in an isolated same-policy process before each measured worker. Prime
JSON/log pairs are retained under `prime-workers/` and `prime-logs/`; they do
not enter timing medians, telemetry sampling, or verifier inputs.

The r5 and r6 remote source suites passed `126` tests in `4.19s` and `3.92s`.
All six timing source hashes and all five telemetry source hashes are
identical across r3/r4/r5/r6.

GPU telemetry coverage and invariants:

```text
r3 target samples per repeat/GPU:  9..15
r3 learned samples per repeat/GPU: 19..27
r4 target samples per repeat/GPU:  7..10
r4 learned samples per repeat/GPU: 11..41
r5 target samples per repeat/GPU:  6..12, total 296
r5 learned samples per repeat/GPU: 10..22, total 500
r6 target samples per repeat/GPU:  7..13, total 296
r6 learned samples per repeat/GPU: 12..23, total 568

SM clock:       1410 MHz only
memory clock:   1512 MHz only
P-state:        P0 only
throttle mask:  0 only
temperature:    38..45 C
```

The four-campaign comparison must not merge medians across campaigns:

```text
target:
  unprimed first r3 E2E:       5.522148 s
  unprimed second r4 E2E:      3.958820 s
  primed first r5 E2E:         3.811664 s
  primed second r6 E2E:        3.932132 s
  primed second versus first: +3.16%

  primed first r5 TPOT:        0.238367 s
  primed second r6 TPOT:       0.243541 s
  primed second versus first: +2.17%

learned:
  unprimed first r4 E2E:      11.641256 s
  unprimed second r3 E2E:      9.840383 s
  primed first r6 E2E:         6.676585 s
  primed second r5 E2E:        5.157472 s
  primed second versus first: -22.75%

  primed first r6 TPOT:        0.424122 s
  primed second r5 TPOT:       0.330360 s
  primed second versus first: -22.11%

  primed first r6 proposal:    3547.134835 ms
  primed second r5 proposal:   3011.710241 ms
  primed second versus first: -15.09%
```

Priming materially lowers absolute medians relative to historical
same-position results:

```text
target first r5 versus r3 E2E:   -30.97%
target second r6 versus r4 E2E:   -0.67%
learned first r6 versus r4 E2E:  -42.65%
learned second r5 versus r3 E2E: -47.59%
learned first proposal:          -45.96%
learned second proposal:         -43.42%
```

Target primed order medians are near-converged, but learned remains
materially faster in the second process position. Under the predeclared
design rules, the unique classification is `POSITION_EFFECT_REMAINS`, not
`PROCESS_CADENCE_EFFECT_SUPPORTED`.

The absolute priming speedups do not establish whether JIT, model page cache,
allocator state, CUDA runtime state, or another process boundary is the
specific cause.

The host logs remain hash-bound retention rather than semantically parsed,
per-repeat verifier evidence. An exploratory `vmstat` summary found median
`us=59% sy=39% id=2% wa=0 st=0` in r5/r6, but host contention is not
excluded.

The selected-GPU process inventory retained the pre-existing GPU-7 `python3`
PID `703088` before and after all campaigns. No selected-GPU process was
terminated.

Strict classification:

```text
QWEN3_TP4_B4_TELEMETRY_COVERAGE=ESTABLISHED
QWEN3_TP4_B4_REVERSE_ORDER_EXECUTION=ESTABLISHED
QWEN3_TP4_B4_SAME_POLICY_PRIMING_EXECUTION=ESTABLISHED
QWEN3_TP4_B4_EXACT_PARITY=ESTABLISHED
QWEN3_TP4_B4_STABLE_GPU_CLOCK_PSTATE_THROTTLE=ESTABLISHED
QWEN3_TP4_B4_PRIMING_CLASSIFICATION=POSITION_EFFECT_REMAINS
TARGET_PRIMED_ORDER_EFFECT=NEAR_CONVERGED
LEARNED_PRIMED_ORDER_EFFECT=REMAINS
QWEN3_TP4_B4_TIMING_STABILITY=NOT_ESTABLISHED
QWEN3_TP4_B4_HOST_CONTENTION=NOT_EXCLUDED
QWEN3_TP4_B4_RUNTIME_ROOT_CAUSE=NOT_ESTABLISHED
CONTROLLED_PERFORMANCE_PROMOTION=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

Do not choose a CUDA Graph, TP collective, or metadata optimization from
these campaigns. The next experiment should target the residual learned
first-position effect with per-repeat host semantic alignment or a more
specific process/JIT boundary control.

### TP4 batch-4 instability triage

The source-bound schema-v3 diagnostic bundle is:

```text
experiments/autoregressive_draft/
  tp4-qwen3-b4-proposal-forward-diagnostic-gpu3467-r2-20260815
```

Read-only recomputation over its retained per-rank rows establishes:

```text
critical-rank sequence:
  1,0,3,0,3,1,1,1

median cross-rank proposal spread:
  23.475 ms = 0.51% of proposal median

critical submit+collective+readback / proposal correlation:
  0.9967

per-rank submit+collective / proposal correlation:
  0.9979-0.9989

critical submit / collective correlation:
  -0.6761
```

The instability is rank-wide rather than a single-rank laggard. Deferred CUDA
completion migrates between the submit and collective wall-clock boundaries,
but their sum tracks the parent proposal interval. Existing before/after GPU
snapshots do not cover per-repeat clocks, throttling, power, utilization,
temperature, or host load. Therefore:

```text
SINGLE_RANK_LAGGARD_EXPLANATION=NOT_SUPPORTED
SUBMIT_VS_COLLECTIVE_BUCKET_OPTIMIZATION=DEFERRED
PER_REPEAT_ENVIRONMENTAL_CAUSE=NOT_ESTABLISHED
NEXT_OPTIMIZATION=INSTABILITY_INVESTIGATION
PHASE_1=NOT_ACHIEVED
```

## TP4 Independent-Draft Proposal-Forward Diagnostic

Evidence bundle:

```text
experiments/autoregressive_draft/
  tp4-qwen3-b4-proposal-forward-diagnostic-gpu3467-r2-20260815
```

Coverage:

```text
TP4 batch 4:                    COVERED
two warmups / eight measured:   COVERED
exact greedy parity:            COVERED
raw per-rank proposal details:  COVERED
critical-rank coherent sum:     COVERED
acceptance:                     COVERED
peak CUDA allocation:           COVERED
source-bound remote verifier:   COVERED
source-bound local verifier:    COVERED
checksum manifest:              COVERED
stable performance baseline:    NOT COVERED
GPU kernel duration:            NOT COVERED
long-context performance:       NOT COVERED
second learned structure:       NOT COVERED
```

The artifact is valid but classified `UNSTABLE`. Target and learned E2E
range-over-median are `66.60%` and `62.67%`; therefore performance deltas are
not promotion evidence. Learned proposal-forward correlates with E2E at
`0.9466`, and its critical-rank submit/collective/readback bucket has a
median share of `94.45%`, but asynchronous completion moves wall time between
those sub-boundaries.

The evidence-based decision is `INSTABILITY_INVESTIGATION`. CUDA Graph,
authority-frequency reduction, and metadata optimization remain deferred
until a repeatable baseline exists.

Strict claim state:

```text
QWEN3_INDEPENDENT_DRAFT_TP4_B4_DETAIL_ATTRIBUTION=ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_B4_EXACT_PARITY_8X=ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_B4_STABLE_BASELINE=NOT_ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_CUDA_GRAPH_DIRECTION=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

### TP4 schema-v2 timing pilot

```text
bundle:
  experiments/autoregressive_draft/
    tp4-qwen3-controlled-performance-timing-gpu3467-r4b-20260815

artifact status:
  PASS

classification:
  PILOT_ONLY

direction:
  NEGATIVE

remote verifier:
  PASS

local verifier:
  PASS

refreshed manifest:
  PASS
```

The schema-v2 worker captures seven additive runtime stages and four-rank
executor deltas. Executor timing is nested inside
`first_target_batch_ms`; it is not an additional additive stage.

Median comparison:

```text
batch 1:
  target TPOT:       0.256894 s
  learned TPOT:      0.374803 s
  TPOT direction:    +45.90%
  target throughput: 3.858257 tok/s
  learned throughput:2.718225 tok/s
  throughput:        -29.55%
  acceptance:        15 / 15 = 100.00%

batch 4:
  target TPOT:       0.281634 s
  learned TPOT:      0.535204 s
  TPOT direction:    +90.03%
  target throughput: 14.254814 tok/s
  learned throughput:6.885352 tok/s
  throughput:        -51.70%
  acceptance:        53 / 72 = 73.61%
```

Learned batch-1 median stage totals:

```text
first target plus proposal batch: 4019.264 ms
target verify tail batch:          935.312 ms
transactional commit metadata:     390.031 ms
reserve blocks:                      1.289 ms
KV materialize:                      0.092 ms
accept/sample:                       0.055 ms
```

Nested batch-1 executor median:

```text
prompt bootstrap:   350.626 ms
proposal forward:  2693.178 ms
proposal finalize:  312.353 ms
```

Learned batch-4 median stage totals:

```text
first target plus proposal batch: 6402.739 ms
target verify tail batch:         1288.624 ms
transactional commit metadata:     962.029 ms
reserve blocks:                      4.706 ms
KV materialize:                      0.217 ms
accept/sample:                       0.146 ms
```

Nested batch-4 executor median:

```text
prompt bootstrap:   335.970 ms
proposal forward:  4411.708 ms
proposal finalize:  529.206 ms
```

The primary bottleneck family is independent-draft `proposal_forward`.
Block reservation, KV materialization, and greedy accept/sample are
negligible at this workload. Proposal finalization and publication are
secondary but material.

Batch-4 learned E2E is not stable across the three measured repeats:

```text
6082.036 ms
9295.094 ms
11607.905 ms
```

The increase appears on every TP rank and is concentrated in
`proposal_forward` and the enclosing `first_target_batch_ms`; it is not a
single observed rank-3 timing outlier. Therefore this artifact establishes
negative direction and the bottleneck family, but it is not a stable
before/after optimization baseline.

Fresh validation:

```text
local performance/publication timing tests:
  24 passed, 23 deselected

remote full autoregressive draft executor:
  75 passed

remote performance/publication timing tests:
  24 passed, 23 deselected

py_compile:
  PASS

bash -n:
  PASS

scoped git diff --check:
  PASS
```

One broad remote combination produced `121 passed, 1 failed`; the only
failure is the pre-existing
`test_partition_preserves_selected_and_suppressed_order` expectation drift.
The full executor and the exact timing/publication set pass separately.

The first `r4` launch failed during a transient SSH banner exchange and
retained only its local `command.txt` and `source.tar`. The successful
authority uses the unique `r4b` tag and does not overwrite that evidence.

Next evidence step:

```text
1. split proposal_forward into backend decode, TP greedy selection,
   Proposal-KV bookkeeping, and authority convergence evidence;
2. run an isolated repeated batch-4 diagnostic;
3. only then choose between exact-shape independent-draft CUDA Graph work
   and metadata/authority optimization.
```

Strict classification:

```text
QWEN3_INDEPENDENT_DRAFT_TP4_TIMING_SCHEMA_V2=ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_PRIMARY_BOTTLENECK=PROPOSAL_FORWARD
QWEN3_INDEPENDENT_DRAFT_TP4_BATCH4_STABLE_BASELINE=NOT_ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_PERFORMANCE_DIRECTION=NEGATIVE
QWEN3_INDEPENDENT_DRAFT_4K_PERFORMANCE=NOT_ESTABLISHED
SECOND_LEARNED_MODEL_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Qwen3 B4 Repeat-Aligned Host-Semantic Authority

The controlled-performance instability follow-up now has two source-identical,
opposite-order campaigns and one independently verified comparison.

Canonical campaign artifacts:

```text
r7 target then learned:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815/
      host-semantic.json

r8 learned then target:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815/
      host-semantic.json

comparison:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json

comparison receipt:
  experiments/autoregressive_draft/
    tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.verify.json
```

Campaign identity and coverage:

```text
r7:
  policy order: target,learned
  prime each policy: true
  exact parity: true
  host samples: target 1585, learned 1069
  aligned samples per repeat:
    target  14,13,19,13,21,15,12,11
    learned 58,61,54,39,43,63,52,40
  maximum repeat-local host gap:
    target  0.201811240 s
    learned 0.203207182 s

r8:
  policy order: learned,target
  prime each policy: true
  exact parity: true
  host samples: learned 940, target 2579
  aligned samples per repeat:
    learned 47,42,29,44,47,31,30,25
    target  16,22,16,24,25,22,17,20
  maximum repeat-local host gap:
    learned 0.234627638 s
    target  0.222141446 s

allowed maximum repeat-local gap:
  0.600000000 s
```

Both campaign artifacts bind the same six source hashes:

```text
timing diagnostic:
  d2fc63df070602403163c87e2d3a7244d373522e38ec694482f48bb0f7a87b4b
host sampler:
  6245dc19c9f56cf1530181a5be9a4df606f96adc71c69ff42bf7f533d4eac986
host semantic diagnostic:
  e3b1a4ed9dbfc769ab4baed2356f6d63760eefd9ab7919585a3f169fb8bf49ee
GPU instability telemetry:
  b74acc04ddbb5557c65d9e983e73b96764cb272bd9a384ed6d71babe405c79fa
performance worker:
  fdc81278a137218b66e4057ae08d52dfa52ed30c426c394e51b171bc2893b7c5
host verifier:
  110556d81c4b119ca740deb83080d53958c7885c6bfb6a609534bd6cc8ea0a8a
```

r7 is a source-bound postprocessing-only recovery. Its initial raw campaign
and initial verifier exits remain `1`; the raw worker, timing, GPU telemetry,
and host JSONL inputs were preserved. `recovery-provenance.json`,
`recovery-raw-inputs.sha256`, and `recovery-source-files.sha256` bind the
recovery. The canonical runner/recovery and all remote/local verifier exits
are `0`. r8 is a clean post-fix launch with all exits `0`.

Verifier evidence:

```text
r7 remote/local timing:
  PASS / UNSTABLE
r7 remote/local GPU telemetry:
  PASS
r7 remote/local host semantic:
  PASS / ALIGNED_CAMPAIGN

r8 remote/local timing:
  PASS / UNSTABLE
r8 remote/local GPU telemetry:
  PASS
r8 remote/local host semantic:
  PASS / ALIGNED_CAMPAIGN

comparison independent verifier:
  PASS
  campaign artifacts verified: 2
  source files verified per campaign: 6

manifests:
  r7:         PASS / 72 entries
  r8:         PASS / 54 entries
  comparison: PASS / 4 entries
```

The learned position effect reversed rather than reproducing the earlier
positive learned-first slowdown:

```text
E2E:
  learned first median:  6.1944397425 s
  learned second median: 9.0720287193 s
  relative delta:       -31.7193547971%

TPOT:
  learned first median:  0.3971080518 s
  learned second median: 0.5817995088 s
  relative delta:       -31.7448629878%

proposal forward:
  learned first median:  3468.1439884007 ms
  learned second median: 5174.6466420591 ms
  relative delta:       -32.9781484940%
```

Primary host comparison:

```text
metrics compared: 9
metrics worse in learned-first: 2

CPU iowait:
  learned first:  0.000033576805
  learned second: 0.000003701729
  relative increase: +807.0572302179%

major faults/s:
  learned first:  0.187509227588
  learned second: 0.164828530622
  relative increase: +13.7601766399%
```

No primary metric reaches the required positive Spearman `rho >= 0.6`.
Across the 16 learned repeats, the largest positive E2E correlation is memory
writeback at `rho=0.3589400087`; I/O PSI is `0.3529411765`. Run queue is
negatively correlated with E2E at `rho=-0.55`. All nine metrics and both E2E
and proposal-forward correlations retain rank variance and 16 samples.

Final comparison:

```text
status:
  PASS

classification:
  HOST_ALIGNMENT_INCONCLUSIVE

classification reason:
  learned E2E position effect is below 10%
```

This classification is caused by effect reversal, not an alignment, sample,
source-identity, or verifier evidence gap. Because these campaigns contain no
positive learned-first slowdown, they cannot support or refute the hypothesis
that such a positive slowdown is caused by host pressure. The next useful
measurement is a primed learned/learned process-boundary A/A campaign to
quantify same-policy launch/order variance before another causal attribution
attempt.

Strict boundary:

```text
REPEAT_ALIGNED_HOST_TELEMETRY=ESTABLISHED
R7_R8_SOURCE_IDENTITY=ESTABLISHED
R7_R8_EXACT_GREEDY_PARITY=ESTABLISHED
POSITIVE_LEARNED_FIRST_SLOWDOWN=NOT_REPRODUCED
POSITION_EFFECT_REMAINS=NOT_ESTABLISHED
SPECIFIC_RUNTIME_ROOT_CAUSE=NOT_ESTABLISHED
HOST_CONTENTION=NOT_EXCLUDED
STABLE_PERFORMANCE_BASELINE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

Fresh final authority verification:

```text
focused host/timing/telemetry tests:
  98 passed in 0.89s

r7 independent verifier:
  PASS / ALIGNED_CAMPAIGN
  input files verified: 6
  source files verified: 6
  repeat coverage: target 8, learned 8

r8 independent verifier:
  PASS / ALIGNED_CAMPAIGN
  input files verified: 6
  source files verified: 6
  repeat coverage: target 8, learned 8

comparison independent verifier:
  PASS / HOST_ALIGNMENT_INCONCLUSIVE
  campaign artifacts verified: 2
  source files verified per campaign: 6

manifest verification:
  r7:         PASS / 72 entries
  r8:         PASS / 54 entries
  comparison: PASS / 4 entries

runner bash syntax:
  PASS

Python compile:
  PASS

scoped git diff --check:
  PASS
```

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
ran during this re-audit.

## Cross-Family Transactional Direct-Commit Re-Audit

The current transactional paths were re-audited across model-free n-gram and
SAM, Qwen3.5 native MTP, and independent Qwen3 drafting.

The ownership boundary is:

```text
n-gram / SAM:
  model-free proposals
  -> generic target-KV prepared selection/commit/rollback
  -> no Proposal-KV ownership

Qwen3.5 native MTP / independent Qwen3 draft:
  generic target-KV prepared selection/commit/rollback
  + ProposalKVLifecycleCoordinator finalize/rollback/release
```

Fresh unified regression:

```text
source adapters, target prepared commit, Engine/ModelRunner callbacks,
Proposal-KV cache/lifecycle/residency, native MTP, independent draft,
and local gate validators:
  531 passed in 3.42s

transactional production py_compile:
  PASS
```

Production-only source assertions:

```text
tinyvllm/** rematerialize_accepted_kv symbols:
  ABSENT

accepted-entry copy/replay/rematerialization counter increment paths:
  ABSENT

generic Engine/runtime source-type routing branches:
  ABSENT

Proposal-KV accepted commit count:
  max(accepted_proposal_tokens - 1, 0)

rejected or rolled-back Proposal-KV retirement:
  writeback=False
```

`tools/profile_ngram_commit.py` still contains the historical profiling-only
`rematerialize_accepted_kv()` helper, with corresponding legacy profiler
tests in `tools/test_ngram_speculative.py`. That helper is outside the
`tinyvllm/` production runtime path. It must not be described as the current
production correctness fallback, and it is not direct-commit authority.

```text
CROSS_FAMILY_TARGET_KV_PREPARED_COMMIT_CONTRACT=ESTABLISHED_LOCAL
NATIVE_MTP_PROPOSAL_KV_DIRECT_COMMIT_CONTRACT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_DIRECT_COMMIT_CONTRACT=ESTABLISHED_LOCAL
PRODUCTION_ACCEPTED_KV_REMATERIALIZATION_SYMBOLS=ABSENT
CROSS_FAMILY_LOADED_ZERO_REMATERIALIZATION=NOT_ESTABLISHED
REAL_GPU_TP_ZERO_REMATERIALIZATION=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
ran during this re-audit.

## Generic Proposal-KV Residency Plan Reconciliation

The current Proposal-KV residency plan has 53 real task lines:

```text
PROPOSAL_KV_RESIDENCY_PLAN_TOTAL_STEPS=53
PROPOSAL_KV_RESIDENCY_PLAN_CHECKED=44
PROPOSAL_KV_RESIDENCY_PLAN_INTENTIONALLY_OPEN=9
```

The nine open steps are six historical RED executions without retained
failure transcripts plus the three explicit Task 8 authorization-boundary
steps for a future real GPU/remote/NCCL authority campaign.

Fresh current-source evidence:

```text
allocator/residency/cache/lifecycle/Qwen3.5 MTP focused suite:
  100 passed in 2.80s

generic speculative runtime plus independent-drafter adjacent suite:
  275 passed in 2.14s

fresh total:
  375 passed

production/test py_compile:
  PASS

interface, durable-slot-removal, no-rematerialization, local-gate,
and scoped diff assertions:
  PASS
```

Current source directly establishes durable logical identities decoupled from
temporary GPU slots, generation-checked leases, deterministic committed-LRU
residency, dirty writeback, batched transfer contracts, completion-aware
retirement, rejected-suffix zero-D2H, accepted-prefix
zero-copy/zero-replay/zero-rematerialization, and default-off zero movement.

The local pressure tests use synchronous CPU or fake copy backends. They
validate the state machine, ordering, and counters, but are not real pinned
memory, CUDA stream/event, H2D/D2H movement, or performance authority.

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

No Task 8 producer, worker, verifier, or remote runner was created. No GPU,
remote, NCCL, loaded-checkpoint, performance workload, or remote probe ran
during this reconciliation.

## Autoregressive Draft Model Executor Plan Reconciliation

The previously all-unchecked
`docs/superpowers/plans/2026-08-14-autoregressive-draft-model-executor.md`
was reconciled against current source, tests, fresh GREEN runs, static checks,
and durable documentation:

```text
AUTOREGRESSIVE_DRAFT_EXECUTOR_PLAN_TOTAL_STEPS=34
AUTOREGRESSIVE_DRAFT_EXECUTOR_PLAN_CHECKED=26
AUTOREGRESSIVE_DRAFT_EXECUTOR_PLAN_INTENTIONALLY_OPEN=8
```

The eight open steps are the historical RED executions for Tasks 1-8. The
plan specifies their expected failures, but no retained failure transcript
was found. All directly provable test-writing, implementation, GREEN,
regression, static-check, and handoff steps are checked.

Fresh current-source evidence:

```text
offline CPU Torch full regression matrix:
  437 passed in 23.60s

executor/backend/registration/config/ModelRunner/TP1-gate interfaces:
  PASS

production and gate py_compile:
  PASS

generic Engine/Scheduler/verifier/lifecycle source-neutral scan:
  PASS
```

This establishes the local generic batch-native autoregressive executor,
chunked-prefill prompt accumulation, transactional
bootstrap/propose/finalize/release, tensor-free authority, Qwen3 dense
backend, exact tokenizer/checkpoint compatibility, source-neutral ModelRunner
registration, and TP1 gate harness.

Later Proposal-KV residency and TP4 work moved
`Qwen3DraftPhysicalSlotStore` from `qwen3_draft_backend.py` into
`qwen3_draft_proposal_kv.py`; the backend continues to import/export the
symbol and the current regression matrix covers the superseding
implementation.

The real Qwen3 draft plus Qwen3.5 target checkpoint gate did not run:

```text
AUTOREGRESSIVE_DRAFT_EXECUTOR_CONTRACT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_TP1_GATE_HARNESS=ESTABLISHED_LOCAL
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
ran during this reconciliation.

## Learned-Drafter TP4 Source-Bound Readiness Recheck

The existing loaded-direct and source-bound learned-drafter TP4 plans are
fully executed:

```text
docs/superpowers/plans/
  2026-08-15-autoregressive-draft-tp4-loaded-direct-gate.md
    12 / 12 checked
  2026-08-15-autoregressive-draft-tp4-source-bound-bundle.md
    23 / 23 checked

combined:
  35 / 35 checked
```

Fresh current-environment evidence:

```text
dependency-light learned-drafter / Proposal-KV suite:
  142 passed in 1.12s

offline CPU Torch tensor integration suite:
  211 passed in 19.90s

fresh split matrix total:
  353 passed

TP4 gate, archived verifier, local validator, and snapshot transport
py_compile:
  PASS

deterministic source inventory:
  PASS, 30 sorted unique files

unchecked tar extractall():
  ABSENT
```

The passing suite directly covers schema-v2 accepted-prefix identity,
direct-allocator TP4 configuration and rank authority, normalized
deterministic tar bytes, full inventory and hash validation, rejection of
unsafe/missing/duplicate/unexpected archive members, safe extraction, result
tamper rejection, current-versus-archived verifier receipt equality, failed
bundle preservation, and exclusive atomic publication.

The initial combined regression under CommandLineTools Python was blocked
during collection for five real tensor integration files because that
interpreter has no `torch` installation:

```text
tools/test_autoregressive_draft_executor.py
tools/test_autoregressive_draft_model_runner_integration.py
tools/test_autoregressive_draft_registration.py
tools/test_autoregressive_draft_tp.py
tools/test_qwen3_draft_proposal_kv_storage.py
```

The same five files were then rerun through the existing offline dependency
cache:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  <the five tensor integration files>

211 passed in 19.90s
```

The collection failure was therefore an interpreter-selection issue, not a
source or assertion failure. No network dependency installation or fake
Torch/CUDA module was introduced. The previously recorded isolated CPU Torch
result of `217 passed` remains historical evidence; the new `211 passed`
result is fresh current-source evidence. These local CPU tests still do not
replace real GPU, NCCL, or loaded-checkpoint execution.

```text
TP4_SOURCE_BOUND_BUNDLE_CONTRACT=ESTABLISHED_LOCAL
TP4_ARCHIVED_VERIFIER=ESTABLISHED_LOCAL
TP4_ATOMIC_ARTIFACT_BUNDLE=ESTABLISHED_LOCAL
TP4_RETAINED_SOURCE_BOUND_EXECUTION_ARTIFACT=ABSENT
LEARNED_DRAFTER_TP4_LOADED_EXECUTION=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote
probe ran during this recheck.

## Generic Runtime Performance Frozen Source Reconstruction

The legacy `20260812T085852Z` performance artifact embeds SHA-256 identities
for exactly nine source files. All nine were recovered exactly into:

```text
/tmp/speculative-runtime-performance-frozen-2026-08-15
```

Recovery provenance:

```text
current checkout exact matches: 5
snapshot blob exact matches:    4
unrecoverable files:            0
manifest:
  /tmp/speculative-runtime-performance-frozen-manifest.json
GENERIC_PERFORMANCE_FROZEN_SOURCE_CLOSURE=RECOVERED_9_OF_9
```

The four drifted files recovered from
`/Users/bytedance/.trae/hooks/ai-contribution-sdk/snapshot-blobs/<sha256>`
were:

```text
tinyvllm/engine/llm_engine.py
tinyvllm/engine/model_runner.py
tinyvllm/engine/speculative_runtime.py
tinyvllm/speculative/batch_runtime.py
```

The artifact-bound verifier from the reconstructed tree was run against the
unchanged retained artifact and exited zero:

```text
verification output:
  /tmp/speculative-runtime-performance-frozen-verify.json
artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f
status:         PASS
classification: NOT_PROMOTABLE
direction:      POSITIVE
batch 1:        IMPROVED
batch 4:        IMPROVED
GENERIC_PERFORMANCE_FROZEN_VERIFICATION=PASS_NOT_PROMOTABLE
```

This establishes local reconstructability of the embedded nine-file closure
and independent re-verification under the artifact-bound source. It does not
turn the result into a current-source PASS. The temporary `/tmp` tree is not a
retained standalone source archive, and the artifact still lacks a
checkpoint-bound model manifest and the broader two-structure, TP4,
16K/32K, and learned-drafter promotion evidence.

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
ran during reconstruction.

```text
GENERIC_PERFORMANCE_HISTORICAL_AUTHORITY=PASS_NOT_PROMOTABLE
GENERIC_PERFORMANCE_CURRENT_SOURCE_VERIFICATION=FAIL_CLOSED_SOURCE_DRIFT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Generic Runtime Performance Plan Reconciliation

The generic TP1/4K performance plan now records 29 proven steps and keeps six
historical RED observations open:

```text
GENERIC_PERFORMANCE_PLAN_TOTAL_STEPS=35
GENERIC_PERFORMANCE_PLAN_CHECKED=29
GENERIC_PERFORMANCE_PLAN_INTENTIONALLY_OPEN=6
GENERIC_PERFORMANCE_FRESH_TESTS=107_PASSED
GENERIC_PERFORMANCE_SOURCE_PYCOMPILE=PASS
GENERIC_PERFORMANCE_RUNNER_BASH_SYNTAX=PASS
```

The retained `20260812T085852Z` artifact still has matching historical local
and remote `PASS / NOT_PROMOTABLE` receipts. A fresh current-source verifier
invocation correctly fails closed:

```text
artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f
artifact-bound llm_engine.py SHA-256:
  baf26ee14d3cfe1dbeb0d897e8d4572460c1d9376ac92bbbff7fac718a7e5e12
current llm_engine.py SHA-256:
  2ffceaccfb1ff9e0cd2aa6506a1e6cdda588e71bc17368f2411974918543096b
verification:
  FAIL_CLOSED_SOURCE_HASH_MISMATCH
```

The historical result remains valid within its bound source and measured
Qwen3-0.6B TP1/4K batch-1/4 scope. The embedded nine-file closure was
subsequently reconstructed from current exact matches plus content-addressed
snapshot blobs and re-verified as `PASS / NOT_PROMOTABLE`; see the preceding
section. It is not a fresh-current-source result, does not retain a standalone
deterministic source archive or checkpoint-bound model manifest, and does not
satisfy the broader two-structure, TP4, 16K/32K, or learned-source promotion
gates.

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
ran during this reconciliation.

```text
GENERIC_PERFORMANCE_HISTORICAL_AUTHORITY=PASS_NOT_PROMOTABLE
GENERIC_PERFORMANCE_CURRENT_SOURCE_VERIFICATION=FAIL_CLOSED_SOURCE_DRIFT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Variable-Q CUDA Graph Plan Reconciliation

The stale documentation checkboxes in
`docs/superpowers/plans/2026-08-12-variable-q-speculative-cuda-graph.md`
were reconciled against current source, fresh local tests, retained blocked
preflight records, the durable handoff, and this audit.

```text
exact-family smoke/verifier/remote-runner tests: 93 passed in 0.20s
producer/verifier/runner py_compile:             PASS
VARIABLE_Q_PLAN_TOTAL_STEPS=48
VARIABLE_Q_PLAN_CHECKED=44
VARIABLE_Q_PLAN_INTENTIONALLY_OPEN=4
```

The handoff and prompt-to-artifact audit steps are now checked. This records
the local exact-family contract and its limits; it does not promote either
blocked preflight record into a CUDA result. The CUDA correctness,
performance, final-full-gate, and plan-completion steps remain open.

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
ran during this reconciliation.

```text
VARIABLE_Q_EXACT_FAMILY_PASS_ARTIFACT=ABSENT
VARIABLE_Q_SOURCE_BOUND_ARCHIVED_VERIFIER=NOT_ESTABLISHED
VARIABLE_Q_TP4=NOT_ESTABLISHED
VARIABLE_Q_OFFLOAD=NOT_ESTABLISHED
VARIABLE_Q_LONG_CONTEXT=NOT_ESTABLISHED
VARIABLE_Q_PERFORMANCE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Next Loaded-Campaign Readiness Recheck

The highest-value remaining promotion action is still a separately authorized
loaded Qwen3 independent-draft plus Qwen3.5 target campaign. The required
order remains:

1. TP1 exact parity plus real bidirectional Proposal-KV movement;
2. TP4 batch 1/4 parity with all-rank terminal snapshots and a source-bound
   bundle;
3. controlled TPOT/TTFT/throughput, memory, H2D/D2H byte, and acceptance
   comparison;
4. native MTP TP4/32K paired-trace collection and root-cause resolution.

Fresh dependency-light TP1 readiness evidence:

```text
tools/test_autoregressive_draft_tp1_engine_gate.py: 34 passed in 0.04s
TP1 gate/test py_compile:                            PASS
TP1 CLI --help:                                      PASS
TP1_OUTPUT_DIR_SURFACE=ABSENT
AUTOREGRESSIVE_DRAFT_TP1_OFFLOAD_AUTHORITY_HARNESS=ESTABLISHED_LOCAL
TP1_HARNESS_FRESH=PASS
TP1_SOURCE_BOUND_BUNDLE=ABSENT
TP1_INDEPENDENT_VERIFIER=ABSENT
TP1_REMOTE_LAUNCHER=ABSENT
```

The TP1 schema-v2 gate fixes TP1, bfloat16, temperature zero, and
`MAX_PROPOSAL_TOKENS=4`. Its validator requires target/draft checkpoint and
tokenizer identities, batch 1/4 exact output-token equality, nonempty
acceptance rows, positive real-draft forwards, zero extra target forwards,
distinct positive-byte Proposal-KV and target-KV storage, zero live Proposal-KV
slots after release, and internally consistent direct/residency capacities and
nested allocator deltas. It also requires accepted-entry copy, replay, and
rematerialization counters all to remain zero.

The movement classification is fail closed: H2D and D2H entry counts and bytes
must all be positive for
`real_proposal_kv_bidirectional_movement=true`; direct mode must report zero
movement. The payload permanently rejects a performance pass criterion.

These are execution-result validation rules, not evidence that a loaded run
occurred. The CLI exposes only a single `--output` JSON written with
`Path.write_text()`. It has no `--output-dir`, exclusive/atomic publisher,
frozen source inventory, `source.tar`, `source_manifest.json`, artifact/source
SHA-256 binding, current/archive receipt comparison, or independent verifier.

By contrast, the local TP4 contract publishes canonical `result.json`,
deterministic `source.tar`, `source_manifest.json`, and `verify.json` through a
temporary directory followed by rename. Its independent verifier validates
the gate payload and source/artifact hashes against both current source and a
safely extracted archived source tree; both receipts must match before
publication.

No new bundle or launcher was implemented because the current constraints
forbid creating a GPU launcher and do not separately approve a new TP1 bundle
design. No GPU, remote, NCCL, loaded-checkpoint, performance workload, or
remote probe ran. Local harness readiness cannot substitute for real pinned
backing, CUDA copies, stream/event ordering, loaded parity, or controlled
performance. No autoregressive-draft TP1 source-bound authority bundle was
found under `artifacts/` or `experiments/`.

```text
NEXT_PHASE1_PROMOTION_ACTION=LOADED_QWEN3_DRAFT_QWEN35_TARGET_TP1
NEXT_PHASE1_PROMOTION_ACTION_AUTHORIZATION=NOT_GRANTED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## SAM Drafter Gate Plan Reconciliation

The 90 unchecked steps in
`docs/superpowers/plans/2026-07-15-sam-drafter-gate.md` were stale. Current
source, focused tests, retained implementation commits, and the recorded
remote smoke sequence prove 73 steps. Seventeen steps remain intentionally
open:

```text
historical RED observations without retained transcript: 7
dedicated remote block-boundary smoke without record:      1
canonical 175-row run/resume/verify/audit/evidence:        5
canonical-dependent final documentation/audit/clean state: 4

total:                                                     17
```

The checked remote-runner download/verification step is supported only by the
retained five-file 25-row smoke. It must not be read as evidence that the
canonical 175-row matrix ran. The smoke repair/retry step is supported by the
retained `INCOMPLETE` artifact, commits `3f61619` and `b71e7ce`, and the
subsequent strict `NO_GO` artifact. The source-freeze step is supported by the
valid smoke manifest's `source_dirty=false` and exact source commit
`b71e7ceabec211a7e1f5a4e2a942fac9a780c067`.

No GPU, remote, NCCL, loaded-checkpoint, performance workload, or remote probe
was run during this reconciliation.

Fresh local reconciliation verification:

```text
SAM/ngram/gate focused pytest:                    87 passed in 0.30s
SAM/profiler/gate py_compile:                     PASS
remote runner bash -n:                            PASS
two retained 25-row artifact verifier runs:       PASS
plan count/open-boundary assertions:              PASS
adaptive artifact 140 rows / SAM rows 0:          PASS
handoff/audit marker assertions:                  PASS
scoped git diff --check:                          PASS
```

The attempted four-file pytest command including
`tools/test_chunked_prefill.py` was blocked during collection because the only
local interpreter,
`/Library/Developer/CommandLineTools/usr/bin/python3`, has no `torch`
installation (`ModuleNotFoundError: No module named 'torch'`). This is an
environment failure, not a test assertion result. No dependency was installed
and no fake torch/CUDA/FlashAttention module was introduced.

```text
SAM_PLAN_TOTAL_CHECKBOXES=90
SAM_PLAN_CHECKED=73
SAM_PLAN_INTENTIONALLY_OPEN=17
SAM_DEDICATED_REMOTE_BLOCK_BOUNDARY_SMOKE=NOT_FOUND
SAM_RETAINED_CANONICAL_175_ROW_AUTHORITY=ABSENT
SAM_LOADED_PERFORMANCE_AUTHORITY=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Purpose

This audit maps the three objective directions and the frozen Phase 1 exit
criteria to current source, tests, and loaded-model artifacts. It incorporates
the Proposal-KV logical/physical decoupling Tasks 1-7 completed on 2026-08-15.
It does not reinterpret local tests as GPU movement, loaded-checkpoint
correctness, or performance evidence.

The compact promotion authority remains:

- `docs/superpowers/audits/2026-08-14-phase1-promotion-checklist.md`

## Generic Qwen3 TP4 Correctness/Collective Plan Reconciliation

The all-unchecked state of
`docs/superpowers/plans/2026-08-13-generic-speculative-runtime-tp4-correctness-collective-authority.md`
was stale. Current source, focused tests, the retained real TP4 authority,
four verifier receipts, the historical execution record, and both durable
project documents directly establish that the plan was executed.

Retained authority:

```text
directory:
  artifacts/generic_speculative_tp4/
    tp4-opaque-48d18e4aba16756d/authority

result.json SHA-256:
  4e504110074eb8c6a5d449d381d599d5e4303ac05371ad8c40cf9cea50955e9b

source_manifest.json SHA-256:
  0d5c2984fe000627ceb46ed3a9f5c5762d50b07fac1596c84dd0c82f4a9b36a5

source tree SHA-256:
  88d30b69246ac9c15caab5ce3c7f5f82fad00d7ec24e4c005e8ab31beed97546

model manifest SHA-256:
  6bb7f90f4ad46c059c9e3df600532147ecc00683e58e96ce9dd6bc5084f2c90e

scope:
  Qwen3-0.6B
  generic host n-gram
  TP4
  4096-token context
  batch 1 and 4
  exact greedy parity

classification:
  NOT_PROMOTABLE
```

Fresh current-source tests and static checks:

```text
tools/test_generic_speculative_tp4_gate.py
tools/test_model_runner_spec_verify.py:
  133 passed in 0.30s

changed source py_compile:
  PASS

remote runner bash syntax:
  PASS
```

A fresh verifier invocation against the current checkout fails closed with:

```text
{"classification":"FAIL",
 "failures":["current source file identity mismatch"]}
```

This is expected source drift, not an invalidation of the source-bound
historical artifact. Its manifest binds eleven source files. The authority
does not contain a `source.tar`, but every bound content hash was present in
the local AI contribution snapshot blob store and was restored to:

```text
/tmp/generic-speculative-tp4-frozen-2026-08-15
```

After exact per-file SHA-256 verification, the frozen verifier produced:

```text
{"classification":"PASS","failures":[]}
```

The fresh receipt exactly matches all four retained `verify*.json` receipts.
This establishes local frozen-source recoverability and replay through an
external snapshot store; it does not provide the durability of a
self-contained source archive.

The retained loaded rows establish, within this fixed scope:

- exact batch-1 and batch-4 baseline/candidate token parity;
- speculative first-target and tail activity with nonzero proposals and
  accepted tokens;
- ranks 0 through 3 each recording eight speculative callback profile rows
  per candidate cell;
- ranks 0 through 3 each recording 456 collective rows with cross-rank
  callback/collective identity agreement;
- two complete `prepare -> precommit -> seal` acknowledgement sequences per
  candidate cell;
- all-rank clean shutdown;
- real target-KV D2H counters, including 20 copies / 146,800,640 bytes per
  rank for candidate batch 1 and 80 copies / 587,202,560 bytes per rank for
  candidate batch 4;
- zero rejected speculative D2H copies.

The old schema must not be overinterpreted. Loaded speculative
committed/rejected block-count deltas are zero. Existing mutation tests show
the validator does not independently bind every rejected aggregate and
treats `kv_decision` as an opaque non-empty string. Collective presence does
not establish overlap, fusion, launch reduction, or performance.

Plan synchronization:

```text
current implementation/execution/GREEN checked: 47
historical RED unchecked:                       9
```

No new GPU, remote, NCCL, loaded-checkpoint, or performance workload was run
during this reconciliation:

```text
GENERIC_QWEN3_TP4_4K_PARITY=ESTABLISHED_WITHIN_RETAINED_SCOPE
GENERIC_QWEN3_TP4_ALL_RANK_CALLBACKS=ESTABLISHED_WITHIN_RETAINED_SCOPE
GENERIC_QWEN3_TP4_COLLECTIVE_PRESENCE=ESTABLISHED_WITHIN_RETAINED_SCOPE
GENERIC_QWEN3_TP4_REAL_TARGET_KV_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPE
GENERIC_QWEN3_TP4_FROZEN_SOURCE_RECOVERY=ESTABLISHED_LOCAL_EXTERNAL_SNAPSHOT
GENERIC_QWEN3_TP4_FROZEN_SOURCE_REPLAY=PASS
GENERIC_QWEN3_TP4_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
GENERIC_QWEN3_TP4_SOURCE_ARCHIVE=ABSENT
TP_COLLECTIVE_OVERLAP=NOT_ESTABLISHED
TP4_PERFORMANCE=NOT_ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Generic Proposal-KV Residency Plan Reconciliation

The all-unchecked state of
`docs/superpowers/plans/2026-08-14-generic-proposal-kv-residency-transaction.md`
was stale relative to the implementation and the existing Tasks 1–7 handoff.
This reconciliation inspected the current interfaces and reran the planned
local matrices without running GPU, remote, NCCL, loaded-checkpoint,
authority, or performance workloads.

Current implementation coverage:

```text
generation-bearing logical entry identities
temporary generation-checked physical leases
direct allocator with literal zero movement counters
GPU/CPU residency state machine
deterministic committed-entry LRU
dirty writeback and clean prefetch batching contracts
deferred retirement behind consumer completions
logical-identity-only durable transaction state
Qwen3.5 MTP storage adapter and allocator construction
lease-derived attention mappings
default-off configuration and V1 validation
separate proposal movement authority snapshots
```

Fresh Torch-enabled evidence:

```text
Tasks 1-7 focused suite:
  100 passed in 2.87s

source-neutral neighboring regression:
  275 passed in 1.92s

Task 1-7 production/test py_compile:
  PASS

scoped git diff --check:
  PASS

durable staged_slot_ids/committed_slot_ids scan:
  PASS

Task 8 authority files:
  ABSENT
```

The focused suite covers allocator generation reuse, lease validation,
deterministic eviction, synchronous fake H2D/D2H batching, rejected-entry
zero-D2H, cache/lifecycle accepted-prefix transactions, Qwen3.5 storage and
executor integration, configuration validation, ModelRunner construction,
default-off behavior, and the local terminal gate. The neighboring suite
covers the generic runtime, batch runtime, target-KV transaction, side state,
independent autoregressive drafter, and Qwen3 draft backend.

These tests do not execute the CUDA copy backend. CPU tensors and the
synchronous injectable backend prove state-machine behavior, not pinned
backing, real H2D/D2H, CUDA stream/event ordering, loaded exact parity, memory
reduction, or performance.

Plan synchronization:

```text
Tasks 1-7 current implementation/GREEN checked: 44
Tasks 1-7 historical RED unchecked:             6
Task 8 authorization steps unchecked:           3
whole-plan checked:                            44
whole-plan unchecked:                           9
```

Task 8 remains an explicit authorization boundary, and none of its five
future gate/worker/verifier/runner/test files exists. Therefore this closes
only the local-plan documentation mismatch:

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=ESTABLISHED
PROPOSAL_KV_TORCH_ENABLED_LOCAL_REGRESSION=375_PASSED
PROPOSAL_KV_TASK8_AUTHORITY_FILES=ABSENT
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PINNED_CPU_BACKING_RUNTIME_AUTHORITY=NOT_ESTABLISHED
CUDA_STREAM_EVENT_ORDERING_AUTHORITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Generic Speculative Runtime Performance Gate Reconciliation

The all-unchecked state of
`docs/superpowers/plans/2026-08-12-generic-speculative-runtime-performance-gate.md`
was stale relative to the current source tree, retained execution artifact,
historical verifier receipts, audit, and handoff.

The implementation still exposes every planned production and gate interface:

```text
ModelRunner.reset_peak_memory_stats()
LLMEngine.reset_peak_memory_stats()
build_prompt_token_batches()
subtract_counter_summaries()
build_run_metrics()
aggregate_measurements()
classify_batch_direction()
run_request_batch()
run_policy_campaign()
validate_worker_result()
build_performance_artifact()
validate_performance_artifact()
verify_performance_artifact()
```

Fresh dependency-light verification:

```text
tools/test_engine_speculative_runtime.py
tools/test_kv_offload_generation_metadata.py
tools/test_speculative_runtime_performance_gate.py:
  107 passed in 0.77s

performance source py_compile:
  PASS

tools/run_speculative_runtime_performance_gate_remote.sh bash syntax:
  PASS
```

The retained campaign remains:

```text
artifact:
  artifacts/speculative_runtime_performance/
    20260812T085852Z/result.json

artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f

schema/status:
  1 / PASS

matrix:
  Qwen3-0.6B
  TP1
  4096 prompt tokens
  64 greedy output tokens
  batch 1 and 4
  baseline versus generic n-gram runtime
  1 warmup + 1 parity + 5 measured runs per cell

batch directions:
  1=IMPROVED
  4=IMPROVED

campaign direction:
  POSITIVE

classification:
  NOT_PROMOTABLE
```

The artifact binds nine source files by SHA-256. A fresh verifier invocation
against the current checkout correctly fails closed because
`tinyvllm/engine/llm_engine.py` has evolved since the campaign:

```text
ValueError: source hash mismatch: tinyvllm/engine/llm_engine.py
```

All nine artifact-bound content hashes were present in the local AI
contribution content-addressed snapshot store. They were restored to:

```text
/tmp/speculative-runtime-performance-frozen-2026-08-15
```

After rechecking every restored file hash, the frozen verifier produced a
fresh `PASS` receipt. Its artifact hash, schema, classification, campaign
direction, and per-batch directions exactly match both historical receipts:

```text
fresh frozen-source status:
  PASS

fresh receipt semantics == historical verify.json:
  true

fresh receipt semantics == historical verify.remote.json:
  true
```

The historical campaign audit also records a direct remote A100
`tools/test_kv_offload.py` result of `kv offload tests passed`; no GPU
regression was rerun during this reconciliation.

Plan synchronization follows the same evidence discipline used for the TP4
reconciliation: current implementation, fresh GREEN, retained execution, and
historically recorded execution steps are checked, while original RED states
not observed in this continuation remain unchecked:

```text
checked:                 29
historical RED unchecked: 6
```

This repairs plan-state drift but does not broaden the artifact's narrow TP1
4K Qwen3/n-gram authority:

```text
GENERIC_NGRAM_TP1_4K_PERFORMANCE_ARTIFACT=ESTABLISHED_WITHIN_RETAINED_SCOPE
GENERIC_NGRAM_TP1_4K_FROZEN_SOURCE_RECOVERY=ESTABLISHED_LOCAL_EXTERNAL_SNAPSHOT
GENERIC_NGRAM_TP1_4K_FROZEN_SOURCE_REPLAY=PASS
GENERIC_NGRAM_TP1_4K_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
GENERIC_NGRAM_TP1_4K_DIRECTION=POSITIVE
SECOND_MODEL_PERFORMANCE=NOT_ESTABLISHED
TP4_PERFORMANCE=NOT_ESTABLISHED
16K_32K_CONTROLLED_PERFORMANCE=NOT_ESTABLISHED
LEARNED_DRAFTER_PERFORMANCE=NOT_ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## 2026-08-15 Completion-Audit Reconciliation

The Phase 1 objective is complete only if the repository has all of the
following deliverables, not merely local contracts:

| Explicit objective requirement | Concrete evidence inspected | Current result |
| --- | --- | --- |
| One source-neutral speculative runtime for MTP, independent draft, n-gram, and SAM | generic adapter/runtime, ModelRunner executor registry, native MTP registration, Qwen3 autoregressive draft backend, n-gram/SAM adapters and focused tests | Runtime abstraction is established; SAM still lacks current loaded performance authority |
| Batch-native multi-token verification with exact greedy semantics | batch verifier, selection/commit bridge, TP1/TP4 correctness artifacts cited by the promotion checklist | Established only within cited model/context cells |
| Transactional Proposal-KV accepted-prefix commit and rejected-suffix rollback without accepted-token rematerialization | Proposal-KV cache/lifecycle, native-MTP executor, autoregressive-draft executor, production-symbol scan, focused local regressions | Shared local contract established; real learned-source zero-rematerialization movement remains unproved |
| Two model structures | retained Qwen3 and Qwen3.5 authorities plus independent Qwen3-draft/Qwen3.5-target local gate | Existing source-neutral correctness authorities cover two target structures, but the independent learned-drafter loaded authority is absent |
| TP1 and TP4 | retained TP1/TP4 artifacts and local TP4 learned-drafter gate | Partial; learned-drafter TP4 loaded execution is absent |
| 4K, 16K, and 32K or longer | retained generic/native authority artifacts | Partial; native MTP TP4/32K exact parity failed |
| Batch 1, batch 4, and multi-sequence | retained raw rows and validators cited by the promotion checklist | Established only within cited cells, not uniformly across every learned source and optimization |
| Exact greedy parity | retained raw output rows and independent verifier receipts | Partial; learned-drafter TP1/TP4 loaded parity and native MTP TP4/32K parity are absent |
| Real KV H2D/D2H rather than simulated copies | retained ordinary target-KV movement artifacts and counters | Established for ordinary target KV in cited scopes; real Proposal-KV movement is absent |
| TPOT, TTFT, throughput, memory, H2D bytes, and acceptance | retained performance/correctness artifacts and payload re-audit | Partial; there is no controlled learned-source performance and Proposal-KV movement campaign |
| Variable proposal-length CUDA Graph | local exact-family producer/verifier contract and legacy retained artifact | Partial; no source-bound PASS artifact, archived verifier receipt, TP4/offload/long-context or performance authority |
| Source-bound, independently verifiable promotion artifacts | nine retained manifest-backed authorities plus learned-drafter schema-v2 bundle machinery | Partial; provenance is not uniform and the learned-drafter bundle has no loaded retained execution |

Fresh reconciliation evidence:

```text
learned-drafter schema-v2 gate, archived verifier, TP4 validator,
snapshot transport, executor, registration, and ModelRunner integration:
  217 passed in 6.25s

learned-drafter frozen source inventory:
  SOURCE_INVENTORY_CHECK=PASS files=30 missing=[]

learned-drafter gate/verifier py_compile:
  PASS

unchecked extractall() scan:
  PASS

paired verify trace complete local regression:
  294 passed in 9.62s

paired verify trace final focused rerun:
  50 passed in 4.27s
```

The first attempted learned-drafter rerun used the `pytest` console script and
failed collection because the isolated console entry did not place the
repository root on `sys.path`. The same unchanged test matrix passed under
`python -m pytest`, which preserves the repository root and imports the
`tools` namespace. This was an invocation-environment failure, not a product
test failure.

No file or directory matching a retained learned-drafter TP4 loaded execution
artifact was found under `artifacts/` or `experiments/`. Therefore the local
schema-v2 publisher and archived verifier must not be classified as a loaded
authority.

## Current Objective Coverage

### Direction 1: Improve KV-Cache Utilization

| Requirement | Current artifact or source | Classification | Remaining boundary |
| --- | --- | --- | --- |
| Ordinary target-KV logical/physical decoupling | `KVOffloadMVP0` in `tinyvllm/engine/model_runner.py`; loaded real-movement artifacts cited by the promotion checklist | `ESTABLISHED` within cited scopes | No unified KV4/KV8 plus offload matrix |
| Async and batched target-KV migration | Dedicated copy-stream/event and batched-pair implementation; retained production H2D/D2H counters | Real movement `ESTABLISHED`; async/batch runtime classification `PARTIAL` | Result schemas omit async mode, batch counts, and batch spans |
| Dirty target-KV writeback | Source-bound `mark_dirty()`, `writeback_dirty()`, dirty eviction, and positive D2H counters | Real writeback `ESTABLISHED` within cited scopes | No retained `evict_dirty` or explicit-versus-eviction writeback receipt |
| Proposal-KV logical identity decoupled from physical slots | `tinyvllm/engine/proposal_kv_allocator.py`; `tools/test_proposal_kv_allocator.py` | `ESTABLISHED` as a local contract | Real learned-source movement is not established |
| Proposal-KV GPU/CPU residency and batching | `tinyvllm/engine/proposal_kv_residency.py`; `tools/test_proposal_kv_residency.py` | `ESTABLISHED` as a dependency-light local contract | No CUDA copy, pinned-memory, or performance authority |
| Proposal-KV accepted-prefix commit and rejected-suffix retirement | `tinyvllm/engine/proposal_kv_cache.py`; `tinyvllm/engine/proposal_kv_lifecycle.py`; native-MTP and independent-drafter executor tests | `ESTABLISHED` as a shared local transaction contract | Loaded learned-source zero-rematerialization remains unproved |
| Native Qwen3.5 MTP allocator integration | `tinyvllm/engine/qwen35_mtp_registration.py`; `tinyvllm/engine/qwen35_mtp_executor.py`; local physical-KV and lease tests | `ESTABLISHED` locally | Real proposal-KV H2D/D2H remains unproved |
| Independent Qwen3 drafter allocator integration | `tinyvllm/engine/autoregressive_draft_executor.py`; `tinyvllm/engine/qwen3_draft_backend.py`; learned-drafter local TP1/TP4 matrix | `ESTABLISHED` locally | Loaded TP1/TP4 parity remains unproved |
| Independent Qwen3 multi-layer residency payload | `tinyvllm/engine/qwen3_draft_proposal_kv.py`; learned-drafter configuration and ModelRunner registration; focused local tests | `ESTABLISHED` as local default-off runtime wiring | No real H2D/D2H or loaded parity authority |
| Prefix sharing, deduplication, and refcounts | Ordinary KV block hashing/refcounts plus local prefix/offload composition tests; Qwen3.5 tensor-interning tests | `ESTABLISHED` as local contracts | No retained loaded artifact records a cross-request prefix hit, shared-ref lifetime, or deduplicated-byte receipt; hybrid-state genericization remains incomplete |
| Independent KV4/KV8 storage and routing | Packed INT4/INT8 allocation, per-group scales, cache writes, and dequantization paths; fresh KV4 CPU reference and KV8 CPU-reference-plus-actual-dequant round-trips; fresh KV8 cached-prefill routing test | `PARTIAL` | No authorized Triton store-kernel execution, loaded parity, memory, performance, or retained execution receipt |
| KV4/KV8 plus offload | Configuration and blockwise attention intentionally reject this combination; local verifier paths reject quantized KV before transaction work | `NOT_ESTABLISHED` | Separate precision/residency design and authority required |
| Per-layer/per-token heat grading | Physical-slot LRU and fixed dirty/future/pending eviction penalties only | `MISSING` | No layer/token heat, hot/warm/cold identities, precision transitions, or execution artifact; separate design required |

### Direction 2: Support Longer Context

| Requirement | Current artifact or source | Classification | Remaining boundary |
| --- | --- | --- | --- |
| Chunked prefill | Scheduler and chunked-prefill tests; Qwen3/Qwen3.5 long-context authorities | `ESTABLISHED` within cited exact-greedy scopes | No universal performance claim |
| Blockwise online-softmax prefill/decode/spec-verify | `tinyvllm/layers/attention.py`, focused dense-oracle tests, and native TP4/16K source-bound dispatch/configuration | `PARTIAL` end to end | No retained result directly observes the selected attention path; generic 16K/32K manifests omit the blockwise attention implementation; native MTP TP4/32K batch-1 parity remains failed |
| Bounded GPU residency with future-window staging | Target-KV offload manager and blockwise staging paths; native TP4/16K peak/capacity rows | `ESTABLISHED` for native TP4/16K bounded capacity; `PARTIAL` for direct future-window execution observation | Generic authorities establish real movement/lifecycle but not blockwise source binding; proposal-KV movement has no real authority |
| Prefix cache plus CPU-resident backing | `tools/test_prefix_kv_offload_integration.py` | `ESTABLISHED` as a dependency-light scheduling/identity contract | The test records requested H2D pairs rather than executing CUDA copies; no loaded prefix-restore authority or standalone performance promotion |

### Direction 3: Lower TPOT

| Requirement | Current artifact or source | Classification | Remaining boundary |
| --- | --- | --- | --- |
| Source-neutral speculative runtime | Generic adapter/runtime, ModelRunner executor registry, batch-native verifier, Scheduler transaction bridge | `ESTABLISHED` |
| Model-free n-gram | Two target structures, TP1/TP4, 4K/16K/32K, batch 1/4 exact-parity authorities | `ESTABLISHED` within cited scopes |
| Model-free SAM | Adapter and local lifecycle tests | `PARTIAL` | No current loaded performance authority |
| Native Qwen3.5 MTP | TP1/4K, TP4/4K, and TP4/16K target-KV-offload authorities; local paired target-forward trace implementation | `PARTIAL` | TP4/32K failed; paired-trace readiness does not provide a first-divergence artifact, root cause, controlled MTP performance, or proposal-KV movement authority |
| Independent learned Qwen3 drafter | Concrete model/backend/TP contracts, generation-aware allocator leases, multi-layer storage adapter, TP1 authority harness, schema-v2 TP4 gate, reconstructable accepted-prefix rows, deterministic source archive, archived verifier, and atomic bundle publisher | `PARTIAL` | The source-bound bundle contract is established locally, but no loaded TP4 execution, retained exact-parity bundle, real Proposal-KV movement, or controlled performance result exists |
| Variable-Q graph families | Legacy TP1/no-offload artifact with `Q=(1,2,3,4)`, batch `(1,4)`, top-level graph booleans/counts, and 28 transaction rows; separate exact `(B,Q,W)` producer/verifier contract with raw per-family outputs | `PARTIAL` | Legacy graph/eager claims remain non-reconstructable. The exact-family semantic verifier is established locally with current-source hashes, but no PASS artifact, archived verifier, TP4, offload, long-context, or performance authority exists |
| Verifier/sampling/KV-commit fusion | Batch-native verification, token-free transactional KV commit, and prepared Scheduler publication exist as separate ordered phases | `MISSING` | No fused kernel/runtime, launch-count receipt, fused graph node, or fusion-specific performance attribution |
| TP collective overlap and reduction fusion | Cited TP4 artifacts record synchronous AllReduce participation on all ranks; source uses blocking collectives and replicated row-parallel outputs | `MISSING` | No async overlap, AllReduce fusion, ReduceScatter, persistent hidden-state sharding, or optimization-specific performance authority |

## Variable-Q CUDA Graph Authority Re-Audit Delta

Two different evidence campaigns must not be conflated.

The retained legacy Qwen3.5 MTP artifact:

```text
artifacts/qwen35-mtp-runs/
  qwen35-mtp-graph-gate-opaque-7/
    qwen35_mtp_real_checkpoint_gate.json
```

records `Q=(1,2,3,4)`, batch `(1,4)`, six captures, twelve replays,
graph/eager equality booleans, and 28 transaction rows. The transaction rows
cover every `(batch_size, q, accepted=0..q)` tuple, and their
staged/committed/released slot sets are independently recomputable. The
artifact does not retain per-family graph/eager token arrays, logits,
capture/replay rows, backend receipts, or source closure, so its parity
booleans and aggregate graph counters remain producer assertions.

The newer exact-family gate fixes the semantic schema gap:

```text
tools/spec_verify_cuda_graph_smoke.py
tools/verify_spec_verify_cuda_graph_gate.py
tools/run_spec_verify_cuda_graph_gate_remote.py
```

For the fixed TP1/no-offload/4K matrix, it requires all eight exact
`(B,Q,W)` families across batch `(1,4)`, query length `(1,3)`, and page-table
width `(1,2)`. Its independent standard-library verifier recomputes:

```text
eager-versus-graph logits SHA equality
target-token equality
accepted-length equality
final-token equality
accepted-prefix KV SHA equality
one warmed graph replay and zero eager forward
materialized/committed/released transaction-set equations
post-replay failure propagation, no eager retry, and stable quarantine
```

The producer binds an explicit 11-file inventory through per-file SHA-256 and
an aggregate `source_sha256`. Verification still reads those files from the
current checkout; the runner does not retain a deterministic source archive,
execute an archived verifier, or publish matching current/archived receipts.
The only retained exact-family records are fail-closed preflight blockers:

```text
experiments/spec_verify_cuda_graph/
  preflight-20260812-idle-gpu-blocked.json
  preflight-20260812-task9-refresh-idle-gpu-blocked.json
```

Neither record uploaded source nor started the CUDA gate. No correctness or
performance PASS artifact exists.

Fresh local evidence:

```text
variable-Q config/cache/ModelRunner/smoke/verifier/runner plus
legacy Qwen3.5 MTP gate:
  248 passed in 0.94s

changed gate/verifier py_compile:
  PASS

explicit exact-family source inventory:
  11 files
```

```text
VARIABLE_Q_RUNTIME_CONTRACT=ESTABLISHED_LOCAL
VARIABLE_Q_EXACT_FAMILY_SEMANTIC_VERIFIER_CONTRACT=ESTABLISHED_LOCAL
VARIABLE_Q_EXACT_FAMILY_CURRENT_SOURCE_HASH_BINDING=ESTABLISHED_LOCAL
VARIABLE_Q_LEGACY_RETAINED_ARTIFACT_AUTHORITY=PARTIAL
VARIABLE_Q_EXACT_FAMILY_PASS_ARTIFACT=ABSENT
VARIABLE_Q_SOURCE_BOUND_ARCHIVED_VERIFIER=NOT_ESTABLISHED
VARIABLE_Q_TP4=NOT_ESTABLISHED
VARIABLE_Q_OFFLOAD=NOT_ESTABLISHED
VARIABLE_Q_LONG_CONTEXT=NOT_ESTABLISHED
VARIABLE_Q_PERFORMANCE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## KV4/KV8 Completion-Audit Delta

The current checkout has independent quantized target-KV implementations:

```text
KV4:
  packed int8 storage with head_dim/2 final dimension
  symmetric per-group scales
  dedicated store and dequantization functions

KV8:
  int8 storage with head_dim final dimension
  symmetric per-group scales
  dedicated store and dequantization functions
```

Composition remains fail closed:

```text
kv_offload_mvp0 => kv_quant_bits == 0
blockwise prefill/decode => kv_offload_mvp0
blockwise attention asserts kv_quant_bits == 0
spec_verify rejects kv_quant_bits != 0
AM compact rejects KV4 and permits only FP KV or KV8
```

Fresh local evidence:

```text
focused verifier/fail-closed matrix: 3 passed

KV4 NumPy reference round-trip:
  group_size=32  max_err=2.9071 <= bound=3.9056
  group_size=64  max_err=3.1450 <= bound=3.9056
  group_size=128 max_err=3.1450 <= bound=3.9056

KV8 CPU reference quantization plus actual dequant_kv_blocks_q8():
  group_size=32  max_err=0.15293193 <= bound=0.16280018
  group_size=64  max_err=0.15620777 <= bound=0.16280018
  group_size=128 max_err=0.16151990 <= bound=0.16280018
  exact dequant parity and padded block-table handling passed

KV8 cached-prefill routing:
  tools/test_qwen35_full_attention_shell.py::
    test_cached_prefill_quantized_kv_uses_original_backend
  1 passed in 1.38s
```

The KV8 numerical check used an isolated CPU Torch environment. It AST-extracted
and executed the current pure-Torch `dequant_kv_blocks_q8()` implementation
from `tinyvllm/layers/attention.py`; inputs came from a CPU reference quantizer
matching the current `store_kvcache_q8_kernel` formula. This is local numerical
and routing evidence only. It does not execute the Triton store kernel and is
not a loaded-model, memory-reduction, performance, or retained production
execution receipt. No GPU workload was authorized or run.

A structured-key scan of every retained JSON/JSONL artifact found no
`kv_quant_bits`, quantized cache dtype, KV-scale, group-size, or
quantized-execution field. Existing relevant results explicitly retain
`no KV8/KV4 evidence` limitations. The exact classification is:

```text
KV4_CPU_REFERENCE_ROUNDTRIP=ESTABLISHED
KV4_KV8_STORAGE_AND_ROUTING_CONTRACT=PARTIAL
KV4_KV8_OFFLOAD_BLOCKWISE_COMPOSITION=INTENTIONALLY_REJECTED
KV4_REAL_GPU_TRITON_ROUNDTRIP=NOT_RUN
KV8_CPU_REFERENCE_ACTUAL_DEQUANT_ROUNDTRIP=ESTABLISHED
KV8_CACHED_PREFILL_ROUTING=ESTABLISHED_LOCAL
KV8_TRITON_STORE_KERNEL_ROUNDTRIP=NOT_RUN
KV4_KV8_LOADED_PARITY=NOT_ESTABLISHED
KV4_KV8_MEMORY_REDUCTION=NOT_ESTABLISHED
KV4_KV8_PERFORMANCE=NOT_ESTABLISHED
KV4_KV8_RETAINED_EXECUTION_ARTIFACT=ABSENT
```

## Heat-Tiering Completion-Audit Delta

The nearest existing mechanism is `KVOffloadMVP0` eviction scoring:

```text
_touch(slot):
  records one recency clock per physical GPU slot

_victim_score(slot, future_logical_blocks):
  LRU recency
  + fixed dirty penalty
  + fixed future-window penalty
  + fixed pending-H2D penalty

blockwise staging:
  supplies bounded future block sets to eviction scoring
```

This is a homogeneous FP staging-cache eviction policy, not a generic
per-layer/per-token heat-tier policy. There are no:

```text
layer_heat or token_heat states
frequency counters with decay
hot/warm/cold block identities
residency or precision tier enums
promotion/demotion thresholds or transactions
FP/KV8/KV4 transition ownership
tier-aware speculative commit/rollback receipts
```

Fresh local evidence:

```text
production _victim_score AST probe:
  PASS

fixed-score progression:
  base=5.0
  future=805.0
  dirty+future=1205.0
  pending+dirty+future=1805.0

prefix CPU-backing identity/generation matrix:
  3 passed
```

The selected direct `tools/test_kv_offload.py` nodes did not collect because
the local interpreter lacks `torch`; that limitation is not converted into a
PASS or implementation failure.

A strict retained-artifact scan found zero structured KV heat/tier fields.
Unrelated names such as `promotion_classification`, `warmup_runs`, and
`rank_snapshots` were rejected as false positives.

```text
KV_LRU_RECENCY_POLICY=ESTABLISHED_IN_SOURCE
KV_FUTURE_DIRTY_PENDING_EVICTION_BIAS=ESTABLISHED_IN_SOURCE
PER_LAYER_KV_HEAT=NOT_IMPLEMENTED
PER_TOKEN_KV_HEAT=NOT_IMPLEMENTED
HOT_WARM_COLD_KV_STATE_MACHINE=NOT_IMPLEMENTED
PRECISION_TIER_TRANSITIONS=NOT_IMPLEMENTED
HEAT_TIER_TRANSACTIONAL_KV_COMPOSITION=NOT_IMPLEMENTED
HEAT_TIER_RETAINED_EXECUTION_ARTIFACT=ABSENT
HEAT_TIER_LOADED_PARITY=NOT_ESTABLISHED
HEAT_TIER_MEMORY_REDUCTION=NOT_ESTABLISHED
HEAT_TIER_PERFORMANCE=NOT_ESTABLISHED
```

## Verifier/Sampling/Commit Fusion Completion-Audit Delta

The current production workflow is an explicit multi-phase transaction:

```text
target verification forward(s)
  -> target token extraction
  -> acceptance / greedy selection
  -> prepared runtime result
  -> prepared Scheduler output rows
  -> prepared KV commit plans
  -> optional residency precommit
  -> side-state apply
  -> token-free KV batch commit
  -> Scheduler publication
  -> optional residency seal
```

Separate runtime observability confirms the phase boundaries:

```text
target_forward_ms
accept_sample_ms
commit_metadata_ms
prepared side-state state
prepared KV plans
prepared Scheduler publication
```

`verify_and_commit_block()` is an aggregate Python function name, not evidence
of a fused kernel. Its body still performs verification, host-visible
argmax/acceptance, KV materialization, commit, and finish checks separately.

Fresh local evidence:

```text
phase-order and publication matrix:
  5 passed
```

The matrix establishes callback ordering, token-free exactly-once KV batch
commit, and exactly-once Scheduler token publication. It does not establish
kernel fusion or launch reduction.

A strict retained JSON/JSONL scan found zero structured fusion, kernel-launch,
or fused graph-node fields.

```text
BATCH_NATIVE_MULTI_TOKEN_VERIFICATION=ESTABLISHED_WITHIN_CITED_SCOPES
TRANSACTIONAL_TOKEN_FREE_KV_COMMIT=ESTABLISHED
PREPARED_SCHEDULER_PUBLICATION=ESTABLISHED
VERIFY_SAMPLE_COMMIT_PHASE_ORDER=ESTABLISHED_LOCALLY
VERIFY_SAMPLE_KV_COMMIT_KERNEL_FUSION=NOT_IMPLEMENTED
VERIFY_SAMPLE_KV_COMMIT_RUNTIME_FUSION=NOT_IMPLEMENTED
FUSION_KERNEL_LAUNCH_REDUCTION=NOT_ESTABLISHED
FUSION_RETAINED_EXECUTION_ARTIFACT=ABSENT
FUSION_PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
```

## TP Collective Optimization Completion-Audit Delta

Current production collective boundaries:

```text
embedding:
  blocking dist.all_reduce

row-parallel decode:
  blocking dist.all_reduce per output/chunk

dense prefill preservation:
  blocking dist.all_gather plus concatenation

profiler:
  times the complete blocking call
```

There is no `async_op=True`, pending-work ownership, communication stream, or
deferred synchronization contract. Row-parallel partial outputs are reduced
to replicated outputs at each boundary, so current TP weight sharding does not
preserve hidden-state sharding across layers.

Fresh retained-artifact inventory:

```text
Qwen3 generic TP4:
  10032 collective rows
  9856 row-parallel AllReduce
  176 embedding AllReduce

Qwen3.5 generic TP4/4K:
  5856 collective rows
  5760 row-parallel AllReduce
  96 embedding AllReduce

combined:
  15888 synchronous collective rows
  0 async_op fields
```

A strict artifact scan found zero overlap, AllReduce-fusion, ReduceScatter, or
persistent-hidden-sharding fields. Source inspection found no production
`tinyvllm` ReduceScatter call.

Fresh local evidence:

```text
collective profiler/inventory/mutation matrix:
  6 passed
```

The matrix proves collective observability and cross-rank identity checking,
not communication/compute overlap.

```text
TP4_ALL_RANK_COLLECTIVE_PARTICIPATION=ESTABLISHED_WITHIN_CITED_SCOPES
TP4_COLLECTIVE_IDENTITY_VALIDATION=ESTABLISHED
TP_COLLECTIVE_CALLS_SYNCHRONOUS=ESTABLISHED_IN_SOURCE
TP_COLLECTIVE_COMPUTE_OVERLAP=NOT_IMPLEMENTED
SPECULATIVE_ALLREDUCE_FUSION=NOT_IMPLEMENTED
SPECULATIVE_REDUCESCATTER=NOT_IMPLEMENTED
PERSISTENT_HIDDEN_STATE_SHARDING=NOT_IMPLEMENTED
TP_OVERLAP_FUSION_RETAINED_ARTIFACT=ABSENT
TP_OVERLAP_FUSION_PERFORMANCE=NOT_ESTABLISHED
```

## Proposal-KV Tasks 1-7 Delta

The 2026-08-15 local implementation newly establishes:

1. generation-aware `ProposalKVEntryIdentity`;
2. direct and residency-backed allocator contracts;
3. temporary readable/writable leases with physical occupancy generations;
4. committed LRU, H2D/D2H batching, dirty writeback, and deferred retirement
   state machines;
5. `ProposalKVCache` durable state expressed only in logical identities;
6. accepted-prefix in-place commit and rejected-suffix retirement with
   `writeback=False`;
7. Qwen3.5 MTP storage, registration, executor lease mapping, configuration,
   and local terminal classification gates.

Fresh recorded local evidence before this audit:

```text
Proposal-KV Tasks 1-7 dependency-light suite: 78 passed
generic speculative runtime regression:         170 passed
py_compile:                                     PASS
scoped git diff --check:                        PASS
```

The following classifications remain mandatory:

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Independent Learned-Drafter Local Delta

The 2026-08-15 learned-drafter continuation now establishes:

1. default `Qwen3DraftPhysicalSlotStore -> DirectProposalKVAllocator ->
   ProposalKVCache` construction;
2. executor-owned readable and writable leases around prompt and decode
   forwards;
3. durable logical identities with physical mappings confined to ephemeral
   rows;
4. a reusable `Qwen3DraftProposalKVStorage` whose logical entry contains K/V
   payload for every local draft-model layer;
5. exact all-layer entry-byte accounting and validated batched copy methods;
6. failure-atomic attention-backend binding;
7. backend compatibility with either direct allocator `.physical_store` or
   residency manager `.storage`;
8. explicit learned-drafter offload configuration with
   `logical == cpu > gpu > 0` validation;
9. one storage-aware allocator builder that selects direct or generic
   residency allocation;
10. ModelRunner registration that resolves either allocator storage shape
    before failure-atomic all-rank publication;
11. nested backend/cache/allocator authority snapshots that expose allocator
    mode, H2D/D2H entries and bytes, and accepted copy/replay/rematerialization
    counters without tensors;
12. default-off registration with no CPU backing, copy backend, or transfer
    stream;
13. a schema-v2 loaded TP1 authority gate with explicit direct versus
    residency configuration, workload-derived logical/CPU capacity, and a
    caller-supplied bounded GPU slot capacity;
14. before/after deltas from
    `executor.backend.proposal_kv_cache.entry_allocator` for H2D, D2H,
    accepted copy, replay, and rematerialization counters;
15. separate exact-parity and real-bidirectional-movement classifications, so
    configuration or simulated copies cannot establish movement; and
16. the current `owned_entry_count` cache snapshot field in place of the
    stale `owned_slot_count` harness lookup;
17. an engine-level
    `autoregressive_draft_authority_snapshots(timeout_s=...)` transport that
    collects rank 0 plus acknowledged worker snapshots and rejects field,
    rank, world-size, duplicate-rank, or inventory mismatches; and
18. a production `tools/autoregressive_draft_tp4_local_gate.py` validator,
    promoted out of its former test-only location, that requires four
    registered direct-allocator ranks, identical registration/model/tokenizer
    authority, zero accepted Proposal-KV copy/replay/rematerialization, and
    terminal transaction/entry/physical-slot cleanup;
19. a schema-v2 `tools/autoregressive_draft_tp4_engine_gate.py` that runs
    target-only before learned execution, passes direct-only TP4 engine
    configuration, compares batch-1 and batch-4 exact greedy outputs, retains
    every acceptance event with reconstructable prompt/step/output/proposal
    and accepted-prefix identity, and validates raw four-rank snapshots
    through the production TP4 validator;
20. a production TinyLLM adapter that activates
    `EngineSpeculativeRuntime(model_runner_executor=descriptor)`, flushes
    pending hybrid-state releases, and collects
    `autoregressive_draft_authority_snapshots(timeout_s=60)` after each learned
    case; and
21. a prompt-file CLI with four-GPU/positive-port validation, scoped
    `CUDA_VISIBLE_DEVICES`/`TINYVLLM_DIST_PORT`/`MASTER_PORT` restoration, and
    legacy exclusive JSON output plus source-bound atomic bundle publication
    that refuses to replace an existing artifact; and
22. direct independent-drafter reuse of `ProposalKVLifecycle`: proposal
    transactions stage `exact_q - 1` physical entries, partial acceptance
    commits exactly `max(accepted - 1, 0)`, rollback retains the prompt and
    releases the staged suffix, and repeated accepted rounds append without
    bootstrap replay.

Fresh local evidence for the combined learned-drafter campaign:

```text
TP1 loaded authority gate contract:       34 passed
TP4 snapshot transport/direct validator:  15 passed
TP4 loaded gate contract:                  25 passed
learned-drafter full local matrix:        305 passed in 15.22s
Proposal-KV Tasks 1-7 regression:         78 passed
generic speculative runtime regression: 170 passed
focused 18-file py_compile:              PASS
TP4 gate CLI --help:                     PASS
stale production symbol scan:            PASS
scoped git diff --check:                 PASS
```

Fresh focused direct-commit re-audit:

```text
autoregressive executor, registration, ModelRunner integration,
TP4 direct validator, and TP4 snapshot transport:
  176 passed in 15.29s

production learned-drafter rematerialization symbol scan:
  PASS
```

The scan covers the production autoregressive executor, ModelRunner,
`LLMEngine`, proposal cache, and proposal lifecycle. None references
`rematerialize_accepted_kv`, `legacy_rematerialize`, or the legacy
`commit_accepted_tokens` helper.

The 305-test matrix, focused `py_compile`, and TP4 gate `--help` checks were
freshly rerun after the completion-audit fixture/schema repairs. They remain
dependency-light local evidence only; no GPU, NCCL, remote host,
loaded-checkpoint, or performance workload was used.

The TP4 gate tests above are dependency-injected local contract evidence.
No GPU, NCCL, remote host, real Qwen3/Qwen3.5 checkpoint, or performance
workload was executed for this gate.

Required classifications:

```text
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_ALLOCATOR_REUSE=ESTABLISHED
QWEN3_DRAFT_MULTILAYER_PROPOSAL_KV_STORAGE_ADAPTER=ESTABLISHED
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_RUNTIME_WIRING=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_DEFAULT=DISABLED
AUTOREGRESSIVE_DRAFT_OFFLOAD_OBSERVABILITY_CONTRACT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_TP1_OFFLOAD_AUTHORITY_HARNESS=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_TP4_SNAPSHOT_TRANSPORT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_TP4_DIRECT_VALIDATOR=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_TP4_LOADED_GATE_CONTRACT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_ACCEPTED_PREFIX_DIRECT_COMMIT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_REJECTED_SUFFIX_ROLLBACK=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_PRODUCTION_REMATERIALIZATION_SYMBOLS=ABSENT
LEARNED_DRAFTER_TP4_LOADED_EXECUTION=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Target Accepted-KV Direct-Commit Re-Audit

The current production generic speculative runtime no longer uses the older
accepted-token decode replay path. The locally established target-KV sequence
is:

1. `prepare_native_speculative_batch()` reserves speculative target-KV blocks;
2. the verifier tail forward writes directly into the reserved physical slots;
3. `mark_speculative_kv_materialized(transaction, plan.query_len)` records the
   exact written prefix;
4. `prepare_speculative_kv_commit()` creates a token-free publication plan;
5. `commit_speculative_kv_commit_batch()` transfers required block ownership,
   publishes eligible full-block hashes, and releases the unused reserved
   suffix;
6. one prepared Scheduler metadata commit publishes the accepted tokens.

No production reference to `rematerialize_accepted_kv` or
`legacy_rematerialize` exists in `tinyvllm/engine/llm_engine.py`,
`tinyvllm/speculative/batch_runtime.py`, or
`tinyvllm/engine/block_manager.py`. The remaining
`legacy_rematerialize` implementation is an explicit comparison mode in
`tools/profile_ngram_commit.py`; its native mode reports zero accepted-KV
decode replay and copy calls.

Fresh focused local evidence:

```text
target-KV transaction, prepared batch/engine commit, rollback, and
native verifier zero-replay gate: 31 passed in 0.34s
production legacy-rematerialization symbol scan: PASS
```

This closes the old local production-runtime gap, but it does not replace a
loaded GPU/TP trace or establish a speedup:

```text
ACCEPTED_TARGET_KV_DIRECT_COMMIT=ESTABLISHED_LOCAL
LEGACY_REMATERIALIZATION_PROFILER_COMPARATOR=RETAINED
REAL_GPU_TP_ZERO_REMATERIALIZATION=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Completion-Audit Regression

The final prompt-to-artifact audit ran the Phase 1 contracts in isolated
processes because several dependency-light tests intentionally install
file-local module stubs. The first combined run exposed two stale
`KVOffloadMVP0` AST fixtures, not a production initialization omission:

- `tools/test_prefix_kv_offload_integration.py` compiled the class without
  injecting `H2DSlotReuseDiagnostic` and constructed it through `__new__`
  without calling `_initialize_h2d_slot_reuse_diagnostic()`;
- `tools/test_kv_offload_generation_metadata.py` had the same incomplete
  extracted-class construction contract.

Both fixtures now inject the real diagnostic type and initialize it in
default-off mode with an event factory that fails if an event is allocated.
This preserves the required default-off boundary: no CUDA event, CPU backing,
or transfer stream is created by these dependency-light managers.

The audit also found one stale exact-dictionary assertion in
`tools/test_chunked_prefill.py`. `LLMEngine.step()` now intentionally records
both `speculative_proposal_token_ids_by_seq` and
`speculative_accepted_draft_token_ids_by_seq`; the non-speculative branch
correctly emits empty mappings. The test expectation was updated to the
current observation schema without changing production behavior.

Fresh isolated evidence:

```text
proposal-KV, generic runtime, engine transaction, and prefix group:
  199 passed
model-free n-gram:
   59 passed
SAM gate:
   16 passed
chunked prefill and Engine observation schema:
  100 passed
native verifier attention dispatch:
    4 passed
dependency-light Phase 1 regression total:
  378 passed
H2D diagnostic ownership + generation metadata + prefix composition:
   37 passed
changed-test scoped git diff --check:
  PASS
```

`tools/test_blockwise_attention_planning.py` was not collected in the current
host environment because importing the production package requires
`flash_attn`, which is unavailable locally. This is an environment
limitation, not a passing result and not a new functional failure. Existing
source review and source-bound loaded artifacts remain the available
blockwise evidence; rerun this file in the approved CUDA/FlashAttention
environment before treating it as fresh current-worktree test evidence.

The completion audit therefore confirms the local contracts are internally
consistent after fixture/schema maintenance, but it does not change any
promotion classification:

```text
LEARNED_DRAFTER_TP4_LOADED_EXECUTION=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Focused H2D Causal-Diagnostic Readiness Delta

The ordinary TP4/32K H2D physical-slot-reuse diagnostic has a local
producer/gate/verifier implementation. It is constrained to baseline
`observe/control` cells at batch sizes 1 and 4 and retains compact logits only
for prediction indices 0 and 1. Non-baseline policy is rejected before loading
the frozen 32K worker.

Fresh local validation:

```text
attention marker contract
diagnostic state machine
KV-offload manager integration
focused campaign/gate/worker/verifier

result:
  99 passed in 0.45s

adjacent dependency-light regressions:
  169 passed in 3.98s

diagnostic/runtime/gate/worker/verifier py_compile:
  PASS
```

The audit found a real local path mismatch before any loaded run. The focused
worker emits `prefill` for the first ordinary scheduling step, while the real
`ModelRunner` wrapper accepted only `decode`. A RED receipt test reproduced
the failure. The wrapper now accepts exactly `prefill` and `decode`, rejects
`spec_verify`, and the focused/adjacent suites above are GREEN.

The gate audit additionally produced two RED→GREEN workload-identity
regressions. Before the correction, coordinated mutation of both batch-4
observe/control fixtures could preserve the causal matrix shape and produce a
false `SUPPORTED` result even though the compared workloads differed. The
gate now requires exact prompt-0 token identity across all four cells and exact
prediction-index-0/1 semantic identity
(`input_token_id`, `position`, `context_length`) across all four cells, with
`context_length == position + 1`. Missing or mismatched identity is
`INCONCLUSIVE`.

The default system Python lacks Torch. An existing Homebrew Python 3.12
environment provides CPU-only `torch 2.12.0`; an isolated `/tmp` bridge reused
only the pure-Python pytest runner dependencies. With that environment:

```text
test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py:
  35 passed in 3.57s

test_kv_offload.py:
test_blockwise_attention_planning.py:
  collection blocked by missing flash_attn
  production import path also requires triton
```

The latter test files explicitly require a Torch/CUDA environment. No existing
dependency-light flash-attn/triton stub harness was found, and fabricating one
would not prove the CUDA contract. Those two collection errors are not treated
as test passes.

Task-8 prompt-to-artifact mapping:

| Requirement | Local artifact coverage | Boundary |
| --- | --- | --- |
| Physical slot plus occupancy generation | `test_slot_generation_increments_for_every_assignment`, stale-generation rejection | `ESTABLISHED_LOCALLY` |
| All mapping transitions covered | Initial assignment, reassignment, clear, discard, eviction, contiguous reorder, atomic restore tests | `ESTABLISHED_LOCALLY` |
| Decode/spec-verify/prefill read-marker placement | Four AST placement/forwarding tests | `ESTABLISHED_LOCALLY` |
| Observe records without waits | Observe/control wait parameterization | `ESTABLISHED_LOCALLY` |
| Control waits unique predecessors only | Multi-stream retention and control predecessor coverage tests | `ESTABLISHED_LOCALLY` |
| Diagnostic events separate from production H2D/D2H events | Inventory-preservation and exact-operation-order manager tests | `ESTABLISHED_LOCALLY` |
| Fixed timing epsilon and classification | Frozen `0.2 ms` gate constant and timing classification tests | `ESTABLISHED_LOCALLY` |
| Immutable tensor-free drain | Frozen-row/tensor-free drain and event-release test | `ESTABLISHED_LOCALLY` |
| Ordinary prefill and decode context propagation | RED→GREEN ModelRunner receipt test; `spec_verify` remains rejected | `ESTABLISHED_LOCALLY` |
| Baseline batch 1 and batch 4 | Exact four-cell campaign inventory | `ESTABLISHED_LOCALLY` |
| Original and synchronization-control modes | `observe` and `control` only | `ESTABLISHED_LOCALLY` |
| Cross-batch prompt-0 workload identity | Four-cell exact token-sequence validation plus coordinated batch-4 mutation regression | `ENFORCED` |
| Cross-batch prediction-index-0/1 semantic identity | Four-cell `input_token_id`/`position`/`context_length` validation plus coordinated batch-4 mutation regression | `ENFORCED` |
| Prediction-index-0 control row | Required by evaluator | `ESTABLISHED_LOCALLY` |
| Prediction-index-1 drift row | Required by evaluator | `ESTABLISHED_LOCALLY` |
| All-rank lifecycle | Exact ranks 0-3 and mode/schema validation | `ESTABLISHED_LOCALLY` |
| PyTorch/CUDA/driver/device metadata | Required and validated as non-empty/four-device fields; no loaded runtime receipt | `CONTRACT_ONLY` |
| Movement and target-forward invariants | Observe/control invariant projection and mutation tests | `ESTABLISHED_LOCALLY` |
| Supported/rejected/inconclusive exclusivity | Dedicated terminal matrix tests | `ESTABLISHED_LOCALLY` |
| Default-off non-invasiveness | No event creation or stream request in off mode; source checks show no new diagnostic hot-path synchronize | `ESTABLISHED_LOCALLY` |
| Cleanup after failure | Diagnostic and logit recording disabled in `finally` | `ESTABLISHED_LOCALLY` |
| Positive causal classification | Requires observed unsafe overlap plus control removal of overlap and index-1 drift | `DEFINED_NOT_EXECUTED` |
| GPU/NCCL execution | No focused launcher or CLI exists | `NOT_EXECUTED_NOT_APPROVED` |
| Production correction | No slot-reuse correctness fix was inferred from static evidence | `NOT_IMPLEMENTED` |
| Claim boundary | GPU causality, TP4/32K parity, performance and promotion remain negative | `PRESERVED` |
| Code-enforced human authorization | No approval token/file/argument is required by the callable campaign API | `MISSING` |
| Focused source manifest | Seven files required by the written plan are hashed | `PLAN_CONFORMANT` |
| Full producer source closure | Dynamically imported frozen 32K worker and its execution closure are not hashed | `INCOMPLETE` |

The absence of a launcher prevents accidental execution by running the
focused files directly, but it is not equivalent to a code-enforced
authorization gate. Likewise, verifier recomputation of the seven-file source
digest is useful tamper detection but is not complete producer provenance.

The inherited 32K authority gate already enumerates a candidate full source
closure of 126 files: 115 `tinyvllm` Python files and 11 authority
gate/worker/verifier files. It includes all dynamically loaded 32K→16K→4K
worker layers and the relevant runtime files. The focused repetition currently
does not retain that inherited authority digest, so this inventory is
available for a future schema decision but does not repair current provenance.

An import-only CPU audit refined that inventory. The focused producer loads
eleven tool modules, but the inherited eleven-tool set omits the focused
gate/worker and includes two inherited verifier-only files instead. The
conservative union is therefore 128 producer files
(`115 tinyvllm + 13 tools`) or 129 authority files when the focused independent
verifier is included. All files exist, but neither union is retained by the
artifact. The approved seven-file plan contract remains unchanged.

```text
FOCUSED_H2D_DIAGNOSTIC_LOCAL_CONTRACT=ESTABLISHED
FOCUSED_H2D_CROSS_BATCH_PROMPT_IDENTITY=ENFORCED
FOCUSED_H2D_CROSS_BATCH_PREDICTION_IDENTITY=ENFORCED
FOCUSED_H2D_DIAGNOSTIC_DEPENDENCY_LIGHT_TESTS=99_PASSED
FOCUSED_H2D_DIAGNOSTIC_ADJACENT_REGRESSIONS=169_PASSED
FOCUSED_H2D_PREFILL_CONTEXT_BLOCKER=FIXED_LOCALLY
FOCUSED_H2D_32K_AUTHORITY_GATE_TESTS=35_PASSED_CPU_TORCH
FOCUSED_H2D_PLAN_LISTED_CUDA_EXTENSION_TESTS=2_COLLECTION_BLOCKED
FOCUSED_H2D_DIAGNOSTIC_CODE_ENFORCED_AUTHORIZATION=ABSENT
FOCUSED_H2D_DIAGNOSTIC_AUTHORIZATION_BOUNDARY=OPERATIONAL_ONLY
FOCUSED_H2D_SOURCE_MANIFEST_PLAN_CONFORMANCE=ESTABLISHED
FOCUSED_H2D_FROZEN_AUTHORITY_SOURCE_CLOSURE_CANDIDATE=126_FILES
FOCUSED_H2D_CONSERVATIVE_PRODUCER_SOURCE_CLOSURE_CANDIDATE=128_FILES
FOCUSED_H2D_CONSERVATIVE_AUTHORITY_SOURCE_CLOSURE_CANDIDATE=129_FILES
FOCUSED_H2D_FROZEN_32K_WORKER_SOURCE_BINDING=MISSING
FOCUSED_H2D_COMPLETE_PRODUCER_PROVENANCE=NOT_ESTABLISHED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
PAIRED_TRACE_REMOTE_DIAGNOSTIC=NOT_APPROVED
```

## Highest-Value Remaining Boundary

No additional local runtime/scheduler duplication is justified. The next
strong evidence requires a separately authorized loaded
Qwen3-draft/Qwen3.5-target TP1/TP4 GPU exact-parity campaign. The offload
runtime path is now locally wired and the TP1 gate can collect its nested
movement counters, but promotion still requires actually running that gate
with real pinned backing and CUDA copies. The TP4 snapshot transport and
direct validator are now callable production tools, but a loaded TP4 worker
and authority artifact have not been produced. Loaded TP4 parity,
stream/event ordering, measured H2D/D2H bytes, and controlled performance
evidence remain required.

The previously listed deterministic CPU-only learned-drafter verifier gap is
closed. `tools/verify_autoregressive_draft_tp4_engine_gate.py` now validates
the exact TP4/direct-only/batch-1-and-4/temperature-zero contract, recomputes
output parity and accepted-prefix identity from raw rows, validates all four
terminal authority snapshots, freezes a 30-file source inventory, executes
again from a safely extracted deterministic source archive, and rejects
Proposal-KV movement or performance promotion claims. Current-source and
archived-source receipts must match before atomic publication.

The paired TP4/32K target-forward trace is also locally ready, but no remote
or GPU diagnostic was authorized and no first-divergence artifact exists.
Consequently there is no remaining approved CPU-only runtime branch or
learned-drafter artifact-durability task that can substitute for the missing
authority. Other local hardening ideas, such as a source archive for the
variable-Q CUDA Graph gate or a larger focused-H2D producer closure, are
separate designs and do not establish Phase 1 by themselves.

The next promotion-relevant action is therefore the separately authorized
loaded campaign, in this order:

1. Qwen3 draft plus Qwen3.5 target TP1 exact-parity and real Proposal-KV
   movement;
2. the same learned pair under TP4 batch 1/4 with all-rank terminal
   snapshots and a retained schema-v2 source-bound bundle;
3. controlled learned-source TPOT/TTFT/throughput, memory, H2D/D2H bytes, and
   acceptance comparison;
4. separately, native MTP TP4/32K paired-trace collection and root-cause
   resolution.

## Retained Artifact Integrity Re-Audit

The compact promotion checklist's cited artifact paths were revalidated
against the current filesystem rather than trusted from documentation alone:

```text
referenced JSON artifacts:                         11
present and JSON-decodable:                       11
manifest-backed result artifacts:                  9
result/manifest source-tree matches:               9 / 9
result/manifest model or checkpoint hash matches:  9 / 9
authority verifier receipts present:               9 / 9
retained source.tar archives:                      8 / 9
result payloads with source_archive_sha256:         0 / 11
```

The nine manifest-backed authorities retain valid source-tree and
model/checkpoint bindings. Their provenance is not uniform archive binding:
the older generic Qwen3 TP4 authority has no `source.tar`, and none of the
result schemas hashes the archive itself.

Two legacy artifacts are weaker still:

```text
Qwen3 TP1/4K performance:
  embedded per-file source hashes and PASS verifier receipts
  no standalone source manifest/tree/archive or model-manifest hash

Qwen3.5 TP1 CUDA Graph:
  checkpoint-manifest hash and explicit limitations
  no source binding, source archive, or independent verifier receipt
```

Therefore source/config/checkpoint integrity is `PARTIAL` across the full
cited set, not uniformly established. The recorded within-scope measurements
remain useful, but the weaker artifacts cannot serve as templates for the new
learned-drafter TP4 bundle.

## Retained Artifact Payload Coverage Re-Audit

Raw prompt and output rows were recomputed independently of producer-supplied
`parity` and `campaign_direction` fields.

Correctness coverage:

```text
Qwen3 generic TP4 nominal 4K:
  4096-token prompts, batch 1/4 raw parity PASS

Qwen3.5 generic TP1 nominal 4K:
  4048-token prompts, batch 1/4 raw parity PASS

Qwen3.5 generic TP4:
  4096/16384/32768-token prompts, batch 1/4 raw parity PASS

Qwen3.5 native MTP:
  TP1 4096, TP4 4096, TP4 16384-token prompts
  batch 1/4 raw parity PASS for retained passing cells
```

The Qwen3.5 generic TP1 artifact must therefore remain described as
**nominal-4K**, not as an exact 4096-token prompt. This does not remove exact
4096-token TP1 coverage from the broader retained set: the Qwen3 TP1
performance artifact and native-MTP TP1 authority both use 4096-token prompts.

Performance ratios recomputed from aggregate medians:

```text
Qwen3 TP1/4K:
  batch 1: TPOT 0.777298, throughput 1.250800,
           TTFT 0.932292, H2D bytes 1.000000
  batch 4: TPOT 0.519766, throughput 1.668044,
           TTFT 0.662030, H2D bytes 1.000000

Qwen3.5 TP4/16K:
  batch 1: TPOT 0.567749, throughput 1.427909,
           TTFT 1.140597, H2D bytes 0.540897
  batch 4: TPOT 0.544378, throughput 1.782389,
           TTFT 1.060917, H2D bytes 0.539683
```

Both artifacts have raw parity, positive real H2D/D2H counters, proposals,
and accepted draft tokens. The Qwen3 artifact establishes movement presence
but no H2D-byte reduction. Only the Qwen3.5 TP4/16K artifact establishes a
within-campaign H2D-byte reduction, while also recording TTFT regressions.

The retained CUDA Graph payload records `Q=(1,2,3,4)`, batch `(1,4)`, six
captures, twelve replays, graph/eager equality booleans, and 28 transaction
rows. The transaction domain and slot-set equations are independently
recomputable. The graph/eager token parity and per-family capture/replay
counts are not: the legacy JSON retains no raw per-family outputs or counter
rows and has no source-bound independent verifier. Its scope remains TP1, no
offload, no long context, and no performance claim.

## Transactional KV Raw-Receipt Re-Audit

A read-only assertion script traversed the retained passing cells and
recomputed acceptance, rejection, replay, release, and terminal-cleanup
invariants from raw payload fields:

```text
TRANSACTIONAL_KV_RAW_RECEIPT_ASSERTIONS=PASS

Qwen3.5 generic correctness:
  cells=8
  accepted=125
  rejected=55
  TP4 rank transaction rows=140

Qwen3.5 generic TP4/16K performance:
  cells=2
  runs=14
  accepted=2170

native MTP TP1:
  cells=2
  proposal-KV receipts=40
  accepted=150
  rejected=10

native MTP TP4:
  cells=4
  rank snapshots=16
  per-rank transaction rows=212
  canonical accepted=174
  canonical rejected=32
```

The eight Qwen3.5 generic correctness cells all have non-zero accepted and
rejected draft tokens. Their raw mapping rows exactly reproduce both
aggregates, and every cell records `accepted_prefix_replays=0`. TP4 rank rows
add acceptance masks plus `commit_prefix_*_rollback_suffix` decisions, while
the TP1 schema retains aggregate committed-input mappings and terminal
leak/cleanup receipts without the TP4 rank-level decision rows.

The generic TP4/16K performance payload has fourteen raw
warmup/parity/measured runs with non-zero acceptance and terminal cell cleanup,
but it does not retain explicit replay, rejection, or transaction-mask
fields. It is performance evidence, not independent transactional zero-replay
or rollback evidence.

Native MTP TP1 provides the strongest explicit identity receipt: all forty
receipts have `accepted_slot_identity_preserved=true` and
`rejected_slots_released=true`, their counts exactly reproduce runtime totals,
and both cells record zero accepted-prefix target replay and zero terminal
proposal ownership.

Native MTP TP4 provides four-rank transaction and terminal-state evidence
rather than the TP1 identity boolean. Across all sixteen rank snapshots,
transaction sums match rank totals, replay is zero, rejected proposal tokens
are present, and terminal executor/cache state has zero active transactions,
allocated/owned physical slots, prepared tickets, and active sequences.

Two schema boundaries remain explicit:

```text
QWEN3_GENERIC_TP4_TRANSACTIONAL_CLASSIFICATION=
  ACCEPTANCE_ESTABLISHED_BUT_ZERO_REPLAY_FIELD_MISSING

QWEN35_GENERIC_TP4_16K_PERFORMANCE_TRANSACTIONAL_CLASSIFICATION=
  ACCEPTANCE_AND_TERMINAL_CLEANUP_ONLY
```

Neither artifact may be promoted to zero-replay or rejected-suffix authority
from classification text alone. These boundaries narrow the evidence source;
they do not change the mandatory Phase 1 or promotion verdict.

## Archived Transactional-Verifier Coverage

The retained verifier receipts were audited semantically, not only checked
for file presence. The audit loaded each frozen gate from `source.tar`,
validated the unmodified result, and applied controlled in-memory mutations:

```text
ARCHIVED_GATE_ORIGINAL_RESULTS=PASS

generic TP1 accepted aggregate +1:                  ACCEPT
generic TP1 rejected aggregate +1:                  ACCEPT
generic TP4 rejected aggregate +1:                  ACCEPT
generic TP4 opaque kv_decision on all ranks:        ACCEPT
native MTP TP1 accepted aggregate +1:               ACCEPT
native MTP TP1 rejected aggregate +1:               ACCEPT
native MTP TP1 identity boolean false:              REJECT
native MTP TP4 all-rank accepted/rejected +1:       ACCEPT
native MTP TP4 release ticket +1:                   REJECT

ARCHIVED_GATE_MUTATION_EXPECTATIONS=PASS
```

The archived verifiers therefore provide real but incomplete transactional
tamper resistance:

- generic TP1 validates positive counters, zero replay, mapping shapes, lease
  cleanup, and process cleanup, but not aggregate-to-mapping equality;
- generic TP4 validates proposed/accepted mapping totals, acceptance masks,
  cross-rank transaction equality, zero replay, and terminal cleanup, but not
  rejected-total equality or the semantic content of `kv_decision`;
- native MTP TP1 validates identity/release booleans, zero replay, lifecycle
  receipts, and terminal cleanup, but uses `receipt_sum <= runtime_total`
  rather than equality;
- native MTP TP4 validates transaction rows, exact-Q accounting,
  commit/release tickets, all-rank parity, and terminal zero ownership, but
  does not bind rank accepted/rejected aggregates back to transaction sums.

The raw payload values still pass the stronger completion-audit equalities.
The distinction is durability: those stronger checks are not frozen into the
retained artifact verifier.

```text
AUTHORITY_VERIFIER_RECEIPT_PRESENCE=ESTABLISHED
ARCHIVED_TRANSACTIONAL_VERIFIER_SEMANTIC_COVERAGE=PARTIAL
CURRENT_RAW_PAYLOAD_TRANSACTIONAL_REAUDIT=PASS
TAMPER_RESISTANT_TRANSACTIONAL_ARTIFACT_AUTHORITY=NOT_ESTABLISHED_UNIFORMLY
```

This is directly relevant to the future independent learned-drafter verifier:
it must reject these mutation classes instead of inheriting the older gate
pattern.

## Performance Verifier Mutation Coverage

The Qwen3 TP1/4K and Qwen3.5 TP4/16K controlled-performance artifacts were
tested against their retained verifier logic. The Qwen3 artifact has no
archive. Its nine embedded hashes currently produce five matches and four
mismatches because `llm_engine.py`, `model_runner.py`,
`speculative_runtime.py`, and `batch_runtime.py` have since changed. The
performance gate, worker, and verifier files match exactly, so the semantic
probe uses byte-identical verifier logic without treating the current runtime
tree as frozen execution source. Qwen3.5 was loaded from manifest-selected
regular files in `source.tar`.

Both original results validate. The gates reject mutations to:

```text
aggregate TPOT
aggregate TTFT, throughput, and peak memory
aggregate accepted-token totals
movement totals that no longer equal rank sums
raw parity output tokens
Qwen3 campaign direction
Qwen3.5 comparison ratios
per-request latency metrics
stored throughput metrics
```

Qwen3.5 also rejects a standalone `batch_elapsed_s` mutation because it
cross-checks elapsed time and token throughput. The older Qwen3 verifier does
not: changing only `batch_elapsed_s` is accepted while the stored throughput
and request-rate fields remain unchanged.

Simulation-marker rejection is not uniform:

```text
Qwen3 run-level simulated_upload_mb:     ACCEPT
Qwen3 top-level simulate_kv_upload_mb:   ACCEPT
Qwen3.5 run-level simulated_upload_mb:   REJECT
Qwen3.5 top-level simulate_kv_upload_mb: ACCEPT
```

The original artifacts contain none of these fields, and their real movement
totals still match the per-rank `KVOffloadMVP0` deltas. This preserves the
within-scope movement observation while preventing a stronger claim that the
artifact verifier universally excludes simulated-copy evidence.

```text
PERFORMANCE_AGGREGATE_RECOMPUTATION=ESTABLISHED
MOVEMENT_RANK_SUM_VALIDATION=ESTABLISHED
EXACT_PARITY_MUTATION_REJECTION=ESTABLISHED
QWEN3_BATCH_ELAPSED_RATE_CONSISTENCY=NOT_ENFORCED
NO_SIMULATION_VERIFIER_ENFORCEMENT=PARTIAL
PERFORMANCE_ARTIFACT_TAMPER_RESISTANCE=PARTIAL
```

The future promotion verifier must use an exact schema, reject simulation
markers at all nesting levels, and derive throughput/request rate from elapsed
time and token/request counts.

## Long-Context Blockwise Evidence Boundary

Frozen-verifier mutation probes were run against the retained Qwen3.5 generic
TP4/16K, generic TP4/32K, and native-MTP TP4/16K target-KV-offload results:

```text
LONG_CONTEXT_ORIGINAL_RESULTS=PASS

generic TP4/16K and TP4/32K reject:
  context length +1
  profiling disabled
  movement provenance changed
  all batch-4 movement counters zeroed
  residency phases emptied

native MTP TP4/16K rejects:
  blockwise prefill or decode disabled
  max_num_prefill_tokens_per_step changed from 1024
  peak residency greater than the 68-block GPU capacity
  movement provenance changed
  all native batch-4 movement counters zeroed
  residency phases emptied
```

The native raw rows retain bounded capacity:

```text
batch 1: peak/resident blocks=65, GPU capacity=68
batch 4: peak/resident blocks=68, GPU capacity=68
```

All three payloads therefore strongly establish their retained prompt length,
exact greedy parity, real target-KV movement, and lifecycle evidence. The
native schema additionally binds the blockwise flags, prefill chunk, and
bounded capacity. These fields still do not directly observe the selected
attention implementation: none of the results contains
`_blockwise_online_*`, `blockwise_online`, or `attention_path`.

The source manifests further narrow the claim:

```text
generic TP4/16K: 16 files
generic TP4/32K: 16 files
  campaign workers are source-bound
  tinyvllm/layers/attention.py is absent
  tinyvllm/layers/qwen35_full_attention.py is absent

native MTP TP4/16K: 112 files
  tinyvllm/config.py is present
  tinyvllm/engine/scheduler.py is present
  tinyvllm/layers/attention.py is present
  tinyvllm/layers/qwen35_full_attention.py is present
```

The archived native attention implementation dispatches to blockwise online
prefill, decode, and speculative-verify paths under the frozen enabled flags.
That is source-bound dispatch/configuration evidence, not a runtime
path/kernel receipt. Consequently, the generic artifacts remain valid
long-context parity/movement authorities, but not source-bound blockwise
kernel authorities.

The retained TP1 blockwise speculative-verifier authority was also fresh
re-audited:

```text
artifacts/blockwise_speculative_verifier/
  blockwise-tp1-opaque-17786-19070/
```

Its result payload validates all four required `16K/32K x batch 1/4` cells,
and the historical local and remote receipts are byte-equivalent PASS
receipts bound to artifact SHA-256
`2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600`.
Unlike the generic TP4 manifests above, this artifact's seven-file inventory
does include `tinyvllm/layers/attention.py`,
`tinyvllm/engine/model_runner.py`, speculative residency, Context, producer,
worker, and verifier.

Its durability is weaker than the newer learned-drafter bundle. The authority
directory contains no source archive, and fresh verification against the
current checkout fails closed on:

```text
source hash drift: tinyvllm/engine/model_runner.py
```

Fresh current-code dependency-light verification, run in isolated processes
because the tests deliberately install incompatible module stubs, produced
`147 passed`. The production `tools/test_kv_offload.py` collection remains
blocked on the current host by missing `flash_attn`. Compilation, remote
wrapper shell syntax, repo-global diff hygiene, and staged-empty checks pass.
Therefore the historical blockwise authority remains valid within its
recorded source hashes, but Task 8 Step 3 cannot be reclassified as fresh
current-source completion.

The previously missing historical `model_runner.py` was subsequently
recovered from the local AI contribution content-addressed snapshot store.
The blob name and its recomputed content digest both equal the artifact's
expected SHA-256:

```text
037bf4f4a6aff6e19b19493e8fb6b316abdf827a74564b76291089ae83d12f42
```

The other six recorded source digests also exist as exact blobs. Rebuilding
all seven paths under
`/tmp/blockwise-speculative-verifier-frozen-2026-08-15` and running the
recovered historical verifier produced a fresh `PASS / NOT_PROMOTABLE`
receipt for artifact SHA-256
`2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600`.
The fresh receipt equals both retained `verify.json` and
`verify.remote.json`, including all four `16K/32K x batch 1/4` PASS cells.

This establishes local historical frozen-source replay, not fresh
current-source authority. The authority directory itself still has no source
archive and depends on an external local hook blob store for reconstruction.
The current checkout still fails the source-bound verifier on
`model_runner.py`. Production `tools/test_kv_offload.py` also remains
uncollected: current `/usr/bin/python3` stops first on missing `torch`, while
the earlier Torch-enabled entry point stopped on missing `flash_attn`. No GPU,
remote, NCCL, loaded-checkpoint, performance, or direct runtime-path
observation was run.

```text
LONG_CONTEXT_PROMPT_AND_EXACT_PARITY=ESTABLISHED_WITHIN_RETAINED_SCOPES
LONG_CONTEXT_REAL_TARGET_KV_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPES
LONG_CONTEXT_RESIDENCY_LIFECYCLE=ESTABLISHED_WITHIN_RETAINED_SCOPES
NATIVE_TP4_16K_BOUNDED_GPU_RESIDENCY=ESTABLISHED
NATIVE_TP4_16K_BLOCKWISE_SOURCE_AND_CONFIG_BINDING=ESTABLISHED
BLOCKWISE_TP1_16K_32K_HISTORICAL_ARTIFACT=ESTABLISHED_WITHIN_RETAINED_SCOPE
BLOCKWISE_TP1_FROZEN_SOURCE_RECOVERY=ESTABLISHED_LOCAL_EXTERNAL_SNAPSHOT
BLOCKWISE_TP1_FROZEN_SOURCE_REPLAY=PASS
BLOCKWISE_TP1_FROZEN_RECEIPT_EQUAL_HISTORICAL_LOCAL_REMOTE=PASS
BLOCKWISE_TP1_FRESH_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
BLOCKWISE_CURRENT_SYSTEM_PYTHON_KV_OFFLOAD_TEST=NOT_COLLECTED_MISSING_TORCH
BLOCKWISE_TORCH_ENABLED_KV_OFFLOAD_TEST=NOT_COLLECTED_MISSING_FLASH_ATTN
GENERIC_16K_32K_BLOCKWISE_IMPLEMENTATION_SOURCE_BINDING=MISSING
DIRECT_BLOCKWISE_RUNTIME_PATH_OBSERVATION=MISSING
END_TO_END_BLOCKWISE_ONLINE_SOFTMAX_AUTHORITY=PARTIAL
```

## Migration and Dirty-Writeback Artifact Boundary

The retained long-context/performance authorities bind production
`llm_engine.py`, `model_runner.py`, workers, and gates, and contain positive
per-rank H2D/D2H copy and byte deltas. Real target-KV movement and real D2H
writeback therefore remain established within those scopes.

An exact retained-file search covered the current checkout,
`/Users/bytedance/dev`, Trae session JSONL/tool-results, Downloads, Desktop,
selected cache/tmp locations, and Spotlight-indexed paths. The historical
files

```text
profile_out/kv_offload_batched_dirty_evict_migration_20260708_r2.json
profile_out/kv_offload_batched_dirty_evict_thrash_20260708_r2.json
```

were not recovered as raw JSON. Session records contain only handoff summaries,
not payloads, checksums, source manifests, or verifier receipts.

Those summaries record:

```text
migration:
  gate_pass=true
  h2d_copies=2
  d2h_copies=4
  h2d_batches=1
  d2h_batches=2
  d2h_batch_spans=2
  copy_waits=6

thrash:
  gate_pass=true
  h2d_copies=8
  d2h_copies=6
  h2d_batches=4
  d2h_batches=2
  d2h_batch_spans=2
  prefetch_plans=4
```

They remain historical run summaries only.

The retained schemas split into two evidence classes:

- `speculative_residency_boundary` and `speculative_tp1_parity` retain
  `evict_dirty=0` with positive D2H movement.
- `speculative_runtime_performance` retains positive movement totals but omits
  `evict_dirty`.

All of these retained schemas omit:

```text
dirty_blocks
async_copy
batch_copy
h2d_batches / d2h_batches
h2d_batch_spans / d2h_batch_spans
writeback_on_evict
```

Production source increments `evict_dirty` in `_evict_slot()` but not in
explicit `writeback_dirty()`. The residency/parity receipts therefore establish
non-dirty-eviction writeback within those scopes. The performance receipt
cannot classify its D2H, no retained receipt records `evict_dirty > 0`, and no
receipt proves which copy-stream mode or transfer coalescing path executed.

Independent movement arithmetic was recomputed over seven retained aggregate
records: one residency record, two TP1 parity records, and four performance
cells. Every record satisfies:

```text
h2d_bytes == h2d_copies * 29360128
d2h_bytes == d2h_copies * 29360128
MOVEMENT_ARITHMETIC=PASS
positive retained evict_dirty records=0
```

Frozen mutation probe:

```text
generic TP4/16K added evict_dirty field: PASS
generic TP4/32K added evict_dirty field: PASS
native MTP TP4/16K added evict_dirty field: FAIL
```

The generic schemas ignore the unknown field after normalization. The native
schema is canonical/fail-closed for unknown fields, but still has no required
dirty counter.

Fresh local verification:

```text
dependency-light dirty/D2H selection:
  2 passed, 32 deselected in 0.17s

AST-loaded writeback_dirty pair batching:
  DEPENDENCY_LIGHT_WRITEBACK_BATCH_CONTRACT=PASS

production CUDA dirty-batch test:
  NOT COLLECTED
  ModuleNotFoundError: No module named 'flash_attn'
```

The CUDA test failure occurs during production-package import, before its
internal no-CUDA early return. It is an environment limitation, not a passing
test and not evidence of an implementation regression.

```text
REAL_TARGET_KV_H2D_D2H_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPES
DIRTY_WRITEBACK_IMPLEMENTATION=ESTABLISHED
REAL_TARGET_KV_D2H_WRITEBACK=ESTABLISHED_WITHIN_RETAINED_SCOPES
DEPENDENCY_LIGHT_WRITEBACK_BATCH_CONTRACT=ESTABLISHED
EXPLICIT_WRITEBACK_RETAINED_SCOPE=ESTABLISHED_WITHIN_RESIDENCY_AND_TP1_PARITY
DIRTY_EVICTION_HISTORICAL_RUN_SUMMARY=RECORDED
DIRTY_EVICTION_EXACT_ARTIFACT_RECEIPT=MISSING
EXPLICIT_VS_EVICTION_WRITEBACK_ARTIFACT_CLASSIFICATION=MISSING
ASYNC_COPY_RUNTIME_RECEIPT=MISSING
BATCHED_COPY_RUNTIME_RECEIPT=MISSING
CURRENT_HOST_CUDA_DIRTY_BATCH_TEST=NOT_COLLECTED_MISSING_FLASH_ATTN
GENERIC_MOVEMENT_SCHEMA_UNKNOWN_FIELD_REJECTION=NOT_ENFORCED
```

## Prefix-Cache Artifact Coverage Boundary

Fresh focused verification:

```text
tools/test_prefix_kv_offload_integration.py
tools/test_qwen35_hybrid_prefix_cache.py
tools/test_chunked_prefill.py

selection:
  prefix or reusable or hash_collision or ref_count or deduplic or intern

result:
  45 passed, 84 deselected in 9.59s
```

The tests establish three local contracts:

```text
ordinary KV:
  exact hash-plus-token prefix reuse
  live/idle reuse and collision handling
  multi-owner Block.ref_count lifetime
  reservation attach/release/rollback and generation safety

Qwen3.5 hybrid state:
  exact and partial tensor interning
  physical/logical/deduplicated byte equations
  final-reference release and failure-atomic replacement

ordinary KV plus CPU backing:
  same-generation cpu_valid identity schedules H2D residency
  recycled generation invalidates stale CPU backing
  cached-prefix reads require valid backing
```

The third test family is not a real transfer authority:
`KVOffloadMVP0` is created through `__new__`, its state is list-backed, and
the H2D/D2H enqueue methods are replaced by Python pair recorders. No CUDA
copy, pinned-memory transfer, asynchronous stream, model forward, or
cross-request loaded execution occurs.

No artifact JSON in the retained tree contains an explicit
`num_cached_tokens`, `prefix_cache_hits`, `prefix_hit`,
`prefix_block_count`, `deduplicated_bytes`, `current_intern_references`,
`qwen35_hybrid_prefix`, or `reused_prompt_tokens` field. Native source
manifests may bind `block_manager.py` and
`qwen35_hybrid_prefix_cache.py`, but source binding alone does not prove
runtime cache reuse. Existing `accepted_prefix_*` fields are speculative
transaction receipts, not prefix-cache hit receipts.

```text
ORDINARY_PREFIX_HASH_TOKEN_REUSE_LOCAL_CONTRACT=ESTABLISHED
ORDINARY_PREFIX_MULTI_OWNER_REFCOUNT_LOCAL_CONTRACT=ESTABLISHED
QWEN35_HYBRID_TENSOR_DEDUP_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_RESIDENCY_SCHEDULING_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_REAL_CUDA_COPY=NOT_ESTABLISHED
LOADED_CROSS_REQUEST_PREFIX_HIT_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_REFCOUNT_LIFETIME_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_DEDUP_BYTE_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_CPU_RESTORE_AUTHORITY=NOT_ESTABLISHED
GENERIC_HYBRID_STATE_PREFIX_DEDUP=NOT_ESTABLISHED
```

## Legacy CUDA Graph Verifier Coverage

The retained graph artifact has checkpoint identity but no source manifest,
source archive, frozen verifier, or independent verification receipt.

Independent raw recomputation:

```text
q_values=[1,2,3,4]
batch_sizes=[1,4]
transaction cases=28
complete (batch, q, accepted=0..q) domain
staged/committed/released slot counts and set partition: PASS
top-level graph_capture_count=6
top-level graph_replay_count=12
```

The JSON contains no per-family graph/eager token IDs, logits, capture rows,
replay rows, or backend receipts. Thus top-level graph/eager parity booleans
and aggregate counts cannot be independently reconstructed.

Current-validator mutation coverage:

```text
capture count reduced to 1:          ACCEPT
replay count reduced to 1:           ACCEPT
eager max_abs_diff set to 999999:    ACCEPT
transaction slot sets made arbitrary: ACCEPT
graph argmax boolean=false:          REJECT
accepted identity boolean=false:     REJECT
Q=4 removed from domain:             REJECT
```

Fresh current-worktree unit tests:

```text
tools/test_qwen35_mtp_real_checkpoint_gate.py:
  53 passed in 0.26s
```

```text
CUDA_GRAPH_Q1_Q4_BATCH1_BATCH4_RECORDED=ESTABLISHED
CUDA_GRAPH_TRANSACTION_RAW_SET_EQUATIONS=ESTABLISHED
CUDA_GRAPH_TOP_LEVEL_CAPTURE_REPLAY_COUNTS=PRODUCER_ASSERTED
CUDA_GRAPH_EAGER_TOKEN_PARITY=NOT_INDEPENDENTLY_RECOMPUTABLE
CUDA_GRAPH_PER_FAMILY_CAPTURE_REPLAY=NOT_INDEPENDENTLY_RECOMPUTABLE
CUDA_GRAPH_SOURCE_BOUND_PROVENANCE=MISSING
CUDA_GRAPH_FROZEN_INDEPENDENT_VERIFIER=MISSING
CUDA_GRAPH_CURRENT_VALIDATOR_SEMANTIC_COVERAGE=PARTIAL
VARIABLE_Q_CUDA_GRAPH_AUTHORITY=PARTIAL
```

## Independent TP4 Verifier Readiness Checklist

The local evidence layer now defines and tests an independent source-bound
bundle contract. This is a producer/verifier contract result only: no retained
bundle from a real TP4 loaded-checkpoint execution exists.

| Requirement | Established local contract | Retained execution evidence |
| --- | --- | --- |
| Fixed TP4/direct/BF16/temperature-zero/max-proposal-four configuration | Schema-v2 producer and independent verifier require the exact configuration | Absent |
| Batch 1 and batch 4 workload presence | Verifier requires the exact case inventory | Absent |
| Exact greedy output parity | Verifier recomputes parity from raw target and learned token-ID rows | Absent |
| Non-empty proposal/acceptance activity | Verifier requires proposal rows and accepted-prefix activity | Absent |
| Accepted-prefix identity | Every row binds event, step, sequence, prompt, output boundary, proposal IDs, count, and exact accepted proposal prefix | Absent |
| Four-rank topology and terminal authority | Archived production validator recomputes the rank summary from all four raw snapshots | Absent |
| Direct allocator and zero accepted-copy/replay/rematerialization counters | Checked from every raw rank snapshot | Absent |
| Terminal transaction, owned-entry, and physical-slot cleanup | Checked from every raw rank snapshot | Absent |
| Checkpoint and tokenizer identity | Stored in the canonical result and bound by the result digest | Absent |
| Canonical result schema | Independent verifier reloads, validates, canonicalizes, and requires byte-equivalent canonical JSON | Absent |
| Result artifact SHA-256 | `source_manifest.json` binds `result.json` | Absent |
| Frozen source-file inventory and per-file SHA-256 | Exact 30-file inventory is required and hashed | Absent |
| Frozen source-tree SHA-256 | Canonical path/payload framing is hashed and verified | Absent |
| Archived verifier/source closure | Deterministic `source.tar` contains the verifier and its exact declared dependencies | Absent |
| Independent verifier receipt | Current-source and safely extracted archived-source verifier processes must emit identical canonical receipts | Absent |
| Atomic multi-file publication | Temporary bundle verification precedes exclusive directory publication; failures retain bounded `.failed/failure.json` evidence | Absent |
| Real Proposal-KV movement claim | Producer and verifier require the negative claim | Not established |
| Performance promotion claim | Producer and verifier require the negative claim | Not established |
| Loaded TP4 authority | Contract exists, but only an authorized real-checkpoint TP4 run can establish authority | Not established |

The frozen source inventory is:

```text
tinyvllm/__init__.py
tinyvllm/llm.py
tinyvllm/config.py
tinyvllm/sampling_params.py
tinyvllm/engine/llm_engine.py
tinyvllm/engine/model_runner.py
tinyvllm/engine/model_runner_command_ack.py
tinyvllm/engine/autoregressive_draft_registration.py
tinyvllm/engine/autoregressive_draft_tp.py
tinyvllm/engine/autoregressive_draft_executor.py
tinyvllm/engine/qwen3_draft_backend.py
tinyvllm/engine/qwen3_draft_proposal_kv.py
tinyvllm/engine/proposal_kv_allocator.py
tinyvllm/engine/proposal_kv_cache.py
tinyvllm/engine/proposal_kv_lifecycle.py
tinyvllm/engine/proposal_kv_residency.py
tinyvllm/engine/speculative_proposal_executor.py
tinyvllm/engine/speculative_runtime.py
tinyvllm/engine/speculative_selection.py
tinyvllm/engine/tensor_parallel_greedy.py
tinyvllm/models/qwen3.py
tinyvllm/speculative/adapter.py
tinyvllm/speculative/batch_runtime.py
tinyvllm/speculative/verifier.py
tinyvllm/utils/context.py
tinyvllm/utils/loader.py
tools/autoregressive_draft_tp1_engine_gate.py
tools/autoregressive_draft_tp4_engine_gate.py
tools/autoregressive_draft_tp4_local_gate.py
tools/verify_autoregressive_draft_tp4_engine_gate.py
```

The inventory deliberately includes the public engine/config entry, registration
fingerprinting and TP consensus, Qwen3 model loading, proposal executor and
KV lifecycle, token selection/verifier path, snapshot acknowledgement
transport, producer gate, local authority validator, and independent
verifier. It deliberately excludes unrelated Qwen3.5 projection experiments,
hybrid-prefix research modules, native-MTP-only modules, and performance
workers. Tests fail when a named path is absent and require the archived
verifier to load only from the safely extracted archive root.

### Source-Archive Safety Boundary

The learned-drafter TP4 bundle implements the following deterministic archive
contract:

1. include only explicitly named regular files;
2. use sorted POSIX relative paths with no empty, absolute, `.` or `..`
   component;
3. reject symlinks, hard links, devices, FIFOs, duplicate names, and
   unexpected members;
4. normalize uid/gid to zero, owner/group names to empty strings, mode to
   `0644`, and mtime to zero;
5. hash `source.tar` itself in `source_manifest.json`;
6. validate every archive member and its SHA-256 before extraction;
7. extract into a fresh temporary directory without unchecked
   `extractall()`;
8. load the verifier module from the extracted archive, not from the current
   checkout;
9. execute verification with the extracted archive root as `source_root`;
10. require the archived verifier result to match the pre-publication
    verifier receipt before the temporary run directory is published.

Fresh local evidence:

```text
schema-v2 gate focused tests:                27 passed
gate plus bundle/verifier focused tests:     41 passed in 0.74s
broader learned-drafter CPU regressions:    217 passed in 6.86s
frozen source inventory:                     PASS, 30 files
changed-file py_compile:                     PASS
unchecked tar extractall() scan:             PASS
```

These tests establish reconstructability, deterministic archive construction,
safe extraction, current/archived verifier equality, tamper rejection, and
atomic publication behavior. They did not execute a real model, GPU, NCCL,
remote host, loaded checkpoint, Proposal-KV transfer, or performance workload.

Readiness classification:

```text
TP4_RESULT_CONTENT_FOR_INDEPENDENT_VERIFICATION=ESTABLISHED_LOCAL_CONTRACT
TP4_ACCEPTED_PREFIX_IDENTITY=RECONSTRUCTABLE_LOCAL_CONTRACT
TP4_SOURCE_BOUND_BUNDLE_CONTRACT=ESTABLISHED_LOCAL
TP4_ARCHIVED_VERIFIER=ESTABLISHED_LOCAL
TP4_ATOMIC_ARTIFACT_BUNDLE=ESTABLISHED_LOCAL
TP4_RETAINED_SOURCE_BOUND_EXECUTION_ARTIFACT=ABSENT
LEARNED_DRAFTER_TP4_LOADED_EXECUTION=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Autoregressive Draft TP4 Task 1-7 Reconciliation

The pre-existing TP4 extension plan was checked against current source
rather than trusted from its stale all-unchecked state:

```text
docs/superpowers/plans/
  2026-08-14-autoregressive-draft-tp4-extension.md
```

Tasks 1-7 now have direct source and test evidence for the tensor-parallel
coordinator, canonical logical digests, stage convergence, registry preflight,
topology-aware root/non-root logits, rank-local proposal-KV evidence,
synchronized proposal and lifecycle transitions, private registration
candidates, identity consensus, failure-atomic ModelRunner publication, and
root-only fused proposal return.

Fresh focused evidence:

```text
Task 1-7 focused matrix:
  271 passed in 5.95s

source-neutral neighboring regressions:
  224 passed in isolated Python processes

Task 1-7 source/test py_compile:
  PASS

obsolete TP1-only message scan:
  PASS

git diff --check:
  PASS

staged diff:
  empty
```

The first combined neighboring regression collection was invalidated by
dependency-light `sys.modules` stub pollution: a stubbed
`tinyvllm.utils.context` lacked `temporary_context`. Isolated runs all passed.
Production `tools/test_kv_offload.py` independently remains uncollected in
the offline Python 3.12 + Torch 2.12 environment because `flash_attn` is
unavailable.

Only current implementation and freshly observed GREEN steps were checked.
Historical RED/test-first steps remain unchecked because their original
failure state was not observed during this continuation. Task 8 was not
executed or synchronized:

```text
whole-plan checked:     24
whole-plan unchecked:   29
Task 8 unchecked:        8
```

This closes a documentation/evidence mismatch; it does not create new loaded
authority:

```text
AUTOREGRESSIVE_DRAFT_TP4_LOCAL_IMPLEMENTATION=ESTABLISHED
AUTOREGRESSIVE_DRAFT_TP4_TASK1_7_GREEN=271_PASSED
SOURCE_NEUTRAL_NEIGHBOR_REGRESSION=224_PASSED_ISOLATED
AUTOREGRESSIVE_DRAFT_TP4_LOCAL_GATE_TASK8=NOT_EXECUTED_IN_THIS_RECONCILIATION
TP4_INDEPENDENT_DRAFT_REAL_CHECKPOINT=NOT_ESTABLISHED
TP4_4K_ENGINE_PARITY=NOT_ESTABLISHED
TP4_16K_ENGINE_PARITY=NOT_ESTABLISHED
TP4_32K_ENGINE_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Source-Neutral N-Gram/SAM Drafter Family Audit

The source-neutral drafter family has fresh dependency-light local evidence
covering its adapter API, runtime integration, verified-history lifecycle,
transaction boundaries, algorithm behavior, and canonical SAM gate logic:

```text
adapter/public API/runtime/transaction matrix: 157 passed
tools/test_ngram_speculative.py:                 59 passed
SAM algorithm plus gate tests:                  28 passed
```

The inspected implementation establishes the following local contracts:

- `NGramDraftAdapter` is host-domain, batch-capable, and stateless.
- `SAMDraftAdapter` owns a per-sequence suffix-automaton index.
- SAM proposal lookup is read-only; only explicitly verified history updates
  the index.
- Sequence release removes the corresponding SAM state.
- Both adapters are exported through `tinyvllm.speculative`.
- Runtime and transaction tests preserve exact accepted-prefix and rollback
  boundaries without granting either drafter authority over target-KV state.

### Retained SAM Artifact Regeneration

Only two retained SAM experiment directories were found. Both are 25-row,
single-repetition, single-sequence, profiler-owned smoke artifacts rather than
the canonical 175-row gate authority.

The current verifier requires all canonical artifact files, recomputes the
SHA-256 values for `manifest.json`, `raw_rows.json`, and `event_rows.json`,
regenerates `summary.json`, regenerates `report.md`, and rejects any mismatch.
Fresh verification produced exit zero for both retained directories:

```text
experiments/sam_drafter/
  qwen3-06b-sam-remat-smoke-20260715-162323

expected/observed rows:              25/25
input artifact hashes:               PASS
summary regeneration:                PASS
report regeneration:                 PASS
structural failures:                 none
trace reconciliation:                PASS
policy exercise:                     PASS
correctness:                         PASS
decision:                            NO_GO
median SAM vs baseline:             -0.10721453066361797
median SAM vs ngram-k4:              0.08438688372122316
median verify-attempt reduction:     0.25
median draft-waste reduction:       -2.1818181818181817
```

The `NO_GO` decision is authoritative within this smoke's narrow scope. SAM
regressed against baseline on the median and on the `natural_prose`,
`structured_code_like`, `repeated_long_context`, and
`prompt_copy_retrieval` critical prompts.

```text
experiments/sam_drafter/
  qwen3-06b-sam-smoke3-reconciled-20260715

expected/observed rows:              25/25
input artifact hashes:               PASS
summary regeneration:                PASS
report regeneration:                 PASS
structural failures:                 none
trace reconciliation:                PASS
policy exercise:                     PASS
correctness:                         FAIL
correctness failures:                6 output mismatches
decision:                            INCOMPLETE
```

The second artifact's positive throughput figures cannot be interpreted as
correctness-preserving performance evidence because three policies mismatch
on `natural_prose` and three policies mismatch on `structured_code_like`.

Both artifacts declare the same narrow claim boundary:

```text
greedy_only=true
single_sequence=true
profiler_owned=true
production_batch_throughput=false
memory_reduction=false
ragged_batched_verify=false
queue_tail_latency=false
```

An additional retained directory has a misleading `sam-canonical` suffix:

```text
experiments/adaptive_ngram/
  20260717-k1-sam-canonical
```

It is not a SAM-drafter artifact. Its manifest declares 140 expected rows,
seven repetitions, four prompts, and these five policies:

```text
baseline
fixed_k1
fixed_k2
fixed_k4
adaptive
```

Fresh inspection found 140 raw rows, only `fixed` and `adaptive`
`draft_policy` values, and zero rows whose `policy` or `draft_policy` is
`sam`. Its report is titled `Adaptive N-Gram Speculation Gate` and records a
reproducible `NO_GO`. It remains useful adaptive n-gram evidence, but its
directory name, row count, schema, policies, and producer contract cannot
satisfy or substitute for the missing canonical 175-row SAM authority.

No GPU, remote, NCCL, loaded-checkpoint, canonical 175-row, multi-sequence,
production throughput, real KV-movement, or queue-tail workload was executed
for this audit. Therefore local adapter correctness and reproducible retained
smoke decisions do not establish loaded performance authority:

```text
NGRAM_SOURCE_NEUTRAL_ADAPTER=ESTABLISHED_LOCAL
SAM_SOURCE_NEUTRAL_ADAPTER=ESTABLISHED_LOCAL
SAM_VERIFIED_HISTORY_LIFECYCLE=ESTABLISHED_LOCAL
ADAPTIVE_NGRAM_140_ROW_CANONICAL_ARTIFACT=RETAINED_NO_GO
ADAPTIVE_NGRAM_ARTIFACT_SAM_ROWS=0
ADAPTIVE_NGRAM_ARTIFACT_AS_SAM_AUTHORITY=REJECTED
SAM_RETAINED_ARTIFACT_REGENERATION=PASS_FOR_TWO_25_ROW_SMOKES
SAM_RETAINED_CANONICAL_175_ROW_AUTHORITY=ABSENT
SAM_RETAINED_SMOKE_25_ROW_DECISIONS=NO_GO_AND_INCOMPLETE
SAM_LOADED_PERFORMANCE_AUTHORITY=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Qwen3.5 Native-MTP Proposal Executor Plan Reconciliation

The original native-MTP executor plan is no longer an unimplemented
`0 checked` design. Current source, contract tests, retained loaded-checkpoint
evidence, and the canonical handoff establish its narrow TP1/no-offload
implementation path:

- source-neutral proposal lifecycle and two-phase finalization;
- explicit final target-prefill hidden observation;
- exact immutable 15-tensor BF16 MTP checkpoint planning and binding;
- native Qwen3.5 MTP module with an independent math oracle;
- metadata-first reversible proposal-KV transactions;
- exact-Q proposal semantics for Q 1 through 4;
- distinct unpadded exact-Q CUDA Graph identities with private scratch;
- model-specific registration after target loading and MTP binding;
- pre-publication rollback and post-publication poison-without-retry;
- a fail-closed real-checkpoint gate and serial remote wrapper.

Fresh current local execution produced:

```text
full 13-file plan matrix:
  collection blocked by ModuleNotFoundError: torch

direct torch-dependent files:
  tools/test_qwen35_mtp.py
  tools/test_qwen35_mtp_executor.py

remaining 11 dependency-light files:
  386 passed in 1.35s
```

The two direct-Torch files are not reported as fresh GREEN. No dependency was
installed and no fake Torch/CUDA/FlashAttention module was injected. Historical
local regression records support the implementation mapping, while loaded
execution authority remains the retained GPU artifact:

```text
artifacts/qwen35-mtp-runs/
  qwen35-mtp-graph-gate-opaque-7/
    qwen35_mtp_real_checkpoint_gate.json

status:                         PASS
promotion classification:      NOT_PROMOTABLE
device:                         NVIDIA A100 80GB PCIe
Q values:                       1,2,3,4
batch sizes:                    1,4
loader:                         PASS
eager/reference greedy argmax:  equal
graph/eager greedy argmax:      equal
graph captures / replays:       6 / 12
accepted slot identity:         preserved
rejected suffix:                released
rollback-safe continuation:     equal
post-replay eager retry:        zero
```

The remote-gate step is checked from the retained artifact plus its existing
execution record; no remote workload was rerun during this reconciliation.
Historical test-first RED steps remain unchecked because their original
failure state was not observed in this continuation:

```text
whole-plan checked:             77
historical RED unchecked:       12
```

The original micro-gate does not by itself establish later TP4, offload,
long-context, second-model, real proposal-KV movement, or performance claims.
Later production gates are classified separately:

```text
NATIVE_MTP_MICRO_GATE=PASS_WITHIN_RETAINED_SCOPE
NATIVE_MTP_PRODUCTION_ENGINE_TP1_4K=ESTABLISHED_BY_LATER_GATE
NATIVE_MTP_PRODUCTION_ENGINE_TP4_4K=ESTABLISHED_BY_LATER_GATE
NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD=ESTABLISHED_BY_LATER_GATE
NATIVE_MTP_TP4_32K_PARITY=FAILED_OR_NOT_ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
CONTROLLED_NATIVE_MTP_PERFORMANCE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Native-MTP TP1/4K Production Engine Plan Reconciliation

The production Engine plan
`2026-08-13-qwen35-native-mtp-tp1-engine-transactional-correctness.md`
was stale at `0 checked`. Its implementation, historical execution record,
source-bound loaded artifact, and independent verifier establish the approved
TP1/4K domain.

Authoritative artifact:

```text
artifacts/qwen35_native_mtp_tp1_4k_engine/
  opaque-57a3a62810d43636b96295da/
    local-authority/result.json
    local-authority/source_manifest.json
    local-authority/verify.json
    verify.remote.json
    verify.local.json
    source.tar
```

The result establishes:

```text
classification:
  QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED

promotion:
  NOT_PROMOTABLE

scope:
  Qwen3.5 native MTP
  production LLMEngine.step()
  TP1
  4K prompt
  batch 1 and 4
  greedy
  eager native MTP
  target KV offload disabled

exact parity:
  batch 1 PASS
  batch 4 PASS

native proposal totals:
  batch 1 proposed / accepted / rejected = 32 / 30 / 2
  batch 4 proposed / accepted / rejected = 128 / 120 / 8

target-forward ownership:
  first-target forward count equals callback count
  verify forward count equals callback count
  accepted-prefix target replay count is zero

cleanup:
  zero pending prefixes
  zero bootstrapped sequences
  zero proposal transactions
  zero finalize tickets
  zero allocated proposal slots
  no runtime poison
  Engine exit called
```

The current checkout no longer has the frozen 106-file inventory, so its
verifier correctly returns:

```text
{"classification":"FAIL",
 "failures":["source file inventory mismatch"]}
```

The artifact is self-contained. Extracting its `source.tar`, checking all
106 manifest-bound file hashes, and running the archived independent verifier
produced:

```text
{"classification":"PASS","failures":[]}
```

This fresh result matches all three retained verifier receipts. Current local
plan-specific regression also passed:

```text
ModelRunner executor registry and release
ModelRunner callback bridge
Engine runtime ordering and poison behavior
native-MTP executor
TP1/4K gate producer/verifier/runner contracts

244 passed in 5.33s
```

A broader historical seven-file combination currently reports
`219 passed / 8 failed`. The failures are confined to
`test_qwen35_mtp_real_transaction_probe.py` and
`test_qwen35_mtp_real_eager_reference_probe.py`. Those old micro-gate
fixtures still request FP32 proposal storage and directly use the physical
store as `ProposalKVCache`'s allocator. Current production code requires
FP16/BF16 storage and a logical-entry allocator with generation-bearing
identities and residency leases. This is a current micro-probe compatibility
gap, not evidence against the frozen TP1 Engine authority. It must be migrated
separately without restoring the removed physical-store allocator interface.

Plan synchronization:

```text
current implementation/execution/GREEN checked: 26
historical RED unchecked:                        5
```

Strict classification:

```text
NATIVE_MTP_PRODUCTION_ENGINE_TP1_4K=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP1_4K_FROZEN_SOURCE_ARCHIVE=SELF_CONTAINED
NATIVE_MTP_TP1_4K_FROZEN_SOURCE_REPLAY=PASS
NATIVE_MTP_TP1_4K_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
NATIVE_MTP_CURRENT_MICRO_PROBE_COMPATIBILITY=NEEDS_ALLOCATOR_IDENTITY_MIGRATION
NATIVE_MTP_TP1_4K_PERFORMANCE=NOT_ESTABLISHED
SECOND_LEARNED_MODEL_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Native-MTP TP4/4K Production Engine Plan Reconciliation

The TP4 production Engine plan
`2026-08-14-qwen35-native-mtp-tp4-4k-engine-transactional-correctness.md`
was stale at `0 checked`. The final retained loaded authority, its source
archive, independent verifier, current source contracts, and fresh local
tests establish the approved TP4/4K domain.

Only the final passing run is positive authority:

```text
artifacts/qwen35_native_mtp_tp4_4k_engine/
  opaque-95aa0889f8365beac8be2b6f/
    artifacts/authority/result.json
    artifacts/authority/source_manifest.json
    artifacts/authority/verify.json
    verify.remote.json
    verify.local.json
    source.tar
    source/
```

Earlier opaque directories are failed or diagnostic campaigns and are not
used for positive conclusions.

The result establishes:

```text
classification:
  QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED

promotion:
  NOT_PROMOTABLE

scope:
  Qwen3.5 native MTP
  production LLMEngine.step()
  TP4 ranks 0,1,2,3
  4K prompt
  batch 1 and 4
  exact greedy
  eager target and MTP
  target KV offload disabled
  MTP CUDA Graph disabled

parity:
  baseline/native batch 1 and 4 true
  TP1/TP4 native batch 1 and 4 true

native batch 1 per rank:
  proposal rows 8
  accepted / rejected draft tokens 30 / 2
  selected proposal tokens 24
  proposal-KV history rows 9

native batch 4 per rank:
  proposal rows 32
  accepted / rejected draft tokens 120 / 8
  selected proposal tokens 96
  proposal-KV history rows 36

cleanup:
  zero live proposal transactions, tickets, sequences and slots
  rank exit codes [0,0,0,0]
  Engine exit, process-group destroy and shared-memory release true
  no owned children
  runtime not poisoned
```

The history rows include real 4K bootstrap transactions as well as proposal
transactions. Rank 0 owns greedy host output; all four ranks execute matching
MTP and transaction lifecycles, and only contiguous int64 token tensors are
broadcast.

Current-source verification fails closed because the source inventory has
evolved:

```text
{"classification":"FAIL",
 "failures":["source file inventory mismatch"]}
```

The artifact's frozen source inventory contains 109 files. Fresh per-file
hash validation reported 109/109 matches, and the archived verifier produced:

```text
{"classification":"PASS","failures":[]}
```

This matches all retained verifier receipts. The current equivalent local
matrix also passed:

```text
395 passed in 15.98s
```

The plan's original matrix names
`tools/test_qwen35_mtp_tp4_rank_evidence.py`, which has since been removed.
Its authority-snapshot and rank-evidence contracts are now covered in
`test_qwen35_mtp_executor.py`, ModelRunner integration tests, and the TP4 gate
tests. The unmodified original command therefore fails before collection due
to an absent path, while the current equivalent matrix is GREEN.

Plan synchronization:

```text
current implementation/execution/GREEN checked: 41
historical RED unchecked:                        8
```

Strict classification:

```text
NATIVE_MTP_PRODUCTION_ENGINE_TP4_4K=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP4_4K_ALL_RANK_EXECUTION=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP4_4K_TOKEN_ONLY_BROADCAST=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP4_4K_FROZEN_SOURCE_ARCHIVE=SELF_CONTAINED
NATIVE_MTP_TP4_4K_FROZEN_SOURCE_REPLAY=PASS
NATIVE_MTP_TP4_4K_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
NATIVE_MTP_TP4_4K_PERFORMANCE=NOT_ESTABLISHED
NATIVE_MTP_TP4_4K_KV_OFFLOAD=DISABLED_BY_SCOPE
SECOND_LEARNED_MODEL_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Deferred After the Local Gap

The following remain separate authority or design campaigns:

1. real independent-drafter TP1 and TP4 loaded-checkpoint parity;
2. real proposal-KV H2D/D2H for Qwen3.5 MTP and the Qwen3 drafter;
3. native MTP TP4/32K correctness resolution;
4. controlled learned-source TPOT/TTFT/throughput and memory measurements;
5. KV4/KV8 loaded parity/performance and KV4/KV8 plus offload;
6. heat-tier policy;
7. verifier/sampling/commit fusion and TP collective overlap.

## Native-MTP TP4/16K Target-KV-Offload Plan Reconciliation

The production target-KV-offload plan
`2026-08-14-qwen35-native-mtp-tp4-16k-target-kv-offload.md`
was stale at `0 checked`. Its implementation, retained campaign history,
source-bound loaded artifact, frozen source tree, independent verifier, and
fresh local pure-Python gates establish the approved TP4/16K domain.

Canonical positive authority:

```text
artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/
  lifecycle-release-fix-20260814-2/
    artifacts/authority/result.json
    artifacts/authority/source_manifest.json
    artifacts/authority/verify.json
    verify.remote.json
    verify.local.json
    source.tar
    source/
```

`lifecycle-release-fix-20260814-1` is excluded from positive evidence. An
unrelated process occupied selected GPU 3 after idle preflight, leaving
1.05 GiB free; native batch 4 failed a 2.00 GiB allocation with CUDA OOM.
No unrelated process was terminated.

The retained result establishes:

```text
classification:
  QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED

promotion:
  NOT_PROMOTABLE

scope:
  Qwen3.5 native learned MTP
  production LLMEngine.step()
  TP4 ranks 0,1,2,3
  16K prompt
  batch 1 and 4
  exact greedy
  maximum proposal tokens 4
  eager target and MTP
  target-KV capacity 68 GPU / 640 logical blocks
  block size 256
  blockwise window 8

exact baseline/native parity:
  batch 1 true
  batch 4 true

native batch 1 per rank:
  proposed / accepted / rejected = 10 / 5 / 5
  accepted-prefix target replay = 0
  D2H copies / bytes = 69 / 434110464
  H2D copies / bytes = 0 / 0
  resident / peak blocks = 65 / 65

native batch 4 per rank:
  proposed / accepted / rejected = 36 / 19 / 17
  accepted-prefix target replay = 0
  D2H copies / bytes = 273 / 1717567488
  H2D copies / bytes = 6762 / 42542825472
  resident / peak blocks = 68 / 68
```

Every movement row uses production
`engine.kv_offload_summaries` provenance. Target-KV receipts are
`prepare -> commit`; side-state receipts are
`prepare -> select -> apply -> seal`; residency receipts are
`prepare -> precommit -> seal`. The campaign records zero live proposal
transactions, tickets, sequences and slots, no runtime poison, rank exit
codes `[0,0,0,0]`, Engine exit, process-group destruction, shared-memory
release, no owned children, and unchanged selected-GPU process inventory.

The artifact is self-contained. Fresh validation found all 112 frozen source
files present with matching SHA-256 digests, and the archived verifier
returned:

```text
{"classification":"PASS","failures":[]}
```

This matches the retained remote, runner-local, and explicit local verifier
receipts. The current checkout has evolved in nine bound files, so the current
verifier correctly fails closed before accepting the old authority:

```text
{"classification":"FAIL",
 "failures":["source file inventory mismatch"]}
```

Fresh current local evidence:

```text
TP4/16K plus TP4/4K pure-Python gate matrix:
  100 passed in 4.21s

authority files and ModelRunner py_compile:
  PASS

remote-runner bash syntax:
  PASS
```

The current host cannot freshly collect the Torch-backed
`test_kv_offload.py` or `test_qwen35_mtp_executor.py` because system Python
does not provide `torch`. That is an environment blocker, not a test failure;
the retained remote Torch runs and source-bound loaded campaign remain the
execution evidence.

The final retained fix addressed ordinary-only completion omitting
ModelRunner proposal-executor release. It restored lifecycle symmetry and
complete release-row inventory without changing exact greedy selection.

Plan synchronization:

```text
current implementation/execution/GREEN checked: 35
historical RED unchecked:                         6
```

Strict classification:

```text
NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD=ESTABLISHED_BY_LATER_GATE
NATIVE_MTP_TP4_16K_EXACT_GREEDY_PARITY=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP4_16K_REAL_TARGET_KV_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP4_16K_BOUNDED_RESIDENCY=ESTABLISHED_WITHIN_RETAINED_SCOPE
NATIVE_MTP_TP4_16K_FROZEN_SOURCE_ARCHIVE=SELF_CONTAINED
NATIVE_MTP_TP4_16K_FROZEN_SOURCE_REPLAY=PASS
NATIVE_MTP_TP4_16K_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
NATIVE_MTP_TP4_32K_PARITY=FAILED_OR_NOT_ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
CONTROLLED_NATIVE_MTP_PERFORMANCE=NOT_ESTABLISHED
SECOND_LEARNED_MODEL_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Native-MTP TP4/32K Target-KV-Offload Plan Reconciliation

The production 32K plan
`2026-08-14-qwen35-native-mtp-tp4-32k-target-kv-offload.md`
was stale at `0 checked`. Its local overlay, worker, verifier, bounded runner,
tests, and two retained campaigns exist, but no correctness authority was
established.

Retained run classification:

```text
native-mtp-tp4-32k-20260814-1:
  FAILED
  result missing
  overlay CLI did not dispatch main()
  no GPU correctness conclusion

native-mtp-tp4-32k-20260814-2:
  FAILED
  four production Engine cells completed
  authority.failed retained
  baseline/native batch-1 exact parity mismatch
```

Run 1's CLI defect is covered by a subprocess regression and the overlay now
dispatches `sys.exit(main())`. Run 2 failed at canonical result assembly:

```text
baseline batch 1:
  [220,15,15,15,15,15,15,15]

native MTP batch 1:
  [220,15,15,220,15,15,220,15]

differing indices:
  3,6

batch 4:
  exact baseline/native parity for all four prompts
```

The identical prompt-0 baseline row differs between batch 1 and batch 4, while
native batch 1 matches baseline batch 4. The retained evidence therefore does
not isolate the defect to MTP verify-tail behavior; ordinary target execution
also has a batch/query-shape-sensitive inconsistency at 32K.

The failed run nevertheless confirms real production target-KV movement and
clean lifecycle evidence:

```text
native batch 1 per rank:
  proposed / accepted / rejected = 10 / 5 / 5
  H2D copies / bytes = 2257 / 14199816192
  D2H copies / bytes = 133 / 836763648

native batch 4 per rank:
  proposed / accepted / rejected = 36 / 19 / 17
  H2D copies / bytes = 15159 / 95372181504
  D2H copies / bytes = 530 / 3334471680

every cell:
  production engine.kv_offload_summaries provenance
  resident / peak / GPU / logical blocks = 68 / 68 / 68 / 640
  rank exit codes [0,0,0,0]
  no runtime poison
  Engine/process-group/shared-memory/child cleanup complete

every native rank:
  zero accepted-prefix target replay
  zero live proposal transactions, tickets, sequences and slots
  complete target-KV, side-state and release receipts
```

This is movement evidence from a failed correctness campaign, not a promoted
32K authority.

The retained failed bundle is source-bound:

```text
source manifest:
  115 files

frozen hash check:
  115/115 match

source tree:
  d722bc58f309c21695ea406035d69638d87337e7bac6ce0779e8848eb92fa6b8

archived verifier:
  {"classification":"FAIL","failures":["result is missing"]}
```

The verifier correctly rejects the failed bundle because no canonical
`result.json` exists. Current `DEFAULT_SOURCE_FILES` contains 126 files and
ten of the old bound files have changed, so neither the failed bundle nor its
movement counters prove current-source correctness.

Fresh local validation:

```text
32K gate/worker/verifier/runner:
  35 passed in 3.42s

16K gate, generic TP4/32K gate, Engine runtime,
and ModelRunner spec-verify regressions:
  244 passed in 5.09s

py_compile:
  PASS

bounded runner bash syntax:
  PASS
```

Plan synchronization:

```text
implementation/local GREEN/failed-run audit checked: 22
historical RED or unachieved PASS steps unchecked:   8
```

The unachieved production steps remain explicit verifier PASS, PASS-authority
documentation, and final PASS verification. No new GPU campaign or diagnostic
was authorized during reconciliation.

Strict classification:

```text
NATIVE_MTP_TP4_32K_LOCAL_GATE=ESTABLISHED
NATIVE_MTP_TP4_32K_FOUR_CELL_EXECUTION=OBSERVED_IN_FAILED_RUN
NATIVE_MTP_TP4_32K_REAL_TARGET_KV_MOVEMENT=OBSERVED_IN_FAILED_RUN
NATIVE_MTP_TP4_32K_BATCH4_PARITY=OBSERVED_IN_FAILED_RUN
NATIVE_MTP_TP4_32K_BATCH1_PARITY=FAILED
NATIVE_MTP_TP4_32K_PARITY=FAILED_OR_NOT_ESTABLISHED
NATIVE_MTP_TP4_32K_CORRECTNESS_AUTHORITY=NOT_ESTABLISHED
NATIVE_MTP_TP4_32K_ROOT_CAUSE=NOT_ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
CONTROLLED_NATIVE_MTP_PERFORMANCE=NOT_ESTABLISHED
SECOND_LEARNED_MODEL_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Qwen3 Independent-Draft Authority and Performance Coverage

The independent learned-drafter branch now has three distinct source-bound
artifacts rather than one mixed correctness/performance claim.

### TP1 Proposal-KV offload authority

```text
bundle:
  experiments/autoregressive_draft/
    tp1-qwen3-loaded-offload-gpu4-20260815

exact greedy parity:
  established

real Proposal-KV H2D:
  18266 entries
  2094891008 bytes

real Proposal-KV D2H:
  83 entries
  9519104 bytes

performance pass criterion:
  false
```

This is real offload movement evidence, not a synthetic copy benchmark.

### TP4 direct correctness authority

```text
bundle:
  experiments/autoregressive_draft/
    tp4-qwen3-loaded-direct-gpu3467-authority-r2-20260815

batch 1 exact parity:
  true

batch 4 exact parity:
  true

independent verifier:
  PASS

accepted entry copy/replay/rematerialization:
  zero on every rank

free slots after release:
  90 on every rank

performance pass criterion:
  false
```

The correctness authority binds the small-page dense read-view fix for
ordinary FP decode. It does not alter Proposal-KV page ownership or claim
physical savings from the temporary read view.

### TP4 controlled-performance pilot

```text
bundle:
  experiments/autoregressive_draft/
    tp4-qwen3-controlled-performance-gpu3467-r3-20260815

workload:
  TP4
  direct Proposal-KV
  256 prompt tokens
  16 output tokens
  batch 1 / 4
  one warmup
  three measured runs
  max proposal tokens 4

artifact status:
  PASS

classification:
  PILOT_ONLY

direction:
  NEGATIVE

remote verifier:
  PASS

local verifier:
  PASS

manifest:
  PASS
```

Every measured repeat has exact target/learned output parity. Raw timing,
distributed CUDA peak memory, acceptance, and Proposal-KV counter rows are
retained for all four isolated cells.

Median comparison:

```text
batch 1:
  target TPOT:       0.277665 s
  learned TPOT:      0.353775 s
  TPOT direction:    +27.41%
  target throughput: 3.590582 tok/s
  learned throughput:2.849243 tok/s
  throughput:        -20.65%
  acceptance:        15 / 15 = 100.00%

batch 4:
  target TPOT:       0.401188 s
  learned TPOT:      0.796710 s
  TPOT direction:    +98.59%
  target throughput: 9.799624 tok/s
  learned throughput:4.761981 tok/s
  throughput:        -51.41%
  acceptance:        53 / 72 = 73.61%
```

The direct allocator records zero Proposal-KV H2D/D2H bytes. This is honest
zero movement, not missing or synthetic evidence, and therefore cannot
support an offload-benefit claim.

Peak allocated memory direction:

```text
batch 1 learned minus target:
  +287.992 MiB

batch 4 learned minus target:
  +309.826 MiB
```

The pilot proves that the controlled measurement path works and that the
current learned runtime is slower at this workload. It does not prove 4K or
long-context performance, statistical significance, a second model
structure, or promotion readiness.

The diagnostic chain is retained:

```text
r1:
  environment/source package boundary failed
  run_packages omitted, flash_attn unavailable

r2:
  both target cells completed
  learned bootstrap exposed fixed 90-slot capacity mismatch

r3:
  exact workload-derived capacities:
    batch 1 = 276
    batch 4 = 1104
  four cells completed
  dual verifier PASS
```

Strict classification:

```text
QWEN3_INDEPENDENT_DRAFT_TP1_REAL_PROPOSAL_KV_MOVEMENT=ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_DIRECT_PARITY=ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_CONTROLLED_MEASUREMENT=ESTABLISHED
QWEN3_INDEPENDENT_DRAFT_TP4_PERFORMANCE_DIRECTION=NEGATIVE
QWEN3_INDEPENDENT_DRAFT_4K_PERFORMANCE=NOT_ESTABLISHED
SECOND_LEARNED_MODEL_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
