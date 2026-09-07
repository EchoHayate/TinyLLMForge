# TP4 Segmented Capture Attribution and Scratch-KV Repair Design

**Date:** 2026-09-07

**Status:** Approved direction; design awaiting user review

**Primary model:** Qwen/Qwen3.8-27B BF16, tensor parallel size 4

**Runtime scope:** diagnostic follow-up for the default-disabled segmented
`lease_pool_index_v1` exact decode graph path

## Problem

The strict-clean r60 Stage 0 census established a terminal
`NO_GO_CORRECTNESS_OR_LIFECYCLE` result for equal-layer segmented capture.
The result is complete negative evidence, not an unfinished run:

- p2 captured `[0,32)` and `[32,64)` with a maximum segment duration of
  `3_039_488_211 ns` and a TP-wide lifecycle of `8_869_714_806 ns`;
- p3 captured `[0,22)`, `[22,43)`, and `[43,64)` with a maximum segment
  duration of `2_739_897_045 ns` and a lifecycle of `9_726_856_076 ns`;
- p4 captured four 16-layer ranges with a maximum segment duration of
  `2_553_261_459 ns` and a lifecycle of `9_935_843_505 ns`;
- every plan produced exact output, exact selected hybrid state, unchanged
  unselected state, and successful graph reset;
- every plan failed exact scratch-KV restoration;
- no plan met the frozen Stage 0 limits of `1_800_000_000 ns` per segment and
  `4_500_000_000 ns` per complete lifecycle.

The p4 timings contain a stronger diagnostic signal than a simple
"segments are still too large" conclusion:

```text
[0,16):   680,478,231 ns
[16,32): 2,553,261,459 ns
[32,48):   619,027,595 ns
[48,64): 2,502,394,024 ns
```

Qwen3.8 repeats three linear-attention layers followed by one full-attention
layer. Each 16-layer range therefore contains the same 12:4 layer-type mix.
The alternating fast/slow pattern is unlikely to be explained by layer count
or attention composition alone. Candidate causes include shared graph-pool
growth, capture-end graph instantiation, NCCL graph registration, cumulative
candidate-state ownership, stream synchronization, or another ordinal effect.

The scratch-KV failure is similarly under-localized. The current census
compares scratch KV only after replay and a restore call. It does not identify
whether the first divergence occurs during eager reference execution,
capture, replay, restore, or a later asynchronous write.

Adding more segments before resolving these two unknowns would create more
graphs, more launch overhead, more retained buffers, and a longer lifecycle
without establishing a path to correctness or the frozen ceilings.

## Goals

1. Attribute every material component of segment capture latency without
   moving work outside the measured lifecycle.
2. Distinguish layer-range cost from segment-ordinal, shared-pool, collective,
   and capture-finalization cost.
3. Identify the first exact scratch-KV divergence boundary on every rank.
4. Prove whether snapshot/restore is independently bit-exact before involving
   CUDA Graph capture.
5. Produce a bounded go/pivot decision before any production integration.
6. Preserve the existing source identity, strict-clean admission, dual
   verification, manifest, and exact-tag cleanup contracts.
7. Report both the diagnostic information gained and the added time, memory,
   synchronization, and implementation cost.

## Non-Goals

- Do not integrate segmented graphs into `model_runner` production dispatch.
- Do not run the conditional Q1 smoke or the full schema-v2 gate.
- Do not lower the `1.8 s` Stage 0 segment limit or `4.5 s` lifecycle limit.
- Do not weaken exact output, state, scratch-KV, cleanup, admission, replay,
  throughput, latency, or memory gates.
- Do not add 8- or 16-segment production plans merely to reduce individual
  capture size.
- Do not move gather, commit, final norm, LM head, synchronization, or restore
  outside measured accounting to manufacture a passing duration.
- Do not terminate, suspend, adopt, or inspect private data from foreign GPU
  workloads.
- Do not mutate or reclassify r57-r60 artifacts.
- Do not claim throughput, TTFT, TPOT, P99, or steady-state memory benefit from
  this attribution run.
- Do not begin communication-compute fusion implementation in this design.
  That becomes a separate design only if the segmented route reaches the
  pivot condition below.

## Considered Approaches

### A. Phase attribution plus scratch-KV forensic checkpoints

Instrument the existing bounded census so each segment exposes capture-entry,
body, capture-exit/instantiation, synchronization, restore, and graph-reset
time. Add isolated range and pool controls, plus exact scratch-KV checkpoints
around eager, capture, replay, and restore.

Advantages:

- directly addresses both r60 blockers;
- preserves the current implementation and evidence boundaries;
- can distinguish a fixable protocol cost from an inherent graph cost;
- requires one bounded strict-clean attribution run after local tests, followed
  by at most one fresh repaired-source validation run;
- supplies an explicit stop rule instead of open-ended optimization.

Costs and risks:

- adds diagnostic synchronization and snapshots that cannot represent
  production steady-state behavior;
- exact CUDA/PyTorch internals may combine graph finalization and
  instantiation into one observable host interval;
- one attribution run is required, and a source repair requires one additional
  fresh validation run;
- a negative result may close the segmented route without producing a runtime
  speedup.

This is the selected approach.

### B. Increase segmentation to eight or sixteen graphs

Continue dividing the 64-layer stack until every individual graph is below
the segment ceiling.

Advantages:

- simple extension of the existing plan contract;
- likely to reduce the amount of model work in each graph body.

Costs and risks:

- r60 lifecycle already increased to approximately 9-10 seconds;
- graph launches and stable boundary buffers grow with segment count;
- the alternating slow segments may persist independently of layer count;
- scratch-KV restoration remains unresolved;
- it optimizes one failed metric while making the other failed metrics worse.

This approach is rejected unless attribution first proves a bounded segment
count can satisfy both frozen timing limits.

### C. Close segmented capture and pivot to TP4 communication-compute fusion

Stop CUDA Graph work and optimize the eager TP4 decode path through
collective decomposition, stream scheduling, or operation overlap.

Advantages:

- targets steady-state tokens/s and TPOT directly;
- avoids first-capture lifecycle and graph-retained-memory costs;
- matches the broader goal of multi-GPU communication-compute fusion.

Costs and risks:

- larger and more invasive runtime change;
- dependencies between attention, residual, and MLP collectives constrain
  legal overlap;
- requires a separate profile, design, correctness proof, and benchmark;
- abandoning segmented capture without attribution would discard a
  potentially fixable 36.47% maximum-capture reduction.

This is the mandatory pivot if Approach A reaches a terminal pivot condition.

## Selected Architecture

### 1. Diagnostic-only protocol

Introduce a new attribution schema and worker mode. It remains default-off and
cannot create a production graph-cache entry.

The program has at most two hardware phases:

```text
Phase A1: source-bound attribution
Phase A2: optional fresh-source repair validation
```

Phase A2 is allowed only when Phase A1 returns `REPAIR_CANDIDATE`. It uses a
new committed source revision and a fresh immutable run tag. A failed Phase A2
cannot authorize a second repair iteration inside this program.

The diagnostic consumes the same frozen r60 workload:

```text
model:                 Qwen/Qwen3.8-27B
revision:              1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
dtype:                 BF16
tensor parallel size:  4
batch/concurrency:     8
prompt length:         256
worker max tokens:     2
model length:          384
```

Every hardware run must use a fresh immutable tag, committed source, the
existing Kerberos TTL guard, mounted remote storage, and `strict_clean`
admission. A shared-capacity run may be used only as an explicitly labeled
preflight and cannot determine the route decision.

### 2. Capture phase accounting

For every rank and segment, record the following non-overlapping host
intervals:

```text
snapshot_and_prepare_ns
graph_object_create_ns
capture_context_enter_ns
capture_body_ns
capture_context_exit_and_instantiate_ns
post_capture_synchronize_ns
post_capture_restore_ns
graph_reset_ns
segment_total_ns
program_lifecycle_ns
```

The timing wrapper places a timestamp immediately before entering
`torch.cuda.graph`, as the first operation inside the context, as the last
operation inside the context, immediately after context exit, and after the
required CUDA synchronization. PyTorch may not expose graph instantiation as
a separately stable API, so the design deliberately names the observable
interval `capture_context_exit_and_instantiate_ns` rather than claiming
unsupported precision.

Accounting must satisfy:

```text
segment_total_ns >=
    graph_object_create_ns
  + capture_context_enter_ns
  + capture_body_ns
  + capture_context_exit_and_instantiate_ns
  + post_capture_synchronize_ns
```

The complete lifecycle starts before the first snapshot or static diagnostic
buffer allocation and stops only after final restore, graph reset, memory
sampling, and process synchronization. Attribution may add diagnostic work,
but it may not subtract that work from the lifecycle.

Each row also records:

- segment ordinal and exact half-open layer range;
- count of linear- and full-attention layers;
- count and bytes of candidate convolution/recurrent tensors;
- stable hidden, candidate, and logits buffer bytes;
- CUDA allocated/reserved deltas;
- shared capture-pool identity;
- collective count by operation class when available from existing receipts;
- current CUDA stream identity;
- exact source and plan hash.

### 3. Bounded range and pool controls

The attribution matrix is intentionally small:

#### Control 1: Stitched p4 repeat

Run the existing p4 plan twice in the same worker after an explicit restore
and graph reset boundary. This determines whether the alternating pattern is
stable by ordinal and whether second-pass capture is materially different
after lazy initialization.

The second repeat is diagnostic only and does not replace the first formal
row.

#### Control 2: Isolated 16-layer ranges

Capture `[0,16)`, `[16,32)`, `[32,48)`, and `[48,64)` independently. For each
range, produce its input hidden state through an eager prefix from the same
restored snapshot, then start the measured isolated-range lifecycle before
the range-specific capture.

This distinguishes:

- a range-specific slow body, if the same range remains slow in isolation;
- an ordinal/shared-pool effect, if isolated ranges converge but stitched
  ranges alternate.

Eager-prefix preparation is reported separately and cannot be used as a
production capture measurement.

#### Control 3: Shared-pool versus isolated-pool

Repeat only the slowest and fastest isolated ranges with:

- the current shared capture pool;
- a fresh per-range capture pool.

This is capped at four additional captures. If isolated pools reduce
capture-exit cost, the artifact must also report their additional reserved
memory and reject the approach if it violates the existing memory gate.

No other partition search or autotuning is permitted in this stage.

### 4. Scratch-KV forensic state machine

Fill the eight approved scratch slots with a deterministic nonzero sentinel
before the initial snapshot. The sentinel is derived from run tag, rank,
K/V selector, layer index, scratch slot ordinal, head index, and element
offset. This prevents an all-zero buffer from hiding a missed restore.

Record exact scratch snapshots at these boundaries:

```text
S0  after deterministic sentinel initialization
S1  after eager reference execution
S2  after restoring S0 following eager
S3  after each segment capture
S4  after restoring S0 following capture
S5  after stitched graph replay
S6  after restoring S0 following replay
S7  after final cleanup synchronization
```

For every transition, record separately for keys and values:

- exact equality to S0;
- SHA-256 of canonical CPU bytes plus dtype and shape;
- mismatching element count;
- first mismatching layer, scratch slot, head, and element offset;
- maximum absolute difference for floating-point values;
- whether a required CUDA synchronization completed before the snapshot.

The full tensor contents must not be written to artifacts. Hashes and bounded
diff summaries are sufficient.

The worker applies these fail-closed rules:

1. If an immediate `restore_kv_slots(S0)` round trip fails before eager or
   capture, record reason `scratch_restore_primitive` and classify
   `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`.
2. If S1 differs but S2 equals S0, eager scratch writes are expected and the
   restore primitive is correct.
3. If S4 differs from S0, localize the failure to capture or its restore.
4. If S4 equals S0 but S6 differs, localize the failure to replay or its
   restore.
5. If S6 equals S0 but S7 differs, classify a post-restore asynchronous or
   cleanup mutation.
6. No timing result is eligible for route selection unless S2, S4, S6, and S7
   all equal S0 on every rank.

### 5. Rank agreement and independent verification

The producer and two independent verifiers reconstruct:

- source, model, workload, admission, and plan identity;
- phase-duration non-negativity and accounting consistency;
- rank inventory and TP-wide maxima;
- layer-type and candidate inventories for each range;
- shared/isolated pool control identity;
- every scratch checkpoint hash and diff summary;
- exact output, selected-state, unselected-state, and graph-reset results;
- memory deltas and cleanup receipts;
- final decision classification.

TP-wide phase values use the maximum rank duration. A plan cannot pass by
averaging a slow rank with faster ranks.

The final manifest binds all producer inputs, phase rows, scratch rows,
process receipts, cleanup, and report artifacts. A post-verification manifest
binds both verifier outputs.

### 6. Decision classifier

Each immutable run emits exactly one terminal state.

#### `REPAIR_CANDIDATE`

Allowed only for the Phase A1 attribution run when:

- the first scratch-KV divergence boundary is exact and rank-consistent;
- the slow-segment phase is exact and rank-consistent;
- one minimal source-controlled repair can be stated without changing the
  frozen gates or moving measured work;
- the evidence supplies a conservative path to both timing ceilings;
- cleanup is `CLEAN`.

This classification authorizes strict RED/GREEN implementation of that one
repair and one fresh Phase A2 run. It is not a correctness, capture, replay,
or performance GO.

#### `GO_SEGMENTED_REPAIR`

Allowed only for the fresh-source Phase A2 validation run when all of the
following hold:

- every required scratch restore checkpoint is exact on every rank;
- the Phase A1 divergence and the implemented repair are source-controlled
  and independently verified;
- attribution identifies a specific removable phase that explains the slow
  segments;
- after subtracting no measured work, the repaired source measures
  `<= 1.8 s` maximum segment and `<= 4.5 s` lifecycle;
- the repair does not require more than four production segment graphs;
- measured retained memory remains inside the existing production gate;
- exact output and selected/unselected state checks pass;
- cleanup is `CLEAN`.

This classification authorizes a new implementation design and plan. It does
not authorize production integration or a performance claim.

#### `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`

Required when any of the following is true:

- scratch restore cannot be made exact without weakening semantics;
- capture-exit/instantiation, collective registration, or other mandatory
  per-graph overhead makes the `4.5 s` lifecycle infeasible;
- the best conservative repaired segment remains above `1.8 s`;
- passing requires more than four graph launches;
- passing requires moving measured work outside the lifecycle;
- isolated-pool capture requires memory beyond the frozen gate;
- no single source-supported cause explains the alternating slow segments;
- Phase A2 fails any correctness, timing, memory, verification, or lifecycle
  requirement.

This classification closes the current segmented-capture line. The next
artifact must be a separate TP4 communication-compute fusion design based on
steady-state profiling.

#### `INCOMPLETE`

Used for missing artifacts, rank disagreement, verifier disagreement,
incomplete cleanup, infrastructure interruption, or unbound source. An
`INCOMPLETE` run cannot choose either technical route and requires a fresh
tag after the evidence defect is corrected.

## Error Handling and Cleanup

- Snapshot, capture, replay, restore, verification, or reset failure preserves
  the first operational error while attaching later cleanup errors.
- Every graph is reset in reverse creation order.
- Every owned child and process group must terminate through the existing
  exact-tag lifecycle owner.
- Final cleanup requires rank exit code zero, process-group destruction, no
  owned child, and repeated empty exact-tag scans.
- SSH return code 255 receives the existing bounded full-pipeline retry only
  where source/archive staging is idempotent. A worker disconnect never
  authorizes duplicate launch.
- Foreign GPU processes are never terminated or adopted.
- All remote source, logs, temporary files, and artifacts remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

## Evidence and Reporting

The final report must include both benefit and cost.

Benefit fields:

- exact phase attribution for every segment;
- first scratch-KV divergence boundary;
- whether the restore primitive is independently exact;
- whether a bounded source repair exists;
- maximum-capture reduction attributable to that repair, if measured.

Cost fields:

- added diagnostic captures and synchronizations;
- attribution lifecycle duration;
- CPU snapshot/hash time;
- allocated and reserved memory deltas;
- stable boundary-buffer bytes;
- graph count and projected replay launches;
- whether the result closes the segmented route without an E2E speedup.

The report must state explicitly:

```text
capture attribution is not steady-state performance
scratch repair is not replay qualification
a GO diagnosis is not production GO
a pivot is a technically complete negative result
```

## Testing Strategy

### Pure local tests

- phase accounting accepts valid non-overlapping intervals and rejects
  negative, missing, overlapping, or inconsistent totals;
- layer-type inventory matches exact half-open ranges;
- scratch sentinel generation is deterministic and rank-sensitive;
- scratch diff summaries identify K/V, layer, slot, head, and offset without
  serializing full tensors;
- restore state-machine classification covers failures at S2, S4, S6, and S7;
- route classifier covers repair-candidate, GO, pivot, and incomplete
  conditions;
- rank aggregation uses TP-wide maxima rather than averages;
- shared-pool and isolated-pool rows cannot be confused;
- manifest verification detects any modified phase or scratch row.

### Backend contract tests

- eager, capture, replay, and restore checkpoints occur in the required order;
- every diagnostic synchronization is explicit and recorded;
- source errors remain primary when restore/reset also fail;
- graph reset remains reverse-ordered and idempotent;
- non-root TP ranks may legally own no logits;
- snapshot and restore preserve dtype, shape, and device semantics.

### Hardware gate

After all local tests and review pass:

1. commit and push the exact diagnostic source;
2. confirm the fresh tag is unused locally and remotely;
3. require sufficient Kerberos lifetime and mounted-storage preflight;
4. wait for four `strict_clean` GPUs without touching foreign work;
5. run one bounded attribution matrix;
6. produce local and remote independent verification;
7. require `CLEAN` cleanup and a final live exact-tag scan;
8. if Phase A1 returns `REPAIR_CANDIDATE`, implement only the admitted repair
   through RED/GREEN, commit and push it, then use a fresh tag for one bounded
   Phase A2 validation;
9. require local and remote independent verification, manifest integrity,
   `CLEAN` cleanup, and a final live exact-tag scan for Phase A2;
10. stop at `GO_SEGMENTED_REPAIR`,
    `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`, or `INCOMPLETE`.

No production smoke or full performance gate is part of this design.

## Success Criteria

This design succeeds when it produces one source-bound, independently
verified, clean terminal decision:

- `GO_SEGMENTED_REPAIR` with an exact scratch-KV repair and measured compliance
  with both Stage 0 timing ceilings; or
- `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION` with enough attribution to close
  segmented capture without further blind partitioning.

It does not succeed merely because another TP4 process ran, another segment
became shorter, or an unverified restore appeared to work.
