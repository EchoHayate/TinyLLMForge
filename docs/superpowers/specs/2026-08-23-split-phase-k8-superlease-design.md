# Split-Phase K8 Exact-Burst Superlease Design

**Date:** 2026-08-23
**Status:** Approved under the standing autonomous-optimization authorization
**Stage-1 model:** Qwen3-0.6B
**Primary target:** retain K8 exact-burst efficiency while restoring K4-scale
host-visible token cadence

## Objective

Run eight exact greedy decode forwards under one scheduler authorization, but
publish their tokens to the host-visible sequence in two ordered phases:

```text
K8 GPU work:       replay 1 2 3 4 5 6 7 8
host publication:              prefix 1..4
                                      suffix 5..8
```

The optimization targets the tension already measured by the exact-burst
gate:

- K8 has better TPOT and throughput than K4;
- K8 has roughly twice K4's maximum host-visible publication gap;
- waiting for all eight tokens before any host-visible progress makes K8 less
  attractive for interactive decode even when aggregate throughput improves.

The design must preserve exact token and bounded logit parity. It must not
weaken scheduler authority, KV-block generation checks, lease identity,
publication ordering, rollback, stop handling, or terminal quarantine
behavior. The feature remains default-disabled until a source-bound hardware
gate proves both benefit and cost.

## Scope

Stage 1 is deliberately narrow:

- TP1, rank 0, batch size 1;
- exact greedy completion-only decode;
- `temperature == 0`;
- `ignore_eos == true`;
- eight authorized tokens that stay inside one writable KV block;
- one prefix publication of four tokens and one suffix publication of four
  tokens;
- no continuation-epoch composition;
- no speculative decoding, KV offload, CPU offload, KV quantization, Quest,
  or compact-attention composition;
- no new scheduling decision between the prefix and suffix publication.

The implementation may reuse existing CUDA stream, event, and pinned-memory
patterns, but it must not claim general asynchronous serving support from this
Stage-1 result.

## Observed Opportunity

The canonical exact-burst Stage-1 result showed that K8 improves aggregate
decode efficiency over K4:

- aggregate median TPOT improved by `6.562618%`;
- aggregate P95 TPOT improved by `9.329137%`;
- aggregate E2E improved by `6.111457%`.

The cost was host-visible cadence:

- K4 maximum host-visible gap: `12.633577 ms`;
- K8 maximum host-visible gap: `24.035218 ms`.

The current runtime performs all K replays, converts the complete
`token_history` slice to a Python tuple, then commits all K tokens in one
scheduler transaction. Therefore K8's compute benefit and K8's publication
gap are coupled by the host protocol, not by the model's autoregressive
dependency itself.

After replay four, the first four tokens and their KV writes are already
complete and immutable. Replays five through eight consume that resident
state but do not require the host to wait before initiating a D2H copy of the
prefix. A copy stream can transfer tokens 1..4 while the compute stream
continues tokens 5..8.

## Considered Approaches

### A. Split-phase K8 superlease

Authorize one K8 lease, enqueue eight exact graph replays, transfer the first
four and last four tokens into separate pinned mailboxes, publish the prefix,
and drain the suffix before the next scheduler decision.

Advantages:

- retains one K8 scheduling and graph setup decision;
- overlaps the prefix D2H with replays five through eight;
- restores a K4-scale first publication boundary;
- keeps one parent authority across both publications;
- directly addresses the measured K8 cadence cost.

Costs and risks:

- two D2H transfers instead of one;
- two pinned host mailboxes and CUDA completion events;
- a pending suffix spans two engine calls;
- cancellation and failure must respect GPU completion before discarding
  unpublished data;
- scheduler commit and rollback must support a lease that remains active
  after prefix publication.

This is the selected approach.

### B. Latency-budget adaptive burst width

Select K4 or K8 from a predicted host-visible latency budget.

This is simpler and useful as a later policy layer, but it only chooses
between throughput and cadence. It does not improve the K8 Pareto point and
cannot prove that K8 compute can coexist with K4 publication granularity.

### C. K4 macrograph

Capture four complete decode steps in one CUDA Graph and publish every four
tokens.

This has lower lifecycle risk, but earlier evidence indicates that per-replay
launch overhead is not the dominant remaining cost. It also gives up K8's
measured scheduling and setup amortization.

## Terminology

### Parent superlease

One immutable scheduler authorization covering all eight write positions,
their physical slots, the complete block-table identity, graph generation,
and initial sequence state.

### Prefix

Tokens and KV writes for ordinals `[0, 4)`. Prefix publication advances the
host-visible sequence by four tokens but does not release the parent
superlease.

### Suffix

Tokens and KV writes for ordinals `[4, 8)`. Suffix publication advances the
same sequence by the remaining four tokens and releases the parent
superlease.

### Publication ticket

An immutable child identity derived from the parent lease and phase:

```text
parent_lease_identity_sha256
phase = prefix | suffix
phase_start_ordinal
phase_token_count
first_write_position
last_write_position
first_physical_slot
last_physical_slot
ticket_identity_sha256
```

The ticket is evidence and validation input. It is not independent scheduling
authority.

### Pending suffix

Engine-owned state proving that GPU work for the suffix was enqueued under a
specific parent lease and that no new scheduler decision may occur until the
suffix is committed or terminally discarded.

## Architecture

### 1. Immutable parent lease

The existing `ExactGreedyDecodeBurstLease` remains the source of authority.
For the split-phase path:

- `requested_token_count == 8`;
- `authorized_token_count == 8`;
- all eight physical slots belong to one writable block;
- remaining output budget is at least eight;
- the sequence is completion-only and ignores EOS;
- no existing exact-burst lease or pending split-phase suffix exists.

The scheduler records one pending parent lease. Prefix publication must not
replace, shorten, or regenerate it.

### 2. Split result

The model-runner path returns a split result rather than one ordinary burst
result:

```text
parent lease identity
graph identity
replay count = 8
prefix ticket
suffix ticket
prefix mailbox handle
suffix mailbox handle
prefix completion handle
suffix completion handle
enqueue timestamps
copy byte counts
```

Mailbox and completion handles remain process-local runtime objects. They are
not serialized into benchmark artifacts. Artifact rows contain stable
identities, counts, bytes, and timings.

The result constructor validates that prefix and suffix are contiguous,
non-overlapping, exhaustive children of the same K8 lease.

### 3. CUDA execution and copy order

The exact-burst graph continues to run on the current compute stream. The
split path owns one dedicated D2H copy stream and two reusable pinned CPU
mailboxes, each sized for four `int64` tokens.

Execution order:

1. bind graph state and validate the K8 parent lease;
2. replay ordinals 0 through 3 on the compute stream;
3. record `prefix_compute_done` on the compute stream;
4. make the copy stream wait on `prefix_compute_done`;
5. enqueue `token_history[0:4]` D2H into prefix mailbox A;
6. record `prefix_copy_done` on the copy stream;
7. continue replay ordinals 4 through 7 on the compute stream;
8. record `suffix_compute_done` on the compute stream;
9. make the copy stream wait on `suffix_compute_done`;
10. enqueue `token_history[4:8]` D2H into suffix mailbox B;
11. record `suffix_copy_done` on the copy stream;
12. return a split result without synchronizing the entire device.

The host waits only for `prefix_copy_done` before constructing and committing
the prefix. It must not wait for `suffix_copy_done` on the prefix critical
path.

The copy stream must not read a token-history region before the corresponding
compute event. Mailbox reuse is forbidden while any pending result still owns
that mailbox generation.

### 4. Engine state machine

The engine owns at most one split-phase transaction:

```text
IDLE
  -> ENQUEUED
  -> PREFIX_READY
  -> PREFIX_COMMITTED
  -> SUFFIX_READY
  -> SUFFIX_COMMITTED
  -> IDLE
```

Terminal error states:

```text
ENQUEUED | PREFIX_READY
  -> PRE_PREFIX_FAILED

PREFIX_COMMITTED | SUFFIX_READY
  -> POST_PREFIX_FAILED
```

State transitions are monotonic. Repeating a transition, skipping a phase, or
using a ticket from another parent lease is an invariant violation.

At the beginning of every engine step, before calling
`scheduler.schedule()`:

1. inspect the pending split-phase transaction;
2. if no pending suffix exists, continue normally;
3. if a pending suffix exists, wait for its completion handle;
4. validate the suffix ticket and current parent lease;
5. construct the suffix token tuple from mailbox B;
6. prepare and commit suffix publication;
7. clear the pending transaction only after scheduler commit succeeds;
8. return the newly published output without making a new scheduling
   decision in that engine call.

This ordering prevents another request, prefill chunk, block allocation, or
ordinary decode step from observing an intermediate sequence state while the
parent K8 lease is still active.

### 5. Scheduler publication phases

The scheduler adds explicit prefix and suffix prepare/commit entrypoints.

Prefix prepare validates:

- the parent lease is the unique pending exact-burst lease;
- the sequence still matches the parent's initial state;
- the prefix ticket covers ordinals `[0, 4)`;
- prefix tokens count is exactly four;
- parent block identities and generations remain valid;
- all prefix KV writes are inside the parent's authorized range;
- no prefix was previously prepared or committed.

Prefix commit:

- appends four host-visible tokens;
- publishes only full KV blocks whose materialized boundary is justified by
  the prefix;
- advances completion and sequence length by four;
- records prefix cadence and cost statistics;
- changes the pending lease phase to `prefix_committed`;
- keeps the parent lease pending;
- does not decrement the parent pending-lease count.

Suffix prepare validates:

- prefix commit completed for the same parent lease;
- the sequence length and completion count equal the parent initial values
  plus four;
- the suffix ticket covers ordinals `[4, 8)`;
- suffix tokens count is exactly four;
- parent block identities and generations remain valid;
- no scheduler generation advanced after parent authorization;
- no new scheduling decision occurred between phases.

Suffix commit:

- appends four host-visible tokens;
- publishes full KV blocks up to the parent's final materialized boundary;
- records suffix cadence and aggregate superlease statistics;
- releases the parent lease;
- decrements the pending-lease count exactly once;
- returns the scheduler to ordinary eligibility.

The existing one-phase commit remains unchanged for K2 through K8 when
split-phase mode is disabled.

### 6. Scheduler generation

Prefix and suffix are two publication transactions under one scheduling
generation. Prefix commit must not increment `schedule_generation`. The
suffix drain occurs before `schedule()`, so the scheduler generation must
still equal the parent lease generation.

Any observed generation drift is terminal after GPU enqueue because eight KV
writes may already exist.

### 7. Rollback boundary

Each publication phase uses the existing scheduler postprocess journal with a
phase-specific snapshot.

Prefix rollback restores the host-visible sequence and scheduler metadata to
the parent's initial state. It does not undo GPU KV writes. Because those
writes may include all eight authorized positions, a prefix commit failure
after GPU enqueue quarantines the exact-burst graph and terminally fails the
parent lease.

Suffix rollback restores the host-visible sequence to the already-committed
prefix state. It must never remove the prefix. A suffix commit failure also
quarantines the exact-burst graph and terminally fails the parent lease.

No path retries the same parent lease after GPU mutation.

## Cancellation and Failure Semantics

### Before GPU enqueue

Eligibility or materialization failure returns the ordinary exact-burst
fallback. The pending parent lease may be cancelled without quarantine because
no authorized KV position was mutated.

### After enqueue, before prefix publication

The runtime must first reach a GPU-safe point:

- wait for the suffix completion event, or
- synchronize the owning streams if event observation itself failed.

It then:

- discards both mailbox generations;
- terminally fails the parent lease;
- invalidates continuation state;
- quarantines the exact-burst graph;
- exposes no generated tokens to the host.

The scheduler must not reuse or re-authorize the affected KV positions.

### After prefix publication, before suffix publication

The prefix is already externally visible and cannot be rolled back by a later
engine call. The runtime must:

- wait for the suffix GPU-safe point;
- attempt exactly one validated suffix drain;
- if suffix reconstruction or commit fails, mark the request failed,
  quarantine the graph, and preserve the four-token prefix as the last
  committed host state;
- never schedule ordinary work while the suffix is unresolved.

User cancellation arriving after prefix commit does not silently drop already
computed suffix KV writes. Stage 1 drains the suffix transaction to a safe
terminal state before applying request cancellation. This cost is measured.

### Copy failure

A D2H enqueue, event record, event wait, or mailbox conversion failure after
any replay is terminal. There is no synchronous `.tolist()` fallback on the
same mutated transaction.

## Configuration

Add one strict boolean:

```text
exact_greedy_decode_burst_split_phase = false
```

It is effective only when:

```text
exact_greedy_decode_burst = true
exact_greedy_decode_burst_tokens = 8
exact_greedy_decode_burst_continuation = false
```

Non-boolean values are rejected. Enabling split phase with a non-K8 width or
with continuation enabled is a configuration error rather than a silent
fallback, because those combinations would invalidate the frozen gate
contract.

## Observability

Expose cumulative counters and stable reason maps for:

- split-phase attempts, acceptances, fallbacks, and quarantines;
- parent leases created, completed, failed, and pending;
- prefix and suffix tickets created and committed;
- prefix and suffix replay counts;
- prefix and suffix D2H calls and bytes;
- prefix and suffix event records and waits;
- prefix and suffix mailbox allocations, reuses, and generation conflicts;
- current and peak pinned mailbox bytes;
- prefix-ready wait time;
- suffix-ready wait time;
- enqueue-to-prefix-publication gap;
- prefix-to-suffix-publication gap;
- maximum host-visible gap;
- pending-suffix engine drains;
- scheduler calls skipped because a suffix was drained;
- cancellation drains and their duration;
- failures before prefix and after prefix;
- fallback, failure, and quarantine counts by stable reason.

Each engine observation row reports:

```text
split_phase_attempted
split_phase_accepted
parent_lease_identity_sha256
prefix_ticket_identity_sha256
suffix_ticket_identity_sha256
phase_published
phase_token_count
replay_count
prefix_d2h_calls
suffix_d2h_calls
prefix_d2h_bytes
suffix_d2h_bytes
prefix_wait_ns
suffix_wait_ns
host_visible_gap_ns
pending_suffix
fallback_reason
quarantine_reason
```

## Cost Model

The gate must report both benefit and cost.

Expected benefit:

- K8-level setup and scheduling amortization;
- K4-scale first host-visible publication gap;
- overlap of the first four-token D2H with replays five through eight.

Explicit costs:

- D2H calls increase from one K8 transfer to two K4 transfers;
- CUDA events: two compute-done and two copy-done events per superlease;
- one dedicated copy stream per graph/runtime owner;
- two pinned mailboxes of four `int64` tokens, plus ownership metadata;
- pending suffix state across engine calls;
- one extra scheduler publication transaction;
- suffix drain can consume a complete engine call without a new schedule;
- cancellation may wait for all eight GPU replays and both copy safe points;
- implementation and invariant surface are larger than ordinary K8.

The optimization is not a GO if cadence improves only by materially
regressing TPOT, throughput, TTFT, E2E, or memory.

## Correctness Strategy

### Token parity

For every paired request:

- `host_greedy`, K4, K8, and split K8 produce exactly the same token IDs;
- decoded text is exactly equal;
- prefix plus suffix equals the ordinary K8 result;
- prefix and suffix are contiguous and exhaustive;
- every parent lease produces exactly one prefix and one suffix commit.

### Logit parity

Correctness runs collect bounded float32 logits at frozen global decode
ordinals. Split publication must not change graph replay order or sampled
logits.

Thresholds:

- maximum absolute logit difference `<= 0.25`;
- per-pair mean absolute difference `<= 0.05`;
- argmax token equal at every sampled point.

### Lifecycle invariants

The producer and independent verifier reject:

- suffix publication before prefix;
- duplicate prefix or suffix;
- missing suffix;
- parent lease release after prefix;
- more than one pending parent lease;
- scheduler generation drift;
- a scheduler call while suffix is pending;
- mailbox reuse before completion;
- ticket/lease identity mismatch;
- incomplete event, D2H, or byte inventory;
- unexpected fallback, failure, quarantine, or pending state.

## Stage-1 Benchmark Matrix

Model:

```text
Qwen3-0.6B
```

Arms:

```text
host_greedy
decode_burst_k4
decode_burst_k8
decode_burst_k8_split_phase
```

Workload:

- short, medium, and long aligned prompt buckets;
- fixed 128 generated tokens;
- batch size 1;
- five paired repetitions per arm and bucket;
- frozen request order and seed;
- 60 performance rows total;
- complete correctness rows and logit sidecars;
- source commit and dirty-patch hashes bound in the manifest.

The benchmark runner, producer gate, independent verifier, and remote
controller follow the existing exact-burst artifact layout. Run tags are
immutable and never reused.

## Stage-1 GO/NO_GO Gate

Correctness and lifecycle requirements are mandatory:

- exact tokens and text for all paired groups;
- all logit thresholds pass;
- exactly one prefix and one suffix per accepted parent lease;
- zero unexpected fallback, failure, quarantine, or pending transaction;
- complete parent/ticket/event/D2H/mailbox inventory;
- producer and independent verifier agree.

Performance requirements compare split K8 with ordinary K8 unless otherwise
stated:

- aggregate median TPOT regression `<= 2%`;
- aggregate throughput regression `<= 2%`;
- every bucket TTFT regression `<= 3%`;
- every bucket E2E regression `<= 3%`;
- reserved-memory regression `<= 3%`;
- split K8 maximum host-visible gap `<= 60%` of paired K8;
- split K8 median maximum gap must not regress more than `3%` versus paired
  K4;
- no bucket median or P95 TPOT regression above `3%` versus K8;
- cost fields are complete and internally consistent.

Classification:

- `GO`: every mandatory requirement passes;
- `NO_GO_CORRECTNESS`: any token, text, logit, ordering, identity, or lifecycle
  requirement fails;
- `NO_GO_PERFORMANCE`: correctness passes but any frozen performance or cost
  threshold fails;
- `INCOMPLETE_EVIDENCE`: the matrix, sidecars, manifest, source binding, or
  verifier agreement is incomplete.

The frozen matrix is completed even if an early row suggests failure.

## Stage-2 Promotion

Only a Stage-1 `GO` authorizes a Qwen3-8B gate using the same four arms,
matched inputs, output lengths, concurrency, and hardware. Stage 2 must report
the same benefit and cost fields and may not reuse Qwen3-0.6B thresholds as a
claim of production generality.

## Test Plan

### Pure contracts

- configuration validation;
- prefix/suffix ticket construction and digest stability;
- child range contiguity and exhaustiveness;
- state-machine transition validation;
- cost and inventory schema validation;
- stable fallback and terminal reason validation.

### Graph/runtime tests

- exact replay order remains 0 through 7;
- prefix event is recorded after replay four;
- prefix D2H reads only history `[0:4]`;
- suffix event is recorded after replay eight;
- suffix D2H reads only history `[4:8]`;
- prefix wait does not wait on suffix completion;
- mailbox generation prevents premature reuse;
- copy or event failure quarantines after mutation.

### Scheduler tests

- parent lease remains pending after prefix commit;
- prefix advances sequence and completion counts by four;
- suffix validation uses prefix-advanced state;
- suffix advances counts by another four and releases once;
- schedule generation does not advance between phases;
- prefix rollback restores the initial host state;
- suffix rollback preserves the committed prefix;
- cancellation drains to a GPU-safe terminal state.

### Engine tests

- pending suffix drains before `scheduler.schedule()`;
- a suffix-drain step makes zero scheduler calls;
- ordinary K8 and non-split widths retain existing behavior;
- suffix failure cannot fall through to ordinary decode;
- observation rows distinguish prefix and suffix publication.

### Benchmark tooling tests

- frozen 60-row matrix;
- correctness row and sidecar completeness;
- producer/verifier agreement;
- threshold boundary tests;
- source-head, manifest, and run-tag immutability;
- remote paths remain under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.

## Non-Goals

This design does not claim:

- a new decoding algorithm;
- academic originality;
- reduced target-model forward count;
- speculative acceptance gains;
- multi-sequence or TP>1 support;
- safe EOS-sensitive split publication;
- arbitrary phase widths;
- composition with continuation epochs;
- production readiness before both hardware stages.

It is an original engineering proposal for this runtime's measured data flow:
decouple K8 compute authorization from K8 host-publication granularity while
preserving one scheduler-owned transaction.
