# TP4 Completion-Owned Overlap Stage-0.1 Design

**Date:** 2026-09-08
**Status:** Written design pending user review
**Source anchor:** `001ca59f8037f79725c1eb21ff2f7b79d47bad6e`
**Target branch:** `feat/kv-sparse-attention`
**Scope:** model-neutral four-GPU mechanism qualification
**Default:** disabled
**Qwen integration:** prohibited unless this design's fresh Stage-0.1 gate returns
`GO_COMPLETION_OWNED_OVERLAP_MICROGATE`

## 1. Decision

Run one correctness-first redesign of the lease-sealed state-commit overlap
microgate. The redesign transfers collective-completion authority from a CUDA
event recorded on the caller-created communication stream to the asynchronous
NCCL `Work` returned by `dist.all_reduce(..., async_op=True)`.

The formal experiment has three arms:

1. **Synchronous baseline:** complete the AllReduce, copy the BF16 output, then
   copy the side-effect payload.
2. **Event-only diagnostic control:** preserve the Stage-0 ordering bug so the
   gate can prove that its correctness oracle detects premature consumption.
   This arm is never promotion-eligible and contributes no performance result.
3. **Completion-owned candidate:** launch asynchronous AllReduce and the
   side-effect copy independently, call `Work.wait()` from the consuming CUDA
   stream before reading the reduced tensor, wait for side-effect completion
   before seal or publish, and then copy the exact BF16 output.

This is Stage 0.1 rather than a reinterpretation of Stage 0. The immutable r5
result remains `NO_GO_CORRECTNESS_OR_LIFECYCLE`. Stage 0.1 uses a new source
revision, a fresh immutable attempt tag, and a separately sealed evidence
bundle.

The redesign changes synchronization ownership and measurement semantics. It
does not change model math, reduction order, tensor shapes, accumulation
precision, output dtype, state payload, or transaction semantics.

## 2. Why Stage 0 failed

The terminal Stage-0 r5 attempt completed all 180 rows and all lifecycle
checks, but:

- `reduced_output_exact` failed on 12 of 180 rows;
- `final_output_exact` failed on 176 of 180 rows;
- median reported overlap was zero for active-token groups 1, 4, and 8; and
- candidate host submission regressed by 67.95% to 77.87%.

The current runtime stores the returned collective `Work`, but its normal
`join()` path never waits on that object. It instead waits on a
`consumer_ready_event` recorded on the caller-created communication stream
immediately after asynchronous submission.

That event is not authoritative for ProcessGroupNCCL completion. NCCL work is
issued on ProcessGroupNCCL-owned CUDA streams. The supported ownership
transfer is `Work.wait()` or its synchronization equivalent, which makes the
current user-facing stream wait for NCCL completion. A later event on the
caller stream can only be used as completion evidence after that ownership
transfer has occurred.

The observed r5 failure pattern is consistent with premature consumption:
the immediate BF16 output copy failed much more frequently than the later
FP32 reduced-result check. This remains a hypothesis until Stage 0.1
reproduces the event-only failure and makes the completion-owned arm exact in
the same process and workload.

## 3. Goals

Stage 0.1 has six goals:

1. Establish correct NCCL completion ownership for the consuming stream.
2. Reproduce the unsafe event-only behavior as a diagnostic control.
3. Preserve exact FP32 reduction, BF16 output, shadow state, lifecycle, and
   transaction identity.
4. Measure useful concurrency without treating a caller-stream post-submit
   event as NCCL completion.
5. Determine whether the corrected mechanism provides at least a modest,
   repeatable critical-path benefit at active-token groups 4 and 8.
6. Produce a complete, source-bound, independently reconstructed terminal
   result before any Qwen integration.

## 4. Non-goals

Stage 0.1 does not:

- integrate with Qwen3.8 or any other model;
- modify `tinyvllm/layers/linear.py`;
- change `RowParallelLinear` semantics;
- replace NCCL or implement a custom reduction;
- introduce ReduceScatter, AllGather, tensor chunking, cross-layer pipelining,
  or cross-request wavefront scheduling;
- claim end-to-end QPS, TPOT, TTFT, or model-level latency improvement;
- use profiler traces as formal timing evidence;
- repair, overwrite, or reclassify any Stage-0 attempt;
- permit the event-only arm to pass, promote, or contribute favorable timing;
- lower correctness, lifecycle, overlap, tail-latency, host-submission,
  allocation, memory, identity, or cleanup gates.

If Stage 0.1 is correct but misses its performance gate, this mechanism stops.
The next optimization family must move to a larger-granularity TP design such
as chunked collective/compute pipelining or collective decomposition under a
separately reviewed design.

## 5. Completion ownership

### 5.1 Ownership rule

The asynchronous collective's returned `Work` is the sole authority for when
the reduced tensor can become consumable.

A CUDA event recorded after asynchronous submission is not a substitute for
`Work.wait()`. The completion-owned arm must perform the ownership transfer
from ProcessGroupNCCL to the consuming stream before any read, cast, digest,
seal, publication, allocator reuse, or request-slot reuse involving the
reduced tensor.

### 5.2 Required consuming-stream sequence

The completion-owned join sequence is:

```text
consumer stream becomes current
    |
    +-- Work.wait()
    |     establishes the NCCL-stream -> consumer-stream dependency
    |
    +-- record collective_visible_event on the consumer stream
    |
    +-- wait_event(side_effect_ready_event)
    |
    +-- read/copy reduced FP32 tensor into BF16 output
    |
    +-- seal
    |
    +-- publish or abort
```

`collective_visible_event` means "the reduced result is ordered before this
point on the consumer stream." It does not claim to identify the exact start
or end timestamp of NCCL kernels.

### 5.3 Host behavior

The formal candidate path may call `Work.wait()`, but may not use:

- `torch.cuda.synchronize()` or device-wide synchronization;
- `event.synchronize()` in the timed success path;
- `.item()` for readiness or identity decisions;
- Python busy polling;
- host polling of CUDA events;
- a second synchronous collective used to validate the first;
- a hidden synchronous fallback after either candidate branch launches.

Host-submission time remains a separately measured cost. If the installed
PyTorch/NCCL combination makes `Work.wait()` host-blocking enough to violate
the frozen host ceiling, the result is a performance no-go rather than a
reason to weaken the ceiling.

### 5.4 Abort ownership

An abort after launch must retire all work owned by the transaction:

- wait for or safely terminate the owned collective according to the
  supported `Work` contract;
- wait for the owned side-effect copy;
- preserve the old active state;
- prevent shadow publication;
- release pooled resources only after their final owning dependency.

Abort may not affect foreign processes, unrelated collectives, other request
slots, or a different transaction generation.

## 6. Three-arm data flow

### 6.1 Synchronous baseline

```text
producer stream: create deterministic FP32 local result
    |
synchronous NCCL AllReduce
    |
copy exact reduced FP32 result to BF16 output
    |
copy BF16 side-effect payload to shadow state
    |
seal and publish
```

The baseline is the correctness and performance reference. It executes the
same reduction, output cast/copy, state bytes, identity checks, and
transaction lifecycle as the completion-owned candidate.

### 6.2 Event-only diagnostic control

```text
producer stream: create deterministic FP32 local result
    |
caller communication stream: submit async AllReduce
    |
record caller-stream event without Work.wait()
    |
consumer waits only on that event and reads reduced tensor
```

The diagnostic arm deliberately retains the Stage-0 completion error. It runs
only in untimed diagnostic rows. Its required outcome is at least one
correctness mismatch across the frozen matrix. If it unexpectedly becomes
exact, Stage 0.1 is `INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED`; the experiment
must not silently promote the completion-owned candidate because the proposed
root cause was not discriminated.

### 6.3 Completion-owned candidate

```text
producer stream creates FP32 local result
  |
  +-- ProcessGroupNCCL internal stream: asynchronous AllReduce
  |
  +-- side-effect stream: copy BF16 payload to shadow state

consumer stream:
  Work.wait()
  record collective_visible_event
  wait side_effect_ready_event
  copy exact reduced result to BF16 output
  seal and publish
```

Both branches may execute concurrently, but the consumer cannot use the
reduced tensor before `Work.wait()` transfers completion ownership. Seal and
publication require both branches.

## 7. Runtime boundaries

### 7.1 Generic runtime

The model-neutral runtime owns:

- one active ticket per resource bundle;
- the returned collective `Work`;
- producer, side-effect, and collective-visible events;
- consuming-stream join;
- launched, joined, sealed, published, and aborted state transitions;
- exact transaction identity;
- resource retirement after success or failure.

It accepts opaque tensors and callbacks. It does not know model names, layer
types, request workloads, Qwen state layouts, or benchmark thresholds.

### 7.2 Benchmark worker

The worker owns:

- real first-adopter tensor shapes and dtypes;
- deterministic input initialization;
- three-arm execution order;
- CUDA event placement;
- host-submission measurement;
- correctness digests and rank agreement;
- lifecycle fault probes;
- raw row emission.

Benchmark-only instrumentation must not leak into the generic runtime API.

### 7.3 Classifier and verifiers

The assembler and both independent verifiers own:

- required row and artifact inventory;
- schema validation;
- source, attempt, environment, and GPU identity;
- recomputation of correctness and lifecycle results;
- recomputation of overlap, latency, host, and memory aggregates;
- precedence-ordered terminal classification;
- immutable manifest verification.

No producer-provided classification is trusted without reconstruction.

## 8. Formal Stage-0.1 protocol

### 8.1 Topology and admission

- one host;
- world size four;
- exactly four admitted CUDA GPUs;
- NVIDIA A100 80GB PCIe for comparability with r5;
- FP32 collective tensors;
- BF16 final outputs and side-effect state;
- each selected GPU at admission:
  - at most 1,024 MiB used memory;
  - at most 5% utilization;
  - no compute process;
- exact physical index, UUID, rank mapping, driver, CUDA, PyTorch, and NCCL
  versions recorded;
- no foreign process killed, paused, adopted, or modified.

All remote source, caches, logs, artifacts, temporary files, compiler output,
and task-local process records remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

Nothing task-owned may be written to remote `/`, `/tmp`, or another
root-filesystem path.

### 8.2 Shape matrix

The frozen active-token groups are:

```text
(1, 4, 8)
```

For every group:

- hidden width is 5,120;
- each rank contributes a deterministic FP32 local result;
- all arms produce the same BF16 output;
- side-effect payload is deterministic BF16 data;
- side-effect bytes equal `271,360 * active_tokens`;
- all destinations, streams, events, and reusable buffers are allocated before
  formal timing.

### 8.3 Diagnostic rows

Before formal timing, each rank runs 15 untimed diagnostic iterations for all
three arms and all three shapes. Across four ranks, this produces 180
rank-local diagnostic triplets with identical inputs inside each triplet.

The diagnostic contract requires:

- baseline exact on every row;
- completion-owned candidate exact on every row;
- event-only arm produces at least one exactness failure across the 180
  rank-local diagnostic triplets;
- the event-only mismatch is reported, never repaired in place;
- no diagnostic timing contributes to a promotion metric.

If baseline or completion-owned correctness fails, classification is
`NO_GO_CORRECTNESS_OR_LIFECYCLE`. If both are exact but event-only failure is
not reproduced, classification is `INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED`.

### 8.4 Repetition protocol

For each shape:

- two paired warmup repetitions;
- fifteen measured baseline/completion-owned pairs;
- deterministic balanced AB/BA ordering;
- identical initialized bytes within every pair;
- no threshold, workload, event placement, or policy tuning after any formal
  row becomes visible;
- CUDA-event device timing;
- separate monotonic-clock host-submission timing;
- no profiler in the formal run.

A profiler trace may be captured only after the evidence bundle is sealed. It
is diagnostic and cannot replace, amend, or reclassify formal measurements.

## 9. Measurement semantics

### 9.1 Critical-path intervals

The baseline critical interval starts before the baseline result copy and ends
after reduced-output materialization and side-effect copy.

The candidate critical interval starts at the same logical boundary and ends
after:

- `Work.wait()` has established the consuming-stream dependency;
- the side-effect event has been joined;
- the exact BF16 output has been materialized; and
- the candidate completion event has been recorded.

### 9.2 Collective outstanding window

Stage 0 incorrectly treated a caller communication-stream event as the NCCL
completion boundary. Stage 0.1 replaces that metric with:

```text
collective_outstanding_window =
    [producer_ready_event, collective_visible_event]
```

`collective_visible_event` is recorded on the consuming stream immediately
after `Work.wait()`. This interval includes submission and dependency
transfer; it is intentionally named an outstanding window rather than a pure
NCCL kernel interval.

The side-effect interval is:

```text
side_effect_window =
    [state_copy_started_event, state_copy_completed_event]
```

Formal useful concurrency is:

```text
overlap_intersection_ns =
    intersection(collective_outstanding_window, side_effect_window)

realized_overlap =
    overlap_intersection_ns
    / min(collective_outstanding_window_ns, side_effect_window_ns)
```

All events must share a valid timing origin and be synchronized only after the
timed pair has completed. The event-only arm's intervals are excluded.

### 9.3 Required reported metrics

For every shape report:

- baseline and candidate critical-path median, P90, P95, and P99;
- paired absolute time saved and paired speed ratio;
- collective outstanding-window duration;
- side-effect-copy duration;
- intersection and realized-overlap ratio;
- baseline and candidate host-submission median and P99;
- pair-direction agreement;
- peak allocated and peak reserved memory delta by rank;
- theoretical shadow bytes;
- post-warmup allocation count;
- all correctness, lifecycle, identity, timeout, and cleanup results.

Every claimed benefit must be shown beside its cost.

## 10. Correctness and lifecycle oracle

Every formal baseline and completion-owned row requires:

- a deterministic expected FP32 sum computed independently from the baseline
  and candidate execution paths;
- bitwise equality of both reduced FP32 outputs against that expected sum;
- bitwise equality of both BF16 final outputs against the expected BF16 cast;
- bitwise equality between the baseline and completion-owned outputs;
- byte-equal shadow payload;
- identical rank digests for reduced output, final output, and shadow payload;
- finite output;
- old active state unchanged before publication;
- exact active state after successful publication;
- exact old active state after abort;
- matching commit identity on all four ranks;
- no timed-out operation.

Lifecycle tests additionally cover:

- join before launch or against a non-active ticket;
- seal before join;
- publish before seal;
- duplicate launch on one resource bundle;
- stale or mismatched commit identity;
- collective failure after launch;
- side-effect failure after launch;
- cancellation before seal;
- double publish and double abort;
- request-slot or pooled-resource reuse before terminal ownership release;
- cleanup after every injected failure.

The verifier must distinguish:

- collective launched;
- `Work.wait()` invoked;
- collective completion dependency transferred to the consumer stream;
- side-effect completion joined;
- output consumed;
- transaction sealed;
- transaction published or aborted.

No event flag supplied by the producer may stand in for this ordered evidence
without matching raw lifecycle records.

## 11. Intended implementation scope

The subsequent implementation plan may change only the model-neutral Stage-0.1
surface:

- `tinyvllm/engine/collective_side_effect_overlap.py`;
- its focused runtime tests;
- the lease-sealed overlap benchmark worker and focused worker tests;
- the Stage-0.1 schema, assembler, independent verifier, controller, and their
  focused tests;
- the new Stage-0.1 audit, handoff checkpoint, and compact evidence manifest.

It may reuse existing Stage-0 infrastructure where behavior and artifact
identity remain explicit. It may not edit Qwen model code, generic linear
layers, unrelated optimization mechanisms, old attempt directories, or the
sealed r5 bundle.

## 12. Performance and resource gate

Performance is eligible only after all baseline and completion-owned
correctness, lifecycle, identity, and cleanup checks pass and the diagnostic
control behaves as required.

Return `GO_COMPLETION_OWNED_OVERLAP_MICROGATE` only if:

- active-token shapes 4 and 8 each achieve at least 20% median realized
  overlap;
- the geometric mean of their paired median critical-path improvements is at
  least 5%;
- neither shape 4 nor shape 8 regresses in median critical path;
- shape 1 median critical-path regression is at most 1%;
- no shape has more than 3% P99 critical-path regression;
- no shape has more than 3% median host-submission regression;
- shapes 4 and 8 each have at least 11 of 15 pairs agreeing with their
  aggregate improvement direction;
- no request-path allocation occurs after warmup;
- candidate peak reserved-memory increase is no more than theoretical shadow
  bytes plus 64 MiB per rank;
- all four ranks exit zero;
- cleanup is `CLEAN`;
- producer, remote independent verifier, and local streaming independent
  verifier reconstruct the same classification.

Missing, non-finite, malformed, contradictory, or unverifiable measurements
never pass.

## 13. Classifier precedence

The terminal classifier is ordered:

1. `NO_GO_CORRECTNESS_OR_LIFECYCLE`
2. `NO_GO_RESOURCE_IDENTITY`
3. `NO_GO_MEMORY_OR_ALLOCATION`
4. `INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT`
5. `INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED`
6. `NO_GO_INSUFFICIENT_OVERLAP`
7. `NO_GO_PERFORMANCE`
8. `GO_COMPLETION_OWNED_OVERLAP_MICROGATE`

Correctness has absolute precedence. No latency, overlap, memory, or host
metric is eligible after a correctness or lifecycle failure.

The event-only diagnostic arm can only:

- establish that the oracle discriminates the known unsafe ordering; or
- force `INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED`.

It can never cause or strengthen a GO.

## 14. Failure handling

| Failure | Required behavior |
|---|---|
| async collective launch failure | abort without consuming result or publishing shadow |
| `Work.wait()` failure | abort; preserve old active state; retire owned work safely |
| side-effect copy failure | abort; preserve old active state |
| output materialization failure | abort before seal |
| stale or mismatched identity | abort all participating ranks |
| missing or duplicate contribution | reject seal |
| timeout | terminal no-go; exact-tag-owned cleanup only |
| unsupported backend or `Work` contract | reject before formal timing |
| event-only arm unexpectedly exact | terminal inconclusive; no performance promotion |
| foreign GPU activity after admission | environment failure; do not touch foreign work |
| incomplete artifact or verifier disagreement | terminal inconclusive; no claim |

Fallback is allowed only before candidate work launches. Once launched, the
transaction must complete or abort through its owned resources.

## 15. Test strategy

Implementation follows RED, minimal implementation, then GREEN.

### 15.1 CPU contract tests

Add or update focused tests that initially fail because the current join does
not wait on `collective_work`:

- `join()` invokes `Work.wait()` exactly once;
- `Work.wait()` occurs while the consumer stream is current;
- the collective-visible event is recorded after `Work.wait()`;
- side-effect dependency is joined before output consumption;
- join-state transition occurs only after both dependencies are installed;
- abort waits for owned collective and side-effect work;
- failed wait cannot seal or publish;
- the event-only diagnostic path is structurally excluded from promotion;
- classifier precedence matches Section 13.

Mocks must record call order, not merely call presence.

### 15.2 CUDA and distributed tests

Before the formal campaign:

- two-rank smoke verifies completion-owned exactness;
- four-rank smoke verifies exact rank agreement;
- an untimed event-only negative reproduces at least one mismatch;
- injected collective and copy failures preserve old active state;
- no device-wide synchronization appears in the candidate success path;
- pooled events and buffers are not reused before ownership release;
- no post-warmup allocation occurs.

If the event-only mismatch is timing-sensitive and cannot be reproduced in the
frozen diagnostic matrix, the result remains inconclusive. The test must not
insert arbitrary sleeps or data corruption solely to manufacture a mismatch.

### 15.3 Static safety checks

The implementation scope must pass scans rejecting:

- `torch.cuda.synchronize()` in the timed candidate path;
- `.item()` in the timed candidate path;
- host event polling or Python busy loops;
- event-only completion authority in the promotion candidate;
- writes outside the approved remote mounted root;
- Qwen, model adapter, or `tinyvllm/layers/linear.py` changes.

## 16. Evidence and immutability

Each attempt uses a fresh tag and source revision. Required compact final
evidence includes:

```text
source_manifest.json
environment_manifest.json
gpu_rank_manifest.json
workload_manifest.json
admission.json
diagnostic_rows.jsonl
paired_rows.jsonl
lifecycle_rows.jsonl
memory_rows.jsonl
producer_result.json
remote_independent_verification.json
local_streaming_independent_verification.json
cleanup.json
report.md
manifest.json
manifest.sha256
```

The manifest binds every input, row, derived report, verifier, source
revision, source-tree hash, environment identity, GPU UUID, rank mapping,
attempt tag, and cleanup record.

Large raw files stay under the approved remote mounted root. Only the compact
sealed final bundle may be downloaded to the Mac.

A process ID, background shell, controller log, zero worker exit code,
producer classification, or complete manifest is not sufficient evidence by
itself.

## 17. Stage-1 stop rule

Qwen integration remains prohibited unless one fresh Stage-0.1 attempt:

- returns `GO_COMPLETION_OWNED_OVERLAP_MICROGATE`;
- has producer, remote verifier, and local verifier agreement;
- passes post-seal `--check-only` verification;
- has clean exact-tag-owned teardown; and
- is documented in a terminal audit committed from the same evidence.

Even a Stage-0.1 GO proves only that the model-neutral mechanism is correct,
measurably concurrent, and locally beneficial for the frozen real-shape
microgate. It does not prove Qwen3.8 end-to-end benefit.

After a Stage-0.1 GO, a separately reviewed Stage-1 plan may integrate the
default-disabled mechanism at the model adapter boundary and run the frozen
Qwen3.8-27B TP4 end-to-end gate.

Every other classification stops this mechanism. In particular:

- correctness failure stops immediately;
- exact but insufficient overlap stops;
- exact and overlapping but below-threshold performance stops;
- diagnostic non-reproduction stops pending a new root-cause design;
- incomplete evidence or verifier disagreement stops without a claim.

## 18. Claim boundary

Before Stage-0.1 execution, the only valid claim is:

> Stage 0 exposed an NCCL completion-ownership defect, and Stage 0.1 specifies
> a correctness-first experiment to test a `Work.wait()`-owned repair.

After execution:

- a GO permits planning a model integration experiment;
- a no-go records a complete negative mechanism result;
- an inconclusive result permits only diagnosis under a new reviewed design.

No Stage-0.1 outcome by itself supports a claim of Qwen3.8 QPS, TPOT, TTFT,
throughput, production, or cost improvement.
