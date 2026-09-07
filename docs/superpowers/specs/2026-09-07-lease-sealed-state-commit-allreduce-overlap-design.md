# Lease-Sealed State-Commit / AllReduce Overlap Design

**Date:** 2026-09-07
**Status:** Written design pending user review
**Source anchor:** `342bd8637453f010ccc782f6e7cfde19cae787a5`
**Target branch:** `feat/kv-sparse-attention`
**First adopter:** `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, tensor parallel size
four
**Default:** disabled
**Stage-1 authorization:** forbidden until the Stage-0 classifier is
`GO_LEASE_SEALED_OVERLAP_MICROGATE`

## 1. Decision

Implement a default-disabled runtime mechanism that overlaps an asynchronous
tensor-parallel AllReduce with an otherwise serial state-commit copy.

The first adopter is the Qwen3.8 linear-attention path. A linear-attention
layer already produces its candidate convolution and recurrent states before
the row-parallel output projection completes. The proposed path:

1. computes the local output projection;
2. launches the output AllReduce asynchronously on a communication stream;
3. copies the already-produced candidate state into an invisible shadow
   generation on a side-effect stream;
4. joins the collective and copy completion events before the layer result is
   consumed;
5. keeps the shadow generation invisible through the rest of the model step;
6. publishes the complete cross-layer shadow transaction only after model-step
   success and lease validation.

This changes scheduling and state ownership, not model math. It does not split
batches, replace NCCL, alter accumulation precision, or approximate attention.

The design is a reusable mechanism candidate with a first-adopter-only
integration verdict. It must not be described as a generic framework until a
second independent caller, or a model-neutral synthetic caller exercising the
same production contract, demonstrates reuse.

## 2. Motivation and evidence boundary

The current row-parallel attention output path performs a local projection
followed by a blocking `dist.all_reduce`. Candidate hybrid state has already
been computed at that point, but its copy into persistent state remains a
later serial transaction step.

The earlier Qwen3.8 TP4 communication profile established:

- complete four-rank alignment and exact correctness;
- exposed communication ratios between about `13.36%` and `38.82%` across the
  frozen workloads;
- overlap-headroom lower bounds of only about `10.42%` to `10.77%`;
- profiler overhead of `38.194519%`.

That profile classified the opportunity as `INCONCLUSIVE_LOW_HEADROOM`. Its
trace timings are useful for locating the blocking boundary, but they are not
steady-state speedup evidence.

Other completed experiments close two tempting alternatives:

- fixed-slot CUDA IPC peer reduction is `NO_GO_MICROGATE` because device
  latency regressed despite lower host-submission time;
- split-cohort cross-request wavefront is
  `NO_GO_INSUFFICIENT_OVERLAP` because it damaged GEMM efficiency and increased
  host submission.

This design therefore preserves the full batch and the existing NCCL
collective. It seeks independent work already present in the same layer:
copying candidate state into a transactionally invisible destination.

No prior profile, microbenchmark, segmented-capture result, or theoretical
headroom number establishes an end-to-end benefit for this mechanism.

## 3. Goals

The design has five goals:

1. Preserve exact greedy output and hybrid-state semantics.
2. Hide part of the output AllReduce behind useful state-commit work without
   fragmenting the batch.
3. Improve online request QPS and P99 end-to-end latency on the frozen
   Qwen3.8-27B TP4 workload.
4. Bound the added memory, stream, event, and transaction costs.
5. Produce source-bound, independently verifiable evidence reporting both
   benefit and cost.

## 4. Non-goals

This design does not:

- replace NCCL with a custom reduction kernel;
- split one batch into communication and compute cohorts;
- overlap an AllReduce with compute that consumes its result;
- change FP32 accumulation, BF16 output, greedy sampling, or token semantics;
- modify the 16 full-attention layers in the first adopter;
- make state visible before complete model-step success;
- introduce request-path host synchronization through `.item()` or
  `cuda.synchronize()`;
- remove the embedding AllReduce;
- claim production benefit from Stage 0;
- revive old run tags or retroactively reclassify existing evidence;
- make model-specific state names, layer counts, workload names, or checkpoint
  identities part of the generic runtime API.

## 5. Architectural boundary

The design separates four ownership domains.

### 5.1 Generic overlap core

The generic core understands only these roles:

- `local_result`: a tensor requiring a collective before consumer use;
- `side_effect_payload`: data that can be materialized independently while the
  collective is active;
- `consumer_ready_event`: completion of the collective result;
- `side_effect_ready_event`: completion of side-effect materialization;
- `commit_identity`: an opaque identity used to validate publication.

The core owns:

- communication-stream launch;
- side-effect-stream launch;
- event creation, recording, waiting, and reuse;
- joining both branches without a device-wide synchronization;
- lifecycle state transitions;
- abort propagation;
- observability counters.

It does not understand convolution state, recurrent state, Qwen layer
placement, request IDs, prompt classes, or benchmark thresholds.

### 5.2 Model adapter

The Qwen adapter owns:

- identifying eligible linear-attention layers;
- packaging convolution and recurrent candidate state as the side-effect
  payload;
- mapping request slots into shadow-state destinations;
- constructing and validating the model-specific commit identity;
- leaving full-attention layers on the synchronous path;
- exposing the final cross-layer publish or abort action.

### 5.3 Policy

The policy owns whether the mechanism is enabled. The initial policy is
deliberately narrow:

- disabled by default;
- enabled only for the explicitly qualified Qwen3.8 BF16 TP4 configuration;
- enabled only for eligible linear-attention output projections;
- no dynamic threshold selected from measured repetitions;
- any unsupported topology or state layout falls back to the existing
  synchronous path before timed execution.

### 5.4 Benchmark profile

The benchmark profile owns model revision, topology, tensor shapes, workload
names, repetition counts, thresholds, admission rules, and terminal
classifications. None of these values belong in the generic mechanism.

## 6. State and stream ownership

### 6.1 Existing serial path

The effective serial dependency is:

```text
candidate state produced
        |
local output projection
        |
blocking AllReduce
        |
residual / norm / MLP
        |
cross-layer state commit copy
        |
publish
```

The later state copy does useful work, but it cannot currently hide collective
latency.

### 6.2 Candidate path

The candidate dependency is:

```text
candidate state produced ---------> side-effect stream: copy to shadow
        |                                      |
local output projection                         +--> side_effect_ready_event
        |
communication stream: async AllReduce
        |
consumer_ready_event
        |
join consumer_ready_event + side_effect_ready_event
        |
residual / norm / MLP
        |
complete model step
        |
validate lease + generation + rank identity
        |
publish all shadow state, or abort all
```

The critical interval changes from approximately:

```text
AllReduce duration + state-copy duration
```

to:

```text
max(AllReduce duration, state-copy duration) + event/launch overhead
```

The optimization is useful only if the saved overlap exceeds stream, event,
shadow-memory, and publication overhead.

### 6.3 Tensor lifetime

The producer stream retains the local projection tensor until the
communication stream has accepted its dependency. The reduced result remains
owned by the overlap operation until `consumer_ready_event`.

The candidate-state payload remains alive until `side_effect_ready_event`.
The shadow destination remains owned by the transaction until publish or
abort. Neither allocator reuse nor request-slot reuse may occur before the
owning event and transaction reach a terminal state.

### 6.4 Event semantics

The timed path may use stream-local events and `wait_event`. It may not use:

- device-wide `cuda.synchronize()`;
- host polling of a CUDA event;
- `.item()` for readiness or identity decisions;
- a Python busy loop;
- implicit default-stream synchronization as the correctness mechanism.

The join must be explicit and visible to instrumentation.

## 7. Lease-sealed transaction

### 7.1 Commit identity

`commit_identity` is opaque to the generic core. The first adopter binds it to:

- request identity;
- pool slot;
- lease generation;
- model-step epoch;
- participating layer set;
- tensor-parallel world and rank mapping.

Every rank must agree on the logical commit identity before publication.

### 7.2 Prepare

Prepare reserves an inactive shadow generation and records the old active
generation. It must not mutate the active state. Reservation failure falls
back or aborts before the overlap operation launches.

### 7.3 Seal

A transaction becomes sealed only when:

- every eligible layer has completed its side-effect copy;
- every corresponding collective has completed;
- the model step has produced a valid result;
- the lease and generation still match;
- all ranks agree on success and commit identity.

Seal does not itself expose partially written state.

### 7.4 Publish

Publication switches the request's active-generation identity only after the
entire cross-layer transaction is sealed. Publication must be all-or-nothing
from the request's next-step perspective.

No layer may publish independently.

### 7.5 Abort

Any failure before publication:

- marks the transaction aborted;
- preserves the old active generation;
- prevents shadow reads;
- waits only for work owned by the failed transaction;
- releases or recycles shadow storage after its final owning event;
- emits a reasoned terminal record.

An abort may not kill, reset, or adopt unrelated GPU work.

## 8. First-adopter integration

Qwen3.8-27B has 64 decoder layers:

- 48 linear-attention layers;
- 16 full-attention layers.

Only the 48 linear-attention layers are eligible initially. Their candidate
convolution and recurrent states exist before the attention output projection.
Their row-parallel output AllReduce is the communication branch.

The 16 full-attention layers retain the existing synchronous behavior. The
embedding collective and final greedy-token broadcast are unchanged.

The integration must be opt-in at the Qwen adapter boundary. It must not change
the semantics of every `RowParallelLinear` caller merely because the generic
linear layer gains an asynchronous capability.

The existing `prepare_step()` and `commit_prepared_step()` split remains the
outer model-step transaction boundary. The overlap mechanism changes where the
candidate state is copied, not when the model state becomes visible.

## 9. Memory model

For the first adopter:

- key heads: 16;
- value heads: 16;
- key dimension: 128;
- value dimension: 128;
- convolution kernel: 4;
- tensor parallel size: 4;
- state dtype: BF16.

Per request, per linear-attention layer:

- convolution state: `9,216 bytes`;
- recurrent state: `262,144 bytes`;
- total: `271,360 bytes`.

Across 48 eligible layers:

```text
271,360 * 48 = 13,025,280 bytes
```

This is approximately `12.42 MiB/request/rank`.

At active batch eight:

```text
13,025,280 * 8 = 104,202,240 bytes
```

This is approximately `99.38 MiB/rank`.

These are layout estimates, not allocator evidence. The formal gate measures
both peak allocated and peak reserved memory. Stage 1 caps the paired
candidate-minus-baseline peak reserved increase at `160 MiB/rank`.

Shadow buffers and reusable CUDA events must be preallocated or pooled before
timed execution. No per-token or per-request allocation is allowed after
warmup.

## 10. Failure handling

The implementation must define and test these failures:

| Failure | Required behavior |
|---|---|
| stale lease or generation | abort; preserve old active state |
| rank commit-identity mismatch | abort all participating ranks |
| collective launch failure | do not publish; retire owned shadow safely |
| collective completion failure | do not consume result or publish shadow |
| side-effect copy failure | do not publish; preserve old active state |
| missing eligible-layer contribution | reject seal |
| duplicate layer contribution | reject transaction |
| request cancellation before seal | abort owned transaction |
| request-slot reuse attempt | deny reuse until terminal event ownership clears |
| unsupported topology or layout | deterministic synchronous fallback before timing |
| process failure | terminal failure record; exact-tag-owned cleanup only |

Fallback is valid only before candidate work is launched. Once either branch
has launched, the operation must complete or abort through its owned
transaction; it may not silently rerun the same step on the synchronous path.

## 11. Stage 0: isolated real-shape mechanism gate

Stage 0 determines whether the mechanism creates useful physical overlap
before model integration.

### 11.1 Topology and admission

- one host;
- four CUDA GPUs;
- tensor parallel size four;
- FP32 NCCL AllReduce followed by the unchanged BF16 output cast;
- all four GPUs pass `strict_clean` admission:
  - memory usage at most `1,024 MiB`;
  - utilization at most `5%`;
  - no compute process;
- exact GPU UUID and rank mapping frozen in the attempt manifest;
- all remote source, cache, logs, artifacts, and temporary task data remain
  below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.

No foreign process may be killed, paused, adopted, or cleaned.

### 11.2 Shape matrix

The benchmark profile uses active-token counts `1`, `4`, and `8`.

For each shape:

- `local_result` uses the first adopter's real hidden width and FP32
  accumulation dtype;
- the final output cast uses BF16;
- side-effect bytes equal `271,360 * active_tokens` in BF16;
- destinations are preallocated shadow storage;
- the same deterministic input bytes feed baseline and candidate.

These values belong to the benchmark profile, not the generic API.

### 11.3 Paired variants

Baseline:

```text
local projection result
-> synchronous AllReduce
-> state copy
-> completion
```

Candidate:

```text
local projection result
-> asynchronous AllReduce
|| state copy to shadow
-> event join
-> completion
```

Both variants execute the same FP32 collective, the same BF16 output cast, and
copy the same BF16 state bytes.

### 11.4 Repetition protocol

For every shape:

- two paired warmups;
- fifteen measured baseline/candidate pairs;
- balanced deterministic AB/BA ordering;
- no tuning after any measured result is visible;
- enough transactions per repetition to exceed timer resolution;
- CUDA-event timing around device intervals;
- host submission time measured separately;
- no Nsight Systems in formal timing.

A separate diagnostic trace may be captured only after the formal rows are
sealed. Its timing cannot replace or reclassify the formal rows.

### 11.5 Correctness oracle

Stage 0 requires:

- bitwise-equal FP32 reduced output and BF16 final output;
- byte-equal shadow payload;
- old active state unchanged before publish;
- exact active state after successful publish;
- exact old state after injected abort;
- no consumer access before `consumer_ready_event`;
- no publish before `side_effect_ready_event`;
- matching commit identity on all four ranks;
- finite outputs;
- no leaked transaction, event, stream-owned tensor, or NCCL work.

Lifecycle fault tests cover stale generation, rank mismatch, missing
contribution, duplicate contribution, collective failure, copy failure, and
cancellation.

### 11.6 Metrics

Report for every shape:

- serial baseline device critical interval;
- candidate device critical interval;
- AllReduce interval;
- state-copy interval;
- intersection of the AllReduce and state-copy intervals;
- realized overlap:

```text
intersection_ns / min(allreduce_ns, state_copy_ns)
```

- median, P90, P95, and P99;
- paired speed ratio and absolute time saved;
- host-submission time;
- peak allocated and reserved memory;
- post-warmup allocation count;
- correctness and lifecycle outcomes.

### 11.7 Stage-0 GO gate

Return `GO_LEASE_SEALED_OVERLAP_MICROGATE` only if all conditions hold:

- all correctness and lifecycle checks pass;
- active-token shapes four and eight each have at least `20%` median realized
  overlap;
- the geometric mean of their paired median critical-path improvement is at
  least `5%`;
- neither shape's median critical path regresses;
- active-token shape one has at most `1%` median regression;
- no shape has more than `3%` P99 regression;
- host-submission time does not regress by more than `3%`;
- at least eleven of fifteen pairs agree with the aggregate improvement
  direction for shapes four and eight;
- no request-path allocation occurs after warmup;
- candidate peak reserved-memory increase is no more than theoretical shadow
  bytes plus `64 MiB/rank`;
- resource identity, manifest, cleanup, producer, remote independent verifier,
  and local streaming independent verifier all pass and reconstruct the same
  classification.

Any missing measurement is not a pass.

### 11.8 Stage-0 terminal classifications

The classifier uses this precedence:

1. `NO_GO_CORRECTNESS_OR_LIFECYCLE`
2. `NO_GO_RESOURCE_IDENTITY`
3. `NO_GO_MEMORY_OR_ALLOCATION`
4. `INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT`
5. `NO_GO_INSUFFICIENT_OVERLAP`
6. `NO_GO_PERFORMANCE`
7. `GO_LEASE_SEALED_OVERLAP_MICROGATE`

Stage 1 is prohibited for every result except the final GO.

## 12. Stage 1: Qwen3.8-27B TP4 end-to-end gate

Stage 1 is the first point that can establish a model-level performance
result.

### 12.1 Frozen model identity

- model: `Qwen/Qwen3.8-27B`;
- revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- dtype: BF16;
- tensor parallel size: four;
- decoding: greedy;
- baseline and candidate use the same committed source revision;
- the candidate differs only by the frozen, default-disabled feature flag.

### 12.2 Frozen workloads

| Workload | Prompt tokens | Output tokens | Concurrency | Role |
|---|---:|---:|---:|---|
| P0 | 256 | 128 | 1 | single-request protection |
| P1 | 2,048 | 128 | 1 | long-prompt protection |
| Q0 | 256 | 128 | 4 | online promotion |
| Q1 | 256 | 128 | 8 | online promotion |
| Q2 | 2,048 | 128 | 4 | mixed prefill/decode promotion |

Each workload has:

- two paired warmups;
- seven measured baseline/candidate pairs;
- balanced deterministic AB/BA ordering;
- identical prompt and request-arrival identities within a pair;
- no threshold or policy tuning after measured rows are visible.

### 12.3 Correctness

A separate untimed correctness pass contains twenty deterministic requests per
workload, for one hundred rows total.

Every row requires:

- exact generated-token sequence;
- exact stop position and reason;
- finite logits;
- exact argmax;
- zero maximum absolute and relative logit error;
- exact post-step convolution and recurrent state;
- matching active generation and commit identity across ranks.

The performance run also records exact token and lifecycle checks. Passing the
separate correctness pass cannot excuse a timed-run mismatch.

### 12.4 Benefit metrics

Primary benefit metrics:

- request QPS;
- output tokens per second;
- request-level P99 end-to-end latency.

Secondary metrics:

- TPOT P50/P95/P99;
- TTFT P50/P95/P99;
- queueing time;
- per-layer overlap interval;
- collective wait exposed to the consumer;
- host-submission time.

For Q0, Q1, and Q2, aggregate throughput uses the geometric mean of paired
candidate-to-baseline ratios. Aggregate latency uses the geometric mean of
paired baseline-to-candidate ratios. A reported percentage must include its
absolute baseline and candidate values.

### 12.5 Cost metrics

Every result reports:

- peak allocated memory by rank;
- peak reserved memory by rank;
- shadow bytes by active request count;
- event and stream pool size;
- host-submission overhead;
- abort/fallback count and reason;
- post-warmup allocation count;
- cleanup duration;
- implementation coverage: eligible layers, synchronous layers, and fallback
  invocations.

### 12.6 Stage-1 GO gate

Return `GO_LEASE_SEALED_STATE_COMMIT_OVERLAP` only if all conditions hold:

- all correctness and lifecycle checks pass;
- Q0/Q1/Q2 aggregate request QPS improves by at least `5%`;
- at least two of Q0/Q1/Q2 individually improve request QPS by at least `3%`;
- none of Q0/Q1/Q2 regresses request QPS by more than `1%`;
- Q0/Q1/Q2 aggregate request-level P99 end-to-end latency improves by at least
  `3%`;
- at least two of Q0/Q1/Q2 individually improve P99 end-to-end latency;
- no workload's P99 end-to-end latency regresses by more than `3%`;
- no workload's TTFT regresses by more than `3%`;
- P0 and P1 median TPOT regression is at most `1%`;
- each candidate online workload has nonzero measured AllReduce/state-copy
  overlap;
- at least five of seven measured pairs agree with the aggregate QPS direction
  for each promoted online workload;
- candidate peak reserved-memory increase is at most `160 MiB/rank`;
- no allocation occurs on the request path after warmup;
- there is no leaked shadow generation, CUDA event, stream-owned tensor, NCCL
  work, or owned process;
- strict-clean admission, source identity, model identity, rank identity,
  manifest, and cleanup pass;
- the producer, remote independent verifier, and local streaming independent
  verifier reconstruct the same terminal classification.

No positive metric offsets a failed correctness, lifecycle, resource,
tail-latency, memory, or evidence-chain gate.

### 12.7 Stage-1 terminal classifications

The classifier uses this precedence:

1. `NO_GO_CORRECTNESS_OR_LIFECYCLE`
2. `NO_GO_RESOURCE_IDENTITY`
3. `NO_GO_MEMORY_OR_ALLOCATION`
4. `INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT`
5. `NO_GO_TAIL_OR_TTFT`
6. `NO_GO_PERFORMANCE`
7. `GO_LEASE_SEALED_STATE_COMMIT_OVERLAP`

## 13. Evidence package

Every stage uses a fresh immutable tag and evidence directory.

The final bundle must contain at least:

```text
source_manifest.json
model_manifest.json
environment_manifest.json
gpu_rank_manifest.json
workload_manifest.json
admission.json
paired_rows.jsonl
correctness_rows.jsonl
lifecycle_rows.jsonl
memory_rows.jsonl
overlap_rows.jsonl
producer_result.json
remote_independent_verification.json
local_streaming_independent_verification.json
cleanup.json
report.md
manifest.json
manifest.sha256
```

Stage 0 may omit `model_manifest.json` only if its synthetic profile records the
complete first-adopter shape identity in `workload_manifest.json`. Stage 1 may
not omit it.

The manifest binds:

- source commit and source-tree hash;
- dirty-patch hash when applicable;
- model and tokenizer revision;
- environment and dependency identity;
- GPU UUID, driver, CUDA, NCCL, world size, and rank mapping;
- exact prompts, arrival schedule, seeds, and workload parameters;
- baseline/candidate flag identity;
- every raw row and derived report;
- verifier source identity;
- cleanup result.

The local streaming verifier consumes the bounded final evidence package, not
model weights, caches, large traces, or scratch tensors. Large experiment data
stays on the remote mounted storage.

Worker exit zero, a complete manifest, a passing unit suite, or a producer GO
is insufficient by itself.

## 14. Verification layers

### 14.1 Focused CPU tests

Before GPU execution, tests cover:

- lifecycle state machine;
- commit-identity comparison;
- exact layer-set accounting;
- stale lease rejection;
- duplicate/missing contribution rejection;
- publish/abort atomicity;
- fallback eligibility;
- classifier precedence;
- manifest reconstruction.

### 14.2 CUDA unit and fault tests

Tests cover:

- stream/event dependency without device-wide synchronization;
- destination lifetime through event completion;
- asynchronous collective completion;
- copy failure and collective failure;
- cancellation and slot-reuse races;
- event-pool reuse;
- no post-warmup allocation.

### 14.3 Stage 0

Stage 0 proves only mechanism correctness, lifecycle safety, real-shape
physical overlap, and bounded local cost.

### 14.4 Stage 1

Stage 1 proves first-adopter integration and, only if all frozen gates pass,
end-to-end model-level benefit.

## 15. Observability

The candidate path records bounded counters without host synchronization:

- eligible operation count;
- launched overlap count;
- synchronous fallback count by reason;
- committed transaction count;
- aborted transaction count by reason;
- stale-identity rejection count;
- outstanding and high-water shadow generations;
- event-pool high-water mark;
- post-warmup allocation count.

Per-operation diagnostic timestamps may be enabled only in non-formal
diagnostic runs unless their overhead passes a separately frozen ceiling.

## 16. Rollout

1. Implement the generic primitive, lifecycle state machine, and Stage-0
   harness without Qwen model integration.
2. Run CPU and CUDA fault tests.
3. Run a fresh strict-clean Stage-0 campaign.
4. Stop permanently if Stage 0 is not
   `GO_LEASE_SEALED_OVERLAP_MICROGATE`.
5. If Stage 0 passes, implement the Qwen adapter and default-disabled policy.
6. Run focused and adjacent regression tests.
7. Run a fresh strict-clean Stage-1 campaign.
8. Publish benefit and cost together with producer and both independent
   verifiers.

The implementation plan may refine file-level sequencing, but it may not lower
or reinterpret these gates.

## 17. Claim boundary

A Stage-0 GO supports only:

> A four-GPU real-shape mechanism gate demonstrated that lease-sealed shadow
> state copies can overlap NCCL AllReduce with exact output and bounded memory
> and lifecycle cost.

It does not support a Qwen3.8 performance claim.

A verified Stage-1 GO may support:

> On the frozen Qwen3.8-27B BF16 TP4 online workload, TinyLLMForge's
> default-disabled lease-sealed state-commit path improved request QPS and P99
> end-to-end latency while preserving exact greedy output, transactional state
> semantics, TTFT, and the declared memory bound.

The claim must include the measured baseline/candidate values and cannot be
generalized to another model, topology, dtype, scheduler, or workload without
new evidence.

A NO_GO remains useful evidence about why the overlap did not survive launch,
memory-bandwidth, event, transaction, or end-to-end costs. It must not be
described as an optimization win.

## 18. Rejected alternatives

### 18.1 Custom peer reduction

Rejected because completed measurements showed device-latency regressions even
when host submission improved.

### 18.2 Split-cohort wavefront

Rejected because splitting the batch reduced GEMM efficiency and failed the
minimum realized-overlap gate.

### 18.3 Embedding replication

Rejected because about `1.776 GiB/rank` of persistent and peak memory would
remove only one of the corrected decode graph's 66 synchronous boundaries.

### 18.4 Immediate full-model integration

Rejected because it would mix primitive viability, transaction correctness,
and model-level performance into one expensive experiment. Stage 0 provides a
bounded stop condition.

### 18.5 Early per-layer publication

Rejected because later model-step failure would leave hybrid state partially
advanced and violate existing transaction semantics.

## 19. Design acceptance checklist

| Requirement | Design evidence |
|---|---|
| Preserve full-batch GEMM | no cohort split; same local projection |
| Preserve NCCL and exact math | same FP32 AllReduce, BF16 cast, and greedy path |
| Create real overlap | async collective concurrent with shadow copy |
| Preserve state atomicity | invisible generation plus cross-layer seal |
| Protect stale requests | lease/generation identity at seal and publish |
| Keep core model-agnostic | role-based generic contract; Qwen adapter owns state meaning |
| Bound memory cost | theoretical accounting plus Stage-0 and Stage-1 caps |
| Prevent premature E2E claims | Stage 1 forbidden without Stage-0 GO |
| Measure benefit | QPS, output throughput, P99, TPOT, TTFT |
| Measure cost | memory, host overhead, allocations, aborts, cleanup |
| Preserve reproducibility | immutable tag, identities, raw rows, manifest hash |
| Require independent evidence | producer plus remote and local verifiers |
| Protect shared hardware | strict-clean admission and owned cleanup only |
| Keep large data remote | mounted `/data00/home/sitian` workspace only |

This checklist defines the design contract. Passing an implementation test
that does not cover the corresponding row does not satisfy that row.
