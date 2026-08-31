# Cross-Request Wavefront Collective Overlap Design

**Date:** 2026-08-31

**Status:** Approved direction; written-spec review pending

**Stage-0 topology:** single host, four CUDA GPUs, tensor parallel size four

**Stage-1 first adopter:** `Qwen/Qwen3.8-27B`

**Model revision:** `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

## 1. Objective

Determine whether TinyLLMForge can reduce tensor-parallel decode latency and
increase online throughput by overlapping one request cohort's collective
with another cohort's independent computation.

The selected execution pattern is:

```text
cohort A local compute
  -> publish A-ready event
  -> ordered asynchronous AllReduce(A) on the communication stream

cohort B local compute overlaps AllReduce(A)
  -> publish B-ready event
  -> ordered asynchronous AllReduce(B) on the communication stream

cohort A post-reduction work overlaps AllReduce(B)
  -> cohort B post-reduction work
  -> restore original request order
```

This design does not replace NCCL, change tensor-parallel math, compress
collective payloads, or claim that asynchronous collectives and multi-stream
execution are novel. The TinyLLMForge-specific contribution is a
model-neutral, fail-closed runtime protocol that:

- identifies independent request cohorts;
- preserves globally identical collective order on all ranks;
- owns compute, communication, and completion events explicitly;
- measures realized device overlap rather than inferring it from host launch
  order;
- activates only when the expected net benefit is positive; and
- falls back to the unchanged synchronous path outside the qualified
  workload.

The first optimization target is online decode at concurrency four and eight.
Single-request decode remains on the synchronous baseline.

## 2. Existing evidence and why this is a new direction

Three earlier TP4 communication investigations are terminal and must not be
repeated or reinterpreted.

### 2.1 Communication-exposure profile

The Qwen3.8 TP4 profile observed material exposed communication, but the
matched Nsight runs added as much as `38.194519%` runtime overhead. The
terminal classification was:

```text
INCONCLUSIVE_LOW_HEADROOM
```

That result is useful for locating the boundary, but it does not authorize an
overlap implementation or establish attainable speedup.

### 2.2 Synchronous collective reduction

The corrected static opportunity found only one directly removable
steady-decode collective, the embedding-input AllReduce. Removing it through
replication would add about `1.776 GiB` of persistent device memory per rank.
No nonzero measurement budget passed the frozen overhead gate. The terminal
classification was:

```text
INCONCLUSIVE_PROFILER_OVERHEAD
```

### 2.3 Fixed-slot peer reduction

The custom CUDA IPC peer-reduction path preserved correctness and reduced host
submission time, but increased median CUDA transaction latency:

```text
active tokens 1:  +9.85%
active tokens 4: +29.98%
active tokens 8: +43.73%
```

The terminal classification was:

```text
NO_GO_MICROGATE
```

The current design therefore keeps NCCL and changes scheduling across
independent request cohorts. It does not revive peer-memory reduction,
projection replication, or the old 130-site collective catalog.

## 3. Success definition

The project succeeds only if a real four-GPU end-to-end gate proves both
benefit and cost under a frozen workload.

The final candidate must:

- preserve generated-token, output-length, and stop-reason equality;
- improve aggregate Q0/Q1/Q2 output throughput by at least `5%`;
- improve aggregate Q0/Q1/Q2 median TPOT by at least `5%`;
- keep every workload P99 E2E regression within `3%`;
- keep every workload TTFT regression within `3%`;
- add no more than `256 MiB` peak allocated memory per rank;
- use no host synchronization in the timed decode path;
- preserve deterministic collective order on every rank;
- prove nonzero device-side communication/compute overlap; and
- pass producer, independent verifier, manifest, and cleanup checks.

A mechanism microbenchmark, host submission reduction, or trace screenshot is
not an end-to-end success.

## 4. Considered approaches

### A. Cross-request two-cohort wavefront — selected

Split an admitted decode batch into two stable cohorts. Use independent
compute streams and one ordered communication stream. While cohort A's
AllReduce executes, launch cohort B's independent local computation. Once A
is reduced, continue A's dependent work while B communicates.

Benefits:

- attacks exposed communication without replacing NCCL;
- uses independence already present between requests;
- can preserve the baseline for unsupported batch sizes;
- does not require lossy communication; and
- has a bounded Stage-0 experiment before model integration.

Costs:

- benefits only workloads with at least two useful cohorts;
- smaller GEMMs may lose efficiency after batch splitting;
- NCCL may contend with GEMM for SM and memory-bandwidth resources;
- stream/event and tensor-lifetime ownership become part of correctness; and
- graph capture is out of scope until a separate replay gate.

### B. Reduced-precision collective — deferred

Cast local FP32 projection partials to BF16 or FP8 before reduction.

Potential benefit:

- reduces payload bytes.

Reasons not to select it first:

- the current one-token payload is small and may be launch-latency dominated;
- it changes numerical behavior;
- FP8 needs scale ownership and adds kernels or fused implementation work;
- it does not directly realize computation/communication overlap; and
- it would require a separate quality gate.

### C. TP2 replicas on four GPUs — control, not candidate

Run two TP2 replicas and route requests between them.

Potential benefit:

- fewer ranks per collective;
- higher aggregate throughput when the model fits TP2.

Reasons not to select it as the implementation:

- it changes serving topology and replication semantics;
- it may change KV capacity and admission behavior;
- it does not answer whether TP4 communication can overlap with computation;
- it is still valuable as a later horizontal control if the checkpoint fits.

## 5. Model-agnostic runtime verdict

### 5.1 Two-axis verdict

```text
mechanism:   reusable candidate
integration: first adopter only
```

The stream/event/cohort transaction can be model-neutral. Qwen3.8 is the
first adopter and performance profile, not a runtime semantic.

The mechanism becomes `generic` only after either:

- a second model or second tensor-parallel caller uses the same contract; or
- a synthetic second caller proves that no Qwen-specific metadata is required.

### 5.2 Layer map

| Layer | Responsibility |
| --- | --- |
| Mechanism | cohort identity, stream/event lifecycle, ordered collective transaction, completion, cancellation, poisoning, cleanup |
| Adapter | expose a splittable local-compute operation and its dependent continuation |
| Policy/config | minimum active requests, cohort split, enable flag, memory budget, fallback choice |
| Benchmark/profile | checkpoint, topology, hidden dimensions, workload matrix, repetitions, thresholds |

Core runtime code must not consume:

- `Qwen`;
- attention layer numbers;
- model repository names;
- prompt class names such as `Q0`;
- hidden size `5120` as a global constant; or
- workload-specific speed thresholds.

## 6. Runtime contracts

### 6.1 Cohort descriptor

The mechanism consumes a role-based immutable descriptor:

```python
@dataclass(frozen=True)
class CollectiveOverlapCohort:
    cohort_id: int
    request_indices: tuple[int, ...]
    active_token_count: int
```

Requirements:

- exactly two non-empty cohorts in the first implementation;
- each active request index appears exactly once;
- indices are strictly increasing within each cohort;
- the union equals the original active request set;
- all ranks receive the same descriptor digest; and
- output restoration uses indices, not completion order.

The default split is contiguous and balanced:

```text
cohort A = first ceil(N / 2) requests
cohort B = remaining requests
```

The split is frozen before timing and is not tuned per observed repetition.

### 6.2 Operation adapter

The generic executor receives callbacks rather than model semantics:

```python
class CollectiveOverlapOperation(Protocol):
    def launch_local(
        self,
        cohort: CollectiveOverlapCohort,
        *,
        stream,
        output,
    ) -> None: ...

    def launch_collective(
        self,
        cohort: CollectiveOverlapCohort,
        *,
        stream,
        tensor,
    ) -> object: ...

    def launch_dependent(
        self,
        cohort: CollectiveOverlapCohort,
        *,
        stream,
        reduced,
        output,
    ) -> None: ...
```

The adapter owns tensor meaning. The executor owns ordering, event
dependencies, completion, lifetime, and failure convergence.

### 6.3 Stream ownership

The first implementation owns:

```text
compute stream A
compute stream B
one communication stream
one completion event per phase and cohort
```

The communication stream is singular so every rank enqueues collectives in
the same total order:

```text
layer L cohort A
layer L cohort B
layer L+1 cohort A
layer L+1 cohort B
```

No rank may select order from local readiness. Readiness affects stream waits,
not collective sequence.

### 6.4 Device dependency protocol

For each cohort:

1. the caller records an input-ready event on its producing stream;
2. the cohort compute stream waits on input readiness;
3. local computation writes a cohort-owned output slice;
4. the compute stream records a local-ready event;
5. the communication stream waits on local readiness;
6. the communication stream enqueues the asynchronous collective;
7. a communication-complete event is recorded after the collective;
8. the cohort compute stream waits on communication completion;
9. dependent computation writes into the original full-batch indices; and
10. the caller's stream waits on both cohort-done events.

The timed path must not use:

- `torch.cuda.synchronize()`;
- device-wide synchronization;
- host polling;
- object collectives;
- request-path stream construction;
- request-path event construction; or
- dynamic output allocation.

PyTorch asynchronous collective work is not treated as complete merely
because the host enqueue returned. Completion is represented by the
communication stream's event dependency and verified device results.

### 6.5 Tensor lifetime

All tensors participating in another stream must remain alive until the final
consumer completes. The owner must attach stream usage through the framework's
stream-recording mechanism where applicable and retain:

- cohort input views;
- local partial buffers;
- reduced buffers;
- restored output views; and
- events and collective work handles

until both cohort-done events are visible to the caller stream.

Buffer reuse is generation tagged. A generation cannot reuse a slot while any
owned event from the previous generation is incomplete.

## 7. Activation and fallback policy

The candidate is default-disabled.

It is eligible only when:

- tensor parallel world size is four;
- all ranks are on one host;
- the backend supports CUDA asynchronous collectives;
- the active request count is at least four;
- two non-empty balanced cohorts are possible;
- the operation adapter declares cohort independence;
- CUDA Graph replay is disabled for this path;
- fixed buffers and events are ready;
- no prior transaction poisoned the executor; and
- all ranks agree on the policy and cohort digest.

Otherwise execution uses the unchanged synchronous baseline.

Single-request P0 and P1 workloads are protected baseline controls. They are
not routed through an empty or duplicated second cohort.

## 8. Failure and cancellation semantics

### 8.1 Pre-launch failure

Unsupported topology, invalid descriptors, allocation failure, policy drift,
or rank disagreement fails closed before candidate work begins. The current
request uses the baseline only if all ranks agree before entering the first
candidate collective.

### 8.2 In-flight failure

After the first candidate collective is enqueued, a rank cannot independently
fall back. Any local failure poisons the executor and aborts the distributed
request consistently.

The runtime must never allow:

```text
rank 0: cohort A candidate collective
rank 1: full-batch baseline collective
```

### 8.3 Cancellation

Cancellation stops future request admission but does not invalidate tensors
or events used by an in-flight collective. The transaction reaches one of:

```text
completed
failed_convergently
```

Only then may its buffers be reused.

### 8.4 Close

`close()` is idempotent. It:

- stops candidate admission;
- waits only for owned stream work;
- releases work handles;
- drops event and buffer ownership;
- destroys no external process group;
- restores no global CUDA state; and
- reports any live generation.

## 9. Stage 0: isolated real-shape qualification

Stage 0 answers whether the wavefront has a physical opportunity before any
model runtime integration.

### 9.1 Frozen shapes

Use the current Qwen3.8 attention-output transaction dimensions as a
benchmark profile:

```text
world size:          4
active tokens:       4 and 8
hidden size:         5120
rank-local input:    1536
input dtype:         bfloat16
local accumulation:  float32
collective dtype:    float32
output dtype:        bfloat16
```

The benchmark engine accepts dimensions from a profile and contains no model
name.

### 9.2 Arms

Baseline:

```text
one full-batch local GEMM
  -> blocking FP32 NCCL AllReduce
  -> BF16 cast
  -> residual add
  -> dependent pointwise/normalization surrogate
```

Candidate:

```text
cohort A local GEMM
  -> async AllReduce(A)
cohort B local GEMM overlaps AllReduce(A)
  -> async AllReduce(B)
cohort A dependent work overlaps AllReduce(B)
  -> cohort B dependent work
  -> restore full-batch order
```

The dependent surrogate must use the reduced values and must not be removable
by dead-code elimination.

### 9.3 Measurement

Use:

- two warmup pairs;
- at least 300 measured alternating AB/BA pairs per active-token count;
- four-rank maximum CUDA duration as transaction latency;
- host submission duration;
- P50, P90, P95, and P99;
- per-rank start and completion skew;
- collective duration;
- local-compute duration;
- dependent-compute duration;
- interval-union realized overlap;
- no-profiler paired wall-clock controls;
- peak allocated and reserved memory;
- numerical error, NaN, and Inf counts; and
- cleanup state.

CUDA events are preallocated. The primary timing path may not perform a
device-wide synchronization per internal operation. A bounded final
synchronization after the measured transaction is allowed to read results.

### 9.4 Stage-0 gate

Stage 0 returns `GO_WAVEFRONT_MICROGATE` only if:

- active tokens four median transaction latency improves by at least `5%`;
- active tokens eight median transaction latency improves by at least `8%`;
- neither shape's P99 regresses by more than `3%`;
- realized overlap is at least `20%` of candidate communication interval for
  both shapes;
- candidate host submission does not regress by more than `10%`;
- all ranks agree on outputs within `atol=2e-4, rtol=2e-4`;
- candidate versus baseline is within `atol=2e-2, rtol=2e-3`;
- no NaN or Inf appears;
- no collective timeout or rank-order mismatch occurs;
- additional peak allocated memory is at most `128 MiB` per rank in the
  isolated gate; and
- cleanup and the independent verifier pass.

If both median improvements are below `3%`, stop the complete direction.
Results between the stop threshold and GO threshold permit one bounded
scheduling refinement, not threshold retuning.

## 10. Stage 1: model-neutral executor integration

Stage 1 is authorized only by `GO_WAVEFRONT_MICROGATE`.

Create a reusable overlap executor under the engine runtime and a Qwen3.8
attention-output adapter. The first integration is limited to:

- eager decode;
- TP4;
- active request counts four and eight;
- greedy decoding;
- no speculative branch;
- no KV offload;
- no CUDA Graph candidate path; and
- the existing FP32 collective semantics.

The adapter exposes local attention-output projection and dependent
residual/norm continuation without placing layer numbers or model names in
the executor.

A synthetic second adapter must prove that the executor contract does not
depend on Qwen metadata.

## 11. Stage 2: paired end-to-end gate

Use the frozen workload matrix:

```text
P0: prompt=256,  output=128, concurrency=1
P1: prompt=2048, output=128, concurrency=1
Q0: prompt=256,  output=128, concurrency=4
Q1: prompt=256,  output=128, concurrency=8
Q2: prompt=2048, output=128, concurrency=4
```

Compare:

```text
baseline synchronous TP4
candidate cross-request wavefront TP4
```

Use identical checkpoint, source tree, GPUs, prompt rows, request order,
sampling, scheduler policy, memory capacity, and graph policy. Run two
warmups and at least seven alternating measured pairs per workload.

P0 and P1 must prove fallback identity and no material regression. Q0, Q1,
and Q2 determine performance promotion.

The terminal result is
`GO_CROSS_REQUEST_WAVEFRONT_OVERLAP` only if:

- every generated token, output length, and stop reason matches;
- aggregate Q0/Q1/Q2 output tokens per second improves by at least `5%`;
- aggregate Q0/Q1/Q2 median TPOT improves by at least `5%`;
- at least two of Q0/Q1/Q2 individually improve median TPOT by `5%`;
- no Q0/Q1/Q2 median TPOT regresses;
- no workload P99 E2E latency regresses by more than `3%`;
- no workload TTFT regresses by more than `3%`;
- P0/P1 candidate fallback invokes zero wavefront collectives;
- P0/P1 median TPOT regression is at most `1%`;
- measured device overlap is nonzero in every candidate online workload;
- candidate peak allocated memory delta is at most `256 MiB` per rank;
- no host synchronization exists in the timed path;
- collective order and cohort digest agree on all ranks;
- cleanup is complete; and
- producer and both independent verifiers agree.

## 12. Evidence bundle

The compact terminal bundle contains:

```text
source_identity.json
model_manifest.json
gpu_topology.json
runtime_capabilities.json
workload_manifest.json
cohort_policy.json
collective_order.jsonl
microgate_rows.jsonl
microgate_summary.json
paired_online_metrics.json
correctness.jsonl
overlap_intervals.jsonl
memory_summary.json
resource_samples.jsonl
cleanup.json
classification.json
independent_verification.json
manifest.sha256
```

Large profiler traces remain remote. The final result must be independently
reconstructable from compact raw intervals and paired measurements.

## 13. Classifications

```text
GO_WAVEFRONT_MICROGATE
GO_CROSS_REQUEST_WAVEFRONT_OVERLAP
NO_GO_INSUFFICIENT_OVERLAP
NO_GO_GEMM_FRAGMENTATION
NO_GO_PERFORMANCE
NO_GO_TAIL_OR_TTFT
NO_GO_CORRECTNESS
NO_GO_MEMORY
INELIGIBLE_TOPOLOGY
INCONCLUSIVE_RESOURCE_IDENTITY
INCONCLUSIVE_EVIDENCE
```

`NO_GO_GEMM_FRAGMENTATION` means splitting the batch reduced GEMM efficiency
enough to erase communication overlap. That is a terminal mechanism result,
not a reason to weaken cohort sizes or thresholds after seeing the data.

## 14. Remote execution constraints

All remote task files, build products, caches, logs, and temporary data must
remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

The controller must:

- inspect Kerberos lifetime without running `kinit` or `krenew`;
- admit exactly four strict-clean GPUs;
- perform a second strict-clean check immediately before launch;
- keep monitoring and launch authority in the local agent;
- never terminate, adopt, or clean an external GPU process;
- use a fresh immutable attempt tag;
- place `TMPDIR`, CUDA cache, extension cache, source, logs, and raw evidence
  under the approved remote root;
- never write task data to `/`, `/tmp`, a model-cache directory, or a retired
  checkout;
- preserve every failed or partial attempt; and
- download only compact evidence.

## 15. Test strategy

### 15.1 CPU contract tests

Test:

- balanced deterministic cohort construction;
- exact request-index coverage;
- duplicate and missing index rejection;
- identical cohort digest generation;
- activation and fallback policy;
- state transitions;
- poisoning and cancellation;
- idempotent close;
- classification boundaries; and
- model-neutral synthetic adapters.

### 15.2 CUDA single-process tests

Test:

- stream dependency construction;
- preallocated event reuse;
- output restoration;
- tensor lifetime recording;
- no request-path allocation after warmup; and
- baseline fallback.

### 15.3 Distributed four-GPU tests

Test:

- identical collective order on all ranks;
- deliberate rank-order mismatch fails before a collective deadlock;
- numerical parity;
- bounded completion;
- no host synchronization in timed code;
- cleanup and process-group survival; and
- real overlap interval reconstruction.

### 15.4 Real-model evidence

Only Stage 2 can establish a Qwen3.8 end-to-end performance result. Stage 0
proves a mechanism opportunity, and Stage 1 proves integration correctness.

## 16. Recommended contribution split

1. **Design and gate contract:** this document and the Stage-0 classifier.
2. **Generic core:** cohort descriptor, wavefront executor, lifecycle tests.
3. **First adopter:** Qwen3.8 attention-output adapter, default disabled.
4. **Second proof path:** synthetic independent operation adapter.
5. **Validation:** four-GPU microgate followed conditionally by model E2E.

No production default is authorized in the first implementation.

## 17. Claim boundary

A successful Stage-0 result supports only:

> On the frozen four-GPU real-shape transaction, cross-request wavefront
> scheduling realized communication/computation overlap and reduced the
> isolated transaction latency within the declared correctness and memory
> bounds.

A successful Stage-2 result may support:

> On the frozen Qwen3.8-27B BF16 TP4 online workload at concurrency four and
> eight, TinyLLMForge's default-disabled cross-request wavefront reduced TPOT
> and increased output throughput while preserving output behavior and the
> protected tail, TTFT, and memory bounds.

Neither result supports claims about:

- other checkpoints;
- TP2 or TP8;
- multi-node deployments;
- NVLink systems;
- stochastic sampling;
- speculative decoding;
- CUDA Graph replay;
- KV offload;
- maximum serving capacity; or
- production deployment.

## 18. External semantic references

- PyTorch distributed collectives:
  `https://docs.pytorch.org/docs/stable/distributed.html`
- NCCL collective ordering and CUDA stream behavior:
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html`
