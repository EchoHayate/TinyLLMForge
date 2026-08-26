# Qwen3.8-27B TP4 Communication-Exposure Profile Design

**Date:** 2026-08-26

**Authoritative checkout:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

## Goal

Establish whether tensor-parallel communication is materially exposed on the
critical path of Qwen3.8-27B inference before authorizing any
communication/computation-overlap implementation.

The gate must:

1. run the official `Qwen/Qwen3.8-27B` checkpoint on four strictly clean GPUs;
2. establish a source-bound TP4 correctness baseline;
3. measure per-layer GEMM, NCCL collective, and GPU idle intervals;
4. compute how much NCCL time is actually exposed on the critical path rather
   than summing overlapping durations;
5. classify the opportunity as `GO`, `NO_GO`, or `INCONCLUSIVE`;
6. leave the current synchronous collective path unchanged unless the result
   is `GO_COMMUNICATION_OVERLAP`.

This design does not pre-authorize a production overlap implementation. A
`GO_COMMUNICATION_OVERLAP` result authorizes a separate implementation design
whose candidate must be compared against the frozen baseline produced here.

## Current State

TinyLLMForge already has:

- a native Qwen3.5 hybrid text runtime;
- TP4 checkpoint sharding and real-checkpoint correctness gates;
- row-parallel `all_reduce` and replicated-weight `all_gather` call sites;
- process-local decode-step and collective profiling through CUDA Events;
- NVTX annotations and an Nsight Systems remote profiling path;
- strict source, workload, rank, process, and artifact identity validation.

The current Qwen3.8-27B checkpoint advertises:

```text
repository: Qwen/Qwen3.8-27B
top-level model_type: qwen3_5
architecture: Qwen3_5ForConditionalGeneration
text model_type: qwen3_5_text
text hidden size: 5120
text intermediate size: 17408
text layers: 64
layer cadence: three linear-attention layers followed by one full-attention
full-attention interval: 4
text dtype: bfloat16
```

Sharing `model_type=qwen3_5` is compatibility evidence, not compatibility
proof. The 27B checkpoint has a materially different shape and a multimodal
top-level wrapper. The profiling campaign therefore requires an explicit
text-only adopter and a fresh TP1/TP4 correctness authority before any
performance result is accepted.

The completed elastic exact-burst work is outside this gate. It is TP1,
batch-one host/device synchronization amortization and must not be described
as TP4 communication overlap.

## Considered Approaches

### A. Profile Qwen3.8-27B in an external serving framework

This is the shortest route to a hardware timeline, but it would measure that
framework's model adapter, scheduler, collective implementation, and kernels.
It cannot establish a TinyLLMForge bottleneck or authorize a TinyLLMForge
runtime change.

Rejected as the primary gate. External-framework traces may be retained only
as non-comparable reference material.

### B. Reuse the existing Qwen3.5 runtime through a validated Qwen3.8 adopter

The adapter unwraps the official checkpoint's `text_config`, validates every
shape and layer type needed by the existing hybrid runtime, maps checkpoint
tensors through the established TP sharding contracts, and keeps the
profiling mechanism model-agnostic.

This is selected because it separates four questions:

1. can TinyLLMForge execute the checkpoint correctly;
2. where time is spent in the unchanged synchronous baseline;
3. how much communication is exposed;
4. whether a later overlap mechanism improves end-to-end behavior.

### C. Build the profiler on Qwen3.5-2B and infer the 27B result

The existing 2B model is useful for CPU tests and remote smoke tests, but its
GEMM-to-communication ratio is not representative of a 27B model. A profiler
that works on 2B is implementation evidence only.

Rejected for the terminal performance classification.

## Two-Axis Runtime Verdict

- **Mechanism:** `reusable candidate`
- **Integration:** `first adopter only`

Per-layer interval capture, timeline validation, and exposed-communication
classification are generic TP runtime capabilities. Qwen3.8 checkpoint
translation and workload selection are first-adopter concerns. The mechanism
must consume layer roles, operation classes, ranks, streams, and timestamps;
it must not branch on `qwen3_8`, prompt text, or checkpoint-specific layer
numbers.

## Layer Map

### Mechanism

- process-local profile activation and finalization;
- layer and operation range lifecycle;
- CUDA Event ownership;
- NVTX range emission;
- rank/step/layer alignment;
- interval-union and overlap calculations;
- critical-rank selection;
- fail-closed classification.

### Adapter

- top-level `Qwen3_5ForConditionalGeneration` to text-only model translation;
- `text_config` extraction;
- checkpoint tensor-name translation;
- vision-component exclusion for text-only inference;
- TP4 shard construction through existing linear and attention contracts.

### Policy and Configuration

- profile enable flag;
- allowed operation classes;
- representative Nsight repetition;
- strict GPU admission thresholds;
- materiality thresholds;
- remote timeouts and retry limits.

### Benchmark Profile

- checkpoint repository and resolved immutable revision;
- text-only prompt corpus;
- context and output lengths;
- concurrency;
- warmup and measured repetitions;
- four selected GPU UUIDs;
- dtype and sampling configuration.

## Phase 0: Source-Bound Qwen3.8-27B TP4 Baseline

### Checkpoint Identity

The acquisition receipt must bind:

- repository exactly `Qwen/Qwen3.8-27B`;
- resolved immutable Hugging Face revision SHA;
- every downloaded file name, byte size, and SHA-256;
- top-level and text configuration SHA-256;
- tokenizer file inventory and hashes;
- source-tree inventory and SHA-256;
- Transformers, PyTorch, CUDA, driver, NCCL, and GPU identities.

Floating revisions such as `main` are not valid execution identities.

### Text-Only Adopter Boundary

The first adopter executes text-only prompts. It must:

1. read topology from `text_config`;
2. reject image or video tokens;
3. avoid constructing or loading the vision encoder;
4. preserve the official vocabulary, embeddings, output head, layer cadence,
   RMSNorm, rotary, DeltaNet, and full-attention semantics;
5. use existing generic TP linear and collective primitives;
6. keep checkpoint-specific names and transforms outside scheduler and generic
   profiler code.

The adapter may reuse Qwen3.5 implementation units only when shape,
parameter, and semantic validation succeeds. Unsupported fields fail before
GPU mutation.

### Correctness Authority

The baseline must pass all of the following:

- one TP1 official-reference run;
- one TinyLLMForge TP1 run;
- one TinyLLMForge TP4 run with four real ranks;
- exact prompt-token identity;
- exact generated-token identity under greedy decoding;
- exact argmax identity for every checked decode position;
- finite logits on every rank;
- top-k token identity and explicitly recorded numeric logit error;
- all ranks load distinct expected TP shards;
- every rank exits successfully and destroys its process group;
- no owned child process remains.

Numeric tolerance is recorded as evidence, not silently accepted. Token and
argmax parity remain hard requirements even when BF16 reduction ordering
changes internal values.

No performance result is valid until this authority is complete.

## Phase 1: Communication-Exposure Profiling

### Frozen Workload Matrix

All performance rows use:

```text
checkpoint: Qwen/Qwen3.8-27B at the acquired immutable revision
mode: text-only
dtype: BF16
tensor parallel size: 4
sampling: greedy, temperature 0
EOS handling: ignored until the fixed output length
CUDA Graph policy: identical for all compared rows
scheduler policy: identical for all compared rows
```

Two workload families are required.

#### Causal batch-one profile

```text
P0: 256 prompt tokens, 128 output tokens, concurrency 1
P1: 2048 prompt tokens, 128 output tokens, concurrency 1
```

These rows provide minimally confounded per-layer decode timelines.

#### Online throughput profile

```text
Q0: 256 prompt tokens, 128 output tokens, concurrency 4
Q1: 256 prompt tokens, 128 output tokens, concurrency 8
Q2: 2048 prompt tokens, 128 output tokens, concurrency 4
```

The online rows establish QPS, TPOT, TTFT, queueing behavior, and whether the
batch-one bottleneck survives realistic concurrency.

Each workload has two warmups followed by five measured repetitions. The
measured order is deterministic and preserved in the manifest. No measured
row may reuse a process whose profile finalization failed.

### Strict GPU Admission

At controller entry and immediately before worker launch, the selected set
must contain exactly four distinct GPU UUIDs. Every selected GPU must have:

```text
memory used <= 1024 MiB
GPU utilization <= 5 percent
compute processes == []
```

After worker launch, only PIDs owned by the current attempt may appear on
those GPUs. The controller never kills, pauses, adopts, or signals an
unrelated process. Loss of the clean window invalidates the attempt.

GPU indices are selected from the current inventory rather than frozen to
historical indices. UUIDs, PCI topology, NVLink/NVSwitch topology, and rank
mapping are frozen in the attempt manifest.

### Remote Storage

All remote source snapshots, logs, traces, model-acquisition receipts, and
artifacts created by this work must remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

The campaign must not write task data beneath remote `/`, `/tmp`,
`/private/tmp`, the old remote checkout, or an adaptive-ngram checkout.
Temporary files are created inside the attempt directory and renamed
atomically.

### Instrumentation Events

Every rank emits an aligned event stream keyed by:

```text
attempt
workload
repetition
request-set digest
decode ordinal
rank
layer index
layer role
operation ordinal
```

Layer roles are:

```text
linear_attention
full_attention
mlp
normalization
residual
embedding
output_head
```

Operation classes are:

```text
gemm
attention
recurrent
collective
memory
other_compute
```

Each structured row records CPU enqueue timestamps and unresolved CUDA Event
pairs. CUDA Event durations are resolved only during bounded finalization;
the measured hot path must not call per-event `torch.cuda.synchronize()`,
`.item()`, or emit per-operation log lines.

The existing collective wrapper is extended with:

- collective kind;
- tensor shape and dtype;
- process-group identity;
- synchronous or asynchronous mode;
- source CUDA stream;
- completion CUDA stream;
- layer and operation identity.

The baseline remains synchronous. Recording asynchronous fields does not
authorize asynchronous execution.

### Nsight Timeline

One representative repetition per workload is captured with Nsight Systems.
The selected repetition is the one nearest the five-run median decode time,
chosen only after the structured runs complete.

The trace must include:

- CUDA kernels and runtime calls;
- NVTX layer and operation ranges;
- NCCL kernels when the installed Nsight version exposes them;
- stream identities;
- CPU launch threads;
- context-switch and scheduling evidence when available.

If NCCL kernels cannot be identified and correlated to the structured
collective inventory, exposed communication is not estimated from names or
wall-time subtraction. That workload is `INCONCLUSIVE_TRACE_COVERAGE`.

### Metric Definitions

For each aligned decode step and rank:

```text
step_critical_interval
    = first device operation start .. final required device operation end

gemm_union
    = union of all GEMM kernel intervals inside the step

collective_union
    = union of all validated NCCL collective intervals inside the step

compute_union
    = union of GEMM, attention, recurrent, normalization, and other
      required compute intervals

exposed_collective
    = collective_union minus overlap with compute_union

gpu_idle
    = step_critical_interval minus union of all required GPU work
```

Durations are computed from interval unions, never by summing kernel
durations. Concurrent kernels therefore do not double-count time.

The critical rank for an aligned step is the rank whose final required event
finishes last relative to the aligned step start. Primary TP4 metrics use
that rank. Per-rank distributions remain in the artifact.

Report per layer type and layer index:

- GEMM union duration;
- collective union duration;
- exposed collective duration;
- compute/collective overlap duration;
- GPU idle duration;
- collective count and bytes;
- critical-path contribution.

Report end-to-end:

- request QPS;
- output tokens per second;
- TTFT P50/P95/P99;
- TPOT P50/P95/P99;
- end-to-end latency P50/P95/P99;
- peak allocated and reserved CUDA memory per rank;
- GPU utilization and power samples;
- exact-token and argmax correctness.

### Exposure Classification

For every workload:

```text
exposed_communication_ratio
    = critical-rank exposed collective duration
      / critical-rank step critical-interval duration

overlap_headroom_lower_bound
    = min(exposed collective duration, independent compute duration)
      / step critical-interval duration
```

The terminal classifier uses the median of measured repetitions and requires
the same direction in at least four of five repetitions.

Classification precedence:

1. `INVALID_CORRECTNESS`
2. `INVALID_RESOURCE_IDENTITY`
3. `INCONCLUSIVE_TRACE_COVERAGE`
4. `INCONCLUSIVE_VARIANCE`
5. `GO_COMMUNICATION_OVERLAP`
6. `NO_GO_ALREADY_HIDDEN`
7. `INCONCLUSIVE_LOW_HEADROOM`

`GO_COMMUNICATION_OVERLAP` requires:

- all correctness and resource gates pass;
- all five workload profiles have complete four-rank alignment;
- at least one causal profile and one online profile have median exposed
  communication ratio at least 10 percent;
- those qualifying workloads have overlap-headroom lower bound at least
  5 percent;
- at least four of five repetitions agree in direction;
- profiler overhead is no more than 3 percent in paired profiled/unprofiled
  control runs.

`NO_GO_ALREADY_HIDDEN` requires every workload to have:

- median exposed communication ratio below 5 percent; and
- overlap-headroom lower bound below 2 percent.

Results between those boundaries are
`INCONCLUSIVE_LOW_HEADROOM`. High variation or incomplete trace correlation
cannot be promoted to `GO`.

## Conditional Phase 2: Overlap Authorization Boundary

A `GO_COMMUNICATION_OVERLAP` result authorizes writing a separate design that
chooses one exact ownership transformation. It does not authorize all of the
following at once.

Candidate mechanisms are evaluated in this order:

1. asynchronous existing collective with useful independent downstream work;
2. chunked local projection plus chunked collective;
3. `ReduceScatter` with a deliberately sharded consumer;
4. delayed `AllGather` at the first consumer that requires replicated state.

An `all_reduce(async_op=True)` call followed immediately by `wait()` is not
communication/computation fusion. A `ReduceScatter` followed immediately by
an `AllGather` with no intervening sharded consumer is also not an acceptable
optimization.

The later implementation design must define:

- chunk ownership and ordering;
- producer and consumer streams;
- CUDA Event wait edges;
- buffer lifetime and reuse;
- residual and normalization ownership;
- cancellation, exception, timeout, and rank-failure cleanup;
- deterministic fallback to the frozen synchronous baseline;
- CUDA Graph compatibility;
- correctness tolerances and performance gates.

## Failure Semantics

The profiler and campaign fail closed on:

- unsupported checkpoint fields or shapes;
- incomplete checkpoint inventory;
- wrong model revision;
- rank, GPU UUID, process, or topology drift;
- missing layer or operation ranges;
- non-monotonic or cross-step timestamps;
- events associated with the wrong rank, layer, or request;
- profiler finalization failure;
- missing Nsight correlation needed for exposed-time calculation;
- NaN or Inf;
- token or argmax mismatch;
- unrelated GPU process appearance;
- remote output escaping the approved root;
- incomplete cleanup.

Failure artifacts are preserved under their fresh attempt tag and are never
overwritten or relabeled as valid measurements.

## Artifact Contract

The terminal local bundle contains:

```text
source_manifest.json
model_manifest.json
environment.json
gpu_topology.json
workload_manifest.json
correctness_rows.jsonl
profile_rows.jsonl
layer_summary.json
communication_exposure_summary.json
online_metrics.json
memory_summary.json
resource_samples.jsonl
nsys/
independent_verification.json
manifest.sha256
report.md
```

The producer and an independent verifier must agree on:

- source and model identity;
- exact row inventory;
- four-rank alignment;
- interval-union calculations;
- exposed communication ratios;
- profiler overhead;
- correctness;
- terminal classification.

The local downloader recomputes every listed artifact hash. A successful
worker exit, a complete manifest, or a passing unit suite is not sufficient
without semantic verification of every gate requirement.

## Prompt-to-Artifact Completion Checklist

| Requirement | Required artifact or evidence |
| --- | --- |
| Official Qwen3.8-27B | immutable repository revision and complete model manifest |
| Four clean GPUs | entry and worker-entry inventories with UUIDs, memory, utilization, and process lists |
| Per-layer GEMM | aligned `gemm` intervals plus layer-index summaries |
| Per-layer NCCL | structured collective rows correlated with NCCL kernel intervals |
| GPU idle gap | interval-union complement inside each critical step |
| Exposed communication | independently recomputed critical-path interval subtraction |
| No pointless fusion | terminal `NO_GO_ALREADY_HIDDEN` prevents an implementation spec |
| Chunked collective | deferred to a separate approved design after `GO` |
| Compute/communication streams | deferred ownership and event graph after `GO` |
| CUDA Event dependencies | required by the conditional implementation design |
| Synchronization barriers | baseline inventory and later before/after comparison |
| QPS | online profile rows and aggregate distribution |
| TPOT | P50/P95/P99 from fixed-output workloads |
| TTFT | P50/P95/P99 from the same request records |
| Memory | per-rank peak allocated/reserved and external telemetry |
| Correctness | TP1 reference, TP4 token/argmax parity, finite logits, numeric error |
| Reproducibility | source, model, environment, workload, topology, and manifest hashes |
| Independent verification | separate verifier output matching producer classification |

## Non-Goals

This gate does not:

- claim that QPS `1.55 -> 2.34` is reproducible;
- compare TinyLLMForge with another framework;
- optimize or load the vision encoder;
- add quantization;
- add speculative decoding;
- change scheduler fairness;
- change KV-cache layout or offload;
- weaken strict clean-GPU admission;
- treat Qwen3.5-2B results as Qwen3.8-27B performance evidence;
- authorize overlap when trace coverage or correctness is incomplete.

## Completion Criteria

The profiling gate is complete only when:

1. the spec and implementation plan are committed and pushed;
2. the Qwen3.8 text-only adapter passes focused and adjacent tests;
3. TP1 reference and real TP4 correctness authorities pass;
4. a fresh strict-clean four-GPU campaign completes all workload rows;
5. structured and Nsight evidence covers GEMM, NCCL, idle, and exposed time;
6. producer and independent verifier agree;
7. the terminal bundle and manifest validate locally;
8. audit and handoff documents state the exact result and claim boundary;
9. all intended files are exact-staged, committed with one required trailer,
   and pushed to `origin/feat/kv-sparse-attention`.

Only a verified `GO_COMMUNICATION_OVERLAP` advances to the separate
communication/computation-overlap design.
