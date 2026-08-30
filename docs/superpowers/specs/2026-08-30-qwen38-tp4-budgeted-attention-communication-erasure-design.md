# Qwen3.8 TP4 Budgeted Attention Communication Erasure Design

**Date:** 2026-08-30

**Status:** Approved design; implementation plan pending

**Stage-1 model:** `Qwen/Qwen3.8-27B`

**Model revision:** `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

**Runtime topology:** BF16, tensor parallel size four

## 1. Objective

Determine whether TinyLLMForge can improve Qwen3.8-27B TP4 decode latency by
replacing a memory-budgeted subset of attention output projections with
replicated projections, thereby removing their synchronous output
AllReduces.

The first implementation must remain default-disabled. It may be promoted
only if one immutable paired campaign proves:

- exact source, model, topology, workload, and layer-selection identities;
- the real steady-decode collective sequence;
- correctness against the unchanged TP4 baseline;
- lower TPOT without unacceptable TTFT, throughput, tail-latency, or memory
  regressions;
- cleanup and independent-verifier agreement; and
- both the realized benefit and the measured cost.

This design does not claim that projection replication, tensor parallelism,
or communication avoidance is novel. The TinyLLMForge-specific contribution
is the end-to-end policy that derives the actual collective graph, measures
the per-layer compute-versus-communication break-even, selects a bounded
replication mask under a hard memory budget, and validates that mask with a
frozen paired gate.

## 2. Corrected evidence boundary

The earlier synchronous collective-reduction design assumed:

```text
embedding AllReduce
+ 2 row-parallel AllReduces per decoder layer
+ greedy-token broadcast
= 130 collectives per decode token
```

That assumption does not match the implemented Qwen3.8 runtime.

The current Qwen3.8 model construction uses:

- one vocabulary-parallel embedding AllReduce;
- one row-parallel attention output projection in each of 64 layers;
- a replicated MLP gate/up projection and replicated MLP down projection;
- one greedy-token broadcast after rank-0 exact-vocabulary selection.

Therefore the expected steady-decode sequence is:

```text
1 embedding AllReduce
+ 64 attention-output AllReduces
+ 1 greedy-token broadcast
= 66 collectives per decode token
```

The immutable r10 bundle cannot establish the 130-site dynamic sequence:

- `selected_event_budget` is `null`;
- `coverage_complete` is `false`;
- `collective_census.jsonl` is empty;
- `collective_timing_samples.jsonl` is empty; and
- `paired_online_metrics.json` contains no terminal workloads.

Its valid terminal classification remains
`INCONCLUSIVE_PROFILER_OVERHEAD`, but its handwritten 130-site static catalog
must not be reused as runtime truth.

The earlier complete communication-profile bundle independently observed 65
model collectives per single-sequence decode step: one embedding collective
and one collective in each of 64 decoder layers. The token broadcast was
outside that model-layer summary. This evidence supports a 66-site target,
but the new campaign must still regenerate the count from the current frozen
source.

## 3. Why attention output projection is the target

Every decoder layer currently computes an attention output projection as:

```text
rank-local attention heads
  -> rank-local FP32 output-projection GEMM
  -> synchronous FP32 AllReduce of [active_tokens, 5120]
  -> BF16 conversion
  -> residual addition
  -> RMSNorm
```

The residual addition is an immediate consumer. Merely changing the
AllReduce to `async_op=True` cannot hide useful work because the next
semantic operation needs the reduced tensor.

The MLP is not a candidate for collective removal because its projection
weights are already replicated and its decode path has no MLP output
AllReduce.

The embedding collective is not the first candidate because eliminating one
call requires approximately 1.91 GB of extra persistent memory per rank and
removes only one of 66 calls.

Attention output projection offers 64 independently selectable sites with a
uniform shape. A selected site can trade more local GEMM work and model
memory for removal of one synchronous AllReduce.

## 4. Considered approaches

### A. Budgeted attention projection replication — selected

For selected layers, retain a full FP32 accumulation weight on every TP rank
and execute the complete attention output projection locally. The layer then
skips its output AllReduce.

Advantages:

- removes a whole collective rather than attempting to hide an immediate
  dependency;
- preserves the existing FP32 accumulation boundary;
- layer selection provides an explicit memory and compute budget;
- can be tested first with isolated real-shape kernels and then with an
  end-to-end paired run; and
- follows an existing runtime principle: use replicated computation only
  where it is cheaper than communication.

Costs:

- four times the projection GEMM work at selected sites;
- approximately 90 MiB of additional persistent FP32 weight per selected
  layer relative to the current local FP32 accumulation weight;
- possible TTFT increase if initialization or materialization is not kept
  outside measured request execution; and
- topology- and workload-dependent break-even.

### B. FP32 peer-reduction plus residual fused kernel — deferred fallback

Implement a topology-specific CUDA kernel that reads rank-local output
partials through peer mappings, performs the FP32 reduction, adds the
residual, and writes the replicated result.

Advantages:

- avoids full projection-weight replication;
- could remove both NCCL launch overhead and a residual-add kernel; and
- directly targets tiny-message collective latency.

Costs:

- substantially larger correctness and failure-safety surface;
- requires peer-access, synchronization, process-lifetime, and topology
  contracts;
- needs a new extension and multi-process test harness; and
- reduction ordering may differ from NCCL and must be checked explicitly.

This approach is considered only if approach A returns a clean performance
NO_GO while proving that attention collectives still provide at least five
percent removable TPOT opportunity.

### C. Chunked GEMM/AllReduce wavefront — rejected for Stage 1

Split the output projection into chunks, overlap a chunk's collective with
the next chunk's GEMM, and incrementally prepare residual/RMSNorm inputs.

This creates a legal overlap window, but each current payload is only:

```text
active_tokens * 5120 * sizeof(float32)
```

or 20 KiB per active token. Splitting it increases collective and GEMM launch
counts, while RMSNorm still cannot finish until all hidden-dimension chunks
are available. Stage 1 therefore rejects this option unless later evidence
shows a large-message regime where the launch penalty is amortized.

## 5. Memory model and candidate masks

Both full-attention and linear-attention output projections have shape:

```text
[5120, 6144]
```

Per layer:

```text
full BF16 prefill weight:            60 MiB
current rank-local BF16 shard:       15 MiB
current rank-local FP32 weight:      30 MiB
full FP32 accumulation weight:      120 MiB
increment for replicated FP32:       90 MiB
```

Replicating all 64 layers would add:

```text
64 * 90 MiB = 5.625 GiB per rank
```

The previous Q2 baseline reached approximately 77.57 GiB reserved memory, so
full replication is excluded before benchmarking.

The Stage-1 candidate masks are:

| Candidate | Selected layers | Extra persistent memory per rank | Removed attention AllReduces |
| --- | ---: | ---: | ---: |
| `R0` | 0 | 0 MiB | 0 |
| `R8` | 8 | 720 MiB | 8 |
| `R16` | 16 | 1,440 MiB | 16 |

Layer masks must be deterministic and independent of observed end-to-end
scores. Stage 1 uses evenly spaced layer indices:

```text
R8:
  3, 11, 19, 27, 35, 43, 51, 59

R16:
  3, 7, 11, 15, 19, 23, 27, 31,
  35, 39, 43, 47, 51, 55, 59, 63
```

These are the 16 full-attention positions in the checkpoint's fixed
three-linear/one-full layer pattern. `R8` takes every second full-attention
position. This avoids selecting layers from noisy profiler rankings and
keeps the first implementation limited to one attention implementation.

No adaptive post-result mask tuning is allowed in the same campaign.

## 6. Architecture

### 6.1 Runtime policy

Add a default-disabled runtime policy with:

```text
enabled
selected_layer_indices
expected_tensor_parallel_size=4
expected_input_size=6144
expected_output_size=5120
accumulation_dtype=float32
```

The policy fails closed when:

- TP size is not four;
- a selected layer is not a full-attention layer;
- a selected projection has a different shape;
- the checkpoint binding cannot provide the full source weight;
- the current projection is quantized;
- the full accumulation weight cannot be materialized before request
  admission; or
- projected memory exceeds the frozen per-rank budget.

### 6.2 Projection implementation

Introduce a dedicated replicated FP32 accumulation projection rather than
changing generic `ReplicatedLinear` behavior.

For a selected full-attention layer:

```text
local attention output [tokens, 1536]
  -> unavailable as a complete projection input
```

The implementation must not incorrectly feed the local attention-head shard
to a full `[5120, 6144]` projection. It must first prove one of these exact
data paths:

1. the attention backend can expose the already-computed full 6144-wide
   output on every rank without extra attention or KV-cache replication; or
2. an input AllGather plus full projection is faster than the current local
   projection plus output AllReduce.

Current source inspection does not prove path 1. Therefore the first
executable candidate is path 2:

```text
rank-local BF16 attention output [tokens, 1536]
  -> BF16 AllGather [tokens, 6144]
  -> full FP32 projection [tokens, 5120]
  -> BF16 result
  -> residual addition
```

The communication operation changes from:

```text
FP32 AllReduce payload: 20,480 bytes per active token
```

to:

```text
BF16 full input size: 12,288 bytes per active token
```

The gate must report actual transport semantics and observed latency; it must
not equate these logical tensor sizes with wire bytes.

The existing full BF16 prefill weight may be reused only as checkpoint
source material. Decode must use a full FP32 accumulation weight so that the
candidate does not silently conflate communication optimization with reduced
arithmetic precision.

### 6.3 Model wiring

Only selected full-attention `output_projection` modules use the candidate.
All other modules remain byte-for-byte on the baseline path.

The candidate must expose a stable per-layer receipt:

```text
layer_index
projection_mode
input_collective_kind
input_tensor_shape
input_tensor_dtype
output_collective_kind
output_tensor_shape
output_tensor_dtype
additional_persistent_bytes
```

Expected candidate behavior:

```text
input_collective_kind=all_gather
output_collective_kind=none
```

Expected baseline behavior:

```text
input_collective_kind=none
output_collective_kind=all_reduce
```

## 7. Qualification stages

### Stage 0: collective-contract repair

Before candidate timing:

1. derive the static catalog from the actual model module graph;
2. require exactly 66 decode collectives for the frozen checkpoint;
3. require exactly 64 attention-output sites and zero MLP-output sites;
4. require one embedding AllReduce and one greedy-token broadcast;
5. obtain a real four-rank dynamic census for at least two decode steps; and
6. make the old 130-site fixture fail.

This stage updates the old audit's claim boundary but does not rewrite or
delete immutable r10 artifacts.

### Stage 1: real-shape break-even microgate

Run baseline and candidate projection transactions on four strict-clean
A100 GPUs for:

```text
active_tokens = 1, 4, 8
input width  = 6144
output width = 5120
dtype        = BF16 input, FP32 accumulation
```

Each pair uses the same frozen input and weight tensors. Measure:

- transaction median and P95 latency;
- GEMM CUDA duration;
- collective CUDA duration;
- host submission duration;
- allocated and reserved memory delta;
- maximum absolute and relative output error; and
- exact BF16 output equality after final conversion.

The microgate is diagnostic. It cannot establish end-to-end GO.

An `R8` or `R16` end-to-end candidate is eligible only if:

- candidate median transaction latency is at least five percent lower for
  active-token counts one and four;
- active-token count eight does not regress by more than two percent;
- output passes the frozen correctness contract; and
- measured persistent memory is within five percent of the predicted budget.

### Stage 2: paired end-to-end gate

Reuse the frozen workload matrix:

```text
P0: prompt=256,  output=128, concurrency=1
P1: prompt=2048, output=128, concurrency=1
Q0: prompt=256,  output=128, concurrency=4
Q1: prompt=256,  output=128, concurrency=8
Q2: prompt=2048, output=128, concurrency=4
```

Compare:

```text
baseline R0
candidate R8
candidate R16, only if admitted by memory preflight
```

Use alternating paired order, two warmups, and at least five measured
repetitions per workload and arm. No Nsight profiling is used in the
performance decision. A separate bounded diagnostic may run after the gate,
but it cannot replace paired online timing.

## 8. Correctness contract

For every candidate:

- generated token IDs must exactly match baseline for every request;
- output lengths and stop reasons must match;
- all logits and hidden states sampled by the frozen verifier must be finite;
- selected-layer receipts must match on all four ranks;
- input AllGather shapes, dtypes, and order must match on all ranks;
- candidate layers must have no output AllReduce;
- unselected layers must retain the baseline output AllReduce;
- no MLP-output collective may appear;
- startup materialization must complete before measured request admission;
  and
- candidate activation or teardown must not change another runtime policy.

Exact token equality is required, but it is not described as bitwise
floating-point equivalence. Any nonzero sampled logit or hidden-state
difference must be reported with maximum absolute and relative error.

## 9. Performance and cost gate

The best admitted candidate receives `GO` only if all conditions hold:

### Benefit

- aggregate paired median TPOT improvement is at least five percent;
- P0 and P1 each improve median TPOT by at least three percent;
- Q0, Q1, and Q2 have no median TPOT regression greater than two percent;
- aggregate output tokens/s improves by at least three percent;
- no workload's P99 E2E latency regresses by more than three percent; and
- observed attention-output collective count changes exactly according to
  the selected mask.

### Cost

- TTFT regression is at most three percent for every workload;
- peak allocated and reserved memory remain below the physical limit with at
  least 512 MiB headroom per rank;
- measured persistent-byte increase is within five percent of the declared
  budget;
- no new CPU offload or host-memory staging occurs;
- no extra model initialization is included inside measured request timing;
  and
- no worker, child, shared-memory object, or GPU allocation remains after
  cleanup.

### Classification

```text
GO_BUDGETED_ATTENTION_COMMUNICATION_ERASURE
NO_GO_CORRECTNESS
NO_GO_MICROGATE
NO_GO_PERFORMANCE
NO_GO_MEMORY
NO_GO_TAIL_OR_TTFT
INCONCLUSIVE_RESOURCE_IDENTITY
INCONCLUSIVE_EVIDENCE
```

The optimization remains default-disabled for every classification except
`GO_BUDGETED_ATTENTION_COMMUNICATION_ERASURE`.

## 10. Evidence and verifier

The immutable bundle must contain:

```text
source_identity.json
model_manifest.json
gpu_topology.json
runtime_collective_catalog.json
dynamic_collective_census.jsonl
layer_selection_policy.json
projection_receipts.jsonl
microgate_rows.jsonl
paired_online_metrics.json
correctness.jsonl
memory_summary.json
resource_samples.jsonl
cleanup.json
classification.json
independent_verification.json
manifest.sha256
```

The independent verifier must reconstruct:

- source and model identity;
- the 66-site baseline sequence;
- selected and unselected layer sets;
- expected collective substitutions;
- memory accounting;
- correctness;
- paired performance statistics;
- every threshold and terminal classification; and
- the final manifest.

An empty dynamic census, a handwritten-only catalog, missing paired arm,
missing workload, missing repetition, or missing cleanup proof is
`INCONCLUSIVE_EVIDENCE`.

## 11. Remote execution constraints

All remote task data must remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

The controller must:

- verify Kerberos lifetime without running `kinit` or `krenew`;
- admit exactly four strict-clean GPUs;
- retain ownership of monitoring and launch decisions locally;
- never terminate, adopt, or clean an external GPU process;
- use a fresh immutable attempt tag;
- never overwrite or delete a failed or partial attempt;
- keep large raw traces remote;
- download only compact evidence; and
- perform a second strict-clean check immediately before launch.

## 12. Stop rules

Stop before end-to-end implementation if:

- Stage 0 cannot prove the 66-site sequence;
- the candidate merely exchanges AllReduce for an equal-or-slower AllGather;
- neither R8 nor R16 passes the microgate;
- correctness requires weakening exact token equality;
- R16 cannot retain 512 MiB physical-memory headroom;
- the observed benefit depends on profiler timing; or
- the implementation would require modifying non-attention model semantics.

If approach A stops, preserve the result as a measured NO_GO and use its
real-shape transaction evidence to decide whether approach B has sufficient
absolute headroom. Do not proceed to the chunked wavefront without a new
design.

## 13. Claim boundary

A successful gate may support this claim:

> TinyLLMForge uses a runtime-derived, memory-budgeted layer mask to exchange
> selected Qwen3.8 TP4 attention output AllReduces for lower-volume input
> AllGathers plus replicated FP32 projection, and the frozen paired campaign
> measured the reported latency benefit and memory/compute cost.

It must not support these broader claims without separate evidence:

- all Qwen3.8 TP4 communication was removed;
- communication and computation were generally overlapped;
- the mechanism is faster on another GPU topology, TP size, model, dtype, or
  workload;
- every component is novel; or
- a diagnostic ceiling is a production speedup.
