# Qwen3.8 Topology-Local TP2 Linear-Attention Islands Design

**Date:** 2026-09-08

**Status:** Approved design under the standing autonomous optimization
authorization

**Source anchor:** `838614322fd99be3c90d4745d9a6a539b4d428bd`

**Target branch:** `feat/kv-sparse-attention`

**Model:** `Qwen/Qwen3.8-27B`

**Model revision:** `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

**Default:** disabled

**First gate:** real-checkpoint four-rank linear-attention microgate

**End-to-end integration:** prohibited unless Stage 0 returns
`GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE`

## 1. Decision

The next TP4 optimization will test **topology-local TP2 linear-attention
islands**.

The four global tensor-parallel ranks are divided into two topology-local
pairs. During each of Qwen3.8-27B's 48 linear-attention layers, both pairs
execute the same logical TP2 mixer:

```text
global TP4 ranks:       0        1        2        3
pair replica:           A        A        B        B
logical TP2 rank:       0        1        0        1

linear-attention layer:
  pair A computes one complete mixer output with a two-rank AllReduce
  pair B computes the same complete mixer output with a two-rank AllReduce

full-attention layer:
  all four ranks execute the existing global TP4 path
```

The candidate deliberately spends more linear-attention computation and state
memory to avoid cross-NUMA four-rank synchronization at 48 of 64 attention
output boundaries. It does not split requests between replicas. Every global
rank continues to hold the same logical request batch and the same replicated
hidden state at layer boundaries.

Stage 0 is a real-checkpoint mixer microgate, not production integration. It
must measure the complete affected linear-attention path, the one-time
TP4-to-TP2 state transformation, output and state correctness, host cost, and
memory cost. A projection-only win is insufficient.

## 2. Why this follows the completed evidence

The terminal completion-owned overlap campaign proved that correct fine-grain
overlap is possible but not profitable:

- active-token 4 median critical latency regressed `28.670041%`;
- active-token 8 median critical latency regressed `10.280405%`;
- the active-token 4/8 geometric aggregate speedup was `-19.120881%`;
- host submission regressed by `99.162755%` to `127.951833%`; and
- the terminal classification was `NO_GO_PERFORMANCE`.

Earlier alternatives also reached terminal no-go results:

- fixed-slot CUDA IPC peer reduction was `NO_GO_MICROGATE`;
- cross-request wavefront overlap was `NO_GO_INSUFFICIENT_OVERLAP`; and
- segmented capture pivoted to communication-compute fusion without providing
  steady-state performance evidence.

Those results rule out another wrapper around every small AllReduce. The next
candidate must alter the communication structure over a larger semantic
region.

The current Qwen3.8 implementation provides a model-specific opportunity:

- hidden size is 5,120;
- the model has 64 layers;
- 48 layers use linear attention and 16 use full attention;
- the measured steady-decode path has one attention-output AllReduce per
  layer, plus embedding reduction and token broadcast;
- each one-token attention-output collective carries 20,480 FP32 bytes;
- the linear-attention input projections already store complete replicated
  weights and compute their complete outputs before selecting a TP-local
  slice;
- the linear-attention output projection already preserves a complete dense
  BF16 weight for the prefill path; and
- the MLP is already replicated rather than tensor parallel in the current
  Qwen3.8 assembly.

On the profiled four-GPU placement, ranks 0/1 and ranks 2/3 are local to
separate PCIe/NUMA regions, while a global four-rank collective crosses the
system interconnect. The candidate therefore targets both collective
participant count and topology, not asynchronous launch overhead.

## 3. Current and candidate data flow

### 3.1 Current linear-attention decode layer

Each rank begins with the same BF16 hidden state:

```text
replicated hidden [tokens, 5120]
  -> complete input projections, then select global TP4 head quarter
  -> local convolution and recurrent delta-rule state update
  -> local gated value slice
  -> local FP32 output-projection contribution [tokens, 5120]
  -> global four-rank FP32 AllReduce
  -> BF16 attention output, replicated on all ranks
  -> residual + post-attention RMSNorm
  -> replicated MLP
```

The AllReduce result has an immediate nonlinear consumer. It cannot simply be
deferred past residual addition and RMSNorm.

### 3.2 Candidate linear-attention decode layer

The candidate changes only the linear-attention mixer:

```text
replicated hidden [tokens, 5120]
  -> complete input projections, then select logical TP2 head half
  -> pair-local convolution and recurrent delta-rule state update
  -> pair-local gated value half
  -> pair-local FP32 output-projection contribution [tokens, 5120]
  -> topology-local two-rank FP32 AllReduce
  -> complete BF16 attention output in each pair replica
  -> residual + post-attention RMSNorm
  -> existing replicated MLP
```

Pair A and pair B start from identical hidden states, parameters, request
identity, and logical TP2 state. They must independently produce equivalent
complete outputs. No cross-pair collective is inserted in the timed path.

At each full-attention layer, all ranks execute the unchanged global TP4
implementation. Because the preceding pair replicas must agree on the full
hidden state, no layout conversion is needed at a linear-to-full or
full-to-linear layer boundary.

## 4. Why persistent hidden sharding is not the first candidate

A ReduceScatter plus delayed-AllGather region was evaluated as the initial
leading option. It is not selected for Stage 0.

The current layer boundary requires:

```text
attention output
  -> residual addition
  -> RMSNorm
  -> replicated MLP input
```

Hidden-dimension sharding can carry the residual addition locally, but RMSNorm
requires at least a distributed scalar reduction, and the current replicated
MLP requires the full normalized hidden vector. The candidate would therefore
pay an AllGather almost immediately. Converting the MLP and the next
attention input projections to consume hidden shards would require a new 2D
weight layout or additional reductions over the much larger intermediate
dimension.

Sequence-parallel sharding has the same structural problem for small decode
batches: the TP projections need an AllGather before each column-parallel
projection, and active-token groups smaller than four cannot usefully occupy
all sequence shards.

Persistent sharding remains a possible future architecture, but the current
model layout offers no narrow region that both:

1. crosses multiple substantial operations; and
2. avoids an immediate compensating collective.

Implementing it first would therefore be a broad model rewrite without a
defensible communication-count reduction.

## 5. Alternatives considered

| Approach | Potential benefit | Required cost | Decision |
|---|---|---|---|
| Topology-local TP2 linear-attention islands | Replaces 48 cross-NUMA TP4 attention reductions with local two-rank reductions; reuses already replicated projection weights and full prefill output weights | Doubles linear-attention state per rank and duplicates half of the logical mixer computation; requires prefill-to-decode state remap | **Recommended** |
| Persistent sharded hidden region | Could replace full materialization with longer-lived shards if several consumers accepted them | Current replicated MLP forces early AllGather; distributed RMSNorm and weight relayout are invasive | Reject for this stage |
| TP4 Exact Greedy K8 collective epoch | Amortizes scheduler, lease, and graph-control overhead over eight decode steps | Token dependencies remain sequential; 64 attention collectives per token remain; TP4 token agreement, state rollback, and graph ownership are unresolved | Defer |
| Two complete TP2 serving replicas | Often improves aggregate QPS under concurrency and avoids TP4 collectives | Changes service topology and request routing rather than one-request TP communication; cannot establish a same-request fusion claim | Retain only as an end-to-end control |

TP4 Exact K8 may still be valuable for host and scheduling overhead, but it
does not attack the measured communication structure. Eight exact steps would
still execute the model's sequential per-layer collectives eight times unless
combined with a separate communication-changing design.

## 6. Topology and process-group contract

### 6.1 Frozen pair map

Stage 0 accepts an explicit pair map and records both global ranks and physical
GPU UUIDs. The default map for a four-rank job is:

```text
pair A: global ranks [0, 1]
pair B: global ranks [2, 3]
```

The controller may choose a different perfect matching only before attempt
creation. The selected matching is immutable within an attempt.

All ranks must create all pair process groups in the same global order.
Each rank records:

- global rank and world size;
- pair identifier;
- logical pair rank;
- physical GPU index and UUID;
- pair link class from the captured topology;
- whether the pair crosses a NUMA boundary; and
- the global source revision and source-tree digest.

### 6.2 Admission

The candidate requires:

- exactly four admitted GPUs;
- no modification or termination of foreign processes;
- a topology record that distinguishes the three possible pair matchings;
- two disjoint two-rank pairs;
- no selected pair worse than the best available perfect matching under the
  frozen topology ranking; and
- enough free memory for the measured candidate state and temporary migration
  workspace.

GPU cleanliness, memory allowance, and Kerberos lifetime remain controller
admission conditions. They are not performance results.

The launch-time Kerberos floor is 10,800 seconds. This covers the two-hour
worker timeout plus one hour for staging, verification, and compact download.
The controller never renews credentials itself; an external credential agent
may refresh the fixed cache while the campaign is running.

Each SSH command retries transport exit status 255 within the frozen retry
budget. Retries use bounded exponential delays of one, then two, then four
seconds so a transient proxy close cannot exhaust the entire budget
instantaneously. Non-255 command failures return immediately, and an attempt
that was created remotely is never recreated or repaired.

### 6.3 No hidden global synchronization

The timed candidate linear-attention path may not contain:

- a global four-rank AllReduce;
- an AllGather of the output activation;
- a global barrier;
- device-wide synchronization;
- cross-pair correctness checks;
- host polling for pair completion; or
- a synchronous fallback after pair execution begins.

Cross-pair comparison is allowed only outside the timed interval.

## 7. Logical TP2 weight view

### 7.1 Input projections

`ReplicatedSegmentedColumnParallelLinear`,
`ReplicatedColumnParallelLinear`, and
`ReplicatedLocalOutputLinear` already own complete source weights on every
rank. Stage 0 adds an explicit logical-parallel view:

```text
logical_parallel_size = 2
logical_parallel_rank = global_rank % 2
```

The candidate retains the existing complete checkpoint weights. Q, K, and V
must use the same complete fused QKV projection as the TP4 baseline before
selecting the logical TP2 segments. A diagnostic using three smaller Q/K/V row
view GEMMs changed the BF16 GEMM numerical path: token-1 remained correct, but
the first token-4 case changed the downstream greedy argmax despite all tensor
tolerance checks passing. The complete fused QKV projection is therefore a
correctness requirement for this Stage-0 comparison, not an optional fallback.

Z must likewise use the complete baseline-shaped projection before selecting
the logical TP2 half. Its half-row GEMM differed in only 19--21 of 12,288
projection elements with maximum absolute error at most `2.44140625e-4`, but a
later token-8 diagnostic still changed the downstream greedy argmax. The
logical A and B halves are concatenated once before warmup into one contiguous
48-row BF16 weight and computed by one fused GEMM. Across token groups 1, 4,
and 8, this fused result was bitwise equal to the corresponding slices from
the two baseline-shaped A and B GEMMs.

Computing complete A and B projections and discarding the unused half is
rejected because it spends avoidable input-projection FLOPs and one extra
kernel launch. The fused A/B half costs 0.46875 MiB per linear-attention layer,
or 22.5 MiB per rank across 48 layers. Materializing fused TP2 copies for QKV
or Z remains rejected because their persistent memory cost would invalidate
the integrated memory budget, and their smaller GEMM shapes failed the frozen
exact-greedy correctness gate.

The logical view must not mutate the module's global TP identity or change the
full-attention path.

### 7.2 Output projection

The linear-attention `RowParallelLinear` already retains:

- the current TP4 decode shard in `weight`; and
- the complete BF16 dense output weight in `prefill_weight`.

Stage 0 derives the logical TP2 input-column half from `prefill_weight`. It
materializes one contiguous FP32 logical half for the measured layer. The
remaining 47 layers are represented by an exact-size resident allocation in
the Stage-0 memory projection, so the microgate cannot pass by measuring only
one layer's weight cost.

The pair-local output projection:

1. consumes the logical value-head half;
2. computes an FP32 local contribution using the matching input-column half;
3. applies a two-rank pair AllReduce; and
4. casts the complete result to the input BF16 dtype.

The baseline remains the current global TP4 row-parallel path.

An integrated candidate would replace each 30 MiB TP4 FP32 accumulation shard
with a 60 MiB TP2 FP32 accumulation shard. Across 48 linear-attention layers,
the incremental output-projection weight is therefore 1,440 MiB
(1.40625 GiB) per rank. The preserved complete BF16 prefill weight already
exists in the baseline and is not counted as a candidate-only allocation.

The Stage-0 process retains the measured layer's baseline quarter and
candidate half simultaneously for paired timing. Its projected allocation for
the other 47 layers is adjusted so the total resident candidate increment
matches the integrated layout, plus at most one measured-layer quarter as
explicit microgate overhead. Both values must be reported.

### 7.3 Linear-state parameters

The convolution weight, `A_log`, and `dt_bias` currently follow the TP4 head
quarter. Stage 0 loads the logical TP2 half directly from the pinned
checkpoint rather than communicating parameter shards at runtime.

`norm_weight` is already replicated over the per-head feature dimension and
does not change. All candidate parameter allocations are established before
warmup and remain immutable during timing.

### 7.4 Parameter identity

The assembler records digests proving:

- both replicas use the same logical-rank-0 parameter slices;
- both replicas use the same logical-rank-1 parameter slices;
- concatenating logical TP2 slices reconstructs the same complete checkpoint
  parameter used by the baseline; and
- no full-attention or MLP parameter is changed.

### 7.5 Short-chunk gated-delta specialization

The formal r5 result showed that the fused A/B candidate is correct but does
not have a stable performance margin: active-token 4/8 geometric aggregate
speedup was `4.0542%`, and active-token 4 P99 regressed `5.2860%`. Comparing
r5 with diagnostic-r15 also showed that the diagnostic GO depended partly on
the second GPU pair having a slower baseline; it is not sufficient evidence
for a stable software win.

The dominant candidate component for active-token 4 and 8 is the
gated-delta core, at approximately 10.4 ms of an approximately 10.8 ms
candidate path. The current chunk implementation pads every non-recurrent
call to `chunk_size=64`, then executes the 64-step triangular recurrence even
when only four or eight tokens are active.

The next candidate revision therefore specializes the chunk size:

```text
token_count == 1       -> recurrent path, unchanged
2 <= token_count <= 8  -> chunk_size = token_count
token_count > 8        -> chunk_size = 64, unchanged
```

This is preferred over a fixed short chunk of eight because token-4 would
still execute four padded rows, and preferred over token-by-token recurrent
execution because that would introduce multiple sequential launches. A
single-GPU CUDA probe over 12- and 24-head geometries measured the
gated-delta core falling from approximately 6.14 ms at chunk size 64 to
1.38--1.71 ms at chunk sizes 4 and 8. The probe observed maximum output
difference `1.220703125e-4` and maximum state difference
`2.384185791015625e-7`, both within the existing tensor tolerances, but it is
diagnostic evidence only.

The specialization is part of the candidate composition, not silently
applied to the TP4 baseline. This makes the comparison answer the practical
question "does the revised topology-local runtime beat the current TP4
runtime?" rather than attempting to attribute every nanosecond solely to
communication topology. Reports must therefore name both ingredients:
topology-local TP2 islands and short-chunk gated-delta specialization.

The specialization remains default-off outside this Stage-0 worker. It must
not modify the production Qwen3.5/Qwen3.8 linear-attention path. The existing
full output/state tolerance, pair-replica equality, downstream exact-greedy,
allocation, lifecycle, memory, migration, tail, and performance gates remain
unchanged.

## 8. State layout and migration

### 8.1 Exact state sizes

The pinned model configuration has:

- 16 linear key heads;
- 48 linear value heads;
- key and value head dimensions of 128;
- convolution kernel width 4; and
- 48 linear-attention layers.

With BF16 convolution state and FP32 recurrent state, one request slot uses:

| Layout | Per-rank state across 48 linear layers |
|---|---:|
| TP4 baseline | 36.9375 MiB |
| TP2 island candidate | 73.875 MiB |
| Increment | 36.9375 MiB |

At the formal Stage-0 capacity of eight slots, the static incremental
per-rank state is 295.5 MiB. Temporary migration storage is measured
separately and must be released before steady-state timing.

The integrated candidate's calculated persistent increment is therefore:

| Component | Per-rank increment |
|---|---:|
| FP32 linear-attention output-projection accumulation weights | 1,440 MiB |
| Linear-attention state at capacity eight | 295.5 MiB |
| Fused BF16 logical-half A/B projection weights | 22.5 MiB |
| Convolution and scalar state parameters | less than 2 MiB |
| **Calculated subtotal** | **about 1,760 MiB** |

These are calculated logical sizes, not physical-memory evidence. The worker
must report allocator-observed peak allocated and reserved bytes, including
temporary migration storage and the measured-layer paired-arm overhead.
Temporary-release proof is object-lifecycle based: the worker records weak
references to all eight per-rank `all_gather` destination tensors, removes
every strong reference, synchronizes the device, and requires zero live
temporary tensors before warmup. Allocator-observed steady bytes remain a
separate physical measurement because CUDA allocation bins need not equal the
logical tensor byte count.

### 8.2 Prefill policy

Stage 0 preserves the existing TP4 prefill path. It does not duplicate the
linear-attention core over long prompts merely to simplify state ownership.

Before the first candidate decode step, the worker transforms the active
request's TP4 state quarters into logical TP2 halves:

```text
logical half 0 = TP4 quarters 0 + 1
logical half 1 = TP4 quarters 2 + 3

pair A ranks receive logical halves 0 and 1
pair B ranks receive the same logical halves 0 and 1
```

The transformation includes convolution and recurrent state for all 48
linear-attention layers. It is generation-bound and request-bound.

### 8.3 Migration accounting

State migration is not hidden from the result. The gate records:

- migration latency median, P95, and P99;
- bytes read, transferred, and retained per rank;
- temporary peak allocated and reserved bytes;
- steady-state incremental bytes;
- state digest before and after transformation; and
- break-even output-token count:

```text
ceil(median migration latency /
     median per-token candidate savings)
```

If candidate per-token savings are non-positive, break-even is infinite and
the candidate cannot pass.

The isolated migration microbenchmark aligns all four ranks immediately
before recording its start event. The alignment is outside the measured
interval and prevents per-rank CPU digest and garbage-collection work from
being misclassified as state-transfer latency. The measured interval still
contains the complete state collectives, layout conversion, and retained
candidate-state creation.

### 8.4 Lifecycle

The logical TP2 state remains owned by the same request lease and generation
as the TP4 source state. It may be published only after both pair replicas
produce valid candidate state for the same decode step.

Abort, slot reuse, prefix restoration, and speculative rollback are outside
the Stage-0 integration scope, but the microgate must prove:

- no state publication before the timed step succeeds;
- baseline state remains unchanged during candidate execution;
- stale generation identifiers are rejected;
- candidate state cannot be reused by a different request identity; and
- all temporary state is retired before process-group destruction.

Production integration remains prohibited until those lifecycle operations
receive their own plan and tests.

## 9. Stage-0 microgate

### 9.1 Scope

Stage 0 runs on four real GPUs with one real checkpoint-backed
linear-attention layer. It covers the complete mixer boundary:

```text
normalized hidden
  -> input projections
  -> convolution state update
  -> recurrent delta rule
  -> gated RMSNorm
  -> output projection
  -> collective materialization
  -> BF16 mixer output and candidate states
```

It excludes the unchanged residual, post-attention RMSNorm, MLP, scheduler,
sampling, KV cache, and HTTP serving layers. Therefore a Stage-0 pass is
mechanism evidence only, not an end-to-end claim.

The representative layer is layer 0, which is a linear-attention layer in the
pinned checkpoint. The worker must load its real checkpoint parameters and
record the exact tensor bindings.

### 9.2 Arms

Each measured pair has two arms:

1. **Global TP4 baseline**
   - global TP4 quarter heads and state;
   - current FP32 output-projection accumulation;
   - current global four-rank FP32 AllReduce.
2. **Topology-local TP2 island**
   - logical TP2 half heads and state in both replicas;
   - FP32 output-projection accumulation;
   - pair-local two-rank FP32 AllReduce.

Both arms start from cloned, digest-bound hidden and state inputs. Execution
order alternates by pair index. Untimed correctness projection occurs after
timing.

### 9.3 Shapes and repetitions

The frozen matrix is:

- active-token groups: 1, 4, and 8;
- two warmup pairs per shape;
- 15 measured pairs per shape;
- four ranks per pair;
- a separate state-migration matrix with two warmups and 15 measured
  repetitions across all four ranks; and
- no profiler enabled during formal timing.

The worker may run a small untimed smoke before the immutable attempt. It may
not repair or reuse a failed formal attempt.

### 9.4 Timing

CUDA event timing surrounds the full mixer path for each arm. The pair latency
is the maximum rank latency across all four global ranks, because the next
global operation cannot safely proceed until both replicas have completed.

Report:

- median, P90, P95, and P99 critical latency;
- ratio-of-medians speedup;
- median paired speedup;
- improving-pair count;
- pair A and pair B latency independently;
- output-projection GEMM time;
- collective time;
- linear-attention core time;
- host submission median and P99;
- migration latency from the separate 60-row measured matrix and break-even
  token count for each active-token shape; and
- peak allocated and reserved bytes by rank.

Profiler-derived component times are diagnostic only. The classifier uses the
unprofiled enclosing critical interval.

## 10. Correctness gates

All correctness gates are mandatory:

1. Every output and state tensor is finite.
2. Each pair replica's logical input and parameter digests match its partner
   replica.
3. Pair A and pair B mixer outputs agree within
   `atol=2e-4, rtol=2e-4`.
4. Pair A and pair B candidate convolution and recurrent states agree within
   `atol=2e-4, rtol=2e-4`.
5. Candidate versus TP4 baseline mixer output is within
   `atol=2e-2, rtol=2e-3`.
6. Candidate versus TP4 baseline convolution and recurrent states are within
   `atol=2e-2, rtol=2e-3` after conversion to the same logical full-head
   order.
7. The projected full hidden state produces the same greedy argmax in a
   frozen downstream projection check.
8. Request, generation, layer, pair, logical-rank, and state-slot identities
   agree with the manifest.
9. No timed-path allocation occurs after warmup.
10. All four ranks complete every row without timeout or hidden fallback.

Any correctness, identity, lifecycle, or rank-completion failure terminates
the attempt as `NO_GO_CORRECTNESS_OR_LIFECYCLE`.

## 11. Performance and cost gates

Stage 0 returns `GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE` only if every
correctness gate passes and all of the following hold:

- active-token 1 ratio-of-medians critical-latency speedup is at least `5%`;
- active-token 4 and 8 geometric aggregate speedup is at least `5%`;
- neither active-token 4 nor 8 median regresses;
- no shape's P99 critical latency regresses by more than `3%`;
- at least 11 of 15 pairs improve for active-token 4 and for active-token 8;
- host-submission median does not regress by more than `10%` for any shape;
- state-migration median break-even is at most 32 generated tokens for every
  shape;
- projected integrated steady-state incremental allocated memory is at most
  1,920 MiB per rank at capacity eight;
- peak allocated memory remains below `98%` of physical memory on every rank;
- temporary migration storage is released before steady-state timing; and
- cleanup leaves no owned child, process group, CUDA IPC handle, or task file
  outside the approved attempt root.

The 1,920 MiB steady-state ceiling permits the calculated approximately
1,737.5 MiB weight-and-state increment plus bounded allocator overhead. It
does not permit another complete dense output-projection copy or retention of
both TP4 and TP2 FP32 accumulation weights for all 48 layers.

## 12. Classification

The producer and independent verifier reconstruct one of:

- `GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE`
- `NO_GO_CORRECTNESS_OR_LIFECYCLE`
- `NO_GO_PERFORMANCE`
- `NO_GO_MEMORY`
- `NO_GO_MIGRATION_AMORTIZATION`
- `BLOCKED_ADMISSION`
- `INVALID_EVIDENCE`

Only the exact `GO` classification authorizes an end-to-end integration
design. A `GO` does not itself establish improved TPOT, TTFT, QPS, or
whole-model latency.

A performance no-go terminates that exact frozen candidate revision for this
checkpoint and hardware topology. It may not be rescued by rerunning until a
lucky sample appears, dropping active-token shapes, ignoring migration,
weakening tail gates, or reporting only the faster pair. A later revision may
continue only after a measured root cause motivates a source change, and it
must use a fresh immutable attempt tag and preserve the failed evidence.

## 13. Stage-1 boundary after a microgate GO

If and only if Stage 0 passes, a separate reviewed Stage-1 plan may integrate:

- topology-aware pair-group construction;
- logical TP views for linear-attention-only projections;
- active-slot TP4-to-TP2 state migration;
- paired state ownership and transaction semantics;
- prefix-cache restore conversion;
- speculative rollback and abort;
- CUDA graph capture compatibility;
- full 64-layer decode;
- exact greedy token agreement across all global ranks; and
- the frozen P0/P1/Q0/Q1/Q2 end-to-end workload suite.

The Stage-1 end-to-end gate must report benefit and cost together:

- TPOT, TTFT, E2E latency, output tokens/s, and request QPS;
- P50/P95/P99 tails;
- migration latency and observed break-even;
- peak allocated and reserved memory;
- power/utilization;
- exact-token and numeric correctness;
- prefix-cache and lifecycle correctness; and
- cleanup.

TP2×2 whole-model replicas must be measured as a service-topology control for
concurrent workloads, but cannot replace the same-request TP4 baseline.

## 14. Borrowed components and original contribution boundary

This design does not claim that tensor parallelism, process subgroups,
topology-aware placement, redundant computation, or heterogeneous parallel
layouts are new.

The project-specific contribution being tested is the composition:

1. retain one global TP4 request and hidden-state timeline;
2. exploit Qwen3.8's 48/16 hybrid layer structure;
3. execute only linear-attention mixers as two topology-local TP2 replicas;
4. rejoin the unchanged global TP4 path without a cross-pair activation
   exchange;
5. reuse already replicated input weights and preserved dense output weights;
6. transform state once at the prefill-to-decode boundary; and
7. classify the design using same-request benefit, duplicated-compute cost,
   state-memory cost, migration break-even, and tail latency.

That exact end-to-end composition may be original within TinyLLMForge. No
claim of first publication or global novelty is permitted without a separate
literature and prior-art review.

## 15. Non-goals

Stage 0 does not:

- edit the production `RowParallelLinear` path;
- change full-attention TP4 behavior;
- change the replicated MLP;
- run two independently scheduled serving replicas;
- change sampling, request routing, or admission policy;
- optimize prefill;
- enable prefix cache, speculative decoding, quantization, KV offload, or
  sparse attention;
- claim a whole-model or production speedup;
- infer performance from a PID, profiler trace, complete manifest, passing
  correctness suite, or verifier alone;
- write task-owned files to remote `/` or `/tmp`; or
- terminate, pause, adopt, or modify foreign workloads.

## 16. Evidence and storage

The formal attempt uses a fresh immutable tag under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/attempts/
```

Large source snapshots, raw rows, caches, traces, and temporary tensors remain
remote. Only the compact sealed final bundle is downloaded under:

```text
artifacts/qwen38_topology_local_tp2_islands/
  20260908-qwen38-topology-local-tp2-island-stage0-r1/final_bundle/
```

The bundle must contain:

- admission and topology records;
- model and source identity;
- frozen workload manifest;
- parameter-slice manifest;
- state-layout and migration manifest;
- correctness rows;
- paired timing rows;
- component diagnostic rows;
- memory rows;
- lifecycle rows;
- cleanup record;
- producer result;
- remote independent verification;
- downloaded local independent verification;
- terminal report;
- manifest; and
- `manifest.sha256`.

The verifier independently reconstructs every threshold and classification
from raw rows. Manifest completeness is not a substitute for semantic
coverage.

## 17. Expected implementation boundaries

The implementation plan should prefer new model-neutral or Qwen3.8-specific
helpers over changing global linear-layer semantics. Expected boundaries are:

- a topology-local pair identity and process-group helper;
- logical TP projection-view helpers;
- a deterministic TP4-to-TP2 state transformer;
- a checkpoint-backed Stage-0 worker;
- controller, assembler, and independent verifier;
- focused CPU tests for topology, slicing, state order, classification, and
  evidence integrity; and
- four-rank GPU correctness and performance evidence.

No implementation task may broaden the candidate into a production model path
before the Stage-0 result is sealed and independently verified.
