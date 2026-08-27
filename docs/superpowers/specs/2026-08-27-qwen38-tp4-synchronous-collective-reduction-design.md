# Qwen3.8 TP4 Synchronous Collective Reduction Design

**Date:** 2026-08-27

**Status:** Approved direction; this document authorizes qualification only

**Stage-1 model:** Qwen/Qwen3.8-27B

**Primary target:** decode TPOT and P99 latency on TP4 without communication
and computation overlap

## Objective

Determine whether TinyLLMForge can materially reduce Qwen3.8-27B TP4 decode
latency by eliminating, packing, or avoiding synchronous tensor-parallel
collectives.

The first stage builds a low-disturbance collective inventory and a static
consumer-dependency proof. It does not modify collective execution. A
production optimization is authorized only when the qualification gate
identifies one exact transformation with:

- an explicit numerical-equivalence argument;
- an explicit producer/consumer and rank-ownership argument;
- an attributable decode critical-path reduction ceiling of at least 5%;
- measured profiler overhead within the frozen limits; and
- no use of asynchronous collectives or communication/computation overlap.

The final optimization must report benefit and cost together. Benefit means
paired TPOT, P99, throughput, and collective-count or collective-byte
improvement. Cost means additional device memory, prefill impact, capture or
startup cost, supported topology, and implementation complexity.

## Frozen Non-Goals

The terminal communication-exposure campaign classified the previous
direction as `INCONCLUSIVE_LOW_HEADROOM`. The following boundaries remain
frozen:

```text
OVERLAP_DESIGN_AUTHORIZED=false
ASYNC_COLLECTIVES_AUTHORIZED=false
```

This design does not authorize:

- `async_op=True`;
- pending work handles;
- a dedicated communication stream;
- dual-stream execution;
- new CUDA Event dependencies between compute and communication;
- chunked ReduceScatter or AllGather;
- overlap-aware scheduling;
- speculative execution across an unresolved collective;
- a new Nsight replay campaign;
- retuning the qualification thresholds after observing terminal results.

Synchronous transformation is not synonymous with overlap. A later
implementation may change the number, payload, or materialization of
blocking collectives only after this qualification returns
`GO_SYNC_COLLECTIVE_REDUCTION`.

## Existing Runtime Boundary

The current production TP path is synchronous:

```text
VocabParallelEmbedding
  -> local embedding lookup
  -> blocking AllReduce

Qwen3/Qwen3.5 attention output
  -> local RowParallelLinear
  -> blocking AllReduce
  -> residual / normalization consumer

Qwen3/Qwen3.5 MLP output
  -> local RowParallelLinear
  -> blocking AllReduce
  -> next-layer residual / normalization consumer

Qwen3.8 exact-full-vocabulary greedy selection
  -> rank-0 full-vocabulary logits
  -> rank-0 float32 argmax
  -> blocking token-ID Broadcast
  -> next-token input on every rank
```

For Qwen3.8-27B, the text model contains 64 decoder layers. The ordinary
decode topology therefore has an expected lower-level pattern of one
embedding collective plus two row-parallel collectives per decoder layer,
followed by one greedy-token broadcast:

```text
1 + 2 * 64 + 1 = 130 synchronous collective boundaries per decode token
```

This count is a topology expectation, not terminal runtime evidence. The
qualification campaign must reconstruct the exact count from all four ranks
and reject rank disagreement.

The two row-parallel results in each decoder layer have immediate consumers.
Cross-layer packing is therefore not presumed legal: the next computation
cannot begin from an unresolved partial result. Qualification must prove a
consumer-independent packing window rather than infer one from repeated
operation names.

Qwen3.8's `exact_full_vocab` path also gathers the complete LM-head weight to
rank 0 during initialization. That setup collective and its retained full
weight are startup and memory costs, not per-token decode collectives. The
qualification inventory records them separately and must not attribute their
one-time duration to steady-state TPOT.

The embedding collective has a different trade-off. Replicating a
248,320-by-5,120 BF16 embedding would consume about 2.37 GiB per rank in
total, versus about 0.59 GiB per rank for TP4 sharding, or roughly 1.78 GiB
additional memory per rank before allocator effects. That cost may remove
only one collective per decode token. The gate must measure its absolute
critical-path contribution before any replication design is allowed.

## Considered Approaches

### A. Static dependency proof plus sampled CUDA-event census

Build a model-agnostic collective census that records every synchronous
collective's stable identity, payload, rank, layer, and consumer boundary.
Use CUDA events only for a bounded rotating subset of collectives in selected
steady-decode steps. Calibrate the sampling budget against matched
uninstrumented controls before collecting terminal evidence.

Advantages:

- avoids the 30.77% median and 38.19% maximum Nsight perturbation observed in
  the previous campaign;
- separates exact call-count and byte evidence from sampled duration
  evidence;
- can identify a legal elimination or packing window without changing
  runtime semantics;
- reuses existing operation names and decode-step identities;
- can fail closed before any production collective mutation.

Costs:

- rotating cohorts require multiple matched runs for full timing coverage;
- sampled per-call timings are estimates, not a continuous timeline;
- static consumer analysis must be maintained alongside runtime identities;
- a low sampling budget may yield an inconclusive result.

This is the selected approach.

### B. Full CUDA-event timing of every operation and collective

Reuse `DecodeInternalProfiler` unchanged and enable all layer, operation,
step, and collective events.

Advantages:

- already has structured rank, step, layer, operation, shape, dtype, and CUDA
  duration output;
- produces a complete per-step decomposition in one run.

Costs:

- creates CUDA events around substantially more than the 130 collective
  boundaries per token;
- profiles compute that this qualification does not need;
- has a higher event-allocation, Python bookkeeping, and finalization burden;
- does not independently prove that repeated collectives can be combined.

This approach is retained only as a non-terminal debugging mode. Its output
cannot authorize implementation unless it independently passes the same
overhead gate.

### C. Implement an embedding or row-parallel transformation first

Immediately replicate embeddings, pack selected buffers, or change a
collective primitive, then compare end-to-end performance.

Advantages:

- produces a concrete candidate quickly;
- avoids building a separate qualification profiler.

Costs:

- risks spending substantial implementation and GPU time on a sub-1% target;
- can accidentally change numerical order, rank ownership, or memory limits;
- makes a negative result ambiguous between a weak target and a flawed
  implementation;
- violates the evidence-first boundary established after the perturbed
  Nsight campaign.

This approach is rejected.

## Qualification Architecture

### 1. Static collective catalog

Create a dependency-light catalog from the instantiated model topology. Each
entry describes one logical collective site:

```text
site_id
module_path
layer_index
layer_role
operation_name
collective_kind
process_group
expected_calls_per_decode_step
local_tensor_shape_formula
local_tensor_dtype
producer
first_consumer
requires_replicated_result
packing_window
elimination_precondition
```

`site_id` is derived only from runtime roles and module paths. It must not
contain a workload, prompt, checkpoint revision, or rank.

The catalog classifies each site into exactly one of:

```text
MANDATORY_IMMEDIATE_CONSUMER
REDUNDANT_IF_PRECONDITION_PROVEN
PACKABLE_IF_WINDOW_PROVEN
MATERIALIZATION_ALTERNATIVE
UNRESOLVED
```

Classification is conservative:

- a row-parallel output consumed by residual addition or normalization before
  another collective is `MANDATORY_IMMEDIATE_CONSUMER`;
- identical operation names do not imply packability;
- rank-local equality does not imply replicated-value equality;
- a materialization alternative must name the extra memory and the consumer
  layout it changes;
- an unresolved consumer or alias makes the site `UNRESOLVED`.

The static catalog is a proof input, not performance evidence.

### 2. Runtime count-and-byte census

Add a separate lightweight observer to the existing `profile_collective`
boundary. Route every steady-decode collective through that boundary,
including the greedy-token broadcast in
`select_tensor_parallel_greedy_tokens()`. Initialization-only parameter
gathers use the same metadata schema but are assigned the `startup` phase and
are excluded from decode timing and count totals. When active, the observer
records:

```text
attempt
workload
repetition
rank
step identity
decode ordinal
site_id
operation name
collective kind
process group
execution phase
tensor shape
tensor dtype
tensor bytes
source stream identity
async_mode=false
```

The observer must:

- record only decode steps selected by the campaign;
- use preallocated or append-only bounded records;
- create no CUDA event for count-only observations;
- perform no synchronization;
- call the same blocking collective exactly once;
- preserve the existing broadcast source rank and gather destination rank;
- preserve the original tensor object and return value;
- reject `async_mode=True`;
- reject a missing or unknown static `site_id`;
- record no prompt text, token values, logits, or model weights.

The count-and-byte census is complete only when all four ranks report the
same ordered `site_id` sequence, collective kind, shape, dtype, and byte count
for every compared decode step.

### 3. Rotating CUDA-event sampler

Timing is enabled for a bounded subset of census ordinals. For a sampled
collective, the observer records one start event immediately before the
blocking call and one end event immediately after it on the current stream.
It does not synchronize inside the collective wrapper.

Sampling uses a deterministic cohort:

```text
cohort = hash(
    source_revision,
    attempt,
    workload,
    repetition,
    decode_ordinal,
) mod cohort_count
```

The selected cohort and ordinal membership are frozen in the campaign plan.
All ranks must sample the same logical sites. The verifier rejects missing,
extra, or rank-divergent timed sites.

Before the terminal campaign, calibrate event budgets:

```text
0, 8, 16, and 32 collective pairs per sampled decode step
```

Choose the largest budget satisfying both:

```text
median matched overhead <= 3.0%
maximum matched overhead <= 5.0%
```

The zero-event count-only arm must also pass these limits. If no nonzero
budget passes, duration qualification is
`INCONCLUSIVE_PROFILER_OVERHEAD`; count and byte evidence may still be
reported but cannot authorize implementation.

Events are resolved only after the measured request completes and the
campaign reaches an existing safe synchronization point. Event resolution
must not introduce a per-token or per-collective synchronization.

### 4. Consumer-window verifier

The independent verifier joins the static catalog with the runtime sequence.
For each potential transformation it reconstructs:

- the local producer;
- the blocking collective;
- every read of the collective result before the next candidate site;
- rank ownership and replicated/sharded state;
- aliasing or in-place mutation;
- payload layout and dtype;
- the first operation that requires the globally reduced value.

A packing candidate is legal only if two or more payloads are simultaneously
available before any candidate payload has a consumer. Their collective kind,
process group, dtype, and numerical reduction must match. The verifier must
name the exact contiguous or explicitly packed storage and the unpack cost.

An elimination candidate is legal only if the verifier proves that every
rank can construct the same required result without the collective. A
replication candidate must include exact persistent and peak-memory deltas.

No timing signal can override a failed consumer-window proof.

### 5. Reduction-ceiling estimator

For each legally reducible candidate, report:

```text
calls removed per decode step
bytes removed or repacked per decode step
sampled collective CUDA duration
packing/unpacking or local-compute estimate
additional persistent and peak device bytes
estimated TPOT reduction lower bound
estimated TPOT reduction upper bound
affected workloads
unsupported topologies
```

The lower bound subtracts all measured or conservatively bounded replacement
work. It also subtracts profiler uncertainty. The upper bound may assume the
entire sampled collective duration disappears, but it must be labeled a
ceiling rather than a predicted speedup.

The estimator must not sum durations from incompatible cohorts or treat
overlapped intervals as additive. This qualification targets blocking
collective removal, not exposure or overlap.

## Workloads and Pairing

Reuse the frozen Qwen3.8-27B TP4 workload definitions:

```text
P0: prompt=256,  output=128, concurrency=1, family=causal
P1: prompt=2048, output=128, concurrency=1, family=causal
Q0: prompt=256,  output=128, concurrency=4, family=online
Q1: prompt=256,  output=128, concurrency=8, family=online
Q2: prompt=2048, output=128, concurrency=4, family=online
```

Every arm uses:

- the same immutable model revision;
- the same runtime source;
- BF16;
- TP4;
- greedy temperature-zero decoding;
- exactly 128 output tokens;
- identical scheduler and CUDA Graph policy;
- the same physical GPU UUID-to-rank map within a pair;
- two warmups followed by five measured repetitions;
- alternating instrumented/control order.

The profiler calibration may use a strict subset of P0, P1, and Q1, but the
terminal count, byte, and timing coverage must include all five workloads.

## Correctness and Safety Invariants

Qualification must preserve:

- exact generated token IDs and decoded-text hashes;
- exact rank-local collective order;
- exact number of collective invocations;
- exact tensor shape, dtype, and byte count at each collective site;
- the original blocking call and current-stream execution;
- one model execution and one sampling decision per decode step;
- scheduler order, completion, and stop behavior;
- model weights, KV writes, attention math, and logits;
- strict-clean GPU admission and terminal cleanup.

The observer may not retain tensor references after the wrapper returns.
Malformed records, event failures, rank divergence, source drift, model
revision drift, or GPU identity drift fail the attempt closed.

Remote task data must remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

No task data may be written to remote `/`, remote `/tmp`, an old checkout, or
the retired adaptive-ngram checkout. The campaign does not run `kinit` or
`krenew`, signal unrelated processes, or claim GPUs used by another task.

## Classification

The producer and independent verifier must each return exactly one terminal
classification.

### `GO_SYNC_COLLECTIVE_REDUCTION`

All of the following are required:

```text
static catalog complete
all-rank runtime sequence exact
all-rank count and byte coverage complete
median matched profiler overhead <= 3.0%
maximum matched profiler overhead <= 5.0%
at least one consumer-window proof PASS
candidate lower-bound attributable TPOT opportunity >= 5.0%
correctness PASS
resource and cleanup PASS
producer/verifier classification agreement
```

Only the named candidate transformation is authorized. The classification
does not authorize a different collective, topology, model, or overlap
mechanism.

### `NO_GO_NO_REDUCIBLE_COLLECTIVE`

Use this classification when coverage and profiler cost pass, but every
candidate is mandatory, fails the consumer-window proof, or has a lower-bound
opportunity below 5%.

### `INCONCLUSIVE_PROFILER_OVERHEAD`

Use this classification when no nonzero timing budget satisfies the frozen
3% median and 5% maximum overhead limits.

### `INCONCLUSIVE_INCOMPLETE_COVERAGE`

Use this classification for missing ranks, missing sites, rank-divergent
sequences, unresolved catalog entries, incomplete workload coverage, source
drift, or missing terminal artifacts.

Thresholds are frozen before the terminal run and must not be retuned from
the result.

## Implementation Gate After Qualification

If and only if producer and verifier both return
`GO_SYNC_COLLECTIVE_REDUCTION`, write a second candidate-specific design.
That design must name:

- the exact collective sites changed;
- the before/after ownership and data flow;
- the exact numerical equivalence;
- the expected call and byte reduction;
- replacement local work;
- persistent and peak memory cost;
- prefill behavior;
- fallback and quarantine behavior;
- TP-size and model-support boundaries;
- a paired hardware promotion gate.

The implementation gate requires:

```text
exact token equality
equal argmax and bounded logit error under the existing correctness contract
median TPOT improvement >= 5.0%
P99 TPOT improvement >= 5.0%
throughput regression <= 1.0%
TTFT regression <= 1.0%
no workload median TPOT regression > 1.0%
collective count or bytes improve exactly as designed
memory and startup costs reported
```

If qualification returns any other classification, no production collective
mutation is authorized from this evidence.

## Artifact Contract

The terminal qualification bundle contains:

```text
source_identity.json
model_manifest.json
gpu_topology.json
workload_manifest.json
static_collective_catalog.json
consumer_dependency_proofs.json
profiler_calibration.json
collective_census.jsonl
collective_timing_samples.jsonl
paired_online_metrics.json
correctness.jsonl
resource_samples.jsonl
reduction_ceiling.json
classification.json
independent_verification.json
cleanup.json
manifest.sha256
```

The manifest rejects missing and extra terminal artifacts. Producer and
independent verifier separately recompute:

- rank sequence equality;
- count and byte totals;
- timing-cohort coverage;
- matched overhead;
- consumer-window legality;
- reduction ceilings;
- correctness;
- resource identity;
- cleanup;
- terminal classification.

Local verification may stream bounded immutable artifacts from the approved
remote bundle, but it must not require retaining all raw campaign data on the
Mac.

## Evidence Boundary

Dependency-light tests can prove schema validation, deterministic cohort
selection, rank-sequence reconciliation, consumer-window rules, arithmetic,
classification, and fail-closed behavior.

Only a source-bound four-GPU run can prove:

- runtime collective counts and bytes;
- profiler overhead;
- collective duration;
- end-to-end TPOT, P99, TTFT, and throughput;
- device-memory cost;
- exact multi-rank correctness;
- a candidate-specific performance opportunity.

`GO_SYNC_COLLECTIVE_REDUCTION` is authorization to design one measured
synchronous reduction candidate. It is not a speedup claim. A speedup claim
requires the later candidate implementation and its separate paired hardware
gate.
