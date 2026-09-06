# TP4 Segmented Exact Decode Graph Program Design

**Date:** 2026-09-06

**Status:** Approved direction; capture-cost census required before runtime
integration

**Primary model:** Qwen/Qwen3.8-27B BF16, tensor parallel size 4

**Runtime scope:** exact multi-sequence decode CUDA Graphs using the
`lease_pool_index_v1` Qwen3.8 hybrid-state path

## Problem

The r54 strict-clean Q1 smoke proved that the dynamic pool-index protocol is
correct at the eager boundary but cannot enter replay:

- exact outputs matched for all eight requests;
- all four ranks attempted the same graph program;
- the TP-wide capture duration was `4_019_119_030 ns`;
- the frozen single-capture ceiling is `2_000_000_000 ns`;
- the cache therefore rejected the program with
  `single_capture_budget`;
- measured graph dispatches and cross-lease replays were both zero.

Capture receipts localize the cost. On rank zero, approximately:

- `76.7 ms` elapsed before the hot-path prerequisite receipt;
- `4.8 ms` elapsed before graph capture began;
- `3.541 s` elapsed inside the CUDA Graph capture body;
- `396.7 ms` elapsed in capture-end synchronization;
- `51.1 ms` elapsed restoring transactional state and scratch KV.

The dominant cost is therefore the captured 64-layer model program, not
logging, host preparation, or rollback. The current graph contains embedding,
all decoder layers, tensor-indexed hybrid-state gather and commit, final norm,
and LM head in one capture.

The next optimization must reduce the size of each real CUDA Graph capture.
It must not hide setup work outside the measured interval, weaken the
two-second ceiling, or trade away exactness and steady-state performance.

## Goals

1. Replace one monolithic 64-layer capture with a bounded composite program of
   contiguous CUDA Graph segments.
2. Keep every individual segment capture at or below
   `2_000_000_000 ns`.
3. Keep the complete per-rank capture lifecycle at or below
   `5_000_000_000 ns`.
4. Preserve exact output equality, ordered lease validation, dynamic physical
   state-slot selection, and isolation of unselected state slots.
5. Preserve one logical decode step: the segment graphs replay in order on
   one CUDA stream without host synchronization between segments.
6. Make all additional cost visible: segment capture times, total capture
   time, retained memory, first-capture request latency, replay launch
   overhead, TTFT, TPOT, P99 E2E, and throughput.
7. Stop before broad runtime integration if a bounded hardware census cannot
   establish sufficient capture-budget headroom.

## Non-Goals

- Do not change the frozen correctness, capture, replay-coverage, memory,
  throughput, latency, TTL, or GPU-admission gates.
- Do not change the ordinary `forward_v1` or
  `lease_transaction_v1` paths.
- Do not change exact-prefill graphs, speculative verification, Exact Greedy
  K8, KV offload, Quest, graph-resident greedy tail, or sampling semantics.
- Do not move state gather, state commit, final norm, or LM head to an
  unmeasured host-side path merely to pass the capture limit.
- Do not retain warmup graphs across the warmup/measured reset boundary.
- Do not make segmented capture production-default.
- Do not modify or reclassify r50, r52, or r54 evidence.
- Do not claim an end-to-end benefit from a capture-only census.

## Considered Approaches

### A. Protocol-specific prewarm before capture

Run the pool-index path eagerly with the exact capture shapes immediately
before entering `torch.cuda.graph(...)`, forcing lazy kernels, allocators, and
collective setup to become hot.

Advantages:

- small implementation surface;
- useful as a diagnostic control;
- may reduce one-time lazy setup inside capture.

Costs and risks:

- r54 already reaches capture from a successful hot eager path;
- approximately 88% of the measured duration is inside the capture body;
- moving initialization earlier increases first-request or setup cost even if
  the recorded graph interval becomes shorter;
- it is unlikely to reduce a roughly four-second lifecycle below two seconds.

This is retained only as a census control. It is not the selected
optimization.

### B. Capture only the model core

Gather hybrid state before graph replay, capture only decoder computation, and
commit state after replay.

Advantages:

- smaller captured program;
- simpler graph body and fewer captured indexing operations.

Costs and risks:

- adds per-token launches and state movement outside the graph;
- weakens the existing one-program transactional shape;
- creates new host-visible failure boundaries between compute and commit;
- can pass the capture gate while regressing TPOT or tail latency.

This is not selected because it optimizes the measured setup metric by adding
steady-state work to every token.

### C. Composite contiguous graph segments

Capture embedding, decoder-layer ranges, final norm, LM head, and state commit
as an ordered program of two to four CUDA Graphs. Stable device buffers connect
the segments. Hybrid-state candidates remain graph-owned until the final
commit stage.

Advantages:

- directly reduces the amount of work in each individual capture;
- retains device-side pool-index gather and commit;
- adds only bounded graph-launch overhead during replay;
- preserves the existing cache, identity, and lease-manifest model through a
  wrapper that still exposes `replay()`, `reset()`, and `pool()`;
- permits an early hardware stop rule before broad integration.

Costs and risks:

- total capture time may increase because each segment has capture-finalization
  overhead;
- stable hidden and candidate buffers increase retained memory;
- multiple graph launches may regress TPOT or P99;
- replay failure after an earlier segment has been enqueued cannot fall back
  to eager execution.

This is the selected direction.

## Stage 0: Bounded Capture-Cost Census

No production dispatch or cache behavior may change until a source-bound
strict-clean census evaluates candidate segment plans.

### Candidate plans

The census evaluates contiguous two-, three-, and four-compute-segment plans.
A separate commit graph, when required, is an additional captured segment. A
plan must:

- cover every layer exactly once and in model order;
- place embedding only in the first segment;
- place final norm and LM head only in the last compute segment;
- place the pool-index state commit only in the final commit stage;
- use the same token, position, attention-context, state-slot, and scratch-KV
  shapes as the r54 Q1 graph arm;
- use one CUDA memory pool per composite program;
- restore state and scratch KV after every candidate plan.

The initial equal-layer partitions are:

```text
2 segments: [0, 32), [32, 64)
3 segments: [0, 22), [22, 43), [43, 64)
4 segments: [0, 16), [16, 32), [32, 48), [48, 64)
```

The last segment includes final norm and LM head, so the census may move only
the final interior boundary earlier to balance measured capture duration.
Any adjusted plan must remain deterministic and be recorded verbatim in the
artifact. The census may test at most one adjusted plan per segment count.

### Census measurements

For every rank, plan, and captured segment, record:

- segment ordinal and exact half-open layer range;
- whether embedding, final norm, LM head, and state commit are included;
- capture-body duration;
- capture-end synchronization duration;
- complete segment capture duration;
- complete program capture lifecycle duration;
- allocated and reserved memory deltas;
- stable boundary-buffer bytes;
- exact output equality after one stitched replay;
- selected-slot state equality and unselected-slot immutability;
- cleanup and graph reset completion.

The complete program lifecycle timer starts before static tensor allocation
and snapshot creation and stops only after state and scratch-KV restoration.
No setup work may be excluded from this total.

### Census selection rule

Select the smallest compute-segment count that satisfies all of:

- every TP-wide segment duration is at most `1_800_000_000 ns`;
- TP-wide complete program capture lifecycle is at most
  `4_500_000_000 ns`;
- one stitched replay is exactly equal to eager output;
- selected hybrid-state slots equal eager post-state;
- every unselected hybrid-state slot is unchanged;
- cleanup is `CLEAN`.

The `1.8 s` and `4.5 s` census thresholds deliberately preserve 10% headroom
inside the frozen `2 s` and `5 s` production gates. If no plan passes, publish
`NO_GO_SEGMENTED_CAPTURE_CEILING` and stop without runtime integration.

The census is diagnostic evidence only. It cannot authorize a performance
claim or replace the Q1 smoke.

## Selected Runtime Architecture

### 1. Segment plan identity

Introduce an immutable segment plan containing:

- schema version;
- model layer count;
- ordered half-open layer ranges;
- embedding owner segment;
- final norm/LM-head owner segment;
- state-commit owner stage;
- segment-plan SHA-256.

The selected plan hash becomes part of the stable graph-program cache key.
Two different partitions must never share one cache entry.

The segmented protocol is explicitly named
`lease_pool_index_segmented_v1`. Existing protocols retain their current
identity and cache-key behavior.

### 2. Layer-range execution

`Qwen35PackedHeterogeneousLayerStack` gains a range-bounded pool-index
execution primitive. It accepts:

- `start_layer` and `end_layer`;
- fixed-shape token counts and position IDs;
- a stable input hidden-state tensor;
- a stable output hidden-state tensor;
- the graph-owned physical state-slot tensor;
- graph-owned candidate storage for linear-attention layers in the range.

The primitive executes exactly the layers in `[start_layer, end_layer)`.
Linear-attention adapters are selected by their actual model-layer indices,
not by assuming every layer is stateful. Each layer's candidate is written to
the stable storage assigned to that layer. No segment commits state.

The first segment performs embedding before its layer range. Intermediate
segments consume the preceding stable hidden buffer. The final compute segment
runs final norm and LM head and writes stable logits.

### 3. Deferred state commit

All pool-index state gathers remain inside their owning compute segments.
Candidate convolution and recurrent states are retained in graph-owned stable
buffers.

A final commit graph performs tensor-indexed commit for every stateful layer
using the same validated `state_slot_ids` tensor. This preserves the semantic
boundary that no pool state becomes authoritative until all compute segments
have been enqueued successfully.

The commit graph is part of the composite program and has its own capture
duration row. It counts toward both the single-capture and total-capture
budgets.

If the selected census shows that folding commit into the final compute
segment remains below the internal `1.8 s` limit, the implementation may use
that simpler form. The artifact must state whether commit is folded or
separate.

### 4. Stable boundary buffers

The composite entry owns:

- existing static input, position, attention-context, and state-slot tensors;
- one hidden-state boundary tensor per inter-segment edge;
- stable candidate tensors for every stateful layer;
- stable final logits;
- the ordered graph objects;
- the selected segment plan and per-segment capture measurements.

Segments communicate only through these fixed device addresses. They do not
allocate boundary tensors during replay and do not read host-mutated state
after replay begins.

### 5. Composite graph wrapper

Add a small `CompositeExactCudaGraph` object with:

```text
replay() -> enqueue every segment graph in order on the current stream
reset()  -> reset every segment graph exactly once
pool()   -> return the shared capture pool handle
```

`replay()` performs no synchronization between segments. If any enqueue raises,
the cache entry is disabled, the request fails, and eager retry is forbidden.
The engine must not continue serving from a potentially partially executed
request state.

The wrapper lets the existing cache and replay call site retain one logical
entry while making the number of underlying CUDA Graphs explicit in evidence.

### 6. Capture and budget accounting

`ExactCudaGraphEntry` gains explicit composite accounting:

- ordered `segment_capture_durations_ns`;
- `max_segment_capture_duration_ns`;
- `total_capture_duration_ns`;
- segment-plan hash, compute-segment count, and total captured-graph count.

For legacy entries, the one existing duration populates all three views.

TP4 synchronization performs an element-wise maximum across ranks for every
segment duration. Production admission then requires:

```text
max(TP-wide segment durations) <= 2_000_000_000 ns
sum(TP-wide segment durations and measured lifecycle overhead)
    <= 5_000_000_000 ns
```

The cache's process-wide total-capture accounting adds the complete program
capture duration, not only the largest segment. A rejected composite resets
all already captured segment graphs and never becomes replayable.

### 7. Capture transaction

Capture remains post-success on the hot eager path:

1. validate the full lease manifest;
2. allocate and populate all stable tensors;
3. snapshot selected hybrid state and scratch KV;
4. set the exact decode context;
5. capture each compute segment in order using one shared pool;
6. capture or fold the final state commit;
7. synchronize and record every segment;
8. restore hybrid state and scratch KV;
9. rebuild and compare invocation identity and segment-plan identity;
10. TP-synchronize the duration vector;
11. atomically commit or reject the complete composite entry.

No partially captured plan may enter the ready cache.

### 8. Replay transaction

Replay retains the current pool-index protocol order:

1. rebuild the full invocation identity;
2. validate ordered `slot_id + generation + request_id`;
3. verify the stable program key and segment-plan hash;
4. copy static inputs and current physical slot IDs;
5. set the exact decode context;
6. call the composite wrapper once;
7. consume stable logits;
8. reset context and record replay evidence.

The underlying segment launches occur consecutively on one stream. There is no
host synchronization or per-segment lease check between them.

## Failure Semantics

- Invalid or non-covering segment plan: reject before capture.
- Segment capture failure: reset all captured segment graphs, restore state and
  scratch KV, then reject with `capture_failed`.
- State or scratch restore failure: preserve `scratch_unavailable` and the
  original chained failures.
- Segment above `2 s`: reject the whole entry with
  `single_capture_budget`.
- Complete program above `5 s`: reject the whole entry with
  `total_capture_budget`.
- TP disagreement in segment count, plan hash, or duration-vector shape:
  reject before cache commit.
- Replay identity or lease-manifest drift: disable the entry before launch.
- Replay enqueue failure after the first segment: disable the entry, fail the
  request, and do not eager-retry.
- Graph reset failure: fail the phase boundary; measured evidence is invalid.
- Census failure: stop before production integration.

## Configuration

Add a strict default-false feature:

```text
multi_sequence_cuda_graph_segmented_capture = false
```

It is valid only when:

- `multi_sequence_cuda_graphs` is enabled;
- `multi_sequence_cuda_graph_dynamic_pool_indices` is enabled;
- the model exposes the segmented pool-index hooks;
- decode is exact greedy and otherwise eligible for the current graph path.

The selected segment plan is source-controlled and model-profile-bound. There
is no runtime autotuning in the request path.

## Verification Strategy

### Pure and CPU-level tests

- segment plans are canonical, contiguous, non-overlapping, and cover all
  layers exactly once;
- plan hashes change when any boundary or stage owner changes;
- stateful adapter selection follows actual model-layer indices;
- composite replay and reset preserve order and are idempotent where required;
- legacy one-graph entries retain current budget behavior;
- composite entries use max-segment for the single limit and complete-program
  duration for the total limit;
- partial capture cannot enter the ready cache;
- replay failure disables the entry and cannot eager-fallback;
- feature dependencies fail closed.

### CUDA integration tests

- segment boundary buffers retain stable addresses;
- each segment captures and replays on one stream without an inter-segment
  synchronization;
- outputs equal eager execution exactly;
- selected hybrid-state post-state equals eager;
- unselected state slots remain byte-identical;
- scratch KV is restored after capture;
- all segment graphs reset during phase reset.

### Remote evidence ladder

1. Run the bounded strict-clean capture-cost census.
2. If and only if the census passes, integrate the selected plan.
3. Run a fresh strict-clean Q1 smoke with a new immutable tag.
4. Require exactly `SMOKE_PASS` before creating a new full-gate tag.
5. Run the complete schema-v2 8-pair gate.
6. Require exactly `GO_STAGE1_JUSTIFIED` for a positive claim.
7. Run producer, remote independent verifier, local independent verifier,
   manifest, post-verification hash, cleanup, audit, handoff, commit, push, and
   remote SHA checks.

## Success Criteria

The optimization is successful only if the fresh full gate proves:

- every individual TP-wide segment capture is at most `2_000_000_000 ns`;
- complete per-rank capture cost is at most `5_000_000_000 ns`;
- replay coverage is at least `0.80`;
- cross-lease replay is exercised;
- exact token and text output equality holds;
- selected state equals eager and unselected state is unchanged;
- added allocated and reserved memory remain within the frozen gate;
- throughput, TTFT, median TPOT, and P99 E2E do not regress under the existing
  acceptance rules;
- cleanup is `CLEAN`;
- both independent verifiers agree on `GO_STAGE1_JUSTIFIED`.

Anything less remains diagnostic or negative evidence.

## Benefit and Cost Contract

A positive report must state both:

- **benefit:** capture admission, replay coverage, throughput, TTFT, TPOT, and
  P99 E2E relative to eager;
- **cost:** segment count, additional graph launches per token, total capture
  lifecycle, first-capture request latency, retained boundary/candidate memory,
  peak allocated/reserved memory, and any failure quarantine behavior.

Passing the capture ceiling alone is not a performance win.
