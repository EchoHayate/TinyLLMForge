# Exact Burst GPU-Resident Continuation Epoch Design

**Date:** 2026-08-23
**Status:** Approved under the standing autonomous-optimization authorization
**Stage-1 model:** Qwen3-0.6B
**Primary target:** exact greedy batch-1 decode TPOT with bounded host-visible
token cadence

## Objective

Reduce the fixed setup cost paid before each exact greedy decode burst by
reusing the graph-resident token, position, context-length, slot, block-table,
and history-cursor state left by the preceding successful burst.

The optimization must preserve exact output tokens and bounded logit parity.
It remains default-disabled until a source-bound hardware gate proves both
benefit and cost. It must not weaken scheduler ownership, KV-block generation
checks, lease validation, or terminal quarantine behavior.

## Observed Opportunity

The current exact-burst graph advances all decode state on device:

```text
input_token
position
context_length
slot_mapping
history_index
```

After a successful burst, those tensors already describe the next decode
position. The next engine step nevertheless performs a full setup:

1. clear six static tensors and zero the history index;
2. fill four scalar tensors;
3. build a padded block table through a temporary CUDA tensor;
4. copy that block table into graph-owned storage;
5. replay the same complete-step graph.

The frozen Stage-1 workload uses a KV block size of 256 and generates 128
tokens. For its aligned prompts, the decode tail remains in one writable
block epoch. A K4 request therefore has one cold bind followed by 31
opportunities to reuse already-correct device state.

This is different from replay-aware metadata landing. That optimization
reduced ordinary per-token staging but still rebuilt every step from host
metadata. This design preserves the graph's own successfully advanced state
across engine calls.

## Considered Approaches

### A. Verified GPU-resident continuation epoch

Keep a host-owned receipt describing the expected next graph state. If the
next lease exactly matches that receipt and the same KV block epoch, replay
without clearing or rebinding graph state.

Advantages:

- removes nearly all repeated setup operations within a block epoch;
- reuses state already produced by the exact graph;
- retains the existing K4 host-publication cadence;
- fails closed before any current-step GPU mutation;
- adds only bounded history capacity and a small host receipt.

Costs and risks:

- correctness depends on strict continuity validation;
- token history must span the complete continuation epoch;
- any intervening ordinary decode, sequence switch, block change, graph
  generation change, or failed transaction invalidates reuse;
- state ownership becomes explicit across engine calls.

This is the selected approach.

### B. Packed replay seed slabs

Pack dynamic `int64` and `int32` controls into two pinned-host/device slabs and
perform two H2D copies per burst.

This has a narrower state-lifetime contract, but it still pays setup on every
burst. It remains the fallback research direction if continuation cannot pass
the correctness or lifecycle gate.

### C. Unrolled K4 supergraph

Capture four complete decode steps in one CUDA Graph and replay it once.

This removes graph-launch overhead, but capture time and graph-private memory
can grow substantially. The current evidence points to pre-replay setup as the
larger opportunity, so supergraph capture is deferred.

## Architecture

### 1. Continuation receipt

The exact-burst graph owns one optional immutable continuation receipt:

```text
sequence_id
graph_generation
block_table_identity
write_block_id
write_block_generation
next_input_token
next_position
next_context_length
next_physical_slot
history_cursor
```

The receipt describes what the graph-owned tensors must contain after the
last successful replay. It is not proof by itself; every continuation attempt
also validates the incoming lease and current graph identity.

The receipt is private to one graph instance. It is never serialized as
request authority and never replaces scheduler or block-manager validation.

### 2. Cold bind

A cold bind is required when no valid receipt exists or continuity is not
proven. It preserves the current setup behavior:

1. reset graph-owned static state;
2. bind initial token, position, context length, and physical slot;
3. copy the complete padded block table;
4. set `history_index` to zero;
5. replay the authorized token count;
6. publish only the newly written history slice;
7. install a continuation receipt after successful result construction.

Cold bind remains the semantic baseline and fallback.

### 3. Continuation hit

Before any GPU mutation, continuation requires all of:

- the feature flag is enabled;
- the ordinary exact-burst eligibility contract already passed;
- the previous receipt is valid and belongs to the same graph instance;
- sequence ID and graph generation match;
- the complete block-table identity matches;
- write block ID and generation match;
- incoming `initial_token` equals `next_input_token`;
- lease first write position equals `next_position`;
- lease initial sequence length equals `next_context_length`;
- lease first physical slot equals `next_physical_slot`;
- the authorized range remains in the same physical block;
- `history_cursor + authorized_token_count` fits retained history capacity;
- the graph is not quarantined.

On a hit, replay begins immediately from the existing graph-owned state.
There is no static reset, scalar fill, block-table construction, temporary
CUDA metadata tensor, or block-table copy.

The result reads exactly:

```text
token_history[
  history_cursor:
  history_cursor + authorized_token_count
]
```

After successful result construction, the receipt advances by the replay
count.

### 4. History capacity

Production token history capacity becomes one KV block:

```text
history_capacity = block_size
```

The history index is reset only on a cold bind. It advances monotonically
inside the epoch. The default block size of 256 retains 2,048 bytes of int64
token history, replacing the current 64-byte eight-token history.

The implementation must report exact retained-byte deltas. It must not grow
history without a configured bound.

### 5. Invalidation

The continuation receipt is invalidated on:

- graph capture or graph-generation replacement;
- sequence or block-table identity change;
- write-block generation change;
- scalar continuity mismatch;
- history-capacity exhaustion;
- graph replay exception;
- result-construction or final-token D2H exception;
- engine failure after replay and before scheduler commit;
- explicit graph quarantine.

An invalid receipt causes a cold bind when cold bind is still safe. A replay
failure or any failure after current-step KV mutation remains terminal and
must not retry or rebind the same step.

### 6. Engine and scheduler ownership

The scheduler continues to:

- decide exact-burst eligibility and width;
- create and own the pending lease;
- validate block generations and sequence state;
- prepare and commit postprocessing;
- publish generated tokens to the host-visible sequence;
- own stop conditions and request completion.

The graph continuation receipt only avoids redundant device setup. It does
not allocate KV blocks, alter queue order, commit tokens, or authorize work.

## Correctness Trace

The correctness graph uses epoch-relative history indices. For the frozen
aligned 128-token workload, sampled decode ordinals are:

```text
0, 63, 126
```

The graph stores bounded float32 logits when its monotonic epoch history index
matches one of those ordinals. This exercises continuation itself rather than
resetting the correctness graph for every burst.

If a correctness workload crosses a block boundary, the benchmark must map
each requested global sample point to its block-epoch-relative ordinal and
record the epoch identity. Ambiguous mapping is an evidence failure.

## Configuration

Add one strict boolean flag:

```text
exact_greedy_decode_burst_continuation = false
```

It is effective only when `exact_greedy_decode_burst` is enabled. Non-boolean
values are rejected. Unsupported modes use the current exact-burst cold-bind
path without changing output semantics.

## Observability

Expose cumulative counters for:

- continuation attempts and hits;
- cold binds;
- misses by stable reason;
- receipt invalidations by stable reason;
- continuation span in tokens and bursts;
- skipped static-reset operations;
- skipped scalar-bind operations;
- skipped block-table construction and copy calls;
- skipped block-table H2D bytes;
- history capacity and retained bytes;
- final token D2H calls and bytes;
- replay, failure, and quarantine counts.

Counters describe path use and eliminated work. They are not performance
proof.

## Correctness Invariants

The candidate must preserve:

- exact generated token IDs and decoded-text hashes;
- float32 sampled logits with `max_abs <= 0.25`,
  per-pair `mean_abs <= 0.05`, and equal argmax;
- exact scheduler lease, commit, completion, and stop behavior;
- exact KV block IDs, generations, write positions, and physical slots;
- one target-model forward per generated token;
- zero intermediate token D2H and one final token D2H per burst;
- no replay or fallback after a current-step mutation failure;
- unchanged cold-bind behavior when continuation is disabled or misses.

No continuity check may trust only a sequence ID or host counter. Device-state
reuse requires the complete receipt match.

## Stage-1 Benchmark

Run one source-bound four-arm Qwen3-0.6B matrix:

```text
host_greedy
decode_burst_k4
decode_burst_k4_continuation
decode_burst_k8
```

Use:

- TP1 and batch size one;
- exact temperature zero and `ignore_eos=true`;
- prompt lengths 256, 2048, and 8192;
- 128 generated tokens;
- two warmups and five measured repetitions;
- alternating arm order;
- identical input IDs, output budget, model, source commit, and physical GPU.

This produces 60 performance rows. Correctness uses four sample points per
arm and context, producing 48 rows plus manifest-bound float32 sidecars.

The current K4 and K8 implementations are rerun in the same matrix. Results
from another immutable run are context, not a paired baseline.

## Gate

`GO_EXACT_BURST_CONTINUATION_EPOCH` requires:

1. exact output token and decoded-text equality in every paired group;
2. bounded logit parity and equal argmax at every correctness point;
3. exactly one cold bind and at least 31 continuation hits in each measured
   K4-continuation request for the frozen aligned workload;
4. zero unexpected misses, failures, quarantines, or pending receipts;
5. at least 5% K4-continuation median TPOT improvement over current K4 in at
   least two of three context buckets;
6. at least 3% aggregate nearest-rank P95 TPOT improvement over current K4;
7. aggregate median TPOT no more than 2% slower than current K8;
8. maximum host-visible gap no more than 60% of current K8's paired maximum;
9. no bucket median or P95 TPOT regression above 3% versus current K4;
10. no TTFT or E2E regression above 3% versus current K4;
11. no throughput regression above 2% versus current K4;
12. no CUDA reserved-memory regression above 3% versus current K4;
13. exact retained host and device memory costs are reported;
14. producer and independent verifier agree on classification, selected arm,
    row counts, and comparison digest.

Failure produces a specific NO-GO classification. A correct implementation
with insufficient speedup remains default-disabled.

## Promotion Boundary

Stage 1 proves only Qwen3-0.6B TP1 batch-1 completion-only greedy decode under
the frozen workload.

Only a Stage-1 GO permits:

- enabling continuation for the proven exact-burst scope;
- a separate Qwen3-8B gate;
- testing K8 continuation;
- testing non-aligned prompts or block-boundary-heavy workloads;
- considering multi-sequence or tensor-parallel adoption.

No benefit may be claimed for ordinary decode, stochastic sampling, mixed
batches, speculative verification, other models, or production concurrency.

## Deliverables

- default-disabled continuation implementation;
- dependency-light unit and integration tests;
- source-bound paired benchmark runner;
- producer gate and independent verifier;
- immutable primary and controller artifacts;
- benefit-and-cost report;
- canonical audit and handoff reconciliation;
- exact-path commits pushed to `origin/feat/kv-sparse-attention`.
