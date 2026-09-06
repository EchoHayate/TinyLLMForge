# TP4 Dynamic Pool-Index Graph Protocol Design

**Date:** 2026-09-06
**Status:** Approved design; implementation not started
**Primary model:** Qwen/Qwen3.8-27B BF16, tensor parallel size 4
**Runtime scope:** exact multi-sequence decode CUDA Graphs using the
`lease_transaction_v1` Qwen3.8 hybrid-state path

## Context

The r49 Q1 smoke established that the existing hot-path capture and
warmup/measured phase isolation are correct on real TP4 hardware:

- eager and graph outputs matched exactly;
- all four ranks completed the capture and replay receipt protocols;
- graph state was reset between warmup and measurement;
- owned processes and process groups exited cleanly.

The smoke was nevertheless terminally negative:

- the TP-wide maximum single-capture duration was
  `4_110_665_805 ns`, above the frozen `2_000_000_000 ns` ceiling;
- graph throughput was approximately 3.25% below eager for the one
  diagnostic pair;
- the complete 30-case/15-pair gate was therefore not launched.

The current graph identity includes an ordered lease seal derived from
every active sequence's:

```text
slot_id + generation + request_id
```

That seal is necessary for safety in the current implementation because
the captured Qwen3.8 hybrid-state gather and commit paths resolve Python
lease objects to concrete physical pool slots. It also means that a
request or generation rotation creates a new graph identity even when
the executable graph program, tensor shapes, attention policy, and
state layout are unchanged.

The next bounded optimization is to separate:

1. the stable identity of the captured graph program; and
2. the dynamic ownership identity that must be validated before every
   replay.

This design does not claim that removing lease-rotation recapture will
make the first capture satisfy the two-second ceiling. It creates a
safe way to measure that question without deleting ownership identity
or weakening any frozen gate.

## Goals

1. Reuse one captured transactional decode graph across changes in
   request ID, lease generation, and physical hybrid-state slot.
2. Preserve ordered `slot_id + generation + request_id` validation
   before every graph replay.
3. Make physical hybrid-state slot selection a graph input rather than
   a Python constant embedded during capture.
4. Preserve exact output parity and transactional hybrid-state commit
   semantics.
5. Keep the feature default-disabled and limited to the Qwen3.8
   `lease_transaction_v1` path.
6. Produce explicit evidence for capture count, cross-lease replay,
   correctness, latency, throughput, memory, and cleanup.

## Non-Goals

- Do not change the ordinary `forward_v1` CUDA Graph path.
- Do not change exact-prefill graphs, speculative verification graphs,
  Exact Greedy K8, KV offload, Quest, or graph-resident greedy tail.
- Do not remove `slot_id`, `generation`, or `request_id` from runtime
  ownership validation or evidence.
- Do not retain warmup graphs across the warmup/measured reset boundary.
- Do not lower the frozen capture, replay-coverage, correctness, memory,
  or performance gates.
- Do not make the feature production-default.
- Do not modify or reclassify r48 or r49 artifacts.

## Current Data Flow

For each multi-sequence decode step, `ModelRunner` currently:

1. prepares the active `HybridStateLease` tuple in sequence-row order;
2. builds `FlashAttentionGraphIdentity`;
3. asks the model for a state-schema hash and full ordered lease seal;
4. includes that lease seal in `identity.sha256`;
5. looks up the graph cache by that full identity;
6. runs eager on a miss;
7. captures after sufficient successful observations;
8. during capture, resolves Python leases to concrete slot IDs;
9. during replay, rebuilds and compares the same full identity.

The Qwen3.8 state transaction validates each lease against
`HybridStateTensorPool`, gathers state from concrete pool rows, computes
new state, and commits the candidates back to those same rows.

The graph therefore captures the addresses and indexing operations for
one concrete lease assignment. The cache cannot safely reuse it for a
different assignment.

## Considered Approaches

### A. Key by physical slot IDs but ignore generation and request ID

The graph would remain bound to concrete physical slots while
generation and request identity moved to replay-time validation.

Advantages:

- smallest code change;
- generation reuse of the same physical slots would avoid some
  recaptures;
- existing Python-indexed gather and commit could remain.

Disadvantages:

- a different physical slot assignment still requires another graph;
- reuse depends on allocator behavior and is not stable;
- it does not implement a true dynamic pool-index protocol.

This is not selected because its benefit is workload- and
allocator-dependent.

### B. Stable program key plus dynamic device-side pool indices

The graph owns a fixed-shape device tensor containing one physical
hybrid-state slot ID per batch row. Capture records tensor-indexed
gather and commit operations. Before each replay, the runtime validates
the current full lease manifest and copies its slot IDs into that
tensor.

Advantages:

- one graph can serve arbitrary valid lease rotations with the same
  structural program identity;
- no full recurrent-state staging copy is added per token;
- complete ownership validation remains outside and before graph
  launch;
- cache behavior becomes deterministic with respect to structural
  shape rather than allocator history.

Disadvantages:

- graph-specific tensor-indexed gather and commit APIs are required;
- index operations must be proven capture-safe on the target runtime;
- the first capture may remain above the frozen two-second ceiling.

This is the selected approach.

### C. Graph-private state staging

Before replay, copy each active request's complete recurrent and
convolution state into graph-private contiguous buffers; after replay,
copy the result back to the owned pool slots.

Advantages:

- strongest separation between graph storage and runtime ownership;
- minimal dependence on dynamic indexing support inside the graph.

Disadvantages:

- copies the large hybrid state twice per decode token;
- increases memory footprint;
- likely removes the latency and throughput benefit being sought.

This remains a fallback diagnostic only if device-side indexed commit
cannot be captured correctly.

## Selected Architecture

### 1. Separate program identity from lease manifest

The existing `FlashAttentionGraphIdentity` fields remain intact,
including `lease_seal`. The implementation must not erase or silently
repurpose ownership evidence.

For `lease_pool_index_v1`, the runtime derives two hashes:

#### Full invocation identity

The existing `identity.sha256`, including the full ordered lease seal.
It identifies one concrete invocation and remains available in dispatch
events and receipts.

#### Stable graph-program key

A deterministic hash over:

- graph batch size;
- active batch size;
- page-table width;
- effective FlashAttention split count;
- FlashAttention version;
- multiprocessor count;
- local query/KV head counts;
- head dimension;
- page block size;
- maximum query length;
- execution protocol `lease_pool_index_v1`;
- hybrid-state schema hash.

It excludes only the concrete lease seal. For every protocol other than
`lease_pool_index_v1`, the cache key remains the existing full
`identity.sha256`.

The implementation should expose this distinction explicitly, for
example as `identity.cache_key_sha256`, rather than duplicating
ad-hoc field filtering in the cache and runner.

### 2. Ordered lease manifest

Before capture or replay, the runtime builds an immutable manifest in
the same order as the input batch:

```text
LeaseManifestRow(
    batch_index,
    slot_id,
    generation,
    request_id,
)
```

Validation must prove:

- the row count equals the active batch size;
- batch indices are canonical and contiguous;
- each row's request ID equals the corresponding sequence ID;
- every lease is currently owned by the hybrid-state allocator and
  tensor pool;
- each generation equals the pool's current binding;
- physical slot IDs are distinct;
- all slot IDs are in range;
- the manifest order is the input/token row order.

The manifest receives its own deterministic SHA-256. The digest is
recorded for diagnosis, but it is not part of the graph-program cache
key.

An invalid or stale manifest fails before `graph.replay()`. It must not
be converted into a silent graph launch or treated as a cache miss.

### 3. Graph-owned dynamic slot tensor

Each ready `lease_pool_index_v1` graph entry owns a fixed-shape device
tensor:

```text
state_slot_ids: int64[active_batch_size]
```

The exact dtype may be narrowed only if every target indexing operator
requires and validates that dtype. Its shape, dtype, device, and storage
address remain stable for the graph lifetime.

At replay admission:

1. validate the full ordered lease manifest on the host;
2. materialize its ordered physical slot IDs;
3. copy them into `entry.tensors["state_slot_ids"]`;
4. copy the existing input IDs, positions, KV slot mapping, context
   lengths, and block tables;
5. set the attention context;
6. call `graph.replay()`.

The copy must complete on the same ordered stream used by replay. No
host synchronization is added solely for the slot tensor.

### 4. Graph-specific hybrid-state access

The eager lease API remains unchanged. New graph-specific methods
accept the graph-owned slot tensor:

```text
Qwen35LayerStateAdapter.gather_batch_by_slot_tensor(slot_ids)
Qwen35LayerStateAdapter.commit_batch_by_slot_tensor(slot_ids, ...)

Qwen35CrossLayerStateTransaction.gather_by_slot_tensor(slot_ids)
Qwen35CrossLayerStateTransaction.commit_by_slot_tensor(slot_ids, ...)

Qwen35PackedForCausalLM.run_exact_cuda_graph_step_by_pool_index(
    state_slot_ids,
    token_counts,
    input_ids,
    position_ids,
)
```

The graph path uses tensor operations such as `index_select` and
`index_copy_` so slot values can change without changing tensor
addresses or graph topology. It must not call `.item()`, convert slot
values to Python integers, or branch on device values inside capture.

The eager transaction path keeps its current lease validation,
rollback behavior, and Python-facing API.

### 5. Capture lifecycle

Capture remains post-success: at least one successful eager execution
must precede capture admission.

For a selected structural program key:

1. validate the current full manifest;
2. allocate graph-owned static input tensors, including
   `state_slot_ids`;
3. initialize `state_slot_ids` from the validated capture manifest;
4. snapshot the active hybrid state and scratch KV;
5. capture one call to
   `run_exact_cuda_graph_step_by_pool_index`;
6. synchronize and record the TP-wide maximum capture duration;
7. restore hybrid state and scratch KV on success or failure;
8. rebuild both the program key and current full invocation identity;
9. commit the graph only if the program key is unchanged and all
   rollback obligations succeeded.

The entry records:

- stable program-key SHA;
- capture invocation identity SHA;
- capture lease-manifest SHA;
- graph object and static tensors;
- static/reserved bytes;
- capture duration;
- replay count and last replay step.

### 6. Replay lifecycle

On a structural cache hit:

1. rebuild the current identity and stable program key;
2. require the entry to be ready and keyed by the same program hash;
3. build and validate the current ordered lease manifest;
4. verify all static tensor shapes, dtypes, devices, and protocol
   version;
5. copy current slot IDs and existing dynamic inputs;
6. record the current invocation and manifest digests;
7. launch the graph;
8. compute or return logits as the protocol requires;
9. increment replay counters only after successful replay.

A lease rotation is expected to change the full invocation hash while
leaving the program key unchanged. That transition is counted as a
cross-lease replay, not identity drift.

### 7. Cache behavior

`ExactCudaGraphCache` uses the protocol-aware stable key for:

- observation counts;
- in-progress capture ownership;
- ready entries;
- terminal rejection by program key.

For legacy protocols, behavior is byte-for-byte compatible with the
existing full-identity key.

The cache summary adds:

- `cross_lease_replays`;
- `lease_manifest_rejections`;
- `unique_invocation_identities`;
- `unique_program_keys`.

The full invocation identity remains in events so a stable-key hit
cannot conceal which concrete request ownership was used.

### 8. Phase isolation

`reset_exact_cuda_graph_cache()` retains its current semantics:

- release ready graphs;
- synchronize after release;
- clear observations, rejections, counters, capture time, and graph
  pool ownership.

Warmup graphs are not carried into measured execution. The measured
phase must perform and report its own capture. This preserves the r49
evidence boundary and prevents warmup work from being presented as
zero-cost measured capture.

Within the measured phase, subsequent valid lease rotations may reuse
the measured graph.

## Failure Handling

The following failures occur before graph launch and fail closed:

- stale generation;
- request ownership mismatch;
- duplicate physical slot;
- manifest/input order mismatch;
- slot out of range;
- program-key mismatch;
- static slot-tensor shape, dtype, device, or storage drift.

An ownership failure is not a normal cache miss. The runtime must raise
or use the existing deterministic error path before state mutation.

Capture failure rejects only the affected program key. Replay failure
quarantines the ready entry as `replay_disabled`. Existing graph
release, process-group shutdown, and owned-process cleanup behavior
remain mandatory.

## Observability

Capture and replay receipts add:

- `execution_protocol`;
- `program_key_sha256`;
- `invocation_identity_sha256`;
- `lease_manifest_sha256`;
- ordered slot IDs;
- whether the replay crossed a lease identity;
- whether replay was reached after successful manifest validation.

Receipts must not claim that generation/request identity is part of the
stable cache key. They must make the separation directly auditable.

For TP4 evidence, all ranks must report the same program key and
ordered manifest digest for a dispatch. Cross-rank disagreement is a
correctness failure even if process exit codes are zero.

## TDD Verification

Implementation proceeds strictly RED to GREEN.

### Identity and cache tests

- same structural fields and different lease seals produce different
  full invocation hashes but the same v2 program key;
- legacy v1 identities retain the existing key behavior;
- different batch size, page-table width, split count, schema, model
  topology, or protocol produces a different program key;
- observations and ready lookup reuse the v2 structural key;
- cache entry and static-storage limits remain enforced.

### Lease-manifest tests

- valid manifests preserve batch-row order;
- stale generation is rejected;
- wrong request ID is rejected;
- duplicate slot is rejected;
- request/input ordering mismatch is rejected;
- validation failure occurs before slot-tensor copy or graph replay.

### Dynamic state-access tests

- tensor-indexed gather returns the same values as eager lease gather;
- tensor-indexed commit updates only the selected slots;
- changing slot-tensor values redirects both gather and commit without
  changing tensor storage;
- unselected slots remain byte-identical;
- capture-safe code performs no device-to-host scalar extraction.

### Model-runner tests

- one graph capture serves at least two distinct valid lease manifests;
- the second manifest changes request ID, generation, and physical
  slots;
- full invocation identity changes while the program key remains
  stable;
- the second dispatch is graph replay, not capture;
- stale or reordered ownership cannot launch the graph;
- legacy `lease_transaction_v1` and `forward_v1` behavior remains
  unchanged when the feature flag is off.

### Lifecycle and evidence tests

- warmup/measured reset still releases all graphs;
- capture and replay receipts carry both identity layers;
- TP4 receipt assembly rejects cross-rank program or manifest mismatch;
- cleanup remains `CLEAN` after success, rejection, capture failure,
  and replay failure.

## Hardware Qualification

Use a fresh immutable tag and the approved remote root:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

The first Q1 smoke must deliberately rotate leases while holding the
structural program identity constant. It must prove:

- exact eager/graph token equality;
- one successful measured capture per rank for the structural key;
- at least one successful replay after a different full lease
  manifest;
- identical TP4 program and manifest evidence;
- no state written to stale or unselected slots;
- clean four-rank lifecycle;
- complete capture and replay receipts.

The unchanged frozen gates are:

```text
single capture per rank:       <= 2_000_000_000 ns
total capture per rank:        <= 5_000_000_000 ns
measured replay coverage:      >= 0.80
exact output equality:         required
memory budget:                 unchanged
throughput/latency regression: prohibited by the existing contract
```

If the smoke demonstrates cross-lease reuse but the first capture still
exceeds two seconds, classify it as a useful mechanism result but
`NO_GO_CAPTURE_BUDGET`; do not launch the complete gate.

Only a smoke that passes correctness, lifecycle, capture, memory, and
performance admission may launch the complete 30-case/15-pair gate and
dual-verifier workflow.

## Rollback

The feature is controlled by a dedicated default-false configuration
flag. Disabling it restores the existing lease-sealed v1 identity and
capture path without artifact migration.

No existing graph entry is compatible across protocol versions.
Changing the protocol invalidates the process-local cache by
construction.

## Claim Boundary

A local unit-test pass establishes only interface and state-transition
correctness.

A successful Q1 smoke may establish safe cross-lease graph reuse for
that model, topology, workload, and source revision. It is not a
complete performance result.

Only a fresh complete gate with producer classification, immutable
manifest, remote independent verifier, and local frozen-source verifier
may support a performance claim.

The method must not be described as a performance improvement if it
only reduces capture count while first-capture cost, steady-state
throughput, latency, memory, or correctness gates fail.
