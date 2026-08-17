# Variable-Q Speculative CUDA Graph Design

**Date:** 2026-08-12
**Status:** Design approved; written specification awaiting user review

## Goal

Add an opt-in CUDA Graph path for the batch-native speculative tail verifier
without changing its exact-Q batching or transactional KV semantics.

The first version supports:

```text
TP1 only
KV offload disabled
one graph family per distinct exact Q
no padding
greedy deterministic verification
```

Variable proposal lengths continue to be supported by grouping rows into
distinct exact query lengths before entering `ModelRunner`. Each fixed-Q group
is independently dispatched to either its exact graph family or eager
execution.

The path must preserve:

- exact greedy parity with the eager verifier;
- one target forward per fixed-Q group;
- direct writes into the live speculative transaction's physical KV slots;
- in-place commit of accepted KV;
- rollback of the rejected suffix;
- safe eager fallback before replay starts;
- fail-closed handling after replay starts.

## Non-Goals

This slice does not:

- support TP greater than one;
- support `kv_offload_mvp0`;
- support blockwise speculative verification;
- pad, round, or merge different query lengths;
- capture proposal generation, sampling, acceptance, transaction finalization,
  prefix refcount changes, or scheduler metadata mutation;
- fuse the LM head, greedy selection, verifier acceptance, or KV commit into
  the graph;
- support non-zero temperature;
- support non-transactional recurrent or convolution state;
- change the source-neutral speculative runtime contract;
- claim 16K/32K, TP4, second-model, learned-drafter, MTP-executor, or production
  promotion coverage.

Unsupported cases remain on the existing eager path when eager execution is
safe. Cases that violate deterministic or transactional correctness fail
closed rather than silently changing execution semantics.

## Current State

The callback bridge already groups variable proposal lengths by distinct exact
query length:

```text
run_model_runner_tail_batch(items, residency_ticket_id)
  -> build_fixed_q_tail_batches(items)
  -> one run_spec_verify_batch(group.items, ...) per exact Q
```

`prepare_spec_verify_batch()` already produces the graph-compatible flattened
layout for a homogeneous group:

```text
B = number of rows
Q = homogeneous query length
N = B * Q

input_ids:    int64 [N]
positions:    int64 [N]
slot_mapping: int32 [N]
context_lens: int32 [B]
block_tables: int32 [B, W]
spec_verify_query_lens = (Q,) * B
```

`run_model(..., execution_mode="spec_verify")` currently forces every such
forward to eager execution. The existing `ExactCudaGraphCache` cannot be
reused unchanged because:

- its identity is decode-specific and requires `max_seqlen_q == 1`;
- its batch validator rejects `B == 1`;
- its input/output layout assumes one query token per sequence;
- its failure and lifecycle rules do not include speculative transaction
  ownership;
- decode and speculative verification must remain independently disableable
  and observable.

## Considered Approaches

### A. Independent Exact-Q Spec-Verify Cache

Create a dedicated cache, identity, entry type, admission policy, dispatch
event, and capture helper for speculative verification.

Advantages:

- isolates verifier correctness from the existing decode graph path;
- supports `B == 1` without weakening decode validation;
- makes exact Q part of the canonical identity;
- gives speculative replay its own transaction and failure semantics;
- allows focused rollback or disablement without affecting decode graphs.

Costs:

- duplicates a small amount of cache and budget machinery;
- requires a second observability schema;
- requires separate tests for lifecycle and memory accounting.

### B. Generalize the Existing Decode Cache

Extend the current exact-width cache with a mode field and optional Q
dimensions.

Rejected for the first version. The resulting union type would mix incompatible
batch rules, tensor layouts, transaction semantics, and failure behavior. A
shared abstraction can be considered only after both paths are independently
correct and stable.

### C. Pre-Capture a Fixed `(B, Q, W)` Matrix at Startup

Capture every configured family before serving requests.

Rejected. Page-table width depends on live sequence state, most combinations
may never be used, and startup capture cannot naturally obtain a valid
speculative transaction without manufacturing scheduler-visible state.

## Decision

Use approach A: a dedicated `SpecVerifyExactCudaGraphCache` with lazy
post-step capture.

A cold exact identity executes eagerly. Successful eager observations are
counted. Once the observation threshold is reached, the same request still
returns its eager result, while a separate capture-only transaction is used to
capture the family after the live forward. Later exact matches replay the
ready graph.

No request waits for an identity to become generally compatible by rounding
its batch size, query length, or page-table width.

## Exact Graph Identity

Add an immutable `SpecVerifyGraphIdentity` owned by
`tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`.

Its canonical fields are:

```text
active_batch_size
query_len
total_query_tokens
page_table_width
flash_attn_num_splits
attention_backend
attention_backend_version
input_dtype
output_dtype
num_query_heads
num_kv_heads
head_dim
page_block_size
device_compute_capability
```

Required invariants:

```text
active_batch_size == B > 0
query_len == Q > 0
total_query_tokens == B * Q
page_table_width == W > 0
flash_attn_num_splits == SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS
```

The canonical SHA-256 is computed from all fields. Runtime dispatch rebuilds
the identity from the prepared tensors and active model/backend metadata. An
entry is replayable only when both the dataclass value and SHA-256 match
exactly.

`B`, `Q`, and `W` are never rounded. `total_query_tokens` is retained even
though it is derived because it makes flattening drift explicit and
independently validates the output shape.

## Static Entry Layout

Each `SpecVerifyExactCudaGraphEntry` owns graph-private tensors sized exactly
for one identity:

```text
input_ids:    int64 [B * Q]
positions:    int64 [B * Q]
slot_mapping: int32 [B * Q]
context_lens: int32 [B]
block_tables: int32 [B, W]
outputs:      output_dtype [B * Q, hidden_size]
```

It also owns:

- the canonical identity and SHA-256;
- one `torch.cuda.CUDAGraph`;
- capture duration;
- static tensor bytes;
- allocated-memory delta;
- reserved-memory delta;
- replay count and last replay step;
- last-use step for LRU ordering;
- an in-flight replay count;
- state: `ready`, `quarantined`, or `evicted`;
- one stable terminal reason when quarantined.

The graph contains the target model's pure GPU transformer forward, including
the KV writes performed by attention layers. `compute_logits()`, argmax,
acceptance, and transaction finalization remain outside the graph and use the
same code for eager and replay results.

Static tensors are never shared across identities. Replay copies exact-shaped
live inputs and context metadata into the entry's tensors before launching the
graph.

## Admission Rules

Graph dispatch is considered only when all of the following hold:

- execution mode is `spec_verify`;
- tensor parallel size is exactly one;
- `kv_offload_mvp0` is disabled;
- blockwise KV-offload verification is disabled;
- the batch contains a positive homogeneous exact Q;
- `B` is present in the configured batch allowlist;
- `Q` is present in the configured query-length allowlist;
- the exact identity can be built and fully matches a ready entry;
- execution is greedy and deterministic;
- no input embeddings or hidden-state return mode is active;
- the live verifier call is authorized by a valid speculative transaction;
- recurrent or convolution state is absent, or is covered by a separately
  approved transactional state contract.

The first version has no separately approved transactional recurrent or
convolution state contract, so any such active state fails closed.

The transaction authorization check is distinct from KV offload residency.
The first version requires no `SpeculativeResidencyParticipant` ticket because
KV offload is disabled, but it must still prove that the prepared logical and
physical slots belong to the live speculative KV transaction and that this
fixed-Q group has not already been materialized or finalized.

Admission failure before replay starts produces either:

- eager execution with a specific fallback reason when the current eager
  verifier contract remains valid; or
- a raised error when deterministic or transactional correctness cannot be
  established.

## Cache Policy

Add:

```python
@dataclass(frozen=True)
class SpecVerifyExactCudaGraphCacheConfig:
    enabled: bool
    batch_allowlist: tuple[int, ...]
    query_len_allowlist: tuple[int, ...]
    min_observations: int
    max_entries: int
    max_static_bytes: int
    max_reserved_bytes: int
    max_total_capture_ns: int
    max_single_capture_ns: int
```

The cache tracks:

- successful eager observation counts by exact identity SHA;
- ready entries;
- quarantined identities and stable reasons;
- entries currently capturing or replaying;
- static bytes and retained CUDA reserved-memory deltas;
- cumulative capture duration;
- hit, miss, capture, fallback, quarantine, and eviction counters.

Policy:

1. An unseen identity executes eagerly.
2. Each successful eligible eager forward increments its observation count.
3. Capture is considered when the count reaches `min_observations`.
4. Pre-capture and post-capture resource budgets must both pass.
5. When `max_entries` is reached, the least-recently-used entry may be evicted
   only if it is `ready` and has zero in-flight replays.
6. Capturing, replaying, or quarantined entries are never selected as eviction
   victims.
7. Eviction removes the entry from lookup and releases Python references. Its
   observed reserved-memory delta remains conservative cache accounting until
   current CUDA allocator measurements prove that memory was returned.
8. A quarantined identity remains unavailable for the process lifetime.

The cache does not recapture a quarantined identity.

## Configuration

Add independent fields to `tinyvllm/config.py`:

```text
spec_verify_cuda_graphs = False
spec_verify_cuda_graph_batch_allowlist = (1, 4)
spec_verify_cuda_graph_query_len_allowlist = ()
spec_verify_cuda_graph_min_observations = 2
spec_verify_cuda_graph_max_entries = 8
spec_verify_cuda_graph_max_static_bytes = 64 * 1024 * 1024
spec_verify_cuda_graph_max_reserved_bytes = 512 * 1024 * 1024
spec_verify_cuda_graph_max_total_capture_ns = 5_000_000_000
spec_verify_cuda_graph_max_single_capture_ns = 2_000_000_000
```

The empty default Q allowlist means no Q family is admitted even if a caller
turns on the feature without selecting query lengths. This is deliberately
fail-closed.

Validation is independent from the decode graph validator:

- the enable field is boolean;
- the batch allowlist is a non-empty canonical tuple of positive integers and
  explicitly permits `1`;
- the Q allowlist is a canonical tuple of positive integers and may be empty;
- booleans are rejected as integers;
- observation, entry, byte, and capture-time limits are positive integers;
- `enforce_eager=True` overrides replay and capture;
- no configuration permits batch, Q, or width rounding.

The cache receives an immutable normalized snapshot of these fields during
`ModelRunner` initialization.

## ModelRunner Components

Add dedicated helpers rather than extending decode helpers:

```text
_spec_verify_graph_incompatible_reason(...)
_build_spec_verify_graph_identity(...)
_estimate_spec_verify_graph_static_bytes(...)
_replay_spec_verify_graph(...)
_attempt_post_step_spec_verify_capture(...)
_capture_spec_verify_graph(...)
_publish_spec_verify_graph_dispatch_event(...)
spec_verify_graph_dispatch_observation()
```

`run_model()` keeps decode behavior unchanged. Its `spec_verify` branch becomes
an explicit dispatcher:

```text
preflight admission
  -> incompatible but eager-safe: eager + fallback event
  -> invalid transaction/determinism: fail closed

build exact identity
  -> invalid but eager-safe: eager + identity-invalid event

ready exact entry
  -> copy exact-shaped tensors
  -> revalidate transaction and identity
  -> mark entry in-flight
  -> launch replay
  -> clear in-flight
  -> common logits/argmax path

cache miss
  -> eager live forward
  -> record successful observation
  -> optionally run post-step capture-only forward
  -> return original eager result
```

`run_spec_verify_batch()` remains the public fixed-Q verifier method. It
continues to prepare one batch, invoke `run_model()` once, mark the live group
materialized once, compute greedy target tokens, and split rows in the original
order.

No graph helper precommits, seals, rolls back, or otherwise finalizes the live
speculative transaction. The capture helper may roll back only its private
capture-only scratch transaction. No graph helper performs sequence commit,
prefix publication, or scheduler mutation.

## Transaction and Capture Semantics

### Live Eager or Replay Forward

The live speculative transaction owns the verifier write slots. Both eager and
graph execution write directly to those slots. After the target tokens are
returned, the existing single finalize path:

1. determines the accepted prefix;
2. precommits the accepted/rejected partition;
3. commits accepted KV in place;
4. discards or rolls back rejected suffix storage;
5. seals metadata exactly once.

Graph replay never creates a second finalize path and never replays accepted
tokens to reconstruct KV.

### Capture-Only Forward

Post-step capture must not reuse the live transaction's write slots. It creates
a private capture-only transaction/context with:

- read-only access to the same valid prefix KV;
- exclusive speculative write slots sufficient for the exact `(B, Q, W)`
  family;
- no sequence ownership transfer;
- no scheduler-visible request, sequence, block, prefix, or generation
  mutation;
- no precommit, seal, publication, or acceptance operation.

Each scratch row must cover the worst valid terminal-prefix offset. For exact
query length `Q` and page block size `P`, reserve
`ceil((P - 1 + Q) / P)` private blocks per row. This capacity includes both a
private clone of a partially occupied terminal prefix block and any following
blocks crossed by the speculative writes.

The capture forward may overwrite only those exclusive scratch slots. Its
logits and hidden outputs are discarded. On both success and failure, the
capture-only transaction is completely rolled back and its scratch slot
ownership is released.

The captured entry is published to the cache only after:

- graph capture succeeds;
- the scratch transaction rolls back successfully;
- identity and tensor-shape checks still match;
- post-capture memory and duration budgets pass.

Failure to roll back the capture-only transaction is fatal for that verifier
call and quarantines the family.

## Replay Error Semantics

Before `graph.replay()` starts, any mismatch in identity, tensor shape,
allowlist, transaction ownership, or cache state disables the attempted replay
and may safely execute the live eager verifier once.

After `graph.replay()` starts:

- no eager retry is allowed for the same live transaction;
- the live transaction is aborted or rolled back by its existing owner;
- the graph family is quarantined with a stable reason;
- the CUDA error propagates;
- no target-token row is returned;
- no accepted KV or scheduler metadata is committed.

This rule prevents duplicate writes and ambiguous partial execution.

Capture failure is different from replay failure: the request's already
completed eager result remains valid only when the private capture transaction
was fully rolled back. The failed family is quarantined or rejected according
to the failure reason.

## Observability

Add a spec-verify-specific dispatch event rather than reusing
`last_cuda_graph_dispatch_event`.

Every eligible or rejected spec-verify dispatch records:

```text
step_id
request_ids_hash
mode
active_batch_size
query_len
total_query_tokens
page_table_width
flash_attn_num_splits
graph_identity_sha256
feature_enabled
dispatch
decision
fallback_reason
cache_state
observation_count
capture_attempted
capture_duration_ns
capture_static_bytes
capture_allocated_delta_bytes
capture_reserved_delta_bytes
cache_ready_entries
cache_static_bytes
cache_reserved_delta_bytes
cache_total_capture_ns
cache_hits
cache_misses
cache_evictions
cache_quarantines
transaction_authorized
source_sha256
```

`dispatch` is one of `eager` or `graph`. `decision` distinguishes at least:

```text
feature_disabled
incompatible
not_allowlisted
cold
capture
hit
evicted
quarantined
fail_closed
```

The event schema is fixed and tested for drift.

Performance reports must separate:

1. warmed exact-family graph-hit latency, where every measured verifier group
   is a graph hit; and
2. mixed-hit-rate end-to-end TPOT, including eager misses, capture stalls,
   grouping, transaction finalization, and normal scheduling.

Capture latency is never hidden inside a warmed graph-hit claim.

## File and API Placement

### New File

`tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`

- `SpecVerifyGraphIdentity`
- `SpecVerifyExactCudaGraphCacheConfig`
- `SpecVerifyExactCudaGraphEntry`
- `SpecVerifyGraphAdmissionDecision`
- `SpecVerifyExactCudaGraphCache`
- canonical fallback/quarantine reason definitions

### Modified Files

`tinyvllm/config.py`

- independent spec-verify graph fields and validator;
- no change to decode graph defaults or validator.

`tinyvllm/engine/model_runner.py`

- initialize the independent cache;
- add identity, admission, capture, replay, and dispatch helpers;
- replace unconditional spec-verify eager dispatch with the approved exact-Q
  decision tree;
- preserve the single public verifier and finalize path.

`tinyvllm/engine/speculative_model_runner.py`

- no grouping algorithm change;
- only contract or observability plumbing if the implementation requires
  transaction authorization metadata at the fixed-Q call boundary.

`tinyvllm/engine/speculative_residency.py` and
`tinyvllm/engine/block_manager.py`

- no offload behavior change;
- add only the minimal capture-only scratch transaction API if existing
  transaction primitives cannot express exclusive reversible capture slots.

## Testing

### Dependency-Light Cache and Configuration Tests

Add focused tests for:

- independent batch validation accepting `B == 1`;
- canonical positive batch and Q allowlists;
- empty Q allowlist failing closed without invalid configuration;
- exact identity equality and SHA drift;
- `total_query_tokens == B * Q`;
- observation threshold `2`;
- max-entry and byte/time budgets;
- ready, in-flight, quarantined, and evicted states;
- LRU eviction selecting only ready non-in-flight entries;
- stable quarantine reasons;
- conservative reserved-memory accounting after eviction;
- deterministic cache summaries and dispatch schema.

### ModelRunner Dispatch Tests

Extend or add focused tests covering:

- feature disabled remains eager;
- TP greater than one remains eager;
- KV offload remains eager;
- blockwise spec-verify remains eager;
- batch and Q allowlist misses remain eager;
- `B == 1` eligible exact family;
- `B == 4` eligible exact family;
- exact `Q` mismatch is a miss, never padding;
- exact `W` mismatch is a miss;
- cold observations execute eager;
- the threshold-triggering request returns its eager result;
- successful post-step capture creates a ready entry;
- ready identity replays once;
- pre-replay mismatch safely falls back to eager once;
- post-replay CUDA failure does not call eager;
- post-replay failure quarantines the family and aborts the live transaction;
- eager and graph results use the same logits/argmax and row splitter path;
- `mark_materialized()` occurs once for the live group;
- no graph helper invokes precommit, seal, or rollback on the live
  transaction, and no graph helper mutates prefix refcounts or scheduler
  state;
- the capture helper invokes rollback only for its private scratch
  transaction.

### Capture Transaction Tests

Tests must prove:

- capture uses slots disjoint from every live transaction;
- prefix KV is read-only during capture;
- scratch writes can be fully rolled back;
- capture produces no scheduler-visible metadata;
- capture produces no prefix refcount or generation mutation;
- successful capture publishes only after rollback;
- capture failure rolls back scratch ownership;
- rollback failure is fatal and quarantines the identity;
- capture never marks the live group materialized twice.

### CUDA Correctness Tests

On a CUDA-capable environment:

- compare eager and warmed graph logits for exact `(B, Q, W)` identities;
- compare greedy target tokens row by row;
- compare accepted lengths and final sequence tokens;
- compare committed KV for the accepted prefix;
- prove rejected suffix slots are not live after finalize;
- cover `B in {1, 4}`;
- cover every explicitly enabled Q in the first evaluation allowlist;
- cover at least two exact page-table widths;
- prove no extra target forward occurs during warmed replay;
- inject a replay failure and prove there is no eager retry.

### Regression Tests

Retain focused coverage for:

- exact-Q grouping and original row order;
- one target forward per fixed-Q group;
- transactional speculative KV commit/rollback;
- speculative selection and scheduler commit;
- decode exact-width CUDA Graph behavior;
- ordinary eager spec-verify behavior when the feature is disabled;
- prefix KV reference counting;
- chunked prefill and blockwise attention gates.

The implementation plan must name the exact existing test files to extend and
introduce dedicated cache, dispatch, capture-transaction, and CUDA integration
test files where isolation is clearer.

## Evaluation and Success Criteria

Correctness gate:

- exact greedy parity against eager;
- identical accepted lengths and final tokens;
- accepted KV committed in place;
- rejected suffix rolled back;
- no duplicate materialization or finalize;
- no eager retry after replay starts;
- deterministic fallback and quarantine evidence.

Coverage gate for this first version:

```text
TP1
KV offload disabled
context length 4K
batch 1 and 4
explicit exact Q families only
at least two exact W identities
```

Performance gate:

- warmed exact-family graph hits show repeatable verifier latency improvement;
- end-to-end TPOT at batch 1 and 4 improves under a source-bound mixed-hit-rate
  workload;
- TTFT has no material regression;
- capture stalls, graph memory, hit rate, eviction count, and quarantine count
  are reported;
- no H2D or D2H benefit is claimed because KV offload is disabled.

The benchmark report must include pinned source identity, model/checkpoint,
device, software versions, configuration, prompt/proposal distribution, warmup,
measurement count, and eager baseline.

## Promotion Boundary

Passing this design's implementation and tests proves only a TP1,
no-KV-offload, exact-Q speculative verifier CUDA Graph path.

It does not satisfy the broader generic optimization promotion gate, which
still requires two model structures, TP1 and TP4, 4K/16K/32K or longer
contexts, broader multi-sequence coverage, real KV-offload counters, and
learned/MTP execution evidence.

The repository-level classification remains:

```text
NOT_PROMOTABLE
```
