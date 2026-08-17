# Qwen3.5 MTP Exact-Q CUDA Graph Design

**Date:** 2026-08-12
**Status:** Approved design; implementation pending

## Goal

Install a real CUDA Graph capture/replay backend behind the existing
`Qwen35MTPExactGraphRunner` so native Qwen3.5 MTP proposal generation can
execute fixed exact-Q families without changing its transactional proposal-KV
semantics.

The first slice is intentionally narrow:

```text
TP1 only
KV offload disabled
greedy only
one MTP layer
shared embedding and LM head
exact Q in {2, 3, 4}
exact batch size in {1, 4}
Q1 remains eager passthrough
```

Each ready graph family must execute the complete `Q - 1` MTP-forward chain
for all rows in the fixed batch. The sampled token remains on the GPU between
steps: `argmax(logits)` writes the next step's static input-ID buffer without
calling `.item()` or synchronizing with Python.

The graph path must return ordinary `DraftProposal` objects backed by live
`ProposalKVCache` transactions. The executor registers those transaction IDs
exactly as it does for eager proposals, so existing prepare/commit/abort
finalization remains authoritative.

## Non-Goals

This slice does not:

- support tensor parallelism greater than one;
- support KV offload or blockwise KV-offload execution;
- support Q values outside `2`, `3`, and `4`;
- support graph batches other than `1` and `4`;
- capture Q1, because Q1 performs no MTP forward and stages no proposal KV;
- pad or merge distinct Q or batch sizes;
- support more than one MTP layer;
- support non-greedy sampling;
- move scheduler, verifier, target-KV, residency, or generic speculative
  runtime policy into Qwen-specific code;
- change accepted-prefix commit or rejected-suffix rollback behavior;
- replay, copy, or rematerialize accepted proposal KV;
- claim TP4, KV-offload, second-model, long-context, throughput, latency, or
  complete MTP-correctness coverage.

The repository remains `NOT_PROMOTABLE` until the real-checkpoint graph gate
passes and the broader phase-one promotion criteria are independently met.

## Current State

`Qwen35MTPProposalExecutor` already groups rows by exact Q:

```text
propose_batch(inputs)
  -> exact_q = min(remaining, requested, configured maximum)
  -> group rows by exact_q
  -> Q1 or no runner: _run_exact_q_eager(...)
  -> otherwise: graph_runner.run(exact_q, rows, eager)
```

`Qwen35MTPExactGraphRunner` already owns:

- immutable identities;
- exact Q and batch allowlists;
- observation thresholds;
- capture byte/time budgets;
- capture scratch rollback;
- permanent quarantine reasons;
- eager fallback before replay;
- `Qwen35MTPGraphReplayError` after replay starts, with no eager retry.

It currently receives fake backends in unit tests. Production registration in
`model_runner.py` raises:

```text
Qwen3.5 MTP CUDA graph capture backend is not installed
```

The eager proposal loop itself is not capture-safe:

- `torch.argmax(...).item()` synchronizes each proposal step with Python;
- each row is executed serially rather than as a fixed batch;
- `qwen35_cached_decode_eager_attention()` reads
  `context_lens.item()` and uses the result in Python shape control.

The generic `Attention` decode backend is the capture candidate because its
decode path uses `store_kvcache + flash_attn_with_kvcache` with tensor
metadata supplied by `temporary_context`.

## Considered Approaches

### A. Capture the Existing Eager Callback

Rejected. The callback contains CPU scalar extraction and row/step Python
loops. Capturing it would either fail or preserve host synchronization and
would not establish a real batched graph backend.

### B. Add a Dedicated Batched CUDA Backend and Scratch Owner

Selected. Keep the existing runner's admission and quarantine contract, but
provide production implementations in focused files:

```text
qwen35_mtp_cuda_graph_backend.py
qwen35_mtp_graph_scratch.py
```

The backend owns static tensors, graph capture, replay preflight, GPU argmax
chaining, and conversion of graph outputs into proposal records. The scratch
owner acquires isolated proposal-KV transactions for capture and rolls them
back after capture.

This keeps `model_runner.py` limited to construction and avoids mixing CUDA
graph lifecycle code into model registration.

### C. Generalize the Target Spec-Verify Graph Backend

Rejected for the first slice. Target tail verification and MTP proposal
generation have different tensor layouts, transaction owners, output types,
and graph contents. Sharing a backend now would create a union abstraction
before either path is stable.

## Architecture

### Existing Policy Layer

`Qwen35MTPExactGraphRunner` remains the policy layer. Its public call stays:

```python
run(*, exact_q: int, rows: tuple, eager)
```

Cold, disabled, unsupported, over-budget, or quarantined families return the
single eager result already produced by the runner. A ready family delegates
to backend replay. Once backend replay starts, any exception is wrapped in
`Qwen35MTPGraphReplayError`, quarantines the family, and propagates without an
eager retry.

### New Capture Backend

`Qwen35MTPCudaGraphBackend` owns:

```python
estimate_static_bytes(identity, rows) -> int
capture(identity, rows, eager, scratch_lease) -> Qwen35MTPGraphEntry
replay(entry, rows) -> tuple[DraftProposal, ...]
```

The backend is constructed with:

- the loaded Qwen3.5 MTP module;
- the live `ProposalKVCache`;
- the fixed model maximum block-table width;
- the MTP attention backend;
- the CUDA device and compute dtype.

It must not allocate or finalize live proposal transactions during capture.
It receives only scratch rows whose transactions are owned by the scratch
lease.

### New Scratch Owner

`Qwen35MTPGraphScratchOwner` owns capture-only proposal transactions through a
dedicated `ProposalKVCache` that shares only the production physical store.
It must not use the live executor cache: capture is attempted after a
successful eager observation, and that eager result intentionally leaves one
active live transaction per source sequence awaiting verifier finalization.

```python
acquire(identity, rows) -> Qwen35MTPGraphScratchLease
rollback(lease) -> None
```

For each source row it:

1. reads committed source slots from the live cache without mutating them;
2. maps each source row to an owner-private positive synthetic sequence ID;
3. reserves exactly `Q - 1` output slots through the dedicated scratch cache;
4. returns row-shaped capture inputs bound to those scratch transactions and
   source committed-slot snapshots;
5. never inserts scratch transaction IDs into the executor's
   `_proposal_transactions`;
6. aborts every scratch transaction on rollback;
7. releases each synthetic scratch sequence after rollback;
8. treats rollback failure as a hard capture-lifecycle error.

The capture graph may retain CUDA allocator reservations, but it must not
retain materialized logical proposal state. Sharing the physical store is
required so captured slot IDs address the same attention KV tensors; sharing
the live cache's transaction namespace is forbidden.

## Exact Graph Identity

The existing `Qwen35MTPGraphIdentity` remains canonical:

```text
exact_q
exact_batch_size
device_index
compute_dtype
hidden_size
mtp_layer_count
block_table_width
```

For the first slice:

```text
exact_q in (2, 3, 4)
exact_batch_size in (1, 4)
mtp_layer_count == 1
block_table_width == Config.max_model_len
```

No field is rounded. Q2/B1 and Q2/B4 are different entries. The backend
revalidates shapes and identity before replay begins. A mismatch is a
pre-replay error and must not call `graph.replay()`.

## Static Tensor Layout

For identity `(B, Q, H, W)`, each entry owns graph-private CUDA buffers:

```text
first_tokens:       int64 [B]
current_tokens:     int64 [B]
positions:          int64 [Q - 1, B]
initial_hidden:     compute_dtype [B, H]
current_hidden:     compute_dtype [B, H]
next_hidden:        compute_dtype [B, H]
slot_mapping:       int32 [Q - 1, B]
context_lens:       int32 [Q - 1, B]
block_tables:       int32 [Q - 1, B, W]
proposal_tokens:    int64 [B, Q]
```

The proposal output includes the target-provided first token in column zero.
At graph step `s`:

```text
input_ids      = current_tokens
hidden_states  = current_hidden
positions      = positions[s]
slot_mapping   = slot_mapping[s]
context_lens   = context_lens[s]
block_tables   = block_tables[s]

next_hidden, logits = module.forward_step(...)
next_tokens = argmax(logits, dim=-1)

proposal_tokens[:, s + 1].copy_(next_tokens)
current_tokens.copy_(next_tokens)
current_hidden.copy_(next_hidden)
```

All state transitions above execute on the capture stream. Python iterates
over the fixed `Q - 1` steps only while recording the graph; replay launches
one graph and performs no per-step host scalar extraction.

The implementation may alias `next_hidden` and `current_hidden` only if the
module and capture tests prove no read/write overlap. The initial
implementation should use distinct buffers and explicit `copy_` operations.

## Capture-Safe Attention

During graph capture and replay, Qwen3.5 full attention must use the generic
`Attention` decode backend rather than
`qwen35_cached_decode_eager_attention`.

For each fixed step, `temporary_context` binds static:

```text
mode="decode"
is_prefill=False
slot_mapping=[B]
context_lens=[B]
block_tables=[B, W]
max_seqlen_q=1
max_seqlen_k=fixed identity-compatible maximum
kv_offload_manager=None
kv_offload_blockwise_decode=False
kv_offload_blockwise_prefill=False
```

The graph path must call the already-bound attention backend's
`store_kvcache + flash_attn_with_kvcache` route. It must not mutate the global
attention mode for unrelated target-model execution. Any temporary override
is scoped to the graph backend call and restored even when capture fails.

## Live Replay Preparation

Before `graph.replay()`:

1. validate exact Q, exact B, device, dtype, hidden size, MTP layer count, and
   fixed table width;
2. begin one live proposal transaction per row for exactly `Q - 1` slots;
3. build every step's visible block table from committed slots plus the
   transaction's staged prefix through that step;
4. reject any visible table longer than `W`;
5. copy first tokens, initial hidden states, positions, slot mappings,
   context lengths, and zero-padded block tables into entry buffers;
6. record the live transactions locally but do not publish them to executor
   finalization maps yet.

Any failure in these steps is pre-replay. The backend aborts every transaction
it began and raises a fallback-safe exception. The runner may use eager only
when no graph replay has started.

The runner/backend interface must therefore distinguish:

```python
class Qwen35MTPGraphPreReplayError(RuntimeError):
    pass
```

`Qwen35MTPExactGraphRunner.run()` catches this error from `replay()` and
executes the eager callback once after the backend has proved that all partial
live graph transactions were aborted. Other replay exceptions retain the
existing hard-failure behavior.

## Replay and Transaction Publication

After static inputs are prepared:

```text
graph.replay()
torch-visible output buffers now contain proposal token IDs
```

No eager retry is permitted after the call to `graph.replay()` begins.

On successful replay:

1. mark every live transaction materialized for exactly `Q - 1` entries;
2. create one `DraftProposal` per row using the static proposal-token output;
3. include `exact_q`, `staged_entry_count`, and
   `execution_mode="cuda_graph"` metadata;
4. return proposal transaction IDs with the proposals.

The executor, not the backend, publishes successful transaction IDs into
`_proposal_transactions`. This keeps finalization ownership in one place and
makes eager and graph results follow the same registration validation.

If any error occurs after replay starts, the backend aborts still-abortable
transactions only as cleanup, then re-raises. Cleanup must never convert the
failure into an eager retry.

## Executor Integration

`Qwen35MTPProposalExecutor` gains one result-registration helper:

```python
def _register_group_proposals(
    self,
    proposals: tuple[DraftProposal, ...],
    rows: tuple[tuple[ModelRunnerProposalInput, _BootstrappedSequence], ...],
) -> tuple[DraftProposal, ...]:
    ...
```

Both eager and graph paths return proposals whose transaction IDs are
validated against the row sequence and epoch. The helper inserts IDs into
`_proposal_transactions` exactly once.

To avoid double registration, `_run_proposal()` stops directly mutating
`_proposal_transactions`; registration occurs only after the entire exact-Q
group succeeds. If group execution fails, all group transactions must already
be aborted by their execution owner.

Q1 stays unchanged: it uses eager `_run_proposal()` with zero staged entries
and never enters the graph runner.

## Production Construction

`model_runner.py` replaces the placeholder exception with focused
construction:

```text
build_graph_runner(config, module, proposal_kv_cache)
  -> validate TP1 / no KV offload / one MTP layer
  -> Qwen35MTPGraphScratchOwner(...)
  -> Qwen35MTPCudaGraphBackend(...)
  -> Qwen35MTPExactGraphRunner(...)
```

The default remains disabled. The first real gate overrides the configured
batch allowlist to `(1, 4)`, even though older config normalization tests may
still exercise broader valid tuples.

No CUDA graph implementation logic is added to `model_runner.py`.

## Failure Semantics

### Before Replay Starts

These cases may fall back to exactly one eager group execution:

- feature disabled;
- exact Q or batch not allowlisted;
- family cold or below observation threshold;
- family quarantined by capture/admission policy;
- static input mismatch;
- page-table capacity mismatch;
- failure to reserve all live proposal transactions;
- failure copying live data into static buffers;
- explicit `Qwen35MTPGraphPreReplayError`.

All graph-created live transactions must be aborted before eager fallback.

### After Replay Starts

These cases are hard failures:

- `graph.replay()` raises;
- a CUDA error is observed after replay launch;
- output shape or dtype is invalid after replay;
- materialized count cannot be published;
- transaction cleanup fails.

The family is quarantined as `replay_failed`,
`Qwen35MTPGraphReplayError` propagates, and no eager callback runs.

## Test Strategy

### Dependency-Light Unit Tests

Add focused tests for:

- static tensor shapes for Q2/Q3/Q4 and B1/B4;
- static-byte estimation;
- GPU argmax chaining source contract: no `.item()` in backend replay/capture;
- block-table padding and overflow rejection;
- scratch lease acquire/rollback and no executor publication;
- pre-replay partial-allocation rollback;
- explicit pre-replay fallback;
- replay-started failure quarantine and no eager retry;
- Q1 bypass;
- executor registration of graph-produced transactions;
- production builder installation when enabled;
- disabled registration returning no graph runner.

### Local Regression

Run graph, executor, model-runner integration, physical-KV, eager/reference,
and real-gate contract suites. CUDA-only tests may skip locally when no device
is available; a skip is not evidence that the backend works.

### Remote Real-Checkpoint Gate

Use only:

```text
host: sitian@10.232.195.203
Kerberos cache: FILE:/Users/bytedance/krb5cc_sitian
SSH: ControlMaster=no, ControlPath=none
GPU: CUDA_VISIBLE_DEVICES=7
```

The remote artifact must exercise:

```text
Q2/B1 eager cold observation
Q2/B1 capture
Q2/B1 replay
Q3/B1 replay
Q4/B1 replay
at least one B4 family
fresh-sequence eager/reference comparison
accepted-prefix commit
rejected-suffix rollback
injected post-replay failure with zero eager retry
```

Required artifact facts include:

```text
graph_backend_installed=true
graph_capture_count > 0
graph_replay_count > 0
graph_eager_argmax_equal=true
graph_eager_proposal_tokens_equal=true
graph_transaction_commit=true
graph_transaction_rollback=true
replay_failure_quarantined=true
replay_failure_eager_retry_count=0
backend_failures excludes graph_eager
```

The artifact token is an opaque run ID. A token containing `20260813` must not
be treated as chronological evidence because 2026-08-13 is in the future
relative to this specification date.

## Promotion Boundary

Passing this design's tests proves only the first-slice exact-Q graph backend
for the exercised Qwen3.5 checkpoint, TP1, no KV offload, greedy decode, one
MTP layer, Q2/Q3/Q4, and batches 1/4.

It does not prove:

- TP4 or any distributed graph behavior;
- KV-offload compatibility;
- arbitrary batch or Q support;
- long-context capacity;
- performance benefit;
- second-model portability;
- complete end-to-end MTP correctness.

