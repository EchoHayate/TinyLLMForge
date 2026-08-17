# Qwen3 Draft Blockwise Proposal-KV Offload Design

## Problem

The real TP1 gate uses Qwen3-1.7B as target, Qwen3-0.6B as the independent
draft model, prompt lengths `13/9/10/10`, and Proposal-KV GPU capacity `8`.
The current draft bootstrap calls `ensure_writable()` for an entire prompt,
and each draft decode calls `ensure_readable()` for the complete committed
history of every batch row. A single 13-token prompt therefore exceeds the
physical capacity before any committed entry exists to evict. The observed
failure is:

```text
no eligible committed proposal KV eviction victim
```

Increasing the capacity would avoid the failure but would not establish real
H2D/D2H movement. Preloading unrelated KV would be synthetic evidence and is
forbidden.

## Selected Approach

Reuse the existing exact blockwise online-softmax attention implementation.
Add a narrow adapter over `ProposalKVResidencyManager` so attention layers can
stage committed logical Proposal-KV entries window by window while protecting
the current writable entries.

The direct allocator path remains unchanged.

## Components

### Proposal-KV blockwise staging adapter

`ProposalKVResidencyManager` will expose a dedicated adapter rather than
imitating `KVOffloadMVP0` directly. The adapter owns no tensors. It translates
logical entry IDs into current `ProposalKVEntryIdentity` values and exposes the
minimal interface consumed by `_blockwise_online_decode_attention()`:

- `gpu_blocks`
- `logical_to_slot`
- `pending_wait_blocks`
- `stats`
- `_touch(slot_id)`
- `ensure_resident(...)`
- `wait_for_blocks(...)`
- `mark_dirty(...)`
- `record_h2d_slot_read_window(...)`

`ensure_resident()` must protect every currently required logical entry and
every active staged write entry from eviction. Actual copies continue through
`TorchProposalKVCopyBackend`, so existing allocator counters remain the
authority for H2D/D2H bytes and operations.

### Incremental prompt bootstrap

When the allocator is residency-backed, prompt bootstrap will materialize one
real prompt token per sequence per round. Each round:

1. starts a one-entry Proposal-KV transaction;
2. acquires one writable physical slot per active sequence;
3. runs the real Qwen3 draft model in blockwise decode mode using all previously
   committed logical prompt entries plus the current writable entry;
4. records completion, materializes, prepares, and commits the one entry.

This is numerically equivalent to incremental autoregressive prompt ingestion.
It avoids requiring the entire prompt to be GPU resident. The existing packed
prefill path remains active for direct allocators.

### Blockwise proposal decode

For residency-backed allocators, the executor will not call
`ensure_readable()` on the full history before the model forward. It passes
logical entry IDs to the backend and acquires only the current writable entry.

The backend sets:

- `kv_offload_manager` to the Proposal-KV staging adapter;
- `kv_offload_blockwise_decode=True`;
- `kv_offload_blockwise_blocks=1`;
- logical block tables to committed plus prior staged IDs;
- write blocks to the current staged logical IDs.

With batch four and capacity eight, each attention window uses at most four
historical entries plus four protected write entries.

### Lifecycle and authority

Accepted-prefix semantics do not change:

- accepted entries commit through `ProposalKVCache.commit_finalize()`;
- rejected suffix entries retire without replay or rematerialization;
- sequence release retires committed prompt/proposal entries;
- target verification remains the only acceptance authority.

Real movement remains classified only from allocator copy counters. No
synthetic copy hooks or fabricated bytes are added.

## Failure Handling

- A window whose required entries plus protected writes exceed physical
  capacity fails closed with an explicit capacity error.
- Missing or stale logical-entry generations fail before attention.
- Any bootstrap round failure aborts or rolls back every transaction created in
  that round in reverse order.
- TP convergence behavior remains unchanged; TP1 rethrows the local root cause.

## Validation

1. Unit RED/GREEN for protected window staging and real dirty D2H plus clean H2D.
2. Backend RED/GREEN proving blockwise context contains logical IDs and only
   writable physical slots.
3. Executor RED/GREEN proving a prompt longer than GPU capacity bootstraps
   incrementally and batch-four proposal decode does not full-pin history.
4. Existing Proposal-KV, Qwen3 backend, executor, registration, model-runner,
   TP1-gate, storage, and integration suites.
5. Remote GPU 4 TP1 gate with capacity eight, exact batch 1/4 greedy parity,
   zero extra target forwards, zero accepted-entry copy/replay/rematerialization,
   zero leaked Proposal-KV slots, and nonzero real H2D and D2H bytes.

## Explicit Non-Goals

- No change to target-model KV offload.
- No performance claim from this correctness implementation.
- No CUDA Graph support in this patch.
- No change to TP4 producer/worker/verifier authority files.
- No change to `MAX_PROPOSAL_TOKENS=4`.
