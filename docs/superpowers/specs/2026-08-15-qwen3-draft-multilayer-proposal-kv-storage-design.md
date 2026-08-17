# Qwen3 Draft Multi-Layer Proposal-KV Storage Design

Date: 2026-08-15

Status: approved by the active Phase 1 continuation directive

## Goal

Extract the independent Qwen3 drafter's physical KV payload into a reusable
multi-layer `ProposalKVStorageAdapter` implementation. One logical proposal-KV
entry represents the key and value rows for every local draft-model layer.

This milestone prepares the storage boundary for a later residency allocator.
It does not wire offload into ModelRunner, allocate CPU backing by default,
create a CUDA transfer stream, or claim real H2D/D2H movement.

## Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve exact greedy selection, accepted-prefix semantics, fallback
  indexing, target-KV transactions, Scheduler behavior, and
  `MAX_PROPOSAL_TOKENS=4`.
- Preserve the learned-drafter executor-owned lease contract.
- Default registration must continue to construct
  `DirectProposalKVAllocator`.
- Default registration must allocate no CPU proposal-KV backing and create no
  copy backend or transfer stream.
- Do not add configuration that enables learned-drafter proposal-KV offload.
- Do not run GPU, remote, NCCL, loaded-checkpoint, or performance authority.
- Local copy tests are storage-contract evidence only, not real movement
  evidence.

## Considered Approaches

### 1. Extend `Qwen3DraftPhysicalSlotStore` in the backend module

Add logical capacity, optional CPU tensors, and copy methods directly to the
existing class.

Rejected because `qwen3_draft_backend.py` would continue to own model
execution, physical allocation, attention-cache binding, and residency
payload movement in one large unit.

### 2. Wrap the current physical store with a residency adapter

Keep the current GPU tensors and slot allocator unchanged, then add a wrapper
that owns CPU tensors and forwards GPU access.

Rejected because two objects would participate in capacity validation,
authority reporting, and storage ownership. This makes it easier to bind the
allocator or backend to the wrong object.

### 3. Extract a multi-layer storage class and retain a direct-store subclass

Create `Qwen3DraftProposalKVStorage` as the complete multi-layer GPU/optional
CPU payload owner. Keep `Qwen3DraftPhysicalSlotStore` as a thin direct-mode
subclass with equal logical and GPU capacities and no CPU backing.

Selected because it gives one object full storage ownership, matches the
generic `ProposalKVStorageAdapter` contract, preserves existing direct-mode
callers, and keeps runtime behavior default-off.

## Architecture

### Storage unit

Add:

```text
tinyvllm/engine/qwen3_draft_proposal_kv.py
```

`Qwen3DraftProposalKVStorage` owns:

- `gpu_key_cache` and `gpu_value_cache` with shape
  `[layer_count, gpu_capacity, 1, local_kv_heads, head_dim]`;
- optional `cpu_key_cache` and `cpu_value_cache` with shape
  `[layer_count, logical_capacity, 1, local_kv_heads, head_dim]`;
- attention-backend binding for every local layer;
- full-entry byte accounting across all local layers;
- batched logical-entry/physical-slot copy methods;
- tensor-free authority metadata.

The existing `key_cache` and `value_cache` aliases continue to point at GPU
storage so learned-drafter diagnostics and backend byte accounting remain
stable.

### Direct compatibility

`Qwen3DraftPhysicalSlotStore` subclasses the new storage class and calls it
with:

```text
logical_capacity == gpu_capacity == capacity
allocate_cpu_backing == False
```

The direct subclass alone owns `reserve_slots()`, `release_slots()`, and
direct slot-allocation authority. Generic residency storage leaves occupancy
authority entirely to `ProposalKVResidencyManager`.

The direct subclass public constructor remains unchanged. Existing
registration continues to construct this class, then wraps it in
`DirectProposalKVAllocator`.

### Full-entry movement semantics

`entry_nbytes()` counts both K and V for every local layer:

```text
2 * layer_count * block_size * local_kv_heads * head_dim * element_size
```

For each `(logical_entry_id, physical_slot_id)` row:

- `copy_gpu_to_cpu()` copies all layers for that slot into the logical row;
- `copy_cpu_to_gpu()` restores all layers from the logical row into that slot.

Rows are validated before any copy starts. Duplicate logical IDs, duplicate
physical slots, out-of-range indices, malformed rows, or absent CPU backing
fail closed without partial mutation.

### Module ownership

`qwen3_draft_backend.py` imports the storage classes and continues to expose
`Qwen3DraftPhysicalSlotStore` for existing callers. Backend execution remains
unchanged.

`autoregressive_draft_registration.py` imports the direct store from the new
module. No offload flag or residency manager is added to registration.

## Failure Boundaries

- Model geometry is validated before tensors are allocated.
- Every attention backend is preflighted before any layer is rebound, so a
  foreign cache on a later layer cannot partially publish earlier bindings.
- CPU backing is allocated only when explicitly requested.
- CUDA CPU backing requires pinned memory.
- Copy rows are fully validated before the first tensor mutation.
- Direct-mode release zeroes all layers of released GPU slots.
- A pre-existing foreign attention cache is rejected before rebinding.
- Runtime registration remains failure-atomic and direct-only.

## Testing

Focused tests must prove:

1. GPU and optional CPU tensors contain every local layer.
2. `entry_nbytes()` includes K, V, and all layers.
3. D2H and H2D copy all layers for every requested row.
4. Batched row validation fails before partial mutation.
5. absent CPU backing fails closed.
6. generic storage exposes payload/backing authority but no competing slot
   occupancy authority.
7. direct compatibility preserves allocation, release, zeroing, attention
   binding, and its legacy authority schema.
8. default registration still constructs `DirectProposalKVAllocator` around
   `Qwen3DraftPhysicalSlotStore`.
9. learned-drafter executor/backend and Proposal-KV regressions remain green.

## Completion Boundary

The strongest allowed result is:

```text
QWEN3_DRAFT_MULTILAYER_PROPOSAL_KV_STORAGE_ADAPTER=ESTABLISHED
QWEN3_DRAFT_RUNTIME_ALLOCATOR_MODE=DIRECT_ONLY
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD=NOT_ENABLED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
