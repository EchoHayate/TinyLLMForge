# Autoregressive Draft Proposal-KV Offload Runtime Wiring Design

Date: 2026-08-15

Repository: `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`

## Goal

Connect the existing multi-layer `Qwen3DraftProposalKVStorage` to the existing
generic `ProposalKVResidencyManager` through the learned-drafter registration
path, while preserving the current direct allocator as the default.

This is a local runtime/configuration milestone. It does not establish real
CUDA H2D/D2H movement, loaded-checkpoint parity, TPOT, TTFT, throughput, memory
savings, or promotion readiness.

## Frozen Constraints

1. `MAX_PROPOSAL_TOKENS` remains 4.
2. Exact greedy parity requirements are unchanged.
3. Verifier selection, fallback indexing, accepted-prefix semantics, target-KV
   transactions, side-state, Scheduler behavior, n-gram, SAM, and native MTP
   behavior are unchanged.
4. Default-off construction allocates no CPU backing, creates no transfer
   stream, and constructs no `ProposalKVResidencyManager`.
5. Existing Qwen3.5 MTP proposal-KV configuration and construction semantics
   remain unchanged.
6. Durable learned-drafter state continues to store
   `ProposalKVEntryIdentity`; physical slot mappings remain temporary leases.
7. No GPU, remote, NCCL, loaded-checkpoint, or performance workload is part of
   this milestone.

## Approaches Considered

### A. Reuse the Existing Qwen3.5 Offload Enable and Capacities

This would let `proposal_kv_offload_enabled` control both learned sources.
It minimizes fields, but it becomes ambiguous when Qwen3.5 MTP and the
independent Qwen3 drafter are both registered. One capacity tuple would
silently configure two independent caches with different entry sizes.

Rejected because the configuration would not identify which cache owns the
capacity budget.

### B. Add Learned-Drafter-Specific Enable and Backing Capacities

Add:

```text
autoregressive_draft_proposal_kv_offload_enabled
autoregressive_draft_logical_entry_capacity
autoregressive_draft_cpu_backing_capacity
```

Continue using:

```text
autoregressive_draft_gpu_slot_capacity
proposal_kv_async_copy
proposal_kv_batch_copy
```

This keeps ownership explicit, preserves the existing GPU-capacity field, and
reuses the generic copy policy without duplicating the residency state machine.

Selected.

### C. Replace Flat Fields with a Per-Source Configuration Object

This would provide the cleanest long-term schema, but it would require broad
CLI/config serialization migration and compatibility work unrelated to the
local runtime gap.

Rejected as unnecessary for this milestone.

## Configuration Contract

Defaults:

```text
autoregressive_draft_proposal_kv_offload_enabled = False
autoregressive_draft_logical_entry_capacity = 0
autoregressive_draft_cpu_backing_capacity = 0
```

When offload is disabled:

- both new capacities must be nonnegative;
- normal learned-drafter validation still requires
  `autoregressive_draft_gpu_slot_capacity > 0` when the drafter is enabled;
- construction uses `Qwen3DraftPhysicalSlotStore` and
  `DirectProposalKVAllocator`;
- no CPU backing or copy backend is created.

When offload is enabled:

```text
autoregressive_draft_enabled == True
logical_entry_capacity == cpu_backing_capacity
logical_entry_capacity > gpu_slot_capacity > 0
```

Invalid combinations fail during `Config.__post_init__` before model loading.

## Construction Boundary

Add a source-specific allocator builder:

```python
build_qwen3_draft_proposal_kv_allocator(
    model,
    *,
    offload_enabled: bool,
    logical_entry_capacity: int,
    gpu_slot_capacity: int,
    cpu_backing_capacity: int,
    async_copy: bool,
    batch_copy: bool,
    dtype: torch.dtype,
    device: str | torch.device,
    _copy_backend=None,
)
```

Direct mode:

1. require `logical_entry_capacity == gpu_slot_capacity`;
2. build `Qwen3DraftPhysicalSlotStore`;
3. return `DirectProposalKVAllocator(storage)`.

Offload mode:

1. require `logical_entry_capacity == cpu_backing_capacity`;
2. require `logical_entry_capacity > gpu_slot_capacity`;
3. build `Qwen3DraftProposalKVStorage` with CPU backing;
4. use `TorchProposalKVCopyBackend` when `async_copy=True`, otherwise
   `SynchronousProposalKVCopyBackend`;
5. return `ProposalKVResidencyManager`.

The `_copy_backend` argument exists only for dependency-light tests. Production
construction never injects it.

## Registration Data Flow

`ModelRunner._register_autoregressive_draft()` will call one allocator builder
instead of separately constructing a physical store and direct allocator:

```text
validated Config
  -> loaded and device-bound Qwen3 draft model
  -> build_qwen3_draft_proposal_kv_allocator(...)
  -> resolve allocator.storage or allocator.physical_store
  -> ProposalKVCache(allocator)
  -> Qwen3AutoregressiveDraftBackend
  -> AutoregressiveDraftProposalExecutor
```

The resolved storage object remains available through the existing
`autoregressive_draft_physical_store` field for compatibility. Its value may be
either `Qwen3DraftPhysicalSlotStore` in direct mode or
`Qwen3DraftProposalKVStorage` in residency mode. No durable cache state depends
on that field.

## Failure and Publication Semantics

1. Configuration failures occur before checkpoint/model construction.
2. Allocator/storage/copy-backend construction remains inside the existing
   all-rank registration candidate phase.
3. A rank-local failure produces the existing structured registration status.
4. Cross-rank consensus must succeed before the executor, storage, and cache
   are published to `ModelRunner`.
5. Failed registration leaves all published learned-drafter fields unchanged.
6. No new fallback path is introduced.

## Test Strategy

### Configuration

- defaults remain off and zero;
- enabled offload accepts `logical == cpu > gpu > 0`;
- enabled offload rejects a disabled drafter;
- enabled offload rejects mismatched logical/CPU capacity;
- enabled offload rejects `logical <= gpu`;
- existing direct-mode validation remains unchanged.

### Builder

- direct mode returns `DirectProposalKVAllocator`;
- direct mode allocates no CPU backing and creates no copy backend;
- residency mode returns `ProposalKVResidencyManager`;
- residency mode exposes the multi-layer Qwen3 storage;
- synchronous dependency-light copy round-trip preserves every layer;
- async production selection is tested through an injected fake backend rather
  than constructing CUDA resources.

### ModelRunner Registration

- direct registration preserves the existing construction order and authority;
- offload registration passes exact configuration values to the allocator
  builder;
- registration resolves `.storage` from a residency manager;
- successful publication preserves the existing descriptor and executor
  topology;
- injected allocator failure remains failure-atomic.

### Regression

- learned-drafter focused tests;
- Proposal-KV Tasks 1-7;
- generic speculative runtime tests;
- changed production/test `py_compile`;
- default-off and stale-symbol scans;
- scoped `git diff --check`.

## Non-Goals

- no real pinned-memory or CUDA stream/event authority;
- no real H2D/D2H byte claim;
- no loaded Qwen3 draft checkpoint run;
- no TP1/TP4 exact-parity claim;
- no TPOT, TTFT, throughput, or memory claim;
- no changes to target-KV offload;
- no changes to Qwen3.5 MTP proposal-KV wiring;
- no automatic policy that enables offload based on memory pressure.

## Terminal Classification

If the local tests pass:

```text
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_RUNTIME_WIRING=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_DEFAULT=DISABLED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
