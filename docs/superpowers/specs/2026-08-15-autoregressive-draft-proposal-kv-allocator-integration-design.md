# Autoregressive Draft Proposal-KV Allocator Integration Design

Date: 2026-08-15

Status: approved continuation of the Proposal-KV reuse milestone

## Goal

Migrate the independent Qwen3 autoregressive draft executor from physical
slot ownership to the same generation-aware logical identity and temporary
residency-lease contract already used by native Qwen3.5 MTP.

The milestone is local and default-direct. It must not enable proposal-KV
offload, start CUDA copies, or claim loaded-checkpoint correctness or
performance.

## Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve exact greedy selection, accepted-prefix semantics, fallback
  indexing, target-KV transactions, Scheduler behavior, and
  `MAX_PROPOSAL_TOKENS=4`.
- Preserve TP1 and TP4 logical authority rules.
- Do not add compatibility aliases such as `staged_slot_ids` or
  `committed_slot_ids` to `ProposalKVCache`.
- Do not expose persistent physical slot IDs through cache durable state.
- Default mode must use `DirectProposalKVAllocator` and allocate no CPU
  backing or transfer stream.
- Do not add GPU, remote, NCCL, or performance authority artifacts.

## Considered Approaches

### 1. Restore physical-slot compatibility properties

Add `ProposalKVCache.physical_store`, `staged_slot_ids`, and
`committed_slot_ids` as aliases.

Rejected because it would reintroduce physical residency into durable cache
state, make eviction unsafe, and defeat the logical/physical decoupling
contract.

### 2. Let the Qwen3 backend resolve allocator identities

Keep row types unchanged and let `Qwen3AutoregressiveDraftBackend` acquire
leases from the allocator.

Rejected because the backend would own both model execution and transaction
residency completion. Failure cleanup and TP stage convergence would become
split across executor and backend.

### 3. Executor-owned leases with physical mappings in ephemeral rows

The executor acquires readable/writable leases immediately before each
backend forward, passes only temporary physical slot tuples in the row, and
records completion immediately after the forward, including the existing
exception path convention.

Selected because it matches the Qwen3.5 MTP implementation, keeps durable
state logical, and leaves the backend responsible only for building one
batched model context from already-authorized physical mappings.

## Architecture

### Registration

`build_autoregressive_draft_registration_dependencies()` continues to build
`Qwen3DraftPhysicalSlotStore`, then wraps it in
`DirectProposalKVAllocator`, then constructs `ProposalKVCache`.

ModelRunner retains the physical store only for diagnostics. The cache owns
the allocator contract, not the store directly.

### Ephemeral row contracts

`AutoregressiveDraftPrefillRow` adds:

```python
physical_slot_ids: tuple[int, ...]
```

The tuple must have one slot per prompt token.

`AutoregressiveDraftDecodeRow` adds:

```python
writable_physical_slot_id: int
visible_physical_slot_ids: tuple[int, ...]
```

The visible tuple contains committed entries, earlier staged entries, and the
current writable entry in causal order.

These fields are temporary forward mappings. They are never stored in
`ProposalKVCache`, lifecycle registrations, TP logical authority rows, or
final artifacts.

### Bootstrap flow

For each new prompt transaction:

1. reserve logical staged identities through `ProposalKVCache.begin()`;
2. acquire one writable lease for all staged identities;
3. pass the lease physical slots to the prefill row;
4. execute one batched backend prefill;
5. record each writable lease complete;
6. mark all staged identities materialized;
7. prepare and commit all prompt entries.

If the backend raises after dispatch, completion is still recorded using the
same fail-closed convention as Qwen3.5 MTP, then the existing transaction
rollback path runs.

### Proposal decode flow

For each sequence and proposal step:

1. compute the read prefix as committed identities plus staged identities
   before the current step;
2. acquire a readable lease for that prefix;
3. acquire a writable lease for the current staged identity;
4. build the visible physical tuple as read slots plus the writable slot;
5. pass the temporary mappings to the backend;
6. record read and write completion after the forward, including exception
   cleanup;
7. continue unchanged greedy token selection and TP broadcast.

The backend must not call allocator methods or derive slots from logical IDs.

### Backend validation

The backend verifies:

- prefill physical-slot count equals token count;
- decode writable slot is the final visible slot;
- visible slot count equals committed logical length plus `step + 1`;
- all slots are nonnegative integers and unique within one visible row;
- transaction ownership and row ordering remain unchanged.

It then builds the same `temporary_context` tensors as before.

## Failure Boundaries

- Allocator lease failure occurs before the backend forward.
- Backend failure records acquired lease completion before propagating.
- Transaction rollback retires staged logical identities with
  `writeback=False`.
- Registration failure publishes no executor and retains no live proposal
  entries.
- TP logical authority rows remain physical-identity-free.

## Testing

Focused tests must prove:

1. registration wraps the Qwen3 store in `DirectProposalKVAllocator`;
2. cache construction no longer accepts a physical store directly;
3. bootstrap uses writable leases and records completion;
4. decode uses readable and writable leases with exact visible ordering;
5. backend consumes only row-provided physical mappings;
6. stale removed slot APIs are absent from production learned-drafter code;
7. release drains logical entries and physical slots;
8. neighboring Proposal-KV and generic speculative runtime tests remain
   unchanged.

## Completion Boundary

The strongest allowed result is:

```text
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_ALLOCATOR_REUSE=ESTABLISHED
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD=NOT_ENABLED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

