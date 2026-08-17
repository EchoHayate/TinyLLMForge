# Generic Proposal-KV Residency Transaction Design

## Status

The user approved this design on 2026-08-14.

This approval authorizes an uncommitted design document only. It does not
authorize implementation, staging, committing, pushing, a GPU workload,
remote access, NCCL, or an authority campaign.

Repository constraints require all work to remain in:

```text
/Users/bytedance/dev/TinyLLMForge-adaptive-ngram
```

Current classifications remain:

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=NOT_ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=NOT_ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=NOT_ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Problem Statement

The generic speculative runtime already has a proposal lifecycle shared by
native Qwen3.5 MTP and an independent Qwen3 draft-model path. Its proposal-KV
transaction, however, still uses physical GPU slot IDs as the durable identity
of committed and staged entries:

```text
ProposalKVCache
  -> ProposalKVTransaction.staged_slot_ids
  -> ProposalKVSequenceState.committed_slot_ids
  -> executor attention slot_mapping and block_tables
```

The current Qwen3.5 MTP and Qwen3 draft stores reserve every proposal-KV entry
from a GPU-only tensor. A committed entry therefore cannot leave GPU memory
without invalidating the transaction's identity. This prevents:

- a logical proposal-KV capacity larger than the GPU proposal-KV capacity;
- eviction of committed proposal K/V to fixed CPU backing;
- asynchronous prefetch and batched H2D restore;
- dirty writeback before committed-entry eviction;
- one residency implementation shared by MTP and an independent drafter.

The problem is not the accepted-prefix transaction itself. The existing
accepted-entry rule is correct:

```text
accepted proposal tokens = A
committed proposal-KV entries = max(A - 1, 0)
```

The missing boundary is between durable logical ownership and temporary
physical residency.

## Goal

Introduce a rank-local, model-agnostic proposal-KV residency transaction that:

1. gives every proposal-KV entry a stable logical identity with a generation;
2. gives an executor temporary leases over physical GPU slots;
3. keeps active staged entries pinned through transaction finalization;
4. allows committed entries to move between GPU and fixed CPU backing;
5. performs asynchronous, batchable H2D prefetch and dirty D2H writeback;
6. commits an accepted prefix in place without KV copy, replay, or
   rematerialization;
7. retires a rejected suffix without D2H;
8. delays physical-slot reuse until prior GPU reads or writes complete;
9. preserves exact greedy proposal and target semantics;
10. supplies authority counters that distinguish real proposal-KV movement
    from logical bookkeeping.

Qwen3.5 MTP is the first consumer. The independent Qwen3 drafter must later be
able to reuse the same allocator and residency contracts without duplicating
the transaction or transfer state machine.

## Non-Goals

This design does not:

- alter verifier token selection, fallback indexing, accepted-prefix
  semantics, target-KV transactions, recurrent side state, Scheduler
  behavior, n-gram, SAM, or unrelated MTP behavior;
- change `MAX_PROPOSAL_TOKENS=4`;
- add accepted-token target replay or proposal-KV rematerialization;
- adapt target `KVOffloadMVP0` as the proposal-KV owner;
- share physical storage between target K/V and proposal K/V;
- implement prefix KV sharing, deduplication, or cross-request reference
  counting for proposal K/V;
- implement KV8 or KV4 proposal-KV storage;
- implement layer-aware or token-aware proposal-KV heat policies;
- combine proposal-KV offload with proposal CUDA Graph execution;
- change the target block size or proposal block size from the current
  proposal value of one token per entry;
- add a third residency tier, pageable-memory fallback, NVMe, or remote
  storage;
- claim lower TPOT, lower memory use, higher throughput, or longer-context
  support from local tests;
- authorize a GPU, remote, NCCL, or authority workload.

## Considered Approaches

### A. Generic Proposal-KV Residency Store

Place a small logical-entry allocator boundary below `ProposalKVCache`.
Provide a direct allocator for default-off behavior and a residency manager
for GPU/CPU movement. Executors consume temporary physical leases.

This is selected because proposal ownership and target ownership have
different layouts and lifecycles, while native MTP and independent draft
models need the same proposal transaction.

### B. Reuse Target `KVOffloadMVP0`

Translate proposal entries into target logical blocks and use the target
offload manager directly.

This is rejected because it couples proposal K/V to target layer layout,
target block size, sequence ownership, target commit timing, and target
movement counters. It would make proposal rollback depend on target-cache
invariants and would blur authority evidence.

### C. Separate MTP and Qwen3 Offload Implementations

Add one offload manager to `Qwen35MTPPhysicalSlotStore` and another to
`Qwen3DraftPhysicalSlotStore`.

This is rejected because logical generations, lease validation, transfer
ordering, dirty state, retirement, and movement accounting would be
duplicated. The two implementations would likely diverge at exactly the
transaction boundary Phase 1 is intended to make generic.

## Architecture

The selected architecture is:

```text
ProposalKVCache
  -> ProposalKVEntryAllocator protocol
       -> DirectProposalKVAllocator
       -> ProposalKVResidencyManager
            -> ProposalKVStorageAdapter
            -> ProposalKVCopyBackend

Qwen35MTPProposalExecutor
  -> ensure_writable / ensure_readable
  -> temporary ProposalKVResidencyLease
  -> attention slot_mapping / block_tables
  -> read/write completion marker
```

The components have the following responsibilities.

### `ProposalKVCache`

Owns only logical sequence and transaction semantics:

- sequence epoch validation;
- transaction begin/materialized/prepare/finalize/rollback states;
- committed logical-entry order;
- accepted-prefix commit count;
- rejected-suffix retirement;
- sequence release;
- lifecycle authority rows.

It must not choose a GPU victim, enqueue a copy, inspect model tensor layout,
or expose a durable physical slot as proposal identity.

### `ProposalKVEntryAllocator`

Is the only storage-facing dependency of `ProposalKVCache`. It owns logical
entry generations and accepts lifecycle transitions. Its public operations
are intentionally smaller than the full residency-manager API.

### `DirectProposalKVAllocator`

Preserves default-off behavior using a real generation-aware allocator:

- logical capacity equals GPU slot capacity;
- each live logical entry is permanently bound to one GPU slot;
- no CPU tensor or copy stream is created;
- no H2D or D2H operation is possible;
- movement counters remain exactly zero;
- leases still carry occupancy generations and receive completion markers.

This is not an identity shim. It exercises the same logical-entry and lease
contract as offload mode, so enabled and disabled paths cannot disagree about
ownership.

### `ProposalKVResidencyManager`

Implements the GPU/CPU hierarchy:

- fixed logical-entry table;
- fixed pinned-CPU backing tensor;
- smaller fixed GPU slot tensor;
- logical-to-residency and slot-to-occupancy maps;
- dirty and validity state;
- asynchronous H2D/D2H scheduling;
- batched copy coalescing;
- victim selection among eligible committed entries;
- stale-generation and stale-lease rejection;
- deferred retirement and physical-slot recycling;
- movement and residency authority counters.

### `ProposalKVStorageAdapter`

Describes model-local tensor geometry without owning transaction policy.
Qwen3.5 MTP supplies a one-layer adapter. The independent Qwen3 drafter later
supplies a multi-layer adapter. The manager sees an entry as an opaque pair
of K/V payloads and asks the adapter to expose:

- GPU K/V tensors;
- pinned CPU K/V tensors;
- entry axis and shape metadata;
- dtype and device;
- copy-span validation.

### `ProposalKVCopyBackend`

Owns stream and event mechanics:

- enqueue batched H2D spans;
- enqueue batched D2H spans;
- make the consumer stream wait for transfer completion;
- record consumer read/write completion;
- expose completion polling for deferred retirement;
- provide a synchronous test backend.

It does not select victims or mutate logical ownership.

## Stable Identities and Temporary Leases

The durable proposal-KV identity is:

```python
@dataclass(frozen=True)
class ProposalKVEntryIdentity:
    logical_entry_id: int
    generation: int
```

Rules:

- `logical_entry_id` addresses a fixed-capacity allocator row;
- every reuse of that row increments `generation`;
- a stale generation is rejected even if the numeric row is live again;
- sequence state and transactions store entry identities, never GPU slots;
- equality of entry identities does not imply current GPU residency.

An executor receives:

```python
@dataclass(frozen=True)
class ProposalKVResidencyLease:
    identities: tuple[ProposalKVEntryIdentity, ...]
    physical_slot_ids: tuple[int, ...]
    occupancy_generations: tuple[int, ...]
```

Lease rules:

- tuple order is significant and matches the requested identity order;
- each physical slot has a monotonically increasing occupancy generation;
- rebinding any slot invalidates all leases for its prior occupancy;
- a lease is valid only for the operation for which it was acquired;
- an executor may build attention tensors from `physical_slot_ids`, but may
  not persist them in sequence or transaction state;
- the manager validates identity, physical slot, and occupancy generation
  when a completion marker is recorded;
- a stale lease fails closed and cannot release or overwrite a newer
  occupancy.

## Allocator Contract

The implementation plan must define a protocol equivalent to:

```python
class ProposalKVEntryAllocator(Protocol):
    def reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]: ...

    def ensure_writable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease: ...

    def ensure_readable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease: ...

    def record_write_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None: ...

    def record_read_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None: ...

    def commit_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> None: ...

    def retire_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        writeback: bool,
    ) -> None: ...
```

The exact Python names may change in the implementation plan, but the
separation of responsibilities may not:

- reservation returns logical identities;
- residency lookup returns temporary leases;
- completion is explicit;
- accepted entries become committed without data movement;
- rejected or released entries retire with `writeback=False`;
- the cache does not call a model-specific store.

`retire_entries(..., writeback=False)` is asynchronous with respect to slot
recycling. Logical ownership ends immediately, while any physical occupancy
with pending GPU completion enters `retiring` until the copy backend confirms
that reuse is safe.

## Capacity and Configuration

V1 adds configuration equivalent to:

```text
proposal_kv_offload_enabled = false
proposal_kv_logical_entry_capacity
proposal_kv_gpu_slot_capacity
proposal_kv_cpu_backing_capacity
proposal_kv_async_copy = true
proposal_kv_batch_copy = true
```

Enabled-mode validation is:

```text
logical_entry_capacity == cpu_backing_capacity
logical_entry_capacity > gpu_slot_capacity > 0
```

Additional constraints:

- proposal block size is exactly one;
- proposal K/V dtype is unquantized FP16 or BF16;
- CPU backing is allocated eagerly with fixed capacity and pinned memory;
- pinned allocation failure is a construction error;
- no implicit pageable fallback is allowed;
- GPU capacity exhaustion is recoverable only if an eligible committed victim
  exists;
- active staged entries, pending transfers, and retiring occupancies are not
  victims;
- proposal-KV offload and proposal CUDA Graph execution are mutually
  exclusive in V1;
- target-KV offload configuration and counters remain independent.

Disabled mode constructs `DirectProposalKVAllocator`. It must not allocate CPU
backing, create a copy stream, or increment movement counters.

## Entry State Machine

Each logical identity has one authoritative state:

```text
free
  -> reserved
  -> active_staged
  -> committed
  -> retiring
  -> free with generation incremented on next reservation
```

Residency is orthogonal to logical state. A committed entry can be:

```text
gpu_dirty
gpu_clean
cpu_clean
h2d_pending
d2h_pending
```

The valid combinations are:

- `reserved`: GPU slot assigned for a future write; not readable;
- `active_staged`: GPU-resident, dirty, transaction-pinned, and readable after
  its write completion;
- `committed + gpu_dirty`: accepted in place; CPU copy is absent or stale;
- `committed + d2h_pending`: dirty writeback is in flight; GPU occupancy is
  not reusable until the transfer and prior reads complete;
- `committed + gpu_clean`: CPU and GPU copies represent the same generation;
- `committed + cpu_clean`: no GPU occupancy; prefetch is required before use;
- `committed + h2d_pending`: CPU-to-GPU restore is in flight;
- `retiring`: no new lease may be issued; pending events must drain before
  physical and logical resources return to free pools.

Active staged entries may not be evicted or written back before transaction
finalization. This bounds V1 proposal pressure because
`MAX_PROPOSAL_TOKENS=4`, while preventing proposal generation from observing
its own entries through unstable physical bindings.

## Transaction Lifecycle

### Begin

For a proposal with `P` tokens:

1. `ProposalKVCache.begin()` reserves `max(P - 1, 0)` logical identities.
2. The identities are attached to the active transaction in proposal order.
3. The allocator binds writable GPU occupancies.
4. No physical slot is stored in sequence state or the proposal object.

If reservation or residency fails, all identities reserved by that begin
attempt retire without writeback. The sequence has no active transaction
after cleanup.

### Materialize

For each proposal forward:

1. acquire readable leases for committed entries needed by attention;
2. acquire a writable lease for the staged destination entry or entries;
3. wait only for required H2D completion;
4. build temporary `slot_mapping` and `block_tables` from the leases;
5. enqueue the unchanged model forward;
6. record read and write completion on the consumer stream;
7. mark entries materialized only after the executor has issued all required
   writes.

The completion marker may represent queued GPU completion rather than a host
synchronization. It exists to order future eviction and physical-slot reuse.

### Prepare Finalize

For `A` accepted proposal tokens:

```text
commit_entry_count = max(A - 1, 0)
accepted identities = staged[:commit_entry_count]
rejected identities = staged[commit_entry_count:]
```

Prepare validates sequence epoch, transaction ownership, complete
materialization, accepted range, and exact identity order. It does not move
K/V.

### Commit Finalize

Commit performs:

1. append accepted identities to the sequence's committed identity list;
2. transition accepted identities from `active_staged` to `committed`;
3. retain their existing GPU payload and occupancy;
4. mark their CPU copy absent or stale until a later dirty writeback;
5. transition rejected identities to `retiring` with `writeback=False`;
6. clear active transaction and ticket ownership.

Accepted-prefix commit must produce:

```text
accepted_entry_copy_count = 0
accepted_entry_replay_count = 0
accepted_entry_rematerialization_count = 0
```

Rejected suffix handling must produce:

```text
rejected_entry_d2h_count = 0
rejected_entry_d2h_bytes = 0
```

### Rollback or Abort

Rollback and abort transition every staged identity to `retiring` with
`writeback=False`. They do not restore a prior physical mapping because the
committed logical identity list never changed.

No retired occupancy may return to the free GPU-slot pool until:

- all prior reads of that occupancy complete;
- all prior writes to that occupancy complete;
- any already-enqueued transfer touching it completes.

### Sequence Release

Sequence release requires no active transaction or finalize ticket. All
committed identities retire with `writeback=False`; preserving proposal K/V
after sequence destruction has no semantic value.

## Residency and Transfer Policy

### Writable Acquisition

`ensure_writable` requires reserved identities. It:

1. rejects stale or already-materialized identities;
2. allocates free GPU slots or evicts eligible committed victims;
3. increments each destination slot's occupancy generation;
4. binds the logical identity to the new occupancy;
5. returns a writable lease.

New proposal entries have no valid CPU payload and never require H2D.

### Readable Acquisition

`ensure_readable` preserves request order and deduplicates transfer work:

- GPU-resident entries return their current occupancy;
- `h2d_pending` entries reuse the in-flight transfer and completion event;
- CPU-resident entries receive destination slots and batched H2D;
- stale, free, reserved-but-unwritten, or retiring identities fail closed.

The consumer stream waits on each unique transfer completion event before
using the returned physical slots.

### Victim Eligibility

An entry is eligible for eviction only if it is:

- logically committed;
- not transaction-pinned;
- not already retiring;
- not participating in a pending H2D or D2H;
- not protected by an outstanding consumer completion that would make slot
  overwrite unsafe.

V1 uses a deterministic committed-entry LRU policy. Policy choice must remain
replaceable and must not leak into `ProposalKVCache`.

### Dirty Writeback

Evicting `committed + gpu_dirty`:

1. records the current occupancy and identity generation;
2. orders D2H after prior consumer writes;
3. copies K and V into that identity's fixed CPU row;
4. marks CPU data valid only after transfer completion;
5. prevents slot reuse until prior reads and D2H complete;
6. transitions the entry to `cpu_clean`.

Evicting `committed + gpu_clean` requires no D2H.

Rejected, aborted, rolled-back, and sequence-released entries never write
back.

### Batched Copies

When `proposal_kv_batch_copy=true`, the manager groups compatible entries into
the largest spans supported by the storage adapter. A batch remains
semantically a set of per-entry generation-checked movements.

Counters must report both operation count and entry count so one batched copy
cannot be misreported as one moved entry.

## Qwen3.5 MTP Integration

Qwen3.5 MTP is the first integration target.

### Bootstrap

The bootstrap forward acquires one writable lease for all staged identities.
Its `slot_mapping` is built from lease slots in staged-entry order. After the
forward is enqueued, the executor records write completion and then marks the
transaction materialized.

### Autoregressive Proposal Step

At proposal step `i`:

```text
visible logical identities =
    committed identities
    + staged identities[:i + 1]

writable logical identity =
    staged identities[i]
```

The executor:

1. ensures committed identities are readable, prefetching if needed;
2. obtains the current staged occupancy for read/write use;
3. builds the temporary block table in the exact logical order above;
4. uses the writable slot as `slot_mapping`;
5. runs the unchanged MTP module;
6. records one consumer completion that protects all leased occupancies.

The final token in a proposal has no proposal-KV entry under the current
`P - 1` representation. This design does not change that representation.

### Lifecycle Coordinator

`ProposalKVLifecycleCoordinator` keeps the existing registration and
two-phase finalize ownership. Its authority rows change from staged slot
counts to staged logical-entry counts. It must not expose physical slot IDs as
durable proposal evidence.

### CUDA Graph Boundary

V1 rejects construction when both proposal-KV offload and proposal CUDA Graph
execution are enabled. Variable physical lease bindings and copy-event waits
must first receive a separate graph design and authority campaign.

## Independent Qwen3 Drafter Reuse

The Qwen3 draft path is not the first implementation consumer, but the generic
design is incomplete if it requires a second transaction implementation.

The later Qwen3 adapter may differ only in payload geometry:

```text
Qwen3.5 MTP entry:
  [K/V, one proposal layer, one token, local KV heads, head dim]

Qwen3 draft entry:
  [K/V, all draft layers, one token, local KV heads, head dim]
```

It must reuse:

- `ProposalKVEntryIdentity`;
- allocator lifecycle;
- lease validation;
- residency state machine;
- copy backend;
- accepted-prefix and rejected-suffix semantics;
- movement counters.

The Qwen3 adapter may provide layer-major copy spans, but it may not fork
generation, retirement, or writeback policy.

## Error Handling and Cleanup

The implementation must fail closed for:

- stale sequence epochs;
- stale logical-entry generations;
- stale occupancy generations;
- duplicate identities in one lease request;
- an identity owned by another active transaction;
- reads before materialization;
- writes after materialization or commit;
- eviction of active staged entries;
- reuse of a retiring slot;
- CPU or GPU capacity mismatches;
- unsupported dtype, block size, or graph combination;
- incomplete or failed transfer events.

Partial-operation cleanup rules are:

- failed reservation retires only identities reserved by that attempt;
- failed H2D leaves CPU data authoritative and does not expose the
  destination occupancy as readable;
- failed D2H leaves GPU data authoritative and does not mark CPU data valid;
- executor failure retires staged identities without writeback after queued
  GPU work drains;
- cleanup preserves the first failure while still attempting all owned
  retirements;
- no error path may recycle a slot merely because logical ownership was
  cleared.

## Authority Snapshot and Counters

The allocator snapshot must expose rank-local, integer counters and current
state counts sufficient to prove real movement and safe cleanup.

Required cumulative counters:

```text
logical_entries_reserved
logical_entries_committed
logical_entries_retired
lease_read_count
lease_write_count
lease_stale_rejections
gpu_slot_rebindings
h2d_operation_count
h2d_entry_count
h2d_bytes
d2h_operation_count
d2h_entry_count
d2h_bytes
dirty_writeback_entry_count
clean_eviction_entry_count
rejected_entry_count
rejected_entry_d2h_count
rejected_entry_d2h_bytes
accepted_entry_copy_count
accepted_entry_replay_count
accepted_entry_rematerialization_count
retirement_wait_count
```

Required current gauges:

```text
free_logical_entry_count
reserved_entry_count
active_staged_entry_count
committed_entry_count
retiring_entry_count
free_gpu_slot_count
gpu_dirty_entry_count
gpu_clean_entry_count
cpu_clean_entry_count
h2d_pending_entry_count
d2h_pending_entry_count
```

Required invariants:

```text
default-off h2d/d2h operations, entries, and bytes == 0
accepted copy/replay/rematerialization counters == 0
rejected D2H entries and bytes == 0
no live identity appears in two logical states
no physical slot has two live occupancies
all drained sequences leave no active, committed, or retiring entries
```

Target-KV and proposal-KV movement must remain separately labeled in all
combined runtime artifacts.

## Local TDD Strategy

Implementation must be test-driven and dependency-light before any GPU
authority request.

### Logical Allocator Tests

Cover:

- monotonically increasing entry generations;
- stale identity rejection after logical-row reuse;
- direct allocator leases and zero movement;
- capacity exhaustion and exact cleanup;
- duplicate and foreign identity rejection.

### Residency State Tests

Use a deterministic synchronous copy backend to cover:

- committed dirty eviction and one D2H;
- clean eviction without D2H;
- CPU-resident prefetch and one H2D;
- batched operation count versus moved-entry count;
- deterministic LRU victim choice;
- active staged entries never selected as victims;
- pending and retiring occupancies never reused;
- stale occupancy lease rejection.

### Transaction Tests

Cover all accepted counts for proposals up to
`MAX_PROPOSAL_TOKENS=4`:

- exact `max(A - 1, 0)` committed-entry count;
- accepted identities preserved in place;
- rejected suffix retires without writeback;
- abort and rollback retire all staged entries without D2H;
- sequence release drains committed entries;
- prepared-ticket ownership and epoch checks remain unchanged.

### Qwen3.5 MTP Integration Tests

Use fake tensors/modules and a fake event backend to cover:

- bootstrap uses lease-derived slots;
- autoregressive block-table order matches logical order;
- committed-entry prefetch occurs before use;
- read/write completion is recorded after forward enqueue;
- default-off output and lifecycle snapshots remain compatible;
- offload mode reaches logical capacity greater than GPU capacity;
- exact greedy proposal tokens match direct mode;
- unsupported graph/offload combination fails at construction.

### Source and Configuration Tests

Cover:

- explicit new configuration defaults;
- enabled-mode capacity validation;
- FP16/BF16 and block-size restrictions;
- fixed pinned CPU allocation requirement;
- target-KV counters remain untouched;
- no `staged_slot_ids` or `committed_slot_ids` remain as durable
  `ProposalKVCache` state.

Passing local tests establishes contracts only. It does not establish CUDA
event ordering, real H2D/D2H movement, exact model parity, or performance.

## Separately Authorized GPU Authority

A future GPU campaign requires independent user authorization. Its minimum
Qwen3.5 MTP matrix must include:

```text
offload: direct and proposal-KV offload
TP:      TP1 and TP4
context: at least 4K and one pressure case at 16K or 32K
batch:   1 and 4
decode:  exact greedy
proposal length ceiling: 4
```

The pressure cells must satisfy:

```text
logical_entry_capacity > gpu_slot_capacity
h2d_entry_count > 0
d2h_entry_count > 0
gpu_slot_rebindings > 0
```

Correctness requirements:

- generated tokens exactly match direct proposal-KV mode;
- accepted proposal counts match;
- target-forward counts match;
- accepted-prefix transaction counters remain zero for copy, replay, and
  rematerialization;
- rejected suffix D2H remains zero;
- all ranks agree on configuration and drain cleanly;
- no authority cell relies on simulated copies.

Required reported metrics:

- TPOT, TTFT, throughput, and peak GPU memory;
- target-KV H2D bytes separately from proposal-KV H2D bytes;
- proposal-KV H2D/D2H operations, entries, bytes, and batch spans;
- acceptance, proposal length, target-forward count, and fallback count;
- direct-versus-offload exact greedy parity.

Performance improvement is established only by controlled before/after
measurements on identical model, prompt, output length, TP, batch, and
sampling settings. Real movement and exact parity are prerequisites, not
performance proof.

The independent Qwen3 drafter requires a later reuse gate proving that the
same residency manager operates with its multi-layer storage adapter. It is
not required for the first Qwen3.5 MTP local integration milestone, but it is
required before calling the subsystem generic across learned drafters.

## Implementation Boundaries

The future implementation plan must decompose the work into independently
green steps:

1. logical identity and allocator protocol;
2. direct generation-aware allocator migration;
3. dependency-light residency manager and copy backend;
4. `ProposalKVCache` migration from slots to identities;
5. Qwen3.5 MTP lease-based integration;
6. configuration and authority snapshot wiring;
7. local contract gate;
8. separately authorized GPU authority;
9. later Qwen3 multi-layer adapter reuse.

No step may silently enable proposal-KV offload by default. No production
performance policy is promoted from local fake-backend tests.

## Completion Boundary

After local implementation and local tests pass, the strongest allowed
classification is:

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

After an authorized GPU campaign proves real copies and exact parity, only
the corresponding movement and correctness classifications may advance.
Phase 1 remains incomplete until the broader prompt-to-artifact promotion
checklist covers both model structures, TP1/TP4, required context lengths,
batch sizes, exact greedy parity, performance metrics, memory, movement
bytes, and acceptance without proxy evidence.
