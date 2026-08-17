# Generic Proposal-KV Residency Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Do not use subagents.

## 2026-08-15 Evidence Reconciliation

This plan was reconciled against current allocator/residency/cache/lifecycle
source, Qwen3.5 MTP integration, the local terminal-classification gate, and
fresh offline CPU Torch tests:

```text
PROPOSAL_KV_RESIDENCY_PLAN_TOTAL_STEPS=53
PROPOSAL_KV_RESIDENCY_PLAN_CHECKED=44
PROPOSAL_KV_RESIDENCY_PLAN_INTENTIONALLY_OPEN=9

focused allocator/residency/MTP integration suite:
  100 passed in 2.80s

adjacent generic speculative runtime and independent-drafter suite:
  275 passed in 2.14s

fresh total:
  375 passed

production/test py_compile:
  PASS

interface, durable-slot-removal, no-rematerialization, and local-gate
assertions:
  PASS
```

The nine open steps are intentional:

- six historical RED executions have no retained failure transcript;
- three Task 8 steps are the explicit authorization boundary for creating
  and running real GPU/remote/NCCL authority artifacts.

Current source establishes logical/physical identity decoupling, generation
checked leases, deterministic committed-LRU residency, dirty writeback,
batched H2D/D2H scheduling contracts, completion-aware retirement, rejected
suffix zero-D2H, accepted-prefix no-copy/no-replay/no-rematerialization, and
default-off zero-movement behavior. The synchronous CPU backend and fake
movement rows are contract evidence only; they are not real CUDA movement
authority.

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

**Goal:** Decouple durable proposal-KV logical identities from temporary GPU
slots, add a default-off GPU/CPU residency manager with asynchronous batched
movement and safe retirement, and integrate it first with Qwen3.5 MTP without
changing speculative token semantics.

**Architecture:** Put immutable entry identities, physical leases, and the
small allocator protocol in one dependency-light module. Keep movement,
residency state, transfer ordering, and authority counters in a separate
manager that consumes an opaque model-storage adapter. Migrate
`ProposalKVCache` to logical identities, make Qwen3.5 MTP construct attention
tables only from temporary leases, and select either a generation-aware
direct allocator or the residency manager at registration time.

**Tech Stack:** Python 3, dataclasses, typing protocols, PyTorch tensors,
pinned CPU memory, CUDA streams/events behind an injectable backend,
TinyLLMForge proposal lifecycle and Qwen3.5 MTP executor, pytest, and existing
dependency-light source-contract tests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or
  clean.
- Do not use subagents.
- Do not run GPU, remote, NCCL, or authority workloads while implementing
  this plan; those require separate explicit authorization.
- Do not terminate unrelated GPU processes.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- Exact greedy parity remains mandatory.
- Do not change verifier selection, fallback indexing, accepted-prefix
  semantics, target-KV transactions, recurrent side state, Scheduler
  behavior, n-gram, SAM, or unrelated MTP behavior.
- Do not add accepted-token replay, proposal-KV copy-on-commit, or
  proposal-KV rematerialization.
- Rejected, aborted, rolled-back, and sequence-released entries must retire
  with zero D2H.
- Active staged proposal entries remain GPU-pinned until finalize.
- Enabled V1 supports only block size one and unquantized FP16/BF16 proposal
  K/V.
- Enabled V1 requires
  `logical_entry_capacity == cpu_backing_capacity >
  gpu_slot_capacity > 0`.
- Fixed pinned CPU backing is mandatory; do not add pageable-memory fallback.
- Proposal-KV offload and Qwen3.5 MTP CUDA Graph execution are mutually
  exclusive in V1.
- Target-KV and proposal-KV storage, ownership, transfer streams, and counters
  remain separate.
- Default-off mode must allocate no CPU backing, create no transfer stream,
  enqueue no movement, and report zero H2D/D2H operations, entries, and bytes.
- Every task ends with local verification instead of a git commit.
- Local completion may establish only:

```text
PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT=ESTABLISHED
PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT=ESTABLISHED
QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION=ESTABLISHED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## File Structure

- Create `tinyvllm/engine/proposal_kv_allocator.py`
  - Frozen logical identity and physical lease records, allocator protocol,
    entry/lease validation helpers, and generation-aware direct allocator.
- Create `tinyvllm/engine/proposal_kv_residency.py`
  - Storage-adapter protocol, copy-backend protocol, synchronous test backend,
    CUDA copy backend, residency state machine, deterministic committed LRU,
    deferred retirement, and movement authority snapshot.
- Modify `tinyvllm/engine/proposal_kv_cache.py`
  - Store committed/staged logical identities, call only the allocator
    lifecycle, and preserve accepted-prefix finalize semantics.
- Modify `tinyvllm/engine/proposal_kv_lifecycle.py`
  - Report entry counts and logical ownership without durable physical slots.
- Modify `tinyvllm/engine/qwen35_mtp_registration.py`
  - Turn the current GPU tensor owner into a storage adapter and construct
    direct or offloaded proposal-KV allocators.
- Modify `tinyvllm/engine/qwen35_mtp_executor.py`
  - Acquire readable/writable leases, derive temporary attention mappings,
    and record consumer completion after each forward enqueue.
- Modify `tinyvllm/config.py`
  - Add default-off proposal-KV residency options and V1 validation.
- Modify `tinyvllm/engine/model_runner.py`
  - Pass proposal-KV configuration into Qwen3.5 MTP construction and expose
    allocator authority in existing snapshots.
- Create `tools/test_proposal_kv_allocator.py`
  - Direct allocator generation, lease, stale-identity, retirement, and
    zero-movement tests.
- Create `tools/test_proposal_kv_residency.py`
  - Dependency-light fake-storage/fake-copy tests for eviction, prefetch,
    writeback, batching, pending events, retirement, and counters.
- Modify `tools/test_proposal_kv_cache.py`
  - Logical-identity transaction matrix and no-copy/no-replay/no-D2H checks.
- Modify `tools/test_proposal_kv_lifecycle.py`
  - Logical-entry registration/finalize/release authority checks.
- Modify `tools/test_qwen35_mtp_physical_kv.py`
  - Qwen3.5 adapter geometry, direct mode, fixed CPU backing, and in-place
    accepted payload tests.
- Modify `tools/test_qwen35_mtp_executor.py`
  - Lease-derived bootstrap/decode attention mappings and completion markers.
- Modify `tools/test_qwen35_mtp_model_runner_integration.py`
  - Default-off registration, enabled-mode construction, graph conflict, and
    authority snapshot wiring.
- Modify `tools/test_qwen35_config_compatibility.py`
  - Configuration defaults and invalid-combination coverage.
- Create `tools/test_proposal_kv_residency_local_gate.py`
  - One dependency-light local contract gate covering direct mode,
    pressure-mode movement, transaction invariants, and Qwen3.5 integration.
- Modify `AGENT_HANDOFF_STATE.md`
  - Record commands, results, limitations, and the explicit no-GPU boundary.

---

### Task 1: Add Logical Identities, Leases, and the Direct Allocator

**Files:**
- Create: `tinyvllm/engine/proposal_kv_allocator.py`
- Create: `tools/test_proposal_kv_allocator.py`

**Interfaces:**
- Produces:
  - `ProposalKVEntryIdentity(logical_entry_id: int, generation: int)`
  - `ProposalKVResidencyLease(identities, physical_slot_ids,
    occupancy_generations)`
  - `ProposalKVEntryAllocator`
  - `DirectProposalKVAllocator`
- Consumes:
  - A physical store exposing `capacity`, `reserve_slots(count)`,
    `release_slots(slot_ids)`, and optional `authority_snapshot()`.

- [x] **Step 1: Write RED tests for generations, leases, and zero movement**

Create `tools/test_proposal_kv_allocator.py` with:

```python
from __future__ import annotations

import pytest

from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
    ProposalKVEntryIdentity,
    ProposalKVResidencyLease,
)


class _Store:
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.free = list(range(capacity))
        self.release_calls = []

    def reserve_slots(self, count):
        if count > len(self.free):
            raise RuntimeError("slots exhausted")
        result = tuple(self.free[:count])
        del self.free[:count]
        return result

    def release_slots(self, slot_ids):
        self.release_calls.append(tuple(slot_ids))
        self.free.extend(slot_ids)
        self.free.sort()


def test_direct_allocator_reuses_logical_row_with_new_generation():
    allocator = DirectProposalKVAllocator(_Store(capacity=1))
    first = allocator.reserve_entries(1)
    lease = allocator.ensure_writable(first)
    allocator.record_write_complete(lease)
    allocator.retire_entries(first, writeback=False)
    second = allocator.reserve_entries(1)
    assert first == (ProposalKVEntryIdentity(0, 1),)
    assert second == (ProposalKVEntryIdentity(0, 2),)
    with pytest.raises(RuntimeError, match="stale"):
        allocator.ensure_readable(first)


def test_direct_allocator_returns_generation_checked_lease():
    allocator = DirectProposalKVAllocator(_Store(capacity=2))
    identities = allocator.reserve_entries(2)
    lease = allocator.ensure_writable(identities)
    assert isinstance(lease, ProposalKVResidencyLease)
    assert lease.identities == identities
    assert lease.physical_slot_ids == (0, 1)
    assert lease.occupancy_generations == (1, 1)
    allocator.record_write_complete(lease)
    assert allocator.ensure_readable(identities) == lease


def test_direct_allocator_default_off_movement_is_exactly_zero():
    allocator = DirectProposalKVAllocator(_Store(capacity=2))
    identities = allocator.reserve_entries(1)
    lease = allocator.ensure_writable(identities)
    allocator.record_write_complete(lease)
    allocator.commit_entries(identities)
    snapshot = allocator.authority_snapshot()
    assert snapshot["h2d_operation_count"] == 0
    assert snapshot["h2d_entry_count"] == 0
    assert snapshot["h2d_bytes"] == 0
    assert snapshot["d2h_operation_count"] == 0
    assert snapshot["d2h_entry_count"] == 0
    assert snapshot["d2h_bytes"] == 0
```

- [ ] **Step 2: Run the new tests RED**

Run:

```bash
python3 -m pytest -q tools/test_proposal_kv_allocator.py
```

Expected: collection fails because
`tinyvllm.engine.proposal_kv_allocator` does not exist.

- [x] **Step 3: Implement frozen records and the allocator protocol**

Create `tinyvllm/engine/proposal_kv_allocator.py` with these public
definitions:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProposalKVEntryIdentity:
    logical_entry_id: int
    generation: int


@dataclass(frozen=True)
class ProposalKVResidencyLease:
    identities: tuple[ProposalKVEntryIdentity, ...]
    physical_slot_ids: tuple[int, ...]
    occupancy_generations: tuple[int, ...]


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

    def authority_snapshot(self) -> dict: ...
```

Add internal entry records with states `free`, `reserved`,
`active_staged`, and `committed`. Validate positive generations,
nonnegative logical/physical IDs, tuple lengths, duplicate identities, and
stale occupancy generations.

- [x] **Step 4: Implement `DirectProposalKVAllocator` minimally**

Implement a real logical-row table and generation counter:

```python
class DirectProposalKVAllocator:
    def __init__(self, physical_store):
        self.physical_store = physical_store
        self.logical_capacity = int(physical_store.capacity)
        self._free_logical_ids = list(range(self.logical_capacity))
        self._generations = [0] * self.logical_capacity
        self._entries = {}
        self._slot_occupancy_generations = [
            0
        ] * self.logical_capacity
```

Required behavior:

- reserve one physical slot per logical identity;
- increment logical and occupancy generations on every reuse;
- `ensure_writable` accepts only `reserved`;
- `record_write_complete` transitions leased entries to `active_staged`;
- `ensure_readable` accepts `active_staged` or `committed`;
- `commit_entries` accepts only `active_staged`;
- `retire_entries(..., writeback=True)` raises because direct mode has no
  offload need;
- `retire_entries(..., writeback=False)` releases slots and logical rows;
- stale identity or occupancy generation raises `RuntimeError`;
- authority movement counters are literal zeros, not derived estimates.

- [x] **Step 5: Run allocator tests GREEN**

Run:

```bash
python3 -m pytest -q tools/test_proposal_kv_allocator.py
```

Expected: all tests pass.

- [x] **Step 6: Run static verification**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/proposal_kv_allocator.py \
  tools/test_proposal_kv_allocator.py
git diff --check -- \
  tinyvllm/engine/proposal_kv_allocator.py \
  tools/test_proposal_kv_allocator.py
```

Expected: both commands exit zero.

---

### Task 2: Add the Dependency-Light Residency State Machine

**Files:**
- Create: `tinyvllm/engine/proposal_kv_residency.py`
- Create: `tools/test_proposal_kv_residency.py`

**Interfaces:**
- Consumes:
  - `ProposalKVEntryIdentity`
  - `ProposalKVResidencyLease`
  - `ProposalKVEntryAllocator`
- Produces:
  - `ProposalKVStorageAdapter`
  - `ProposalKVCopyBackend`
  - `SynchronousProposalKVCopyBackend`
  - `TorchProposalKVCopyBackend`
  - `ProposalKVResidencyManager`

- [x] **Step 1: Write RED fake-storage tests for dirty eviction and H2D**

Create `tools/test_proposal_kv_residency.py` with a CPU-only adapter:

```python
from __future__ import annotations

import torch

from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
)


class _Storage:
    def __init__(self, logical_capacity=4, gpu_capacity=2):
        self.logical_capacity = logical_capacity
        self.gpu_capacity = gpu_capacity
        self.block_size = 1
        self.dtype = torch.float16
        self.gpu_key_cache = torch.zeros(gpu_capacity, 1, 1, 2)
        self.gpu_value_cache = torch.zeros_like(self.gpu_key_cache)
        self.cpu_key_cache = torch.zeros(logical_capacity, 1, 1, 2)
        self.cpu_value_cache = torch.zeros_like(self.cpu_key_cache)

    def entry_nbytes(self):
        return 16

    def copy_gpu_to_cpu(self, rows):
        for logical_id, slot_id in rows:
            self.cpu_key_cache[logical_id].copy_(
                self.gpu_key_cache[slot_id]
            )
            self.cpu_value_cache[logical_id].copy_(
                self.gpu_value_cache[slot_id]
            )

    def copy_cpu_to_gpu(self, rows):
        for logical_id, slot_id in rows:
            self.gpu_key_cache[slot_id].copy_(
                self.cpu_key_cache[logical_id]
            )
            self.gpu_value_cache[slot_id].copy_(
                self.cpu_value_cache[logical_id]
            )
```

Add tests:

```python
def _manager():
    return ProposalKVResidencyManager(
        storage=_Storage(),
        copy_backend=SynchronousProposalKVCopyBackend(),
    )


def test_dirty_committed_victim_writes_back_then_prefetches():
    manager = _manager()
    first = manager.reserve_entries(1)
    first_write = manager.ensure_writable(first)
    manager.storage.gpu_key_cache[
        first_write.physical_slot_ids[0]
    ].fill_(7)
    manager.record_write_complete(first_write)
    manager.commit_entries(first)

    second = manager.reserve_entries(1)
    second_write = manager.ensure_writable(second)
    manager.record_write_complete(second_write)
    manager.commit_entries(second)

    third = manager.reserve_entries(1)
    manager.ensure_writable(third)
    assert manager.authority_snapshot()["d2h_entry_count"] == 1

    restored = manager.ensure_readable(first)
    assert manager.authority_snapshot()["h2d_entry_count"] == 1
    torch.testing.assert_close(
        manager.storage.gpu_key_cache[
            restored.physical_slot_ids[0]
        ],
        torch.full((1, 1, 2), 7, dtype=torch.float16),
    )


def test_rejected_entry_never_writes_back():
    manager = _manager()
    identity = manager.reserve_entries(1)
    lease = manager.ensure_writable(identity)
    manager.record_write_complete(lease)
    manager.retire_entries(identity, writeback=False)
    snapshot = manager.authority_snapshot()
    assert snapshot["rejected_entry_count"] == 1
    assert snapshot["rejected_entry_d2h_count"] == 0
    assert snapshot["rejected_entry_d2h_bytes"] == 0
```

- [x] **Step 2: Add RED tests for LRU, batching, and retirement waits**

Use a fake completion object:

```python
class _Completion:
    def __init__(self, complete=False):
        self.complete = complete

    def query(self):
        return self.complete
```

Test:

- deterministic least-recently-read committed victim;
- active staged entries are never victims;
- two compatible dirty victims increment `d2h_operation_count` once and
  `d2h_entry_count` twice;
- two compatible prefetches increment `h2d_operation_count` once and
  `h2d_entry_count` twice;
- retirement with an incomplete consumer completion leaves the slot
  unavailable;
- draining the completion recycles the slot;
- old leases fail after slot occupancy generation increments;
- clean eviction performs no D2H.

- [ ] **Step 3: Run residency tests RED**

Run:

```bash
python3 -m pytest -q tools/test_proposal_kv_residency.py
```

Expected: collection fails because
`tinyvllm.engine.proposal_kv_residency` does not exist.

- [x] **Step 4: Implement storage and copy protocols**

Create `tinyvllm/engine/proposal_kv_residency.py` with:

```python
class ProposalKVStorageAdapter(Protocol):
    logical_capacity: int
    gpu_capacity: int
    block_size: int
    dtype: torch.dtype

    def entry_nbytes(self) -> int: ...
    def copy_gpu_to_cpu(
        self,
        rows: tuple[tuple[int, int], ...],
    ) -> None: ...
    def copy_cpu_to_gpu(
        self,
        rows: tuple[tuple[int, int], ...],
    ) -> None: ...


class ProposalKVCopyCompletion(Protocol):
    def query(self) -> bool: ...
    def wait_current_stream(self) -> None: ...


class ProposalKVCopyBackend(Protocol):
    def enqueue_h2d(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion: ...

    def enqueue_d2h(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion: ...

    def record_consumer_completion(
        self,
    ) -> ProposalKVCopyCompletion: ...
```

The synchronous backend performs copies immediately and returns a completion
whose `query()` is always true. The torch backend owns one copy stream,
records CUDA events, orders D2H after the current stream, and makes the
current stream wait for H2D completion. Do not synchronize the device.

- [x] **Step 5: Implement residency records and invariants**

Use focused internal records:

```python
@dataclass
class _LogicalEntry:
    generation: int = 0
    logical_state: str = "free"
    residency_state: str = "none"
    physical_slot_id: int | None = None
    cpu_valid: bool = False
    dirty: bool = False
    last_access_ordinal: int = 0
    pending_completion: object | None = None


@dataclass
class _PhysicalOccupancy:
    generation: int = 0
    identity: ProposalKVEntryIdentity | None = None
    consumer_completion: object | None = None
    retiring: bool = False
```

Implement exact states from the spec:

```text
logical: free, reserved, active_staged, committed, retiring
residency: none, gpu_dirty, gpu_clean, cpu_clean, h2d_pending, d2h_pending
```

Reject combinations not listed in the design. Keep fixed logical and physical
tables; never grow them dynamically.

- [x] **Step 6: Implement writable/readable acquisition and deterministic LRU**

Implement:

```python
class ProposalKVResidencyManager:
    def reserve_entries(self, count): ...
    def ensure_writable(self, identities): ...
    def ensure_readable(self, identities): ...
    def record_write_complete(self, lease): ...
    def record_read_complete(self, lease): ...
    def commit_entries(self, identities): ...
    def retire_entries(self, identities, *, writeback): ...
    def poll_retirements(self): ...
    def authority_snapshot(self): ...
```

Rules:

- writable entries are new and never H2D;
- readable CPU entries are grouped into one backend call when batching is
  enabled;
- victim key is `(last_access_ordinal, logical_entry_id)`;
- only committed, nonpending, nonretiring entries are victims;
- dirty victim D2H completes before CPU becomes authoritative;
- slot occupancy generation increments on every binding;
- return leases only after required H2D waits are installed;
- `record_write_complete` marks payload dirty and active staged;
- `record_read_complete` attaches one consumer completion to every occupancy
  in the lease;
- rejected retirement increments rejected counters and never calls D2H;
- physical slot and logical row recycling wait for all associated
  completions.

- [x] **Step 7: Implement exact authority counters**

Initialize every required counter to zero and expose all gauges named in the
spec. Counter increments must occur at enqueue time for operations/entries/
bytes and at lifecycle transition time for logical counters.

Include:

```python
"accepted_entry_copy_count": 0,
"accepted_entry_replay_count": 0,
"accepted_entry_rematerialization_count": 0,
```

These values remain literal zero in V1; no code path may increment them.

- [x] **Step 8: Run residency tests GREEN**

Run:

```bash
python3 -m pytest -q tools/test_proposal_kv_residency.py
```

Expected: all tests pass.

- [x] **Step 9: Run focused static verification**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/proposal_kv_residency.py \
  tools/test_proposal_kv_residency.py
git diff --check -- \
  tinyvllm/engine/proposal_kv_residency.py \
  tools/test_proposal_kv_residency.py
```

Expected: both commands exit zero.

---

### Task 3: Migrate `ProposalKVCache` from Slots to Logical Entries

**Files:**
- Modify: `tinyvllm/engine/proposal_kv_cache.py`
- Modify: `tinyvllm/engine/proposal_kv_lifecycle.py`
- Modify: `tools/test_proposal_kv_cache.py`
- Modify: `tools/test_proposal_kv_lifecycle.py`

**Interfaces:**
- Consumes:
  - `ProposalKVEntryAllocator`
  - `ProposalKVEntryIdentity`
- Produces:
  - `ProposalKVSequenceState.committed_entry_identities`
  - `ProposalKVTransaction.staged_entry_identities`
  - `ProposalKVFinalizeTicket.retire_entry_identities`
  - `ProposalKVCache.committed_entry_identities(sequence_id)`

- [x] **Step 1: Convert cache tests to logical identities and keep them RED**

Replace `_PhysicalStore` with:

```python
class _Allocator:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.free = list(range(capacity))
        self.generations = [0] * capacity
        self.commit_calls = []
        self.retire_calls = []

    def reserve_entries(self, count):
        selected = tuple(self.free[:count])
        del self.free[:count]
        result = []
        for logical_id in selected:
            self.generations[logical_id] += 1
            result.append(
                ProposalKVEntryIdentity(
                    logical_id,
                    self.generations[logical_id],
                )
            )
        return tuple(result)

    def commit_entries(self, identities):
        self.commit_calls.append(tuple(identities))

    def retire_entries(self, identities, *, writeback):
        assert writeback is False
        self.retire_calls.append(tuple(identities))
        self.free.extend(
            identity.logical_entry_id for identity in identities
        )
        self.free.sort()

    def authority_snapshot(self):
        return {"owned_entry_count": self.capacity - len(self.free)}
```

Update the finalize matrix to assert:

```python
assert ticket.retire_entry_identities == staged[commit_count:]
assert cache.committed_entry_identities(7) == staged[:commit_count]
assert allocator.commit_calls == (
    [staged[:commit_count]] if commit_count else []
)
assert allocator.retire_calls == (
    [staged[commit_count:]] if staged[commit_count:] else []
)
```

Add explicit tests that rollback, abort, and sequence release call
`retire_entries(..., writeback=False)`.

- [ ] **Step 2: Run cache/lifecycle tests RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py
```

Expected: failures reference missing logical-entry fields or methods.

- [x] **Step 3: Rename durable cache records**

In `tinyvllm/engine/proposal_kv_cache.py`, use:

```python
@dataclass
class ProposalKVSequenceState:
    sequence_id: int
    sequence_epoch: int
    committed_entry_identities: tuple[
        ProposalKVEntryIdentity, ...
    ] = ()
    active_transaction_id: str | None = None
    active_ticket_id: str | None = None


@dataclass
class ProposalKVTransaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    original_committed_length: int
    staged_entry_identities: tuple[
        ProposalKVEntryIdentity, ...
    ]
    materialized_entry_count: int = 0
    state: str = "reserved"


@dataclass
class ProposalKVFinalizeTicket:
    ticket_id: str
    transaction_id: str
    commit_entry_count: int
    retire_entry_identities: tuple[
        ProposalKVEntryIdentity, ...
    ]
    state: str = "prepared"
```

Remove durable `staged_slot_ids`, `committed_slot_ids`, and
`release_slot_ids`. Do not leave aliases that encourage executors to persist
physical slots.

- [x] **Step 4: Replace physical-store ownership with allocator lifecycle**

Change construction to:

```python
def __init__(
    self,
    entry_allocator: ProposalKVEntryAllocator,
):
    self._entry_allocator = entry_allocator
```

Then:

- `begin()` calls `reserve_entries`;
- `commit_finalize()` calls `commit_entries(accepted)` and
  `retire_entries(rejected, writeback=False)`;
- `rollback_finalize()` and `abort()` retire all staged entries without
  writeback;
- `release_sequence()` retires all committed entries without writeback;
- sequence and transaction validation remains unchanged apart from field
  names;
- accepted-prefix order remains exact.

Expose:

```python
@property
def entry_allocator(self) -> ProposalKVEntryAllocator: ...

def committed_entry_identities(
    self,
    sequence_id: int,
) -> tuple[ProposalKVEntryIdentity, ...]: ...
```

- [x] **Step 5: Update lifecycle authority rows**

In `proposal_kv_lifecycle.py`, derive counts from
`transaction.staged_entry_identities`. Preserve token rule:

```python
len(proposal.token_ids) == staged_entry_count + 1
```

Snapshots may include logical IDs and generations for diagnostics, but no
physical slot IDs.

- [x] **Step 6: Run cache/lifecycle tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py
```

Expected: all tests pass.

- [x] **Step 7: Prove no durable slot fields remain**

Run:

```bash
if rg -n \
  'staged_slot_ids|committed_slot_ids|release_slot_ids|owned_slot_count' \
  tinyvllm/engine/proposal_kv_cache.py \
  tinyvllm/engine/proposal_kv_lifecycle.py; then
  exit 1
fi
```

Expected: exit zero with no matches.

---

### Task 4: Adapt Qwen3.5 MTP Storage and Allocator Construction

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_registration.py`
- Modify: `tools/test_qwen35_mtp_physical_kv.py`

**Interfaces:**
- Consumes:
  - `DirectProposalKVAllocator`
  - `ProposalKVResidencyManager`
  - `SynchronousProposalKVCopyBackend`
  - `TorchProposalKVCopyBackend`
- Produces:
  - `Qwen35MTPProposalKVStorage`
  - `build_qwen35_mtp_proposal_kv_allocator(...)`

- [x] **Step 1: Write RED adapter and construction tests**

Replace store-only helpers with:

```python
def _storage(
    *,
    logical_capacity=8,
    gpu_capacity=4,
    dtype=torch.float16,
):
    return Qwen35MTPProposalKVStorage(
        logical_capacity=logical_capacity,
        gpu_capacity=gpu_capacity,
        num_kv_heads=2,
        head_dim=4,
        dtype=dtype,
        device="cpu",
        allocate_pinned_cpu=False,
    )
```

The `allocate_pinned_cpu=False` escape hatch is test-only and accepted only
for CPU storage. Production CUDA construction must always pin CPU backing.

Test:

- GPU shape `(gpu_capacity, 1, num_kv_heads, head_dim)`;
- CPU shape `(logical_capacity, 1, num_kv_heads, head_dim)`;
- `entry_nbytes()` equals K plus V payload bytes;
- attention backend binds only GPU tensors;
- direct builder allocates no CPU backing;
- offload builder returns `ProposalKVResidencyManager`;
- float32, block size other than one, and pageable CUDA backing are rejected;
- accepted payload remains in the same GPU occupancy after
  `commit_entries`.

- [ ] **Step 2: Run Qwen3.5 physical-KV tests RED**

Run:

```bash
python3 -m pytest -q tools/test_qwen35_mtp_physical_kv.py
```

Expected: failures reference missing storage adapter and builder.

- [x] **Step 3: Split GPU tensor ownership from allocation policy**

Replace `Qwen35MTPPhysicalSlotStore` with a storage class exposing:

```python
class Qwen35MTPProposalKVStorage:
    block_size = 1

    def __init__(
        self,
        *,
        logical_capacity: int,
        gpu_capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
        allocate_cpu_backing: bool,
        allocate_pinned_cpu: bool = True,
    ): ...

    def bind_attention_backend(self, backend) -> None: ...
    def entry_nbytes(self) -> int: ...
    def copy_gpu_to_cpu(self, rows) -> None: ...
    def copy_cpu_to_gpu(self, rows) -> None: ...
    def reserve_slots(self, count: int) -> tuple[int, ...]: ...
    def release_slots(self, slot_ids: tuple[int, ...]) -> None: ...
```

Direct mode sets `allocate_cpu_backing=False` and uses
`logical_capacity == gpu_capacity`. Offload mode allocates fixed CPU K/V rows
with `pin_memory=True`. The direct allocator alone uses `reserve_slots` and
`release_slots`; the residency manager owns its own GPU free-slot table and
must not call those two compatibility methods.

- [x] **Step 4: Add the allocator builder**

Implement:

```python
def build_qwen35_mtp_proposal_kv_allocator(
    *,
    offload_enabled: bool,
    logical_entry_capacity: int,
    gpu_slot_capacity: int,
    cpu_backing_capacity: int,
    async_copy: bool,
    batch_copy: bool,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str | torch.device,
):
    ...
```

Rules:

- disabled: construct a GPU-only storage compatible with
  `DirectProposalKVAllocator`;
- enabled: validate fixed capacities/dtype/device and construct
  `ProposalKVResidencyManager`;
- asynchronous mode uses `TorchProposalKVCopyBackend`;
- dependency-light CPU tests may inject the synchronous backend through a
  private keyword used only by tests;
- bind the same GPU cache tensors to the MTP attention backend after
  construction.

- [x] **Step 5: Run Qwen3.5 physical-KV tests GREEN**

Run:

```bash
python3 -m pytest -q tools/test_qwen35_mtp_physical_kv.py
```

Expected: all tests pass.

- [x] **Step 6: Run focused static verification**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/qwen35_mtp_registration.py \
  tools/test_qwen35_mtp_physical_kv.py
git diff --check -- \
  tinyvllm/engine/qwen35_mtp_registration.py \
  tools/test_qwen35_mtp_physical_kv.py
```

Expected: both commands exit zero.

---

### Task 5: Integrate Temporary Leases into the Qwen3.5 MTP Executor

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`
- Modify: `tools/test_qwen35_mtp_executor.py`

**Interfaces:**
- Consumes:
  - `ProposalKVCache.entry_allocator`
  - `ProposalKVTransaction.staged_entry_identities`
  - `ProposalKVCache.committed_entry_identities`
- Produces:
  - Lease-derived `slot_mapping` and `block_tables`
  - Explicit read/write completion recording after forward enqueue

- [x] **Step 1: Add RED bootstrap lease tests**

Extend the fake allocator in `tools/test_qwen35_mtp_executor.py` to return
nonidentity physical slots:

```python
logical identities:  (entry 0, entry 1)
physical slots:      (7, 3)
occupancy generations: (4, 9)
```

Assert bootstrap context receives:

```python
slot_mapping.tolist() == [7, 3]
```

Assert the transaction itself still contains only logical identities and that
`record_write_complete()` receives the exact writable lease after
`forward_hidden()` returns.

- [x] **Step 2: Add RED autoregressive visible-order tests**

Set committed identities to leases on slots `(6, 2)` and staged identities to
slots `(7, 3, 5)`. At step one, assert:

```python
slot_mapping.tolist() == [3]
block_tables.tolist() == [[6, 2, 7, 3]]
context_lens.tolist() == [4]
```

Assert one read completion protects every visible occupancy and one write
completion protects the current destination. The current destination is also
read inside the same attention forward, but the post-forward write completion
event is ordered after both operations and therefore protects that occupancy
from reuse without a duplicate read marker.

The current staged entry is not passed to `ensure_readable()` before the
forward. Its K/V is produced and then consumed inside the same attention
forward. The executor obtains it through `ensure_writable()` and appends that
physical slot to the readable-prefix slots when constructing the block table.

- [ ] **Step 3: Run executor tests RED**

Run:

```bash
python3 -m pytest -q tools/test_qwen35_mtp_executor.py
```

Expected: failures show direct use of removed `staged_slot_ids` or missing
completion calls.

- [x] **Step 4: Add focused lease helpers to the executor**

Implement helpers equivalent to:

```python
def _writable_slots(self, identities):
    return self.proposal_kv_cache.entry_allocator.ensure_writable(
        identities
    )

def _readable_slots(self, identities):
    return self.proposal_kv_cache.entry_allocator.ensure_readable(
        identities
    )
```

Do not cache returned physical slots across forward calls.

- [x] **Step 5: Rewrite bootstrap and proposal-step mapping construction**

Bootstrap:

```python
writable_lease = allocator.ensure_writable(
    transaction.staged_entry_identities
)
slot_mapping = torch.tensor(
    writable_lease.physical_slot_ids,
    dtype=torch.int32,
    device=target_hidden.device,
)
```

Proposal step:

```python
committed = cache.committed_entry_identities(
    transaction.sequence_id
)
read_prefix = (
    committed
    + transaction.staged_entry_identities[:step]
)
read_lease = allocator.ensure_readable(read_prefix)
write_lease = allocator.ensure_writable(
    (transaction.staged_entry_identities[step],)
)
visible_physical_slots = (
    read_lease.physical_slot_ids
    + write_lease.physical_slot_ids
)
```

The current staged identity must still be `reserved`; a second writable
acquisition after it becomes active fails closed. Build `block_tables` from
`visible_physical_slots` and `slot_mapping` from the one-slot writable lease.
Preserve all token, position, hidden, greedy-selection, and TP broadcast
logic.

- [x] **Step 6: Record completion after forward enqueue**

Wrap only the model call:

```python
try:
    output = self.module.forward_step(...)
except BaseException:
    allocator.record_read_complete(read_lease)
    allocator.record_write_complete(write_lease)
    raise
allocator.record_read_complete(read_lease)
allocator.record_write_complete(write_lease)
return output
```

The allocator backend records current-stream completion without host
synchronization. The write completion is recorded after the entire attention
forward, so it protects both the current-slot write and its same-forward read.
Ensure cleanup also records queued work before transaction abort can retire
slots.

- [x] **Step 7: Run executor and lifecycle tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_mtp_executor.py \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py
```

Expected: all tests pass.

- [x] **Step 8: Prove the executor no longer persists proposal slots**

Run:

```bash
if rg -n 'staged_slot_ids|committed_slot_ids' \
  tinyvllm/engine/qwen35_mtp_executor.py; then
  exit 1
fi
rg -n \
  'ensure_readable|ensure_writable|record_read_complete|record_write_complete' \
  tinyvllm/engine/qwen35_mtp_executor.py
```

Expected: first command exits zero with no matches; second prints all four
lease operations.

---

### Task 6: Add Configuration and Model-Runner Construction

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_qwen35_config_compatibility.py`
- Modify: `tools/test_qwen35_mtp_model_runner_integration.py`

**Interfaces:**
- Consumes:
  - `build_qwen35_mtp_proposal_kv_allocator`
- Produces configuration fields:
  - `proposal_kv_offload_enabled`
  - `proposal_kv_logical_entry_capacity`
  - `proposal_kv_gpu_slot_capacity`
  - `proposal_kv_cpu_backing_capacity`
  - `proposal_kv_async_copy`
  - `proposal_kv_batch_copy`

- [x] **Step 1: Write RED configuration tests**

Assert defaults:

```python
assert config.proposal_kv_offload_enabled is False
assert config.proposal_kv_logical_entry_capacity == 0
assert config.proposal_kv_gpu_slot_capacity == 0
assert config.proposal_kv_cpu_backing_capacity == 0
assert config.proposal_kv_async_copy is True
assert config.proposal_kv_batch_copy is True
```

Assert enabled validation accepts:

```text
logical=16, cpu=16, gpu=4
qwen35_mtp_enabled=true
qwen35_mtp_cuda_graphs=false
```

Assert it rejects:

- logical not equal to CPU;
- logical less than or equal to GPU;
- zero GPU capacity;
- offload enabled without `qwen35_mtp_enabled=true`; the independent Qwen3
  storage adapter is deliberately deferred;
- `qwen35_mtp_cuda_graphs=true`;
- boolean values supplied where integer capacities are required.

- [x] **Step 2: Write RED model-runner construction tests**

Inject builder dependencies and assert:

- default-off passes `offload_enabled=False` and constructs
  `DirectProposalKVAllocator`;
- enabled mode forwards all six configuration values;
- the executor receives `ProposalKVCache(entry_allocator)`;
- authority snapshot includes
  `proposal_kv_cache["entry_allocator"]`;
- target-KV authority keys and counters are unchanged.

- [ ] **Step 3: Run configuration/integration tests RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_config_compatibility.py \
  tools/test_qwen35_mtp_model_runner_integration.py
```

Expected: failures reference missing configuration fields and old physical
store construction.

- [x] **Step 4: Add configuration fields and exact validation**

Add the six fields to `Config`. In `__post_init__`:

```python
if self.proposal_kv_offload_enabled:
    if (
        self.proposal_kv_logical_entry_capacity
        != self.proposal_kv_cpu_backing_capacity
        or self.proposal_kv_logical_entry_capacity
        <= self.proposal_kv_gpu_slot_capacity
        or self.proposal_kv_gpu_slot_capacity <= 0
    ):
        raise ValueError(
            "proposal KV offload requires logical == cpu > gpu > 0"
        )
    if self.qwen35_mtp_cuda_graphs:
        raise ValueError(
            "proposal KV offload is incompatible with "
            "Qwen3.5 MTP CUDA graphs"
        )
```

Validate every boolean and nonnegative capacity with the repository's
existing strict bool/int style. Disabled mode permits zero capacities and
derives direct capacity from the existing MTP registration requirement.

- [x] **Step 5: Replace MTP physical-store construction**

In model-runner registration dependencies, replace
`build_physical_slot_store` with
`build_proposal_kv_allocator`. Construct:

```python
entry_allocator = build_qwen35_mtp_proposal_kv_allocator(...)
proposal_kv_cache = ProposalKVCache(entry_allocator)
```

Bind the allocator's storage GPU tensors to the MTP attention backend before
executor construction. Preserve descriptor validation and all-rank
registration consensus.

- [x] **Step 6: Extend authority snapshots without changing target counters**

`ProposalKVCache.authority_snapshot()` must nest:

```python
"entry_allocator": allocator.authority_snapshot()
```

Do not merge proposal movement into `KVOffloadMVP0` fields.

- [x] **Step 7: Run configuration/integration tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_config_compatibility.py \
  tools/test_qwen35_mtp_model_runner_integration.py
```

Expected: all tests pass.

- [x] **Step 8: Run default-off regression tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_mtp.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_qwen35_mtp_executor_graph_registration.py \
  tools/test_qwen35_mtp_cuda_graph_backend.py \
  tools/test_model_runner_proposal_executor.py
```

Expected: all tests pass with proposal-KV offload disabled.

---

### Task 7: Add the Local Contract Gate and Full Dependency-Light Suite

**Files:**
- Create: `tools/test_proposal_kv_residency_local_gate.py`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes all earlier local contracts.
- Produces one local-only evidence command and explicit terminal
  classifications.

- [x] **Step 1: Write the local gate**

Create `tools/test_proposal_kv_residency_local_gate.py` with one test per
classification:

```python
def test_logical_physical_decoupling_contract():
    ...

def test_residency_transaction_contract():
    ...

def test_qwen35_mtp_local_integration_contract():
    ...

def test_default_off_zero_movement_contract():
    ...

def test_rejected_suffix_zero_d2h_contract():
    ...

def test_local_gate_keeps_gpu_and_performance_unestablished():
    assert TERMINAL_CLASSIFICATIONS == {
        "PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT": (
            "ESTABLISHED"
        ),
        "PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT": "ESTABLISHED",
        "QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION": (
            "ESTABLISHED"
        ),
        "REAL_PROPOSAL_KV_MOVEMENT": "NOT_ESTABLISHED",
        "PERFORMANCE_IMPROVEMENT": "NOT_ESTABLISHED",
        "PHASE_1": "NOT_ACHIEVED",
        "PROMOTION": "NOT_PROMOTABLE",
    }
```

The first three tests must instantiate real local classes, not inspect source
strings alone. Use the synchronous backend and CPU tensors only.

- [x] **Step 2: Run the focused suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_proposal_kv_allocator.py \
  tools/test_proposal_kv_residency.py \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_config_compatibility.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_proposal_kv_residency_local_gate.py
```

Expected: all tests pass and pytest reports zero failures.

- [x] **Step 3: Run relevant speculative-runtime regressions**

Run:

```bash
python3 -m pytest -q \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_side_state.py \
  tools/test_autoregressive_draft_executor.py \
  tools/test_qwen3_draft_backend.py
```

Expected: all tests pass. These tests do not establish Qwen3 residency reuse;
they prove the MTP-first change did not break the independent drafter or
generic target transaction.

- [x] **Step 4: Run compile and scoped diff checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/proposal_kv_allocator.py \
  tinyvllm/engine/proposal_kv_residency.py \
  tinyvllm/engine/proposal_kv_cache.py \
  tinyvllm/engine/proposal_kv_lifecycle.py \
  tinyvllm/engine/qwen35_mtp_registration.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/test_proposal_kv_allocator.py \
  tools/test_proposal_kv_residency.py \
  tools/test_proposal_kv_residency_local_gate.py

git diff --check -- \
  tinyvllm/engine/proposal_kv_allocator.py \
  tinyvllm/engine/proposal_kv_residency.py \
  tinyvllm/engine/proposal_kv_cache.py \
  tinyvllm/engine/proposal_kv_lifecycle.py \
  tinyvllm/engine/qwen35_mtp_registration.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/test_proposal_kv_allocator.py \
  tools/test_proposal_kv_residency.py \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_config_compatibility.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_proposal_kv_residency_local_gate.py
```

Expected: both commands exit zero.

- [x] **Step 5: Record evidence and limitations**

Update the top of `AGENT_HANDOFF_STATE.md` with:

- exact files changed;
- exact test commands and pass counts;
- compile and diff-check results;
- no GPU/remote/NCCL statement;
- default-off movement counters;
- pressure-mode fake-backend H2D/D2H evidence;
- rejected-suffix zero-D2H evidence;
- explicit statement that fake/synchronous copy is not real offload evidence;
- terminal classifications from the spec.

Do not overwrite older handoff evidence.

- [x] **Step 6: Re-run the focused suite after handoff edit**

Run the exact focused suite from Step 2 again.

Expected: all tests pass with zero failures.

---

### Task 8: Prepare but Do Not Run the Separate GPU Authority

**Files:**
- Create only after separate authorization:
  - `tools/qwen35_mtp_proposal_kv_residency_gate.py`
  - `tools/qwen35_mtp_proposal_kv_residency_worker.py`
  - `tools/verify_qwen35_mtp_proposal_kv_residency.py`
  - `tools/run_qwen35_mtp_proposal_kv_residency_remote.sh`
  - `tools/test_qwen35_mtp_proposal_kv_residency_gate.py`

**Interfaces:**
- Consumes local GREEN contracts and a separately approved GPU route.
- Produces real-copy, exact-parity, cleanup, memory, and performance
  artifacts.

- [ ] **Step 1: Stop at the authorization boundary**

Before creating or running any GPU authority artifact, report the completed
local evidence and request explicit permission for:

```text
GPU execution
remote access
NCCL / TP4
authority artifact creation
```

Expected: no GPU, remote, NCCL, or authority command runs without a fresh
user approval.

- [ ] **Step 2: Freeze the future matrix only after authorization**

The minimum matrix is:

```text
mode: direct, proposal-KV offload
TP: 1, 4
context: 4K and one of 16K/32K under proposal-GPU pressure
batch: 1, 4
decode: exact greedy
max proposal tokens: 4
```

The pressure cells must prove:

```text
logical_entry_capacity > gpu_slot_capacity
h2d_entry_count > 0
d2h_entry_count > 0
gpu_slot_rebindings > 0
```

- [ ] **Step 3: Freeze future acceptance criteria**

The verifier must require:

- exact generated-token parity with direct mode;
- equal accepted-token counts and target-forward counts;
- zero accepted copy/replay/rematerialization;
- zero rejected-suffix D2H;
- all-rank cleanup;
- real CUDA movement counters and bytes;
- separate target-KV and proposal-KV movement;
- TPOT, TTFT, throughput, peak GPU memory, and acceptance;
- no simulated copy accepted as movement evidence.

This task remains unexecuted until independently authorized.

## Plan Completion Boundary

Tasks 1-7 complete the local MTP-first subsystem. They do not complete Phase
1 and do not prove real movement or performance.

Task 8 is an authorization boundary, not permission to run. The independent
Qwen3 multi-layer storage adapter is a later plan after the Qwen3.5 MTP local
contract is GREEN; it must reuse the same identity, lease, residency, transfer,
retirement, and counter implementations.
