from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
)


class _Storage:

    def __init__(self, logical_capacity=4, gpu_capacity=2):
        self.logical_capacity = logical_capacity
        self.gpu_capacity = gpu_capacity
        self.block_size = 1
        self.dtype = "float16"
        self.gpu_key_cache = _FakeCache(gpu_capacity)
        self.gpu_value_cache = _FakeCache(gpu_capacity)
        self.cpu_key_cache = _FakeCache(logical_capacity)
        self.cpu_value_cache = _FakeCache(logical_capacity)

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


class _FakeRow:

    def __init__(self):
        self.values = [0, 0]

    def fill_(self, value):
        self.values[:] = [value, value]
        return self

    def copy_(self, other):
        self.values[:] = other.values
        return self


class _FakeCache:

    def __init__(self, capacity):
        self.rows = [_FakeRow() for _ in range(capacity)]

    def __getitem__(self, index):
        return self.rows[index]


class _Completion:

    def __init__(self, complete=False):
        self.complete = complete
        self.wait_count = 0

    def query(self):
        return self.complete

    def wait_current_stream(self):
        self.wait_count += 1


class _ManualConsumerBackend(SynchronousProposalKVCopyBackend):

    def __init__(self):
        self.consumer_completions = []

    def record_consumer_completion(self):
        completion = _Completion()
        self.consumer_completions.append(completion)
        return completion


def _manager(
    *,
    logical_capacity=4,
    gpu_capacity=2,
    copy_backend=None,
    batch_copy=True,
):
    return ProposalKVResidencyManager(
        storage=_Storage(
            logical_capacity=logical_capacity,
            gpu_capacity=gpu_capacity,
        ),
        copy_backend=(
            SynchronousProposalKVCopyBackend()
            if copy_backend is None
            else copy_backend
        ),
        batch_copy=batch_copy,
    )


def _materialize_and_commit(manager, value):
    identities = manager.reserve_entries(1)
    lease = manager.ensure_writable(identities)
    manager.storage.gpu_key_cache[
        lease.physical_slot_ids[0]
    ].fill_(value)
    manager.storage.gpu_value_cache[
        lease.physical_slot_ids[0]
    ].fill_(value + 1)
    manager.record_write_complete(lease)
    manager.commit_entries(identities)
    return identities, lease


def test_dirty_committed_victim_writes_back_then_prefetches():
    manager = _manager()
    first, _ = _materialize_and_commit(manager, 7)
    _materialize_and_commit(manager, 11)

    third = manager.reserve_entries(1)
    manager.ensure_writable(third)
    assert manager.authority_snapshot()["d2h_entry_count"] == 1

    manager.retire_entries(third, writeback=False)
    restored = manager.ensure_readable(first)
    assert manager.authority_snapshot()["h2d_entry_count"] == 1
    assert manager.storage.gpu_key_cache[
        restored.physical_slot_ids[0]
    ].values == [7, 7]


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
    assert snapshot["d2h_entry_count"] == 0


def test_deterministic_lru_evicts_least_recently_read_committed_entry():
    manager = _manager(logical_capacity=4, gpu_capacity=2)
    first, _ = _materialize_and_commit(manager, 3)
    second, _ = _materialize_and_commit(manager, 5)
    first_read = manager.ensure_readable(first)
    manager.record_read_complete(first_read)

    third = manager.reserve_entries(1)
    manager.ensure_writable(third)

    before = manager.authority_snapshot()["h2d_entry_count"]
    manager.ensure_readable(first)
    assert manager.authority_snapshot()["h2d_entry_count"] == before
    manager.retire_entries(third, writeback=False)
    manager.ensure_readable(second)
    assert manager.authority_snapshot()["h2d_entry_count"] == before + 1


def test_active_staged_entries_are_never_eviction_victims():
    manager = _manager(logical_capacity=2, gpu_capacity=1)
    first = manager.reserve_entries(1)
    first_lease = manager.ensure_writable(first)
    manager.record_write_complete(first_lease)
    second = manager.reserve_entries(1)
    with pytest.raises(RuntimeError, match="eligible"):
        manager.ensure_writable(second)


def test_compatible_dirty_evictions_and_prefetches_are_batched():
    manager = _manager(logical_capacity=6, gpu_capacity=2)
    first, _ = _materialize_and_commit(manager, 3)
    second, _ = _materialize_and_commit(manager, 5)

    replacements = manager.reserve_entries(2)
    replacement_lease = manager.ensure_writable(replacements)
    snapshot = manager.authority_snapshot()
    assert snapshot["d2h_operation_count"] == 1
    assert snapshot["d2h_entry_count"] == 2
    manager.record_write_complete(replacement_lease)
    manager.retire_entries(replacements, writeback=False)

    restored = manager.ensure_readable(first + second)
    snapshot = manager.authority_snapshot()
    assert snapshot["h2d_operation_count"] == 1
    assert snapshot["h2d_entry_count"] == 2
    assert len(restored.physical_slot_ids) == 2


def test_batch_copy_false_counts_one_operation_per_entry():
    manager = _manager(
        logical_capacity=6,
        gpu_capacity=2,
        batch_copy=False,
    )
    _materialize_and_commit(manager, 3)
    _materialize_and_commit(manager, 5)
    replacements = manager.reserve_entries(2)
    manager.ensure_writable(replacements)
    snapshot = manager.authority_snapshot()
    assert snapshot["d2h_operation_count"] == 2
    assert snapshot["d2h_entry_count"] == 2


def test_retirement_waits_for_incomplete_consumer_completion():
    backend = _ManualConsumerBackend()
    manager = _manager(
        logical_capacity=2,
        gpu_capacity=1,
        copy_backend=backend,
    )
    first = manager.reserve_entries(1)
    first_lease = manager.ensure_writable(first)
    manager.record_write_complete(first_lease)
    manager.retire_entries(first, writeback=False)
    snapshot = manager.authority_snapshot()
    assert snapshot["retiring_entry_count"] == 1
    assert snapshot["free_gpu_slot_count"] == 0
    assert snapshot["retirement_wait_count"] == 1

    second = manager.reserve_entries(1)
    with pytest.raises(RuntimeError, match="eligible"):
        manager.ensure_writable(second)

    backend.consumer_completions[0].complete = True
    manager.poll_retirements()
    second_lease = manager.ensure_writable(second)
    assert second_lease.physical_slot_ids == first_lease.physical_slot_ids


def test_old_lease_fails_after_slot_occupancy_generation_changes():
    manager = _manager(logical_capacity=2, gpu_capacity=1)
    first = manager.reserve_entries(1)
    old_lease = manager.ensure_writable(first)
    manager.record_write_complete(old_lease)
    manager.retire_entries(first, writeback=False)

    second = manager.reserve_entries(1)
    new_lease = manager.ensure_writable(second)
    assert new_lease.occupancy_generations[0] > (
        old_lease.occupancy_generations[0]
    )
    with pytest.raises(RuntimeError, match="stale"):
        manager.record_read_complete(old_lease)


def test_clean_eviction_performs_no_additional_d2h():
    manager = _manager(logical_capacity=4, gpu_capacity=1)
    first, _ = _materialize_and_commit(manager, 7)
    replacement = manager.reserve_entries(1)
    replacement_lease = manager.ensure_writable(replacement)
    manager.record_write_complete(replacement_lease)
    manager.retire_entries(replacement, writeback=False)
    manager.ensure_readable(first)
    d2h_before = manager.authority_snapshot()["d2h_entry_count"]

    third = manager.reserve_entries(1)
    manager.ensure_writable(third)
    snapshot = manager.authority_snapshot()
    assert snapshot["d2h_entry_count"] == d2h_before
    assert snapshot["clean_eviction_entry_count"] == 1


def test_blockwise_adapter_stages_window_without_evicting_protected_writes():
    manager = _manager(logical_capacity=5, gpu_capacity=2)
    first, _ = _materialize_and_commit(manager, 7)
    second, _ = _materialize_and_commit(manager, 11)

    protected = manager.reserve_entries(1)
    protected_lease = manager.ensure_writable(protected)
    manager.record_write_complete(protected_lease)

    adapter = manager.blockwise_attention_adapter
    first_id = first[0].logical_entry_id
    second_id = second[0].logical_entry_id
    protected_id = protected[0].logical_entry_id
    assert protected_id in adapter.logical_to_slot
    assert first_id not in adapter.logical_to_slot
    assert second_id in adapter.logical_to_slot

    adapter.ensure_resident(
        [first_id],
        require_valid=True,
        future_logical_blocks={first_id},
        protected_logical_blocks={protected_id},
    )
    adapter.wait_for_blocks([first_id], clear_pending=True)

    assert first_id in adapter.logical_to_slot
    assert protected_id in adapter.logical_to_slot
    assert second_id not in adapter.logical_to_slot
    snapshot = manager.authority_snapshot()
    assert snapshot["d2h_entry_count"] == 2
    assert snapshot["h2d_entry_count"] == 1


def test_blockwise_adapter_accepts_resident_reserved_write_in_read_window():
    manager = _manager(logical_capacity=5, gpu_capacity=2)
    first, _ = _materialize_and_commit(manager, 7)
    _materialize_and_commit(manager, 11)

    write_identity = manager.reserve_entries(1)
    write_lease = manager.ensure_writable(write_identity)
    first_id = first[0].logical_entry_id
    write_id = write_identity[0].logical_entry_id
    assert write_lease.physical_slot_ids == (0,)

    mapping = manager.blockwise_attention_adapter.ensure_resident(
        [first_id, write_id],
        require_valid=True,
        future_logical_blocks={first_id, write_id},
        protected_logical_blocks={write_id},
    )

    assert set(mapping) == {first_id, write_id}
    assert write_id in manager.blockwise_attention_adapter.logical_to_slot
    assert first_id in manager.blockwise_attention_adapter.logical_to_slot
