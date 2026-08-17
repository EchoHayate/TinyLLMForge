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
