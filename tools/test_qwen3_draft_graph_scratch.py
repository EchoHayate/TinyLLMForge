from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.autoregressive_draft_graph import (
    AutoregressiveDraftGraphIdentity,
)
from tinyvllm.engine.qwen3_draft_graph_scratch import (
    Qwen3DraftGraphScratchLease,
    Qwen3DraftGraphScratchOwner,
)


@dataclass(frozen=True)
class _InputRow:
    sequence_id: int


@dataclass(frozen=True)
class _Identity:
    logical_entry_id: int


@dataclass(frozen=True)
class _Lease:
    identities: tuple[_Identity, ...]
    physical_slot_ids: tuple[int, ...]


@dataclass
class _Transaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    staged_entry_identities: tuple[_Identity, ...]
    state: str = "reserved"


class _LiveAllocator:

    def __init__(self):
        self.read_leases = []
        self.completed = []

    def ensure_readable(self, identities):
        lease = _Lease(
            identities=identities,
            physical_slot_ids=tuple(
                identity.logical_entry_id + 100
                for identity in identities
            ),
        )
        self.read_leases.append(lease)
        return lease

    def record_read_complete(self, lease):
        self.completed.append(lease)


class _LiveCache:

    def __init__(self):
        self.entry_allocator = _LiveAllocator()
        self.committed = {
            7: (_Identity(1), _Identity(2)),
            9: (_Identity(3),),
            11: (_Identity(4), _Identity(5), _Identity(6)),
            13: (),
        }
        self.transactions = {
            sequence_id: f"live-{sequence_id}"
            for sequence_id in self.committed
        }

    def committed_entry_identities(self, sequence_id):
        return self.committed[sequence_id]


class _ScratchCache:

    def __init__(self):
        self.begin_calls = []
        self.abort_calls = []
        self.release_calls = []
        self.transactions = {}
        self.fail_begin_call = None

    def begin(
        self,
        sequence_id,
        sequence_epoch,
        staged_entry_count,
    ):
        self.begin_calls.append((
            sequence_id,
            sequence_epoch,
            staged_entry_count,
        ))
        if self.fail_begin_call == len(self.begin_calls):
            raise RuntimeError("scratch begin failed")
        transaction = _Transaction(
            transaction_id=(
                f"scratch-{len(self.begin_calls)}"
            ),
            sequence_id=sequence_id,
            sequence_epoch=sequence_epoch,
            staged_entry_identities=tuple(
                _Identity(
                    1000
                    + 10 * len(self.begin_calls)
                    + index
                )
                for index in range(staged_entry_count)
            ),
        )
        self.transactions[
            transaction.transaction_id
        ] = transaction
        return transaction

    def abort(self, transaction_id):
        transaction = self.transactions[transaction_id]
        assert transaction.state in ("reserved", "materialized")
        transaction.state = "aborted"
        self.abort_calls.append(transaction_id)

    def release_sequence(self, sequence_id, *, sequence_epoch):
        assert sequence_id > 0
        assert sequence_epoch == 0
        self.release_calls.append((sequence_id, sequence_epoch))


def _identity():
    return AutoregressiveDraftGraphIdentity(
        exact_q=4,
        exact_batch_size=4,
        tensor_parallel_size=4,
        tensor_parallel_rank=0,
        device_index=3,
        compute_dtype="torch.bfloat16",
        backend_identity="qwen3",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
        local_query_heads=8,
        local_kv_heads=2,
        kv_block_table_width=512,
        proposal_kv_capacity=4096,
        blockwise_offload=False,
    )


def _rows():
    return tuple(
        (
            index,
            _InputRow(sequence_id),
            context_token_count,
        )
        for index, sequence_id, context_token_count in (
            (0, 7, 4),
            (1, 9, 3),
            (2, 11, 5),
            (3, 13, 2),
        )
    )


def _owner():
    live_cache = _LiveCache()
    scratch_cache = _ScratchCache()
    owner = Qwen3DraftGraphScratchOwner(
        live_cache=live_cache,
        scratch_cache=scratch_cache,
    )
    return owner, live_cache, scratch_cache


def test_acquire_uses_private_transactions_and_source_read_leases():
    owner, live_cache, scratch_cache = _owner()
    live_before = dict(live_cache.transactions)

    lease = owner.acquire(_identity(), _rows())

    assert isinstance(lease, Qwen3DraftGraphScratchLease)
    assert tuple(
        row.indexed_row[1].sequence_id
        for row in lease.rows
    ) == (7, 9, 11, 13)
    assert tuple(
        row.source_committed_physical_slot_ids
        for row in lease.rows
    ) == ((101, 102), (103,), (104, 105, 106), ())
    assert tuple(
        len(row.transaction.staged_entry_identities)
        for row in lease.rows
    ) == (3, 3, 3, 3)
    assert all(
        row.transaction.transaction_id.startswith("scratch-")
        for row in lease.rows
    )
    assert live_cache.transactions == live_before
    assert len(scratch_cache.begin_calls) == 4
    assert live_cache.entry_allocator.completed == []


def test_rollback_releases_transactions_and_source_read_leases():
    owner, live_cache, scratch_cache = _owner()
    lease = owner.acquire(_identity(), _rows())
    lease.rows[1].transaction.state = "materialized"

    owner.rollback(lease)

    assert scratch_cache.abort_calls == [
        "scratch-4",
        "scratch-3",
        "scratch-2",
        "scratch-1",
    ]
    assert len(scratch_cache.release_calls) == 4
    assert live_cache.entry_allocator.completed == list(
        reversed(live_cache.entry_allocator.read_leases)
    )
    assert lease.rolled_back is True


def test_partial_acquire_failure_releases_every_acquired_resource():
    owner, live_cache, scratch_cache = _owner()
    scratch_cache.fail_begin_call = 3

    with pytest.raises(RuntimeError, match="scratch begin failed"):
        owner.acquire(_identity(), _rows())

    assert scratch_cache.abort_calls == [
        "scratch-2",
        "scratch-1",
    ]
    assert len(scratch_cache.release_calls) == 2
    assert live_cache.entry_allocator.completed == list(
        reversed(live_cache.entry_allocator.read_leases)
    )


def test_double_rollback_is_rejected():
    owner, _, _ = _owner()
    lease = owner.acquire(_identity(), _rows())
    owner.rollback(lease)

    with pytest.raises(RuntimeError, match="already rolled back"):
        owner.rollback(lease)


def test_batch_mismatch_and_offload_identity_fail_before_leasing():
    owner, live_cache, scratch_cache = _owner()

    with pytest.raises(ValueError, match="batch size"):
        owner.acquire(
            AutoregressiveDraftGraphIdentity(
                **{
                    **_identity().__dict__,
                    "exact_batch_size": 1,
                }
            ),
            _rows(),
        )
    with pytest.raises(ValueError, match="offload"):
        owner.acquire(
            AutoregressiveDraftGraphIdentity(
                **{
                    **_identity().__dict__,
                    "blockwise_offload": True,
                }
            ),
            _rows(),
        )

    assert live_cache.entry_allocator.read_leases == []
    assert scratch_cache.begin_calls == []
