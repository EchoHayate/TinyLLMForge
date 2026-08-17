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
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.qwen35_mtp_graph import (
    Qwen35MTPGraphIdentity,
)
from tinyvllm.engine.qwen35_mtp_graph_scratch import (
    Qwen35MTPGraphScratchOwner,
)


@dataclass(frozen=True)
class _InputRow:
    sequence_id: int


@dataclass(frozen=True)
class _Bootstrap:
    sequence_epoch: int


@dataclass
class _Transaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    staged_slot_ids: tuple[int, ...]
    state: str = "reserved"


class _LiveCache:

    def __init__(self):
        self.physical_store = object()
        self.active_transaction_ids = {
            7: "live-7",
            9: "live-9",
            11: "live-11",
            13: "live-13",
        }
        self.committed = {
            7: (1, 2),
            9: (3,),
            11: (4, 5, 6),
            13: (),
        }

    def committed_slot_ids(self, sequence_id):
        assert sequence_id in self.active_transaction_ids
        return self.committed[sequence_id]


class _ScratchCache:

    def __init__(self, physical_store):
        self.physical_store = physical_store
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
            transaction_id=f"scratch-{len(self.begin_calls)}",
            sequence_id=sequence_id,
            sequence_epoch=sequence_epoch,
            staged_slot_ids=tuple(
                range(
                    100 * len(self.begin_calls),
                    100 * len(self.begin_calls)
                    + staged_entry_count,
                )
            ),
        )
        self.transactions[transaction.transaction_id] = transaction
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


def _identity(*, exact_q=3, exact_batch_size=4):
    return Qwen35MTPGraphIdentity(
        exact_q=exact_q,
        exact_batch_size=exact_batch_size,
        device_index=0,
        compute_dtype="torch.bfloat16",
        hidden_size=2048,
        mtp_layer_count=1,
        block_table_width=17,
    )


def _rows():
    return tuple(
        (_InputRow(sequence_id), _Bootstrap(sequence_epoch=5))
        for sequence_id in (7, 9, 11, 13)
    )


def _owner():
    live_cache = _LiveCache()
    scratch_cache = _ScratchCache(live_cache.physical_store)
    owner = Qwen35MTPGraphScratchOwner(
        live_cache=live_cache,
        scratch_cache=scratch_cache,
    )
    return owner, live_cache, scratch_cache


def test_scratch_acquire_uses_private_cache_and_preserves_row_order():
    owner, live_cache, scratch_cache = _owner()

    lease = owner.acquire(_identity(), _rows())

    assert owner.live_cache is live_cache
    assert owner.scratch_cache is scratch_cache
    assert owner.live_cache is not owner.scratch_cache
    assert tuple(
        row.input_row.sequence_id for row in lease.rows
    ) == (7, 9, 11, 13)
    assert tuple(
        row.source_committed_slot_ids for row in lease.rows
    ) == ((1, 2), (3,), (4, 5, 6), ())
    assert tuple(
        len(row.transaction.staged_slot_ids)
        for row in lease.rows
    ) == (2, 2, 2, 2)
    synthetic_ids = tuple(
        row.transaction.sequence_id for row in lease.rows
    )
    assert len(set(synthetic_ids)) == 4
    assert all(sequence_id > 0 for sequence_id in synthetic_ids)
    assert set(synthetic_ids).isdisjoint(
        live_cache.active_transaction_ids
    )


def test_active_live_transactions_do_not_block_scratch_acquire():
    owner, live_cache, scratch_cache = _owner()
    assert live_cache.active_transaction_ids

    lease = owner.acquire(_identity(), _rows())

    assert len(lease.rows) == 4
    assert len(scratch_cache.begin_calls) == 4


def test_partial_acquire_failure_aborts_and_releases_created_rows():
    owner, _, scratch_cache = _owner()
    scratch_cache.fail_begin_call = 3

    with pytest.raises(RuntimeError, match="scratch begin failed"):
        owner.acquire(_identity(), _rows())

    assert scratch_cache.abort_calls == [
        "scratch-2",
        "scratch-1",
    ]
    assert len(scratch_cache.release_calls) == 2
    assert {
        sequence_id
        for sequence_id, _ in scratch_cache.release_calls
    } == {
        scratch_cache.begin_calls[0][0],
        scratch_cache.begin_calls[1][0],
    }


def test_rollback_aborts_all_rows_and_releases_synthetic_sequences():
    owner, _, scratch_cache = _owner()
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
    assert lease.rolled_back is True


def test_double_rollback_is_rejected():
    owner, _, _ = _owner()
    lease = owner.acquire(_identity(), _rows())
    owner.rollback(lease)

    with pytest.raises(RuntimeError, match="already rolled back"):
        owner.rollback(lease)


def test_scratch_owner_rejects_identity_batch_mismatch():
    owner, _, scratch_cache = _owner()

    with pytest.raises(ValueError, match="batch size"):
        owner.acquire(
            _identity(exact_batch_size=1),
            _rows(),
        )

    assert scratch_cache.begin_calls == []

