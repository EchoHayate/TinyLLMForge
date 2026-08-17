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

from tinyvllm.engine.proposal_kv_cache import (
    ProposalKVCache,
    ProposalKVFinalizeTicket,
    ProposalKVSequenceState,
    ProposalKVTransaction,
)
from tinyvllm.engine.proposal_kv_allocator import (
    ProposalKVEntryIdentity,
)


class _Allocator:

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.free = list(range(capacity))
        self.generations = [0] * capacity
        self.reserve_calls = []
        self.commit_calls = []
        self.retire_calls = []

    def reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        self.reserve_calls.append(count)
        if count > len(self.free):
            raise RuntimeError("insufficient proposal KV entries")
        selected = tuple(self.free[:count])
        del self.free[:count]
        identities = []
        for logical_entry_id in selected:
            self.generations[logical_entry_id] += 1
            identities.append(
                ProposalKVEntryIdentity(
                    logical_entry_id,
                    self.generations[logical_entry_id],
                )
            )
        return tuple(identities)

    def commit_entries(self, identities) -> None:
        self.commit_calls.append(tuple(identities))

    def retire_entries(self, identities, *, writeback) -> None:
        assert writeback is False
        identities = tuple(identities)
        self.retire_calls.append(identities)
        self.free.extend(
            identity.logical_entry_id for identity in identities
        )
        self.free.sort()

    def authority_snapshot(self):
        return {
            "owned_entry_count": self.capacity - len(self.free),
        }


@pytest.mark.parametrize("q", (1, 2, 3, 5))
def test_finalize_matrix_commits_prefix_in_place_and_releases_suffix(q):
    for accepted in range(q + 1):
        allocator = _Allocator()
        cache = ProposalKVCache(allocator)
        staged_count = max(q - 1, 0)
        transaction = cache.begin(
            sequence_id=7,
            sequence_epoch=3,
            staged_entry_count=staged_count,
        )
        assert isinstance(transaction, ProposalKVTransaction)
        staged = transaction.staged_entry_identities
        cache.mark_materialized(transaction, staged_count)
        free_before_prepare = tuple(allocator.free)
        ticket = cache.prepare_finalize(
            transaction.transaction_id,
            accepted_proposal_tokens=accepted,
        )
        assert isinstance(ticket, ProposalKVFinalizeTicket)
        assert tuple(allocator.free) == free_before_prepare
        assert cache.committed_length(7) == 0
        commit_count = max(accepted - 1, 0)
        assert ticket.commit_entry_count == commit_count
        assert ticket.retire_entry_identities == staged[commit_count:]

        cache.commit_finalize(ticket.ticket_id)

        assert cache.committed_length(7) == commit_count
        assert cache.committed_entry_identities(7) == staged[:commit_count]
        assert allocator.commit_calls == (
            [staged[:commit_count]] if commit_count else []
        )
        assert allocator.retire_calls == (
            [staged[commit_count:]]
            if staged[commit_count:]
            else []
        )


def test_multiple_commits_append_without_replaying_existing_entries():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    first = cache.begin(7, 3, 3)
    cache.mark_materialized(first, 3)
    first_ticket = cache.prepare_finalize(
        first.transaction_id,
        accepted_proposal_tokens=3,
    )
    cache.commit_finalize(first_ticket.ticket_id)
    first_entries = cache.committed_entry_identities(7)

    second = cache.begin(7, 3, 2)
    cache.mark_materialized(second, 2)
    second_ticket = cache.prepare_finalize(
        second.transaction_id,
        accepted_proposal_tokens=3,
    )
    cache.commit_finalize(second_ticket.ticket_id)

    assert cache.committed_entry_identities(7) == (
        first_entries + second.staged_entry_identities
    )
    assert cache.committed_entry_identities(7)[
        :len(first_entries)
    ] == first_entries
    assert allocator.reserve_calls == [3, 2]
    assert allocator.retire_calls == [
        first.staged_entry_identities[2:]
    ]


def test_prepare_is_side_effect_free_and_rollback_releases_all_staging():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(7, 3, 4)
    cache.mark_materialized(transaction, 4)
    free_before_prepare = tuple(allocator.free)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=3,
    )
    assert tuple(allocator.free) == free_before_prepare
    assert cache.committed_entry_identities(7) == ()

    cache.rollback_finalize(ticket.ticket_id)

    assert cache.committed_entry_identities(7) == ()
    assert allocator.retire_calls == [
        transaction.staged_entry_identities
    ]
    assert all(
        identity.logical_entry_id in allocator.free
        for identity in transaction.staged_entry_identities
    )


def test_abort_releases_reserved_or_materialized_transaction():
    for materialized_count in (0, 2):
        allocator = _Allocator()
        cache = ProposalKVCache(allocator)
        transaction = cache.begin(7, 3, 2)
        if materialized_count:
            cache.mark_materialized(transaction, materialized_count)
        cache.abort(transaction.transaction_id)
        assert transaction.state == "aborted"
        assert allocator.retire_calls == [
            transaction.staged_entry_identities
        ]
        replacement = cache.begin(7, 3, 1)
        assert replacement.staged_entry_identities[
            0
        ].logical_entry_id in (
            identity.logical_entry_id
            for identity in transaction.staged_entry_identities
        )


def test_release_sequence_retires_only_committed_entries():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(7, 3, 3)
    cache.mark_materialized(transaction, 3)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=3,
    )
    cache.commit_finalize(ticket.ticket_id)
    committed = cache.committed_entry_identities(7)

    cache.release_sequence(7, sequence_epoch=3)

    assert allocator.retire_calls[-1] == committed
    assert cache.sequence_state(7) is None
    assert all(
        identity.logical_entry_id in allocator.free
        for identity in committed
    )


def test_runtime_sequence_zero_completes_proposal_kv_lifecycle():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(
        sequence_id=0,
        sequence_epoch=0,
        staged_entry_count=2,
    )
    cache.mark_materialized(transaction, 2)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=3,
    )
    cache.commit_finalize(ticket.ticket_id)

    assert cache.committed_entry_identities(
        0
    ) == transaction.staged_entry_identities

    cache.release_sequence(0, sequence_epoch=0)

    assert cache.sequence_state(0) is None
    assert all(
        identity.logical_entry_id in allocator.free
        for identity in transaction.staged_entry_identities
    )


def test_authority_snapshot_reports_history_and_zero_live_leaks():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(7, 3, 3)
    cache.mark_materialized(transaction, 3)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=2,
    )
    cache.commit_finalize(ticket.ticket_id)
    cache.release_sequence(7, sequence_epoch=3)

    snapshot = cache.authority_snapshot()

    assert snapshot["active_sequence_count"] == 0
    assert snapshot["active_transaction_count"] == 0
    assert snapshot["prepared_ticket_count"] == 0
    assert snapshot["owned_entry_count"] == 0
    assert snapshot["entry_allocator"] == {
        "owned_entry_count": 0,
    }
    assert snapshot["transactions"] == [{
        "transaction_id": "proposal-kv-transaction-1",
        "sequence_id": 7,
        "sequence_epoch": 3,
        "original_committed_length": 0,
        "staged_entry_count": 3,
        "materialized_entry_count": 3,
        "state": "committed",
    }]
    assert snapshot["tickets"] == [{
        "ticket_id": "proposal-kv-ticket-1",
        "transaction_id": "proposal-kv-transaction-1",
        "commit_entry_count": 1,
        "release_entry_count": 2,
        "state": "committed",
    }]


def test_public_records_expose_explicit_metadata_state():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(7, 3, 2)
    state = cache.sequence_state(7)
    assert isinstance(state, ProposalKVSequenceState)
    assert state.sequence_id == 7
    assert state.sequence_epoch == 3
    assert state.committed_entry_identities == ()
    assert transaction.original_committed_length == 0
    assert transaction.state == "reserved"
    assert transaction.materialized_entry_count == 0


def test_rejects_overlapping_transactions_and_stale_epochs():
    cache = ProposalKVCache(_Allocator())
    transaction = cache.begin(7, 3, 2)
    with pytest.raises(RuntimeError, match="active transaction"):
        cache.begin(7, 3, 1)
    cache.abort(transaction.transaction_id)
    with pytest.raises(RuntimeError, match="epoch"):
        cache.begin(7, 4, 1)
    with pytest.raises(RuntimeError, match="epoch"):
        cache.release_sequence(7, sequence_epoch=4)


def test_rejects_invalid_materialization_and_finalize_counts():
    cache = ProposalKVCache(_Allocator())
    transaction = cache.begin(7, 3, 2)
    with pytest.raises(ValueError, match="materialized"):
        cache.mark_materialized(transaction, 3)
    with pytest.raises(ValueError, match="materialized"):
        cache.mark_materialized(transaction, -1)
    with pytest.raises(RuntimeError, match="materialized"):
        cache.prepare_finalize(
            transaction.transaction_id,
            accepted_proposal_tokens=1,
        )
    cache.mark_materialized(transaction, 2)
    with pytest.raises(ValueError, match="accepted"):
        cache.prepare_finalize(
            transaction.transaction_id,
            accepted_proposal_tokens=4,
        )


def test_rejects_unknown_duplicate_and_cross_cache_transactions():
    first_cache = ProposalKVCache(_Allocator())
    second_cache = ProposalKVCache(_Allocator())
    transaction = first_cache.begin(7, 3, 1)
    with pytest.raises(ValueError, match="transaction"):
        second_cache.mark_materialized(transaction, 1)
    with pytest.raises(ValueError, match="unknown transaction"):
        first_cache.prepare_finalize(
            "missing",
            accepted_proposal_tokens=1,
        )
    first_cache.mark_materialized(transaction, 1)
    ticket = first_cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=2,
    )
    with pytest.raises(RuntimeError, match="prepared"):
        first_cache.prepare_finalize(
            transaction.transaction_id,
            accepted_proposal_tokens=2,
        )
    first_cache.commit_finalize(ticket.ticket_id)
    with pytest.raises(RuntimeError, match="already committed"):
        first_cache.commit_finalize(ticket.ticket_id)
    with pytest.raises(RuntimeError, match="committed"):
        first_cache.rollback_finalize(ticket.ticket_id)
    with pytest.raises(RuntimeError, match="transaction"):
        first_cache.abort(transaction.transaction_id)


def test_rejects_commit_after_rollback_and_ticket_reuse():
    cache = ProposalKVCache(_Allocator())
    transaction = cache.begin(7, 3, 2)
    cache.mark_materialized(transaction, 2)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=2,
    )
    cache.rollback_finalize(ticket.ticket_id)
    with pytest.raises(RuntimeError, match="rolled back"):
        cache.commit_finalize(ticket.ticket_id)
    with pytest.raises(RuntimeError, match="already rolled back"):
        cache.rollback_finalize(ticket.ticket_id)


def test_release_rejects_active_transaction_or_ticket():
    cache = ProposalKVCache(_Allocator())
    transaction = cache.begin(7, 3, 1)
    with pytest.raises(RuntimeError, match="active transaction"):
        cache.release_sequence(7, sequence_epoch=3)
    cache.mark_materialized(transaction, 1)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=2,
    )
    with pytest.raises(RuntimeError, match="active"):
        cache.release_sequence(7, sequence_epoch=3)
    cache.rollback_finalize(ticket.ticket_id)
    cache.release_sequence(7, sequence_epoch=3)


def test_allocator_contract_has_no_copy_or_forward_path():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(7, 3, 3)
    cache.mark_materialized(transaction, 3)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=3,
    )
    cache.commit_finalize(ticket.ticket_id)

    assert tuple(vars(allocator)) == (
        "capacity",
        "free",
        "generations",
        "reserve_calls",
        "commit_calls",
        "retire_calls",
    )
    assert not hasattr(allocator, "copy")
    assert not hasattr(allocator, "rematerialize")
    assert not hasattr(allocator, "forward")
