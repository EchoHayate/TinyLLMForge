from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.proposal_kv_allocator import (
    ProposalKVEntryIdentity,
)
from tinyvllm.engine.proposal_kv_lifecycle import (
    ProposalKVLifecycleCoordinator,
    ProposalKVRegistration,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ProposalFinalizeRow,
)
from tinyvllm.speculative.adapter import DraftProposal


class _Allocator:

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.free = list(range(capacity))
        self.generations = [0] * capacity
        self.commit_calls = []
        self.retire_calls = []

    def reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
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


def _materialized(
    cache: ProposalKVCache,
    sequence_id: int,
    *,
    sequence_epoch: int = 0,
    staged_entry_count: int = 1,
):
    transaction = cache.begin(
        sequence_id,
        sequence_epoch,
        staged_entry_count,
    )
    cache.mark_materialized(transaction, staged_entry_count)
    return transaction


def _proposal(
    sequence_id: int,
    transaction_id: str | None,
    *,
    token_ids: tuple[int, ...] = (11, 12),
) -> DraftProposal:
    return DraftProposal(
        sequence_id=sequence_id,
        token_ids=token_ids,
        source_type="learned",
        metadata={
            "exact_q": len(token_ids),
            "staged_entry_count": max(len(token_ids) - 1, 0),
        },
        proposal_transaction_id=transaction_id,
    )


def test_register_batch_preserves_order_and_tracks_materialized_transactions():
    cache = ProposalKVCache(_Allocator())
    transaction = _materialized(cache, 1)
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    empty = _proposal(2, None, token_ids=())
    proposed = _proposal(1, transaction.transaction_id)

    rows = coordinator.register_batch((
        ProposalKVRegistration(2, 0, empty),
        ProposalKVRegistration(1, 0, proposed),
    ))

    assert rows == (empty, proposed)
    assert coordinator.active_transaction_count == 1
    assert coordinator.authority_snapshot()["transactions"] == [{
        "sequence_id": 1,
        "sequence_epoch": 0,
        "transaction_id": transaction.transaction_id,
        "token_ids": [11, 12],
        "staged_entry_count": 1,
        "staged_entry_identities": [{
            "logical_entry_id": (
                transaction.staged_entry_identities[
                    0
                ].logical_entry_id
            ),
            "generation": (
                transaction.staged_entry_identities[0].generation
            ),
        }],
        "accepted_proposal_tokens": None,
        "rejected_proposal_tokens": None,
        "finalize_ticket_id": None,
        "state": "active",
    }]


def test_register_batch_rejects_reserved_transaction_and_aborts_it():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(1, 0, 1)
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )

    with pytest.raises(RuntimeError, match="materialized"):
        coordinator.register_batch((
            ProposalKVRegistration(
                1,
                0,
                _proposal(1, transaction.transaction_id),
            ),
        ))

    assert transaction.state == "aborted"
    assert allocator.retire_calls == [
        transaction.staged_entry_identities
    ]
    assert coordinator.active_transaction_count == 0


def test_failed_registration_aborts_only_new_unregistered_transactions():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    existing = _materialized(cache, 1)
    coordinator.register_batch((
        ProposalKVRegistration(
            1,
            0,
            _proposal(1, existing.transaction_id),
        ),
    ))
    first_new = _materialized(cache, 2)
    stale = _materialized(cache, 3, sequence_epoch=4)

    with pytest.raises(RuntimeError, match="epoch"):
        coordinator.register_batch((
            ProposalKVRegistration(
                2,
                0,
                _proposal(2, first_new.transaction_id),
            ),
            ProposalKVRegistration(
                3,
                5,
                _proposal(3, stale.transaction_id),
            ),
        ))

    assert existing.state == "materialized"
    assert first_new.state == "aborted"
    assert stale.state == "aborted"
    assert coordinator.active_transaction_count == 1
    assert allocator.retire_calls[-2:] == [
        stale.staged_entry_identities,
        first_new.staged_entry_identities,
    ]


def test_early_registration_validation_aborts_the_whole_new_batch():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    first = _materialized(cache, 1)
    mismatched = _materialized(cache, 2)

    with pytest.raises(ValueError, match="sequence"):
        coordinator.register_batch((
            ProposalKVRegistration(
                1,
                0,
                _proposal(1, first.transaction_id),
            ),
            ProposalKVRegistration(
                2,
                0,
                _proposal(3, mismatched.transaction_id),
            ),
        ))

    assert first.state == "aborted"
    assert mismatched.state == "aborted"
    assert allocator.retire_calls == [
        mismatched.staged_entry_identities,
        first.staged_entry_identities,
    ]
    assert coordinator.active_transaction_count == 0


@pytest.mark.parametrize(
    "rows,match",
    (
        (
            (
                ProposalKVRegistration(
                    1,
                    0,
                    _proposal(1, "proposal-kv-transaction-1"),
                ),
                ProposalKVRegistration(
                    1,
                    0,
                    _proposal(1, "proposal-kv-transaction-2"),
                ),
            ),
            "sequence",
        ),
        (
            (
                ProposalKVRegistration(
                    1,
                    0,
                    _proposal(1, "proposal-kv-transaction-1"),
                ),
                ProposalKVRegistration(
                    2,
                    0,
                    _proposal(2, "proposal-kv-transaction-1"),
                ),
            ),
            "transaction",
        ),
    ),
)
def test_register_batch_rejects_duplicate_identities(rows, match):
    coordinator = ProposalKVLifecycleCoordinator(
        ProposalKVCache(_Allocator()),
        ticket_namespace="fixture",
    )

    with pytest.raises(ValueError, match=match):
        coordinator.register_batch(rows)


def _registered(
    coordinator: ProposalKVLifecycleCoordinator,
    cache: ProposalKVCache,
    sequence_id: int,
    *,
    sequence_epoch: int = 0,
    staged_entry_count: int = 1,
):
    transaction = _materialized(
        cache,
        sequence_id,
        sequence_epoch=sequence_epoch,
        staged_entry_count=staged_entry_count,
    )
    proposal = _proposal(
        sequence_id,
        transaction.transaction_id,
        token_ids=tuple(
            range(100, 100 + staged_entry_count + 1)
        ),
    )
    coordinator.register_batch((
        ProposalKVRegistration(
            sequence_id,
            sequence_epoch,
            proposal,
        ),
    ))
    return transaction, proposal


def test_finalize_commit_preserves_accepted_prefix_and_consumes_ticket():
    cache = ProposalKVCache(_Allocator())
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    transaction, _ = _registered(
        coordinator,
        cache,
        1,
        staged_entry_count=2,
    )

    ticket_id = coordinator.prepare_finalize_batch((
        ProposalFinalizeRow(
            sequence_id=1,
            proposal_transaction_id=transaction.transaction_id,
            accepted_proposal_tokens=2,
        ),
    ))

    assert ticket_id == "fixture-finalize-1"
    assert coordinator.prepared_ticket_count == 1
    coordinator.commit_finalize_batch(ticket_id)
    assert cache.committed_entry_identities(1) == (
        transaction.staged_entry_identities[:1]
    )
    assert coordinator.active_transaction_count == 0
    assert coordinator.prepared_ticket_count == 0
    with pytest.raises(ValueError, match="not active"):
        coordinator.commit_finalize_batch(ticket_id)
    assert coordinator.authority_snapshot()["transactions"][0][
        "state"
    ] == "committed"


def test_finalize_rollback_retires_all_staged_entries():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    transaction, _ = _registered(
        coordinator,
        cache,
        1,
        staged_entry_count=2,
    )
    ticket_id = coordinator.prepare_finalize_batch((
        ProposalFinalizeRow(
            sequence_id=1,
            proposal_transaction_id=transaction.transaction_id,
            accepted_proposal_tokens=2,
        ),
    ))

    coordinator.rollback_finalize_batch(ticket_id)

    assert transaction.state == "rolled_back"
    assert allocator.retire_calls[
        -1
    ] == transaction.staged_entry_identities
    assert coordinator.active_transaction_count == 0
    assert coordinator.authority_snapshot()["transactions"][0][
        "state"
    ] == "rolled_back"


def test_partial_finalize_prepare_failure_rolls_back_prior_tickets(
    monkeypatch,
):
    cache = ProposalKVCache(_Allocator())
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    first, _ = _registered(coordinator, cache, 1)
    second, _ = _registered(coordinator, cache, 2)
    original_prepare = cache.prepare_finalize
    original_rollback = cache.rollback_finalize
    prepared = []
    rolled_back = []

    def prepare(transaction_id, *, accepted_proposal_tokens):
        if transaction_id == second.transaction_id:
            raise RuntimeError("prepare failed")
        ticket = original_prepare(
            transaction_id,
            accepted_proposal_tokens=accepted_proposal_tokens,
        )
        prepared.append(ticket.ticket_id)
        return ticket

    def rollback(ticket_id):
        rolled_back.append(ticket_id)
        return original_rollback(ticket_id)

    monkeypatch.setattr(cache, "prepare_finalize", prepare)
    monkeypatch.setattr(cache, "rollback_finalize", rollback)

    with pytest.raises(RuntimeError, match="prepare failed"):
        coordinator.prepare_finalize_batch((
            ProposalFinalizeRow(
                1,
                first.transaction_id,
                1,
            ),
            ProposalFinalizeRow(
                2,
                second.transaction_id,
                1,
            ),
        ))

    assert rolled_back == list(reversed(prepared))
    assert coordinator.prepared_ticket_count == 0
    assert first.state == "rolled_back"
    assert second.state == "aborted"
    assert coordinator.active_transaction_count == 0
    states = {
        row["transaction_id"]: row["state"]
        for row in coordinator.authority_snapshot()["transactions"]
    }
    assert states == {
        first.transaction_id: "rolled_back",
        second.transaction_id: "aborted",
    }


def test_release_rejects_active_transaction_and_stale_epoch():
    cache = ProposalKVCache(_Allocator())
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    transaction, _ = _registered(
        coordinator,
        cache,
        1,
        sequence_epoch=3,
    )

    with pytest.raises(RuntimeError, match="active"):
        coordinator.assert_sequence_releasable(1, 3)
    ticket_id = coordinator.prepare_finalize_batch((
        ProposalFinalizeRow(
            1,
            transaction.transaction_id,
            2,
        ),
    ))
    coordinator.commit_finalize_batch(ticket_id)
    with pytest.raises(RuntimeError, match="epoch"):
        coordinator.release_sequence(1, 4)


def test_release_sequence_retires_committed_entries_and_records_evidence():
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    transaction, _ = _registered(
        coordinator,
        cache,
        1,
        sequence_epoch=3,
        staged_entry_count=2,
    )
    ticket_id = coordinator.prepare_finalize_batch((
        ProposalFinalizeRow(
            1,
            transaction.transaction_id,
            3,
        ),
    ))
    coordinator.commit_finalize_batch(ticket_id)
    committed_entries = cache.committed_entry_identities(1)

    coordinator.release_sequence(1, 3)

    assert cache.sequence_state(1) is None
    assert allocator.retire_calls[-1] == committed_entries
    assert coordinator.authority_snapshot()["release_rows"] == [{
        "sequence_id": 1,
        "sequence_epoch": 3,
    }]
