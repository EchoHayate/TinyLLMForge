from __future__ import annotations

from collections import deque
import hashlib
import os
import sys
import types

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)
sys.modules.pop("tinyvllm.engine.sequence", None)
sys.modules.pop("tinyvllm.engine.block_manager", None)


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


xxhash_module = types.ModuleType("xxhash")
xxhash_module.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_module)

from tinyvllm.engine.block_manager import BlockManager
from tinyvllm.engine.sequence import Sequence
from tinyvllm.engine.spec_verify_exact_cuda_graph_cache import (
    SpecVerifyGraphReplayError,
)
from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
)
from tinyvllm.speculative.batch_runtime import (
    FirstTargetResult,
    NativeSpeculativeBatchError,
    prepare_native_speculative_batch,
)


@pytest.fixture(autouse=True)
def _restore_sequence_block_size():
    original = Sequence.block_size
    Sequence.block_size = 4
    try:
        yield
    finally:
        Sequence.block_size = original


def _allocated_sequence(
    token_ids: list[int],
    *,
    num_blocks: int = 8,
) -> tuple[BlockManager, Sequence]:
    manager = BlockManager(num_blocks=num_blocks, block_size=4)
    sequence = Sequence(token_ids)
    manager.allocate(sequence)
    return manager, sequence


def _allocator_snapshot(manager: BlockManager) -> dict[str, object]:
    return {
        "free": tuple(manager.free_block_ids),
        "used": frozenset(manager.used_block_ids),
        "hash_to_block_id": dict(manager.hash_to_block_id),
        "hash_to_block_ids": {
            block_hash: frozenset(block_ids)
            for block_hash, block_ids
            in manager.hash_to_block_ids.items()
        },
        "blocks": tuple(
            (
                block.ref_count,
                block.generation,
                block.hash,
                tuple(block.token_ids),
            )
            for block in manager.blocks
        ),
    }


def test_replay_failure_uses_existing_batch_rollback_owner():
    manager = BlockManager(num_blocks=8, block_size=4)
    sequences = (
        Sequence([1, 2, 3, 4]),
        Sequence([9, 10, 11, 12]),
    )
    for sequence in sequences:
        manager.allocate(sequence)
    sequence_snapshot = tuple(
        (
            tuple(sequence.token_ids),
            tuple(sequence.block_table),
        )
        for sequence in sequences
    )
    allocator_snapshot = _allocator_snapshot(manager)
    rolled_back_transactions = []
    original_rollback = (
        manager.rollback_speculative_kv_transaction
    )

    def record_rollback(transaction, sequence):
        rolled_back_transactions.append(transaction)
        return original_rollback(transaction, sequence)

    manager.rollback_speculative_kv_transaction = record_rollback

    class Adapter:
        capabilities = DraftCapabilities(
            source_type="fixture",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=2,
        )

        def propose_batch(self, contexts):
            return tuple(
                DraftProposal(
                    sequence_id=context.sequence_id,
                    token_ids=(
                        context.first_target_token,
                        context.first_target_token + 1,
                    ),
                    source_type="fixture",
                )
                for context in contexts
            )

    def run_first_targets(active_sequences):
        return tuple(
            FirstTargetResult(
                sequence_id=sequence.seq_id,
                target_token=sequence.last_token + 1,
            )
            for sequence in active_sequences
        )

    def run_tail_batch(items):
        assert len(items) == 2
        raise SpecVerifyGraphReplayError(
            "f" * 64,
            RuntimeError("replay failed"),
        )

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="tail_batch",
    ) as exc_info:
        prepare_native_speculative_batch(
            block_manager=manager,
            seqs=sequences,
            draft_adapter=Adapter(),
            eos_token=99,
            run_first_targets=run_first_targets,
            run_tail_batch=run_tail_batch,
        )

    error = exc_info.value
    assert isinstance(error.cause, SpecVerifyGraphReplayError)
    assert error.cause.identity_sha256 == "f" * 64
    assert set(error.rolled_back_sequence_ids) == {
        sequence.seq_id for sequence in sequences
    }
    assert len(rolled_back_transactions) == 2
    assert all(
        transaction.state == "rolled_back"
        for transaction in rolled_back_transactions
    )
    assert set(manager.free_block_ids) == set(
        allocator_snapshot["free"]
    )
    assert manager.used_block_ids == set(
        allocator_snapshot["used"]
    )
    assert (
        manager.hash_to_block_id
        == allocator_snapshot["hash_to_block_id"]
    )
    assert {
        block_hash: frozenset(block_ids)
        for block_hash, block_ids
        in manager.hash_to_block_ids.items()
    } == allocator_snapshot["hash_to_block_ids"]
    for transaction in rolled_back_transactions:
        for block_id in transaction.reserved_block_ids:
            block = manager.blocks[block_id]
            assert block.ref_count == 0
            assert block.generation == 1
            assert block.hash == -1
            assert block.token_ids == []
    assert tuple(
        (
            tuple(sequence.token_ids),
            tuple(sequence.block_table),
        )
        for sequence in sequences
    ) == sequence_snapshot


def test_begin_transaction_reserves_verifier_visible_capacity_without_sequence_mutation():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    original_tokens = list(sequence.token_ids)
    original_table = list(sequence.block_table)

    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )

    assert transaction.sequence_id == sequence.seq_id
    assert transaction.original_num_tokens == 4
    assert transaction.original_last_token == 4
    assert transaction.original_block_table == tuple(original_table)
    assert transaction.original_block_generations == tuple(
        manager.blocks[block_id].generation
        for block_id in original_table
    )
    assert transaction.proposed_token_count == 2
    assert transaction.materialized_token_count == 0
    assert transaction.state == "reserved"
    assert len(transaction.reserved_block_ids) == 1
    assert transaction.reserved_block_generations == tuple(
        manager.blocks[block_id].generation
        for block_id in transaction.reserved_block_ids
    )
    assert sequence.token_ids == original_tokens
    assert sequence.block_table == original_table


def test_authorize_speculative_kv_write_returns_canonical_frozen_snapshot():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )

    authorization = manager.authorize_speculative_kv_write(
        transaction,
        sequence,
    )

    original_identities = tuple(zip(
        transaction.original_block_table,
        transaction.original_block_generations,
    ))
    reserved_identities = tuple(zip(
        transaction.reserved_block_ids,
        transaction.reserved_block_generations,
    ))
    payload = (
        sequence.seq_id,
        len(sequence),
        transaction.proposed_token_count,
        0,
        "reserved",
        original_identities,
        reserved_identities,
    )
    assert authorization.sequence_id == sequence.seq_id
    assert authorization.original_num_tokens == len(sequence)
    assert (
        authorization.proposed_token_count
        == transaction.proposed_token_count
    )
    assert authorization.materialized_token_count == 0
    assert authorization.state == "reserved"
    assert authorization.original_block_identities == original_identities
    assert authorization.reserved_block_identities == reserved_identities
    assert authorization.authorization_sha256 == hashlib.sha256(
        repr(payload).encode("utf-8")
    ).hexdigest()
    with pytest.raises((AttributeError, TypeError)):
        authorization.state = "materialized"


@pytest.mark.parametrize(
    "state",
    ("materialized", "committed", "rolled_back"),
)
def test_authorize_speculative_kv_write_rejects_nonreserved_state(state):
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    transaction.state = state

    with pytest.raises(RuntimeError, match="authoriz"):
        manager.authorize_speculative_kv_write(
            transaction,
            sequence,
        )


def test_authorize_speculative_kv_write_rejects_materialized_count():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    transaction.materialized_token_count = 1

    with pytest.raises(RuntimeError, match="materialized"):
        manager.authorize_speculative_kv_write(
            transaction,
            sequence,
        )


def test_authorize_speculative_kv_write_rejects_owner_and_snapshot_drift():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    other = Sequence([9])
    with pytest.raises(ValueError, match="different sequence"):
        manager.authorize_speculative_kv_write(
            transaction,
            other,
        )

    sequence.append_token(5)
    with pytest.raises(RuntimeError, match="snapshot is stale"):
        manager.authorize_speculative_kv_write(
            transaction,
            sequence,
        )


def test_authorize_speculative_kv_write_rejects_stale_generations():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    original_id = transaction.original_block_table[0]
    manager.blocks[original_id].generation += 1
    with pytest.raises(
        RuntimeError,
        match="original block ownership is stale",
    ):
        manager.authorize_speculative_kv_write(
            transaction,
            sequence,
        )

    manager.blocks[original_id].generation -= 1
    reserved_id = transaction.reserved_block_ids[0]
    manager.blocks[reserved_id].generation += 1
    with pytest.raises(RuntimeError, match="block ownership is stale"):
        manager.authorize_speculative_kv_write(
            transaction,
            sequence,
        )


def test_authorize_speculative_kv_write_rejects_stale_reserved_ownership():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    reserved_id = transaction.reserved_block_ids[0]
    manager.blocks[reserved_id].hash = 123

    with pytest.raises(RuntimeError, match="block ownership is stale"):
        manager.authorize_speculative_kv_write(
            transaction,
            sequence,
        )


def test_block_identities_returns_allocator_generations_in_order():
    manager, sequence = _allocated_sequence([1, 2, 3, 4, 5])

    identities = manager.block_identities(
        tuple(sequence.block_table)
    )

    assert identities == tuple(
        (block_id, manager.blocks[block_id].generation)
        for block_id in sequence.block_table
    )


def test_block_identities_rejects_duplicate_ids():
    manager, sequence = _allocated_sequence([1, 2, 3])
    block_id = sequence.block_table[0]

    with pytest.raises(ValueError, match="unique"):
        manager.block_identities((block_id, block_id))


def test_speculative_transaction_rejects_stale_original_generation():
    manager, sequence = _allocated_sequence([1, 2, 3])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    original_block_id = sequence.block_table[0]
    manager.blocks[original_block_id].generation += 1

    with pytest.raises(
        RuntimeError,
        match="original block ownership is stale",
    ):
        manager.mark_speculative_kv_materialized(
            transaction,
            1,
        )

    assert transaction.state == "reserved"


def test_begin_transaction_uses_n_minus_one_appended_kv_positions():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])

    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=1,
    )

    assert transaction.reserved_block_ids == ()
    assert transaction.reserved_block_generations == ()


def test_begin_transaction_rejects_invalid_proposal_counts():
    manager, sequence = _allocated_sequence([1, 2, 3])

    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            manager.begin_speculative_kv_transaction(
                sequence,
                proposed_token_count=value,
            )


def test_begin_transaction_capacity_failure_is_atomic():
    manager, sequence = _allocated_sequence(
        [1, 2, 3, 4],
        num_blocks=1,
    )
    sequence_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
        sequence.num_tokens,
        sequence.last_token,
    )
    allocator_snapshot = _allocator_snapshot(manager)

    with pytest.raises(RuntimeError, match="insufficient KV blocks"):
        manager.begin_speculative_kv_transaction(
            sequence,
            proposed_token_count=2,
        )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
        sequence.num_tokens,
        sequence.last_token,
    ) == sequence_snapshot
    assert _allocator_snapshot(manager) == allocator_snapshot


def test_begin_transaction_rolls_back_partial_allocation(monkeypatch):
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    allocator_snapshot = _allocator_snapshot(manager)
    original_allocate = manager._allocate_block
    calls = 0

    def fail_second_allocation(block_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return original_allocate(block_id)

    monkeypatch.setattr(manager, "_allocate_block", fail_second_allocation)

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        manager.begin_speculative_kv_transaction(
            sequence,
            proposed_token_count=6,
        )

    assert _allocator_snapshot(manager) == allocator_snapshot


def test_mark_materialized_accepts_zero_and_maximum_counts():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    zero = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=1,
    )

    manager.mark_speculative_kv_materialized(zero, 0)

    assert zero.materialized_token_count == 0
    assert zero.state == "materialized"

    maximum = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=3,
    )

    manager.mark_speculative_kv_materialized(maximum, 2)

    assert maximum.materialized_token_count == 2
    assert maximum.state == "materialized"


def test_mark_materialized_rejects_invalid_counts_without_transition():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])

    for value in (-1, 2, True, 1.5):
        transaction = manager.begin_speculative_kv_transaction(
            sequence,
            proposed_token_count=2,
        )
        with pytest.raises(ValueError):
            manager.mark_speculative_kv_materialized(
                transaction,
                value,
            )
        assert transaction.state == "reserved"
        assert transaction.materialized_token_count == 0
        manager.release_reserved_blocks(
            list(transaction.reserved_block_ids)
        )


def test_mark_materialized_is_exactly_once():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    manager.mark_speculative_kv_materialized(transaction, 1)

    with pytest.raises(RuntimeError, match="not materializable"):
        manager.mark_speculative_kv_materialized(transaction, 1)


def test_mark_materialized_rejects_stale_reserved_generation():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    reserved_id = transaction.reserved_block_ids[0]
    block = manager.blocks[reserved_id]
    block.generation += 1

    with pytest.raises(RuntimeError, match="stale"):
        manager.mark_speculative_kv_materialized(transaction, 1)

    assert transaction.state == "reserved"


def test_begin_transaction_does_not_depend_on_free_list_container_identity():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    manager.free_block_ids = deque(manager.free_block_ids)

    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )

    assert len(transaction.reserved_block_ids) == 1


def _materialized_transaction(
    manager: BlockManager,
    sequence: Sequence,
    proposed_token_count: int,
    materialized_token_count: int,
):
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=proposed_token_count,
    )
    manager.mark_speculative_kv_materialized(
        transaction,
        materialized_token_count,
    )
    return transaction


def test_commit_requires_materialized_state():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    sequence_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
    )

    with pytest.raises(RuntimeError, match="not committable"):
        manager.commit_speculative_kv_transaction(
            transaction,
            sequence,
            [5],
        )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
    ) == sequence_snapshot
    assert transaction.state == "reserved"


def test_prepare_speculative_kv_commit_is_non_mutating():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=6,
        materialized_token_count=5,
    )
    sequence_snapshot = (
        tuple(sequence.token_ids),
        tuple(sequence.block_table),
        sequence.num_tokens,
        sequence.last_token,
    )
    allocator_snapshot = _allocator_snapshot(manager)

    plan = manager.prepare_speculative_kv_commit(
        transaction,
        sequence,
        (5, 6, 7, 8, 9, 10),
    )

    assert plan.sequence_id == sequence.seq_id
    assert plan.sequence is sequence
    assert plan.transaction is transaction
    assert plan.accepted_tokens == (5, 6, 7, 8, 9, 10)
    assert plan.committed_block_ids == (
        transaction.reserved_block_ids[:2]
    )
    assert plan.unused_block_ids == ()
    assert plan.materialized_end == 9
    assert tuple(
        (row.block_id, row.token_ids)
        for row in plan.publications
    ) == (
        (transaction.reserved_block_ids[0], (5, 6, 7, 8)),
    )
    assert (
        tuple(sequence.token_ids),
        tuple(sequence.block_table),
        sequence.num_tokens,
        sequence.last_token,
    ) == sequence_snapshot
    assert _allocator_snapshot(manager) == allocator_snapshot
    assert transaction.state == "materialized"


def test_commit_speculative_kv_batch_is_token_free():
    manager = BlockManager(num_blocks=8, block_size=4)
    first = Sequence([1, 2, 3, 4])
    second = Sequence([9, 10, 11, 12])
    manager.allocate(first)
    manager.allocate(second)
    first_transaction = _materialized_transaction(
        manager,
        first,
        proposed_token_count=2,
        materialized_token_count=1,
    )
    second_transaction = _materialized_transaction(
        manager,
        second,
        proposed_token_count=2,
        materialized_token_count=1,
    )
    plans = (
        manager.prepare_speculative_kv_commit(
            first_transaction,
            first,
            (5, 6),
        ),
        manager.prepare_speculative_kv_commit(
            second_transaction,
            second,
            (13, 14),
        ),
    )
    token_snapshots = (
        tuple(first.token_ids),
        tuple(second.token_ids),
    )

    manager.commit_speculative_kv_commit_batch(plans)

    assert tuple(first.token_ids) == token_snapshots[0]
    assert tuple(second.token_ids) == token_snapshots[1]
    assert first.block_table[-1] == (
        first_transaction.reserved_block_ids[0]
    )
    assert second.block_table[-1] == (
        second_transaction.reserved_block_ids[0]
    )
    assert first_transaction.state == "committed"
    assert second_transaction.state == "committed"


def test_commit_speculative_kv_batch_failure_restores_every_plan(
    monkeypatch,
):
    manager = BlockManager(num_blocks=8, block_size=4)
    sequences = (
        Sequence([1, 2, 3, 4]),
        Sequence([9, 10, 11, 12]),
    )
    for sequence in sequences:
        manager.allocate(sequence)
    transactions = tuple(
        _materialized_transaction(
            manager,
            sequence,
            proposed_token_count=2,
            materialized_token_count=1,
        )
        for sequence in sequences
    )
    plans = tuple(
        manager.prepare_speculative_kv_commit(
            transaction,
            sequence,
            accepted_tokens,
        )
        for transaction, sequence, accepted_tokens in zip(
            transactions,
            sequences,
            ((5, 6), (13, 14)),
        )
    )
    sequence_snapshots = tuple(
        (
            tuple(sequence.token_ids),
            tuple(sequence.block_table),
            sequence.num_tokens,
            sequence.last_token,
        )
        for sequence in sequences
    )
    allocator_snapshot = _allocator_snapshot(manager)
    original_apply = manager._apply_speculative_kv_commit_plan
    calls = 0

    def fail_second_plan(plan):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected batch commit failure")
        return original_apply(plan)

    monkeypatch.setattr(
        manager,
        "_apply_speculative_kv_commit_plan",
        fail_second_plan,
    )

    with pytest.raises(
        RuntimeError,
        match="injected batch commit failure",
    ):
        manager.commit_speculative_kv_commit_batch(plans)

    assert tuple(
        (
            tuple(sequence.token_ids),
            tuple(sequence.block_table),
            sequence.num_tokens,
            sequence.last_token,
        )
        for sequence in sequences
    ) == sequence_snapshots
    assert _allocator_snapshot(manager) == allocator_snapshot
    assert tuple(
        transaction.state for transaction in transactions
    ) == ("materialized", "materialized")


@pytest.mark.parametrize(
    (
        "accepted_tokens",
        "expected_block_count",
        "expected_reserved_used",
    ),
    (
        ([], 1, 0),
        ([5], 1, 0),
        ([5, 6, 7], 2, 1),
        ([5, 6, 7, 8, 9, 10], 3, 2),
    ),
)
def test_commit_zero_one_partial_and_full_acceptance(
    accepted_tokens,
    expected_block_count,
    expected_reserved_used,
):
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    original_table = list(sequence.block_table)
    transaction = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=6,
        materialized_token_count=5,
    )
    reserved_ids = transaction.reserved_block_ids

    manager.commit_speculative_kv_transaction(
        transaction,
        sequence,
        list(accepted_tokens),
    )

    assert sequence.token_ids == [1, 2, 3, 4] + accepted_tokens
    assert len(sequence.block_table) == expected_block_count
    assert sequence.block_table[:1] == original_table
    assert sequence.block_table[1:] == list(
        reserved_ids[:expected_reserved_used]
    )
    assert transaction.state == "committed"
    for block_id in reserved_ids[:expected_reserved_used]:
        assert block_id in manager.used_block_ids
        assert manager.blocks[block_id].ref_count == 1
    for block_id in reserved_ids[expected_reserved_used:]:
        assert block_id not in manager.used_block_ids
        assert manager.blocks[block_id].ref_count == 0


def test_commit_publishes_only_full_materialized_blocks():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=6,
        materialized_token_count=5,
    )

    manager.commit_speculative_kv_transaction(
        transaction,
        sequence,
        [5, 6, 7, 8, 9, 10],
    )

    second_block = manager.blocks[sequence.block_table[1]]
    third_block = manager.blocks[sequence.block_table[2]]
    assert second_block.hash != -1
    assert second_block.token_ids == [5, 6, 7, 8]
    assert third_block.hash == -1
    assert third_block.token_ids == []


def test_commit_rejects_wrong_sequence_and_sequence_drift_before_mutation():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=2,
        materialized_token_count=1,
    )
    other = Sequence([9])
    other_snapshot = (list(other.token_ids), list(other.block_table))

    with pytest.raises(ValueError, match="different sequence"):
        manager.commit_speculative_kv_transaction(
            transaction,
            other,
            [5],
        )

    assert (list(other.token_ids), list(other.block_table)) == other_snapshot
    sequence.append_token(99)
    drifted_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
    )

    with pytest.raises(RuntimeError, match="sequence snapshot is stale"):
        manager.commit_speculative_kv_transaction(
            transaction,
            sequence,
            [5],
        )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
    ) == drifted_snapshot
    assert transaction.state == "materialized"


def test_commit_rejects_stale_or_published_reserved_block_before_mutation():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=2,
        materialized_token_count=1,
    )
    reserved_id = transaction.reserved_block_ids[0]
    sequence_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
    )
    manager.blocks[reserved_id].generation += 1

    with pytest.raises(RuntimeError, match="stale"):
        manager.commit_speculative_kv_transaction(
            transaction,
            sequence,
            [5, 6],
        )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
    ) == sequence_snapshot
    manager.blocks[reserved_id].generation -= 1
    manager.blocks[reserved_id].hash = 123

    with pytest.raises(RuntimeError, match="stale"):
        manager.commit_speculative_kv_transaction(
            transaction,
            sequence,
            [5, 6],
        )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
    ) == sequence_snapshot
    assert transaction.state == "materialized"


def test_commit_rejects_reserved_block_overlapping_original_ownership():
    manager, sequence = _allocated_sequence([1, 2, 3])
    transaction = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=3,
        materialized_token_count=2,
    )
    original_block_id = sequence.block_table[0]
    transaction.reserved_block_ids = (original_block_id,)
    transaction.reserved_block_generations = (
        manager.blocks[original_block_id].generation,
    )
    sequence_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
    )

    with pytest.raises(ValueError, match="overlap"):
        manager.commit_speculative_kv_transaction(
            transaction,
            sequence,
            [4, 5, 6],
        )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
    ) == sequence_snapshot
    assert transaction.state == "materialized"


def test_commit_rejects_invalid_acceptance_before_mutation():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    sequence_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
    )

    cases = (
        (3, 2, [5, 6, 7, 8], "exceeds proposal"),
        (3, 1, [5, 6, 7], "exceeds materialized KV"),
        (2, 1, [True], "accepted token"),
        (2, 1, [5.5], "accepted token"),
    )
    for proposed, materialized, accepted, message in cases:
        transaction = _materialized_transaction(
            manager,
            sequence,
            proposed_token_count=proposed,
            materialized_token_count=materialized,
        )
        with pytest.raises((ValueError, RuntimeError), match=message):
            manager.commit_speculative_kv_transaction(
                transaction,
                sequence,
                accepted,
            )
        assert (
            list(sequence.token_ids),
            list(sequence.block_table),
        ) == sequence_snapshot
        manager.rollback_speculative_kv_transaction(
            transaction,
            sequence,
        )


@pytest.mark.parametrize("materialize", (False, True))
def test_rollback_releases_reserved_blocks_without_sequence_mutation(
    materialize,
):
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    sequence_snapshot = (
        list(sequence.token_ids),
        list(sequence.block_table),
    )
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    reserved_id = transaction.reserved_block_ids[0]
    if materialize:
        manager.mark_speculative_kv_materialized(transaction, 1)

    manager.rollback_speculative_kv_transaction(
        transaction,
        sequence,
    )

    assert (
        list(sequence.token_ids),
        list(sequence.block_table),
    ) == sequence_snapshot
    assert reserved_id not in manager.used_block_ids
    assert manager.blocks[reserved_id].ref_count == 0
    assert transaction.state == "rolled_back"


def test_rollback_allows_same_owner_sequence_drift_but_rejects_wrong_owner():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    other = Sequence([9])

    with pytest.raises(ValueError, match="different sequence"):
        manager.rollback_speculative_kv_transaction(
            transaction,
            other,
        )

    sequence.append_token(99)
    manager.rollback_speculative_kv_transaction(
        transaction,
        sequence,
    )

    assert sequence.token_ids == [1, 2, 3, 4, 99]
    assert transaction.state == "rolled_back"


def test_commit_and_rollback_are_exactly_once():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    committed = _materialized_transaction(
        manager,
        sequence,
        proposed_token_count=1,
        materialized_token_count=0,
    )
    manager.commit_speculative_kv_transaction(
        committed,
        sequence,
        [5],
    )

    with pytest.raises(RuntimeError, match="not committable"):
        manager.commit_speculative_kv_transaction(
            committed,
            sequence,
            [],
        )
    with pytest.raises(RuntimeError, match="not rollbackable"):
        manager.rollback_speculative_kv_transaction(
            committed,
            sequence,
        )

    rolled_back = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    manager.rollback_speculative_kv_transaction(
        rolled_back,
        sequence,
    )

    with pytest.raises(RuntimeError, match="not rollbackable"):
        manager.rollback_speculative_kv_transaction(
            rolled_back,
            sequence,
        )
    with pytest.raises(RuntimeError, match="not committable"):
        manager.commit_speculative_kv_transaction(
            rolled_back,
            sequence,
            [],
        )


def test_rollback_rejects_stale_generation_without_releasing_new_owner():
    manager, sequence = _allocated_sequence([1, 2, 3, 4])
    transaction = manager.begin_speculative_kv_transaction(
        sequence,
        proposed_token_count=2,
    )
    reserved_id = transaction.reserved_block_ids[0]
    manager.blocks[reserved_id].generation += 1

    with pytest.raises(RuntimeError, match="stale"):
        manager.rollback_speculative_kv_transaction(
            transaction,
            sequence,
        )

    assert reserved_id in manager.used_block_ids
    assert transaction.state == "reserved"
