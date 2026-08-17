from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(ROOT, "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [
    os.path.join(ROOT, "tinyvllm", "engine")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)

from tinyvllm.engine.speculative_residency import (
    KVBlockIdentityRow,
    SpeculativeResidencyParticipant,
    SpeculativeResidencyPrecommitRow,
    SpeculativeResidencyPrepareRow,
    build_kv_block_identity_rows,
)


class _BlockManager:
    def __init__(self):
        self.identities = {
            (1, 3): ((1, 4), (3, 2)),
            (5,): ((5, 8),),
        }
        self.calls = []

    def block_identities(self, block_ids):
        self.calls.append(block_ids)
        return self.identities[block_ids]


def test_build_identity_rows_uses_allocator_generation_order():
    block_manager = _BlockManager()
    sequence_a = SimpleNamespace(
        seq_id=7,
        block_table=[1, 3],
    )
    sequence_b = SimpleNamespace(
        seq_id=9,
        block_table=[5],
    )

    rows = build_kv_block_identity_rows(
        block_manager,
        (sequence_a, sequence_b),
    )

    assert rows == (
        KVBlockIdentityRow(7, ((1, 4), (3, 2))),
        KVBlockIdentityRow(9, ((5, 8),)),
    )
    assert block_manager.calls == [(1, 3), (5,)]


def test_build_identity_rows_rejects_duplicate_sequence_ids():
    block_manager = _BlockManager()
    sequences = (
        SimpleNamespace(seq_id=7, block_table=[1, 3]),
        SimpleNamespace(seq_id=7, block_table=[5]),
    )

    with pytest.raises(ValueError, match="unique"):
        build_kv_block_identity_rows(
            block_manager,
            sequences,
        )


class _ResidencyManager:
    def __init__(self):
        self.bound = {1: 1}
        self.logical_to_slot = {1: 0}
        self.cpu_valid = {1}
        self.next_slot = 1
        self.dirty = set()
        self.discarded = []
        self.ensure_calls = []
        self.writeback_calls = []
        self.writeback_on_evict = False
        self.fail_history = False
        self.stats = {
            "speculative_residency_prepares": 0,
            "speculative_residency_precommits": 0,
            "speculative_residency_seals": 0,
            "speculative_residency_rollbacks": 0,
            "speculative_residency_committed_blocks": 0,
            "speculative_residency_rejected_blocks": 0,
            "speculative_residency_rejected_d2h_copies": 0,
        }

    def bind_logical_block_identity(self, block_id, generation):
        current = self.bound.get(block_id)
        if current is not None and current != generation:
            raise RuntimeError("generation mismatch")
        self.bound[block_id] = generation

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
        wait=False,
    ):
        self.ensure_calls.append((
            tuple(logical_blocks),
            require_valid,
            frozenset(protected_logical_blocks or ()),
            wait,
        ))
        if require_valid and self.fail_history:
            raise RuntimeError("unreadable historical block")
        mapping = {}
        for block_id in logical_blocks:
            if (
                require_valid
                and block_id not in self.cpu_valid
                and block_id not in self.logical_to_slot
            ):
                raise RuntimeError("unreadable historical block")
            if block_id not in self.logical_to_slot:
                self.logical_to_slot[block_id] = self.next_slot
                self.next_slot += 1
            mapping[block_id] = self.logical_to_slot[block_id]
        return mapping

    def discard_resident_blocks(
        self,
        block_identities,
        *,
        allow_dirty,
    ):
        for block_id, _ in block_identities:
            if not allow_dirty and block_id in self.dirty:
                raise RuntimeError("dirty discard")
        for block_id, _ in block_identities:
            self.logical_to_slot.pop(block_id)
            self.dirty.discard(block_id)
        self.discarded.append(tuple(block_identities))
        return tuple(block_identities)

    def mark_dirty(self, logical_blocks):
        self.dirty.update(logical_blocks)

    def writeback_dirty(self, logical_blocks=None):
        blocks = tuple(logical_blocks or self.dirty)
        self.writeback_calls.append(blocks)
        self.dirty.difference_update(blocks)


class _CapacityResidencyManager(_ResidencyManager):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.bound = {99: 1}
        self.logical_to_slot = {99: 0}
        self.cpu_valid = {10, 20, 99}
        self.next_slot = 1

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
        wait=False,
    ):
        protected = set(protected_logical_blocks or ())
        self.ensure_calls.append((
            tuple(logical_blocks),
            require_valid,
            frozenset(protected),
            wait,
        ))
        mapping = {}
        for block_id in logical_blocks:
            if (
                require_valid
                and block_id not in self.cpu_valid
                and block_id not in self.logical_to_slot
            ):
                raise RuntimeError("unreadable historical block")
            if block_id not in self.logical_to_slot:
                if len(self.logical_to_slot) >= self.capacity:
                    victim = min(
                        resident
                        for resident in self.logical_to_slot
                        if resident not in protected
                    )
                    self.logical_to_slot.pop(victim)
                self.logical_to_slot[block_id] = self.next_slot
                self.next_slot += 1
            mapping[block_id] = self.logical_to_slot[block_id]
        return mapping


def _prepare_row():
    return SpeculativeResidencyPrepareRow(
        sequence_id=7,
        original_block_identities=((1, 1),),
        reserved_block_identities=((2, 1),),
        proxy_block_table=(1, 2),
        logical_slots=(3, 4),
    )


def _second_prepare_row():
    return SpeculativeResidencyPrepareRow(
        sequence_id=9,
        original_block_identities=((1, 1),),
        reserved_block_identities=((3, 1),),
        proxy_block_table=(1, 3),
        logical_slots=(3, 4),
    )


def test_prepare_batch_streaming_stages_only_materialized_write_blocks():
    manager = _ResidencyManager()
    manager.cpu_valid.update({4, 5})
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    row = SpeculativeResidencyPrepareRow(
        sequence_id=7,
        original_block_identities=(
            (1, 1),
            (4, 1),
            (5, 1),
        ),
        reserved_block_identities=((2, 1),),
        proxy_block_table=(1, 4, 5, 2),
        logical_slots=(11, 12),
    )

    participant.prepare_batch(
        41,
        (row,),
        stage_all_original_blocks=False,
    )

    assert manager.bound == {
        1: 1,
        2: 1,
        4: 1,
        5: 1,
    }
    assert manager.ensure_calls == [
        (
            (5,),
            True,
            frozenset({2, 5}),
            True,
        ),
        (
            (2,),
            False,
            frozenset({2, 5}),
            True,
        ),
    ]
    assert 4 not in manager.logical_to_slot
    assert set(manager.logical_to_slot) == {1, 2, 5}


def test_prepare_batch_protects_materialized_blocks_across_all_rows():
    manager = _CapacityResidencyManager(capacity=4)
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    rows = (
        SpeculativeResidencyPrepareRow(
            sequence_id=7,
            original_block_identities=((10, 1),),
            reserved_block_identities=((11, 1),),
            proxy_block_table=(10, 11),
            logical_slots=(3, 4),
        ),
        SpeculativeResidencyPrepareRow(
            sequence_id=9,
            original_block_identities=((20, 1),),
            reserved_block_identities=((21, 1),),
            proxy_block_table=(20, 21),
            logical_slots=(3, 4),
        ),
    )

    participant.prepare_batch(
        41,
        rows,
        stage_all_original_blocks=False,
    )

    assert set(manager.logical_to_slot) == {10, 11, 20, 21}
    assert all(
        protected == frozenset({10, 11, 20, 21})
        for _, _, protected, _ in manager.ensure_calls
    )


def test_ensure_materialized_for_restores_evicted_ticket_blocks():
    manager = _CapacityResidencyManager(capacity=4)
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    rows = (
        SpeculativeResidencyPrepareRow(
            sequence_id=7,
            original_block_identities=((10, 1),),
            reserved_block_identities=((11, 1),),
            proxy_block_table=(10, 11),
            logical_slots=(3, 4),
        ),
        SpeculativeResidencyPrepareRow(
            sequence_id=9,
            original_block_identities=((20, 1),),
            reserved_block_identities=((21, 1),),
            proxy_block_table=(20, 21),
            logical_slots=(3, 4),
        ),
    )
    participant.prepare_batch(
        41,
        rows,
        stage_all_original_blocks=False,
    )
    evicted_slot = manager.logical_to_slot.pop(21)
    manager.logical_to_slot[30] = evicted_slot

    participant.ensure_materialized_for(41, (7, 9))

    assert set(manager.logical_to_slot) == {10, 11, 20, 21}
    assert manager.ensure_calls[-2:] == [
        (
            (10, 20),
            True,
            frozenset({10, 11, 20, 21}),
            True,
        ),
        (
            (11, 21),
            False,
            frozenset({10, 11, 20, 21}),
            True,
        ),
    ]


def test_prepare_batch_rejects_non_boolean_staging_policy():
    manager = _ResidencyManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        participant.prepare_batch(
            41,
            (_prepare_row(),),
            stage_all_original_blocks=1,
        )


def test_residency_ticket_accumulates_fixed_q_materialization_groups():
    manager = _ResidencyManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    participant.prepare_batch(
        41,
        (_prepare_row(), _second_prepare_row()),
    )

    assert participant.is_prepared_for(41, (9,))
    assert participant.is_prepared_for(41, (7,))
    assert not participant.is_prepared_for(41, (11,))
    participant.mark_materialized(41, (9,))
    participant.mark_materialized(41, (7,))

    result = participant.precommit_batch(
        41,
        (
            SpeculativeResidencyPrecommitRow(
                sequence_id=7,
                committed_block_identities=((2, 1),),
                rejected_block_identities=(),
                accepted_materialized_end=5,
            ),
            SpeculativeResidencyPrecommitRow(
                sequence_id=9,
                committed_block_identities=((3, 1),),
                rejected_block_identities=(),
                accepted_materialized_end=5,
            ),
        ),
    )

    assert result.status == "precommitted"


def test_residency_ticket_full_commit_seals_in_place():
    manager = _ResidencyManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )

    prepare = participant.prepare_batch(41, (_prepare_row(),))
    participant.mark_materialized(41, (7,))
    precommit = participant.precommit_batch(
        41,
        (
            SpeculativeResidencyPrecommitRow(
                sequence_id=7,
                committed_block_identities=((2, 1),),
                rejected_block_identities=(),
                accepted_materialized_end=5,
            ),
        ),
    )
    seal = participant.seal_batch(41)

    assert prepare.status == "prepared"
    assert precommit.status == "precommitted"
    assert seal.status == "sealed"
    assert manager.logical_to_slot == {1: 0, 2: 1}
    assert manager.discarded == []
    assert manager.writeback_calls == [(1, 2)]
    assert manager.stats["speculative_residency_prepares"] == 1
    assert manager.stats["speculative_residency_precommits"] == 1
    assert manager.stats["speculative_residency_seals"] == 1
    assert manager.stats[
        "speculative_residency_committed_blocks"
    ] == 1


def test_residency_ticket_partial_commit_discards_rejected_without_d2h():
    manager = _ResidencyManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    participant.prepare_batch(41, (_prepare_row(),))
    participant.mark_materialized(41, (7,))
    participant.precommit_batch(
        41,
        (
            SpeculativeResidencyPrecommitRow(
                sequence_id=7,
                committed_block_identities=(),
                rejected_block_identities=((2, 1),),
                accepted_materialized_end=4,
            ),
        ),
    )

    result = participant.seal_batch(41)

    assert result.rejected_block_identities == ((2, 1),)
    assert manager.logical_to_slot == {1: 0}
    assert manager.discarded == [((2, 1),)]
    assert manager.writeback_calls == [(1,)]
    assert manager.stats[
        "speculative_residency_rejected_blocks"
    ] == 1
    assert manager.stats[
        "speculative_residency_rejected_d2h_copies"
    ] == 0


def test_residency_ticket_rollback_discards_all_reserved_blocks_once():
    manager = _ResidencyManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    participant.prepare_batch(41, (_prepare_row(),))

    result = participant.rollback_batch(41)

    assert result.status == "rolled_back"
    assert manager.logical_to_slot == {1: 0}
    assert manager.discarded == [((2, 1),)]
    assert manager.stats["speculative_residency_rollbacks"] == 1
    with pytest.raises(RuntimeError, match="not rollbackable"):
        participant.rollback_batch(41)


def test_residency_precommit_requires_exact_reserved_partition():
    manager = _ResidencyManager()
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )
    participant.prepare_batch(41, (_prepare_row(),))
    participant.mark_materialized(41, (7,))

    with pytest.raises(RuntimeError, match="partition"):
        participant.precommit_batch(
            41,
            (
                SpeculativeResidencyPrecommitRow(
                    sequence_id=7,
                    committed_block_identities=(),
                    rejected_block_identities=(),
                    accepted_materialized_end=4,
                ),
            ),
        )


def test_prepare_failure_discards_only_reserved_mappings():
    manager = _ResidencyManager()
    manager.fail_history = True
    participant = SpeculativeResidencyParticipant(
        participant_id=0,
        manager=manager,
        block_size=4,
    )

    with pytest.raises(
        RuntimeError,
        match="unreadable historical block",
    ):
        participant.prepare_batch(41, (_prepare_row(),))

    assert manager.logical_to_slot == {1: 0}
    assert 2 not in manager.logical_to_slot
