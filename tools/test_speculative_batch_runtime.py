from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
speculative_package = types.ModuleType("tinyvllm.speculative")
speculative_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "speculative")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.speculative", speculative_package)

import tinyvllm.speculative.batch_runtime as batch_runtime_module
from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
)
from tinyvllm.speculative.batch_runtime import (
    FirstTargetProposalResult,
    FirstTargetResult,
    NativeSpeculativeBatchError,
    TailBatchResult,
    apply_prepared_speculative_side_state,
    commit_prepared_native_speculative_batch,
    execute_native_speculative_batch,
    prepare_native_speculative_batch,
    rollback_prepared_native_speculative_batch,
    rollback_prepared_speculative_side_state,
    seal_prepared_speculative_side_state,
)
from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateCallbacks,
)


class _Sequence:
    block_size = 4

    def __init__(
        self,
        sequence_id,
        *,
        token_ids=(1, 2, 3),
        max_tokens=16,
        ignore_eos=False,
    ):
        self.seq_id = sequence_id
        self.token_ids = list(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.last_token = self.token_ids[-1]
        self.block_table = [sequence_id * 10]
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

    def __len__(self):
        return self.num_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def append_token(self, token_id):
        self.token_ids.append(int(token_id))
        self.num_tokens += 1
        self.last_token = int(token_id)


class _AcceptanceFailSequence(_Sequence):
    def __getattribute__(self, name):
        if name == "ignore_eos":
            raise RuntimeError("acceptance failed")
        return super().__getattribute__(name)


class _BlockManager:
    def __init__(
        self,
        *,
        fail_begin_id=None,
        fail_authorize_id=None,
        fail_mark_id=None,
        fail_commit_id=None,
        fail_rollback_ids=(),
    ):
        self.fail_begin_id = fail_begin_id
        self.fail_authorize_id = fail_authorize_id
        self.fail_mark_id = fail_mark_id
        self.fail_commit_id = fail_commit_id
        self.fail_rollback_ids = set(fail_rollback_ids)
        self.calls = []
        self.transactions = {}
        self.authorizations = {}

    def begin_speculative_kv_transaction(
        self,
        seq,
        proposed_token_count,
    ):
        self.calls.append(
            ("begin", seq.seq_id, proposed_token_count)
        )
        if seq.seq_id == self.fail_begin_id:
            raise RuntimeError(f"begin {seq.seq_id} failed")
        reserved = (
            ()
            if proposed_token_count <= 1
            else (seq.seq_id * 100 + 1,)
        )
        transaction = SimpleNamespace(
            sequence_id=seq.seq_id,
            proposed_token_count=proposed_token_count,
            original_block_table=tuple(seq.block_table),
            original_block_generations=tuple(
                10 + index
                for index, _ in enumerate(seq.block_table)
            ),
            reserved_block_ids=reserved,
            reserved_block_generations=tuple(
                20 + index
                for index, _ in enumerate(reserved)
            ),
            materialized_token_count=None,
            state="reserved",
        )
        self.transactions[seq.seq_id] = transaction
        return transaction

    def authorize_speculative_kv_write(
        self,
        transaction,
        seq,
    ):
        self.calls.append(("authorize", seq.seq_id))
        if seq.seq_id == self.fail_authorize_id:
            raise RuntimeError(
                f"authorize {seq.seq_id} failed"
            )
        authorization = object()
        self.authorizations[seq.seq_id] = authorization
        return authorization

    def mark_speculative_kv_materialized(
        self,
        transaction,
        materialized_token_count,
    ):
        sequence_id = transaction.sequence_id
        self.calls.append(
            ("mark", sequence_id, materialized_token_count)
        )
        if sequence_id == self.fail_mark_id:
            raise RuntimeError(f"mark {sequence_id} failed")
        transaction.materialized_token_count = (
            materialized_token_count
        )
        transaction.state = "materialized"

    def commit_speculative_kv_transaction(
        self,
        transaction,
        seq,
        accepted_tokens,
    ):
        sequence_id = transaction.sequence_id
        self.calls.append(
            (
                "commit",
                sequence_id,
                tuple(accepted_tokens),
            )
        )
        if sequence_id == self.fail_commit_id:
            raise RuntimeError(f"commit {sequence_id} failed")
        for token_id in accepted_tokens:
            seq.append_token(token_id)
        if (
            len(accepted_tokens) > 1
            and transaction.reserved_block_ids
        ):
            seq.block_table.append(
                transaction.reserved_block_ids[0]
            )
        transaction.state = "committed"

    def prepare_speculative_kv_commit(
        self,
        transaction,
        seq,
        accepted_tokens,
    ):
        sequence_id = transaction.sequence_id
        self.calls.append(
            (
                "prepare_commit",
                sequence_id,
                tuple(accepted_tokens),
            )
        )
        committed = (
            transaction.reserved_block_ids[:1]
            if len(accepted_tokens) > 1
            else ()
        )
        return SimpleNamespace(
            sequence_id=sequence_id,
            sequence=seq,
            transaction=transaction,
            accepted_tokens=tuple(accepted_tokens),
            committed_block_ids=committed,
            unused_block_ids=tuple(
                block_id
                for block_id
                in transaction.reserved_block_ids
                if block_id not in committed
            ),
        )

    def commit_speculative_kv_commit_batch(self, plans):
        self.calls.append(
            (
                "commit_batch",
                tuple(plan.sequence_id for plan in plans),
            )
        )
        if any(
            plan.sequence_id == self.fail_commit_id
            for plan in plans
        ):
            raise RuntimeError(
                f"commit {self.fail_commit_id} failed"
            )
        for plan in plans:
            plan.sequence.block_table.extend(
                plan.committed_block_ids
            )
            plan.transaction.state = "committed"

    def rollback_speculative_kv_transaction(
        self,
        transaction,
        seq,
    ):
        sequence_id = transaction.sequence_id
        self.calls.append(("rollback", sequence_id))
        if sequence_id in self.fail_rollback_ids:
            raise RuntimeError(
                f"rollback {sequence_id} failed"
            )
        transaction.state = "rolled_back"


class _Adapter:
    def __init__(
        self,
        proposals_by_id,
        *,
        fail=False,
        requires_hidden=False,
    ):
        self._proposals_by_id = {
            int(sequence_id): tuple(tokens)
            for sequence_id, tokens
            in proposals_by_id.items()
        }
        self._fail = fail
        self.calls = []
        self._capabilities = DraftCapabilities(
            source_type="fixture",
            supports_batch=True,
            requires_target_hidden=requires_hidden,
            requires_target_logits=False,
            max_proposal_tokens=8,
        )

    @property
    def capabilities(self):
        return self._capabilities

    def propose_batch(self, contexts):
        self.calls.append(contexts)
        if self._fail:
            raise RuntimeError("proposal failed")
        return tuple(
            DraftProposal(
                sequence_id=context.sequence_id,
                token_ids=self._proposals_by_id[
                    context.sequence_id
                ],
                source_type="fixture",
                metadata={
                    "first_target": context.first_target_token,
                },
                timing_ms={"draft_ms": 0.1},
            )
            for context in reversed(contexts)
        )


class _RecordingSideState:
    def __init__(
        self,
        events,
        *,
        fail_select=False,
        fail_rollback=False,
    ):
        self.events = events
        self.fail_select = fail_select
        self.fail_rollback = fail_rollback
        self.handle = {
            "transaction_id": "side-1",
        }

    def callbacks(self):
        return SpeculativeSideStateCallbacks(
            prepare=self.prepare,
            select=self.select,
            apply=self.apply,
            seal=self.seal,
            rollback=self.rollback,
        )

    def prepare(self, sequences):
        self.events.append((
            "side_prepare",
            tuple(seq.seq_id for seq in sequences),
        ))
        return self.handle

    def select(self, handle, rows):
        assert handle is self.handle
        self.events.append(("side_select", rows))
        if self.fail_select:
            raise RuntimeError("side select failed")
        return {
            "transaction_id": handle["transaction_id"],
        }

    def apply(self, handle):
        assert handle is self.handle
        self.events.append((
            "side_apply",
            handle["transaction_id"],
        ))
        return {
            "operation": "apply",
        }

    def seal(self, handle):
        assert handle is self.handle
        self.events.append((
            "side_seal",
            handle["transaction_id"],
        ))
        return {
            "operation": "seal",
        }

    def rollback(self, handle):
        assert handle is self.handle
        self.events.append((
            "side_rollback",
            handle["transaction_id"],
        ))
        if self.fail_rollback:
            raise RuntimeError("side rollback failed")
        return {
            "operation": "rollback",
        }


def _first_target_callback(
    targets_by_id,
    calls,
    *,
    fail=False,
):
    def callback(seqs):
        calls.append(tuple(seq.seq_id for seq in seqs))
        if fail:
            raise RuntimeError("first target failed")
        return tuple(
            FirstTargetResult(
                sequence_id=seq.seq_id,
                target_token=targets_by_id[seq.seq_id],
                target_hidden={
                    "sequence_id": seq.seq_id,
                },
                metadata={"batch": True},
            )
            for seq in reversed(seqs)
        )

    return callback


def test_side_state_prepare_precedes_first_target_and_selects_after_acceptance():
    events = []
    side_state = _RecordingSideState(events)

    def first_target(sequences):
        events.append(("first_target",))
        return (
            FirstTargetResult(
                sequence_id=sequences[0].seq_id,
                target_token=10,
            ),
        )

    def tail(items):
        events.append(("tail",))
        return (
            TailBatchResult(
                sequence_id=items[0].sequence_id,
                target_tokens=(11, 99),
            ),
        )

    prepared = prepare_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(_Sequence(1),),
        draft_adapter=_Adapter({
            1: (10, 11, 12),
        }),
        eos_token=999,
        run_first_targets=first_target,
        run_tail_batch=tail,
        side_state_callbacks=side_state.callbacks(),
    )

    assert [event[0] for event in events] == [
        "side_prepare",
        "first_target",
        "tail",
        "side_select",
    ]
    assert prepared.side_state_handle is side_state.handle
    assert prepared.side_state_state == "selected"
    assert prepared.side_state_selection[0].sequence_id == 1
    assert (
        prepared.side_state_selection[0]
        .committed_input_count
        == 3
    )


def test_side_state_rolls_back_once_when_first_target_fails():
    events = []
    side_state = _RecordingSideState(events)

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="first target failed",
    ):
        prepare_native_speculative_batch(
            block_manager=_BlockManager(),
            seqs=(_Sequence(1),),
            draft_adapter=_Adapter({
                1: (10, 11),
            }),
            eos_token=999,
            run_first_targets=_first_target_callback(
                {1: 10},
                [],
                fail=True,
            ),
            run_tail_batch=_tail_callback({}, []),
            side_state_callbacks=side_state.callbacks(),
        )

    assert [event[0] for event in events] == [
        "side_prepare",
        "side_rollback",
    ]


def test_side_state_rolls_back_once_when_kv_reservation_fails():
    events = []
    side_state = _RecordingSideState(events)

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="begin 2 failed",
    ):
        prepare_native_speculative_batch(
            block_manager=_BlockManager(fail_begin_id=2),
            seqs=(_Sequence(1), _Sequence(2)),
            draft_adapter=_Adapter({
                1: (10, 11),
                2: (20, 21),
            }),
            eos_token=999,
            run_first_targets=_first_target_callback(
                {1: 10, 2: 20},
                [],
            ),
            run_tail_batch=_tail_callback({}, []),
            side_state_callbacks=side_state.callbacks(),
        )

    assert [
        event[0]
        for event in events
        if event[0] == "side_rollback"
    ] == ["side_rollback"]


def test_side_state_select_failure_rolls_back_kv_and_side_state():
    events = []
    side_state = _RecordingSideState(
        events,
        fail_select=True,
    )
    manager = _BlockManager()

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="side select failed",
    ):
        prepare_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1),),
            draft_adapter=_Adapter({
                1: (10, 11),
            }),
            eos_token=999,
            run_first_targets=_first_target_callback(
                {1: 10},
                [],
            ),
            run_tail_batch=_tail_callback(
                {1: (11,)},
                [],
            ),
            side_state_callbacks=side_state.callbacks(),
        )

    assert manager.transactions[1].state == "rolled_back"
    assert [event[0] for event in events][-2:] == [
        "side_select",
        "side_rollback",
    ]


def test_side_state_helpers_enforce_apply_seal_and_rollback_states():
    events = []
    side_state = _RecordingSideState(events)
    prepared = prepare_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(_Sequence(1),),
        draft_adapter=_Adapter({
            1: (10,),
        }),
        eos_token=999,
        run_first_targets=_first_target_callback(
            {1: 10},
            [],
        ),
        run_tail_batch=_tail_callback({}, []),
        side_state_callbacks=side_state.callbacks(),
    )

    apply_prepared_speculative_side_state(prepared)
    assert prepared.side_state_state == "applied"
    rollback_prepared_speculative_side_state(prepared)
    assert prepared.side_state_state == "rolled_back"
    rollback_prepared_speculative_side_state(prepared)
    assert [
        event[0]
        for event in events
        if event[0] == "side_rollback"
    ] == ["side_rollback"]
    with pytest.raises(RuntimeError, match="selected before apply"):
        apply_prepared_speculative_side_state(prepared)

    second_events = []
    second_side_state = _RecordingSideState(second_events)
    second = prepare_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(_Sequence(2),),
        draft_adapter=_Adapter({
            2: (20,),
        }),
        eos_token=999,
        run_first_targets=_first_target_callback(
            {2: 20},
            [],
        ),
        run_tail_batch=_tail_callback({}, []),
        side_state_callbacks=second_side_state.callbacks(),
    )
    apply_prepared_speculative_side_state(second)
    seal_prepared_speculative_side_state(second)
    assert second.side_state_state == "sealed"
    with pytest.raises(RuntimeError, match="sealed.*rolled back"):
        rollback_prepared_speculative_side_state(second)


def test_side_state_seal_accepts_committed_publication_container():
    events = []
    side_state = _RecordingSideState(events)
    prepared = prepare_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(_Sequence(1),),
        draft_adapter=_Adapter({1: (10,)}),
        eos_token=999,
        run_first_targets=_first_target_callback(
            {1: 10},
            [],
        ),
        run_tail_batch=_tail_callback({}, []),
        side_state_callbacks=side_state.callbacks(),
    )
    apply_prepared_speculative_side_state(prepared)
    prepared.state = "committed"

    seal_prepared_speculative_side_state(prepared)

    assert prepared.side_state_state == "sealed"
    assert events[-1][0] == "side_seal"


def test_no_side_state_provider_keeps_disabled_lifecycle():
    prepared = prepare_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(_Sequence(1),),
        draft_adapter=_Adapter({
            1: (10,),
        }),
        eos_token=999,
        run_first_targets=_first_target_callback(
            {1: 10},
            [],
        ),
        run_tail_batch=_tail_callback({}, []),
    )

    assert prepared.side_state_callbacks is None
    assert prepared.side_state_handle is None
    assert prepared.side_state_selection == ()
    assert prepared.side_state_state == "disabled"
    assert apply_prepared_speculative_side_state(prepared) is None
    assert seal_prepared_speculative_side_state(prepared) is None
    assert rollback_prepared_speculative_side_state(prepared) is None


def _tail_callback(
    targets_by_id,
    calls,
    *,
    fail=False,
    omit_id=None,
):
    def callback(items):
        calls.append(items)
        if fail:
            raise RuntimeError("tail failed")
        return tuple(
            TailBatchResult(
                sequence_id=item.sequence_id,
                target_tokens=tuple(
                    targets_by_id[item.sequence_id]
                ),
                metadata={
                    "query_len": item.plan.query_len,
                },
                auxiliary={"tail": True},
            )
            for item in reversed(items)
            if item.sequence_id != omit_id
        )

    return callback


def _prepare_provider_proposal(
    *,
    proposal,
    target_tokens,
):
    sequence_id = proposal.sequence_id
    manager = _BlockManager()

    def provider(active_sequences):
        assert tuple(
            seq.seq_id for seq in active_sequences
        ) == (sequence_id,)
        return (
            FirstTargetProposalResult(
                sequence_id=sequence_id,
                target_token=target_tokens[0],
                proposal=proposal,
            ),
        )

    return (
        manager,
        prepare_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(sequence_id),),
            eos_token=999,
            run_first_targets_and_proposals=provider,
            run_tail_batch=_tail_callback(
                {
                    sequence_id: target_tokens[1:],
                },
                [],
            ),
        ),
    )


def test_prepared_batch_exposes_lifecycle_finalize_rows():
    _, prepared = _prepare_provider_proposal(
        proposal=DraftProposal(
            sequence_id=1,
            token_ids=(11, 12, 13),
            source_type="fixture",
            proposal_transaction_id="tx-1",
        ),
        target_tokens=(11, 12, 99),
    )

    rows = (
        batch_runtime_module
        .build_prepared_proposal_finalize_rows(prepared)
    )

    assert rows == (
        batch_runtime_module.PreparedProposalFinalizeRow(
            sequence_id=1,
            proposal_transaction_id="tx-1",
            accepted_proposal_tokens=2,
        ),
    )


@pytest.mark.parametrize(
    "proposal_tokens,target_tokens,accepted_count",
    [
        ((11, 12), (99, 12), 0),
        ((11,), (11,), 1),
    ],
)
def test_prepared_finalize_rows_preserve_exact_acceptance(
    proposal_tokens,
    target_tokens,
    accepted_count,
):
    _, prepared = _prepare_provider_proposal(
        proposal=DraftProposal(
            sequence_id=1,
            token_ids=proposal_tokens,
            source_type="fixture",
            proposal_transaction_id="tx-1",
        ),
        target_tokens=target_tokens,
    )

    rows = (
        batch_runtime_module
        .build_prepared_proposal_finalize_rows(prepared)
    )

    assert rows[0].accepted_proposal_tokens == accepted_count


@pytest.mark.parametrize("proposal_tokens", [(), (11,)])
def test_prepared_finalize_rows_skip_host_proposals(
    proposal_tokens,
):
    target_tokens = (11,)
    _, prepared = _prepare_provider_proposal(
        proposal=DraftProposal(
            sequence_id=1,
            token_ids=proposal_tokens,
            source_type="fixture",
        ),
        target_tokens=target_tokens,
    )

    rows = (
        batch_runtime_module
        .build_prepared_proposal_finalize_rows(prepared)
    )

    assert rows == ()


def test_prepared_finalize_rows_reject_duplicate_transactions():
    manager = _BlockManager()

    def provider(active_sequences):
        return tuple(
            FirstTargetProposalResult(
                sequence_id=seq.seq_id,
                target_token=seq.seq_id * 10,
                proposal=DraftProposal(
                    sequence_id=seq.seq_id,
                    token_ids=(seq.seq_id * 10,),
                    source_type="fixture",
                    proposal_transaction_id="duplicate",
                ),
            )
            for seq in active_sequences
        )

    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=(_Sequence(1), _Sequence(2)),
        eos_token=999,
        run_first_targets_and_proposals=provider,
        run_tail_batch=_tail_callback({}, []),
    )

    with pytest.raises(ValueError, match="transaction.*unique"):
        (
            batch_runtime_module
            .build_prepared_proposal_finalize_rows(prepared)
        )


def test_provider_rows_reuse_transaction_and_tail_runtime():
    seqs = (_Sequence(1), _Sequence(2))
    manager = _BlockManager()
    provider_calls = []
    tail_calls = []

    def provider(active_sequences):
        provider_calls.append(
            tuple(seq.seq_id for seq in active_sequences)
        )
        return (
            FirstTargetProposalResult(
                sequence_id=2,
                target_token=20,
                proposal=DraftProposal(
                    sequence_id=2,
                    token_ids=(20, 21, 22),
                    source_type="fixture",
                ),
            ),
            FirstTargetProposalResult(
                sequence_id=1,
                target_token=10,
                proposal=DraftProposal(
                    sequence_id=1,
                    token_ids=(),
                    source_type="fixture",
                ),
            ),
        )

    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=seqs,
        eos_token=99,
        run_first_targets_and_proposals=provider,
        run_tail_batch=_tail_callback(
            {2: (21, 22)},
            tail_calls,
        ),
    )

    assert provider_calls == [(1, 2)]
    assert len(tail_calls) == 1
    assert tuple(
        row.sequence_id for row in prepared.sequences
    ) == (1, 2)
    assert prepared.sequences[0].plan is None
    assert prepared.sequences[1].plan.query_len == 2
    assert (
        prepared.sequences[1].proposal.token_ids
        == (20, 21, 22)
    )


def test_provider_failure_creates_no_transactions():
    manager = _BlockManager()

    def provider(active_sequences):
        raise RuntimeError("provider failed")

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="first_target_proposal_batch",
    ) as exc_info:
        prepare_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1),),
            eos_token=99,
            run_first_targets_and_proposals=provider,
            run_tail_batch=_tail_callback({}, []),
        )

    assert manager.calls == []
    assert isinstance(exc_info.value.cause, RuntimeError)


def test_batch4_uses_one_first_target_and_one_tail_callback():
    seqs = tuple(_Sequence(index) for index in range(1, 5))
    manager = _BlockManager()
    adapter = _Adapter({
        1: (),
        2: (20,),
        3: (30, 31),
        4: (40, 41, 42, 43),
    })
    first_calls = []
    tail_calls = []

    result = execute_native_speculative_batch(
        block_manager=manager,
        seqs=seqs,
        draft_adapter=adapter,
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10, 2: 20, 3: 30, 4: 40},
            first_calls,
        ),
        run_tail_batch=_tail_callback(
            {
                3: (31,),
                4: (41, 999, 43),
            },
            tail_calls,
        ),
    )

    assert first_calls == [(1, 2, 3, 4)]
    assert len(tail_calls) == 1
    assert tuple(
        item.sequence_id for item in tail_calls[0]
    ) == (3, 4)
    assert tail_calls[0][0].original_block_identities == (
        (30, 10),
    )
    assert tail_calls[0][0].reserved_block_identities == (
        (301, 20),
    )
    assert result.first_target_callback_count == 1
    assert result.tail_callback_count == 1
    assert tuple(
        item.sequence_id for item in result.sequences
    ) == (1, 2, 3, 4)
    assert tuple(
        item.accepted_tokens for item in result.sequences
    ) == (
        (),
        (20,),
        (30, 31),
        (40, 41),
    )
    assert result.sequences[0].plan is None
    assert result.sequences[1].plan.query_len == 0
    assert result.sequences[2].plan.query_len == 1
    assert result.sequences[3].plan.query_len == 3
    assert (
        result.sequences[3].proxy_block_table
        == (40, 401)
    )
    assert (
        result.sequences[3].committed_blocks
        == (401,)
    )
    assert result.sequences[3].released_blocks == ()
    assert [
        call for call in manager.calls if call[0] == "mark"
    ] == [
        ("mark", 2, 0),
        ("mark", 3, 1),
        ("mark", 4, 3),
    ]


def test_tail_returns_before_each_transaction_is_materialized_once():
    seq = _Sequence(3)
    manager = _BlockManager()
    events = []
    original_mark = manager.mark_speculative_kv_materialized

    def tail_callback(items):
        events.append(("tail", tuple(
            item.sequence_id for item in items
        )))
        result = (
            TailBatchResult(
                sequence_id=3,
                target_tokens=(31,),
            ),
        )
        events.append(("tail_returning",))
        return result

    def mark_after_tail(transaction, materialized_token_count):
        assert events[-1] == ("tail_returning",)
        events.append((
            "mark",
            transaction.sequence_id,
            materialized_token_count,
        ))
        original_mark(
            transaction,
            materialized_token_count,
        )

    manager.mark_speculative_kv_materialized = mark_after_tail

    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=(seq,),
        draft_adapter=_Adapter({3: (30, 31)}),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {3: 30},
            [],
        ),
        run_tail_batch=tail_callback,
    )

    assert prepared.state == "prepared"
    assert events == [
        ("tail", (3,)),
        ("tail_returning",),
        ("mark", 3, 1),
    ]
    assert [
        call
        for call in manager.calls
        if call[0] == "mark"
    ] == [("mark", 3, 1)]


def test_prepare_attaches_exact_transaction_authorization_to_tail_items():
    seqs = (_Sequence(3), _Sequence(4))
    manager = _BlockManager()
    tail_calls = []

    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=seqs,
        draft_adapter=_Adapter({
            3: (30, 31),
            4: (40, 41, 42),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {3: 30, 4: 40},
            [],
        ),
        run_tail_batch=_tail_callback(
            {3: (31,), 4: (41, 42)},
            tail_calls,
        ),
    )

    assert prepared.state == "prepared"
    assert len(tail_calls) == 1
    for item in tail_calls[0]:
        assert (
            item.transaction_authorization
            is manager.authorizations[item.sequence_id]
        )


def test_authorization_failure_rolls_back_reserved_transaction():
    seqs = (_Sequence(3),)
    manager = _BlockManager(fail_authorize_id=3)

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="authorize 3 failed",
    ):
        prepare_native_speculative_batch(
            block_manager=manager,
            seqs=seqs,
            draft_adapter=_Adapter({
                3: (30, 31),
            }),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {3: 30},
                [],
            ),
            run_tail_batch=_tail_callback(
                {3: (31,)},
                [],
            ),
        )

    assert manager.transactions[3].state == "rolled_back"
    assert ("rollback", 3) in manager.calls


def test_prepare_batch4_keeps_live_sequence_metadata_uncommitted():
    seqs = tuple(_Sequence(index) for index in range(1, 5))
    manager = _BlockManager()
    before_tokens = tuple(tuple(seq.token_ids) for seq in seqs)
    before_block_tables = tuple(
        tuple(seq.block_table) for seq in seqs
    )

    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=seqs,
        draft_adapter=_Adapter({
            1: (),
            2: (20,),
            3: (30, 31),
            4: (40, 41, 42, 43),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10, 2: 20, 3: 30, 4: 40},
            [],
        ),
        run_tail_batch=_tail_callback(
            {
                3: (31,),
                4: (41, 999, 43),
            },
            [],
        ),
    )

    assert prepared.state == "prepared"
    assert tuple(
        row.sequence_id for row in prepared.sequences
    ) == (1, 2, 3, 4)
    assert tuple(tuple(seq.token_ids) for seq in seqs) == before_tokens
    assert tuple(
        tuple(seq.block_table) for seq in seqs
    ) == before_block_tables
    assert prepared.sequences[0].transaction is None
    assert all(
        row.transaction is None
        or row.transaction.state == "materialized"
        for row in prepared.sequences
    )


def test_prepared_rollback_releases_every_active_transaction():
    seqs = (_Sequence(1), _Sequence(2))
    manager = _BlockManager()
    before_tokens = tuple(tuple(seq.token_ids) for seq in seqs)
    before_block_tables = tuple(
        tuple(seq.block_table) for seq in seqs
    )
    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=seqs,
        draft_adapter=_Adapter({
            1: (10, 11),
            2: (20, 21),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10, 2: 20},
            [],
        ),
        run_tail_batch=_tail_callback(
            {1: (11,), 2: (21,)},
            [],
        ),
    )
    proposal_transaction_ids = tuple(
        row.proposal.proposal_transaction_id
        for row in prepared.sequences
    )

    rolled_back = rollback_prepared_native_speculative_batch(
        block_manager=manager,
        prepared=prepared,
    )

    assert rolled_back == (1, 2)
    assert prepared.state == "rolled_back"
    assert all(
        row.transaction.state == "rolled_back"
        for row in prepared.sequences
    )
    assert tuple(tuple(seq.token_ids) for seq in seqs) == before_tokens
    assert tuple(
        tuple(seq.block_table) for seq in seqs
    ) == before_block_tables
    assert tuple(
        row.proposal.proposal_transaction_id
        for row in prepared.sequences
    ) == proposal_transaction_ids
    with pytest.raises(RuntimeError, match="rolled_back"):
        rollback_prepared_native_speculative_batch(
            block_manager=manager,
            prepared=prepared,
        )


def test_prepared_commit_is_token_free_and_exactly_once():
    seqs = (_Sequence(1), _Sequence(2))
    manager = _BlockManager()
    prepared = prepare_native_speculative_batch(
        block_manager=manager,
        seqs=seqs,
        draft_adapter=_Adapter({
            1: (10, 11),
            2: (20,),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10, 2: 20},
            [],
        ),
        run_tail_batch=_tail_callback(
            {1: (11,)},
            [],
        ),
    )
    before_tokens = tuple(tuple(seq.token_ids) for seq in seqs)

    result = commit_prepared_native_speculative_batch(
        block_manager=manager,
        prepared=prepared,
    )

    assert prepared.state == "committed"
    assert tuple(tuple(seq.token_ids) for seq in seqs) == before_tokens
    assert result.sequences[0].committed_blocks == (101,)
    assert result.sequences[0].released_blocks == ()
    assert result.sequences[1].committed_blocks == ()
    assert result.sequences[1].released_blocks == ()
    with pytest.raises(RuntimeError, match="committed"):
        commit_prepared_native_speculative_batch(
            block_manager=manager,
            prepared=prepared,
        )


def test_all_empty_or_k1_proposals_skip_tail_callback():
    seqs = (_Sequence(1), _Sequence(2))
    tail_calls = []

    result = execute_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=seqs,
        draft_adapter=_Adapter({
            1: (),
            2: (20,),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10, 2: 20},
            [],
        ),
        run_tail_batch=_tail_callback({}, tail_calls),
    )

    assert tail_calls == []
    assert result.tail_callback_count == 0
    assert result.sequences[1].accepted_tokens == (20,)


def test_eos_and_output_budget_truncate_per_sequence():
    eos_seq = _Sequence(1, ignore_eos=False)
    budget_seq = _Sequence(
        2,
        token_ids=(1, 2, 3, 4),
        max_tokens=2,
    )
    budget_seq.num_prompt_tokens = 3

    result = execute_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(eos_seq, budget_seq),
        draft_adapter=_Adapter({
            1: (10, 99, 12),
            2: (20, 21, 22),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10, 2: 20},
            [],
        ),
        run_tail_batch=_tail_callback(
            {
                1: (99, 12),
                2: (21, 22),
            },
            [],
        ),
    )

    assert result.sequences[0].accepted_tokens == (10, 99)
    assert result.sequences[0].eos_truncated is True
    assert result.sequences[1].accepted_tokens == (20,)
    assert (
        result.sequences[1].output_budget_truncated
        is True
    )


def test_partial_acceptance_releases_uncommitted_reserved_block():
    result = execute_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(_Sequence(1),),
        draft_adapter=_Adapter({
            1: (10, 11, 12, 13),
        }),
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10},
            [],
        ),
        run_tail_batch=_tail_callback(
            {1: (999, 12, 13)},
            [],
        ),
    )

    row = result.sequences[0]
    assert row.accepted_tokens == (10,)
    assert row.committed_blocks == ()
    assert row.released_blocks == (101,)


def test_adapter_receives_immutable_history_and_first_target_payload():
    seq = _Sequence(1)
    adapter = _Adapter(
        {1: (10,)},
        requires_hidden=True,
    )

    execute_native_speculative_batch(
        block_manager=_BlockManager(),
        seqs=(seq,),
        draft_adapter=adapter,
        eos_token=99,
        run_first_targets=_first_target_callback(
            {1: 10},
            [],
        ),
        run_tail_batch=_tail_callback({}, []),
    )

    context = adapter.calls[0][0]
    assert context.token_ids == (1, 2, 3)
    assert context.first_target_token == 10
    assert context.target_hidden == {"sequence_id": 1}


def test_first_target_failure_creates_no_transactions():
    manager = _BlockManager()

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="first_target_batch",
    ) as exc_info:
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1),),
            draft_adapter=_Adapter({1: (10,)}),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10},
                [],
                fail=True,
            ),
            run_tail_batch=_tail_callback({}, []),
        )

    assert manager.calls == []
    assert isinstance(
        exc_info.value.cause,
        RuntimeError,
    )


def test_adapter_failure_creates_no_transactions():
    manager = _BlockManager()

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="draft_proposal",
    ):
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1),),
            draft_adapter=_Adapter(
                {1: (10,)},
                fail=True,
            ),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10},
                [],
            ),
            run_tail_batch=_tail_callback({}, []),
        )

    assert manager.calls == []


def test_reserve_failure_rolls_back_prior_transactions():
    manager = _BlockManager(fail_begin_id=2)

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="reserve",
    ) as exc_info:
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1), _Sequence(2)),
            draft_adapter=_Adapter({
                1: (10, 11),
                2: (20, 21),
            }),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10, 2: 20},
                [],
            ),
            run_tail_batch=_tail_callback({}, []),
        )

    assert ("rollback", 1) in manager.calls
    assert exc_info.value.rolled_back_sequence_ids == (1,)
    assert exc_info.value.committed_sequence_ids == ()


@pytest.mark.parametrize(
    "failure_kind,expected_phase",
    [
        ("tail", "tail_batch"),
        ("tail_validation", "tail_batch"),
        ("mark", "kv_materialize"),
    ],
)
def test_precommit_failures_roll_back_all_transactions(
    failure_kind,
    expected_phase,
):
    manager = _BlockManager(
        fail_mark_id=2 if failure_kind == "mark" else None
    )
    tail = _tail_callback(
        {1: (11,), 2: (21,)},
        [],
        fail=failure_kind == "tail",
        omit_id=2 if failure_kind == "tail_validation" else None,
    )

    with pytest.raises(
        NativeSpeculativeBatchError,
        match=expected_phase,
    ) as exc_info:
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1), _Sequence(2)),
            draft_adapter=_Adapter({
                1: (10, 11),
                2: (20, 21),
            }),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10, 2: 20},
                [],
            ),
            run_tail_batch=tail,
        )

    assert set(
        exc_info.value.rolled_back_sequence_ids
    ) == {1, 2}
    assert exc_info.value.committed_sequence_ids == ()


def test_acceptance_failure_rolls_back_all_transactions():
    manager = _BlockManager()

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="acceptance",
    ) as exc_info:
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=(
                _Sequence(1),
                _AcceptanceFailSequence(2),
            ),
            draft_adapter=_Adapter({
                1: (10, 11),
                2: (20, 21),
            }),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10, 2: 20},
                [],
            ),
            run_tail_batch=_tail_callback(
                {1: (11,), 2: (21,)},
                [],
            ),
        )

    assert set(
        exc_info.value.rolled_back_sequence_ids
    ) == {1, 2}
    assert [
        call for call in manager.calls
        if call[0] == "rollback"
    ] == [
        ("rollback", 1),
        ("rollback", 2),
    ]


def test_commit_failure_rolls_back_every_prepared_transaction():
    manager = _BlockManager(fail_commit_id=2)
    seqs = (_Sequence(1), _Sequence(2), _Sequence(3))

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="metadata_commit",
    ) as exc_info:
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=seqs,
            draft_adapter=_Adapter({
                1: (10, 11),
                2: (20, 21),
                3: (30, 31),
            }),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10, 2: 20, 3: 30},
                [],
            ),
            run_tail_batch=_tail_callback(
                {
                    1: (11,),
                    2: (21,),
                    3: (31,),
                },
                [],
            ),
        )

    error = exc_info.value
    assert error.committed_sequence_ids == ()
    assert set(error.rolled_back_sequence_ids) == {1, 2, 3}
    assert seqs[0].token_ids == [1, 2, 3]
    assert seqs[1].token_ids == [1, 2, 3]
    assert seqs[2].token_ids == [1, 2, 3]


def test_rollback_failure_preserves_original_cause():
    manager = _BlockManager(
        fail_mark_id=2,
        fail_rollback_ids=(1,),
    )

    with pytest.raises(
        NativeSpeculativeBatchError,
        match="mark 2 failed",
    ) as exc_info:
        execute_native_speculative_batch(
            block_manager=manager,
            seqs=(_Sequence(1), _Sequence(2)),
            draft_adapter=_Adapter({
                1: (10, 11),
                2: (20, 21),
            }),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10, 2: 20},
                [],
            ),
            run_tail_batch=_tail_callback(
                {1: (11,), 2: (21,)},
                [],
            ),
        )

    error = exc_info.value
    assert isinstance(error.cause, RuntimeError)
    assert "mark 2 failed" in str(error.cause)
    assert set(error.rollback_errors) == {1}
    assert "rollback 1 failed" in str(
        error.rollback_errors[1]
    )


def test_rejects_duplicate_sequence_ids_before_callbacks():
    calls = []

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        execute_native_speculative_batch(
            block_manager=_BlockManager(),
            seqs=(_Sequence(1), _Sequence(1)),
            draft_adapter=_Adapter({1: (10,)}),
            eos_token=99,
            run_first_targets=_first_target_callback(
                {1: 10},
                calls,
            ),
            run_tail_batch=_tail_callback({}, []),
        )

    assert calls == []
