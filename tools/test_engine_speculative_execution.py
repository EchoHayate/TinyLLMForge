from __future__ import annotations

import os
import ast
import sys
import types
from types import SimpleNamespace

import pytest
from contextlib import contextmanager


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_LLM_ENGINE_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "llm_engine.py",
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "engine")
]
speculative_package = types.ModuleType("tinyvllm.speculative")
speculative_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "speculative")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)
sys.modules.setdefault("tinyvllm.speculative", speculative_package)

import tinyvllm.engine.speculative_execution as speculative_execution
from tinyvllm.engine.engine_step_timeline import (
    EngineStepTimelineRecorder,
)
from tinyvllm.engine.speculative_execution import (
    build_engine_speculative_commit_rows,
    build_engine_speculative_partition,
)
from tinyvllm.engine.speculative_residency import (
    SpeculativeResidencyPrecommitRow,
    SpeculativeResidencyPrepareRow,
)
from tinyvllm.engine.speculative_selection import (
    SpeculativeSelectionConfig,
    build_speculative_selection_record,
)
from tinyvllm.speculative.adapter import DraftProposal
from tinyvllm.speculative.batch_runtime import (
    NativeSpeculativeBatchResult,
    NativeSpeculativeSequenceResult,
    PreparedNativeSpeculativeBatch,
    PreparedNativeSpeculativeSequence,
    TailBatchItem,
)
from tinyvllm.speculative.verifier import SpecVerifyPlan


class SpeculativeKVCommitRollbackError(RuntimeError):
    def __init__(self, commit_error, rollback_error):
        super().__init__(
            "speculative KV commit rollback failed: "
            f"{rollback_error}"
        )
        self.commit_error = commit_error
        self.rollback_error = rollback_error


class SchedulerPostprocessRollbackError(RuntimeError):
    def __init__(self, commit_error, rollback_error):
        super().__init__(
            "scheduler postprocess rollback failed: "
            f"{rollback_error}"
        )
        self.commit_error = commit_error
        self.rollback_error = rollback_error


def _load_engine_helper(name, namespace):
    tree = ast.parse(
        open(_LLM_ENGINE_PATH).read(),
        filename=_LLM_ENGINE_PATH,
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            _LLM_ENGINE_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def test_proposal_lifecycle_ack_requires_all_workers_and_records_history():
    helper = _load_engine_helper(
        "_call_speculative_proposal_lifecycle",
        {},
    )
    poisoned = []
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=4),
        speculative_proposal_lifecycle_ack_rows=[],
        call_model_runner_acknowledged=(
            lambda method_name, *args, timeout_s: (
                "ticket-1",
                tuple(
                    SimpleNamespace(
                        rank=rank,
                        result="ticket-1",
                    )
                    for rank in (1, 2, 3)
                ),
            )
        ),
        _poison_model_runner_ack_collector=poisoned.append,
    )

    assert helper(engine, "prepare", "row") == "ticket-1"
    assert engine.speculative_proposal_lifecycle_ack_rows == [{
        "method_name": "prepare",
        "worker_ranks": [1, 2, 3],
    }]
    assert poisoned == []

    engine.call_model_runner_acknowledged = (
        lambda method_name, *args, timeout_s: (
            None,
            (
                SimpleNamespace(rank=1, result=None),
                SimpleNamespace(rank=3, result=None),
            ),
        )
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        helper(engine, "release", 7)
    assert poisoned[-1].endswith("ranks are incomplete")


def _publication_helper(events, *, fail=None, finalize_rows=True):
    prepared = SimpleNamespace(
        state="prepared",
        side_state_callbacks=None,
        side_state_state="disabled",
        timing_ms={"commit_metadata_ms": 0.0},
    )
    transaction = SimpleNamespace(state="materialized")
    plan = SimpleNamespace(transaction=transaction)

    class Journal:
        def extend_speculative_kv_plans(self, scheduler, plans):
            assert scheduler is engine.scheduler
            assert plans == (plan,)
            events.append("scheduler_journal_extended")

    prepared_scheduler = SimpleNamespace(
        state="prepared",
        snapshot=Journal(),
    )
    engine = SimpleNamespace(
        model_runner=object(),
        scheduler=SimpleNamespace(
            block_manager=SimpleNamespace(
                commit_speculative_kv_commit_batch=(
                    lambda plans: (
                        events.append("target_kv_committed")
                        if fail != "target"
                        else (_ for _ in ()).throw(
                            RuntimeError("target failed")
                        )
                    )
                )
            ),
            commit_prepared_postprocess=(
                lambda scheduler: (
                    events.append("scheduler_committed")
                    if fail != "scheduler"
                    else (_ for _ in ()).throw(
                        RuntimeError("scheduler failed")
                    )
                )
            ),
        ),
        speculative_runtime_poisoned=False,
        speculative_runtime_poison_reason=None,
    )
    rows = (
        (
            SimpleNamespace(
                sequence_id=7,
                proposal_transaction_id="tx-7",
                accepted_proposal_tokens=2,
            ),
        )
        if finalize_rows
        else ()
    )

    def prepare(
        _model_runner,
        _descriptor,
        actual_rows,
        *,
        dispatch,
    ):
        assert callable(dispatch)
        assert actual_rows == rows
        assert actual_rows[0].accepted_proposal_tokens == 2
        events.append("proposal_finalize_prepared")
        return "ticket-7"

    def commit(
        _model_runner,
        _descriptor,
        ticket,
        *,
        dispatch,
    ):
        assert callable(dispatch)
        assert ticket == "ticket-7"
        if fail == "proposal_commit":
            raise RuntimeError("proposal commit failed")
        events.append("proposal_finalize_committed")

    def rollback(
        _model_runner,
        _descriptor,
        ticket,
        *,
        dispatch,
    ):
        assert callable(dispatch)
        assert ticket == "ticket-7"
        events.append("proposal_finalize_rolled_back")

    helper = _load_engine_helper(
        "_commit_prepared_speculative_publication",
        {
            "build_prepared_proposal_finalize_rows": (
                lambda actual: rows
            ),
            "prepare_model_runner_proposal_finalize_batch": (
                prepare
            ),
            "commit_model_runner_proposal_finalize_batch": (
                commit
            ),
            "rollback_model_runner_proposal_finalize_batch": (
                rollback
            ),
            "apply_prepared_speculative_side_state": (
                lambda prepared: None
            ),
            "seal_prepared_speculative_side_state": (
                lambda prepared: None
            ),
            "rollback_prepared_speculative_side_state": (
                lambda prepared: None
            ),
            "_call_speculative_proposal_lifecycle": (
                lambda engine, method_name, *args: (
                    engine.model_runner,
                    method_name,
                    args,
                )[0]
            ),
            "SpeculativeKVCommitRollbackError": (
                SpeculativeKVCommitRollbackError
            ),
            "SchedulerPostprocessRollbackError": (
                SchedulerPostprocessRollbackError
            ),
        },
    )
    runtime = SimpleNamespace(
        model_runner_executor=(
            SimpleNamespace(executor_id="fixture")
            if finalize_rows
            else None
        )
    )
    return (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    )


def test_two_phase_publication_orders_proposal_lifecycle_exactly():
    events = [
        "verify_complete",
        "target_commit_plans_prepared",
    ]
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events)
    )

    helper(
        engine,
        runtime,
        prepared,
        (plan,),
        prepared_scheduler,
    )

    assert events == [
        "verify_complete",
        "target_commit_plans_prepared",
        "proposal_finalize_prepared",
        "scheduler_journal_extended",
        "target_kv_committed",
        "scheduler_committed",
        "proposal_finalize_committed",
    ]


def test_publication_timeline_phases_wrap_existing_operation_order():
    events = []
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events)
    )

    class Timeline:
        enabled = True

        @contextmanager
        def phase(self, name):
            events.append(("phase_start", name))
            try:
                yield
            finally:
                events.append(("phase_end", name))

    engine.engine_step_timeline = Timeline()
    helper(
        engine,
        runtime,
        prepared,
        (plan,),
        prepared_scheduler,
    )

    assert events == [
        ("phase_start", "proposal_lifecycle_finalize_prepare"),
        "proposal_finalize_prepared",
        ("phase_end", "proposal_lifecycle_finalize_prepare"),
        "scheduler_journal_extended",
        ("phase_start", "proposal_kv_prepare_commit"),
        "target_kv_committed",
        ("phase_end", "proposal_kv_prepare_commit"),
        ("phase_start", "scheduler_commit_postprocess"),
        "scheduler_committed",
        ("phase_end", "scheduler_commit_postprocess"),
        ("phase_start", "proposal_lifecycle_finalize_commit"),
        "proposal_finalize_committed",
        ("phase_end", "proposal_lifecycle_finalize_commit"),
        ("phase_start", "side_state_seal"),
        ("phase_end", "side_state_seal"),
    ]


def test_disabled_recorder_publication_does_not_request_phase_contexts():
    events = []
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events)
    )
    recorder = EngineStepTimelineRecorder(enabled=False)
    phase_requests = []
    recorder_phase = recorder.phase

    def record_phase_request(name):
        phase_requests.append(name)
        return recorder_phase(name)

    recorder.phase = record_phase_request
    engine.engine_step_timeline = recorder

    helper(
        engine,
        runtime,
        prepared,
        (plan,),
        prepared_scheduler,
    )

    assert events == [
        "proposal_finalize_prepared",
        "scheduler_journal_extended",
        "target_kv_committed",
        "scheduler_committed",
        "proposal_finalize_committed",
    ]
    assert phase_requests == []


def test_publication_records_transactional_commit_timing():
    events = []
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events)
    )
    clock = iter((10.0, 10.007))

    helper(
        engine,
        runtime,
        prepared,
        (plan,),
        prepared_scheduler,
        clock=lambda: next(clock),
    )

    assert prepared.timing_ms["commit_metadata_ms"] == (
        pytest.approx(7.0)
    )
    assert prepared.state == "committed"
    assert engine.speculative_runtime_poisoned is False


@pytest.mark.parametrize("failure", ("target", "scheduler"))
def test_prepublication_failure_rolls_back_proposal_ticket(failure):
    events = []
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events, fail=failure)
    )

    with pytest.raises(RuntimeError, match="failed"):
        helper(
            engine,
            runtime,
            prepared,
            (plan,),
            prepared_scheduler,
        )

    assert events[-1] == "proposal_finalize_rolled_back"
    assert prepared.state == "prepared"
    assert engine.speculative_runtime_poisoned is False


def test_postpublication_finalize_failure_poisoned_without_retry():
    events = []
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events, fail="proposal_commit")
    )

    with pytest.raises(RuntimeError, match="proposal commit failed"):
        helper(
            engine,
            runtime,
            prepared,
            (plan,),
            prepared_scheduler,
        )

    assert prepared.state == "committed"
    assert engine.speculative_runtime_poisoned is True
    assert (
        "proposal finalization commit failed"
        in engine.speculative_runtime_poison_reason
    )
    assert events == [
        "proposal_finalize_prepared",
        "scheduler_journal_extended",
        "target_kv_committed",
        "scheduler_committed",
    ]


def test_host_proposals_make_zero_lifecycle_calls():
    events = []
    (
        helper,
        engine,
        runtime,
        prepared,
        plan,
        prepared_scheduler,
    ) = (
        _publication_helper(events, finalize_rows=False)
    )

    helper(
        engine,
        runtime,
        prepared,
        (plan,),
        prepared_scheduler,
    )

    assert events == [
        "scheduler_journal_extended",
        "target_kv_committed",
        "scheduler_committed",
    ]


class _Sequence:
    def __init__(
        self,
        sequence_id,
        *,
        num_tokens=8,
        completion_tokens=1,
        max_tokens=8,
        ignore_eos=False,
        temperature=0.0,
    ):
        self.seq_id = sequence_id
        self.num_tokens = num_tokens
        self.num_completion_tokens = completion_tokens
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.temperature = temperature
        self.step_is_decode = False
        self.step_do_sample = True


def _selection_record(seqs, generation=3):
    return build_speculative_selection_record(
        seqs=tuple(seqs),
        is_prefill=False,
        do_sample=True,
        batch_kind=None,
        policy_branch="decode",
        schedule_generation=generation,
        config=SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        ),
    )


def _runtime_row(
    sequence_id,
    *,
    proposal,
    first_target,
    target_tokens,
    greedy_count,
    accepted_tokens,
):
    return NativeSpeculativeSequenceResult(
        sequence_id=sequence_id,
        first_target_token=first_target,
        proposal=DraftProposal(
            sequence_id=sequence_id,
            token_ids=tuple(proposal),
            source_type="fixture",
        ),
        plan=None,
        target_tokens=tuple(target_tokens),
        greedy_accepted_count=greedy_count,
        accepted_tokens=tuple(accepted_tokens),
        eos_truncated=False,
        output_budget_truncated=False,
        reserved_blocks=(),
        proxy_block_table=(),
        committed_blocks=(),
        released_blocks=(),
    )


def _batch(*rows):
    return NativeSpeculativeBatchResult(
        sequences=tuple(rows),
        first_target_callback_count=1,
        tail_callback_count=0,
        timing_ms={},
    )


def test_prepared_commit_row_builder_is_exposed():
    assert callable(
        getattr(
            speculative_execution,
            "build_engine_prepared_speculative_commit_rows",
            None,
        )
    )


def test_residency_prepare_rows_preserve_tail_identity_order():
    item = TailBatchItem(
        sequence_id=7,
        plan=SpecVerifyPlan(
            input_tokens=(11, 12),
            positions=(3, 4),
            logical_slots=(3, 4),
            context_len=5,
            visible_block_count=2,
        ),
        proxy_block_table=(1, 2),
        original_block_identities=((1, 4),),
        reserved_block_identities=((2, 9),),
    )

    rows = (
        speculative_execution
        .build_speculative_residency_prepare_rows(
            (item,)
        )
    )

    assert rows == (
        SpeculativeResidencyPrepareRow(
            sequence_id=7,
            original_block_identities=((1, 4),),
            reserved_block_identities=((2, 9),),
            proxy_block_table=(1, 2),
            logical_slots=(3, 4),
        ),
    )


def test_residency_precommit_rows_use_transaction_generations():
    transaction = SimpleNamespace(
        reserved_block_ids=(2, 4, 6),
        reserved_block_generations=(9, 3, 8),
    )
    plan = SimpleNamespace(
        sequence_id=7,
        transaction=transaction,
        committed_block_ids=(2, 4),
        unused_block_ids=(6,),
        materialized_end=11,
    )

    rows = (
        speculative_execution
        .build_speculative_residency_precommit_rows(
            (plan,)
        )
    )

    assert rows == (
        SpeculativeResidencyPrecommitRow(
            sequence_id=7,
            committed_block_identities=((2, 9), (4, 3)),
            rejected_block_identities=((6, 8),),
            accepted_materialized_end=11,
        ),
    )


def test_prepared_commit_rows_use_unmutated_sequence_snapshot():
    sequence = _Sequence(
        5,
        completion_tokens=1,
        max_tokens=5,
    )
    proposal = DraftProposal(
        sequence_id=5,
        token_ids=(11, 12, 13),
        source_type="fixture",
    )
    prepared = PreparedNativeSpeculativeBatch(
        sequences=(
            PreparedNativeSpeculativeSequence(
                sequence_id=5,
                sequence=sequence,
                first_target_token=11,
                proposal=proposal,
                plan=None,
                target_tokens=(11, 12, 99),
                greedy_accepted_count=2,
                accepted_tokens=(11, 12),
                eos_truncated=False,
                output_budget_truncated=False,
                transaction=None,
                reserved_blocks=(),
                proxy_block_table=(),
            ),
        ),
        first_target_callback_count=1,
        tail_callback_count=1,
        timing_ms={},
    )

    rows = (
        speculative_execution
        .build_engine_prepared_speculative_commit_rows(
            prepared,
            (sequence,),
            eos_token=-1,
        )
    )

    assert rows[0].sequence_id == 5
    assert rows[0].output_tokens == (11, 12, 99)
    assert rows[0].accepted_draft_tokens == (11, 12)
    assert sequence.num_completion_tokens == 1
    assert prepared.state == "prepared"


def test_partition_preserves_selected_and_suppressed_order():
    seqs = (
        _Sequence(8),
        _Sequence(4, completion_tokens=7, max_tokens=8),
        _Sequence(2),
    )
    record = _selection_record(seqs)

    partition = build_engine_speculative_partition(
        record,
        seqs,
        expected_schedule_generation=3,
    )

    assert partition.scheduled_sequence_ids == (8, 4, 2)
    assert partition.selected_sequence_ids == (8, 2)
    assert partition.suppressed_sequence_ids == (4,)
    assert partition.selected_sequences == (seqs[0], seqs[2])
    assert partition.suppressed_sequences == (seqs[1],)


def test_partition_delegates_stale_snapshot_validation():
    seqs = (_Sequence(1),)
    record = _selection_record(seqs)
    seqs[0].num_tokens += 1

    with pytest.raises(ValueError, match="token"):
        build_engine_speculative_partition(
            record,
            seqs,
            expected_schedule_generation=3,
        )


@pytest.mark.parametrize(
    "row,sequence,expected,accepted,fallback,eos,budget",
    [
        (
            _runtime_row(
                1,
                proposal=(),
                first_target=9,
                target_tokens=(9,),
                greedy_count=0,
                accepted_tokens=(),
            ),
            _Sequence(1),
            (9,),
            (),
            9,
            False,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1, 2),
                first_target=9,
                target_tokens=(9, 8),
                greedy_count=0,
                accepted_tokens=(),
            ),
            _Sequence(1),
            (9,),
            (),
            9,
            False,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1,),
                first_target=1,
                target_tokens=(1,),
                greedy_count=1,
                accepted_tokens=(1,),
            ),
            _Sequence(1),
            (1,),
            (1,),
            None,
            False,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1, 2, 3),
                first_target=1,
                target_tokens=(1, 9, 8),
                greedy_count=1,
                accepted_tokens=(1,),
            ),
            _Sequence(1),
            (1, 9),
            (1,),
            9,
            False,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1, 2, 3),
                first_target=1,
                target_tokens=(1, 2, 3),
                greedy_count=3,
                accepted_tokens=(1, 2, 3),
            ),
            _Sequence(1),
            (1, 2, 3),
            (1, 2, 3),
            None,
            False,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1, 2, 3),
                first_target=1,
                target_tokens=(1, 2, 9),
                greedy_count=2,
                accepted_tokens=(1, 2),
            ),
            _Sequence(1),
            (1, 2),
            (1, 2),
            None,
            True,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1, 3),
                first_target=1,
                target_tokens=(1, 2),
                greedy_count=1,
                accepted_tokens=(1,),
            ),
            _Sequence(1),
            (1, 2),
            (1,),
            2,
            True,
            False,
        ),
        (
            _runtime_row(
                1,
                proposal=(1, 3),
                first_target=1,
                target_tokens=(1, 9),
                greedy_count=1,
                accepted_tokens=(1,),
            ),
            _Sequence(
                1,
                completion_tokens=2,
                max_tokens=3,
            ),
            (1,),
            (1,),
            None,
            False,
            True,
        ),
    ],
)
def test_builds_exact_output_commit_semantics(
    row,
    sequence,
    expected,
    accepted,
    fallback,
    eos,
    budget,
):
    commit = build_engine_speculative_commit_rows(
        _batch(row),
        (sequence,),
        eos_token=2,
    )[0]

    assert commit.output_tokens == expected
    assert commit.accepted_draft_tokens == accepted
    assert commit.fallback_target_token == fallback
    assert commit.finished_by_eos is eos
    assert commit.finished_by_output_budget is budget


def test_ignore_eos_allows_fallback_after_accepted_eos():
    row = _runtime_row(
        1,
        proposal=(1, 2, 3),
        first_target=1,
        target_tokens=(1, 2, 9),
        greedy_count=2,
        accepted_tokens=(1, 2),
    )

    commit = build_engine_speculative_commit_rows(
        _batch(row),
        (_Sequence(1, ignore_eos=True),),
        eos_token=2,
    )[0]

    assert commit.output_tokens == (1, 2, 9)
    assert commit.fallback_target_token == 9
    assert commit.finished_by_eos is False


def test_commit_rows_preserve_result_order():
    seqs = (_Sequence(8), _Sequence(4))
    result = _batch(
        _runtime_row(
            8,
            proposal=(),
            first_target=80,
            target_tokens=(80,),
            greedy_count=0,
            accepted_tokens=(),
        ),
        _runtime_row(
            4,
            proposal=(),
            first_target=40,
            target_tokens=(40,),
            greedy_count=0,
            accepted_tokens=(),
        ),
    )

    rows = build_engine_speculative_commit_rows(
        result,
        seqs,
        eos_token=-1,
    )

    assert tuple(row.sequence_id for row in rows) == (8, 4)
    assert tuple(row.output_tokens for row in rows) == (
        (80,),
        (40,),
    )


@pytest.mark.parametrize(
    "result,seqs,match",
    [
        (
            _batch(
                _runtime_row(
                    2,
                    proposal=(),
                    first_target=9,
                    target_tokens=(9,),
                    greedy_count=0,
                    accepted_tokens=(),
                ),
            ),
            (_Sequence(1),),
            "order",
        ),
        (
            _batch(
                _runtime_row(
                    1,
                    proposal=(1, 2),
                    first_target=1,
                    target_tokens=(1, 2),
                    greedy_count=1,
                    accepted_tokens=(9,),
                ),
            ),
            (_Sequence(1),),
            "prefix",
        ),
        (
            _batch(
                _runtime_row(
                    1,
                    proposal=(1, 2),
                    first_target=1,
                    target_tokens=(1, 2),
                    greedy_count=2,
                    accepted_tokens=(1,),
                ),
            ),
            (_Sequence(1),),
            "accepted",
        ),
        (
            _batch(
                _runtime_row(
                    1,
                    proposal=(1,),
                    first_target=1,
                    target_tokens=(1,),
                    greedy_count=2,
                    accepted_tokens=(1,),
                ),
            ),
            (_Sequence(1),),
            "greedy",
        ),
        (
            _batch(
                _runtime_row(
                    1,
                    proposal=(1, 2),
                    first_target=1,
                    target_tokens=(1,),
                    greedy_count=1,
                    accepted_tokens=(1,),
                ),
            ),
            (_Sequence(1),),
            "target",
        ),
    ],
)
def test_commit_rows_reject_invalid_runtime_results(
    result,
    seqs,
    match,
):
    with pytest.raises(ValueError, match=match):
        build_engine_speculative_commit_rows(
            result,
            seqs,
            eos_token=-1,
        )
