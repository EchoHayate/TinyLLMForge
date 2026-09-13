from __future__ import annotations

import ast
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT / "tinyvllm" / "engine" / "slo_cohort_burst.py"
)
POLICY_SPEC = importlib.util.spec_from_file_location(
    "scheduler_slo_cohort_policy_under_test",
    POLICY_PATH,
)
policy = importlib.util.module_from_spec(POLICY_SPEC)
sys.modules[POLICY_SPEC.name] = policy
POLICY_SPEC.loader.exec_module(policy)

CONTRACT_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_cohort_burst.py"
)
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "scheduler_exact_greedy_cohort_contract_under_test",
    CONTRACT_PATH,
)
contract = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = contract
CONTRACT_SPEC.loader.exec_module(contract)


@dataclass(frozen=True)
class _ScheduledOutputRow:
    sequence_id: int
    output_tokens: tuple[int, ...]
    speculative: bool
    accepted_draft_tokens: tuple[int, ...] = ()
    exact_burst: bool = False
    exact_burst_gate_only: bool = False
    exact_burst_phase: str | None = None
    exact_cohort_burst: bool = False


def _load_scheduler_method(name: str):
    path = REPO_ROOT / "tinyvllm" / "engine" / "scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scheduler_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        (
            node
            for node in scheduler_node.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    assert method is not None, f"Scheduler.{name} is missing"
    method = deepcopy(method)
    method.decorator_list = []
    method.returns = None
    for argument in (
        method.args.posonlyargs
        + method.args.args
        + method.args.kwonlyargs
    ):
        argument.annotation = None
    namespace = {
        "replace": replace,
        "RequestSLOState": policy.RequestSLOState,
        "ProtectedRequestSnapshot": policy.ProtectedRequestSnapshot,
        "SLOCohortBurstObservation": policy.SLOCohortBurstObservation,
        "SLOCohortBurstDecision": policy.SLOCohortBurstDecision,
        "build_slo_cohort_decision_telemetry": (
            policy.build_slo_cohort_decision_telemetry
        ),
        "CohortWriteAuthority": contract.CohortWriteAuthority,
        "ExactGreedyCohortBurstFallback": (
            contract.ExactGreedyCohortBurstFallback
        ),
        "ExactGreedyCohortBurstLease": (
            contract.ExactGreedyCohortBurstLease
        ),
        "ExactGreedyCohortBurstResult": (
            contract.ExactGreedyCohortBurstResult
        ),
        "ExactGreedyCohortBurstTransaction": (
            contract.ExactGreedyCohortBurstTransaction
        ),
        "build_exact_greedy_cohort_burst_lease": (
            contract.build_exact_greedy_cohort_burst_lease
        ),
        "validate_exact_greedy_cohort_burst_result": (
            contract.validate_exact_greedy_cohort_burst_result
        ),
        "ScheduledOutputRow": _ScheduledOutputRow,
        "SequenceStatus": SimpleNamespace(RUNNING="running"),
        "INT64_MAX": (1 << 63) - 1,
    }
    module = ast.Module(body=[method], type_ignores=[])
    code = compile(
        ast.fix_missing_locations(module),
        str(path),
        "exec",
    )
    exec(code, namespace)
    return namespace[name]


class FakeScheduler:
    register_slo_request = _load_scheduler_method(
        "register_slo_request"
    )
    record_slo_prefill_start = _load_scheduler_method(
        "record_slo_prefill_start"
    )
    record_slo_prefill_completion = _load_scheduler_method(
        "record_slo_prefill_completion"
    )
    record_slo_publication = _load_scheduler_method(
        "record_slo_publication"
    )
    remove_slo_request = _load_scheduler_method(
        "remove_slo_request"
    )
    slo_cohort_telemetry_snapshot = _load_scheduler_method(
        "slo_cohort_telemetry_snapshot"
    )
    build_slo_cohort_observation = _load_scheduler_method(
        "build_slo_cohort_observation"
    )
    _record_decode_progress = _load_scheduler_method(
        "_record_decode_progress"
    )
    _remove_finished_progress = _load_scheduler_method(
        "_remove_finished_progress"
    )
    add = _load_scheduler_method("add")
    _clear_exact_greedy_cohort_burst = _load_scheduler_method(
        "_clear_exact_greedy_cohort_burst"
    )
    _advance_exact_greedy_cohort_sequence_generations = (
        _load_scheduler_method(
            "_advance_exact_greedy_cohort_sequence_generations"
        )
    )
    _validate_pending_exact_greedy_cohort_burst = (
        _load_scheduler_method(
            "_validate_pending_exact_greedy_cohort_burst"
        )
    )
    prepare_exact_greedy_cohort_burst = _load_scheduler_method(
        "prepare_exact_greedy_cohort_burst"
    )
    cancel_exact_greedy_cohort_burst = _load_scheduler_method(
        "cancel_exact_greedy_cohort_burst"
    )
    fail_exact_greedy_cohort_burst = _load_scheduler_method(
        "fail_exact_greedy_cohort_burst"
    )
    prepare_exact_greedy_cohort_burst_commit = (
        _load_scheduler_method(
            "prepare_exact_greedy_cohort_burst_commit"
        )
    )

    def __init__(self):
        self.slo_request_state_by_seq_id = {}
        self.completed_slo_request_state_by_seq_id = {}
        self.slo_clock_invalid = False
        self.slo_clock_invalid_reason = None
        self.exact_greedy_cohort_burst = True
        self.exact_greedy_cohort_burst_widths = (1, 2, 4, 8)
        self.exact_greedy_cohort_burst_target_itl_ns = 100
        self.exact_greedy_cohort_burst_target_ttft_ns = 100
        self.exact_greedy_cohort_burst_reserve_ns = 10
        self._exact_greedy_cohort_burst_pending_lease = None
        self._exact_greedy_cohort_burst_pending_transaction = None
        self._exact_greedy_cohort_burst_sequence_generations = {}
        self.schedule_generation = 11
        self.eos = 2
        self._slo_cohort_cost_table = SimpleNamespace(
            table_sha256="b" * 64
        )
        self._last_slo_cohort_decision_telemetry = None
        self.chunked_prefill_slo_mixed = False
        self.decode_progress_ns_by_seq_id = {}
        self.block_manager = _BlockManager()
        self.running = deque()
        self.waiting = deque()
        self.prefilling = deque()
        self.prepared_calls = []

    def _invalidate_slo_clock(self, reason: str) -> None:
        if not self.slo_clock_invalid:
            self.slo_clock_invalid = True
            self.slo_clock_invalid_reason = reason

    def _validate_admission(self, seq) -> None:
        del seq

    def prepare_postprocess(self, seqs, rows, **kwargs):
        prepared = SimpleNamespace(
            scheduled_sequence_ids=tuple(
                seq.seq_id for seq in seqs
            ),
            rows=tuple(rows),
            **kwargs,
        )
        self.prepared_calls.append(prepared)
        return prepared


class _BlockManager:
    block_size = 8

    def __init__(self):
        self.blocks = [
            SimpleNamespace(generation=100 + index)
            for index in range(64)
        ]

    def block_identities(self, block_ids):
        return tuple(
            (block_id, self.blocks[block_id].generation)
            for block_id in block_ids
        )

    def validate_block_identities(self, identities):
        if self.block_identities(
            tuple(block_id for block_id, _ in identities)
        ) != identities:
            raise RuntimeError("block identity is stale")


class FakeSequence:
    def __init__(
        self,
        sequence_id: int,
        max_tokens: int = 16,
        *,
        num_tokens: int = 5,
        block_id: int | None = None,
    ):
        self.seq_id = sequence_id
        self.num_tokens = num_tokens
        self.num_prompt_tokens = 4
        self.max_tokens = max_tokens
        self.block_table = [
            sequence_id if block_id is None else block_id
        ]
        self.status = "running"
        self.ignore_eos = False

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens


def _scheduler():
    return FakeScheduler()


def _sequence(sequence_id: int, *, max_tokens: int = 16):
    return FakeSequence(sequence_id, max_tokens)


def _decision(width: int = 8):
    return policy.SLOCohortBurstDecision(
        selected_width=width,
        reason="selected",
        global_slack_ns=100,
        predicted_cost_ns_by_width=(
            (8, 80),
            (4, 40),
            (2, 20),
        ),
        protected_sequence_ids=(7, 9, 11, 13),
    )


def _graph_capability():
    return {
        "available": True,
        "quarantined": False,
        "shape_supported": True,
        "graph_identity_sha256": "a" * 64,
        "graph_generation": 7,
    }


def _cohort_result(lease, tokens):
    rows = tuple(
        contract.ExactGreedyCohortBurstRowResult(
            sequence_id=authority.sequence_id,
            sequence_generation=authority.sequence_generation,
            tokens=tuple(row_tokens),
            final_position=(
                authority.first_write_position
                + lease.authorized_width
            ),
            final_context_length=(
                authority.initial_sequence_length
                + lease.authorized_width
            ),
            final_physical_slot=(
                authority.last_physical_slot + 1
            ),
        )
        for authority, row_tokens in zip(lease.rows, tokens)
    )
    return contract.ExactGreedyCohortBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        graph_identity_sha256=lease.graph_identity_sha256,
        graph_generation=lease.graph_generation,
        replay_count=lease.authorized_width,
        rows=rows,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=0,
    )


def test_slo_request_lifecycle_uses_immutable_replacement() -> None:
    scheduler = _scheduler()
    seq = _sequence(7)

    state0 = scheduler.register_slo_request(
        seq,
        arrival_ns=10,
        service_class="default",
    )
    assert scheduler.slo_request_state_by_seq_id == {7: state0}

    state1 = scheduler.record_slo_publication(7, visible_ns=20)
    assert state1 is not state0
    assert state1.first_token_visible_ns == 20
    assert state1.last_token_visible_ns == 20

    state2 = scheduler.record_slo_publication(7, visible_ns=30)
    assert state2 is not state1
    assert state2.first_token_visible_ns == 20
    assert state2.last_token_visible_ns == 30

    scheduler.remove_slo_request(7)
    assert 7 not in scheduler.slo_request_state_by_seq_id


def test_prefill_timeline_uses_immutable_monotonic_replacement() -> None:
    scheduler = _scheduler()
    seq = _sequence(7)
    state0 = scheduler.register_slo_request(
        seq,
        arrival_ns=10,
        service_class="default",
    )

    state1 = scheduler.record_slo_prefill_start(7, start_ns=12)
    assert (
        scheduler.record_slo_prefill_start(7, start_ns=15)
        is state1
    )
    state2 = scheduler.record_slo_prefill_completion(
        7,
        complete_ns=18,
    )

    assert state1 is not state0
    assert state2 is not state1
    assert state2.prefill_start_ns == 12
    assert state2.prefill_complete_ns == 18
    with pytest.raises(ValueError, match="precedes"):
        scheduler.record_slo_prefill_completion(
            7,
            complete_ns=11,
        )


def test_add_requires_engine_clock_and_registers_before_enqueue() -> None:
    scheduler = _scheduler()
    seq = _sequence(5)
    with pytest.raises(ValueError, match="requires arrival_ns"):
        scheduler.add(seq)
    assert not scheduler.waiting

    scheduler.add(seq, arrival_ns=10, service_class="interactive")
    assert tuple(scheduler.waiting) == (seq,)
    state = scheduler.slo_request_state_by_seq_id[5]
    assert state.arrival_ns == 10
    assert state.service_class == "interactive"


def test_engine_defaults_to_its_clock_but_accepts_frozen_arrival() -> None:
    source = (
        REPO_ROOT / "tinyvllm" / "engine" / "llm_engine.py"
    ).read_text(encoding="utf-8")
    assert (
        "arrival_ns: int | None = None"
    ) in source
    assert (
        "arrival_ns=(\n"
        "                    self._clock_ns()\n"
        "                    if arrival_ns is None\n"
        "                    else arrival_ns\n"
        "                )"
    ) in source


def test_slo_clock_rollback_is_sticky_and_does_not_mutate_state() -> None:
    scheduler = _scheduler()
    seq = _sequence(8)
    scheduler.register_slo_request(
        seq,
        arrival_ns=10,
        service_class="default",
    )
    state = scheduler.record_slo_publication(8, visible_ns=20)

    with pytest.raises(ValueError, match="regressed"):
        scheduler.record_slo_publication(8, visible_ns=19)

    assert scheduler.slo_request_state_by_seq_id[8] == state
    assert scheduler.slo_clock_invalid is True
    assert scheduler.slo_clock_invalid_reason == (
        "cohort_publication_clock_regressed"
    )


def test_build_observation_protects_every_queue_without_reordering() -> None:
    scheduler = _scheduler()
    cohort = (_sequence(7), _sequence(9))
    omitted = _sequence(11)
    waiting = _sequence(13)
    prefilling = _sequence(15)
    scheduler.running.extend((*cohort, omitted))
    scheduler.waiting.append(waiting)
    scheduler.prefilling.append(prefilling)

    for offset, seq in enumerate(
        (*cohort, omitted, waiting, prefilling)
    ):
        scheduler.register_slo_request(
            seq,
            arrival_ns=offset,
            service_class="default",
        )
    for seq in (*cohort, omitted):
        scheduler.record_slo_publication(
            seq.seq_id,
            visible_ns=10,
        )

    observation = scheduler.build_slo_cohort_observation(
        cohort,
        decision_now_ns=20,
        graph_capability={
            "available": True,
            "quarantined": False,
            "shape_supported": True,
        },
        all_greedy=True,
        mixed_mode_unsupported=False,
    )

    assert tuple(row.sequence_id for row in observation.cohort) == (7, 9)
    assert tuple(
        row.sequence_id
        for row in observation.omitted_runnable_decode
    ) == (11,)
    assert tuple(row.sequence_id for row in observation.waiting) == (13,)
    assert tuple(
        row.sequence_id for row in observation.incomplete_prefill
    ) == (15,)


def test_missing_slo_state_is_preserved_for_fail_closed_selection() -> None:
    scheduler = _scheduler()
    cohort = (_sequence(7),)
    missing = _sequence(13)
    scheduler.running.append(cohort[0])
    scheduler.waiting.append(missing)
    scheduler.register_slo_request(
        cohort[0],
        arrival_ns=0,
        service_class="default",
    )
    scheduler.record_slo_publication(7, visible_ns=10)

    observation = scheduler.build_slo_cohort_observation(
        cohort,
        decision_now_ns=20,
        graph_capability={
            "available": True,
            "quarantined": False,
            "shape_supported": True,
        },
        all_greedy=True,
        mixed_mode_unsupported=False,
    )
    assert observation.waiting[0].slo_state is None


def test_decode_publication_and_terminal_cleanup_update_slo_state() -> None:
    scheduler = _scheduler()
    seq = _sequence(21)
    scheduler.register_slo_request(
        seq,
        arrival_ns=10,
        service_class="default",
    )

    scheduler._record_decode_progress(
        seq,
        step_end_ns=20,
        progress_updates={},
    )
    state = scheduler.slo_request_state_by_seq_id[21]
    assert state.first_token_visible_ns == 20
    assert state.last_token_visible_ns == 20

    removed = []
    scheduler._remove_finished_progress(seq, removed)
    assert 21 not in scheduler.slo_request_state_by_seq_id
    assert (
        scheduler.completed_slo_request_state_by_seq_id[21][1]
        == "length"
    )
    request_row = scheduler.slo_cohort_telemetry_snapshot()[
        "requests"
    ][0]
    assert (
        request_row["schema_version"]
        == "slo-cohort-burst.request.v1"
    )
    assert request_row["completion_ns"] == 20
    assert request_row["terminal_reason"] == "length"
    drained = scheduler.slo_cohort_telemetry_snapshot(
        drain_completed=True,
    )
    assert len(drained["requests"]) == 1
    assert scheduler.slo_cohort_telemetry_snapshot()["requests"] == []


def test_draining_request_rows_also_consumes_the_decision_row() -> None:
    scheduler = _scheduler()
    scheduler._last_slo_cohort_decision_telemetry = SimpleNamespace(
        to_payload=lambda: {"selected_width": 4},
    )

    drained = scheduler.slo_cohort_telemetry_snapshot(
        drain_completed=True,
    )

    assert drained["decision"] == {"selected_width": 4}
    assert (
        scheduler.slo_cohort_telemetry_snapshot()["decision"]
        is None
    )


def test_cohort_publication_records_each_token_at_one_visible_time() -> None:
    scheduler = _scheduler()
    seq = _sequence(21)
    scheduler.register_slo_request(
        seq,
        arrival_ns=10,
        service_class="default",
    )

    scheduler._record_decode_progress(
        seq,
        step_end_ns=20,
        progress_updates={},
        visible_token_ids=(31, 32, 33, 34),
    )

    state = scheduler.slo_request_state_by_seq_id[21]
    assert state.host_visible_token_timestamps_ns == (20, 20, 20, 20)
    assert state.output_token_ids == (31, 32, 33, 34)


def test_postprocess_journals_snapshot_cohort_slo_state() -> None:
    source = (
        REPO_ROOT / "tinyvllm" / "engine" / "scheduler.py"
    ).read_text(encoding="utf-8")
    assert source.count(
        "slo_request_states=dict(\n"
        "                scheduler.slo_request_state_by_seq_id"
    ) == 2
    assert source.count(
        "scheduler.slo_request_state_by_seq_id.update(\n"
        "                self.slo_request_states"
    ) == 2


def test_cohort_lease_preserves_order_and_clips_width_to_shared_capacity():
    scheduler = _scheduler()
    cohort = tuple(
        FakeSequence(sequence_id, max_tokens=16)
        for sequence_id in (7, 9, 11, 13)
    )

    lease = scheduler.prepare_exact_greedy_cohort_burst(
        cohort,
        _decision(8),
        schedule_generation=11,
        decision_now_ns=100,
        graph_capability=_graph_capability(),
    )

    assert lease.ordered_sequence_ids == (7, 9, 11, 13)
    assert lease.requested_width == 8
    assert lease.authorized_width == 4
    assert tuple(
        row.sequence_id for row in lease.rows
    ) == (7, 9, 11, 13)
    assert tuple(
        row.block_table_identity for row in lease.rows
    ) == (
        ((7, 107),),
        ((9, 109),),
        ((11, 111),),
        ((13, 113),),
    )
    assert tuple(
        (
            row.first_physical_slot,
            row.last_physical_slot,
        )
        for row in lease.rows
    ) == (
        (60, 63),
        (76, 79),
        (92, 95),
        (108, 111),
    )


@pytest.mark.parametrize(
    ("remaining", "writable", "expected"),
    (
        (8, 8, 8),
        (7, 8, 4),
        (4, 7, 4),
        (3, 8, 2),
        (8, 3, 2),
        (1, 8, 1),
        (8, 1, 1),
    ),
)
def test_cohort_lease_clips_width_over_supported_ladder(
    remaining,
    writable,
    expected,
):
    scheduler = _scheduler()
    num_tokens = scheduler.block_manager.block_size - writable + 1
    sequence = FakeSequence(
        7,
        max_tokens=remaining,
        num_tokens=num_tokens,
    )
    sequence.num_prompt_tokens = num_tokens

    lease = scheduler.prepare_exact_greedy_cohort_burst(
        (sequence,),
        _decision(8),
        schedule_generation=11,
        decision_now_ns=100,
        graph_capability=_graph_capability(),
    )

    if expected == 1:
        assert lease is None
        assert (
            scheduler._exact_greedy_cohort_burst_pending_transaction
            is None
        )
    else:
        assert lease.authorized_width == expected


def test_cohort_scheduler_allows_only_one_pending_transaction():
    scheduler = _scheduler()
    cohort = (FakeSequence(7), FakeSequence(9))
    lease = scheduler.prepare_exact_greedy_cohort_burst(
        cohort,
        _decision(4),
        schedule_generation=11,
        decision_now_ns=100,
        graph_capability=_graph_capability(),
    )

    with pytest.raises(RuntimeError, match="pending"):
        scheduler.prepare_exact_greedy_cohort_burst(
            cohort,
            _decision(4),
            schedule_generation=11,
            decision_now_ns=100,
            graph_capability=_graph_capability(),
        )

    scheduler.cancel_exact_greedy_cohort_burst(
        lease,
        "pre_replay_bind_failure",
    )
    assert scheduler._exact_greedy_cohort_burst_pending_lease is None
    assert (
        scheduler._exact_greedy_cohort_burst_pending_transaction
        is None
    )


def test_cohort_lease_rejects_exhausted_generation_before_reservation():
    scheduler = _scheduler()
    cohort = (FakeSequence(7), FakeSequence(9))
    scheduler._exact_greedy_cohort_burst_sequence_generations[9] = (
        (1 << 63) - 1
    )

    with pytest.raises(
        OverflowError,
        match="cohort sequence generation exhausted",
    ):
        scheduler.prepare_exact_greedy_cohort_burst(
            cohort,
            _decision(4),
            schedule_generation=11,
            decision_now_ns=100,
            graph_capability=_graph_capability(),
        )

    assert (
        scheduler._exact_greedy_cohort_burst_pending_lease is None
    )
    assert (
        scheduler._exact_greedy_cohort_burst_pending_transaction
        is None
    )
    assert 7 not in (
        scheduler._exact_greedy_cohort_burst_sequence_generations
    )


def test_terminal_cohort_failure_closes_pending_transaction():
    scheduler = _scheduler()
    cohort = (FakeSequence(7), FakeSequence(9))
    lease = scheduler.prepare_exact_greedy_cohort_burst(
        cohort,
        _decision(4),
        schedule_generation=11,
        decision_now_ns=100,
        graph_capability=_graph_capability(),
    )

    scheduler.fail_exact_greedy_cohort_burst(
        lease,
        terminal=True,
        reason="graph replay failed",
        completed_replays=2,
    )

    assert scheduler._exact_greedy_cohort_burst_pending_lease is None
    assert (
        scheduler._exact_greedy_cohort_burst_pending_transaction
        is None
    )


def test_prepare_cohort_commit_validates_every_row_and_truncates_at_eos():
    scheduler = _scheduler()
    cohort = (FakeSequence(7), FakeSequence(9))
    lease = scheduler.prepare_exact_greedy_cohort_burst(
        cohort,
        _decision(4),
        schedule_generation=11,
        decision_now_ns=100,
        graph_capability=_graph_capability(),
    )
    result = _cohort_result(
        lease,
        (
            (31, 2, 91, 92),
            (41, 42, 43, 44),
        ),
    )

    prepared = scheduler.prepare_exact_greedy_cohort_burst_commit(
        cohort,
        lease,
        result,
        decision_now_ns=100,
        step_end_ns=140,
    )

    assert len(scheduler.prepared_calls) == 1
    assert tuple(
        row.sequence_id for row in prepared.rows
    ) == (7, 9)
    assert tuple(
        row.output_tokens for row in prepared.rows
    ) == ((31, 2), (41, 42, 43, 44))
    assert all(row.exact_cohort_burst for row in prepared.rows)
    assert prepared.exact_cohort_burst_lease is lease
    assert prepared.exact_cohort_burst_result is result
    assert (
        prepared.exact_cohort_burst_transaction.state
        == "reserved"
    )
