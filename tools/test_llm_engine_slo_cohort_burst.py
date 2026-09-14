"""Dependency-light LLMEngine SLO cohort burst integration tests."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm" / "engine" / "llm_engine.py"
CONTRACT_PATH = (
    ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_cohort_burst.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "llm_engine_slo_cohort_contract_under_test",
    CONTRACT_PATH,
)
ExactGreedyCohortBurstFallback = (
    contract.ExactGreedyCohortBurstFallback
)
CohortWriteAuthority = contract.CohortWriteAuthority
ExactGreedyCohortBurstLease = (
    contract.ExactGreedyCohortBurstLease
)
ExactGreedyCohortBurstResult = (
    contract.ExactGreedyCohortBurstResult
)
ExactGreedyCohortBurstTerminalError = (
    contract.ExactGreedyCohortBurstTerminalError
)
build_exact_greedy_cohort_burst_execution_telemetry = (
    contract.build_exact_greedy_cohort_burst_execution_telemetry
)
build_terminal_exact_greedy_cohort_burst_execution_telemetry = (
    contract
    .build_terminal_exact_greedy_cohort_burst_execution_telemetry
)
build_exact_greedy_cohort_burst_lease = (
    contract.build_exact_greedy_cohort_burst_lease
)


def _load_engine_method(name: str):
    tree = ast.parse(
        ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(ENGINE_PATH),
    )
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method = next(
        (
            node
            for node in engine_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == name
        ),
        None,
    )
    assert method is not None, f"LLMEngine.{name} is missing"
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=None,
        type_comment=method.type_comment,
    )
    namespace = {
        "hashlib": hashlib,
        "ExactGreedyCohortBurstFallback": (
            ExactGreedyCohortBurstFallback
        ),
        "ExactGreedyCohortBurstLease": (
            ExactGreedyCohortBurstLease
        ),
        "ExactGreedyCohortBurstTerminalError": (
            ExactGreedyCohortBurstTerminalError
        ),
        "ExactGreedyCohortBurstResult": (
            ExactGreedyCohortBurstResult
        ),
        "build_exact_greedy_cohort_burst_execution_telemetry": (
            build_exact_greedy_cohort_burst_execution_telemetry
        ),
        "build_terminal_exact_greedy_cohort_burst_execution_telemetry": (
            build_terminal_exact_greedy_cohort_burst_execution_telemetry
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class _Sequence:
    def __init__(self, sequence_id: int):
        self.seq_id = sequence_id
        self.block_table = [sequence_id]
        self.temperature = 0.0
        self.token_ids = [1, 2]


class _Scheduler:
    schedule_generation = 11

    def __init__(self, *, decision_width=4):
        self.decision = SimpleNamespace(
            selected_width=decision_width,
        )
        self.pending_cohort_lease = None
        self.commit_count = 0
        self.cancel_count = 0
        self.fail_count = 0
        self.fail_kwargs = None
        self.events = []

    def select_slo_cohort_burst(self, seqs, **kwargs):
        self.events.append(("select", tuple(seq.seq_id for seq in seqs)))
        return self.decision

    def prepare_exact_greedy_cohort_burst(
        self,
        seqs,
        decision,
        **kwargs,
    ):
        self.events.append(
            ("prepare", tuple(seq.seq_id for seq in seqs))
        )
        width = decision.selected_width
        rows = tuple(
            CohortWriteAuthority(
                sequence_id=seq.seq_id,
                sequence_generation=3,
                block_table_identity=((seq.seq_id, 5),),
                writable_block_identities=((seq.seq_id, 5),),
                first_write_position=len(seq.token_ids) - 1,
                last_write_position=(
                    len(seq.token_ids) + width - 2
                ),
                first_physical_slot=seq.seq_id * 256,
                last_physical_slot=seq.seq_id * 256 + width - 1,
                initial_completion_count=0,
                initial_sequence_length=len(seq.token_ids),
                remaining_output_tokens=width,
            )
            for seq in seqs
        )
        self.pending_cohort_lease = build_exact_greedy_cohort_burst_lease(
            schedule_generation=kwargs["schedule_generation"],
            graph_generation=7,
            graph_identity_sha256="a" * 64,
            requested_width=width,
            authorized_width=decision.selected_width,
            decision_now_ns=kwargs["decision_now_ns"],
            cost_table_sha256="b" * 64,
            predicted_duration_ns=20,
            global_slack_ns=30,
            rows=rows,
        )
        return self.pending_cohort_lease

    def cancel_exact_greedy_cohort_burst(self, lease, reason):
        assert lease is self.pending_cohort_lease
        self.events.append(("cancel", reason))
        self.cancel_count += 1
        self.pending_cohort_lease = None

    def fail_exact_greedy_cohort_burst(self, lease, **kwargs):
        assert lease is self.pending_cohort_lease
        self.events.append(("fail", kwargs))
        self.fail_count += 1
        self.fail_kwargs = kwargs
        self.pending_cohort_lease = None

    def prepare_exact_greedy_cohort_burst_commit(
        self,
        seqs,
        lease,
        result,
        **kwargs,
    ):
        self.events.append(("prepare_commit",))
        return SimpleNamespace(
            seqs=tuple(seqs),
            lease=lease,
            result=result,
            rows=tuple(
                SimpleNamespace(
                    sequence_id=seq.seq_id,
                    output_tokens=row_tokens,
                )
                for seq, row_tokens in zip(seqs, result.tokens)
            ),
        )

    def commit_prepared_postprocess(self, prepared):
        self.events.append(
            (
                "commit",
                tuple(row.sequence_id for row in prepared.rows),
            )
        )
        self.commit_count += 1
        for seq, row in zip(prepared.seqs, prepared.rows):
            seq.token_ids.extend(row.output_tokens)
        self.pending_cohort_lease = None


class _ModelRunner:
    world_size = 1
    rank = 0
    block_size = 16

    def __init__(self, outcome):
        self.config = SimpleNamespace(
            exact_greedy_cohort_burst=True,
            max_model_len=2_048,
        )
        self.outcome = outcome
        self.replay_count = 0
        self.ordinary_forward_count = 0
        self.capability_calls = []

    def exact_greedy_cohort_burst_capability(self, **kwargs):
        self.capability_calls.append(dict(kwargs))
        return {
            "available": True,
            "quarantined": False,
            "shape_supported": True,
            "graph_identity_sha256": "a" * 64,
            "graph_generation": 7,
        }

    def call(self, method_name, *args):
        if method_name == "run_exact_greedy_cohort_burst":
            lease = args[0]
            self.replay_count += lease.authorized_width
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome
        if method_name == "run":
            self.ordinary_forward_count += 1
            return (99,)
        raise AssertionError(method_name)


class _Engine:
    _execute_slo_cohort_burst = _load_engine_method(
        "_execute_slo_cohort_burst"
    )
    _snapshot_slo_cohort_telemetry = _load_engine_method(
        "_snapshot_slo_cohort_telemetry"
    )

    def __init__(self, outcome, *, decision_width=4):
        self.scheduler = _Scheduler(
            decision_width=decision_width
        )
        self.model_runner = _ModelRunner(outcome)
        self._clock_values = iter((140,))

    def _clock_ns(self):
        return next(self._clock_values)


def _result(tokens):
    return SimpleNamespace(tokens=tuple(tokens))


def _run(engine, seqs):
    return engine._execute_slo_cohort_burst(
        tuple(seqs),
        decision_now_ns=100,
        completion_only=True,
        is_prefill=False,
        do_sample=True,
        batch_kind=None,
        exact_burst_gate_width=None,
        exact_burst_correctness_trace=False,
    )


def test_engine_commits_all_cohort_prefixes_once_in_scheduler_order():
    seqs = tuple(_Sequence(seq_id) for seq_id in (7, 9, 11, 13))
    engine = _Engine(
        _result(
            (
                (31, 32, 33, 34),
                (41, 42),
                (51, 52, 53, 54),
                (61, 62, 63, 64),
            )
        )
    )

    committed, step_end_ns, committed_tokens = _run(engine, seqs)

    assert committed is True
    assert step_end_ns == 140
    assert committed_tokens == 14
    assert engine.scheduler.commit_count == 1
    assert engine.model_runner.replay_count == 4
    assert engine.scheduler.events[-1] == (
        "commit",
        (7, 9, 11, 13),
    )


def test_post_replay_failure_is_terminal_without_k1_retry():
    seqs = tuple(_Sequence(seq_id) for seq_id in (7, 9, 11, 13))
    engine = _Engine(
        ExactGreedyCohortBurstTerminalError(
            "cohort burst replay failed",
            completed_replays=2,
        )
    )

    with pytest.raises(RuntimeError, match="cohort burst"):
        _run(engine, seqs)

    assert engine.model_runner.ordinary_forward_count == 0
    assert engine.scheduler.pending_cohort_lease is None
    assert engine.scheduler.fail_count == 1


def test_failed_quarantine_is_not_reported_as_successful():
    seqs = tuple(_Sequence(seq_id) for seq_id in (7, 9))
    engine = _Engine(
        ExactGreedyCohortBurstTerminalError(
            "cohort burst replay failed",
            completed_replays=2,
        )
    )

    def fail_quarantine(_graph_identity, _reason):
        raise RuntimeError("quarantine failed")

    engine.model_runner.quarantine_exact_greedy_cohort_burst_graph = (
        fail_quarantine
    )

    with pytest.raises(RuntimeError, match="cohort burst"):
        _run(engine, seqs)

    telemetry = engine._last_slo_cohort_execution_telemetry
    assert telemetry.quarantined is False
    assert telemetry.quarantine_reason is None


def test_invalid_result_replay_count_is_bounded_for_terminal_cleanup():
    seqs = tuple(_Sequence(seq_id) for seq_id in (7, 9, 11, 13))
    engine = _Engine(
        SimpleNamespace(replay_count=99),
    )

    with pytest.raises(AttributeError, match="tokens"):
        _run(engine, seqs)

    assert engine.scheduler.fail_kwargs["completed_replays"] == 4
    assert engine.scheduler.pending_cohort_lease is None


def test_malformed_result_replay_count_still_closes_terminal_inventory():
    seqs = tuple(_Sequence(seq_id) for seq_id in (7, 9, 11, 13))
    engine = _Engine(
        SimpleNamespace(replay_count=None),
    )

    with pytest.raises(AttributeError, match="tokens"):
        _run(engine, seqs)

    assert engine.scheduler.fail_kwargs["completed_replays"] == 1
    assert engine.scheduler.pending_cohort_lease is None


def test_pre_replay_fallback_cancels_and_returns_k1_eligibility():
    seqs = tuple(_Sequence(seq_id) for seq_id in (7, 9))
    engine = _Engine(
        ExactGreedyCohortBurstFallback("row_bind_failure")
    )

    committed, step_end_ns, committed_tokens = _run(engine, seqs)

    assert (committed, step_end_ns, committed_tokens) == (
        False,
        None,
        0,
    )
    assert engine.scheduler.cancel_count == 1
    assert engine.scheduler.pending_cohort_lease is None


def test_engine_queries_the_exact_active_block_table_width():
    seq = _Sequence(7)
    engine = _Engine(_result(((31, 32),)), decision_width=1)

    committed, step_end_ns, committed_tokens = _run(engine, (seq,))

    assert (committed, step_end_ns, committed_tokens) == (
        False,
        None,
        0,
    )
    assert engine.model_runner.capability_calls == [{
        "batch_size": 1,
        "block_table_width": 1,
        "correctness_trace": False,
    }]


def test_step_calls_cohort_orchestration_only_on_non_speculative_path():
    tree = ast.parse(
        ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(ENGINE_PATH),
    )
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    step = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
    )
    calls = [
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_execute_slo_cohort_burst"
    ]
    assert len(calls) == 1


def test_cohort_mode_suppresses_legacy_single_request_burst_fallthrough():
    tree = ast.parse(
        ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(ENGINE_PATH),
    )
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    step = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "step"
    )
    candidate_assignment = next(
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "exact_burst_candidate"
            for target in node.targets
        )
    )
    assert any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "cohort_burst_enabled"
        for node in ast.walk(candidate_assignment.value)
    )


def test_step_observation_exposes_closed_cohort_telemetry() -> None:
    source = ENGINE_PATH.read_text(encoding="utf-8")
    for key in (
        '"slo_cohort_decision_telemetry"',
        '"exact_greedy_cohort_burst_execution_telemetry"',
        '"slo_cohort_request_telemetry"',
    ):
        assert key in source


def test_request_telemetry_hash_failure_is_non_interfering_and_not_drained():
    row = {
        "terminal_reason": "eos",
        "output_token_ids": [7, 8],
    }

    class Scheduler:
        def __init__(self):
            self.calls = []

        def slo_cohort_telemetry_snapshot(
            self,
            *,
            drain_completed=False,
        ):
            self.calls.append(drain_completed)
            return {
                "decision": {"selected_width": 4},
                "requests": [dict(row)],
            }

    class Tokenizer:
        def decode(self, _token_ids):
            raise RuntimeError("decode failed")

    engine = object.__new__(_Engine)
    engine.scheduler = Scheduler()
    engine.tokenizer = Tokenizer()
    engine._last_slo_cohort_telemetry_error = None

    snapshot = engine._snapshot_slo_cohort_telemetry()

    assert snapshot == {"decision": None, "requests": []}
    assert engine.scheduler.calls == [False]
    assert engine._last_slo_cohort_telemetry_error == (
        "RuntimeError: decode failed"
    )
