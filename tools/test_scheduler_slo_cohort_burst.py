from __future__ import annotations

import ast
from collections import deque
from copy import deepcopy
from dataclasses import replace
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
    record_slo_publication = _load_scheduler_method(
        "record_slo_publication"
    )
    remove_slo_request = _load_scheduler_method(
        "remove_slo_request"
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

    def __init__(self):
        self.slo_request_state_by_seq_id = {}
        self.slo_clock_invalid = False
        self.slo_clock_invalid_reason = None
        self.exact_greedy_cohort_burst = True
        self.exact_greedy_cohort_burst_widths = (1, 2, 4, 8)
        self.exact_greedy_cohort_burst_target_itl_ns = 100
        self.exact_greedy_cohort_burst_target_ttft_ns = 100
        self.exact_greedy_cohort_burst_reserve_ns = 10
        self._exact_greedy_cohort_burst_pending_lease = None
        self.chunked_prefill_slo_mixed = False
        self.decode_progress_ns_by_seq_id = {}
        self.block_manager = SimpleNamespace(block_size=4)
        self.running = deque()
        self.waiting = deque()
        self.prefilling = deque()

    def _invalidate_slo_clock(self, reason: str) -> None:
        if not self.slo_clock_invalid:
            self.slo_clock_invalid = True
            self.slo_clock_invalid_reason = reason

    def _validate_admission(self, seq) -> None:
        del seq


class FakeSequence:
    def __init__(self, sequence_id: int, max_tokens: int = 16):
        self.seq_id = sequence_id
        self.num_tokens = 4
        self.num_prompt_tokens = 4
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens


def _scheduler():
    return FakeScheduler()


def _sequence(sequence_id: int, *, max_tokens: int = 16):
    return FakeSequence(sequence_id, max_tokens)


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


def test_engine_passes_its_monotonic_clock_to_scheduler_admission() -> None:
    source = (
        REPO_ROOT / "tinyvllm" / "engine" / "llm_engine.py"
    ).read_text(encoding="utf-8")
    assert (
        "self.scheduler.add(\n"
        "                seq,\n"
        "                arrival_ns=self._clock_ns(),\n"
        "            )"
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
