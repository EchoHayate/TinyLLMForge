from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass, replace
from importlib import util
from itertools import count
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tinyvllm" / "engine" / "engine_step_timeline.py"
MODEL_RUNNER_PATH = ROOT / "tinyvllm" / "engine" / "model_runner.py"
LLM_ENGINE_PATH = ROOT / "tinyvllm" / "engine" / "llm_engine.py"

PHASES = (
    "scheduler_schedule",
    "partition_and_step_setup",
    "ordinary_or_first_target_dispatch",
    "speculative_prepare",
    "scheduler_prepare_postprocess",
    "proposal_kv_prepare_commit",
    "proposal_lifecycle_finalize_prepare",
    "scheduler_commit_postprocess",
    "proposal_lifecycle_finalize_commit",
    "side_state_seal",
    "residency_precommit_or_seal",
    "ordinary_scheduler_postprocess",
)


def load_module():
    name = "tinyvllm.engine.engine_step_timeline"
    sys.modules.pop(name, None)
    spec = util.spec_from_file_location(name, MODULE_PATH)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model_runner_method(name):
    tree = ast.parse(MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    @dataclass(frozen=True)
    class CommandTraceIdentity:
        command_id: int
        method_name: str
        requires_ack: bool
        engine_step_id: int | None
        repeat_index: int | None
        request_set_sha256: str | None
        batch_kind: str | None
        speculative_selected_sequence_ids_sha256: str | None
        dispatch_started_monotonic_ns: int
        dispatch_published_monotonic_ns: int

    @dataclass(frozen=True)
    class ModelRunnerCommandEnvelope:
        command_id: int
        method_name: str
        args: tuple
        requires_ack: bool
        trace_identity: CommandTraceIdentity | None = None

    namespace = {
        "CommandTraceIdentity": CommandTraceIdentity,
        "ModelRunnerCommandEnvelope": ModelRunnerCommandEnvelope,
        "replace": replace,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def load_llm_engine_method(name, namespace=None):
    tree = ast.parse(LLM_ENGINE_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    globals_namespace = {} if namespace is None else dict(namespace)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(LLM_ENGINE_PATH),
            "exec",
        ),
        globals_namespace,
    )
    return globals_namespace[name]


def begin(recorder, **overrides):
    values = {
        "repeat_index": 0,
        "request_set_sha256": "a" * 64,
        "batch_kind": "decode",
        "speculative_selected_sequence_ids_sha256": "b" * 64,
    }
    values.update(overrides)
    return recorder.begin_step(**values)


def phase_inventory(**executed):
    phases = {
        phase: {
            "executed": False,
            "started_monotonic_ns": None,
            "finished_monotonic_ns": None,
            "duration_ns": 0,
        }
        for phase in PHASES
    }
    phases.update(executed)
    return phases


def test_step_identity_is_immutable_and_phase_inventory_is_fixed():
    module = load_module()
    identity = module.EngineStepTraceIdentity(
        engine_step_id=4,
        repeat_index=2,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
    )

    assert module.ENGINE_STEP_PHASES == PHASES
    with pytest.raises(FrozenInstanceError):
        identity.repeat_index = 3


def test_step_recorder_emits_explicit_skipped_phases_and_deep_copies():
    module = load_module()
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        max_steps=4,
        clock_ns=iter((100, 120, 140, 180)).__next__,
    )
    identity = begin(recorder)
    with recorder.phase("scheduler_schedule"):
        pass
    recorder.finish_step(identity)

    snapshot = recorder.snapshot()
    row = snapshot["steps"][0]
    assert row["phases"]["scheduler_schedule"] == {
        "executed": True,
        "started_monotonic_ns": 120,
        "finished_monotonic_ns": 140,
        "duration_ns": 20,
    }
    assert row["phases"]["speculative_prepare"] == {
        "executed": False,
        "started_monotonic_ns": None,
        "finished_monotonic_ns": None,
        "duration_ns": 0,
    }
    assert row["status"] == "ok"
    assert row["detail"] == ""

    snapshot["steps"][0]["phases"]["scheduler_schedule"][
        "duration_ns"
    ] = -1
    assert (
        recorder.snapshot()["steps"][0]["phases"][
            "scheduler_schedule"
        ]["duration_ns"]
        == 20
    )


def test_recorder_rejects_nested_repeated_and_mismatched_finish():
    module = load_module()
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=count(100).__next__,
    )
    identity = begin(recorder)

    with recorder.phase("scheduler_schedule"):
        with pytest.raises(RuntimeError, match="active phase"):
            with recorder.phase("partition_and_step_setup"):
                pass
    with pytest.raises(RuntimeError, match="already executed"):
        with recorder.phase("scheduler_schedule"):
            pass
    with pytest.raises(ValueError, match="active step identity"):
        recorder.finish_step(
            module.EngineStepTraceIdentity(
                engine_step_id=identity.engine_step_id + 1,
                repeat_index=identity.repeat_index,
                request_set_sha256=identity.request_set_sha256,
                batch_kind=identity.batch_kind,
                speculative_selected_sequence_ids_sha256=(
                    identity.speculative_selected_sequence_ids_sha256
                ),
            )
        )

    recorder.finish_step(identity)
    with pytest.raises(RuntimeError, match="no active step"):
        recorder.finish_step(identity)


def test_nested_step_attempt_preserves_active_context_and_cleanup():
    module = load_module()
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=count(10).__next__,
    )
    identity = begin(recorder)
    assert module.active_engine_step_trace() == identity

    with pytest.raises(RuntimeError, match="active step"):
        begin(recorder, repeat_index=1)
    assert module.active_engine_step_trace() == identity

    recorder.finish_step(
        identity,
        error=RuntimeError("x" * 5000),
    )
    assert module.active_engine_step_trace() is None
    row = recorder.snapshot()["steps"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"
    assert len(row["detail"].encode("utf-8")) <= 4096


def test_finish_clock_failure_still_cleans_active_context():
    module = load_module()
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=iter((10,)).__next__,
    )
    identity = begin(recorder)

    with pytest.raises(StopIteration):
        recorder.finish_step(identity, command_rows=[])
    assert module.active_engine_step_trace() is None
    assert recorder.active is False


def test_phase_failure_and_scope_nesting_reset_context_safely():
    module = load_module()
    outer = module.EngineStepTraceIdentity(
        engine_step_id=1,
        repeat_index=0,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256=None,
    )
    inner = module.EngineStepTraceIdentity(
        engine_step_id=2,
        repeat_index=1,
        request_set_sha256="b" * 64,
        batch_kind="mixed",
        speculative_selected_sequence_ids_sha256="c" * 64,
    )

    assert module.active_engine_step_trace() is None
    with module.engine_step_trace_scope(outer):
        assert module.active_engine_step_trace() == outer
        with pytest.raises(RuntimeError, match="boom"):
            with module.engine_step_trace_scope(inner):
                assert module.active_engine_step_trace() == inner
                raise RuntimeError("boom")
        assert module.active_engine_step_trace() == outer
    assert module.active_engine_step_trace() is None

    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=count(20).__next__,
    )
    identity = begin(recorder)
    with pytest.raises(ValueError, match="phase failed"):
        with recorder.phase("speculative_prepare"):
            raise ValueError("phase failed")
    assert module.active_engine_step_trace() == identity
    recorder.finish_step(identity, error=ValueError("phase failed"))
    assert module.active_engine_step_trace() is None


def test_disabled_recorder_is_a_clock_free_noop():
    module = load_module()
    recorder = module.EngineStepTimelineRecorder(
        enabled=False,
        clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("disabled recorder read clock")
        ),
    )

    assert begin(recorder) is None
    with recorder.phase("scheduler_schedule"):
        pass
    assert recorder.bind_step_identity(
        None,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256=None,
    ) is None
    recorder.finish_step(None)
    assert recorder.snapshot() == {
        "schema_version": 1,
        "enabled": False,
        "max_steps": 0,
        "dropped_steps": 0,
        "steps": [],
    }
    assert module.active_engine_step_trace() is None


def test_step_conservation_uses_larger_absolute_or_relative_tolerance():
    module = load_module()
    result = module.compute_step_conservation(
        {
            "step_wall_ns": 100_000_000,
            "phases": {
                "scheduler_schedule": {"duration_ns": 10_000_000},
                "scheduler_commit_postprocess": {
                    "duration_ns": 20_000_000
                },
            },
        },
        command_critical_path_ns=69_000_000,
        acknowledged_wait_ns=0,
    )

    assert result["serial_phase_sum_ns"] == 30_000_000
    assert result["residual_ns"] == 1_000_000
    assert result["tolerance_ns"] == 2_000_000
    assert result["status"] == "ok"
    assert result["passed"] is True

    relative = module.compute_step_conservation(
        {
            "step_wall_ns": 400_000_000,
            "phases": {
                "scheduler_schedule": {"duration_ns": 396_000_000},
            },
        },
        command_critical_path_ns=0,
        acknowledged_wait_ns=0,
    )
    assert relative["residual_ns"] == 4_000_000
    assert relative["tolerance_ns"] == 4_000_000
    assert relative["passed"] is True


@pytest.mark.parametrize(
    "step,critical_path,ack_wait,detail",
    [
        (
            {
                "step_wall_ns": 10,
                "phases": {
                    "scheduler_schedule": {"duration_ns": 11},
                },
            },
            0,
            0,
            "over-attributed",
        ),
        (
            {
                "step_wall_ns": 10,
                "phases": {
                    "scheduler_schedule": {"duration_ns": -1},
                },
            },
            0,
            0,
            "duration_ns",
        ),
        (
            {"step_wall_ns": 10, "phases": {}},
            None,
            None,
            "command rows",
        ),
    ],
)
def test_step_conservation_fails_closed(
    step,
    critical_path,
    ack_wait,
    detail,
):
    module = load_module()
    kwargs = {}
    if critical_path is not None:
        kwargs = {
            "command_critical_path_ns": critical_path,
            "acknowledged_wait_ns": ack_wait,
        }
    result = module.compute_step_conservation(step, **kwargs)

    assert result["status"] == "invalid"
    assert result["passed"] is False
    assert detail in result["detail"]


def test_command_row_conservation_uses_post_local_ack_without_double_count():
    module = load_module()
    step = {
        "engine_step_id": 7,
        "started_monotonic_ns": 100,
        "finished_monotonic_ns": 1_100,
        "step_wall_ns": 1_000,
        "phases": phase_inventory(
            proposal_lifecycle_finalize_prepare={
                "executed": True,
                "started_monotonic_ns": 200,
                "finished_monotonic_ns": 800,
                "duration_ns": 600,
            },
        ),
    }
    command_rows = [{
        "rank": 0,
        "engine_step_id": 7,
        "local_method_started_monotonic_ns": 300,
        "local_method_finished_monotonic_ns": 600,
        "ack_wait_started_monotonic_ns": 350,
        "ack_wait_finished_monotonic_ns": 700,
    }]

    result = module.compute_step_conservation(step, command_rows)

    assert result["serial_phase_sum_ns"] == 200
    assert result["command_critical_path_ns"] == 300
    assert result["acknowledged_wait_ns"] == 100
    assert result["residual_ns"] == 400
    assert result["passed"] is True


def test_command_row_conservation_rejects_missing_required_command_data():
    module = load_module()
    result = module.compute_step_conservation(
        {
            "engine_step_id": 7,
            "started_monotonic_ns": 100,
            "finished_monotonic_ns": 1_100,
            "step_wall_ns": 1_000,
            "phases": phase_inventory(
                ordinary_or_first_target_dispatch={
                    "executed": True,
                    "started_monotonic_ns": 200,
                    "finished_monotonic_ns": 800,
                    "duration_ns": 600,
                },
            ),
        },
        [],
    )

    assert result["status"] == "invalid"
    assert result["passed"] is False
    assert "matching rank-zero command rows" in result["detail"]


def test_task2_dispatch_receives_live_step_and_repeat_identity():
    module = load_module()
    dispatch = load_model_runner_method("dispatch_command")

    class Timeline:
        enabled = True

        def __init__(self):
            self.rows = []

        def record_dispatch(self, identity):
            self.rows.append(identity)

    timeline = Timeline()
    runner = types.SimpleNamespace(
        rank=0,
        world_size=1,
        _command_ids=count(9),
        command_timeline=timeline,
        _command_timeline_clock_ns=count(100, 10).__next__,
        _active_command_timeline_trace=(
            module.active_engine_step_trace
        ),
        write_shm=lambda envelope: envelope,
    )
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=count(1).__next__,
    )
    identity = begin(
        recorder,
        repeat_index=3,
        batch_kind="unknown",
        speculative_selected_sequence_ids_sha256=None,
    )
    identity = recorder.bind_step_identity(
        identity,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
    )

    envelope = dispatch(
        runner,
        "run",
        requires_ack=False,
    )

    trace = envelope.trace_identity
    assert trace.engine_step_id == identity.engine_step_id
    assert trace.repeat_index == 3
    assert trace.request_set_sha256 == "a" * 64
    assert trace.batch_kind == "decode"
    assert trace.speculative_selected_sequence_ids_sha256 == "b" * 64
    recorder.finish_step(identity)
    assert module.active_engine_step_trace() is None


def test_bounded_snapshot_drops_new_rows_without_mutating_old_rows():
    module = load_module()
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        max_steps=1,
        clock_ns=count(1).__next__,
    )
    first = begin(recorder)
    recorder.finish_step(first)
    second = begin(recorder, repeat_index=1)
    recorder.finish_step(second)

    snapshot = recorder.snapshot()
    assert len(snapshot["steps"]) == 1
    assert snapshot["steps"][0]["engine_step_id"] == first.engine_step_id
    assert snapshot["dropped_steps"] == 1


def test_engine_repeat_lifecycle_and_snapshot_are_explicit():
    module = load_module()
    begin_repeat = load_llm_engine_method(
        "begin_command_timeline_repeat"
    )
    end_repeat = load_llm_engine_method(
        "end_command_timeline_repeat"
    )
    snapshot = load_llm_engine_method(
        "engine_step_timeline_snapshot"
    )
    engine = types.SimpleNamespace(
        engine_step_timeline=module.EngineStepTimelineRecorder(
            enabled=True,
            clock_ns=count(1).__next__,
        ),
        _command_timeline_repeat_index=None,
        _command_timeline_request_set_sha256=None,
    )

    assert begin_repeat(
        engine,
        3,
        request_set_sha256="a" * 64,
    ) == {
        "repeat_index": 3,
        "request_set_sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="already active"):
        begin_repeat(engine, 4)
    assert snapshot(engine)["steps"] == []
    assert end_repeat(engine) == {
        "repeat_index": 3,
        "request_set_sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="not active"):
        end_repeat(engine)


def test_engine_command_timeline_reset_clears_step_rows_and_repeat():
    module = load_module()
    reset = load_llm_engine_method("reset_command_timeline")
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        max_steps=4,
        clock_ns=count(1).__next__,
    )
    identity = begin(recorder)
    recorder.finish_step(identity, command_rows=[])
    engine = types.SimpleNamespace(
        call_model_runner_acknowledged=(
            lambda method_name, timeout_s: (
                {"rank": 0, "enabled": True, "max_rows": 4},
                (),
            )
        ),
        engine_step_timeline=recorder,
        _command_timeline_repeat_index=3,
        _command_timeline_request_set_sha256="a" * 64,
    )

    assert reset(engine, timeout_s=5.0) == (
        {"rank": 0, "enabled": True, "max_rows": 4},
    )
    assert engine.engine_step_timeline.snapshot() == {
        "schema_version": 1,
        "enabled": True,
        "max_steps": 4,
        "dropped_steps": 0,
        "steps": [],
    }
    assert engine._command_timeline_repeat_index is None
    assert engine._command_timeline_request_set_sha256 is None


def test_engine_command_timeline_reset_rejects_active_step_before_dispatch():
    module = load_module()
    reset = load_llm_engine_method("reset_command_timeline")
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        max_steps=4,
        clock_ns=count(1).__next__,
    )
    identity = begin(recorder)
    engine = types.SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("reset dispatched during active step")),
        engine_step_timeline=recorder,
    )

    with pytest.raises(RuntimeError, match="active step"):
        reset(engine, timeout_s=5.0)
    recorder.finish_step(identity, command_rows=[])


def test_engine_step_failure_finalizes_telemetry_and_preserves_exception():
    module = load_module()
    step = load_llm_engine_method("step")
    expected = RuntimeError("scheduler failed")

    class Scheduler:
        def observation_snapshot(self):
            return {"running": [7]}

        def schedule(self, decision_now_ns):
            assert decision_now_ns == 100
            raise expected

    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=iter((10, 20, 30, 40)).__next__,
    )
    engine = types.SimpleNamespace(
        _clock_ns=lambda: 100,
        scheduler=Scheduler(),
        engine_step_timeline=recorder,
        _command_timeline_repeat_index=2,
        _command_timeline_request_set_sha256="a" * 64,
        model_runner=types.SimpleNamespace(
            command_timeline=types.SimpleNamespace(
                snapshot=lambda: {"rows": []}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="scheduler failed") as raised:
        step(engine)

    assert raised.value is expected
    assert module.active_engine_step_trace() is None
    row = recorder.snapshot()["steps"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"
    assert row["detail"] == "scheduler failed"
    assert row["phases"]["scheduler_schedule"]["executed"] is True
