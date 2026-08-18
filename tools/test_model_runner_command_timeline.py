from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tinyvllm" / "engine" / "model_runner_command_timeline.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "model_runner_command_timeline",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identity(module, command_id=7, requires_ack=False):
    return module.CommandTraceIdentity(
        command_id=command_id,
        method_name="run",
        requires_ack=requires_ack,
        engine_step_id=3,
        repeat_index=2,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
        dispatch_started_monotonic_ns=1_000,
        dispatch_published_monotonic_ns=1_100,
    )


def clock_identity(module, captured_at_unix_ns=5_000):
    return module.CommandClockIdentity(
        boot_id="boot",
        implementation="clock_gettime(CLOCK_MONOTONIC)",
        resolution_s=1e-9,
        monotonic=True,
        adjustable=False,
        captured_at_unix_ns=captured_at_unix_ns,
    )


def test_disabled_recorder_is_empty_and_side_effect_free():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder.disabled(rank=2)
    recorder.record_worker_receive(
        identity(module),
        event_woken_monotonic_ns=1_200,
        envelope_read_monotonic_ns=1_250,
    )
    assert recorder.snapshot() == {
        "schema_version": 1,
        "rank": 2,
        "enabled": False,
        "clock": None,
        "rows": [],
        "dropped_rows": 0,
    }


def test_clock_identity_records_wall_clock_capture(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.time, "time_ns", lambda: 987_654_321)
    monkeypatch.setattr(
        module.Path,
        "read_text",
        lambda self, encoding: "boot-id\n",
    )

    result = module.read_command_clock_identity()

    assert result.boot_id == "boot-id"
    assert result.captured_at_unix_ns == 987_654_321


def test_command_trace_scope_restores_prior_identity():
    module = load_module()
    outer = identity(module, 1)
    inner = identity(module, 2)

    assert module.active_model_runner_command_trace() is None
    with module.command_trace_scope(outer):
        assert module.active_model_runner_command_trace() == outer
        with module.command_trace_scope(inner):
            assert module.active_model_runner_command_trace() == inner
        assert module.active_model_runner_command_trace() == outer
    assert module.active_model_runner_command_trace() is None


def test_recorder_requires_strict_command_order():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder(
        rank=1,
        max_rows=8,
        clock_identity=clock_identity(module),
    )
    recorder.record_worker_receive(
        identity(module, 1),
        event_woken_monotonic_ns=1_200,
        envelope_read_monotonic_ns=1_250,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.record_worker_receive(
            identity(module, 1),
            event_woken_monotonic_ns=1_300,
            envelope_read_monotonic_ns=1_350,
        )


def test_recorder_captures_worker_lifecycle_and_returns_deep_copy():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder(
        rank=1,
        max_rows=8,
        clock_identity=clock_identity(module),
    )
    recorder.record_worker_receive(
        identity(module, 1, requires_ack=True),
        event_woken_monotonic_ns=1_200,
        envelope_read_monotonic_ns=1_250,
    )
    recorder.record_method_start(1, started_ns=1_300)
    recorder.record_method_end(
        1,
        finished_ns=1_500,
        status="error",
        error_type="x" * 200,
    )
    recorder.record_ack_send_start(1, started_ns=1_550)
    recorder.record_ack_send_end(1, finished_ns=1_600)

    snapshot = recorder.snapshot()

    assert snapshot["clock"]["captured_at_unix_ns"] == 5_000
    assert snapshot["rows"][0]["method_started_monotonic_ns"] == 1_300
    assert snapshot["rows"][0]["method_finished_monotonic_ns"] == 1_500
    assert snapshot["rows"][0]["ack_send_started_monotonic_ns"] == 1_550
    assert snapshot["rows"][0]["ack_send_finished_monotonic_ns"] == 1_600
    assert snapshot["rows"][0]["status"] == "error"
    assert snapshot["rows"][0]["error_type"] == "x" * 128

    snapshot["rows"][0]["status"] = "mutated"
    assert recorder.snapshot()["rows"][0]["status"] == "error"


def test_recorder_rejects_snapshot_during_active_phase():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder(
        rank=0,
        max_rows=8,
        clock_identity=clock_identity(module),
    )
    recorder.record_dispatch(identity(module, 1))
    recorder.record_method_start(1, started_ns=1_200)

    with pytest.raises(ValueError, match="unfinished"):
        recorder.snapshot()


def test_recorder_drops_rows_beyond_capacity():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder(
        rank=0,
        max_rows=1,
        clock_identity=clock_identity(module),
    )
    recorder.record_dispatch(identity(module, 1))
    recorder.record_method_start(1, started_ns=1_200)
    recorder.record_method_end(1, finished_ns=1_300)
    recorder.record_dispatch(identity(module, 2))
    recorder.record_method_start(2, started_ns=1_400)
    recorder.record_method_end(2, finished_ns=1_500)

    snapshot = recorder.snapshot()

    assert [row["command_id"] for row in snapshot["rows"]] == [1]
    assert snapshot["dropped_rows"] == 1


def test_queue_debt_arithmetic_separates_prior_overlap():
    module = load_module()
    rows = [
        {
            "rank": 1,
            "command_id": 1,
            "dispatch_published_monotonic_ns": 100,
            "method_started_monotonic_ns": 120,
            "method_finished_monotonic_ns": 260,
            "cuda_ns": 100,
        },
        {
            "rank": 1,
            "command_id": 2,
            "dispatch_published_monotonic_ns": 200,
            "method_started_monotonic_ns": 260,
            "method_finished_monotonic_ns": 320,
            "cuda_ns": 40,
        },
    ]
    result = module.compute_command_decomposition(rows)
    assert result[1]["worker_queue_wait_ns"] == 60
    assert result[1]["queued_behind_prior_command_ns"] == 60
    assert result[1]["worker_ready_delay_ns"] == 0
    assert result[1]["worker_non_cuda_upper_bound_ns"] == 20


def test_command_decomposition_rejects_missing_predecessor():
    module = load_module()
    rows = [
        {
            "rank": 1,
            "command_id": 1,
            "dispatch_published_monotonic_ns": 100,
            "method_started_monotonic_ns": 120,
            "method_finished_monotonic_ns": 160,
            "cuda_ns": 20,
        },
        {
            "rank": 1,
            "command_id": 3,
            "dispatch_published_monotonic_ns": 200,
            "method_started_monotonic_ns": 220,
            "method_finished_monotonic_ns": 260,
            "cuda_ns": 20,
        },
    ]

    with pytest.raises(ValueError, match="missing predecessor"):
        module.compute_command_decomposition(rows)
