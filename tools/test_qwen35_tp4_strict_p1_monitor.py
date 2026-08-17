from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load_monitor():
    path = TOOLS / "qwen35_tp4_strict_p1_monitor.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_strict_p1_monitor",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _blocked(sample_id):
    return {
        "classification": "BLOCKED_RESOURCES",
        "sample_id": sample_id,
        "gpu_rows": [],
    }


def _ready(sample_id):
    return {
        "classification": "READY",
        "sample_id": sample_id,
        "selected_gpus": [
            {
                "gpu_index": index,
                "free_bytes": 30 * 1024**3,
                "compute_processes": [],
            }
            for index in (2, 4, 5, 6)
        ],
    }


def test_monitor_waits_sixty_seconds_and_requires_consecutive_ready():
    monitor = _load_monitor()
    samples = iter([
        _blocked(1),
        _ready(2),
        _blocked(3),
        _ready(4),
        _ready(5),
    ])
    sleeps = []
    launches = []
    with tempfile.TemporaryDirectory() as temporary:
        result = monitor.monitor_until_launch(
            monitor_tag="monitor-test",
            output_dir=Path(temporary) / "monitor",
            sample_fn=lambda: next(samples),
            launch_fn=lambda: launches.append("launch") or {
                "classification": "PASS",
            },
            cleanup_fn=lambda: {
                "classification": "CLEAN",
                "matched_pids": [],
            },
            sleep_fn=sleeps.append,
            interval_s=60,
            required_ready_samples=2,
            max_samples=5,
        )

        assert sleeps == [60, 60, 60, 60]
        assert launches == ["launch"]
        assert result["classification"] == "PASS"
        assert result["trigger_sample_ids"] == [4, 5]
        rows = [
            json.loads(line)
            for line in (
                Path(temporary)
                / "monitor"
                / "resource_samples.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert [row["sample_id"] for row in rows] == [1, 2, 3, 4, 5]


def test_monitor_does_not_launch_when_sample_budget_expires():
    monitor = _load_monitor()
    launches = []
    with tempfile.TemporaryDirectory() as temporary:
        result = monitor.monitor_until_launch(
            monitor_tag="bounded-monitor",
            output_dir=Path(temporary) / "monitor",
            sample_fn=lambda: _blocked(1),
            launch_fn=lambda: launches.append("launch"),
            cleanup_fn=lambda: {
                "classification": "CLEAN",
                "matched_pids": [],
            },
            sleep_fn=lambda seconds: None,
            interval_s=60,
            required_ready_samples=2,
            max_samples=3,
        )

        assert launches == []
        assert result["classification"] == "MONITOR_EXPIRED"
        assert result["sample_count"] == 3


def test_monitor_records_launch_failure_and_always_runs_scoped_cleanup():
    monitor = _load_monitor()
    cleanup_calls = []
    with tempfile.TemporaryDirectory() as temporary:
        try:
            monitor.monitor_until_launch(
                monitor_tag="failure-monitor",
                output_dir=Path(temporary) / "monitor",
                sample_fn=lambda: _ready(1),
                launch_fn=lambda: (_ for _ in ()).throw(
                    RuntimeError("worker failed")
                ),
                cleanup_fn=lambda: cleanup_calls.append("cleanup") or {
                    "classification": "CLEAN",
                    "matched_pids": [123],
                },
                sleep_fn=lambda seconds: None,
                interval_s=60,
                required_ready_samples=1,
                max_samples=1,
            )
        except RuntimeError as error:
            assert str(error) == "worker failed"
        else:
            raise AssertionError("launch failure was hidden")

        assert cleanup_calls == ["cleanup"]
        failure = json.loads(
            (
                Path(temporary) / "monitor" / "monitor_failure.json"
            ).read_text(encoding="utf-8")
        )
        assert failure["classification"] == "FAILED"
        assert failure["error"] == "worker failed"
        assert failure["cleanup"]["matched_pids"] == [123]


def test_monitor_persists_launch_and_cleanup_failures_together():
    monitor = _load_monitor()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "monitor"
        try:
            monitor.monitor_until_launch(
                monitor_tag="double-failure-monitor",
                output_dir=output_dir,
                sample_fn=lambda: _ready(1),
                launch_fn=lambda: (_ for _ in ()).throw(
                    RuntimeError("worker failed")
                ),
                cleanup_fn=lambda: (_ for _ in ()).throw(
                    RuntimeError("cleanup transport failed")
                ),
                sleep_fn=lambda seconds: None,
                interval_s=60,
                required_ready_samples=1,
                max_samples=1,
            )
        except RuntimeError as error:
            assert str(error) == "worker failed"
        else:
            raise AssertionError("launch failure was hidden")

        failure = json.loads(
            (output_dir / "monitor_failure.json").read_text(
                encoding="utf-8"
            )
        )
        assert failure["classification"] == "FAILED"
        assert failure["error"] == "worker failed"
        assert failure["cleanup"]["classification"] == "CLEANUP_FAILED"
        assert failure["cleanup"]["error"] == "cleanup transport failed"


def test_monitor_classifies_successful_launch_with_failed_cleanup():
    monitor = _load_monitor()
    with tempfile.TemporaryDirectory() as temporary:
        result = monitor.monitor_until_launch(
            monitor_tag="cleanup-failure-monitor",
            output_dir=Path(temporary) / "monitor",
            sample_fn=lambda: _ready(1),
            launch_fn=lambda: {"classification": "PASS"},
            cleanup_fn=lambda: (_ for _ in ()).throw(
                RuntimeError("cleanup transport failed")
            ),
            sleep_fn=lambda seconds: None,
            interval_s=60,
            required_ready_samples=1,
            max_samples=1,
        )

        assert result["classification"] == "CLEANUP_FAILED"
        assert result["launch_result"]["classification"] == "PASS"
        assert result["cleanup"]["classification"] == "CLEANUP_FAILED"
        assert result["cleanup"]["error"] == "cleanup transport failed"
        persisted = json.loads(
            (
                Path(temporary)
                / "monitor"
                / "monitor_result.json"
            ).read_text(encoding="utf-8")
        )
        assert persisted == result


def test_monitor_records_sample_failure_and_keeps_polling():
    monitor = _load_monitor()
    events = iter([
        RuntimeError("temporary transport failure"),
        _ready(2),
        _ready(3),
    ])
    launches = []

    def sample():
        event = next(events)
        if isinstance(event, BaseException):
            raise event
        return event

    with tempfile.TemporaryDirectory() as temporary:
        result = monitor.monitor_until_launch(
            monitor_tag="sample-failure-monitor",
            output_dir=Path(temporary) / "monitor",
            sample_fn=sample,
            launch_fn=lambda: launches.append("launch") or {
                "classification": "PASS",
            },
            cleanup_fn=lambda: {
                "classification": "CLEAN",
                "matched_pids": [],
            },
            sleep_fn=lambda seconds: None,
            interval_s=60,
            required_ready_samples=2,
            max_samples=3,
        )

        rows = [
            json.loads(line)
            for line in (
                Path(temporary)
                / "monitor"
                / "resource_samples.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert rows[0]["classification"] == "SAMPLE_FAILED"
        assert rows[0]["error"] == "temporary transport failure"
        assert launches == ["launch"]
        assert result["trigger_sample_ids"] == [2, 3]


def test_monitor_resumes_after_launch_preflight_race():
    monitor = _load_monitor()
    samples = iter([
        _ready(1),
        _ready(2),
        _ready(3),
        _ready(4),
    ])
    launch_results = iter([
        {"classification": "BLOCKED_RESOURCES"},
        {"classification": "PASS"},
    ])
    cleanup_calls = []
    with tempfile.TemporaryDirectory() as temporary:
        result = monitor.monitor_until_launch(
            monitor_tag="launch-race-monitor",
            output_dir=Path(temporary) / "monitor",
            sample_fn=lambda: next(samples),
            launch_fn=lambda: next(launch_results),
            cleanup_fn=lambda: cleanup_calls.append("cleanup") or {
                "classification": "CLEAN",
                "matched_pids": [],
            },
            sleep_fn=lambda seconds: None,
            interval_s=60,
            required_ready_samples=2,
            max_samples=4,
        )

        assert cleanup_calls == ["cleanup", "cleanup"]
        assert result["classification"] == "PASS"
        assert result["trigger_sample_ids"] == [3, 4]
        attempts = [
            json.loads(line)
            for line in (
                Path(temporary)
                / "monitor"
                / "launch_attempts.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert attempts[0]["classification"] == "BLOCKED_RESOURCES"
        assert attempts[0]["trigger_sample_ids"] == [1, 2]


def test_monitor_stops_when_launch_race_cleanup_fails():
    monitor = _load_monitor()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "monitor"
        result = monitor.monitor_until_launch(
            monitor_tag="launch-race-cleanup-failure",
            output_dir=output_dir,
            sample_fn=lambda: _ready(1),
            launch_fn=lambda: {
                "classification": "BLOCKED_RESOURCES",
            },
            cleanup_fn=lambda: {
                "classification": "CLEANUP_FAILED",
                "remaining_pids": [123],
            },
            sleep_fn=lambda seconds: None,
            interval_s=60,
            required_ready_samples=1,
            max_samples=2,
        )

        assert result["classification"] == "CLEANUP_FAILED"
        assert result["sample_count"] == 1
        assert result["launch_result"]["classification"] == (
            "BLOCKED_RESOURCES"
        )
        assert result["cleanup"]["remaining_pids"] == [123]
        persisted = json.loads(
            (output_dir / "monitor_result.json").read_text(
                encoding="utf-8"
            )
        )
        assert persisted == result


def test_monitor_rejects_unbounded_or_unsafe_configuration():
    monitor = _load_monitor()
    cases = [
        {"interval_s": 59, "required_ready_samples": 2, "max_samples": 3},
        {"interval_s": 60, "required_ready_samples": 0, "max_samples": 3},
        {"interval_s": 60, "required_ready_samples": 2, "max_samples": 0},
    ]
    for case in cases:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                monitor.monitor_until_launch(
                    monitor_tag="invalid",
                    output_dir=Path(temporary) / "monitor",
                    sample_fn=lambda: _blocked(1),
                    launch_fn=lambda: None,
                    cleanup_fn=lambda: None,
                    sleep_fn=lambda seconds: None,
                    **case,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe monitor configuration accepted")


def test_monitor_resumes_existing_sample_ledger_without_reusing_ready_gate():
    monitor = _load_monitor()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "monitor"
        output_dir.mkdir()
        samples_path = output_dir / "resource_samples.jsonl"
        samples_path.write_text(
            "".join(
                json.dumps(
                    _ready(sample_id)
                    if sample_id == 7
                    else _blocked(sample_id),
                    sort_keys=True,
                )
                + "\n"
                for sample_id in range(1, 8)
            ),
            encoding="utf-8",
        )
        sample_ids = []

        def sample():
            sample_id = 8 + len(sample_ids)
            sample_ids.append(sample_id)
            return _ready(sample_id)

        result = monitor.monitor_until_launch(
            monitor_tag="resume-monitor",
            output_dir=output_dir,
            sample_fn=sample,
            launch_fn=lambda: {"classification": "PASS"},
            cleanup_fn=lambda: {"classification": "CLEAN"},
            sleep_fn=lambda seconds: None,
            interval_s=60,
            required_ready_samples=2,
            max_samples=9,
            resume_existing=True,
        )

        assert sample_ids == [8, 9]
        assert result["sample_count"] == 9
        assert result["trigger_sample_ids"] == [8, 9]
        rows = [
            json.loads(line)
            for line in samples_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [row["sample_id"] for row in rows] == list(range(1, 10))


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 strict-P1 monitor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
