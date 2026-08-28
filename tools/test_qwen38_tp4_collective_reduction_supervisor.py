from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import tools.qwen38_tp4_collective_reduction_supervisor as supervisor


ATTEMPT = "20260827-qwen38-tp4-collective-reduction-r1"


def _gpu(index, *, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": 20_000 + index,
        "utilization_percent": 80,
        "compute_processes": list(processes),
    }


def _selected():
    return [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_used_mib": 3,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in range(4)
    ]


def test_runtime_sample_accepts_only_attempt_owned_gpu_processes():
    observed = [
        _gpu(
            index,
            processes=[{
                "pid": 101 + index,
                "process_name": "python",
                "used_memory_mib": 20_000 + index,
            }],
        )
        for index in range(4)
    ]

    sample = supervisor.build_runtime_sample(
        case_id="case-1",
        selected_gpus=_selected(),
        observed_gpus=observed,
        owned_pids={101, 102, 103, 104},
        captured_at_unix_ns=123,
    )

    assert sample["owned_pids"] == [101, 102, 103, 104]
    assert sample["selected_gpus"] == observed

    observed[0]["compute_processes"].append({
        "pid": 999,
        "process_name": "foreign",
        "used_memory_mib": 1,
    })
    with pytest.raises(
        ValueError,
        match=(
            "foreign GPU process detected: "
            "gpu_uuid=GPU-0 pid=999 process_name=foreign"
        ),
    ):
        supervisor.build_runtime_sample(
            case_id="case-1",
            selected_gpus=_selected(),
            observed_gpus=observed,
            owned_pids={101, 102, 103, 104},
            captured_at_unix_ns=124,
        )


def test_process_group_scan_ignores_process_exit_race(monkeypatch):
    class VanishedProcess:
        name = "123"

        def __truediv__(self, name):
            assert name == "stat"
            return self

        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            raise ProcessLookupError(3, "No such process")

    class ProcRoot:
        def __init__(self, path):
            assert path == "/proc"

        def iterdir(self):
            return [VanishedProcess()]

    monkeypatch.setattr(supervisor, "Path", ProcRoot)

    assert supervisor.process_group_pids(101) == []


def test_exact_tag_scan_ignores_process_exit_race(monkeypatch):
    class VanishedProcess:
        name = "123"

        def __truediv__(self, name):
            assert name == "cmdline"
            return self

        def read_bytes(self):
            raise ProcessLookupError(3, "No such process")

    class ProcRoot:
        def __init__(self, path):
            assert path == "/proc"

        def iterdir(self):
            return [VanishedProcess()]

    monkeypatch.setattr(supervisor, "Path", ProcRoot)
    monkeypatch.setattr(supervisor.os, "getpid", lambda: 999)

    assert supervisor.exact_tag_processes(ATTEMPT) == []


def test_worker_environment_keeps_writable_paths_below_attempt(tmp_path):
    attempt_root = tmp_path / ATTEMPT

    environment = supervisor.build_worker_environment(
        attempt_root=attempt_root,
        selected_gpus=_selected(),
        base_environment={
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/approved/existing/lib",
        },
        dist_port=29671,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["TINYVLLM_DIST_PORT"] == "29671"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["LD_LIBRARY_PATH"] == (
        "/data00/home/sitian/tllm/miniforge/lib"
        ":/approved/existing/lib"
    )
    assert environment["PYTHONPATH"] == str(attempt_root / "source")
    for name in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
    ):
        assert Path(environment[name]).is_relative_to(attempt_root)


class _FakeProcess:
    def __init__(self):
        self.pid = 101
        self._polls = iter((None, 0))

    def poll(self):
        return next(self._polls)

    def wait(self):
        return 0


class _TwoPollProcess:
    def __init__(self):
        self.pid = 101
        self._polls = iter((None, None, 0))

    def poll(self):
        return next(self._polls)

    def wait(self):
        return 0


def test_supervisor_retains_known_owned_pids_during_gpu_telemetry_lag(
    tmp_path,
):
    attempt_root = tmp_path / ATTEMPT
    controller = attempt_root / "controller"
    cases = attempt_root / "cases"
    source = attempt_root / "source"
    controller.mkdir(parents=True)
    cases.mkdir()
    source.mkdir()
    (attempt_root / "worker.json").write_text(
        json.dumps({
            "classification": "PASS",
            "attempt": ATTEMPT,
            "source_revision": "a" * 40,
            "selected_budget": None,
            "owned_pids": [101],
            "cases": [{
                "case_id": "case-1",
                "classification": "PASS",
            }],
            "phase_cleanups": [{
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            }],
        }),
        encoding="utf-8",
    )
    process_groups = iter((
        [101, 102],
        [101, 102],
        [101],
        [101],
        [],
    ))

    receipt = supervisor.supervise_worker(
        attempt=ATTEMPT,
        source_revision="a" * 40,
        attempt_root=attempt_root,
        source_root=source,
        model_root=tmp_path / "model",
        python_path=Path("/approved/read-only/python"),
        selected_gpus=_selected(),
        dist_port=29671,
        poll_interval_s=1,
        worker_timeout_s=10,
        launcher=lambda *_args, **_kwargs: _TwoPollProcess(),
        inventory_query=lambda: [
            _gpu(
                index,
                processes=[{
                    "pid": 102,
                    "process_name": "python",
                    "used_memory_mib": 20_000,
                }],
            )
            if index == 0 else _gpu(index)
            for index in range(4)
        ],
        pgid_resolver=lambda pid: pid,
        process_group_pids=lambda _pgid: next(process_groups),
        exact_tag_scan=lambda _tag: [],
        sleep=lambda _seconds: None,
        clock_ns=iter(range(100, 1000)).__next__,
        monotonic=iter((0.0, 1.0, 2.0, 3.0, 4.0)).__next__,
    )

    assert receipt["classification"] == "PASS"
    assert receipt["violations"] == []


def test_supervisor_writes_resource_and_cleanup_evidence_without_signals(
    tmp_path,
):
    attempt_root = tmp_path / ATTEMPT
    controller = attempt_root / "controller"
    cases = attempt_root / "cases"
    source = attempt_root / "source"
    controller.mkdir(parents=True)
    cases.mkdir()
    source.mkdir()
    worker_result = {
        "classification": "PASS",
        "attempt": ATTEMPT,
        "source_revision": "a" * 40,
        "selected_budget": None,
        "owned_pids": [101],
        "cases": [
            {"case_id": "case-1", "classification": "PASS"},
            {"case_id": "case-2", "classification": "PASS"},
        ],
        "phase_cleanups": [{
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
        }],
    }
    (attempt_root / "worker.json").write_text(
        json.dumps(worker_result),
        encoding="utf-8",
    )
    process = _FakeProcess()
    launches = []
    sleeps = []
    scans = iter(([], [], []))
    process_groups = iter((
        [101],
        [101, 102, 103, 104],
        [],
    ))

    receipt = supervisor.supervise_worker(
        attempt=ATTEMPT,
        source_revision="a" * 40,
        attempt_root=attempt_root,
        source_root=source,
        model_root=tmp_path / "model",
        python_path=Path("/approved/read-only/python"),
        selected_gpus=_selected(),
        dist_port=29671,
        poll_interval_s=1,
        worker_timeout_s=10,
        launcher=lambda *args, **kwargs: (
            launches.append((args, kwargs)) or process
        ),
        inventory_query=lambda: [
            _gpu(
                index,
                processes=[{
                    "pid": 101 + index,
                    "process_name": "python",
                    "used_memory_mib": 20_000,
                }],
            )
            for index in range(4)
        ],
        pgid_resolver=lambda pid: pid,
        process_group_pids=lambda _pgid: next(process_groups),
        exact_tag_scan=lambda _tag: next(scans),
        sleep=lambda seconds: sleeps.append(seconds),
        clock_ns=iter(range(100, 1000)).__next__,
        monotonic=iter((0.0, 1.0, 2.0, 3.0, 4.0)).__next__,
    )

    assert receipt["classification"] == "PASS"
    assert receipt["worker_returncode"] == 0
    assert receipt["owned_children_remaining"] == []
    assert receipt["exact_tag_scans"] == [[], [], []]
    assert len(launches) == 1
    assert launches[0][1]["start_new_session"] is True
    samples = json.loads(
        (controller / "resource_samples.json").read_text()
    )
    assert [row["case_id"] for row in samples] == [
        "case-1",
        "case-2",
    ]
    assert all(row["owned_pids"] == [101, 102, 103, 104] for row in samples)
    cleanup = json.loads((controller / "cleanup.json").read_text())
    assert cleanup == {
        "schema_version": (
            "qwen38.tp4-collective-reduction-cleanup.v1"
        ),
        "complete": True,
        "process_group_destroyed": True,
        "owned_children_remaining": [],
        "exact_tag_scans": [[], [], []],
    }


def test_supervisor_implementation_has_no_signal_or_kill_path():
    source = inspect.getsource(supervisor)

    assert "import signal" not in source
    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
