import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

import run_tp4_segmented_capture_census as controller


def _source(run_tag="census-r1"):
    return {
        "schema_version": controller.SOURCE_SCHEMA,
        "run_tag": run_tag,
        "source_revision": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "model_repository": controller.MODEL_REPOSITORY,
        "model_revision": controller.MODEL_REVISION,
    }


def _gpus():
    return [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in range(4)
    ]


def test_plan_keeps_every_remote_path_under_large_mount():
    plan = controller.build_plan(
        run_tag="census-r1",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )
    root = PurePosixPath(controller.REMOTE_ROOT)
    assert plan["admission_mode"] == "strict_clean"
    assert plan["selected_gpu_indices"] == [0, 1, 2, 3]
    assert all(
        PurePosixPath(path).is_relative_to(root)
        for path in plan["paths"].values()
    )
    assert all(
        PurePosixPath(path).is_relative_to(root)
        for path in plan["environment"].values()
    )


def test_plan_rejects_non_strict_admission_and_unsafe_tag():
    with pytest.raises(ValueError, match="strict_clean"):
        controller.build_plan(
            run_tag="census-r1",
            source_identity=_source(),
            selected_gpus=_gpus(),
            admission_mode="shared_capacity",
        )
    with pytest.raises(ValueError, match="run tag"):
        controller.build_plan(
            run_tag="../escape",
            source_identity=_source("../escape"),
            selected_gpus=_gpus(),
            admission_mode="strict_clean",
        )


class _Adapter:
    def __init__(self, *, fail_at=None):
        self.fail_at = fail_at
        self.events = []

    def _step(self, name, value):
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")
        return value

    def freeze_source(self, seed):
        return self._step("freeze", _source(seed["run_tag"]))

    def ssh_storage_preflight(self, seed, source):
        del source
        return self._step(
            "preflight",
            {
                "classification": "PASS",
                "attempt_exists": False,
                "remote_root": seed["remote_root"],
            },
        )

    def gpu_admission(self, plan, preflight):
        del preflight
        return self._step(
            "admission",
            {
                "classification": "READY",
                "selected_gpus": plan["selected_gpus"],
            },
        )

    def launch(self, plan, admission):
        del plan, admission
        return self._step("launch", {"owned_pids": [123]})

    def wait(self, plan, launch):
        del plan, launch
        return self._step("wait", {"exit_code": 0})

    def download(self, plan, waited):
        del plan, waited
        return self._step("download", {"downloaded": True})

    def remote_verify(self, plan, downloaded):
        del plan, downloaded
        return self._step(
            "remote_verify",
            {"classification": "GO_SEGMENT_PLAN_SELECTED"},
        )

    def local_verify(self, plan, downloaded):
        del plan, downloaded
        return self._step(
            "local_verify",
            {"classification": "GO_SEGMENT_PLAN_SELECTED"},
        )

    def validate_cleanup(self, plan, launch):
        del plan, launch
        return self._step(
            "cleanup",
            {
                "classification": "CLEAN",
                "exact_tag_scans": [[], [], []],
            },
        )


def test_attempt_runs_both_verifiers_and_always_validates_cleanup():
    adapter = _Adapter()
    result = controller.monitor_and_run(
        run_tag="census-r1",
        admission_mode="strict_clean",
        gpu_monitor=lambda: {
            "classification": "READY",
            "selected_gpus": _gpus(),
        },
        adapter=adapter,
    )
    assert result["classification"] == "GO_SEGMENT_PLAN_SELECTED"
    assert adapter.events == [
        "freeze",
        "preflight",
        "admission",
        "launch",
        "wait",
        "download",
        "remote_verify",
        "local_verify",
        "cleanup",
    ]

    failing = _Adapter(fail_at="wait")
    with pytest.raises(RuntimeError, match="wait failed"):
        controller.monitor_and_run(
            run_tag="census-r2",
            admission_mode="strict_clean",
            gpu_monitor=lambda: {
                "classification": "READY",
                "selected_gpus": _gpus(),
            },
            adapter=failing,
        )
    assert failing.events[-1] == "cleanup"


def test_attempt_rejects_verifier_plan_disagreement():
    class DisagreeingAdapter(_Adapter):
        def remote_verify(self, plan, downloaded):
            del plan, downloaded
            return self._step(
                "remote_verify",
                {
                    "classification": "GO_SEGMENT_PLAN_SELECTED",
                    "selected_plan_id": "p2",
                    "selected_plan_sha256": "2" * 64,
                },
            )

        def local_verify(self, plan, downloaded):
            del plan, downloaded
            return self._step(
                "local_verify",
                {
                    "classification": "GO_SEGMENT_PLAN_SELECTED",
                    "selected_plan_id": "p3",
                    "selected_plan_sha256": "3" * 64,
                },
            )

    adapter = DisagreeingAdapter()
    with pytest.raises(RuntimeError, match="verifiers disagree"):
        controller.monitor_and_run(
            run_tag="census-verifier-disagreement",
            admission_mode="strict_clean",
            gpu_monitor=lambda: {
                "classification": "READY",
                "selected_gpus": _gpus(),
            },
            adapter=adapter,
        )
    assert adapter.events[-1] == "cleanup"


def test_main_wires_monitor_and_production_adapter_contract(
    tmp_path,
    capsys,
):
    adapter = _Adapter()
    exit_code = controller.main(
        [
            "monitor-and-run",
            "--run-tag",
            "census-r3",
            "--admission-mode",
            "strict_clean",
            "--local-attempt-root",
            str(tmp_path / "attempt"),
        ],
        gpu_monitor=lambda: {
            "classification": "READY",
            "selected_gpus": _gpus(),
        },
        adapter_factory=lambda **kwargs: adapter,
    )
    assert exit_code == 0
    assert '"classification": "GO_SEGMENT_PLAN_SELECTED"' in (
        capsys.readouterr().out
    )


def test_validate_cleanup_reaps_only_exact_tag_owned_processes(tmp_path):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path
    adapter._cleanup = None
    scans = iter(
        (
            [
                {
                    "pid": 123,
                    "command": "python census-r4",
                    "matched_cmdline": True,
                    "matched_environment": True,
                }
            ],
            [],
            [],
            [],
        )
    )
    reaped = []
    adapter._scan_exact_tag = lambda plan: next(scans)
    adapter._reap_exact_tag = (
        lambda plan, rows: reaped.append((plan["run_tag"], rows))
        or {
            "requested_pids": [123],
            "terminated_pids": [123],
            "killed_pids": [],
            "remaining_pids": [],
        }
    )
    adapter._process = SimpleNamespace(
        poll=lambda: 0,
    )
    plan = controller.build_plan(
        run_tag="census-r4",
        source_identity=_source("census-r4"),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    cleanup = adapter.validate_cleanup(plan, {"owned_pids": [999]})

    assert [entry[0] for entry in reaped] == ["census-r4"]
    assert [row["pid"] for row in reaped[0][1]] == [123]
    assert cleanup["classification"] == "CLEAN"
    assert cleanup["exact_tag_scans"][0][0]["pid"] == 123
    assert cleanup["exact_tag_scans"][1:] == [[], [], []]
    assert cleanup["reap_receipt"]["remaining_pids"] == []


def test_validate_cleanup_preserves_failed_rank_lifecycle(tmp_path):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path
    adapter._cleanup = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-capture-cleanup.v1"
        ),
        "run_tag": "census-r5",
        "classification": "DIRTY",
        "owned_children_remaining": [],
        "exact_tag_scans": [[], [], []],
        "rank_rows": [
            {
                "rank": rank,
                "exit_code": 1 if rank == 2 else 0,
                "process_group_destroyed": rank != 2,
            }
            for rank in range(4)
        ],
    }
    adapter._scan_exact_tag = lambda plan: []
    adapter._process = SimpleNamespace(poll=lambda: 1)
    plan = controller.build_plan(
        run_tag="census-r5",
        source_identity=_source("census-r5"),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    cleanup = adapter.validate_cleanup(plan, {"owned_pids": [999]})

    assert cleanup["classification"] == "DIRTY"
    assert cleanup["final_exact_tag_scans"] == [[], [], []]


def test_exact_run_tag_environment_match_is_not_a_substring():
    assert controller._matches_exact_tag_identity(
        b"python worker.py\x00",
        b"TINYLLMFORGE_RUN_TAG=census-r6\x00",
        run_tag="census-r6",
        attempt_root="/remote/census-r6",
    )
    assert not controller._matches_exact_tag_identity(
        b"python worker.py\x00",
        b"TINYLLMFORGE_RUN_TAG=census-r6-extra\x00",
        run_tag="census-r6",
        attempt_root="/remote/census-r6",
    )
    assert controller._matches_exact_tag_identity(
        b"python\x00/remote/census-r6/source/worker.py\x00",
        b"",
        run_tag="census-r6",
        attempt_root="/remote/census-r6",
    )
    assert not controller._matches_exact_tag_identity(
        b"python\x00/remote/census-r6-extra/source/worker.py\x00",
        b"",
        run_tag="census-r6",
        attempt_root="/remote/census-r6",
    )


def test_freeze_source_rejects_existing_local_attempt_root(
    tmp_path,
    monkeypatch,
):
    attempt_root = tmp_path / "existing"
    attempt_root.mkdir()
    (attempt_root / "immutable.txt").write_text(
        "preserve",
        encoding="utf-8",
    )
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_attempt_root = attempt_root
    adapter.local_controller_root = attempt_root / "controller"
    adapter._source = None
    monkeypatch.setattr(
        controller,
        "_capture_source_identity",
        lambda run_tag: _source(run_tag),
    )

    with pytest.raises(ValueError, match="local attempt"):
        adapter.freeze_source({"run_tag": "census-existing"})

    assert (attempt_root / "immutable.txt").read_text(
        encoding="utf-8"
    ) == "preserve"


def test_kerberos_preflight_failure_is_persisted(tmp_path):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path
    adapter._query_kerberos_window = lambda: {
        "classification": "BLOCKED_KERBEROS_TTL",
        "remaining_lifetime_seconds": 22_123,
        "minimum_required_lifetime_seconds": 22_500,
    }

    receipt = adapter.ssh_storage_preflight(
        {"run_tag": "census-ttl"},
        _source("census-ttl"),
    )

    assert receipt["classification"] == "INCOMPLETE"
    assert receipt["reason"] == "Kerberos TTL preflight failed"
    assert json.loads(
        (tmp_path / "ssh_storage_preflight.json").read_text(
            encoding="utf-8"
        )
    ) == receipt


def test_wait_finalizes_cleanup_before_bundle_download(tmp_path):
    class Process:
        returncode = 0

        @staticmethod
        def communicate(timeout):
            assert timeout == 30
            return ("", "")

        @staticmethod
        def poll():
            return 0

    adapter = object.__new__(controller.ProductionAdapter)
    adapter._process = Process()
    adapter.command_timeout_s = 30
    adapter.local_controller_root = tmp_path
    adapter._cleanup = None
    calls = []
    adapter.validate_cleanup = (
        lambda plan, launch: calls.append((plan, launch))
        or {
            "classification": "CLEAN",
            "final_exact_tag_scans": [[], [], []],
        }
    )
    plan = {
        "run_tag": "census-wait-cleanup",
        "paths": {
            "worker_stderr_path": "/remote/unused.stderr",
        },
    }
    launch = {"owned_pids": [123]}

    result = adapter.wait(plan, launch)

    assert result["exit_code"] == 0
    assert calls == [(plan, launch)]


def test_gpu_admission_preserves_strict_clean_measurements(
    tmp_path,
    monkeypatch,
):
    observed = _gpus()
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path
    adapter.ssh_target = "unused"
    adapter.control_path = None
    adapter.command_timeout_s = 30
    adapter.retry_count = 1
    adapter._require_kerberos_window = lambda: {"classification": "READY"}
    monkeypatch.setattr(
        controller,
        "query_remote_gpu_inventory",
        lambda **kwargs: observed,
    )
    monkeypatch.setattr(
        controller,
        "validate_selected_gpu_processes",
        lambda **kwargs: observed,
    )
    plan = controller.build_plan(
        run_tag="census-admission",
        source_identity=_source("census-admission"),
        selected_gpus=observed,
        admission_mode="strict_clean",
    )

    adapter.gpu_admission(plan, {})

    assert adapter._admission["selected_gpus"] == [
        {
            "rank": rank,
            "index": rank,
            "uuid": f"GPU-{rank}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for rank in range(4)
    ]
