from __future__ import annotations

import io
import json
from pathlib import PurePosixPath
import signal
from types import SimpleNamespace

import pytest

import tools.run_lease_sealed_state_commit_overlap as controller_module
from tools.run_lease_sealed_state_commit_overlap import (
    APPROVED_REMOTE_ROOT,
    _prepare_local_attempt_root,
    _run_with_terminal_receipt,
    _stage_committed_source,
    _terminate_owned_process_groups,
    build_attempt_plan,
    build_remote_worker_commands,
    run_attempt,
    run_ssh_with_retry,
    supervise_remote_workers,
    verify_local_downloaded_bundle,
)
from tools.assemble_lease_sealed_state_commit_overlap import assemble_bundle
from tools.test_assemble_lease_sealed_state_commit_overlap import clone_inputs
from tools.verify_lease_sealed_state_commit_overlap import verify_bundle


def gpu(index, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def plan(**overrides):
    values = {
        "attempt_tag": (
            "20260907-lease-sealed-state-commit-overlap-stage0-r1"
        ),
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "selected_gpus": [gpu(index) for index in range(4)],
        "remote_path_state": {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
            "remote_root_exists": True,
            "remote_root_is_directory": True,
            "remote_root_on_distinct_filesystem": True,
            "resolved_remote_root": APPROVED_REMOTE_ROOT,
            "resolved_attempt_root": (
                f"{APPROVED_REMOTE_ROOT}/attempts/"
                "20260907-lease-sealed-state-commit-overlap-stage0-r1"
            ),
        },
    }
    values.update(overrides)
    return build_attempt_plan(**values)


def test_every_remote_path_is_below_approved_mount():
    root = PurePosixPath(APPROVED_REMOTE_ROOT)
    candidate = plan()

    for name in (
        "attempt_root",
        "source_root",
        "raw_root",
        "bundle_root",
        "controller_root",
    ):
        assert PurePosixPath(candidate[name]).is_relative_to(root)
    for path in candidate["environment"].values():
        assert PurePosixPath(path).is_relative_to(
            PurePosixPath(candidate["attempt_root"])
        )


def test_plan_requires_fresh_path_and_four_strict_clean_gpus():
    with pytest.raises(ValueError, match="fresh"):
        plan(remote_path_state={
            "attempt_exists": True,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
            "remote_root_exists": True,
            "remote_root_is_directory": True,
            "remote_root_on_distinct_filesystem": True,
            "resolved_remote_root": APPROVED_REMOTE_ROOT,
            "resolved_attempt_root": (
                f"{APPROVED_REMOTE_ROOT}/attempts/"
                "20260907-lease-sealed-state-commit-overlap-stage0-r1"
            ),
        })
    with pytest.raises(ValueError, match="four strict-clean"):
        plan(selected_gpus=[
            gpu(0),
            gpu(1),
            gpu(2),
            gpu(3, memory=1025),
        ])


def test_plan_rejects_remote_root_on_root_filesystem():
    state = {
        "attempt_exists": False,
        "attempt_parent_is_symlink": False,
        "remote_root_is_symlink": False,
        "remote_root_exists": True,
        "remote_root_is_directory": True,
        "remote_root_on_distinct_filesystem": False,
        "resolved_remote_root": APPROVED_REMOTE_ROOT,
        "resolved_attempt_root": (
            f"{APPROVED_REMOTE_ROOT}/attempts/"
            "20260907-lease-sealed-state-commit-overlap-stage0-r1"
        ),
    }

    with pytest.raises(ValueError, match="mounted filesystem"):
        plan(remote_path_state=state)


def test_worker_commands_freeze_rank_world_size_port_and_gpu_mapping():
    commands = build_remote_worker_commands(plan())

    assert len(commands) == 4
    for rank, command in enumerate(commands):
        assert f"--rank {rank}" in command
        assert "--world-size 4" in command
        assert "--dist-port 29741" in command
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in command


def test_run_attempt_checks_auth_then_gpu_twice_and_both_verifiers():
    events = []
    clean = [gpu(index) for index in range(4)]
    result = run_attempt(
        plan(),
        kerberos_probe=lambda: events.append("kerberos")
        or {"classification": "PASS"},
        gpu_probe=lambda: events.append("gpu") or clean,
        remote_writer=lambda _plan: events.append("write")
        or {"classification": "PASS"},
        launch_admission_writer=lambda _plan, _observed: events.append(
            "write_launch_admission"
        )
        or {"classification": "PASS"},
        worker_runner=lambda _plan: events.append("worker")
        or {"classification": "PASS"},
        assembler=lambda _plan: events.append("assemble")
        or {"classification": "GO_LEASE_SEALED_OVERLAP_MICROGATE"},
        remote_verifier=lambda _plan: events.append("remote_verify")
        or {
            "status": "PASS",
            "reconstructed_classification": (
                "GO_LEASE_SEALED_OVERLAP_MICROGATE"
            ),
        },
        downloader=lambda _plan: events.append("download")
        or {"classification": "PASS"},
        local_verifier=lambda _plan: events.append("local_verify")
        or {
            "status": "PASS",
            "reconstructed_classification": (
                "GO_LEASE_SEALED_OVERLAP_MICROGATE"
            ),
        },
    )

    assert result["classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert events == [
        "kerberos",
        "gpu",
        "write",
        "gpu",
        "write_launch_admission",
        "worker",
        "assemble",
        "remote_verify",
        "download",
        "local_verify",
    ]


def test_expired_auth_stops_before_remote_or_gpu_access():
    events = []
    result = run_attempt(
        plan(),
        kerberos_probe=lambda: events.append("kerberos")
        or {"classification": "BLOCKED"},
        gpu_probe=lambda: events.append("gpu") or [],
        remote_writer=lambda _plan: events.append("write"),
        worker_runner=lambda _plan: events.append("worker"),
    )
    assert result["classification"] == "BLOCKED_KERBEROS"
    assert result["worker_started"] is False
    assert events == ["kerberos"]


def test_second_admission_requires_the_frozen_four_gpus_to_stay_clean():
    initially_clean = [gpu(index) for index in range(4)]
    selected_gpu_dirty = [
        gpu(0, memory=1025),
        gpu(1),
        gpu(2),
        gpu(3),
        gpu(4),
        gpu(5),
        gpu(6),
        gpu(7),
    ]
    observations = iter((initially_clean, selected_gpu_dirty))
    events = []

    with pytest.raises(ValueError, match="strict-clean"):
        run_attempt(
            plan(),
            kerberos_probe=lambda: {"classification": "PASS"},
            gpu_probe=lambda: next(observations),
            remote_writer=lambda _plan: {"classification": "PASS"},
            launch_admission_writer=lambda _plan, _observed: events.append(
                "write_launch_admission"
            )
            or {"classification": "PASS"},
            worker_runner=lambda _plan: events.append("worker")
            or {"classification": "PASS"},
            assembler=lambda _plan: {},
            remote_verifier=lambda _plan: {},
            downloader=lambda _plan: {},
            local_verifier=lambda _plan: {},
        )

    assert events == []


def test_ssh_255_retries_only_within_fixed_budget():
    returncodes = iter((255, 255, 0))
    calls = []
    result = run_ssh_with_retry(
        ["ssh", "host", "true"],
        retry_count=2,
        runner=lambda argv, **kwargs: (
            calls.append(list(argv))
            or SimpleNamespace(
                returncode=next(returncodes),
                stdout="",
                stderr="",
            )
        ),
    )
    assert result.returncode == 0
    assert len(calls) == 3


def test_local_verifier_preserves_remote_receipt_and_seals_bundle(tmp_path):
    bundle = tmp_path / "final_bundle"
    assemble_bundle(output_root=bundle, **clone_inputs())
    verify_bundle(
        bundle,
        receipt_name="remote_independent_verification.json",
    )
    remote_bytes = (
        bundle / "remote_independent_verification.json"
    ).read_bytes()

    result = verify_local_downloaded_bundle(bundle)

    assert result["status"] == "PASS"
    assert (
        bundle / "remote_independent_verification.json"
    ).read_bytes() == remote_bytes
    assert (
        bundle / "local_streaming_independent_verification.json"
    ).is_file()
    assert (bundle / "manifest.json").is_file()


def test_local_attempt_root_is_fresh_and_failures_get_terminal_receipt(
    tmp_path,
):
    attempt_root = tmp_path / "attempt"
    controller_root = _prepare_local_attempt_root(attempt_root)
    receipt_path = controller_root / "result.json"

    def fail():
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        _run_with_terminal_receipt(receipt_path, fail)

    receipt = __import__("json").loads(receipt_path.read_text())
    assert receipt["classification"] == "CONTROLLER_ERROR"
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["worker_started"] is None
    with pytest.raises(ValueError, match="fresh"):
        _prepare_local_attempt_root(attempt_root)


def test_timeout_cleanup_signals_only_owned_process_groups():
    processes = [
        SimpleNamespace(pid=101),
        SimpleNamespace(pid=202),
    ]
    signals = []
    alive = {101, 202}

    def signal_group(pgid, signal_number):
        signals.append((pgid, signal_number))
        if signal_number != 0:
            alive.discard(pgid)
        elif pgid not in alive:
            raise ProcessLookupError

    _terminate_owned_process_groups(
        processes,
        get_process_group=lambda pid: pid,
        signal_group=signal_group,
        sleeper=lambda _seconds: None,
    )

    assert {pgid for pgid, _signal in signals} == {101, 202}


def test_cleanup_uses_registered_group_after_leader_exits():
    process = SimpleNamespace(pid=101)
    signals = []
    alive = {101}

    def signal_group(pgid, signal_number):
        signals.append((pgid, signal_number))
        if signal_number == signal.SIGTERM:
            alive.discard(pgid)
        elif signal_number == 0 and pgid not in alive:
            raise ProcessLookupError

    _terminate_owned_process_groups(
        [process],
        owned_process_groups=(101,),
        get_process_group=lambda _pid: (_ for _ in ()).throw(
            ProcessLookupError
        ),
        signal_group=signal_group,
        sleeper=lambda _seconds: None,
    )

    assert signals == [
        (101, signal.SIGTERM),
        (101, 0),
    ]


def test_resource_identity_violation_stops_owned_workers_immediately(
    tmp_path,
    monkeypatch,
):
    current = plan()
    current.update({
        "source_root": str(tmp_path / "source"),
        "raw_root": str(tmp_path / "raw"),
        "controller_root": str(tmp_path / "controller"),
        "environment": {
            "TMPDIR": str(tmp_path / "runtime" / "tmp"),
        },
    })
    processes = [
        SimpleNamespace(pid=100 + rank, returncode=None)
        for rank in range(4)
    ]
    launched = iter(processes)
    terminated = []

    for process in processes:
        process.poll = lambda process=process: process.returncode
        process.wait = lambda process=process: process.returncode

    monkeypatch.setattr(
        controller_module,
        "_validate_plan",
        lambda _plan: tuple(gpu(index) for index in range(4)),
    )
    monkeypatch.setattr(
        controller_module,
        "build_remote_worker_commands",
        lambda *_args, **_kwargs: tuple("worker" for _ in range(4)),
    )
    monkeypatch.setattr(
        controller_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(launched),
    )
    monkeypatch.setattr(
        controller_module,
        "_descendant_pids",
        lambda parents: set(parents),
    )
    monkeypatch.setattr(
        controller_module,
        "_remote_inventory_local",
        lambda: (_ for _ in ()).throw(
            RuntimeError("unrelated GPU process detected")
        ),
    )
    monkeypatch.setattr(
        controller_module,
        "_exact_tag_worker_pids",
        lambda _attempt: [],
    )

    def terminate(owned, **_kwargs):
        terminated.append([process.pid for process in owned])
        for process in owned:
            process.returncode = -signal.SIGTERM

    monkeypatch.setattr(
        controller_module,
        "_terminate_owned_process_groups",
        terminate,
    )
    monkeypatch.setattr(
        controller_module.time,
        "sleep",
        lambda _seconds: [
            setattr(process, "returncode", process.returncode or 0)
            for process in processes
        ],
    )

    result = supervise_remote_workers(
        current,
        poll_interval_s=0,
    )

    assert terminated == [[100, 101, 102, 103]]
    assert result["classification"] == "FAIL"
    assert result["violations"] == [
        "RuntimeError: unrelated GPU process detected"
    ]


def test_early_controller_exception_writes_terminal_receipt(
    tmp_path,
    monkeypatch,
):
    attempt_root = tmp_path / "attempt"
    monkeypatch.setattr(
        controller_module,
        "query_local_kerberos",
        lambda **_kwargs: {"classification": "READY"},
    )
    monkeypatch.setattr(
        controller_module,
        "capture_source_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("source capture failed")
        ),
    )

    with pytest.raises(RuntimeError, match="source capture failed"):
        controller_module.main([
            "--attempt-tag",
            "early-failure",
            "--source-revision",
            "a" * 40,
            "--local-attempt-root",
            str(attempt_root),
        ])

    receipt = json.loads(
        (attempt_root / "controller" / "result.json").read_text()
    )
    assert receipt["classification"] == "CONTROLLER_ERROR"
    assert receipt["worker_started"] is None
    assert receipt["error_type"] == "RuntimeError"


def test_remote_supervisor_exception_writes_terminal_receipt(
    tmp_path,
    monkeypatch,
):
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    current = {
        "controller_root": str(controller_root),
    }
    monkeypatch.setattr(
        controller_module,
        "build_attempt_plan",
        lambda **_kwargs: current,
    )
    monkeypatch.setattr(
        controller_module,
        "supervise_remote_workers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("supervision failed")
        ),
    )

    with pytest.raises(RuntimeError, match="supervision failed"):
        controller_module.main([
            "--remote-supervise",
            "--attempt-tag",
            "remote-failure",
            "--source-revision",
            "a" * 40,
            "--source-tree-sha256",
            "b" * 64,
            "--selected-gpus-json",
            "[]",
        ])

    receipt = json.loads(
        (controller_root / "supervisor_receipt.json").read_text()
    )
    assert receipt["classification"] == "CONTROLLER_ERROR"
    assert receipt["error_type"] == "RuntimeError"


def test_remote_supervisor_supplies_complete_validated_path_state(
    tmp_path,
    monkeypatch,
):
    captured = {}
    controller_root = tmp_path / "controller"
    controller_root.mkdir()

    def fake_build_attempt_plan(**kwargs):
        captured.update(kwargs)
        return {"controller_root": str(controller_root)}

    monkeypatch.setattr(
        controller_module,
        "build_attempt_plan",
        fake_build_attempt_plan,
    )
    monkeypatch.setattr(
        controller_module,
        "supervise_remote_workers",
        lambda *_args, **_kwargs: {"classification": "PASS"},
    )

    assert controller_module.main([
        "--remote-supervise",
        "--attempt-tag",
        "remote-path-state",
        "--source-revision",
        "a" * 40,
        "--source-tree-sha256",
        "b" * 64,
        "--selected-gpus-json",
        "[]",
    ]) == 0
    assert captured["remote_path_state"] == {
        "attempt_exists": False,
        "attempt_parent_is_symlink": False,
        "remote_root_is_symlink": False,
        "remote_root_exists": True,
        "remote_root_is_directory": True,
        "remote_root_on_distinct_filesystem": True,
        "resolved_remote_root": APPROVED_REMOTE_ROOT,
        "resolved_attempt_root": (
            f"{APPROVED_REMOTE_ROOT}/attempts/remote-path-state"
        ),
    }


def test_source_staging_timeout_terminates_owned_archive_process(
    tmp_path,
    monkeypatch,
):
    class FakeArchive:
        def __init__(self):
            self.stdout = io.BytesIO(b"archive")
            self.stderr = io.BytesIO()
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return -signal.SIGTERM

    archive = FakeArchive()
    monkeypatch.setattr(
        controller_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: archive,
    )
    monkeypatch.setattr(
        controller_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            controller_module.subprocess.TimeoutExpired("ssh", 1)
        ),
    )

    with pytest.raises(controller_module.subprocess.TimeoutExpired):
        _stage_committed_source(
            {
                "source_revision": "a" * 40,
                "source_root": "/data00/source",
            },
            repo_root=tmp_path,
            ssh_target="host",
            proxy_host="proxy",
            timeout_s=1,
        )

    assert archive.stdout.closed
    assert archive.terminated is True
    assert archive.waited is True


def test_cli_dry_run_is_forwarded_to_execution_gate(tmp_path, monkeypatch):
    clean = [gpu(index) for index in range(4)]
    captured = {}
    monkeypatch.setattr(
        controller_module,
        "query_local_kerberos",
        lambda **_kwargs: {"classification": "READY"},
    )
    monkeypatch.setattr(
        controller_module,
        "capture_source_identity",
        lambda **_kwargs: {
            "schema_version": (
                "lease-sealed-state-commit-overlap-source.v1"
            ),
            "attempt": "dry-run",
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        controller_module,
        "query_remote_path_state",
        lambda **_kwargs: {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
            "remote_root_exists": True,
            "remote_root_is_directory": True,
            "remote_root_on_distinct_filesystem": True,
            "resolved_remote_root": APPROVED_REMOTE_ROOT,
            "resolved_attempt_root": (
                f"{APPROVED_REMOTE_ROOT}/attempts/dry-run"
            ),
        },
    )
    monkeypatch.setattr(
        controller_module,
        "wait_for_strict_clean_gpus",
        lambda **_kwargs: {
            "classification": "READY",
            "selected_gpus": clean,
        },
    )

    def fake_run_attempt(_plan, **kwargs):
        captured.update(kwargs)
        return {
            "classification": "DRY_RUN_READY",
            "worker_started": False,
        }

    monkeypatch.setattr(
        controller_module,
        "run_attempt",
        fake_run_attempt,
    )

    assert controller_module.main([
        "--attempt-tag",
        "dry-run",
        "--source-revision",
        "a" * 40,
        "--local-attempt-root",
        str(tmp_path / "attempt"),
        "--dry-run",
    ]) == 0
    assert captured["dry_run"] is True
