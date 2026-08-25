from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import (
    run_exact_burst_generation_sealed_lease_identity_remote as remote,
)


def _gpu(index, *, memory=0, utilization=0, processes=()):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _args(tmp_path, *, mode="--launch", run_tag="fresh"):
    return remote.parse_args([
        mode,
        "--run-tag",
        run_tag,
        "--source-sha",
        "a" * 40,
        "--local-output-dir",
        str(tmp_path / run_tag),
    ])


def test_paths_and_runtime_are_confined_to_remote_mount():
    run_tag = "20260825-generation-sealed-r1"
    paths = remote.remote_paths(run_tag)
    dist_port = remote.dist_port_for_run_tag(run_tag)

    assert remote.DEFAULT_CONTROL_PATH == "none"
    assert remote.REMOTE_PYTHON == (
        "/data00/home/sitian/tllm/env/bin/python3.11"
    )
    assert all(
        path.startswith(remote.TASK_REMOTE_ROOT + "/")
        for path in paths.values()
    )
    assert remote.TASK_REMOTE_ROOT == (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/"
        "exact-burst-generation-sealed-lease-identity"
    )
    prelude = remote.remote_runtime_prelude(
        source=paths["staging"] + "/source",
        gpu_index=2,
        dist_port=dist_port,
    )
    assert paths["staging"] + "/runtime" in prelude
    assert "export CUDA_VISIBLE_DEVICES=2" in prelude
    assert f"export TINYVLLM_DIST_PORT={dist_port}" in prelude
    assert f"export MASTER_PORT={dist_port}" in prelude
    for variable in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_EXTENSIONS_DIR",
    ):
        assert f"export {variable}=" in prelude
    for forbidden in (
        "export TMPDIR=/tmp",
        "export TMP=/tmp",
        "export TEMP=/tmp",
        "/private/tmp",
        "/data00/home/sitian/tllm/TinyLLMForge",
    ):
        assert forbidden not in prelude
    with pytest.raises(ValueError, match="approved mounted root"):
        remote.validate_remote_task_root("/tmp")


def test_parse_args_requires_exactly_one_mode(tmp_path):
    common = [
        "--run-tag",
        "mode-contract",
        "--source-sha",
        "a" * 40,
        "--local-output-dir",
        str(tmp_path / "mode-contract"),
    ]
    with pytest.raises(SystemExit):
        remote.parse_args(common)
    with pytest.raises(SystemExit):
        remote.parse_args(["--launch", "--resume-existing", *common])
    assert remote.parse_args(["--launch", *common]).launch is True
    assert (
        remote.parse_args(["--resume-existing", *common]).resume_existing
        is True
    )


def test_source_and_gpu_admission_are_strict():
    commit = "a" * 40
    assert remote.validate_source_commit(
        commit,
        pushed_head=commit,
    ) == commit
    with pytest.raises(ValueError, match="requested source commit"):
        remote.validate_source_commit(
            "b" * 40,
            pushed_head=commit,
        )
    rows = [
        _gpu(0, memory=1024, utilization=5),
        _gpu(1, memory=1025),
        _gpu(2, utilization=6),
        _gpu(3, processes=({"pid": 9},)),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]
    assert remote.SOURCE_PATCH_SHA256 == hashlib.sha256(
        b""
    ).hexdigest()
    assert remote.MINIMUM_KERBEROS_LIFETIME_SECONDS == 5_400


def test_worker_and_verifier_commands_are_source_bound(monkeypatch):
    captured = []

    def fake_remote(command, **_kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="321\n",
            stderr="",
        )

    monkeypatch.setattr(remote, "_run_remote_checked", fake_remote)
    run_tag = "source-bound"
    paths = remote.remote_paths(run_tag)
    dist_port = remote.dist_port_for_run_tag(run_tag)
    assert remote._launch_worker(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        controller=paths["controller"],
        run_tag=run_tag,
        source_commit="a" * 40,
        gpu_index=1,
        dist_port=dist_port,
    ) == {"worker_pid": 321, "worker_pgid": 321}
    remote._run_remote_verifier(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        gpu_index=1,
        dist_port=dist_port,
    )

    joined = "\n".join(captured)
    assert (
        "tools/exact_burst_generation_sealed_lease_identity_gate.py"
        in joined
    )
    assert "--source-sha " + "a" * 40 in joined
    assert "--run-tag source-bound" in joined
    assert (
        "tools/exact_burst_generation_sealed_lease_identity_verify.py"
        in joined
    )


def test_worker_launch_is_receipt_driven_exactly_once(monkeypatch):
    captured = {}

    def fake_remote(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"worker_pid":321,"worker_pgid":321}\n',
            stderr="",
        )

    monkeypatch.setattr(remote, "_run_remote_checked", fake_remote)
    paths = remote.remote_paths("exactly-once")
    receipt = remote._launch_worker(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        controller=paths["controller"],
        run_tag="exactly-once",
        source_commit="a" * 40,
        gpu_index=2,
        dist_port=remote.dist_port_for_run_tag("exactly-once"),
    )

    assert receipt == {"worker_pid": 321, "worker_pgid": 321}
    command = captured["command"]
    assert "worker.launch.lock" in command
    assert "launch_receipt.json" in command
    assert "existing launch receipt mismatch" in command
    assert captured["kwargs"]["retry_attempts"] == 3
    assert "set -eu; umask 077" in command
    assert command.index("launch_receipt.json") < command.index(
        "set +e"
    )
    syntax = subprocess.run(
        ["bash", "-n"],
        input=command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_resume_receipts_must_match_run_source_pid_and_pgid():
    local = {
        "schema": (
            "exact_burst_generation_sealed_lease_identity_remote_v1"
        ),
        "status": "LAUNCHED",
        "run_tag": "resume-me",
        "source_sha": "a" * 40,
        "worker_pid": 321,
        "worker_pgid": 321,
    }
    assert remote.validate_resume_receipts(local, dict(local)) == {
        "worker_pid": 321,
        "worker_pgid": 321,
    }
    for key, value in (
        ("run_tag", "other"),
        ("source_sha", "b" * 40),
        ("worker_pid", 322),
        ("worker_pgid", 322),
    ):
        changed = dict(local)
        changed[key] = value
        with pytest.raises(ValueError, match="resume receipt"):
            remote.validate_resume_receipts(local, changed)


def test_resume_existing_never_launches(monkeypatch, tmp_path):
    run_tag = "resume-me"
    destination = tmp_path / run_tag
    destination.mkdir()
    receipt = {
        "schema": (
            "exact_burst_generation_sealed_lease_identity_remote_v1"
        ),
        "status": "LAUNCHED",
        "run_tag": run_tag,
        "source_sha": "a" * 40,
        "worker_pid": 321,
        "worker_pgid": 321,
        "selected_gpu": _gpu(2),
        "dist_port": remote.dist_port_for_run_tag(run_tag),
    }
    (destination / "launch_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        remote.base,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "_read_remote_launch_receipt",
        lambda **_kwargs: dict(receipt),
    )
    monkeypatch.setattr(
        remote,
        "_launch_worker",
        lambda **_kwargs: pytest.fail("resume relaunched worker"),
    )
    monkeypatch.setattr(
        remote,
        "_poll_worker",
        lambda **_kwargs: calls.append("poll") or 0,
    )
    monkeypatch.setattr(
        remote,
        "_run_remote_verifier",
        lambda **_kwargs: calls.append("verify"),
    )
    monkeypatch.setattr(
        remote,
        "_write_remote_completion",
        lambda **_kwargs: calls.append("completion"),
    )
    monkeypatch.setattr(
        remote,
        "_download_terminal_bundle",
        lambda **_kwargs: {
            "local_verification": {
                "verified": True,
                "classification": "GO",
            }
        },
    )

    result = remote.run_controller(
        _args(tmp_path, mode="--resume-existing", run_tag=run_tag)
    )

    assert calls == ["poll", "verify", "completion"]
    assert result["status"] == "COMPLETE"


def test_launch_rejects_attempted_local_tag(monkeypatch, tmp_path):
    destination = tmp_path / "attempted"
    destination.mkdir()
    monkeypatch.setattr(
        remote,
        "_require_task_tracked_diff_clean",
        lambda _root: None,
    )
    with pytest.raises(ValueError, match="already exists"):
        remote.run_controller(
            _args(tmp_path, run_tag="attempted")
        )


def test_launch_rejects_attempted_remote_tag_before_upload(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.setattr(
        remote,
        "_require_task_tracked_diff_clean",
        lambda _root: None,
    )
    monkeypatch.setattr(
        remote.base,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "_probe_remote_requirements",
        lambda: {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "_require_remote_destinations_absent",
        lambda _paths: (_ for _ in ()).throw(
            ValueError("remote destination already exists")
        ),
    )
    monkeypatch.setattr(
        remote,
        "_upload_source_archive",
        lambda **_kwargs: calls.append("upload"),
    )

    with pytest.raises(ValueError, match="already exists"):
        remote.run_controller(
            _args(tmp_path, run_tag="remote-attempted")
        )
    assert calls == []


def test_launch_rechecks_gpu_immediately_before_worker(monkeypatch, tmp_path):
    selected = _gpu(2)
    calls = []
    monkeypatch.setattr(
        remote,
        "_require_task_tracked_diff_clean",
        lambda _root: calls.append("tracked"),
    )
    monkeypatch.setattr(
        remote.base,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos")
        or {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "_probe_remote_requirements",
        lambda: {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "_require_remote_destinations_absent",
        lambda _paths: calls.append("destinations"),
    )
    monkeypatch.setattr(
        remote,
        "_wait_for_clean_gpu",
        lambda **_kwargs: calls.append("monitor")
        or ([selected], selected),
    )
    monkeypatch.setattr(
        remote,
        "committed_archive",
        lambda *_args, **_kwargs: b"archive",
    )
    monkeypatch.setattr(
        remote,
        "_upload_source_archive",
        lambda **_kwargs: calls.append("upload")
        or (
            remote.TASK_REMOTE_ROOT
            + "/staging/fresh/source"
        ),
    )
    monkeypatch.setattr(
        remote,
        "_run_remote_preflight",
        lambda **_kwargs: calls.append("preflight"),
    )
    monkeypatch.setattr(
        remote,
        "_query_remote_gpu_rows",
        lambda: calls.append("gpu_recheck") or [selected],
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda chosen, _rows: calls.append("validate_recheck")
        or chosen,
    )
    monkeypatch.setattr(
        remote,
        "_create_controller_dir",
        lambda **_kwargs: calls.append("controller"),
    )
    monkeypatch.setattr(
        remote,
        "_launch_worker",
        lambda **_kwargs: calls.append("launch")
        or {"worker_pid": 321, "worker_pgid": 321},
    )
    monkeypatch.setattr(
        remote,
        "_read_remote_launch_receipt",
        lambda **_kwargs: {
            "schema": remote.REMOTE_SCHEMA,
            "status": "LAUNCHED",
            "run_tag": "fresh",
            "source_sha": "a" * 40,
            "worker_pid": 321,
            "worker_pgid": 321,
        },
    )
    monkeypatch.setattr(
        remote,
        "_poll_worker",
        lambda **_kwargs: calls.append("poll") or 0,
    )
    monkeypatch.setattr(
        remote,
        "_run_remote_verifier",
        lambda **_kwargs: calls.append("verify"),
    )
    monkeypatch.setattr(
        remote,
        "_write_remote_completion",
        lambda **_kwargs: calls.append("completion"),
    )
    monkeypatch.setattr(
        remote,
        "_download_terminal_bundle",
        lambda **_kwargs: {
            "local_verification": {
                "verified": True,
                "classification": "GO",
            }
        },
    )

    result = remote.run_controller(_args(tmp_path))

    assert calls.index("gpu_recheck") < calls.index("launch")
    assert calls.index("validate_recheck") < calls.index("launch")
    assert calls.index("preflight") < calls.index("gpu_recheck")
    assert calls.index("controller") < calls.index("gpu_recheck")
    assert calls[calls.index("gpu_recheck") - 1] == "kerberos"
    assert calls.count("kerberos") == 3
    assert result["status"] == "COMPLETE"


def test_controller_receipt_retry_accepts_existing_parent(monkeypatch):
    captured = {}

    def fake_remote(command, payload, **kwargs):
        captured["command"] = command
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"stored\n",
            stderr=b"",
        )

    monkeypatch.setattr(
        remote,
        "_run_remote_with_input_checked",
        fake_remote,
    )
    controller = remote.remote_paths(
        "receipt-parent-retry"
    )["controller"]
    remote._create_controller_dir(
        controller=controller,
        receipt={"status": "READY"},
    )

    assert "exist_ok=True" in captured["command"]
    assert captured["kwargs"] == {
        "context": "controller receipt upload",
        "retry_attempts": 3,
        "idempotent": True,
    }


def test_remote_preflight_covers_controller_and_adjacent_suites(
    monkeypatch,
):
    captured = {}

    def fake_remote(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(remote, "_run_remote_checked", fake_remote)
    paths = remote.remote_paths("preflight-inventory")
    remote._run_remote_preflight(
        source=paths["staging"] + "/source",
        gpu_index=2,
        dist_port=remote.dist_port_for_run_tag(
            "preflight-inventory"
        ),
    )

    command = captured["command"]
    for path in (
        "tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py",
        "tools/test_llm_engine_exact_greedy_decode_burst.py",
        "tools/test_exact_burst_continuation_epoch_gate.py",
        "tools/test_exact_burst_continuation_epoch_verify.py",
        "tools/test_exact_burst_ragged_coalescing_gate.py",
        "tools/test_exact_burst_ragged_coalescing_verify.py",
        "tools/test_exact_burst_split_phase_gate.py",
        "tools/test_exact_burst_split_phase_verify.py",
        "tools/test_exact_burst_one_phase_lease_local_journal_gate.py",
        "tools/test_exact_burst_one_phase_lease_local_journal_verify.py",
    ):
        assert path in command
    assert captured["kwargs"] == {
        "context": "remote source-bound preflight",
        "retry_attempts": 3,
        "idempotent": True,
    }


def test_remote_preflight_inventory_names_existing_files():
    assert remote.REMOTE_PREFLIGHT_TEST_FILES
    assert all(
        (remote.REPO_ROOT / path).is_file()
        for path in remote.REMOTE_PREFLIGHT_TEST_FILES
    )


def test_terminal_download_reuses_verified_existing_tree(
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "primary"
    destination.mkdir()
    inventory = [{"path": "summary.json", "size_bytes": 3}]
    calls = []
    monkeypatch.setattr(
        remote.base,
        "fetch_remote_inventory",
        lambda root: calls.append(("inventory", root)) or inventory,
    )
    monkeypatch.setattr(
        remote.base,
        "verify_downloaded_tree",
        lambda target, rows: calls.append(
            ("verify", target, rows)
        ),
    )
    monkeypatch.setattr(
        remote,
        "download_remote_tree_preserving_partial",
        lambda *_args, **_kwargs: pytest.fail(
            "verified tree was downloaded again"
        ),
    )

    assert remote._download_remote_tree_idempotent(
        "/approved/remote",
        destination,
    ) == inventory
    assert calls == [
        ("inventory", "/approved/remote"),
        ("verify", destination, inventory),
    ]


def test_terminal_download_preserves_stale_partial_before_retry(
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "controller"
    partial = tmp_path / "controller.partial"
    partial.mkdir()
    (partial / "partial.log").write_text("evidence", encoding="utf-8")
    monkeypatch.setattr(
        remote,
        "download_remote_tree_preserving_partial",
        lambda root, target: [{"path": root, "target": str(target)}],
    )

    inventory = remote._download_remote_tree_idempotent(
        "/approved/controller",
        destination,
    )

    assert inventory == [{
        "path": "/approved/controller",
        "target": str(destination),
    }]
    preserved = list(tmp_path.glob("controller.partial.preserved-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "partial.log").read_text(
        encoding="utf-8"
    ) == "evidence"
    assert not partial.exists()


def test_retry_requires_explicit_idempotent_authorization(monkeypatch):
    attempts = []

    def operation():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return "ok"

    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    assert remote._retry_idempotent(
        operation,
        attempts=2,
        idempotent=True,
    ) == "ok"
    with pytest.raises(ValueError, match="idempotent"):
        remote._retry_idempotent(
            operation,
            attempts=2,
            idempotent=False,
        )


def test_controller_source_contains_no_refresh_or_process_kill():
    source = Path(remote.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "kinit",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "os.kill(",
        "os.killpg(",
        "TMPDIR=/tmp",
        "TMP=/tmp",
        "TEMP=/tmp",
        "/private/tmp",
    ):
        assert forbidden not in source
