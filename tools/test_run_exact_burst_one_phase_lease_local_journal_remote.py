from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from tools import (
    run_exact_burst_one_phase_lease_local_journal_remote as remote,
)


def _gpu(index, *, memory=0, utilization=0, processes=()):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100-SXM4-80GB",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def test_paths_and_runtime_are_confined_to_remote_mount():
    run_tag = "20260824-one-phase-journal-r1"
    assert remote.DEFAULT_CONTROL_PATH == "none"
    paths = remote.remote_paths(run_tag)
    dist_port = remote.dist_port_for_run_tag(run_tag)
    assert all(
        path.startswith(remote.TASK_REMOTE_ROOT + "/")
        for path in paths.values()
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
    assert 20_000 <= dist_port < 50_000
    assert dist_port != remote.dist_port_for_run_tag(
        "20260824-one-phase-journal-r2"
    )
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
    for invalid in ("", "../escape", "nested/tag", "-bad", "bad tag"):
        with pytest.raises(ValueError):
            remote.remote_paths(invalid)
    with pytest.raises(ValueError, match="approved mounted root"):
        remote.validate_remote_task_root("/tmp")


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


def test_worker_and_verifier_commands_are_source_bound(
    monkeypatch,
):
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
    run_tag = "fresh-one-phase-tag"
    paths = remote.remote_paths(run_tag)
    dist_port = remote.dist_port_for_run_tag(run_tag)
    pid = remote._launch_worker(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        controller=paths["controller"],
        run_tag=run_tag,
        source_commit="a" * 40,
        gpu_index=1,
        dist_port=dist_port,
    )
    remote._run_remote_verifier(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        gpu_index=1,
        dist_port=dist_port,
    )

    assert pid == 321
    joined = "\n".join(captured)
    assert (
        "tools/exact_burst_one_phase_lease_local_journal_gate.py"
        in joined
    )
    assert "--source-sha " + "a" * 40 in joined
    assert "--run-tag fresh-one-phase-tag" in joined
    assert (
        "tools/exact_burst_one_phase_lease_local_journal_verify.py"
        in joined
    )
    assert str(remote.MODEL_PATH) in joined
    assert f"export TINYVLLM_DIST_PORT={dist_port}" in joined
    assert f"export MASTER_PORT={dist_port}" in joined


def test_controller_monitors_before_launch_and_verifies_locally(
    monkeypatch,
    tmp_path,
):
    selected = _gpu(2)
    calls = []
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos")
        or {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote.base,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "_require_task_tracked_diff_clean",
        lambda _root: calls.append("tracked_clean"),
    )
    monkeypatch.setattr(
        remote.base,
        "require_remote_destinations_absent",
        lambda _paths: calls.append("destinations"),
    )
    monkeypatch.setattr(
        remote,
        "_probe_remote_requirements",
        lambda: {"status": "PASS"},
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
        remote.base,
        "query_remote_gpu_rows",
        lambda: [selected],
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda chosen, _rows: chosen,
    )
    monkeypatch.setattr(
        remote,
        "_create_controller_dir",
        lambda **_kwargs: calls.append("controller"),
    )
    monkeypatch.setattr(
        remote,
        "_launch_worker",
        lambda **_kwargs: calls.append("launch") or 321,
    )
    monkeypatch.setattr(
        remote,
        "_poll_worker",
        lambda **_kwargs: calls.append("poll") or 0,
    )
    monkeypatch.setattr(
        remote,
        "_run_remote_verifier",
        lambda **_kwargs: calls.append("remote_verify"),
    )
    monkeypatch.setattr(
        remote,
        "_write_remote_completion",
        lambda **_kwargs: calls.append("complete"),
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

    result = remote.run_controller(remote.parse_args([
        "--run-tag",
        "fresh",
        "--source-sha",
        "a" * 40,
        "--local-output-dir",
        str(tmp_path / "result"),
    ]))

    assert calls.index("monitor") < calls.index("launch")
    assert calls.index("poll") < calls.index("remote_verify")
    assert calls.count("kerberos") == 2
    assert result["status"] == "COMPLETE"


def test_worker_poll_tolerates_one_transient_remote_failure(
    monkeypatch,
):
    calls = iter((
        RuntimeError("transient SSH failure"),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"state":"finished","exitcode":0}\n',
            stderr="",
        ),
    ))

    def fake_remote(*_args, **_kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(remote, "_run_remote_checked", fake_remote)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)

    assert remote._poll_worker(
        controller=remote.remote_paths(
            "poll-retry"
        )["controller"],
        worker_pid=321,
        poll_interval_seconds=1,
    ) == 0


def test_requirement_probe_retries_transient_ssh_failures(
    monkeypatch,
):
    calls = []
    results = iter((
        RuntimeError(
            "remote requirement probe failed: "
            "Connection closed by UNKNOWN port 65535"
        ),
        RuntimeError(
            "remote requirement probe failed: "
            "Connection closed by UNKNOWN port 65535"
        ),
        {"status": "PASS"},
    ))

    def fake_probe(model):
        calls.append(model)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    sleeps = []
    monkeypatch.setattr(
        remote.base,
        "probe_remote_requirements",
        fake_probe,
    )
    monkeypatch.setattr(
        remote.legacy,
        "validate_remote_requirements",
        lambda result: result,
    )
    monkeypatch.setattr(
        remote.time,
        "sleep",
        sleeps.append,
    )

    assert remote._probe_remote_requirements() == {
        "status": "PASS",
    }
    assert calls == ["qwen3-0.6b"] * 3
    assert sleeps == [1, 2]


def test_controller_rejects_attempted_local_tag(
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "attempted"
    destination.mkdir()
    monkeypatch.setattr(
        remote,
        "_require_task_tracked_diff_clean",
        lambda _root: None,
    )

    with pytest.raises(ValueError, match="already exists"):
        remote.run_controller(remote.parse_args([
            "--run-tag",
            "attempted",
            "--source-sha",
            "a" * 40,
            "--local-output-dir",
            str(destination),
        ]))


def test_controller_source_contains_no_refresh_or_process_kill():
    source = Path(remote.__file__).read_text()
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
