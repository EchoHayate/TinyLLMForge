from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import run_context_gated_elastic_exact_burst_remote as remote


def _gpu(index: int, *, memory=0, utilization=0, processes=()):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100-SXM4-80GB",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def test_paths_runtime_and_source_patch_are_strict() -> None:
    run_tag = "20260825-elastic-k16-ceiling-r1"
    paths = remote.remote_paths(run_tag)
    port = remote.dist_port_for_run_tag(run_tag)
    assert remote.DEFAULT_CONTROL_PATH == remote.base.CONTROL_PATH
    assert remote.DEFAULT_CONTROL_PATH == (
        "/tmp/ssh-sitian-10.232.195.203"
    )
    assert all(
        path.startswith(remote.TASK_REMOTE_ROOT + "/")
        for path in paths.values()
    )
    prelude = remote.remote_runtime_prelude(
        source=paths["staging"] + "/source",
        gpu_index=2,
        dist_port=port,
    )
    assert "export CUDA_VISIBLE_DEVICES=2" in prelude
    assert paths["staging"] + "/runtime" in prelude
    assert f"export TINYVLLM_DIST_PORT={port}" in prelude
    assert f"export MASTER_PORT={port}" in prelude
    assert "/private/tmp" not in prelude
    assert "export TMPDIR=/tmp" not in prelude
    assert remote.SOURCE_PATCH_SHA256 == hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError, match="approved mounted root"):
        remote.validate_remote_task_root("/tmp")


def test_source_and_gpu_admission_are_strict() -> None:
    commit = "a" * 40
    assert remote.validate_source_commit(
        commit,
        pushed_head=commit,
    ) == commit
    with pytest.raises(ValueError, match="requested source commit"):
        remote.validate_source_commit("b" * 40, pushed_head=commit)
    rows = [
        _gpu(0, memory=1024, utilization=5),
        _gpu(1, memory=1025),
        _gpu(2, utilization=6),
        _gpu(3, processes=({"pid": 9},)),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]


def test_remote_destination_check_preserves_named_path_inventory(
    monkeypatch,
) -> None:
    captured = []
    monkeypatch.setattr(
        remote.base,
        "require_remote_destinations_absent",
        captured.append,
    )
    paths = remote.remote_paths("destination-contract-r1")

    remote._require_remote_destinations_absent(paths)

    assert captured == [paths]


def test_gpu_monitor_waits_then_returns_first_strict_clean_gpu(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inventories = iter((
        [_gpu(0, memory=2048)],
        [_gpu(0, memory=0)],
    ))
    kerberos = []
    sleeps = []
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **kwargs: kerberos.append(kwargs)
        or {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "_query_remote_gpu_rows",
        lambda: next(inventories),
    )
    monkeypatch.setattr(remote.time, "sleep", sleeps.append)
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(
        remote.time,
        "monotonic",
        lambda: next(monotonic),
    )

    rows, selected = remote._wait_for_clean_gpu(
        timeout_seconds=60,
        poll_interval_seconds=1,
        local_destination=tmp_path,
    )

    assert rows == [_gpu(0, memory=0)]
    assert selected == _gpu(0, memory=0)
    assert len(kerberos) == 2
    assert sleeps == [1]
    assert (
        tmp_path / "gpu_inventory.jsonl"
    ).read_text(encoding="utf-8").count("\n") == 2


def test_worker_and_remote_verifier_are_source_bound(monkeypatch) -> None:
    commands = []

    def fake_remote(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="321\n",
            stderr="",
        )

    monkeypatch.setattr(remote, "_run_remote_checked", fake_remote)
    paths = remote.remote_paths("elastic-ceiling-r1")
    port = remote.dist_port_for_run_tag("elastic-ceiling-r1")
    pid = remote._launch_worker(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        controller=paths["controller"],
        run_tag="elastic-ceiling-r1",
        source_commit="a" * 40,
        gpu_index=1,
        dist_port=port,
    )
    remote._run_remote_verifier(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        gpu_index=1,
        dist_port=port,
    )

    joined = "\n".join(commands)
    assert pid == 321
    assert "profile_context_gated_elastic_exact_burst.py" in joined
    assert "context_gated_elastic_exact_burst_ceiling.py" in joined
    assert "--source-commit " + "a" * 40 in joined
    assert "--run-tag elastic-ceiling-r1" in joined
    assert "--repetitions 3" in joined
    assert str(remote.MODEL_PATH) in joined
    assert "(set -e;" in commands[0]


def test_controller_monitors_rechecks_and_verifies(monkeypatch, tmp_path):
    selected = _gpu(2)
    calls = []
    kerberos_lifetimes = []

    def fake_kerberos(*, minimum_lifetime_seconds):
        kerberos_lifetimes.append(minimum_lifetime_seconds)
        calls.append("kerberos")
        return {"status": "PASS"}

    monkeypatch.setattr(remote, "validate_kerberos", fake_kerberos)
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
        remote,
        "_require_remote_destinations_absent",
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
        or remote.TASK_REMOTE_ROOT + "/staging/fresh/source",
    )
    monkeypatch.setattr(
        remote,
        "_run_remote_preflight",
        lambda **_kwargs: calls.append("preflight"),
    )
    monkeypatch.setattr(
        remote,
        "_query_remote_gpu_rows",
        lambda: [selected],
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda chosen, _rows: calls.append("recheck") or chosen,
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
        "_download_ceiling_bundle",
        lambda **_kwargs: {
            "local_verification": {
                "verified": True,
                "classification": "CEILING_GO",
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

    assert calls.index("monitor") < calls.index("recheck")
    assert calls.index("recheck") < calls.index("launch")
    assert calls.index("poll") < calls.index("remote_verify")
    assert kerberos_lifetimes == [
        remote.MINIMUM_KERBEROS_LIFETIME_SECONDS,
        remote.MINIMUM_KERBEROS_LIFETIME_SECONDS,
    ]
    assert result["status"] == "COMPLETE"


def test_frozen_source_local_verifier_uses_downloaded_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    frozen_source = Path("frozen-source")
    primary = Path("primary")
    output = Path("verify.json")
    script = (
        frozen_source
        / "tools"
        / "context_gated_elastic_exact_burst_ceiling.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n", encoding="utf-8")
    primary.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output.resolve().write_text(
            json.dumps({
                "verified": True,
                "classification": "CEILING_GO",
                "performance_row_count": 24,
                "correctness_row_count": 32,
            }),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    receipt = remote._run_frozen_source_local_verifier(
        frozen_source=frozen_source,
        primary=primary,
        output=output,
    )
    command, kwargs = calls[0]
    assert Path(command[1]) == script.resolve()
    assert Path(command[2]) == primary.resolve()
    assert Path(command[-1]) == output.resolve()
    assert kwargs["cwd"] == frozen_source.resolve()
    assert receipt["verified"] is True


def test_controller_source_has_no_kinit_or_process_kill() -> None:
    source = Path(remote.__file__).read_text(encoding="utf-8")
    forbidden = ("kinit", "kill -", "pkill", "killall")
    assert all(token not in source for token in forbidden)
