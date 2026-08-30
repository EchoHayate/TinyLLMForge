#!/usr/bin/env python3
"""Tests for the mounted-only persistent-decode remote controller."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

import pytest

from tools import profile_lease_sealed_persistent_decode_ceiling as profile
from tools import run_lease_sealed_persistent_decode_ceiling_remote as remote


def test_direct_script_entrypoint_can_import_tools():
    result = subprocess.run(
        [sys.executable, str(Path(remote.__file__)), "--help"],
        cwd=Path(remote.__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _gpu(
    index: int,
    *,
    memory: int = 0,
    utilization: int = 0,
    processes=None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def test_remote_paths_stay_below_approved_task_root():
    paths = remote.remote_paths(
        "20260830-persistent-decode-ceiling-test"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(
        value.startswith(remote.APPROVED_ROOT + "/")
        for value in paths.values()
    )
    with pytest.raises(ValueError, match="run tag"):
        remote.remote_paths("../escape")


def test_validate_kerberos_uses_fixed_file_cache_and_ttl():
    now = datetime(2026, 8, 30, 18, 0, tzinfo=timezone.utc)
    expiry = now + timedelta(seconds=remote.MINIMUM_KERBEROS_LIFETIME_SECONDS)
    output = (
        f"Credentials cache: {remote.KRB5_CACHE}\n"
        "Principal: sitian@BYTEDANCE.COM\n\n"
        "Issued                Expires               Principal\n"
        f"Aug 30 17:00:00 2026  {expiry.strftime('%b %d %H:%M:%S %Y')}  "
        "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM\n"
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=output,
            stderr="",
        )

    receipt = remote.validate_kerberos(
        environ={},
        command_runner=runner,
        now=lambda: now,
        minimum_lifetime_seconds=
            remote.MINIMUM_KERBEROS_LIFETIME_SECONDS,
    )

    assert receipt["status"] == "PASS"
    assert calls[0][1]["env"]["KRB5CCNAME"] == remote.KRB5_CACHE


def test_strict_clean_a100_requires_zero_everything():
    rows = [
        _gpu(0),
        _gpu(1, memory=1),
        _gpu(2, utilization=1),
        _gpu(3, processes=[{"pid": 7}]),
        {**_gpu(4), "name": "NVIDIA H100 80GB HBM3"},
    ]

    assert remote.strict_clean_a100s(rows) == [rows[0]]


def test_committed_source_archive_contains_exact_qualification_sources():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for relative in profile.SOURCE_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n")
        unrelated = root / "docs/private.txt"
        unrelated.parent.mkdir(parents=True, exist_ok=True)
        unrelated.write_text("no\n")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "add", "--", *profile.SOURCE_FILES, "docs/private.txt"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Remote Test",
                "-c",
                "user.email=remote@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=root,
            check=True,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        payload = remote.committed_source_archive(root, commit)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as bundle:
        names = {member.name for member in bundle.getmembers()}
    assert {
        "source/" + relative
        for relative in profile.SOURCE_FILES
    }.issubset(names)
    assert "source/docs/private.txt" not in names


def test_nsys_command_is_bounded_and_mounted_only():
    paths = remote.remote_paths("20260830-persistent-decode-r1")
    command = remote.build_nsys_command(
        source_dir=paths["staging"] + "/source",
        output_dir=paths["primary"],
        run_tag="20260830-persistent-decode-r1",
        source_commit="a" * 40,
        gpu_index=2,
        prompt_tokens=2048,
    )
    joined = " ".join(command)

    assert command[0] == "/usr/local/bin/nsys"
    assert "tools.profile_lease_sealed_persistent_decode_ceiling" in joined
    assert "--mode structural" in joined
    assert "--prompt-tokens 2048" in joined
    assert "CUDA_VISIBLE_DEVICES=2" in joined
    assert paths["primary"] in joined
    assert "/tmp" not in joined
    assert "/private/tmp" not in joined


def test_worker_plan_has_one_timing_three_traces_and_remote_verifier():
    paths = remote.remote_paths("20260830-persistent-decode-r2")
    plan = remote.build_worker_plan(
        paths=paths,
        run_tag="20260830-persistent-decode-r2",
        source_commit="a" * 40,
        gpu=_gpu(3),
    )
    commands = plan["commands"]
    joined = "\n".join(
        " ".join(command) if isinstance(command, list) else command
        for command in commands
    )

    assert sum("--mode timing" in str(command) for command in commands) == 1
    assert sum(
        isinstance(command, list)
        and command[:2] == [remote.NSYS_PATH, "profile"]
        for command in commands
    ) == 3
    assert sum(
        isinstance(command, str)
        and command.startswith(
            remote.NSYS_PATH + " export --type=sqlite"
        )
        for command in commands
    ) == 3
    assert "tools.verify_lease_sealed_persistent_decode_ceiling" in joined
    assert "gpu_admission_second.json" in joined
    assert paths["controller"] + "/remote-verify-bundle" in joined
    assert paths["controller"] + "/remote_verification.json" in joined
    assert "kinit" not in joined
    assert "krenew" not in joined
    assert "pkill" not in joined
    assert "killall" not in joined
    assert "/tmp" not in joined
    assert "/private/tmp" not in joined


def test_controller_source_has_no_process_kill_or_legacy_paths():
    source = Path(remote.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "kinit",
        "krenew",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "TinyLLMForge-adaptive-ngram",
    ):
        assert forbidden not in source


def test_compact_member_filter_excludes_raw_profiler_files():
    assert remote.is_compact_artifact("ceiling.json") is True
    assert remote.is_compact_artifact("nsys/context-256.nsys-rep") is False
    assert remote.is_compact_artifact("nsys/context-256.sqlite") is False
    assert remote.is_compact_artifact("runtime/hf-cache/state.json") is False
    assert remote.is_compact_artifact("gpu_admission_second.json") is False
    with pytest.raises(ValueError, match="artifact path"):
        remote.is_compact_artifact("../ceiling.json")


def test_streamed_raw_trace_verification_uses_temporary_files_only(
    monkeypatch,
    tmp_path: Path,
):
    compact = tmp_path / "compact"
    compact.mkdir()
    payloads = {
        context: f"sqlite-{context}".encode()
        for context in remote.CONTEXT_LENGTHS
    }
    raw_traces = [
        {
            "context_length": context,
            "remote_path": (
                remote.TASK_REMOTE_ROOT
                + "/runs/run-a/nsys/"
                + f"context-{context}.sqlite"
            ),
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for context, payload in payloads.items()
    ]
    (compact / "trace_inventory.json").write_text(
        json.dumps({"raw_traces": raw_traces}),
        encoding="utf-8",
    )
    inventory = [
        {
            "path": f"nsys/context-{context}.sqlite",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "chunks": [{
                "offset": 0,
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        }
        for context, payload in payloads.items()
    ]
    monkeypatch.setattr(
        remote.base,
        "fetch_remote_inventory",
        lambda _path: inventory,
    )
    monkeypatch.setattr(
        remote.base,
        "download_chunk",
        lambda path, **_kwargs: payloads[
            int(Path(path).stem.removeprefix("context-"))
        ],
    )
    monkeypatch.setattr(
        remote,
        "verify_local_bundle",
        lambda path: {"verified": path == compact},
    )

    receipt = remote.stream_and_verify_raw_traces(
        remote_path=remote.TASK_REMOTE_ROOT + "/runs/run-a",
        compact_path=compact,
        temporary_parent=tmp_path,
    )

    assert receipt["verified"] is True
    assert receipt["raw_trace_count"] == 3
    assert not list(tmp_path.rglob("*.sqlite"))


def test_download_compact_bundle_rejects_existing_destination(
    tmp_path: Path,
):
    destination = (
        tmp_path / "20260830-persistent-decode-ceiling-existing"
    )
    destination.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        remote.download_compact_bundle(
            remote_path=(
                remote.TASK_REMOTE_ROOT
                + "/runs/20260830-persistent-decode-ceiling-existing"
            ),
            local_parent=tmp_path,
        )


def test_write_controller_receipts_records_plan_admission_and_download(
    tmp_path: Path,
):
    destination = tmp_path / "run-a"
    destination.mkdir()
    for relative in remote.COMPACT_FILES:
        (destination / relative).write_bytes(relative.encode())

    controller = remote.write_controller_receipts(
        destination=destination,
        plan={"run_tag": "run-a"},
        gpu_inventory=[_gpu(0)],
        selected_gpu=_gpu(0),
        worker={"status": "COMPLETE"},
        verification={"verified": True, "raw_trace_count": 3},
    )

    assert {
        path.name for path in controller.iterdir()
    } == {
        "plan.json",
        "launch_admission.json",
        "download_manifest.json",
        "local-verification.json",
    }
    download = json.loads(
        (controller / "download_manifest.json").read_text()
    )
    assert {
        row["path"] for row in download["artifacts"]
    } == set(remote.COMPACT_FILES)


def test_run_controller_rechecks_gpu_immediately_before_launch(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    selected = _gpu(2)
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: calls.append("head") or "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "require_remote_destinations_absent",
        lambda _paths: calls.append("destinations"),
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: (
            calls.append("first_admission") or ([selected], selected)
        ),
    )
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda _root, _commit: calls.append("archive") or b"archive",
    )
    monkeypatch.setattr(
        remote,
        "upload_source_archive",
        lambda **kwargs: (
            calls.append("upload")
            or kwargs["staging"] + "/source"
        ),
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda gpu: calls.append("second_admission") or gpu,
    )
    monkeypatch.setattr(
        remote,
        "run_worker_plan",
        lambda _plan: calls.append("launch") or {"status": "COMPLETE"},
    )
    monkeypatch.setattr(
        remote,
        "verify_remote_bundle_with_streamed_traces",
        lambda **_kwargs: (
            calls.append("stream_verify") or {"verified": True}
        ),
    )
    monkeypatch.setattr(
        remote,
        "download_compact_bundle",
        lambda **_kwargs: calls.append("download") or tmp_path / "bundle",
    )
    monkeypatch.setattr(
        remote,
        "write_controller_receipts",
        lambda **_kwargs: calls.append("receipts") or tmp_path / "controller",
    )

    result = remote.run_controller(
        remote.parse_args([
            "--run-tag",
            "20260830-persistent-decode-ceiling-controller",
            "--source-commit",
            "a" * 40,
            "--local-artifact-root",
            str(tmp_path),
        ])
    )

    assert result["status"] == "COMPLETE"
    assert calls.index("second_admission") < calls.index("launch")
    assert calls == [
        "head",
        "kerberos",
        "destinations",
        "first_admission",
        "archive",
        "upload",
        "kerberos",
        "second_admission",
        "launch",
        "stream_verify",
        "download",
        "receipts",
    ]
