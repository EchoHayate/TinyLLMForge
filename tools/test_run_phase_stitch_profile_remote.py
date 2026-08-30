#!/usr/bin/env python3
"""Safety contracts for the Phase-Stitch remote controller."""

from __future__ import annotations

import io
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

import pytest


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))


def _remote():
    from tools import run_phase_stitch_profile_remote

    return run_phase_stitch_profile_remote


def _gpu_row(index, *, memory=0, utilization=0, processes=None):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def _commit_source_fixture(root, tool_source_files):
    paths = ("tinyvllm/config.py", *tool_source_files)
    for relative_path in paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"source_file = {relative_path!r}\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "--", *paths], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Phase Stitch Test",
            "-c",
            "user.email=phase-stitch@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return paths


def test_paths_and_runtime_are_confined_to_approved_data_root():
    remote = _remote()
    paths = remote.remote_paths("20260830-qwen3-06b-r1")
    expected_root = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/phase-stitch-profile"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(
        path.startswith(expected_root + "/")
        for path in paths.values()
    )
    prelude = remote.remote_runtime_prelude(
        source=paths["staging"] + "/source",
        gpu_index=2,
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
    assert paths["staging"] + "/runtime" in prelude
    assert "export CUDA_VISIBLE_DEVICES=2" in prelude
    for forbidden in (
        "export TMPDIR=/tmp",
        "export TMP=/tmp",
        "export TEMP=/tmp",
        "/private/tmp",
        "/data00/home/sitian/tllm/TinyLLMForge",
    ):
        assert forbidden not in prelude
    for tag in ("", "../escape", "nested/tag", "-leading", "space tag"):
        with pytest.raises(ValueError, match="run tag"):
            remote.remote_paths(tag)


def test_gpu_admission_requires_one_clean_a100_and_never_kills():
    remote = _remote()
    rows = [
        _gpu_row(0),
        _gpu_row(1, memory=1025),
        _gpu_row(2, utilization=6),
        _gpu_row(3, processes=[{"pid": 7, "process_name": "python"}]),
        {
            **_gpu_row(4),
            "name": "NVIDIA H100 80GB HBM3",
        },
    ]

    assert remote.strict_clean_a100s(rows) == [rows[0]]
    source = Path(remote.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "kinit",
        "krenew",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "os.kill(",
        "os.killpg(",
    ):
        assert forbidden not in source


def test_source_archive_contains_only_frozen_allowlist():
    remote = _remote()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        _commit_source_fixture(root, remote.TOOL_SOURCE_FILES)
        payload, inventory = remote.build_source_archive(root)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
    assert inventory
    assert {member.name for member in members} == {
        f"source/{relative_path}" for relative_path in inventory
    } | {"source"}
    assert all(
        member.isdir() or member.isfile()
        for member in members
    )
    assert all(
        relative_path == "tinyvllm"
        or relative_path.startswith("tinyvllm/")
        or relative_path in remote.TOOL_SOURCE_FILES
        for relative_path in inventory
    )
    assert "tools/phase_stitch_profile_worker.py" in inventory
    assert "tools/phase_stitch_profile_gate.py" in inventory
    assert "tools/phase_stitch_profile_verify.py" in inventory
    assert "tools/test_phase_stitch_profile.py" in inventory
    assert "tools/test_phase_stitch_profile_benchmark.py" in inventory
    assert (
        "tools/test_llm_engine_exact_greedy_decode_burst.py"
        in inventory
    )
    assert not any(
        relative_path.startswith("experiments/")
        for relative_path in inventory
    )


def test_source_archive_reads_pushed_commit_not_dirty_worktree():
    remote = _remote()
    original_tool_source_files = remote.TOOL_SOURCE_FILES
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        _commit_source_fixture(root, ())
        source = root / "tinyvllm/config.py"
        source.write_text("dirty = True\n", encoding="utf-8")
        remote.TOOL_SOURCE_FILES = ()
        try:
            payload, inventory = remote.build_source_archive(root)
        finally:
            remote.TOOL_SOURCE_FILES = original_tool_source_files

    assert inventory == ("tinyvllm/config.py",)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        archived = archive.extractfile("source/tinyvllm/config.py")
        assert archived is not None
        assert archived.read() == (
            b"source_file = 'tinyvllm/config.py'\n"
        )


def test_source_hashes_read_pushed_commit_not_dirty_worktree():
    remote = _remote()
    original_source_files = remote.contract.SOURCE_FILES
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        _commit_source_fixture(root, ())
        source = root / "tinyvllm/config.py"
        committed = source.read_bytes()
        source.write_text("dirty = True\n", encoding="utf-8")
        remote.contract.SOURCE_FILES = ("tinyvllm/config.py",)
        try:
            hashes = remote._source_hashes(root)
        finally:
            remote.contract.SOURCE_FILES = original_source_files

    assert hashes == {
        "tinyvllm/config.py": hashlib.sha256(committed).hexdigest(),
    }


def test_remote_commands_capture_case_gate_and_verifier_exit_codes():
    remote = _remote()
    paths = remote.remote_paths("fresh-phase-stitch-r1")
    commands = remote.remote_execution_commands(
        source=paths["staging"] + "/source",
        primary=paths["primary"],
        controller=paths["controller"],
        model=remote.MODEL_PATH,
        gpu_index=1,
    )
    joined = "\n".join(commands)

    assert len([
        command
        for command in commands
        if "tools.phase_stitch_profile_worker" in command
    ]) == 8
    assert "tools.phase_stitch_profile_gate" in joined
    assert "tools.phase_stitch_profile_verify" in joined
    assert "producer_exitcode" in joined
    assert "verifier_exitcode" in joined
    assert "--model " + remote.MODEL_PATH in joined
    assert "download" not in joined.lower()
    for case in remote.contract.build_case_matrix():
        case_id = case["case_id"]
        assert (
            f"--spec {paths['controller']}/case-specs/{case_id}.json"
            in joined
        )
        assert (
            f"--output-dir {paths['primary']}/cases/{case_id}"
            in joined
        )
        assert f"/cases/{case_id}/measurement" not in joined


def test_create_remote_run_places_specs_outside_measurement_dirs():
    remote = _remote()
    original_runner = remote.base._run_remote_with_input
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = {
            "staging": os.fspath(root / "staging"),
            "primary": os.fspath(root / "primary"),
            "controller": os.fspath(root / "controller"),
        }

        def run_locally(command, payload):
            return subprocess.run(
                command,
                shell=True,
                input=payload,
                capture_output=True,
                check=False,
            )

        remote.base._run_remote_with_input = run_locally
        try:
            remote.create_remote_run(
                paths=paths,
                run_tag="fixture-r1",
                source_commit="a" * 40,
                source_hashes={
                    path: "b" * 64
                    for path in remote.contract.SOURCE_FILES
                },
                gpu_inventory=[_gpu_row(0)],
                selected_gpu=_gpu_row(0),
            )
        finally:
            remote.base._run_remote_with_input = original_runner

        assert (root / "primary/run_manifest.json").is_file()
        for case_id in remote.contract.expected_case_ids():
            assert (
                root / f"controller/case-specs/{case_id}.json"
            ).is_file()
            assert not (
                root / f"primary/cases/{case_id}"
            ).exists()


def test_controller_waits_then_auto_launches_without_overwrite():
    remote = _remote()
    selected = _gpu_row(2)
    calls = []
    originals = {
        name: getattr(remote, name)
        for name in (
            "validate_kerberos",
            "require_pushed_head",
            "_source_hashes",
            "require_remote_destinations_absent",
            "wait_for_clean_a100",
            "build_source_archive",
            "upload_source_archive",
            "run_remote_preflight",
            "validate_selected_gpu_still_clean",
            "create_remote_run",
            "run_remote_pipeline",
            "download_terminal_bundle",
        )
    }
    remote.validate_kerberos = lambda **_kwargs: (
        calls.append("kerberos") or {"status": "PASS"}
    )
    remote.require_pushed_head = lambda _root: (
        calls.append("pushed-head") or "a" * 40
    )
    remote._source_hashes = lambda _root: {
        path: "b" * 64
        for path in remote.contract.SOURCE_FILES
    }
    remote.require_remote_destinations_absent = (
        lambda _paths: calls.append("destinations")
    )
    remote.wait_for_clean_a100 = lambda **_kwargs: (
        calls.append("wait") or ([selected], selected)
    )
    remote.build_source_archive = lambda _root: (
        b"archive",
        ("tinyvllm/config.py",),
    )
    remote.upload_source_archive = lambda **_kwargs: (
        calls.append("upload")
        or remote.remote_paths("fresh-phase-stitch-r1")["staging"]
        + "/source"
    )
    remote.run_remote_preflight = lambda **_kwargs: (
        calls.append("preflight")
    )
    remote.validate_selected_gpu_still_clean = (
        lambda chosen: calls.append("recheck") or chosen
    )
    remote.create_remote_run = lambda **_kwargs: calls.append("create")
    remote.run_remote_pipeline = lambda **_kwargs: (
        calls.append("pipeline")
        or {"producer_exitcode": 0, "verifier_exitcode": 0}
    )
    remote.download_terminal_bundle = lambda **_kwargs: (
        calls.append("download")
        or {
            "verified": True,
            "classification": "GO_PHASE_STITCH_PROFILE",
        }
    )
    try:
        with TemporaryDirectory() as temporary:
            result = remote.run_controller(remote.parse_args([
                "--run-tag", "fresh-phase-stitch-r1",
                "--source-commit", "a" * 40,
                "--local-artifact-root", temporary,
            ]))
    finally:
        for name, value in originals.items():
            setattr(remote, name, value)

    assert calls.index("wait") < calls.index("pipeline")
    assert calls.index("wait") < calls.index(
        "pushed-head",
        calls.index("wait"),
    )
    assert calls.index("recheck") < calls.index("pipeline")
    assert calls.count("kerberos") == 2
    assert calls.count("pushed-head") == 2
    assert result["classification"] == "GO_PHASE_STITCH_PROFILE"

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "used-tag").mkdir()
        with pytest.raises(ValueError, match="already exists"):
            remote.ensure_local_destination_absent(root, "used-tag")


def test_failed_remote_subprocess_is_not_silently_accepted():
    remote = _remote()
    failure = subprocess.CompletedProcess(
        args=["ssh"],
        returncode=9,
        stdout="",
        stderr="boom",
    )
    with pytest.raises(RuntimeError, match="boom"):
        remote.require_success(failure, "remote stage")
