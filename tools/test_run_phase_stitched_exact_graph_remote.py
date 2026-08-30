#!/usr/bin/env python3
"""Safety contracts for the phase-stitched exact-graph controller."""

from __future__ import annotations

import hashlib
import io
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
    from tools import run_phase_stitched_exact_graph_remote

    return run_phase_stitched_exact_graph_remote


def _gpu_row(index, *, memory=0, utilization=0, processes=None):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def _commit_fixture(root: Path, paths):
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"value = {relative!r}\n", encoding="utf-8")
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


def test_remote_paths_runtime_and_gpu_admission_are_safe():
    remote = _remote()
    paths = remote.remote_paths("20260830-qwen3-06b-r1")
    approved = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/phase-stitched-exact-graph"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(path.startswith(approved + "/") for path in paths.values())
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
    assert "export CUDA_VISIBLE_DEVICES=2" in prelude
    assert "/tmp" not in prelude
    rows = [
        _gpu_row(0),
        _gpu_row(1, memory=1),
        _gpu_row(2, utilization=1),
        _gpu_row(3, processes=[{"pid": 7}]),
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


def test_source_archive_is_built_from_pushed_head_allowlist():
    remote = _remote()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = ("tinyvllm/config.py", *remote.TOOL_SOURCE_FILES)
        _commit_fixture(root, paths)
        (root / "tinyvllm/config.py").write_text(
            "dirty = True\n",
            encoding="utf-8",
        )
        payload, inventory = remote.build_source_archive(root)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        names = {member.name for member in archive.getmembers()}
        source = archive.extractfile("source/tinyvllm/config.py")
        assert source is not None
        assert source.read() == b"value = 'tinyvllm/config.py'\n"
    assert names == {
        "source",
        *(f"source/{relative}" for relative in inventory),
    }
    assert not any(path.startswith("experiments/") for path in inventory)


def test_source_archive_uses_head_even_if_tracked_file_is_deleted_locally():
    remote = _remote()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = ("tinyvllm/config.py", *remote.TOOL_SOURCE_FILES)
        _commit_fixture(root, paths)
        (root / "tinyvllm/config.py").unlink()
        payload, inventory = remote.build_source_archive(root)

    assert "tinyvllm/config.py" in inventory
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        source = archive.extractfile("source/tinyvllm/config.py")
        assert source is not None
        assert source.read() == b"value = 'tinyvllm/config.py'\n"


def test_remote_commands_run_all_cases_and_both_verifiers():
    remote = _remote()
    paths = remote.remote_paths("fresh-phase-stitched-r1")
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
        if "tools.phase_stitched_exact_graph_worker" in command
    ]) == 16
    assert "tools.phase_stitched_exact_graph_gate" in joined
    assert "tools.phase_stitched_exact_graph_verify" in joined
    assert "producer_exitcode" in joined
    assert "verifier_exitcode" in joined
    assert "--model " + remote.MODEL_PATH in joined
    assert "download" not in joined.lower()


def test_controller_requires_ticket_to_outlive_wait_and_rechecks_head(
    tmp_path: Path,
):
    remote = _remote()
    selected = _gpu_row(2)
    calls = []
    lifetimes = []
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
    remote.validate_kerberos = lambda **kwargs: (
        lifetimes.append(kwargs["minimum_lifetime_seconds"])
        or {"status": "PASS"}
    )
    remote.require_pushed_head = lambda _root: (
        calls.append("head") or "a" * 40
    )
    remote._source_hashes = lambda _root: {
        path: hashlib.sha256(path.encode()).hexdigest()
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
        or remote.remote_paths("fresh-phase-stitched-r1")["staging"]
        + "/source"
    )
    remote.run_remote_preflight = lambda **_kwargs: calls.append(
        "preflight"
    )
    remote.validate_selected_gpu_still_clean = (
        lambda chosen: calls.append("recheck") or chosen
    )
    remote.create_remote_run = lambda **_kwargs: calls.append("create")
    remote.run_remote_pipeline = lambda **_kwargs: {
        "case_exitcodes": [0] * 16,
        "producer_exitcode": 0,
        "verifier_exitcode": 0,
    }
    remote.download_terminal_bundle = lambda **_kwargs: {
        "verified": True,
        "classification": "GO_PHASE_STITCHED_EXACT_GRAPH",
    }
    try:
        args = remote.parse_args([
            "--run-tag",
            "fresh-phase-stitched-r1",
            "--source-commit",
            "a" * 40,
            "--local-artifact-root",
            str(tmp_path),
            "--gpu-wait-timeout-seconds",
            "7200",
        ])
        result = remote.run_controller(args)
    finally:
        for name, value in originals.items():
            setattr(remote, name, value)

    assert calls.count("head") == 2
    assert calls.index("wait") < calls.index("head", 1)
    assert lifetimes[0] >= (
        7200 + remote.MINIMUM_LAUNCH_KERBEROS_SECONDS
    )
    assert result["classification"] == (
        "GO_PHASE_STITCHED_EXACT_GRAPH"
    )
