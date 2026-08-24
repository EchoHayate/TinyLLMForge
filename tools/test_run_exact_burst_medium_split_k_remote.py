#!/usr/bin/env python3
"""Safety contracts for the medium split-K remote controller."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import pytest

from tools import run_exact_burst_medium_split_k_remote as remote


def _gpu(index: int, *, memory=0, utilization=0, processes=None):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def test_paths_runtime_and_unique_port_are_confined() -> None:
    paths = remote.remote_paths("20260824-medium-split-k-r1")
    root = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/"
        "exact-burst-medium-split-k"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(path.startswith(root + "/") for path in paths.values())
    prelude = remote.remote_runtime_prelude(
        source=paths["staging"] + "/source",
        gpu_index=2,
        dist_port=remote.dist_port_for_run_tag(
            "20260824-medium-split-k-r1"
        ),
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
    assert remote.dist_port_for_run_tag(
        "20260824-medium-split-k-r1"
    ) != remote.dist_port_for_run_tag(
        "20260824-medium-split-k-r2"
    )
    for forbidden in (
        "export TMPDIR=/tmp",
        "export TMP=/tmp",
        "export TEMP=/tmp",
        "/private/tmp",
        "/data00/home/sitian/tllm/TinyLLMForge",
    ):
        assert forbidden not in prelude
    for tag in ("", "../escape", "nested/tag", "-leading", "space tag"):
        with pytest.raises(ValueError):
            remote.remote_paths(tag)


def test_strict_clean_gpu_and_second_check_precede_launch() -> None:
    rows = [
        _gpu(0, memory=1024, utilization=5),
        _gpu(1, memory=1025),
        _gpu(2, utilization=6),
        _gpu(3, processes=[{"pid": 7, "process_name": "python"}]),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]
    selected = _gpu(2)
    calls = []
    replacements = {
        "validate_kerberos": lambda **_kwargs: {"status": "PASS"},
        "_probe_remote_requirements": lambda: {"status": "PASS"},
        "_wait_for_clean_gpu": lambda **_kwargs: (
            calls.append("wait") or ([selected], selected)
        ),
        "committed_archive": lambda *_args, **_kwargs: b"archive",
        "_upload_source_archive": lambda **_kwargs: (
            calls.append("upload")
            or (
                remote.TASK_REMOTE_ROOT
                + "/staging/fresh/source"
            )
        ),
        "_run_remote_preflight": lambda **_kwargs: (
            calls.append("preflight")
        ),
        "validate_selected_gpu_still_clean": (
            lambda chosen, _rows: (
                calls.append("second-clean") or chosen
            )
        ),
        "_create_controller_dir": lambda **_kwargs: None,
        "_launch_worker": lambda **_kwargs: (
            calls.append("launch") or 321
        ),
        "_poll_worker": lambda **_kwargs: 0,
        "_run_remote_gates": lambda **_kwargs: None,
        "_write_remote_completion": lambda **_kwargs: None,
        "_download_terminal_bundle": lambda **_kwargs: {
            "local_verification": {
                "verified": True,
                "classification":
                    "GO_EXACT_BURST_MEDIUM_SPLIT_K",
            },
        },
    }
    originals = {
        name: getattr(remote, name) for name in replacements
    }
    base_originals = {
        "require_pushed_head": remote.base.require_pushed_head,
        "require_remote_destinations_absent":
            remote.base.require_remote_destinations_absent,
        "query_remote_gpu_rows": remote.base.query_remote_gpu_rows,
    }
    for name, value in replacements.items():
        setattr(remote, name, value)
    remote.base.require_pushed_head = lambda _root: "a" * 40
    remote.base.require_remote_destinations_absent = (
        lambda _paths: None
    )
    remote.base.query_remote_gpu_rows = lambda: [selected]
    try:
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "fresh"
            result = remote.run_controller(
                remote.parse_args([
                    "--run-tag", "fresh",
                    "--source-commit", "a" * 40,
                    "--model", remote.MODEL_PATH,
                    "--local-destination", str(destination),
                    "--repetitions", "3",
                    "--warmup-repetitions", "1",
                ])
            )
    finally:
        for name, value in originals.items():
            setattr(remote, name, value)
        for name, value in base_originals.items():
            setattr(remote.base, name, value)
    assert calls.index("wait") < calls.index("second-clean")
    assert calls.index("second-clean") < calls.index("launch")
    assert result["status"] == "COMPLETE"


def test_worker_gate_and_verifier_are_source_bound() -> None:
    assert all(
        "/data00/home/sitian/pytest-site" in command
        for command in remote.preflight_commands()
    )
    captured = []
    original = remote._run_remote_checked
    remote._run_remote_checked = lambda command, **_kwargs: (
        captured.append(command)
        or subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="321\n",
            stderr="",
        )
    )
    paths = remote.remote_paths("fresh-tag")
    try:
        pid = remote._launch_worker(
            source=paths["staging"] + "/source",
            primary=paths["primary"],
            controller=paths["controller"],
            run_tag="fresh-tag",
            source_commit="a" * 40,
            model=remote.MODEL_PATH,
            repetitions=3,
            warmup_repetitions=1,
            gpu_index=1,
            dist_port=remote.dist_port_for_run_tag("fresh-tag"),
        )
        remote._run_remote_gates(
            source=paths["staging"] + "/source",
            primary=paths["primary"],
            controller=paths["controller"],
            gpu_index=1,
            dist_port=remote.dist_port_for_run_tag("fresh-tag"),
        )
    finally:
        remote._run_remote_checked = original
    assert pid == 321
    joined = "\n".join(captured)
    assert (
        joined.index("profile_exact_burst_medium_split_k.py")
        < joined.index("exact_burst_medium_split_k_gate.py")
        < joined.index("exact_burst_medium_split_k_verify.py")
    )
    for required in (
        "--generated-tokens 128",
        "--repetitions 3",
        "--warmup-repetitions 1",
        "--context-lengths 1025,1537,2049,2561,3073,3585,4090,6145",
        "--source-commit " + "a" * 40,
    ):
        assert required in joined


def test_download_inventory_receipts_and_source_are_strict() -> None:
    artifacts = {
        "performance_rows.jsonl": "a" * 64,
        "correctness_rows.jsonl": "b" * 64,
        "summary.json": "c" * 64,
    }
    inventory = [
        {"path": name} for name in (*artifacts, "manifest.json")
    ]
    assert remote.validate_terminal_download_inventory(
        inventory,
        manifest_artifacts=artifacts,
    ) == inventory
    receipt = {
        "verified": True,
        "classification": "GO_EXACT_BURST_MEDIUM_SPLIT_K",
        "manifest_verified": True,
        "raw_metrics_reconstructed": True,
    }
    assert remote.validate_verification_receipt_agreement(
        receipt,
        dict(receipt),
    ) == receipt
    with pytest.raises(ValueError, match="receipt disagreement"):
        remote.validate_verification_receipt_agreement(
            receipt,
            receipt | {"classification": "NO_GO_PERFORMANCE"},
        )
    source = Path(
        "tools/run_exact_burst_medium_split_k_remote.py"
    ).read_text(encoding="utf-8")
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
    assert remote.SOURCE_PATCH_SHA256 == hashlib.sha256(b"").hexdigest()
