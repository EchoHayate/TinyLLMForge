#!/usr/bin/env python3
"""Safety contracts for the ragged-coalescing remote controller."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import run_exact_burst_ragged_coalescing_remote as remote


def _raises(expected, callback, *args, message=None, **kwargs):
    try:
        callback(*args, **kwargs)
    except expected as error:
        if message is not None:
            assert message in str(error), str(error)
        return error
    raise AssertionError(f"did not raise {expected.__name__}")


def _gpu_row(index: int, *, memory=0, utilization=0, processes=None):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100-SXM4-80GB",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def test_paths_runtime_and_tags_are_strictly_confined() -> None:
    paths = remote.remote_paths(
        "20260823-qwen3-06b-ragged-r1"
    )
    expected_root = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/"
        "exact-burst-ragged-coalescing"
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
        _raises(ValueError, remote.remote_paths, tag)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        remote.ensure_local_destination_absent(root, "fresh-tag")
        (root / "used-tag").mkdir()
        _raises(
            ValueError,
            remote.ensure_local_destination_absent,
            root,
            "used-tag",
            message="local run tag already exists",
        )


def test_source_gpu_admission_and_automatic_launch() -> None:
    commit = "a" * 40
    assert remote.validate_source_commit(
        commit,
        pushed_head=commit,
    ) == commit
    _raises(
        ValueError,
        remote.validate_source_commit,
        "b" * 40,
        pushed_head=commit,
        message="requested source commit",
    )
    assert remote.COMMITTED_ARCHIVE_PATHS == ("tinyvllm", "tools")
    assert remote.SOURCE_PATCH_SHA256 == hashlib.sha256(b"").hexdigest()
    rows = [
        _gpu_row(0, memory=1024, utilization=5),
        _gpu_row(1, memory=1025),
        _gpu_row(2, utilization=6),
        _gpu_row(3, processes=[{"pid": 7, "process_name": "python"}]),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]

    selected = _gpu_row(2)
    calls = []
    originals = {
        name: getattr(remote, name)
        for name in (
            "validate_kerberos",
            "_probe_remote_requirements",
            "_wait_for_clean_gpu",
            "committed_archive",
            "_upload_source_archive",
            "_run_remote_preflight",
            "validate_selected_gpu_still_clean",
            "_create_controller_dir",
            "_launch_worker",
            "_poll_worker",
            "_run_remote_gates",
            "_write_remote_completion",
            "_download_terminal_bundle",
        )
    }
    base_originals = {
        "require_pushed_head": remote.base.require_pushed_head,
        "require_remote_destinations_absent":
            remote.base.require_remote_destinations_absent,
        "query_remote_gpu_rows": remote.base.query_remote_gpu_rows,
    }
    remote.validate_kerberos = lambda **kwargs: (
        calls.append(("kerberos", kwargs["minimum_lifetime_seconds"]))
        or {"status": "PASS"}
    )
    remote.base.require_pushed_head = lambda _root: "a" * 40
    remote._probe_remote_requirements = lambda: {"status": "PASS"}
    remote.base.require_remote_destinations_absent = (
        lambda _paths: calls.append(("destinations", None))
    )
    remote._wait_for_clean_gpu = lambda **_kwargs: (
        calls.append(("wait", None)) or ([selected], selected)
    )
    remote.committed_archive = lambda *_args, **_kwargs: b"archive"
    remote._upload_source_archive = lambda **_kwargs: (
        calls.append(("upload", None))
        or "/approved/staging/source"
    )
    remote._run_remote_preflight = lambda **_kwargs: (
        calls.append(("preflight", None))
    )
    remote.base.query_remote_gpu_rows = lambda: [selected]
    remote.validate_selected_gpu_still_clean = (
        lambda chosen, _rows: chosen
    )
    remote._create_controller_dir = lambda **_kwargs: (
        calls.append(("controller", None))
    )
    remote._launch_worker = lambda **_kwargs: (
        calls.append(("launch", None)) or 321
    )
    remote._poll_worker = lambda **_kwargs: (
        calls.append(("poll", None)) or 0
    )
    remote._run_remote_gates = lambda **_kwargs: (
        calls.append(("gates", None))
    )
    remote._write_remote_completion = lambda **_kwargs: None
    remote._download_terminal_bundle = lambda **_kwargs: {
        "local_verification": {"status": "PASS"},
    }
    try:
        with TemporaryDirectory() as temporary:
            result = remote.run_controller(remote.parse_args([
                "--run-tag", "fresh-controller-tag",
                "--source-commit", "a" * 40,
                "--local-artifact-root", temporary,
            ]))
    finally:
        for name, value in originals.items():
            setattr(remote, name, value)
        for name, value in base_originals.items():
            setattr(remote.base, name, value)
    names = [name for name, _value in calls]
    assert names.index("wait") < names.index("launch")
    assert [
        value for name, value in calls if name == "kerberos"
    ] == [5400, 5400]
    kerberos_positions = [
        index
        for index, name in enumerate(names)
        if name == "kerberos"
    ]
    assert names.index("upload") < kerberos_positions[1] < (
        names.index("preflight")
    )
    assert result["status"] == "COMPLETE"


def test_preflight_worker_gate_and_verifier_are_source_bound() -> None:
    commands = remote.preflight_commands()
    required = {
        "tools/test_exact_greedy_decode_burst.py",
        "tools/test_exact_greedy_decode_burst_split_phase.py",
        "tools/test_model_runner_spec_verify.py",
        "tools/test_profile_exact_greedy_decode_burst.py",
        "tools/test_profile_exact_burst_split_phase.py",
        "tools/test_profile_exact_burst_ragged_coalescing.py",
        "tools/test_exact_burst_ragged_coalescing_gate.py",
        "tools/test_exact_burst_ragged_coalescing_verify.py",
        "tools/test_scheduler_prepared_postprocess.py",
        "tools/test_llm_engine_exact_greedy_decode_burst.py",
        "tools/test_multi_sequence_cuda_graph_gate.py",
        "tools/test_chunked_prefill.py",
    }
    assert required == {command.split()[-1] for command in commands}
    captured = []
    original = remote._run_remote_checked
    remote._run_remote_checked = lambda command, **_kwargs: (
        captured.append(command)
        or subprocess.CompletedProcess(
            args=[], returncode=0, stdout="321\n", stderr=""
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
            gpu_index=1,
        )
        remote._run_remote_gates(
            source=paths["staging"] + "/source",
            primary=paths["primary"],
            gpu_index=1,
        )
    finally:
        remote._run_remote_checked = original
    assert pid == 321
    joined = "\n".join(captured)
    profiler = joined.index(
        "tools/profile_exact_burst_ragged_coalescing.py"
    )
    patch = joined.index("source.patch")
    gate = joined.index(
        "tools/exact_burst_ragged_coalescing_gate.py"
    )
    verifier = joined.index(
        "tools/exact_burst_ragged_coalescing_verify.py"
    )
    assert profiler < patch < gate < verifier
    assert "open('xb')" in joined
    for required_arg in (
        "--generated-tokens 128",
        "--repetitions 5",
        "--warmup-repetitions 2",
        "--prompt-lengths 256,2048,8192",
    ):
        assert required_arg in joined


def test_terminal_bundle_and_controller_source_are_strict() -> None:
    artifacts = {
        name: "a" * 64 for name in remote.MANIFEST_PRIMARY_FILES
    }
    artifacts.update({
        f"logits/sidecar-{index}.f32": "b" * 64
        for index in range(36)
    })
    inventory = [
        {"path": name}
        for name in (
            *artifacts,
            "manifest.sha256",
            "independent-verification.json",
        )
    ]
    assert remote.validate_terminal_download_inventory(
        inventory,
        manifest_artifacts=artifacts,
    ) == inventory
    receipt = {
        "schema_version": (
            "exact-burst-ragged-coalescing."
            "independent-verification.v1"
        ),
        "status": "PASS",
        "reconstructed_classification":
            "GO_EXACT_BURST_RAGGED_COALESCING",
        "reconstructed_selected_policy":
            "decode_burst_k8_split_phase_ragged",
    }
    assert remote.validate_verification_receipt_agreement(
        receipt,
        dict(receipt),
    ) == receipt
    source = (
        REPO_ROOT
        / "tools/run_exact_burst_ragged_coalescing_remote.py"
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


def main() -> None:
    test_paths_runtime_and_tags_are_strictly_confined()
    test_source_gpu_admission_and_automatic_launch()
    test_preflight_worker_gate_and_verifier_are_source_bound()
    test_terminal_bundle_and_controller_source_are_strict()
    print("exact burst ragged-coalescing remote controller tests passed")


if __name__ == "__main__":
    main()
