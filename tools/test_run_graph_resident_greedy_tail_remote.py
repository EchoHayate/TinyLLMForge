#!/usr/bin/env python3
"""Safety contracts for the graph-tail remote controller."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import run_graph_resident_greedy_tail_remote as remote


def _gpu_row(
    index: int,
    *,
    memory_used_mib: int = 0,
    utilization_percent: int = 0,
    compute_processes=None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100-SXM4-80GB",
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": (
            []
            if compute_processes is None
            else compute_processes
        ),
    }


def test_remote_paths_are_confined_to_mounted_root() -> None:
    paths = remote.remote_paths(
        "20260822-qwen3-06b-graph-greedy-tail-r1"
    )
    assert set(paths) == {
        "staging",
        "primary",
        "controller",
    }
    assert all(
        path.startswith(remote.APPROVED_ROOT + "/")
        for path in paths.values()
    )
    assert all(
        "/graph-resident-greedy-tail/" in path
        for path in paths.values()
    )
    for forbidden in (
        "/tmp",
        "/private/tmp",
        "/data00/home/sitian/tllm/TinyLLMForge",
    ):
        assert all(
            forbidden not in path
            for path in paths.values()
        )


def test_gpu_admission_requires_strict_clean_state() -> None:
    rows = [
        _gpu_row(
            0,
            memory_used_mib=1024,
            utilization_percent=5,
        ),
        _gpu_row(1, memory_used_mib=1025),
        _gpu_row(2, utilization_percent=6),
        _gpu_row(
            3,
            compute_processes=[
                {"pid": 12, "process_name": "python"}
            ],
        ),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]


def test_second_admission_requires_same_clean_uuid() -> None:
    selected = _gpu_row(1)
    assert remote.validate_selected_gpu_still_clean(
        selected,
        [_gpu_row(0), _gpu_row(1)],
    ) == selected
    with pytest.raises(RuntimeError, match="no longer strict-clean"):
        remote.validate_selected_gpu_still_clean(
            selected,
            [_gpu_row(1, memory_used_mib=1025)],
        )
    changed_uuid = _gpu_row(1)
    changed_uuid["uuid"] = "GPU-replaced"
    with pytest.raises(RuntimeError, match="no longer strict-clean"):
        remote.validate_selected_gpu_still_clean(
            selected,
            [changed_uuid],
        )


def test_run_tag_and_local_destination_are_immutable() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        remote.ensure_local_destination_absent(root, "fresh-tag")
        (root / "used-tag").mkdir()
        with pytest.raises(
            ValueError,
            match="local run tag already exists",
        ):
            remote.ensure_local_destination_absent(
                root,
                "used-tag",
            )
    for tag in (
        "",
        "../escape",
        "nested/tag",
        "-leading",
        "space tag",
    ):
        with pytest.raises(ValueError):
            remote.remote_paths(tag)


def test_kerberos_guard_rejects_ttl_below_5400() -> None:
    now = datetime(
        2026,
        8,
        22,
        12,
        0,
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    output = "\n".join((
        "Credentials cache: FILE:/Users/bytedance/krb5cc_sitian",
        "Principal: sitian@BYTESEE.NET",
        (
            "Aug 22 11:00:00 2026  "
            "Aug 22 13:00:00 2026  "
            "krbtgt/BYTESEE.NET@BYTESEE.NET"
        ),
    ))

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=output,
            stderr="",
        )

    with pytest.raises(ValueError, match="remaining lifetime"):
        remote.validate_kerberos(
            command_runner=runner,
            now=lambda: now,
        )


def test_source_commit_must_equal_pushed_head() -> None:
    commit = "a" * 40
    assert remote.validate_source_commit(
        commit,
        pushed_head=commit,
    ) == commit
    with pytest.raises(
        ValueError,
        match="requested source commit",
    ):
        remote.validate_source_commit(
            "b" * 40,
            pushed_head=commit,
        )


def test_source_archive_is_limited_to_runtime_and_tools() -> None:
    assert remote.COMMITTED_ARCHIVE_PATHS == (
        "tinyvllm",
        "tools",
    )


def test_remote_preflight_uses_all_required_tests() -> None:
    commands = remote.preflight_commands()
    assert all(
        command.startswith(remote.REMOTE_PYTHON + " ")
        for command in commands
    )
    assert {
        "tools/test_graph_resident_greedy_tail.py",
        "tools/test_greedy_sampling_fast_path.py",
        "tools/test_model_runner_spec_verify.py",
        "tools/test_multi_sequence_cuda_graph_gate.py",
        "tools/test_chunked_prefill.py",
        "tools/test_profile_graph_resident_greedy_tail.py",
        "tools/test_graph_resident_greedy_tail_gate.py",
        "tools/test_graph_resident_greedy_tail_verify.py",
    } == {command.split()[-1] for command in commands}


def test_model_runner_preflight_runs_without_pytest_installed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            os.fspath(
                REPO_ROOT
                / "tools"
                / "test_model_runner_spec_verify.py"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "model runner spec_verify tests passed"
        in result.stdout
    )


def test_runtime_environment_is_isolated_below_staging() -> None:
    source = (
        remote.APPROVED_ROOT
        + "/graph-resident-greedy-tail/staging/tag/source"
    )
    prelude = remote.remote_runtime_prelude(
        source=source,
        gpu_index=2,
    )
    runtime = source.rsplit("/", 1)[0] + "/runtime"
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
    assert runtime in prelude
    assert "export CUDA_VISIBLE_DEVICES=2" in prelude
    assert "export PYTHONNOUSERSITE=1" in prelude


def test_worker_launch_is_source_bound_and_uses_new_worker() -> None:
    commands = []
    original = remote._run_remote_checked
    remote._run_remote_checked = (
        lambda command, **_kwargs: (
            commands.append(command)
            or subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="321\n",
                stderr="",
            )
        )
    )
    paths = remote.remote_paths("tag")
    source = paths["staging"] + "/source"
    try:
        assert remote._launch_worker(
            source=source,
            primary=paths["primary"],
            controller=paths["controller"],
            run_tag="tag",
            source_commit="a" * 40,
            gpu_index=1,
        ) == 321
    finally:
        remote._run_remote_checked = original
    assert len(commands) == 1
    assert (
        "tools/profile_graph_resident_greedy_tail.py"
        in commands[0]
    )
    assert "--generated-tokens 128" in commands[0]
    assert "--repetitions 5" in commands[0]
    assert "--warmup-repetitions 2" in commands[0]
    assert "CUDA_VISIBLE_DEVICES=1" in commands[0]


def test_terminal_inventory_requires_manifest_listed_files() -> None:
    manifest_artifacts = {
        name: "a" * 64
        for name in remote.MANIFEST_PRIMARY_FILES
    }
    manifest_artifacts.update({
        f"logits/sidecar-{index}.f32": "b" * 64
        for index in range(27)
    })
    inventory = [
        {"path": name}
        for name in (
            *manifest_artifacts,
            "manifest.sha256",
            "independent-verification.json",
        )
    ]
    assert remote.validate_terminal_download_inventory(
        inventory,
        manifest_artifacts=manifest_artifacts,
    ) == inventory
    with pytest.raises(
        ValueError,
        match="download is incomplete",
    ):
        remote.validate_terminal_download_inventory(
            inventory[:-3],
            manifest_artifacts=manifest_artifacts,
        )


def test_remote_requirements_pin_python_model_and_free_space() -> None:
    valid = {
        "python": {
            "path": remote.REMOTE_PYTHON,
            "is_file": True,
            "is_executable": True,
        },
        "model": {
            "path": remote.MODEL_PATH,
            "is_dir": True,
            "config_is_file": True,
        },
        "approved_root": {
            "path": remote.APPROVED_ROOT,
            "is_dir": True,
            "free_bytes": 100 * 1024**3,
        },
    }
    assert remote.validate_remote_requirements(valid) == valid
    invalid = json.loads(json.dumps(valid))
    invalid["approved_root"]["free_bytes"] = 1
    with pytest.raises(
        ValueError,
        match="remote requirements",
    ):
        remote.validate_remote_requirements(invalid)


def main() -> None:
    test_remote_paths_are_confined_to_mounted_root()
    test_gpu_admission_requires_strict_clean_state()
    test_second_admission_requires_same_clean_uuid()
    test_run_tag_and_local_destination_are_immutable()
    test_kerberos_guard_rejects_ttl_below_5400()
    test_source_commit_must_equal_pushed_head()
    test_source_archive_is_limited_to_runtime_and_tools()
    test_remote_preflight_uses_all_required_tests()
    test_model_runner_preflight_runs_without_pytest_installed()
    test_runtime_environment_is_isolated_below_staging()
    test_worker_launch_is_source_bound_and_uses_new_worker()
    test_terminal_inventory_requires_manifest_listed_files()
    test_remote_requirements_pin_python_model_and_free_space()
    print("graph-tail remote controller tests passed")


if __name__ == "__main__":
    main()
