#!/usr/bin/env python3
"""Safety contracts for the exact-burst remote controller."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

try:
    import pytest
except ModuleNotFoundError:
    class _Raises:
        def __init__(self, expected, *, match=None):
            self.expected = expected
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, _traceback):
            if exception_type is None:
                raise AssertionError(
                    f"did not raise {self.expected!r}"
                )
            if not issubclass(exception_type, self.expected):
                return False
            if (
                self.match is not None
                and re.search(self.match, str(exception)) is None
            ):
                raise AssertionError(
                    f"{exception!r} does not match {self.match!r}"
                )
            return True

    class _PytestCompat:
        @staticmethod
        def raises(expected, *, match=None):
            return _Raises(expected, match=match)

    pytest = _PytestCompat()


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import run_exact_greedy_decode_burst_remote as remote


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
            [] if compute_processes is None else compute_processes
        ),
    }


def test_remote_paths_and_runtime_are_confined_to_mounted_root() -> None:
    paths = remote.remote_paths(
        "20260822-qwen3-06b-exact-burst-r1"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    expected_root = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/exact-greedy-decode-burst"
    )
    assert all(
        path.startswith(expected_root + "/")
        for path in paths.values()
    )
    source = paths["staging"] + "/source"
    prelude = remote.remote_runtime_prelude(
        source=source,
        gpu_index=2,
    )
    runtime = paths["staging"] + "/runtime"
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
    for forbidden in (
        "export TMPDIR=/tmp",
        "export TMP=/tmp",
        "export TEMP=/tmp",
        "/private/tmp",
        "/data00/home/sitian/tllm/TinyLLMForge",
    ):
        assert forbidden not in prelude


def test_run_tag_and_source_commit_are_immutable() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        remote.ensure_local_destination_absent(root, "fresh-tag")
        (root / "used-tag").mkdir()
        with pytest.raises(
            ValueError,
            match="local run tag already exists",
        ):
            remote.ensure_local_destination_absent(root, "used-tag")
    for tag in ("", "../escape", "nested/tag", "-leading", "space tag"):
        with pytest.raises(ValueError):
            remote.remote_paths(tag)
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


def test_kerberos_guard_is_fail_fast_and_never_refreshes() -> None:
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
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=output,
            stderr="",
        )

    with pytest.raises(ValueError, match="remaining lifetime"):
        remote.validate_kerberos(
            command_runner=runner,
            now=lambda: now,
        )
    assert calls == [["klist"]]


def test_gpu_admission_and_second_probe_require_same_clean_uuid() -> None:
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
    selected = _gpu_row(0)
    assert remote.validate_selected_gpu_still_clean(
        selected,
        [selected],
    ) == selected
    changed = _gpu_row(0)
    changed["uuid"] = "GPU-replaced"
    with pytest.raises(RuntimeError, match="no longer strict-clean"):
        remote.validate_selected_gpu_still_clean(
            selected,
            [changed],
        )


def test_archive_preflight_and_worker_are_source_bound() -> None:
    assert remote.COMMITTED_ARCHIVE_PATHS == ("tinyvllm", "tools")
    commands = remote.preflight_commands()
    required = {
        "tools/test_exact_greedy_decode_burst.py",
        "tools/test_scheduler_prepared_postprocess.py",
        "tools/test_model_runner_spec_verify.py",
        "tools/test_llm_engine_exact_greedy_decode_burst.py",
        "tools/test_profile_exact_greedy_decode_burst.py",
        "tools/test_exact_greedy_decode_burst_gate.py",
        "tools/test_exact_greedy_decode_burst_verify.py",
        "tools/test_multi_sequence_cuda_graph_gate.py",
        "tools/test_chunked_prefill.py",
    }
    assert required == {command.split()[-1] for command in commands}
    pytest_commands = {
        command.split()[-1]: command
        for command in commands
        if "-m pytest -q" in command
    }
    assert set(pytest_commands) == {
        "tools/test_scheduler_prepared_postprocess.py",
        "tools/test_llm_engine_exact_greedy_decode_burst.py",
    }
    assert all(
        remote.REMOTE_PYTEST_SITE in command
        for command in pytest_commands.values()
    )

    captured = []
    original = remote._run_remote_checked
    remote._run_remote_checked = (
        lambda command, **_kwargs: (
            captured.append(command)
            or subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="321\n",
                stderr="",
            )
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
    finally:
        remote._run_remote_checked = original
    assert pid == 321
    assert len(captured) == 1
    command = captured[0]
    assert "tools/profile_exact_greedy_decode_burst.py" in command
    assert "--generated-tokens 128" in command
    assert "--repetitions 5" in command
    assert "--warmup-repetitions 2" in command
    assert "CUDA_VISIBLE_DEVICES=1" in command
    for forbidden in (
        "kinit",
        "renew",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
    ):
        assert forbidden not in command


def test_remote_gates_keep_selected_gpu_and_never_construct_kill() -> None:
    captured = []
    original = remote._run_remote_checked
    remote._run_remote_checked = (
        lambda command, **_kwargs: captured.append(command)
    )
    paths = remote.remote_paths("fresh-tag")
    try:
        remote._run_remote_gates(
            source=paths["staging"] + "/source",
            primary=paths["primary"],
            gpu_index=3,
        )
    finally:
        remote._run_remote_checked = original
    assert len(captured) == 1
    command = captured[0]
    assert "export CUDA_VISIBLE_DEVICES=3" in command
    assert "tools/exact_greedy_decode_burst_gate.py" in command
    assert "tools/exact_greedy_decode_burst_verify.py" in command
    controller_source = (
        REPO_ROOT / "tools/run_exact_greedy_decode_burst_remote.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "kinit",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "os.kill(",
        "os.killpg(",
    ):
        assert forbidden not in controller_source


def test_worker_poll_fails_if_wrapper_disappears_before_exit_receipt() -> None:
    calls = []
    original_run_remote = remote.base._run_remote
    original_validate_kerberos = remote.validate_kerberos
    remote.base._run_remote = lambda command: (
        calls.append(command)
        or subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"state": "missing"}),
            stderr="",
        )
    )
    remote.validate_kerberos = lambda **_kwargs: {"status": "PASS"}
    try:
        with pytest.raises(
            RuntimeError,
            match="disappeared before writing exit code",
        ):
            remote._poll_worker(
                controller="/approved/controller/fresh-tag",
                worker_pid=321,
                poll_interval_seconds=1,
            )
    finally:
        remote.base._run_remote = original_run_remote
        remote.validate_kerberos = original_validate_kerberos
    assert len(calls) == 1
    assert "/proc/321" in calls[0]


def test_worker_poll_does_not_reapply_launch_kerberos_threshold() -> None:
    calls = []
    original_run_remote = remote.base._run_remote
    original_validate_kerberos = remote.validate_kerberos
    remote.base._run_remote = lambda command: (
        calls.append(command)
        or subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "state": "finished",
                "exitcode": 0,
            }),
            stderr="",
        )
    )

    def reject_launch_threshold(**_kwargs):
        raise AssertionError(
            "worker polling reapplied the launch Kerberos threshold"
        )

    remote.validate_kerberos = reject_launch_threshold
    try:
        assert remote._poll_worker(
            controller="/approved/controller/fresh-tag",
            worker_pid=321,
            poll_interval_seconds=1,
        ) == 0
    finally:
        remote.base._run_remote = original_run_remote
        remote.validate_kerberos = original_validate_kerberos
    assert len(calls) == 1
    assert "/proc/321" in calls[0]


def test_controller_applies_strict_kerberos_only_before_launch() -> None:
    selected = _gpu_row(2)
    kerberos_calls = []
    patches = {
        "validate_kerberos": remote.validate_kerberos,
        "_probe_remote_requirements": remote._probe_remote_requirements,
        "_wait_for_clean_gpu": remote._wait_for_clean_gpu,
        "committed_archive": remote.committed_archive,
        "_upload_source_archive": remote._upload_source_archive,
        "_run_remote_preflight": remote._run_remote_preflight,
        "validate_selected_gpu_still_clean":
            remote.validate_selected_gpu_still_clean,
        "_create_controller_dir": remote._create_controller_dir,
        "_launch_worker": remote._launch_worker,
        "_poll_worker": remote._poll_worker,
        "_run_remote_gates": remote._run_remote_gates,
        "_write_remote_completion": remote._write_remote_completion,
        "_download_terminal_bundle": remote._download_terminal_bundle,
        "require_pushed_head": remote.base.require_pushed_head,
        "require_remote_destinations_absent":
            remote.base.require_remote_destinations_absent,
        "query_remote_gpu_rows": remote.base.query_remote_gpu_rows,
    }

    def launch_only_kerberos(**_kwargs):
        kerberos_calls.append(True)
        if len(kerberos_calls) > 2:
            raise AssertionError(
                "controller reapplied the launch Kerberos threshold"
            )
        return {"status": "PASS"}

    remote.validate_kerberos = launch_only_kerberos
    remote.base.require_pushed_head = lambda _root: "a" * 40
    remote._probe_remote_requirements = lambda: {"status": "PASS"}
    remote.base.require_remote_destinations_absent = lambda _paths: None
    remote._wait_for_clean_gpu = lambda **_kwargs: ([selected], selected)
    remote.committed_archive = lambda *_args, **_kwargs: b"archive"
    remote._upload_source_archive = (
        lambda **_kwargs: "/approved/staging/source"
    )
    remote._run_remote_preflight = lambda **_kwargs: None
    remote.base.query_remote_gpu_rows = lambda: [selected]
    remote.validate_selected_gpu_still_clean = (
        lambda chosen, _rows: chosen
    )
    remote._create_controller_dir = lambda *_args, **_kwargs: None
    remote._launch_worker = lambda **_kwargs: 321
    remote._poll_worker = lambda **_kwargs: 0
    remote._run_remote_gates = lambda **_kwargs: None
    remote._write_remote_completion = lambda **_kwargs: None
    remote._download_terminal_bundle = lambda **_kwargs: {
        "local_verification": {"status": "PASS"},
    }
    try:
        with TemporaryDirectory() as temporary:
            result = remote.run_controller(remote.parse_args([
                "--run-tag",
                "fresh-controller-tag",
                "--source-commit",
                "a" * 40,
                "--local-artifact-root",
                temporary,
            ]))
    finally:
        remote.validate_kerberos = patches["validate_kerberos"]
        remote._probe_remote_requirements = patches[
            "_probe_remote_requirements"
        ]
        remote._wait_for_clean_gpu = patches["_wait_for_clean_gpu"]
        remote.committed_archive = patches["committed_archive"]
        remote._upload_source_archive = patches[
            "_upload_source_archive"
        ]
        remote._run_remote_preflight = patches[
            "_run_remote_preflight"
        ]
        remote.validate_selected_gpu_still_clean = patches[
            "validate_selected_gpu_still_clean"
        ]
        remote._create_controller_dir = patches[
            "_create_controller_dir"
        ]
        remote._launch_worker = patches["_launch_worker"]
        remote._poll_worker = patches["_poll_worker"]
        remote._run_remote_gates = patches["_run_remote_gates"]
        remote._write_remote_completion = patches[
            "_write_remote_completion"
        ]
        remote._download_terminal_bundle = patches[
            "_download_terminal_bundle"
        ]
        remote.base.require_pushed_head = patches[
            "require_pushed_head"
        ]
        remote.base.require_remote_destinations_absent = patches[
            "require_remote_destinations_absent"
        ]
        remote.base.query_remote_gpu_rows = patches[
            "query_remote_gpu_rows"
        ]
    assert len(kerberos_calls) == 2
    assert result["status"] == "COMPLETE"


def test_dependency_light_preflight_runs_without_site_packages() -> None:
    expected_markers = {
        "tools/test_exact_greedy_decode_burst.py":
            "exact greedy decode burst tests passed",
        "tools/test_model_runner_spec_verify.py":
            "model runner spec_verify tests passed",
        "tools/test_profile_exact_greedy_decode_burst.py":
            "exact greedy decode burst profile tests passed",
        "tools/test_exact_greedy_decode_burst_gate.py":
            "exact greedy decode-burst gate tests passed",
        "tools/test_exact_greedy_decode_burst_verify.py":
            "exact greedy decode-burst verifier tests passed",
    }
    for script, marker in expected_markers.items():
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                os.fspath(REPO_ROOT / script),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, script + "\n" + result.stderr
        assert marker in result.stdout


def test_terminal_inventory_requires_all_manifest_files() -> None:
    manifest_artifacts = {
        name: "a" * 64
        for name in remote.MANIFEST_PRIMARY_FILES
    }
    manifest_artifacts.update({
        f"logits/sidecar-{index}.f32": "b" * 64
        for index in range(48)
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
    with pytest.raises(ValueError, match="download is incomplete"):
        remote.validate_terminal_download_inventory(
            inventory[:-3],
            manifest_artifacts=manifest_artifacts,
        )


def test_remote_and_local_verifier_receipts_must_match() -> None:
    receipt = {
        "schema_version":
            "exact-greedy-decode-burst.independent-verification.v1",
        "status": "PASS",
        "run_tag": "fresh-tag",
        "source_commit": "a" * 40,
        "reconstructed_classification":
            "GO_EXACT_GREEDY_DECODE_BURST",
        "reconstructed_selected_policy": "decode_burst_k8",
        "reconstructed_selected_burst_width": 8,
        "performance_row_count": 60,
        "correctness_row_count": 48,
        "comparison_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    assert remote.validate_verification_receipt_agreement(
        receipt,
        dict(receipt),
    ) == receipt
    changed = dict(receipt)
    changed["reconstructed_selected_policy"] = "decode_burst_k4"
    with pytest.raises(
        ValueError,
        match="verification receipt disagreement",
    ):
        remote.validate_verification_receipt_agreement(
            receipt,
            changed,
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
    with pytest.raises(ValueError, match="remote requirements"):
        remote.validate_remote_requirements(invalid)


def test_cli_exposes_frozen_gpu_wait_policy() -> None:
    args = remote.parse_args([
        "--run-tag",
        "fresh-tag",
        "--gpu-wait-timeout-seconds",
        "21600",
        "--gpu-poll-interval-seconds",
        "60",
    ])
    assert args.gpu_wait_timeout_seconds == 21600
    assert args.gpu_poll_interval_seconds == 60


def main() -> None:
    test_remote_paths_and_runtime_are_confined_to_mounted_root()
    test_run_tag_and_source_commit_are_immutable()
    test_kerberos_guard_is_fail_fast_and_never_refreshes()
    test_gpu_admission_and_second_probe_require_same_clean_uuid()
    test_archive_preflight_and_worker_are_source_bound()
    test_remote_gates_keep_selected_gpu_and_never_construct_kill()
    test_worker_poll_fails_if_wrapper_disappears_before_exit_receipt()
    test_worker_poll_does_not_reapply_launch_kerberos_threshold()
    test_controller_applies_strict_kerberos_only_before_launch()
    test_dependency_light_preflight_runs_without_site_packages()
    test_terminal_inventory_requires_all_manifest_files()
    test_remote_and_local_verifier_receipts_must_match()
    test_remote_requirements_pin_python_model_and_free_space()
    test_cli_exposes_frozen_gpu_wait_policy()
    print("exact burst remote controller tests passed")


if __name__ == "__main__":
    main()
