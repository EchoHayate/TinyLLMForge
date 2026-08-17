from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace


RUNNER_PATH = (
    Path(__file__).resolve().parent
    / "run_qwen35_tp4_real_root_logit_gate_remote.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_root_logit_remote_runner_under_test",
        os.fspath(RUNNER_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {fragment!r}")


def test_runner_freezes_remote_identity_and_exact_five_inventory():
    runner = _load_runner()
    assert runner.REMOTE_TARGET == "sitian@10.232.195.203"
    assert runner.REMOTE_PYTHON.endswith("/tllm/env/bin/python")
    assert runner.FROZEN_SOURCE_TAG == (
        "qwen35-tp4-source-prep-20260729-170818"
    )
    assert runner.FROZEN_SOURCE_TREE_SHA256 == (
        "ec19a8fa68abfba72e9594bdd1e05428b0add9169d3dbdde24190686c013411f"
    )
    assert runner.EXACT_ARTIFACT_NAMES == {
        "tp4_real_root_logit_correctness.json",
        "reference_logits.pt",
        "native_rank0_logits.pt",
        "rank_evidence.json",
        "source_manifest.json",
    }
    assert runner.NATIVE_SMOKE_ARTIFACT_NAMES == {
        "native_smoke.json",
        "native_rank0_logits.pt",
        "rank_evidence.json",
    }
    assert "native-smoke" in runner.MODES


def test_run_tags_and_local_destinations_fail_closed():
    runner = _load_runner()
    assert runner.validate_run_tag("qwen35-tp4-authority-20260728-r1") == (
        "qwen35-tp4-authority-20260728-r1"
    )
    for value in ("", "../escape", "a/b", "a b", "a.b", "中文"):
        _expect_value_error(
            lambda value=value: runner.validate_run_tag(value),
            "run tag",
        )
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        destination = runner.require_new_local_run_dir(
            root,
            "qwen35-tp4-authority-r1",
        )
        assert destination == (
            root
            / runner.LOCAL_RUN_ROOT
            / "qwen35-tp4-authority-r1"
        )
        destination.mkdir(parents=True)
        _expect_value_error(
            lambda: runner.require_new_local_run_dir(
                root,
                "qwen35-tp4-authority-r1",
            ),
            "already exists",
        )


def test_ssh_command_is_batch_mode_without_shell_interpolation():
    runner = _load_runner()
    command = runner.build_ssh_command([
        runner.REMOTE_PYTHON,
        "-c",
        "print('ok')",
    ])
    assert command[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert "ControlMaster=no" in command
    assert "ServerAliveInterval=30" in command
    assert "ServerAliveCountMax=3" in command
    assert command[-2] == runner.REMOTE_TARGET
    assert command[-1].endswith(
        """-c 'print('"'"'ok'"'"')'"""
    )


def test_preflight_classification_requires_four_selected_rows():
    runner = _load_runner()
    blocked = runner.classify_preflight_payload({
        "eligible": False,
        "error": "four eligible GPUs are required",
        "source_tree_sha256": runner.FROZEN_SOURCE_TREE_SHA256,
        "rows": [{"gpu_index": index} for index in range(8)],
    })
    assert blocked["status"] == "BLOCKED"
    assert blocked["source_tree_sha256"] == (
        runner.FROZEN_SOURCE_TREE_SHA256
    )
    assert blocked["selected"] == []
    ready = runner.classify_preflight_payload({
        "eligible": True,
        "source_tree_sha256": runner.FROZEN_SOURCE_TREE_SHA256,
        "selected": [
            {
                "rank": rank,
                "gpu_index": rank + 4,
                "gpu_uuid": f"GPU-{rank}",
                "compute_processes": [],
                "free_bytes": 30 * 1024**3,
            }
            for rank in range(4)
        ],
        "rows": [],
    })
    assert ready["status"] == "READY"
    assert ready["source_tree_sha256"] == (
        runner.FROZEN_SOURCE_TREE_SHA256
    )
    assert [row["rank"] for row in ready["selected"]] == [0, 1, 2, 3]
    _expect_value_error(
        lambda: runner.classify_preflight_payload({
            "eligible": True,
            "source_tree_sha256": runner.FROZEN_SOURCE_TREE_SHA256,
            "selected": [{"rank": 0}],
            "rows": [],
        }),
        "four selected",
    )
    _expect_value_error(
        lambda: runner.classify_preflight_payload({
            "eligible": False,
            "error": "four eligible GPUs are required",
            "source_tree_sha256": "0" * 64,
            "rows": [],
        }),
        "source tree",
    )
    assert runner.result_exit_code("preflight", blocked) == 2
    assert runner.result_exit_code("preflight", ready) == 0
    assert runner.result_exit_code(
        "verify-only",
        {"classification": "PASS"},
    ) == 0


def _tar_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_extract_exact_five_rejects_missing_extra_and_unsafe_paths():
    runner = _load_runner()
    valid = {
        name: name.encode("utf-8")
        for name in runner.EXACT_ARTIFACT_NAMES
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        destination = Path(temporary_directory) / "run"
        runner.extract_exact_five_tar(
            _tar_bytes(valid),
            destination,
        )
        assert {path.name for path in destination.iterdir()} == (
            runner.EXACT_ARTIFACT_NAMES
        )
    for files, fragment in (
        (
            {
                name: payload
                for name, payload in valid.items()
                if name != "rank_evidence.json"
            },
            "inventory",
        ),
        ({**valid, "extra.txt": b"x"}, "inventory"),
        ({**valid, "../escape": b"x"}, "path"),
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _expect_value_error(
                lambda files=files: runner.extract_exact_five_tar(
                    _tar_bytes(files),
                    Path(temporary_directory) / "run",
                ),
                fragment,
            )


def test_remote_download_rejects_any_extra_top_level_entry():
    runner = _load_runner()
    command = runner._remote_tar_command("qwen35-tp4-authority-r1")
    remote = command[-1]
    assert "find . -mindepth 1 -maxdepth 1 | wc -l" in remote
    assert "find . -mindepth 1 -maxdepth 1 -type f | wc -l" in remote


def test_streamed_artifact_verifier_retries_transient_ssh_disconnect():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact_dir = Path(temporary_directory) / "artifacts"
        artifact_dir.mkdir(parents=True)
        for name in runner.EXACT_ARTIFACT_NAMES:
            (artifact_dir / name).write_bytes(name.encode("utf-8"))
        attempts = 0

        def command_runner(_command, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return SimpleNamespace(
                    returncode=255,
                    stdout=b"",
                    stderr=b"Connection closed by UNKNOWN port 65535",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "classification": "PASS",
                    "case_ids": ["p17", "p65", "synthetic"],
                    "ranks": [0, 1, 2, 3],
                    "checks": 1,
                }).encode("utf-8"),
                stderr=b"",
            )

        result = runner.verify_downloaded_artifacts(
            artifact_dir,
            command_runner=command_runner,
        )

        assert attempts == 2
        assert result["classification"] == "PASS"
        assert not (artifact_dir.parent / "independent_verification.json").exists()


def test_default_runner_injects_exact_kerberos_environment():
    runner = _load_runner()
    calls = []
    original = runner.subprocess.run

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    runner.subprocess.run = fake_run
    try:
        runner._run(["ssh", "sitian@10.232.195.203", "true"])
    finally:
        runner.subprocess.run = original

    assert calls[0][1]["env"]["KRB5CCNAME"] == (
        "FILE:/Users/bytedance/krb5cc_sitian"
    )


def test_authority_orders_preflight_run_download_verify():
    runner = _load_runner()
    events = []

    def preflight(**_kwargs):
        events.append("preflight")
        return {"status": "READY", "selected": [{}, {}, {}, {}]}

    def run(**_kwargs):
        events.append("run")
        return {"status": "REMOTE_PASS"}

    def download(**_kwargs):
        events.append("download")
        return {"status": "DOWNLOADED"}

    def verify(**_kwargs):
        events.append("verify")
        return {"classification": "PASS"}

    result = runner.execute_authority(
        run_tag="qwen35-tp4-authority-r1",
        repo_root=Path("/repo"),
        preflight=preflight,
        run=run,
        download=download,
        verify=verify,
    )
    assert events == ["preflight", "run", "download", "verify"]
    assert result["classification"] == "PASS"
    assert runner.result_exit_code("authority", result) == 0

    events.clear()

    def blocked(**_kwargs):
        events.append("preflight")
        return {"status": "BLOCKED", "selected": []}

    blocked_result = runner.execute_authority(
        run_tag="qwen35-tp4-authority-r2",
        repo_root=Path("/repo"),
        preflight=blocked,
        run=run,
        download=download,
        verify=verify,
    )
    assert blocked_result == {
        "status": "BLOCKED",
        "selected": [],
    }
    assert runner.result_exit_code("authority", blocked_result) == 2
    assert events == ["preflight"]


def test_native_smoke_orders_preflight_before_native_and_blocks_cleanly():
    runner = _load_runner()
    events = []

    def ready(**_kwargs):
        events.append("preflight")
        return {"status": "READY", "selected": [{}, {}, {}, {}]}

    def smoke(**_kwargs):
        events.append("native-smoke-run")
        return {
            "classification": "NATIVE_SMOKE_PASS",
            "artifact_names": sorted(
                runner.NATIVE_SMOKE_ARTIFACT_NAMES
            ),
        }

    result = runner.execute_native_smoke(
        run_tag="qwen35-tp4-native-smoke-r1",
        repo_root=Path("/repo"),
        preflight=ready,
        run_smoke=smoke,
    )
    assert events == ["preflight", "native-smoke-run"]
    assert result["classification"] == "NATIVE_SMOKE_PASS"
    assert runner.result_exit_code("native-smoke", result) == 0

    events.clear()

    def blocked(**_kwargs):
        events.append("preflight")
        return {"status": "BLOCKED", "selected": []}

    blocked_result = runner.execute_native_smoke(
        run_tag="qwen35-tp4-native-smoke-r2",
        repo_root=Path("/repo"),
        preflight=blocked,
        run_smoke=smoke,
    )
    assert blocked_result == {
        "status": "BLOCKED",
        "selected": [],
    }
    assert runner.result_exit_code(
        "native-smoke",
        blocked_result,
    ) == 2
    assert events == ["preflight"]


def test_native_smoke_remote_script_is_source_bound_and_reference_free():
    runner = _load_runner()
    script = runner._native_smoke_script(
        "qwen35-tp4-native-smoke-r1"
    )
    assert runner.FROZEN_PREFLIGHT in script
    assert runner.FROZEN_MANIFEST in script
    assert runner.FROZEN_SOURCE_TREE_SHA256 in script
    assert "internal-native-rank" in script
    assert "internal-reference" not in script
    assert "execute_source_bound_run" not in script
    assert "validate_rank_evidence" in script
    assert "NATIVE_SMOKE_PASS" in script
    assert "native_smoke.json" in script
    assert "reference_logits.pt" not in script
    assert "tp4_real_root_logit_correctness.json" not in script


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"qwen35 TP4 remote runner tests passed ({len(tests)} tests)")


if __name__ == "__main__":
    _run()
