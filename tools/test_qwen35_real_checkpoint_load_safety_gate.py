from __future__ import annotations

import importlib.util
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "tools/qwen35_real_checkpoint_load_contract.py"
)
RUNNER_PATH = (
    ROOT / "tools/run_qwen35_real_checkpoint_load_gate_remote.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("qwen35_real_load_contract_test", CONTRACT_PATH)
runner = _load("qwen35_real_load_runner_test", RUNNER_PATH)


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _json_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _ready_preflight():
    source_hashes = {
        name: "1" * 64
        for name in runner.OWNED_SOURCE_FILES
    }
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "READY",
        "failure_reasons": [],
        "checks": {
            "source_identity": True,
            "remote_identity": True,
            "runtime_dependencies": True,
            "model_identity": True,
            "model_files": True,
            "proc_telemetry": True,
            "run_root_space": True,
            "cuda_disabled": True,
            "gpu0_idle": True,
            "payload_zero": True,
        },
        "remote_target": contract.REMOTE_TARGET,
        "observed_user": "sitian",
        "observed_hostname": "n232-195-203",
        "remote_python": contract.REMOTE_PYTHON,
        "python_version": "3.11.15",
        "packages": {
            "torch": "2.4.1",
            "safetensors": "0.4.5",
            "transformers": "5.8.1",
        },
        "model_repository": contract.MODEL_REPOSITORY,
        "model_revision": contract.MODEL_REVISION,
        "approved_model_manifest_path": (
            contract.APPROVED_MODEL_MANIFEST_PATH
        ),
        "approved_model_dir": contract.APPROVED_MODEL_DIR,
        "cuda_visible_devices": "",
        "cuda_initialized": False,
        "gpu_processes": [],
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "gpu_uuid": "GPU-57be086f-e967-c022-3832-93df4fc77bd0",
        "driver_version": "535.261.03",
        "source_tree_sha256": runner._source_tree_sha256(source_hashes),
        "source_file_sha256": source_hashes,
        "remote_source_file_sha256": dict(source_hashes),
        "model_manifest_sha256": (
            contract.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_index_header_sha256": "c" * 64,
        "config_sha256": contract.APPROVED_CONFIG_SHA256,
        "index_sha256": contract.APPROVED_INDEX_SHA256,
        "shards": [{
            "name": contract.APPROVED_SHARD_NAME,
            "expected_size": contract.APPROVED_SHARD_SIZE,
            "observed_size": contract.APPROVED_SHARD_SIZE,
            "expected_sha256": contract.APPROVED_SHARD_SHA256,
            "resolved_path": (
                f"{contract.APPROVED_MODEL_DIR}/"
                f"{contract.APPROVED_SHARD_NAME}"
            ),
            "inode": 123,
            "device": 456,
            "mode": 33188,
        }],
        "payload_open_count": 0,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "payload_identity_source": "approved_model_manifest",
        "proc_telemetry_available": True,
        "proc_meminfo": {
            "MemTotal": 1024,
            "MemAvailable": 512,
        },
        "proc_status_fields": {
            "VmRSS": True,
            "VmHWM": True,
        },
        "run_root_filesystem": {
            "device": "/dev/nvme0n1",
            "fstype": "ext4",
            "mountpoint": "/data00",
            "source": "/dev/nvme0n1",
        },
        "model_root_filesystem": {
            "device": "/dev/nvme0n1",
            "fstype": "ext4",
            "mountpoint": "/data00",
            "source": "/dev/nvme0n1",
        },
        "free_run_root_bytes": 2 << 30,
        "required_run_root_bytes": 1 << 30,
    }


def _case_row(budget_mib, repeat_index, order_index):
    return {
        "case_id": f"budget-{budget_mib}-repeat-{repeat_index}",
        "budget_bytes": budget_mib << 20,
        "repeat_index": repeat_index,
        "order_index": order_index,
        "warmup": False,
        "tensor_parallel_size": 2,
        "tensor_parallel_rank": 0,
        "assigned_bindings": 320,
        "source_tensors": 320,
        "tile_count": 488 if budget_mib == 8 else 386,
        "destination_bytes": 1881935712,
        "peak_tile_bytes": budget_mib << 20,
        "assignment_digest": "d" * 64,
        "expected_assignment_digest": "d" * 64,
        "all_handles_closed": True,
        "cuda_initialized": False,
        "cuda_allocated_bytes": 0,
        "wall_seconds": (
            10.0 + repeat_index
            if budget_mib == 8
            else 9.0 + repeat_index
        ),
        "user_cpu_seconds": 5.0,
        "system_cpu_seconds": 1.0,
        "minor_faults": 100,
        "major_faults": 0,
        "vmrss_bytes": 4 << 30,
        "vmhwm_bytes": 5 << 30,
        "voluntary_context_switches": 10,
        "involuntary_context_switches": 2,
        "returncode": 0,
    }


def _restore_record(status="RESTORED"):
    checks = {
        "target_state": True,
        "model_directory": True,
        "non_payload_files": True,
        "config_identity": True,
        "index_identity": True,
        "shard_inventory": True,
        "payload_zero": True,
    }
    reasons = []
    write_performed = status == "RESTORED"
    if status in ("INCOMPLETE", "CONFLICT"):
        checks["target_state"] = False
        reasons = [
            (
                "target manifest conflicts with approved bytes"
                if status == "CONFLICT"
                else "restore prerequisite failed"
            )
        ]
        write_performed = False
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "status": status,
        "checks": checks,
        "failure_reasons": reasons,
        "remote_target": contract.REMOTE_TARGET,
        "approved_model_manifest_path": (
            contract.APPROVED_MODEL_MANIFEST_PATH
        ),
        "approved_model_dir": contract.APPROVED_MODEL_DIR,
        "approved_manifest_sha256": (
            contract.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "observed_manifest_sha256": (
            contract.APPROVED_MODEL_MANIFEST_SHA256
            if status in ("RESTORED", "ALREADY_PRESENT")
            else None
        ),
        "non_payload_files": {
            name: dict(entry)
            for name, entry in contract.APPROVED_MODEL_FILES.items()
            if not name.endswith(".safetensors")
        },
        "config_model_type": "qwen3_5",
        "index_shard_names": [contract.APPROVED_SHARD_NAME],
        "observed_shard_names": [contract.APPROVED_SHARD_NAME],
        "shard": {
            "name": contract.APPROVED_SHARD_NAME,
            "expected_size": contract.APPROVED_SHARD_SIZE,
            "observed_size": contract.APPROVED_SHARD_SIZE,
            "inode": 123,
            "device": 456,
            "mode": 33188,
        },
        "payload_open_count": 0,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "write_performed": write_performed,
    }


def test_frozen_identity_matrix_thresholds_and_artifacts():
    assert contract.REMOTE_TARGET == "sitian@10.232.195.203"
    assert contract.REMOTE_PYTHON == (
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
    )
    assert contract.MODEL_REPOSITORY == "Qwen/Qwen3.5-2B"
    assert contract.MODEL_REVISION == (
        "15852e8c16360a2fea060d615a32b45270f8a8fc"
    )
    assert contract.APPROVED_MODEL_MANIFEST_PATH.endswith(
        "qwen35-2b-hybrid-acquire-20260723-222004/"
        "model_manifest.json"
    )
    assert contract.APPROVED_MODEL_DIR.endswith(
        "qwen35-2b-hybrid-acquire-20260723-222004/model"
    )
    assert contract.APPROVED_CONFIG_SHA256 == (
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
    )
    assert contract.APPROVED_INDEX_SHA256 == (
        "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
    )
    assert contract.APPROVED_SHARD_NAME == (
        "model.safetensors-00001-of-00001.safetensors"
    )
    assert contract.APPROVED_SHARD_SIZE == 4548221488
    assert contract.APPROVED_SHARD_SHA256 == (
        "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
    )
    canonical = contract.canonical_approved_model_manifest_bytes()
    assert hashlib.sha256(canonical).hexdigest() == (
        contract.APPROVED_MODEL_MANIFEST_SHA256
    )
    assert json.loads(canonical) == contract.APPROVED_MODEL_MANIFEST
    assert contract.APPROVED_MODEL_MANIFEST["files"] == (
        contract.APPROVED_MODEL_FILES
    )
    assert contract.CASE_ORDER_MIB == (8, 16, 16, 8, 8, 16)
    assert contract.MEASURED_REPEATS_PER_BUDGET == 3
    assert contract.MIN_WALL_TIME_IMPROVEMENT_FRACTION == 0.05
    assert contract.MAX_VMHWM_REGRESSION_BYTES == 16 << 20
    assert contract.REQUIRED_ARTIFACTS == (
        "manifest.json",
        "source_manifest.json",
        "preflight.json",
        "environment.json",
        "model_manifest.json",
        "processes.json",
        "gpu_processes.json",
        "case_rows.jsonl",
        "telemetry.jsonl",
        "summary.json",
        "independent_verification.json",
        "report.md",
        "stdout/worker.log",
        "stderr/worker.log",
    )


def test_preflight_and_result_validation_are_fail_closed():
    ready = _ready_preflight()
    assert contract.validate_preflight(ready) == ready
    for field, value, message in (
        ("remote_target", "wrong", "remote target"),
        ("cuda_visible_devices", "0", "CUDA_VISIBLE_DEVICES"),
        ("cuda_initialized", True, "CUDA"),
        ("gpu_processes", ["123 python"], "GPU process"),
        ("payload_open_count", 1, "payload open"),
        ("proc_telemetry_available", False, "/proc telemetry"),
    ):
        invalid = dict(ready)
        invalid[field] = value
        _expect_error(
            lambda invalid=invalid: contract.validate_preflight(invalid),
            message,
        )
    incomplete = dict(ready)
    incomplete["status"] = "INCOMPLETE"
    incomplete["failure_reasons"] = ["GPU0 is occupied"]
    incomplete["checks"] = {
        **ready["checks"],
        "gpu0_idle": False,
    }
    incomplete["gpu_processes"] = ["1234, python, 1024"]
    assert contract.validate_preflight(incomplete) == incomplete
    mismatched = dict(ready)
    mismatched["status"] = "INCOMPLETE"
    mismatched["failure_reasons"] = ["approved model identity mismatch"]
    mismatched["checks"] = {
        **ready["checks"],
        "model_identity": False,
    }
    mismatched["model_manifest_sha256"] = "f" * 64
    assert contract.validate_preflight(mismatched) == mismatched
    missing_model = dict(ready)
    missing_model["status"] = "INCOMPLETE"
    missing_model["failure_reasons"] = [
        "approved model manifest is missing",
        "approved model shard stat inventory is unavailable",
    ]
    missing_model["checks"] = {
        **ready["checks"],
        "model_identity": False,
        "model_files": False,
    }
    missing_model["model_manifest_sha256"] = None
    missing_model["config_index_header_sha256"] = None
    missing_model["config_sha256"] = None
    missing_model["index_sha256"] = None
    missing_model["shards"] = []
    assert contract.validate_preflight(missing_model) == missing_model
    invalid_status = dict(ready)
    invalid_status["status"] = "GO"
    _expect_error(
        lambda: contract.validate_preflight(invalid_status),
        "status",
    )

    rows = []
    order_index = 0
    for budget_mib in contract.CASE_ORDER_MIB:
        repeat_index = sum(
            row["budget_bytes"] == budget_mib << 20
            for row in rows
        )
        rows.append(_case_row(budget_mib, repeat_index, order_index))
        order_index += 1
    result = contract.classify_case_rows(rows)
    assert result["classification"] == "GO"
    assert result["correctness_passed"] is True
    assert result["wall_time_improvement_fraction"] > 0.05

    corrupt = [dict(row) for row in rows]
    corrupt[-1]["assignment_digest"] = "e" * 64
    incomplete = contract.classify_case_rows(corrupt)
    assert incomplete["classification"] == "INCOMPLETE"
    assert "assignment digest" in incomplete["reason"]


def test_restore_record_validation_is_fail_closed():
    for status in (
        "RESTORED",
        "ALREADY_PRESENT",
        "INCOMPLETE",
        "CONFLICT",
    ):
        record = _restore_record(status)
        assert contract.validate_restore_record(record) == record

    invalid = _restore_record()
    invalid["payload_open_count"] = 1
    _expect_error(
        lambda: contract.validate_restore_record(invalid),
        "payload open",
    )
    invalid = _restore_record()
    invalid["write_performed"] = False
    _expect_error(
        lambda: contract.validate_restore_record(invalid),
        "write",
    )
    invalid = _restore_record("CONFLICT")
    invalid["observed_manifest_sha256"] = (
        contract.APPROVED_MODEL_MANIFEST_SHA256
    )
    _expect_error(
        lambda: contract.validate_restore_record(invalid),
        "conflict",
    )


def test_runner_commands_are_bound_and_non_destructive():
    assert runner.OWNED_SOURCE_FILES == (
        "tools/qwen35_real_checkpoint_load_contract.py",
        "tools/qwen35_real_checkpoint_load_authorization.py",
        "tools/qwen35_real_checkpoint_load_worker.py",
        "tools/verify_qwen35_real_checkpoint_load_gate.py",
        "tools/run_qwen35_real_checkpoint_load_gate_remote.py",
        "tools/test_qwen35_real_checkpoint_load_authorization.py",
        "tools/test_qwen35_real_checkpoint_load_safety_gate.py",
    )
    command = runner.build_ssh_command(["python3", "-V"])
    assert command[:2] == ["ssh", "-S"]
    assert runner.SSH_CONTROL_PATH in command
    assert "BatchMode=yes" in command
    assert command[-2] == contract.REMOTE_TARGET
    assert command[-1] == "python3 -V"

    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "rm -rf",
        "pkill",
        "killall",
        "git reset",
        "git clean",
        "rsync ",
    ):
        assert forbidden not in source
    runner.reject_unimplemented_execution("preflight")
    runner.reject_unimplemented_execution("restore-model-manifest")
    runner.reject_unimplemented_execution("verify-only")
    runner.reject_unimplemented_execution("download-only")
    for mode in ("run",):
        _expect_error(
            lambda mode=mode: runner.reject_unimplemented_execution(mode),
            "not implemented",
        )


def test_source_tar_staging_and_manifest_are_exact():
    payload = runner.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == runner.OWNED_SOURCE_FILES
        for member in archive.getmembers():
            assert member.uid == 0
            assert member.gid == 0
            assert member.mtime == 0

    calls = []
    local_hashes = runner._source_hashes(ROOT)

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("input") is not None:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(local_hashes),
            "",
        )

    staged = runner.stage_owned_source(
        ROOT,
        "read-only-preflight-test",
        command_runner=command_runner,
    )
    assert staged["local_file_sha256"] == local_hashes
    assert staged["remote_file_sha256"] == local_hashes
    assert staged["source_tree_sha256"] == runner._source_tree_sha256(
        local_hashes
    )
    assert len(calls) == 2
    assert calls[0][0][:2] == ["ssh", "-S"]
    assert calls[0][1]["input"] == payload
    assert "test ! -e" in calls[0][0][-1]


def test_remote_preflight_script_is_stat_only_for_payloads():
    source_hashes = runner._source_hashes(ROOT)
    script = runner.build_remote_preflight_script(
        run_tag="read-only-script-test",
        source_file_sha256=source_hashes,
    )
    for required in (
        "model_manifest.json",
        "model.safetensors.index.json",
        "path.stat()",
        "payload_open_count",
        "payload_bytes_read",
        "nvidia-smi",
        "/proc/meminfo",
        "findmnt",
        "MANIFEST_PATH.is_file()",
    ):
        assert required in script
    for forbidden in (
        "snapshot_download",
        "HfApi",
        "torch.load",
        "safe_open",
        "read_bytes()",
        "mmap",
        "open(payload",
    ):
        assert forbidden not in script


def test_restore_script_is_zero_payload_and_conflict_rejecting():
    script = runner.build_restore_model_manifest_script(
        "restore-script-test",
    )
    for required in (
        "model.safetensors.index.json",
        "path.stat()",
        "os.link",
        "ALREADY_PRESENT",
        "CONFLICT",
        "payload_open_count",
        "payload_bytes_read",
        "restore_model_manifest.json",
    ):
        assert required in script
    for forbidden in (
        "safe_open",
        "torch.load",
        "snapshot_download",
        "HfApi",
        "mmap",
        "read_bytes()",
        "unlink(",
        "os.replace",
    ):
        assert forbidden not in script


def test_preflight_classification_is_explicit_and_fail_closed():
    ready = _ready_preflight()
    classified = runner.classify_preflight_payload(dict(ready))
    assert classified["status"] == "READY"
    assert classified["failure_reasons"] == []
    assert all(classified["checks"].values())

    occupied = dict(ready)
    occupied["gpu_processes"] = ["1234, python, 1024"]
    occupied = runner.classify_preflight_payload(occupied)
    assert occupied["status"] == "INCOMPLETE"
    assert occupied["checks"]["gpu0_idle"] is False
    assert any("GPU0" in reason for reason in occupied["failure_reasons"])

    payload_read = dict(ready)
    payload_read["payload_open_count"] = 1
    payload_read = runner.classify_preflight_payload(payload_read)
    assert payload_read["status"] == "INCOMPLETE"
    assert payload_read["checks"]["payload_zero"] is False


def test_run_remote_preflight_persists_only_authorized_artifacts():
    ready = _ready_preflight()
    staged = {
        "remote_source_dir": (
            f"{runner.REMOTE_RUN_ROOT}/read-only-orchestration/source"
        ),
        "local_file_sha256": ready["source_file_sha256"],
        "remote_file_sha256": ready["remote_source_file_sha256"],
        "source_tree_sha256": ready["source_tree_sha256"],
    }
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("input") is not None:
            payload = json.loads(kwargs["input"])
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(payload),
                "",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(ready),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        destination = Path(temporary_directory)
        record = runner.run_remote_preflight(
            ROOT,
            "read-only-orchestration",
            staged=staged,
            destination=destination,
            command_runner=command_runner,
        )
        assert record["status"] == "READY"
        assert json.loads(
            (destination / "preflight.json").read_text(encoding="utf-8")
        ) == record
        source_manifest = json.loads(
            (destination / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert source_manifest["source_tree_sha256"] == (
            staged["source_tree_sha256"]
        )
        assert sorted(path.name for path in destination.iterdir()) == [
            "preflight.json",
            "source_manifest.json",
        ]
        assert len(calls) == 2
        assert "CUDA_VISIBLE_DEVICES=" in calls[0][0][-1]
        assert calls[1][1]["input"]
        assert "preflight.json" in calls[1][0][-1]
        assert "source_manifest.json" in calls[1][0][-1]
        assert "atomic_json" in calls[1][0][-1]


def test_remote_artifact_round_trip_rejects_mismatch():
    ready = _ready_preflight()
    source_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "source_tree_sha256": ready["source_tree_sha256"],
    }

    def command_runner(command, **kwargs):
        payload = json.loads(kwargs["input"])
        payload["preflight"]["payload_open_count"] = 1
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload),
            "",
        )

    _expect_error(
        lambda: runner.persist_and_download_preflight_artifacts(
            "read-only-round-trip",
            preflight=ready,
            source_manifest=source_manifest,
            command_runner=command_runner,
        ),
        "round-trip mismatch",
    )


def test_run_remote_restore_persists_only_authorized_artifacts():
    restored = _restore_record()
    staged = {
        "remote_source_dir": (
            f"{runner.REMOTE_RUN_ROOT}/restore-orchestration/source"
        ),
        "local_file_sha256": {
            name: "1" * 64 for name in runner.OWNED_SOURCE_FILES
        },
        "remote_file_sha256": {
            name: "1" * 64 for name in runner.OWNED_SOURCE_FILES
        },
        "source_tree_sha256": runner._source_tree_sha256({
            name: "1" * 64 for name in runner.OWNED_SOURCE_FILES
        }),
    }
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("input") is not None:
            payload = json.loads(kwargs["input"])
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(payload),
                "",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(restored),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        destination = Path(temporary_directory)
        record = runner.run_remote_model_manifest_restore(
            ROOT,
            "restore-orchestration",
            staged=staged,
            destination=destination,
            command_runner=command_runner,
        )
        assert record == restored
        assert sorted(path.name for path in destination.iterdir()) == [
            "restore_model_manifest.json",
            "source_manifest.json",
        ]
        assert len(calls) == 2
        assert "CUDA_VISIBLE_DEVICES=" in calls[0][0][-1]
        assert "restore_model_manifest.json" in calls[1][0][-1]
        assert json.loads(
            (destination / "restore_model_manifest.json").read_text()
        ) == restored


def test_restore_artifact_round_trip_rejects_mismatch():
    restored = _restore_record()
    source_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "source_tree_sha256": "1" * 64,
    }

    def command_runner(command, **kwargs):
        payload = json.loads(kwargs["input"])
        payload["restore"]["payload_open_count"] = 1
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload),
            "",
        )

    _expect_error(
        lambda: runner.persist_and_download_restore_artifacts(
            "restore-round-trip",
            restore=restored,
            source_manifest=source_manifest,
            command_runner=command_runner,
        ),
        "round-trip mismatch",
    )


def test_execute_preflight_only_stages_and_audits():
    events = []
    staged = {
        "remote_source_dir": "/remote/source",
        "local_file_sha256": {"tools/example.py": "1" * 64},
        "remote_file_sha256": {"tools/example.py": "1" * 64},
        "source_tree_sha256": runner._source_tree_sha256(
            {"tools/example.py": "1" * 64}
        ),
    }
    expected = _ready_preflight()

    def stage(_repo_root, run_tag, command_runner):
        events.append(("stage", run_tag, command_runner))
        return staged

    def audit(
        _repo_root,
        run_tag,
        *,
        staged,
        destination,
        command_runner,
    ):
        events.append((
            "audit",
            run_tag,
            staged,
            Path(destination),
            command_runner,
        ))
        return expected

    marker = object()
    record = runner.execute_preflight(
        ROOT,
        "read-only-execute-test",
        command_runner=marker,
        stage_function=stage,
        audit_function=audit,
    )
    assert record == expected
    assert [event[0] for event in events] == ["stage", "audit"]
    assert events[0][2] is marker
    assert events[1][2] is staged
    assert events[1][3] == (
        ROOT
        / runner.LOCAL_RUN_ROOT
        / "read-only-execute-test"
    )
    assert events[1][4] is marker


def test_execute_restore_only_stages_and_restores():
    events = []
    staged = {
        "remote_source_dir": "/remote/source",
        "local_file_sha256": {"tools/example.py": "1" * 64},
        "remote_file_sha256": {"tools/example.py": "1" * 64},
        "source_tree_sha256": runner._source_tree_sha256(
            {"tools/example.py": "1" * 64}
        ),
    }
    expected = _restore_record()

    def stage(_repo_root, run_tag, command_runner):
        events.append(("stage", run_tag, command_runner))
        return staged

    def restore(
        _repo_root,
        run_tag,
        *,
        staged,
        destination,
        command_runner,
    ):
        events.append((
            "restore",
            run_tag,
            staged,
            Path(destination),
            command_runner,
        ))
        return expected

    marker = object()
    record = runner.execute_model_manifest_restore(
        ROOT,
        "restore-execute-test",
        command_runner=marker,
        stage_function=stage,
        restore_function=restore,
    )
    assert record == expected
    assert [event[0] for event in events] == ["stage", "restore"]
    assert events[0][2] is marker
    assert events[1][2] is staged
    assert events[1][3] == (
        ROOT / runner.LOCAL_RUN_ROOT / "restore-execute-test"
    )
    assert events[1][4] is marker


def test_execute_verify_only_is_local_and_writes_reports():
    events = []
    expected = {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": "INCOMPLETE",
        "expected_case_count": 6,
        "observed_case_count": 0,
        "guards": {},
        "reasons": ["synthetic local verification"],
        "claim_boundary": "test",
    }

    def verify(run_dir, write_report=False):
        events.append((Path(run_dir), write_report))
        return expected

    result = runner.execute_verify_only(
        ROOT,
        "verify-only-local-test",
        verifier_function=verify,
    )
    assert result == expected
    assert events == [(
        ROOT
        / runner.LOCAL_RUN_ROOT
        / "verify-only-local-test",
        True,
    )]


def test_execute_authorization_only_is_local():
    events = []
    expected = {
        "schema_version": contract.SCHEMA_VERSION,
        "decision": "BLOCKED",
        "worker_implementation_authorized": False,
        "worker_execution_authorized": False,
    }

    def authorize(run_dir, owned_source_files):
        events.append((Path(run_dir), tuple(owned_source_files)))
        return expected

    result = runner.execute_authorization_only(
        ROOT,
        "authorization-only-test",
        authorization_function=authorize,
    )
    assert result == expected
    assert events == [(
        ROOT
        / runner.LOCAL_RUN_ROOT
        / "authorization-only-test",
        runner.OWNED_SOURCE_FILES,
    )]


def test_download_only_is_manifest_first_atomic_and_verified():
    with tempfile.TemporaryDirectory() as temporary_directory:
        repo_root = Path(temporary_directory)
        run_tag = "download-only-test"
        input_paths = sorted(
            set(contract.REQUIRED_ARTIFACTS)
            - {
                "manifest.json",
                "independent_verification.json",
                "report.md",
            }
        )
        remote_payloads = {
            relative: (
                b"worker completed\n"
                if relative == "stdout/worker.log"
                else b""
                if relative == "stderr/worker.log"
                else _json_bytes({"artifact": relative})
            )
            for relative in input_paths
        }
        manifest = {
            "schema_version": contract.SCHEMA_VERSION,
            "artifacts": [{
                "path": relative,
                "size": len(remote_payloads[relative]),
                "sha256": hashlib.sha256(
                    remote_payloads[relative]
                ).hexdigest(),
            } for relative in input_paths],
        }
        remote_payloads["manifest.json"] = _json_bytes(manifest)
        reads = []
        verification = []

        def read_artifact(_run_tag, relative_path, command_runner):
            reads.append((relative_path, command_runner))
            return remote_payloads[relative_path]

        expected = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "GO",
        }

        def verify(run_dir, write_report=False):
            destination = Path(run_dir)
            verification.append((destination, write_report))
            assert destination.is_dir()
            assert (destination / "manifest.json").read_bytes() == (
                remote_payloads["manifest.json"]
            )
            for relative, payload in remote_payloads.items():
                assert (destination / relative).read_bytes() == payload
            return expected

        marker = object()
        result = runner.execute_download_only(
            repo_root,
            run_tag,
            command_runner=marker,
            artifact_reader=read_artifact,
            verifier_function=verify,
        )
        destination = (
            repo_root / runner.LOCAL_RUN_ROOT / run_tag
        )
        assert result == expected
        assert reads[0] == ("manifest.json", marker)
        assert [relative for relative, _ in reads[1:]] == input_paths
        assert verification == [(destination, True)]
        assert not tuple(destination.parent.glob(f".{run_tag}.*.tmp"))


def test_download_only_rejects_hash_mismatch_and_existing_destination():
    with tempfile.TemporaryDirectory() as temporary_directory:
        repo_root = Path(temporary_directory)
        run_tag = "download-only-reject-test"
        payload = b"{}\n"
        manifest = {
            "schema_version": contract.SCHEMA_VERSION,
            "artifacts": [{
                "path": relative,
                "size": len(payload),
                "sha256": "f" * 64,
            } for relative in sorted(
                set(contract.REQUIRED_ARTIFACTS)
                - {
                    "manifest.json",
                    "independent_verification.json",
                    "report.md",
                }
            )],
        }

        def read_artifact(_run_tag, relative_path, _command_runner):
            if relative_path == "manifest.json":
                return _json_bytes(manifest)
            return payload

        _expect_error(
            lambda: runner.execute_download_only(
                repo_root,
                run_tag,
                artifact_reader=read_artifact,
                verifier_function=lambda *_args, **_kwargs: {},
            ),
            "sha256 mismatch",
        )
        destination = repo_root / runner.LOCAL_RUN_ROOT / run_tag
        assert not destination.exists()
        destination.mkdir(parents=True)
        _expect_error(
            lambda: runner.execute_download_only(
                repo_root,
                run_tag,
                artifact_reader=read_artifact,
                verifier_function=lambda *_args, **_kwargs: {},
            ),
            "local run directory already exists",
        )


def test_download_manifest_and_remote_reader_are_fail_closed():
    input_paths = sorted(
        set(contract.REQUIRED_ARTIFACTS)
        - {
            "manifest.json",
            "independent_verification.json",
            "report.md",
        }
    )

    def manifest_with(paths):
        return _json_bytes({
            "schema_version": contract.SCHEMA_VERSION,
            "artifacts": [{
                "path": relative,
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            } for relative in paths],
        })

    unsafe = list(input_paths)
    unsafe[0] = "../escape.json"
    _expect_error(
        lambda: runner._validate_download_manifest(
            manifest_with(unsafe)
        ),
        "unsafe artifact path",
    )
    unexpected = [
        *input_paths,
        "independent_verification.json",
    ]
    _expect_error(
        lambda: runner._validate_download_manifest(
            manifest_with(unexpected)
        ),
        "unexpected artifact",
    )

    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"artifact-bytes",
            stderr=b"",
        )

    payload = runner.read_remote_artifact(
        "reader-test",
        "stdout/worker.log",
        command_runner,
    )
    assert payload == b"artifact-bytes"
    command, kwargs = calls[0]
    assert command[:2] == ["ssh", "-S"]
    assert command[-2] == contract.REMOTE_TARGET
    assert command[-1].startswith("cat -- ")
    assert command[-1].endswith(
        "/reader-test/stdout/worker.log"
    )
    assert kwargs == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }


def test_dry_run_json_persistence_executes_no_subprocess_or_ssh():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "nested/dry-run.json"
        completed = subprocess.run(
            (
                sys.executable,
                str(RUNNER_PATH),
                "dry-run",
                "--run-tag",
                "qwen35-real-load-dry-run",
                "--output-json",
                str(output),
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "QWEN35_FAIL_ON_SUBPROCESS": "1",
                "QWEN35_FAIL_ON_SSH": "1",
                "QWEN35_FAIL_ON_SAFETENSORS_PAYLOAD_OPEN": "1",
            },
        )
        printed = json.loads(completed.stdout)
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert printed == persisted
        assert persisted["mode"] == "dry-run"
        assert persisted["remote_target"] == contract.REMOTE_TARGET
        assert persisted["subprocess_count"] == 0
        assert persisted["ssh_count"] == 0
        assert persisted["payload_open_count"] == 0
        assert persisted["execution_authorized"] is False
        assert persisted["case_order_mib"] == [8, 16, 16, 8, 8, 16]
        assert persisted["required_artifacts"] == list(
            contract.REQUIRED_ARTIFACTS
        )
        assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_verify_only_cli_executes_no_ssh_or_worker():
    run_tag = "verify-only-cli-missing-run"
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "verify-only.json"
        completed = subprocess.run(
            (
                sys.executable,
                str(RUNNER_PATH),
                "verify-only",
                "--run-tag",
                run_tag,
                "--output-json",
                str(output),
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "QWEN35_FAIL_ON_SSH": "1",
                "QWEN35_FAIL_ON_WORKER": "1",
                "QWEN35_FAIL_ON_SAFETENSORS_PAYLOAD_OPEN": "1",
            },
        )
        printed = json.loads(completed.stdout)
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert printed == persisted
        assert persisted["classification"] == "INCOMPLETE"
        assert persisted["reasons"] == ["run directory does not exist"]
        assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_authorization_only_cli_executes_no_ssh_or_worker():
    run_tag = (
        "qwen35-real-load-download-only-current-source-"
        "preflight-r2-20260727-233352"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "authorization.json"
        completed = subprocess.run(
            (
                sys.executable,
                str(RUNNER_PATH),
                "authorization-only",
                "--run-tag",
                run_tag,
                "--output-json",
                str(output),
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "QWEN35_FAIL_ON_SSH": "1",
                "QWEN35_FAIL_ON_WORKER": "1",
                "QWEN35_FAIL_ON_SAFETENSORS_PAYLOAD_OPEN": "1",
            },
        )
        printed = json.loads(completed.stdout)
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert printed == persisted
        assert persisted["decision"] == "BLOCKED"
        assert persisted["worker_implementation_authorized"] is False
        assert persisted["worker_execution_authorized"] is False
        assert "preflight is not READY" in persisted["reasons"]


def main():
    test_frozen_identity_matrix_thresholds_and_artifacts()
    test_preflight_and_result_validation_are_fail_closed()
    test_restore_record_validation_is_fail_closed()
    test_runner_commands_are_bound_and_non_destructive()
    test_source_tar_staging_and_manifest_are_exact()
    test_remote_preflight_script_is_stat_only_for_payloads()
    test_restore_script_is_zero_payload_and_conflict_rejecting()
    test_preflight_classification_is_explicit_and_fail_closed()
    test_run_remote_preflight_persists_only_authorized_artifacts()
    test_remote_artifact_round_trip_rejects_mismatch()
    test_run_remote_restore_persists_only_authorized_artifacts()
    test_restore_artifact_round_trip_rejects_mismatch()
    test_execute_preflight_only_stages_and_audits()
    test_execute_restore_only_stages_and_restores()
    test_execute_verify_only_is_local_and_writes_reports()
    test_execute_authorization_only_is_local()
    test_download_only_is_manifest_first_atomic_and_verified()
    test_download_only_rejects_hash_mismatch_and_existing_destination()
    test_download_manifest_and_remote_reader_are_fail_closed()
    test_dry_run_json_persistence_executes_no_subprocess_or_ssh()
    test_verify_only_cli_executes_no_ssh_or_worker()
    test_authorization_only_cli_executes_no_ssh_or_worker()
    print(
        "qwen35 real checkpoint load safety gate tests passed (23 tests)"
    )


if __name__ == "__main__":
    main()
