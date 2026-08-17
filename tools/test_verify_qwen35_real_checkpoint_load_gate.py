"""Dependency-light tests for the real-checkpoint-load verifier.

Run: python3 tools/test_verify_qwen35_real_checkpoint_load_gate.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_real_checkpoint_load_contract.py"
VERIFIER_PATH = THIS_DIR / "verify_qwen35_real_checkpoint_load_gate.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "qwen35_real_checkpoint_load_contract_for_verifier_tests",
    CONTRACT_PATH,
)
verifier = _load_module(
    "qwen35_real_checkpoint_load_verifier_under_test",
    VERIFIER_PATH,
)


SOURCE_COMMIT = "1" * 40
SOURCE_TREE_SHA256 = "2" * 64
SOURCE_FILES = {
    "tools/qwen35_real_checkpoint_load_contract.py": "3" * 64,
    "tools/qwen35_real_checkpoint_load_worker.py": "4" * 64,
}
ASSIGNMENT_DIGEST = "5" * 64


def _json_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_entry(run_dir, relative_path):
    path = run_dir / relative_path
    return {
        "path": relative_path,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _refresh_manifest_artifact(run_dir, relative_path):
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = next(
        entry
        for entry in manifest["artifacts"]
        if entry["path"] == relative_path
    )
    target.update(_artifact_entry(run_dir, relative_path))
    _write_json(manifest_path, manifest)


def _mutate_json(run_dir, relative_path, mutator):
    path = run_dir / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    _write_json(path, payload)
    _refresh_manifest_artifact(run_dir, relative_path)


def _mutate_jsonl(run_dir, relative_path, mutator):
    path = run_dir / relative_path
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    mutator(rows)
    _write_jsonl(path, rows)
    _refresh_manifest_artifact(run_dir, relative_path)


def _preflight():
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "READY",
        "checks": {
            "source_clean": True,
            "source_identity": True,
            "model_identity": True,
            "model_files": True,
            "proc_telemetry": True,
            "gpu0_idle": True,
        },
        "failure_reasons": [],
        "remote_target": contract.REMOTE_TARGET,
        "remote_python": contract.REMOTE_PYTHON,
        "observed_user": "sitian",
        "observed_hostname": "approved-host",
        "python_version": "3.11.9",
        "packages": {
            "torch": "2.7.0",
            "safetensors": "0.5.3",
            "transformers": "4.52.0",
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
        "payload_open_count": 0,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "payload_identity_source": "approved_model_manifest",
        "proc_telemetry_available": True,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "model_manifest_sha256": (
            contract.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_index_header_sha256": "6" * 64,
        "config_sha256": contract.APPROVED_CONFIG_SHA256,
        "index_sha256": contract.APPROVED_INDEX_SHA256,
        "source_file_sha256": SOURCE_FILES,
        "remote_source_file_sha256": SOURCE_FILES,
        "shards": [{
            "name": contract.APPROVED_SHARD_NAME,
            "expected_size": contract.APPROVED_SHARD_SIZE,
            "expected_sha256": contract.APPROVED_SHARD_SHA256,
            "observed_size": contract.APPROVED_SHARD_SIZE,
            "inode": 17,
            "device": 18,
            "mode": 33188,
            "resolved_path": (
                f"{contract.APPROVED_MODEL_DIR}/"
                f"{contract.APPROVED_SHARD_NAME}"
            ),
        }],
        "proc_meminfo": {"MemAvailable": 1 << 40},
        "proc_status_fields": {"VmHWM": True, "VmRSS": True},
        "run_root_filesystem": {"type": "ext4"},
        "model_root_filesystem": {"type": "ext4"},
        "free_run_root_bytes": 1 << 40,
        "required_run_root_bytes": 1 << 30,
    }


def _case_rows(classification):
    if classification == "GO":
        wall_by_budget = {
            8: (10.0, 10.2, 9.8),
            16: (9.0, 8.9, 9.1),
        }
    elif classification == "NO_GO":
        wall_by_budget = {
            8: (10.0, 10.2, 9.8),
            16: (9.7, 9.8, 9.6),
        }
    else:
        raise ValueError("classification must be GO or NO_GO")
    counts = {8: 0, 16: 0}
    rows = []
    for order_index, budget_mib in enumerate(contract.CASE_ORDER_MIB):
        repeat_index = counts[budget_mib]
        counts[budget_mib] += 1
        rows.append({
            "case_id": (
                f"budget-{budget_mib}-repeat-{repeat_index}"
            ),
            "order_index": order_index,
            "repeat_index": repeat_index,
            "warmup": False,
            "budget_bytes": budget_mib << 20,
            "tensor_parallel_size": 2,
            "tensor_parallel_rank": 0,
            "assigned_bindings": 320,
            "source_tensors": 320,
            "tile_count": 640 if budget_mib == 8 else 320,
            "destination_bytes": contract.DESTINATION_BYTES,
            "peak_tile_bytes": budget_mib << 20,
            "assignment_digest": ASSIGNMENT_DIGEST,
            "expected_assignment_digest": ASSIGNMENT_DIGEST,
            "all_handles_closed": True,
            "cuda_initialized": False,
            "cuda_allocated_bytes": 0,
            "wall_seconds": wall_by_budget[budget_mib][repeat_index],
            "user_cpu_seconds": 8.0,
            "system_cpu_seconds": 1.0,
            "minor_faults": 100 + order_index,
            "major_faults": 0,
            "vmrss_bytes": 3 << 30,
            "vmhwm_bytes": (
                (3 << 30)
                + (4 << 20 if budget_mib == 16 else 0)
            ),
            "voluntary_context_switches": 20,
            "involuntary_context_switches": 2,
            "returncode": 0,
        })
    return rows


def _telemetry_rows(case_rows):
    metric_names = (
        "wall_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "minor_faults",
        "major_faults",
        "vmrss_bytes",
        "vmhwm_bytes",
        "voluntary_context_switches",
        "involuntary_context_switches",
    )
    return [{
        "case_id": row["case_id"],
        "order_index": row["order_index"],
        "repeat_index": row["repeat_index"],
        "budget_bytes": row["budget_bytes"],
        **{name: row[name] for name in metric_names},
    } for row in case_rows]


def write_complete_run(run_dir, classification="GO"):
    run_dir.mkdir(parents=True)
    (run_dir / "stdout").mkdir()
    (run_dir / "stderr").mkdir()
    (run_dir / "stdout" / "worker.log").write_text(
        "worker completed\n",
        encoding="utf-8",
    )
    (run_dir / "stderr" / "worker.log").write_text(
        "",
        encoding="utf-8",
    )
    source_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "clean": True,
        "commit": SOURCE_COMMIT,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "local_file_sha256": SOURCE_FILES,
        "remote_file_sha256": SOURCE_FILES,
    }
    environment = {
        "schema_version": contract.SCHEMA_VERSION,
        "remote_target": contract.REMOTE_TARGET,
        "remote_python": contract.REMOTE_PYTHON,
        "user": "sitian",
        "hostname": "approved-host",
        "model_repository": contract.MODEL_REPOSITORY,
        "model_revision": contract.MODEL_REVISION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
        "cuda_initialized": False,
        "cuda_allocated_bytes": 0,
    }
    processes = {
        "schema_version": contract.SCHEMA_VERSION,
        "processes": [{
            "role": "worker",
            "attempted": True,
            "started": True,
            "exited": True,
            "returncode": 0,
            "signal": None,
            "timed_out": False,
            "cuda_visible_devices": "",
            "cuda_initialized": False,
            "cuda_allocated_bytes": 0,
            "stdout_path": "stdout/worker.log",
            "stderr_path": "stderr/worker.log",
        }],
    }
    gpu_processes = {
        "schema_version": contract.SCHEMA_VERSION,
        "before": [],
        "after": [],
    }
    case_rows = _case_rows(classification)
    telemetry_rows = _telemetry_rows(case_rows)
    classified = contract.classify_case_rows(case_rows)
    summary = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "COMPLETE",
        "classification": classified["classification"],
        "case_row_count": len(case_rows),
        "telemetry_row_count": len(telemetry_rows),
        "successful_worker_process_count": 1,
        "gpu_process_count_before": 0,
        "gpu_process_count_after": 0,
        "assigned_bindings": 320,
        "source_tensors": 320,
        "destination_bytes": contract.DESTINATION_BYTES,
        "all_handles_closed": True,
        "cuda_initialized": False,
        "cuda_allocated_bytes": 0,
        "wall_time_improvement_fraction": (
            classified["wall_time_improvement_fraction"]
        ),
        "vmhwm_regression_bytes": (
            classified["vmhwm_regression_bytes"]
        ),
        "median_wall_seconds": classified["median_wall_seconds"],
        "median_vmhwm_bytes": classified["median_vmhwm_bytes"],
    }
    artifacts = {
        "source_manifest.json": source_manifest,
        "preflight.json": _preflight(),
        "environment.json": environment,
        "model_manifest.json": contract.APPROVED_MODEL_MANIFEST,
        "processes.json": processes,
        "gpu_processes.json": gpu_processes,
        "summary.json": summary,
    }
    for relative_path, payload in artifacts.items():
        _write_json(run_dir / relative_path, payload)
    _write_jsonl(run_dir / "case_rows.jsonl", case_rows)
    _write_jsonl(run_dir / "telemetry.jsonl", telemetry_rows)
    input_paths = [
        relative_path
        for relative_path in contract.REQUIRED_ARTIFACTS
        if relative_path not in {
            "manifest.json",
            "independent_verification.json",
            "report.md",
        }
    ]
    manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "model_repository": contract.MODEL_REPOSITORY,
        "model_revision": contract.MODEL_REVISION,
        "model_manifest_sha256": (
            contract.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "artifacts": [
            _artifact_entry(run_dir, relative_path)
            for relative_path in input_paths
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def convert_to_ready_run(run_dir):
    _write_jsonl(run_dir / "case_rows.jsonl", [])
    _write_jsonl(run_dir / "telemetry.jsonl", [])
    _write_json(
        run_dir / "processes.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "processes": [],
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "status": "READY",
            "classification": "READY",
            "case_row_count": 0,
            "telemetry_row_count": 0,
            "successful_worker_process_count": 0,
            "gpu_process_count_before": 0,
            "gpu_process_count_after": 0,
            "cuda_initialized": False,
            "cuda_allocated_bytes": 0,
        },
    )
    (run_dir / "stdout" / "worker.log").write_text("", encoding="utf-8")
    for relative_path in (
        "case_rows.jsonl",
        "telemetry.jsonl",
        "processes.json",
        "summary.json",
        "stdout/worker.log",
    ):
        _refresh_manifest_artifact(run_dir, relative_path)
    return run_dir


def test_ready_fixture_is_not_promoted_to_go():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = convert_to_ready_run(
            write_complete_run(Path(temporary) / "ready")
        )
        result = verifier.verify_run(run_dir)
        assert result["classification"] == "READY"
        assert result["observed_case_count"] == 0


def test_complete_go_and_no_go_fixtures():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        go_dir = write_complete_run(root / "go", "GO")
        no_go_dir = write_complete_run(root / "no-go", "NO_GO")
        go = verifier.verify_run(go_dir, write_report=True)
        no_go = verifier.verify_run(no_go_dir, write_report=True)
        assert go["classification"] == "GO"
        assert no_go["classification"] == "NO_GO"
        assert go["observed_case_count"] == len(contract.CASE_ORDER_MIB)
        assert no_go["observed_case_count"] == len(
            contract.CASE_ORDER_MIB
        )
        assert (go_dir / "independent_verification.json").is_file()
        assert (go_dir / "report.md").is_file()


def test_repeated_output_generation_preserves_input_inventory():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = write_complete_run(Path(temporary) / "go", "GO")
        first = verifier.verify_run(run_dir, write_report=True)
        second = verifier.verify_run(run_dir, write_report=True)
        assert second == first


def _expect_incomplete(base_run, name, mutator, message):
    run_dir = base_run.parent / name
    shutil.copytree(base_run, run_dir)
    mutator(run_dir)
    result = verifier.verify_run(run_dir)
    assert result["classification"] == "INCOMPLETE", (name, result)
    assert any(message in reason for reason in result["reasons"]), (
        name,
        result,
    )


def _add_unlisted_artifact(run_dir):
    (run_dir / "unexpected.json").write_text("{}\n", encoding="utf-8")


def _corrupt_listed_artifact(run_dir):
    with (run_dir / "summary.json").open("ab") as destination:
        destination.write(b" ")


def _unsafe_manifest_path(run_dir):
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["artifacts"][0]["path"] = "../source_manifest.json"
    _write_json(run_dir / "manifest.json", manifest)


def _dirty_source(run_dir):
    _mutate_json(
        run_dir,
        "source_manifest.json",
        lambda payload: payload.__setitem__("clean", False),
    )


def _mismatch_source_hash(run_dir):
    def mutate(payload):
        name = sorted(payload["remote_file_sha256"])[0]
        payload["remote_file_sha256"][name] = "f" * 64

    _mutate_json(run_dir, "source_manifest.json", mutate)


def _mismatch_model_revision(run_dir):
    _mutate_json(
        run_dir,
        "model_manifest.json",
        lambda payload: payload.__setitem__(
            "resolved_revision",
            "7" * 40,
        ),
    )


def _mismatch_model_config(run_dir):
    def mutate(payload):
        payload["files"]["config.json"]["sha256"] = "8" * 64

    _mutate_json(run_dir, "model_manifest.json", mutate)


def _incomplete_preflight(run_dir):
    def mutate(payload):
        payload["status"] = "INCOMPLETE"
        payload["checks"]["gpu0_idle"] = False
        payload["failure_reasons"] = ["GPU0 has active compute processes"]
        payload["gpu_processes"] = [{"pid": 123}]

    _mutate_json(run_dir, "preflight.json", mutate)


def _environment_cuda_visible(run_dir):
    def mutate(payload):
        payload["environment"]["CUDA_VISIBLE_DEVICES"] = "0"

    _mutate_json(run_dir, "environment.json", mutate)


def _worker_failed(run_dir):
    def mutate(payload):
        payload["processes"][0]["returncode"] = 1

    _mutate_json(run_dir, "processes.json", mutate)


def _worker_timed_out(run_dir):
    def mutate(payload):
        payload["processes"][0]["timed_out"] = True

    _mutate_json(run_dir, "processes.json", mutate)


def _worker_cuda_initialized(run_dir):
    def mutate(payload):
        payload["processes"][0]["cuda_initialized"] = True

    _mutate_json(run_dir, "processes.json", mutate)


def _worker_cuda_allocated(run_dir):
    def mutate(payload):
        payload["processes"][0]["cuda_allocated_bytes"] = 1

    _mutate_json(run_dir, "processes.json", mutate)


def _gpu_before_occupied(run_dir):
    def mutate(payload):
        payload["before"] = [{"pid": 321, "gpu_uuid": "GPU-test"}]

    _mutate_json(run_dir, "gpu_processes.json", mutate)


def _gpu_after_occupied(run_dir):
    def mutate(payload):
        payload["after"] = [{"pid": 654, "gpu_uuid": "GPU-test"}]

    _mutate_json(run_dir, "gpu_processes.json", mutate)


def _remove_worker_log(run_dir):
    (run_dir / "stdout" / "worker.log").unlink()


def _remove_telemetry_row(run_dir):
    _mutate_jsonl(
        run_dir,
        "telemetry.jsonl",
        lambda rows: rows.pop(),
    )


def _duplicate_telemetry_case(run_dir):
    def mutate(rows):
        rows[-1]["case_id"] = rows[0]["case_id"]

    _mutate_jsonl(run_dir, "telemetry.jsonl", mutate)


def _mismatch_telemetry_metric(run_dir):
    def mutate(rows):
        rows[0]["minor_faults"] += 1

    _mutate_jsonl(run_dir, "telemetry.jsonl", mutate)


def _non_finite_telemetry(run_dir):
    def mutate(rows):
        rows[0]["vmrss_bytes"] = float("nan")

    _mutate_jsonl(run_dir, "telemetry.jsonl", mutate)


def _remove_case_row(run_dir):
    _mutate_jsonl(
        run_dir,
        "case_rows.jsonl",
        lambda rows: rows.pop(),
    )


def _add_extra_case_row(run_dir):
    def mutate(rows):
        row = dict(rows[-1])
        row["case_id"] = "extra-case"
        row["order_index"] = len(rows)
        rows.append(row)

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _reorder_case_budget(run_dir):
    def mutate(rows):
        rows[0]["budget_bytes"] = 16 << 20

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _overflow_tile_peak(run_dir):
    def mutate(rows):
        rows[0]["peak_tile_bytes"] = (8 << 20) + 1

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _mismatch_binding_count(run_dir):
    def mutate(rows):
        rows[0]["assigned_bindings"] = 319

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _mismatch_source_tensor_count(run_dir):
    def mutate(rows):
        rows[0]["source_tensors"] = 319

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _mismatch_destination_bytes(run_dir):
    def mutate(rows):
        rows[0]["destination_bytes"] -= 1

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _mismatch_assignment_digest(run_dir):
    def mutate(rows):
        rows[0]["assignment_digest"] = "9" * 64

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _leave_handle_open(run_dir):
    def mutate(rows):
        rows[0]["all_handles_closed"] = False

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _summary_count_mismatch(run_dir):
    _mutate_json(
        run_dir,
        "summary.json",
        lambda payload: payload.__setitem__("case_row_count", 5),
    )


def _summary_classification_mismatch(run_dir):
    _mutate_json(
        run_dir,
        "summary.json",
        lambda payload: payload.__setitem__("classification", "NO_GO"),
    )


def test_inventory_and_provenance_tampering_is_incomplete():
    cases = (
        ("unlisted", _add_unlisted_artifact, "unlisted artifact"),
        ("hash", _corrupt_listed_artifact, "artifact size mismatch"),
        ("unsafe", _unsafe_manifest_path, "unsafe artifact path"),
        ("dirty", _dirty_source, "source snapshot is not clean"),
        ("source-hash", _mismatch_source_hash, "source hash mismatch"),
        ("model-revision", _mismatch_model_revision, "model revision"),
        ("model-config", _mismatch_model_config, "model manifest"),
        ("preflight", _incomplete_preflight, "preflight is not READY"),
        (
            "environment-cuda",
            _environment_cuda_visible,
            "CUDA_VISIBLE_DEVICES",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base_run = write_complete_run(Path(temporary) / "complete")
        for name, mutator, message in cases:
            _expect_incomplete(base_run, name, mutator, message)


def test_process_gpu_telemetry_case_and_summary_tampering_is_incomplete():
    cases = (
        ("worker-failed", _worker_failed, "worker returncode"),
        ("worker-timeout", _worker_timed_out, "worker timed out"),
        (
            "worker-cuda",
            _worker_cuda_initialized,
            "worker CUDA initialized",
        ),
        (
            "worker-cuda-allocation",
            _worker_cuda_allocated,
            "worker CUDA allocation",
        ),
        ("gpu-before", _gpu_before_occupied, "GPU occupancy before"),
        ("gpu-after", _gpu_after_occupied, "GPU occupancy after"),
        ("worker-log", _remove_worker_log, "missing listed artifact"),
        (
            "telemetry-missing",
            _remove_telemetry_row,
            "telemetry row count",
        ),
        (
            "telemetry-duplicate",
            _duplicate_telemetry_case,
            "duplicate telemetry case",
        ),
        (
            "telemetry-metric",
            _mismatch_telemetry_metric,
            "telemetry metric mismatch",
        ),
        (
            "telemetry-non-finite",
            _non_finite_telemetry,
            "telemetry metric is invalid",
        ),
        ("case-missing", _remove_case_row, "case row count"),
        ("case-extra", _add_extra_case_row, "case row count"),
        ("case-order", _reorder_case_budget, "case budget order"),
        ("tile-peak", _overflow_tile_peak, "peak tile bytes"),
        ("bindings", _mismatch_binding_count, "assigned binding"),
        (
            "source-tensors",
            _mismatch_source_tensor_count,
            "source tensor count",
        ),
        (
            "destination",
            _mismatch_destination_bytes,
            "destination byte",
        ),
        (
            "assignment-digest",
            _mismatch_assignment_digest,
            "assignment digest",
        ),
        ("open-handle", _leave_handle_open, "handles"),
        ("summary-count", _summary_count_mismatch, "summary case count"),
        (
            "summary-classification",
            _summary_classification_mismatch,
            "summary classification",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base_run = write_complete_run(Path(temporary) / "complete")
        for name, mutator, message in cases:
            _expect_incomplete(base_run, name, mutator, message)


def test_cli_prints_json_and_persists_report_for_all_classifications():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dirs = {
            "GO": write_complete_run(root / "go", "GO"),
            "NO_GO": write_complete_run(root / "no-go", "NO_GO"),
            "READY": convert_to_ready_run(
                write_complete_run(root / "ready", "GO")
            ),
            "INCOMPLETE": write_complete_run(root / "incomplete", "GO"),
        }
        _dirty_source(run_dirs["INCOMPLETE"])
        for classification, run_dir in run_dirs.items():
            completed = subprocess.run(
                [
                    sys.executable,
                    os.fspath(VERIFIER_PATH),
                    "--run-dir",
                    os.fspath(run_dir),
                    "--write-report",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
            result = json.loads(completed.stdout)
            assert result["classification"] == classification
            report = (run_dir / "report.md").read_text(
                encoding="utf-8"
            )
            assert f"Classification: `{classification}`" in report
            assert "does not establish inference speed" in report


if __name__ == "__main__":
    test_ready_fixture_is_not_promoted_to_go()
    test_complete_go_and_no_go_fixtures()
    test_repeated_output_generation_preserves_input_inventory()
    test_inventory_and_provenance_tampering_is_incomplete()
    test_process_gpu_telemetry_case_and_summary_tampering_is_incomplete()
    test_cli_prints_json_and_persists_report_for_all_classifications()
    print("qwen35 real checkpoint load verifier tests passed")
