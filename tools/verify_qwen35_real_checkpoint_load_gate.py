"""Independent verifier for Qwen3.5 real-checkpoint-load artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path, PurePosixPath


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_real_checkpoint_load_contract.py"
OUTPUT_FILES = {
    "independent_verification.json",
    "report.md",
}
TELEMETRY_FIELDS = (
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
CLAIM_BOUNDARY = (
    "Synthetic or real checkpoint-load verification only; this result does "
    "not establish inference speed, production cache or memory reduction, "
    "compression safety, quality retention, or native Qwen3.5 execution."
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "qwen35_real_checkpoint_load_contract_for_verifier",
        os.fspath(CONTRACT_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()
REQUIRED_INPUT_FILES = (
    set(contract.REQUIRED_ARTIFACTS)
    - OUTPUT_FILES
    - {"manifest.json"}
)


class VerificationError(ValueError):
    pass


def _fail(detail):
    raise VerificationError(detail)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON artifact {path.name}: {exc}")


def _read_jsonl(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"invalid JSONL artifact {path.name}: {exc}")
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _fail(
                f"invalid JSONL artifact {path.name}:{line_number}: {exc}"
            )
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"cannot hash artifact {path.name}: {exc}")
    return digest.hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        _fail("artifact path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        _fail(f"unsafe artifact path: {value}")
    return path


def _verify_inventory(run_dir, manifest):
    if manifest.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("manifest schema version mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        _fail("manifest artifacts must be a list")
    listed = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            _fail("manifest artifact entry must be an object")
        relative = _safe_relative_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in listed:
            _fail(f"duplicate manifest artifact: {relative_text}")
        path = run_dir.joinpath(*relative.parts)
        if not path.is_file():
            _fail(f"missing listed artifact: {relative_text}")
        actual_size = path.stat().st_size
        if entry.get("size") != actual_size:
            _fail(f"artifact size mismatch: {relative_text}")
        if entry.get("sha256") != _sha256(path):
            _fail(f"artifact sha256 mismatch: {relative_text}")
        listed[relative_text] = entry
    if set(listed) != REQUIRED_INPUT_FILES:
        missing = sorted(REQUIRED_INPUT_FILES - set(listed))
        extra = sorted(set(listed) - REQUIRED_INPUT_FILES)
        if missing:
            _fail(f"canonical artifact inventory is missing: {missing[0]}")
        _fail(f"unexpected manifest artifact: {extra[0]}")
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in OUTPUT_FILES
        and path.name != "manifest.json"
    }
    if actual_files != set(listed):
        unlisted = sorted(actual_files - set(listed))
        absent = sorted(set(listed) - actual_files)
        if unlisted:
            _fail(f"unlisted artifact: {unlisted[0]}")
        _fail(f"missing listed artifact: {absent[0]}")
    for directory in ("stdout", "stderr"):
        if not (run_dir / directory).is_dir():
            _fail(f"missing canonical directory: {directory}")


def _verify_source(source_manifest, manifest, preflight):
    if source_manifest.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("source manifest schema version mismatch")
    if source_manifest.get("clean") is not True:
        _fail("source snapshot is not clean")
    commit = source_manifest.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        _fail("source commit is not immutable")
    if manifest.get("source_commit") != commit:
        _fail("source commit mismatch")
    tree_digest = source_manifest.get("source_tree_sha256")
    if not _is_sha256(tree_digest):
        _fail("source tree SHA256 is invalid")
    if manifest.get("source_tree_sha256") != tree_digest:
        _fail("manifest source tree SHA256 mismatch")
    if preflight.get("source_tree_sha256") != tree_digest:
        _fail("preflight source tree SHA256 mismatch")
    local_hashes = source_manifest.get("local_file_sha256")
    remote_hashes = source_manifest.get("remote_file_sha256")
    if not isinstance(local_hashes, dict) or not local_hashes:
        _fail("local source hashes are missing")
    if local_hashes != remote_hashes:
        _fail("source hash mismatch")
    if preflight.get("source_file_sha256") != local_hashes:
        _fail("preflight local source hash mismatch")
    if preflight.get("remote_source_file_sha256") != remote_hashes:
        _fail("preflight remote source hash mismatch")
    for name, digest in local_hashes.items():
        _safe_relative_path(name)
        if not _is_sha256(digest):
            _fail(f"source SHA256 is invalid: {name}")


def _verify_model(
    model_manifest,
    model_manifest_path,
    manifest,
    preflight,
):
    if model_manifest.get("repository") != contract.MODEL_REPOSITORY:
        _fail("model repository mismatch")
    if model_manifest.get("resolved_revision") != contract.MODEL_REVISION:
        _fail("model revision mismatch")
    if model_manifest.get("trust_remote_code") is not False:
        _fail("model trust_remote_code mismatch")
    observed_files = model_manifest.get("files")
    if not isinstance(observed_files, dict):
        _fail("model file inventory is missing")
    for name, expected in contract.APPROVED_MODEL_FILES.items():
        if observed_files.get(name) != expected:
            _fail(f"model manifest file mismatch: {name}")
    if model_manifest != contract.APPROVED_MODEL_MANIFEST:
        _fail("model manifest content mismatch")
    if _sha256(model_manifest_path) != (
        contract.APPROVED_MODEL_MANIFEST_SHA256
    ):
        _fail("model manifest SHA256 mismatch")
    if (
        manifest.get("model_manifest_sha256")
        != contract.APPROVED_MODEL_MANIFEST_SHA256
    ):
        _fail("manifest model manifest SHA256 mismatch")
    if manifest.get("model_repository") != contract.MODEL_REPOSITORY:
        _fail("manifest model repository mismatch")
    if manifest.get("model_revision") != contract.MODEL_REVISION:
        _fail("manifest model revision mismatch")
    if preflight.get("model_repository") != contract.MODEL_REPOSITORY:
        _fail("preflight model repository mismatch")
    if preflight.get("model_revision") != contract.MODEL_REVISION:
        _fail("preflight model revision mismatch")
    if (
        preflight.get("model_manifest_sha256")
        != contract.APPROVED_MODEL_MANIFEST_SHA256
    ):
        _fail("preflight model manifest SHA256 mismatch")
    if preflight.get("config_sha256") != contract.APPROVED_CONFIG_SHA256:
        _fail("preflight config SHA256 mismatch")
    if preflight.get("index_sha256") != contract.APPROVED_INDEX_SHA256:
        _fail("preflight index SHA256 mismatch")


def _verify_preflight(preflight):
    try:
        contract.validate_preflight(preflight)
    except (TypeError, ValueError) as exc:
        _fail(f"invalid preflight: {exc}")
    if preflight.get("status") != "READY":
        _fail("preflight is not READY")


def _verify_environment(environment, source_tree_sha256):
    if environment.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("environment schema version mismatch")
    if environment.get("remote_target") != contract.REMOTE_TARGET:
        _fail("environment remote target mismatch")
    if environment.get("remote_python") != contract.REMOTE_PYTHON:
        _fail("environment remote Python mismatch")
    if environment.get("user") != "sitian":
        _fail("environment user mismatch")
    if not isinstance(environment.get("hostname"), str):
        _fail("environment hostname is invalid")
    if environment.get("model_repository") != contract.MODEL_REPOSITORY:
        _fail("environment model repository mismatch")
    if environment.get("model_revision") != contract.MODEL_REVISION:
        _fail("environment model revision mismatch")
    if environment.get("source_tree_sha256") != source_tree_sha256:
        _fail("environment source tree SHA256 mismatch")
    variables = environment.get("environment")
    if not isinstance(variables, dict):
        _fail("execution environment is missing")
    if variables.get("CUDA_VISIBLE_DEVICES") != "":
        _fail("CUDA_VISIBLE_DEVICES must be empty")
    if environment.get("cuda_initialized") is not False:
        _fail("environment CUDA initialized")
    if environment.get("cuda_allocated_bytes") != 0:
        _fail("environment CUDA allocation must be zero")


def _verify_processes(processes):
    if processes.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("process schema version mismatch")
    rows = processes.get("processes")
    if not isinstance(rows, list) or len(rows) != 1:
        _fail("exactly one worker process row is required")
    row = rows[0]
    if not isinstance(row, dict) or row.get("role") != "worker":
        _fail("worker process row is invalid")
    for name in ("attempted", "started", "exited"):
        if row.get(name) is not True:
            _fail(f"worker process was not {name}")
    if row.get("returncode") != 0:
        _fail("worker returncode must be zero")
    if row.get("signal") is not None:
        _fail("worker signal must be absent")
    if row.get("timed_out") is not False:
        _fail("worker timed out")
    if row.get("cuda_visible_devices") != "":
        _fail("worker CUDA_VISIBLE_DEVICES must be empty")
    if row.get("cuda_initialized") is not False:
        _fail("worker CUDA initialized")
    if row.get("cuda_allocated_bytes") != 0:
        _fail("worker CUDA allocation must be zero")
    if row.get("stdout_path") != "stdout/worker.log":
        _fail("worker stdout path mismatch")
    if row.get("stderr_path") != "stderr/worker.log":
        _fail("worker stderr path mismatch")


def _verify_ready_processes(processes):
    if processes.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("process schema version mismatch")
    if processes.get("processes") != []:
        _fail("READY evidence must not contain worker process rows")


def _verify_gpu_processes(gpu_processes):
    if gpu_processes.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("GPU process schema version mismatch")
    before = gpu_processes.get("before")
    after = gpu_processes.get("after")
    if not isinstance(before, list) or not isinstance(after, list):
        _fail("GPU process snapshots are invalid")
    if before:
        _fail("GPU occupancy before worker is non-empty")
    if after:
        _fail("GPU occupancy after worker is non-empty")


def _verify_telemetry(case_rows, telemetry_rows):
    if len(telemetry_rows) != len(case_rows):
        _fail("telemetry row count does not match case rows")
    by_case_id = {}
    for row in telemetry_rows:
        if not isinstance(row, dict):
            _fail("telemetry row must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            _fail("telemetry case ID is invalid")
        if case_id in by_case_id:
            _fail(f"duplicate telemetry case: {case_id}")
        by_case_id[case_id] = row
    seen_case_ids = set()
    for case in case_rows:
        if not isinstance(case, dict):
            _fail("case row must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            _fail("case ID is invalid")
        if case_id in seen_case_ids:
            _fail(f"duplicate case row: {case_id}")
        seen_case_ids.add(case_id)
        telemetry = by_case_id.get(case_id)
        if telemetry is None:
            _fail(f"missing telemetry case: {case_id}")
        for name in TELEMETRY_FIELDS:
            value = telemetry.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                _fail(f"telemetry metric is invalid for {case_id}: {name}")
        for name in (
            "order_index",
            "repeat_index",
            "budget_bytes",
            *TELEMETRY_FIELDS,
        ):
            if telemetry.get(name) != case.get(name):
                _fail(f"telemetry metric mismatch for {case_id}: {name}")
    extra = set(by_case_id) - seen_case_ids
    if extra:
        _fail(f"extra telemetry case: {sorted(extra)[0]}")


def _normalized_budget_map(value):
    if not isinstance(value, dict):
        _fail("summary median map is invalid")
    try:
        return {int(key): item for key, item in value.items()}
    except (TypeError, ValueError):
        _fail("summary median map budget key is invalid")


def _numbers_equal(left, right):
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _verify_summary(summary, case_rows, telemetry_rows, classified):
    if summary.get("schema_version") != contract.SCHEMA_VERSION:
        _fail("summary schema version mismatch")
    if summary.get("status") != "COMPLETE":
        _fail("summary status is not COMPLETE")
    exact = {
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
        "vmhwm_regression_bytes": classified["vmhwm_regression_bytes"],
    }
    labels = {
        "classification": "summary classification mismatch",
        "case_row_count": "summary case count mismatch",
        "telemetry_row_count": "summary telemetry count mismatch",
        "successful_worker_process_count": (
            "summary worker process count mismatch"
        ),
        "gpu_process_count_before": "summary GPU-before count mismatch",
        "gpu_process_count_after": "summary GPU-after count mismatch",
        "assigned_bindings": "summary assigned binding count mismatch",
        "source_tensors": "summary source tensor count mismatch",
        "destination_bytes": "summary destination byte count mismatch",
        "all_handles_closed": "summary handle state mismatch",
        "cuda_initialized": "summary CUDA initialized mismatch",
        "cuda_allocated_bytes": "summary CUDA allocation mismatch",
        "vmhwm_regression_bytes": "summary VmHWM regression mismatch",
    }
    for name, expected in exact.items():
        if summary.get(name) != expected:
            _fail(labels[name])
    if not _numbers_equal(
        summary.get("wall_time_improvement_fraction"),
        classified["wall_time_improvement_fraction"],
    ):
        _fail("summary wall-time improvement mismatch")
    for name in ("median_wall_seconds", "median_vmhwm_bytes"):
        observed = _normalized_budget_map(summary.get(name))
        expected = classified[name]
        if set(observed) != set(expected):
            _fail(f"summary {name} budget coverage mismatch")
        for budget in expected:
            if not _numbers_equal(observed[budget], expected[budget]):
                _fail(f"summary {name} mismatch for budget {budget}")


def _verify_ready_summary(summary):
    expected = {
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
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            _fail(f"READY summary mismatch: {name}")


def _write_atomic(path, content):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8")
    os.replace(partial, path)


def _write_outputs(run_dir, result):
    _write_atomic(
        run_dir / "independent_verification.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    report = "\n".join([
        "# Qwen3.5 Real Checkpoint Load Independent Verification",
        "",
        f"Classification: `{result['classification']}`",
        "",
        "## Reasons",
        "",
        *(
            [f"- {reason}" for reason in result["reasons"]]
            or ["- None."]
        ),
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ])
    _write_atomic(run_dir / "report.md", report)


def _observed_case_count(run_dir):
    try:
        return len(_read_jsonl(run_dir / "case_rows.jsonl"))
    except VerificationError:
        return 0


def _incomplete(reason, run_dir=None):
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": "INCOMPLETE",
        "expected_case_count": len(contract.CASE_ORDER_MIB),
        "observed_case_count": (
            _observed_case_count(run_dir) if run_dir is not None else 0
        ),
        "guards": {},
        "reasons": [reason],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _verify_complete_run(run_dir):
    manifest = _read_json(run_dir / "manifest.json")
    _verify_inventory(run_dir, manifest)
    source_manifest = _read_json(run_dir / "source_manifest.json")
    preflight = _read_json(run_dir / "preflight.json")
    environment = _read_json(run_dir / "environment.json")
    model_manifest_path = run_dir / "model_manifest.json"
    model_manifest = _read_json(model_manifest_path)
    processes = _read_json(run_dir / "processes.json")
    gpu_processes = _read_json(run_dir / "gpu_processes.json")
    case_rows = _read_jsonl(run_dir / "case_rows.jsonl")
    telemetry_rows = _read_jsonl(run_dir / "telemetry.jsonl")
    summary = _read_json(run_dir / "summary.json")
    _verify_preflight(preflight)
    _verify_source(source_manifest, manifest, preflight)
    _verify_model(
        model_manifest,
        model_manifest_path,
        manifest,
        preflight,
    )
    _verify_environment(
        environment,
        source_manifest["source_tree_sha256"],
    )
    _verify_gpu_processes(gpu_processes)
    if not case_rows and not telemetry_rows:
        _verify_ready_processes(processes)
        _verify_ready_summary(summary)
        return {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "READY",
            "expected_case_count": len(contract.CASE_ORDER_MIB),
            "observed_case_count": 0,
            "guards": {
                "artifact_inventory_pass": True,
                "source_provenance_pass": True,
                "model_provenance_pass": True,
                "preflight_pass": True,
                "environment_pass": True,
                "process_pass": True,
                "gpu_occupancy_pass": True,
                "ready_state_pass": True,
            },
            "reasons": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }
    if not case_rows or not telemetry_rows:
        _fail("partial case or telemetry coverage is incomplete")
    _verify_processes(processes)
    classified = contract.classify_case_rows(case_rows)
    if classified["classification"] == "INCOMPLETE":
        _fail(classified["reason"])
    _verify_telemetry(case_rows, telemetry_rows)
    _verify_summary(summary, case_rows, telemetry_rows, classified)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": classified["classification"],
        "expected_case_count": len(contract.CASE_ORDER_MIB),
        "observed_case_count": len(case_rows),
        "guards": {
            "artifact_inventory_pass": True,
            "source_provenance_pass": True,
            "model_provenance_pass": True,
            "preflight_pass": True,
            "environment_pass": True,
            "process_pass": True,
            "gpu_occupancy_pass": True,
            "telemetry_pass": True,
            "case_correctness_pass": True,
            "summary_consistency_pass": True,
        },
        "reasons": (
            []
            if classified["classification"] == "GO"
            else [classified["reason"]]
        ),
        "correctness_passed": classified["correctness_passed"],
        "wall_time_improvement_fraction": (
            classified["wall_time_improvement_fraction"]
        ),
        "vmhwm_regression_bytes": classified["vmhwm_regression_bytes"],
        "median_wall_seconds": classified["median_wall_seconds"],
        "median_vmhwm_bytes": classified["median_vmhwm_bytes"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def verify_run(run_dir, write_report=False):
    destination = Path(run_dir)
    if not destination.is_dir():
        result = _incomplete("run directory does not exist")
    else:
        try:
            result = _verify_complete_run(destination)
        except VerificationError as exc:
            result = _incomplete(str(exc), destination)
    if write_report and destination.is_dir():
        _write_outputs(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-report", action="store_true")
    arguments = parser.parse_args(argv)
    result = verify_run(
        Path(arguments.run_dir),
        write_report=arguments.write_report,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
