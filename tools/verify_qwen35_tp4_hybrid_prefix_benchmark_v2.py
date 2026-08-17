from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import tempfile

import qwen35_tp4_hybrid_prefix_benchmark_v2_contract as contract
import torch


LOGIT_ATOL = 2e-5
LOGIT_RTOL = 0.0
FORBIDDEN_COUNTERS = (
    "hybrid_cache_evictions",
    "hybrid_cache_validation_failures",
    "hybrid_cache_failed_restores",
    "hybrid_cache_quarantines",
    "hybrid_cache_failed_rollbacks",
    "hybrid_cache_corruption_events",
    "hybrid_cache_partial_restore_attempts",
    "hybrid_cache_fallbacks",
    "hybrid_cache_mixed_representation_events",
    "hybrid_cache_missing_layer_events",
    "oom_events",
    "undeclared_eviction_events",
)


class VerificationError(RuntimeError):
    classification = "INVALID_ARTIFACT"

    def __init__(self, message):
        super().__init__(message)
        self.classification = "INVALID_ARTIFACT"


def _fail(message):
    raise VerificationError(message)


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(f"failed to hash artifact: {error}")
    return digest.hexdigest()


def _load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"failed to load JSON {Path(path).name}: {error}")
    _require_finite(value, Path(path).name)
    return value


def _load_jsonl(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        _fail(f"failed to load JSONL {Path(path).name}: {error}")
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            _fail(f"invalid JSONL row {Path(path).name}:{line_number}: {error}")
        _require_finite(row, f"{Path(path).name}:{line_number}")
        rows.append(row)
    return rows


def _require_finite(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite number in {label}")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child, label)
    elif isinstance(value, list):
        for child in value:
            _require_finite(child, label)


def _safe_file(run_dir, relative, label):
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} path is invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        _fail(f"{label} path is unsafe")
    path = run_dir / relative_path
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} file is missing")
    return path


def _verify_closed_inventory(run_dir):
    expected_top_level = {
        "artifact_manifest.json",
        *contract.ARTIFACT_MANIFEST_HASH_DOMAIN,
    }
    allowed_outputs = set(contract.VERIFIER_TRUST_DOMAIN)
    expected_directories = set(contract.NESTED_ARTIFACT_DIRECTORIES)
    actual_top_level = set()
    actual_directories = set()
    for entry in run_dir.iterdir():
        if entry.is_symlink():
            _fail(f"artifact symlink is forbidden: {entry.name}")
        if entry.is_dir():
            actual_directories.add(entry.name)
        elif entry.is_file():
            actual_top_level.add(entry.name)
        else:
            _fail(f"unsupported artifact type: {entry.name}")
    if actual_directories != expected_directories:
        _fail("nested artifact directory inventory mismatch")
    if not expected_top_level.issubset(actual_top_level):
        _fail("required artifact inventory is incomplete")
    if actual_top_level - expected_top_level - allowed_outputs:
        _fail("unexpected top-level artifact")

def _verify_artifact_manifest(run_dir):
    manifest = _load_json(run_dir / "artifact_manifest.json")
    try:
        contract.validate_artifact_manifest(manifest)
    except ValueError as error:
        _fail(f"artifact manifest is invalid: {error}")
    entries = manifest["entries"]
    for entry in entries:
        path = _safe_file(run_dir, entry["path"], "manifest artifact")
        if path.stat().st_size != entry["bytes"]:
            _fail(f"artifact byte count mismatch: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            _fail(f"artifact hash mismatch: {entry['path']}")
    return manifest


def _nested_regular_files(run_dir):
    files = []
    for root_name in contract.NESTED_ARTIFACT_DIRECTORIES:
        root = run_dir / root_name
        for directory, directory_names, file_names in os.walk(
            root,
            followlinks=False,
        ):
            directory_path = Path(directory)
            for name in directory_names:
                path = directory_path / name
                if path.is_symlink():
                    _fail(
                        "nested artifact symlink is forbidden: "
                        f"{path.relative_to(run_dir).as_posix()}"
                    )
                if not path.is_dir():
                    _fail(
                        "unsupported nested artifact type: "
                        f"{path.relative_to(run_dir).as_posix()}"
                    )
            for name in file_names:
                path = directory_path / name
                relative = path.relative_to(run_dir).as_posix()
                if path.is_symlink():
                    _fail(f"nested artifact symlink is forbidden: {relative}")
                if not path.is_file():
                    _fail(f"unsupported nested artifact type: {relative}")
                files.append(relative)
    return sorted(files)


def _verify_manifests(run_dir):
    manifests = {
        kind: _load_json(
            run_dir / contract.NESTED_MANIFEST_ARTIFACT_PATHS[kind]
        )
        for kind in contract.NESTED_MANIFEST_KINDS
    }
    expected_roots = {
        "prerequisites": {"prerequisites"},
        "tokens": {"tokens"},
        "logits": {"logits"},
        "logs": {"logs"},
        "snapshots": {"snapshots"},
        "tensor_inventories": {"snapshots"},
    }
    file_inventory = []
    for kind in contract.NESTED_MANIFEST_KINDS:
        manifest = manifests[kind]
        if (
            not isinstance(manifest, dict)
            or set(manifest) != set(contract.NESTED_MANIFEST_FIELDS)
            or not isinstance(manifest.get("files"), list)
        ):
            _fail(f"{kind} canonical manifest schema mismatch")
        for row in manifest["files"]:
            if (
                not isinstance(row, dict)
                or set(row) != set(contract.NESTED_FILE_FIELDS)
            ):
                _fail(f"{kind} canonical file row mismatch")
            relative = row["path"]
            if (
                not isinstance(relative, str)
                or not Path(relative).parts
                or Path(relative).parts[0] not in expected_roots[kind]
            ):
                _fail(f"{kind} canonical path domain mismatch")
            if row["type"] != "regular_file":
                _fail(f"{kind} canonical file type mismatch")
            path = _safe_file(run_dir, relative, f"{kind} evidence")
            if (
                path.stat().st_size != row["bytes"]
                or _sha256(path) != row["sha256"]
            ):
                _fail(f"{kind} canonical file binding mismatch")
            file_inventory.append(row)
    file_inventory = sorted(file_inventory, key=lambda row: row["path"])
    if [row["path"] for row in file_inventory] != _nested_regular_files(
        run_dir
    ):
        _fail("canonical nested file inventory mismatch")
    for row in manifests["logs"]["rows"]:
        if not isinstance(row, dict) or "file" not in row:
            _fail("canonical worker log row mismatch")
        path = _safe_file(run_dir, row["file"]["path"], "worker log")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            _fail(f"failed to read worker log: {error}")
        if (
            "QWEN35_TP4_BENCHMARK_V2_WORKER_COMPLETE" not in text
            or "Traceback (most recent call last)" in text
        ):
            _fail("worker log authority marker mismatch")
    return manifests, file_inventory


def _verify_authorities(run_dir, case_rows):
    source = _load_json(run_dir / "source_manifest.json")
    try:
        contract.validate_source_manifest(source)
    except ValueError as error:
        _fail(f"source manifest is invalid: {error}")
    workload_path = run_dir / "workload_manifest.json"
    workload = _load_json(workload_path)
    if workload != contract.workload_manifest_payload():
        _fail("workload manifest authority mismatch")
    prerequisite_path = run_dir / "correctness_prerequisites.json"
    calibration_path = run_dir / "calibration_binding.json"
    calibration = _load_json(calibration_path)
    p1_path = run_dir / "p1_authority_binding.json"
    p1 = _load_json(p1_path)
    try:
        contract.validate_calibration_binding(calibration)
        contract.validate_p1_authority_binding(p1)
    except ValueError as error:
        _fail(f"runtime authority binding is invalid: {error}")
    authorization_path = run_dir / "consumed_authorization.json"
    authorization = _load_json(authorization_path)
    if (
        authorization.get("schema_version")
        != contract.EVIDENCE_SCHEMA_VERSIONS["consumed_authorization"]
        or authorization.get("classification") != "AUTHORIZED"
        or authorization.get("consumed") is not True
    ):
        _fail("consumed authorization is invalid")
    expected_hashes = {
        "source_tree_sha256": source["source_tree_sha256"],
        "workload_manifest_sha256": _sha256(workload_path),
        "correctness_prerequisites_sha256": _sha256(prerequisite_path),
        "calibration_artifact_sha256": _sha256(calibration_path),
        "p1_authority_artifact_sha256": _sha256(p1_path),
        "preflight_receipt_sha256": _sha256(run_dir / "preflight.json"),
        "authorization_receipt_sha256": _sha256(authorization_path),
    }
    source_bindings = {
        field: source[field]
        for field in contract.SOURCE_MANIFEST_FIELDS
        if field.endswith("_sha256")
    }
    expected_hashes.update(source_bindings)
    for row in case_rows:
        for field, expected in expected_hashes.items():
            if row[field] != expected:
                _fail(f"case row authority binding mismatch: {field}")


def _verify_static_documents(run_dir):
    source = _load_json(run_dir / "source_manifest.json")
    expected = {
        "environment.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "world_size": contract.WORLD_SIZE,
        },
        "gpu_assignments.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        },
        "commands.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "case_ids": [case.case_id for case in contract.build_case_matrix()],
        },
        "preflight.json": {"classification": "READY", "resources_blocked": False},
        "execution_plan.json": {
            "classification": "READY",
            "sha256": source["execution_plan_sha256"],
        },
        "source_bundle_manifest.json": {
            "sha256": source["source_bundle_sha256"]
        },
        "source_package_manifest.json": {
            "sha256": source["source_package_sha256"]
        },
        "resource_guards.json": {"before": "PASS", "after": "PASS"},
    }
    for name, expected_payload in expected.items():
        if _load_json(run_dir / name) != expected_payload:
            _fail(f"{name} authority mismatch")
    gate = _load_json(run_dir / "gate1_audit.json")
    if gate != {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS["gate1_audit"],
        "classification": "PASS",
        "source_tree_sha256": source["source_tree_sha256"],
    }:
        _fail("gate1_audit.json authority mismatch")


def _verify_summary(run_dir, case_rows, process_rows):
    summary = _load_json(run_dir / "summary.json")
    expected = {
        "schema_version": contract.SCHEMA_VERSION,
        "producer_claim": "UNVERIFIED",
        "thresholds": contract.THRESHOLDS,
        "case_rows": len(case_rows),
        "process_rows": len(process_rows),
    }
    if summary != expected:
        _fail("producer summary integrity mismatch")


def _load_token_file(run_dir, row, role):
    path_field = f"{role}_token_ids_path"
    hash_field = f"{role}_token_ids_sha256"
    path = _safe_file(run_dir, row[path_field], f"{role} token IDs")
    if _sha256(path) != row[hash_field]:
        _fail(f"{role} token ID row hash mismatch")
    values = _load_json(path)
    if (
        not isinstance(values, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
    ):
        _fail(f"{role} token IDs are invalid")
    return values


def _load_logits(run_dir, row):
    path = _safe_file(run_dir, row["final_logits_path"], "final logits")
    if _sha256(path) != row["final_logits_sha256"]:
        _fail("final logits row hash mismatch")
    expected_bytes = contract.MODEL_VOCAB_SIZE * 4
    data = path.read_bytes()
    if len(data) != expected_bytes:
        _fail("final logits byte length mismatch")
    values = torch.frombuffer(bytearray(data), dtype=torch.float32)
    if not torch.isfinite(values).all():
        _fail("final logits contain non-finite values")
    return values


def _verify_tokens_and_logits(run_dir, case_rows):
    grouped = defaultdict(dict)
    token_ids_equal = True
    logits_assert_close = True
    for row in case_rows:
        key = (row["workload"], row["phase"], row["repetition"], row["request_id"])
        grouped[key][row["profile"]] = row
        prompt = _load_token_file(run_dir, row, "prompt")
        output = _load_token_file(run_dir, row, "output")
        if len(prompt) != row["prompt_tokens"] or len(output) != row["generated_tokens"]:
            _fail("token file length mismatch")
    for key, profiles in grouped.items():
        if set(profiles) != set(contract.PROFILES):
            _fail(f"profile comparison set is incomplete: {key}")
        reference_output = _load_token_file(run_dir, profiles["recompute"], "output")
        for profile in contract.PROFILES[1:]:
            if _load_token_file(run_dir, profiles[profile], "output") != reference_output:
                token_ids_equal = False
        exact = _load_logits(run_dir, profiles["exact_restore"])
        candidate_logits = _load_logits(run_dir, profiles[contract.P2_PROFILE])
        recompute_logits = _load_logits(run_dir, profiles["recompute"])
        for candidate in (candidate_logits, recompute_logits):
            try:
                torch.testing.assert_close(
                    candidate,
                    exact,
                    atol=2e-5,
                    rtol=0.0,
                )
            except AssertionError:
                logits_assert_close = False
    return {
        "token_ids_equal": token_ids_equal,
        "logits_assert_close": logits_assert_close,
    }


def _verify_tensor_evidence(run_dir, tensor_manifest):
    for row in tensor_manifest["rows"]:
        if not isinstance(row, dict) or "file" not in row or "evidence" not in row:
            _fail("canonical tensor inventory row mismatch")
        evidence = _load_json(
            _safe_file(
                run_dir,
                row["file"]["path"],
                "tensor inventory evidence",
            )
        )
        if evidence != row["evidence"]:
            _fail("tensor inventory file content mismatch")
        try:
            contract.validate_tensor_storage_evidence(evidence)
            contract.recompute_tensor_storage_accounting(evidence)
        except ValueError as error:
            _fail(f"tensor inventory evidence is invalid: {error}")


def _verify_process_rows(process_rows):
    for row in process_rows:
        if not isinstance(row, dict) or set(row) != set(contract.PROCESS_ROW_FIELDS):
            _fail("canonical process row schema mismatch")
    try:
        contract.validate_process_rows(process_rows)
    except ValueError as error:
        _fail(f"canonical process row evidence is invalid: {error}")
    rows_by_case = defaultdict(list)
    for row in process_rows:
        rows_by_case[row["case_id"]].append(row)
    for case_rows in rows_by_case.values():
        if len(case_rows) != contract.WORLD_SIZE:
            _fail("process rank coverage mismatch")
        rank_invariant_fields = contract.PROCESS_ROW_FIELDS[
            contract.PROCESS_ROW_FIELDS.index("initialization_ns"):
        ]
        reference = case_rows[0]
        for row in case_rows[1:]:
            for field in rank_invariant_fields:
                if row[field] != reference[field]:
                    _fail(f"process rank accounting mismatch: {field}")


def _median(values, label):
    if not values:
        _fail(f"missing values for {label}")
    return float(statistics.median(values))


def _ratio(numerator, denominator, label):
    if denominator <= 0:
        _fail(f"invalid denominator for {label}")
    return numerator / denominator


def _aggregate_metrics(case_rows, process_rows):
    measured_cases = [row for row in case_rows if row["phase"] == "measured"]
    measured_processes = [
        row for row in process_rows if row["phase"] == "measured"
    ]

    def case_values(workload, profile, field):
        return [
            row[field]
            for row in measured_cases
            if row["workload"] == workload and row["profile"] == profile
        ]

    def repetition_ratios(workload, numerator_profile, denominator_profile):
        ratios = []
        for repetition in range(contract.MEASURED_REPETITIONS):
            numerator = [
                row["ttft_ns"]
                for row in measured_cases
                if row["workload"] == workload
                and row["profile"] == numerator_profile
                and row["repetition"] == repetition
            ]
            denominator = [
                row["ttft_ns"]
                for row in measured_cases
                if row["workload"] == workload
                and row["profile"] == denominator_profile
                and row["repetition"] == repetition
            ]
            ratios.append(
                _ratio(
                    _median(numerator, "TTFT numerator"),
                    _median(denominator, "TTFT denominator"),
                    "TTFT repetition ratio",
                )
            )
        return ratios

    performance = {}
    for short_name, workload in (
        ("w1", "w1_medium_reuse"),
        ("w2", "w2_long_reuse"),
        ("w3", "w3_batched_fanout"),
    ):
        int8_ttft = _median(
            case_values(workload, contract.P2_PROFILE, "ttft_ns"),
            f"{short_name} int8 TTFT",
        )
        exact_ttft = _median(
            case_values(workload, "exact_restore", "ttft_ns"),
            f"{short_name} exact TTFT",
        )
        recompute_ttft = _median(
            case_values(workload, "recompute", "ttft_ns"),
            f"{short_name} recompute TTFT",
        )
        int8_e2e = _median(
            case_values(workload, contract.P2_PROFILE, "e2e_ns"),
            f"{short_name} int8 E2E",
        )
        exact_e2e = _median(
            case_values(workload, "exact_restore", "e2e_ns"),
            f"{short_name} exact E2E",
        )
        recompute_e2e = _median(
            case_values(workload, "recompute", "e2e_ns"),
            f"{short_name} recompute E2E",
        )
        performance[short_name] = {
            "int8_to_exact_median_ttft_ratio": _ratio(
                int8_ttft, exact_ttft, f"{short_name} int8/exact TTFT"
            ),
            "int8_to_exact_every_ttft_max_ratio": max(
                repetition_ratios(workload, contract.P2_PROFILE, "exact_restore")
            ),
            "int8_to_recompute_median_ttft_ratio": _ratio(
                int8_ttft, recompute_ttft, f"{short_name} int8/recompute TTFT"
            ),
            "int8_to_exact_throughput_ratio": _ratio(
                exact_e2e, int8_e2e, f"{short_name} int8/exact throughput"
            ),
            "int8_to_recompute_throughput_ratio": _ratio(
                recompute_e2e, int8_e2e, f"{short_name} int8/recompute throughput"
            ),
        }

    def process_values(profile, field):
        return [
            row[field]
            for row in measured_processes
            if row["profile"] == profile
        ]

    exact_physical = _median(
        process_values("exact_restore", "hybrid_cache_current_unique_physical_bytes"),
        "exact physical bytes",
    )
    int8_physical = _median(
        process_values(
            contract.P2_PROFILE, "hybrid_cache_current_unique_physical_bytes"
        ),
        "int8 physical bytes",
    )
    exact_capacity = _median(
        process_values("exact_restore", "same_budget_entry_capacity"),
        "exact capacity",
    )
    int8_capacity = _median(
        process_values(contract.P2_PROFILE, "same_budget_entry_capacity"),
        "int8 capacity",
    )
    exact_reserved = max(
        process_values("exact_restore", "cuda_peak_reserved_bytes")
    )
    int8_reserved = max(
        process_values(contract.P2_PROFILE, "cuda_peak_reserved_bytes")
    )
    int8_decode = _median(
        case_values("w3_batched_fanout", contract.P2_PROFILE, "decode_step_ns"),
        "int8 decode latency",
    )
    recompute_decode = _median(
        case_values("w3_batched_fanout", "recompute", "decode_step_ns"),
        "recompute decode latency",
    )
    forbidden = {
        field: sum(row[field] for row in process_rows)
        for field in FORBIDDEN_COUNTERS
    }
    return {
        "cache": {
            "int8_to_exact_unique_physical_bytes_ratio": _ratio(
                int8_physical, exact_physical, "physical bytes ratio"
            )
        },
        "capacity": {
            "int8_to_exact_same_budget_capacity_ratio": _ratio(
                int8_capacity, exact_capacity, "capacity ratio"
            )
        },
        "performance": {
            **performance,
            "int8_to_recompute_decode_latency_ratio": _ratio(
                int8_decode, recompute_decode, "decode latency ratio"
            ),
        },
        "memory": {
            "int8_to_exact_peak_cuda_reserved_ratio": _ratio(
                int8_reserved, exact_reserved, "peak reserved ratio"
            )
        },
        "safety": {"forbidden_event_counts": forbidden},
    }


def _threshold_rows(metrics):
    return (
        (
            "int8_to_exact_unique_physical_bytes_max_ratio",
            metrics["cache"]["int8_to_exact_unique_physical_bytes_ratio"],
            "<=",
            contract.THRESHOLDS[
                "int8_to_exact_unique_physical_bytes_max_ratio"
            ],
            "cache",
        ),
        (
            "int8_to_exact_same_budget_capacity_min_ratio",
            metrics["capacity"]["int8_to_exact_same_budget_capacity_ratio"],
            ">=",
            contract.THRESHOLDS[
                "int8_to_exact_same_budget_capacity_min_ratio"
            ],
            "cache",
        ),
        (
            "w1_int8_to_exact_median_ttft_max_ratio",
            metrics["performance"]["w1"][
                "int8_to_exact_median_ttft_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "w1_int8_to_exact_median_ttft_max_ratio"
            ],
            "performance",
        ),
        (
            "w1_int8_to_exact_every_ttft_max_ratio",
            metrics["performance"]["w1"][
                "int8_to_exact_every_ttft_max_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "w1_int8_to_exact_every_ttft_max_ratio"
            ],
            "performance",
        ),
        (
            "w2_int8_to_exact_median_ttft_max_ratio",
            metrics["performance"]["w2"][
                "int8_to_exact_median_ttft_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "w2_int8_to_exact_median_ttft_max_ratio"
            ],
            "performance",
        ),
        (
            "w2_int8_to_exact_every_ttft_max_ratio",
            metrics["performance"]["w2"][
                "int8_to_exact_every_ttft_max_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "w2_int8_to_exact_every_ttft_max_ratio"
            ],
            "performance",
        ),
        (
            "w1_int8_to_recompute_median_ttft_max_ratio",
            metrics["performance"]["w1"][
                "int8_to_recompute_median_ttft_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "w1_int8_to_recompute_median_ttft_max_ratio"
            ],
            "performance",
        ),
        (
            "w2_int8_to_recompute_median_ttft_max_ratio",
            metrics["performance"]["w2"][
                "int8_to_recompute_median_ttft_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "w2_int8_to_recompute_median_ttft_max_ratio"
            ],
            "performance",
        ),
        (
            "w3_int8_to_exact_throughput_min_ratio",
            metrics["performance"]["w3"][
                "int8_to_exact_throughput_ratio"
            ],
            ">=",
            contract.THRESHOLDS[
                "w3_int8_to_exact_throughput_min_ratio"
            ],
            "performance",
        ),
        (
            "w3_int8_to_recompute_throughput_min_ratio",
            metrics["performance"]["w3"][
                "int8_to_recompute_throughput_ratio"
            ],
            ">=",
            contract.THRESHOLDS[
                "w3_int8_to_recompute_throughput_min_ratio"
            ],
            "performance",
        ),
        (
            "int8_to_recompute_decode_latency_max_ratio",
            metrics["performance"][
                "int8_to_recompute_decode_latency_ratio"
            ],
            "<=",
            contract.THRESHOLDS[
                "int8_to_recompute_decode_latency_max_ratio"
            ],
            "performance",
        ),
        (
            "int8_to_exact_peak_cuda_reserved_max_ratio",
            metrics["memory"]["int8_to_exact_peak_cuda_reserved_ratio"],
            "<=",
            contract.THRESHOLDS[
                "int8_to_exact_peak_cuda_reserved_max_ratio"
            ],
            "performance",
        ),
    )

def _threshold_checks(metrics):
    rows = _threshold_rows(metrics)

    def passes(group):
        return all(
            (operator == "<=" and value <= threshold)
            or (operator == ">=" and value >= threshold)
            for _, value, operator, threshold, row_group in rows
            if row_group == group
        )

    return passes("cache"), passes("performance")


def _atomic_stage(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _render_report(result, bindings):
    performance = result["performance"]
    lines = [
        "# Qwen3.5 TP4 Hybrid-Prefix Benchmark V2 Verification",
        "",
        f"- Classification: `{result['classification']}`",
        f"- Token IDs equal: `{result['correctness']['token_ids_equal']}`",
        f"- Logits assert-close: `{result['correctness']['logits_assert_close']}`",
        (
            "- Claim boundary: this classification applies only to the bound "
            "model, source, workload, configuration, thresholds, and "
            "artifact hashes listed below."
        ),
        "",
        "## Bound Authority",
        "",
        f"- Source tree SHA256: `{bindings['source_tree_sha256']}`",
        f"- Model manifest SHA256: `{contract.MODEL_MANIFEST_SHA256}`",
        f"- Workload manifest SHA256: `{bindings['workload_manifest_sha256']}`",
        f"- TP world size: `{contract.WORLD_SIZE}`",
        f"- Assigned GPU indices: `{list(contract.REQUIRED_GPU_INDICES)}`",
        f"- Profiles: `{list(contract.PROFILES)}`",
        f"- Workloads: `{list(contract.WORKLOADS)}`",
        f"- Sampling temperature: `{contract.SAMPLING_TEMPERATURE}`",
        f"- Sampling ignore EOS: `{contract.SAMPLING_IGNORE_EOS}`",
        f"- Warmup repetitions: `{contract.WARMUP_REPETITIONS}`",
        f"- Measured repetitions: `{contract.MEASURED_REPETITIONS}`",
        (
            "- Hybrid-prefix entry limit: "
            f"`{contract.HYBRID_PREFIX_MAX_ENTRIES}`"
        ),
        (
            "- Hybrid-prefix byte limit: "
            f"`{contract.HYBRID_PREFIX_MAX_BYTES}`"
        ),
        (
            "- W3 measured concurrency: "
            f"`{contract.WORKLOAD_SPECS['w3_batched_fanout']['continuations']}`"
        ),
        "",
        "## Measured Ratios",
        "",
        (
            "- W1 int8/exact median TTFT ratio: "
            f"`{performance['w1']['int8_to_exact_median_ttft_ratio']:.6f}`"
        ),
        (
            "- W1 int8/exact every-repetition TTFT max ratio: "
            f"`{performance['w1']['int8_to_exact_every_ttft_max_ratio']:.6f}`"
        ),
        (
            "- W1 int8/recompute median TTFT ratio: "
            f"`{performance['w1']['int8_to_recompute_median_ttft_ratio']:.6f}`"
        ),
        (
            "- W2 int8/exact median TTFT ratio: "
            f"`{performance['w2']['int8_to_exact_median_ttft_ratio']:.6f}`"
        ),
        (
            "- W2 int8/exact every-repetition TTFT max ratio: "
            f"`{performance['w2']['int8_to_exact_every_ttft_max_ratio']:.6f}`"
        ),
        (
            "- W2 int8/recompute median TTFT ratio: "
            f"`{performance['w2']['int8_to_recompute_median_ttft_ratio']:.6f}`"
        ),
        (
            "- W3 int8/exact concurrent E2E proxy ratio: "
            f"`{performance['w3']['int8_to_exact_throughput_ratio']:.6f}`"
        ),
        (
            "- W3 int8/recompute concurrent E2E proxy ratio: "
            f"`{performance['w3']['int8_to_recompute_throughput_ratio']:.6f}`"
        ),
        (
            "- Int8/recompute decode latency ratio: "
            f"`{performance['int8_to_recompute_decode_latency_ratio']:.6f}`"
        ),
        (
            "- Int8/exact peak CUDA reserved ratio: "
            f"`{result['memory']['int8_to_exact_peak_cuda_reserved_ratio']:.6f}`"
        ),
        (
            "- Int8/exact unique physical cache bytes ratio: "
            f"`{result['cache']['int8_to_exact_unique_physical_bytes_ratio']:.6f}`"
        ),
        (
            "- Int8/exact same-budget capacity ratio: "
            f"`{result['capacity']['int8_to_exact_same_budget_capacity_ratio']:.6f}`"
        ),
        (
            "- W3 boundary: the ratios above are concurrent per-request E2E "
            "proxies, not sustained serving QPS, tokens/s, arrival-rate "
            "saturation, or batch makespan."
        ),
        "",
        "## Frozen Threshold Gates",
        "",
    ]
    for name, threshold in contract.THRESHOLDS.items():
        lines.append(f"- `{name}`: `{threshold}`")
    lines.extend(
        (
            "",
        "| Threshold | Measured | Operator | Frozen threshold | Status |",
        "| --- | ---: | :---: | ---: | :---: |",
        )
    )
    for name, value, operator, threshold, _ in _threshold_rows(result):
        passed = (
            (operator == "<=" and value <= threshold)
            or (operator == ">=" and value >= threshold)
        )
        lines.append(
            f"| `{name}` | `{value:.6f}` | `{operator}` | "
            f"`{threshold}` | `{'PASS' if passed else 'FAIL'}` |"
        )
    lines.extend(("", "## Artifact Hashes", ""))
    for entry in bindings["artifact_entries"]:
        lines.append(f"- `{entry['path']}`: `{entry['sha256']}`")
    lines.append("")
    return "\n".join(lines)


def verify_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _fail("run directory is missing")
    for relative in contract.VERIFIER_TRUST_DOMAIN:
        (run_dir / relative).unlink(missing_ok=True)
    _verify_closed_inventory(run_dir)
    artifact_manifest = _verify_artifact_manifest(run_dir)
    manifests, file_inventory = _verify_manifests(run_dir)
    case_rows = _load_jsonl(run_dir / "case_rows.jsonl")
    process_rows = _load_jsonl(run_dir / "process_rows.jsonl")
    try:
        contract.validate_case_rows(case_rows)
    except ValueError as error:
        _fail(f"canonical case row evidence is invalid: {error}")
    _verify_process_rows(process_rows)
    try:
        contract.validate_case_process_row_bindings(case_rows, process_rows)
    except ValueError as error:
        _fail(f"case/process evidence binding is invalid: {error}")
    try:
        contract.validate_artifact_evidence(
            case_rows,
            process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )
    except ValueError as error:
        _fail(f"canonical artifact evidence is invalid: {error}")
    _verify_static_documents(run_dir)
    _verify_authorities(run_dir, case_rows)
    _verify_summary(run_dir, case_rows, process_rows)
    _verify_tensor_evidence(run_dir, manifests["tensor_inventories"])
    correctness = _verify_tokens_and_logits(run_dir, case_rows)
    metrics = _aggregate_metrics(case_rows, process_rows)
    cache_pass, performance_pass = _threshold_checks(metrics)
    classification = contract.classify_run(
        {
            "artifact_invalid": False,
            "resources_blocked": False,
            "correctness_pass": all(correctness.values()),
            "runtime_safety_pass": not any(
                metrics["safety"]["forbidden_event_counts"].values()
            ),
            "cache_pass": cache_pass,
            "performance_pass": performance_pass,
        }
    )
    result = {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": classification,
        "logit_tolerance": {"atol": LOGIT_ATOL, "rtol": LOGIT_RTOL},
        "correctness": correctness,
        **metrics,
    }
    verification_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    source = _load_json(run_dir / "source_manifest.json")
    report_bytes = _render_report(
        result,
        {
            "source_tree_sha256": source["source_tree_sha256"],
            "workload_manifest_sha256": _sha256(
                run_dir / "workload_manifest.json"
            ),
            "artifact_entries": artifact_manifest["entries"],
        },
    ).encode("utf-8")
    verification_temporary = _atomic_stage(
        run_dir / "independent_verification.json", verification_bytes
    )
    report_temporary = _atomic_stage(run_dir / "report.md", report_bytes)
    try:
        verification_temporary.replace(
            run_dir / "independent_verification.json"
        )
        report_temporary.replace(run_dir / "report.md")
    except BaseException:
        (run_dir / "independent_verification.json").unlink(missing_ok=True)
        (run_dir / "report.md").unlink(missing_ok=True)
        raise
    finally:
        verification_temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
    return result
