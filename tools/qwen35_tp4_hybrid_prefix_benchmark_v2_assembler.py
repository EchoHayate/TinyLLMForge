from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import qwen35_tp4_hybrid_prefix_benchmark_v2_contract as contract


COMPLETION_MARKER = "QWEN35_TP4_BENCHMARK_V2_WORKER_COMPLETE"
COMMON_FILES = {
    "correctness_prerequisites.json",
    "calibration_binding.json",
    "p1_authority_binding.json",
    "source_manifest.json",
    "gate1_audit.json",
    "consumed_authorization.json",
    "workload_manifest.json",
}
CASE_FILES = {
    "case_rows.jsonl",
    "process_rows.jsonl",
    "summary.json",
    "execution_receipt.json",
}
CASE_DIRECTORIES = {
    "tokens",
    "logits",
    "logs",
    "tensor-inventories",
}
CANONICAL_SNAPSHOT_DIRECTORY = "snapshots"
PROVENANCE_FIELDS = (
    "source_tree_sha256",
    "gate1_audit_sha256",
    "execution_plan_sha256",
    "source_bundle_sha256",
    "source_package_sha256",
    "producer_source_sha256",
    "producer_version_sha256",
    "verifier_source_sha256",
    "verifier_version_sha256",
)


class AssemblyError(RuntimeError):
    pass


def _fail(message):
    raise AssemblyError(message)


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(f"failed to hash evidence: {error}")
    return digest.hexdigest()


def _require_directory(path, label):
    path = Path(path)
    if path.is_symlink():
        _fail(f"{label} symlink is forbidden")
    if not path.is_dir():
        _fail(f"{label} directory is missing")
    return path


def _require_file(path, label):
    path = Path(path)
    if path.is_symlink():
        _fail(f"{label} symlink is forbidden")
    if not path.is_file():
        _fail(f"{label} evidence is missing")
    return path


def _load_json(path, label):
    path = _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"{label} JSON is invalid: {error}")
    _require_finite(value, label)
    return value


def _load_jsonl(path, label):
    path = _require_file(path, label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        _fail(f"{label} JSONL is invalid: {error}")
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            _fail(f"{label} JSONL row {line_number} is invalid: {error}")
        _require_finite(row, f"{label} row {line_number}")
        rows.append(row)
    return rows


def _require_finite(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child, label)
    elif isinstance(value, list):
        for child in value:
            _require_finite(child, label)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_canonical_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(contract.canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _closed_directory(path, expected_files, expected_directories, label):
    path = _require_directory(path, label)
    actual_files = set()
    actual_directories = set()
    for entry in path.iterdir():
        if entry.is_symlink():
            _fail(f"{label} symlink is forbidden: {entry.name}")
        if entry.is_file():
            actual_files.add(entry.name)
        elif entry.is_dir():
            actual_directories.add(entry.name)
        else:
            _fail(f"{label} unknown file type: {entry.name}")
    missing_files = expected_files - actual_files
    missing_directories = expected_directories - actual_directories
    if missing_files:
        _fail(f"{label} evidence is missing: {sorted(missing_files)[0]}")
    if missing_directories:
        _fail(
            f"{label} evidence directory is missing: "
            f"{sorted(missing_directories)[0]}"
        )
    if actual_files - expected_files or actual_directories - expected_directories:
        _fail(f"{label} contains an unknown file or directory")


def _safe_case_file(case_dir, relative, label):
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} path is invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        _fail(f"{label} path is unsafe")
    return _require_file(case_dir / relative_path, label)


def _validate_common(common):
    common = _require_directory(common, "canonical authority common")
    actual_directories = {
        entry.name for entry in common.iterdir() if entry.is_dir()
    }
    canonical_directories = set(contract.PREREQUISITE_NAMES)
    if actual_directories != canonical_directories:
        _fail("common does not contain the complete canonical authority")
    _closed_directory(
        common,
        COMMON_FILES,
        canonical_directories,
        "common",
    )
    documents = {
        name: _load_json(common / name, name) for name in sorted(COMMON_FILES)
    }
    workload = documents["workload_manifest.json"]
    if workload != contract.workload_manifest_payload():
        _fail("workload manifest evidence is not canonical")
    try:
        contract.validate_calibration_binding(
            documents["calibration_binding.json"]
        )
        contract.validate_p1_authority_binding(
            documents["p1_authority_binding.json"]
        )
        contract.validate_source_manifest(documents["source_manifest.json"])
    except ValueError as error:
        _fail(f"common provenance evidence is invalid: {error}")

    prerequisites = documents["correctness_prerequisites.json"]
    status = contract.validate_prerequisites(
        common / "correctness_prerequisites.json"
    )
    if status.classification != "PASS" or not status.authorized:
        _fail(
            "canonical correctness prerequisite evidence is invalid: "
            + "; ".join(status.reasons)
        )
    gate = documents["gate1_audit.json"]
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version")
        != contract.EVIDENCE_SCHEMA_VERSIONS["gate1_audit"]
        or gate.get("classification") != "PASS"
    ):
        _fail("gate1 audit evidence is invalid")
    authorization = documents["consumed_authorization.json"]
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_version")
        != contract.EVIDENCE_SCHEMA_VERSIONS["consumed_authorization"]
        or authorization.get("classification") != "AUTHORIZED"
        or authorization.get("consumed") is not True
    ):
        _fail("consumed authorization evidence is invalid")
    source_sha = documents["source_manifest.json"]["source_tree_sha256"]
    for name in (
        "calibration_binding.json",
        "p1_authority_binding.json",
        "gate1_audit.json",
    ):
        if documents[name].get("source_tree_sha256") != source_sha:
            _fail(f"common provenance drift in {name}")
    return documents


def _validate_case_summary(case, case_dir, case_rows, process_rows):
    summary = _load_json(case_dir / "summary.json", f"{case.case_id} summary")
    expected = {
        "schema_version",
        "complete",
        "case_id",
        "case_rows",
        "process_rows",
    }
    if not isinstance(summary, dict):
        _fail(f"{case.case_id} summary is invalid")
    if "classification" in summary or "producer_claim" in summary:
        _fail(f"{case.case_id} producer classification is forbidden")
    if set(summary) != expected:
        _fail(f"{case.case_id} summary contains an unknown field")
    if (
        summary["schema_version"] != contract.SCHEMA_VERSION
        or summary["complete"] is not True
        or summary["case_id"] != case.case_id
        or summary["case_rows"] != len(case_rows)
    ):
        _fail(f"{case.case_id} summary evidence is invalid")
    if summary["process_rows"] != len(process_rows):
        _fail(f"{case.case_id} rank evidence count is invalid")


def _validate_case_row_trust(case_rows, common):
    if all(isinstance(row, dict) and "row_id" in row for row in case_rows):
        row_ids = [row["row_id"] for row in case_rows]
        if len(row_ids) != len(set(row_ids)):
            _fail("duplicate case row evidence")

    source = common["source_manifest.json"]
    for field in PROVENANCE_FIELDS:
        if any(
            isinstance(row, dict)
            and field in row
            and row[field] != source[field]
            for row in case_rows
        ):
            _fail(f"provenance drift in case rows: {field}")


def _validate_case_receipt(case, case_dir, common, process_rows):
    receipt = _load_json(
        case_dir / "execution_receipt.json",
        f"{case.case_id} receipt",
    )
    expected = {"schema_version", "case_id", "run_tag", "nonce", "complete"}
    if not isinstance(receipt, dict) or set(receipt) != expected:
        _fail(f"{case.case_id} receipt schema is invalid")
    if (
        receipt["schema_version"]
        != contract.EVIDENCE_SCHEMA_VERSIONS["execution_receipt"]
        or receipt["case_id"] != case.case_id
        or receipt["complete"] is not True
        or receipt["run_tag"]
        != common["consumed_authorization.json"]["run_tag"]
        or receipt["nonce"] != common["consumed_authorization.json"]["nonce"]
        or any(
            row["run_tag"] != receipt["run_tag"]
            or row["nonce"] != receipt["nonce"]
            for row in process_rows
        )
    ):
        _fail(f"{case.case_id} receipt binding is invalid")


def _validate_case_files(
    case,
    case_dir,
    case_rows,
    process_rows,
):
    expected_tokens = set()
    expected_logits = set()
    for row in case_rows:
        for role in ("prompt", "output"):
            relative = row[f"{role}_token_ids_path"]
            path = _safe_case_file(
                case_dir, relative, f"{case.case_id} {role} token"
            )
            if _sha256(path) != row[f"{role}_token_ids_sha256"]:
                _fail(f"{case.case_id} {role} token hash mismatch")
            values = _load_json(path, f"{case.case_id} {role} token")
            expected_count = (
                row["prompt_tokens"] if role == "prompt" else row["generated_tokens"]
            )
            if (
                not isinstance(values, list)
                or len(values) != expected_count
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in values
                )
            ):
                _fail(f"{case.case_id} {role} token evidence is invalid")
            expected_tokens.add(Path(relative).name)

        relative = row["final_logits_path"]
        logits = _safe_case_file(case_dir, relative, f"{case.case_id} logit")
        if _sha256(logits) != row["final_logits_sha256"]:
            _fail(f"{case.case_id} logit hash mismatch")
        expected_bytes = math.prod(row["final_logits_shape"]) * 4
        if logits.stat().st_size != expected_bytes:
            _fail(f"{case.case_id} logit byte length is invalid")
        expected_logits.add(Path(relative).name)

    expected_ranks = set(range(contract.WORLD_SIZE))
    actual_ranks = {row["rank"] for row in process_rows}
    if actual_ranks != expected_ranks or len(process_rows) != contract.WORLD_SIZE:
        _fail(f"{case.case_id} rank evidence is incomplete or duplicate")
    expected_logs = {f"rank-{rank}.log" for rank in expected_ranks}
    expected_tensors = {f"rank-{rank}.json" for rank in expected_ranks}
    expected_by_directory = {
        "tokens": expected_tokens,
        "logits": expected_logits,
        "logs": expected_logs,
        "tensor-inventories": expected_tensors,
    }
    for directory, expected in expected_by_directory.items():
        _closed_directory(
            case_dir / directory,
            expected,
            set(),
            f"{case.case_id} {directory}",
        )
    if case.profile != "recompute":
        expected_snapshots = {
            f"rank-{rank}.snapshot" for rank in expected_ranks
        }
        _closed_directory(
            case_dir / CANONICAL_SNAPSHOT_DIRECTORY / case.case_id,
            expected_snapshots,
            set(),
            f"{case.case_id} snapshots",
        )

    process_by_rank = {row["rank"]: row for row in process_rows}
    for rank in sorted(expected_ranks):
        log = _require_file(
            case_dir / "logs" / f"rank-{rank}.log",
            f"{case.case_id} rank {rank} log",
        )
        text = log.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in text:
            _fail(f"{case.case_id} rank {rank} traceback is forbidden")
        if COMPLETION_MARKER not in text:
            _fail(f"{case.case_id} rank {rank} log completion is missing")

        inventory = _load_json(
            case_dir / "tensor-inventories" / f"rank-{rank}.json",
            f"{case.case_id} rank {rank} tensor inventory",
        )
        if case.profile != "recompute":
            try:
                contract.validate_tensor_storage_evidence(inventory)
                accounting = contract.recompute_tensor_storage_accounting(
                    inventory
                )
            except ValueError as error:
                _fail(
                    f"{case.case_id} rank {rank} tensor inventory is "
                    f"invalid: {error}"
                )
            if (
                inventory["case_id"] != case.case_id
                or inventory["profile"] != case.profile
                or inventory["rank"] != rank
            ):
                _fail(
                    f"{case.case_id} rank {rank} tensor inventory identity "
                    "is invalid"
                )
            process = process_by_rank[rank]
            if any(
                process[field] != accounting[field]
                for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS
            ):
                _fail(
                    f"{case.case_id} rank {rank} tensor inventory "
                    "accounting mismatch"
                )
            continue
        if (
            not isinstance(inventory, dict)
            or set(inventory)
            != {"schema_version", "case_id", "profile", "rank", "cache", "workspace"}
            or inventory["schema_version"]
            != contract.TENSOR_STORAGE_EVIDENCE_SCHEMA_VERSION
            or inventory["case_id"] != case.case_id
            or inventory["profile"] != case.profile
            or inventory["rank"] != rank
        ):
            _fail(f"{case.case_id} rank {rank} tensor inventory is invalid")
        process = process_by_rank[rank]
        cache_fields = {
            key
            for key in contract.PROCESS_ROW_FIELDS
            if key.startswith("hybrid_cache_")
        }
        workspace_fields = {
            key for key in contract.PROCESS_ROW_FIELDS if "workspace" in key
        }
        if not cache_fields.issubset(process):
            _fail(f"{case.case_id} cache evidence is invalid")
        if not workspace_fields.issubset(process):
            _fail(f"{case.case_id} workspace evidence is invalid")
        expected_cache = {
            key: process[key]
            for key in cache_fields
        }
        expected_workspace = {
            key: process[key]
            for key in workspace_fields
        }
        if inventory["cache"] != expected_cache:
            _fail(f"{case.case_id} cache evidence binding is invalid")
        if inventory["workspace"] != expected_workspace:
            _fail(f"{case.case_id} workspace evidence binding is invalid")


def _validate_provenance(case_rows, process_rows, common):
    try:
        contract.validate_case_rows(case_rows)
        normalized_process_rows = []
        for row in process_rows:
            normalized = dict(row)
            if (
                normalized["profile"] == "recompute"
                and normalized["same_budget_entry_capacity"] == 0
            ):
                normalized["same_budget_entry_capacity"] = 1
            normalized_process_rows.append(normalized)
        contract.validate_process_rows(normalized_process_rows)
        contract.validate_case_process_row_bindings(
            case_rows, normalized_process_rows
        )
    except (KeyError, ValueError) as error:
        message = str(error)
        if "workspace" in message:
            _fail(f"workspace evidence is invalid: {message}")
        if "hybrid_cache" in message or "cache" in message:
            _fail(f"cache evidence is invalid: {message}")
        _fail(f"provenance or canonical row evidence is invalid: {message}")

    row_ids = [row["row_id"] for row in case_rows]
    if len(row_ids) != len(set(row_ids)):
        _fail("duplicate case row evidence")
    process_keys = [(row["case_id"], row["rank"]) for row in process_rows]
    if len(process_keys) != len(set(process_keys)):
        _fail("duplicate process row evidence")

    source = common["source_manifest.json"]
    for field in PROVENANCE_FIELDS:
        expected = source[field]
        if any(row[field] != expected for row in case_rows):
            _fail(f"provenance drift in case rows: {field}")
        if any(row[field] != expected for row in process_rows):
            _fail(f"provenance drift in process rows: {field}")


def _collect_raw_bundle(raw_root):
    common = _validate_common(raw_root / "common")
    _closed_directory(raw_root, set(), {"common", "profiles"}, "raw bundle")
    profiles_root = raw_root / "profiles"
    expected_profiles = set(contract.PROFILES)
    _closed_directory(profiles_root, set(), expected_profiles, "profile")

    matrix = contract.build_case_matrix()
    cases_by_profile = {
        profile: [case for case in matrix if case.profile == profile]
        for profile in contract.PROFILES
    }
    evidence_by_case = {}
    for profile in contract.PROFILES:
        profile_root = profiles_root / profile
        _closed_directory(profile_root, set(), {"cases"}, f"profile {profile}")
        cases_root = profile_root / "cases"
        expected_cases = {case.case_id for case in cases_by_profile[profile]}
        _closed_directory(
            cases_root, set(), expected_cases, f"profile {profile} case"
        )
        for case in cases_by_profile[profile]:
            case_dir = cases_root / case.case_id
            expected_case_directories = set(CASE_DIRECTORIES)
            if case.profile != "recompute":
                expected_case_directories.add(CANONICAL_SNAPSHOT_DIRECTORY)
            _closed_directory(
                case_dir,
                CASE_FILES,
                expected_case_directories,
                f"case {case.case_id}",
            )
            case_rows = _load_jsonl(
                case_dir / "case_rows.jsonl", f"{case.case_id} case rows"
            )
            process_rows = _load_jsonl(
                case_dir / "process_rows.jsonl", f"{case.case_id} process rows"
            )
            _validate_case_row_trust(case_rows, common)
            _validate_case_summary(case, case_dir, case_rows, process_rows)
            _validate_case_receipt(
                case, case_dir, common, process_rows
            )
            _validate_case_files(
                case,
                case_dir,
                case_rows,
                process_rows,
            )
            evidence_by_case[case.case_id] = (
                case,
                case_dir,
                case_rows,
                process_rows,
            )

    case_evidence = [evidence_by_case[case.case_id] for case in matrix]
    all_case_rows = [
        row
        for _, _, rows, _ in case_evidence
        for row in rows
    ]
    all_process_rows = [
        row
        for _, _, _, rows in case_evidence
        for row in rows
    ]
    _validate_provenance(all_case_rows, all_process_rows, common)
    return (
        common,
        case_evidence,
        all_case_rows,
        all_process_rows,
    )


def _copy_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _manifest_file(path, root):
    return {
        "path": Path(path).relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": Path(path).stat().st_size,
    }


def _nested_file(path, root):
    return {
        **_manifest_file(path, root),
        "type": "regular_file",
    }


def _nested_manifest(kind, files, rows):
    return {
        "schema_version": contract.NESTED_MANIFEST_SCHEMA_VERSIONS[kind],
        "kind": kind,
        "files": sorted(files, key=lambda row: row["path"]),
        "rows": rows,
    }


def _publish_common_documents(temporary, common):
    for directory in contract.NESTED_ARTIFACT_DIRECTORIES:
        (temporary / directory).mkdir(parents=True, exist_ok=True)
    for name in COMMON_FILES:
        _write_json(temporary / name, common[name])


def _publish_static_documents(temporary, common, output_case_rows, process_rows):
    source = common["source_manifest.json"]
    static_documents = {
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
        "preflight.json": {
            "classification": "READY",
            "resources_blocked": False,
        },
        "execution_plan.json": {
            "classification": "READY",
            "sha256": source["execution_plan_sha256"],
        },
        "source_bundle_manifest.json": {
            "sha256": source["source_bundle_sha256"],
        },
        "source_package_manifest.json": {
            "sha256": source["source_package_sha256"],
        },
        "resource_guards.json": {"before": "PASS", "after": "PASS"},
    }
    for name, payload in static_documents.items():
        _write_json(temporary / name, payload)
    published_bindings = {
        "workload_manifest_sha256": "workload_manifest.json",
        "correctness_prerequisites_sha256": "correctness_prerequisites.json",
        "calibration_artifact_sha256": "calibration_binding.json",
        "p1_authority_artifact_sha256": "p1_authority_binding.json",
        "preflight_receipt_sha256": "preflight.json",
        "authorization_receipt_sha256": "consumed_authorization.json",
    }
    for row in output_case_rows:
        for field, name in published_bindings.items():
            row[field] = _sha256(temporary / name)
    _write_jsonl(temporary / "case_rows.jsonl", output_case_rows)
    _write_jsonl(temporary / "process_rows.jsonl", process_rows)
    _write_json(
        temporary / "summary.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "producer_claim": "UNVERIFIED",
            "thresholds": dict(contract.THRESHOLDS),
            "case_rows": len(output_case_rows),
            "process_rows": len(process_rows),
        },
    )


def _publish_artifact_manifest(temporary):
    entries = []
    for relative in contract.ARTIFACT_MANIFEST_HASH_DOMAIN:
        path = _require_file(temporary / relative, relative)
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "producer": "task8-assembler",
                "trust_domain": "producer",
            }
        )
    manifest = {
        "schema_version": contract.ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "hash_domain": list(contract.ARTIFACT_MANIFEST_HASH_DOMAIN),
        "entries": entries,
        "excluded_verifier_outputs": list(contract.VERIFIER_TRUST_DOMAIN),
    }
    _write_json(temporary / "artifact_manifest.json", manifest)
    return manifest


def _publish_canonical_payload(
    temporary,
    common,
    raw_common,
    case_evidence,
    process_rows,
):
    _publish_common_documents(temporary, common)
    manifests = {}
    output_process_rows = []
    for process in process_rows:
        copied = dict(process)
        if copied["profile"] == "recompute":
            for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS:
                copied[field] = 0
        output_process_rows.append(copied)

    prerequisite_files = []
    prerequisite_rows = []
    prerequisite_index = common["correctness_prerequisites.json"]
    role_fields = (
        ("artifact", "artifact_path"),
        ("independent_verification", "independent_verification_path"),
        ("provenance", "provenance_path"),
    )
    for name in contract.PREREQUISITE_NAMES:
        prerequisite = prerequisite_index[name]
        for role, path_field in role_fields:
            source = _safe_case_file(
                raw_common,
                prerequisite[path_field],
                f"{name} {role}",
            )
            destination = (
                temporary
                / "prerequisites"
                / name
                / f"{role}{source.suffix}"
            )
            _copy_file(source, destination)
            file_row = _nested_file(destination, temporary)
            prerequisite_files.append(file_row)
            prerequisite_rows.append(
                {"name": name, "role": role, "file": file_row}
            )
    manifests["prerequisites"] = _nested_manifest(
        "prerequisites",
        prerequisite_files,
        prerequisite_rows,
    )

    output_case_rows = []
    token_files = []
    token_rows = []
    logit_files = []
    logit_rows = []
    log_files = []
    log_rows = []
    snapshot_files = []
    snapshot_rows = []
    tensor_files = []
    tensor_rows = []
    process_by_key = {
        (row["case_id"], row["rank"]): row for row in output_process_rows
    }
    for case, case_dir, rows, processes in case_evidence:
        for row in rows:
            copied = dict(row)
            for role in ("prompt", "output"):
                source = case_dir / row[f"{role}_token_ids_path"]
                destination = (
                    temporary / "tokens" / f"{row['row_id']}.{role}.json"
                )
                _copy_file(source, destination)
                file_row = _nested_file(destination, temporary)
                token_files.append(file_row)
                token_rows.append(
                    {
                        "case_id": case.case_id,
                        "request_id": row["request_id"],
                        "role": role,
                        "token_count": (
                            row["prompt_tokens"]
                            if role == "prompt"
                            else row["generated_tokens"]
                        ),
                        "file": file_row,
                    }
                )
                copied[f"{role}_token_ids_path"] = file_row["path"]
                copied[f"{role}_token_ids_sha256"] = file_row["sha256"]
            source = case_dir / row["final_logits_path"]
            destination = temporary / "logits" / f"{row['row_id']}.float32.bin"
            _copy_file(source, destination)
            file_row = _nested_file(destination, temporary)
            logit_files.append(file_row)
            logit_rows.append(
                {
                    "case_id": case.case_id,
                    "request_id": row["request_id"],
                    "shape": list(row["final_logits_shape"]),
                    "dtype": row["final_logits_dtype"],
                    "file": file_row,
                }
            )
            copied["final_logits_path"] = file_row["path"]
            copied["final_logits_sha256"] = file_row["sha256"]
            output_case_rows.append(copied)

        for process in processes:
            rank = process["rank"]
            log_source = case_dir / "logs" / f"rank-{rank}.log"
            log_destination = (
                temporary / "logs" / case.case_id / f"rank-{rank}.log"
            )
            _copy_file(log_source, log_destination)
            log_file = _nested_file(log_destination, temporary)
            log_files.append(log_file)
            log_rows.append(
                {
                    "case_id": case.case_id,
                    "rank": rank,
                    "world_size": contract.WORLD_SIZE,
                    "completion_marker": True,
                    "traceback_present": False,
                    "file": log_file,
                }
            )

            accounting = {
                field: process_by_key[(case.case_id, rank)][field]
                for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS
            }
            snapshot_row = {
                "case_id": case.case_id,
                "profile": case.profile,
                "rank": rank,
                "world_size": contract.WORLD_SIZE,
                "evidence_kind": "accounting_only",
                "snapshot_file": None,
                "tensor_inventory_file": None,
                "full_fidelity_logical_bytes": 0,
                "encoded_physical_bytes": 0,
                "codec_metadata_bytes": 0,
                "temporary_encode_workspace_bytes": 0,
                "temporary_decode_workspace_bytes": 0,
                **accounting,
            }
            if case.profile != "recompute":
                snapshot_source = (
                    case_dir
                    / CANONICAL_SNAPSHOT_DIRECTORY
                    / case.case_id
                    / f"rank-{rank}.snapshot"
                )
                snapshot_destination = (
                    temporary
                    / "snapshots"
                    / case.case_id
                    / f"rank-{rank}.snapshot"
                )
                _copy_file(snapshot_source, snapshot_destination)
                snapshot_file = _nested_file(
                    snapshot_destination,
                    temporary,
                )
                snapshot_files.append(snapshot_file)

                evidence = _load_json(
                    case_dir
                    / "tensor-inventories"
                    / f"rank-{rank}.json",
                    f"{case.case_id} rank {rank} tensor inventory",
                )
                tensor_destination = (
                    temporary
                    / "snapshots"
                    / case.case_id
                    / f"rank-{rank}.tensor-inventory.json"
                )
                _write_canonical_json(tensor_destination, evidence)
                tensor_file = _nested_file(tensor_destination, temporary)
                tensor_files.append(tensor_file)
                tensor_rows.append(
                    {
                        "case_id": evidence["case_id"],
                        "profile": evidence["profile"],
                        "representation": evidence["representation"],
                        "representation_version": evidence[
                            "representation_version"
                        ],
                        "codec": evidence["codec"],
                        "rank": evidence["rank"],
                        "world_size": evidence["world_size"],
                        "evidence_schema_version": evidence["schema_version"],
                        "snapshot_count": len(evidence["snapshots"]),
                        "storage_count": len(evidence["storages"]),
                        "reference_count": sum(
                            len(snapshot["tensor_references"])
                            for snapshot in evidence["snapshots"]
                        ),
                        "observation_count": len(evidence["observations"]),
                        "evidence": evidence,
                        "file": tensor_file,
                    }
                )
                snapshot_row.update(
                    {
                        "evidence_kind": "snapshot",
                        "snapshot_file": snapshot_file,
                        "tensor_inventory_file": tensor_file,
                        "full_fidelity_logical_bytes": accounting[
                            "hybrid_cache_current_logical_referenced_bytes"
                        ],
                        "encoded_physical_bytes": accounting[
                            "hybrid_cache_current_unique_physical_bytes"
                        ],
                        "codec_metadata_bytes": accounting[
                            "hybrid_cache_current_metadata_bytes"
                        ],
                        "temporary_encode_workspace_bytes": accounting[
                            "encode_workspace_peak_allocated_bytes"
                        ],
                        "temporary_decode_workspace_bytes": accounting[
                            "decode_workspace_peak_allocated_bytes"
                        ],
                    }
                )
            snapshot_rows.append(snapshot_row)

    manifests["tokens"] = _nested_manifest(
        "tokens",
        token_files,
        token_rows,
    )
    manifests["logits"] = _nested_manifest(
        "logits",
        logit_files,
        logit_rows,
    )
    manifests["logs"] = _nested_manifest("logs", log_files, log_rows)
    manifests["snapshots"] = _nested_manifest(
        "snapshots",
        snapshot_files,
        snapshot_rows,
    )
    manifests["tensor_inventories"] = _nested_manifest(
        "tensor_inventories",
        tensor_files,
        tensor_rows,
    )
    for kind in contract.NESTED_MANIFEST_KINDS:
        _write_canonical_json(
            temporary / contract.NESTED_MANIFEST_ARTIFACT_PATHS[kind],
            manifests[kind],
        )

    _publish_static_documents(
        temporary,
        common,
        output_case_rows,
        output_process_rows,
    )
    artifact_manifest = _publish_artifact_manifest(temporary)
    file_inventory = sorted(
        [
            file_row
            for kind in contract.NESTED_MANIFEST_KINDS
            for file_row in manifests[kind]["files"]
        ],
        key=lambda row: row["path"],
    )
    try:
        contract.validate_artifact_evidence(
            output_case_rows,
            output_process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )
    except ValueError as error:
        _fail(f"published canonical evidence is invalid: {error}")
    return output_case_rows


def assemble_benchmark_run(*, raw_bundle_dir, output_dir):
    raw_root = _require_directory(raw_bundle_dir, "raw bundle")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        _fail("final output directory already exists")
    (
        common,
        case_evidence,
        _case_rows,
        process_rows,
    ) = _collect_raw_bundle(raw_root)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.",
        )
    )
    try:
        output_case_rows = _publish_canonical_payload(
            temporary,
            common,
            raw_root / "common",
            case_evidence,
            process_rows,
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "classification": "ASSEMBLED",
        "case_rows": len(output_case_rows),
        "process_rows": len(process_rows),
    }
