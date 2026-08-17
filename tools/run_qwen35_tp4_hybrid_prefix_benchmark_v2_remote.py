from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import sys
import tarfile


def _load_contract():
    name = "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_remote"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / (
        "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()

SSH_TARGET = contract.EXECUTION_SSH_TARGET
REQUIRED_GPU_INDICES = contract.REQUIRED_GPU_INDICES
MIN_GPU_FREE_BYTES = contract.MIN_GPU_FREE_BYTES
KRB5CCNAME = contract.EXECUTION_ENV["KRB5CCNAME"]
SSH_OPTIONS = contract.EXECUTION_SSH_OPTIONS


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle_file(output_dir, relative_path, label):
    if output_dir is None:
        raise ValueError(f"{label} output directory is required")
    root = Path(output_dir).resolve()
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{label} path is invalid")
    candidate = root / Path(*pure.parts)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes output directory") from error
    return candidate


def _validate_source_bundle_bytes(source_bundle, output_dir):
    tar_path = _resolve_bundle_file(
        output_dir,
        source_bundle["path"],
        "source bundle",
    )
    inventory_path = _resolve_bundle_file(
        output_dir,
        source_bundle["inventory_path"],
        "source inventory",
    )
    tar_bytes = contract.read_regular_file_once(tar_path, "source bundle")
    inventory_bytes = contract.read_regular_file_once(
        inventory_path,
        "source inventory",
    )
    if hashlib.sha256(tar_bytes).hexdigest() != source_bundle["sha256"]:
        raise ValueError("source bundle byte sha256 mismatch")
    try:
        inventory = json.loads(inventory_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source inventory JSON is invalid") from error
    if inventory != source_bundle["inventory"]:
        raise ValueError("source inventory file does not match document")
    if inventory_bytes != (
        contract.canonical_json_bytes(inventory) + b"\n"
    ):
        raise ValueError("source inventory file is not canonical JSON")
    if contract.canonical_json_sha256(inventory) != source_bundle[
        "inventory_sha256"
    ]:
        raise ValueError("source inventory hash is invalid")
    expected = {row["path"]: row for row in inventory}
    actual = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ValueError("source bundle tar inventory is invalid")
            for member in members:
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError("source bundle tar member is unsafe")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source bundle tar member is unreadable")
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                actual[member.name] = {
                    "path": member.name,
                    "sha256": digest.hexdigest(),
                    "bytes": size,
                    "type": "file",
                }
    except (OSError, tarfile.TarError) as error:
        raise ValueError("source bundle tar is invalid") from error
    if actual != expected:
        raise ValueError("source bundle tar content does not match inventory")


def _load_authoritative_document(row, label):
    if (
        not isinstance(row, dict)
        or set(row) != {"classification", "sha256", "path"}
    ):
        raise ValueError(f"{label} prerequisite document is required")
    path = Path(row["path"])
    try:
        data, document = contract.load_json_file_once(path, label)
    except ValueError as error:
        raise ValueError(f"{label} prerequisite path is invalid") from error
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"{label} prerequisite SHA mismatch")
    return path, data, document


def _validate_binding_reference(document_path, relative_path, expected_sha, label):
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    trusted_root = document_path.parent.resolve()
    candidate = trusted_root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(trusted_root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes trusted root") from error
    data = contract.read_regular_file_once(candidate, label)
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"{label} SHA mismatch")


def _validate_authoritative_prerequisites(prerequisites):
    if not isinstance(prerequisites, dict):
        raise ValueError("prerequisite authority is invalid")
    required = {
        "correctness_prerequisites",
        "calibration",
        "p1_authority",
        "gate1_audit",
        *contract.EXECUTION_PROVENANCE_FIELDS,
    }
    if set(prerequisites) != required:
        raise ValueError("prerequisite authority schema is invalid")
    loaded = {}
    for name, classification in (
        ("correctness_prerequisites", "PASS"),
        ("calibration", "PASS"),
        ("p1_authority", "GO"),
        ("gate1_audit", "PASS"),
    ):
        row = prerequisites[name]
        if not isinstance(row, dict) or row.get(
            "classification"
        ) != classification:
            raise ValueError(f"{name} prerequisite is not authorized")
        loaded[name] = _load_authoritative_document(row, name)

    correctness_path, correctness_bytes, _ = loaded[
        "correctness_prerequisites"
    ]
    correctness_status = contract.validate_prerequisites(
        correctness_path,
        file_bytes=correctness_bytes,
    )
    if not correctness_status.authorized:
        raise ValueError(
            "correctness prerequisite authority is invalid: "
            + "; ".join(correctness_status.reasons)
        )
    calibration_path, _, calibration = loaded["calibration"]
    p1_path, _, p1_authority = loaded["p1_authority"]
    gate1_audit = loaded["gate1_audit"][2]
    contract.validate_calibration_binding(calibration)
    contract.validate_p1_authority_binding(p1_authority)
    contract.validate_evidence_document("gate1_audit", gate1_audit)
    _validate_binding_reference(
        calibration_path,
        calibration["artifact_path"],
        calibration["artifact_sha256"],
        "calibration artifact",
    )
    _validate_binding_reference(
        p1_path,
        p1_authority["artifact_path"],
        p1_authority["artifact_sha256"],
        "P1 authority artifact",
    )
    _validate_binding_reference(
        p1_path,
        p1_authority["independent_verification_path"],
        p1_authority["independent_verification_sha256"],
        "P1 independent verification",
    )
    if prerequisites["correctness_prerequisites_sha256"] != (
        prerequisites["correctness_prerequisites"]["sha256"]
    ):
        raise ValueError("correctness prerequisite provenance mismatch")
    expected_bindings = {
        "calibration_artifact_sha256": calibration["artifact_sha256"],
        "p1_authority_artifact_sha256": p1_authority["artifact_sha256"],
        "gate1_audit_sha256": gate1_audit["gate1_audit_sha256"],
    }
    for field, expected in expected_bindings.items():
        if prerequisites[field] != expected:
            raise ValueError(f"prerequisite provenance mismatch: {field}")
    for document in (calibration, p1_authority):
        for field in (
            "source_tree_sha256",
            "model_manifest_sha256",
            "workload_manifest_sha256",
        ):
            if document[field] != prerequisites[field]:
                raise ValueError(f"prerequisite binding mismatch: {field}")
    if (
        gate1_audit["source_tree_sha256"]
        != prerequisites["source_tree_sha256"]
    ):
        raise ValueError("prerequisite binding mismatch: source_tree_sha256")
    try:
        contract._validate_execution_provenance(prerequisites)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "prerequisite execution provenance is invalid"
        ) from error


def _base_result(run_tag, nonce, prerequisites):
    return {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS["preflight"],
        "classification": None,
        "run_tag": run_tag,
        "nonce": nonce,
        **{
            field: prerequisites.get(field)
            for field in contract.EXECUTION_PROVENANCE_FIELDS
        },
        "required_gpu_indices": list(REQUIRED_GPU_INDICES),
        "world_size": contract.WORLD_SIZE,
        "minimum_free_bytes_per_gpu": MIN_GPU_FREE_BYTES,
        "gpu_query_rows": [],
        "blocking_reasons": [],
        "worker_authorized": False,
        "remote_path_created": False,
        "source_staged": False,
        "worker_launched": False,
    }


def run_preflight(
    *,
    run_tag,
    nonce,
    prerequisites,
    gpu_query,
    source_bundle_builder,
    source_bundle_output_dir=None,
    remote_path_creator=None,
    process_launcher=None,
):
    del remote_path_creator, process_launcher
    _validate_authoritative_prerequisites(prerequisites)
    result = _base_result(run_tag, nonce, prerequisites)

    rows = gpu_query()
    result["gpu_query_rows"] = rows
    resource_reasons = []
    if (
        not isinstance(rows, list)
        or [row.get("gpu_index") for row in rows]
        != list(REQUIRED_GPU_INDICES)
    ):
        resource_reasons.append("GPU identity is invalid")
    else:
        for row in rows:
            if (
                type(row.get("free_bytes")) is not int
                or row["free_bytes"] < MIN_GPU_FREE_BYTES
                or row.get("compute_processes") != []
            ):
                resource_reasons.append(
                    f"GPU {row.get('gpu_index')} is unavailable"
                )
    if resource_reasons:
        result.update(
            classification="BLOCKED_RESOURCES",
            blocking_reasons=resource_reasons,
        )
        contract.validate_evidence_document("preflight", result)
        return result

    source_bundle = source_bundle_builder(
        output_dir=source_bundle_output_dir,
        run_tag=run_tag,
        nonce=nonce,
        source_tree_sha256=prerequisites["source_tree_sha256"],
    )
    try:
        contract.validate_evidence_document(
            "source_bundle",
            source_bundle,
        )
        contract._validate_source_inventory(
            source_bundle["inventory"],
            "source bundle inventory",
        )
        if source_bundle["inventory_sha256"] != (
            contract.canonical_json_sha256(source_bundle["inventory"])
        ):
            raise ValueError("source bundle inventory hash is invalid")
        for field in (
            "run_tag",
            "nonce",
            *contract.EXECUTION_PROVENANCE_FIELDS,
        ):
            expected = (
                run_tag
                if field == "run_tag"
                else nonce
                if field == "nonce"
                else prerequisites[field]
            )
            if source_bundle[field] != expected:
                raise ValueError(f"source bundle binding mismatch: {field}")
        if source_bundle["sha256"] != prerequisites[
            "source_bundle_sha256"
        ]:
            raise ValueError("source bundle sha256 binding mismatch")
        _validate_source_bundle_bytes(
            source_bundle,
            source_bundle_output_dir,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source bundle authority is invalid") from error
    result.update(
        classification="READY",
        worker_authorized=True,
    )
    contract.validate_evidence_document("preflight", result)
    return result
