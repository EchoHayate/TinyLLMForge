from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "qwen35_tp4_hybrid_prefix_benchmark_contract_for_root_receipt",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
resource_policy_module = _load_module(
    "qwen35_tp4_correctness_resource_policy_for_root_receipt",
    "qwen35_tp4_correctness_resource_policy.py",
)


SCHEMA_VERSION = (
    "qwen35.tp4-root-logit-remote-execution-receipt.v1"
)
MAX_ERROR_BYTES = 64 * 1024
MIN_GPU_FREE_BYTES = 24 * 1024**3


def _canonical_bytes(payload):
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not canonical JSON") from error


def _canonical_sha(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def _default_root_verifier(path):
    verifier = _load_module(
        "verify_qwen35_tp4_real_root_logit_correctness_gate_for_receipt",
        "verify_qwen35_tp4_real_root_logit_correctness_gate.py",
    )
    return verifier.verify_run(path)


def _validate_plan(plan):
    required = {
        "run_tag",
        "local_run_dir",
        "remote_run_dir",
        "frozen_source_tree_sha256",
        "model_manifest_sha256",
        "exact_artifact_names",
        "minimum_free_bytes_per_gpu",
        "requires_no_active_compute_processes",
        "stage_order",
    }
    resource_policy = plan.get(
        "resource_policy",
        resource_policy_module.STRICT_EXCLUSIVE,
    )
    if (
        not isinstance(plan, dict)
        or not required.issubset(plan)
        or plan["stage_order"]
        != ["preflight", "run", "download", "verify"]
        or plan["minimum_free_bytes_per_gpu"]
        != MIN_GPU_FREE_BYTES
        or plan["requires_no_active_compute_processes"]
        is not (
            resource_policy
            == resource_policy_module.STRICT_EXCLUSIVE
        )
    ):
        raise ValueError("root execution receipt plan is invalid")
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        baseline = resource_policy_module.validate_baseline_manifest(
            plan.get("resource_baseline_path", ""),
            ssh_target=plan.get("ssh_target"),
            gpu_indices=plan.get("gpu_indices"),
        )
        if (
            resource_policy_module.sha256(
                plan["resource_baseline_path"]
            )
            != plan.get("resource_baseline_sha256")
            or [row["gpu_uuid"] for row in baseline["selected"]]
            != plan.get("gpu_uuids")
            or plan.get("benchmark_execution_authorized") is not False
        ):
            raise ValueError(
                "root execution receipt resource binding is invalid"
            )
    elif resource_policy != resource_policy_module.STRICT_EXCLUSIVE:
        raise ValueError("root execution receipt resource policy invalid")
    return plan


def _validate_authorization(plan, payload, authorization_record):
    if (
        not isinstance(authorization_record, dict)
        or authorization_record.get("consumed") is not True
        or payload["authorization_sha256"]
        != _canonical_sha(authorization_record)
        or payload["authorization_nonce"]
        != authorization_record.get("nonce")
        or authorization_record.get("plan_sha256")
        != payload["plan_sha256"]
        or authorization_record.get("run_tag") != plan["run_tag"]
        or authorization_record.get("frozen_source_tree_sha256")
        != plan["frozen_source_tree_sha256"]
        or authorization_record.get("model_manifest_sha256")
        != plan["model_manifest_sha256"]
        or authorization_record.get("stage_order")
        != plan["stage_order"]
        or (
            plan.get("resource_policy") is not None
            and (
                authorization_record.get("resource_policy")
                != plan.get("resource_policy")
                or authorization_record.get(
                    "resource_baseline_sha256"
                )
                != plan.get("resource_baseline_sha256")
            )
        )
    ):
        raise ValueError("root execution receipt authorization mismatch")


def _validate_stage_rows(plan, stage_results):
    if (
        not isinstance(stage_results, list)
        or [row.get("name") for row in stage_results]
        != plan["stage_order"]
    ):
        raise ValueError("root execution receipt stage order mismatch")
    for row in stage_results:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "result_sha256", "result"}
            or row["result_sha256"] != _canonical_sha(row["result"])
            or not isinstance(row["result"], dict)
        ):
            raise ValueError("root execution receipt stage mismatch")
    return {
        row["name"]: row["result"] for row in stage_results
    }


def _validate_preflight(plan, payload):
    selected = payload.get("selected")
    if (
        payload.get("status") != "READY"
        or payload.get("run_tag") != plan["run_tag"]
        or payload.get("frozen_source_tree_sha256")
        != plan["frozen_source_tree_sha256"]
        or payload.get("source_tree_sha256")
        != plan["frozen_source_tree_sha256"]
        or not isinstance(selected, list)
        or len(selected) != 4
        or [row.get("rank") for row in selected] != [0, 1, 2, 3]
    ):
        raise ValueError("root preflight evidence is invalid")
    if (
        len({row.get("gpu_index") for row in selected}) != 4
        or len({row.get("gpu_uuid") for row in selected}) != 4
    ):
        raise ValueError("root preflight GPU identity is invalid")
    if any(
            row.get("world_size", 4) != 4
            or isinstance(row.get("free_bytes"), bool)
            or not isinstance(row.get("free_bytes"), int)
            or row["free_bytes"] < MIN_GPU_FREE_BYTES
            for row in selected
    ):
        raise ValueError("root preflight GPU resource is invalid")
    resource_policy = plan.get(
        "resource_policy",
        resource_policy_module.STRICT_EXCLUSIVE,
    )
    if resource_policy == resource_policy_module.STRICT_EXCLUSIVE:
        guard_payload = {
            "classification": "READY",
            "selected": [
                {
                    key: row[key]
                    for key in (
                        "gpu_index",
                        "gpu_uuid",
                        "free_bytes",
                        "compute_processes",
                    )
                }
                for row in selected
            ],
        }
        baseline = None
        baseline_sha256 = None
    else:
        guard_payload = {
            "classification": payload.get("status"),
            "resource_policy": payload.get("resource_policy"),
            "baseline_sha256": payload.get("baseline_sha256"),
            "benchmark_execution_authorized": payload.get(
                "benchmark_execution_authorized"
            ),
            "selected": [
                {
                    key: row[key]
                    for key in (
                        "gpu_index",
                        "gpu_uuid",
                        "free_bytes",
                        "compute_processes",
                    )
                }
                for row in selected
            ],
        }
        baseline = resource_policy_module.validate_baseline_manifest(
            plan["resource_baseline_path"],
            ssh_target=plan["ssh_target"],
            gpu_indices=plan["gpu_indices"],
        )
        baseline_sha256 = plan["resource_baseline_sha256"]
    try:
        resource_policy_module.validate_guard_payload(
            resource_policy,
            guard_payload,
            gpu_indices=[
                row["gpu_index"] for row in selected
            ],
            baseline=baseline,
            baseline_sha256=baseline_sha256,
        )
    except ValueError as error:
        raise ValueError(
            f"root preflight resource process is invalid: {error}"
        ) from error


def _validate_run(plan, payload, preflight):
    resource_policy = plan.get(
        "resource_policy",
        resource_policy_module.STRICT_EXCLUSIVE,
    )
    required = {
        "status",
        "run_tag",
        "remote_run_dir",
        "artifact_names",
    }
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        required.add("final_resource")
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or
        payload.get("status") != "REMOTE_PASS"
        or payload.get("run_tag") != plan["run_tag"]
        or payload.get("remote_run_dir") != plan["remote_run_dir"]
        or payload.get("artifact_names")
        != plan["exact_artifact_names"]
    ):
        raise ValueError("root run inventory is invalid")
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        baseline = resource_policy_module.validate_baseline_manifest(
            plan["resource_baseline_path"],
            ssh_target=plan["ssh_target"],
            gpu_indices=plan["gpu_indices"],
        )
        try:
            resource_policy_module.validate_guard_payload(
                resource_policy,
                payload["final_resource"],
                gpu_indices=plan["gpu_indices"],
                baseline=baseline,
                baseline_sha256=plan["resource_baseline_sha256"],
            )
        except ValueError as error:
            raise ValueError(
                f"root final resource process is invalid: {error}"
            ) from error
        if [
            (row["gpu_index"], row["gpu_uuid"])
            for row in payload["final_resource"]["selected"]
        ] != [
            (row["gpu_index"], row["gpu_uuid"])
            for row in preflight["selected"]
        ]:
            raise ValueError("root final resource GPU drift")


def _artifact_inventory(plan):
    artifact_dir = Path(plan["local_run_dir"]) / "artifacts"
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise ValueError("root artifact inventory is invalid")
    paths = list(artifact_dir.iterdir())
    if (
        sorted(path.name for path in paths)
        != plan["exact_artifact_names"]
        or any(not path.is_file() or path.is_symlink() for path in paths)
    ):
        raise ValueError("root artifact inventory is invalid")
    return artifact_dir


def _validate_download(plan, payload):
    if (
        payload.get("status") != "DOWNLOADED"
        or payload.get("artifact_names")
        != plan["exact_artifact_names"]
    ):
        raise ValueError("root download inventory is invalid")
    return _artifact_inventory(plan)


def _validate_disk_evidence(plan, by_name):
    local_run = Path(plan["local_run_dir"])
    paths = {
        "preflight": local_run / "remote_resource_preflight.json",
        "run": local_run / "remote_run.json",
        "download": local_run / "download.json",
        "verify": local_run / "independent_verification.json",
    }
    for name, path in paths.items():
        if _load_json(path, f"root {name} disk evidence") != by_name[name]:
            raise ValueError("root stage disk evidence mismatch")


def _validate_verification(plan, artifact_dir, payload, root_verifier):
    expected = {
        "classification": "PASS",
        "case_ids": list(contract.TP4_ROOT_CASE_IDS),
        "ranks": [0, 1, 2, 3],
    }
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"classification", "case_ids", "ranks", "checks"}
        or any(payload.get(name) != value for name, value in expected.items())
        or isinstance(payload.get("checks"), bool)
        or not isinstance(payload.get("checks"), int)
        or payload["checks"] <= 0
    ):
        raise ValueError("root verification evidence is invalid")
    verified = root_verifier(artifact_dir)
    if (
        not isinstance(verified, dict)
        or set(verified)
        != {"classification", "case_ids", "ranks", "checks"}
        or any(
            verified.get(name) != payload[name]
            for name in ("classification", "case_ids", "ranks")
        )
        or isinstance(verified.get("checks"), bool)
        or not isinstance(verified.get("checks"), int)
        or verified["checks"] <= 0
    ):
        raise ValueError("root verifier payload mismatch")
    artifact = _load_json(
        artifact_dir / "tp4_real_root_logit_correctness.json",
        "root correctness artifact",
    )
    manifest = _load_json(
        artifact_dir / "source_manifest.json",
        "root source manifest",
    )
    if (
        manifest.get("source_tree_sha256")
        != plan["frozen_source_tree_sha256"]
        or manifest.get("model_manifest_sha256")
        != plan["model_manifest_sha256"]
    ):
        raise ValueError("root source identity mismatch")
    contract.validate_authority_documents(
        "tp4_root_logit",
        artifact,
        payload,
        plan["frozen_source_tree_sha256"],
    )
    return payload


def validate_execution_receipt(
    plan,
    payload,
    *,
    authorization_record,
    root_verifier=None,
):
    plan = _validate_plan(plan)
    required = {
        "schema_version",
        "classification",
        "plan_sha256",
        "authorization_sha256",
        "authorization_nonce",
        "run_tag",
        "stages",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["classification"] != "PASS"
        or payload["plan_sha256"] != _canonical_sha(plan)
        or payload["run_tag"] != plan["run_tag"]
    ):
        raise ValueError("root execution receipt schema mismatch")
    _validate_authorization(plan, payload, authorization_record)
    by_name = _validate_stage_rows(plan, payload["stages"])
    _validate_preflight(plan, by_name["preflight"])
    _validate_run(plan, by_name["run"], by_name["preflight"])
    artifact_dir = _validate_download(plan, by_name["download"])
    _validate_disk_evidence(plan, by_name)
    verification = _validate_verification(
        plan,
        artifact_dir,
        by_name["verify"],
        root_verifier or _default_root_verifier,
    )
    return {
        "classification": "PASS",
        "run_tag": plan["run_tag"],
        "plan_sha256": payload["plan_sha256"],
        "authorization_sha256": payload["authorization_sha256"],
        "authorization_nonce": payload["authorization_nonce"],
        "source_tree_sha256": plan["frozen_source_tree_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "case_ids": verification["case_ids"],
        "ranks": verification["ranks"],
        "checks": verification["checks"],
        "artifact_names": plan["exact_artifact_names"],
        "stage_count": len(plan["stage_order"]),
    }


def _atomic_write(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("execution receipt output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def produce_execution_receipt(
    *,
    plan,
    stage_results,
    output_path,
    authorization_record,
    root_verifier=None,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "PASS",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(
            authorization_record
        ),
        "authorization_nonce": authorization_record.get("nonce"),
        "run_tag": plan.get("run_tag"),
        "stages": stage_results,
    }
    summary = validate_execution_receipt(
        plan,
        payload,
        authorization_record=authorization_record,
        root_verifier=root_verifier,
    )
    _atomic_write(output_path, payload)
    return summary


def verify_receipt_files(
    *,
    plan_path,
    receipt_path,
    authorization_path,
    plan_verifier,
    root_verifier=None,
):
    if not callable(plan_verifier):
        raise ValueError("explicit plan verifier is required")
    plan = plan_verifier(plan_path)
    payload = _load_json(receipt_path, "root execution receipt")
    authorization_record = _load_json(
        authorization_path,
        "root consumed authorization",
    )
    return validate_execution_receipt(
        plan,
        payload,
        authorization_record=authorization_record,
        root_verifier=root_verifier,
    )
