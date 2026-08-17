from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-correctness-authority-campaign-receipt.v1"
)
CHILD_ORDER = (
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
)
CHILD_STAGE_NAMES = {
    "tp4_root_logit": "root_logit",
    "cached_continuation": "cached_continuation",
    "engine_correctness": "engine_correctness",
}
STAGE_ORDER = (
    "root_logit",
    "cached_continuation",
    "engine_correctness",
    "adapt_authorities",
    "build_bundle",
    "verify_bundle",
)


def _canonical_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular_file(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve()


def _directory(path, label):
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    return path.resolve()


def _validate_child(plan_row, stage, verifier):
    result = stage.get("result")
    expected = {
        "classification",
        "name",
        "run_tag",
        "plan_path",
        "plan_sha256",
        "source_tree_sha256",
        "model_manifest_sha256",
        "authority_dir",
        "authorization_path",
        "consumed_authorization_path",
        "authorization_sha256",
        "receipt_path",
        "receipt_sha256",
        "failure_path",
    }
    if plan_row.get("resource_policy") is not None:
        expected.update({
            "resource_policy",
            "resource_baseline_sha256",
        })
    if (
        not isinstance(result, dict)
        or set(result) != expected
        or result["classification"] != "PASS"
        or {
            key: result[key]
            for key in set(plan_row)
        } != plan_row
    ):
        raise ValueError(f"{plan_row['name']} campaign result mismatch")
    plan_path = _regular_file(
        result["plan_path"],
        f"{plan_row['name']} plan",
    )
    authorization_path = _regular_file(
        result["consumed_authorization_path"],
        f"{plan_row['name']} consumed authorization",
    )
    receipt_path = _regular_file(
        result["receipt_path"],
        f"{plan_row['name']} receipt",
    )
    _directory(
        result["authority_dir"],
        f"{plan_row['name']} authority directory",
    )
    if _sha256(plan_path) != result["plan_sha256"]:
        raise ValueError(f"{plan_row['name']} plan SHA mismatch")
    if _sha256(authorization_path) != result["authorization_sha256"]:
        raise ValueError(
            f"{plan_row['name']} authorization SHA mismatch"
        )
    if _sha256(receipt_path) != result["receipt_sha256"]:
        raise ValueError(f"{plan_row['name']} receipt SHA mismatch")
    if not callable(verifier):
        raise ValueError(f"{plan_row['name']} receipt verifier is required")
    summary = verifier(
        plan_path=plan_path,
        authorization_path=authorization_path,
        receipt_path=receipt_path,
    )
    if (
        not isinstance(summary, dict)
        or summary.get("classification") != "PASS"
    ):
        raise ValueError(f"{plan_row['name']} receipt did not prove PASS")


def _validate_adapter(stage):
    result = stage.get("result")
    if (
        not isinstance(result, dict)
        or set(result) != {"classification", "authorities"}
        or result["classification"] != "PASS"
        or not isinstance(result["authorities"], list)
        or [row.get("name") for row in result["authorities"]]
        != list(CHILD_ORDER)
    ):
        raise ValueError("campaign adapter result mismatch")
    for row in result["authorities"]:
        if set(row) != {
            "name",
            "run_tag",
            "source_tree_sha256",
            "artifact_path",
            "artifact_sha256",
            "independent_verification_path",
            "independent_verification_sha256",
            "provenance_path",
            "provenance_sha256",
        }:
            raise ValueError("campaign adapter result mismatch")
        for path_field, sha_field, label in (
            ("artifact_path", "artifact_sha256", "artifact"),
            (
                "independent_verification_path",
                "independent_verification_sha256",
                "independent verification",
            ),
            ("provenance_path", "provenance_sha256", "provenance"),
        ):
            path = _regular_file(
                row[path_field],
                f"{row['name']} {label}",
            )
            if _sha256(path) != row[sha_field]:
                raise ValueError(f"{row['name']} {label} SHA mismatch")


def _owned_files(bundle_root):
    return sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _validate_bundle(plan, build_stage, verify_stage, validator):
    build = build_stage.get("result")
    verify = verify_stage.get("result")
    if (
        not isinstance(build, dict)
        or set(build) != {
            "classification",
            "prerequisite_path",
            "prerequisite_sha256",
            "owned_files",
        }
        or build["classification"] != "PASS"
        or build["prerequisite_path"] != plan["prerequisite_path"]
        or not isinstance(build["owned_files"], list)
        or build["owned_files"] != sorted(set(build["owned_files"]))
    ):
        raise ValueError("campaign bundle result mismatch")
    prerequisite = _regular_file(
        build["prerequisite_path"],
        "campaign prerequisite",
    )
    if _sha256(prerequisite) != build["prerequisite_sha256"]:
        raise ValueError("campaign prerequisite SHA mismatch")
    bundle_root = _directory(
        plan["bundle_output_dir"],
        "campaign bundle",
    )
    if _owned_files(bundle_root) != build["owned_files"]:
        raise ValueError("campaign bundle inventory mismatch")
    if not callable(validator):
        raise ValueError("prerequisite validator is required")
    validation = validator(prerequisite)
    if (
        not isinstance(validation, dict)
        or validation.get("classification") != "PASS"
        or validation.get("authorized") is not True
    ):
        raise ValueError("campaign prerequisite is not authorized")
    if (
        not isinstance(verify, dict)
        or verify.get("classification") != "PASS"
        or verify.get("authorized") is not True
        or verify.get("prerequisite_sha256")
        != build["prerequisite_sha256"]
    ):
        raise ValueError("campaign bundle verification mismatch")


def validate_campaign_receipt(
    plan,
    payload,
    *,
    authorization_record,
    child_receipt_verifiers,
    prerequisite_validator,
):
    required = {
        "schema_version",
        "classification",
        "plan_sha256",
        "authorization_sha256",
        "authorization_nonce",
        "campaign_tag",
        "stages",
        "benchmark_execution_authorized",
    }
    if (
        not isinstance(plan, dict)
        or not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["classification"] != "PASS"
        or payload["plan_sha256"] != _canonical_sha(plan)
        or not isinstance(authorization_record, dict)
        or authorization_record.get("consumed") is not True
        or authorization_record.get("plan_sha256")
        != payload["plan_sha256"]
        or payload["authorization_sha256"]
        != _canonical_sha(authorization_record)
        or payload["authorization_nonce"]
        != authorization_record.get("nonce")
        or payload["campaign_tag"] != plan.get("campaign_tag")
        or payload["benchmark_execution_authorized"] is not False
        or plan.get("benchmark_execution_authorized") is not False
        or (
            plan.get("resource_policy") is not None
            and (
                plan.get("resource_policy") != "controlled_shared"
                or authorization_record.get("resource_policy")
                != plan.get("resource_policy")
                or authorization_record.get(
                    "resource_baseline_sha256"
                )
                != plan.get("resource_baseline_sha256")
                or authorization_record.get(
                    "benchmark_execution_authorized"
                )
                is not False
            )
        )
    ):
        raise ValueError("campaign receipt schema mismatch")
    stages = payload["stages"]
    if (
        not isinstance(stages, list)
        or [row.get("name") for row in stages] != list(STAGE_ORDER)
        or [row.get("name") for row in stages]
        != plan.get("stage_order")
    ):
        raise ValueError("campaign receipt stage order mismatch")
    for row in stages:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "result_sha256", "result"}
            or row["result_sha256"] != _canonical_sha(row["result"])
        ):
            raise ValueError("campaign receipt stage evidence mismatch")
    children = plan.get("children")
    if (
        not isinstance(children, list)
        or [row.get("name") for row in children] != list(CHILD_ORDER)
        or set(child_receipt_verifiers) != set(CHILD_ORDER)
    ):
        raise ValueError("campaign child inventory mismatch")
    for index, child in enumerate(children):
        _validate_child(
            child,
            stages[index],
            child_receipt_verifiers[child["name"]],
        )
    _validate_adapter(stages[3])
    _validate_bundle(
        plan,
        stages[4],
        stages[5],
        prerequisite_validator,
    )
    summary = {
        "classification": "PASS",
        "campaign_tag": plan["campaign_tag"],
        "stage_count": len(stages),
        "prerequisite_path": plan["prerequisite_path"],
        "benchmark_execution_authorized": False,
    }
    if plan.get("resource_policy") is not None:
        summary.update({
            "resource_policy": plan["resource_policy"],
            "resource_baseline_sha256": plan[
                "resource_baseline_sha256"
            ],
        })
    return summary


def _atomic_write(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("campaign receipt output already exists")
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


def produce_campaign_receipt(
    *,
    plan,
    stage_results,
    authorization_record,
    output_path,
    child_receipt_verifiers,
    prerequisite_validator,
):
    stages = [
        {
            "name": row["name"],
            "result_sha256": _canonical_sha(row["result"]),
            "result": row["result"],
        }
        for row in stage_results
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "PASS",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(authorization_record),
        "authorization_nonce": authorization_record.get("nonce"),
        "campaign_tag": plan.get("campaign_tag"),
        "stages": stages,
        "benchmark_execution_authorized": False,
    }
    summary = validate_campaign_receipt(
        plan,
        payload,
        authorization_record=authorization_record,
        child_receipt_verifiers=child_receipt_verifiers,
        prerequisite_validator=prerequisite_validator,
    )
    _atomic_write(output_path, payload)
    return summary
