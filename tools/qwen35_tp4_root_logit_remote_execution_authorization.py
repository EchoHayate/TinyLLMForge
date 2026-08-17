from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-root-logit-remote-execution-authorization.v1"
)


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


def _safe_nonce(value):
    if (
        not isinstance(value, str)
        or not value
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in value
        )
    ):
        raise ValueError("authorization nonce is unsafe")
    return value


def _payload(plan, nonce):
    required = {
        "run_tag",
        "ssh_target",
        "frozen_source_tree_sha256",
        "model_manifest_sha256",
        "stage_order",
    }
    if not isinstance(plan, dict) or not required.issubset(plan):
        raise ValueError("execution plan is invalid")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "AUTHORIZED",
        "plan_sha256": _canonical_sha(plan),
        "run_tag": plan["run_tag"],
        "ssh_target": plan["ssh_target"],
        "frozen_source_tree_sha256": (
            plan["frozen_source_tree_sha256"]
        ),
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "stage_order": list(plan["stage_order"]),
        "nonce": _safe_nonce(nonce),
        "consumed": False,
    }
    resource_policy = plan.get("resource_policy")
    if resource_policy is not None:
        baseline_sha256 = plan.get("resource_baseline_sha256")
        if (
            resource_policy != "controlled_shared"
            or not isinstance(baseline_sha256, str)
            or len(baseline_sha256) != 64
        ):
            raise ValueError("root resource binding is invalid")
        payload.update({
            "resource_policy": resource_policy,
            "resource_baseline_sha256": baseline_sha256,
        })
    return payload


def validate_authorization(plan, payload):
    expected = _payload(plan, "validation-nonce")
    if (
        not isinstance(payload, dict)
        or set(payload) != set(expected)
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["classification"] != "AUTHORIZED"
        or payload["consumed"] is not False
        or payload["plan_sha256"] != expected["plan_sha256"]
        or payload["run_tag"] != expected["run_tag"]
        or payload["ssh_target"] != expected["ssh_target"]
        or payload["frozen_source_tree_sha256"]
        != expected["frozen_source_tree_sha256"]
        or payload["model_manifest_sha256"]
        != expected["model_manifest_sha256"]
        or payload["stage_order"] != expected["stage_order"]
        or (
            plan.get("resource_policy") is not None
            and (
                payload.get("resource_policy")
                != plan.get("resource_policy")
                or payload.get("resource_baseline_sha256")
                != plan.get("resource_baseline_sha256")
            )
        )
    ):
        raise ValueError("root execution authorization mismatch")
    _safe_nonce(payload["nonce"])
    return {
        "classification": "AUTHORIZED",
        "run_tag": payload["run_tag"],
        "plan_sha256": payload["plan_sha256"],
        "nonce": payload["nonce"],
    }


def _atomic_write(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("authorization output already exists")
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


def produce_authorization(*, plan, output_path, nonce):
    payload = _payload(plan, nonce)
    validate_authorization(plan, payload)
    _atomic_write(output_path, payload)
    return payload


def _load_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("authorization is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authorization is invalid") from error


def _rewrite_consumed(path, payload):
    path = Path(path)
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


def consume_authorization(
    *,
    plan,
    authorization_path,
    consumed_path,
):
    authorization_path = Path(authorization_path)
    consumed_path = Path(consumed_path)
    payload = _load_json(authorization_path)
    validate_authorization(plan, payload)
    if consumed_path.exists():
        raise ValueError("consumed authorization already exists")
    consumed = {**payload, "consumed": True}
    consumed_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=authorization_path.parent,
        prefix=f".{authorization_path.name}.claim.",
        delete=False,
    ) as handle:
        claim_path = Path(handle.name)
    claim_path.unlink()
    os.replace(authorization_path, claim_path)
    try:
        _rewrite_consumed(consumed_path, consumed)
    except BaseException:
        os.replace(claim_path, authorization_path)
        raise
    claim_path.unlink()
    return consumed
