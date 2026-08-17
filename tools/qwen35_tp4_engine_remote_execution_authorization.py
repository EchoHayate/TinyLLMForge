from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _load_receipt():
    module_name = "qwen35_tp4_engine_remote_execution_receipt"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_engine_remote_execution_receipt.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


receipt = _load_receipt()
SCHEMA_VERSION = (
    "qwen35.tp4-engine-remote-execution-authorization.v1"
)


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


def _require_sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _payload(plan, nonce):
    local_inputs = plan.get("local_inputs")
    if not isinstance(local_inputs, dict):
        raise ValueError("plan local inputs are missing")
    model_sha = plan.get("model_manifest_sha256")
    if model_sha is None:
        model_sha = local_inputs.get("model_manifest_sha256")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "AUTHORIZED",
        "plan_sha256": receipt._canonical_sha(plan),
        "run_tag": plan.get("run_tag"),
        "source_tree_sha256": plan.get("source_tree_sha256"),
        "model_manifest_sha256": model_sha,
        "workload_manifest_sha256": local_inputs.get(
            "workload_manifest_sha256"
        ),
        "gpu_indices": plan.get("gpu_indices"),
        "ports": plan.get("ports"),
        "nonce": _safe_nonce(nonce),
        "consumed": False,
    }
    resource_policy = plan.get("resource_policy")
    if resource_policy is not None:
        baseline_sha256 = plan.get("resource_baseline_sha256")
        if (
            resource_policy != "controlled_shared"
            or local_inputs.get("resource_baseline_sha256")
            != baseline_sha256
        ):
            raise ValueError("plan resource binding is invalid")
        payload.update({
            "resource_policy": resource_policy,
            "resource_baseline_sha256": _require_sha(
                baseline_sha256,
                "resource baseline SHA",
            ),
        })
    return payload


def validate_authorization(plan, payload):
    expected = set(_payload(plan, "validation-nonce"))
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["classification"] != "AUTHORIZED"
        or payload["consumed"] is not False
        or payload["plan_sha256"] != receipt._canonical_sha(plan)
        or payload["run_tag"] != plan.get("run_tag")
        or payload["source_tree_sha256"]
        != plan.get("source_tree_sha256")
        or payload["model_manifest_sha256"]
        != plan.get("model_manifest_sha256")
        or payload["gpu_indices"] != plan.get("gpu_indices")
        or payload["ports"] != plan.get("ports")
        or payload["workload_manifest_sha256"]
        != plan["local_inputs"]["workload_manifest_sha256"]
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
        raise ValueError("execution authorization mismatch")
    _safe_nonce(payload["nonce"])
    _require_sha(payload["plan_sha256"], "plan SHA")
    _require_sha(payload["source_tree_sha256"], "source tree SHA")
    _require_sha(payload["model_manifest_sha256"], "model manifest SHA")
    _require_sha(
        payload["workload_manifest_sha256"],
        "workload manifest SHA",
    )
    if "resource_baseline_sha256" in payload:
        _require_sha(
            payload["resource_baseline_sha256"],
            "resource baseline SHA",
        )
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
        handle.write(receipt._canonical_bytes(payload) + b"\n")
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


def _rewrite_consumed_authorization(consumed_path, payload):
    consumed_path = Path(consumed_path)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=consumed_path.parent,
        prefix=f".{consumed_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(receipt._canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, consumed_path)
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
        _rewrite_consumed_authorization(consumed_path, consumed)
    except BaseException:
        os.replace(claim_path, authorization_path)
        raise
    claim_path.unlink()
    return consumed
