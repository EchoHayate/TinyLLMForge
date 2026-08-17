from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-benchmark-"
    "remote-execution-authorization.v1"
)
CASE_COUNT = 70


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


def _require_sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _case_port_pairs(plan):
    rows = plan.get("case_commands")
    if not isinstance(rows, list) or len(rows) != CASE_COUNT:
        raise ValueError("authorization requires exactly 70 cases")
    result = []
    case_ids = set()
    ports = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("authorization case command is invalid")
        case_id = row.get("case_id")
        dist_port = row.get("dist_port")
        master_port = row.get("master_port")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
            or isinstance(dist_port, bool)
            or not isinstance(dist_port, int)
            or isinstance(master_port, bool)
            or not isinstance(master_port, int)
            or dist_port <= 0
            or master_port <= 0
            or dist_port == master_port
            or dist_port in ports
            or master_port in ports
        ):
            raise ValueError(
                "authorization case identity or port is invalid"
            )
        case_ids.add(case_id)
        ports.update((dist_port, master_port))
        result.append({
            "case_id": case_id,
            "dist_port": dist_port,
            "master_port": master_port,
        })
    return result


def _payload(plan, nonce):
    worker = plan.get("worker_authorization")
    if not isinstance(worker, dict):
        raise ValueError("plan worker authorization is missing")
    gpu_indices = worker.get("gpu_indices")
    if (
        not isinstance(gpu_indices, list)
        or len(gpu_indices) != 4
        or len(set(gpu_indices)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("plan GPU identity is invalid")
    run_tag = plan.get("run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("plan run tag is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "AUTHORIZED",
        "plan_sha256": _canonical_sha(plan),
        "run_tag": run_tag,
        "prerequisites_sha256": _require_sha(
            worker.get("prerequisites_sha256"),
            "prerequisites SHA",
        ),
        "source_tree_sha256": _require_sha(
            worker.get("source_tree_sha256"),
            "source tree SHA",
        ),
        "model_manifest_sha256": _require_sha(
            worker.get("model_manifest_sha256"),
            "model manifest SHA",
        ),
        "workload_manifest_sha256": _require_sha(
            worker.get("workload_manifest_sha256"),
            "workload manifest SHA",
        ),
        "gpu_indices": list(gpu_indices),
        "case_port_pairs": _case_port_pairs(plan),
        "nonce": _safe_nonce(nonce),
        "consumed": False,
    }


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
        or payload["prerequisites_sha256"]
        != expected["prerequisites_sha256"]
        or payload["source_tree_sha256"]
        != expected["source_tree_sha256"]
        or payload["model_manifest_sha256"]
        != expected["model_manifest_sha256"]
        or payload["workload_manifest_sha256"]
        != expected["workload_manifest_sha256"]
        or payload["gpu_indices"] != expected["gpu_indices"]
        or payload["case_port_pairs"] != expected["case_port_pairs"]
    ):
        raise ValueError("benchmark execution authorization mismatch")
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


def _rewrite_consumed_authorization(consumed_path, payload):
    consumed_path = Path(consumed_path)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=consumed_path.parent,
        prefix=f".{consumed_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_bytes(payload) + b"\n")
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
    if authorization_path.parent != consumed_path.parent:
        raise ValueError(
            "authorization consume paths must share one directory"
        )
    consumed = {**payload, "consumed": True}
    os.replace(authorization_path, consumed_path)
    _rewrite_consumed_authorization(consumed_path, consumed)
    return consumed
