from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-correctness-authority-campaign-authorization.v1"
)
_SAFE_NONCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _canonical_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _nonce(value):
    if not isinstance(value, str) or not _SAFE_NONCE.fullmatch(value):
        raise ValueError("authorization nonce is invalid")
    return value


def _child_plan_sha256s(plan):
    rows = plan.get("children")
    order = plan.get("child_order")
    if (
        not isinstance(rows, list)
        or not isinstance(order, list)
        or [row.get("name") for row in rows] != order
    ):
        raise ValueError("campaign plan child inventory is invalid")
    result = {}
    for row in rows:
        digest = row.get("plan_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("campaign child plan SHA is invalid")
        result[row["name"]] = digest
    return result


def _payload(plan, nonce):
    if (
        not isinstance(plan, dict)
        or plan.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("campaign plan is invalid")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": _canonical_sha(plan),
        "campaign_tag": plan.get("campaign_tag"),
        "ssh_target": plan.get("ssh_target"),
        "execution_env": plan.get("execution_env"),
        "child_order": plan.get("child_order"),
        "child_plan_sha256s": _child_plan_sha256s(plan),
        "adapter_output_dir": plan.get("adapter_output_dir"),
        "bundle_output_dir": plan.get("bundle_output_dir"),
        "nonce": _nonce(nonce),
        "consumed": False,
        "benchmark_execution_authorized": False,
    }
    if plan.get("resource_policy") is not None:
        if (
            plan["resource_policy"] != "controlled_shared"
            or not isinstance(
                plan.get("resource_baseline_sha256"),
                str,
            )
            or len(plan["resource_baseline_sha256"]) != 64
        ):
            raise ValueError("campaign resource identity is invalid")
        payload.update({
            "resource_policy": plan["resource_policy"],
            "resource_baseline_sha256": plan[
                "resource_baseline_sha256"
            ],
        })
    return payload


def validate_authorization(plan, payload):
    if not isinstance(payload, dict):
        raise ValueError("campaign authorization is invalid")
    expected = _payload(plan, payload.get("nonce"))
    if payload.get("consumed") is True:
        expected["consumed"] = True
    if payload != expected:
        raise ValueError("campaign authorization plan mismatch")
    return payload


def _atomic_write(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("campaign authorization output already exists")
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
        raise ValueError("campaign authorization is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("campaign authorization is invalid") from error
    return payload


def _rewrite(path, payload):
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
    if payload.get("consumed") is not False:
        raise ValueError("campaign authorization is already consumed")
    if consumed_path.exists():
        raise ValueError("consumed campaign authorization already exists")
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
        _rewrite(consumed_path, consumed)
    except BaseException:
        os.replace(claim_path, authorization_path)
        raise
    claim_path.unlink()
    return consumed
