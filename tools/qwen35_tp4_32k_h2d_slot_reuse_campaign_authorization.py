from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _load_campaign():
    name = "qwen35_tp4_32k_h2d_slot_reuse_campaign"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_32k_h2d_slot_reuse_campaign.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load focused H2D campaign")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign = _load_campaign()
SCHEMA = "qwen35.tp4-32k-h2d-campaign-authorization.v1"
AUTHORIZATION_FIELDS = frozenset({
    "schema",
    "classification",
    "authorization_text",
    "plan_sha256",
    "run_tag",
    "source_tree_sha256",
    "source_tar_sha256",
    "checkpoint_manifest_sha256",
    "cells",
    "repetitions_per_cell",
    "gpu_indices",
    "ports",
    "ssh_target",
    "remote_run",
    "nonce",
    "consumed",
})


def _safe_nonce(value: object) -> str:
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


def _atomic_write_new(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("authorization output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(campaign.canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_authorization(
    *,
    plan: dict,
    authorization_text: str,
    nonce: str,
) -> dict:
    plan = campaign.validate_campaign_plan(plan)
    if authorization_text != campaign.AUTHORIZATION_TEXT:
        raise ValueError("exact authorization text is required")
    return {
        "schema": SCHEMA,
        "classification": "AUTHORIZED",
        "authorization_text": authorization_text,
        "plan_sha256": campaign.canonical_sha256(plan),
        "run_tag": plan["run_tag"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "source_tar_sha256": plan["source_tar_sha256"],
        "checkpoint_manifest_sha256": plan[
            "checkpoint_manifest_sha256"
        ],
        "cells": list(plan["cells"]),
        "repetitions_per_cell": plan["repetitions_per_cell"],
        "gpu_indices": list(plan["gpu_indices"]),
        "ports": dict(plan["ports"]),
        "ssh_target": plan["ssh_target"],
        "remote_run": plan["remote_run"],
        "nonce": _safe_nonce(nonce),
        "consumed": False,
    }


def validate_authorization(plan: dict, value: object) -> dict:
    plan = campaign.validate_campaign_plan(plan)
    if (
        not isinstance(value, dict)
        or set(value) != AUTHORIZATION_FIELDS
    ):
        raise ValueError("execution authorization mismatch")
    payload = dict(value)
    expected = build_authorization(
        plan=plan,
        authorization_text=campaign.AUTHORIZATION_TEXT,
        nonce=payload.get("nonce"),
    )
    if payload != expected:
        raise ValueError("execution authorization mismatch")
    return payload


def produce_authorization(
    *,
    plan: dict,
    authorization_text: str,
    nonce: str,
    output_path: str | Path,
) -> dict:
    payload = build_authorization(
        plan=plan,
        authorization_text=authorization_text,
        nonce=nonce,
    )
    validate_authorization(plan, payload)
    _atomic_write_new(Path(output_path), payload)
    return payload


def _load_active_authorization(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("authorization is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authorization is invalid") from error


def consume_authorization(
    *,
    plan: dict,
    authorization_path: str | Path,
    consumed_path: str | Path,
) -> dict:
    active = Path(authorization_path)
    consumed = Path(consumed_path)
    payload = validate_authorization(
        plan,
        _load_active_authorization(active),
    )
    if consumed.exists() or consumed.is_symlink():
        raise ValueError("consumed authorization already exists")
    consumed.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=active.parent,
        prefix=f".{active.name}.claim.",
        delete=False,
    ) as handle:
        claim = Path(handle.name)
    claim.unlink()
    os.replace(active, claim)
    result = {**payload, "consumed": True}
    try:
        _atomic_write_new(consumed, result)
    except BaseException:
        os.replace(claim, active)
        raise
    claim.unlink()
    return result
