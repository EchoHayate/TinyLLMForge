from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, filename: str):
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to load focused H2D module: {filename}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign = _load_module(
    "qwen35_tp4_32k_h2d_slot_reuse_campaign",
    "qwen35_tp4_32k_h2d_slot_reuse_campaign.py",
)
authorization = _load_module(
    "qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization",
    "qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization.py",
)
RESULT_SCHEMA = (
    "qwen35.tp4-32k-h2d-authorized-executor-result.v1"
)


def execute_authorized_campaign(
    *,
    plan: dict,
    authorization_path: str | Path,
    consumed_authorization_path: str | Path,
    command_runner,
) -> dict:
    plan = campaign.validate_campaign_plan(plan)
    if not callable(command_runner):
        raise ValueError("explicit command runner is required")
    authorization_record = authorization.consume_authorization(
        plan=plan,
        authorization_path=authorization_path,
        consumed_path=consumed_authorization_path,
    )
    command_result = command_runner(plan, authorization_record)
    if (
        not isinstance(command_result, dict)
        or set(command_result)
        != {"classification", "remote_command_count"}
        or not isinstance(command_result["classification"], str)
        or not command_result["classification"]
        or isinstance(command_result["remote_command_count"], bool)
        or not isinstance(command_result["remote_command_count"], int)
        or command_result["remote_command_count"] < 0
    ):
        raise ValueError("authorized command runner result mismatch")
    return {
        "schema": RESULT_SCHEMA,
        "classification": command_result["classification"],
        "plan_sha256": campaign.canonical_sha256(plan),
        "authorization_nonce": authorization_record["nonce"],
        "remote_command_count": command_result[
            "remote_command_count"
        ],
    }
