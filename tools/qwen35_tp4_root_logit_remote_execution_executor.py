from __future__ import annotations

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


authorization = _load_module(
    "qwen35_tp4_root_logit_remote_execution_authorization_for_executor",
    "qwen35_tp4_root_logit_remote_execution_authorization.py",
)
receipt = _load_module(
    "qwen35_tp4_root_logit_remote_execution_receipt_for_executor",
    "qwen35_tp4_root_logit_remote_execution_receipt.py",
)


REQUIRED_EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}
FAILURE_SCHEMA_VERSION = (
    "qwen35.tp4-root-logit-remote-execution-failure.v1"
)
MAX_ERROR_CHARS = 4096


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_failure_evidence(
    plan,
    payload,
    *,
    authorization_record,
):
    required = {
        "schema_version",
        "classification",
        "plan_sha256",
        "authorization_sha256",
        "authorization_nonce",
        "run_tag",
        "failed_stage",
        "completed_stages",
        "error",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != FAILURE_SCHEMA_VERSION
        or payload["classification"] != "FAILED"
        or payload["plan_sha256"] != receipt._canonical_sha(plan)
        or payload["authorization_sha256"]
        != receipt._canonical_sha(authorization_record)
        or payload["authorization_nonce"]
        != authorization_record.get("nonce")
        or payload["run_tag"] != plan.get("run_tag")
        or not isinstance(payload["completed_stages"], list)
        or not isinstance(payload["error"], str)
        or not payload["error"]
        or len(payload["error"]) > MAX_ERROR_CHARS
    ):
        raise ValueError("root failure evidence schema mismatch")
    completed = payload["completed_stages"]
    order = plan.get("stage_order")
    if (
        [row.get("name") for row in completed]
        != order[:len(completed)]
        or len(completed) >= len(order)
        or payload["failed_stage"] != order[len(completed)]
    ):
        raise ValueError("root failure evidence is not a stage prefix")
    for row in completed:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "result_sha256", "result"}
            or row["result_sha256"]
            != receipt._canonical_sha(row["result"])
        ):
            raise ValueError("root failure stage evidence is invalid")
    return {
        "classification": "FAILED",
        "run_tag": payload["run_tag"],
        "failed_stage": payload["failed_stage"],
        "completed_stage_count": len(completed),
    }


def _write_failure(
    *,
    plan,
    failure_path,
    failed_stage,
    completed_stages,
    error,
    authorization_record,
):
    payload = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "classification": "FAILED",
        "plan_sha256": receipt._canonical_sha(plan),
        "authorization_sha256": receipt._canonical_sha(
            authorization_record
        ),
        "authorization_nonce": authorization_record.get("nonce"),
        "run_tag": plan.get("run_tag"),
        "failed_stage": failed_stage,
        "completed_stages": completed_stages,
        "error": str(error)[-MAX_ERROR_CHARS:],
    }
    validate_failure_evidence(
        plan,
        payload,
        authorization_record=authorization_record,
    )
    _atomic_write(failure_path, payload)


def execute_plan(
    *,
    plan,
    receipt_path,
    failure_path,
    stage_runner,
    authorization_record,
    execution_env,
    root_verifier=None,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError(
            "exact KRB5CCNAME execution environment is required"
        )
    if not callable(stage_runner):
        raise ValueError("explicit stage runner is required")
    completed = []
    for name in plan["stage_order"]:
        try:
            result = stage_runner(
                name=name,
                plan=plan,
                execution_env=dict(execution_env),
            )
            if not isinstance(result, dict):
                raise ValueError("root stage result is invalid")
        except BaseException as error:
            _write_failure(
                plan=plan,
                failure_path=failure_path,
                failed_stage=name,
                completed_stages=completed,
                error=error,
                authorization_record=authorization_record,
            )
            raise
        completed.append({
            "name": name,
            "result_sha256": receipt._canonical_sha(result),
            "result": result,
        })
    return receipt.produce_execution_receipt(
        plan=plan,
        stage_results=completed,
        output_path=receipt_path,
        authorization_record=authorization_record,
        root_verifier=root_verifier,
    )


def execute_verified_plan_file(
    *,
    plan_path,
    authorization_path,
    consumed_authorization_path,
    receipt_path,
    failure_path,
    stage_runner,
    plan_verifier,
    execution_env,
    root_verifier=None,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError(
            "exact KRB5CCNAME execution environment is required"
        )
    if not callable(plan_verifier):
        raise ValueError("explicit plan verifier is required")
    if not callable(stage_runner):
        raise ValueError("explicit stage runner is required")
    plan = plan_verifier(plan_path)
    targets = (
        Path(receipt_path),
        Path(failure_path),
        Path(consumed_authorization_path),
        Path(plan["local_run_dir"]),
    )
    if any(path.exists() for path in targets):
        raise ValueError("root execution output target exists")
    authorization_record = authorization.consume_authorization(
        plan=plan,
        authorization_path=authorization_path,
        consumed_path=consumed_authorization_path,
    )
    return execute_plan(
        plan=plan,
        receipt_path=receipt_path,
        failure_path=failure_path,
        stage_runner=stage_runner,
        authorization_record=authorization_record,
        execution_env=execution_env,
        root_verifier=root_verifier,
    )
