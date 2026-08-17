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
    "qwen35_tp4_correctness_authority_campaign_authorization_for_executor",
    "qwen35_tp4_correctness_authority_campaign_authorization.py",
)
receipt = _load_module(
    "qwen35_tp4_correctness_authority_campaign_receipt_for_executor",
    "qwen35_tp4_correctness_authority_campaign_receipt.py",
)


CHILD_ORDER = receipt.CHILD_ORDER
STAGE_ORDER = receipt.STAGE_ORDER
REQUIRED_EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}
FAILURE_SCHEMA_VERSION = (
    "qwen35.tp4-correctness-authority-campaign-failure.v1"
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


def _failure(
    *,
    plan,
    authorization_record,
    failed_stage,
    completed_stages,
    error,
):
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "classification": "FAILED",
        "plan_sha256": receipt._canonical_sha(plan),
        "authorization_sha256": receipt._canonical_sha(
            authorization_record
        ),
        "authorization_nonce": authorization_record.get("nonce"),
        "campaign_tag": plan.get("campaign_tag"),
        "failed_stage": failed_stage,
        "completed_stages": completed_stages,
        "error": str(error)[-MAX_ERROR_CHARS:],
        "benchmark_execution_authorized": False,
    }


def validate_failure_evidence(
    plan,
    payload,
    *,
    authorization_record,
):
    if (
        not isinstance(payload, dict)
        or set(payload) != {
            "schema_version",
            "classification",
            "plan_sha256",
            "authorization_sha256",
            "authorization_nonce",
            "campaign_tag",
            "failed_stage",
            "completed_stages",
            "error",
            "benchmark_execution_authorized",
        }
        or payload["schema_version"] != FAILURE_SCHEMA_VERSION
        or payload["classification"] != "FAILED"
        or payload["plan_sha256"] != receipt._canonical_sha(plan)
        or payload["authorization_sha256"]
        != receipt._canonical_sha(authorization_record)
        or payload["authorization_nonce"]
        != authorization_record.get("nonce")
        or payload["campaign_tag"] != plan.get("campaign_tag")
        or payload["benchmark_execution_authorized"] is not False
        or not isinstance(payload["error"], str)
        or not payload["error"]
        or len(payload["error"]) > MAX_ERROR_CHARS
    ):
        raise ValueError("campaign failure evidence schema mismatch")
    completed = payload["completed_stages"]
    order = plan.get("stage_order")
    if (
        not isinstance(completed, list)
        or [row.get("name") for row in completed]
        != order[:len(completed)]
        or len(completed) >= len(order)
        or payload["failed_stage"] != order[len(completed)]
    ):
        raise ValueError("campaign failure is not a stage prefix")
    for row in completed:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "result_sha256", "result"}
            or row["result_sha256"]
            != receipt._canonical_sha(row["result"])
        ):
            raise ValueError("campaign failure stage evidence mismatch")
    return payload


def _completed(name, result):
    return {
        "name": name,
        "result_sha256": receipt._canonical_sha(result),
        "result": result,
    }


def _child_result(child):
    authorization_path = receipt._regular_file(
        child["consumed_authorization_path"],
        f"{child['name']} consumed authorization",
    )
    receipt_path = receipt._regular_file(
        child["receipt_path"],
        f"{child['name']} receipt",
    )
    receipt._directory(
        child["authority_dir"],
        f"{child['name']} authority directory",
    )
    return {
        "classification": "PASS",
        **child,
        "authorization_sha256": receipt._sha256(authorization_path),
        "receipt_sha256": receipt._sha256(receipt_path),
    }


def execute_plan(
    *,
    plan,
    authorization_record,
    receipt_path,
    failure_path,
    child_executors,
    child_receipt_verifiers,
    adapt_callback,
    build_callback,
    prerequisite_validator,
    execution_env,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError(
            "exact KRB5CCNAME execution environment is required"
        )
    if (
        set(child_executors) != set(CHILD_ORDER)
        or set(child_receipt_verifiers) != set(CHILD_ORDER)
        or not all(callable(value) for value in child_executors.values())
        or not all(
            callable(value) for value in child_receipt_verifiers.values()
        )
        or not callable(adapt_callback)
        or not callable(build_callback)
        or not callable(prerequisite_validator)
    ):
        raise ValueError("explicit campaign callbacks are required")
    completed = []
    try:
        for child in plan["children"]:
            stage_name = receipt.CHILD_STAGE_NAMES[child["name"]]
            result = child_executors[child["name"]](
                child=child,
                execution_env=dict(execution_env),
            )
            if (
                not isinstance(result, dict)
                or result.get("classification") != "PASS"
            ):
                raise ValueError(
                    f"{child['name']} execution did not prove PASS"
                )
            child_result = _child_result(child)
            summary = child_receipt_verifiers[child["name"]](
                plan_path=child["plan_path"],
                authorization_path=child[
                    "consumed_authorization_path"
                ],
                receipt_path=child["receipt_path"],
            )
            if (
                not isinstance(summary, dict)
                or summary.get("classification") != "PASS"
            ):
                raise ValueError(
                    f"{child['name']} receipt did not prove PASS"
                )
            completed.append(_completed(stage_name, child_result))

        runs = [
            {
                "name": child["name"],
                "run_tag": child["run_tag"],
                "authority_dir": child["authority_dir"],
                "plan_path": child["plan_path"],
                "consumed_authorization_path": child[
                    "consumed_authorization_path"
                ],
                "receipt_path": child["receipt_path"],
            }
            for child in plan["children"]
        ]
        authorities = adapt_callback(
            runs=runs,
            verification_output_dir=plan["adapter_output_dir"],
        )
        adapter_result = {
            "classification": "PASS",
            "authorities": authorities,
        }
        completed.append(_completed("adapt_authorities", adapter_result))

        bundle_result = build_callback(
            authorities=authorities,
            output_dir=plan["bundle_output_dir"],
        )
        completed.append(_completed("build_bundle", bundle_result))

        validation = prerequisite_validator(plan["prerequisite_path"])
        verify_result = {
            "classification": validation.get("classification"),
            "authorized": validation.get("authorized"),
            "prerequisite_sha256": bundle_result.get(
                "prerequisite_sha256"
            ),
        }
        if (
            verify_result["classification"] != "PASS"
            or verify_result["authorized"] is not True
        ):
            raise ValueError("campaign prerequisite is not authorized")
        completed.append(_completed("verify_bundle", verify_result))
    except BaseException as error:
        failed_stage = plan["stage_order"][len(completed)]
        payload = _failure(
            plan=plan,
            authorization_record=authorization_record,
            failed_stage=failed_stage,
            completed_stages=completed,
            error=error,
        )
        validate_failure_evidence(
            plan,
            payload,
            authorization_record=authorization_record,
        )
        _atomic_write(failure_path, payload)
        raise
    return receipt.produce_campaign_receipt(
        plan=plan,
        stage_results=completed,
        authorization_record=authorization_record,
        output_path=receipt_path,
        child_receipt_verifiers=child_receipt_verifiers,
        prerequisite_validator=prerequisite_validator,
    )


def execute_verified_campaign_file(
    *,
    plan_path,
    authorization_path,
    consumed_authorization_path,
    receipt_path,
    failure_path,
    plan_verifier,
    child_executors,
    child_receipt_verifiers,
    adapt_callback,
    build_callback,
    prerequisite_validator,
    execution_env,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError(
            "exact KRB5CCNAME execution environment is required"
        )
    if not callable(plan_verifier):
        raise ValueError("explicit campaign plan verifier is required")
    plan = plan_verifier(plan_path)
    targets = (
        Path(receipt_path),
        Path(failure_path),
        Path(consumed_authorization_path),
        Path(plan["adapter_output_dir"]),
        Path(plan["bundle_output_dir"]),
    )
    if any(path.exists() for path in targets):
        raise ValueError("campaign output target exists")
    authorization_record = authorization.consume_authorization(
        plan=plan,
        authorization_path=authorization_path,
        consumed_path=consumed_authorization_path,
    )
    return execute_plan(
        plan=plan,
        authorization_record=authorization_record,
        receipt_path=receipt_path,
        failure_path=failure_path,
        child_executors=child_executors,
        child_receipt_verifiers=child_receipt_verifiers,
        adapt_callback=adapt_callback,
        build_callback=build_callback,
        prerequisite_validator=prerequisite_validator,
        execution_env=execution_env,
    )
