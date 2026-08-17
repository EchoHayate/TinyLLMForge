from __future__ import annotations

import importlib.util
import hashlib
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


def _load_authorization():
    module_name = "qwen35_tp4_engine_remote_execution_authorization"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_engine_remote_execution_authorization.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


authorization = _load_authorization()
MAX_ERROR_CHARS = 4096
REQUIRED_EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}


def _result_payload(result, name, *, output_path=None):
    required = {"returncode", "stdout", "stderr"}
    if (
        output_path is not None
        and isinstance(result, dict)
        and result.get("returncode") == 0
    ):
        required |= {"output_sha256", "output_size"}
    if not isinstance(result, dict) or set(result) != required:
        if output_path is not None:
            raise ValueError(
                f"{name} package output result schema mismatch"
            )
        raise ValueError(f"{name} command result schema mismatch")
    payload = {
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }
    if output_path is not None and result["returncode"] == 0:
        output_path = Path(output_path)
        if not output_path.is_file() or output_path.is_symlink():
            raise ValueError(f"{name} package output is missing")
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        size = output_path.stat().st_size
        if (
            result["output_sha256"] != digest
            or result["output_size"] != size
        ):
            raise ValueError(f"{name} package output identity mismatch")
        payload.update({
            "output_sha256": digest,
            "output_size": size,
        })
    return payload


def _run_single(
    command_runner,
    *,
    name,
    argv,
    stdout_path=None,
    execution_env=None,
):
    result = _result_payload(
        command_runner(
            name=name,
            argv=argv,
            stdout_path=stdout_path,
            env=dict(execution_env or {}),
        ),
        name,
        output_path=stdout_path,
    )
    if result["returncode"] != 0:
        raise ValueError(f"{name} command returncode mismatch")
    if (
        not isinstance(result["stdout"], str)
        or not isinstance(result["stderr"], str)
        or len(result["stdout"].encode("utf-8"))
        > receipt.MAX_LOG_BYTES
        or len(result["stderr"].encode("utf-8"))
        > receipt.MAX_LOG_BYTES
    ):
        raise ValueError(f"{name} command logs are not bounded")
    return result


def _execute_command(
    command_runner,
    name,
    command,
    *,
    execution_env=None,
):
    if name == "upload":
        argv_rows = command.get("argv")
        if not isinstance(argv_rows, list) or not argv_rows:
            raise ValueError("upload command inventory mismatch")
        stdout = []
        stderr = []
        for index, argv in enumerate(argv_rows):
            result = _run_single(
                command_runner,
                name=f"upload[{index}]",
                argv=argv,
                execution_env=execution_env,
            )
            stdout.append(result["stdout"])
            stderr.append(result["stderr"])
        return {
            "returncode": 0,
            "stdout": "\n".join(stdout),
            "stderr": "\n".join(stderr),
        }
    if name == "guarded_authority":
        return _run_single(
            command_runner,
            name=name,
            argv=command.get("ssh_argv"),
            execution_env=execution_env,
        )
    if name == "package_download":
        return _run_single(
            command_runner,
            name=name,
            argv=command.get("remote_argv"),
            stdout_path=command.get("local_output"),
            execution_env=execution_env,
        )
    return _run_single(
        command_runner,
        name=name,
        argv=command.get("argv"),
        execution_env=execution_env,
    )


def _write_failure(
    *,
    plan,
    failure_path,
    failed_step,
    completed_steps,
    error,
    authorization_record,
):
    failure_path = Path(failure_path)
    payload = {
        "schema_version": (
            "qwen35.tp4-engine-remote-execution-failure.v1"
        ),
        "classification": "FAILED",
        "plan_sha256": receipt._canonical_sha(plan),
        "authorization_sha256": receipt._canonical_sha(
            authorization_record
        ),
        "authorization_nonce": authorization_record.get("nonce"),
        "run_tag": plan.get("run_tag"),
        "failed_step": failed_step,
        "completed_steps": completed_steps,
        "error": str(error)[-MAX_ERROR_CHARS:],
    }
    validate_failure_evidence(
        plan,
        payload,
        authorization_record=authorization_record,
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=failure_path.parent,
        prefix=f".{failure_path.name}.",
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
        os.replace(temporary, failure_path)
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
        "failed_step",
        "completed_steps",
        "error",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"]
        != "qwen35.tp4-engine-remote-execution-failure.v1"
        or payload["classification"] != "FAILED"
        or payload["plan_sha256"] != receipt._canonical_sha(plan)
        or not isinstance(authorization_record, dict)
        or authorization_record.get("consumed") is not True
        or payload["authorization_sha256"]
        != receipt._canonical_sha(authorization_record)
        or payload["authorization_nonce"]
        != authorization_record.get("nonce")
        or authorization_record.get("plan_sha256")
        != payload["plan_sha256"]
        or payload["run_tag"] != plan.get("run_tag")
        or not isinstance(payload["completed_steps"], list)
        or not isinstance(payload["error"], str)
        or not payload["error"]
        or len(payload["error"]) > MAX_ERROR_CHARS
    ):
        raise ValueError("failure evidence schema or plan mismatch")
    order = plan.get("command_order")
    commands = plan.get("commands")
    completed = payload["completed_steps"]
    if (
        not isinstance(order, list)
        or not isinstance(commands, dict)
        or len(completed) >= len(order)
        or [row.get("name") for row in completed]
        != order[:len(completed)]
        or payload["failed_step"] != order[len(completed)]
    ):
        raise ValueError("failure evidence is not an execution prefix")
    for row in completed:
        name = row["name"]
        required_row = {
            "name",
            "command_sha256",
            "returncode",
            "stdout",
            "stderr",
        }
        package_output = (
            name == "package_download"
            and isinstance(commands[name], dict)
            and isinstance(commands[name].get("local_output"), str)
        )
        if package_output:
            required_row |= {"output_sha256", "output_size"}
        if (
            not isinstance(row, dict)
            or set(row) != required_row
            or row["command_sha256"]
            != receipt._canonical_sha(commands[name])
            or row["returncode"] != 0
            or not isinstance(row["stdout"], str)
            or not isinstance(row["stderr"], str)
            or len(row["stdout"].encode("utf-8"))
            > receipt.MAX_LOG_BYTES
            or len(row["stderr"].encode("utf-8"))
            > receipt.MAX_LOG_BYTES
            or (
                package_output
                and (
                    not isinstance(row["output_sha256"], str)
                    or len(row["output_sha256"]) != 64
                    or isinstance(row["output_size"], bool)
                    or not isinstance(row["output_size"], int)
                    or row["output_size"] <= 0
                )
            )
        ):
            raise ValueError("failure evidence completed command mismatch")
    return {
        "classification": "FAILED",
        "run_tag": payload["run_tag"],
        "plan_sha256": payload["plan_sha256"],
        "failed_step": payload["failed_step"],
        "completed_step_count": len(completed),
    }


def execute_plan(
    *,
    plan,
    output_path,
    command_runner,
    authorization_record,
    failure_path=None,
    execution_env=None,
):
    if not callable(command_runner):
        raise ValueError("explicit command runner is required")
    if failure_path is not None and Path(failure_path).exists():
        raise ValueError("failure evidence already exists")
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("command_order"), list)
        or not isinstance(plan.get("commands"), dict)
        or set(plan["commands"]) != set(plan["command_order"])
    ):
        raise ValueError("execution plan schema mismatch")
    steps = []
    for name in plan["command_order"]:
        command = plan["commands"][name]
        try:
            result = _execute_command(
                command_runner,
                name,
                command,
                execution_env=execution_env,
            )
        except BaseException as error:
            if failure_path is not None:
                _write_failure(
                    plan=plan,
                    failure_path=failure_path,
                    failed_step=name,
                    completed_steps=steps,
                    error=error,
                    authorization_record=authorization_record,
                )
            raise
        steps.append({
            "name": name,
            "command_sha256": receipt._canonical_sha(command),
            **result,
        })
    return receipt.produce_execution_receipt(
        plan=plan,
        step_results=steps,
        output_path=output_path,
        authorization_record=authorization_record,
    )


def execute_verified_plan_file(
    *,
    plan_path,
    authorization_path,
    consumed_authorization_path,
    output_path,
    failure_path,
    command_runner,
    plan_verifier,
    execution_env,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError(
            "exact KRB5CCNAME execution environment is required"
        )
    if not callable(plan_verifier):
        raise ValueError("explicit plan verifier is required")
    plan = plan_verifier(plan_path)
    package_path = Path(
        plan["commands"]["package_download"]["local_output"]
    )
    safe_extract_argv = plan["commands"]["safe_extract"]["argv"]
    prepare_argv = plan["commands"]["prepare_local_verifier"]["argv"]
    local_targets = (
        Path(output_path),
        Path(failure_path),
        package_path,
        Path(safe_extract_argv[-1]),
        Path(prepare_argv[-1]),
    )
    if any(path.exists() for path in local_targets):
        raise ValueError("local output target already exists")
    authorization_record = authorization.consume_authorization(
        plan=plan,
        authorization_path=authorization_path,
        consumed_path=consumed_authorization_path,
    )
    return execute_plan(
        plan=plan,
        output_path=output_path,
        failure_path=failure_path,
        command_runner=command_runner,
        authorization_record=authorization_record,
        execution_env=execution_env,
    )
