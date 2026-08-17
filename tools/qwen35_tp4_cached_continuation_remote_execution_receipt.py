from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-cached-continuation-remote-execution-receipt.v1"
)
CACHED_VERIFICATION_SCHEMA_VERSION = (
    "qwen35.tp4-cached-continuation-correctness.v1"
)
MAX_LOG_BYTES = 64 * 1024
MIN_GPU_FREE_BYTES = 24 * 1024**3


def _load_resource_policy():
    module_name = (
        "qwen35_tp4_correctness_resource_policy_for_cached_receipt"
    )
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_correctness_resource_policy.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


resource_policy_module = _load_resource_policy()


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


def _load_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"{label} must be a regular non-symlink file"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error


def _require_sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _parse_json_log(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def _validate_resource_payload(plan, payload):
    gpu_indices = plan.get("gpu_indices")
    resource_policy = plan.get(
        "resource_policy",
        resource_policy_module.STRICT_EXCLUSIVE,
    )
    baseline = None
    baseline_sha256 = None
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        local_inputs = plan.get("local_inputs")
        if not isinstance(local_inputs, dict):
            raise ValueError("resource baseline input is missing")
        baseline_sha256 = plan.get("resource_baseline_sha256")
        if (
            local_inputs.get("resource_baseline_sha256")
            != baseline_sha256
        ):
            raise ValueError("resource baseline identity mismatch")
        baseline = resource_policy_module.validate_baseline_manifest(
            local_inputs.get("resource_baseline", ""),
            ssh_target="sitian@10.232.195.203",
            gpu_indices=gpu_indices,
        )
        if (
            resource_policy_module.sha256(
                local_inputs["resource_baseline"]
            )
            != baseline_sha256
        ):
            raise ValueError("resource baseline SHA mismatch")
    try:
        resource_policy_module.validate_guard_payload(
            resource_policy,
            payload,
            gpu_indices=gpu_indices,
            baseline=baseline,
            baseline_sha256=baseline_sha256,
        )
    except ValueError as error:
        raise ValueError(
            f"resource guard receipt is invalid: {error}"
        ) from error
    return payload["selected"]


def _validate_pass_payload(value, label):
    payload = _parse_json_log(value, label)
    required = {
        "classification",
        "schema_version",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "checks",
    }
    if (
        set(payload) != required
        or payload["classification"] != "PASS"
        or payload["schema_version"]
        != CACHED_VERIFICATION_SCHEMA_VERSION
        or not isinstance(payload["checks"], dict)
        or not payload["checks"]
    ):
        raise ValueError(f"{label} PASS payload is invalid")
    for name in (
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
    ):
        _require_sha(payload[name], f"{label} {name}")
    _canonical_bytes(payload["checks"])
    return payload


def _validate_authorization(plan, receipt, authorization_record):
    local_inputs = plan.get("local_inputs")
    if (
        not isinstance(authorization_record, dict)
        or authorization_record.get("consumed") is not True
        or receipt["authorization_sha256"]
        != _canonical_sha(authorization_record)
        or receipt["authorization_nonce"]
        != authorization_record.get("nonce")
        or authorization_record.get("plan_sha256")
        != receipt["plan_sha256"]
        or authorization_record.get("run_tag") != plan["run_tag"]
        or authorization_record.get("source_tree_sha256")
        != plan["source_tree_sha256"]
        or authorization_record.get("model_manifest_sha256")
        != plan["model_manifest_sha256"]
        or authorization_record.get("workload_manifest_sha256")
        != local_inputs.get("workload_manifest_sha256")
        or authorization_record.get("gpu_indices")
        != plan["gpu_indices"]
        or authorization_record.get("ports") != plan["ports"]
        or (
            plan.get("resource_policy") is not None
            and (
                authorization_record.get("resource_policy")
                != plan.get("resource_policy")
                or authorization_record.get(
                    "resource_baseline_sha256"
                )
                != plan.get("resource_baseline_sha256")
            )
        )
    ):
        raise ValueError("execution receipt authorization mismatch")


def validate_execution_receipt(
    plan,
    receipt,
    *,
    authorization_record,
):
    plan_required = {
        "schema_version",
        "run_tag",
        "source_tree_sha256",
        "model_manifest_sha256",
        "gpu_indices",
        "ports",
        "local_inputs",
        "command_order",
        "commands",
    }
    if (
        not isinstance(plan, dict)
        or not plan_required.issubset(plan)
        or not isinstance(plan["command_order"], list)
        or not isinstance(plan["commands"], dict)
        or set(plan["commands"]) != set(plan["command_order"])
        or len(plan["command_order"]) != len(set(plan["command_order"]))
    ):
        raise ValueError("execution plan schema is invalid")
    receipt_required = {
        "schema_version",
        "plan_sha256",
        "authorization_sha256",
        "authorization_nonce",
        "run_tag",
        "steps",
        "classification",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_required
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt["classification"] != "PASS"
        or receipt["run_tag"] != plan["run_tag"]
        or receipt["plan_sha256"] != _canonical_sha(plan)
    ):
        raise ValueError("execution receipt plan or schema mismatch")
    _validate_authorization(plan, receipt, authorization_record)
    steps = receipt["steps"]
    if (
        not isinstance(steps, list)
        or len(steps) != len(plan["command_order"])
        or [step.get("name") for step in steps]
        != plan["command_order"]
    ):
        raise ValueError("execution receipt step order mismatch")
    by_name = {}
    for name, step in zip(plan["command_order"], steps):
        required = {
            "name",
            "command_sha256",
            "returncode",
            "stdout",
            "stderr",
        }
        package_output = (
            name == "package_download"
            and isinstance(plan["commands"][name], dict)
            and isinstance(
                plan["commands"][name].get("local_output"),
                str,
            )
        )
        if package_output:
            required |= {"output_sha256", "output_size"}
        if (
            not isinstance(step, dict)
            or set(step) != required
            or step["name"] != name
            or step["command_sha256"]
            != _canonical_sha(plan["commands"][name])
        ):
            raise ValueError("execution receipt command mismatch")
        if (
            isinstance(step["returncode"], bool)
            or not isinstance(step["returncode"], int)
            or step["returncode"] != 0
        ):
            raise ValueError("execution receipt returncode mismatch")
        if (
            not isinstance(step["stdout"], str)
            or not isinstance(step["stderr"], str)
            or len(step["stdout"].encode("utf-8")) > MAX_LOG_BYTES
            or len(step["stderr"].encode("utf-8")) > MAX_LOG_BYTES
        ):
            raise ValueError("execution receipt logs are not bounded")
        if package_output:
            try:
                _require_sha(
                    step["output_sha256"],
                    "execution receipt package SHA",
                )
            except ValueError as error:
                raise ValueError(
                    "execution receipt package output mismatch"
                ) from error
            if (
                isinstance(step["output_size"], bool)
                or not isinstance(step["output_size"], int)
                or step["output_size"] <= 0
            ):
                raise ValueError(
                    "execution receipt package output mismatch"
                )
        by_name[name] = step

    preflight = _validate_resource_payload(
        plan,
        _parse_json_log(
            by_name["resource_guard"]["stdout"],
            "resource guard",
        ),
    )
    guarded_lines = by_name["guarded_authority"]["stdout"].splitlines()
    marker = "QWEN35_FINAL_RESOURCE_JSON="
    marked = [
        line[len(marker):]
        for line in guarded_lines
        if line.startswith(marker)
    ]
    if len(marked) != 1:
        raise ValueError(
            "guarded authority final resource receipt is invalid"
        )
    final_resource = _validate_resource_payload(
        plan,
        _parse_json_log(
            marked[0],
            "guarded authority final resource",
        ),
    )
    if [
        (row["gpu_index"], row["gpu_uuid"])
        for row in final_resource
    ] != [
        (row["gpu_index"], row["gpu_uuid"])
        for row in preflight
    ]:
        raise ValueError("final resource receipt drifted from preflight")
    authority_line = None
    for line in reversed(guarded_lines):
        if not line or line.startswith(marker):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            authority_line = line
            break
    if authority_line is None:
        raise ValueError("guarded authority PASS JSON is invalid")
    authority = _validate_pass_payload(
        authority_line,
        "guarded authority",
    )
    local = _validate_pass_payload(
        by_name["local_verify"]["stdout"],
        "local verification",
    )
    if authority != local:
        raise ValueError("local verification does not match authority")
    local_inputs = plan["local_inputs"]
    if (
        authority["source_tree_sha256"]
        != plan["source_tree_sha256"]
        or authority["model_manifest_sha256"]
        != plan["model_manifest_sha256"]
        or authority["workload_manifest_sha256"]
        != local_inputs["workload_manifest_sha256"]
    ):
        raise ValueError("cached authority identity mismatch")
    package = by_name["package_download"]
    return {
        "classification": "PASS",
        "run_tag": plan["run_tag"],
        "plan_sha256": receipt["plan_sha256"],
        "authorization_sha256": receipt["authorization_sha256"],
        "authorization_nonce": receipt["authorization_nonce"],
        "source_tree_sha256": authority["source_tree_sha256"],
        "model_manifest_sha256": authority[
            "model_manifest_sha256"
        ],
        "workload_manifest_sha256": authority[
            "workload_manifest_sha256"
        ],
        "gpu_indices": [row["gpu_index"] for row in final_resource],
        "gpu_uuids": [row["gpu_uuid"] for row in final_resource],
        "package_sha256": package["output_sha256"],
        "package_size": package["output_size"],
        "step_count": len(steps),
    }


def verify_receipt_files(
    *,
    plan_path,
    receipt_path,
    authorization_path,
):
    return validate_execution_receipt(
        _load_json(plan_path, "execution plan"),
        _load_json(receipt_path, "execution receipt"),
        authorization_record=_load_json(
            authorization_path,
            "consumed authorization",
        ),
    )


def produce_execution_receipt(
    *,
    plan,
    step_results,
    output_path,
    authorization_record,
):
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("execution receipt output already exists")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(
            authorization_record
        ),
        "authorization_nonce": authorization_record.get("nonce"),
        "run_tag": plan.get("run_tag"),
        "steps": step_results,
        "classification": "PASS",
    }
    summary = validate_execution_receipt(
        plan,
        payload,
        authorization_record=authorization_record,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        verify_receipt_files(
            plan_path=args.plan,
            receipt_path=args.receipt,
            authorization_path=args.authorization,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
