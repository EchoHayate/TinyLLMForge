from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-benchmark-"
    "remote-execution-receipt.v1"
)
BENCHMARK_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-performance-cache.v1"
)
MAX_LOG_BYTES = 64 * 1024
MIN_GPU_FREE_BYTES = 24 * 1024**3
CASE_COUNT = 70
CASE_ROW_COUNT = 280
PROCESS_ROW_COUNT = 70


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


def _require_sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


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


def _validate_plan(plan):
    required = {
        "schema_version",
        "run_tag",
        "worker_authorization",
        "case_commands",
        "command_order",
        "commands",
    }
    if (
        not isinstance(plan, dict)
        or not required.issubset(plan)
        or not isinstance(plan["run_tag"], str)
        or not plan["run_tag"]
        or not isinstance(plan["worker_authorization"], dict)
        or not isinstance(plan["case_commands"], list)
        or len(plan["case_commands"]) != CASE_COUNT
        or not isinstance(plan["command_order"], list)
        or not isinstance(plan["commands"], dict)
        or set(plan["commands"]) != set(plan["command_order"])
        or len(plan["command_order"]) != len(set(plan["command_order"]))
    ):
        raise ValueError("execution plan schema is invalid")
    return plan


def _validate_authorization(plan, payload, authorization_record):
    worker = plan["worker_authorization"]
    expected_pairs = [
        {
            "case_id": row.get("case_id"),
            "dist_port": row.get("dist_port"),
            "master_port": row.get("master_port"),
        }
        for row in plan["case_commands"]
    ]
    if (
        not isinstance(authorization_record, dict)
        or authorization_record.get("consumed") is not True
        or payload["authorization_sha256"]
        != _canonical_sha(authorization_record)
        or payload["authorization_nonce"]
        != authorization_record.get("nonce")
        or authorization_record.get("plan_sha256")
        != payload["plan_sha256"]
        or authorization_record.get("run_tag") != plan["run_tag"]
        or authorization_record.get("prerequisites_sha256")
        != worker.get("prerequisites_sha256")
        or authorization_record.get("source_tree_sha256")
        != worker.get("source_tree_sha256")
        or authorization_record.get("model_manifest_sha256")
        != worker.get("model_manifest_sha256")
        or authorization_record.get("workload_manifest_sha256")
        != worker.get("workload_manifest_sha256")
        or authorization_record.get("gpu_indices")
        != worker.get("gpu_indices")
        or authorization_record.get("case_port_pairs")
        != expected_pairs
    ):
        raise ValueError("execution receipt authorization mismatch")


def _validate_resource(plan, payload, command_name):
    selected = payload.get("selected")
    gpu_indices = plan["worker_authorization"].get("gpu_indices")
    command = plan["commands"].get(command_name)
    if not isinstance(command, dict):
        raise ValueError("resource guard command is invalid")
    resource_policy = command.get(
        "resource_policy",
        "strict-exclusive",
    )
    requires_no_active_compute_processes = command.get(
        "requires_no_active_compute_processes"
    )
    maximum_gpu_utilization_percent = command.get(
        "maximum_gpu_utilization_percent"
    )
    shared = resource_policy == "shared-low-utilization"
    if (
        payload.get("classification") != "READY"
        or resource_policy not in {
            "strict-exclusive",
            "shared-low-utilization",
        }
        or requires_no_active_compute_processes is shared
        or (
            shared
            and (
                isinstance(
                    maximum_gpu_utilization_percent,
                    bool,
                )
                or not isinstance(
                    maximum_gpu_utilization_percent,
                    int,
                )
                or not 0 <= maximum_gpu_utilization_percent <= 100
                or payload.get("resource_policy") != resource_policy
                or payload.get(
                    "maximum_gpu_utilization_percent"
                ) != maximum_gpu_utilization_percent
            )
        )
        or not isinstance(gpu_indices, list)
        or len(gpu_indices) != 4
        or not isinstance(selected, list)
        or len(selected) != 4
        or [row.get("gpu_index") for row in selected] != gpu_indices
        or len({row.get("gpu_uuid") for row in selected}) != 4
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("gpu_uuid"), str)
            or not row["gpu_uuid"]
            or isinstance(row.get("free_bytes"), bool)
            or not isinstance(row.get("free_bytes"), int)
            or row["free_bytes"] < MIN_GPU_FREE_BYTES
            or not isinstance(row.get("compute_processes"), list)
            or (
                requires_no_active_compute_processes
                and row["compute_processes"] != []
            )
            or (
                shared
                and (
                    isinstance(row.get("utilization_percent"), bool)
                    or not isinstance(
                        row.get("utilization_percent"),
                        int,
                    )
                    or not 0 <= row["utilization_percent"] <= (
                        maximum_gpu_utilization_percent
                    )
                )
            )
            for row in selected
        )
    ):
        raise ValueError("resource guard receipt is invalid")
    return selected


def _validate_worker_completion(plan, payload):
    expected = [
        row.get("case_id") for row in plan["case_commands"]
    ]
    if (
        payload.get("classification") != "COMPLETE"
        or payload.get("case_ids") != expected
        or len(expected) != len(set(expected))
        or any(not isinstance(value, str) or not value for value in expected)
    ):
        raise ValueError("worker completion inventory is invalid")


def _validate_assembly(payload):
    if payload != {
        "classification": "ASSEMBLED",
        "case_rows": CASE_ROW_COUNT,
        "process_rows": PROCESS_ROW_COUNT,
    }:
        raise ValueError("assembly receipt is invalid")


def _validate_verification(plan, payload):
    required = {
        "schema_version",
        "classification",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "correctness_prerequisites_sha256",
        "case_rows",
        "process_rows",
        "workloads",
        "cache_efficiency",
        "initialization_ratio",
        "peak_cuda_reserved_ratio",
    }
    worker = plan["worker_authorization"]
    if (
        set(payload) != required
        or payload["schema_version"] != BENCHMARK_SCHEMA_VERSION
        or payload["classification"] not in {"GO", "NO_GO"}
        or payload["case_rows"] != CASE_ROW_COUNT
        or payload["process_rows"] != PROCESS_ROW_COUNT
        or not isinstance(payload["workloads"], dict)
        or not payload["workloads"]
        or not isinstance(payload["cache_efficiency"], dict)
        or set(payload["cache_efficiency"])
        != {
            "logical_to_physical_snapshot_ratio",
            "physical_snapshot_bytes_per_reused_token",
            "added_cuda_bytes_per_reused_token",
            "saved_prefill_tokens_per_physical_snapshot_byte",
        }
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in payload["cache_efficiency"].values()
        )
    ):
        raise ValueError("benchmark verification payload is invalid")
    identity_pairs = (
        ("source_tree_sha256", "source_tree_sha256"),
        ("model_manifest_sha256", "model_manifest_sha256"),
        ("workload_manifest_sha256", "workload_manifest_sha256"),
        (
            "correctness_prerequisites_sha256",
            "prerequisites_sha256",
        ),
    )
    for payload_name, worker_name in identity_pairs:
        _require_sha(
            payload[payload_name],
            f"benchmark verification {payload_name}",
        )
        if payload[payload_name] != worker.get(worker_name):
            raise ValueError(
                "benchmark verification identity mismatch"
            )
    _canonical_bytes(payload)
    return payload


def validate_execution_receipt(
    plan,
    payload,
    *,
    authorization_record,
):
    plan = _validate_plan(plan)
    required = {
        "schema_version",
        "classification",
        "plan_sha256",
        "authorization_sha256",
        "authorization_nonce",
        "run_tag",
        "steps",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["classification"] != "PASS"
        or payload["plan_sha256"] != _canonical_sha(plan)
        or payload["run_tag"] != plan["run_tag"]
    ):
        raise ValueError("execution receipt plan or schema mismatch")
    _validate_authorization(plan, payload, authorization_record)
    steps = payload["steps"]
    if (
        not isinstance(steps, list)
        or len(steps) != len(plan["command_order"])
        or [row.get("name") for row in steps]
        != plan["command_order"]
    ):
        raise ValueError("execution receipt step order mismatch")
    by_name = {}
    for name, row in zip(plan["command_order"], steps):
        required_row = {
            "name",
            "command_sha256",
            "returncode",
            "stdout",
            "stderr",
        }
        package_output = name == "package_download"
        if package_output:
            required_row |= {"output_sha256", "output_size"}
        if (
            not isinstance(row, dict)
            or set(row) != required_row
            or row["name"] != name
            or row["command_sha256"]
            != _canonical_sha(plan["commands"][name])
        ):
            raise ValueError("execution receipt command mismatch")
        if (
            isinstance(row["returncode"], bool)
            or not isinstance(row["returncode"], int)
            or row["returncode"] != 0
        ):
            raise ValueError("execution receipt returncode mismatch")
        if (
            not isinstance(row["stdout"], str)
            or not isinstance(row["stderr"], str)
            or len(row["stdout"].encode("utf-8")) > MAX_LOG_BYTES
            or len(row["stderr"].encode("utf-8")) > MAX_LOG_BYTES
        ):
            raise ValueError("execution receipt logs are not bounded")
        if package_output:
            try:
                _require_sha(
                    row["output_sha256"],
                    "package output SHA",
                )
            except ValueError as error:
                raise ValueError(
                    "execution receipt package output mismatch"
                ) from error
            if (
                isinstance(row["output_size"], bool)
                or not isinstance(row["output_size"], int)
                or row["output_size"] <= 0
            ):
                raise ValueError(
                    "execution receipt package output mismatch"
                )
        by_name[name] = row

    preflight = _validate_resource(
        plan,
        _parse_json_log(
            by_name["resource_guard"]["stdout"],
            "resource guard",
        ),
        "resource_guard",
    )
    _validate_worker_completion(
        plan,
        _parse_json_log(
            by_name["workers"]["stdout"],
            "worker completion",
        ),
    )
    _validate_assembly(
        _parse_json_log(
            by_name["assembly"]["stdout"],
            "assembly",
        )
    )
    remote = _validate_verification(
        plan,
        _parse_json_log(
            by_name["remote_verify"]["stdout"],
            "remote verification",
        ),
    )
    final = _validate_resource(
        plan,
        _parse_json_log(
            by_name["final_resource_guard"]["stdout"],
            "final resource guard",
        ),
        "final_resource_guard",
    )
    if [
        (row["gpu_index"], row["gpu_uuid"]) for row in preflight
    ] != [
        (row["gpu_index"], row["gpu_uuid"]) for row in final
    ]:
        raise ValueError("resource guard identity drift")
    local = _validate_verification(
        plan,
        _parse_json_log(
            by_name["local_verify"]["stdout"],
            "local verification",
        ),
    )
    if remote != local:
        raise ValueError(
            "local verification does not match remote verification"
        )
    package = by_name["package_download"]
    return {
        "classification": "PASS",
        "benchmark_classification": remote["classification"],
        "run_tag": plan["run_tag"],
        "plan_sha256": payload["plan_sha256"],
        "authorization_sha256": payload["authorization_sha256"],
        "authorization_nonce": payload["authorization_nonce"],
        "source_tree_sha256": remote["source_tree_sha256"],
        "model_manifest_sha256": remote["model_manifest_sha256"],
        "workload_manifest_sha256": remote[
            "workload_manifest_sha256"
        ],
        "correctness_prerequisites_sha256": remote[
            "correctness_prerequisites_sha256"
        ],
        "case_rows": remote["case_rows"],
        "process_rows": remote["process_rows"],
        "gpu_indices": [
            row["gpu_index"] for row in final
        ],
        "package_sha256": package["output_sha256"],
        "package_size": package["output_size"],
        "step_count": len(steps),
    }


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
        "classification": "PASS",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(
            authorization_record
        ),
        "authorization_nonce": authorization_record.get("nonce"),
        "run_tag": plan.get("run_tag"),
        "steps": step_results,
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
