from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


receipt = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_remote_execution_receipt",
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_receipt.py"
    ),
)


def _canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _plan(*, shared=False):
    order = [
        "reserve_remote",
        "upload",
        "stage",
        "resource_guard",
        "workers",
        "assembly",
        "remote_verify",
        "final_resource_guard",
        "package_download",
        "safe_extract",
        "local_verify",
    ]
    commands = {
        name: {"argv": [name, "--frozen"]}
        for name in order
    }
    commands["package_download"]["local_output"] = (
        "/tmp/benchmark-artifact.tar"
    )
    resource_command = {
        "argv": ["resource-guard", "--frozen"],
        "gpu_indices": [0, 1, 2, 3],
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "requires_no_active_compute_processes": not shared,
        "resource_policy": (
            "shared-low-utilization"
            if shared
            else "strict-exclusive"
        ),
    }
    if shared:
        resource_command["maximum_gpu_utilization_percent"] = 10
    commands["resource_guard"] = copy.deepcopy(resource_command)
    commands["final_resource_guard"] = copy.deepcopy(resource_command)
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-benchmark-"
            "remote-execution-plan.v1"
        ),
        "run_tag": "benchmark-receipt-r1",
        "worker_authorization": {
            "prerequisites_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "model_manifest_sha256": "c" * 64,
            "workload_manifest_sha256": "d" * 64,
            "gpu_indices": [0, 1, 2, 3],
        },
        "case_commands": [
            {
                "case_id": f"case-{index:02d}",
                "dist_port": 22000 + index * 2,
                "master_port": 22001 + index * 2,
            }
            for index in range(70)
        ],
        "command_order": order,
        "commands": commands,
    }


def _authorization(plan):
    worker = plan["worker_authorization"]
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-benchmark-"
            "remote-execution-authorization.v1"
        ),
        "classification": "AUTHORIZED",
        "plan_sha256": _canonical_sha(plan),
        "run_tag": plan["run_tag"],
        "prerequisites_sha256": worker["prerequisites_sha256"],
        "source_tree_sha256": worker["source_tree_sha256"],
        "model_manifest_sha256": worker["model_manifest_sha256"],
        "workload_manifest_sha256": worker[
            "workload_manifest_sha256"
        ],
        "gpu_indices": worker["gpu_indices"],
        "case_port_pairs": [
            {
                "case_id": row["case_id"],
                "dist_port": row["dist_port"],
                "master_port": row["master_port"],
            }
            for row in plan["case_commands"]
        ],
        "nonce": "benchmark-receipt-r1",
        "consumed": True,
    }


def _resource(*, shared=False, utilization_percent=0):
    payload = {
        "classification": "READY",
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": 25 * 1024**3,
                "compute_processes": (
                    [{"pid": 1000 + index}]
                    if shared
                    else []
                ),
                "utilization_percent": utilization_percent,
            }
            for index in range(4)
        ],
    }
    if shared:
        payload.update({
            "resource_policy": "shared-low-utilization",
            "maximum_gpu_utilization_percent": 10,
        })
    return payload


def _verification(classification="GO"):
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-performance-cache.v1"
        ),
        "classification": classification,
        "source_tree_sha256": "b" * 64,
        "model_manifest_sha256": "c" * 64,
        "workload_manifest_sha256": "d" * 64,
        "correctness_prerequisites_sha256": "a" * 64,
        "case_rows": 280,
        "process_rows": 70,
        "workloads": {
            f"w{index}": {
                "median_ttft_ratio": 0.9,
                "max_repetition_ttft_ratio": 0.95,
                "throughput_ratio": 1.1,
                "median_e2e_ratio": 0.91,
                "median_decode_ratio": 1.0,
            }
            for index in range(5)
        },
        "cache_efficiency": {
            "logical_to_physical_snapshot_ratio": 1.5,
            "physical_snapshot_bytes_per_reused_token": 0.5,
            "added_cuda_bytes_per_reused_token": 1.0,
            "saved_prefill_tokens_per_physical_snapshot_byte": 2.0,
        },
        "initialization_ratio": 1.0,
        "peak_cuda_reserved_ratio": 1.0,
    }


def _execution_receipt(
    plan,
    classification="GO",
    *,
    shared=False,
    utilization_percent=0,
):
    steps = []
    for name in plan["command_order"]:
        stdout = ""
        if name in {"resource_guard", "final_resource_guard"}:
            stdout = json.dumps(_resource(
                shared=shared,
                utilization_percent=utilization_percent,
            ))
        elif name == "workers":
            stdout = json.dumps({
                "classification": "COMPLETE",
                "case_ids": [
                    row["case_id"]
                    for row in plan["case_commands"]
                ],
            })
        elif name == "assembly":
            stdout = json.dumps({
                "classification": "ASSEMBLED",
                "case_rows": 280,
                "process_rows": 70,
            })
        elif name in {"remote_verify", "local_verify"}:
            stdout = json.dumps(_verification(classification))
        row = {
            "name": name,
            "command_sha256": _canonical_sha(
                plan["commands"][name]
            ),
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        }
        if name == "package_download":
            row.update({
                "output_sha256": "e" * 64,
                "output_size": 12345,
            })
        steps.append(row)
    authorization = _authorization(plan)
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-benchmark-"
            "remote-execution-receipt.v1"
        ),
        "classification": "PASS",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(authorization),
        "authorization_nonce": authorization["nonce"],
        "run_tag": plan["run_tag"],
        "steps": steps,
    }


def _expect_reject(plan, payload, fragment):
    try:
        receipt.validate_execution_receipt(
            plan,
            payload,
            authorization_record=_authorization(plan),
        )
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected rejection containing {fragment!r}"
        )


def test_receipt_accepts_verified_go_and_no_go_artifacts():
    plan = _plan()
    for classification in ("GO", "NO_GO"):
        payload = _execution_receipt(plan, classification)
        summary = receipt.validate_execution_receipt(
            plan,
            payload,
            authorization_record=_authorization(plan),
        )
        assert summary["classification"] == "PASS"
        assert summary["benchmark_classification"] == classification
        assert summary["case_rows"] == 280
        assert summary["process_rows"] == 70
        assert summary["package_sha256"] == "e" * 64


def test_receipt_rejects_worker_inventory_or_assembly_tamper():
    plan = _plan()
    payload = _execution_receipt(plan)
    changed = copy.deepcopy(payload)
    workers = json.loads(changed["steps"][4]["stdout"])
    workers["case_ids"].pop()
    changed["steps"][4]["stdout"] = json.dumps(workers)
    _expect_reject(plan, changed, "worker")

    changed = copy.deepcopy(payload)
    assembly = json.loads(changed["steps"][5]["stdout"])
    assembly["case_rows"] = 279
    changed["steps"][5]["stdout"] = json.dumps(assembly)
    _expect_reject(plan, changed, "assembly")


def test_receipt_rejects_resource_or_verification_drift():
    plan = _plan()
    payload = _execution_receipt(plan)
    changed = copy.deepcopy(payload)
    final = json.loads(changed["steps"][7]["stdout"])
    final["selected"][0]["gpu_uuid"] = "GPU-drift"
    changed["steps"][7]["stdout"] = json.dumps(final)
    _expect_reject(plan, changed, "resource")

    changed = copy.deepcopy(payload)
    local = json.loads(changed["steps"][-1]["stdout"])
    local["peak_cuda_reserved_ratio"] = 0.5
    changed["steps"][-1]["stdout"] = json.dumps(local)
    _expect_reject(plan, changed, "verification")

    changed = copy.deepcopy(payload)
    remote = json.loads(changed["steps"][6]["stdout"])
    remote["source_tree_sha256"] = "0" * 64
    changed["steps"][6]["stdout"] = json.dumps(remote)
    _expect_reject(plan, changed, "identity")


def test_receipt_accepts_shared_processes_below_utilization_limit():
    plan = _plan(shared=True)
    payload = _execution_receipt(
        plan,
        shared=True,
        utilization_percent=10,
    )
    assert receipt.validate_execution_receipt(
        plan,
        payload,
        authorization_record=_authorization(plan),
    )["classification"] == "PASS"

    payload = _execution_receipt(
        plan,
        shared=True,
        utilization_percent=11,
    )
    _expect_reject(plan, payload, "resource")


def test_receipt_rejects_command_package_or_authorization_tamper():
    plan = _plan()
    payload = _execution_receipt(plan)
    changed = copy.deepcopy(payload)
    changed["steps"][2]["command_sha256"] = "0" * 64
    _expect_reject(plan, changed, "command")

    changed = copy.deepcopy(payload)
    changed["steps"][8]["output_size"] = 0
    _expect_reject(plan, changed, "package")

    changed = copy.deepcopy(payload)
    changed["authorization_nonce"] = "other"
    _expect_reject(plan, changed, "authorization")


def test_receipt_producer_and_file_verifier_publish_atomically():
    plan = _plan()
    payload = _execution_receipt(plan)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "receipt.json"
        summary = receipt.produce_execution_receipt(
            plan=plan,
            step_results=payload["steps"],
            output_path=output,
            authorization_record=_authorization(plan),
        )
        assert summary["classification"] == "PASS"
        plan_path = root / "plan.json"
        authorization_path = root / "authorization.consumed.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        authorization_path.write_text(
            json.dumps(_authorization(plan)) + "\n"
        )
        assert receipt.verify_receipt_files(
            plan_path=plan_path,
            receipt_path=output,
            authorization_path=authorization_path,
        )["benchmark_classification"] == "GO"
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                step_results=payload["steps"],
                output_path=output,
                authorization_record=_authorization(plan),
            )
        except ValueError as error:
            assert "exists" in str(error), str(error)
        else:
            raise AssertionError("receipt output was overwritten")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark remote execution "
        f"receipt tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
