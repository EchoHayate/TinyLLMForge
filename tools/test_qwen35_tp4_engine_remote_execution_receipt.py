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
    "qwen35_tp4_engine_remote_execution_receipt",
    "qwen35_tp4_engine_remote_execution_receipt.py",
)


def _canonical_sha(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _plan():
    commands = {
        name: {"argv": [name, "--frozen"]}
        for name in (
            "reserve_remote",
            "upload",
            "stage",
            "resource_guard",
            "guarded_authority",
            "package_download",
            "safe_extract",
            "prepare_local_verifier",
            "local_verify",
        )
    }
    return {
        "schema_version": "qwen35.tp4-engine-remote-execution-plan.v1",
        "run_tag": "authority-r1",
        "source_tree_sha256": "a" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "ports": {"dist_port": 32001, "master_port": 32002},
        "command_order": list(commands),
        "commands": commands,
    }


def _local_pass_payload():
    return {
        "classification": "PASS",
        "model_manifest_sha256": "b" * 64,
        "source_tree_sha256": "a" * 64,
        "workload_manifest_sha256": "c" * 64,
        "reference_classification": "PASS",
        "engine_classification": "PASS",
    }


def _authority_pass_payload():
    return {
        **_local_pass_payload(),
        "inventory": [
            "reference_authority",
            "reference_independent_verification.json",
            "engine_authority",
            "authority_summary.json",
        ],
    }


def _authorization(plan):
    payload = {
        "schema_version": (
            "qwen35.tp4-engine-remote-execution-authorization.v1"
        ),
        "classification": "AUTHORIZED",
        "plan_sha256": _canonical_sha(plan),
        "run_tag": plan["run_tag"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "model_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "gpu_indices": plan["gpu_indices"],
        "ports": plan["ports"],
        "nonce": "receipt-r1",
        "consumed": True,
    }
    if plan.get("resource_policy") is not None:
        payload.update({
            "resource_policy": plan["resource_policy"],
            "resource_baseline_sha256": plan[
                "resource_baseline_sha256"
            ],
        })
    return payload


def _resource(plan, *, new_pid=False):
    if plan.get("resource_policy") != "controlled_shared":
        return {
            "classification": "READY",
            "selected": [
                {
                    "gpu_index": index,
                    "gpu_uuid": f"GPU-{index}",
                    "free_bytes": 25 * 1024**3,
                    "compute_processes": [],
                }
                for index in plan["gpu_indices"]
            ],
        }
    return {
        "classification": "READY",
        "resource_policy": "controlled_shared",
        "baseline_sha256": plan["resource_baseline_sha256"],
        "benchmark_execution_authorized": False,
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": 25 * 1024**3,
                "compute_processes": [{
                    "pid": (9000 if new_pid else 1000 + index),
                    "process_name": "python3",
                    "used_memory_mib": 436,
                    "start_time_ticks": 2000 + index,
                }],
            }
            for index in plan["gpu_indices"]
        ],
    }


def _receipt(plan):
    steps = []
    for name in plan["command_order"]:
        stdout = ""
        if name == "resource_guard":
            stdout = json.dumps(_resource(plan))
        elif name == "guarded_authority":
            stdout = "\n".join([
                "QWEN35_FINAL_RESOURCE_JSON="
                + json.dumps(_resource(plan)),
                "ordinary authority log",
                json.dumps(_authority_pass_payload()),
            ])
        elif name == "local_verify":
            stdout = json.dumps(_local_pass_payload())
        command = plan["commands"][name]
        steps.append({
            "name": name,
            "command_sha256": _canonical_sha(command),
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        })
    return {
        "schema_version": (
            "qwen35.tp4-engine-remote-execution-receipt.v1"
        ),
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(
            _authorization(plan)
        ),
        "authorization_nonce": "receipt-r1",
        "run_tag": plan["run_tag"],
        "steps": steps,
        "classification": "PASS",
    }


def _shared_plan(root):
    baseline = root / "resource_baseline.json"
    payload = {
        "schema_version": (
            "qwen35.tp4-controlled-shared-resource-baseline.v1"
        ),
        "classification": "READY",
        "ssh_target": "sitian@10.232.195.203",
        "captured_at": "2026-07-29T15:00:00+08:00",
        "gpu_indices": [0, 1, 2, 3],
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": 64 * 1024**3,
                "compute_processes": [{
                    "pid": 1000 + index,
                    "process_name": "python3",
                    "used_memory_mib": 436,
                    "start_time_ticks": 2000 + index,
                }],
            }
            for index in [0, 1, 2, 3]
        ],
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "benchmark_execution_authorized": False,
    }
    baseline.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    plan = _plan()
    plan.update({
        "resource_policy": "controlled_shared",
        "resource_baseline_sha256": hashlib.sha256(
            baseline.read_bytes()
        ).hexdigest(),
    })
    plan["local_inputs"] = {
        "resource_baseline": str(baseline),
        "resource_baseline_sha256": plan[
            "resource_baseline_sha256"
        ],
    }
    plan["commands"]["resource_guard"].update({
        "resource_policy": "controlled_shared",
        "resource_baseline_sha256": plan[
            "resource_baseline_sha256"
        ],
    })
    return plan


def test_receipt_accepts_controlled_shared_subset_and_rejects_new_pid():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _shared_plan(Path(temporary))
        payload = _receipt(plan)
        result = receipt.validate_execution_receipt(
            plan,
            payload,
            authorization_record=_authorization(plan),
        )
        assert result["classification"] == "PASS"
        changed = copy.deepcopy(payload)
        changed["steps"][3]["stdout"] = json.dumps(
            _resource(plan, new_pid=True)
        )
        _expect_reject(plan, changed, "process")


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


def test_receipt_requires_exact_steps_guard_and_two_passes():
    plan = _plan()
    payload = _receipt(plan)
    result = receipt.validate_execution_receipt(
        plan,
        payload,
        authorization_record=_authorization(plan),
    )
    assert result == {
        "classification": "PASS",
        "run_tag": "authority-r1",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(
            _authorization(plan)
        ),
        "authorization_nonce": "receipt-r1",
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "step_count": 9,
    }


def test_receipt_rejects_invalid_pass_inventory():
    plan = _plan()
    payload = _receipt(plan)
    guarded_authority = payload["steps"][4]
    pass_payload = _authority_pass_payload()
    pass_payload["inventory"] = [
        "reference_authority",
        "engine_authority",
        "authority_summary.json",
    ]
    guarded_authority["stdout"] = "\n".join([
        "QWEN35_FINAL_RESOURCE_JSON="
        + json.dumps(_resource(plan)),
        "ordinary authority log",
        json.dumps(pass_payload),
    ])
    _expect_reject(plan, payload, "inventory")


def test_receipt_rejects_plan_command_order_and_returncode_tamper():
    plan = _plan()
    payload = _receipt(plan)
    changed = copy.deepcopy(payload)
    changed["plan_sha256"] = "f" * 64
    _expect_reject(plan, changed, "plan")

    changed = copy.deepcopy(payload)
    changed["steps"][2]["command_sha256"] = "f" * 64
    _expect_reject(plan, changed, "command")

    changed = copy.deepcopy(payload)
    changed["steps"][1], changed["steps"][2] = (
        changed["steps"][2],
        changed["steps"][1],
    )
    _expect_reject(plan, changed, "order")

    changed = copy.deepcopy(payload)
    changed["steps"][4]["returncode"] = 1
    changed["classification"] = "PASS"
    _expect_reject(plan, changed, "returncode")


def test_receipt_rejects_unsafe_logs_resource_drift_and_pass_drift():
    plan = _plan()
    payload = _receipt(plan)
    changed = copy.deepcopy(payload)
    changed["steps"][0]["stdout"] = "x" * (receipt.MAX_LOG_BYTES + 1)
    _expect_reject(plan, changed, "bounded")

    changed = copy.deepcopy(payload)
    guard = json.loads(changed["steps"][3]["stdout"])
    guard["selected"][0]["compute_processes"] = [{"pid": 123}]
    changed["steps"][3]["stdout"] = json.dumps(guard)
    _expect_reject(plan, changed, "resource")

    changed = copy.deepcopy(payload)
    local = json.loads(changed["steps"][-1]["stdout"])
    local["source_tree_sha256"] = "d" * 64
    changed["steps"][-1]["stdout"] = json.dumps(local)
    _expect_reject(plan, changed, "verification")

    changed = copy.deepcopy(payload)
    changed["steps"][-1]["stdout"] = "not-json"
    _expect_reject(plan, changed, "JSON")

    changed = copy.deepcopy(payload)
    guarded_lines = changed["steps"][4]["stdout"].splitlines()
    guarded_lines.pop(0)
    changed["steps"][4]["stdout"] = "\n".join(guarded_lines)
    _expect_reject(plan, changed, "final resource")

    changed = copy.deepcopy(payload)
    guarded_lines = changed["steps"][4]["stdout"].splitlines()
    resource = json.loads(guarded_lines[0].split("=", 1)[1])
    resource["selected"][0]["gpu_uuid"] = "GPU-drift"
    guarded_lines[0] = (
        "QWEN35_FINAL_RESOURCE_JSON=" + json.dumps(resource)
    )
    changed["steps"][4]["stdout"] = "\n".join(guarded_lines)
    _expect_reject(plan, changed, "drift")


def test_receipt_file_verifier_rejects_extra_or_symlink():
    plan = _plan()
    payload = _receipt(plan)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan_path = root / "plan.json"
        receipt_path = root / "receipt.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        receipt_path.write_text(json.dumps(payload) + "\n")
        authorization_path = root / "authorization.consumed.json"
        authorization_path.write_text(
            json.dumps(_authorization(plan)) + "\n"
        )
        assert receipt.verify_receipt_files(
            plan_path=plan_path,
            receipt_path=receipt_path,
            authorization_path=authorization_path,
        )["classification"] == "PASS"
        extra = dict(payload)
        extra["extra"] = True
        receipt_path.write_text(json.dumps(extra) + "\n")
        try:
            receipt.verify_receipt_files(
                plan_path=plan_path,
                receipt_path=receipt_path,
                authorization_path=authorization_path,
            )
        except ValueError as error:
            assert "schema" in str(error)
        else:
            raise AssertionError("extra receipt field was accepted")


def test_receipt_producer_validates_then_publishes_atomically():
    plan = _plan()
    payload = _receipt(plan)
    results = payload["steps"]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "execution_receipt.json"
        summary = receipt.produce_execution_receipt(
            plan=plan,
            step_results=results,
            output_path=output,
            authorization_record=_authorization(plan),
        )
        assert summary["classification"] == "PASS"
        assert json.loads(output.read_text()) == payload
        _expect_reject(
            plan,
            {
                **payload,
                "steps": payload["steps"][:-1],
            },
            "order",
        )
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                step_results=results,
                output_path=output,
                authorization_record=_authorization(plan),
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing receipt was overwritten")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "execution_receipt.json"
        failed = copy.deepcopy(results)
        failed[4]["returncode"] = 1
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                step_results=failed,
                output_path=output,
                authorization_record=_authorization(plan),
            )
        except ValueError as error:
            assert "returncode" in str(error)
        else:
            raise AssertionError("failed execution was published")
        assert not output.exists()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine remote execution receipt tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
