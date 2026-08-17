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
    "qwen35_tp4_cached_continuation_remote_execution_receipt",
    "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
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
    commands["package_download"] = {
        "remote_argv": ["ssh", "package"],
        "local_output": "/tmp/cached-authority.tar",
    }
    return {
        "schema_version": (
            "qwen35.tp4-cached-continuation-remote-execution-plan.v1"
        ),
        "run_tag": "cached-authority-r1",
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "ports": {"dist_port": 32001, "master_port": 32002},
        "local_inputs": {
            "workload_manifest_sha256": "c" * 64,
        },
        "command_order": list(commands),
        "commands": commands,
    }


def _pass_payload():
    return {
        "classification": "PASS",
        "schema_version": (
            "qwen35.tp4-cached-continuation-correctness.v1"
        ),
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "checks": {
            "row_count": 19,
            "token_match": True,
            "registered_logits_allclose": True,
        },
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
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "workload_manifest_sha256": (
            plan["local_inputs"]["workload_manifest_sha256"]
        ),
        "gpu_indices": plan["gpu_indices"],
        "ports": plan["ports"],
        "nonce": "cached-receipt-r1",
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


def _resource(
    plan,
    *,
    free_bytes=25 * 1024**3,
    new_pid=False,
):
    payload = {
        "classification": "READY",
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": free_bytes,
                "compute_processes": (
                    [{
                        "pid": (
                            9000 if new_pid else 1000 + index
                        ),
                        "process_name": "python3",
                        "used_memory_mib": 436,
                        "start_time_ticks": 2000 + index,
                    }]
                    if plan.get("resource_policy")
                    == "controlled_shared"
                    else []
                ),
            }
            for index in plan["gpu_indices"]
        ],
    }
    if plan.get("resource_policy") == "controlled_shared":
        payload.update({
            "resource_policy": "controlled_shared",
            "baseline_sha256": plan[
                "resource_baseline_sha256"
            ],
            "benchmark_execution_authorized": False,
        })
    return payload


def _steps(plan):
    steps = []
    for name in plan["command_order"]:
        stdout = ""
        if name == "resource_guard":
            stdout = json.dumps(_resource(plan))
        elif name == "guarded_authority":
            stdout = "\n".join([
                "QWEN35_FINAL_RESOURCE_JSON="
                + json.dumps(_resource(plan)),
                "ordinary cached authority log",
                json.dumps(_pass_payload()),
            ])
        elif name == "local_verify":
            stdout = json.dumps(_pass_payload())
        step = {
            "name": name,
            "command_sha256": _canonical_sha(
                plan["commands"][name]
            ),
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        }
        if name == "package_download":
            step.update({
                "output_sha256": "d" * 64,
                "output_size": 4096,
            })
        steps.append(step)
    return steps


def _receipt(plan):
    authorization = _authorization(plan)
    return {
        "schema_version": (
            "qwen35.tp4-cached-continuation-remote-execution-receipt.v1"
        ),
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(authorization),
        "authorization_nonce": authorization["nonce"],
        "run_tag": plan["run_tag"],
        "steps": _steps(plan),
        "classification": "PASS",
    }


def _shared_plan(root):
    baseline = root / "resource_baseline.json"
    baseline.write_text(
        json.dumps({
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
        }, sort_keys=True, separators=(",", ":")) + "\n"
    )
    plan = _plan()
    plan.update({
        "resource_policy": "controlled_shared",
        "resource_baseline_sha256": hashlib.sha256(
            baseline.read_bytes()
        ).hexdigest(),
    })
    plan["local_inputs"] = {
        **plan["local_inputs"],
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


def _expect_reject(plan, payload, fragment, authorization=None):
    try:
        receipt.validate_execution_receipt(
            plan,
            payload,
            authorization_record=(
                _authorization(plan)
                if authorization is None
                else authorization
            ),
        )
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected rejection containing {fragment!r}"
        )


def test_receipt_accepts_cached_pass_without_engine_phase_fields():
    plan = _plan()
    payload = _receipt(plan)
    result = receipt.validate_execution_receipt(
        plan,
        payload,
        authorization_record=_authorization(plan),
    )
    assert result == {
        "classification": "PASS",
        "run_tag": "cached-authority-r1",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(
            _authorization(plan)
        ),
        "authorization_nonce": "cached-receipt-r1",
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "gpu_uuids": ["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
        "package_sha256": "d" * 64,
        "package_size": 4096,
        "step_count": 9,
    }


def test_receipt_rejects_engine_fields_and_remote_local_drift():
    plan = _plan()
    payload = _receipt(plan)
    changed = copy.deepcopy(payload)
    remote = json.loads(
        changed["steps"][4]["stdout"].splitlines()[-1]
    )
    remote["reference_classification"] = "PASS"
    changed["steps"][4]["stdout"] = "\n".join([
        *changed["steps"][4]["stdout"].splitlines()[:-1],
        json.dumps(remote),
    ])
    _expect_reject(plan, changed, "PASS payload")

    changed = copy.deepcopy(payload)
    local = json.loads(changed["steps"][-1]["stdout"])
    local["checks"]["row_count"] = 18
    changed["steps"][-1]["stdout"] = json.dumps(local)
    _expect_reject(plan, changed, "does not match")


def test_receipt_rejects_identity_authorization_and_resource_drift():
    plan = _plan()
    payload = _receipt(plan)
    changed = copy.deepcopy(payload)
    local = json.loads(changed["steps"][-1]["stdout"])
    local["model_manifest_sha256"] = "e" * 64
    changed["steps"][-1]["stdout"] = json.dumps(local)
    _expect_reject(plan, changed, "does not match")

    changed = copy.deepcopy(payload)
    lines = changed["steps"][4]["stdout"].splitlines()
    final_resource = json.loads(lines[0].split("=", 1)[1])
    final_resource["selected"][0]["gpu_uuid"] = "GPU-drift"
    lines[0] = (
        "QWEN35_FINAL_RESOURCE_JSON="
        + json.dumps(final_resource)
    )
    changed["steps"][4]["stdout"] = "\n".join(lines)
    _expect_reject(plan, changed, "drift")

    authorization = _authorization(plan)
    authorization["model_manifest_sha256"] = "e" * 64
    changed = copy.deepcopy(payload)
    changed["authorization_sha256"] = _canonical_sha(authorization)
    _expect_reject(
        plan,
        changed,
        "authorization",
        authorization=authorization,
    )


def test_receipt_requires_exact_commands_and_nonempty_package_identity():
    plan = _plan()
    payload = _receipt(plan)
    changed = copy.deepcopy(payload)
    changed["steps"][2]["command_sha256"] = "f" * 64
    _expect_reject(plan, changed, "command")

    changed = copy.deepcopy(payload)
    changed["steps"][5]["output_size"] = 0
    _expect_reject(plan, changed, "package")

    changed = copy.deepcopy(payload)
    changed["steps"][5]["output_sha256"] = "not-a-sha"
    _expect_reject(plan, changed, "package")


def test_receipt_producer_validates_then_publishes_atomically():
    plan = _plan()
    authorization = _authorization(plan)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "execution_receipt.json"
        summary = receipt.produce_execution_receipt(
            plan=plan,
            step_results=_steps(plan),
            output_path=output,
            authorization_record=authorization,
        )
        assert summary["classification"] == "PASS"
        assert json.loads(output.read_text()) == _receipt(plan)
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                step_results=_steps(plan),
                output_path=output,
                authorization_record=authorization,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing receipt was overwritten")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation remote execution receipt tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
