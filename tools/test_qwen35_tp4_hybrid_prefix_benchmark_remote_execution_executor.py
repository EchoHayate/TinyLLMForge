from __future__ import annotations

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


executor = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_remote_execution_executor",
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_executor.py"
    ),
)
authorization = _load(
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_authorization_for_executor_test"
    ),
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_authorization.py"
    ),
)
receipt = _load(
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_receipt_for_executor_test"
    ),
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_receipt.py"
    ),
)


def _plan(root):
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
        name: {"argv": [name]}
        for name in order
    }
    commands["upload"] = {
        "argv": [["scp", "source"], ["scp", "inputs"]],
    }
    package = Path(root) / "benchmark-artifact.tar"
    downloaded = Path(root) / "downloaded-benchmark"
    commands["package_download"] = {
        "remote_argv": ["ssh", "tar"],
        "local_output": str(package),
    }
    commands["safe_extract"] = {
        "argv": ["extract", str(package), str(downloaded)],
    }
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-benchmark-"
            "remote-execution-plan.v1"
        ),
        "run_tag": "benchmark-executor-r1",
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


def _resource():
    return {
        "classification": "READY",
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": 25 * 1024**3,
                "compute_processes": [],
            }
            for index in range(4)
        ],
    }


def _verification():
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-performance-cache.v1"
        ),
        "classification": "NO_GO",
        "source_tree_sha256": "b" * 64,
        "model_manifest_sha256": "c" * 64,
        "workload_manifest_sha256": "d" * 64,
        "correctness_prerequisites_sha256": "a" * 64,
        "case_rows": 280,
        "process_rows": 70,
        "workloads": {
            "w0": {
                "median_ttft_ratio": 1.0,
                "max_repetition_ttft_ratio": 1.0,
                "throughput_ratio": 1.0,
                "median_e2e_ratio": 1.0,
                "median_decode_ratio": 1.0,
            },
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


def _runner(events):
    def run(*, name, argv, stdout_path=None, env=None):
        events.append((name, argv, stdout_path, env))
        stdout = ""
        extra = {}
        if name in {"resource_guard", "final_resource_guard"}:
            stdout = json.dumps(_resource())
        elif name == "workers":
            stdout = json.dumps({
                "classification": "COMPLETE",
                "case_ids": [
                    f"case-{index:02d}" for index in range(70)
                ],
            })
        elif name == "assembly":
            stdout = json.dumps({
                "classification": "ASSEMBLED",
                "case_rows": 280,
                "process_rows": 70,
            })
        elif name in {"remote_verify", "local_verify"}:
            stdout = json.dumps(_verification())
        elif name == "package_download":
            path = Path(stdout_path)
            path.write_bytes(b"benchmark-artifact")
            extra = {
                "output_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "output_size": path.stat().st_size,
            }
        return {
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            **extra,
        }
    return run


def _write_authorization(plan, root):
    path = Path(root) / "authorization.json"
    authorization.produce_authorization(
        plan=plan,
        output_path=path,
        nonce="benchmark-executor-r1",
    )
    return path


def test_executor_requires_explicit_runner_and_exact_environment():
    with tempfile.TemporaryDirectory() as temporary:
        plan = _plan(temporary)
        try:
            executor.execute_plan(
                plan=plan,
                output_path=Path(temporary) / "never.json",
            )
        except TypeError as error:
            assert "command_runner" in str(error), str(error)
        else:
            raise AssertionError("executor ran without injected runner")

        authorization_path = _write_authorization(plan, temporary)
        try:
            executor.execute_verified_plan_file(
                plan_path=Path(temporary) / "plan.json",
                authorization_path=authorization_path,
                consumed_authorization_path=(
                    Path(temporary) / "authorization.consumed.json"
                ),
                output_path=Path(temporary) / "receipt.json",
                failure_path=Path(temporary) / "failure.json",
                command_runner=_runner([]),
                plan_verifier=lambda path: plan,
                execution_env={},
            )
        except ValueError as error:
            assert "KRB5CCNAME" in str(error), str(error)
        else:
            raise AssertionError("wrong execution environment was accepted")


def test_verified_executor_consumes_before_commands_and_runs_frozen_order():
    events = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        active = _write_authorization(plan, root)
        consumed = root / "authorization.consumed.json"
        output = root / "receipt.json"
        summary = executor.execute_verified_plan_file(
            plan_path=plan_path,
            authorization_path=active,
            consumed_authorization_path=consumed,
            output_path=output,
            failure_path=root / "failure.json",
            command_runner=_runner(events),
            plan_verifier=lambda path: (
                plan if active.exists() else plan
            ),
            execution_env=executor.REQUIRED_EXECUTION_ENV,
        )

        assert summary["classification"] == "PASS"
        assert summary["benchmark_classification"] == "NO_GO"
        assert not active.exists()
        assert consumed.exists()
        assert [event[0] for event in events] == [
            "reserve_remote",
            "upload[0]",
            "upload[1]",
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
        assert all(
            event[3] == executor.REQUIRED_EXECUTION_ENV
            for event in events
        )


def test_executor_failure_publishes_only_bounded_prefix_evidence():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        consumed = {
            **authorization._payload(plan, "failure-r1"),
            "consumed": True,
        }
        output = root / "receipt.json"
        failure = root / "failure.json"

        def fail(**kwargs):
            if kwargs["name"] == "assembly":
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "assembly failed",
                }
            return _runner([])(**kwargs)

        try:
            executor.execute_plan(
                plan=plan,
                output_path=output,
                failure_path=failure,
                command_runner=fail,
                authorization_record=consumed,
                execution_env=executor.REQUIRED_EXECUTION_ENV,
            )
        except ValueError as error:
            assert "assembly" in str(error), str(error)
        else:
            raise AssertionError("failed execution published PASS")
        assert not output.exists()
        payload = json.loads(failure.read_text(encoding="utf-8"))
        assert payload["classification"] == "FAILED"
        assert payload["failed_step"] == "assembly"
        assert [row["name"] for row in payload["completed_steps"]] == [
            "reserve_remote",
            "upload",
            "stage",
            "resource_guard",
            "workers",
        ]
        assert executor.validate_failure_evidence(
            plan,
            payload,
            authorization_record=consumed,
        )["failed_step"] == "assembly"


def test_verified_executor_rejects_preexisting_local_outputs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        active = _write_authorization(plan, root)
        output = root / "receipt.json"
        output.write_text("existing\n")
        try:
            executor.execute_verified_plan_file(
                plan_path=plan_path,
                authorization_path=active,
                consumed_authorization_path=(
                    root / "authorization.consumed.json"
                ),
                output_path=output,
                failure_path=root / "failure.json",
                command_runner=_runner([]),
                plan_verifier=lambda path: plan,
                execution_env=executor.REQUIRED_EXECUTION_ENV,
            )
        except ValueError as error:
            assert "already exists" in str(error), str(error)
        else:
            raise AssertionError("preexisting output was accepted")
        assert active.exists()


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
        f"executor tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
