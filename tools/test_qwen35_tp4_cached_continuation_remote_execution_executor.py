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
    "qwen35_tp4_cached_continuation_remote_execution_executor",
    "qwen35_tp4_cached_continuation_remote_execution_executor.py",
)
authorization = _load(
    "qwen35_tp4_engine_remote_execution_authorization_for_cached_test",
    "qwen35_tp4_engine_remote_execution_authorization.py",
)
receipt = _load(
    "qwen35_tp4_cached_continuation_remote_execution_receipt_for_executor",
    "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
)


def _plan(root):
    order = [
        "reserve_remote",
        "upload",
        "stage",
        "resource_guard",
        "guarded_authority",
        "package_download",
        "safe_extract",
        "prepare_local_verifier",
        "local_verify",
    ]
    commands = {name: {"argv": [name]} for name in order}
    commands["upload"] = {"argv": [["scp", "a"], ["scp", "b"]]}
    commands["guarded_authority"] = {
        "authority_argv": ["env", "authority"],
        "ssh_argv": ["ssh", "guarded-authority"],
        "final_resource_recheck": True,
    }
    package = root / "cached-authority.tar"
    commands["package_download"] = {
        "remote_argv": ["ssh", "tar"],
        "local_output": str(package),
    }
    commands["safe_extract"]["argv"] = [
        "extract",
        str(package),
        str(root / "downloaded"),
    ]
    commands["prepare_local_verifier"]["argv"] = [
        "prepare",
        str(root / "verifier-source"),
    ]
    return {
        "schema_version": (
            "qwen35.tp4-cached-continuation-remote-execution-plan.v1"
        ),
        "run_tag": "cached-executor-r1",
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "ports": {"dist_port": 32001, "master_port": 32002},
        "local_inputs": {
            "workload_manifest_sha256": "c" * 64,
        },
        "command_order": order,
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
        "checks": {"row_count": 19, "token_match": True},
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


def _runner(events):
    def run(*, name, argv, stdout_path=None, env=None):
        events.append((name, argv, stdout_path, env))
        stdout = ""
        extra = {}
        if name == "package_download":
            path = Path(stdout_path)
            path.write_bytes(b"cached-authority-tar")
            extra = {
                "output_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "output_size": path.stat().st_size,
            }
        elif name == "resource_guard":
            stdout = json.dumps(_resource())
        elif name == "guarded_authority":
            stdout = "\n".join([
                "QWEN35_FINAL_RESOURCE_JSON="
                + json.dumps(_resource()),
                json.dumps(_pass_payload()),
            ])
        elif name == "local_verify":
            stdout = json.dumps(_pass_payload())
        return {
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            **extra,
        }

    return run


def _consumed_authorization(plan):
    return {
        "schema_version": (
            "qwen35.tp4-engine-remote-execution-authorization.v1"
        ),
        "classification": "AUTHORIZED",
        "plan_sha256": receipt._canonical_sha(plan),
        "run_tag": plan["run_tag"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "workload_manifest_sha256": (
            plan["local_inputs"]["workload_manifest_sha256"]
        ),
        "gpu_indices": plan["gpu_indices"],
        "ports": plan["ports"],
        "nonce": "cached-direct-r1",
        "consumed": True,
    }


def test_executor_requires_injected_runner_and_executes_frozen_order():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        try:
            executor.execute_plan(
                plan=plan,
                output_path=root / "never.json",
                authorization_record=_consumed_authorization(plan),
            )
        except TypeError as error:
            assert "command_runner" in str(error)
        else:
            raise AssertionError("executor accepted no runner")
        events = []
        summary = executor.execute_plan(
            plan=plan,
            output_path=root / "receipt.json",
            command_runner=_runner(events),
            authorization_record=_consumed_authorization(plan),
        )
        assert summary["classification"] == "PASS"
        assert [event[0] for event in events] == [
            "reserve_remote",
            "upload[0]",
            "upload[1]",
            "stage",
            "resource_guard",
            "guarded_authority",
            "package_download",
            "safe_extract",
            "prepare_local_verifier",
            "local_verify",
        ]
        assert all(event[3] == {} for event in events)


def test_verified_entrypoint_verifies_before_consuming_authorization():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        calls = []
        events = []
        authorization_path = root / "authorization.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=authorization_path,
            nonce="cached-verified-r1",
        )

        def verify(path):
            calls.append(("verify", Path(path)))
            assert authorization_path.exists()
            return plan

        executor.execute_verified_plan_file(
            plan_path=root / "plan.json",
            authorization_path=authorization_path,
            consumed_authorization_path=(
                root / "authorization.consumed.json"
            ),
            output_path=root / "receipt.json",
            failure_path=root / "failure.json",
            command_runner=_runner(events),
            plan_verifier=verify,
            execution_env={
                "KRB5CCNAME": (
                    "FILE:/Users/bytedance/krb5cc_sitian"
                ),
            },
        )
        assert calls == [("verify", root / "plan.json")]
        assert not authorization_path.exists()
        assert events


def test_verified_entrypoint_rejects_wrong_environment_before_runner():
    for environment in (
        None,
        {},
        {"KRB5CCNAME": "FILE:/tmp/wrong"},
        {
            "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
            "EXTRA": "forbidden",
        },
    ):
        events = []
        try:
            executor.execute_verified_plan_file(
                plan_path="/tmp/plan.json",
                authorization_path="/tmp/authorization.json",
                consumed_authorization_path="/tmp/consumed.json",
                output_path="/tmp/receipt.json",
                failure_path="/tmp/failure.json",
                command_runner=_runner(events),
                plan_verifier=lambda path: {},
                execution_env=environment,
            )
        except ValueError as error:
            assert "KRB5CCNAME" in str(error)
        else:
            raise AssertionError("wrong Kerberos environment accepted")
        assert events == []


def test_failure_is_bounded_authorization_bound_and_not_pass():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)

        def fail(**kwargs):
            if kwargs["name"] == "stage":
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "stage failed",
                }
            return _runner([])(**kwargs)

        failure_path = root / "failure.json"
        try:
            executor.execute_plan(
                plan=plan,
                output_path=root / "receipt.json",
                failure_path=failure_path,
                command_runner=fail,
                authorization_record=_consumed_authorization(plan),
            )
        except ValueError as error:
            assert "stage" in str(error)
        else:
            raise AssertionError("failed execution published PASS")
        failure = json.loads(failure_path.read_text())
        assert failure["classification"] == "FAILED"
        assert failure["failed_step"] == "stage"
        assert failure["authorization_nonce"] == "cached-direct-r1"
        assert not (root / "receipt.json").exists()


def test_executor_source_has_no_process_execution_surface():
    source = (
        TOOLS
        / "qwen35_tp4_cached_continuation_remote_execution_executor.py"
    ).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    assert "def main(" not in source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation remote execution executor tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
