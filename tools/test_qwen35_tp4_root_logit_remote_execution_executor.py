from __future__ import annotations

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


planner = _load(
    "qwen35_tp4_root_logit_remote_execution_plan_for_executor_test",
    "qwen35_tp4_root_logit_remote_execution_plan.py",
)
authorization = _load(
    "qwen35_tp4_root_logit_remote_execution_authorization_for_executor_test",
    "qwen35_tp4_root_logit_remote_execution_authorization.py",
)
receipt = _load(
    "qwen35_tp4_root_logit_remote_execution_receipt_for_executor_test",
    "qwen35_tp4_root_logit_remote_execution_receipt.py",
)
executor = _load(
    "qwen35_tp4_root_logit_remote_execution_executor",
    "qwen35_tp4_root_logit_remote_execution_executor.py",
)
receipt_fixture = _load(
    "qwen35_tp4_root_receipt_fixture_for_executor_test",
    "test_qwen35_tp4_root_logit_remote_execution_receipt.py",
)


def _plan_and_auth(root):
    plan = planner.build_remote_execution_plan(
        repo_root=root / "repo",
        output_dir=root / "plan",
        run_tag="root-logit-executor-r1",
    )
    active = root / "authorization.json"
    authorization.produce_authorization(
        plan=plan,
        output_path=active,
        nonce="root-logit-executor-nonce",
    )
    return plan, active


def _stage_payloads(root):
    fixture_root = root / "fixture"
    plan, _, _, stages = receipt_fixture._fixture(fixture_root)
    return plan, {row["name"]: row["result"] for row in stages}


def _bind_payloads(plan, payloads):
    payloads["preflight"]["run_tag"] = plan["run_tag"]
    payloads["preflight"]["frozen_source_tag"] = (
        plan["frozen_source_tag"]
    )
    payloads["preflight"]["frozen_source_tree_sha256"] = (
        plan["frozen_source_tree_sha256"]
    )
    payloads["preflight"]["source_tree_sha256"] = (
        plan["frozen_source_tree_sha256"]
    )
    payloads["run"]["run_tag"] = plan["run_tag"]
    payloads["run"]["remote_run_dir"] = plan["remote_run_dir"]
    return payloads


def test_executor_consumes_before_callbacks_and_publishes_pass():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, active = _plan_and_auth(root)
        events = []
        consumed = root / "consumed_authorization.json"

        fixture_plan, fixture_payloads = _stage_payloads(root / "payloads")
        fixture_local_run = Path(fixture_plan["local_run_dir"])
        payloads = _bind_payloads(plan, fixture_payloads)

        def stage_runner(*, name, plan, execution_env):
            assert consumed.is_file()
            events.append(name)
            local_run = Path(plan["local_run_dir"])
            if name == "preflight":
                local_run.mkdir(parents=True)
                receipt_fixture._write(
                    local_run / "remote_resource_preflight.json",
                    payloads[name],
                )
            elif name == "run":
                receipt_fixture._write(
                    local_run / "remote_run.json",
                    payloads[name],
                )
            elif name == "download":
                (local_run / "artifacts").mkdir()
                for source in (
                    fixture_local_run / "artifacts"
                ).iterdir():
                    (local_run / "artifacts" / source.name).write_bytes(
                        source.read_bytes()
                    )
                receipt_fixture._write(
                    local_run / "download.json",
                    payloads[name],
                )
            else:
                receipt_fixture._write(
                    local_run / "independent_verification.json",
                    payloads[name],
                )
            return payloads[name]

        output = root / "execution_receipt.json"
        result = executor.execute_verified_plan_file(
            plan_path=root / "plan" / planner.PLAN_NAME,
            authorization_path=active,
            consumed_authorization_path=consumed,
            receipt_path=output,
            failure_path=root / "failure.json",
            stage_runner=stage_runner,
            plan_verifier=planner.verify_remote_execution_plan,
            execution_env=executor.REQUIRED_EXECUTION_ENV,
            root_verifier=lambda _path: payloads["verify"],
        )

        assert events == plan["stage_order"]
        assert result["classification"] == "PASS"
        assert output.is_file()
        assert not (root / "failure.json").exists()


def test_executor_rejects_environment_or_existing_targets():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, active = _plan_and_auth(root)
        kwargs = {
            "plan_path": root / "plan" / planner.PLAN_NAME,
            "authorization_path": active,
            "consumed_authorization_path": (
                root / "consumed_authorization.json"
            ),
            "receipt_path": root / "execution_receipt.json",
            "failure_path": root / "failure.json",
            "stage_runner": lambda **_kwargs: {},
            "plan_verifier": planner.verify_remote_execution_plan,
            "root_verifier": lambda _path: {},
        }
        try:
            executor.execute_verified_plan_file(
                **kwargs,
                execution_env={},
            )
        except ValueError as error:
            assert "KRB5CCNAME" in str(error), str(error)
        else:
            raise AssertionError("wrong execution environment was accepted")

        kwargs["receipt_path"].write_text("{}")
        try:
            executor.execute_verified_plan_file(
                **kwargs,
                execution_env=executor.REQUIRED_EXECUTION_ENV,
            )
        except ValueError as error:
            assert "exists" in str(error), str(error)
        else:
            raise AssertionError("existing output was accepted")


def test_executor_failure_is_prefix_preserving():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, active = _plan_and_auth(root)
        consumed = root / "consumed_authorization.json"
        events = []

        def stage_runner(*, name, plan, execution_env):
            assert consumed.is_file()
            events.append(name)
            if name == "download":
                raise RuntimeError("download failed")
            return {"stage": name}

        failure = root / "failure.json"
        try:
            executor.execute_verified_plan_file(
                plan_path=root / "plan" / planner.PLAN_NAME,
                authorization_path=active,
                consumed_authorization_path=consumed,
                receipt_path=root / "execution_receipt.json",
                failure_path=failure,
                stage_runner=stage_runner,
                plan_verifier=planner.verify_remote_execution_plan,
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                root_verifier=lambda _path: {},
            )
        except RuntimeError as error:
            assert "download failed" in str(error), str(error)
        else:
            raise AssertionError("executor failure was swallowed")

        payload = json.loads(failure.read_text())
        assert events == ["preflight", "run", "download"]
        assert payload["classification"] == "FAILED"
        assert payload["failed_stage"] == "download"
        assert [
            row["name"] for row in payload["completed_stages"]
        ] == ["preflight", "run"]
        assert not (root / "execution_receipt.json").exists()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 root-logit remote execution executor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
