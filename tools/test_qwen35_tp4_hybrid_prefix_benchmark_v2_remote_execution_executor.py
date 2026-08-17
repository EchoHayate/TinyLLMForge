from __future__ import annotations

import copy
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


contract = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_executor_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
contract_fixture = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_fixture_for_executor",
    "test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
executor = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor.py",
)


def _plan_and_authorization():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    return (
        bundle["execution_plan"],
        bundle["consumed_authorization"],
    )


def _bind_template(template, root):
    artifact_root = Path(root)
    contract_fixture._bind_execution_roots(
        template,
        authority_root=artifact_root / "authorization",
        artifact_root=artifact_root,
    )
    run_dir = (
        artifact_root
        / template["execution_plan"]["artifact_paths"]["local_extract"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return artifact_root, run_dir


def _success_bundle_builder(template):
    def build(**kwargs):
        bundle = copy.deepcopy(template)
        bundle["execution_receipt"]["command_results"] = (
            kwargs["command_results"]
        )
        return bundle
    return build


def _command_results(fail_at=None):
    rows = []
    for name in contract.EXECUTION_COMMAND_ORDER:
        if fail_at is not None and name == fail_at:
            rows.append(
                {
                    "name": name,
                    "command_sha256": "a" * 64,
                    "outcome": "attempted",
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "failure",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
            )
            break
        rows.append(
            {
                "name": name,
                "command_sha256": "a" * 64,
                "outcome": "attempted",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
            }
        )
    for name in contract.EXECUTION_COMMAND_ORDER[len(rows):]:
        rows.append(
            {
                "name": name,
                "command_sha256": "a" * 64,
                "outcome": "skipped",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
            }
        )
    return rows


def test_executor_requires_injected_runner_and_exact_environment():
    try:
        executor.execute_plan(
            plan={},
            authorization_record={},
            detached_receipt_path=Path("/tmp/never.json"),
        )
    except TypeError as error:
        assert "command_runner" in str(error), str(error)
    else:
        raise AssertionError("executor ran without an injected runner")
    assert executor.REQUIRED_EXECUTION_ENV == {
        "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"
    }


def test_executor_runs_exact_order_with_bounded_outputs_and_no_kill():
    events = []
    template = contract_fixture._execution_success_with_full_producer_domain()

    def runner(*, name, argv, timeout_seconds, env):
        events.append((name, argv, timeout_seconds, env))
        return {
            "returncode": 0,
            "stdout": "x" * (contract.MAX_BOUNDED_OUTPUT_BYTES + 10),
            "stderr": "",
        }

    with tempfile.TemporaryDirectory() as temporary:
        artifact_root, run_dir = _bind_template(template, temporary)
        plan = template["execution_plan"]
        authorization_record = template["consumed_authorization"]
        expected_commands = contract.canonical_execution_commands(plan)
        output = artifact_root / "authority" / "receipt.json"

        published = executor.execute_plan(
            plan=plan,
            authorization_record=authorization_record,
            detached_receipt_path=output,
            artifact_root=artifact_root,
            run_dir=run_dir,
            command_runner=runner,
            execution_env=executor.REQUIRED_EXECUTION_ENV,
            receipt_builder=_success_bundle_builder(template),
        )
    assert [event[0] for event in events] == list(
        contract.EXECUTION_COMMAND_ORDER
    )
    summary = published["execution_receipt"]["command_results"]
    assert all(
        event[2] == contract.EXECUTION_COMMAND_TIMEOUT_SECONDS[event[0]]
        for event in events
    )
    assert all(
        event[3] == executor.REQUIRED_EXECUTION_ENV for event in events
    )
    assert all(
        event[1] == expected_commands[event[0]]
        for event in events
    )
    assert all(
        len(row["stdout"].encode("utf-8"))
        <= contract.MAX_BOUNDED_OUTPUT_BYTES
        for row in summary
    )
    assert all(row["stdout_truncated"] is True for row in summary)


def test_executor_failure_records_bounded_prefix_and_skips_suffix():
    events = []
    template = contract_fixture._stage_failure_evidence_bundle("assembly")

    def runner(*, name, argv, timeout_seconds, env):
        events.append(name)
        return {
            "returncode": 1 if name == "assembly" else 0,
            "stdout": "",
            "stderr": "assembly failed" if name == "assembly" else "",
        }

    with tempfile.TemporaryDirectory() as temporary:
        artifact_root, run_dir = _bind_template(template, temporary)
        plan = template["execution_plan"]
        authorization_record = template["consumed_authorization"]
        failure_path = artifact_root / "failure.json"
        try:
            executor.execute_plan(
                plan=plan,
                authorization_record=authorization_record,
                detached_receipt_path=artifact_root / "receipt.json",
                artifact_root=artifact_root,
                run_dir=run_dir,
                failure_path=failure_path,
                command_runner=runner,
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=_success_bundle_builder(template),
            )
        except ValueError as error:
            assert "assembly" in str(error), str(error)
        else:
            raise AssertionError("failed execution published success")
        assert events == list(contract.EXECUTION_COMMAND_ORDER[:6])
        assert failure_path.exists()
        contract.validate_execution_evidence_bundle(
            __import__("json").loads(
                failure_path.read_text(encoding="utf-8")
            )
        )


def test_executor_rejects_noncanonical_commands_before_runner_side_effects():
    events = []
    plan = {
        "command_order": list(contract.EXECUTION_COMMAND_ORDER),
        "commands": {
            name: ["echo", name]
            for name in contract.EXECUTION_COMMAND_ORDER
        },
    }
    plan["commands"]["workers"] = ["rm", "-rf", "/tmp/not-authorized"]
    try:
        executor.execute_plan(
            plan=plan,
            authorization_record={
                "consumed": True,
                "consumed_once": True,
            },
            detached_receipt_path=Path("/tmp/never.json"),
            artifact_root=Path("/tmp"),
            run_dir=Path("/tmp/run"),
            command_runner=lambda **kwargs: events.append(kwargs),
            execution_env=executor.REQUIRED_EXECUTION_ENV,
            receipt_builder=lambda **kwargs: kwargs,
        )
    except ValueError as error:
        assert "canonical" in str(error).lower() or (
            "authorization" in str(error).lower()
        ), str(error)
    else:
        raise AssertionError("noncanonical command plan was executed")
    assert events == []


def test_executor_rejects_incomplete_lifecycle_before_publisher():
    events = []
    template = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        artifact_root, run_dir = _bind_template(template, temporary)
        try:
            executor.execute_plan(
                plan=template["execution_plan"],
                authorization_record=template["consumed_authorization"],
                detached_receipt_path=artifact_root / "receipt.json",
                artifact_root=artifact_root,
                run_dir=run_dir,
                command_runner=lambda **kwargs: {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                },
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=lambda **kwargs: {
                    "classification": "PASS",
                },
            )
        except ValueError as error:
            assert "evidence" in str(error).lower() or (
                "bundle" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "incomplete lifecycle authority was published"
            )
    assert events == []


def test_executor_rejects_writer_bytes_different_from_validated_bundle():
    template = contract_fixture._execution_success_with_full_producer_domain()
    plan = template["execution_plan"]
    authorization_record = template["consumed_authorization"]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"

        def malicious_writer(*, payload, output_path):
            Path(output_path).write_text(
                json.dumps({"invalid": True}) + "\n",
                encoding="utf-8",
            )
            return payload

        try:
            executor.execute_plan(
                plan=plan,
                authorization_record=authorization_record,
                detached_receipt_path=output,
                command_runner=lambda **kwargs: {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                },
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=_success_bundle_builder(template),
                receipt_writer=malicious_writer,
            )
        except TypeError as error:
            assert "receipt_writer" in str(error), str(error)
        else:
            raise AssertionError("writer published bytes that were not validated")


def test_executor_rejects_writer_that_publishes_no_file():
    template = contract_fixture._execution_success_with_full_producer_domain()
    plan = template["execution_plan"]
    authorization_record = template["consumed_authorization"]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"
        try:
            executor.execute_plan(
                plan=plan,
                authorization_record=authorization_record,
                detached_receipt_path=output,
                command_runner=lambda **kwargs: {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                },
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=_success_bundle_builder(template),
                receipt_writer=lambda **kwargs: kwargs["payload"],
            )
        except TypeError as error:
            assert "receipt_writer" in str(error), str(error)
        else:
            raise AssertionError("missing receipt file was accepted")
        assert not output.exists()


def test_executor_rejects_receipt_inside_bound_artifact_domain():
    template = contract_fixture._execution_success_with_full_producer_domain()
    plan = template["execution_plan"]
    authorization_record = template["consumed_authorization"]
    with tempfile.TemporaryDirectory() as temporary:
        artifact_root = Path(temporary)
        contract_fixture._bind_execution_roots(
            template,
            authority_root=artifact_root / "authorization",
            artifact_root=artifact_root,
        )
        plan = template["execution_plan"]
        authorization_record = template["consumed_authorization"]
        run_dir = (
            artifact_root / plan["artifact_paths"]["local_extract"]
        )
        run_dir.mkdir(parents=True)
        output = run_dir / "execution_receipt.json"

        try:
            executor.execute_plan(
                plan=plan,
                authorization_record=authorization_record,
                detached_receipt_path=output,
                artifact_root=artifact_root,
                run_dir=run_dir,
                command_runner=lambda **kwargs: {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                },
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=_success_bundle_builder(template),
            )
        except ValueError as error:
            assert "artifact" in str(error).lower() or (
                "detached" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("executor accepted artifact-local receipt")
        assert not output.exists()


def test_executor_failure_does_not_follow_symlink_or_clobber_target():
    template = contract_fixture._stage_failure_evidence_bundle("assembly")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact_root, run_dir = _bind_template(template, root)
        plan = template["execution_plan"]
        authorization_record = template["consumed_authorization"]
        target = root / "unrelated.json"
        target.write_text("preserve\n", encoding="utf-8")
        failure_path = root / "failure.json"
        failure_path.symlink_to(target)
        try:
            executor.execute_plan(
                plan=plan,
                authorization_record=authorization_record,
                detached_receipt_path=root / "receipt.json",
                artifact_root=artifact_root,
                run_dir=run_dir,
                failure_path=failure_path,
                command_runner=lambda *, name, **kwargs: {
                    "returncode": 1 if name == "assembly" else 0,
                    "stdout": "",
                    "stderr": "assembly failed" if name == "assembly" else "",
                },
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=_success_bundle_builder(template),
            )
        except ValueError as error:
            assert "failure" in str(error).lower() or (
                "assembly" in str(error).lower()
            ) or "symlink" in str(error).lower(), str(error)
        else:
            raise AssertionError("failed execution did not reject symlink")
        assert target.read_text(encoding="utf-8") == "preserve\n"
        assert failure_path.is_symlink()


def test_executor_rejects_legacy_writer_without_bound_roots():
    template = contract_fixture._execution_success_with_full_producer_domain()
    plan = template["execution_plan"]
    authorization_record = template["consumed_authorization"]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"

        def writer(*, payload, output_path):
            Path(output_path).write_bytes(
                contract.canonical_json_bytes(payload) + b"\n"
            )
            return payload

        try:
            executor.execute_plan(
                plan=plan,
                authorization_record=authorization_record,
                detached_receipt_path=output,
                command_runner=lambda **kwargs: {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                },
                execution_env=executor.REQUIRED_EXECUTION_ENV,
                receipt_builder=_success_bundle_builder(template),
                receipt_writer=writer,
            )
        except (TypeError, ValueError) as error:
            assert "root" in str(error).lower() or (
                "publisher" in str(error).lower()
            ) or "writer" in str(error).lower(), str(error)
        else:
            raise AssertionError("legacy arbitrary receipt writer was accepted")
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
        f"qwen35 v2 remote executor tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
