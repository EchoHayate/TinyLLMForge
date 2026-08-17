from __future__ import annotations

import importlib.util
import hashlib
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
    "qwen35_tp4_engine_remote_execution_executor",
    "qwen35_tp4_engine_remote_execution_executor.py",
)
authorization = _load(
    "qwen35_tp4_engine_remote_execution_authorization_for_executor_test",
    "qwen35_tp4_engine_remote_execution_authorization.py",
)
planner = _load(
    "qwen35_tp4_engine_remote_execution_plan_for_executor_test",
    "qwen35_tp4_engine_remote_execution_plan.py",
)
receipt = _load(
    "qwen35_tp4_engine_remote_execution_receipt_for_executor_test",
    "qwen35_tp4_engine_remote_execution_receipt.py",
)


def _plan():
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
    commands = {
        name: {"argv": [name]}
        for name in order
    }
    commands["upload"] = {
        "argv": [["scp", "a"], ["scp", "b"]],
    }
    commands["package_download"] = {
        "remote_argv": ["ssh", "tar"],
        "local_output": "/tmp/authority.tar",
    }
    commands["guarded_authority"] = {
        "authority_argv": ["env", "authority"],
        "ssh_argv": ["ssh", "guarded-authority"],
        "final_resource_recheck": True,
    }
    return {
        "schema_version": "qwen35.tp4-engine-remote-execution-plan.v1",
        "run_tag": "executor-r1",
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
        "model_manifest_sha256": "b" * 64,
        "source_tree_sha256": "a" * 64,
        "workload_manifest_sha256": "c" * 64,
        "reference_classification": "PASS",
        "engine_classification": "PASS",
    }


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
        "nonce": "direct-r1",
        "consumed": True,
    }


def _localize_plan(plan, root):
    plan = json.loads(json.dumps(plan))
    package = Path(root) / "authority.tar"
    downloaded = Path(root) / "downloaded_authority"
    verifier_source = Path(root) / "local_verifier_source"
    plan["commands"]["package_download"]["local_output"] = str(
        package
    )
    plan["commands"]["safe_extract"]["argv"] = [
        "extract",
        str(package),
        str(downloaded),
    ]
    plan["commands"]["prepare_local_verifier"]["argv"] = [
        "prepare",
        str(verifier_source),
    ]
    return plan


def _runner(events):
    resource = json.dumps({
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
    })

    def run(*, name, argv, stdout_path=None, env=None):
        events.append((name, argv, stdout_path, env))
        stdout = ""
        extra = {}
        if name == "package_download":
            path = Path(stdout_path)
            path.write_bytes(b"authority-tar")
            extra = {
                "output_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "output_size": path.stat().st_size,
            }
        if name == "resource_guard":
            stdout = resource
        elif name == "guarded_authority":
            stdout = "\n".join([
                f"QWEN35_FINAL_RESOURCE_JSON={resource}",
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


def test_executor_requires_injected_runner_and_executes_frozen_order():
    plan = _plan()
    try:
        executor.execute_plan(plan=plan, output_path="/tmp/never.json")
    except TypeError as error:
        assert "command_runner" in str(error)
    else:
        raise AssertionError("executor ran without injected runner")

    events = []
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"
        summary = executor.execute_plan(
            plan=plan,
            output_path=output,
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
        payload = json.loads(output.read_text())
        assert [step["name"] for step in payload["steps"]] == (
            plan["command_order"]
        )
        package = next(
            step
            for step in payload["steps"]
            if step["name"] == "package_download"
        )
        assert package["output_size"] == len(b"authority-tar")
        assert package["output_sha256"] == hashlib.sha256(
            b"authority-tar"
        ).hexdigest()


def test_executor_maps_special_command_shapes_and_package_output():
    plan = _plan()
    events = []
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"
        executor.execute_plan(
            plan=plan,
            output_path=output,
            command_runner=_runner(events),
            authorization_record=_consumed_authorization(plan),
        )
        mapping = {
            name: (argv, stdout)
            for name, argv, stdout, _env in events
        }
        assert mapping["guarded_authority"][0] == [
            "ssh",
            "guarded-authority",
        ]
        assert mapping["package_download"] == (
            ["ssh", "tar"],
            "/tmp/authority.tar",
        )


def test_executor_failure_or_oversized_log_does_not_publish():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"

        def fail(**kwargs):
            if kwargs["name"] == "stage":
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "stage failed",
                }
            return _runner([])(**kwargs)

        try:
            executor.execute_plan(
                plan=plan,
                output_path=output,
                failure_path=Path(temporary) / "failed.json",
                command_runner=fail,
                authorization_record=_consumed_authorization(plan),
            )
        except ValueError as error:
            assert "stage" in str(error)
        else:
            raise AssertionError("failed execution published a receipt")
        assert not output.exists()
        failure = json.loads(
            (Path(temporary) / "failed.json").read_text()
        )
        assert failure["classification"] == "FAILED"
        assert failure["failed_step"] == "stage"
        assert failure["authorization_nonce"] == "direct-r1"
        assert failure["authorization_sha256"] == (
            receipt._canonical_sha(_consumed_authorization(plan))
        )
        assert [row["name"] for row in failure["completed_steps"]] == [
            "reserve_remote",
            "upload",
        ]
        assert executor.validate_failure_evidence(
            plan,
            failure,
            authorization_record=_consumed_authorization(plan),
        )["failed_step"] == "stage"

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "receipt.json"
        broken_plan = _plan()
        broken_plan["commands"]["package_download"][
            "local_output"
        ] = str(root / "authority.tar")

        def missing_output(**kwargs):
            result = _runner([])(**kwargs)
            if kwargs["name"] == "package_download":
                Path(kwargs["stdout_path"]).unlink()
                result.pop("output_sha256")
                result.pop("output_size")
            return result

        try:
            executor.execute_plan(
                plan=broken_plan,
                output_path=output,
                failure_path=root / "failed.json",
                command_runner=missing_output,
                authorization_record=_consumed_authorization(
                    broken_plan
                ),
            )
        except ValueError as error:
            assert "package output" in str(error)
        else:
            raise AssertionError("missing package output was accepted")
        failure = json.loads(
            (Path(temporary) / "failed.json").read_text()
        )
        assert failure["classification"] == "FAILED"
        assert failure["failed_step"] == "package_download"
        assert executor.validate_failure_evidence(
            broken_plan,
            failure,
            authorization_record=_consumed_authorization(
                broken_plan
            ),
        )["failed_step"] == "package_download"

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "receipt.json"

        def huge(**kwargs):
            result = _runner([])(**kwargs)
            if kwargs["name"] == "stage":
                result["stdout"] = "x" * (receipt.MAX_LOG_BYTES + 1)
            return result

        try:
            executor.execute_plan(
                plan=plan,
                output_path=output,
                failure_path=Path(temporary) / "failed.json",
                command_runner=huge,
                authorization_record=_consumed_authorization(plan),
            )
        except ValueError as error:
            assert "bounded" in str(error)
        else:
            raise AssertionError("oversized logs were published")
        assert not output.exists()
        failure = json.loads(
            (Path(temporary) / "failed.json").read_text()
        )
        assert failure["classification"] == "FAILED"
        assert failure["failed_step"] == "stage"
        assert len(failure["error"]) <= executor.MAX_ERROR_CHARS


def test_failed_package_download_needs_no_output_identity():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan()
        plan["commands"]["package_download"]["local_output"] = str(
            root / "authority.tar"
        )

        def fail_package(**kwargs):
            if kwargs["name"] == "package_download":
                return {
                    "returncode": 7,
                    "stdout": "",
                    "stderr": "download failed",
                }
            return _runner([])(**kwargs)

        try:
            executor.execute_plan(
                plan=plan,
                output_path=root / "receipt.json",
                failure_path=root / "failed.json",
                command_runner=fail_package,
                authorization_record=_consumed_authorization(plan),
            )
        except ValueError as error:
            assert "package_download command returncode" in str(error)
        else:
            raise AssertionError("failed package download was accepted")
        assert not (root / "authority.tar").exists()
        failure = json.loads((root / "failed.json").read_text())
        assert failure["failed_step"] == "package_download"


def test_failure_verifier_rejects_non_prefix_or_command_tamper():
    plan = _plan()
    completed = [
        {
            "name": name,
            "command_sha256": receipt._canonical_sha(
                plan["commands"][name]
            ),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }
        for name in plan["command_order"][:2]
    ]
    payload = {
        "schema_version": (
            "qwen35.tp4-engine-remote-execution-failure.v1"
        ),
        "classification": "FAILED",
        "plan_sha256": receipt._canonical_sha(plan),
        "authorization_sha256": receipt._canonical_sha(
            _consumed_authorization(plan)
        ),
        "authorization_nonce": "direct-r1",
        "run_tag": plan["run_tag"],
        "failed_step": "stage",
        "completed_steps": completed,
        "error": "stage command returncode mismatch",
    }
    assert executor.validate_failure_evidence(
        plan,
        payload,
        authorization_record=_consumed_authorization(plan),
    )["completed_step_count"] == 2

    changed = json.loads(json.dumps(payload))
    changed["failed_step"] = "local_verify"
    try:
        executor.validate_failure_evidence(
            plan,
            changed,
            authorization_record=_consumed_authorization(plan),
        )
    except ValueError as error:
        assert "prefix" in str(error)
    else:
        raise AssertionError("non-prefix failure was accepted")

    changed = json.loads(json.dumps(payload))
    changed["completed_steps"][0]["command_sha256"] = "f" * 64
    try:
        executor.validate_failure_evidence(
            plan,
            changed,
            authorization_record=_consumed_authorization(plan),
        )
    except ValueError as error:
        assert "command" in str(error)
    else:
        raise AssertionError("tampered completed command was accepted")


def test_failure_evidence_is_non_overwriting_and_not_a_pass_receipt():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        failure_path = root / "failed.json"
        failure_path.write_text("{}\n")
        try:
            executor.execute_plan(
                plan=plan,
                output_path=root / "receipt.json",
                failure_path=failure_path,
                command_runner=lambda **kwargs: {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "failed",
                },
                authorization_record=_consumed_authorization(plan),
            )
        except ValueError as error:
            assert "failure evidence already exists" in str(error)
        else:
            raise AssertionError("failure evidence was overwritten")
        assert failure_path.read_text() == "{}\n"


def test_executor_source_has_no_subprocess_execution_surface():
    source = (
        TOOLS / "qwen35_tp4_engine_remote_execution_executor.py"
    ).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source


def test_verified_file_entrypoint_verifies_before_any_command():
    calls = []
    original = planner.verify_remote_execution_plan
    try:
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            localized = _localize_plan(_plan(), root)
            planner.verify_remote_execution_plan = lambda path: (
                calls.append(("verify", Path(path))) or localized
            )
            plan_path = root / "plan.json"
            plan_path.write_text("{}\n")
            authorization_path = root / "authorization.json"
            authorization.produce_authorization(
                plan=localized,
                output_path=authorization_path,
                nonce="verified-r1",
            )
            output = root / "receipt.json"
            executor.execute_verified_plan_file(
                plan_path=plan_path,
                authorization_path=authorization_path,
                consumed_authorization_path=(
                    root / "authorization.consumed.json"
                ),
                output_path=output,
                failure_path=root / "failure.json",
                command_runner=_runner(events),
                plan_verifier=planner.verify_remote_execution_plan,
                execution_env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
            )
            assert calls == [("verify", plan_path)]
            assert events
            assert not authorization_path.exists()
            consumed_path = (
                root / "authorization.consumed.json"
            )
            assert consumed_path.is_file()
            consumed = json.loads(consumed_path.read_text())
            execution_receipt = json.loads(output.read_text())
            assert execution_receipt["authorization_nonce"] == (
                consumed["nonce"]
            )
            assert execution_receipt["authorization_sha256"] == (
                receipt._canonical_sha(consumed)
            )
    finally:
        planner.verify_remote_execution_plan = original

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        events = []

        def reject(path):
            raise ValueError("plan invalid")

        try:
            executor.execute_verified_plan_file(
                plan_path=root / "missing.json",
                authorization_path=root / "authorization.json",
                consumed_authorization_path=(
                    root / "authorization.consumed.json"
                ),
                output_path=root / "receipt.json",
                failure_path=root / "failure.json",
                command_runner=_runner(events),
                plan_verifier=reject,
                execution_env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
            )
        except ValueError as error:
            assert "plan invalid" in str(error)
        else:
            raise AssertionError("invalid plan reached command runner")
        assert events == []
        assert not (root / "receipt.json").exists()
        assert not (root / "failure.json").exists()


def test_verified_entrypoint_requires_exact_kerberos_environment():
    plan_path = Path("/tmp/plan.json")
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
                plan_path=plan_path,
                authorization_path=Path("/tmp/authorization.json"),
                consumed_authorization_path=Path(
                    "/tmp/authorization.consumed.json"
                ),
                output_path="/tmp/receipt.json",
                failure_path="/tmp/failure.json",
                command_runner=_runner(events),
                plan_verifier=lambda path: _plan(),
                execution_env=environment,
            )
        except ValueError as error:
            assert "KRB5CCNAME" in str(error)
        else:
            raise AssertionError("invalid execution environment accepted")
        assert events == []

    events = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        localized = _localize_plan(_plan(), root)
        authorization_path = root / "authorization.json"
        authorization.produce_authorization(
            plan=localized,
            output_path=authorization_path,
            nonce="environment-r1",
        )
        executor.execute_verified_plan_file(
            plan_path=root / "plan.json",
            authorization_path=authorization_path,
            consumed_authorization_path=(
                root / "authorization.consumed.json"
            ),
            output_path=root / "receipt.json",
            failure_path=root / "failure.json",
            command_runner=_runner(events),
            plan_verifier=lambda path: localized,
            execution_env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
            },
        )
        assert all(
            event[3] == {
                "KRB5CCNAME": (
                    "FILE:/Users/bytedance/krb5cc_sitian"
                ),
            }
            for event in events
        )


def test_verified_entrypoint_rejects_existing_local_outputs_pre_execution():
    targets = (
        "receipt",
        "failure",
        "package",
        "downloaded",
        "verifier_source",
    )
    for target in targets:
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _plan()
            plan["commands"]["package_download"][
                "local_output"
            ] = str(root / "authority.tar")
            plan["commands"]["safe_extract"]["argv"] = [
                "extract",
                str(root / "authority.tar"),
                str(root / "downloaded_authority"),
            ]
            plan["commands"]["prepare_local_verifier"]["argv"] = [
                "prepare",
                str(root / "local_verifier_source"),
            ]
            path = {
                "receipt": root / "receipt.json",
                "failure": root / "failure.json",
                "package": root / "authority.tar",
                "downloaded": root / "downloaded_authority",
                "verifier_source": root / "local_verifier_source",
            }[target]
            if target in {"downloaded", "verifier_source"}:
                path.mkdir()
            else:
                path.write_text("occupied\n")
            try:
                authorization_path = root / "authorization.json"
                authorization.produce_authorization(
                    plan=plan,
                    output_path=authorization_path,
                    nonce=f"existing-{target}",
                )
                executor.execute_verified_plan_file(
                    plan_path=root / "plan.json",
                    authorization_path=authorization_path,
                    consumed_authorization_path=(
                        root / "authorization.consumed.json"
                    ),
                    output_path=root / "receipt.json",
                    failure_path=root / "failure.json",
                    command_runner=_runner(events),
                    plan_verifier=lambda _path, plan=plan: plan,
                    execution_env={
                        "KRB5CCNAME": (
                            "FILE:/Users/bytedance/krb5cc_sitian"
                        ),
                    },
                )
            except ValueError as error:
                assert "local output" in str(error)
            else:
                raise AssertionError(
                    f"existing {target} output was accepted"
                )
            assert events == []


def test_verified_entrypoint_requires_matching_single_use_authorization():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _localize_plan(_plan(), root)
        events = []
        try:
            executor.execute_verified_plan_file(
                plan_path=root / "plan.json",
                authorization_path=root / "missing.json",
                consumed_authorization_path=root / "consumed.json",
                output_path=root / "receipt.json",
                failure_path=root / "failure.json",
                command_runner=_runner(events),
                plan_verifier=lambda path: plan,
                execution_env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
            )
        except ValueError as error:
            assert "authorization" in str(error)
        else:
            raise AssertionError("missing authorization was accepted")
        assert events == []

        authorization_path = root / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=authorization_path,
            nonce="single-use-r1",
        )
        payload["plan_sha256"] = "0" * 64
        authorization_path.write_text(json.dumps(payload) + "\n")
        try:
            executor.execute_verified_plan_file(
                plan_path=root / "plan.json",
                authorization_path=authorization_path,
                consumed_authorization_path=root / "consumed.json",
                output_path=root / "receipt.json",
                failure_path=root / "failure.json",
                command_runner=_runner(events),
                plan_verifier=lambda path: plan,
                execution_env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
            )
        except ValueError as error:
            assert "authorization" in str(error)
        else:
            raise AssertionError("tampered authorization was accepted")
        assert events == []


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine remote execution executor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
