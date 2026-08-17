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


authorization = _load(
    "qwen35_tp4_engine_remote_execution_authorization",
    "qwen35_tp4_engine_remote_execution_authorization.py",
)
receipt = _load(
    "qwen35_tp4_engine_remote_execution_receipt_for_auth_test",
    "qwen35_tp4_engine_remote_execution_receipt.py",
)


def _plan():
    return {
        "schema_version": "qwen35.tp4-engine-remote-execution-plan.v1",
        "run_tag": "authorized-r1",
        "source_tree_sha256": "a" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "ports": {"dist_port": 32001, "master_port": 32002},
        "local_inputs": {
            "configuration": "/tmp/configuration.json",
            "configuration_sha256": "d" * 64,
            "source_inventory": "/tmp/inventory.json",
            "source_inventory_sha256": "e" * 64,
            "source_tar": "/tmp/source.tar",
            "source_tar_sha256": "f" * 64,
            "workload_manifest": "/tmp/workload.json",
            "workload_manifest_sha256": "c" * 64,
        },
        "model_manifest_sha256": "b" * 64,
    }


def test_authorization_binds_plan_identities_and_nonce_atomically():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=output,
            nonce="run-once-20260729-r1",
        )
        assert json.loads(output.read_text()) == payload
        assert payload == {
            "schema_version": (
                "qwen35.tp4-engine-remote-execution-authorization.v1"
            ),
            "classification": "AUTHORIZED",
            "plan_sha256": receipt._canonical_sha(plan),
            "run_tag": "authorized-r1",
            "source_tree_sha256": "a" * 64,
            "model_manifest_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "gpu_indices": [0, 1, 2, 3],
            "ports": {"dist_port": 32001, "master_port": 32002},
            "nonce": "run-once-20260729-r1",
            "consumed": False,
        }
        assert authorization.validate_authorization(
            plan,
            payload,
        )["classification"] == "AUTHORIZED"


def test_authorization_binds_controlled_shared_policy_and_baseline():
    plan = {
        **_plan(),
        "resource_policy": "controlled_shared",
        "resource_baseline_sha256": "9" * 64,
    }
    plan["local_inputs"] = {
        **plan["local_inputs"],
        "resource_baseline": "/tmp/resource_baseline.json",
        "resource_baseline_sha256": "9" * 64,
    }
    with tempfile.TemporaryDirectory() as temporary:
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=Path(temporary) / "authorization.json",
            nonce="shared-r1",
        )
        assert payload["resource_policy"] == "controlled_shared"
        assert payload["resource_baseline_sha256"] == "9" * 64
        for field, value in (
            ("resource_policy", "strict_exclusive"),
            ("resource_baseline_sha256", "8" * 64),
        ):
            changed = copy.deepcopy(payload)
            changed[field] = value
            try:
                authorization.validate_authorization(plan, changed)
            except ValueError as error:
                assert "authorization" in str(error), str(error)
            else:
                raise AssertionError(
                    f"tampered {field} was accepted"
                )


def test_authorization_rejects_tamper_reuse_or_unsafe_nonce():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=output,
            nonce="safe-r1",
        )
        for field, value in (
            ("plan_sha256", "0" * 64),
            ("run_tag", "other"),
            ("source_tree_sha256", "0" * 64),
            ("gpu_indices", [4, 5, 6, 7]),
            ("consumed", True),
        ):
            changed = copy.deepcopy(payload)
            changed[field] = value
            try:
                authorization.validate_authorization(plan, changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"tampered {field} was accepted")
        for nonce in ("", "../escape", "with space", "semi;colon"):
            try:
                authorization.produce_authorization(
                    plan=plan,
                    output_path=Path(temporary) / f"{len(nonce)}.json",
                    nonce=nonce,
                )
            except ValueError as error:
                assert "nonce" in str(error)
            else:
                raise AssertionError("unsafe nonce was accepted")


def test_consume_authorization_is_atomic_and_single_use():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorization_path = root / "authorization.json"
        consumed_path = root / "authorization.consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=authorization_path,
            nonce="consume-r1",
        )
        result = authorization.consume_authorization(
            plan=plan,
            authorization_path=authorization_path,
            consumed_path=consumed_path,
        )
        assert result["consumed"] is True
        assert not authorization_path.exists()
        assert json.loads(consumed_path.read_text()) == result
        try:
            authorization.consume_authorization(
                plan=plan,
                authorization_path=authorization_path,
                consumed_path=root / "again.json",
            )
        except ValueError as error:
            assert "authorization" in str(error)
        else:
            raise AssertionError("authorization was reused")


def test_consume_authorization_supports_production_runtime_directory():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authorization_path = root / "children" / "authorization.json"
        consumed_path = root / "runtime" / "engine" / "consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=authorization_path,
            nonce="consume-production-r1",
        )
        result = authorization.consume_authorization(
            plan=plan,
            authorization_path=authorization_path,
            consumed_path=consumed_path,
        )
        assert result["consumed"] is True
        assert not authorization_path.exists()
        assert json.loads(consumed_path.read_text()) == result


def test_consume_claims_active_path_before_writing_consumed_payload():
    source = (
        TOOLS / "qwen35_tp4_engine_remote_execution_authorization.py"
    ).read_text(encoding="utf-8")
    body = source[source.index("def consume_authorization("):]
    rename_position = body.index("os.replace(authorization_path, claim_path)")
    consumed_write_position = body.index(
        "_rewrite_consumed_authorization("
    )
    assert rename_position < consumed_write_position


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine remote execution authorization tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
