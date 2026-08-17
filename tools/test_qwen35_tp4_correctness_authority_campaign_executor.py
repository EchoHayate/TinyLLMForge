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


executor = _load(
    "qwen35_tp4_correctness_authority_campaign_executor",
    "qwen35_tp4_correctness_authority_campaign_executor.py",
)
authorization = _load(
    "qwen35_tp4_correctness_authority_campaign_authorization_for_executor_test",
    "qwen35_tp4_correctness_authority_campaign_authorization.py",
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _fixture(root):
    children = []
    for index, name in enumerate(executor.CHILD_ORDER):
        child_root = root / name
        plan_path = child_root / "plan.json"
        active = child_root / "authorization.json"
        _write_json(plan_path, {"name": name})
        _write_json(active, {"consumed": False})
        children.append({
            "name": name,
            "run_tag": f"{name}-run",
            "plan_path": str(plan_path),
            "plan_sha256": executor.receipt._sha256(plan_path),
            "source_tree_sha256": str(index + 1) * 64,
            "model_manifest_sha256": "a" * 64,
            "authority_dir": str(child_root / "authority"),
            "authorization_path": str(active),
            "consumed_authorization_path": str(child_root / "consumed.json"),
            "receipt_path": str(child_root / "receipt.json"),
            "failure_path": str(child_root / "failure.json"),
        })
    plan = {
        "campaign_tag": "campaign-r1",
        "stage_order": list(executor.STAGE_ORDER),
        "child_order": list(executor.CHILD_ORDER),
        "children": children,
        "adapter_output_dir": str(root / "adapter"),
        "bundle_output_dir": str(root / "bundle"),
        "prerequisite_path": str(
            root / "bundle/correctness_prerequisites.json"
        ),
        "benchmark_execution_authorized": False,
    }
    campaign_active = root / "campaign_authorization.json"
    authorization.produce_authorization(
        plan=plan,
        output_path=campaign_active,
        nonce="operator-r1",
    )
    return plan, campaign_active


def _callbacks(root, events, *, fail_name=None):
    child_verifiers = {}
    child_executors = {}
    for name in executor.CHILD_ORDER:
        def execute(*, child, execution_env, name=name):
            events.append(name)
            assert not (root / "campaign_authorization.json").exists()
            if name == fail_name:
                raise RuntimeError(f"{name} failed")
            Path(child["authority_dir"]).mkdir(parents=True)
            _write_json(
                Path(child["consumed_authorization_path"]),
                {"consumed": True},
            )
            _write_json(
                Path(child["receipt_path"]),
                {"classification": "PASS"},
            )
            return {"classification": "PASS"}

        child_executors[name] = execute
        child_verifiers[name] = lambda **paths: {
            "classification": "PASS",
        }

    def adapt(*, runs, verification_output_dir):
        events.append("adapt_authorities")
        Path(verification_output_dir).mkdir()
        rows = []
        for run in runs:
            path = Path(verification_output_dir) / f"{run['name']}.json"
            _write_json(path, {"classification": "PASS"})
            rows.append({
                "name": run["name"],
                "run_tag": run["run_tag"],
                "source_tree_sha256": "b" * 64,
                "artifact_path": str(path),
                "artifact_sha256": executor.receipt._sha256(path),
                "independent_verification_path": str(path),
                "independent_verification_sha256": (
                    executor.receipt._sha256(path)
                ),
                "provenance_path": str(path),
                "provenance_sha256": executor.receipt._sha256(path),
            })
        return rows

    def build(*, authorities, output_dir):
        events.append("build_bundle")
        path = Path(output_dir) / "correctness_prerequisites.json"
        _write_json(path, {"classification": "PASS"})
        return {
            "classification": "PASS",
            "prerequisite_path": str(path),
            "prerequisite_sha256": executor.receipt._sha256(path),
            "owned_files": ["correctness_prerequisites.json"],
        }

    def validate(path):
        events.append("verify_bundle")
        return {"classification": "PASS", "authorized": True}

    return child_executors, child_verifiers, adapt, build, validate


def test_executor_consumes_first_runs_serially_and_publishes_receipt():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, active = _fixture(root)
        events = []
        callbacks = _callbacks(root, events)
        summary = executor.execute_verified_campaign_file(
            plan_path=root / "campaign_plan.json",
            authorization_path=active,
            consumed_authorization_path=root / "campaign_consumed.json",
            receipt_path=root / "campaign_receipt.json",
            failure_path=root / "campaign_failure.json",
            plan_verifier=lambda path: plan,
            child_executors=callbacks[0],
            child_receipt_verifiers=callbacks[1],
            adapt_callback=callbacks[2],
            build_callback=callbacks[3],
            prerequisite_validator=callbacks[4],
            execution_env=executor.REQUIRED_EXECUTION_ENV,
        )
        assert events == [
            *executor.CHILD_ORDER,
            "adapt_authorities",
            "build_bundle",
            "verify_bundle",
            "verify_bundle",
        ]
        assert summary["classification"] == "PASS"
        assert (root / "campaign_receipt.json").is_file()
        assert not (root / "campaign_failure.json").exists()


def test_executor_stops_on_first_failure_and_writes_prefix():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, active = _fixture(root)
        events = []
        callbacks = _callbacks(
            root,
            events,
            fail_name="cached_continuation",
        )
        try:
            executor.execute_verified_campaign_file(
                plan_path=root / "campaign_plan.json",
                authorization_path=active,
                consumed_authorization_path=root / "campaign_consumed.json",
                receipt_path=root / "campaign_receipt.json",
                failure_path=root / "campaign_failure.json",
                plan_verifier=lambda path: plan,
                child_executors=callbacks[0],
                child_receipt_verifiers=callbacks[1],
                adapt_callback=callbacks[2],
                build_callback=callbacks[3],
                prerequisite_validator=callbacks[4],
                execution_env=executor.REQUIRED_EXECUTION_ENV,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("campaign failure was not propagated")
        assert events == ["tp4_root_logit", "cached_continuation"]
        failure = json.loads((root / "campaign_failure.json").read_text())
        assert failure["failed_stage"] == "cached_continuation"
        assert [row["name"] for row in failure["completed_stages"]] == [
            "root_logit"
        ]
        assert not (root / "campaign_receipt.json").exists()
        assert not Path(plan["bundle_output_dir"]).exists()


def test_executor_rejects_wrong_environment_before_consumption():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, active = _fixture(root)
        callbacks = _callbacks(root, [])
        try:
            executor.execute_verified_campaign_file(
                plan_path=root / "campaign_plan.json",
                authorization_path=active,
                consumed_authorization_path=root / "campaign_consumed.json",
                receipt_path=root / "campaign_receipt.json",
                failure_path=root / "campaign_failure.json",
                plan_verifier=lambda path: plan,
                child_executors=callbacks[0],
                child_receipt_verifiers=callbacks[1],
                adapt_callback=callbacks[2],
                build_callback=callbacks[3],
                prerequisite_validator=callbacks[4],
                execution_env={},
            )
        except ValueError as error:
            assert "KRB5CCNAME" in str(error), str(error)
        else:
            raise AssertionError("wrong environment was accepted")
        assert active.is_file()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 correctness authority campaign executor tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
