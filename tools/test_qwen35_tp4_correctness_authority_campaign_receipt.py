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


receipt = _load(
    "qwen35_tp4_correctness_authority_campaign_receipt",
    "qwen35_tp4_correctness_authority_campaign_receipt.py",
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _fixture(root):
    children = []
    results = []
    verifiers = {}
    for index, name in enumerate(receipt.CHILD_ORDER):
        child_root = root / name
        plan = child_root / "plan.json"
        authorization = child_root / "consumed.json"
        execution_receipt = child_root / "receipt.json"
        authority_dir = child_root / "authority"
        authority_dir.mkdir(parents=True)
        _write_json(plan, {"name": name})
        _write_json(authorization, {"consumed": True})
        _write_json(execution_receipt, {"classification": "PASS"})
        row = {
            "name": name,
            "run_tag": f"{name}-run",
            "plan_path": str(plan),
            "plan_sha256": receipt._sha256(plan),
            "source_tree_sha256": str(index + 1) * 64,
            "model_manifest_sha256": "a" * 64,
            "authority_dir": str(authority_dir),
            "authorization_path": str(child_root / "active.json"),
            "consumed_authorization_path": str(authorization),
            "receipt_path": str(execution_receipt),
            "failure_path": str(child_root / "failure.json"),
        }
        children.append(row)
        results.append({
            "name": receipt.CHILD_STAGE_NAMES[name],
            "result": {
                "classification": "PASS",
                **row,
                "authorization_sha256": receipt._sha256(authorization),
                "receipt_sha256": receipt._sha256(execution_receipt),
            },
        })
        verifiers[name] = lambda **paths: {
            "classification": "PASS",
        }
    adapter_dir = root / "adapter"
    adapter_dir.mkdir()
    adapter_rows = []
    for name in receipt.CHILD_ORDER:
        path = adapter_dir / f"{name}.json"
        _write_json(path, {"classification": "PASS", "name": name})
        adapter_rows.append({
            "name": name,
            "run_tag": f"{name}-run",
            "source_tree_sha256": "b" * 64,
            "artifact_path": str(path),
            "artifact_sha256": receipt._sha256(path),
            "independent_verification_path": str(path),
            "independent_verification_sha256": receipt._sha256(path),
            "provenance_path": str(path),
            "provenance_sha256": receipt._sha256(path),
        })
    results.append({
        "name": "adapt_authorities",
        "result": {
            "classification": "PASS",
            "authorities": adapter_rows,
        },
    })
    bundle = root / "bundle"
    prerequisite = bundle / "correctness_prerequisites.json"
    _write_json(prerequisite, {"classification": "PASS"})
    results.append({
        "name": "build_bundle",
        "result": {
            "classification": "PASS",
            "prerequisite_path": str(prerequisite),
            "prerequisite_sha256": receipt._sha256(prerequisite),
            "owned_files": ["correctness_prerequisites.json"],
        },
    })
    results.append({
        "name": "verify_bundle",
        "result": {
            "classification": "PASS",
            "authorized": True,
            "prerequisite_sha256": receipt._sha256(prerequisite),
        },
    })
    plan = {
        "campaign_tag": "campaign-r1",
        "stage_order": list(receipt.STAGE_ORDER),
        "child_order": list(receipt.CHILD_ORDER),
        "children": children,
        "adapter_output_dir": str(adapter_dir),
        "bundle_output_dir": str(bundle),
        "prerequisite_path": str(prerequisite),
        "benchmark_execution_authorized": False,
    }
    authorization = {
        "consumed": True,
        "nonce": "operator-r1",
        "plan_sha256": receipt._canonical_sha(plan),
    }
    return plan, authorization, results, verifiers


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {fragment!r}"
        )


def test_receipt_binds_children_adapter_and_authorized_bundle():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization, results, verifiers = _fixture(root)
        path = root / "campaign_receipt.json"
        summary = receipt.produce_campaign_receipt(
            plan=plan,
            stage_results=results,
            authorization_record=authorization,
            output_path=path,
            child_receipt_verifiers=verifiers,
            prerequisite_validator=lambda path: {
                "classification": "PASS",
                "authorized": True,
            },
        )
        assert summary["classification"] == "PASS"
        assert summary["stage_count"] == 6
        assert summary["benchmark_execution_authorized"] is False
        assert path.is_file()


def test_receipt_preserves_controlled_shared_correctness_only_boundary():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization, results, verifiers = _fixture(root)
        plan.update({
            "resource_policy": "controlled_shared",
            "resource_baseline_sha256": "f" * 64,
        })
        authorization.update({
            "plan_sha256": receipt._canonical_sha(plan),
            "resource_policy": "controlled_shared",
            "resource_baseline_sha256": "f" * 64,
            "benchmark_execution_authorized": False,
        })
        summary = receipt.produce_campaign_receipt(
            plan=plan,
            stage_results=results,
            authorization_record=authorization,
            output_path=root / "campaign_receipt.json",
            child_receipt_verifiers=verifiers,
            prerequisite_validator=lambda path: {
                "classification": "PASS",
                "authorized": True,
            },
        )
        assert summary["resource_policy"] == "controlled_shared"
        assert summary["resource_baseline_sha256"] == "f" * 64
        assert summary["benchmark_execution_authorized"] is False


def test_receipt_rejects_child_and_bundle_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization, results, verifiers = _fixture(root)
        Path(plan["children"][0]["receipt_path"]).write_text(
            '{"classification":"DRIFT"}\n'
        )
        _expect_value_error(
            lambda: receipt.produce_campaign_receipt(
                plan=plan,
                stage_results=results,
                authorization_record=authorization,
                output_path=root / "receipt.json",
                child_receipt_verifiers=verifiers,
                prerequisite_validator=lambda path: {
                    "classification": "PASS",
                    "authorized": True,
                },
            ),
            "receipt SHA",
        )


def test_receipt_rejects_blocked_bundle_validation():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization, results, verifiers = _fixture(root)
        _expect_value_error(
            lambda: receipt.produce_campaign_receipt(
                plan=plan,
                stage_results=results,
                authorization_record=authorization,
                output_path=root / "receipt.json",
                child_receipt_verifiers=verifiers,
                prerequisite_validator=lambda path: {
                    "classification": "BLOCKED_CORRECTNESS",
                    "authorized": False,
                },
            ),
            "not authorized",
        )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 correctness authority campaign receipt tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
