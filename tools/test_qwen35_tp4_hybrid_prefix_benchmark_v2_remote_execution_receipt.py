from __future__ import annotations

import copy
import importlib.util
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
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_receipt_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
contract_fixture = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_fixture_for_receipt",
    "test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
receipt = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py",
)


def _inventory(paths, prefix):
    return [
        {
            "path": path,
            "sha256": f"{prefix}{index:063x}"[-64:],
            "bytes": index + 1,
            "type": "file",
        }
        for index, path in enumerate(sorted(paths), start=1)
    ]


def _receipt_payload():
    package_inventory = _inventory(
        contract.ARTIFACT_MANIFEST_HASH_DOMAIN,
        "1",
    )
    final_inventory = _inventory(contract.PRODUCER_TRUST_DOMAIN, "2")
    package_by_path = {row["path"]: row for row in package_inventory}
    for row in final_inventory:
        if row["path"] in package_by_path:
            row.update(package_by_path[row["path"]])
    payload = {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
            "execution_receipt"
        ],
        "classification": "PASS",
        "run_tag": "v2-receipt-r1",
        "nonce": "nonce-v2-receipt-r1",
        **{
            field: f"{index:064x}"
            for index, field in enumerate(
                contract.EXECUTION_PROVENANCE_FIELDS, start=1
            )
        },
        "execution_plan_sha256": "a" * 64,
        "consumed_authorization_sha256": "b" * 64,
        "authorization_id": "auth-v2-r1",
        "command_order": list(contract.EXECUTION_COMMAND_ORDER),
        "command_results": [],
        "artifact_paths": {
            "remote_run": "remote/v2-receipt-r1/run",
            "remote_artifact": "remote/v2-receipt-r1/run/artifact",
            "package": "local/v2-receipt-r1/package.tar",
            "local_extract": "local/v2-receipt-r1/run",
        },
        "source_inventory": [],
        "package_inventory": package_inventory,
        "final_inventory": final_inventory,
        "package_inventory_sha256": contract.canonical_json_sha256(
            package_inventory
        ),
        "final_inventory_sha256": contract.canonical_json_sha256(
            final_inventory
        ),
        "resource_guard_before_sha256": "c" * 64,
        "resource_guard_after_sha256": "d" * 64,
        "remote_path_created": True,
        "source_staged": True,
        "worker_launched": True,
        "cleanup_complete": True,
    }
    return payload


def test_detached_receipt_inventory_domains_are_closed_and_acyclic():
    payload = _receipt_payload()
    detached_path = "authority/v2-receipt-r1/receipt.json"
    package_paths = [row["path"] for row in payload["package_inventory"]]
    final_paths = [row["path"] for row in payload["final_inventory"]]
    assert package_paths == sorted(contract.ARTIFACT_MANIFEST_HASH_DOMAIN)
    assert final_paths == sorted(contract.PRODUCER_TRUST_DOMAIN)
    assert detached_path not in package_paths
    assert detached_path not in final_paths
    assert not set(contract.VERIFIER_TRUST_DOMAIN) & set(package_paths)
    assert not set(contract.VERIFIER_TRUST_DOMAIN) & set(final_paths)
    assert "artifact_manifest.json" not in package_paths
    assert "artifact_manifest.json" in final_paths
    assert payload["package_inventory"] == [
        row
        for row in payload["final_inventory"]
        if row["path"] != "artifact_manifest.json"
    ]
    receipt.validate_detached_inventory_domains(
        payload,
        detached_receipt_path=detached_path,
    )


def test_detached_receipt_rejects_receipt_or_verifier_in_producer_domain():
    for injected in (
        "authority/v2-receipt-r1/receipt.json",
        contract.VERIFIER_TRUST_DOMAIN[0],
    ):
        payload = _receipt_payload()
        payload["final_inventory"].append(
            {
                "path": injected,
                "sha256": "e" * 64,
                "bytes": 1,
                "type": "file",
            }
        )
        payload["final_inventory"].sort(key=lambda row: row["path"])
        payload["final_inventory_sha256"] = contract.canonical_json_sha256(
            payload["final_inventory"]
        )
        try:
            receipt.validate_detached_inventory_domains(
                payload,
                detached_receipt_path=(
                    "authority/v2-receipt-r1/receipt.json"
                ),
            )
        except ValueError as error:
            assert "inventory" in str(error).lower(), str(error)
        else:
            raise AssertionError(f"injected path was accepted: {injected}")


def test_detached_receipt_rejects_package_final_drift_or_hash_cycle():
    payload = _receipt_payload()
    payload["final_inventory"][1]["sha256"] = "f" * 64
    payload["final_inventory_sha256"] = contract.canonical_json_sha256(
        payload["final_inventory"]
    )
    try:
        receipt.validate_detached_inventory_domains(
            payload,
            detached_receipt_path=(
                "authority/v2-receipt-r1/receipt.json"
            ),
        )
    except ValueError as error:
        assert "equality" in str(error).lower(), str(error)
    else:
        raise AssertionError("package/final inventory drift was accepted")

    payload = _receipt_payload()
    try:
        receipt.validate_detached_inventory_domains(
            payload,
            detached_receipt_path="artifact_manifest.json",
        )
    except ValueError as error:
        assert "detached" in str(error).lower(), str(error)
    else:
        raise AssertionError("receipt hash cycle was accepted")


def test_receipt_publication_is_atomic_and_outside_run_directory():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=root / "authorization",
            artifact_root=root,
        )
        run_dir = (
            root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        run_dir.mkdir(parents=True)
        detached = root / "authority" / "receipt.json"
        receipt.publish_execution_evidence_bundle(
            bundle=bundle,
            run_dir=run_dir,
            output_path=detached,
            artifact_root=root,
        )
        assert detached.exists()
        assert not (run_dir / detached.name).exists()
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=run_dir,
                output_path=detached,
                artifact_root=root,
            )
        except ValueError as error:
            assert "exists" in str(error).lower(), str(error)
        else:
            raise AssertionError("detached receipt was overwritten")

        inside = run_dir / "receipt.json"
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=run_dir,
                output_path=inside,
                artifact_root=root,
            )
        except ValueError as error:
            assert "outside" in str(error).lower(), str(error)
        else:
            raise AssertionError("run-local receipt was accepted")


def test_receipt_validation_does_not_mutate_payload():
    payload = _receipt_payload()
    before = copy.deepcopy(payload)
    receipt.validate_detached_inventory_domains(
        payload,
        detached_receipt_path="authority/v2-receipt-r1/receipt.json",
    )
    assert payload == before


def test_legacy_private_receipt_publisher_is_disabled():
    payload = _receipt_payload()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        run_dir.mkdir()
        output = root / "authority" / "receipt.json"
        try:
            receipt.publish_detached_execution_receipt(
                payload=payload,
                run_dir=run_dir,
                output_path=output,
            )
        except ValueError as error:
            assert "bundle" in str(error).lower() or (
                "disabled" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("private receipt authority was published")
        assert not output.exists()


def test_complete_success_bundle_is_contract_validated_before_publish():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=root / "authorization",
            artifact_root=root,
        )
        run_dir = (
            root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        run_dir.mkdir(parents=True)
        output = root / "authority" / "execution_receipt.json"
        result = receipt.publish_execution_evidence_bundle(
            bundle=bundle,
            run_dir=run_dir,
            output_path=output,
            artifact_root=root,
        )
        contract.validate_execution_evidence_bundle(result)
        assert output.exists()
        assert not (run_dir / output.name).exists()


def test_complete_bundle_tamper_is_rejected_without_receipt_output():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    bundle["execution_receipt"]["command_results"][0][
        "command_sha256"
    ] = "f" * 64
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=root / "authorization",
            artifact_root=root,
        )
        run_dir = root / "run"
        run_dir.mkdir()
        output = root / "authority" / "execution_receipt.json"
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=run_dir,
                output_path=output,
                artifact_root=root,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("tampered lifecycle bundle was published")
        assert not output.exists()


def test_detached_bundle_rejects_actual_local_extract_domain():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=root / "authorization",
            artifact_root=root,
        )
        unrelated_run_dir = root / "unrelated"
        unrelated_run_dir.mkdir()
        output = (
            root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
            / "execution_receipt.json"
        )
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=unrelated_run_dir,
                output_path=output,
                artifact_root=root,
            )
        except ValueError as error:
            assert "artifact" in str(error).lower() or (
                "detached" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("receipt was published inside local_extract")
        assert not output.exists()


def test_detached_bundle_rejects_remote_run_with_explicit_artifact_root():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=root / "authorization",
            artifact_root=root,
        )
        actual_local_extract = (
            root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        actual_local_extract.mkdir(parents=True)
        output = (
            root
            / bundle["execution_plan"]["artifact_paths"]["remote_run"]
            / "execution_receipt.json"
        )
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=actual_local_extract,
                output_path=output,
                artifact_root=root,
            )
        except ValueError as error:
            assert "artifact" in str(error).lower() or (
                "detached" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("receipt was published inside remote_run")
        assert not output.exists()


def test_detached_bundle_rejects_false_artifact_root_rebinding():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        actual_root = Path(temporary) / "actual"
        false_root = Path(temporary) / "false"
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=actual_root / "authorization",
            artifact_root=actual_root,
        )
        actual_local_extract = (
            actual_root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        actual_local_extract.mkdir(parents=True)
        false_root.mkdir()
        output = actual_local_extract / "execution_receipt.json"
        unrelated_run_dir = actual_root / "unrelated-run"
        unrelated_run_dir.mkdir()
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=unrelated_run_dir,
                output_path=output,
                artifact_root=false_root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "artifact" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("false artifact root rebinding was accepted")
        assert not output.exists()


def test_detached_bundle_rejects_self_consistent_false_artifact_root():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        actual_root = Path(temporary) / "actual"
        false_root = Path(temporary) / "false"
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=actual_root / "authorization",
            artifact_root=actual_root,
        )
        actual_local_extract = (
            actual_root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        actual_local_extract.mkdir(parents=True)
        false_run_dir = (
            false_root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        false_run_dir.mkdir(parents=True)
        output = actual_local_extract / "execution_receipt.json"
        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=false_run_dir,
                output_path=output,
                artifact_root=false_root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "artifact" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "self-consistent false artifact root was accepted"
            )
        assert not output.exists()


def test_detached_bundle_rejects_same_path_artifact_root_replacement():
    bundle = contract_fixture._execution_success_with_full_producer_domain()
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "actual"
        root.mkdir()
        contract_fixture._bind_execution_roots(
            bundle,
            authority_root=root / "authorization",
            artifact_root=root,
        )
        root.rename(base / "actual-old")
        root.mkdir()
        run_dir = (
            root
            / bundle["execution_plan"]["artifact_paths"]["local_extract"]
        )
        run_dir.mkdir(parents=True)
        output = base / "detached" / "execution_receipt.json"

        try:
            receipt.publish_execution_evidence_bundle(
                bundle=bundle,
                run_dir=run_dir,
                output_path=output,
                artifact_root=root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "identity" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "same-path replacement preserved artifact root identity"
            )
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
        f"qwen35 v2 detached receipt tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
