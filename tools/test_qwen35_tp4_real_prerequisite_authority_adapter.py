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


adapter = _load(
    "qwen35_tp4_real_prerequisite_authority_adapter",
    "qwen35_tp4_real_prerequisite_authority_adapter.py",
)
builder = _load(
    "build_qwen35_tp4_performance_prerequisites_for_real_adapter_test",
    "build_qwen35_tp4_performance_prerequisites.py",
)
contract = _load(
    "qwen35_tp4_hybrid_prefix_contract_for_real_adapter_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
builder_fixture = _load(
    "qwen35_prerequisite_builder_fixture_for_real_adapter_test",
    "test_build_qwen35_tp4_performance_prerequisites.py",
)


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {fragment!r}"
        )


def test_adapter_rejects_naked_summary_files_as_authority_runs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        naked = root / "artifact.json"
        naked.write_text('{"classification":"PASS"}\n')
        run = adapter.RealAuthorityRun(
            name="tp4_root_logit",
            run_tag="root-run",
            authority_dir=naked,
        )

        _expect_value_error(
            lambda: adapter.adapt_real_authorities(
                runs=(run,),
                verification_output_dir=root / "verification",
            ),
            "directory",
        )


def test_adapter_requires_exact_three_authority_names():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authority = root / "root"
        authority.mkdir()
        run = adapter.RealAuthorityRun(
            name="tp4_root_logit",
            run_tag="root-run",
            authority_dir=authority,
        )

        _expect_value_error(
            lambda: adapter.adapt_real_authorities(
                runs=(run,),
                verification_output_dir=root / "verification",
            ),
            "inventory",
        )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _authority_dirs(root):
    root_dir = root / "root"
    cached_dir = root / "cached"
    engine_dir = root / "engine"
    for path in (
        root_dir,
        cached_dir,
        engine_dir / "engine_authority",
    ):
        path.mkdir(parents=True)
    root_artifact, _ = builder_fixture._root_payloads()
    cached_artifact, _ = builder_fixture._cached_payloads("c" * 64)
    engine_artifact, _ = builder_fixture._engine_payloads("d" * 64)
    _write_json(
        root_dir / "tp4_real_root_logit_correctness.json",
        root_artifact,
    )
    _write_json(
        root_dir / "source_manifest.json",
        {
            "source_tree_sha256": (
                contract.TP4_ROOT_SOURCE_TREE_SHA256
            ),
        },
    )
    _write_json(
        cached_dir / "cached_continuation_correctness.json",
        cached_artifact,
    )
    _write_json(
        engine_dir
        / "engine_authority"
        / "engine_correctness.json",
        engine_artifact,
    )
    return root_dir, cached_dir, engine_dir


def _receipt_files(root, name, authority_dir):
    plan = root / f"{name}-plan.json"
    authorization = root / f"{name}-authorization.json"
    receipt = root / f"{name}-receipt.json"
    _write_json(plan, {"downloaded": str(authority_dir)})
    _write_json(authorization, {"consumed": True})
    _write_json(receipt, {"classification": "PASS"})
    return plan, authorization, receipt


def _dependencies(
    *,
    root_dir,
    cached_dir,
    engine_dir,
    root_source=None,
    root_receipt_source=None,
    root_plan_dir=None,
    cached_source="c" * 64,
    cached_receipt_source=None,
    engine_source="d" * 64,
    cached_plan_dir=None,
):
    workload_sha = contract.canonical_json_file_sha256(
        contract.workload_manifest_payload()
    )
    _, cached_verification = builder_fixture._cached_payloads(
        cached_source
    )
    _, engine_verification = builder_fixture._engine_payloads(
        engine_source
    )
    return adapter.VerifierDependencies(
        root_verify=lambda path: {
            "classification": "PASS",
            "case_ids": list(contract.TP4_ROOT_CASE_IDS),
            "ranks": [0, 1, 2, 3],
            "checks": 10,
        },
        root_plan_verify=lambda path: {
            "run_tag": "root-run",
            "frozen_source_tree_sha256": (
                root_source or contract.TP4_ROOT_SOURCE_TREE_SHA256
            ),
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "stage_inputs": {
                "verify": {
                    "local_artifact_dir": str(
                        root_plan_dir or root_dir
                    ),
                }
            },
        },
        root_receipt_verify=lambda **paths: {
            "classification": "PASS",
            "run_tag": "root-run",
            "plan_sha256": "7" * 64,
            "authorization_sha256": "8" * 64,
            "source_tree_sha256": (
                root_receipt_source
                or root_source
                or contract.TP4_ROOT_SOURCE_TREE_SHA256
            ),
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        },
        cached_verify=lambda path: {
            **cached_verification,
        },
        engine_authority_verify=lambda path: {
            "classification": "PASS",
            "source_tree_sha256": engine_source,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": workload_sha,
            "reference_classification": "PASS",
            "engine_classification": "PASS",
        },
        engine_gate_verify=lambda path: {
            **engine_verification,
        },
        cached_plan_verify=lambda path: {
            "run_tag": "cached-run",
            "source_tree_sha256": cached_source,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "commands": {
                "local_verify": {
                    "argv": [
                        "python",
                        str(cached_plan_dir or cached_dir),
                    ]
                }
            },
        },
        cached_receipt_verify=lambda **paths: {
            "classification": "PASS",
            "run_tag": "cached-run",
            "plan_sha256": "1" * 64,
            "authorization_sha256": "2" * 64,
            "source_tree_sha256": (
                cached_receipt_source or cached_source
            ),
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": workload_sha,
        },
        engine_plan_verify=lambda path: {
            "run_tag": "engine-run",
            "source_tree_sha256": engine_source,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "commands": {
                "local_verify": {
                    "argv": ["python", str(engine_dir)]
                }
            },
        },
        engine_receipt_verify=lambda **paths: {
            "classification": "PASS",
            "run_tag": "engine-run",
            "plan_sha256": "4" * 64,
            "authorization_sha256": "5" * 64,
            "source_tree_sha256": engine_source,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": workload_sha,
        },
    )


def _runs(root, root_dir, cached_dir, engine_dir):
    root_receipt = _receipt_files(root, "root", root_dir)
    cached_receipt = _receipt_files(root, "cached", cached_dir)
    engine_receipt = _receipt_files(root, "engine", engine_dir)
    return (
        adapter.RealAuthorityRun(
            name="tp4_root_logit",
            run_tag="root-run",
            authority_dir=root_dir,
            plan_path=root_receipt[0],
            consumed_authorization_path=root_receipt[1],
            receipt_path=root_receipt[2],
        ),
        adapter.RealAuthorityRun(
            name="cached_continuation",
            run_tag="cached-run",
            authority_dir=cached_dir,
            plan_path=cached_receipt[0],
            consumed_authorization_path=cached_receipt[1],
            receipt_path=cached_receipt[2],
        ),
        adapter.RealAuthorityRun(
            name="engine_correctness",
            run_tag="engine-run",
            authority_dir=engine_dir,
            plan_path=engine_receipt[0],
            consumed_authorization_path=engine_receipt[1],
            receipt_path=engine_receipt[2],
        ),
    )


def test_adapter_derives_inputs_from_complete_verified_runs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        root_dir, cached_dir, engine_dir = _authority_dirs(root)
        original = adapter._VERIFIERS
        adapter._VERIFIERS = _dependencies(
            root_dir=root_dir,
            cached_dir=cached_dir,
            engine_dir=engine_dir,
        )
        try:
            result = adapter.adapt_real_authorities(
                runs=_runs(root, root_dir, cached_dir, engine_dir),
                verification_output_dir=root / "verification",
            )
        finally:
            adapter._VERIFIERS = original

        assert [row.name for row in result] == list(
            adapter.AUTHORITY_NAMES
        )
        assert all(
            set(row.__dataclass_fields__) == {
                "name",
                "run_tag",
                "source_tree_sha256",
                "artifact_path",
                "artifact_sha256",
                "independent_verification_path",
                "independent_verification_sha256",
                "provenance_path",
                "provenance_sha256",
            }
            for row in result
        )
        assert result[0].source_tree_sha256 == (
            contract.TP4_ROOT_SOURCE_TREE_SHA256
        )
        assert result[1].source_tree_sha256 == "c" * 64
        assert result[2].source_tree_sha256 == "d" * 64
        assert all(
            row.independent_verification_path.is_file()
            for row in result
        )
        assert all(row.provenance_path.is_file() for row in result)
        root_provenance = json.loads(
            result[0].provenance_path.read_text()
        )
        assert root_provenance["root_logit_receipt_gap"] is False
        assert root_provenance["binding_kind"] == (
            "remote_execution_receipt"
        )
        assert root_provenance["plan_path"].endswith(
            "execution_plan.json"
        )
        assert root_provenance["authorization_path"].endswith(
            "consumed_authorization.json"
        )
        assert root_provenance["receipt_path"].endswith(
            "execution_receipt.json"
        )
        cached_provenance = json.loads(
            result[1].provenance_path.read_text()
        )
        assert cached_provenance["plan_path"].endswith(
            "execution_plan.json"
        )
        assert cached_provenance["authorization_path"].endswith(
            "consumed_authorization.json"
        )
        assert cached_provenance["receipt_path"].endswith(
            "execution_receipt.json"
        )
        for row, provenance in (
            (result[0], root_provenance),
            (result[1], cached_provenance),
        ):
            for path_field, sha_field in (
                ("plan_path", "plan_sha256"),
                ("authorization_path", "authorization_sha256"),
                ("receipt_path", "receipt_sha256"),
            ):
                evidence = (
                    row.provenance_path.parent
                    / provenance[path_field]
                )
                assert evidence.is_file()
                assert adapter._sha256(evidence) == provenance[sha_field]


def test_adapter_requires_root_receipt_chain():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        root_dir, cached_dir, engine_dir = _authority_dirs(root)
        runs = list(_runs(root, root_dir, cached_dir, engine_dir))
        runs[0] = adapter.RealAuthorityRun(
            name="tp4_root_logit",
            run_tag="root-run",
            authority_dir=root_dir,
        )
        original = adapter._VERIFIERS
        adapter._VERIFIERS = _dependencies(
            root_dir=root_dir,
            cached_dir=cached_dir,
            engine_dir=engine_dir,
        )
        try:
            _expect_value_error(
                lambda: adapter.adapt_real_authorities(
                    runs=tuple(runs),
                    verification_output_dir=root / "verification",
                ),
                "plan is required",
            )
        finally:
            adapter._VERIFIERS = original
        assert not (root / "verification").exists()


def test_adapter_requires_cached_and_engine_receipt_chain():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        root_dir, cached_dir, engine_dir = _authority_dirs(root)
        runs = list(_runs(root, root_dir, cached_dir, engine_dir))
        runs[1] = adapter.RealAuthorityRun(
            name="cached_continuation",
            run_tag="cached-run",
            authority_dir=cached_dir,
        )
        original = adapter._VERIFIERS
        adapter._VERIFIERS = _dependencies(
            root_dir=root_dir,
            cached_dir=cached_dir,
            engine_dir=engine_dir,
        )
        try:
            _expect_value_error(
                lambda: adapter.adapt_real_authorities(
                    runs=tuple(runs),
                    verification_output_dir=root / "verification",
                ),
                "plan is required",
            )
        finally:
            adapter._VERIFIERS = original
        assert not (root / "verification").exists()


def test_adapter_outputs_build_authorized_v2_prerequisite_bundle():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        root_dir, cached_dir, engine_dir = _authority_dirs(root)
        original = adapter._VERIFIERS
        adapter._VERIFIERS = _dependencies(
            root_dir=root_dir,
            cached_dir=cached_dir,
            engine_dir=engine_dir,
        )
        try:
            rows = adapter.adapt_real_authorities(
                runs=_runs(root, root_dir, cached_dir, engine_dir),
                verification_output_dir=root / "verification",
            )
            production_builder = adapter._builder_module()
            result = production_builder.build_prerequisite_bundle(
                output_dir=root / "bundle",
                authorities=rows,
            )
        finally:
            adapter._VERIFIERS = original

        prerequisite = root / "bundle/correctness_prerequisites.json"
        status = contract.validate_prerequisites(prerequisite)
        assert result["classification"] == "PASS"
        assert status.authorized is True
        payload = json.loads(prerequisite.read_text())
        assert payload["schema_version"] == (
            contract.PREREQUISITE_SCHEMA_VERSION
        )
        assert payload["tp4_root_logit"]["provenance_path"].endswith(
            "provenance.json"
        )


def test_adapter_rejects_plan_directory_or_source_identity_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        root_dir, cached_dir, engine_dir = _authority_dirs(root)
        runs = _runs(root, root_dir, cached_dir, engine_dir)
        cases = (
            (
                _dependencies(
                    root_dir=root_dir,
                    cached_dir=cached_dir,
                    engine_dir=engine_dir,
                    root_plan_dir=root / "wrong-root",
                ),
                "not plan-bound",
            ),
            (
                _dependencies(
                    root_dir=root_dir,
                    cached_dir=cached_dir,
                    engine_dir=engine_dir,
                    cached_plan_dir=root / "wrong",
                ),
                "not plan-bound",
            ),
            (
                _dependencies(
                    root_dir=root_dir,
                    cached_dir=cached_dir,
                    engine_dir=engine_dir,
                    cached_receipt_source="e" * 64,
                ),
                "source_tree_sha256 mismatch",
            ),
        )
        original = adapter._VERIFIERS
        try:
            for index, (dependencies, fragment) in enumerate(cases):
                adapter._VERIFIERS = dependencies
                output = root / f"verification-{index}"
                _expect_value_error(
                    lambda: adapter.adapt_real_authorities(
                        runs=runs,
                        verification_output_dir=output,
                    ),
                    fragment,
                )
                assert not output.exists()
        finally:
            adapter._VERIFIERS = original


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 real prerequisite authority adapter tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
