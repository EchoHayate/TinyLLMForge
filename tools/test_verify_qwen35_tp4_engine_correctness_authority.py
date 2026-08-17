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


driver = _load(
    "run_qwen35_tp4_engine_authority_for_verifier_test",
    "run_qwen35_tp4_engine_correctness_authority.py",
)
verifier = _load(
    "verify_qwen35_tp4_engine_correctness_authority",
    "verify_qwen35_tp4_engine_correctness_authority.py",
)


def _write(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def _authority(root):
    run = root / "authority"
    run.mkdir()
    (run / driver.REFERENCE_DIR_NAME).mkdir()
    (run / driver.ENGINE_DIR_NAME).mkdir()
    _write(
        run / driver.REFERENCE_VERIFICATION_NAME,
        {"classification": "PASS"},
    )
    summary = {
        "classification": "PASS",
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "reference_classification": "PASS",
        "engine_classification": "PASS",
        "inventory": [
            driver.REFERENCE_DIR_NAME,
            driver.REFERENCE_VERIFICATION_NAME,
            driver.ENGINE_DIR_NAME,
            driver.RUN_SUMMARY_NAME,
        ],
    }
    _write(run / driver.RUN_SUMMARY_NAME, summary)
    return run


def test_verifier_composes_reference_and_engine_verifiers():
    original = (
        verifier.reference_verifier.verify_run,
        verifier.engine_verifier.verify_run,
    )
    calls = []

    def verify_reference(path):
        calls.append(("reference", Path(path)))
        return {
            "schema_version": (
                "qwen35.tp4-engine-reference-tokens.v1"
            ),
            "classification": "PASS",
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "reference_tokens_sha256": "e" * 64,
            "source_manifest_sha256": "f" * 64,
            "scenario_count": 5,
        }

    def verify_engine(path):
        calls.append(("engine", Path(path)))
        return {
            "classification": "PASS",
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
        }

    verifier.reference_verifier.verify_run = verify_reference
    verifier.engine_verifier.verify_run = verify_engine
    try:
        with tempfile.TemporaryDirectory() as temporary:
            run = _authority(Path(temporary))
            _write(
                run / driver.REFERENCE_VERIFICATION_NAME,
                {
                    "schema_version": (
                        "qwen35.tp4-engine-reference-tokens.v1"
                    ),
                    "classification": "PASS",
                    "model_manifest_sha256": "a" * 64,
                    "source_tree_sha256": "b" * 64,
                    "workload_manifest_sha256": "c" * 64,
                    "reference_tokens_sha256": "e" * 64,
                    "source_manifest_sha256": "f" * 64,
                    "scenario_count": 5,
                },
            )
            result = verifier.verify_run(run)
            assert result == {
                "classification": "PASS",
                "model_manifest_sha256": "a" * 64,
                "source_tree_sha256": "b" * 64,
                "workload_manifest_sha256": "c" * 64,
                "reference_classification": "PASS",
                "engine_classification": "PASS",
            }
            assert calls == [
                ("reference", run / driver.REFERENCE_DIR_NAME),
                ("engine", run / driver.ENGINE_DIR_NAME),
            ]
    finally:
        (
            verifier.reference_verifier.verify_run,
            verifier.engine_verifier.verify_run,
        ) = original


def test_verifier_rejects_extra_file_or_identity_mismatch():
    original = (
        verifier.reference_verifier.verify_run,
        verifier.engine_verifier.verify_run,
    )
    verifier.reference_verifier.verify_run = lambda path: {
        "schema_version": "qwen35.tp4-engine-reference-tokens.v1",
        "classification": "PASS",
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "reference_tokens_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "scenario_count": 5,
    }
    verifier.engine_verifier.verify_run = lambda path: {
        "classification": "PASS",
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "d" * 64,
    }
    try:
        with tempfile.TemporaryDirectory() as temporary:
            run = _authority(Path(temporary))
            _write(
                run / driver.REFERENCE_VERIFICATION_NAME,
                {
                    "schema_version": (
                        "qwen35.tp4-engine-reference-tokens.v1"
                    ),
                    "classification": "PASS",
                    "model_manifest_sha256": "a" * 64,
                    "source_tree_sha256": "b" * 64,
                    "workload_manifest_sha256": "c" * 64,
                    "reference_tokens_sha256": "e" * 64,
                    "source_manifest_sha256": "f" * 64,
                    "scenario_count": 5,
                },
            )
            try:
                verifier.verify_run(run)
            except verifier.VerificationError as error:
                assert "identity" in str(error)
            else:
                raise AssertionError("cross-phase identity drift was accepted")

        with tempfile.TemporaryDirectory() as temporary:
            run = _authority(Path(temporary))
            (run / "extra.json").write_text("{}\n")
            try:
                verifier.verify_run(run)
            except verifier.VerificationError as error:
                assert "inventory" in str(error)
            else:
                raise AssertionError("extra root artifact was accepted")

        with tempfile.TemporaryDirectory() as temporary:
            run = _authority(Path(temporary))
            _write(
                run / driver.REFERENCE_VERIFICATION_NAME,
                {"classification": "PASS"},
            )
            verifier.engine_verifier.verify_run = lambda path: {
                "classification": "PASS",
                "model_manifest_sha256": "a" * 64,
                "source_tree_sha256": "b" * 64,
            }
            try:
                verifier.verify_run(run)
            except verifier.VerificationError as error:
                assert "external reference verification" in str(error)
            else:
                raise AssertionError(
                    "tampered external verification was accepted"
                )
    finally:
        (
            verifier.reference_verifier.verify_run,
            verifier.engine_verifier.verify_run,
        ) = original


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine authority verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
