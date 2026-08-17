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


contract = _load(
    "qwen35_tp4_cached_contract_for_authority_driver_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
engine_executor = _load(
    "qwen35_tp4_engine_executor_for_cached_authority_driver_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
driver = _load(
    "run_qwen35_tp4_cached_continuation_authority",
    "run_qwen35_tp4_cached_continuation_authority.py",
)


def _configuration():
    return engine_executor.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256=contract.WORKLOAD_MANIFEST_SHA256,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(0, 1, 2, 3),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=32,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


def test_load_configuration_requires_exact_manifests_and_source_inventory():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_manifest = root / "model_manifest.json"
        workload_manifest = root / "workload_manifest.json"
        model_manifest.write_text('{"model":true}\n')
        benchmark = contract._BENCHMARK
        workload_manifest.write_text(
            json.dumps(
                benchmark.workload_manifest_payload(),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        payload = _configuration().to_payload()
        payload["model_manifest_path"] = str(model_manifest)
        payload["model_manifest_sha256"] = hashlib.sha256(
            model_manifest.read_bytes()
        ).hexdigest()
        payload["workload_manifest_path"] = str(workload_manifest)
        payload["workload_manifest_sha256"] = hashlib.sha256(
            workload_manifest.read_bytes()
        ).hexdigest()
        assert payload["workload_manifest_sha256"] == (
            contract.WORKLOAD_MANIFEST_SHA256
        )
        configuration_path = root / "configuration.json"
        configuration_path.write_text(json.dumps(payload) + "\n")
        inventory_path = root / "source_inventory.json"
        inventory_path.write_text(json.dumps({
            "owned_files": ["tinyvllm/engine/llm_engine.py"],
            "source_tree_sha256": payload["source_tree_sha256"],
        }) + "\n")
        loaded = driver.load_configuration(
            configuration_path,
            source_inventory_path=inventory_path,
        )
        assert loaded.to_payload() == payload

        workload_manifest.write_text('{"tampered":true}\n')
        try:
            driver.load_configuration(
                configuration_path,
                source_inventory_path=inventory_path,
            )
        except ValueError as error:
            assert "workload manifest" in str(error)
        else:
            raise AssertionError("tampered workload manifest was accepted")


def test_run_authority_produces_exact_five_and_external_verification():
    calls = []
    original = (
        driver.producer.produce_configured_authority,
        driver.verifier.verify_and_write,
    )

    def produce(**kwargs):
        calls.append(("produce", dict(kwargs)))
        output = Path(kwargs["output_dir"])
        output.mkdir()
        for name in contract.ARTIFACT_NAMES:
            (output / name).write_text("{}\n")
        return {"classification": "PASS"}

    def verify(run_dir, *, output_path):
        calls.append((
            "verify",
            Path(run_dir),
            Path(output_path),
        ))
        Path(output_path).write_text(
            '{"classification":"PASS"}\n'
        )
        return {
            "classification": "PASS",
            "model_manifest_sha256": "a" * 64,
        }

    driver.producer.produce_configured_authority = produce
    driver.verifier.verify_and_write = verify
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cached_authority"
            verification = root / "cached_verification.json"
            result = driver.run_authority(
                output_dir=output,
                verification_path=verification,
                configuration=_configuration(),
                engine_factory="engine-factory",
                reference_executor_factory="reference-factory",
            )
            assert result == {
                "classification": "PASS",
                "model_manifest_sha256": "a" * 64,
            }
            assert output.is_dir()
            assert verification.is_file()
    finally:
        (
            driver.producer.produce_configured_authority,
            driver.verifier.verify_and_write,
        ) = original

    assert calls[0][0] == "produce"
    assert calls[0][1]["engine_factory"] == "engine-factory"
    assert calls[0][1]["reference_executor_factory"] == (
        "reference-factory"
    )
    assert calls[1][0] == "verify"
    assert calls[1][1] == calls[0][1]["output_dir"]


def test_failure_or_existing_outputs_leave_no_partial_publication():
    original = driver.producer.produce_configured_authority

    def fail(**kwargs):
        Path(kwargs["output_dir"]).mkdir()
        raise RuntimeError("synthetic cached authority failure")

    driver.producer.produce_configured_authority = fail
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cached_authority"
            verification = root / "cached_verification.json"
            try:
                driver.run_authority(
                    output_dir=output,
                    verification_path=verification,
                    configuration=_configuration(),
                )
            except RuntimeError as error:
                assert "synthetic" in str(error)
            else:
                raise AssertionError("producer failure was hidden")
            assert not output.exists()
            assert not verification.exists()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cached_authority"
            output.mkdir()
            try:
                driver.run_authority(
                    output_dir=output,
                    verification_path=root / "verification.json",
                    configuration=_configuration(),
                )
            except ValueError as error:
                assert "already exists" in str(error)
            else:
                raise AssertionError("existing output was overwritten")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cached_authority"
            try:
                driver.run_authority(
                    output_dir=output,
                    verification_path=output / "verification.json",
                    configuration=_configuration(),
                )
            except ValueError as error:
                assert "outside" in str(error)
            else:
                raise AssertionError(
                    "run-internal verification path was accepted"
                )
            assert not output.exists()
    finally:
        driver.producer.produce_configured_authority = original


def test_main_loads_verified_configuration_before_running_authority():
    calls = []
    original = (
        driver.load_configuration,
        driver.run_authority,
    )

    def load(path, *, source_inventory_path):
        calls.append((
            "load",
            path,
            source_inventory_path,
        ))
        return "configuration"

    def run(**kwargs):
        calls.append(("run", dict(kwargs)))
        return {"classification": "PASS"}

    driver.load_configuration = load
    driver.run_authority = run
    try:
        assert driver.main([
            "--configuration",
            "/bundle/executor_configuration.json",
            "--source-inventory",
            "/bundle/source_inventory.json",
            "--output-dir",
            "/runs/cached",
            "--verification-path",
            "/runs/cached_verification.json",
        ]) == 0
    finally:
        (
            driver.load_configuration,
            driver.run_authority,
        ) = original

    assert calls == [
        (
            "load",
            "/bundle/executor_configuration.json",
            "/bundle/source_inventory.json",
        ),
        (
            "run",
            {
                "output_dir": "/runs/cached",
                "verification_path": (
                    "/runs/cached_verification.json"
                ),
                "configuration": "configuration",
            },
        ),
    ]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation authority driver tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
