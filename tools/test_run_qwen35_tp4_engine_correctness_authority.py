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
    "qwen35_tp4_engine_executor_for_authority_driver_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
driver = _load(
    "run_qwen35_tp4_engine_correctness_authority",
    "run_qwen35_tp4_engine_correctness_authority.py",
)


def _configuration():
    return executor.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256="c" * 64,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(0, 1, 2, 3),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=8,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


def test_configuration_json_is_exact_and_canonicalized():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_manifest = root / "model_manifest.json"
        workload_manifest = root / "workload_manifest.json"
        model_manifest.write_text('{"model":true}\n')
        workload_manifest.write_text('{"workload":true}\n')
        payload = _configuration().to_payload()
        payload["model_manifest_path"] = str(model_manifest)
        payload["model_manifest_sha256"] = hashlib.sha256(
            model_manifest.read_bytes()
        ).hexdigest()
        payload["workload_manifest_path"] = str(workload_manifest)
        payload["workload_manifest_sha256"] = hashlib.sha256(
            workload_manifest.read_bytes()
        ).hexdigest()
        path = root / "configuration.json"
        path.write_text(
            json.dumps(payload) + "\n"
        )
        loaded = driver.load_configuration(path)
        assert loaded.to_payload() == payload

        extra = dict(payload)
        extra["extra"] = True
        path.write_text(json.dumps(extra) + "\n")
        try:
            driver.load_configuration(path)
        except ValueError as error:
            assert "schema" in str(error)
        else:
            raise AssertionError("extra configuration field was accepted")

        path.write_text(json.dumps(payload) + "\n")
        workload_manifest.write_text('{"tampered":true}\n')
        try:
            driver.load_configuration(path)
        except ValueError as error:
            assert "workload manifest" in str(error)
        else:
            raise AssertionError(
                "tampered workload manifest was accepted"
            )

        workload_manifest.write_text('{"workload":true}\n')
        inventory = root / "source_inventory.json"
        inventory.write_text(json.dumps({
            "owned_files": ["a.py"],
            "source_tree_sha256": payload["source_tree_sha256"],
        }) + "\n")
        loaded = driver.load_configuration(
            path,
            source_inventory_path=inventory,
        )
        assert loaded.source_tree_sha256 == payload["source_tree_sha256"]
        inventory.write_text(json.dumps({
            "owned_files": ["a.py"],
            "source_tree_sha256": "d" * 64,
        }) + "\n")
        try:
            driver.load_configuration(
                path,
                source_inventory_path=inventory,
            )
        except ValueError as error:
            assert "source inventory" in str(error)
        else:
            raise AssertionError("source inventory drift was accepted")


def test_driver_runs_reference_then_engine_and_atomically_publishes():
    calls = []
    original = (
        driver.reference_producer.produce_reference_authority,
        driver.official.build_official_reference_executor_factory,
        driver.engine_producer.build_source_bound_real_backend_factory,
        driver.engine_producer.build_audited_executor_factory,
        driver.engine_producer.produce_authority,
        driver._verify_complete_authority,
    )

    def build_reference_executor(configuration):
        calls.append(("build_reference_executor", configuration.to_payload()))
        return "reference-executor-factory"

    def produce_reference(**kwargs):
        calls.append(("produce_reference", dict(kwargs)))
        Path(kwargs["output_dir"]).mkdir()
        Path(kwargs["verification_path"]).write_text(
            '{"classification":"PASS"}\n'
        )
        return {"classification": "PASS"}

    def build_backend(configuration, **kwargs):
        calls.append((
            "build_backend",
            configuration.to_payload(),
            dict(kwargs),
        ))
        return "backend-factory"

    def build_audited(configuration, **kwargs):
        calls.append((
            "build_audited",
            configuration.to_payload(),
            dict(kwargs),
        ))
        return "engine-executor-factory"

    def produce_engine(**kwargs):
        calls.append(("produce_engine", dict(kwargs)))
        Path(kwargs["output_dir"]).mkdir()
        return {"classification": "PASS"}

    def verify_complete(path):
        calls.append(("verify_complete", Path(path)))
        return {"classification": "PASS"}

    (
        driver.reference_producer.produce_reference_authority,
        driver.official.build_official_reference_executor_factory,
        driver.engine_producer.build_source_bound_real_backend_factory,
        driver.engine_producer.build_audited_executor_factory,
        driver.engine_producer.produce_authority,
        driver._verify_complete_authority,
    ) = (
        produce_reference,
        build_reference_executor,
        build_backend,
        build_audited,
        produce_engine,
        verify_complete,
    )
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authority"
            result = driver.run_authority(
                output_root=output,
                configuration=_configuration(),
                engine_factory="engine-factory",
            )
            assert result["classification"] == "PASS"
            assert output.is_dir()
            assert {path.name for path in output.iterdir()} == {
                driver.REFERENCE_DIR_NAME,
                driver.REFERENCE_VERIFICATION_NAME,
                driver.ENGINE_DIR_NAME,
                driver.RUN_SUMMARY_NAME,
            }
            summary = json.loads(
                (output / driver.RUN_SUMMARY_NAME).read_text()
            )
            assert summary == result
    finally:
        (
            driver.reference_producer.produce_reference_authority,
            driver.official.build_official_reference_executor_factory,
            driver.engine_producer.build_source_bound_real_backend_factory,
            driver.engine_producer.build_audited_executor_factory,
            driver.engine_producer.produce_authority,
            driver._verify_complete_authority,
        ) = original

    names = [call[0] for call in calls]
    assert names == [
        "build_reference_executor",
        "produce_reference",
        "build_backend",
        "build_audited",
        "produce_engine",
        "verify_complete",
    ]
    reference_call = calls[1][1]
    backend_call = calls[2]
    engine_call = calls[4][1]
    assert reference_call["executor_factory"] == (
        "reference-executor-factory"
    )
    assert backend_call[2]["engine_factory"] == "engine-factory"
    assert backend_call[2]["authority_dir"] == (
        reference_call["output_dir"]
    )
    assert backend_call[2]["verification_path"] == (
        reference_call["verification_path"]
    )
    assert engine_call["executor_factory"] == (
        "engine-executor-factory"
    )
    assert calls[-1][1].name.startswith(".authority.")


def test_driver_failure_or_existing_target_never_publishes():
    original = (
        driver.reference_producer.produce_reference_authority,
        driver.official.build_official_reference_executor_factory,
    )
    driver.official.build_official_reference_executor_factory = (
        lambda configuration: "reference-executor-factory"
    )

    def fail_reference(**kwargs):
        Path(kwargs["output_dir"]).mkdir()
        raise RuntimeError("synthetic reference failure")

    driver.reference_producer.produce_reference_authority = fail_reference
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authority"
            try:
                driver.run_authority(
                    output_root=output,
                    configuration=_configuration(),
                )
            except RuntimeError as error:
                assert "synthetic reference failure" in str(error)
            else:
                raise AssertionError("phase failure was hidden")
            assert not output.exists()

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authority"
            output.mkdir()
            try:
                driver.run_authority(
                    output_root=output,
                    configuration=_configuration(),
                )
            except ValueError as error:
                assert "already exists" in str(error)
            else:
                raise AssertionError("existing authority was overwritten")
    finally:
        (
            driver.reference_producer.produce_reference_authority,
            driver.official.build_official_reference_executor_factory,
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
        "qwen35 TP4 Engine authority driver tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
