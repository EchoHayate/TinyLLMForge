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


contract = _load(
    "qwen35_tp4_cached_continuation_contract_for_producer_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
producer = _load(
    "qwen35_tp4_cached_continuation_correctness_producer",
    "qwen35_tp4_cached_continuation_correctness_producer.py",
)
verifier = _load(
    "verify_qwen35_tp4_cached_continuation_for_producer_test",
    "verify_qwen35_tp4_cached_continuation_correctness_gate.py",
)


MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
SOURCE_TREE_SHA256 = "c" * 64


class FakeExecutor:

    def __init__(self, *, corrupt=False):
        self.calls = []
        self.closed = False
        self.corrupt = corrupt

    def run_continuation(self, *, workload, request_index, payload):
        self.calls.append((workload, request_index, payload))
        spec = payload["spec"]
        expected_hit = workload in contract.HIT_WORKLOADS
        output = list(range(spec["generated_tokens"]))
        if self.corrupt and not self.calls[:-1]:
            output[-1] = 999
        return {
            "workload": workload,
            "request_index": request_index,
            "outcome": "continuation",
            "restore_hit": expected_hit,
            "restore_reason": (
                "exact_hit"
                if expected_hit
                else contract.W4_EXPECTED_REASONS[request_index]
            ),
            "prompt_tokens": (
                spec["shared_prefix_tokens"] + spec["suffix_tokens"]
            ),
            "reused_tokens": (
                spec["shared_prefix_tokens"] if expected_hit else 0
            ),
            "executed_prefill_tokens": (
                spec["suffix_tokens"]
                if expected_hit
                else (
                    spec["shared_prefix_tokens"]
                    + spec["suffix_tokens"]
                )
            ),
            "output_token_ids": output,
            "reference_output_token_ids": list(
                range(spec["generated_tokens"])
            ),
            "logits_max_abs_diff": 0.0,
            "logits_allclose": True,
            "cache_identity_match": True,
            "rank_inventory": [0, 1, 2, 3],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        }

    def close(self):
        self.closed = True


def test_producer_writes_exact_five_and_self_verifies():
    executor = FakeExecutor()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "authority"
        result = producer.produce_authority(
            output_dir=output_dir,
            source_tree_sha256=SOURCE_TREE_SHA256,
            model_manifest_sha256=MODEL_MANIFEST_SHA256,
            executor_factory=lambda: executor,
        )
        verification = verifier.verify_run(output_dir)

        assert set(path.name for path in output_dir.iterdir()) == set(
            contract.ARTIFACT_NAMES
        )
        assert result["classification"] == "PASS"
        assert verification["classification"] == "PASS"
        assert len(executor.calls) == 19
        assert executor.closed is True


def test_producer_rejects_invalid_rows_without_publishing_pass():
    executor = FakeExecutor(corrupt=True)
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "authority"
        try:
            producer.produce_authority(
                output_dir=output_dir,
                source_tree_sha256=SOURCE_TREE_SHA256,
                model_manifest_sha256=MODEL_MANIFEST_SHA256,
                executor_factory=lambda: executor,
            )
        except ValueError as error:
            assert "classification" in str(error)
        else:
            raise AssertionError("invalid rows published PASS")

        assert not output_dir.exists()
        assert executor.closed is True


def test_producer_never_overwrites_existing_output():
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "authority"
        output_dir.mkdir()
        (output_dir / "keep.txt").write_text(
            "keep\n",
            encoding="utf-8",
        )
        try:
            producer.produce_authority(
                output_dir=output_dir,
                source_tree_sha256=SOURCE_TREE_SHA256,
                model_manifest_sha256=MODEL_MANIFEST_SHA256,
                executor_factory=FakeExecutor,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing output was overwritten")
        assert (output_dir / "keep.txt").read_text() == "keep\n"


def test_default_runtime_loader_is_fail_closed():
    try:
        producer._default_executor_factory()
    except RuntimeError as error:
        assert "not implemented" in str(error)
    else:
        raise AssertionError("real TP4 producer was silently enabled")


def test_configured_factory_wires_cached_executor_session_and_reference():
    calls = []

    class FakeReferenceExecutor:
        def close(self):
            calls.append("reference_close")

    class FakeSessionFactory:
        def __init__(
            self,
            configuration,
            *,
            engine_factory,
            reference_executor_factory,
        ):
            calls.append((
                "session_factory",
                configuration.to_payload(),
                engine_factory,
                reference_executor_factory,
            ))
            self.closed = False

        def __call__(self, *args, **kwargs):
            raise AssertionError("row execution was not expected")

        def close(self):
            self.closed = True
            calls.append("session_close")

    configuration = _load(
        "qwen35_tp4_engine_executor_for_cached_producer_test",
        "qwen35_tp4_engine_correctness_executor.py",
    ).ExecutorConfiguration(
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
    engine_factory = lambda configuration: object()
    reference_executor_factory = lambda: FakeReferenceExecutor()
    factory = producer.build_configured_executor_factory(
        configuration,
        engine_factory=engine_factory,
        reference_executor_factory=reference_executor_factory,
        session_factory_type=FakeSessionFactory,
    )
    runtime = factory()
    assert calls == [(
        "session_factory",
        configuration.to_payload(),
        engine_factory,
        reference_executor_factory,
    )]
    runtime.close()
    assert calls[-1] == "session_close"


def test_configured_authority_builds_official_reference_and_produces():
    calls = []
    configuration = _load(
        "qwen35_tp4_engine_executor_for_cached_authority_test",
        "qwen35_tp4_engine_correctness_executor.py",
    ).ExecutorConfiguration(
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
    original_build = producer.build_configured_executor_factory
    original_produce = producer.produce_authority

    def build(configuration_arg, **kwargs):
        calls.append((
            "build",
            configuration_arg.to_payload(),
            dict(kwargs),
        ))
        return "configured-executor-factory"

    def produce(**kwargs):
        calls.append(("produce", dict(kwargs)))
        return {"classification": "PASS"}

    producer.build_configured_executor_factory = build
    producer.produce_authority = produce
    try:
        result = producer.produce_configured_authority(
            output_dir="/authority/cached",
            configuration=configuration,
            engine_factory=lambda value: object(),
            reference_executor_factory=lambda: object(),
        )
    finally:
        producer.build_configured_executor_factory = original_build
        producer.produce_authority = original_produce
    assert result == {"classification": "PASS"}
    assert calls[0][0] == "build"
    assert calls[1] == (
        "produce",
        {
            "output_dir": "/authority/cached",
            "source_tree_sha256": "b" * 64,
            "model_manifest_sha256": "a" * 64,
            "executor_factory": "configured-executor-factory",
        },
    )


def test_configured_authority_defaults_to_official_reference_factory():
    calls = []
    configuration = _load(
        "qwen35_tp4_engine_executor_for_cached_default_reference_test",
        "qwen35_tp4_engine_correctness_executor.py",
    ).ExecutorConfiguration(
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
    original_load = producer._load_module
    original_build = producer.build_configured_executor_factory
    original_produce = producer.produce_authority

    class OfficialModule:
        @staticmethod
        def build_official_reference_executor_factory(value):
            calls.append(("official", value.to_payload()))
            return "official-reference-factory"

    def load(name, filename):
        if filename == "qwen35_tp4_engine_official_reference_executor.py":
            return OfficialModule
        return original_load(name, filename)

    def build(configuration_arg, **kwargs):
        calls.append(("build", dict(kwargs)))
        return "configured-executor-factory"

    producer._load_module = load
    producer.build_configured_executor_factory = build
    producer.produce_authority = lambda **kwargs: {
        "classification": "PASS"
    }
    try:
        result = producer.produce_configured_authority(
            output_dir="/authority/cached",
            configuration=configuration,
            engine_factory=lambda value: object(),
        )
    finally:
        producer._load_module = original_load
        producer.build_configured_executor_factory = original_build
        producer.produce_authority = original_produce
    assert result == {"classification": "PASS"}
    assert calls[0] == ("official", configuration.to_payload())
    assert calls[1][0] == "build"
    assert calls[1][1]["reference_executor_factory"] == (
        "official-reference-factory"
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
        "qwen35 TP4 cached-continuation producer tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
