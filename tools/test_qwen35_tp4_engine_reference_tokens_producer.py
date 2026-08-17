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
    "qwen35_tp4_engine_executor_for_reference_producer_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
reference = _load(
    "qwen35_tp4_engine_reference_tokens_for_producer_test",
    "qwen35_tp4_engine_reference_tokens.py",
)
producer = _load(
    "qwen35_tp4_engine_reference_tokens_producer",
    "qwen35_tp4_engine_reference_tokens_producer.py",
)
verifier = _load(
    "verify_qwen35_tp4_engine_reference_tokens_for_producer_test",
    "verify_qwen35_tp4_engine_reference_tokens.py",
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


class FakeReferenceExecutor:

    def __init__(self, *, corrupt=False):
        self.calls = []
        self.closed = False
        self.corrupt = corrupt

    def generate_reference(
        self,
        *,
        scenario,
        prompt_token_ids,
        generated_tokens,
        generation_policy,
    ):
        self.calls.append((
            scenario,
            list(prompt_token_ids),
            generated_tokens,
            dict(generation_policy),
        ))
        count = generated_tokens + (1 if self.corrupt else 0)
        return list(range(count))

    def close(self):
        self.closed = True


def test_producer_writes_exact_two_and_independent_verification():
    fake = FakeReferenceExecutor()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "authority"
        verification = root / "verified.json"
        result = producer.produce_reference_authority(
            output_dir=output,
            verification_path=verification,
            configuration=_configuration(),
            executor_factory=lambda: fake,
        )
        assert result["classification"] == "PASS"
        assert {path.name for path in output.iterdir()} == set(
            reference.ARTIFACT_NAMES
        )
        assert verifier.verify_run(output)["classification"] == "PASS"
        assert json.loads(verification.read_text())[
            "classification"
        ] == "PASS"
        assert [call[0] for call in fake.calls] == list(
            reference.REFERENCE_SCENARIOS
        )
        assert all(
            call[3] == reference.GENERATION_POLICY
            for call in fake.calls
        )
        assert fake.closed is True


def test_invalid_executor_output_never_publishes_authority():
    fake = FakeReferenceExecutor(corrupt=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "authority"
        verification = root / "verified.json"
        try:
            producer.produce_reference_authority(
                output_dir=output,
                verification_path=verification,
                configuration=_configuration(),
                executor_factory=lambda: fake,
            )
        except ValueError as error:
            assert "output" in str(error)
        else:
            raise AssertionError("invalid reference output was published")
        assert not output.exists()
        assert not verification.exists()
        assert fake.closed is True


def test_default_executor_and_existing_targets_fail_closed():
    try:
        producer._default_executor_factory()
    except RuntimeError as error:
        assert "not implemented" in str(error)
    else:
        raise AssertionError("real reference executor was silently enabled")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "authority"
        output.mkdir()
        verification = root / "verified.json"
        try:
            producer.produce_reference_authority(
                output_dir=output,
                verification_path=verification,
                configuration=_configuration(),
                executor_factory=FakeReferenceExecutor,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing authority was overwritten")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "authority"
        verification = root / "verified.json"
        verification.write_text("{}\n")
        try:
            producer.produce_reference_authority(
                output_dir=output,
                verification_path=verification,
                configuration=_configuration(),
                executor_factory=FakeReferenceExecutor,
            )
        except ValueError as error:
            assert "verification" in str(error)
        else:
            raise AssertionError("existing verification was overwritten")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine reference producer tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
