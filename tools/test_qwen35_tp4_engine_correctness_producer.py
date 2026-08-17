from __future__ import annotations

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
    "qwen35_tp4_engine_contract_for_producer_test",
    "qwen35_tp4_engine_correctness_contract.py",
)
producer = _load(
    "qwen35_tp4_engine_correctness_producer",
    "qwen35_tp4_engine_correctness_producer.py",
)
verifier = _load(
    "verify_qwen35_tp4_engine_for_producer_test",
    "verify_qwen35_tp4_engine_correctness_gate.py",
)
executor_module = _load(
    "qwen35_tp4_engine_executor_for_producer_test",
    "qwen35_tp4_engine_correctness_executor.py",
)


MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
SOURCE_TREE_SHA256 = "d" * 64


class FakeExecutor:

    def __init__(self, *, corrupt=False):
        self.calls = []
        self.closed = False
        self.corrupt = corrupt

    def run_scenario(self, *, scenario, expected):
        self.calls.append(scenario)
        outputs = list(range(expected["generated_tokens"]))
        if self.corrupt and len(self.calls) == 1:
            rank_inventory = [0, 1, 2]
        else:
            rank_inventory = [0, 1, 2, 3]
        return {
            "scenario": scenario,
            "engine_class": contract.ENGINE_CLASS,
            "model_runner_class": contract.MODEL_RUNNER_CLASS,
            "rank_inventory": rank_inventory,
            "ack_ranks": [1, 2, 3],
            "scheduler_steps": expected["scheduler_steps"],
            "model_runner_calls": expected["model_runner_calls"],
            "output_token_ids": outputs,
            "reference_output_token_ids": list(outputs),
            "publication_commits": expected["publication_commits"],
            "restore_hits": expected["restore_hits"],
            "restore_misses": expected["restore_misses"],
            "release_events": expected["release_events"],
            "cache_entries_after": expected["cache_entries_after"],
            "cache_identity_match": True,
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
        }

    def close(self):
        self.closed = True


class FakeRuntime:

    def __init__(self, configuration):
        self.configuration = configuration
        self.closed = False

    def run_scenario(self, *, scenario, expected, plan):
        assert plan["scenario"] == scenario
        executor = FakeExecutor()
        return executor.run_scenario(
            scenario=scenario,
            expected=expected,
        )

    def close(self):
        self.closed = True


def _executor_configuration():
    return executor_module.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256=MODEL_MANIFEST_SHA256,
        source_tree_sha256=SOURCE_TREE_SHA256,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256="e" * 64,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(0, 1, 2, 3),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=8,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


def test_producer_writes_exact_four_and_self_verifies():
    executor = FakeExecutor()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority"
        result = producer.produce_authority(
            output_dir=output,
            source_tree_sha256=SOURCE_TREE_SHA256,
            model_manifest_sha256=MODEL_MANIFEST_SHA256,
            executor_factory=lambda: executor,
        )

        assert result["classification"] == "PASS"
        assert verifier.verify_run(output)["classification"] == "PASS"
        assert set(path.name for path in output.iterdir()) == set(
            contract.ARTIFACT_NAMES
        )
        assert executor.calls == list(contract.SCENARIOS)
        assert executor.closed is True


def test_invalid_rows_do_not_publish_target():
    executor = FakeExecutor(corrupt=True)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority"
        try:
            producer.produce_authority(
                output_dir=output,
                source_tree_sha256=SOURCE_TREE_SHA256,
                model_manifest_sha256=MODEL_MANIFEST_SHA256,
                executor_factory=lambda: executor,
            )
        except ValueError as error:
            assert "classification" in str(error)
        else:
            raise AssertionError("invalid Engine rows published")
        assert not output.exists()
        assert executor.closed is True


def test_existing_output_is_never_overwritten():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority"
        output.mkdir()
        (output / "keep.txt").write_text("keep\n")
        try:
            producer.produce_authority(
                output_dir=output,
                source_tree_sha256=SOURCE_TREE_SHA256,
                model_manifest_sha256=MODEL_MANIFEST_SHA256,
                executor_factory=FakeExecutor,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing output was overwritten")
        assert (output / "keep.txt").read_text() == "keep\n"


def test_default_executor_is_fail_closed():
    try:
        producer._default_executor_factory()
    except RuntimeError as error:
        assert "not implemented" in str(error)
    else:
        raise AssertionError("real Engine executor was silently enabled")


def test_configured_executor_factory_runs_exact_four_through_adapter():
    runtimes = []

    def runtime_factory(configuration):
        runtime = FakeRuntime(configuration)
        runtimes.append(runtime)
        return runtime

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority"
        result = producer.produce_authority(
            output_dir=output,
            source_tree_sha256=SOURCE_TREE_SHA256,
            model_manifest_sha256=MODEL_MANIFEST_SHA256,
            executor_factory=producer.build_configured_executor_factory(
                _executor_configuration(),
                runtime_factory=runtime_factory,
            ),
        )
        assert result["classification"] == "PASS"
        assert verifier.verify_run(output)["classification"] == "PASS"
        assert len(runtimes) == 1
        assert runtimes[0].closed is True


def test_audited_backend_factory_runs_exact_four_through_producer():
    sessions = []

    class Session:

        def __init__(self, scenario, expected):
            self.scenario = scenario
            self.expected = expected
            self.closed = False

        def execute_action(self, *, action, scenario, expected):
            assert scenario == self.scenario
            assert expected == self.expected
            outputs = list(range(expected["generated_tokens"]))
            evidence = {
                "construct_engine": {
                    "engine_class": contract.ENGINE_CLASS,
                    "model_runner_class": contract.MODEL_RUNNER_CLASS,
                },
                "begin_observation": (
                    {
                        "publication_commits": 0,
                        "restore_hits": 0,
                        "restore_misses": 0,
                        "release_events": 0,
                        "cache_entries_after": 0,
                        "cache_identity_match": True,
                    }
                    if scenario == "construct_and_bind"
                    else {}
                ),
                "configure_exact_restore": (
                    {
                        "scheduler_steps": 0,
                        "model_runner_calls": 0,
                        "output_token_ids": [],
                        "reference_output_token_ids": [],
                    }
                    if scenario == "construct_and_bind"
                    else {}
                ),
                "verify_rank_bindings": {
                    "rank_inventory": [0, 1, 2, 3],
                    "ack_ranks": [1, 2, 3],
                },
                "run_to_completion": {
                    "scheduler_steps": expected["scheduler_steps"],
                    "model_runner_calls": expected["model_runner_calls"],
                    "output_token_ids": outputs,
                    "reference_output_token_ids": list(outputs),
                },
                "verify_publication_commit": {
                    "publication_commits": expected[
                        "publication_commits"
                    ],
                },
                "drain_release_events": {
                    "release_events": expected["release_events"],
                },
                "snapshot_cache": {
                    **(
                        {}
                        if scenario == "publish_source"
                        else {
                            "publication_commits": expected[
                                "publication_commits"
                            ],
                        }
                    ),
                    "restore_hits": expected["restore_hits"],
                    "restore_misses": expected["restore_misses"],
                    **(
                        {}
                        if scenario == "restore_w1"
                        else {
                            "release_events": expected[
                                "release_events"
                            ],
                        }
                    ),
                    "cache_entries_after": expected[
                        "cache_entries_after"
                    ],
                    "cache_identity_match": True,
                },
                "close_engine": {
                    "rank_exit_codes": [0, 0, 0, 0],
                },
                "verify_cleanup": {
                    "process_group_destroyed": True,
                    "owned_children_remaining": [],
                },
            }
            return evidence.get(action, {})

        def close(self):
            self.closed = True

    def backend_factory(configuration, *, scenario, expected):
        session = Session(scenario, expected)
        sessions.append(session)
        return session

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority"
        result = producer.produce_authority(
            output_dir=output,
            source_tree_sha256=SOURCE_TREE_SHA256,
            model_manifest_sha256=MODEL_MANIFEST_SHA256,
            executor_factory=producer.build_audited_executor_factory(
                _executor_configuration(),
                backend_factory=backend_factory,
            ),
        )
        assert result["classification"] == "PASS"
        assert len(sessions) == len(contract.SCENARIOS)
        assert all(session.closed for session in sessions)


def test_real_backend_factory_requires_independent_reference_provider():
    try:
        producer.build_real_backend_factory(
            reference_token_provider=None,
        )
    except TypeError as error:
        assert "reference_token_provider" in str(error)
    else:
        raise AssertionError(
            "real backend accepted missing reference provider"
        )

    calls = []

    class Engine:
        pass

    engine = Engine()
    factory = producer.build_real_backend_factory(
        engine_factory=lambda configuration: engine,
        reference_token_provider=lambda **kwargs: [],
    )
    session = factory(
        _executor_configuration(),
        scenario="construct_and_bind",
        expected=contract.SCENARIOS["construct_and_bind"],
    )
    assert session.scenario == "construct_and_bind"
    assert session.engine is None
    session.close()
    assert calls == []


def test_source_bound_real_backend_factory_builds_verified_provider():
    reference_module = producer._load_module(
        "qwen35_tp4_engine_reference_tokens",
        "qwen35_tp4_engine_reference_tokens.py",
    )
    original = reference_module.build_reference_token_provider
    calls = []
    provider = lambda **kwargs: [7]

    def build_provider(**kwargs):
        calls.append(kwargs)
        return provider

    reference_module.build_reference_token_provider = build_provider
    try:
        factory = producer.build_source_bound_real_backend_factory(
            _executor_configuration(),
            authority_dir="/authority/reference",
            verification_path="/authority/reference.verify.json",
            engine_factory=lambda configuration: object(),
        )
        session = factory(
            _executor_configuration(),
            scenario="publish_source",
            expected=contract.SCENARIOS["publish_source"],
        )
    finally:
        reference_module.build_reference_token_provider = original

    assert len(calls) == 1
    assert calls[0]["authority_dir"] == "/authority/reference"
    assert (
        calls[0]["verification_path"]
        == "/authority/reference.verify.json"
    )
    assert (
        calls[0]["configuration"].to_payload()
        == _executor_configuration().to_payload()
    )
    assert session.reference_token_provider is provider
    session.close()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine correctness producer tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
