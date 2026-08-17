from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys


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
    "qwen35_tp4_engine_correctness_executor_for_backend_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
backend = _load(
    "qwen35_tp4_engine_backend_session",
    "qwen35_tp4_engine_backend_session.py",
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


class FakeEngine:

    def __init__(self):
        self.calls = []
        self.model_runner = SimpleNamespace(world_size=4)
        self.ps = [
            SimpleNamespace(exitcode=0, is_alive=lambda: False)
            for _ in range(3)
        ]
        self.outputs = []
        self.snapshot = {
            "current_entries": 0,
            "hits": 0,
            "misses": 0,
            "publication_commits": 0,
            "invalidations": 0,
            "clears": 0,
            "last_publication_block_identities": [],
        }

    def configure_qwen35_hybrid_prefix_publication_runtime(self, **kwargs):
        self.calls.append(("configure", kwargs))

    def qwen35_hybrid_prefix_authority_snapshots(self, *, timeout_s):
        self.calls.append(("snapshot", timeout_s))
        return tuple(
            {"rank": rank, **self.snapshot}
            for rank in range(4)
        )

    def add_request(self, prompt, sampling_params):
        self.calls.append((
            "add_request",
            list(prompt),
            sampling_params.temperature,
            sampling_params.max_tokens,
            sampling_params.ignore_eos,
        ))
        self.outputs = list(range(sampling_params.max_tokens))

    def is_finished(self):
        return not self.outputs

    def hybrid_state_release_event_count(self):
        return 0

    def step(self):
        token = self.outputs.pop(0)
        self.calls.append(("step", token))
        if not self.outputs and self.snapshot["current_entries"] == 0:
            self.snapshot.update({
                "current_entries": 1,
                "publication_commits": (
                    self.snapshot["publication_commits"] + 1
                ),
                "last_publication_block_identities": [[7, 2, 99]],
            })
        return ([(17, [token])], 1)

    def clear_qwen35_hybrid_prefix_caches(self, *, timeout_s):
        self.calls.append(("clear", timeout_s))
        self.snapshot["clears"] += 1
        self.snapshot["current_entries"] = 0
        return tuple(
            {"rank": rank, "cleared_entries": 1}
            for rank in range(4)
        )

    def invalidate_qwen35_hybrid_prefix_blocks(
        self,
        block_identities,
        *,
        timeout_s,
    ):
        self.calls.append((
            "invalidate",
            [list(identity) for identity in block_identities],
            timeout_s,
        ))
        self.snapshot["invalidations"] += 1
        self.snapshot["current_entries"] = 0
        return tuple(
            {"rank": rank, "invalidated_entries": 1}
            for rank in range(4)
        )

    def exit(self):
        self.calls.append(("exit",))
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {"rank": rank, "process_group_destroyed": True}
                for rank in range(4)
            ],
        }


def test_backend_session_constructs_engine_lazily_and_configures_runtime():
    engines = []

    def engine_factory(configuration):
        assert configuration == _configuration()
        engine = FakeEngine()
        engines.append(engine)
        return engine

    session = backend.EngineBackendSession(
        _configuration(),
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
        engine_factory=engine_factory,
    )
    assert engines == []
    assert session.execute_action(
        action="construct_engine",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    ) == {
        "engine_class": executor.contract.ENGINE_CLASS,
        "model_runner_class": executor.contract.MODEL_RUNNER_CLASS,
    }
    assert len(engines) == 1
    session.execute_action(
        action="configure_exact_restore",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    )
    assert engines[0].calls == [(
        "configure",
        {
            "model_fingerprint": "qwen35-m8-authority",
            "max_entries": 8,
            "max_bytes": 1 << 30,
            "timeout_s": 600.0,
        },
    )]
    session.close()


def test_default_engine_factory_reserves_a_release_handoff_slot():
    observed = {}
    original_import = backend.importlib.import_module
    original_values = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "TINYVLLM_DIST_PORT",
            "MASTER_PORT",
        )
    }

    class LLMEngine:
        def __init__(self, model, **kwargs):
            observed["model"] = model
            observed["kwargs"] = dict(kwargs)
            observed["environment"] = {
                name: os.environ.get(name)
                for name in original_values
            }

    backend.importlib.import_module = lambda name: SimpleNamespace(
        LLMEngine=LLMEngine
    )
    try:
        result = backend._default_engine_factory(_configuration())
    finally:
        backend.importlib.import_module = original_import

    assert isinstance(result, LLMEngine)
    assert observed == {
        "model": "/models/qwen35",
        "kwargs": {
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 2,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "TINYVLLM_DIST_PORT": "31001",
            "MASTER_PORT": "31002",
        },
    }
    assert {
        name: os.environ.get(name)
        for name in original_values
    } == original_values


def test_backend_session_requires_structured_cleanup_receipt():
    engine = FakeEngine()
    session = backend.EngineBackendSession(
        _configuration(),
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
        engine_factory=lambda configuration: engine,
    )
    session.execute_action(
        action="construct_engine",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    )
    close_evidence = session.execute_action(
        action="close_engine",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    )
    cleanup_evidence = session.execute_action(
        action="verify_cleanup",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    )
    assert close_evidence == {"rank_exit_codes": [0, 0, 0, 0]}
    assert cleanup_evidence == {
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    }

    bad_engine = FakeEngine()
    bad_engine.exit = lambda: None
    bad_session = backend.EngineBackendSession(
        _configuration(),
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
        engine_factory=lambda configuration: bad_engine,
    )
    bad_session.execute_action(
        action="construct_engine",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    )
    try:
        bad_session.execute_action(
            action="close_engine",
            scenario="construct_and_bind",
            expected=executor.contract.SCENARIOS["construct_and_bind"],
        )
    except ValueError as error:
        assert "cleanup receipt" in str(error)
    else:
        raise AssertionError("missing cleanup receipt was accepted")

    bad_rank_engine = FakeEngine()
    bad_rank_engine.exit = lambda: {
        "process_group_destroyed": False,
        "rank_exit_codes": [0, 0, 7, 0],
        "owned_children_remaining": [2],
        "rank_cleanup_receipts": [
            {"rank": 0, "process_group_destroyed": True},
            {"rank": 1, "process_group_destroyed": True},
            {"rank": 3, "process_group_destroyed": True},
        ],
    }
    bad_rank_session = backend.EngineBackendSession(
        _configuration(),
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
        engine_factory=lambda configuration: bad_rank_engine,
    )
    bad_rank_session.execute_action(
        action="construct_engine",
        scenario="construct_and_bind",
        expected=executor.contract.SCENARIOS["construct_and_bind"],
    )
    try:
        bad_rank_session.execute_action(
            action="close_engine",
            scenario="construct_and_bind",
            expected=executor.contract.SCENARIOS["construct_and_bind"],
        )
    except ValueError as error:
        assert "prove cleanup" in str(error)
    else:
        raise AssertionError("failed child cleanup was accepted")


def test_backend_session_runs_frozen_request_and_uses_independent_reference():
    engine = FakeEngine()
    expected = executor.contract.SCENARIOS["restore_w1"]
    reference_calls = []

    def reference_provider(*, scenario, prompt_token_ids, generated_tokens):
        reference_calls.append((
            scenario,
            list(prompt_token_ids),
            generated_tokens,
        ))
        return list(range(generated_tokens))

    session = backend.EngineBackendSession(
        _configuration(),
        scenario="restore_w1",
        expected=expected,
        engine_factory=lambda configuration: engine,
        reference_token_provider=reference_provider,
    )
    payload = executor.build_scenario_payloads()["restore_w1"]
    for action in (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "begin_observation",
        "submit_cached_continuation",
    ):
        session.execute_action(
            action=action,
            scenario="restore_w1",
            expected=expected,
        )
    evidence = session.execute_action(
        action="run_to_completion",
        scenario="restore_w1",
        expected=expected,
    )
    assert evidence == {
        "scheduler_steps": 64,
        "model_runner_calls": 64,
        "output_token_ids": list(range(64)),
        "reference_output_token_ids": list(range(64)),
    }
    assert reference_calls == [(
        "restore_w1",
        payload["request_prompt_token_ids"],
        64,
    )]
    submitted = [
        call for call in engine.calls if call[0] == "add_request"
    ]
    assert submitted[-1] == (
        "add_request",
        payload["request_prompt_token_ids"],
        0.0,
        64,
        True,
    )
    session.close()

    no_reference = backend.EngineBackendSession(
        _configuration(),
        scenario="restore_w1",
        expected=expected,
        engine_factory=lambda configuration: FakeEngine(),
    )
    for action in (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "begin_observation",
        "submit_cached_continuation",
    ):
        no_reference.execute_action(
            action=action,
            scenario="restore_w1",
            expected=expected,
        )
    try:
        no_reference.execute_action(
            action="run_to_completion",
            scenario="restore_w1",
            expected=expected,
        )
    except RuntimeError as error:
        assert "independent reference" in str(error)
    else:
        raise AssertionError("self-referential output authority was accepted")


def test_backend_session_observes_cache_deltas_and_applies_mutations():
    engine = FakeEngine()
    expected = executor.contract.SCENARIOS["miss_w4_stale"]
    session = backend.EngineBackendSession(
        _configuration(),
        scenario="miss_w4_stale",
        expected=expected,
        engine_factory=lambda configuration: engine,
        reference_token_provider=lambda **kwargs: list(range(32)),
    )
    session.execute_action(
        action="construct_engine",
        scenario="miss_w4_stale",
        expected=expected,
    )
    session.execute_action(
        action="configure_exact_restore",
        scenario="miss_w4_stale",
        expected=expected,
    )
    session.execute_action(
        action="verify_rank_bindings",
        scenario="miss_w4_stale",
        expected=expected,
    )
    engine.snapshot.update({
        "current_entries": 1,
        "publication_commits": 1,
        "last_publication_block_identities": [[7, 2, 99]],
    })
    session.execute_action(
        action="seed_source_fixture",
        scenario="miss_w4_stale",
        expected=expected,
    )
    session.execute_action(
        action="invalidate_block_generation",
        scenario="miss_w4_stale",
        expected=expected,
    )
    session.execute_action(
        action="begin_observation",
        scenario="miss_w4_stale",
        expected=expected,
    )
    engine.snapshot.update({
        "current_entries": 1,
        "misses": 1,
    })
    assert session.execute_action(
        action="snapshot_cache",
        scenario="miss_w4_stale",
        expected=expected,
    ) == {
        "publication_commits": 0,
        "restore_hits": 0,
        "restore_misses": 1,
        "release_events": 0,
        "cache_entries_after": 1,
        "cache_identity_match": True,
    }
    assert ("invalidate", [[7, 2, 99]], 600.0) in engine.calls
    session.close()


def test_backend_session_assigns_each_evidence_field_to_one_action():
    engine = FakeEngine()
    expected = executor.contract.SCENARIOS["publish_source"]
    session = backend.EngineBackendSession(
        _configuration(),
        scenario="publish_source",
        expected=expected,
        engine_factory=lambda configuration: engine,
        reference_token_provider=lambda **kwargs: [0],
    )
    for action in (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "begin_observation",
        "submit_source_request",
        "run_to_completion",
    ):
        session.execute_action(
            action=action,
            scenario="publish_source",
            expected=expected,
        )
    publication = session.execute_action(
        action="verify_publication_commit",
        scenario="publish_source",
        expected=expected,
    )
    snapshot = session.execute_action(
        action="snapshot_cache",
        scenario="publish_source",
        expected=expected,
    )
    assert publication == {"publication_commits": 1}
    assert "publication_commits" not in snapshot
    session.close()

    restore_engine = FakeEngine()
    restore_expected = executor.contract.SCENARIOS["restore_w1"]
    restore = backend.EngineBackendSession(
        _configuration(),
        scenario="restore_w1",
        expected=restore_expected,
        engine_factory=lambda configuration: restore_engine,
        reference_token_provider=lambda **kwargs: list(range(64)),
    )
    for action in (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
    ):
        restore.execute_action(
            action=action,
            scenario="restore_w1",
            expected=restore_expected,
        )
    restore_engine.snapshot.update({
        "current_entries": 1,
        "publication_commits": 1,
        "last_publication_block_identities": [[7, 2, 99]],
    })
    restore.execute_action(
        action="seed_source_fixture",
        scenario="restore_w1",
        expected=restore_expected,
    )
    restore.execute_action(
        action="begin_observation",
        scenario="restore_w1",
        expected=restore_expected,
    )
    release = restore.execute_action(
        action="drain_release_events",
        scenario="restore_w1",
        expected=restore_expected,
    )
    snapshot = restore.execute_action(
        action="snapshot_cache",
        scenario="restore_w1",
        expected=restore_expected,
    )
    assert release == {"release_events": 0}
    assert "release_events" not in snapshot
    restore.close()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine backend session tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
