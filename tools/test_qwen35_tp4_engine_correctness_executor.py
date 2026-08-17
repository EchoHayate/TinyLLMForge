from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXECUTOR_PATH = TOOLS / "qwen35_tp4_engine_correctness_executor.py"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_engine_contract_for_executor_test",
    "qwen35_tp4_engine_correctness_contract.py",
)
executor_module = _load(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)


MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)


def _configuration():
    return executor_module.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256=MODEL_MANIFEST_SHA256,
        source_tree_sha256="d" * 64,
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


def _row(scenario, expected):
    outputs = list(range(expected["generated_tokens"]))
    return {
        "scenario": scenario,
        "engine_class": contract.ENGINE_CLASS,
        "model_runner_class": contract.MODEL_RUNNER_CLASS,
        "rank_inventory": [0, 1, 2, 3],
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


class FakeRuntime:

    def __init__(self, configuration):
        self.configuration = configuration
        self.calls = []
        self.closed = False

    def run_scenario(self, *, scenario, expected, plan):
        self.calls.append((scenario, tuple(plan["actions"])))
        return _row(scenario, expected)

    def close(self):
        self.closed = True


class FakeActionSession:

    def __init__(
        self,
        scenario,
        expected,
        *,
        omit_action=None,
        fail_action=None,
    ):
        self.scenario = scenario
        self.expected = expected
        self.omit_action = omit_action
        self.fail_action = fail_action
        self.actions = []
        self.closed = False

    def execute_action(self, *, action, scenario, expected):
        assert scenario == self.scenario
        assert expected == self.expected
        self.actions.append(action)
        if action == self.fail_action:
            raise RuntimeError("synthetic action failure")
        if action == self.omit_action:
            return {}
        outputs = list(range(expected["generated_tokens"]))
        evidence = {
            "construct_engine": {
                "engine_class": contract.ENGINE_CLASS,
                "model_runner_class": contract.MODEL_RUNNER_CLASS,
            },
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
                "publication_commits": expected["publication_commits"],
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
                        "release_events": expected["release_events"],
                    }
                ),
                "cache_entries_after": expected["cache_entries_after"],
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


def test_module_is_dependency_light():
    tree = ast.parse(EXECUTOR_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert not any(
        name == "tinyvllm" or name.startswith("tinyvllm.")
        for name in imports
    )


def test_configuration_is_strict_and_serializable():
    configuration = _configuration()
    payload = configuration.to_payload()
    assert payload["world_size"] == 4
    assert payload["gpu_indices"] == [0, 1, 2, 3]
    assert payload["dist_port"] != payload["master_port"]
    assert set(payload) == set(executor_module.CONFIGURATION_FIELDS)

    for changes, message in (
        ({"gpu_indices": (0, 1, 1, 3)}, "unique"),
        ({"dist_port": 31002}, "different"),
        ({"timeout_s": 0}, "positive"),
        ({"model_manifest_sha256": "A" * 64}, "lowercase"),
        ({"model_dir": "relative/model"}, "absolute"),
    ):
        values = payload | changes
        values.pop("world_size")
        values["gpu_indices"] = tuple(values["gpu_indices"])
        try:
            executor_module.ExecutorConfiguration(**values)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid executor configuration accepted")


def test_scenario_plan_is_exact_and_carries_expected_counts():
    plans = executor_module.build_scenario_plans()
    assert tuple(plans) == tuple(contract.SCENARIOS)
    assert plans["construct_and_bind"]["actions"] == (
        "construct_engine",
        "begin_observation",
        "configure_exact_restore",
        "verify_rank_bindings",
        "close_engine",
        "verify_cleanup",
    )
    assert plans["publish_source"]["actions"][-3:] == (
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    )
    assert plans["restore_w1"]["actions"] == (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "begin_observation",
        "submit_cached_continuation",
        "run_to_completion",
        "drain_release_events",
        "snapshot_cache",
        "close_engine",
        "verify_cleanup",
    )
    assert plans["miss_w4_clear"]["actions"][:6] == (
        "construct_engine",
        "configure_exact_restore",
        "verify_rank_bindings",
        "seed_source_fixture",
        "clear_reusable_cache",
        "begin_observation",
    )
    for scenario, plan in plans.items():
        assert plan["scenario"] == scenario
        assert plan["expected"] == contract.SCENARIOS[scenario]
        assert plan["isolation"] == "fresh_engine_process_group"
        assert plan["actions"][0] == "construct_engine"
        assert plan["actions"][-2:] == (
            "close_engine",
            "verify_cleanup",
        )


def test_scenario_payloads_reuse_frozen_workload_tokens():
    payloads = executor_module.build_scenario_payloads()
    assert tuple(payloads) == tuple(contract.SCENARIOS)
    construct = payloads["construct_and_bind"]
    assert construct == {
        "workload": None,
        "source_prompt_token_ids": [],
        "request_prompt_token_ids": [],
        "generated_tokens": 0,
        "invalidation": {"kind": "none"},
    }
    publish = payloads["publish_source"]
    restore = payloads["restore_w1"]
    assert publish["workload"] == "w1_medium_reuse"
    assert restore["workload"] == "w1_medium_reuse"
    assert len(publish["source_prompt_token_ids"]) == 1024
    assert len(publish["source_prompt_token_ids"]) % 256 == 0
    assert publish["source_prompt_token_ids"] == (
        restore["source_prompt_token_ids"]
    )
    assert len(restore["request_prompt_token_ids"]) == 1088
    assert publish["generated_tokens"] == 1
    assert restore["generated_tokens"] == 64
    miss_scenarios = (
        "miss_w4_token",
        "miss_w4_stale",
        "miss_w4_clear",
    )
    assert [
        payloads[name]["invalidation"]["kind"]
        for name in miss_scenarios
    ] == [
        "token_mismatch",
        "stale_block_generation",
        "cache_clear",
    ]
    assert all(
        payloads[name]["generated_tokens"] == 32
        for name in miss_scenarios
    )
    assert (
        payloads["miss_w4_token"]["request_prompt_token_ids"][512]
        != payloads["miss_w4_token"]["source_prompt_token_ids"][512]
    )


def test_executor_is_lazy_ordered_and_closes_runtime():
    runtimes = []

    def runtime_factory(configuration):
        runtime = FakeRuntime(configuration)
        runtimes.append(runtime)
        return runtime

    executor = executor_module.EngineCorrectnessExecutor(
        configuration=_configuration(),
        runtime_factory=runtime_factory,
    )
    assert runtimes == []
    rows = []
    for scenario, expected in contract.SCENARIOS.items():
        rows.append(executor.run_scenario(
            scenario=scenario,
            expected=expected,
        ))
    assert contract.classify_rows(rows)["classification"] == "PASS"
    assert len(runtimes) == 1
    assert [call[0] for call in runtimes[0].calls] == list(
        contract.SCENARIOS
    )
    executor.close()
    executor.close()
    assert runtimes[0].closed is True


def test_out_of_order_scenario_fails_before_runtime_construction():
    calls = []

    def runtime_factory(configuration):
        calls.append(configuration)
        return FakeRuntime(configuration)

    executor = executor_module.EngineCorrectnessExecutor(
        configuration=_configuration(),
        runtime_factory=runtime_factory,
    )
    try:
        executor.run_scenario(
            scenario="publish_source",
            expected=contract.SCENARIOS["publish_source"],
        )
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("out-of-order scenario was accepted")
    assert calls == []
    executor.close()


def test_default_runtime_factory_is_fail_closed():
    factory = executor_module.build_executor_factory(_configuration())
    executor = factory()
    try:
        executor.run_scenario(
            scenario="construct_and_bind",
            expected=contract.SCENARIOS["construct_and_bind"],
        )
    except RuntimeError as error:
        assert "real Qwen3.5 TP4 Engine runtime is not implemented" in str(
            error
        )
    else:
        raise AssertionError("real runtime was silently enabled")
    executor.close()


def test_audited_runtime_uses_fresh_session_per_scenario():
    sessions = []

    def backend_factory(configuration, *, scenario, expected):
        session = FakeActionSession(scenario, expected)
        sessions.append(session)
        return session

    runtime = executor_module.AuditedScenarioRuntime(
        _configuration(),
        backend_factory=backend_factory,
    )
    rows = []
    plans = executor_module.build_scenario_plans()
    for scenario, expected in contract.SCENARIOS.items():
        rows.append(runtime.run_scenario(
            scenario=scenario,
            expected=expected,
            plan=plans[scenario],
        ))
    runtime.close()
    assert contract.classify_rows(rows)["classification"] == "PASS"
    assert len(sessions) == len(contract.SCENARIOS)
    assert all(session.closed for session in sessions)
    assert [
        tuple(session.actions)
        for session in sessions
    ] == [
        plans[scenario]["actions"]
        for scenario in contract.SCENARIOS
    ]


def test_audited_runtime_rejects_missing_required_action_evidence():
    sessions = []

    def backend_factory(configuration, *, scenario, expected):
        session = FakeActionSession(
            scenario,
            expected,
            omit_action="verify_rank_bindings",
        )
        sessions.append(session)
        return session

    runtime = executor_module.AuditedScenarioRuntime(
        _configuration(),
        backend_factory=backend_factory,
    )
    plan = executor_module.build_scenario_plans()["construct_and_bind"]
    try:
        runtime.run_scenario(
            scenario="construct_and_bind",
            expected=contract.SCENARIOS["construct_and_bind"],
            plan=plan,
        )
    except ValueError as error:
        assert "classification" in str(error)
    else:
        raise AssertionError("missing action evidence was accepted")
    assert sessions[0].closed is True
    runtime.close()


def test_audited_runtime_closes_session_after_action_failure():
    sessions = []

    def backend_factory(configuration, *, scenario, expected):
        session = FakeActionSession(
            scenario,
            expected,
            fail_action="configure_exact_restore",
        )
        sessions.append(session)
        return session

    runtime = executor_module.AuditedScenarioRuntime(
        _configuration(),
        backend_factory=backend_factory,
    )
    plan = executor_module.build_scenario_plans()["construct_and_bind"]
    try:
        runtime.run_scenario(
            scenario="construct_and_bind",
            expected=contract.SCENARIOS["construct_and_bind"],
            plan=plan,
        )
    except RuntimeError as error:
        assert "synthetic action failure" in str(error)
    else:
        raise AssertionError("action failure was hidden")
    assert sessions[0].closed is True
    runtime.close()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine correctness executor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
