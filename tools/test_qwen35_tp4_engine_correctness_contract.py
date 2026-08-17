from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/qwen35_tp4_engine_correctness_contract.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_engine_correctness_contract",
        PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load()


def _row(scenario):
    expected = contract.SCENARIOS[scenario]
    return {
        "scenario": scenario,
        "engine_class": "tinyvllm.engine.llm_engine.LLMEngine",
        "model_runner_class": (
            "tinyvllm.engine.model_runner.ModelRunner"
        ),
        "rank_inventory": [0, 1, 2, 3],
        "ack_ranks": [1, 2, 3],
        "scheduler_steps": expected["scheduler_steps"],
        "model_runner_calls": expected["model_runner_calls"],
        "output_token_ids": list(range(expected["generated_tokens"])),
        "reference_output_token_ids": list(
            range(expected["generated_tokens"])
        ),
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


def _rows():
    return [_row(name) for name in contract.SCENARIOS]


def test_identity_scenarios_and_artifacts_are_frozen():
    assert contract.SCHEMA_VERSION == (
        "qwen35.tp4-engine-model-runner-correctness.v1"
    )
    assert contract.WORLD_SIZE == 4
    assert tuple(contract.SCENARIOS) == (
        "construct_and_bind",
        "publish_source",
        "restore_w1",
        "miss_w4_token",
        "miss_w4_stale",
        "miss_w4_clear",
    )
    assert contract.ARTIFACT_NAMES == (
        "engine_correctness.json",
        "scheduler_observations.json",
        "rank_events.json",
        "source_manifest.json",
    )
    assert contract.MODEL_RUNNER_CALL_SEMANTICS == (
        "model execution calls issued by LLMEngine.step; "
        "control-plane acknowledged commands are excluded"
    )
    assert contract.SCENARIOS["construct_and_bind"][
        "model_runner_calls"
    ] == 0
    assert all(
        expected["model_runner_calls"]
        == expected["scheduler_steps"]
        for expected in contract.SCENARIOS.values()
    )
    assert {
        scenario: expected["release_events"]
        for scenario, expected in contract.SCENARIOS.items()
    } == {
        "construct_and_bind": 0,
        "publish_source": 1,
        "restore_w1": 0,
        "miss_w4_token": 0,
        "miss_w4_stale": 0,
        "miss_w4_clear": 0,
    }
    assert {
        scenario: {
            name: expected[name]
            for name in (
                "scheduler_steps",
                "model_runner_calls",
                "publication_commits",
                "restore_hits",
                "restore_misses",
                "cache_entries_after",
            )
        }
        for scenario, expected in contract.SCENARIOS.items()
        if scenario.startswith("miss_w4_")
    } == {
        "miss_w4_token": {
            "scheduler_steps": 33,
            "model_runner_calls": 33,
            "publication_commits": 1,
            "restore_hits": 0,
            "restore_misses": 0,
            "cache_entries_after": 2,
        },
        "miss_w4_stale": {
            "scheduler_steps": 33,
            "model_runner_calls": 33,
            "publication_commits": 1,
            "restore_hits": 0,
            "restore_misses": 1,
            "cache_entries_after": 1,
        },
        "miss_w4_clear": {
            "scheduler_steps": 33,
            "model_runner_calls": 33,
            "publication_commits": 1,
            "restore_hits": 0,
            "restore_misses": 1,
            "cache_entries_after": 1,
        },
    }


def test_complete_engine_matrix_passes():
    result = contract.classify_rows(_rows())

    assert result["classification"] == "PASS"
    assert result["checks"]["scenario_count"] == 6
    assert result["checks"]["restore_hits"] == 1
    assert result["checks"]["restore_misses"] == 2


def test_real_class_rank_and_ack_identity_fail_closed():
    for field, value in (
        ("engine_class", "fake.Engine"),
        ("model_runner_class", "fake.Runner"),
        ("rank_inventory", [0, 1, 2]),
        ("ack_ranks", [1, 3]),
    ):
        rows = _rows()
        rows[0][field] = value
        assert contract.classify_rows(rows)["classification"] == "FAIL"


def test_scheduler_model_runner_and_output_equality_are_required():
    for field, delta in (
        ("scheduler_steps", 1),
        ("model_runner_calls", 1),
    ):
        rows = _rows()
        rows[2][field] += delta
        assert contract.classify_rows(rows)["classification"] == "FAIL"
    rows = _rows()
    rows[2]["output_token_ids"][-1] = 999
    assert contract.classify_rows(rows)["classification"] == "FAIL"


def test_cache_lifecycle_is_exact():
    for scenario, field, value in (
        ("publish_source", "publication_commits", 0),
        ("restore_w1", "restore_hits", 0),
        ("miss_w4_token", "restore_misses", 1),
        ("miss_w4_stale", "publication_commits", 0),
        ("restore_w1", "release_events", 1),
        ("miss_w4_clear", "cache_entries_after", 0),
        ("restore_w1", "cache_identity_match", False),
    ):
        rows = _rows()
        row = next(row for row in rows if row["scenario"] == scenario)
        row[field] = value
        assert contract.classify_rows(rows)["classification"] == "FAIL"


def test_cleanup_and_matrix_fail_closed():
    for field, value in (
        ("process_group_destroyed", False),
        ("rank_exit_codes", [0, 0, 0, 1]),
        ("owned_children_remaining", [123]),
    ):
        rows = _rows()
        rows[-1][field] = value
        assert contract.classify_rows(rows)["classification"] == "FAIL"
    assert contract.classify_rows(_rows()[:-1])[
        "classification"
    ] == "FAIL"
    assert contract.classify_rows(_rows() + [_rows()[0]])[
        "classification"
    ] == "FAIL"


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine correctness contract tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
