from __future__ import annotations


SCHEMA_VERSION = "qwen35.tp4-engine-model-runner-correctness.v1"
WORLD_SIZE = 4
ENGINE_CLASS = "tinyvllm.engine.llm_engine.LLMEngine"
MODEL_RUNNER_CLASS = "tinyvllm.engine.model_runner.ModelRunner"
MODEL_RUNNER_CALL_SEMANTICS = (
    "model execution calls issued by LLMEngine.step; "
    "control-plane acknowledged commands are excluded"
)
SCENARIOS = {
    "construct_and_bind": {
        "scheduler_steps": 0,
        "model_runner_calls": 0,
        "generated_tokens": 0,
        "publication_commits": 0,
        "restore_hits": 0,
        "restore_misses": 0,
        "release_events": 0,
        "cache_entries_after": 0,
    },
    "publish_source": {
        "scheduler_steps": 1,
        "model_runner_calls": 1,
        "generated_tokens": 1,
        "publication_commits": 1,
        "restore_hits": 0,
        "restore_misses": 0,
        "release_events": 1,
        "cache_entries_after": 1,
    },
    "restore_w1": {
        "scheduler_steps": 64,
        "model_runner_calls": 64,
        "generated_tokens": 64,
        "publication_commits": 0,
        "restore_hits": 1,
        "restore_misses": 0,
        "release_events": 0,
        "cache_entries_after": 1,
    },
    "miss_w4_token": {
        "scheduler_steps": 33,
        "model_runner_calls": 33,
        "generated_tokens": 32,
        "publication_commits": 1,
        "restore_hits": 0,
        "restore_misses": 0,
        "release_events": 0,
        "cache_entries_after": 2,
    },
    "miss_w4_stale": {
        "scheduler_steps": 33,
        "model_runner_calls": 33,
        "generated_tokens": 32,
        "publication_commits": 1,
        "restore_hits": 0,
        "restore_misses": 1,
        "release_events": 0,
        "cache_entries_after": 1,
    },
    "miss_w4_clear": {
        "scheduler_steps": 33,
        "model_runner_calls": 33,
        "generated_tokens": 32,
        "publication_commits": 1,
        "restore_hits": 0,
        "restore_misses": 1,
        "release_events": 0,
        "cache_entries_after": 1,
    },
}
ARTIFACT_NAMES = (
    "engine_correctness.json",
    "scheduler_observations.json",
    "rank_events.json",
    "source_manifest.json",
)


def _integer(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def _validate_row(row, scenario):
    if not isinstance(row, dict):
        return ["row is not an object"]
    required = {
        "scenario",
        "engine_class",
        "model_runner_class",
        "rank_inventory",
        "ack_ranks",
        "scheduler_steps",
        "model_runner_calls",
        "output_token_ids",
        "reference_output_token_ids",
        "publication_commits",
        "restore_hits",
        "restore_misses",
        "release_events",
        "cache_entries_after",
        "cache_identity_match",
        "process_group_destroyed",
        "rank_exit_codes",
        "owned_children_remaining",
    }
    if set(row) != required:
        return ["row schema mismatch"]
    failures = []
    expected = SCENARIOS[scenario]
    if row["scenario"] != scenario:
        failures.append("scenario identity mismatch")
    if row["engine_class"] != ENGINE_CLASS:
        failures.append("Engine class identity mismatch")
    if row["model_runner_class"] != MODEL_RUNNER_CLASS:
        failures.append("ModelRunner class identity mismatch")
    if row["rank_inventory"] != list(range(WORLD_SIZE)):
        failures.append("rank inventory mismatch")
    if row["ack_ranks"] != list(range(1, WORLD_SIZE)):
        failures.append("ack rank inventory mismatch")
    for name in (
        "scheduler_steps",
        "model_runner_calls",
        "publication_commits",
        "restore_hits",
        "restore_misses",
        "release_events",
        "cache_entries_after",
    ):
        if (
            not _integer(row[name])
            or row[name] != expected[name]
        ):
            failures.append(f"{name} mismatch")
    outputs = row["output_token_ids"]
    reference = row["reference_output_token_ids"]
    if (
        not isinstance(outputs, list)
        or not isinstance(reference, list)
        or len(outputs) != expected["generated_tokens"]
        or len(reference) != expected["generated_tokens"]
        or any(not _integer(value) for value in outputs)
        or any(not _integer(value) for value in reference)
        or outputs != reference
    ):
        failures.append("output token mismatch")
    if row["cache_identity_match"] is not True:
        failures.append("cache identity mismatch")
    if row["process_group_destroyed"] is not True:
        failures.append("process group cleanup mismatch")
    if row["rank_exit_codes"] != [0] * WORLD_SIZE:
        failures.append("rank exit mismatch")
    if row["owned_children_remaining"] != []:
        failures.append("owned child cleanup mismatch")
    return failures


def classify_rows(rows):
    failures = []
    scenario_names = tuple(SCENARIOS)
    if not isinstance(rows, (tuple, list)):
        rows = []
        failures.append("rows must be a list")
    actual = tuple(
        row.get("scenario") if isinstance(row, dict) else None
        for row in rows
    )
    if actual != scenario_names:
        failures.append("Engine correctness scenario matrix mismatch")
    if len(rows) == len(scenario_names):
        for row, scenario in zip(rows, scenario_names):
            failures.extend(_validate_row(row, scenario))
    else:
        failures.append("Engine correctness scenario count mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "PASS" if not failures else "FAIL",
        "checks": {
            "scenario_count": len(rows),
            "restore_hits": sum(
                row.get("restore_hits", 0)
                for row in rows if isinstance(row, dict)
            ),
            "restore_misses": sum(
                row.get("restore_misses", 0)
                for row in rows if isinstance(row, dict)
            ),
        },
        "failures": failures,
    }
