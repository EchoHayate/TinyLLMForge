from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


def _load_benchmark_contract():
    module_name = "qwen35_tp4_hybrid_prefix_benchmark_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_hybrid_prefix_benchmark_contract.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BENCHMARK = _load_benchmark_contract()

SCHEMA_VERSION = "qwen35.tp4-cached-continuation-correctness.v1"
WORLD_SIZE = 4
WORKLOADS = (
    "w1_medium_reuse",
    "w2_long_reuse",
    "w3_batched_fanout",
    "w4_miss_invalidation",
)
HIT_WORKLOADS = WORKLOADS[:3]
W4_EXPECTED_REASONS = (
    "token_mismatch",
    "stale_block_generation",
    "cache_clear",
)
WORKLOAD_MANIFEST_SHA256 = (
    _BENCHMARK.canonical_json_file_sha256(
        _BENCHMARK.workload_manifest_payload()
    )
)
REGISTERED_LOGITS_ATOL = 2e-5
ARTIFACT_NAMES = (
    "cached_continuation_correctness.json",
    "reference_outputs.json",
    "restored_outputs.json",
    "registered_logits.json",
    "source_manifest.json",
)


def workload_payload(workload):
    if workload not in WORKLOADS:
        raise ValueError(f"unsupported cached-continuation workload: {workload}")
    return _BENCHMARK.workload_payload(workload)


def _expected_keys():
    return tuple(
        (workload, request_index)
        for workload in WORKLOADS
        for request_index in range(
            workload_payload(workload)["spec"]["continuations"]
        )
    )


def _nonnegative_integer(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def _validate_row(row, expected_key):
    failures = []
    if not isinstance(row, dict):
        return ["row is not an object"]
    required = {
        "workload",
        "request_index",
        "outcome",
        "restore_hit",
        "restore_reason",
        "prompt_tokens",
        "reused_tokens",
        "executed_prefill_tokens",
        "output_token_ids",
        "reference_output_token_ids",
        "logits_max_abs_diff",
        "logits_allclose",
        "cache_identity_match",
        "rank_inventory",
        "process_group_destroyed",
        "owned_children_remaining",
    }
    if set(row) != required:
        return ["row schema mismatch"]
    workload, request_index = expected_key
    if (
        row["workload"] != workload
        or row["request_index"] != request_index
    ):
        failures.append("row identity mismatch")
    if row["outcome"] != "continuation":
        failures.append("row outcome mismatch")
    spec = workload_payload(workload)["spec"]
    expected_hit = workload in HIT_WORKLOADS
    expected_reason = (
        "exact_hit"
        if expected_hit
        else W4_EXPECTED_REASONS[request_index]
    )
    if (
        not isinstance(row["restore_hit"], bool)
        or row["restore_hit"] is not expected_hit
    ):
        failures.append("restore hit semantics mismatch")
    if row["restore_reason"] != expected_reason:
        failures.append("restore reason mismatch")
    expected_prompt = (
        spec["shared_prefix_tokens"] + spec["suffix_tokens"]
    )
    expected_reused = (
        spec["shared_prefix_tokens"] if expected_hit else 0
    )
    expected_prefill = (
        spec["suffix_tokens"] if expected_hit else expected_prompt
    )
    for name, expected in (
        ("prompt_tokens", expected_prompt),
        ("reused_tokens", expected_reused),
        ("executed_prefill_tokens", expected_prefill),
    ):
        if (
            not _nonnegative_integer(row[name])
            or row[name] != expected
        ):
            failures.append(f"{name} mismatch")
    output = row["output_token_ids"]
    reference = row["reference_output_token_ids"]
    if (
        not isinstance(output, list)
        or not isinstance(reference, list)
        or len(output) != spec["generated_tokens"]
        or len(reference) != spec["generated_tokens"]
        or any(not _nonnegative_integer(value) for value in output)
        or any(not _nonnegative_integer(value) for value in reference)
        or output != reference
    ):
        failures.append("output token mismatch")
    logits_diff = row["logits_max_abs_diff"]
    if (
        isinstance(logits_diff, bool)
        or not isinstance(logits_diff, (int, float))
        or not math.isfinite(logits_diff)
        or logits_diff < 0
        or logits_diff > REGISTERED_LOGITS_ATOL
        or row["logits_allclose"] is not True
    ):
        failures.append("registered logits mismatch")
    if row["cache_identity_match"] is not True:
        failures.append("cache identity mismatch")
    if row["rank_inventory"] != list(range(WORLD_SIZE)):
        failures.append("rank inventory mismatch")
    if row["process_group_destroyed"] is not True:
        failures.append("process group cleanup mismatch")
    if row["owned_children_remaining"] != []:
        failures.append("owned child cleanup mismatch")
    return failures


def classify_rows(rows):
    expected = _expected_keys()
    failures = []
    if not isinstance(rows, (tuple, list)):
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "FAIL",
            "checks": {
                "row_count": 0,
                "restore_hits": 0,
                "w4_misses": 0,
            },
            "failures": ["rows must be a list"],
        }
    actual_keys = []
    for row in rows:
        if isinstance(row, dict):
            actual_keys.append((
                row.get("workload"),
                row.get("request_index"),
            ))
        else:
            actual_keys.append((None, None))
    if tuple(actual_keys) != expected:
        failures.append("cached-continuation row matrix mismatch")
    if len(rows) == len(expected):
        for row, expected_key in zip(rows, expected):
            failures.extend(_validate_row(row, expected_key))
    else:
        failures.append("cached-continuation row count mismatch")
    restore_hits = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("workload") in HIT_WORKLOADS
        and row.get("restore_hit") is True
    )
    w4_misses = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("workload") == "w4_miss_invalidation"
        and row.get("restore_hit") is False
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "PASS" if not failures else "FAIL",
        "checks": {
            "row_count": len(rows),
            "restore_hits": restore_hits,
            "w4_misses": w4_misses,
        },
        "failures": failures,
    }
