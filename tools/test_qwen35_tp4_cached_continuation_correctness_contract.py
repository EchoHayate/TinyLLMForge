from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/qwen35_tp4_cached_continuation_correctness_contract.py"
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_cached_continuation_correctness_contract",
        CONTRACT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _row(
    workload,
    *,
    request_index,
    outcome,
    output_token_ids=None,
    logits_max_abs_diff=0.0,
):
    workload_payload = contract.workload_payload(workload)
    spec = workload_payload["spec"]
    expected_restore = workload in contract.HIT_WORKLOADS
    expected_reason = (
        "exact_hit"
        if expected_restore
        else contract.W4_EXPECTED_REASONS[request_index]
    )
    return {
        "workload": workload,
        "request_index": request_index,
        "outcome": outcome,
        "restore_hit": expected_restore,
        "restore_reason": expected_reason,
        "prompt_tokens": (
            spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        ),
        "reused_tokens": (
            spec["shared_prefix_tokens"] if expected_restore else 0
        ),
        "executed_prefill_tokens": (
            spec["suffix_tokens"]
            if expected_restore
            else spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        ),
        "output_token_ids": (
            list(range(spec["generated_tokens"]))
            if output_token_ids is None
            else output_token_ids
        ),
        "reference_output_token_ids": list(
            range(spec["generated_tokens"])
        ),
        "logits_max_abs_diff": logits_max_abs_diff,
        "logits_allclose": True,
        "cache_identity_match": True,
        "rank_inventory": [0, 1, 2, 3],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    }


def _complete_rows():
    rows = []
    for workload in contract.WORKLOADS:
        continuation_count = contract.workload_payload(workload)[
            "spec"
        ]["continuations"]
        for request_index in range(continuation_count):
            rows.append(_row(
                workload,
                request_index=request_index,
                outcome="continuation",
            ))
    return rows


def test_contract_identity_and_artifact_inventory_are_frozen():
    assert contract.SCHEMA_VERSION == (
        "qwen35.tp4-cached-continuation-correctness.v1"
    )
    assert contract.WORLD_SIZE == 4
    assert contract.HIT_WORKLOADS == (
        "w1_medium_reuse",
        "w2_long_reuse",
        "w3_batched_fanout",
    )
    assert contract.W4_EXPECTED_REASONS == (
        "token_mismatch",
        "stale_block_generation",
        "cache_clear",
    )
    assert contract.ARTIFACT_NAMES == (
        "cached_continuation_correctness.json",
        "reference_outputs.json",
        "restored_outputs.json",
        "registered_logits.json",
        "source_manifest.json",
    )
    assert len(contract.WORKLOAD_MANIFEST_SHA256) == 64


def test_complete_exact_restore_matrix_passes():
    result = contract.classify_rows(_complete_rows())

    assert result["classification"] == "PASS"
    assert result["checks"]["row_count"] == 19
    assert result["checks"]["restore_hits"] == 16
    assert result["checks"]["w4_misses"] == 3


def test_output_or_registered_logits_mismatch_fails():
    rows = _complete_rows()
    rows[0]["output_token_ids"][-1] = 999
    assert contract.classify_rows(rows)["classification"] == "FAIL"

    rows = _complete_rows()
    rows[0]["logits_allclose"] = False
    rows[0]["logits_max_abs_diff"] = 1.0
    assert contract.classify_rows(rows)["classification"] == "FAIL"


def test_restore_semantics_and_accounting_are_exact():
    rows = _complete_rows()
    w1 = next(row for row in rows if row["workload"] == "w1_medium_reuse")
    w1["restore_hit"] = False
    assert contract.classify_rows(rows)["classification"] == "FAIL"

    rows = _complete_rows()
    w4 = next(row for row in rows if row["workload"] == "w4_miss_invalidation")
    w4["restore_hit"] = True
    assert contract.classify_rows(rows)["classification"] == "FAIL"

    rows = _complete_rows()
    rows[0]["executed_prefill_tokens"] += 1
    assert contract.classify_rows(rows)["classification"] == "FAIL"


def test_rank_cache_identity_and_cleanup_fail_closed():
    for field, value in (
        ("cache_identity_match", False),
        ("rank_inventory", [0, 1, 2]),
        ("process_group_destroyed", False),
        ("owned_children_remaining", [12345]),
    ):
        rows = _complete_rows()
        rows[0][field] = value
        assert contract.classify_rows(rows)["classification"] == "FAIL"


def test_duplicate_missing_or_extra_rows_fail():
    rows = _complete_rows()
    assert contract.classify_rows(rows[:-1])["classification"] == "FAIL"
    assert contract.classify_rows(rows + [dict(rows[0])])[
        "classification"
    ] == "FAIL"
    extra = dict(rows[0])
    extra["workload"] = "w0_short_control"
    assert contract.classify_rows(rows + [extra])[
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
        "qwen35 TP4 cached-continuation correctness contract tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
