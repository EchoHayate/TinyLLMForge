from __future__ import annotations

import copy

import pytest

from tools.cross_engine_k8_workload import (
    OPTIONAL_ARM,
    REQUIRED_ARMS,
    aggregate_case_rows,
    arm_order,
    build_workload_manifest,
    classify_comparison,
    expected_case_identities,
    reconcile_correctness,
    reconstruct_metrics,
)


def test_workload_is_frozen_and_prompt_tokens_are_deterministic():
    first = build_workload_manifest("a" * 64)
    second = build_workload_manifest("a" * 64)

    assert first == second
    assert first["model"] == "Qwen3-0.6B"
    assert first["precision"] == "bfloat16"
    assert first["prompt_lengths"] == [256, 2048, 8192]
    assert first["output_tokens"] == 128
    assert first["warmups"] == 2
    assert first["measured_repetitions"] == 7
    assert all(
        len(case["prompt_token_ids"]) == case["prompt_tokens"]
        for case in first["cases"]
    )


def test_workload_uses_model_bos_and_frozen_natural_sentence_cycle():
    manifest = build_workload_manifest("a" * 64)
    pattern = [576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]

    assert manifest["prompt_strategy"] == (
        "periodic_natural_sentence_model_config_bos"
    )
    assert manifest["prompt_bos_token_id"] == 151643
    assert manifest["prompt_pattern_text"] == (
        " The quick brown fox jumps over the lazy dog."
    )
    assert manifest["prompt_pattern_token_ids"] == pattern
    for case in manifest["cases"]:
        prompt = case["prompt_token_ids"]
        assert prompt[0] == manifest["prompt_bos_token_id"]
        expected_body = (
            pattern
            * ((case["prompt_tokens"] - 1 + len(pattern) - 1) // len(pattern))
        )[: case["prompt_tokens"] - 1]
        assert prompt[1:] == expected_body


def test_workload_rejects_invalid_model_inventory_digest():
    with pytest.raises(ValueError, match="model_inventory_sha256"):
        build_workload_manifest("not-a-digest")


def test_rotation_balances_first_position():
    arms = REQUIRED_ARMS + (OPTIONAL_ARM,)
    orders = [arm_order(index, arms) for index in range(8)]

    assert {order[0] for order in orders} == set(arms)
    assert all(set(order) == set(arms) for order in orders)


def test_rotation_rejects_duplicate_or_missing_required_arms():
    with pytest.raises(ValueError, match="eligible_arms"):
        arm_order(0, ("tinyllmforge_exact_k8",) * 2)
    with pytest.raises(ValueError, match="required"):
        arm_order(0, ("tinyllmforge_exact_k8",))


def test_expected_case_identities_cover_complete_matrix():
    manifest = build_workload_manifest("a" * 64)

    identities = expected_case_identities(manifest, REQUIRED_ARMS)

    assert len(identities) == 7 * 3 * 3
    assert len(set(identities)) == len(identities)
    assert identities[0] == (
        0,
        "short",
        "tinyllmforge_host_greedy",
    )


def test_metric_reconstruction_uses_monotonic_token_timestamps():
    metrics = reconstruct_metrics(
        request_start_ns=0,
        token_timestamps_ns=[10, 20, 35, 55],
        request_end_ns=60,
        output_tokens=4,
    )

    assert metrics["ttft_ns"] == 10
    assert metrics["tpot_samples_ns"] == [10, 15, 20]
    assert metrics["median_tpot_ns"] == 15
    assert metrics["p95_tpot_ns"] == 20
    assert metrics["p99_tpot_ns"] == 20
    assert metrics["e2e_ns"] == 60
    assert metrics["output_tokens_per_second"] == pytest.approx(
        4 / 60e-9
    )


@pytest.mark.parametrize(
    ("timestamps", "end", "message"),
    (
        ([], 10, "output token count"),
        ([10, 9], 20, "monotonic"),
        ([10, 20], 19, "request_end_ns"),
    ),
)
def test_metric_reconstruction_rejects_invalid_timeline(
    timestamps,
    end,
    message,
):
    with pytest.raises(ValueError, match=message):
        reconstruct_metrics(
            request_start_ns=0,
            token_timestamps_ns=timestamps,
            request_end_ns=end,
            output_tokens=2,
        )


def test_correctness_requires_all_arms_to_match_reference():
    expected = {
        "short": [1, 2, 3],
        "medium": [4, 5, 6],
        "long": [7, 8, 9],
    }
    rows = [
        {
            "context": context,
            "arm": arm,
            "token_ids": tokens,
            "output_tokens": len(tokens),
        }
        for context, tokens in expected.items()
        for arm in REQUIRED_ARMS
    ]

    result = reconcile_correctness(
        rows,
        expected_tokens=expected,
        eligible_arms=REQUIRED_ARMS,
    )

    assert result["valid"] is True
    assert result["mismatches"] == []


def test_correctness_mismatch_excludes_only_affected_arm():
    expected = {"short": [1, 2, 3]}
    rows = [
        {
            "context": "short",
            "arm": arm,
            "token_ids": [1, 2, 3],
            "output_tokens": 3,
        }
        for arm in REQUIRED_ARMS
    ]
    rows[-1]["token_ids"] = [1, 2, 9]

    result = reconcile_correctness(
        rows,
        expected_tokens=expected,
        eligible_arms=REQUIRED_ARMS,
    )

    assert result["valid"] is False
    assert result["eligible_arms"] == list(REQUIRED_ARMS[:-1])
    assert result["mismatches"][0]["arm"] == "vllm_default_greedy"


def _case_row(arm, context, repetition, *, tpot, throughput):
    return {
        "arm": arm,
        "context": context,
        "repetition": repetition,
        "median_tpot_ns": tpot,
        "p95_tpot_ns": tpot,
        "p99_tpot_ns": tpot,
        "ttft_ns": 1_000,
        "e2e_ns": 10_000,
        "output_tokens_per_second": throughput,
        "peak_gpu_memory_bytes": 1_000,
        "peak_rss_bytes": 2_000,
        "performance_eligible": True,
    }


def test_aggregation_takes_median_per_bucket_then_across_buckets():
    rows = []
    values = {"short": 10.0, "medium": 20.0, "long": 100.0}
    for context, value in values.items():
        for repetition in range(7):
            rows.append(
                _case_row(
                    "tinyllmforge_exact_k8",
                    context,
                    repetition,
                    tpot=value + repetition,
                    throughput=1_000 / (value + repetition),
                )
            )

    result = aggregate_case_rows(rows)

    assert result["tinyllmforge_exact_k8"]["contexts"]["short"][
        "median_tpot_ns"
    ] == 13.0
    assert result["tinyllmforge_exact_k8"]["aggregate"][
        "median_tpot_ns"
    ] == 23.0


def _comparison_fixture():
    return {
        "complete": True,
        "correctness_valid": True,
        "storage_valid": True,
        "terminal_receipts_valid": True,
        "verifiers_agree": True,
        "aggregate": {
            "median_tpot_ratio": 0.94,
            "throughput_ratio": 1.06,
            "ttft_ratio": 1.01,
            "e2e_ratio": 0.98,
            "p95_tpot_ratio": 0.99,
            "p99_tpot_ratio": 1.00,
            "peak_gpu_memory_ratio": 1.02,
            "peak_rss_ratio": 1.01,
        },
        "contexts": {
            "short": {"median_tpot_ratio": 0.99},
            "medium": {"median_tpot_ratio": 0.95},
            "long": {"median_tpot_ratio": 0.92},
        },
    }


def test_gate_requires_both_five_percent_gains_and_protected_metrics():
    comparison = _comparison_fixture()
    assert classify_comparison(comparison)["classification"] == (
        "GO_CROSS_ENGINE_ADVANTAGE"
    )

    slower = copy.deepcopy(comparison)
    slower["aggregate"]["throughput_ratio"] = 1.049
    assert classify_comparison(slower)["classification"] == (
        "NO_CROSS_ENGINE_ADVANTAGE"
    )

    regressed = copy.deepcopy(comparison)
    regressed["aggregate"]["peak_rss_ratio"] = 1.021
    assert classify_comparison(regressed)["classification"] == (
        "NO_CROSS_ENGINE_ADVANTAGE"
    )


def test_gate_classifies_parity_at_inclusive_five_percent_boundary():
    comparison = _comparison_fixture()
    comparison["aggregate"]["median_tpot_ratio"] = 1.05
    comparison["aggregate"]["throughput_ratio"] = 0.95
    comparison["contexts"]["long"]["median_tpot_ratio"] = 1.01

    assert classify_comparison(comparison)["classification"] == (
        "CROSS_ENGINE_PARITY"
    )


@pytest.mark.parametrize(
    "field",
    (
        "complete",
        "correctness_valid",
        "storage_valid",
        "terminal_receipts_valid",
        "verifiers_agree",
    ),
)
def test_gate_is_incomplete_before_performance_if_evidence_missing(field):
    comparison = _comparison_fixture()
    comparison[field] = False

    result = classify_comparison(comparison)

    assert result["classification"] == "INCOMPLETE"
    assert field in result["reasons"]
