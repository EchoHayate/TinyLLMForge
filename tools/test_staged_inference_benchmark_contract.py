"""Dependency-light tests for the staged inference benchmark contract.

Run:
    python3 tools/test_staged_inference_benchmark_contract.py
"""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "tools" / "staged_inference_benchmark_contract.py"


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "staged_inference_benchmark_contract",
        CONTRACT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load staged benchmark contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _prefix_shape(
    *,
    prefix_tokens: int,
    batch_size: int,
    cold_ms: float = 100.0,
    warm_ms: float = 75.0,
    cache_cleared_ms: float = 101.0,
) -> dict:
    expected_cached = prefix_tokens * batch_size
    cold_query = (prefix_tokens + 64) * batch_size
    warm_query = 64 * batch_size
    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": 64,
        "batch_size": batch_size,
        "expected_reusable_tokens": expected_cached,
        "cold": {
            "median_elapsed_ms": cold_ms,
            "p95_elapsed_ms": cold_ms * 1.1,
            "median_cached_prompt_tokens": 0,
            "median_executed_query_tokens": cold_query,
            "median_model_batches": 1 if batch_size == 1 else 2,
            "peak_cuda_reserved_bytes": 1_000,
            "exact_outputs": True,
            "logit_argmax_match": True,
            "logit_max_abs": 0.0,
            "logit_mean_abs": 0.0,
            "samples": 7,
        },
        "warm": {
            "median_elapsed_ms": warm_ms,
            "p95_elapsed_ms": warm_ms * 1.1,
            "median_cached_prompt_tokens": expected_cached,
            "median_executed_query_tokens": warm_query,
            "median_model_batches": 1,
            "peak_cuda_reserved_bytes": 1_040,
            "exact_outputs": True,
            "logit_argmax_match": True,
            "logit_max_abs": 0.0,
            "logit_mean_abs": 0.0,
            "samples": 7,
        },
        "cache_cleared": {
            "median_elapsed_ms": cache_cleared_ms,
            "p95_elapsed_ms": cache_cleared_ms * 1.1,
            "median_cached_prompt_tokens": 0,
            "median_executed_query_tokens": cold_query,
            "median_model_batches": 1 if batch_size == 1 else 2,
            "peak_cuda_reserved_bytes": 1_000,
            "exact_outputs": True,
            "logit_argmax_match": True,
            "logit_max_abs": 0.0,
            "logit_mean_abs": 0.0,
            "samples": 7,
        },
        "retained_reusable_blocks": expected_cached // 256,
        "retained_logical_kv_bytes": expected_cached * 32,
        "median_cache_clear_host_ms": 0.2,
    }


def _complete_prefix_bundle() -> dict:
    return {
        "artifact_complete": True,
        "single": {
            "256": _prefix_shape(prefix_tokens=256, batch_size=1),
            "1024": _prefix_shape(prefix_tokens=1024, batch_size=1),
            "2048": _prefix_shape(prefix_tokens=2048, batch_size=1),
        },
        "batch": {
            "1024": _prefix_shape(prefix_tokens=1024, batch_size=8),
            "2048": _prefix_shape(prefix_tokens=2048, batch_size=8),
        },
    }


def _chunked_repetition(
    repetition: int,
    *,
    favorable: bool = True,
) -> dict:
    ttft_ratio = 0.85 if favorable else 0.95
    return {
        "repetition": repetition,
        "OFF": {
            "short_p99_ttft_ns": 100,
            "short_p99_itl_ns": 100,
            "maximum_decode_gap_ns": 100,
            "service_class_p95_completion_ns": {
                "short__short": 100,
                "short__long": 100,
                "medium__short": 100,
                "medium__long": 100,
                "long__short": 100,
                "long__long": 100,
            },
            "long_p95_completion_ns": 100,
            "request_throughput_rps": 100,
            "output_token_throughput_tps": 100,
            "peak_cuda_reserved_bytes": 1_000,
            "exact_outputs": True,
            "complete_lifecycle": True,
            "dropped_requests": 0,
            "rejected_requests": 0,
            "truncated_requests": 0,
            "unfinished_requests": 0,
            "starved_requests": 0,
        },
        "FAIR_CHUNKED": {
            "short_p99_ttft_ns": 100 * ttft_ratio,
            "short_p99_itl_ns": 103,
            "maximum_decode_gap_ns": 108,
            "service_class_p95_completion_ns": {
                "short__short": 104,
                "short__long": 105,
                "medium__short": 106,
                "medium__long": 107,
                "long__short": 108,
                "long__long": 109,
            },
            "long_p95_completion_ns": 109,
            "request_throughput_rps": 98,
            "output_token_throughput_tps": 98,
            "peak_cuda_reserved_bytes": 1_040,
            "exact_outputs": True,
            "complete_lifecycle": True,
            "dropped_requests": 0,
            "rejected_requests": 0,
            "truncated_requests": 0,
            "unfinished_requests": 0,
            "starved_requests": 0,
        },
    }


def _complete_chunked_bundle(*, favorable_repetitions: int = 5) -> dict:
    return {
        "artifact_complete": True,
        "repetitions": [
            _chunked_repetition(
                repetition,
                favorable=repetition < favorable_repetitions,
            )
            for repetition in range(5)
        ],
    }


def test_canonical_json_sha256_is_stable_and_order_independent():
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    assert contract.canonical_json_sha256(left) == (
        contract.canonical_json_sha256(right)
    )
    assert len(contract.canonical_json_sha256(left)) == 64


def test_prefix_case_matrix_has_exact_frozen_shapes():
    rows = contract.build_prefix_case_matrix(model_tier="qwen3-0.6b")
    assert len(rows) == 15
    assert {
        (row["shape"], row["state"])
        for row in rows
    } == {
        *{
            (f"single-{prefix_tokens}", state)
            for prefix_tokens in (256, 1024, 2048)
            for state in ("cold", "warm", "cache_cleared")
        },
        *{
            (f"batch8-{prefix_tokens}", state)
            for prefix_tokens in (1024, 2048)
            for state in ("cold", "warm", "cache_cleared")
        },
    }
    assert all(row["suffix_tokens"] == 64 for row in rows)
    assert all(row["warmup_repetitions"] == 2 for row in rows)
    assert all(row["measured_repetitions"] == 7 for row in rows)
    assert all(row["enforce_eager"] is True for row in rows)


def test_chunked_workload_has_exact_frozen_shape():
    rows = contract.build_chunked_workload()
    warmup = [row for row in rows if row["warmup"]]
    measured = [row for row in rows if not row["warmup"]]
    assert len(warmup) == 8
    assert len(measured) == 96
    assert Counter(row["prompt_tokens"] for row in measured) == {
        64: 58,
        512: 24,
        4096: 14,
    }
    for prompt_tokens in (64, 512, 4096):
        outputs = Counter(
            row["requested_output_tokens"]
            for row in measured
            if row["prompt_tokens"] == prompt_tokens
        )
        assert set(outputs) == {16, 64}
        assert max(outputs.values()) - min(outputs.values()) <= 1
    assert [row["arrival_offset_ns"] for row in rows] == sorted(
        row["arrival_offset_ns"] for row in rows
    )
    assert len({row["request_id"] for row in rows}) == 104
    assert all(
        row["prompt_tokens"] + row["requested_output_tokens"] <= 4352
        for row in rows
    )
    assert rows == contract.build_chunked_workload()


def test_chunked_case_matrix_is_paired_and_alternates_order():
    rows = contract.build_chunked_case_matrix(model_tier="qwen3-0.6b")
    assert len(rows) == 10
    assert Counter(row["policy"] for row in rows) == {
        "OFF": 5,
        "FAIR_CHUNKED": 5,
    }
    assert {row["repetition"] for row in rows} == set(range(5))
    for repetition in range(5):
        pair = [
            row["policy"]
            for row in rows
            if row["repetition"] == repetition
        ]
        expected = (
            ["OFF", "FAIR_CHUNKED"]
            if repetition % 2 == 0
            else ["FAIR_CHUNKED", "OFF"]
        )
        assert pair == expected
    off = next(row for row in rows if row["policy"] == "OFF")
    fair = next(row for row in rows if row["policy"] == "FAIR_CHUNKED")
    assert off["engine_config"]["max_num_prefill_tokens_per_step"] == 0
    assert fair["engine_config"] == {
        "max_model_len": 4352,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 512,
        "max_num_prefill_tokens_per_step": 128,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    }


def test_prefix_classification_separates_go_no_go_and_incorrect():
    bundle = _complete_prefix_bundle()
    result = contract.classify_prefix_bundle(bundle)
    assert result["classification"] == "PREFIX_CACHE_GO"
    assert result["performance_failures"] == []
    assert result["benefit"]["minimum_primary_improvement_fraction"] == 0.25
    assert result["cost"]["maximum_cuda_reserved_regression_fraction"] == 0.04

    bundle = _complete_prefix_bundle()
    bundle["single"]["1024"]["warm"]["median_elapsed_ms"] = 81.0
    result = contract.classify_prefix_bundle(bundle)
    assert result["classification"] == "PREFIX_CACHE_NO_GO"
    assert any(
        "1024" in failure and "20%" in failure
        for failure in result["performance_failures"]
    )

    bundle = _complete_prefix_bundle()
    bundle["single"]["1024"]["warm"]["exact_outputs"] = False
    result = contract.classify_prefix_bundle(bundle)
    assert result["classification"] == (
        "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    )
    assert result["correctness_failures"]

    bundle = _complete_prefix_bundle()
    bundle["artifact_complete"] = False
    result = contract.classify_prefix_bundle(bundle)
    assert result["classification"] == (
        "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    )
    assert result["structural_failures"]


def test_prefix_classification_preserves_targeted_correctness_failures():
    bundle = _complete_prefix_bundle()
    bundle["correctness_failures"] = [
        "repeat_513: targeted correctness check failed"
    ]

    result = contract.classify_prefix_bundle(bundle)

    assert result["classification"] == (
        "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    )
    assert result["structural_failures"] == []
    assert result["correctness_failures"] == [
        "repeat_513: targeted correctness check failed"
    ]


def test_prefix_classification_enforces_accounting_and_protected_costs():
    bundle = _complete_prefix_bundle()
    bundle["batch"]["1024"]["warm"]["median_model_batches"] = 2
    assert contract.classify_prefix_bundle(bundle)["classification"] == (
        "PREFIX_CACHE_NO_GO"
    )

    bundle = _complete_prefix_bundle()
    bundle["single"]["2048"]["cache_cleared"]["median_elapsed_ms"] = 106.0
    assert contract.classify_prefix_bundle(bundle)["classification"] == (
        "PREFIX_CACHE_NO_GO"
    )

    bundle = _complete_prefix_bundle()
    bundle["single"]["2048"]["warm"]["peak_cuda_reserved_bytes"] = 1_051
    assert contract.classify_prefix_bundle(bundle)["classification"] == (
        "PREFIX_CACHE_NO_GO"
    )

    bundle = _complete_prefix_bundle()
    bundle["single"]["1024"]["warm"]["median_cached_prompt_tokens"] = 768
    assert contract.classify_prefix_bundle(bundle)["classification"] == (
        "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    )


def test_chunked_classification_requires_four_favorable_repetitions():
    result = contract.classify_chunked_bundle(
        _complete_chunked_bundle(favorable_repetitions=4)
    )
    assert result["classification"] == "FAIR_CHUNKED_GO"
    assert result["benefit"]["favorable_repetitions"] == 4

    result = contract.classify_chunked_bundle(
        _complete_chunked_bundle(favorable_repetitions=3)
    )
    assert result["classification"] == "FAIR_CHUNKED_NO_GO"
    assert any(
        "four of five" in failure
        for failure in result["performance_failures"]
    )


def test_chunked_classification_enforces_correctness_and_guards():
    bundle = _complete_chunked_bundle()
    bundle["repetitions"][0]["FAIR_CHUNKED"]["starved_requests"] = 1
    assert contract.classify_chunked_bundle(bundle)["classification"] == (
        "FAIR_CHUNKED_INCOMPLETE"
    )

    bundle = _complete_chunked_bundle()
    bundle["repetitions"][0]["FAIR_CHUNKED"][
        "output_token_throughput_tps"
    ] = 96
    assert contract.classify_chunked_bundle(bundle)["classification"] == (
        "FAIR_CHUNKED_NO_GO"
    )

    bundle = _complete_chunked_bundle()
    bundle["repetitions"][0]["FAIR_CHUNKED"][
        "service_class_p95_completion_ns"
    ]["long__long"] = 111
    assert contract.classify_chunked_bundle(bundle)["classification"] == (
        "FAIR_CHUNKED_NO_GO"
    )

    bundle = _complete_chunked_bundle()
    bundle["artifact_complete"] = False
    assert contract.classify_chunked_bundle(bundle)["classification"] == (
        "FAIR_CHUNKED_INCOMPLETE"
    )


def test_stage2_winner_uses_frozen_order_and_rejects_no_go():
    prefix = contract.classify_prefix_bundle(_complete_prefix_bundle())
    chunked = contract.classify_chunked_bundle(_complete_chunked_bundle())
    winner = contract.select_stage2_winner(prefix, chunked)
    assert winner["winner"] == "prefix"
    assert winner["reason"] == "larger normalized primary benefit"

    tied_prefix = {
        **prefix,
        "benefit": {"minimum_primary_improvement_fraction": 0.15},
        "cost": {
            "worst_protected_metric_regression_fraction": 0.05,
            "maximum_cuda_reserved_regression_fraction": 0.04,
        },
    }
    tied_chunked = {
        **chunked,
        "benefit": {
            **chunked["benefit"],
            "short_p99_ttft_improvement_fraction": 0.15,
        },
        "cost": {
            "worst_protected_metric_regression_fraction": 0.05,
            "maximum_cuda_reserved_regression_fraction": 0.04,
        },
    }
    winner = contract.select_stage2_winner(tied_prefix, tied_chunked)
    assert winner == {
        "winner": "prefix",
        "reason": "exact tie favors lower-occupancy prefix gate",
    }

    prefix_no_go = {
        **prefix,
        "classification": "PREFIX_CACHE_NO_GO",
    }
    chunked_no_go = {
        **chunked,
        "classification": "FAIR_CHUNKED_NO_GO",
    }
    assert contract.select_stage2_winner(
        prefix_no_go,
        chunked_no_go,
    ) == {
        "winner": None,
        "reason": "no Stage-1 gate is eligible",
    }


def main():
    test_canonical_json_sha256_is_stable_and_order_independent()
    test_prefix_case_matrix_has_exact_frozen_shapes()
    test_chunked_workload_has_exact_frozen_shape()
    test_chunked_case_matrix_is_paired_and_alternates_order()
    test_prefix_classification_separates_go_no_go_and_incorrect()
    test_prefix_classification_preserves_targeted_correctness_failures()
    test_prefix_classification_enforces_accounting_and_protected_costs()
    test_chunked_classification_requires_four_favorable_repetitions()
    test_chunked_classification_enforces_correctness_and_guards()
    test_stage2_winner_uses_frozen_order_and_rejects_no_go()
    print("staged inference benchmark contract tests passed")


if __name__ == "__main__":
    main()
