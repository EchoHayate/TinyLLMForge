from __future__ import annotations

from copy import deepcopy
import math

import pytest

from tools.slo_cohort_burst_gate import (
    GO_SLO_AWARE_COHORT_DECODE_BURST,
    INVALID_SOURCE_OR_EVIDENCE,
    classify_slo_cohort_burst,
    nearest_rank,
    summarize_request_rows,
)


def complete_summary(**overrides):
    summary = {
        "evidence_complete": True,
        "source_exact": True,
        "verifier_agreement": True,
        "correctness_passed": True,
        "lifecycle_closed": True,
        "aggregate_throughput_improvement": 0.10,
        "medium_throughput_improvement": 0.10,
        "high_throughput_improvement": 0.10,
        "worst_throughput_regression": 0.02,
        "worst_p99_itl_regression": 0.03,
        "worst_p99_ttft_regression": 0.05,
        "worst_p99_e2e_regression": 0.05,
        "maximum_host_visible_gap_ns": 40_000_000,
        "starved_requests": 0,
        "post_eos_wasted_forward_fraction": 0.10,
        "peak_reserved_memory_regression": 0.05,
    }
    summary.update(overrides)
    return summary


def test_formal_gate_passes_exact_boundaries():
    assert classify_slo_cohort_burst(complete_summary()) == (
        GO_SLO_AWARE_COHORT_DECODE_BURST
    )


def test_tail_failure_precedes_throughput_failure():
    summary = complete_summary(
        worst_p99_itl_regression=0.031,
        aggregate_throughput_improvement=0.01,
    )
    assert classify_slo_cohort_burst(summary) == (
        "NO_GO_TAIL_LATENCY"
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ({"correctness_passed": False}, "NO_GO_CORRECTNESS"),
        ({"lifecycle_closed": False}, "NO_GO_LIFECYCLE"),
        ({"starved_requests": 1}, "NO_GO_STARVATION"),
        (
            {"worst_p99_ttft_regression": 0.051},
            "NO_GO_TAIL_LATENCY",
        ),
        (
            {"peak_reserved_memory_regression": 0.051},
            "NO_GO_MEMORY",
        ),
        (
            {"post_eos_wasted_forward_fraction": 0.101},
            "NO_GO_EOS_WASTE",
        ),
        (
            {"high_throughput_improvement": 0.099},
            "NO_GO_THROUGHPUT",
        ),
    ),
)
def test_formal_gate_uses_frozen_failure_precedence(
    mutation,
    expected,
):
    assert classify_slo_cohort_burst(
        complete_summary(**mutation)
    ) == expected


@pytest.mark.parametrize(
    "mutation",
    (
        {"evidence_complete": False},
        {"source_exact": False},
        {"verifier_agreement": False},
        {"aggregate_throughput_improvement": float("nan")},
        {"starved_requests": True},
    ),
)
def test_invalid_source_or_metric_evidence_fails_closed(mutation):
    assert classify_slo_cohort_burst(
        complete_summary(**mutation)
    ) == INVALID_SOURCE_OR_EVIDENCE


def test_nearest_rank_and_raw_request_timestamps_are_authoritative():
    assert nearest_rank((10, 20, 30, 40, 50), 0.99) == 50
    rows = (
        {
            "request_id": "r0",
            "sequence_id": 7,
            "service_class": "default",
            "arrival_ns": 0,
            "prefill_start_ns": 1,
            "prefill_complete_ns": 9,
            "first_token_visible_ns": 10,
            "token_visible_ns": (10, 10, 30),
            "completion_ns": 30,
            "output_token_ids": (11, 12, 13),
            "output_text_sha256": "a" * 64,
            "terminal_reason": "eos",
        },
        {
            "request_id": "r1",
            "sequence_id": 9,
            "service_class": "default",
            "arrival_ns": 5,
            "prefill_start_ns": 6,
            "prefill_complete_ns": 19,
            "first_token_visible_ns": 20,
            "token_visible_ns": (20, 40),
            "completion_ns": 40,
            "output_token_ids": (21, 22),
            "output_text_sha256": "b" * 64,
            "terminal_reason": "length",
        },
    )

    summary = summarize_request_rows(rows)

    assert summary["committed_output_tokens"] == 5
    assert summary["p99_ttft_ns"] == 15
    assert summary["p99_itl_ns"] == 20
    assert summary["p99_e2e_ns"] == 35
    assert summary["maximum_host_visible_gap_ns"] == 20
    assert summary["output_throughput_tps"] == pytest.approx(
        125_000_000.0
    )


def test_request_summary_rejects_amortized_or_mutated_timeline():
    rows = [{
        "request_id": "r0",
        "sequence_id": 7,
        "service_class": "default",
        "arrival_ns": 0,
        "prefill_start_ns": 1,
        "prefill_complete_ns": 9,
        "first_token_visible_ns": 10,
        "token_visible_ns": (10, 10),
        "completion_ns": 10,
        "output_token_ids": (11, 12),
        "output_text_sha256": "a" * 64,
        "terminal_reason": "eos",
    }]
    mutated = deepcopy(rows)
    mutated[0]["token_visible_ns"] = (10,)
    with pytest.raises(ValueError, match="token timestamp count"):
        summarize_request_rows(mutated)

    for bad_value in (math.nan, math.inf, True):
        mutated = deepcopy(rows)
        mutated[0]["arrival_ns"] = bad_value
        with pytest.raises(ValueError):
            summarize_request_rows(mutated)


def test_request_summary_accepts_versioned_runtime_rows():
    row = {
        "schema_version": "slo-cohort-burst.request.v1",
        "request_id": "r0",
        "sequence_id": 7,
        "service_class": "default",
        "arrival_ns": 0,
        "prefill_start_ns": 1,
        "prefill_complete_ns": 9,
        "first_token_visible_ns": 10,
        "token_visible_ns": (10, 10),
        "completion_ns": 10,
        "output_token_ids": (11, 12),
        "output_text_sha256": "a" * 64,
        "terminal_reason": "eos",
    }

    summary = summarize_request_rows((row,))

    assert summary["committed_output_tokens"] == 2
    assert summary["p99_itl_ns"] == 0


def test_request_summary_rejects_mismatched_first_token_timestamp():
    row = {
        "schema_version": "slo-cohort-burst.request.v1",
        "request_id": "r0",
        "sequence_id": 7,
        "service_class": "default",
        "arrival_ns": 0,
        "prefill_start_ns": 1,
        "prefill_complete_ns": 9,
        "first_token_visible_ns": 11,
        "token_visible_ns": (10, 10),
        "completion_ns": 10,
        "output_token_ids": (11, 12),
        "output_text_sha256": "a" * 64,
        "terminal_reason": "eos",
    }

    with pytest.raises(ValueError, match="first token"):
        summarize_request_rows((row,))
