#!/usr/bin/env python3
"""Tests for the persistent-decode optimistic ceiling classifier."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from tools.lease_sealed_persistent_decode_ceiling import (
    GO_PERSISTENT_DECODE_CEILING,
    INCOMPLETE_EVIDENCE,
    INCONCLUSIVE_CORRECTNESS,
    INCONCLUSIVE_PROFILE_OVERHEAD,
    INCONCLUSIVE_TRACE_COVERAGE,
    NO_GO_PERSISTENT_DECODE_CEILING,
    compute_ceiling,
)


SOURCE_COMMIT = "a" * 40
SOURCE_TREE_SHA256 = "b" * 64
RUNTIME_IDENTITY_SHA256 = "c" * 64
WORKLOAD_IDENTITY_SHA256 = "d" * 64
CONTEXTS = (256, 2048, 8192)


def _tokens(context):
    return [context % 97] * 128


def _timing_rows(*, median_tpot_ns=2_000_000, p95_tpot_ns=2_200_000):
    rows = []
    for repetition in range(5):
        for context in CONTEXTS:
            rows.append({
                "schema_version":
                    "lease-sealed-persistent-decode.timing.v1",
                "arm": "uninstrumented",
                "source_commit": SOURCE_COMMIT,
                "source_tree_sha256": SOURCE_TREE_SHA256,
                "runtime_identity_sha256": RUNTIME_IDENTITY_SHA256,
                "workload_identity_sha256": WORKLOAD_IDENTITY_SHA256,
                "context_length": context,
                "repetition": repetition,
                "generated_tokens": 128,
                "tpot_median_ns": median_tpot_ns,
                "tpot_p95_ns": p95_tpot_ns,
                "output_token_ids": _tokens(context),
                "output_text_sha256": f"{context:064x}",
                "target_model_forwards": 127,
                "committed_tokens": 127,
                "fallback_count": 0,
                "failure_count": 0,
                "rollback_count": 0,
                "quarantine_reason": None,
            })
    return rows


def _trace_summary(
    *,
    eligible_zero_cost_ns=140_000,
    candidate_cuda_duration_ns=100_000,
    total_kernel_duration_ns=2_000_000,
    classified_launch_ratio=0.99,
    classified_duration_ratio=0.995,
    profiled_median_tpot_ns=2_100_000,
    profiled_p95_tpot_ns=2_300_000,
):
    return {
        "schema_version":
            "lease-sealed-persistent-decode.trace-summary.v1",
        "source_commit": SOURCE_COMMIT,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "runtime_identity_sha256": RUNTIME_IDENTITY_SHA256,
        "workload_identity_sha256": WORKLOAD_IDENTITY_SHA256,
        "contexts": [
            {
                "context_length": context,
                "profiled_tpot_median_ns":
                    profiled_median_tpot_ns,
                "profiled_tpot_p95_ns":
                    profiled_p95_tpot_ns,
                "output_token_ids": _tokens(context),
                "output_text_sha256": f"{context:064x}",
                "target_model_forwards": 127,
                "committed_tokens": 127,
                "fallback_count": 0,
                "failure_count": 0,
                "rollback_count": 0,
                "quarantine_reason": None,
                "transaction_count": 16,
                "logical_token_count": 127,
                "eligible_zero_cost_ns_per_token":
                    eligible_zero_cost_ns,
                "candidate_cuda_duration_ns":
                    candidate_cuda_duration_ns,
                "total_kernel_duration_ns":
                    total_kernel_duration_ns,
                "classified_launch_ratio":
                    classified_launch_ratio,
                "classified_duration_ratio":
                    classified_duration_ratio,
                "segment_signatures": ["e" * 64],
            }
            for context in CONTEXTS
        ],
    }


def _classification(**trace_kwargs):
    return compute_ceiling(
        _timing_rows(),
        _trace_summary(**trace_kwargs),
    )["classification"]


def test_complete_headroom_returns_go():
    result = compute_ceiling(
        _timing_rows(),
        _trace_summary(),
    )

    assert result["classification"] == GO_PERSISTENT_DECODE_CEILING
    assert result["aggregate_optimistic_improvement_pct"] == pytest.approx(
        7.0
    )
    assert result["aggregate_candidate_cuda_duration_share_pct"] == (
        pytest.approx(5.0)
    )
    assert result["stable_cross_context_signatures"] == ["e" * 64]


def test_aggregate_optimistic_headroom_below_five_percent_is_no_go():
    assert _classification(
        eligible_zero_cost_ns=90_000,
    ) == NO_GO_PERSISTENT_DECODE_CEILING


def test_one_context_below_three_percent_is_no_go():
    trace = _trace_summary()
    trace["contexts"][0]["eligible_zero_cost_ns_per_token"] = 50_000

    result = compute_ceiling(_timing_rows(), trace)

    assert result["classification"] == (
        NO_GO_PERSISTENT_DECODE_CEILING
    )
    assert "minimum_context_optimistic_improvement_pct" in (
        result["failed_conditions"]
    )


def test_candidate_cuda_share_below_four_percent_is_no_go():
    assert _classification(
        candidate_cuda_duration_ns=70_000,
    ) == NO_GO_PERSISTENT_DECODE_CEILING


def test_no_stable_cross_context_signature_is_no_go():
    trace = _trace_summary()
    for index, row in enumerate(trace["contexts"]):
        row["segment_signatures"] = [f"{index + 1:064x}"]

    assert compute_ceiling(
        _timing_rows(),
        trace,
    )["classification"] == NO_GO_PERSISTENT_DECODE_CEILING


def test_low_launch_coverage_is_inconclusive():
    assert _classification(
        classified_launch_ratio=0.979,
    ) == INCONCLUSIVE_TRACE_COVERAGE


def test_low_duration_coverage_is_inconclusive():
    assert _classification(
        classified_duration_ratio=0.989,
    ) == INCONCLUSIVE_TRACE_COVERAGE


def test_excessive_median_profiler_perturbation_is_inconclusive():
    assert _classification(
        profiled_median_tpot_ns=2_210_000,
    ) == INCONCLUSIVE_PROFILE_OVERHEAD


def test_excessive_p95_profiler_perturbation_is_inconclusive():
    assert _classification(
        profiled_p95_tpot_ns=2_540_000,
    ) == INCONCLUSIVE_PROFILE_OVERHEAD


@pytest.mark.parametrize(
    "mutation",
    ["token", "text"],
)
def test_output_mismatch_is_inconclusive_correctness(mutation):
    trace = _trace_summary()
    if mutation == "token":
        trace["contexts"][1]["output_token_ids"][-1] += 1
    else:
        trace["contexts"][1]["output_text_sha256"] = "f" * 64

    assert compute_ceiling(
        _timing_rows(),
        trace,
    )["classification"] == INCONCLUSIVE_CORRECTNESS


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "source_tree_sha256",
        "runtime_identity_sha256",
        "workload_identity_sha256",
    ],
)
def test_identity_mismatch_is_incomplete(field):
    trace = _trace_summary()
    trace[field] = (
        "f" * 40 if field == "source_commit" else "f" * 64
    )

    assert compute_ceiling(
        _timing_rows(),
        trace,
    )["classification"] == INCOMPLETE_EVIDENCE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_count", 1),
        ("failure_count", 1),
        ("rollback_count", 1),
        ("quarantine_reason", "fault"),
    ],
)
def test_runtime_failure_evidence_is_incomplete(field, value):
    trace = _trace_summary()
    trace["contexts"][0][field] = value

    assert compute_ceiling(
        _timing_rows(),
        trace,
    )["classification"] == INCOMPLETE_EVIDENCE


def test_missing_timing_row_is_incomplete():
    assert compute_ceiling(
        _timing_rows()[:-1],
        _trace_summary(),
    )["classification"] == INCOMPLETE_EVIDENCE


def test_missing_structural_context_is_incomplete():
    trace = _trace_summary()
    trace["contexts"].pop()

    assert compute_ceiling(
        _timing_rows(),
        trace,
    )["classification"] == INCOMPLETE_EVIDENCE


def test_non_finite_metric_is_rejected():
    trace = _trace_summary()
    trace["contexts"][0][
        "eligible_zero_cost_ns_per_token"
    ] = math.inf

    with pytest.raises(ValueError, match="finite"):
        compute_ceiling(_timing_rows(), trace)


def test_classification_precedence_checks_correctness_before_coverage():
    trace = _trace_summary(classified_launch_ratio=0.5)
    trace["contexts"][0]["output_token_ids"][-1] += 1

    assert compute_ceiling(
        _timing_rows(),
        trace,
    )["classification"] == INCONCLUSIVE_CORRECTNESS


def test_duplicate_timing_identity_is_rejected_as_incomplete():
    rows = _timing_rows()
    rows[-1] = deepcopy(rows[0])

    assert compute_ceiling(
        rows,
        _trace_summary(),
    )["classification"] == INCOMPLETE_EVIDENCE
