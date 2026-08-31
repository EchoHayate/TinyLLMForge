from __future__ import annotations

import pytest

from tools.cross_request_wavefront_overlap import (
    build_balanced_cohorts,
    classify_wavefront_microgate,
    cohort_digest,
    interval_overlap_ns,
    interval_union_ns,
)


def _passing_rows():
    return [
        {
            "active_tokens": active_tokens,
            "pair_index": pair_index,
            "rank": rank,
            "arm_order": (
                ["baseline", "candidate"]
                if pair_index % 2 == 0
                else ["candidate", "baseline"]
            ),
            "cohort_digest": "a" * 64,
            "collective_order_digest": "b" * 64,
            "baseline_cuda_ns": 100_000,
            "candidate_cuda_ns": 90_000,
            "baseline_host_submission_ns": 20_000,
            "candidate_host_submission_ns": 21_000,
            "candidate_communication_union_ns": 40_000,
            "candidate_realized_overlap_ns": 12_000,
            "cross_rank_max_abs_error": 0.0,
            "cross_rank_max_rel_error": 0.0,
            "baseline_max_abs_error": 0.0,
            "baseline_max_rel_error": 0.0,
            "nan_count": 0,
            "inf_count": 0,
            "timed_out": False,
        }
        for active_tokens in (4, 8)
        for pair_index in range(300)
        for rank in range(4)
    ]


def _passing_memory():
    return {"maximum_allocated_delta_bytes": 64 * 1024 * 1024}


def _passing_cleanup():
    return {"classification": "CLEAN"}


def _mutated_inputs(mutation):
    rows = _passing_rows()
    memory = _passing_memory()
    cleanup = _passing_cleanup()
    if mutation == "coverage":
        rows.pop()
    elif mutation == "rank_digest":
        rows[0]["cohort_digest"] = "c" * 64
    elif mutation == "correctness":
        rows[0]["baseline_max_abs_error"] = 0.021
        rows[0]["baseline_max_rel_error"] = 0.003
    elif mutation == "memory":
        memory["maximum_allocated_delta_bytes"] = 128 * 1024 * 1024 + 1
    elif mutation == "tail":
        for row in rows:
            if row["pair_index"] >= 296:
                row["candidate_cuda_ns"] = 110_000
    elif mutation == "overlap":
        for row in rows:
            if row["active_tokens"] == 4:
                row["candidate_realized_overlap_ns"] = 4_000
    elif mutation == "fragmentation":
        for row in rows:
            row["candidate_cuda_ns"] = 98_000
    elif mutation == "cleanup":
        cleanup["classification"] = "DIRTY"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    return rows, memory, cleanup


def test_balanced_cohorts_are_contiguous_complete_and_stable():
    assert build_balanced_cohorts(4) == (
        {
            "cohort_id": 0,
            "request_indices": (0, 1),
            "active_token_count": 2,
        },
        {
            "cohort_id": 1,
            "request_indices": (2, 3),
            "active_token_count": 2,
        },
    )
    assert build_balanced_cohorts(8)[0]["request_indices"] == (0, 1, 2, 3)
    assert build_balanced_cohorts(8)[1]["request_indices"] == (4, 5, 6, 7)
    assert cohort_digest(build_balanced_cohorts(8)) == cohort_digest(
        build_balanced_cohorts(8)
    )


@pytest.mark.parametrize("count", (0, 1, 2, 3, 5, True))
def test_balanced_cohorts_reject_unsupported_counts(count):
    with pytest.raises(ValueError, match="active request count"):
        build_balanced_cohorts(count)


def test_interval_math_uses_unions_before_overlap():
    communication = ((10, 30), (25, 40))
    computation = ((0, 15), (20, 35))

    assert interval_union_ns(communication) == 30
    assert interval_overlap_ns(communication, computation) == 20


@pytest.mark.parametrize(
    "intervals",
    (
        (),
        ((True, 10),),
        ((0, False),),
        ((-1, 10),),
        ((10, 9),),
        ((0, float("inf")),),
        ((0, float("nan")),),
    ),
)
def test_interval_math_rejects_invalid_intervals(intervals):
    with pytest.raises(ValueError, match="interval"):
        interval_union_ns(intervals)


def test_classifier_accepts_complete_profitable_gate():
    result = classify_wavefront_microgate(
        _passing_rows(),
        memory=_passing_memory(),
        cleanup=_passing_cleanup(),
    )

    assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
    assert result["runtime_integration_authorized"] is True
    assert [row["active_tokens"] for row in result["shape_summaries"]] == [
        4,
        8,
    ]
    assert all(
        row["pair_count"] == 300 for row in result["shape_summaries"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("coverage", "INCONCLUSIVE_EVIDENCE"),
        ("rank_digest", "INCONCLUSIVE_EVIDENCE"),
        ("correctness", "NO_GO_CORRECTNESS"),
        ("memory", "NO_GO_MEMORY"),
        ("tail", "NO_GO_PERFORMANCE"),
        ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
        ("fragmentation", "NO_GO_GEMM_FRAGMENTATION"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_classifier_fails_closed(mutation, expected):
    rows, memory, cleanup = _mutated_inputs(mutation)
    assert classify_wavefront_microgate(
        rows,
        memory,
        cleanup,
    )["classification"] == expected


def test_classifier_uses_max_rank_duration_and_reports_costs():
    rows = _passing_rows()
    for row in rows:
        if row["active_tokens"] == 4 and row["rank"] == 3:
            row["candidate_cuda_ns"] = 92_000
            row["candidate_host_submission_ns"] = 22_000

    result = classify_wavefront_microgate(
        rows,
        _passing_memory(),
        _passing_cleanup(),
    )
    tokens4 = result["shape_summaries"][0]

    assert tokens4["candidate_median_cuda_ns"] == 92_000
    assert tokens4["median_speedup_ratio"] == pytest.approx(0.08)
    assert tokens4["host_submission_regression_ratio"] == pytest.approx(0.10)
    assert tokens4["realized_overlap_ratio"] == pytest.approx(0.30)


def test_classifier_rejects_nonfinite_and_invalid_arm_order():
    rows = _passing_rows()
    rows[0]["candidate_cuda_ns"] = float("nan")
    rows[1]["arm_order"] = ["baseline", "baseline"]

    result = classify_wavefront_microgate(
        rows,
        _passing_memory(),
        _passing_cleanup(),
    )

    assert result["classification"] == "INCONCLUSIVE_EVIDENCE"


def test_classifier_precedence_prefers_correctness_then_memory():
    rows, memory, cleanup = _mutated_inputs("correctness")
    memory["maximum_allocated_delta_bytes"] = 128 * 1024 * 1024 + 1
    cleanup["classification"] = "DIRTY"

    assert classify_wavefront_microgate(
        rows,
        memory,
        cleanup,
    )["classification"] == "NO_GO_CORRECTNESS"

    rows = _passing_rows()
    assert classify_wavefront_microgate(
        rows,
        memory,
        cleanup,
    )["classification"] == "NO_GO_MEMORY"


def test_classifier_accepts_absolute_tolerance_near_zero():
    rows = _passing_rows()
    for row in rows:
        if row["active_tokens"] == 8:
            row["baseline_max_abs_error"] = 0.00390625
            row["baseline_max_rel_error"] = 0.03448275849223137

    result = classify_wavefront_microgate(
        rows,
        _passing_memory(),
        _passing_cleanup(),
    )

    assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
