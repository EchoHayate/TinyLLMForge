from __future__ import annotations

import hashlib
import json
import math
from statistics import median


WORLD_SIZE = 4
ACTIVE_TOKEN_GROUPS = (4, 8)
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 300
MAX_ALLOCATED_DELTA_BYTES = 128 * 1024 * 1024
MIN_MEDIAN_SPEEDUP = {4: 0.05, 8: 0.08}
STOP_MEDIAN_SPEEDUP = 0.03
MIN_OVERLAP_RATIO = 0.20
MAX_P99_REGRESSION = 0.03
MAX_HOST_SUBMISSION_REGRESSION = 0.10
CROSS_RANK_ATOL = 2e-4
CROSS_RANK_RTOL = 2e-4
BASELINE_ATOL = 2e-2
BASELINE_RTOL = 2e-3


def build_balanced_cohorts(active_request_count):
    if (
        type(active_request_count) is not int
        or active_request_count not in ACTIVE_TOKEN_GROUPS
    ):
        raise ValueError("active request count must be 4 or 8")
    split = (active_request_count + 1) // 2
    return (
        {
            "cohort_id": 0,
            "request_indices": tuple(range(split)),
            "active_token_count": split,
        },
        {
            "cohort_id": 1,
            "request_indices": tuple(range(split, active_request_count)),
            "active_token_count": active_request_count - split,
        },
    )


def cohort_digest(cohorts):
    encoded = json.dumps(
        list(cohorts),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_intervals(intervals):
    normalized = []
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError("interval must contain two endpoints")
        start, end = interval
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
        ):
            raise ValueError("interval endpoints are invalid")
        normalized.append((start, end))
    if not normalized:
        raise ValueError("interval set must not be empty")

    merged = []
    for start, end in sorted(normalized):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def interval_union_ns(intervals):
    return sum(end - start for start, end in _normalize_intervals(intervals))


def interval_overlap_ns(left, right):
    left_intervals = _normalize_intervals(left)
    right_intervals = _normalize_intervals(right)
    left_index = 0
    right_index = 0
    overlap = 0
    while (
        left_index < len(left_intervals)
        and right_index < len(right_intervals)
    ):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        overlap += max(
            0,
            min(left_end, right_end) - max(left_start, right_start),
        )
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _nearest_rank_percentile(values, percentile):
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _classification(classification, shape_summaries=()):
    return {
        "classification": classification,
        "runtime_integration_authorized": (
            classification == "GO_WAVEFRONT_MICROGATE"
        ),
        "shape_summaries": list(shape_summaries),
    }


def classify_wavefront_microgate(rows, memory, cleanup):
    correctness_failed = False
    evidence_incomplete = False
    grouped = {
        active_tokens: {} for active_tokens in ACTIVE_TOKEN_GROUPS
    }
    cohort_digests = {
        active_tokens: set() for active_tokens in ACTIVE_TOKEN_GROUPS
    }
    collective_order_digests = set()
    seen = set()
    metric_names = (
        "baseline_cuda_ns",
        "candidate_cuda_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "candidate_communication_union_ns",
        "candidate_realized_overlap_ns",
        "cross_rank_max_abs_error",
        "cross_rank_max_rel_error",
        "baseline_max_abs_error",
        "baseline_max_rel_error",
    )

    if not isinstance(rows, (list, tuple)) or not rows:
        evidence_incomplete = True
        rows = ()

    for row in rows:
        if not isinstance(row, dict):
            evidence_incomplete = True
            continue
        active_tokens = row.get("active_tokens")
        pair_index = row.get("pair_index")
        rank = row.get("rank")
        identity = (active_tokens, pair_index, rank)
        expected_arm_order = (
            ["baseline", "candidate"]
            if type(pair_index) is int and pair_index % 2 == 0
            else ["candidate", "baseline"]
        )
        if (
            active_tokens not in grouped
            or type(pair_index) is not int
            or pair_index not in range(MEASURED_PAIR_COUNT)
            or type(rank) is not int
            or rank not in range(WORLD_SIZE)
            or identity in seen
        ):
            evidence_incomplete = True
            continue
        seen.add(identity)
        if row.get("arm_order") != expected_arm_order:
            evidence_incomplete = True
        if any(
            not _finite_nonnegative(row.get(name))
            for name in metric_names
        ):
            evidence_incomplete = True
            continue
        if (
            row["baseline_cuda_ns"] <= 0
            or row["baseline_host_submission_ns"] <= 0
            or row["candidate_communication_union_ns"] <= 0
            or row["candidate_realized_overlap_ns"]
            > row["candidate_communication_union_ns"]
            or row.get("timed_out") not in (True, False)
            or type(row.get("nan_count")) is not int
            or row["nan_count"] < 0
            or type(row.get("inf_count")) is not int
            or row["inf_count"] < 0
        ):
            evidence_incomplete = True
            continue

        cohort_hash = row.get("cohort_digest")
        collective_hash = row.get("collective_order_digest")
        if (
            not isinstance(cohort_hash, str)
            or len(cohort_hash) != 64
            or any(character not in "0123456789abcdef" for character in cohort_hash)
            or not isinstance(collective_hash, str)
            or len(collective_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in collective_hash
            )
        ):
            evidence_incomplete = True
            continue
        cohort_digests[active_tokens].add(cohort_hash)
        collective_order_digests.add(collective_hash)

        if (
            row["cross_rank_max_abs_error"] > CROSS_RANK_ATOL
            or row["cross_rank_max_rel_error"] > CROSS_RANK_RTOL
            or row["baseline_max_abs_error"] > BASELINE_ATOL
            or row["baseline_max_rel_error"] > BASELINE_RTOL
            or row["nan_count"] != 0
            or row["inf_count"] != 0
        ):
            correctness_failed = True
        if row["timed_out"]:
            evidence_incomplete = True
        grouped[active_tokens].setdefault(pair_index, {})[rank] = row

    if correctness_failed:
        return _classification("NO_GO_CORRECTNESS")

    allocated_delta = (
        memory.get("maximum_allocated_delta_bytes")
        if isinstance(memory, dict)
        else None
    )
    if not _finite_nonnegative(allocated_delta):
        evidence_incomplete = True
    elif allocated_delta > MAX_ALLOCATED_DELTA_BYTES:
        return _classification("NO_GO_MEMORY")

    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        evidence_incomplete = True
    if len(collective_order_digests) != 1:
        evidence_incomplete = True
    if any(len(digests) != 1 for digests in cohort_digests.values()):
        evidence_incomplete = True

    shape_summaries = []
    for active_tokens in ACTIVE_TOKEN_GROUPS:
        pairs = grouped[active_tokens]
        if set(pairs) != set(range(MEASURED_PAIR_COUNT)):
            evidence_incomplete = True
        complete_pairs = []
        for pair_index in range(MEASURED_PAIR_COUNT):
            rank_rows = pairs.get(pair_index, {})
            if set(rank_rows) != set(range(WORLD_SIZE)):
                evidence_incomplete = True
                continue
            complete_pairs.append(rank_rows)
        if len(complete_pairs) != MEASURED_PAIR_COUNT:
            continue

        baseline_cuda = [
            max(row["baseline_cuda_ns"] for row in rank_rows.values())
            for rank_rows in complete_pairs
        ]
        candidate_cuda = [
            max(row["candidate_cuda_ns"] for row in rank_rows.values())
            for rank_rows in complete_pairs
        ]
        baseline_host = [
            max(
                row["baseline_host_submission_ns"]
                for row in rank_rows.values()
            )
            for rank_rows in complete_pairs
        ]
        candidate_host = [
            max(
                row["candidate_host_submission_ns"]
                for row in rank_rows.values()
            )
            for rank_rows in complete_pairs
        ]
        communication_total = sum(
            row["candidate_communication_union_ns"]
            for rank_rows in complete_pairs
            for row in rank_rows.values()
        )
        overlap_total = sum(
            row["candidate_realized_overlap_ns"]
            for rank_rows in complete_pairs
            for row in rank_rows.values()
        )
        baseline_median = median(baseline_cuda)
        candidate_median = median(candidate_cuda)
        baseline_p99 = _nearest_rank_percentile(baseline_cuda, 0.99)
        candidate_p99 = _nearest_rank_percentile(candidate_cuda, 0.99)
        baseline_host_median = median(baseline_host)
        candidate_host_median = median(candidate_host)
        shape_summaries.append(
            {
                "active_tokens": active_tokens,
                "pair_count": len(complete_pairs),
                "baseline_median_cuda_ns": baseline_median,
                "candidate_median_cuda_ns": candidate_median,
                "median_speedup_ratio": (
                    1.0 - candidate_median / baseline_median
                ),
                "baseline_p99_cuda_ns": baseline_p99,
                "candidate_p99_cuda_ns": candidate_p99,
                "p99_regression_ratio": (
                    candidate_p99 / baseline_p99 - 1.0
                ),
                "baseline_median_host_submission_ns": (
                    baseline_host_median
                ),
                "candidate_median_host_submission_ns": (
                    candidate_host_median
                ),
                "host_submission_regression_ratio": (
                    candidate_host_median / baseline_host_median - 1.0
                ),
                "candidate_communication_union_ns": communication_total,
                "candidate_realized_overlap_ns": overlap_total,
                "realized_overlap_ratio": (
                    overlap_total / communication_total
                ),
            }
        )

    if evidence_incomplete:
        return _classification(
            "INCONCLUSIVE_EVIDENCE",
            shape_summaries,
        )
    if any(
        summary["realized_overlap_ratio"] < MIN_OVERLAP_RATIO
        for summary in shape_summaries
    ):
        return _classification(
            "NO_GO_INSUFFICIENT_OVERLAP",
            shape_summaries,
        )
    if all(
        summary["median_speedup_ratio"] < STOP_MEDIAN_SPEEDUP
        for summary in shape_summaries
    ):
        return _classification(
            "NO_GO_GEMM_FRAGMENTATION",
            shape_summaries,
        )
    if any(
        summary["median_speedup_ratio"]
        < MIN_MEDIAN_SPEEDUP[summary["active_tokens"]]
        or summary["p99_regression_ratio"] > MAX_P99_REGRESSION
        or summary["host_submission_regression_ratio"]
        > MAX_HOST_SUBMISSION_REGRESSION
        for summary in shape_summaries
    ):
        return _classification(
            "NO_GO_PERFORMANCE",
            shape_summaries,
        )
    return _classification(
        "GO_WAVEFRONT_MICROGATE",
        shape_summaries,
    )
