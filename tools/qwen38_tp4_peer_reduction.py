from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median


WORLD_SIZE = 4
HIDDEN_SIZE = 5120
MAX_ACTIVE_TOKENS = 8
SLOT_RING_SIZE = 2
MAX_ALLOCATED_DELTA_BYTES = 48 * 1024 * 1024
MIN_MEDIAN_SPEEDUP = 0.10
MAX_TOKENS8_REGRESSION = 0.02
MAX_P99_REGRESSION = 0.03
CROSS_RANK_ATOL = 2e-4
CROSS_RANK_RTOL = 2e-4
BASELINE_ATOL = 2e-2
BASELINE_RTOL = 2e-3
ACTIVE_TOKEN_GROUPS = (1, 4, 8)
MINIMUM_PAIRED_MEASUREMENTS = 200


@dataclass(frozen=True)
class PeerReductionPolicy:
    world_size: int = WORLD_SIZE
    hidden_size: int = HIDDEN_SIZE
    max_active_tokens: int = MAX_ACTIVE_TOKENS
    slot_ring_size: int = SLOT_RING_SIZE
    maximum_allocated_delta_bytes: int = (
        MAX_ALLOCATED_DELTA_BYTES
    )
    minimum_median_speedup: float = MIN_MEDIAN_SPEEDUP
    maximum_tokens8_regression: float = (
        MAX_TOKENS8_REGRESSION
    )
    maximum_p99_regression: float = MAX_P99_REGRESSION
    cross_rank_atol: float = CROSS_RANK_ATOL
    cross_rank_rtol: float = CROSS_RANK_RTOL
    baseline_atol: float = BASELINE_ATOL
    baseline_rtol: float = BASELINE_RTOL


def validate_peer_topology(rows):
    expected_edges = {
        (source_rank, destination_rank)
        for source_rank in range(WORLD_SIZE)
        for destination_rank in range(WORLD_SIZE)
        if source_rank != destination_rank
    }
    if not isinstance(rows, (list, tuple)):
        raise ValueError("peer topology must contain twelve edges")

    observed_edges = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("peer topology row is invalid")
        edge = (
            row.get("source_rank"),
            row.get("destination_rank"),
        )
        if (
            edge not in expected_edges
            or edge in observed_edges
            or row.get("can_access") is not True
            or row.get("ipc_roundtrip") is not True
        ):
            raise ValueError("peer topology is incomplete")
        observed_edges.add(edge)

    if observed_edges != expected_edges:
        raise ValueError("peer topology is incomplete")
    return {
        "classification": "PASS",
        "world_size": WORLD_SIZE,
        "directed_peer_edge_count": len(observed_edges),
    }


def _finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _percentile_nearest_rank(values, percentile):
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _ratio(candidate, baseline):
    if baseline <= 0:
        raise ValueError("baseline duration must be positive")
    return candidate / baseline - 1.0


def _classification(classification, *, shape_summaries=()):
    return {
        "classification": classification,
        "shape_summaries": list(shape_summaries),
    }


def classify_peer_microgate(rows, cleanup, memory):
    policy = PeerReductionPolicy()
    correctness_failed = False
    microgate_failed = False
    evidence_incomplete = False

    if not isinstance(rows, (list, tuple)) or not rows:
        evidence_incomplete = True
        rows = ()

    grouped = {active_tokens: {} for active_tokens in ACTIVE_TOKEN_GROUPS}
    seen = set()
    required_metrics = (
        "baseline_cuda_ns",
        "candidate_cuda_ns",
        "cross_rank_max_abs_error",
        "cross_rank_max_rel_error",
        "baseline_max_abs_error",
        "baseline_max_rel_error",
    )
    for row in rows:
        if not isinstance(row, dict):
            evidence_incomplete = True
            continue
        active_tokens = row.get("active_tokens")
        pair_index = row.get("pair_index")
        rank = row.get("rank")
        identity = (active_tokens, pair_index, rank)
        if (
            active_tokens not in grouped
            or type(pair_index) is not int
            or pair_index < 0
            or type(rank) is not int
            or rank not in range(WORLD_SIZE)
            or identity in seen
        ):
            evidence_incomplete = True
            continue
        seen.add(identity)
        if any(
            not _finite_nonnegative(row.get(name))
            for name in required_metrics
        ):
            evidence_incomplete = True
            continue
        if (
            row["baseline_cuda_ns"] == 0
            or row.get("timed_out") not in (True, False)
        ):
            evidence_incomplete = True
            continue
        if (
            row["cross_rank_max_abs_error"] > policy.cross_rank_atol
            or row["cross_rank_max_rel_error"] > policy.cross_rank_rtol
            or row["baseline_max_abs_error"] > policy.baseline_atol
            or row["baseline_max_rel_error"] > policy.baseline_rtol
        ):
            correctness_failed = True
        if row["timed_out"]:
            microgate_failed = True
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
    elif allocated_delta > policy.maximum_allocated_delta_bytes:
        return _classification("NO_GO_MEMORY")

    shape_summaries = []
    for active_tokens in ACTIVE_TOKEN_GROUPS:
        pairs = grouped[active_tokens]
        complete_pairs = []
        for pair_index, rank_rows in pairs.items():
            if set(rank_rows) != set(range(WORLD_SIZE)):
                evidence_incomplete = True
                continue
            complete_pairs.append((pair_index, rank_rows))
        complete_pairs.sort()
        if len(complete_pairs) < MINIMUM_PAIRED_MEASUREMENTS:
            evidence_incomplete = True
            continue

        baseline_values = [
            max(
                rank_rows[rank]["baseline_cuda_ns"]
                for rank in range(WORLD_SIZE)
            )
            for _, rank_rows in complete_pairs
        ]
        candidate_values = [
            max(
                rank_rows[rank]["candidate_cuda_ns"]
                for rank in range(WORLD_SIZE)
            )
            for _, rank_rows in complete_pairs
        ]
        baseline_median = median(baseline_values)
        candidate_median = median(candidate_values)
        baseline_p99 = _percentile_nearest_rank(
            baseline_values,
            0.99,
        )
        candidate_p99 = _percentile_nearest_rank(
            candidate_values,
            0.99,
        )
        median_speedup = (
            1.0 - candidate_median / baseline_median
        )
        p99_regression = _ratio(candidate_p99, baseline_p99)
        if active_tokens == MAX_ACTIVE_TOKENS:
            if median_speedup < -policy.maximum_tokens8_regression:
                microgate_failed = True
        elif median_speedup < policy.minimum_median_speedup:
            microgate_failed = True
        if p99_regression > policy.maximum_p99_regression:
            microgate_failed = True
        shape_summaries.append({
            "active_tokens": active_tokens,
            "pair_count": len(complete_pairs),
            "baseline_median_cuda_ns": baseline_median,
            "candidate_median_cuda_ns": candidate_median,
            "median_speedup_ratio": median_speedup,
            "baseline_p99_cuda_ns": baseline_p99,
            "candidate_p99_cuda_ns": candidate_p99,
            "p99_regression_ratio": p99_regression,
        })

    if microgate_failed:
        return _classification(
            "NO_GO_MICROGATE",
            shape_summaries=shape_summaries,
        )
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        evidence_incomplete = True
    if evidence_incomplete:
        return _classification(
            "INCONCLUSIVE_EVIDENCE",
            shape_summaries=shape_summaries,
        )
    return _classification(
        "PASS",
        shape_summaries=shape_summaries,
    )
