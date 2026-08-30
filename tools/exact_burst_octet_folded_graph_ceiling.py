#!/usr/bin/env python3
"""Classify the source-bound octet-folded exact-burst ceiling."""

from __future__ import annotations

import math
import statistics

from tools import profile_exact_burst_octet_folded_graph as profile


SCHEMA_VERSION = "exact-burst-octet-folded.ceiling.v1"
GO_CEILING = "GO_CEILING"
NO_GO_CEILING = "NO_GO_CEILING"

MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT = 1.0
MINIMUM_P95_TPOT_IMPROVEMENT_PCT = 0.5
MAXIMUM_PROTECTED_REGRESSION_PCT = 2.0
MAXIMUM_CAPTURE_MEMORY_RATIO = 0.01
MAXIMUM_RETAINED_STATIC_DELTA_BYTES = 128 * 1024 * 1024
MAXIMUM_FOLDED_CAPTURE_DURATION_NS = 120_000_000_000


def expected_performance_identities(
) -> tuple[tuple[int, int, str], ...]:
    return profile.performance_identities(
        repetitions=profile.REPETITIONS,
    )


def expected_correctness_identities(
) -> tuple[tuple[int, str, str], ...]:
    return profile.correctness_identities()


def inventory_status(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> tuple[bool, str | None]:
    if any(not isinstance(row, dict) for row in performance_rows):
        return False, "performance row payload is not an object"
    if any(not isinstance(row, dict) for row in correctness_rows):
        return False, "correctness row payload is not an object"
    performance = [
        (
            row.get("repetition"),
            row.get("context_length"),
            row.get("policy"),
        )
        for row in performance_rows
    ]
    correctness = [
        (
            row.get("context_length"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        for row in correctness_rows
    ]
    if (
        len(performance) != len(set(performance))
        or len(correctness) != len(set(correctness))
    ):
        return False, "duplicate row identity"
    if set(performance) != set(expected_performance_identities()):
        return False, "performance row inventory is incomplete"
    if set(correctness) != set(expected_correctness_identities()):
        return False, "correctness row inventory is incomplete"
    return True, None


def _finite_number(metrics: dict, field: str) -> float | None:
    value = metrics.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _valid_digest(value, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def classification_reasons(metrics: dict) -> tuple[str, ...]:
    reasons = []
    for field in (
        "evidence_complete",
        "source_exact",
        "workload_identity_exact",
        "execution_order_exact",
        "correctness_exact",
        "runtime_inventory_exact",
        "physical_launch_reduction_exact",
        "no_runtime_anomalies",
    ):
        if metrics.get(field) is not True:
            reasons.append(field)
    thresholds = (
        (
            "aggregate_median_tpot_improvement_pct",
            MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT,
            "minimum",
        ),
        (
            "aggregate_p95_tpot_improvement_pct",
            MINIMUM_P95_TPOT_IMPROVEMENT_PCT,
            "minimum",
        ),
        (
            "maximum_ttft_regression_pct",
            MAXIMUM_PROTECTED_REGRESSION_PCT,
            "maximum",
        ),
        (
            "maximum_e2e_regression_pct",
            MAXIMUM_PROTECTED_REGRESSION_PCT,
            "maximum",
        ),
        (
            "minimum_throughput_improvement_pct",
            -MAXIMUM_PROTECTED_REGRESSION_PCT,
            "minimum",
        ),
        (
            "maximum_capture_allocated_ratio",
            MAXIMUM_CAPTURE_MEMORY_RATIO,
            "maximum",
        ),
        (
            "maximum_capture_reserved_ratio",
            MAXIMUM_CAPTURE_MEMORY_RATIO,
            "maximum",
        ),
        (
            "maximum_retained_static_delta_bytes",
            MAXIMUM_RETAINED_STATIC_DELTA_BYTES,
            "maximum",
        ),
        (
            "maximum_folded_capture_duration_ns",
            MAXIMUM_FOLDED_CAPTURE_DURATION_NS,
            "maximum",
        ),
    )
    for field, threshold, direction in thresholds:
        value = _finite_number(metrics, field)
        if value is None:
            reasons.append(field)
        elif direction == "minimum" and value < threshold:
            reasons.append(field)
        elif direction == "maximum" and value > threshold:
            reasons.append(field)
    if "source_commit" in metrics and not _valid_digest(
        metrics["source_commit"],
        (40, 64),
    ):
        reasons.append("source_commit")
    if "source_patch_sha256" in metrics and not _valid_digest(
        metrics["source_patch_sha256"],
        (64,),
    ):
        reasons.append("source_patch_sha256")
    observed_sources = metrics.get("observed_source_commits")
    if observed_sources is not None and (
        not isinstance(observed_sources, list)
        or len(set(observed_sources)) != 1
        or (
            "source_commit" in metrics
            and observed_sources[0] != metrics["source_commit"]
        )
    ):
        reasons.append("source_mismatch")
    return tuple(dict.fromkeys(reasons))


def classify(metrics: dict) -> str:
    if not isinstance(metrics, dict):
        return NO_GO_CEILING
    return (
        GO_CEILING
        if not classification_reasons(metrics)
        else NO_GO_CEILING
    )


def _improvement_pct(control: float, candidate: float) -> float:
    if (
        not math.isfinite(control)
        or not math.isfinite(candidate)
        or control <= 0.0
        or candidate < 0.0
    ):
        raise ValueError("metric inputs must be finite and valid")
    return (control - candidate) / control * 100.0


def summarize_evidence(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> dict:
    complete, inventory_error = inventory_status(
        performance_rows,
        correctness_rows,
    )
    if not complete:
        metrics = {
            **_empty_metrics(),
            "evidence_complete": False,
            "inventory_error": inventory_error,
        }
        metrics["classification_reasons"] = list(
            classification_reasons(metrics)
        )
        metrics["classification"] = classify(metrics)
        return metrics
    try:
        rows = [
            profile.validate_case_row(row)
            for row in performance_rows
        ]
        validated_correctness = profile.validate_correctness_rows(
            correctness_rows
        )
    except (KeyError, TypeError, ValueError) as error:
        metrics = {
            **_empty_metrics(),
            "inventory_error": str(error),
        }
        metrics["classification_reasons"] = list(
            classification_reasons(metrics)
        )
        metrics["classification"] = classify(metrics)
        return metrics

    pairs = {}
    for row in rows:
        pairs.setdefault(
            (row["repetition"], row["context_length"]),
            {},
        )[row["policy"]] = row
    paired = list(pairs.values())
    controls = [pair["one_token_graph"] for pair in paired]
    candidates = [pair["octet_folded_graph"] for pair in paired]
    median_improvements = [
        _improvement_pct(
            control["tpot_median_ns"],
            candidate["tpot_median_ns"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    p95_improvements = [
        _improvement_pct(
            control["tpot_p95_ns"],
            candidate["tpot_p95_ns"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    ttft_regressions = [
        -_improvement_pct(
            control["ttft_ns"],
            candidate["ttft_ns"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    e2e_regressions = [
        -_improvement_pct(
            control["e2e_ns"],
            candidate["e2e_ns"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    throughput_improvements = [
        -_improvement_pct(
            control["output_tokens_per_second"],
            candidate["output_tokens_per_second"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    correctness_exact = _correctness_exact(validated_correctness)
    runtime_inventory_exact = all(
        row["logical_forwards"] == row["logical_replays"]
        == GENERATED_LOGICAL_REPLAYS
        and row["token_d2h_calls"]
        == math.ceil(GENERATED_LOGICAL_REPLAYS / 8)
        for row in rows
    )
    physical_launch_reduction_exact = all(
        control["one_token_cuda_graph_launches"]
        == GENERATED_LOGICAL_REPLAYS
        and control["folded_cuda_graph_launches"] == 0
        and candidate["folded_cuda_graph_launches"]
        == GENERATED_LOGICAL_REPLAYS // 8
        and candidate["one_token_cuda_graph_launches"]
        == GENERATED_LOGICAL_REPLAYS % 8
        for control, candidate in zip(controls, candidates)
    )
    no_runtime_anomalies = all(
        row["fallback_count"] == 0
        and row["rollback_count"] == 0
        and row["quarantine_reason"] is None
        for row in rows
    )
    allocated_ratios = [
        candidate["capture_allocated_delta_bytes"]
        / max(1, control["cuda_peak_allocated_bytes"])
        for control, candidate in zip(controls, candidates)
    ]
    reserved_ratios = [
        candidate["capture_reserved_delta_bytes"]
        / max(1, control["cuda_peak_reserved_bytes"])
        for control, candidate in zip(controls, candidates)
    ]
    retained_deltas = [
        candidate["capture_retained_static_bytes"]
        - control["capture_retained_static_bytes"]
        for control, candidate in zip(controls, candidates)
    ]
    source_commits = sorted({
        row["source_commit"]
        for row in (*rows, *validated_correctness)
    })
    patch_digests = sorted({
        row["source_patch_sha256"]
        for row in (*rows, *validated_correctness)
    })
    workload_identity_exact = all(
        control["prompt_sha256"] == candidate["prompt_sha256"]
        and control["output_token_ids"]
        == candidate["output_token_ids"]
        and control["output_text_sha256"]
        == candidate["output_text_sha256"]
        for control, candidate in zip(controls, candidates)
    )
    execution_order_exact = all(
        sorted(pair) == sorted(profile.POLICIES)
        and {
            row["order_position"] for row in pair.values()
        } == {0, 1}
        for pair in paired
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": (
            len(source_commits) == 1
            and len(patch_digests) == 1
        ),
        "workload_identity_exact": workload_identity_exact,
        "execution_order_exact": execution_order_exact,
        "correctness_exact": correctness_exact,
        "runtime_inventory_exact": runtime_inventory_exact,
        "physical_launch_reduction_exact":
            physical_launch_reduction_exact,
        "no_runtime_anomalies": no_runtime_anomalies,
        "aggregate_median_tpot_improvement_pct":
            statistics.median(median_improvements),
        "aggregate_p95_tpot_improvement_pct":
            statistics.median(p95_improvements),
        "maximum_ttft_regression_pct": max(ttft_regressions),
        "maximum_e2e_regression_pct": max(e2e_regressions),
        "minimum_throughput_improvement_pct": min(
            throughput_improvements
        ),
        "maximum_capture_allocated_ratio": max(
            allocated_ratios
        ),
        "maximum_capture_reserved_ratio": max(reserved_ratios),
        "maximum_retained_static_delta_bytes": max(
            retained_deltas
        ),
        "maximum_folded_capture_duration_ns": max(
            row["capture_duration_ns"] for row in candidates
        ),
        "source_commit": source_commits[0],
        "source_patch_sha256": patch_digests[0],
        "observed_source_commits": source_commits,
        "performance_row_count": len(rows),
        "correctness_row_count": len(correctness_rows),
        "inventory_error": None,
    }
    metrics["classification_reasons"] = list(
        classification_reasons(metrics)
    )
    metrics["classification"] = classify(metrics)
    return metrics


GENERATED_LOGICAL_REPLAYS = profile.GENERATED_TOKENS - 1


def _correctness_exact(rows: list[dict]) -> bool:
    groups = {}
    for row in rows:
        key = (row["context_length"], row["sampling_point"])
        groups.setdefault(key, {})[row["policy"]] = row
    if len(groups) != (
        len(profile.CONTEXT_LENGTHS)
        * len(profile.SAMPLING_POINTS)
    ):
        return False
    return all(
        set(pair) == set(profile.POLICIES)
        and pair["one_token_graph"].get("output_token_ids")
        == pair["octet_folded_graph"].get("output_token_ids")
        and pair["one_token_graph"].get("output_text_sha256")
        == pair["octet_folded_graph"].get("output_text_sha256")
        and pair["one_token_graph"].get("argmax_token_id")
        == pair["octet_folded_graph"].get("argmax_token_id")
        and pair["one_token_graph"].get("logits_sha256")
        == pair["octet_folded_graph"].get("logits_sha256")
        for pair in groups.values()
    )


def _empty_metrics() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_complete": False,
        "source_exact": False,
        "workload_identity_exact": False,
        "execution_order_exact": False,
        "correctness_exact": False,
        "runtime_inventory_exact": False,
        "physical_launch_reduction_exact": False,
        "no_runtime_anomalies": False,
        "aggregate_median_tpot_improvement_pct": float("-inf"),
        "aggregate_p95_tpot_improvement_pct": float("-inf"),
        "maximum_ttft_regression_pct": float("inf"),
        "maximum_e2e_regression_pct": float("inf"),
        "minimum_throughput_improvement_pct": float("-inf"),
        "maximum_capture_allocated_ratio": float("inf"),
        "maximum_capture_reserved_ratio": float("inf"),
        "maximum_retained_static_delta_bytes": float("inf"),
        "maximum_folded_capture_duration_ns": float("inf"),
    }
