#!/usr/bin/env python3
"""Frozen optimistic ceiling for a lease-scoped persistent decode segment."""

from __future__ import annotations

import math
import statistics


TIMING_SCHEMA_VERSION = "lease-sealed-persistent-decode.timing.v1"
TRACE_SUMMARY_SCHEMA_VERSION = (
    "lease-sealed-persistent-decode.trace-summary.v1"
)
CEILING_SCHEMA_VERSION = "lease-sealed-persistent-decode.ceiling.v1"

GO_PERSISTENT_DECODE_CEILING = "GO_PERSISTENT_DECODE_CEILING"
NO_GO_PERSISTENT_DECODE_CEILING = "NO_GO_PERSISTENT_DECODE_CEILING"
INCONCLUSIVE_PROFILE_OVERHEAD = "INCONCLUSIVE_PROFILE_OVERHEAD"
INCONCLUSIVE_TRACE_COVERAGE = "INCONCLUSIVE_TRACE_COVERAGE"
INCONCLUSIVE_CORRECTNESS = "INCONCLUSIVE_CORRECTNESS"
INCOMPLETE_EVIDENCE = "INCOMPLETE_EVIDENCE"

CONTEXT_LENGTHS = (256, 2048, 8192)
GENERATED_TOKENS = 128
REPETITIONS = 5
MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT = 5.0
MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT = 3.0
MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT = 4.0
MIN_CLASSIFIED_LAUNCH_RATIO = 0.98
MIN_CLASSIFIED_DURATION_RATIO = 0.99
MAX_MEDIAN_PROFILE_PERTURBATION_PCT = 10.0
MAX_P95_PROFILE_PERTURBATION_PCT = 15.0

_IDENTITY_FIELDS = (
    "source_commit",
    "source_tree_sha256",
    "runtime_identity_sha256",
    "workload_identity_sha256",
)


class _IncompleteEvidence(ValueError):
    pass


def _finite(value, field: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return result


def _integer(value, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _IncompleteEvidence(f"{field} must be an integer")
    if value < minimum:
        raise _IncompleteEvidence(
            f"{field} must be at least {minimum}"
        )
    return value


def _digest(value, field: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _IncompleteEvidence(f"{field} is invalid")
    return value


def _tokens(value, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != GENERATED_TOKENS
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in value
        )
    ):
        raise _IncompleteEvidence(f"{field} is invalid")
    return list(value)


def _identity(payload: dict) -> tuple[str, str, str, str]:
    return (
        _digest(payload.get("source_commit"), "source_commit", length=40),
        _digest(
            payload.get("source_tree_sha256"),
            "source_tree_sha256",
        ),
        _digest(
            payload.get("runtime_identity_sha256"),
            "runtime_identity_sha256",
        ),
        _digest(
            payload.get("workload_identity_sha256"),
            "workload_identity_sha256",
        ),
    )


def _runtime_is_clean(row: dict) -> bool:
    return (
        row.get("fallback_count") == 0
        and row.get("failure_count") == 0
        and row.get("rollback_count") == 0
        and row.get("quarantine_reason") is None
        and row.get("target_model_forwards")
        == GENERATED_TOKENS - 1
        and row.get("committed_tokens")
        == GENERATED_TOKENS - 1
    )


def validate_timing_rows(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list):
        raise _IncompleteEvidence("timing rows must be a list")
    expected = {
        (repetition, context)
        for repetition in range(REPETITIONS)
        for context in CONTEXT_LENGTHS
    }
    normalized = []
    identities = set()
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise _IncompleteEvidence("timing row must be an object")
        if row.get("schema_version") != TIMING_SCHEMA_VERSION:
            raise _IncompleteEvidence("timing schema version mismatch")
        if row.get("arm") != "uninstrumented":
            raise _IncompleteEvidence(
                "timing row is not uninstrumented authority"
            )
        identity = _identity(row)
        identities.add(identity)
        context = _integer(
            row.get("context_length"),
            "context_length",
            minimum=1,
        )
        repetition = _integer(
            row.get("repetition"),
            "repetition",
        )
        key = (repetition, context)
        if key in seen:
            raise _IncompleteEvidence("duplicate timing identity")
        seen.add(key)
        if row.get("generated_tokens") != GENERATED_TOKENS:
            raise _IncompleteEvidence("generated token count mismatch")
        normalized.append({
            **row,
            "context_length": context,
            "repetition": repetition,
            "tpot_median_ns": _finite(
                row.get("tpot_median_ns"),
                "tpot_median_ns",
                positive=True,
            ),
            "tpot_p95_ns": _finite(
                row.get("tpot_p95_ns"),
                "tpot_p95_ns",
                positive=True,
            ),
            "output_token_ids": _tokens(
                row.get("output_token_ids"),
                "output_token_ids",
            ),
            "output_text_sha256": _digest(
                row.get("output_text_sha256"),
                "output_text_sha256",
            ),
        })
    if seen != expected:
        raise _IncompleteEvidence("timing row inventory is incomplete")
    if len(identities) != 1:
        raise _IncompleteEvidence("timing identity is inconsistent")
    return normalized


def validate_trace_summary(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise _IncompleteEvidence("trace summary must be an object")
    if payload.get("schema_version") != TRACE_SUMMARY_SCHEMA_VERSION:
        raise _IncompleteEvidence("trace summary schema version mismatch")
    identity = _identity(payload)
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
        raise _IncompleteEvidence("trace contexts must be a list")
    normalized = []
    seen = set()
    for row in contexts:
        if not isinstance(row, dict):
            raise _IncompleteEvidence("trace context must be an object")
        context = _integer(
            row.get("context_length"),
            "trace context_length",
            minimum=1,
        )
        if context in seen:
            raise _IncompleteEvidence("duplicate trace context")
        seen.add(context)
        signatures = row.get("segment_signatures")
        if not isinstance(signatures, list):
            raise _IncompleteEvidence(
                "segment signatures must be a list"
            )
        normalized_signatures = [
            _digest(signature, "segment signature")
            for signature in signatures
        ]
        launch_ratio = _finite(
            row.get("classified_launch_ratio"),
            "classified_launch_ratio",
        )
        duration_ratio = _finite(
            row.get("classified_duration_ratio"),
            "classified_duration_ratio",
        )
        if launch_ratio > 1.0 or duration_ratio > 1.0:
            raise ValueError("classification ratio must not exceed one")
        normalized.append({
            **row,
            "context_length": context,
            "profiled_tpot_median_ns": _finite(
                row.get("profiled_tpot_median_ns"),
                "profiled_tpot_median_ns",
                positive=True,
            ),
            "profiled_tpot_p95_ns": _finite(
                row.get("profiled_tpot_p95_ns"),
                "profiled_tpot_p95_ns",
                positive=True,
            ),
            "output_token_ids": _tokens(
                row.get("output_token_ids"),
                "trace output_token_ids",
            ),
            "output_text_sha256": _digest(
                row.get("output_text_sha256"),
                "trace output_text_sha256",
            ),
            "transaction_count": _integer(
                row.get("transaction_count"),
                "transaction_count",
                minimum=1,
            ),
            "logical_token_count": _integer(
                row.get("logical_token_count"),
                "logical_token_count",
                minimum=1,
            ),
            "eligible_zero_cost_ns_per_token": _finite(
                row.get("eligible_zero_cost_ns_per_token"),
                "eligible_zero_cost_ns_per_token",
            ),
            "candidate_cuda_duration_ns": _finite(
                row.get("candidate_cuda_duration_ns"),
                "candidate_cuda_duration_ns",
            ),
            "total_kernel_duration_ns": _finite(
                row.get("total_kernel_duration_ns"),
                "total_kernel_duration_ns",
                positive=True,
            ),
            "classified_launch_ratio": launch_ratio,
            "classified_duration_ratio": duration_ratio,
            "segment_signatures": normalized_signatures,
        })
    if seen != set(CONTEXT_LENGTHS):
        raise _IncompleteEvidence("trace context inventory is incomplete")
    return {
        **payload,
        "_identity": identity,
        "contexts": normalized,
    }


def _incomplete(reason: str) -> dict:
    return {
        "schema_version": CEILING_SCHEMA_VERSION,
        "classification": INCOMPLETE_EVIDENCE,
        "failed_conditions": [reason],
    }


def _regression_pct(candidate: float, baseline: float) -> float:
    return max(0.0, (candidate / baseline - 1.0) * 100.0)


def compute_ceiling(
    timing_rows: list[dict],
    trace_summary: dict,
) -> dict:
    try:
        timing = validate_timing_rows(timing_rows)
        trace = validate_trace_summary(trace_summary)
    except _IncompleteEvidence as error:
        return _incomplete(str(error))

    timing_identity = _identity(timing[0])
    if trace["_identity"] != timing_identity:
        return _incomplete("source/runtime/workload identity mismatch")
    if any(not _runtime_is_clean(row) for row in timing):
        return _incomplete("timing runtime evidence is not clean")
    if any(
        not _runtime_is_clean(row)
        for row in trace["contexts"]
    ):
        return _incomplete("trace runtime evidence is not clean")

    timing_by_context = {
        context: [
            row for row in timing
            if row["context_length"] == context
        ]
        for context in CONTEXT_LENGTHS
    }
    trace_by_context = {
        row["context_length"]: row for row in trace["contexts"]
    }

    for context, rows in timing_by_context.items():
        expected_tokens = rows[0]["output_token_ids"]
        expected_text = rows[0]["output_text_sha256"]
        if any(
            row["output_token_ids"] != expected_tokens
            or row["output_text_sha256"] != expected_text
            for row in rows[1:]
        ):
            return {
                "schema_version": CEILING_SCHEMA_VERSION,
                "classification": INCONCLUSIVE_CORRECTNESS,
                "failed_conditions": [
                    "uninstrumented output mismatch",
                ],
            }
        traced = trace_by_context[context]
        if (
            traced["output_token_ids"] != expected_tokens
            or traced["output_text_sha256"] != expected_text
        ):
            return {
                "schema_version": CEILING_SCHEMA_VERSION,
                "classification": INCONCLUSIVE_CORRECTNESS,
                "failed_conditions": [
                    "timing and structural output mismatch",
                ],
            }

    minimum_launch_ratio = min(
        row["classified_launch_ratio"]
        for row in trace["contexts"]
    )
    minimum_duration_ratio = min(
        row["classified_duration_ratio"]
        for row in trace["contexts"]
    )
    coverage_failures = []
    if minimum_launch_ratio < MIN_CLASSIFIED_LAUNCH_RATIO:
        coverage_failures.append("classified_launch_ratio")
    if minimum_duration_ratio < MIN_CLASSIFIED_DURATION_RATIO:
        coverage_failures.append("classified_duration_ratio")
    if coverage_failures:
        return {
            "schema_version": CEILING_SCHEMA_VERSION,
            "classification": INCONCLUSIVE_TRACE_COVERAGE,
            "failed_conditions": coverage_failures,
            "minimum_classified_launch_ratio": minimum_launch_ratio,
            "minimum_classified_duration_ratio": minimum_duration_ratio,
        }

    context_metrics = []
    median_perturbations = []
    p95_perturbations = []
    for context in CONTEXT_LENGTHS:
        baseline_rows = timing_by_context[context]
        baseline_median = statistics.median(
            row["tpot_median_ns"] for row in baseline_rows
        )
        baseline_p95 = statistics.median(
            row["tpot_p95_ns"] for row in baseline_rows
        )
        traced = trace_by_context[context]
        optimistic_pct = (
            traced["eligible_zero_cost_ns_per_token"]
            / baseline_median
            * 100.0
        )
        median_perturbation = _regression_pct(
            traced["profiled_tpot_median_ns"],
            baseline_median,
        )
        p95_perturbation = _regression_pct(
            traced["profiled_tpot_p95_ns"],
            baseline_p95,
        )
        median_perturbations.append(median_perturbation)
        p95_perturbations.append(p95_perturbation)
        context_metrics.append({
            "context_length": context,
            "baseline_tpot_median_ns": baseline_median,
            "baseline_tpot_p95_ns": baseline_p95,
            "profiled_tpot_median_ns":
                traced["profiled_tpot_median_ns"],
            "profiled_tpot_p95_ns":
                traced["profiled_tpot_p95_ns"],
            "profile_median_perturbation_pct":
                median_perturbation,
            "profile_p95_perturbation_pct":
                p95_perturbation,
            "eligible_zero_cost_ns_per_token":
                traced["eligible_zero_cost_ns_per_token"],
            "optimistic_improvement_pct": optimistic_pct,
            "candidate_cuda_duration_ns":
                traced["candidate_cuda_duration_ns"],
            "total_kernel_duration_ns":
                traced["total_kernel_duration_ns"],
            "classified_launch_ratio":
                traced["classified_launch_ratio"],
            "classified_duration_ratio":
                traced["classified_duration_ratio"],
            "segment_signatures":
                traced["segment_signatures"],
        })

    maximum_median_perturbation = max(median_perturbations)
    maximum_p95_perturbation = max(p95_perturbations)
    overhead_failures = []
    if (
        maximum_median_perturbation
        > MAX_MEDIAN_PROFILE_PERTURBATION_PCT
    ):
        overhead_failures.append("profile_median_perturbation_pct")
    if (
        maximum_p95_perturbation
        > MAX_P95_PROFILE_PERTURBATION_PCT
    ):
        overhead_failures.append("profile_p95_perturbation_pct")
    if overhead_failures:
        return {
            "schema_version": CEILING_SCHEMA_VERSION,
            "classification": INCONCLUSIVE_PROFILE_OVERHEAD,
            "failed_conditions": overhead_failures,
            "maximum_profile_median_perturbation_pct":
                maximum_median_perturbation,
            "maximum_profile_p95_perturbation_pct":
                maximum_p95_perturbation,
            "contexts": context_metrics,
        }

    optimistic_values = [
        row["optimistic_improvement_pct"] for row in context_metrics
    ]
    aggregate_optimistic = statistics.median(optimistic_values)
    total_candidate_duration = sum(
        row["candidate_cuda_duration_ns"]
        for row in context_metrics
    )
    total_kernel_duration = sum(
        row["total_kernel_duration_ns"]
        for row in context_metrics
    )
    candidate_share = (
        total_candidate_duration / total_kernel_duration * 100.0
    )
    stable_signatures = sorted(set.intersection(*(
        set(row["segment_signatures"])
        for row in context_metrics
    )))

    failed_conditions = []
    if (
        aggregate_optimistic
        < MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT
    ):
        failed_conditions.append(
            "aggregate_optimistic_improvement_pct"
        )
    if (
        min(optimistic_values)
        < MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT
    ):
        failed_conditions.append(
            "minimum_context_optimistic_improvement_pct"
        )
    if (
        candidate_share
        < MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT
    ):
        failed_conditions.append(
            "aggregate_candidate_cuda_duration_share_pct"
        )
    if not stable_signatures:
        failed_conditions.append(
            "stable_cross_context_signatures"
        )
    classification = (
        NO_GO_PERSISTENT_DECODE_CEILING
        if failed_conditions
        else GO_PERSISTENT_DECODE_CEILING
    )
    return {
        "schema_version": CEILING_SCHEMA_VERSION,
        "classification": classification,
        "failed_conditions": failed_conditions,
        "thresholds": {
            "minimum_aggregate_optimistic_improvement_pct":
                MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT,
            "minimum_context_optimistic_improvement_pct":
                MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT,
            "minimum_candidate_cuda_duration_share_pct":
                MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT,
            "minimum_classified_launch_ratio":
                MIN_CLASSIFIED_LAUNCH_RATIO,
            "minimum_classified_duration_ratio":
                MIN_CLASSIFIED_DURATION_RATIO,
            "maximum_profile_median_perturbation_pct":
                MAX_MEDIAN_PROFILE_PERTURBATION_PCT,
            "maximum_profile_p95_perturbation_pct":
                MAX_P95_PROFILE_PERTURBATION_PCT,
        },
        "aggregate_optimistic_improvement_pct":
            aggregate_optimistic,
        "minimum_context_optimistic_improvement_pct":
            min(optimistic_values),
        "aggregate_candidate_cuda_duration_share_pct":
            candidate_share,
        "minimum_classified_launch_ratio": minimum_launch_ratio,
        "minimum_classified_duration_ratio": minimum_duration_ratio,
        "maximum_profile_median_perturbation_pct":
            maximum_median_perturbation,
        "maximum_profile_p95_perturbation_pct":
            maximum_p95_perturbation,
        "stable_cross_context_signatures": stable_signatures,
        "contexts": context_metrics,
    }


def classify_ceiling(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise ValueError("ceiling payload must be an object")
    if payload.get("schema_version") != CEILING_SCHEMA_VERSION:
        raise ValueError("ceiling schema version mismatch")
    classification = payload.get("classification")
    allowed = {
        GO_PERSISTENT_DECODE_CEILING,
        NO_GO_PERSISTENT_DECODE_CEILING,
        INCONCLUSIVE_PROFILE_OVERHEAD,
        INCONCLUSIVE_TRACE_COVERAGE,
        INCONCLUSIVE_CORRECTNESS,
        INCOMPLETE_EVIDENCE,
    }
    if classification not in allowed:
        raise ValueError("ceiling classification is invalid")
    return classification
