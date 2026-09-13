#!/usr/bin/env python3
"""Closed metrics and formal classification for SLO cohort bursts."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


INVALID_SOURCE_OR_EVIDENCE = "INVALID_SOURCE_OR_EVIDENCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_LIFECYCLE = "NO_GO_LIFECYCLE"
NO_GO_STARVATION = "NO_GO_STARVATION"
NO_GO_TAIL_LATENCY = "NO_GO_TAIL_LATENCY"
NO_GO_MEMORY = "NO_GO_MEMORY"
NO_GO_EOS_WASTE = "NO_GO_EOS_WASTE"
NO_GO_THROUGHPUT = "NO_GO_THROUGHPUT"
GO_SLO_AWARE_COHORT_DECODE_BURST = (
    "GO_SLO_AWARE_COHORT_DECODE_BURST"
)

FAILURE_PRECEDENCE = (
    INVALID_SOURCE_OR_EVIDENCE,
    NO_GO_CORRECTNESS,
    NO_GO_LIFECYCLE,
    NO_GO_STARVATION,
    NO_GO_TAIL_LATENCY,
    NO_GO_MEMORY,
    NO_GO_EOS_WASTE,
    NO_GO_THROUGHPUT,
    GO_SLO_AWARE_COHORT_DECODE_BURST,
)

_BOOLEAN_FIELDS = (
    "evidence_complete",
    "source_exact",
    "verifier_agreement",
    "correctness_passed",
    "lifecycle_closed",
)
_FLOAT_FIELDS = (
    "aggregate_throughput_improvement",
    "medium_throughput_improvement",
    "high_throughput_improvement",
    "worst_throughput_regression",
    "worst_p99_itl_regression",
    "worst_p99_ttft_regression",
    "worst_p99_e2e_regression",
    "post_eos_wasted_forward_fraction",
    "peak_reserved_memory_regression",
)


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _non_negative_int(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    text = _non_empty_string(value, name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef"
        for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def nearest_rank(
    values: Sequence[int | float],
    percentile: float,
) -> int | float:
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not math.isfinite(float(percentile))
        or not 0.0 < float(percentile) <= 1.0
    ):
        raise ValueError("percentile must be finite and in (0, 1]")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise ValueError("nearest-rank values must be non-empty")
    normalized = [
        _finite_number(value, "nearest-rank value")
        for value in values
    ]
    ordered = sorted(normalized)
    rank = max(1, math.ceil(float(percentile) * len(ordered)))
    selected = ordered[rank - 1]
    if all(isinstance(value, int) for value in values):
        return int(selected)
    return selected


def _normalize_request_row(
    row: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("request row must be a mapping")
    required = {
        "request_id",
        "sequence_id",
        "service_class",
        "arrival_ns",
        "prefill_start_ns",
        "prefill_complete_ns",
        "first_token_visible_ns",
        "token_visible_ns",
        "completion_ns",
        "output_token_ids",
        "output_text_sha256",
        "terminal_reason",
    }
    fields = set(row)
    if fields not in (required, required | {"schema_version"}):
        raise ValueError("request row fields are incomplete")
    if (
        "schema_version" in row
        and row["schema_version"]
        != "slo-cohort-burst.request.v1"
    ):
        raise ValueError("request row schema version is invalid")
    request_id = _non_empty_string(
        row["request_id"],
        "request_id",
    )
    sequence_id = _non_negative_int(
        row["sequence_id"],
        "sequence_id",
    )
    service_class = _non_empty_string(
        row["service_class"],
        "service_class",
    )
    arrival_ns = _non_negative_int(row["arrival_ns"], "arrival_ns")
    prefill_start_ns = _non_negative_int(
        row["prefill_start_ns"],
        "prefill_start_ns",
    )
    prefill_complete_ns = _non_negative_int(
        row["prefill_complete_ns"],
        "prefill_complete_ns",
    )
    first_token_visible_ns = _non_negative_int(
        row["first_token_visible_ns"],
        "first_token_visible_ns",
    )
    completion_ns = _non_negative_int(
        row["completion_ns"],
        "completion_ns",
    )
    visible = row["token_visible_ns"]
    output_tokens = row["output_token_ids"]
    if not isinstance(visible, (tuple, list)) or not visible:
        raise ValueError("token_visible_ns must be non-empty")
    if not isinstance(output_tokens, (tuple, list)) or not output_tokens:
        raise ValueError("output_token_ids must be non-empty")
    visible_ns = tuple(
        _non_negative_int(value, "token visibility timestamp")
        for value in visible
    )
    token_ids = tuple(
        _non_negative_int(value, "output token ID")
        for value in output_tokens
    )
    if len(visible_ns) != len(token_ids):
        raise ValueError(
            "token timestamp count does not match output token count"
        )
    if first_token_visible_ns != visible_ns[0]:
        raise ValueError(
            "first token timestamp does not match token timeline"
        )
    timeline = (
        arrival_ns,
        prefill_start_ns,
        prefill_complete_ns,
        first_token_visible_ns,
        *visible_ns[1:],
        completion_ns,
    )
    if any(current < prior for prior, current in zip(
        timeline,
        timeline[1:],
    )):
        raise ValueError("request timestamps are not monotonic")
    if completion_ns != visible_ns[-1]:
        raise ValueError(
            "completion timestamp must equal final token visibility"
        )
    return {
        "request_id": request_id,
        "sequence_id": sequence_id,
        "service_class": service_class,
        "arrival_ns": arrival_ns,
        "prefill_start_ns": prefill_start_ns,
        "prefill_complete_ns": prefill_complete_ns,
        "first_token_visible_ns": first_token_visible_ns,
        "token_visible_ns": visible_ns,
        "completion_ns": completion_ns,
        "output_token_ids": token_ids,
        "output_text_sha256": _digest(
            row["output_text_sha256"],
            "output_text_sha256",
        ),
        "terminal_reason": _non_empty_string(
            row["terminal_reason"],
            "terminal_reason",
        ),
    }


def summarize_request_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not rows
    ):
        raise ValueError("request rows must be non-empty")
    normalized = tuple(_normalize_request_row(row) for row in rows)
    request_ids = tuple(row["request_id"] for row in normalized)
    sequence_ids = tuple(row["sequence_id"] for row in normalized)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request rows contain duplicate request IDs")
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("request rows contain duplicate sequence IDs")
    start_ns = min(row["arrival_ns"] for row in normalized)
    end_ns = max(row["completion_ns"] for row in normalized)
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        raise ValueError("request measurement window must be positive")
    ttft_ns = [
        row["token_visible_ns"][0] - row["arrival_ns"]
        for row in normalized
    ]
    itl_ns = [
        current - prior
        for row in normalized
        for prior, current in zip(
            row["token_visible_ns"],
            row["token_visible_ns"][1:],
        )
    ]
    if not itl_ns:
        raise ValueError("request rows contain no ITL samples")
    e2e_ns = [
        row["completion_ns"] - row["arrival_ns"]
        for row in normalized
    ]
    committed_output_tokens = sum(
        len(row["output_token_ids"]) for row in normalized
    )
    return {
        "request_count": len(normalized),
        "committed_output_tokens": committed_output_tokens,
        "measurement_start_ns": start_ns,
        "measurement_end_ns": end_ns,
        "measurement_duration_ns": duration_ns,
        "request_throughput_rps": (
            len(normalized) * 1_000_000_000.0 / duration_ns
        ),
        "output_throughput_tps": (
            committed_output_tokens
            * 1_000_000_000.0
            / duration_ns
        ),
        "p50_ttft_ns": nearest_rank(ttft_ns, 0.50),
        "p95_ttft_ns": nearest_rank(ttft_ns, 0.95),
        "p99_ttft_ns": nearest_rank(ttft_ns, 0.99),
        "p50_itl_ns": nearest_rank(itl_ns, 0.50),
        "p95_itl_ns": nearest_rank(itl_ns, 0.95),
        "p99_itl_ns": nearest_rank(itl_ns, 0.99),
        "p50_e2e_ns": nearest_rank(e2e_ns, 0.50),
        "p95_e2e_ns": nearest_rank(e2e_ns, 0.95),
        "p99_e2e_ns": nearest_rank(e2e_ns, 0.99),
        "maximum_host_visible_gap_ns": max(itl_ns),
        "starved_requests": sum(
            row["terminal_reason"] == "starved"
            for row in normalized
        ),
    }


def classify_slo_cohort_burst(
    summary: Mapping[str, object],
) -> str:
    if not isinstance(summary, Mapping):
        return INVALID_SOURCE_OR_EVIDENCE
    required = set(_BOOLEAN_FIELDS) | set(_FLOAT_FIELDS) | {
        "maximum_host_visible_gap_ns",
        "starved_requests",
    }
    if set(summary) != required:
        return INVALID_SOURCE_OR_EVIDENCE
    if any(
        not isinstance(summary[field], bool)
        for field in _BOOLEAN_FIELDS
    ):
        return INVALID_SOURCE_OR_EVIDENCE
    try:
        metrics = {
            field: _finite_number(summary[field], field)
            for field in _FLOAT_FIELDS
        }
        maximum_gap_ns = _non_negative_int(
            summary["maximum_host_visible_gap_ns"],
            "maximum_host_visible_gap_ns",
        )
        starved_requests = _non_negative_int(
            summary["starved_requests"],
            "starved_requests",
        )
    except ValueError:
        return INVALID_SOURCE_OR_EVIDENCE
    if any(
        metrics[field] < 0.0
        for field in (
            "worst_throughput_regression",
            "worst_p99_itl_regression",
            "worst_p99_ttft_regression",
            "worst_p99_e2e_regression",
            "post_eos_wasted_forward_fraction",
            "peak_reserved_memory_regression",
        )
    ):
        return INVALID_SOURCE_OR_EVIDENCE
    if not (
        summary["evidence_complete"]
        and summary["source_exact"]
        and summary["verifier_agreement"]
    ):
        return INVALID_SOURCE_OR_EVIDENCE
    if not summary["correctness_passed"]:
        return NO_GO_CORRECTNESS
    if not summary["lifecycle_closed"]:
        return NO_GO_LIFECYCLE
    if starved_requests:
        return NO_GO_STARVATION
    if (
        metrics["worst_p99_itl_regression"] > 0.03
        or metrics["worst_p99_ttft_regression"] > 0.05
        or metrics["worst_p99_e2e_regression"] > 0.05
        or maximum_gap_ns > 40_000_000
    ):
        return NO_GO_TAIL_LATENCY
    if metrics["peak_reserved_memory_regression"] > 0.05:
        return NO_GO_MEMORY
    if metrics["post_eos_wasted_forward_fraction"] > 0.10:
        return NO_GO_EOS_WASTE
    if (
        metrics["aggregate_throughput_improvement"] < 0.10
        or metrics["medium_throughput_improvement"] < 0.10
        or metrics["high_throughput_improvement"] < 0.10
        or metrics["worst_throughput_regression"] > 0.02
    ):
        return NO_GO_THROUGHPUT
    return GO_SLO_AWARE_COHORT_DECODE_BURST
