#!/usr/bin/env python3
"""Source-bound contracts for the SLO-aware cohort-burst ceiling gate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence


COST_SAMPLE_SCHEMA_VERSION = "slo-cohort-burst.cost-sample.v1"
COST_TABLE_SCHEMA_VERSION = "slo-cohort-burst.cost-table.v1"
CEILING_SUMMARY_SCHEMA_VERSION = "slo-cohort-burst.ceiling-summary.v1"
ARTIFACT_SCHEMA_VERSION = "slo-cohort-burst.ceiling-artifact.v1"

NO_GO_CEILING = "NO_GO_CEILING"
CONTINUE_RUNTIME = "CONTINUE_RUNTIME"
INVALID_SOURCE_OR_EVIDENCE = "INVALID_SOURCE_OR_EVIDENCE"

MINIMUM_OPTIMISTIC_HEADROOM_RATIO = 0.12
SUPPORTED_BURST_WIDTHS = (1, 2, 4, 8)

_SOURCE_DIGEST_FIELDS = (
    "checkpoint_sha256",
    "config_sha256",
    "source_patch_sha256",
)
_SOURCE_TEXT_FIELDS = (
    "dtype",
    "gpu_name",
    "gpu_uuid",
    "model",
)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_digest(
    value: object,
    name: str,
    *,
    lengths: tuple[int, ...] = (64,),
) -> str:
    text = _require_non_empty_string(value, name)
    if len(text) not in lengths or any(
        character not in "0123456789abcdef"
        for character in text
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return text


def _normalize_source_identity(
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(source_identity, Mapping):
        raise ValueError("source identity must be a mapping")
    required = {
        "source_commit",
        "source_patch_sha256",
        "model",
        "checkpoint_sha256",
        "gpu_uuid",
        "gpu_name",
        "tensor_parallel_size",
        "dtype",
        "config_sha256",
    }
    if set(source_identity) != required:
        raise ValueError("source identity fields are incomplete")
    normalized = dict(source_identity)
    normalized["source_commit"] = _require_digest(
        normalized["source_commit"],
        "source_commit",
        lengths=(40, 64),
    )
    for field in _SOURCE_DIGEST_FIELDS:
        normalized[field] = _require_digest(
            normalized[field],
            field,
        )
    for field in _SOURCE_TEXT_FIELDS:
        normalized[field] = _require_non_empty_string(
            normalized[field],
            field,
        )
    if _require_positive_int(
        normalized["tensor_parallel_size"],
        "tensor_parallel_size",
    ) != 1:
        raise ValueError("ceiling source identity requires TP1")
    return normalized


@dataclass(frozen=True, order=True)
class SLOCohortCostKey:
    batch_size: int
    context_bucket: int
    burst_width: int

    def __post_init__(self) -> None:
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.context_bucket, "context_bucket")
        _require_positive_int(self.burst_width, "burst_width")
        if self.burst_width not in SUPPORTED_BURST_WIDTHS:
            raise ValueError("burst_width is unsupported")

    @property
    def canonical_name(self) -> str:
        return (
            f"b{self.batch_size}-c{self.context_bucket}"
            f"-k{self.burst_width}"
        )


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
    if not isinstance(values, Sequence) or not values:
        raise ValueError("nearest-rank values must be non-empty")
    normalized = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("nearest-rank values must be finite numbers")
        normalized.append(value)
    ordered = sorted(normalized)
    rank = max(1, math.ceil(float(percentile) * len(ordered)))
    return ordered[rank - 1]


def _normalize_cost_row(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("cost row must be a mapping")
    required = {
        "schema_version",
        "sample_id",
        "batch_size",
        "context_bucket",
        "burst_width",
        "duration_ns",
    }
    if set(row) != required:
        raise ValueError("cost row fields are incomplete")
    if row["schema_version"] != COST_SAMPLE_SCHEMA_VERSION:
        raise ValueError("cost row schema version is invalid")
    sample_id = _require_non_empty_string(
        row["sample_id"],
        "sample_id",
    )
    key = SLOCohortCostKey(
        batch_size=_require_positive_int(
            row["batch_size"],
            "batch_size",
        ),
        context_bucket=_require_positive_int(
            row["context_bucket"],
            "context_bucket",
        ),
        burst_width=_require_positive_int(
            row["burst_width"],
            "burst_width",
        ),
    )
    duration_ns = _require_positive_int(
        row["duration_ns"],
        "duration_ns",
    )
    return {
        "schema_version": COST_SAMPLE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "batch_size": key.batch_size,
        "context_bucket": key.context_bucket,
        "burst_width": key.burst_width,
        "duration_ns": duration_ns,
    }


def build_frozen_cost_table(
    rows: Sequence[Mapping[str, object]],
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("cost rows must be non-empty")
    normalized_source = _normalize_source_identity(source_identity)
    normalized_rows = [_normalize_cost_row(row) for row in rows]
    sample_ids = [row["sample_id"] for row in normalized_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample ID")

    grouped: dict[SLOCohortCostKey, list[int]] = defaultdict(list)
    for row in normalized_rows:
        key = SLOCohortCostKey(
            batch_size=row["batch_size"],
            context_bucket=row["context_bucket"],
            burst_width=row["burst_width"],
        )
        grouped[key].append(row["duration_ns"])

    entries = {}
    for key in sorted(grouped):
        samples = sorted(grouped[key])
        entries[key.canonical_name] = {
            "batch_size": key.batch_size,
            "context_bucket": key.context_bucket,
            "burst_width": key.burst_width,
            "sample_count": len(samples),
            "raw_sample_sha256": _sha256(samples),
            "p50_ns": nearest_rank(samples, 0.50),
            "p95_ns": nearest_rank(samples, 0.95),
            "p99_ns": nearest_rank(samples, 0.99),
        }

    payload = {
        "schema_version": COST_TABLE_SCHEMA_VERSION,
        "source_identity": normalized_source,
        "entries": entries,
    }
    return {
        **payload,
        "table_sha256": _sha256(payload),
    }


def _finite_non_negative_ratio(
    summary: Mapping[str, object],
    field: str,
) -> float | None:
    value = summary.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        return None
    return float(value)


def classify_ceiling(summary: Mapping[str, object]) -> str:
    if not isinstance(summary, Mapping):
        return INVALID_SOURCE_OR_EVIDENCE
    if (
        summary.get("evidence_complete") is not True
        or summary.get("source_exact") is not True
    ):
        return INVALID_SOURCE_OR_EVIDENCE
    medium = _finite_non_negative_ratio(
        summary,
        "medium_headroom_ratio",
    )
    high = _finite_non_negative_ratio(
        summary,
        "high_headroom_ratio",
    )
    if medium is None or high is None:
        return INVALID_SOURCE_OR_EVIDENCE
    if (
        medium < MINIMUM_OPTIMISTIC_HEADROOM_RATIO
        and high < MINIMUM_OPTIMISTIC_HEADROOM_RATIO
    ):
        return NO_GO_CEILING
    return CONTINUE_RUNTIME


def verify_ceiling_artifact(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(artifact, Mapping):
        raise ValueError("ceiling artifact must be a mapping")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("ceiling artifact schema version is invalid")
    source_identity = artifact.get("source_identity")
    cost_rows = artifact.get("cost_rows")
    cost_table = artifact.get("cost_table")
    summary = artifact.get("ceiling_summary")
    if (
        not isinstance(source_identity, Mapping)
        or not isinstance(cost_rows, list)
        or not isinstance(cost_table, Mapping)
        or not isinstance(summary, Mapping)
    ):
        raise ValueError("ceiling artifact fields are incomplete")

    rebuilt_table = build_frozen_cost_table(
        cost_rows,
        source_identity,
    )
    if rebuilt_table != dict(cost_table):
        raise ValueError("cost table does not match raw samples")
    if (
        summary.get("schema_version")
        != CEILING_SUMMARY_SCHEMA_VERSION
    ):
        raise ValueError("ceiling summary schema version is invalid")
    classification = classify_ceiling(summary)
    if summary.get("classification") != classification:
        raise ValueError("ceiling classification does not reconstruct")
    return {
        "verified": True,
        "cost_table_sha256": rebuilt_table["table_sha256"],
        "classification": classification,
    }
