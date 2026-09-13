#!/usr/bin/env python3
"""Baseline attribution for the SLO-aware cohort-burst ceiling gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from tools import slo_cohort_burst_ceiling as ceiling


PROFILE_ROW_SCHEMA_VERSION = "slo-cohort-burst.ceiling-profile-row.v1"
FROZEN_LOADS = ("low", "medium", "high")
_RAW_COMPONENT_FIELDS = (
    ("target_cuda_ns", "target_cuda"),
    ("graph_launch_gap_ns", "graph_launch_gap"),
    ("scheduler_ns", "scheduler"),
    ("token_d2h_publication_ns", "token_d2h_publication"),
    ("batch_binding_ns", "batch_binding"),
)
_AMORTIZABLE_COMPONENTS = (
    "graph_launch_gap",
    "scheduler",
    "token_d2h_publication",
    "batch_binding",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def profile_baseline_case(
    engine,
    case,
    *,
    clock_ns,
) -> dict[str, object]:
    measured = engine.profile_baseline_case(case, clock_ns=clock_ns)
    if not isinstance(measured, Mapping):
        raise ValueError("profile result must be a mapping")
    wall_ns = _positive_int(measured.get("wall_ns"), "wall_ns")
    component_ns = {}
    for raw_name, public_name in _RAW_COMPONENT_FIELDS:
        component_ns[public_name] = _non_negative_int(
            measured.get(raw_name),
            raw_name,
        )
    attributed_ns = sum(component_ns.values())
    if attributed_ns > wall_ns:
        raise ValueError("profile components exceed wall time")
    component_ns["unattributed"] = wall_ns - attributed_ns

    load = _text(case.load, "load")
    if load not in FROZEN_LOADS:
        raise ValueError("load is outside the frozen inventory")
    key = ceiling.SLOCohortCostKey(
        batch_size=case.batch_size,
        context_bucket=case.context_bucket,
        burst_width=case.burst_width,
    )
    return {
        "schema_version": PROFILE_ROW_SCHEMA_VERSION,
        "case_id": _text(case.case_id, "case_id"),
        "load": load,
        "source_commit": _text(
            case.source_commit,
            "source_commit",
        ),
        "batch_size": key.batch_size,
        "context_bucket": key.context_bucket,
        "burst_width": key.burst_width,
        "component_ns": component_ns,
        "wall_ns": wall_ns,
        "committed_tokens": _positive_int(
            measured.get("committed_tokens"),
            "committed_tokens",
        ),
        "cuda_reserved_bytes": _non_negative_int(
            measured.get("cuda_reserved_bytes"),
            "cuda_reserved_bytes",
        ),
    }


def _validate_profile_row(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("profile row must be a mapping")
    if row.get("schema_version") != PROFILE_ROW_SCHEMA_VERSION:
        raise ValueError("profile row schema version is invalid")
    load = _text(row.get("load"), "load")
    if load not in FROZEN_LOADS:
        raise ValueError("load is outside the frozen inventory")
    components = row.get("component_ns")
    if not isinstance(components, Mapping):
        raise ValueError("component_ns must be a mapping")
    expected_components = {
        "target_cuda",
        "graph_launch_gap",
        "scheduler",
        "token_d2h_publication",
        "batch_binding",
        "unattributed",
    }
    if set(components) != expected_components:
        raise ValueError("profile component inventory is incomplete")
    normalized_components = {
        name: _non_negative_int(components[name], name)
        for name in sorted(expected_components)
    }
    wall_ns = _positive_int(row.get("wall_ns"), "wall_ns")
    if sum(normalized_components.values()) != wall_ns:
        raise ValueError("profile components do not equal wall time")
    return {
        "schema_version": PROFILE_ROW_SCHEMA_VERSION,
        "case_id": _text(row.get("case_id"), "case_id"),
        "load": load,
        "source_commit": _text(
            row.get("source_commit"),
            "source_commit",
        ),
        "batch_size": _positive_int(
            row.get("batch_size"),
            "batch_size",
        ),
        "context_bucket": _positive_int(
            row.get("context_bucket"),
            "context_bucket",
        ),
        "burst_width": _positive_int(
            row.get("burst_width"),
            "burst_width",
        ),
        "component_ns": normalized_components,
        "wall_ns": wall_ns,
        "committed_tokens": _positive_int(
            row.get("committed_tokens"),
            "committed_tokens",
        ),
        "cuda_reserved_bytes": _non_negative_int(
            row.get("cuda_reserved_bytes"),
            "cuda_reserved_bytes",
        ),
    }


def _optimistic_headroom_ratio(row: Mapping[str, object]) -> float:
    components = row["component_ns"]
    removable_ns = sum(
        components[name] for name in _AMORTIZABLE_COMPONENTS
    )
    irreducible_ns = row["wall_ns"] - removable_ns
    if irreducible_ns <= 0:
        raise ValueError("optimistic irreducible time must be positive")
    value = row["wall_ns"] / irreducible_ns - 1.0
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("optimistic headroom is invalid")
    return value


def build_ceiling_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("profile rows must be non-empty")
    normalized = [_validate_profile_row(row) for row in rows]
    case_ids = [row["case_id"] for row in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case ID")
    observed_loads = {row["load"] for row in normalized}
    if observed_loads != set(FROZEN_LOADS):
        raise ValueError("load inventory is incomplete")
    source_commits = {row["source_commit"] for row in normalized}
    source_exact = (
        len(source_commits) == 1
        and len(next(iter(source_commits))) == 40
        and all(
            character in "0123456789abcdef"
            for character in next(iter(source_commits))
        )
    )
    by_load = {}
    for load in FROZEN_LOADS:
        values = [
            _optimistic_headroom_ratio(row)
            for row in normalized
            if row["load"] == load
        ]
        by_load[load] = statistics.median(values)
    summary = {
        "schema_version": ceiling.CEILING_SUMMARY_SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": source_exact,
        "row_count": len(normalized),
        "low_headroom_ratio": by_load["low"],
        "medium_headroom_ratio": by_load["medium"],
        "high_headroom_ratio": by_load["high"],
    }
    summary["classification"] = ceiling.classify_ceiling(summary)
    return summary


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    destination = Path(path)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ValueError(
            f"artifact already exists: {destination.name}"
        ) from error


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_ceiling_bundle(
    *,
    output_dir: Path,
    profile_rows: Sequence[Mapping[str, object]],
    cost_rows: Sequence[Mapping[str, object]],
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("artifact destination is not empty")
    else:
        destination.mkdir(parents=True)

    normalized_rows = [
        _validate_profile_row(row) for row in profile_rows
    ]
    summary = build_ceiling_summary(normalized_rows)
    table = ceiling.build_frozen_cost_table(
        cost_rows,
        source_identity,
    )
    artifact = {
        "schema_version": ceiling.ARTIFACT_SCHEMA_VERSION,
        "source_identity": dict(source_identity),
        "cost_rows": [dict(row) for row in cost_rows],
        "cost_table": table,
        "ceiling_summary": summary,
    }
    verification = ceiling.verify_ceiling_artifact(artifact)

    _write_bytes_exclusive(
        destination / "raw_rows.jsonl",
        b"".join(
            json.dumps(
                row,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
            for row in normalized_rows
        ),
    )
    _write_bytes_exclusive(
        destination / "cost_table.json",
        _canonical_json_bytes(table),
    )
    _write_bytes_exclusive(
        destination / "ceiling_summary.json",
        _canonical_json_bytes(summary),
    )
    _write_bytes_exclusive(
        destination / "source_manifest.json",
        _canonical_json_bytes(dict(source_identity)),
    )
    _write_bytes_exclusive(
        destination / "remote_verify.json",
        _canonical_json_bytes(verification),
    )
    return verification
