from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from tools import slo_cohort_burst_ceiling as ceiling


SOURCE_IDENTITY = {
    "source_commit": "a" * 40,
    "source_patch_sha256": "b" * 64,
    "model": "Qwen3-0.6B",
    "checkpoint_sha256": "c" * 64,
    "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    "gpu_name": "NVIDIA A100 80GB PCIe",
    "tensor_parallel_size": 1,
    "dtype": "torch.bfloat16",
    "config_sha256": "d" * 64,
}


def _cost_rows() -> list[dict]:
    rows = []
    for duration_ns in (10, 20, 30, 40, 50):
        rows.append({
            "schema_version": ceiling.COST_SAMPLE_SCHEMA_VERSION,
            "sample_id": f"b4-c2048-k4-{duration_ns}",
            "batch_size": 4,
            "context_bucket": 2048,
            "burst_width": 4,
            "duration_ns": duration_ns,
        })
    for duration_ns in (5, 10, 15, 20, 25):
        rows.append({
            "schema_version": ceiling.COST_SAMPLE_SCHEMA_VERSION,
            "sample_id": f"b4-c2048-k2-{duration_ns}",
            "batch_size": 4,
            "context_bucket": 2048,
            "burst_width": 2,
            "duration_ns": duration_ns,
        })
    return rows


def _artifact() -> dict:
    rows = _cost_rows()
    source_identity = deepcopy(SOURCE_IDENTITY)
    table = ceiling.build_frozen_cost_table(
        rows,
        source_identity,
    )
    summary = {
        "schema_version": ceiling.CEILING_SUMMARY_SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": True,
        "medium_headroom_ratio": 0.12,
        "high_headroom_ratio": 0.15,
    }
    summary["classification"] = ceiling.classify_ceiling(summary)
    return {
        "schema_version": ceiling.ARTIFACT_SCHEMA_VERSION,
        "source_identity": source_identity,
        "cost_rows": rows,
        "cost_table": table,
        "ceiling_summary": summary,
    }


def test_cost_key_is_strict_and_canonical() -> None:
    key = ceiling.SLOCohortCostKey(
        batch_size=4,
        context_bucket=2048,
        burst_width=8,
    )
    assert key.canonical_name == "b4-c2048-k8"

    for kwargs in (
        {"batch_size": True, "context_bucket": 2048, "burst_width": 8},
        {"batch_size": 0, "context_bucket": 2048, "burst_width": 8},
        {"batch_size": 4, "context_bucket": -1, "burst_width": 8},
        {"batch_size": 4, "context_bucket": 2048, "burst_width": 3},
    ):
        with pytest.raises(ValueError):
            ceiling.SLOCohortCostKey(**kwargs)


def test_nearest_rank_is_exact_and_rejects_invalid_samples() -> None:
    values = (10, 20, 30, 40, 50)
    assert ceiling.nearest_rank(values, 0.50) == 30
    assert ceiling.nearest_rank(values, 0.95) == 50
    assert ceiling.nearest_rank(values, 0.99) == 50

    for values, percentile in (
        ((), 0.99),
        ((1, float("nan")), 0.99),
        ((1, float("inf")), 0.99),
        ((1, True), 0.99),
        ((1, 2), 0.0),
        ((1, 2), 1.01),
    ):
        with pytest.raises(ValueError):
            ceiling.nearest_rank(values, percentile)


def test_cost_table_is_source_bound_and_uses_nearest_rank_p99() -> None:
    rows = _cost_rows()
    table = ceiling.build_frozen_cost_table(rows, SOURCE_IDENTITY)

    assert table["schema_version"] == ceiling.COST_TABLE_SCHEMA_VERSION
    assert table["source_identity"] == SOURCE_IDENTITY
    assert list(table["entries"]) == [
        "b4-c2048-k2",
        "b4-c2048-k4",
    ]
    assert table["entries"]["b4-c2048-k4"] == {
        "batch_size": 4,
        "context_bucket": 2048,
        "burst_width": 4,
        "sample_count": 5,
        "raw_sample_sha256": hashlib.sha256(
            json.dumps(
                [10, 20, 30, 40, 50],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "p50_ns": 30,
        "p95_ns": 50,
        "p99_ns": 50,
    }
    assert len(table["table_sha256"]) == 64
    assert table == ceiling.build_frozen_cost_table(
        list(reversed(rows)),
        dict(reversed(tuple(SOURCE_IDENTITY.items()))),
    )


def test_cost_table_rejects_duplicate_sample_ids_and_bad_rows() -> None:
    rows = _cost_rows()
    with pytest.raises(ValueError, match="duplicate sample ID"):
        ceiling.build_frozen_cost_table(
            rows + [deepcopy(rows[0])],
            SOURCE_IDENTITY,
        )

    for field, value in (
        ("schema_version", "wrong"),
        ("sample_id", ""),
        ("duration_ns", 0),
        ("duration_ns", True),
        ("burst_width", 3),
    ):
        mutated = deepcopy(rows)
        mutated[0][field] = value
        with pytest.raises(ValueError):
            ceiling.build_frozen_cost_table(
                mutated,
                SOURCE_IDENTITY,
            )


def test_ceiling_boundary_and_invalid_inputs_fail_closed() -> None:
    assert ceiling.classify_ceiling({
        "evidence_complete": True,
        "source_exact": True,
        "medium_headroom_ratio": 0.119,
        "high_headroom_ratio": 0.118,
    }) == ceiling.NO_GO_CEILING
    assert ceiling.classify_ceiling({
        "evidence_complete": True,
        "source_exact": True,
        "medium_headroom_ratio": 0.12,
        "high_headroom_ratio": 0.0,
    }) == ceiling.CONTINUE_RUNTIME
    assert ceiling.classify_ceiling({
        "evidence_complete": True,
        "source_exact": True,
        "medium_headroom_ratio": 0.0,
        "high_headroom_ratio": 0.12,
    }) == ceiling.CONTINUE_RUNTIME

    for mutation in (
        {"evidence_complete": False},
        {"source_exact": False},
        {"medium_headroom_ratio": float("nan")},
        {"high_headroom_ratio": True},
        {"medium_headroom_ratio": -0.001},
    ):
        summary = {
            "evidence_complete": True,
            "source_exact": True,
            "medium_headroom_ratio": 0.12,
            "high_headroom_ratio": 0.12,
        }
        summary.update(mutation)
        assert ceiling.classify_ceiling(summary) == (
            ceiling.INVALID_SOURCE_OR_EVIDENCE
        )


def test_verify_artifact_reconstructs_table_and_classification() -> None:
    artifact = _artifact()
    verified = ceiling.verify_ceiling_artifact(artifact)

    assert verified["verified"] is True
    assert verified["cost_table_sha256"] == (
        artifact["cost_table"]["table_sha256"]
    )
    assert verified["classification"] == ceiling.CONTINUE_RUNTIME


@pytest.mark.parametrize(
    "mutation",
    (
        "source",
        "raw_sample",
        "table_percentile",
        "table_hash",
        "classification",
    ),
)
def test_verify_artifact_rejects_authoritative_mutation(
    mutation: str,
) -> None:
    artifact = _artifact()
    if mutation == "source":
        artifact["source_identity"]["source_commit"] = "9" * 40
    elif mutation == "raw_sample":
        artifact["cost_rows"][0]["duration_ns"] += 1
    elif mutation == "table_percentile":
        artifact["cost_table"]["entries"][
            "b4-c2048-k4"
        ]["p99_ns"] += 1
    elif mutation == "table_hash":
        artifact["cost_table"]["table_sha256"] = "0" * 64
    elif mutation == "classification":
        artifact["ceiling_summary"]["classification"] = (
            ceiling.NO_GO_CEILING
        )

    with pytest.raises(ValueError):
        ceiling.verify_ceiling_artifact(artifact)
