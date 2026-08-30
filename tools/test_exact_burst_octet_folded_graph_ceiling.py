from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import exact_burst_octet_folded_graph_ceiling as ceiling
from tools.test_profile_exact_burst_octet_folded_graph import (
    PATCH_SHA256,
    SOURCE_COMMIT,
    _case_row,
    _correctness_rows,
)


def _valid_metrics() -> dict:
    return {
        "evidence_complete": True,
        "source_exact": True,
        "workload_identity_exact": True,
        "execution_order_exact": True,
        "correctness_exact": True,
        "runtime_inventory_exact": True,
        "physical_launch_reduction_exact": True,
        "no_runtime_anomalies": True,
        "minimum_foldable_launch_reduction_pct": 85.0,
        "aggregate_median_tpot_improvement_pct": 1.0,
        "aggregate_p95_tpot_improvement_pct": 0.5,
        "maximum_context_median_tpot_regression_pct": 2.0,
        "maximum_context_p95_tpot_regression_pct": 2.0,
        "maximum_tpot_p99_regression_pct": 2.0,
        "maximum_ttft_regression_pct": 2.0,
        "maximum_e2e_regression_pct": 2.0,
        "minimum_throughput_improvement_pct": -2.0,
        "maximum_capture_allocated_ratio": 0.01,
        "maximum_capture_reserved_ratio": 0.01,
        "maximum_retained_static_delta_bytes": 128 * 1024 * 1024,
        "maximum_folded_capture_duration_ns": 120_000_000_000,
    }


def test_ceiling_thresholds_are_frozen() -> None:
    assert ceiling.MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT == 1.0
    assert ceiling.MINIMUM_P95_TPOT_IMPROVEMENT_PCT == 0.5
    assert ceiling.MINIMUM_FOLDABLE_LAUNCH_REDUCTION_PCT == 85.0
    assert ceiling.MAXIMUM_PROTECTED_REGRESSION_PCT == 2.0
    assert ceiling.MAXIMUM_CAPTURE_MEMORY_RATIO == 0.01
    assert ceiling.MAXIMUM_RETAINED_STATIC_DELTA_BYTES == (
        128 * 1024 * 1024
    )
    assert ceiling.MAXIMUM_FOLDED_CAPTURE_DURATION_NS == (
        120_000_000_000
    )
    assert ceiling.classify(_valid_metrics()) == (
        ceiling.GO_CEILING
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evidence_complete", False),
        ("source_exact", False),
        ("workload_identity_exact", False),
        ("execution_order_exact", False),
        ("correctness_exact", False),
        ("runtime_inventory_exact", False),
        ("physical_launch_reduction_exact", False),
        ("no_runtime_anomalies", False),
        ("minimum_foldable_launch_reduction_pct", 84.999999),
        ("aggregate_median_tpot_improvement_pct", 0.999999),
        ("aggregate_p95_tpot_improvement_pct", 0.499999),
        (
            "maximum_context_median_tpot_regression_pct",
            2.000001,
        ),
        (
            "maximum_context_p95_tpot_regression_pct",
            2.000001,
        ),
        ("maximum_tpot_p99_regression_pct", 2.000001),
        ("maximum_ttft_regression_pct", 2.000001),
        ("maximum_e2e_regression_pct", 2.000001),
        ("minimum_throughput_improvement_pct", -2.000001),
        ("maximum_capture_allocated_ratio", 0.010001),
        ("maximum_capture_reserved_ratio", 0.010001),
        (
            "maximum_retained_static_delta_bytes",
            128 * 1024 * 1024 + 1,
        ),
        (
            "maximum_folded_capture_duration_ns",
            120_000_000_001,
        ),
    ),
)
def test_each_ceiling_failure_is_independently_no_go(
    field,
    value,
) -> None:
    metrics = _valid_metrics()
    metrics[field] = value
    assert ceiling.classify(metrics) == ceiling.NO_GO_CEILING


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_non_finite_metric_is_no_go(value) -> None:
    metrics = _valid_metrics()
    metrics["aggregate_median_tpot_improvement_pct"] = value
    assert ceiling.classify(metrics) == ceiling.NO_GO_CEILING


def test_inventory_rejects_missing_and_duplicate_rows() -> None:
    performance = [
        {
            "repetition": repetition,
            "context_length": context,
            "policy": policy,
        }
        for repetition, context, policy
        in ceiling.expected_performance_identities()
    ]
    correctness = [
        {
            "context_length": context,
            "policy": policy,
            "sampling_point": point,
        }
        for context, policy, point
        in ceiling.expected_correctness_identities()
    ]
    complete, reason = ceiling.inventory_status(
        performance,
        correctness,
    )
    assert complete is True
    assert reason is None

    complete, reason = ceiling.inventory_status(
        performance[:-1],
        correctness,
    )
    assert complete is False
    assert "performance" in reason

    duplicate = performance + [deepcopy(performance[0])]
    complete, reason = ceiling.inventory_status(
        duplicate,
        correctness,
    )
    assert complete is False
    assert "duplicate" in reason


def test_digest_and_source_mismatch_are_no_go() -> None:
    metrics = _valid_metrics()
    metrics["source_commit"] = "not-a-digest"
    metrics["source_patch_sha256"] = "b" * 64
    assert ceiling.classify(metrics) == ceiling.NO_GO_CEILING

    metrics = _valid_metrics()
    metrics["source_commit"] = "a" * 40
    metrics["source_patch_sha256"] = "b" * 64
    metrics["observed_source_commits"] = ["a" * 40, "c" * 40]
    assert ceiling.classify(metrics) == ceiling.NO_GO_CEILING


def _performance_rows() -> list[dict]:
    rows = []
    for repetition, context_length, policy in (
        ceiling.expected_performance_identities()
    ):
        row = _case_row(
            policy,
            context_length=context_length,
            repetition=repetition,
        )
        if policy == "octet_folded_graph":
            row["tpot_median_ns"] = 990_000.0
            row["tpot_p95_ns"] = 995_000.0
        rows.append(row)
    return rows


def test_correctness_source_drift_is_no_go(tmp_path: Path) -> None:
    correctness = _correctness_rows(tmp_path)
    correctness[-1]["source_commit"] = "9" * 40
    result = ceiling.summarize_evidence(
        _performance_rows(),
        correctness,
    )
    assert result["source_exact"] is False
    assert "source_exact" in result["classification_reasons"]
    assert result["classification"] == ceiling.NO_GO_CEILING


def test_throughput_regression_uses_control_as_denominator(
    tmp_path: Path,
) -> None:
    performance = _performance_rows()
    for row in performance:
        row["output_tokens_per_second"] = (
            100.0 if row["policy"] == "one_token_graph" else 98.0
        )
    result = ceiling.summarize_evidence(
        performance,
        _correctness_rows(tmp_path),
    )
    assert result["minimum_throughput_improvement_pct"] == pytest.approx(
        -2.0
    )
    assert result["classification"] == ceiling.GO_CEILING


def test_swapped_pair_execution_order_is_no_go(
    tmp_path: Path,
) -> None:
    performance = _performance_rows()
    pair = [
        row
        for row in performance
        if row["repetition"] == 0
        and row["context_length"] == 256
    ]
    for row in pair:
        row["order_position"] = 1 - row["order_position"]
    result = ceiling.summarize_evidence(
        performance,
        _correctness_rows(tmp_path),
    )
    assert result["execution_order_exact"] is False
    assert result["classification"] == ceiling.NO_GO_CEILING


def test_summary_enforces_launch_and_tpot_protection_metrics(
    tmp_path: Path,
) -> None:
    performance = _performance_rows()
    result = ceiling.summarize_evidence(
        performance,
        _correctness_rows(tmp_path),
    )
    assert result["minimum_foldable_launch_reduction_pct"] == (
        pytest.approx(87.5)
    )
    assert result[
        "maximum_context_median_tpot_regression_pct"
    ] == pytest.approx(-1.0)
    assert result[
        "maximum_context_p95_tpot_regression_pct"
    ] == pytest.approx(-0.5)
    assert result["maximum_tpot_p99_regression_pct"] == (
        pytest.approx(0.0)
    )

    for row in performance:
        if (
            row["context_length"] == 8192
            and row["policy"] == "octet_folded_graph"
        ):
            row["tpot_p99_ns"] = 1_020_001.0
    result = ceiling.summarize_evidence(
        performance,
        _correctness_rows(tmp_path / "p99"),
    )
    assert result["maximum_tpot_p99_regression_pct"] > 2.0
    assert "maximum_tpot_p99_regression_pct" in (
        result["classification_reasons"]
    )
    assert result["classification"] == ceiling.NO_GO_CEILING
