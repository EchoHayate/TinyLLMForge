from __future__ import annotations

import copy

import pytest

from tools.lease_sealed_state_commit_overlap import (
    ACTIVE_TOKEN_GROUPS,
    MEASURED_PAIR_COUNT,
    WORLD_SIZE,
    classify_stage0,
    interval_intersection_ns,
    validate_measurement_row,
)


def passing_rows():
    return [
        {
            "attempt": "20260907-lease-sealed-stage0-r1",
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "active_tokens": active_tokens,
            "pair_index": pair_index,
            "rank": rank,
            "arm_order": (
                ["baseline", "candidate"]
                if pair_index % 2 == 0
                else ["candidate", "baseline"]
            ),
            "baseline_critical_ns": 100_000,
            "candidate_critical_ns": 90_000,
            "baseline_host_submission_ns": 20_000,
            "candidate_host_submission_ns": 20_400,
            "allreduce_interval_ns": [10_000, 60_000],
            "state_copy_interval_ns": [35_000, 75_000],
            "overlap_intersection_ns": 25_000,
            "reduced_output_exact": True,
            "final_output_exact": True,
            "shadow_payload_exact": True,
            "active_state_preserved_before_publish": True,
            "published_state_exact": True,
            "abort_preserved_old_state": True,
            "commit_identity_match": True,
            "finite_output": True,
            "timed_path_allocation_count": 0,
            "timed_out": False,
        }
        for active_tokens in ACTIVE_TOKEN_GROUPS
        for pair_index in range(MEASURED_PAIR_COUNT)
        for rank in range(WORLD_SIZE)
    ]


def passing_memory():
    return {
        "rank_rows": [
            {
                "rank": rank,
                "maximum_allocated_delta_bytes": 100_000_000,
                "maximum_reserved_delta_bytes": 120_000_000,
                "maximum_theoretical_shadow_bytes": 104_202_240,
            }
            for rank in range(WORLD_SIZE)
        ]
    }


def test_frozen_inventory_and_interval_math():
    assert ACTIVE_TOKEN_GROUPS == (1, 4, 8)
    assert MEASURED_PAIR_COUNT == 15
    assert WORLD_SIZE == 4
    assert interval_intersection_ns((10, 60), (35, 75)) == 25


def test_classifier_accepts_complete_profitable_evidence():
    result = classify_stage0(
        passing_rows(),
        passing_memory(),
        {"classification": "CLEAN"},
    )

    assert result["classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert result["stage1_authorized"] is True
    assert result["measurement_row_count"] == 180


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("coverage", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
        ("correctness", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("final_output", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("identity", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("allocation", "NO_GO_MEMORY_OR_ALLOCATION"),
        ("memory", "NO_GO_MEMORY_OR_ALLOCATION"),
        ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
        ("median", "NO_GO_PERFORMANCE"),
        ("tail", "NO_GO_PERFORMANCE"),
        ("host", "NO_GO_PERFORMANCE"),
        ("direction", "NO_GO_PERFORMANCE"),
        ("cleanup", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
    ),
)
def test_classifier_fails_closed(mutation, expected):
    rows = passing_rows()
    memory = passing_memory()
    cleanup = {"classification": "CLEAN"}
    if mutation == "coverage":
        rows.pop()
    elif mutation == "correctness":
        rows[0]["shadow_payload_exact"] = False
    elif mutation == "final_output":
        rows[0]["final_output_exact"] = False
    elif mutation == "identity":
        rows[0]["commit_identity_match"] = False
    elif mutation == "allocation":
        rows[0]["timed_path_allocation_count"] = 1
    elif mutation == "memory":
        memory["rank_rows"][0]["maximum_reserved_delta_bytes"] = (
            memory["rank_rows"][0]["maximum_theoretical_shadow_bytes"]
            + 64 * 1024 * 1024
            + 1
        )
    elif mutation == "overlap":
        for row in rows:
            if row["active_tokens"] == 4:
                row["state_copy_interval_ns"] = [58_000, 75_000]
                row["overlap_intersection_ns"] = 2_000
    elif mutation == "median":
        for row in rows:
            if row["active_tokens"] in (4, 8):
                row["candidate_critical_ns"] = 98_000
    elif mutation == "tail":
        for row in rows:
            if row["pair_index"] == 14:
                row["candidate_critical_ns"] = 110_000
    elif mutation == "host":
        for row in rows:
            row["candidate_host_submission_ns"] = 21_000
    elif mutation == "direction":
        for row in rows:
            if row["active_tokens"] == 8 and row["pair_index"] >= 10:
                row["candidate_critical_ns"] = 101_000
    elif mutation == "cleanup":
        cleanup["classification"] = "DIRTY"

    assert classify_stage0(rows, memory, cleanup)["classification"] == expected


def test_measurement_row_rejects_nonfinite_or_wrong_order():
    row = passing_rows()[0]
    assert validate_measurement_row(row) == row

    broken = copy.deepcopy(row)
    broken["candidate_critical_ns"] = float("nan")
    with pytest.raises(ValueError, match="candidate_critical_ns"):
        validate_measurement_row(broken)

    broken = copy.deepcopy(row)
    broken["arm_order"] = ["candidate", "baseline"]
    with pytest.raises(ValueError, match="arm_order"):
        validate_measurement_row(broken)
