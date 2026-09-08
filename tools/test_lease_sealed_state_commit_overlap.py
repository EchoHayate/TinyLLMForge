from __future__ import annotations

import copy

import pytest

from tools.lease_sealed_state_commit_overlap import (
    ACTIVE_TOKEN_GROUPS,
    DIAGNOSTIC_ITERATION_COUNT,
    MEASURED_PAIR_COUNT,
    WORLD_SIZE,
    classify_stage01,
    classify_stage0,
    interval_intersection_ns,
    validate_stage01_diagnostic_row,
    validate_stage01_measurement_row,
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


def passing_stage01_rows():
    return [
        {
            "attempt": "20260908-completion-owned-stage01-r1",
            "source_revision": "c" * 40,
            "source_tree_sha256": "d" * 64,
            "active_tokens": active_tokens,
            "pair_index": pair_index,
            "rank": rank,
            "arm_order": (
                ["baseline", "completion_owned"]
                if pair_index % 2 == 0
                else ["completion_owned", "baseline"]
            ),
            "baseline_critical_ns": 100_000,
            "candidate_critical_ns": 90_000,
            "baseline_host_submission_ns": 20_000,
            "candidate_host_submission_ns": 20_400,
            "collective_outstanding_window_ns": [10_000, 60_000],
            "side_effect_window_ns": [35_000, 75_000],
            "overlap_intersection_ns": 25_000,
            "expected_reduced_exact": True,
            "baseline_reduced_exact": True,
            "candidate_reduced_exact": True,
            "baseline_final_exact": True,
            "candidate_final_exact": True,
            "baseline_candidate_exact": True,
            "shadow_payload_exact": True,
            "active_state_preserved_before_publish": True,
            "published_state_exact": True,
            "abort_preserved_old_state": True,
            "commit_identity_match": True,
            "collective_wait_invoked": True,
            "collective_dependency_transferred": True,
            "side_effect_dependency_joined": True,
            "finite_output": True,
            "timed_path_allocation_count": 0,
            "timed_out": False,
        }
        for active_tokens in ACTIVE_TOKEN_GROUPS
        for pair_index in range(MEASURED_PAIR_COUNT)
        for rank in range(WORLD_SIZE)
    ]


def passing_stage01_diagnostics():
    rows = [
        {
            "attempt": "20260908-completion-owned-stage01-r1",
            "source_revision": "c" * 40,
            "source_tree_sha256": "d" * 64,
            "active_tokens": active_tokens,
            "diagnostic_index": diagnostic_index,
            "rank": rank,
            "baseline_reduced_exact": True,
            "baseline_final_exact": True,
            "completion_owned_reduced_exact": True,
            "completion_owned_final_exact": True,
            "event_only_reduced_exact": True,
            "event_only_final_exact": True,
        }
        for active_tokens in ACTIVE_TOKEN_GROUPS
        for diagnostic_index in range(DIAGNOSTIC_ITERATION_COUNT)
        for rank in range(WORLD_SIZE)
    ]
    rows[0]["event_only_reduced_exact"] = False
    rows[0]["event_only_final_exact"] = False
    return rows


def test_frozen_inventory_and_interval_math():
    assert ACTIVE_TOKEN_GROUPS == (1, 4, 8)
    assert MEASURED_PAIR_COUNT == 15
    assert WORLD_SIZE == 4
    assert interval_intersection_ns((10, 60), (35, 75)) == 25


def test_stage01_inventory_and_validators_are_frozen():
    assert DIAGNOSTIC_ITERATION_COUNT == 15
    rows = passing_stage01_rows()
    diagnostics = passing_stage01_diagnostics()

    assert len(rows) == 180
    assert len(diagnostics) == 180
    assert validate_stage01_measurement_row(rows[0]) == rows[0]
    assert validate_stage01_diagnostic_row(diagnostics[0]) == diagnostics[0]


def test_stage01_classifier_accepts_complete_profitable_evidence():
    result = classify_stage01(
        passing_stage01_rows(),
        passing_stage01_diagnostics(),
        passing_memory(),
        {"classification": "CLEAN"},
    )

    assert result["classification"] == (
        "GO_COMPLETION_OWNED_OVERLAP_MICROGATE"
    )
    assert result["stage1_authorized"] is True
    assert result["measurement_row_count"] == 180
    assert result["diagnostic_row_count"] == 180


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("candidate_correctness", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("wait_not_invoked", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("resource_identity", "NO_GO_RESOURCE_IDENTITY"),
        ("allocation", "NO_GO_MEMORY_OR_ALLOCATION"),
        ("diagnostic_missing", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
        (
            "diagnostic_not_reproduced",
            "INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED",
        ),
        ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
        ("median", "NO_GO_PERFORMANCE"),
        ("tail", "NO_GO_PERFORMANCE"),
        ("host", "NO_GO_PERFORMANCE"),
        ("direction", "NO_GO_PERFORMANCE"),
        ("cleanup", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
    ),
)
def test_stage01_classifier_precedence_and_fail_closed(mutation, expected):
    rows = passing_stage01_rows()
    diagnostics = passing_stage01_diagnostics()
    memory = passing_memory()
    cleanup = {"classification": "CLEAN"}
    resource_identity_valid = True

    if mutation == "candidate_correctness":
        rows[0]["candidate_final_exact"] = False
    elif mutation == "wait_not_invoked":
        rows[0]["collective_wait_invoked"] = False
    elif mutation == "resource_identity":
        resource_identity_valid = False
    elif mutation == "allocation":
        rows[0]["timed_path_allocation_count"] = 1
    elif mutation == "diagnostic_missing":
        diagnostics.pop()
    elif mutation == "diagnostic_not_reproduced":
        for row in diagnostics:
            row["event_only_reduced_exact"] = True
            row["event_only_final_exact"] = True
    elif mutation == "overlap":
        for row in rows:
            if row["active_tokens"] == 4:
                row["side_effect_window_ns"] = [58_000, 75_000]
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

    result = classify_stage01(
        rows,
        diagnostics,
        memory,
        cleanup,
        resource_identity_valid=resource_identity_valid,
    )
    assert result["classification"] == expected


def test_stage01_correctness_precedes_resource_and_measurement_failures():
    rows = passing_stage01_rows()
    diagnostics = passing_stage01_diagnostics()
    rows[0]["candidate_reduced_exact"] = False
    diagnostics.pop()

    result = classify_stage01(
        rows,
        diagnostics,
        passing_memory(),
        {"classification": "DIRTY"},
        resource_identity_valid=False,
    )

    assert result["classification"] == "NO_GO_CORRECTNESS_OR_LIFECYCLE"


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
