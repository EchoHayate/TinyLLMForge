from __future__ import annotations

import math

import pytest

from tools.qwen38_tp4_peer_reduction import (
    PeerReductionPolicy,
    classify_peer_microgate,
    validate_peer_topology,
)


def _passing_topology_rows():
    return [
        {
            "source_rank": source_rank,
            "destination_rank": destination_rank,
            "can_access": True,
            "ipc_roundtrip": True,
        }
        for source_rank in range(4)
        for destination_rank in range(4)
        if source_rank != destination_rank
    ]


def _passing_microgate_rows():
    return [
        {
            "active_tokens": active_tokens,
            "pair_index": pair_index,
            "rank": rank,
            "baseline_cuda_ns": 100_000,
            "candidate_cuda_ns": 80_000,
            "cross_rank_max_abs_error": 0.0,
            "cross_rank_max_rel_error": 0.0,
            "baseline_max_abs_error": 0.0,
            "baseline_max_rel_error": 0.0,
            "timed_out": False,
        }
        for active_tokens in (1, 4, 8)
        for pair_index in range(200)
        for rank in range(4)
    ]


def _passing_cleanup():
    return {"classification": "CLEAN"}


def _passing_memory():
    return {
        "maximum_allocated_delta_bytes": 48 * 1024 * 1024,
    }


def _mutated_case(mutation):
    rows = _passing_microgate_rows()
    cleanup = _passing_cleanup()
    memory = _passing_memory()
    if mutation == "correctness":
        rows[0]["baseline_max_abs_error"] = 0.021
    elif mutation == "median":
        for row in rows:
            if row["active_tokens"] == 4:
                row["candidate_cuda_ns"] = 95_000
    elif mutation == "p99":
        for row in rows:
            if (
                row["active_tokens"] == 1
                and row["pair_index"] >= 196
            ):
                row["candidate_cuda_ns"] = 110_000
    elif mutation == "timeout":
        rows[0]["timed_out"] = True
    elif mutation == "memory":
        memory["maximum_allocated_delta_bytes"] += 1
    elif mutation == "cleanup":
        cleanup["classification"] = "DIRTY"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    return rows, cleanup, memory


def test_policy_freezes_supported_shape_and_ring():
    policy = PeerReductionPolicy()

    assert policy.world_size == 4
    assert policy.hidden_size == 5120
    assert policy.max_active_tokens == 8
    assert policy.slot_ring_size == 2
    assert (
        policy.maximum_allocated_delta_bytes
        == 48 * 1024 * 1024
    )


def test_topology_requires_all_twelve_directed_peer_edges():
    rows = _passing_topology_rows()

    assert validate_peer_topology(rows)["classification"] == "PASS"
    rows[-1]["can_access"] = False
    with pytest.raises(ValueError, match="peer topology"):
        validate_peer_topology(rows)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "self"))
def test_topology_rejects_invalid_edge_inventory(mutation):
    rows = _passing_topology_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[-1]["destination_rank"] = rows[-1]["source_rank"]

    with pytest.raises(ValueError, match="peer topology"):
        validate_peer_topology(rows)


def test_microgate_requires_benefit_cost_correctness_and_cleanup():
    result = classify_peer_microgate(
        rows=_passing_microgate_rows(),
        cleanup=_passing_cleanup(),
        memory=_passing_memory(),
    )

    assert result["classification"] == "PASS"
    assert [
        row["active_tokens"] for row in result["shape_summaries"]
    ] == [1, 4, 8]
    assert all(
        row["pair_count"] == 200
        for row in result["shape_summaries"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("correctness", "NO_GO_CORRECTNESS"),
        ("median", "NO_GO_MICROGATE"),
        ("p99", "NO_GO_MICROGATE"),
        ("timeout", "NO_GO_MICROGATE"),
        ("memory", "NO_GO_MEMORY"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_microgate_fails_closed(mutation, expected):
    rows, cleanup, memory = _mutated_case(mutation)

    assert classify_peer_microgate(
        rows=rows,
        cleanup=cleanup,
        memory=memory,
    )["classification"] == expected


@pytest.mark.parametrize(
    "mutation",
    ("missing_group", "short_group", "missing_rank", "duplicate"),
)
def test_microgate_rejects_incomplete_pair_inventory(mutation):
    rows = _passing_microgate_rows()
    if mutation == "missing_group":
        rows = [row for row in rows if row["active_tokens"] != 4]
    elif mutation == "short_group":
        rows = [
            row
            for row in rows
            if not (
                row["active_tokens"] == 4
                and row["pair_index"] == 199
            )
        ]
    elif mutation == "missing_rank":
        rows = [
            row
            for row in rows
            if not (
                row["active_tokens"] == 4
                and row["pair_index"] == 0
                and row["rank"] == 3
            )
        ]
    else:
        rows.append(dict(rows[0]))

    result = classify_peer_microgate(
        rows=rows,
        cleanup=_passing_cleanup(),
        memory=_passing_memory(),
    )

    assert result["classification"] == "INCONCLUSIVE_EVIDENCE"


def test_microgate_rejects_nonfinite_metrics():
    rows = _passing_microgate_rows()
    rows[0]["candidate_cuda_ns"] = math.nan

    result = classify_peer_microgate(
        rows=rows,
        cleanup=_passing_cleanup(),
        memory=_passing_memory(),
    )

    assert result["classification"] == "INCONCLUSIVE_EVIDENCE"


def test_microgate_classification_precedence_is_fail_closed():
    rows, cleanup, memory = _mutated_case("correctness")
    cleanup["classification"] = "DIRTY"
    memory["maximum_allocated_delta_bytes"] += 1

    result = classify_peer_microgate(
        rows=rows,
        cleanup=cleanup,
        memory=memory,
    )

    assert result["classification"] == "NO_GO_CORRECTNESS"
