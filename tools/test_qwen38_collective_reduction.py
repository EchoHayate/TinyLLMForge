from __future__ import annotations

import pytest

from tools.qwen38_collective_reduction import (
    build_consumer_dependency_proofs,
    build_qwen38_static_collective_catalog,
    classify_collective_reduction,
    estimate_reduction_ceiling,
    select_event_budget,
    validate_collective_census,
)


def _profile():
    return {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "vocab_size": 248320,
        "dtype": "bfloat16",
    }


def _catalog():
    return build_qwen38_static_collective_catalog(
        _profile(),
        tensor_parallel_size=4,
    )


def _rank_snapshot(rank):
    rows = []
    for ordinal, site in enumerate(_catalog()):
        rows.append({
            "collective_ordinal": ordinal,
            "site_id": site["site_id"],
            "collective_kind": site["collective_kind"],
            "tensor_shape": [1, 5120],
            "tensor_dtype": "torch.bfloat16",
            "tensor_bytes": 10240,
        })
    return {
        "schema": "tinyllmforge.synchronous-collective-census.v1",
        "rank": rank,
        "enabled": True,
        "finalization_status": "complete",
        "steps": [{"decode_ordinal": 0, "collective_count": 130}],
        "collectives": rows,
    }


def _four_rank_rows():
    return [_rank_snapshot(rank) for rank in range(4)]


def _calibration_rows(values):
    return [
        {
            "budget": budget,
            "median_overhead_ratio": pair[0],
            "maximum_overhead_ratio": pair[1],
        }
        for budget, pair in values.items()
    ]


def test_qwen38_catalog_contains_exactly_130_decode_sites():
    catalog = _catalog()

    assert len(catalog) == 130
    assert catalog[0]["site_role"] == "vocab_parallel_embedding"
    assert sum(
        row["site_role"] == "row_parallel_output"
        for row in catalog
    ) == 128
    assert catalog[-1]["site_role"] == "greedy_token_broadcast"
    assert [row["site_id"] for row in catalog[:4]] == [
        "embedding.input",
        "layer.000.attention.output",
        "layer.000.mlp.output",
        "layer.001.attention.output",
    ]


def test_catalog_assigns_conservative_consumer_classes():
    catalog = _catalog()
    row_parallel = [
        row for row in catalog
        if row["site_role"] == "row_parallel_output"
    ]

    assert {
        row["classification"] for row in row_parallel
    } == {"MANDATORY_IMMEDIATE_CONSUMER"}
    assert catalog[0]["classification"] == "MATERIALIZATION_ALTERNATIVE"
    assert (
        catalog[-1]["classification"]
        == "MANDATORY_IMMEDIATE_CONSUMER"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_hidden_layers", 63),
        ("hidden_size", 4096),
        ("vocab_size", 248319),
        ("dtype", "float16"),
    ),
)
def test_catalog_rejects_non_frozen_profile(field, value):
    profile = _profile()
    profile[field] = value
    with pytest.raises(ValueError, match=field):
        build_qwen38_static_collective_catalog(
            profile,
            tensor_parallel_size=4,
        )


def test_catalog_rejects_non_tp4():
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        build_qwen38_static_collective_catalog(
            _profile(),
            tensor_parallel_size=2,
        )


def test_census_requires_four_identical_rank_sequences():
    rows = _four_rank_rows()
    result = validate_collective_census(rows, _catalog())

    assert result["rank_inventory"] == [0, 1, 2, 3]
    assert result["collective_count_per_rank"] == 130
    rows[-1]["collectives"][7]["tensor_bytes"] += 2
    with pytest.raises(ValueError, match="rank collective sequence"):
        validate_collective_census(rows, _catalog())


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_census_rejects_incomplete_or_duplicate_sites(mutation):
    rows = _four_rank_rows()
    if mutation == "missing":
        rows[0]["collectives"].pop()
    elif mutation == "extra":
        rows[0]["collectives"].append(
            dict(rows[0]["collectives"][-1])
        )
    else:
        rows[0]["collectives"][1]["site_id"] = (
            rows[0]["collectives"][0]["site_id"]
        )
    with pytest.raises(ValueError, match="collective sequence"):
        validate_collective_census(rows, _catalog())


def test_select_event_budget_chooses_largest_passing_budget():
    rows = _calibration_rows({
        0: (0.01, 0.02),
        8: (0.02, 0.03),
        16: (0.03, 0.05),
        32: (0.04, 0.06),
    })
    assert select_event_budget(rows) == 16


def test_event_budget_thresholds_and_count_only_gate():
    exact = _calibration_rows({
        0: (0.03, 0.05),
        8: (0.03, 0.05),
        16: (0.031, 0.05),
        32: (0.03, 0.051),
    })
    assert select_event_budget(exact) == 8

    no_timed = _calibration_rows({
        0: (0.03, 0.05),
        8: (0.031, 0.05),
        16: (0.03, 0.051),
        32: (0.04, 0.06),
    })
    assert select_event_budget(no_timed) is None

    invalid_count_only = _calibration_rows({
        0: (0.031, 0.05),
        8: (0.01, 0.01),
        16: (0.01, 0.01),
        32: (0.01, 0.01),
    })
    with pytest.raises(ValueError, match="count-only"):
        select_event_budget(invalid_count_only)


def test_consumer_proofs_are_conservative_and_named():
    proofs = build_consumer_dependency_proofs(_catalog())

    assert len(proofs) == 130
    embedding = proofs[0]
    assert embedding["candidate_id"] == "replicate_embedding"
    assert embedding["status"] == "PASS"
    assert embedding["additional_persistent_device_bytes_per_rank"] > 0
    assert {
        row["status"]
        for row in proofs
        if row["site_role"] == "row_parallel_output"
    } == {"FAIL_IMMEDIATE_CONSUMER"}


def test_reduction_ceiling_subtracts_replacement_and_uncertainty():
    result = estimate_reduction_ceiling(
        {"coverage_complete": True},
        {
            "replicate_embedding": {
                "sampled_collective_cuda_ns": 80,
                "profiler_uncertainty_ns": 10,
            },
        },
        [{
            "candidate_id": "replicate_embedding",
            "site_role": "vocab_parallel_embedding",
            "status": "PASS",
            "calls_removed_per_decode_step": 1,
            "bytes_removed_per_decode_step": 10240,
            "replacement_cost_ns": 20,
            "additional_persistent_device_bytes_per_rank": 1_904_640_000,
            "additional_peak_device_bytes_per_rank": 1_904_640_000,
            "unsupported_topologies": ["tp!=4"],
        }],
        {"median_tpot_ns": 1000, "workloads": ["P0", "P1"]},
    )

    candidate = result["candidates"][0]
    assert candidate["lower_bound_tpot_reduction_ns"] == 50
    assert candidate["upper_bound_tpot_reduction_ns"] == 80
    assert candidate["lower_bound_tpot_opportunity_ratio"] == 0.05
    assert candidate["upper_bound_tpot_opportunity_ratio"] == 0.08


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"correctness_pass": False}, "INVALID_CORRECTNESS"),
        ({"resource_identity_pass": False}, "INVALID_RESOURCE_IDENTITY"),
        ({"coverage_complete": False}, "INCONCLUSIVE_INCOMPLETE_COVERAGE"),
        ({"profiler_overhead_pass": False}, "INCONCLUSIVE_PROFILER_OVERHEAD"),
        ({}, "GO_SYNC_COLLECTIVE_REDUCTION"),
        (
            {
                "candidates": [{
                    "candidate_id": "replicate_embedding",
                    "proof_status": "PASS",
                    "lower_bound_tpot_opportunity_ratio": 0.049999,
                }],
            },
            "NO_GO_NO_REDUCIBLE_COLLECTIVE",
        ),
    ),
)
def test_terminal_classification_precedence(overrides, expected):
    summary = {
        "correctness_pass": True,
        "resource_identity_pass": True,
        "coverage_complete": True,
        "profiler_overhead_pass": True,
        "candidates": [{
            "candidate_id": "replicate_embedding",
            "proof_status": "PASS",
            "lower_bound_tpot_opportunity_ratio": 0.05,
        }],
    }
    summary.update(overrides)
    assert classify_collective_reduction(summary) == expected
