from __future__ import annotations

import copy
import math

import pytest

from tools.quantized_draft_int4_microgate import (
    DraftLinearShape,
    QuantizedDraftInt4Policy,
    classify_int4_microgate,
    validate_shape_manifest,
)


def _shape_payload():
    return {
        "schema_version": 1,
        "shapes": [
            {
                "shape_id": "m1_k1024_n2048_g128",
                "input_features": 1024,
                "output_features": 2048,
                "execution_count": 28,
                "group_size": 128,
            },
            {
                "shape_id": "m1_k1024_n3072_g128",
                "input_features": 1024,
                "output_features": 3072,
                "execution_count": 14,
                "group_size": 128,
            },
            {
                "shape_id": "m4_k3072_n1024_g128",
                "input_features": 3072,
                "output_features": 1024,
                "execution_count": 14,
                "group_size": 128,
            },
        ],
    }


def _passing_shapes():
    return validate_shape_manifest(_shape_payload())


def _passing_rows():
    rows = []
    for shape in _passing_shapes():
        for pair_index in range(200):
            rows.append({
                "shape_id": shape.shape_id,
                "pair_index": pair_index,
                "arm_order": (
                    ["bf16", "dequant", "fused_int4"]
                    if pair_index % 2 == 0
                    else ["fused_int4", "dequant", "bf16"]
                ),
                "bf16_cuda_ns": 100_000,
                "dequant_cuda_ns": 110_000,
                "fused_int4_cuda_ns": 70_000,
                "bf16_host_submission_ns": 10_000,
                "dequant_host_submission_ns": 12_000,
                "fused_int4_host_submission_ns": 11_000,
                "maximum_absolute_error": 0.01,
                "maximum_relative_error": 0.01,
                "fallback_reason": None,
                "full_dequant_allocation_observed": False,
            })
    return rows


def _expected_weight_bytes(shapes):
    bf16 = sum(
        shape.output_features
        * shape.input_features
        * 2
        * shape.execution_count
        for shape in shapes
    )
    packed = sum(
        (
            shape.output_features * (shape.input_features // 2)
            + shape.output_features
            * (shape.input_features // shape.group_size)
            * 4
        )
        * shape.execution_count
        for shape in shapes
    )
    return bf16, packed


def _passing_memory():
    shapes = _passing_shapes()
    bf16, packed = _expected_weight_bytes(shapes)
    return {
        "classification": "PASS",
        "observed_bf16_weight_bytes": bf16,
        "observed_candidate_weight_bytes": packed,
        "minimum_packed_weight_bytes": packed,
        "maximum_candidate_allocated_delta_bytes": 0,
        "full_dequant_allocation_observed": False,
    }


def _passing_graph():
    return {
        "classification": "PASS",
        "shapes": [
            {
                "shape_id": shape.shape_id,
                "capture_succeeded": True,
                "replay_count": 2,
                "static_pointers_stable": True,
                "maximum_absolute_error": 0.01,
                "maximum_relative_error": 0.01,
            }
            for shape in _passing_shapes()
        ],
    }


def _passing_cleanup():
    return {"classification": "CLEAN"}


def test_policy_freezes_stage0_thresholds():
    policy = QuantizedDraftInt4Policy()

    assert policy.minimum_pairs_per_shape == 200
    assert policy.maximum_candidate_to_bf16_median_ratio == 0.75
    assert policy.maximum_candidate_to_bf16_p99_ratio == 0.95
    assert policy.maximum_weight_bytes_ratio == 0.40
    assert policy.maximum_absolute_error == 0.08
    assert policy.maximum_relative_error == 0.08


def test_shape_manifest_requires_unique_positive_aligned_shapes():
    shapes = validate_shape_manifest(_shape_payload())

    assert shapes == (
        DraftLinearShape(
            shape_id="m1_k1024_n2048_g128",
            input_features=1024,
            output_features=2048,
            execution_count=28,
            group_size=128,
        ),
        DraftLinearShape(
            shape_id="m1_k1024_n3072_g128",
            input_features=1024,
            output_features=3072,
            execution_count=14,
            group_size=128,
        ),
        DraftLinearShape(
            shape_id="m4_k3072_n1024_g128",
            input_features=3072,
            output_features=1024,
            execution_count=14,
            group_size=128,
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "empty",
        "duplicate",
        "shape_id",
        "input_features",
        "output_features",
        "execution_count",
        "group_size",
        "group_alignment",
        "packed_alignment",
    ),
)
def test_shape_manifest_rejects_invalid_inventory(mutation):
    payload = _shape_payload()
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "empty":
        payload["shapes"] = []
    elif mutation == "duplicate":
        payload["shapes"].append(copy.deepcopy(payload["shapes"][0]))
    elif mutation == "shape_id":
        payload["shapes"][0]["shape_id"] = ""
    elif mutation == "input_features":
        payload["shapes"][0]["input_features"] = 0
    elif mutation == "output_features":
        payload["shapes"][0]["output_features"] = True
    elif mutation == "execution_count":
        payload["shapes"][0]["execution_count"] = -1
    elif mutation == "group_size":
        payload["shapes"][0]["group_size"] = 0
    elif mutation == "group_alignment":
        payload["shapes"][0]["input_features"] = 1000
    elif mutation == "packed_alignment":
        payload["shapes"][0]["input_features"] = 1025
        payload["shapes"][0]["group_size"] = 5
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        validate_shape_manifest(payload)


def test_classifier_accepts_complete_profitable_evidence():
    result = classify_int4_microgate(
        shapes=_passing_shapes(),
        rows=_passing_rows(),
        memory=_passing_memory(),
        graph=_passing_graph(),
        cleanup=_passing_cleanup(),
    )

    assert result["classification"] == "GO_FUSED_INT4_DRAFT_KERNEL"
    assert [
        row["shape_id"] for row in result["shape_summaries"]
    ] == [shape.shape_id for shape in _passing_shapes()]
    assert all(
        row["pair_count"] == 200
        for row in result["shape_summaries"]
    )
    assert result["weighted_summary"][
        "candidate_to_bf16_median_ratio"
    ] == pytest.approx(0.70)
    assert result["memory_summary"]["weight_bytes_ratio"] < 0.40


def _mutated_evidence(mutation):
    rows = _passing_rows()
    memory = _passing_memory()
    graph = _passing_graph()
    cleanup = _passing_cleanup()
    first_shape = _passing_shapes()[0].shape_id
    if mutation == "error":
        rows[0]["maximum_absolute_error"] = 0.081
    elif mutation == "missing_pair":
        rows.pop()
    elif mutation == "duplicate_pair":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "nonfinite":
        rows[0]["fused_int4_cuda_ns"] = math.nan
    elif mutation == "full_dequant":
        rows[0]["full_dequant_allocation_observed"] = True
    elif mutation == "weight_bytes":
        memory["observed_candidate_weight_bytes"] = math.ceil(
            memory["observed_bf16_weight_bytes"] * 0.41
        )
    elif mutation == "graph":
        graph["classification"] = "FAIL"
    elif mutation == "median":
        for row in rows:
            if row["shape_id"] == first_shape:
                row["fused_int4_cuda_ns"] = 80_000
    elif mutation == "p99":
        for row in rows:
            if (
                row["shape_id"] == first_shape
                and row["pair_index"] >= 197
            ):
                row["fused_int4_cuda_ns"] = 100_000
    elif mutation == "cleanup":
        cleanup["classification"] = "DIRTY"
    else:
        raise AssertionError(mutation)
    return rows, memory, graph, cleanup


@pytest.mark.parametrize(
    ("mutation", "classification"),
    (
        ("error", "NO_GO_CORRECTNESS"),
        ("missing_pair", "INCONCLUSIVE_EVIDENCE"),
        ("duplicate_pair", "INCONCLUSIVE_EVIDENCE"),
        ("nonfinite", "INCONCLUSIVE_EVIDENCE"),
        ("full_dequant", "INCONCLUSIVE_EVIDENCE"),
        ("weight_bytes", "NO_GO_MEMORY"),
        ("graph", "NO_GO_GRAPH"),
        ("median", "NO_GO_PERFORMANCE"),
        ("p99", "NO_GO_PERFORMANCE"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_classifier_fails_closed(mutation, classification):
    rows, memory, graph, cleanup = _mutated_evidence(mutation)

    result = classify_int4_microgate(
        shapes=_passing_shapes(),
        rows=rows,
        memory=memory,
        graph=graph,
        cleanup=cleanup,
    )

    assert result["classification"] == classification


def test_correctness_has_precedence_over_other_failures():
    rows, memory, graph, cleanup = _mutated_evidence("error")
    memory["observed_candidate_weight_bytes"] = (
        memory["observed_bf16_weight_bytes"]
    )
    graph["classification"] = "FAIL"
    cleanup["classification"] = "DIRTY"

    result = classify_int4_microgate(
        shapes=_passing_shapes(),
        rows=rows,
        memory=memory,
        graph=graph,
        cleanup=cleanup,
    )

    assert result["classification"] == "NO_GO_CORRECTNESS"


def test_incomplete_evidence_has_precedence_over_cost_failures():
    rows, memory, graph, cleanup = _mutated_evidence("missing_pair")
    memory["observed_candidate_weight_bytes"] = (
        memory["observed_bf16_weight_bytes"]
    )
    graph["classification"] = "FAIL"

    result = classify_int4_microgate(
        shapes=_passing_shapes(),
        rows=rows,
        memory=memory,
        graph=graph,
        cleanup=cleanup,
    )

    assert result["classification"] == "INCONCLUSIVE_EVIDENCE"
