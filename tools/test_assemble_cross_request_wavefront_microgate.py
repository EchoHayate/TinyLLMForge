from __future__ import annotations

import copy
import json

import pytest

from tools.assemble_cross_request_wavefront_microgate import (
    PRODUCER_ARTIFACTS,
    _load_json,
    assemble_bundle,
)


ATTEMPT = "20260831-cross-request-wavefront-stage0-r1"
SOURCE_REVISION = "a" * 40
SOURCE_TREE_SHA256 = "b" * 64


def _microgate_rows():
    return [
        {
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "active_tokens": active_tokens,
            "pair_index": pair_index,
            "rank": rank,
            "arm_order": (
                ["baseline", "candidate"]
                if pair_index % 2 == 0
                else ["candidate", "baseline"]
            ),
            "cohort_digest": (
                "c" * 64 if active_tokens == 4 else "d" * 64
            ),
            "collective_order_digest": "e" * 64,
            "candidate_output_digest": "f" * 64,
            "rank_output_digests": ["f" * 64] * 4,
            "baseline_cuda_ns": 100_000,
            "candidate_cuda_ns": 90_000,
            "baseline_host_submission_ns": 20_000,
            "candidate_host_submission_ns": 21_000,
            "candidate_communication_union_ns": 40_000,
            "candidate_realized_overlap_ns": 12_000,
            "cross_rank_max_abs_error": 0.0,
            "cross_rank_max_rel_error": 0.0,
            "baseline_max_abs_error": 0.0,
            "baseline_max_rel_error": 0.0,
            "nan_count": 0,
            "inf_count": 0,
            "timed_out": False,
        }
        for active_tokens in (4, 8)
        for pair_index in range(300)
        for rank in range(4)
    ]


def _inputs():
    return {
        "source_identity": {
            "schema_version": "cross-request-wavefront-source.v1",
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
        },
        "runtime_capabilities": {
            "schema_version": (
                "cross-request-wavefront-runtime-capabilities.v1"
            ),
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "rank_rows": [
                {
                    "rank": rank,
                    "world_size": 4,
                    "local_input_size": 1536,
                    "hidden_size": 5120,
                }
                for rank in range(4)
            ],
        },
        "cohort_policy": {
            "schema_version": "cross-request-wavefront-cohort-policy.v1",
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "active_token_groups": [4, 8],
            "cohort_digests": {
                "4": "c" * 64,
                "8": "d" * 64,
            },
            "collective_order_digest": "e" * 64,
        },
        "rows": _microgate_rows(),
        "memory": {
            "maximum_allocated_delta_bytes": 64 * 1024 * 1024,
            "maximum_reserved_delta_bytes": 80 * 1024 * 1024,
            "rank_shape_rows": [],
        },
        "cleanup": {
            "classification": "CLEAN",
            "rank_rows": [
                {
                    "rank": rank,
                    "streams_released": True,
                    "events_released": True,
                    "timed_out": False,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
            "owned_children_remaining": [],
            "exact_tag_scans": [[], [], []],
        },
    }


def test_assembler_writes_nine_file_manifested_go_bundle(tmp_path):
    result = assemble_bundle(output_root=tmp_path, **_inputs())

    assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
    assert {path.name for path in tmp_path.iterdir()} == set(
        PRODUCER_ARTIFACTS
    )
    assert json.loads(
        (tmp_path / "classification.json").read_text()
    ) == {
        "schema_version": "cross-request-wavefront-classification.v1",
        "classification": "GO_WAVEFRONT_MICROGATE",
        "runtime_integration_authorized": True,
    }
    summary = json.loads(
        (tmp_path / "microgate_summary.json").read_text()
    )
    assert summary["measurement_row_count"] == 2400
    assert [row["active_tokens"] for row in summary["shape_summaries"]] == [
        4,
        8,
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("coverage", "INCONCLUSIVE_EVIDENCE"),
        ("correctness", "NO_GO_CORRECTNESS"),
        ("memory", "NO_GO_MEMORY"),
        ("tail", "NO_GO_PERFORMANCE"),
        ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
        ("fragmentation", "NO_GO_GEMM_FRAGMENTATION"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_assembler_classifies_each_failure_closed(
    tmp_path,
    mutation,
    expected,
):
    inputs = _inputs()
    if mutation == "coverage":
        inputs["rows"].pop()
    elif mutation == "correctness":
        inputs["rows"][0]["baseline_max_abs_error"] = 1.0
        inputs["rows"][0]["baseline_max_rel_error"] = 1.0
    elif mutation == "memory":
        inputs["memory"][
            "maximum_allocated_delta_bytes"
        ] = 128 * 1024 * 1024 + 1
    elif mutation == "tail":
        for row in inputs["rows"]:
            if row["pair_index"] >= 296:
                row["candidate_cuda_ns"] = 110_000
    elif mutation == "overlap":
        for row in inputs["rows"]:
            if row["active_tokens"] == 4:
                row["candidate_realized_overlap_ns"] = 4_000
    elif mutation == "fragmentation":
        for row in inputs["rows"]:
            row["candidate_cuda_ns"] = 98_000
    elif mutation == "cleanup":
        inputs["cleanup"]["classification"] = "DIRTY"

    result = assemble_bundle(output_root=tmp_path, **inputs)

    assert result["classification"] == expected


def test_assembler_rejects_nonempty_output_and_identity_drift(tmp_path):
    (tmp_path / "existing").write_text("occupied")
    with pytest.raises(ValueError, match="must be empty"):
        assemble_bundle(output_root=tmp_path, **_inputs())

    fresh = tmp_path / "fresh"
    inputs = _inputs()
    inputs["rows"][0]["attempt"] = "different-attempt"
    with pytest.raises(ValueError, match="identity"):
        assemble_bundle(output_root=fresh, **inputs)

    policy_drift = tmp_path / "policy-drift"
    inputs = _inputs()
    inputs["rows"][0]["cohort_digest"] = "0" * 64
    with pytest.raises(ValueError, match="policy identity"):
        assemble_bundle(output_root=policy_drift, **inputs)


def test_assembler_rejects_invalid_source_and_nonfinite_evidence(tmp_path):
    inputs = _inputs()
    inputs["source_identity"]["source_revision"] = "A" * 40
    with pytest.raises(ValueError, match="source identity"):
        assemble_bundle(output_root=tmp_path / "source", **inputs)

    inputs = _inputs()
    inputs["memory"]["maximum_reserved_delta_bytes"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assemble_bundle(output_root=tmp_path / "finite", **inputs)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"attempt":"a","attempt":"b"}\n')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_json(path)


def clone_inputs():
    return copy.deepcopy(_inputs())
