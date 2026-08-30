from __future__ import annotations

import copy
import json

import pytest

from tools.assemble_qwen38_tp4_peer_reduction_microgate import (
    PRODUCER_ARTIFACTS,
    assemble_bundle,
)


ATTEMPT = "20260830-qwen38-tp4-peer-reduction-r1"
SOURCE_REVISION = "a" * 40


def _topology_rows():
    return [
        {
            "source_rank": source,
            "destination_rank": destination,
            "can_access": True,
            "ipc_roundtrip": True,
        }
        for source in range(4)
        for destination in range(4)
        if source != destination
    ]


def _microgate_rows():
    rows = []
    for active_tokens in (1, 4, 8):
        for pair_index in range(200):
            for rank in range(4):
                baseline = 100_000 + pair_index + rank
                candidate = (
                    int(baseline * 0.80)
                    if active_tokens in (1, 4)
                    else int(baseline * 0.99)
                )
                rows.append({
                    "attempt": ATTEMPT,
                    "source_revision": SOURCE_REVISION,
                    "active_tokens": active_tokens,
                    "pair_index": pair_index,
                    "rank": rank,
                    "arm_order": (
                        ["baseline", "candidate"]
                        if pair_index % 2 == 0
                        else ["candidate", "baseline"]
                    ),
                    "baseline_cuda_ns": baseline,
                    "candidate_cuda_ns": candidate,
                    "baseline_host_submission_ns": 20_000,
                    "candidate_host_submission_ns": 15_000,
                    "cross_rank_max_abs_error": 0.0,
                    "cross_rank_max_rel_error": 0.0,
                    "baseline_max_abs_error": 0.0,
                    "baseline_max_rel_error": 0.0,
                    "timed_out": False,
                    "device_status": 0,
                })
    return rows


def _inputs():
    topology = _topology_rows()
    return {
        "source_identity": {
            "schema_version": "qwen38.tp4-peer-reduction-source.v1",
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": "b" * 64,
        },
        "peer_access_matrix": {
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "world_size": 4,
            "rows": copy.deepcopy(topology),
        },
        "ipc_roundtrip_rows": copy.deepcopy(topology),
        "microgate_rows": _microgate_rows(),
        "memory_summary": {
            "maximum_allocated_delta_bytes": 32 * 1024 * 1024,
            "rank_rows": [
                {"rank": rank, "allocated_delta_bytes": 32 * 1024 * 1024}
                for rank in range(4)
            ],
        },
        "cleanup": {
            "classification": "CLEAN",
            "rank_rows": [
                {
                    "rank": rank,
                    "peer_group_closed": True,
                    "timed_out": False,
                }
                for rank in range(4)
            ],
            "owned_children_remaining": [],
            "exact_tag_scans": [[], [], []],
        },
    }


def test_assembler_writes_compact_manifested_pass_bundle(tmp_path):
    result = assemble_bundle(output_root=tmp_path, **_inputs())

    assert result["classification"] == "PASS"
    assert {path.name for path in tmp_path.iterdir()} == set(
        PRODUCER_ARTIFACTS
    )
    summary = json.loads(
        (tmp_path / "microgate_summary.json").read_text()
    )
    assert [row["active_tokens"] for row in summary["shape_summaries"]] == [
        1,
        4,
        8,
    ]
    assert all(row["pair_count"] == 200 for row in summary["shape_summaries"])


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("topology", "INELIGIBLE_TOPOLOGY"),
        ("coverage", "INCONCLUSIVE_EVIDENCE"),
        ("correctness", "NO_GO_CORRECTNESS"),
        ("median", "NO_GO_MICROGATE"),
        ("p99", "NO_GO_MICROGATE"),
        ("timeout", "NO_GO_MICROGATE"),
        ("memory", "NO_GO_MEMORY"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_assembler_classifies_each_failure_closed(
    tmp_path,
    mutation,
    expected,
):
    inputs = _inputs()
    if mutation == "topology":
        inputs["peer_access_matrix"]["rows"][-1]["can_access"] = False
    elif mutation == "coverage":
        inputs["microgate_rows"] = inputs["microgate_rows"][:-4]
    elif mutation == "correctness":
        inputs["microgate_rows"][0]["baseline_max_abs_error"] = 1.0
    elif mutation == "median":
        for row in inputs["microgate_rows"]:
            if row["active_tokens"] == 1:
                row["candidate_cuda_ns"] = row["baseline_cuda_ns"]
    elif mutation == "p99":
        for row in inputs["microgate_rows"]:
            if (
                row["active_tokens"] == 4
                and row["pair_index"] >= 197
            ):
                row["candidate_cuda_ns"] = row["baseline_cuda_ns"] * 2
    elif mutation == "timeout":
        inputs["microgate_rows"][0]["timed_out"] = True
    elif mutation == "memory":
        inputs["memory_summary"][
            "maximum_allocated_delta_bytes"
        ] = 49 * 1024 * 1024
    elif mutation == "cleanup":
        inputs["cleanup"]["classification"] = "DIRTY"

    result = assemble_bundle(output_root=tmp_path, **inputs)

    assert result["classification"] == expected
