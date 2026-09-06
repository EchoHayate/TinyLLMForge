from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tinyvllm"
    / "engine"
    / "segmented_exact_cuda_graph.py"
)
assert MODULE_PATH.is_file(), "segmented exact graph contract is missing"
SPEC = importlib.util.spec_from_file_location(
    "segmented_exact_cuda_graph_for_census_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ExactGraphSegment = module.ExactGraphSegment
ExactGraphSegmentPlan = module.ExactGraphSegmentPlan
from verify_tp4_segmented_capture_census import verify_bundle


RANGES = {
    "p2": ((0, 32), (32, 64)),
    "p3": ((0, 22), (22, 43), (43, 64)),
    "p4": ((0, 16), (16, 32), (32, 48), (48, 64)),
}


def _plan(plan_id):
    ranges = RANGES[plan_id]
    return ExactGraphSegmentPlan(
        layer_count=64,
        segments=tuple(
            ExactGraphSegment(
                start_layer=start,
                end_layer=end,
                include_embedding=ordinal == 0,
                include_final=ordinal == len(ranges) - 1,
                include_commit=ordinal == len(ranges) - 1,
            )
            for ordinal, (start, end) in enumerate(ranges)
        ),
    )


def durations(segment_ns, lifecycle_ns):
    return {
        "segment_durations_ns": tuple(segment_ns),
        "lifecycle_duration_ns": lifecycle_ns,
    }


def make_bundle(
    *,
    plans,
    exact=True,
    state_exact=True,
    unselected=True,
    cleanup="CLEAN",
):
    run_tag = "census-r1"
    source = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-capture-source.v1"
        ),
        "run_tag": run_tag,
        "source_revision": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }
    complete_plans = {
        plan_id: durations(
            (1_000_000_000,) * len(ranges),
            4_600_000_000,
        )
        for plan_id, ranges in RANGES.items()
    }
    complete_plans.update(plans)
    rows = []
    for plan_id, timing in complete_plans.items():
        plan = _plan(plan_id)
        for rank in range(4):
            for ordinal, ((start, end), duration) in enumerate(
                zip(
                    RANGES[plan_id],
                    timing["segment_durations_ns"],
                    strict=True,
                )
            ):
                rows.append({
                    "row_id": (
                        f"{plan_id}:segment-{ordinal}:rank-{rank}"
                    ),
                    "plan_id": plan_id,
                    "plan_sha256": plan.sha256,
                    "plan_ranges": [
                        list(value) for value in RANGES[plan_id]
                    ],
                    "segment_ordinal": ordinal,
                    "start_layer": start,
                    "end_layer": end,
                    "include_embedding": ordinal == 0,
                    "include_final": (
                        ordinal == len(RANGES[plan_id]) - 1
                    ),
                    "include_commit": (
                        ordinal == len(RANGES[plan_id]) - 1
                    ),
                    "rank": rank,
                    "world_size": 4,
                    "capture_body_duration_ns": duration - 20,
                    "post_capture_sync_duration_ns": 10,
                    "segment_capture_duration_ns": duration,
                    "lifecycle_duration_ns": timing[
                        "lifecycle_duration_ns"
                    ],
                    "allocated_delta_bytes": 100,
                    "reserved_delta_bytes": 200,
                    "stable_boundary_buffer_bytes": 300,
                    "exact_output": exact,
                    "selected_state_exact": state_exact,
                    "unselected_state_unchanged": unselected,
                    "scratch_kv_restored": True,
                    "graph_reset": True,
                    "complete": True,
                })
    return {
        "schema_version": (
            "tinyllmforge.tp4-segmented-capture-census.v1"
        ),
        "run_tag": run_tag,
        "source_identity": source,
        "source_manifest": deepcopy(source),
        "launch_admission": {
            "schema_version": (
                "tinyllmforge.tp4-segmented-capture-admission.v1"
            ),
            "run_tag": run_tag,
            "admission_mode": "strict_clean",
            "strict_clean": True,
            "world_size": 4,
            "selected_gpus": [
                {
                    "rank": rank,
                    "index": rank,
                    "uuid": f"GPU-{rank}",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                    "compute_processes": [],
                }
                for rank in range(4)
            ],
        },
        "cleanup": {
            "schema_version": (
                "tinyllmforge.tp4-segmented-capture-cleanup.v1"
            ),
            "run_tag": run_tag,
            "classification": cleanup,
            "owned_children_remaining": [],
            "exact_tag_scans": [[], [], []],
            "rank_rows": [
                {
                    "rank": rank,
                    "exit_code": 0,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
        },
        "process_receipts": {
            "schema_version": (
                "tinyllmforge.tp4-segmented-capture-process.v1"
            ),
            "run_tag": run_tag,
            "plans": {
                plan_id: {
                    "rank_exit_codes": [0, 0, 0, 0],
                    "process_group_destroyed": True,
                    "owned_children_remaining": [],
                    "rank_cleanup_receipts": [
                        {
                            "rank": rank,
                            "process_group_destroyed": True,
                        }
                        for rank in range(4)
                    ],
                }
                for plan_id in RANGES
            },
        },
        "rows": rows,
    }


def test_verifier_selects_smallest_plan_with_headroom():
    bundle = make_bundle(
        plans={
            "p2": durations(
                (1_950_000_000, 1_700_000_000),
                4_100_000_000,
            ),
            "p3": durations(
                (1_300_000_000, 1_250_000_000, 1_200_000_000),
                4_200_000_000,
            ),
            "p4": durations(
                (980_000_000,) * 4,
                4_400_000_000,
            ),
        },
    )
    result = verify_bundle(bundle)
    assert result["classification"] == "GO_SEGMENT_PLAN_SELECTED"
    assert result["selected_plan_id"] == "p3"
    assert result["selected_plan_sha256"] == _plan("p3").sha256


def test_verifier_rejects_hidden_total_capture_cost():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_600_000_000,
            ),
        },
    )
    result = verify_bundle(bundle)
    assert result["classification"] == (
        "NO_GO_SEGMENTED_CAPTURE_CEILING"
    )
    assert result["selected_plan_id"] is None


def test_verifier_selects_a_passing_plan_when_another_plan_is_inexact():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    for row in bundle["rows"]:
        if row["plan_id"] == "p2":
            row["exact_output"] = False

    result = verify_bundle(bundle)

    assert result["classification"] == "GO_SEGMENT_PLAN_SELECTED"
    assert result["selected_plan_id"] == "p3"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("missing_rank", "INCOMPLETE"),
        ("plan_hash", "INCOMPLETE"),
        ("forged_consistent_plan_hash", "INCOMPLETE"),
        ("duplicate_row", "INCOMPLETE"),
        ("segment_range", "INCOMPLETE"),
        ("stage_owner", "INCOMPLETE"),
        ("duration_accounting", "INCOMPLETE"),
        ("memory_accounting", "INCOMPLETE"),
        ("boolean_rank", "INCOMPLETE"),
        ("boolean_segment_ordinal", "INCOMPLETE"),
        ("output", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("selected_state", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("unselected_state", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("cleanup", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("non_strict", "INCOMPLETE"),
        ("source", "INCOMPLETE"),
        ("residue", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
    ),
)
def test_verifier_fails_closed_on_invalid_or_negative_evidence(
    mutation,
    expected,
):
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    if mutation == "missing_rank":
        bundle["rows"] = [
            row for row in bundle["rows"] if row["rank"] != 3
        ]
    elif mutation == "plan_hash":
        bundle["rows"][0]["plan_sha256"] = "f" * 64
    elif mutation == "forged_consistent_plan_hash":
        for row in bundle["rows"]:
            row["plan_sha256"] = "f" * 64
    elif mutation == "duplicate_row":
        bundle["rows"].append(deepcopy(bundle["rows"][0]))
    elif mutation == "segment_range":
        bundle["rows"][0]["end_layer"] = 21
    elif mutation == "stage_owner":
        bundle["rows"][0]["include_embedding"] = False
    elif mutation == "duration_accounting":
        bundle["rows"][0]["capture_body_duration_ns"] = (
            bundle["rows"][0]["segment_capture_duration_ns"] + 1
        )
    elif mutation == "memory_accounting":
        bundle["rows"][0]["allocated_delta_bytes"] = -1
    elif mutation == "boolean_rank":
        for row in bundle["rows"]:
            if row["rank"] == 0:
                row["rank"] = False
                row["row_id"] = (
                    f"{row['plan_id']}:segment-"
                    f"{row['segment_ordinal']}:rank-False"
                )
    elif mutation == "boolean_segment_ordinal":
        for row in bundle["rows"]:
            if row["segment_ordinal"] == 0:
                row["segment_ordinal"] = False
                row["row_id"] = (
                    f"{row['plan_id']}:segment-False:"
                    f"rank-{row['rank']}"
                )
    elif mutation == "output":
        for row in bundle["rows"]:
            row["exact_output"] = False
    elif mutation == "selected_state":
        for row in bundle["rows"]:
            row["selected_state_exact"] = False
    elif mutation == "unselected_state":
        for row in bundle["rows"]:
            row["unselected_state_unchanged"] = False
    elif mutation == "cleanup":
        bundle["cleanup"]["classification"] = "DIRTY"
    elif mutation == "non_strict":
        bundle["launch_admission"]["admission_mode"] = (
            "shared_capacity"
        )
        bundle["launch_admission"]["strict_clean"] = False
    elif mutation == "source":
        bundle["source_manifest"]["source_tree_sha256"] = "e" * 64
    elif mutation == "residue":
        bundle["cleanup"]["exact_tag_scans"][1] = [{"pid": 42}]

    result = verify_bundle(bundle)
    assert result["classification"] == expected
    assert result["selected_plan_id"] is None


def test_path_bundle_loader_requires_complete_files(tmp_path: Path):
    result = verify_bundle(tmp_path)
    assert result["classification"] == "INCOMPLETE"
    assert result["selected_plan_id"] is None


def test_verifier_requires_process_receipts_for_every_candidate_plan():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    del bundle["process_receipts"]["plans"]["p4"]

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["selected_plan_id"] is None
    assert result["failed_gates"] == [
        "process receipt plan inventory is incomplete"
    ]


def test_verifier_fails_closed_on_malformed_rank_cleanup_receipt():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["process_receipts"]["plans"]["p3"][
        "rank_cleanup_receipts"
    ][0] = None

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "process receipts are incomplete"
    ]


def test_verifier_rejects_boolean_process_receipt_rank():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["process_receipts"]["plans"]["p3"][
        "rank_cleanup_receipts"
    ][0]["rank"] = False

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "process receipts are incomplete"
    ]


def test_verifier_fails_closed_on_malformed_cleanup_rank_row():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["cleanup"]["rank_rows"][0] = None

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "cleanup inventory is incomplete"
    ]


def test_verifier_rejects_boolean_cleanup_rank():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["cleanup"]["rank_rows"][0]["rank"] = False

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "cleanup inventory is incomplete"
    ]


def test_verifier_requires_rows_for_every_candidate_plan():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["rows"] = [
        row for row in bundle["rows"] if row["plan_id"] != "p4"
    ]

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "candidate plan inventory is incomplete"
    ]


def test_verifier_binds_the_frozen_model_revision():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["source_identity"]["model_revision"] = "f" * 40
    bundle["source_manifest"]["model_revision"] = "f" * 40

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == ["source identity is invalid"]


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_gpu", "memory", "utilization", "process"),
)
def test_verifier_recomputes_strict_clean_admission(mutation):
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    selected = bundle["launch_admission"]["selected_gpus"]
    if mutation == "duplicate_gpu":
        selected[1]["index"] = selected[0]["index"]
    elif mutation == "memory":
        selected[0]["memory_used_mib"] = 1_025
    elif mutation == "utilization":
        selected[0]["utilization_percent"] = 6
    else:
        selected[0]["compute_processes"] = [{"pid": 42}]

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "strict-clean admission is invalid"
    ]


def test_verifier_requires_three_final_cleanup_scans():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    bundle["cleanup"]["final_exact_tag_scans"] = [[]]

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert result["failed_gates"] == [
        "cleanup inventory is incomplete"
    ]


def test_verifier_accepts_successful_owned_reap_with_clean_final_scans():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    residue = [{"pid": 42, "matched_environment": True}]
    bundle["cleanup"]["exact_tag_scans"] = [residue, [], [], []]
    bundle["cleanup"]["final_exact_tag_scans"] = [[], [], []]
    bundle["cleanup"]["reap_receipt"] = {
        "requested_pids": [42],
        "terminated_pids": [42],
        "killed_pids": [],
        "remaining_pids": [],
    }

    result = verify_bundle(bundle)

    assert result["classification"] == "GO_SEGMENT_PLAN_SELECTED"
    assert result["selected_plan_id"] == "p3"


def test_path_bundle_manifest_detects_post_verification_tampering(
    tmp_path: Path,
):
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_200_000_000,
            ),
        },
    )
    files = {
        "source_identity.json": bundle["source_identity"],
        "source_manifest.json": bundle["source_manifest"],
        "launch_admission.json": bundle["launch_admission"],
        "cleanup.json": bundle["cleanup"],
        "process_receipts.json": bundle["process_receipts"],
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rows_path = tmp_path / "segment_rows.jsonl"
    rows_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in bundle["rows"]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-capture-manifest.v1"
        ),
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.iterdir()
            if path.is_file()
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert verify_bundle(tmp_path)["classification"] == (
        "GO_SEGMENT_PLAN_SELECTED"
    )

    rows_path.write_text(
        rows_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert verify_bundle(tmp_path)["classification"] == "INCOMPLETE"
