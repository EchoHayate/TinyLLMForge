from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.assemble_qwen38_topology_local_tp2_island import (
    PRODUCER_ARTIFACTS,
    _load_json,
    assemble_bundle,
    main,
)


ATTEMPT = "20260908-qwen38-topology-local-tp2-island-stage0-r1"
SOURCE_REVISION = "a" * 40
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PAIR_GROUPS = ((0, 1), (2, 3))


def _identity():
    return {
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": MODEL_REVISION,
        "pair_groups": [list(group) for group in PAIR_GROUPS],
    }


def _timing_rows():
    rows = []
    for active_tokens in (1, 4, 8):
        baseline = active_tokens * 1000
        candidate = active_tokens * 900
        for repetition in range(15):
            for rank in range(4):
                rows.append({
                    **_identity(),
                    "schema": (
                        "qwen38.topology-local-tp2-island-worker.v1"
                    ),
                    "active_tokens": active_tokens,
                    "phase": "measured",
                    "repetition": repetition,
                    "rank": rank,
                    "pair_id": 0 if rank < 2 else 1,
                    "logical_rank": rank % 2,
                    "arm_order": (
                        ["baseline", "candidate"]
                        if repetition % 2 == 0
                        else ["candidate", "baseline"]
                    ),
                    "baseline_cuda_ns": baseline + rank,
                    "candidate_cuda_ns": candidate + rank,
                    "baseline_host_submission_ns": 100,
                    "candidate_host_submission_ns": 105,
                    "output_max_abs_error": 0.001,
                    "output_max_rel_error": 0.0001,
                    "convolution_max_abs_error": 0.001,
                    "convolution_max_rel_error": 0.0001,
                    "recurrent_max_abs_error": 0.001,
                    "recurrent_max_rel_error": 0.0001,
                    "pair_replica_output_max_abs_error": 0.0001,
                    "pair_replica_output_max_rel_error": 0.00001,
                    "pair_replica_convolution_max_abs_error": 0.0001,
                    "pair_replica_convolution_max_rel_error": 0.00001,
                    "pair_replica_recurrent_max_abs_error": 0.0001,
                    "pair_replica_recurrent_max_rel_error": 0.00001,
                    "output_within_tolerance": True,
                    "convolution_within_tolerance": True,
                    "recurrent_within_tolerance": True,
                    "pair_replicas_within_tolerance": True,
                    "greedy_argmax_equal": True,
                    "finite": True,
                    "candidate_global_collective_count": 0,
                    "fallback_count": 0,
                    "timed_allocation_count": 0,
                    "parameter_digests": {
                        "qkv_weight": "b" * 64,
                    },
                })
    return rows


def _migration_rows():
    rows = []
    for repetition in range(15):
        for rank in range(4):
            rows.append({
                **_identity(),
                "schema": (
                    "qwen38.topology-local-tp2-island-worker.v1"
                ),
                "phase": "measured",
                "repetition": repetition,
                "rank": rank,
                "pair_id": 0 if rank < 2 else 1,
                "logical_rank": rank % 2,
                "latency_ns": 2000 + rank,
                "source_bytes": 1_000,
                "transferred_bytes": 3_000,
                "retained_bytes": 2_000,
                "temporary_peak_allocated_bytes": 4_000,
                "steady_allocated_bytes": 2_000,
                "temporary_allocated_bytes_after_release": 0,
                "temporary_released_before_timing": True,
                "source_digest": "c" * 64,
                "candidate_digest": (
                    "d" * 64 if rank % 2 == 0 else "e" * 64
                ),
            })
    return rows


def _memory_rows():
    return [
        {
            **_identity(),
            "rank": rank,
            "projected_integrated_increment_bytes": (
                1800 * 1024 * 1024
            ),
            "peak_allocated_bytes": 70 * 1024**3,
            "peak_reserved_bytes": 72 * 1024**3,
            "physical_memory_bytes": 80 * 1024**3,
            "peak_allocated_ratio": 0.875,
        }
        for rank in range(4)
    ]


def passing_inputs():
    identity = _identity()
    checkpoint_digests = {"full": "c" * 64}
    return {
        "source_identity": {
            **identity,
            "schema": "qwen38.topology-local-tp2-island-source.v1",
        },
        "model_identity": {
            **identity,
            "hidden_size": 5120,
            "layer_count": 64,
            "linear_attention_layer_count": 48,
            "full_attention_layer_count": 16,
            "dtype": "bfloat16",
        },
        "admission": {
            **identity,
            "classification": "ADMITTED",
            "rank_rows": [
                {
                    "rank": rank,
                    "device_uuid": f"GPU-{rank}",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                    "foreign_compute_processes": [],
                }
                for rank in range(4)
            ],
        },
        "topology": {
            **identity,
            "selection_frozen": True,
            "selected_pair_groups": [
                list(group) for group in PAIR_GROUPS
            ],
        },
        "workload": {
            **identity,
            "active_token_groups": [1, 4, 8],
            "warmup_pairs_per_shape": 2,
            "measured_pairs_per_shape": 15,
            "migration_warmups": 2,
            "migration_measurements": 15,
        },
        "parameter_slices": {
            **identity,
            "layer_index": 0,
            "linear_attention_only": True,
            "full_attention_parameters_changed": False,
            "mlp_parameters_changed": False,
            "replica_digest_match": True,
            "checkpoint_reconstruction_match": True,
            "rank_parameter_evidence": [
                {
                    "rank": rank,
                    "logical_rank": rank % 2,
                    "parameter_digests": {
                        "slice": ("a" if rank % 2 == 0 else "b") * 64,
                    },
                    "checkpoint_full_parameter_digests": (
                        dict(checkpoint_digests)
                    ),
                    "reconstructed_full_parameter_digests": (
                        dict(checkpoint_digests)
                    ),
                    "checkpoint_reconstruction_match": True,
                }
                for rank in range(4)
            ],
        },
        "timing_rows": _timing_rows(),
        "migration_rows": _migration_rows(),
        "memory_rows": _memory_rows(),
        "lifecycle_rows": [
            {
                **identity,
                "rank": rank,
                "state_identity_match": True,
                "stale_generation_rejected": True,
                "different_request_rejected": True,
                "publish_after_success": True,
                "baseline_state_unchanged": True,
                "temporary_state_retired": True,
                "fallback_count": 0,
            }
            for rank in range(4)
        ],
        "cleanup": {
            **identity,
            "classification": "CLEAN",
            "rank_rows": [
                {
                    "rank": rank,
                    "process_groups_destroyed": 3,
                    "tensor_reservations_released": 6,
                    "candidate_state_unpublished": True,
                    "owned_children_remaining": [],
                    "task_files_outside_attempt_root": [],
                }
                for rank in range(4)
            ],
        },
    }


def mutate(inputs, mutation):
    result = copy.deepcopy(inputs)
    if mutation == "pair_disagreement":
        result["timing_rows"][0][
            "pair_replicas_within_tolerance"
        ] = False
    elif mutation == "baseline_numeric_failure":
        result["timing_rows"][0]["finite"] = False
    elif mutation == "token1_speedup_0049":
        for row in result["timing_rows"]:
            if row["active_tokens"] == 1:
                row["candidate_cuda_ns"] = 956
        for row in result["migration_rows"]:
            row["latency_ns"] = 1000
    elif mutation == "token48_geomean_0049":
        for row in result["timing_rows"]:
            if row["active_tokens"] in (4, 8):
                row["candidate_cuda_ns"] = int(
                    row["baseline_cuda_ns"] / 1.049
                )
    elif mutation == "p99_regression_0031":
        for row in result["timing_rows"]:
            if (
                row["active_tokens"] == 1
                and row["repetition"] == 14
            ):
                row["candidate_cuda_ns"] = int(
                    row["baseline_cuda_ns"] * 1.031
                )
    elif mutation == "improving_pairs_10":
        for row in result["timing_rows"]:
            if (
                row["active_tokens"] == 4
                and row["repetition"] >= 10
            ):
                row["candidate_cuda_ns"] = row["baseline_cuda_ns"]
    elif mutation == "host_regression_0101":
        for row in result["timing_rows"]:
            if row["active_tokens"] == 8:
                row["candidate_host_submission_ns"] = 111
    elif mutation == "break_even_33":
        for row in result["migration_rows"]:
            row["latency_ns"] = 3301
    elif mutation == "steady_increment_over_1920_mib":
        result["memory_rows"][0][
            "projected_integrated_increment_bytes"
        ] = 1920 * 1024 * 1024 + 1
    elif mutation == "peak_ratio_09801":
        result["memory_rows"][0]["peak_allocated_ratio"] = 0.9801
    else:
        raise AssertionError(mutation)
    return result


def test_assembler_classifies_complete_gate_as_go(tmp_path):
    result = assemble_bundle(tmp_path, **passing_inputs())

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
    )
    assert result["measurement_row_count"] == 180
    assert result["migration_row_count"] == 60
    assert {path.name for path in tmp_path.iterdir()} == set(
        PRODUCER_ARTIFACTS
    )


def test_assembler_accepts_any_frozen_rank_partition(tmp_path):
    inputs = passing_inputs()
    pair_groups = [[0, 2], [1, 3]]
    for value in inputs.values():
        if isinstance(value, dict) and "pair_groups" in value:
            value["pair_groups"] = pair_groups
    inputs["topology"]["selected_pair_groups"] = pair_groups
    for row_set in ("timing_rows", "migration_rows"):
        for row in inputs[row_set]:
            rank = row["rank"]
            row["pair_groups"] = pair_groups
            row["pair_id"] = 0 if rank in pair_groups[0] else 1
            row["logical_rank"] = pair_groups[row["pair_id"]].index(rank)
    for row_set in ("memory_rows", "lifecycle_rows"):
        for row in inputs[row_set]:
            row["pair_groups"] = pair_groups
    for row in inputs["admission"]["rank_rows"]:
        row["pair_groups"] = pair_groups
    for row in inputs["cleanup"]["rank_rows"]:
        row["pair_groups"] = pair_groups
    for row in inputs["parameter_slices"]["rank_parameter_evidence"]:
        rank = row["rank"]
        pair_id = 0 if rank in pair_groups[0] else 1
        logical_rank = pair_groups[pair_id].index(rank)
        row["logical_rank"] = logical_rank
        row["parameter_digests"] = {
            "slice": ("a" if logical_rank == 0 else "b") * 64,
        }

    assert assemble_bundle(tmp_path, **inputs)["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
    )


@pytest.mark.parametrize(
    ("mutation", "classification"),
    [
        ("pair_disagreement", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("baseline_numeric_failure", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("token1_speedup_0049", "NO_GO_PERFORMANCE"),
        ("token48_geomean_0049", "NO_GO_PERFORMANCE"),
        ("p99_regression_0031", "NO_GO_PERFORMANCE"),
        ("improving_pairs_10", "NO_GO_PERFORMANCE"),
        ("host_regression_0101", "NO_GO_PERFORMANCE"),
        ("break_even_33", "NO_GO_MIGRATION_AMORTIZATION"),
        ("steady_increment_over_1920_mib", "NO_GO_MEMORY"),
        ("peak_ratio_09801", "NO_GO_MEMORY"),
    ],
)
def test_assembler_enforces_frozen_boundaries(
    tmp_path,
    mutation,
    classification,
):
    inputs = mutate(passing_inputs(), mutation)

    assert assemble_bundle(tmp_path, **inputs)["classification"] == (
        classification
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_rank",
        "wrong_row_count",
        "unfrozen_pair_map",
        "source_drift",
        "model_drift",
        "candidate_global_collective",
        "fallback",
        "incomplete_cleanup",
        "parameter_reconstruction_mismatch",
    ),
)
def test_assembler_rejects_invalid_evidence(tmp_path, mutation):
    inputs = passing_inputs()
    if mutation == "missing_rank":
        inputs["timing_rows"] = [
            row for row in inputs["timing_rows"] if row["rank"] != 3
        ]
    elif mutation == "wrong_row_count":
        inputs["timing_rows"].pop()
    elif mutation == "unfrozen_pair_map":
        inputs["topology"]["selection_frozen"] = False
    elif mutation == "source_drift":
        inputs["timing_rows"][0]["source_revision"] = "f" * 40
    elif mutation == "model_drift":
        inputs["model_identity"]["model_revision"] = "f" * 40
    elif mutation == "candidate_global_collective":
        inputs["timing_rows"][0][
            "candidate_global_collective_count"
        ] = 1
    elif mutation == "fallback":
        inputs["timing_rows"][0]["fallback_count"] = 1
    elif mutation == "incomplete_cleanup":
        inputs["cleanup"]["rank_rows"][0][
            "candidate_state_unpublished"
        ] = False
    elif mutation == "parameter_reconstruction_mismatch":
        inputs["parameter_slices"]["rank_parameter_evidence"][3][
            "reconstructed_full_parameter_digests"
        ]["full"] = "d" * 64

    result = assemble_bundle(tmp_path, **inputs)

    assert result["classification"] == "INVALID_EVIDENCE"


def test_assembler_rejects_nonempty_output_directory(tmp_path):
    (tmp_path / "unexpected.txt").write_text("foreign")

    with pytest.raises(ValueError, match="empty"):
        assemble_bundle(tmp_path, **passing_inputs())


@pytest.mark.parametrize(
    "payload",
    (
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
    ),
)
def test_json_loader_rejects_duplicate_or_nonfinite_values(
    tmp_path,
    payload,
):
    path = tmp_path / "invalid.json"
    path.write_text(payload)

    with pytest.raises(ValueError):
        _load_json(path)


def test_assembler_cli_consumes_controller_aggregate_rows(tmp_path):
    inputs = passing_inputs()
    attempt_root = tmp_path / "attempt"
    controller = attempt_root / "controller"
    raw = attempt_root / "raw"
    controller.mkdir(parents=True)
    raw.mkdir()

    def write_json(path, payload):
        path.write_text(json.dumps(payload) + "\n")

    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write_json(controller / "source_identity.json", inputs["source_identity"])
    write_json(controller / "launch_admission.json", inputs["admission"])
    write_json(raw / "model_identity.json", inputs["model_identity"])
    write_json(raw / "topology.json", inputs["topology"])
    write_json(raw / "workload_manifest.json", inputs["workload"])
    write_json(
        raw / "parameter_slice_manifest.json",
        inputs["parameter_slices"],
    )
    write_jsonl(raw / "measurement_rows.jsonl", inputs["timing_rows"])
    write_jsonl(raw / "migration_rows.jsonl", inputs["migration_rows"])
    write_jsonl(raw / "memory_rows.jsonl", inputs["memory_rows"])
    write_jsonl(raw / "lifecycle_rows.jsonl", inputs["lifecycle_rows"])
    write_json(raw / "cleanup.json", inputs["cleanup"])

    assert main(["--attempt-root", str(attempt_root)]) == 0
    assert _load_json(
        attempt_root / "final_bundle" / "producer_result.json"
    )["classification"] == "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
