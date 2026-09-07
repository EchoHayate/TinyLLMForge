from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from verify_tp4_segmented_capture_attribution import (
    verify_bundle,
    verification_results_equal,
)


MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONTROLS = (
    ("stitched_p4_repeat_0", ((0, 16), (16, 32), (32, 48), (48, 64))),
    ("stitched_p4_repeat_1", ((0, 16), (16, 32), (32, 48), (48, 64))),
    ("isolated_0_16", ((0, 16),)),
    ("isolated_16_32", ((16, 32),)),
    ("isolated_32_48", ((32, 48),)),
    ("isolated_48_64", ((48, 64),)),
    ("pool_fastest_shared", ((0, 16),)),
    ("pool_fastest_isolated", ((0, 16),)),
    ("pool_slowest_shared", ((16, 32),)),
    ("pool_slowest_isolated", ((16, 32),)),
)
PHASES = (
    "snapshot_and_prepare_ns",
    "graph_object_create_ns",
    "capture_context_enter_ns",
    "capture_body_ns",
    "capture_context_exit_and_instantiate_ns",
    "post_capture_synchronize_ns",
    "post_capture_restore_ns",
    "graph_reset_ns",
)


def _digest(selector):
    return {
        "selector": selector,
        "dtype": "torch.bfloat16",
        "shape": [64, 8, 8, 64],
        "byte_count": 524_288,
        "sha256": ("a" if selector == "key" else "b") * 64,
    }


def _diff(equal):
    if equal:
        return {
            "equal_to_s0": True,
            "mismatching_element_count": 0,
            "first_mismatch": None,
            "max_absolute_difference": 0.0,
        }
    return {
        "equal_to_s0": False,
        "mismatching_element_count": 1,
        "first_mismatch": {
            "layer": 0,
            "scratch_slot_ordinal": 0,
            "head": 0,
            "element_offset": 0,
        },
        "max_absolute_difference": 0.5,
    }


def make_bundle():
    run_tag = "phase-a1-test"
    source_revision = "1" * 40
    source_tree_sha256 = "2" * 64
    tools_root = Path(__file__).resolve().parent
    worker_sha256 = hashlib.sha256(
        (
            tools_root
            / "tp4_segmented_capture_attribution_worker.py"
        ).read_bytes()
    ).hexdigest()
    verifier_sha256 = hashlib.sha256(
        (
            tools_root
            / "verify_tp4_segmented_capture_attribution.py"
        ).read_bytes()
    ).hexdigest()
    base_controls = (
        {
            "control_id": "stitched_p4_repeat_0",
            "ranges": ((0, 16), (16, 32), (32, 48), (48, 64)),
            "kind": "stitched",
            "pool_mode": "shared",
            "formal_route_row": True,
        },
        {
            "control_id": "stitched_p4_repeat_1",
            "ranges": ((0, 16), (16, 32), (32, 48), (48, 64)),
            "kind": "stitched",
            "pool_mode": "shared",
            "formal_route_row": False,
        },
        *(
            {
                "control_id": f"isolated_{start}_{end}",
                "ranges": ((start, end),),
                "kind": "isolated",
                "pool_mode": "isolated",
                "formal_route_row": False,
            }
            for start, end in (
                (0, 16),
                (16, 32),
                (32, 48),
                (48, 64),
            )
        ),
    )
    plan_sha256 = hashlib.sha256(
        json.dumps(
            base_controls,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    phase_rows = []
    scratch_rows = []
    rank_results = []
    isolated_body = {
        (0, 16): 100,
        (16, 32): 400,
        (32, 48): 200,
        (48, 64): 300,
    }
    for rank in range(4):
        rank_phase = []
        rank_scratch = []
        for control_id, ranges in CONTROLS:
            for ordinal, (start_layer, end_layer) in enumerate(ranges):
                body = isolated_body.get(
                    (start_layer, end_layer),
                    150,
                ) + rank
                accounting = {
                    name: 10 for name in PHASES
                }
                accounting["capture_body_ns"] = body
                accounting["segment_total_ns"] = body + 80
                accounting["program_lifecycle_ns"] = 2_000
                row = {
                    "row_id": (
                        f"{control_id}:segment-{ordinal}:rank-{rank}"
                    ),
                    "rank": rank,
                    "control_id": control_id,
                    "segment_ordinal": ordinal,
                    "start_layer": start_layer,
                    "end_layer": end_layer,
                    "source_revision": source_revision,
                    "plan_sha256": plan_sha256,
                    "pool_mode": (
                        "shared"
                        if control_id.endswith("_shared")
                        or control_id.startswith("stitched")
                        else "isolated"
                    ),
                    "pool_identity": (
                        f"pool-{rank}-stitched_p4_repeat_1"
                        if control_id in {
                            "pool_fastest_shared",
                            "pool_slowest_shared",
                        }
                        else f"pool-{rank}-{control_id}"
                    ),
                    "linear_attention_layer_count": 12,
                    "full_attention_layer_count": 4,
                    "candidate_tensor_count": 24,
                    "candidate_tensor_bytes": 4_096,
                    "stable_hidden_candidate_logits_bytes": 8_192,
                    "allocated_before_bytes": 100,
                    "allocated_after_bytes": 200,
                    "allocated_delta_bytes": 100,
                    "reserved_before_bytes": 200,
                    "reserved_after_bytes": 400,
                    "reserved_delta_bytes": 200,
                    "collectives": {
                        "available": False,
                        "counts": {},
                        "unavailable_reason": (
                            "existing_receipt_not_exposed"
                        ),
                    },
                    "cuda_stream_identity": f"stream-{rank}",
                    "exact_output": (
                        True
                        if control_id.startswith("stitched")
                        else None
                    ),
                    "exact_output_applicable": (
                        control_id.startswith("stitched")
                    ),
                    "selected_state_exact": True,
                    "unselected_state_unchanged": True,
                    "scratch_kv_restored": True,
                    "graph_reset": True,
                    "formal_route_row": (
                        control_id == "stitched_p4_repeat_0"
                    ),
                    **accounting,
                }
                rank_phase.append(row)
                phase_rows.append(row)
            checkpoints = (
                ("S0", None),
                ("S1", None),
                ("S2", None),
                *((("S3", ordinal) for ordinal in range(len(ranges)))),
                ("S4", None),
                ("S5", None),
                ("S6", None),
                ("S7", None),
            )
            for checkpoint_ordinal, (
                checkpoint,
                segment_ordinal,
            ) in enumerate(checkpoints):
                exact = checkpoint != "S1"
                row = {
                    "row_id": (
                        f"{control_id}:checkpoint-{checkpoint_ordinal}:"
                        f"rank-{rank}"
                    ),
                    "rank": rank,
                    "control_id": control_id,
                    "source_revision": source_revision,
                    "plan_sha256": plan_sha256,
                    "checkpoint": checkpoint,
                    "segment_ordinal": segment_ordinal,
                    "synchronized": True,
                    "keys": _digest("key"),
                    "values": _digest("value"),
                    "key_diff": _diff(exact),
                    "value_diff": _diff(exact),
                    "scratch_snapshot_cpu_ns": 5,
                }
                rank_scratch.append(row)
                scratch_rows.append(row)
        rank_results.append({
            "phase": "A1",
            "rank": rank,
            "run_tag": run_tag,
            "source_revision": source_revision,
            "plan_sha256": plan_sha256,
            "control_ids": [name for name, _ranges in CONTROLS],
            "complete": True,
            "phase_rows": rank_phase,
            "scratch_rows": rank_scratch,
            "benefit": {
                "attributed_segments": 16,
                "first_scratch_divergence": "S1",
                "restore_round_trip_exact": True,
            },
            "cost": {
                "diagnostic_capture_count": 16,
                "diagnostic_synchronization_count": 100,
                "total_worker_duration_ns": 10_000 + rank,
                "scratch_snapshot_cpu_ns": 430,
                "peak_allocated_delta_bytes": 100,
                "peak_reserved_delta_bytes": 200,
            },
        })
    source = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-attribution-source.v1"
        ),
        "phase": "A1",
        "run_tag": run_tag,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "worker_sha256": worker_sha256,
        "verifier_sha256": verifier_sha256,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": MODEL_REVISION,
    }
    selected_gpus = [
        {
            "rank": rank,
            "index": rank,
            "uuid": f"GPU-{rank}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for rank in range(4)
    ]
    remote_base = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818"
    )
    attempt_root = (
        f"{remote_base}/tp4-segmented-capture-attribution/{run_tag}"
    )
    source_root = f"{attempt_root}/source"
    runtime_root = f"{attempt_root}/runtime"
    plan = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-attribution-plan.v1"
        ),
        "phase": "A1",
        "run_tag": run_tag,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "worker_sha256": worker_sha256,
        "verifier_sha256": verifier_sha256,
        "source_identity": source,
        "admission_mode": "strict_clean",
        "strict_clean": True,
        "plan_sha256": plan_sha256,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": MODEL_REVISION,
        "model_root": (
            f"{remote_base}/models/Qwen3.8-27B/snapshots/"
            f"{MODEL_REVISION}"
        ),
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "batch_size": 8,
        "prompt_length": 256,
        "max_tokens": 2,
        "model_length": 384,
        "controls": [
            {
                "control_id": name,
                "ranges": (
                    []
                    if name.startswith("pool_")
                    else [list(value) for value in ranges]
                ),
                "kind": (
                    "stitched"
                    if name.startswith("stitched_")
                    else (
                        "pool_control"
                        if name.startswith("pool_")
                        else "isolated"
                    )
                ),
                "pool_mode": (
                    "shared"
                    if name.startswith("stitched_")
                    or name.endswith("_shared")
                    else "isolated"
                ),
                "formal_route_row": (
                    name == "stitched_p4_repeat_0"
                ),
            }
            for name, ranges in CONTROLS
        ],
        "max_segment_ns": 1_800_000_000,
        "max_lifecycle_ns": 4_500_000_000,
        "max_added_memory_bytes_per_rank": 512 * 1024 * 1024,
        "selected_gpus": [
            {
                "gpu_index": row["index"],
                "gpu_uuid": row["uuid"],
                "memory_used_mib": row["memory_used_mib"],
                "utilization_percent": row["utilization_percent"],
                "compute_processes": row["compute_processes"],
            }
            for row in selected_gpus
        ],
        "selected_gpu_indices": [0, 1, 2, 3],
        "paths": {
            "attempt_root": attempt_root,
            "source_root": source_root,
            "raw_root": f"{attempt_root}/raw",
            "bundle_root": f"{attempt_root}/final_bundle",
            "controller_root": f"{attempt_root}/controller",
            "worker_stdout_path": (
                f"{attempt_root}/controller/worker.stdout"
            ),
            "worker_stderr_path": (
                f"{attempt_root}/controller/worker.stderr"
            ),
            "remote_verification_path": (
                f"{attempt_root}/controller/"
                "remote_independent_verification.json"
            ),
            "post_verification_manifest_path": (
                f"{attempt_root}/controller/"
                "post_verification_manifest.json"
            ),
        },
        "environment": {
            "TMPDIR": f"{runtime_root}/tmp",
            "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
            "HF_HOME": f"{runtime_root}/cache/huggingface",
            "TRANSFORMERS_CACHE": (
                f"{runtime_root}/cache/huggingface/transformers"
            ),
            "TORCH_EXTENSIONS_DIR": (
                f"{runtime_root}/cache/torch-extensions"
            ),
            "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
        },
        "process_environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{source_root}:{source_root}/tools",
            "TINYLLMFORGE_RUN_TAG": run_tag,
        },
    }
    return {
        "schema_version": (
            "tinyllmforge.tp4-segmented-attribution-bundle.v1"
        ),
        "source_identity": source,
        "plan": plan,
        "admission": {
            "schema_version": (
                "tinyllmforge.tp4-segmented-attribution-admission.v1"
            ),
            "run_tag": run_tag,
            "admission_mode": "strict_clean",
            "strict_clean": True,
            "selected_gpus": selected_gpus,
        },
        "phase_rows": phase_rows,
        "scratch_rows": scratch_rows,
        "rank_results": rank_results,
        "process_receipts": {
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
        },
        "cleanup_receipt": {
            "schema_version": (
                "tinyllmforge.tp4-segmented-attribution-cleanup.v1"
            ),
            "run_tag": run_tag,
            "classification": "CLEAN",
            "owned_children_remaining": [],
            "rank_rows": [
                {
                    "rank": rank,
                    "exit_code": 0,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
            "final_exact_tag_scans": [[], [], []],
        },
        "diagnosis": {
            "first_scratch_divergence": "S1",
            "slow_capture_phase": "capture_body_ns",
            "root_cause_kind": "eager_scratch_write",
            "source_path": (
                "tools/tp4_segmented_capture_attribution_worker.py"
            ),
            "source_symbol": "_AttributionCudaBackend.run_eager",
            "repair_statement": "preserve the scratch baseline",
            "repair_count": 1,
            "projected_max_segment_ns": 1_700_000_000,
            "projected_lifecycle_ns": 4_400_000_000,
            "projected_graph_count": 4,
        },
        "worker_summary": {
            "schema_version": (
                "tinyllmforge.tp4-segmented-attribution-worker.v1"
            ),
            "phase": "A1",
            "run_tag": run_tag,
            "source_revision": source_revision,
            "plan_sha256": plan_sha256,
            "control_ids": [name for name, _ranges in CONTROLS],
            "complete": True,
        },
    }


def _write_bundle(root: Path, bundle: dict):
    names = {
        "source_identity.json": "source_identity",
        "plan.json": "plan",
        "admission.json": "admission",
        "phase_rows.jsonl": "phase_rows",
        "scratch_rows.jsonl": "scratch_rows",
        "rank_results.json": "rank_results",
        "process_receipts.json": "process_receipts",
        "cleanup_receipt.json": "cleanup_receipt",
        "diagnosis.json": "diagnosis",
        "worker_summary.json": "worker_summary",
    }
    root.mkdir()
    artifacts = {}
    for filename, key in names.items():
        path = root / filename
        value = bundle[key]
        if filename.endswith(".jsonl"):
            payload = "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in value
            )
        else:
            payload = (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        path.write_text(payload)
        artifacts[filename] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-attribution-manifest.v1"
        ),
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )


def test_verifier_accepts_complete_phase_a1_bundle():
    result = verify_bundle(make_bundle())
    assert result["classification"] == "REPAIR_CANDIDATE"
    assert result["failed_gates"] == []
    assert result["phase"] == "A1"
    assert result["source_revision"] == "1" * 40
    assert result["run_tag"] == "phase-a1-test"
    assert result["scratch_summary"]["first_scratch_divergence"] == "S1"
    assert result["pool_summary"]["fastest_isolated_range"] == [0, 16]
    assert result["pool_summary"]["slowest_isolated_range"] == [16, 32]
    assert result["benefit"]["attributed_segments"] == 16
    assert result["cost"]["diagnostic_capture_count"] == 16


@pytest.mark.parametrize(
    "source_hash_name",
    ("worker_sha256", "verifier_sha256"),
)
def test_verifier_rejects_frozen_source_file_hash_mismatch(
    source_hash_name,
):
    bundle = make_bundle()
    bundle["source_identity"][source_hash_name] = "f" * 64

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert "source file hash mismatch" in " ".join(
        result["failed_gates"]
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda bundle: bundle["source_identity"].update(phase="A2"),
        lambda bundle: bundle["plan"].update(worker_sha256="f" * 64),
        lambda bundle: bundle["plan"].update(verifier_sha256="f" * 64),
        lambda bundle: bundle["plan"].update(source_identity={}),
    ),
)
def test_verifier_rejects_source_and_plan_identity_disagreement(mutate):
    bundle = make_bundle()
    mutate(bundle)

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert "source" in " ".join(result["failed_gates"])


@pytest.mark.parametrize(
    "mutate",
    (
        lambda bundle: bundle["plan"].update(
            selected_gpu_indices=[0, 1, 2, 7]
        ),
        lambda bundle: bundle["plan"]["paths"].update(
            attempt_root="/root/phase-a1-test"
        ),
        lambda bundle: bundle["plan"].update(
            admission_mode="shared_capacity"
        ),
    ),
)
def test_verifier_rejects_plan_admission_or_path_disagreement(mutate):
    bundle = make_bundle()
    mutate(bundle)

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert "plan" in " ".join(result["failed_gates"])


@pytest.mark.parametrize(
    ("mutate", "gate"),
    (
        (
            lambda bundle: bundle["source_identity"].update(
                model_revision="bad"
            ),
            "source",
        ),
        (
            lambda bundle: bundle["admission"][
                "selected_gpus"
            ][0]["compute_processes"].append({"pid": 1}),
            "strict-clean",
        ),
        (
            lambda bundle: bundle["phase_rows"][0].update(
                capture_body_ns=-1
            ),
            "phase",
        ),
        (
            lambda bundle: bundle["scratch_rows"][0].update(
                synchronized=False
            ),
            "scratch",
        ),
        (
            lambda bundle: bundle["cleanup_receipt"].update(
                final_exact_tag_scans=[[], []]
            ),
            "cleanup",
        ),
    ),
)
def test_verifier_fails_closed_on_structural_tampering(mutate, gate):
    bundle = make_bundle()
    mutate(bundle)
    result = verify_bundle(bundle)
    assert result["classification"] == "INCOMPLETE"
    assert any(gate in item for item in result["failed_gates"])


def test_verifier_uses_tp_wide_max_and_excludes_repeat_one():
    bundle = make_bundle()
    for row in bundle["phase_rows"]:
        if (
            row["control_id"] == "isolated_0_16"
            and row["rank"] == 3
        ):
            row["capture_body_ns"] = 199
            row["segment_total_ns"] = 279
        if row["control_id"] == "stitched_p4_repeat_1":
            row["capture_body_ns"] = 9_000_000_000
            row["segment_total_ns"] = 9_000_000_080
            row["program_lifecycle_ns"] = 9_000_000_080
    result = verify_bundle(bundle)
    assert result["classification"] == "REPAIR_CANDIDATE"
    assert result["tp_wide_phase_summary"][
        "isolated_0_16"
    ][0]["capture_body_ns"] == 199
    assert result["tp_wide_phase_summary"][
        "stitched_p4_repeat_1"
    ][0]["formal_timing_eligible"] is False


def test_verifier_reconstructs_pool_selection():
    bundle = make_bundle()
    for row in bundle["phase_rows"]:
        if row["control_id"] == "pool_fastest_shared":
            row["start_layer"], row["end_layer"] = 16, 32
    result = verify_bundle(bundle)
    assert result["classification"] == "INCOMPLETE"
    assert "pool_selection_mismatch" in result["failed_gates"]


def test_verifier_rejects_shared_pool_identity_disagreement():
    bundle = make_bundle()
    for row in bundle["phase_rows"]:
        if (
            row["rank"] == 0
            and row["control_id"] == "pool_slowest_shared"
        ):
            row["pool_identity"] = "different-pool"
    bundle["rank_results"][0]["phase_rows"] = [
        row
        for row in bundle["phase_rows"]
        if row["rank"] == 0
    ]

    result = verify_bundle(bundle)

    assert result["classification"] == "INCOMPLETE"
    assert "pool identity" in " ".join(result["failed_gates"])


def test_verifier_rejects_plan_hash_and_rank_payload_disagreement():
    bundle = make_bundle()
    bundle["plan"]["plan_sha256"] = "f" * 64
    assert verify_bundle(bundle)["classification"] == "INCOMPLETE"

    bundle = make_bundle()
    bundle["rank_results"][0]["phase_rows"][0] = {
        **bundle["rank_results"][0]["phase_rows"][0],
        "capture_body_ns": 999,
    }
    result = verify_bundle(bundle)
    assert result["classification"] == "INCOMPLETE"
    assert "rank payload" in " ".join(result["failed_gates"])


def test_verifier_rejects_modified_fixed_base_range():
    bundle = make_bundle()
    for row in bundle["phase_rows"]:
        if (
            row["control_id"] == "stitched_p4_repeat_0"
            and row["segment_ordinal"] == 0
        ):
            row["end_layer"] = 15
    result = verify_bundle(bundle)
    assert result["classification"] == "INCOMPLETE"
    assert "control range" in " ".join(result["failed_gates"])


def test_phase_a1_rejects_go_segmented_repair_request():
    bundle = make_bundle()
    bundle["diagnosis"]["requested_classification"] = (
        "GO_SEGMENTED_REPAIR"
    )
    result = verify_bundle(bundle)
    assert result["classification"] == "INCOMPLETE"
    assert "GO_SEGMENTED_REPAIR" in " ".join(result["failed_gates"])


def test_verifier_pivots_when_isolated_pool_exceeds_memory_gate():
    bundle = make_bundle()
    for row in bundle["phase_rows"]:
        if row["control_id"] == "pool_fastest_isolated":
            row["reserved_delta_bytes"] = 512 * 1024 * 1024 + 1
    result = verify_bundle(bundle)
    assert result["classification"] == (
        "PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION"
    )
    assert "memory_gate" in result["failed_gates"]


def test_manifest_detects_modified_phase_or_scratch_row(tmp_path):
    root = tmp_path / "bundle"
    _write_bundle(root, make_bundle())
    assert verify_bundle(root)["classification"] == "REPAIR_CANDIDATE"

    phase_path = root / "phase_rows.jsonl"
    phase_path.write_text(phase_path.read_text() + "{}\n")
    result = verify_bundle(root)
    assert result["classification"] == "INCOMPLETE"
    assert any(
        "manifest hash mismatch" in gate
        for gate in result["failed_gates"]
    )

    root = tmp_path / "scratch-bundle"
    _write_bundle(root, make_bundle())
    scratch_path = root / "scratch_rows.jsonl"
    scratch_path.write_text(scratch_path.read_text() + "{}\n")
    result = verify_bundle(root)
    assert result["classification"] == "INCOMPLETE"
    assert any(
        "manifest hash mismatch" in gate
        for gate in result["failed_gates"]
    )


def test_local_and_remote_verifier_results_must_match():
    result = verify_bundle(make_bundle())
    assert verification_results_equal(result, deepcopy(result)) is True
    remote = deepcopy(result)
    remote["classification"] = "INCOMPLETE"
    assert verification_results_equal(result, remote) is False
