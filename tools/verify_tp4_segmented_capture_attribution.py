#!/usr/bin/env python3
"""Independent verifier for TP4 segmented-capture Phase A1 evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


BUNDLE_SCHEMA = "tinyllmforge.tp4-segmented-attribution-bundle.v1"
VERIFICATION_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-verification.v1"
)
SOURCE_SCHEMA = "tinyllmforge.tp4-segmented-attribution-source.v1"
PLAN_SCHEMA = "tinyllmforge.tp4-segmented-attribution-plan.v1"
ADMISSION_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-admission.v1"
)
CLEANUP_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-cleanup.v1"
)
MANIFEST_SCHEMA = (
    "tinyllmforge.tp4-segmented-attribution-manifest.v1"
)
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WORLD_SIZE = 4
MAX_GPU_MEMORY_USED_MIB = 1_024
MAX_GPU_UTILIZATION_PERCENT = 5
MAX_ADDED_MEMORY_BYTES_PER_RANK = 512 * 1024 * 1024
REQUIRED_BUNDLE_FILES = (
    "source_identity.json",
    "plan.json",
    "admission.json",
    "phase_rows.jsonl",
    "scratch_rows.jsonl",
    "rank_results.json",
    "process_receipts.json",
    "cleanup_receipt.json",
    "diagnosis.json",
    "worker_summary.json",
    "manifest.json",
)
EXPECTED_CONTROLS = (
    "stitched_p4_repeat_0",
    "stitched_p4_repeat_1",
    "isolated_0_16",
    "isolated_16_32",
    "isolated_32_48",
    "isolated_48_64",
    "pool_fastest_shared",
    "pool_fastest_isolated",
    "pool_slowest_shared",
    "pool_slowest_isolated",
)
BASE_RANGES = (
    (0, 16),
    (16, 32),
    (32, 48),
    (48, 64),
)
BASE_CONTROL_SPEC = (
    {
        "control_id": "stitched_p4_repeat_0",
        "ranges": BASE_RANGES,
        "kind": "stitched",
        "pool_mode": "shared",
        "formal_route_row": True,
    },
    {
        "control_id": "stitched_p4_repeat_1",
        "ranges": BASE_RANGES,
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
        for start, end in BASE_RANGES
    ),
)
EXPECTED_PLAN_SHA256 = hashlib.sha256(
    json.dumps(
        BASE_CONTROL_SPEC,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _load_contract():
    module_name = "_tinyllmforge_segmented_attribution_verifier"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[1]
        / "tinyvllm"
        / "engine"
        / "segmented_capture_attribution.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Phase A1 contract cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root is missing")
    missing = [
        name
        for name in REQUIRED_BUNDLE_FILES
        if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(
            "bundle files are incomplete: " + ", ".join(missing)
        )
    manifest = _load_json(root / "manifest.json")
    artifacts = (
        manifest.get("artifacts")
        if isinstance(manifest, dict)
        and manifest.get("schema_version") == MANIFEST_SCHEMA
        else None
    )
    expected = set(REQUIRED_BUNDLE_FILES) - {"manifest.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("bundle manifest is incomplete")
    for name, expected_sha256 in artifacts.items():
        path = root / name
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256))
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise ValueError("bundle manifest hash mismatch")
    return {
        "schema_version": BUNDLE_SCHEMA,
        "source_identity": _load_json(root / "source_identity.json"),
        "plan": _load_json(root / "plan.json"),
        "admission": _load_json(root / "admission.json"),
        "phase_rows": _load_jsonl(root / "phase_rows.jsonl"),
        "scratch_rows": _load_jsonl(root / "scratch_rows.jsonl"),
        "rank_results": _load_json(root / "rank_results.json"),
        "process_receipts": _load_json(root / "process_receipts.json"),
        "cleanup_receipt": _load_json(root / "cleanup_receipt.json"),
        "diagnosis": _load_json(root / "diagnosis.json"),
        "worker_summary": _load_json(root / "worker_summary.json"),
        "manifest": manifest,
    }


def _require_source_and_plan(bundle: dict) -> tuple[str, str, str, dict]:
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("bundle schema mismatch")
    source = bundle.get("source_identity")
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != SOURCE_SCHEMA
        or not isinstance(source.get("run_tag"), str)
        or not source["run_tag"]
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(source.get("source_revision")),
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(source.get("source_tree_sha256")),
        )
        or source.get("model_repository") != MODEL_REPOSITORY
        or source.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("source identity is invalid")
    plan = bundle.get("plan")
    expected_workload = {
        "phase": "A1",
        "run_tag": source["run_tag"],
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": WORLD_SIZE,
        "batch_size": 8,
        "prompt_length": 256,
        "max_tokens": 2,
        "model_length": 384,
        "max_segment_ns": 1_800_000_000,
        "max_lifecycle_ns": 4_500_000_000,
        "max_added_memory_bytes_per_rank": (
            MAX_ADDED_MEMORY_BYTES_PER_RANK
        ),
    }
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA
        or any(plan.get(name) != value for name, value in expected_workload.items())
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(plan.get("plan_sha256")),
        )
        or plan.get("plan_sha256") != EXPECTED_PLAN_SHA256
    ):
        raise ValueError("source plan or frozen workload is invalid")
    controls = plan.get("controls")
    if (
        not isinstance(controls, list)
        or tuple(
            row.get("control_id")
            for row in controls
            if isinstance(row, dict)
        )
        != EXPECTED_CONTROLS
    ):
        raise ValueError("plan control inventory is invalid")
    return (
        source["run_tag"],
        source["source_revision"],
        plan["plan_sha256"],
        plan,
    )


def _require_admission(bundle: dict, run_tag: str) -> None:
    admission = bundle.get("admission")
    selected = (
        admission.get("selected_gpus")
        if isinstance(admission, dict)
        else None
    )
    if (
        not isinstance(admission, dict)
        or admission.get("schema_version") != ADMISSION_SCHEMA
        or admission.get("run_tag") != run_tag
        or admission.get("admission_mode") != "strict_clean"
        or admission.get("strict_clean") is not True
        or not isinstance(selected, list)
        or len(selected) != WORLD_SIZE
    ):
        raise ValueError("strict-clean admission is invalid")
    indices = set()
    uuids = set()
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError("strict-clean admission is invalid")
        rank = row.get("rank")
        index = row.get("index")
        uuid = row.get("uuid")
        memory = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= WORLD_SIZE
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or index in indices
            or uuid in uuids
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
            or memory > MAX_GPU_MEMORY_USED_MIB
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or utilization > MAX_GPU_UTILIZATION_PERCENT
            or row.get("compute_processes") != []
        ):
            raise ValueError("strict-clean admission is invalid")
        indices.add(index)
        uuids.add(uuid)
    if {row["rank"] for row in selected} != set(range(WORLD_SIZE)):
        raise ValueError("strict-clean admission rank inventory is invalid")


def _require_process_and_cleanup(bundle: dict, run_tag: str) -> str:
    receipt = bundle.get("process_receipts")
    rank_receipts = (
        receipt.get("rank_cleanup_receipts")
        if isinstance(receipt, dict)
        else None
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("owned_children_remaining") != []
        or not isinstance(rank_receipts, list)
        or len(rank_receipts) != WORLD_SIZE
        or {row.get("rank") for row in rank_receipts}
        != set(range(WORLD_SIZE))
        or any(
            row.get("process_group_destroyed") is not True
            for row in rank_receipts
        )
    ):
        raise ValueError("process receipts are incomplete")
    cleanup = bundle.get("cleanup_receipt")
    scans = (
        cleanup.get("final_exact_tag_scans")
        if isinstance(cleanup, dict)
        else None
    )
    rank_rows = (
        cleanup.get("rank_rows")
        if isinstance(cleanup, dict)
        else None
    )
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("schema_version") != CLEANUP_SCHEMA
        or cleanup.get("run_tag") != run_tag
        or cleanup.get("classification") != "CLEAN"
        or cleanup.get("owned_children_remaining") != []
        or not isinstance(scans, list)
        or len(scans) < 3
        or any(scan != [] for scan in scans)
        or not isinstance(rank_rows, list)
        or len(rank_rows) != WORLD_SIZE
        or {row.get("rank") for row in rank_rows}
        != set(range(WORLD_SIZE))
        or any(
            row.get("exit_code") != 0
            or row.get("process_group_destroyed") is not True
            for row in rank_rows
        )
    ):
        raise ValueError("cleanup receipt is invalid")
    return "CLEAN"


def _phase_summary(
    bundle: dict,
    *,
    source_revision: str,
    plan_sha256: str,
) -> tuple[dict, bool]:
    rows = bundle.get("phase_rows")
    contract = _load_contract()
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE * 16:
        raise ValueError("phase row inventory is incomplete")
    row_ids = [row.get("row_id") for row in rows if isinstance(row, dict)]
    if len(row_ids) != len(rows) or len(set(row_ids)) != len(row_ids):
        raise ValueError("phase row IDs are invalid")
    grouped = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("control_id") not in EXPECTED_CONTROLS
            or row.get("source_revision") != source_revision
            or row.get("plan_sha256") != plan_sha256
            or row.get("rank") not in range(WORLD_SIZE)
        ):
            raise ValueError("phase row identity is invalid")
        accounting = contract.CapturePhaseAccounting(
            **{
                name: row[name]
                for name in (
                    *contract.CAPTURE_PHASE_NAMES,
                    "segment_total_ns",
                    "program_lifecycle_ns",
                )
            }
        )
        if asdict(accounting) != {
            name: row[name]
            for name in asdict(accounting)
        }:
            raise ValueError("phase accounting is invalid")
        identity = (
            row["control_id"],
            row.get("segment_ordinal"),
            row.get("start_layer"),
            row.get("end_layer"),
        )
        grouped.setdefault(identity, []).append(row)
    expected_counts = {
        control_id: len(ranges)
        for control_id, ranges in (
            ("stitched_p4_repeat_0", BASE_RANGES),
            ("stitched_p4_repeat_1", BASE_RANGES),
            *((f"isolated_{start}_{end}", ((start, end),)) for start, end in BASE_RANGES),
            ("pool_fastest_shared", ((0, 16),)),
            ("pool_fastest_isolated", ((0, 16),)),
            ("pool_slowest_shared", ((16, 32),)),
            ("pool_slowest_isolated", ((16, 32),)),
        )
    }
    if {
        control_id: sum(key[0] == control_id for key in grouped)
        for control_id in EXPECTED_CONTROLS
    } != expected_counts:
        raise ValueError("phase control inventory is invalid")
    expected_base_ranges = {
        "stitched_p4_repeat_0": BASE_RANGES,
        "stitched_p4_repeat_1": BASE_RANGES,
        **{
            f"isolated_{start}_{end}": ((start, end),)
            for start, end in BASE_RANGES
        },
    }
    for control_id, ranges in expected_base_ranges.items():
        actual = tuple(sorted(
            (
                row.get("start_layer"),
                row.get("end_layer"),
            )
            for row in rows
            if row.get("rank") == 0
            and row.get("control_id") == control_id
        ))
        if actual != tuple(sorted(ranges)):
            raise ValueError("fixed control range inventory is invalid")
    summary = {control_id: [] for control_id in EXPECTED_CONTROLS}
    formal_correct = True
    for identity, rank_rows in grouped.items():
        if len(rank_rows) != WORLD_SIZE:
            raise ValueError("phase TP4 rank inventory is incomplete")
        metadata_names = (
            "control_id",
            "segment_ordinal",
            "start_layer",
            "end_layer",
            "source_revision",
            "plan_sha256",
            "pool_mode",
            "linear_attention_layer_count",
            "full_attention_layer_count",
            "candidate_tensor_count",
            "candidate_tensor_bytes",
            "stable_hidden_candidate_logits_bytes",
            "formal_route_row",
        )
        if any(
            len({row.get(name) for row in rank_rows}) != 1
            for name in metadata_names
        ):
            raise ValueError("phase metadata disagrees across ranks")
        if any(
            row.get("linear_attention_layer_count") != 12
            or row.get("full_attention_layer_count") != 4
            or row.get("graph_reset") is not True
            or row.get("selected_state_exact") is not True
            or row.get("unselected_state_unchanged") is not True
            or row.get("scratch_kv_restored") is not True
            for row in rank_rows
        ):
            raise ValueError("phase correctness or graph reset failed")
        control_id = identity[0]
        formal = control_id == "stitched_p4_repeat_0"
        if any(row.get("formal_route_row") is not formal for row in rank_rows):
            raise ValueError("formal timing eligibility is invalid")
        if formal and any(
            row.get("exact_output_applicable") is not True
            or row.get("exact_output") is not True
            for row in rank_rows
        ):
            formal_correct = False
        aggregated = contract.aggregate_tp4_phase_rows(rank_rows)
        aggregated["allocated_delta_bytes"] = max(
            row["allocated_delta_bytes"] for row in rank_rows
        )
        aggregated["reserved_delta_bytes"] = max(
            row["reserved_delta_bytes"] for row in rank_rows
        )
        aggregated["formal_timing_eligible"] = formal
        summary[control_id].append(aggregated)
    for values in summary.values():
        values.sort(key=lambda row: row["segment_ordinal"])
    return summary, formal_correct


def _scratch_summary(
    bundle: dict,
    *,
    source_revision: str,
    plan_sha256: str,
) -> dict:
    rows = bundle.get("scratch_rows")
    contract = _load_contract()
    if not isinstance(rows, list):
        raise ValueError("scratch rows are missing")
    row_ids = [row.get("row_id") for row in rows if isinstance(row, dict)]
    if len(row_ids) != len(rows) or len(set(row_ids)) != len(row_ids):
        raise ValueError("scratch row IDs are invalid")
    grouped = {
        (rank, control_id): []
        for rank in range(WORLD_SIZE)
        for control_id in EXPECTED_CONTROLS
    }
    for row in rows:
        key = (row.get("rank"), row.get("control_id"))
        if (
            not isinstance(row, dict)
            or key not in grouped
            or row.get("source_revision") != source_revision
            or row.get("plan_sha256") != plan_sha256
            or row.get("synchronized") is not True
        ):
            raise ValueError("scratch row identity or synchronization failed")
        try:
            key_digest = dict(row["keys"])
            value_digest = dict(row["values"])
            key_digest["shape"] = tuple(key_digest["shape"])
            value_digest["shape"] = tuple(value_digest["shape"])
            record = contract.ScratchCheckpointRecord(
                checkpoint=row["checkpoint"],
                rank=row["rank"],
                synchronized=row["synchronized"],
                keys=contract.ScratchTensorDigest(**key_digest),
                values=contract.ScratchTensorDigest(**value_digest),
                key_diff=contract.ScratchDiffSummary(**row["key_diff"]),
                value_diff=contract.ScratchDiffSummary(**row["value_diff"]),
                segment_ordinal=row.get("segment_ordinal"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("scratch digest or diff is invalid") from error
        grouped[key].append(record)
    divergences = set()
    restored = True
    required_exact = {"S2", "S4", "S6", "S7"}
    for records in grouped.values():
        try:
            contract.validate_checkpoint_sequence(tuple(records))
        except ValueError as error:
            raise ValueError("scratch checkpoint inventory is invalid") from error
        divergence = next(
            (
                record.checkpoint
                for record in records
                if (
                    not record.key_diff.equal_to_s0
                    or not record.value_diff.equal_to_s0
                )
            ),
            None,
        )
        divergences.add(divergence)
        restored = restored and all(
            record.key_diff.equal_to_s0
            and record.value_diff.equal_to_s0
            for record in records
            if record.checkpoint in required_exact
        )
    if len(divergences) != 1:
        raise ValueError("scratch divergence disagrees across ranks")
    return {
        "first_scratch_divergence": next(iter(divergences)),
        "restore_round_trip_exact": restored,
        "timing_evidence_eligible": restored,
        "checkpoint_row_count": len(rows),
    }


def _pool_summary(phase_summary: dict) -> dict:
    isolated = {}
    for start_layer, end_layer in BASE_RANGES:
        control_id = f"isolated_{start_layer}_{end_layer}"
        isolated[(start_layer, end_layer)] = max(
            row["segment_total_ns"]
            for row in phase_summary[control_id]
        )
    fastest = min(
        isolated,
        key=lambda key: (isolated[key], key),
    )
    slowest = min(
        isolated,
        key=lambda key: (-isolated[key], key),
    )
    expected = {
        "pool_fastest_shared": fastest,
        "pool_fastest_isolated": fastest,
        "pool_slowest_shared": slowest,
        "pool_slowest_isolated": slowest,
    }
    for control_id, selected_range in expected.items():
        rows = phase_summary[control_id]
        if (
            len(rows) != 1
            or (
                rows[0]["start_layer"],
                rows[0]["end_layer"],
            )
            != selected_range
        ):
            raise ValueError("pool_selection_mismatch")
    shared = [
        row
        for control_id in (
            "pool_fastest_shared",
            "pool_slowest_shared",
        )
        for row in phase_summary[control_id]
    ]
    isolated_pool = [
        row
        for control_id in (
            "pool_fastest_isolated",
            "pool_slowest_isolated",
        )
        for row in phase_summary[control_id]
    ]
    return {
        "fastest_isolated_range": list(fastest),
        "slowest_isolated_range": list(slowest),
        "shared_max_segment_ns": max(
            row["segment_total_ns"] for row in shared
        ),
        "isolated_max_segment_ns": max(
            row["segment_total_ns"] for row in isolated_pool
        ),
        "isolated_added_allocated_bytes": max(
            row["allocated_delta_bytes"] for row in isolated_pool
        ),
        "isolated_added_reserved_bytes": max(
            row["reserved_delta_bytes"] for row in isolated_pool
        ),
    }


def _require_worker_binding(
    bundle: dict,
    *,
    run_tag: str,
    source_revision: str,
    plan_sha256: str,
) -> list[dict]:
    summary = bundle.get("worker_summary")
    rank_results = bundle.get("rank_results")
    if (
        not isinstance(summary, dict)
        or summary.get("phase") != "A1"
        or summary.get("run_tag") != run_tag
        or summary.get("source_revision") != source_revision
        or summary.get("plan_sha256") != plan_sha256
        or tuple(summary.get("control_ids", ())) != EXPECTED_CONTROLS
        or summary.get("complete") is not True
        or not isinstance(rank_results, list)
        or len(rank_results) != WORLD_SIZE
        or {row.get("rank") for row in rank_results}
        != set(range(WORLD_SIZE))
        or any(
            row.get("phase") != "A1"
            or row.get("run_tag") != run_tag
            or row.get("source_revision") != source_revision
            or row.get("plan_sha256") != plan_sha256
            or tuple(row.get("control_ids", ())) != EXPECTED_CONTROLS
            or row.get("complete") is not True
            for row in rank_results
        )
    ):
        raise ValueError("worker or rank binding is invalid")
    nested_phase = [
        row
        for result in rank_results
        for row in result.get("phase_rows", ())
    ]
    nested_scratch = [
        row
        for result in rank_results
        for row in result.get("scratch_rows", ())
    ]
    canonical = lambda rows: sorted(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for row in rows
        )
    )
    if (
        canonical(nested_phase) != canonical(bundle.get("phase_rows", ()))
        or canonical(nested_scratch)
        != canonical(bundle.get("scratch_rows", ()))
    ):
        raise ValueError("rank payload disagrees with worker rows")
    return rank_results


def _incomplete(reason: str) -> dict:
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "phase": "A1",
        "classification": "INCOMPLETE",
        "failed_gates": [reason],
        "source_revision": None,
        "run_tag": None,
        "tp_wide_phase_summary": {},
        "scratch_summary": {},
        "pool_summary": {},
        "benefit": {},
        "cost": {},
    }


def verify_bundle(bundle_or_root) -> dict:
    try:
        bundle = (
            _load_bundle(Path(bundle_or_root))
            if isinstance(bundle_or_root, (str, Path))
            else bundle_or_root
        )
        if not isinstance(bundle, dict):
            raise ValueError("Phase A1 bundle is invalid")
        run_tag, source_revision, plan_sha256, _plan = (
            _require_source_and_plan(bundle)
        )
        _require_admission(bundle, run_tag)
        cleanup = _require_process_and_cleanup(bundle, run_tag)
        rank_results = _require_worker_binding(
            bundle,
            run_tag=run_tag,
            source_revision=source_revision,
            plan_sha256=plan_sha256,
        )
        phase_summary, formal_correct = _phase_summary(
            bundle,
            source_revision=source_revision,
            plan_sha256=plan_sha256,
        )
        scratch_summary = _scratch_summary(
            bundle,
            source_revision=source_revision,
            plan_sha256=plan_sha256,
        )
        pool_summary = _pool_summary(phase_summary)
        diagnosis_value = bundle.get("diagnosis")
        if (
            isinstance(diagnosis_value, dict)
            and diagnosis_value.get("requested_classification")
            == "GO_SEGMENTED_REPAIR"
        ):
            raise ValueError(
                "GO_SEGMENTED_REPAIR is invalid for Phase A1"
            )
        diagnosis = _load_contract().AttributionDiagnosis(
            **diagnosis_value
        )
        if (
            diagnosis.first_scratch_divergence
            != scratch_summary["first_scratch_divergence"]
        ):
            raise ValueError("diagnosis scratch boundary mismatch")
        formal_rows = phase_summary["stitched_p4_repeat_0"]
        phase_maxima = {
            name: max(row[name] for row in formal_rows)
            for name in _load_contract().CAPTURE_PHASE_NAMES
        }
        slow_phase = max(
            phase_maxima,
            key=lambda name: (phase_maxima[name], name),
        )
        if diagnosis.slow_capture_phase != slow_phase:
            raise ValueError("diagnosis slow phase mismatch")
        memory_gate = (
            pool_summary["isolated_added_allocated_bytes"]
            <= MAX_ADDED_MEMORY_BYTES_PER_RANK
            and pool_summary["isolated_added_reserved_bytes"]
            <= MAX_ADDED_MEMORY_BYTES_PER_RANK
        )
        decision = _load_contract().classify_phase_a1({
            "complete": True,
            "source_bound": True,
            "rank_agreement": True,
            "verifier_agreement": True,
            "cleanup": cleanup,
            "restore_round_trip_exact": scratch_summary[
                "restore_round_trip_exact"
            ],
            "timing_evidence_eligible": (
                scratch_summary["timing_evidence_eligible"]
                and formal_correct
            ),
            "memory_gate_pass": memory_gate,
            "diagnosis": diagnosis,
        })
        phase_row_count = len(bundle["phase_rows"]) // WORLD_SIZE
        scratch_cpu_by_rank = {
            rank: sum(
                int(row.get("scratch_snapshot_cpu_ns", 0))
                for row in bundle["scratch_rows"]
                if row.get("rank") == rank
            )
            for rank in range(WORLD_SIZE)
        }
        benefit = {
            "attributed_segments": phase_row_count,
            "first_scratch_divergence": scratch_summary[
                "first_scratch_divergence"
            ],
            "restore_round_trip_exact": scratch_summary[
                "restore_round_trip_exact"
            ],
        }
        cost = {
            "diagnostic_capture_count": phase_row_count,
            "diagnostic_synchronization_count": max(
                int(row["cost"]["diagnostic_synchronization_count"])
                for row in rank_results
            ),
            "total_worker_duration_ns": max(
                int(row["cost"]["total_worker_duration_ns"])
                for row in rank_results
            ),
            "scratch_snapshot_cpu_ns": max(
                scratch_cpu_by_rank.values()
            ),
            "peak_allocated_delta_bytes": max(
                int(row["allocated_delta_bytes"])
                for row in bundle["phase_rows"]
            ),
            "peak_reserved_delta_bytes": max(
                int(row["reserved_delta_bytes"])
                for row in bundle["phase_rows"]
            ),
        }
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "phase": "A1",
            "classification": decision["classification"],
            "failed_gates": sorted(decision["failed_gates"]),
            "source_revision": source_revision,
            "run_tag": run_tag,
            "tp_wide_phase_summary": phase_summary,
            "scratch_summary": scratch_summary,
            "pool_summary": pool_summary,
            "benefit": benefit,
            "cost": cost,
        }
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
    ) as error:
        return _incomplete(str(error) or type(error).__name__)


def verification_results_equal(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return json.dumps(
        left,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_bundle(args.bundle_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["classification"] != "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
