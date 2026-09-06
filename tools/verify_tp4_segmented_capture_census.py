#!/usr/bin/env python3
"""Independent verifier for the TP4 segmented-capture census."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CENSUS_SCHEMA = "tinyllmforge.tp4-segmented-capture-census.v1"
SOURCE_SCHEMA = "tinyllmforge.tp4-segmented-capture-source.v1"
ADMISSION_SCHEMA = (
    "tinyllmforge.tp4-segmented-capture-admission.v1"
)
CLEANUP_SCHEMA = "tinyllmforge.tp4-segmented-capture-cleanup.v1"
PROCESS_SCHEMA = "tinyllmforge.tp4-segmented-capture-process.v1"
WORLD_SIZE = 4
MAX_SEGMENT_CAPTURE_DURATION_NS = 1_800_000_000
MAX_LIFECYCLE_DURATION_NS = 4_500_000_000
MAX_GPU_MEMORY_USED_MIB = 1_024
MAX_GPU_UTILIZATION_PERCENT = 5
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CANDIDATE_RANGES = {
    "p2": ((0, 32), (32, 64)),
    "p3": ((0, 22), (22, 43), (43, 64)),
    "p4": ((0, 16), (16, 32), (32, 48), (48, 64)),
}
REQUIRED_BUNDLE_FILES = (
    "manifest.json",
    "source_identity.json",
    "source_manifest.json",
    "launch_admission.json",
    "cleanup.json",
    "process_receipts.json",
    "segment_rows.jsonl",
)


def _result(
    classification: str,
    *,
    failed_gates: list[str],
    selected_plan_id: str | None = None,
    selected_plan_sha256: str | None = None,
    plans: dict | None = None,
) -> dict:
    if classification != "GO_SEGMENT_PLAN_SELECTED":
        selected_plan_id = None
        selected_plan_sha256 = None
    return {
        "schema_version": (
            "tinyllmforge.tp4-segmented-capture-verification.v1"
        ),
        "classification": classification,
        "selected_plan_id": selected_plan_id,
        "selected_plan_sha256": selected_plan_sha256,
        "failed_gates": failed_gates,
        "plans": {} if plans is None else plans,
    }


def _incomplete(reason: str) -> dict:
    return _result("INCOMPLETE", failed_gates=[reason])


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
        and manifest.get("schema_version")
        == "tinyllmforge.tp4-segmented-capture-manifest.v1"
        else None
    )
    expected_names = set(REQUIRED_BUNDLE_FILES) - {"manifest.json"}
    if (
        not isinstance(artifacts, dict)
        or not expected_names.issubset(artifacts)
    ):
        raise ValueError("bundle manifest is incomplete")
    for name, expected_sha256 in artifacts.items():
        path = root / name
        if (
            not path.is_file()
            or not isinstance(expected_sha256, str)
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise ValueError("bundle manifest hash mismatch")
    return {
        "schema_version": CENSUS_SCHEMA,
        "source_identity": _load_json(root / "source_identity.json"),
        "source_manifest": _load_json(root / "source_manifest.json"),
        "launch_admission": _load_json(
            root / "launch_admission.json"
        ),
        "cleanup": _load_json(root / "cleanup.json"),
        "process_receipts": _load_json(
            root / "process_receipts.json"
        ),
        "rows": _load_jsonl(root / "segment_rows.jsonl"),
    }


def _require_source(bundle: dict) -> tuple[str, dict]:
    if bundle.get("schema_version") != CENSUS_SCHEMA:
        raise ValueError("census schema mismatch")
    source = bundle.get("source_identity")
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != SOURCE_SCHEMA
        or not isinstance(source.get("run_tag"), str)
        or not source["run_tag"]
        or not isinstance(source.get("source_revision"), str)
        or len(source["source_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source["source_revision"]
        )
        or not isinstance(source.get("source_tree_sha256"), str)
        or len(source["source_tree_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source["source_tree_sha256"]
        )
        or source.get("model_repository") != "Qwen/Qwen3.8-27B"
        or source.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("source identity is invalid")
    if bundle.get("source_manifest") != source:
        raise ValueError("source manifest mismatch")
    return source["run_tag"], source


def _require_admission(bundle: dict, run_tag: str) -> None:
    admission = bundle.get("launch_admission")
    selected = (
        admission.get("selected_gpus")
        if isinstance(admission, dict)
        else None
    )
    structurally_valid = (
        isinstance(admission, dict)
        and admission.get("schema_version") == ADMISSION_SCHEMA
        and admission.get("run_tag") == run_tag
        and admission.get("admission_mode") == "strict_clean"
        and admission.get("strict_clean") is True
        and admission.get("world_size") == WORLD_SIZE
        and isinstance(selected, list)
        and len(selected) == WORLD_SIZE
    )
    if not structurally_valid:
        raise ValueError("strict-clean admission is invalid")
    indices = set()
    uuids = set()
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError("strict-clean admission is invalid")
        rank = row.get("rank")
        index = row.get("index")
        uuid = row.get("uuid")
        memory_used = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        processes = row.get("compute_processes")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or index in indices
            or uuid in uuids
            or isinstance(memory_used, bool)
            or not isinstance(memory_used, int)
            or memory_used < 0
            or memory_used > MAX_GPU_MEMORY_USED_MIB
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or utilization > MAX_GPU_UTILIZATION_PERCENT
            or processes != []
        ):
            raise ValueError("strict-clean admission is invalid")
        indices.add(index)
        uuids.add(uuid)
    if (
        {row.get("rank") for row in selected}
        != set(range(WORLD_SIZE))
    ):
        raise ValueError("strict-clean admission is invalid")


def _require_process_receipts(bundle: dict, run_tag: str) -> None:
    receipts = bundle.get("process_receipts")
    if (
        isinstance(receipts, dict)
        and receipts.get("schema_version") == PROCESS_SCHEMA
        and receipts.get("run_tag") == run_tag
        and isinstance(receipts.get("plans"), dict)
    ):
        plans = receipts["plans"]
        if set(plans) != set(CANDIDATE_RANGES):
            raise ValueError(
                "process receipt plan inventory is incomplete"
            )
        if any(
            not isinstance(receipt, dict)
            or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
            or receipt.get("process_group_destroyed") is not True
            or receipt.get("owned_children_remaining") != []
            or not isinstance(
                receipt.get("rank_cleanup_receipts"),
                list,
            )
            or len(receipt["rank_cleanup_receipts"]) != WORLD_SIZE
            or any(
                not isinstance(row, dict)
                for row in receipt["rank_cleanup_receipts"]
            )
            or any(
                isinstance(row.get("rank"), bool)
                or not isinstance(row.get("rank"), int)
                for row in receipt["rank_cleanup_receipts"]
            )
            or {
                row.get("rank")
                for row in receipt["rank_cleanup_receipts"]
            }
            != set(range(WORLD_SIZE))
            or any(
                row.get("process_group_destroyed") is not True
                for row in receipt["rank_cleanup_receipts"]
            )
            for receipt in plans.values()
        ):
            raise ValueError("process receipts are incomplete")
        return
    raise ValueError("process receipts are incomplete")


def _cleanup_is_clean(bundle: dict, run_tag: str) -> bool:
    cleanup = bundle.get("cleanup")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("schema_version") != CLEANUP_SCHEMA
        or cleanup.get("run_tag") != run_tag
    ):
        raise ValueError("cleanup receipt is invalid")
    scans = cleanup.get(
        "final_exact_tag_scans",
        cleanup.get("exact_tag_scans"),
    )
    rank_rows = cleanup.get("rank_rows")
    if (
        not isinstance(scans, list)
        or len(scans) != 3
        or any(not isinstance(scan, list) for scan in scans)
        or not isinstance(rank_rows, list)
        or any(not isinstance(row, dict) for row in rank_rows)
        or any(
            isinstance(row.get("rank"), bool)
            or not isinstance(row.get("rank"), int)
            for row in rank_rows
        )
    ):
        raise ValueError("cleanup inventory is incomplete")
    reap_receipt = cleanup.get("reap_receipt")
    if (
        reap_receipt is not None
        and (
            not isinstance(reap_receipt, dict)
            or reap_receipt.get("remaining_pids") != []
        )
    ):
        raise ValueError("owned-process reap receipt is invalid")
    residue = any(bool(scan) for scan in scans)
    ranks_complete = (
        len(rank_rows) == WORLD_SIZE
        and {row.get("rank") for row in rank_rows}
        == set(range(WORLD_SIZE))
        and all(
            row.get("exit_code") == 0
            and row.get("process_group_destroyed") is True
            for row in rank_rows
        )
    )
    return (
        cleanup.get("classification") == "CLEAN"
        and cleanup.get("owned_children_remaining") == []
        and not residue
        and ranks_complete
    )


def _canonical_plan_rows(rows: object) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("segment rows are missing")
    row_ids = [row.get("row_id") for row in rows if isinstance(row, dict)]
    if (
        len(row_ids) != len(rows)
        or any(not isinstance(row_id, str) for row_id in row_ids)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise ValueError("segment row IDs are invalid")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        plan_id = row.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan identity is missing")
        grouped.setdefault(plan_id, []).append(row)
    if set(grouped) != set(CANDIDATE_RANGES):
        raise ValueError("candidate plan inventory is incomplete")

    plans = {}
    for plan_id, plan_rows in grouped.items():
        expected_ranges = CANDIDATE_RANGES.get(plan_id)
        if expected_ranges is None:
            raise ValueError("candidate plan is unknown")
        hashes = {row.get("plan_sha256") for row in plan_rows}
        serialized_ranges = {
            json.dumps(
                row.get("plan_ranges"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in plan_rows
        }
        if (
            len(hashes) != 1
            or None in hashes
            or len(serialized_ranges) != 1
        ):
            raise ValueError("plan identity disagrees across ranks")
        ranges = plan_rows[0].get("plan_ranges")
        if (
            not isinstance(ranges, list)
            or not ranges
            or any(
                not isinstance(value, list)
                or len(value) != 2
                or isinstance(value[0], bool)
                or isinstance(value[1], bool)
                or not isinstance(value[0], int)
                or not isinstance(value[1], int)
                or value[1] <= value[0]
                for value in ranges
            )
        ):
            raise ValueError("segment ranges are invalid")
        if tuple(tuple(value) for value in ranges) != expected_ranges:
            raise ValueError("segment ranges do not match candidate plan")
        canonical_plan = {
            "layer_count": 64,
            "schema_version": (
                "tinyllmforge.segmented-exact-graph-plan.v1"
            ),
            "segments": [
                {
                    "end_layer": end_layer,
                    "include_commit": ordinal == len(ranges) - 1,
                    "include_embedding": ordinal == 0,
                    "include_final": ordinal == len(ranges) - 1,
                    "start_layer": start_layer,
                }
                for ordinal, (start_layer, end_layer)
                in enumerate(ranges)
            ],
        }
        expected_plan_sha256 = hashlib.sha256(
            json.dumps(
                canonical_plan,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if hashes != {expected_plan_sha256}:
            raise ValueError("plan hash does not match segment ranges")
        for ordinal, (start_layer, end_layer) in enumerate(ranges):
            if start_layer != (0 if ordinal == 0 else ranges[ordinal - 1][1]):
                raise ValueError("segment ranges are not contiguous")
            rank_rows = [
                row
                for row in plan_rows
                if row.get("segment_ordinal") == ordinal
            ]
            if (
                len(rank_rows) != WORLD_SIZE
                or {row.get("rank") for row in rank_rows}
                != set(range(WORLD_SIZE))
            ):
                raise ValueError("rank inventory is incomplete")
            for row in rank_rows:
                rank = row["rank"]
                segment_ordinal = row.get("segment_ordinal")
                body_ns = row.get("capture_body_duration_ns")
                sync_ns = row.get("post_capture_sync_duration_ns")
                segment_ns = row.get("segment_capture_duration_ns")
                allocated_delta = row.get("allocated_delta_bytes")
                reserved_delta = row.get("reserved_delta_bytes")
                stable_bytes = row.get(
                    "stable_boundary_buffer_bytes"
                )
                if (
                    row.get("row_id")
                    != f"{plan_id}:segment-{ordinal}:rank-{rank}"
                    or isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank < 0
                    or rank >= WORLD_SIZE
                    or isinstance(segment_ordinal, bool)
                    or not isinstance(segment_ordinal, int)
                    or segment_ordinal != ordinal
                    or row.get("world_size") != WORLD_SIZE
                    or row.get("start_layer") != start_layer
                    or row.get("end_layer") != end_layer
                    or row.get("include_embedding")
                    is not (ordinal == 0)
                    or row.get("include_final")
                    is not (ordinal == len(ranges) - 1)
                    or row.get("include_commit")
                    is not (ordinal == len(ranges) - 1)
                    or row.get("complete") is not True
                    or isinstance(body_ns, bool)
                    or not isinstance(body_ns, int)
                    or body_ns < 0
                    or isinstance(sync_ns, bool)
                    or not isinstance(sync_ns, int)
                    or sync_ns < 0
                    or isinstance(segment_ns, bool)
                    or not isinstance(segment_ns, int)
                    or segment_ns < body_ns + sync_ns
                    or isinstance(allocated_delta, bool)
                    or not isinstance(allocated_delta, int)
                    or allocated_delta < 0
                    or isinstance(reserved_delta, bool)
                    or not isinstance(reserved_delta, int)
                    or reserved_delta < 0
                    or isinstance(stable_bytes, bool)
                    or not isinstance(stable_bytes, int)
                    or stable_bytes < 0
                ):
                    raise ValueError("segment inventory disagrees")
        if len(plan_rows) != WORLD_SIZE * len(ranges):
            raise ValueError("segment inventory is incomplete")

        segment_durations = []
        for ordinal in range(len(ranges)):
            durations = [
                row.get("segment_capture_duration_ns")
                for row in plan_rows
                if row["segment_ordinal"] == ordinal
            ]
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in durations
            ):
                raise ValueError("segment duration is invalid")
            segment_durations.append(max(durations))
        lifecycle_values = [
            row.get("lifecycle_duration_ns") for row in plan_rows
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in lifecycle_values
        ):
            raise ValueError("lifecycle duration is invalid")
        for rank in range(WORLD_SIZE):
            rank_rows = [
                row for row in plan_rows if row["rank"] == rank
            ]
            rank_lifecycles = {
                row["lifecycle_duration_ns"] for row in rank_rows
            }
            if (
                len(rank_lifecycles) != 1
                or next(iter(rank_lifecycles))
                < sum(
                    row["segment_capture_duration_ns"]
                    for row in rank_rows
                )
            ):
                raise ValueError(
                    "lifecycle duration accounting is invalid"
                )
        plans[plan_id] = {
            "plan_sha256": next(iter(hashes)),
            "compute_segment_count": len(ranges),
            "tp_wide_segment_durations_ns": segment_durations,
            "max_segment_capture_duration_ns": max(
                segment_durations
            ),
            "tp_wide_lifecycle_duration_ns": max(
                lifecycle_values
            ),
            "exact_output": all(
                row.get("exact_output") is True for row in plan_rows
            ),
            "selected_state_exact": all(
                row.get("selected_state_exact") is True
                for row in plan_rows
            ),
            "unselected_state_unchanged": all(
                row.get("unselected_state_unchanged") is True
                for row in plan_rows
            ),
            "scratch_kv_restored": all(
                row.get("scratch_kv_restored") is True
                for row in plan_rows
            ),
            "graph_reset": all(
                row.get("graph_reset") is True for row in plan_rows
            ),
        }
    return plans


def verify_bundle(bundle_or_root) -> dict:
    try:
        bundle = (
            _load_bundle(bundle_or_root)
            if isinstance(bundle_or_root, (str, Path))
            else bundle_or_root
        )
        if not isinstance(bundle, dict):
            raise ValueError("census bundle is invalid")
        run_tag, _source = _require_source(bundle)
        _require_admission(bundle, run_tag)
        _require_process_receipts(bundle, run_tag)
        cleanup_clean = _cleanup_is_clean(bundle, run_tag)
        plans = _canonical_plan_rows(bundle.get("rows"))
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
    ) as error:
        return _incomplete(str(error) or type(error).__name__)

    lifecycle_failures = []
    ceiling_failures = []
    passing = []
    for plan_id, plan in plans.items():
        correctness = (
            plan["exact_output"]
            and plan["selected_state_exact"]
            and plan["unselected_state_unchanged"]
            and plan["scratch_kv_restored"]
            and plan["graph_reset"]
            and cleanup_clean
        )
        ceiling = (
            plan["max_segment_capture_duration_ns"]
            <= MAX_SEGMENT_CAPTURE_DURATION_NS
            and plan["tp_wide_lifecycle_duration_ns"]
            <= MAX_LIFECYCLE_DURATION_NS
        )
        plan["correctness_and_lifecycle_pass"] = correctness
        plan["capture_ceiling_pass"] = ceiling
        plan["plan_pass"] = correctness and ceiling
        if not correctness:
            lifecycle_failures.append(plan_id)
        elif not ceiling:
            ceiling_failures.append(plan_id)
        else:
            passing.append((plan_id, plan))

    if passing:
        selected_id, selected = min(
            passing,
            key=lambda item: (
                item[1]["compute_segment_count"],
                item[1]["tp_wide_lifecycle_duration_ns"],
                item[0],
            ),
        )
        return _result(
            "GO_SEGMENT_PLAN_SELECTED",
            failed_gates=[],
            selected_plan_id=selected_id,
            selected_plan_sha256=selected["plan_sha256"],
            plans=plans,
        )
    if lifecycle_failures:
        return _result(
            "NO_GO_CORRECTNESS_OR_LIFECYCLE",
            failed_gates=[
                f"correctness_or_lifecycle:{plan_id}"
                for plan_id in sorted(lifecycle_failures)
            ],
            plans=plans,
        )
    if not passing:
        return _result(
            "NO_GO_SEGMENTED_CAPTURE_CEILING",
            failed_gates=[
                f"capture_ceiling:{plan_id}"
                for plan_id in sorted(ceiling_failures)
            ],
            plans=plans,
        )
    raise AssertionError("unreachable segmented-capture classification")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(
        verify_bundle(args.bundle_root),
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
