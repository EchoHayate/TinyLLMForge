#!/usr/bin/env python3
"""Independent verifier for the fused INT4 draft Stage-0 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


EXPECTED_FILES = frozenset({
    "source_identity.json",
    "environment.json",
    "shape_manifest.json",
    "microgate_rows.jsonl",
    "memory.json",
    "graph.json",
    "cleanup.json",
    "summary.json",
    "classification.json",
    "independent_verification.json",
    "manifest.sha256",
})
MINIMUM_PAIRS = 200
MAXIMUM_MEDIAN_RATIO = 0.75
MAXIMUM_P99_RATIO = 0.95
MAXIMUM_WEIGHT_RATIO = 0.40
MAXIMUM_ABSOLUTE_ERROR = 0.08
MAXIMUM_RELATIVE_ERROR = 0.08
_ARM_ORDERS = (
    ["bf16", "dequant", "fused_int4"],
    ["fused_int4", "dequant", "bf16"],
)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _load_rows(path: Path) -> list[object]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        rows.append(json.loads(line, parse_constant=_reject_constant))
    return rows


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _nearest_rank(values: list[float], percentile: float):
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _shape_contract(payload: object) -> list[dict[str, object]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("shapes"), list)
        or not payload["shapes"]
    ):
        raise ValueError("invalid shape manifest")
    result = []
    seen = set()
    for row in payload["shapes"]:
        if not isinstance(row, dict):
            raise ValueError("invalid shape row")
        required = (
            "shape_id",
            "input_features",
            "output_features",
            "execution_count",
            "group_size",
        )
        if any(name not in row for name in required):
            raise ValueError("incomplete shape row")
        shape_id = row["shape_id"]
        integers = [row[name] for name in required[1:]]
        if (
            not isinstance(shape_id, str)
            or not shape_id
            or shape_id in seen
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in integers
            )
            or row["input_features"] % row["group_size"] != 0
            or row["input_features"] % 2 != 0
        ):
            raise ValueError("invalid shape row")
        seen.add(shape_id)
        result.append({name: row[name] for name in required})
    return result


def _independent_recompute(
    *,
    shape_manifest: object,
    rows: object,
    memory: object,
    graph: object,
    cleanup: object,
) -> dict[str, object]:
    shapes = _shape_contract(shape_manifest)
    shape_by_id = {row["shape_id"]: row for row in shapes}
    grouped = {shape_id: {} for shape_id in shape_by_id}
    incomplete = not isinstance(rows, list) or not rows
    correctness_failed = False
    timing_names = (
        "bf16_cuda_ns",
        "dequant_cuda_ns",
        "fused_int4_cuda_ns",
        "bf16_host_submission_ns",
        "dequant_host_submission_ns",
        "fused_int4_host_submission_ns",
    )
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                incomplete = True
                continue
            shape_id = row.get("shape_id")
            pair_index = row.get("pair_index")
            if (
                shape_id not in grouped
                or isinstance(pair_index, bool)
                or not isinstance(pair_index, int)
                or pair_index in grouped.get(shape_id, {})
                or row.get("arm_order") not in _ARM_ORDERS
                or any(
                    not _finite_nonnegative(row.get(name))
                    or row[name] <= 0
                    for name in timing_names
                )
                or not _finite_nonnegative(
                    row.get("maximum_absolute_error")
                )
                or not _finite_nonnegative(
                    row.get("maximum_relative_error")
                )
                or row.get("fallback_reason") is not None
                or row.get("full_dequant_allocation_observed") is not False
            ):
                incomplete = True
                continue
            if (
                row["maximum_absolute_error"] > MAXIMUM_ABSOLUTE_ERROR
                or row["maximum_relative_error"] > MAXIMUM_RELATIVE_ERROR
            ):
                correctness_failed = True
            grouped[shape_id][pair_index] = row

    expected_bf16 = sum(
        shape["output_features"]
        * shape["input_features"]
        * 2
        * shape["execution_count"]
        for shape in shapes
    )
    expected_packed = sum(
        (
            shape["output_features"] * (shape["input_features"] // 2)
            + shape["output_features"]
            * (shape["input_features"] // shape["group_size"])
            * 4
        )
        * shape["execution_count"]
        for shape in shapes
    )
    memory_failed = False
    memory_summary = None
    if not isinstance(memory, dict):
        incomplete = True
    else:
        candidate = memory.get("observed_candidate_weight_bytes")
        if (
            memory.get("classification") != "PASS"
            or memory.get("observed_bf16_weight_bytes") != expected_bf16
            or memory.get("minimum_packed_weight_bytes") != expected_packed
            or isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < expected_packed
            or not _finite_nonnegative(
                memory.get("maximum_candidate_allocated_delta_bytes")
            )
            or memory.get("full_dequant_allocation_observed") is not False
        ):
            incomplete = True
        else:
            ratio = candidate / expected_bf16
            memory_failed = ratio > MAXIMUM_WEIGHT_RATIO
            memory_summary = {
                "observed_bf16_weight_bytes": expected_bf16,
                "observed_candidate_weight_bytes": candidate,
                "minimum_packed_weight_bytes": expected_packed,
                "weight_bytes_ratio": ratio,
                "maximum_candidate_allocated_delta_bytes": memory[
                    "maximum_candidate_allocated_delta_bytes"
                ],
            }

    graph_failed = False
    if not isinstance(graph, dict):
        incomplete = True
    else:
        graph_rows = graph.get("shapes")
        graph_by_id = {}
        if not isinstance(graph_rows, list):
            incomplete = True
            graph_rows = []
        for row in graph_rows:
            if not isinstance(row, dict):
                incomplete = True
                continue
            shape_id = row.get("shape_id")
            if shape_id not in shape_by_id or shape_id in graph_by_id:
                incomplete = True
                continue
            if (
                not _finite_nonnegative(
                    row.get("maximum_absolute_error")
                )
                or not _finite_nonnegative(
                    row.get("maximum_relative_error")
                )
            ):
                incomplete = True
                continue
            if (
                row["maximum_absolute_error"] > MAXIMUM_ABSOLUTE_ERROR
                or row["maximum_relative_error"] > MAXIMUM_RELATIVE_ERROR
            ):
                correctness_failed = True
            if (
                row.get("capture_succeeded") is not True
                or isinstance(row.get("replay_count"), bool)
                or not isinstance(row.get("replay_count"), int)
                or row["replay_count"] < 2
                or row.get("static_pointers_stable") is not True
            ):
                graph_failed = True
            graph_by_id[shape_id] = row
        if set(graph_by_id) != set(shape_by_id):
            incomplete = True
        if graph.get("classification") != "PASS":
            graph_failed = True

    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        incomplete = True

    shape_summaries = []
    performance_failed = False
    for shape in shapes:
        pair_rows = grouped[shape["shape_id"]]
        if set(pair_rows) != set(range(MINIMUM_PAIRS)):
            incomplete = True
            continue
        ordered = [pair_rows[index] for index in range(MINIMUM_PAIRS)]
        first_bf16 = sum(
            row["arm_order"][0] == "bf16" for row in ordered
        )
        first_fused = sum(
            row["arm_order"][0] == "fused_int4" for row in ordered
        )
        if abs(first_bf16 - first_fused) > 1:
            incomplete = True
            continue
        bf16 = [row["bf16_cuda_ns"] for row in ordered]
        dequant = [row["dequant_cuda_ns"] for row in ordered]
        fused = [row["fused_int4_cuda_ns"] for row in ordered]
        bf16_median = median(bf16)
        fused_median = median(fused)
        median_ratio = fused_median / bf16_median
        bf16_p99 = _nearest_rank(bf16, 0.99)
        fused_p99 = _nearest_rank(fused, 0.99)
        p99_ratio = fused_p99 / bf16_p99
        if (
            median_ratio > MAXIMUM_MEDIAN_RATIO
            or p99_ratio > MAXIMUM_P99_RATIO
        ):
            performance_failed = True
        shape_summaries.append({
            "shape_id": shape["shape_id"],
            "execution_count": shape["execution_count"],
            "pair_count": len(ordered),
            "bf16_median_cuda_ns": bf16_median,
            "dequant_median_cuda_ns": median(dequant),
            "fused_int4_median_cuda_ns": fused_median,
            "candidate_to_bf16_median_ratio": median_ratio,
            "bf16_p99_cuda_ns": bf16_p99,
            "fused_int4_p99_cuda_ns": fused_p99,
            "candidate_to_bf16_p99_ratio": p99_ratio,
        })

    if correctness_failed:
        classification = "NO_GO_CORRECTNESS"
    elif incomplete:
        classification = "INCONCLUSIVE_EVIDENCE"
    elif memory_failed:
        classification = "NO_GO_MEMORY"
    elif graph_failed:
        classification = "NO_GO_GRAPH"
    elif performance_failed:
        classification = "NO_GO_PERFORMANCE"
    else:
        classification = "GO_FUSED_INT4_DRAFT_KERNEL"

    result = {
        "classification": classification,
        "shape_summaries": shape_summaries,
    }
    if not incomplete:
        total = sum(row["execution_count"] for row in shape_summaries)
        result["weighted_summary"] = {
            "candidate_to_bf16_median_ratio": sum(
                row["candidate_to_bf16_median_ratio"]
                * row["execution_count"]
                for row in shape_summaries
            ) / total,
            "candidate_to_bf16_p99_ratio": sum(
                row["candidate_to_bf16_p99_ratio"]
                * row["execution_count"]
                for row in shape_summaries
            ) / total,
            "execution_count": total,
        }
        result["memory_summary"] = memory_summary
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest(bundle_dir: Path) -> None:
    manifest_path = bundle_dir / "manifest.sha256"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    entries = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("invalid manifest line")
        digest, name = parts
        if name in entries:
            raise ValueError("duplicate manifest entry")
        entries[name] = digest
    expected = EXPECTED_FILES - {"manifest.sha256"}
    if set(entries) != expected:
        raise ValueError("manifest inventory mismatch")
    for name, digest in entries.items():
        path = bundle_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("manifest target is missing or a symlink")
        if path.resolve().parent != bundle_dir:
            raise ValueError("manifest target escapes bundle directory")
        if _sha256(path) != digest:
            raise ValueError(f"manifest digest mismatch: {name}")


def recompute_bundle_evidence(
    bundle_dir: Path,
    *,
    verify_manifest: bool,
    require_receipt: bool,
) -> dict[str, object]:
    bundle_dir = bundle_dir.expanduser().resolve()
    if not bundle_dir.is_dir():
        raise ValueError("bundle directory does not exist")
    expected = set(EXPECTED_FILES)
    if not verify_manifest:
        expected.remove("manifest.sha256")
    if not require_receipt:
        expected.remove("independent_verification.json")
    actual = {path.name for path in bundle_dir.iterdir()}
    if actual != expected:
        raise ValueError("bundle inventory mismatch")
    for path in bundle_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("bundle contains a symlink or non-file")
        if path.resolve().parent != bundle_dir:
            raise ValueError("bundle path escapes bundle directory")
    if verify_manifest:
        _verify_manifest(bundle_dir)

    source = _load_json(bundle_dir / "source_identity.json")
    source_revision = source.get("source_revision")
    run_tag = source.get("run_tag")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
        or not isinstance(run_tag, str)
        or not run_tag
    ):
        raise ValueError("invalid source identity")

    recomputed = _independent_recompute(
        shape_manifest=_load_json(bundle_dir / "shape_manifest.json"),
        rows=_load_rows(bundle_dir / "microgate_rows.jsonl"),
        memory=_load_json(bundle_dir / "memory.json"),
        graph=_load_json(bundle_dir / "graph.json"),
        cleanup=_load_json(bundle_dir / "cleanup.json"),
    )
    summary = _load_json(bundle_dir / "summary.json")
    classification = _load_json(bundle_dir / "classification.json")
    identity_payloads = (summary, classification)
    if require_receipt:
        receipt = _load_json(
            bundle_dir / "independent_verification.json"
        )
        identity_payloads += (receipt,)
        if receipt.get("status") != "PASS":
            raise ValueError("independent verification receipt is not PASS")
    if any(
        payload.get("source_revision") != source_revision
        or payload.get("run_tag") != run_tag
        for payload in identity_payloads
    ):
        raise ValueError("source or run identity mismatch")
    if summary.get("classification") != recomputed["classification"]:
        raise ValueError("summary classification mismatch")
    if classification.get("classification") != recomputed["classification"]:
        raise ValueError("classification receipt mismatch")
    for key in ("shape_summaries", "weighted_summary", "memory_summary"):
        if key in recomputed and summary.get(key) != recomputed[key]:
            raise ValueError(f"summary metric mismatch: {key}")
    if require_receipt and receipt.get("classification") != recomputed[
        "classification"
    ]:
        raise ValueError("independent receipt classification mismatch")
    return {
        "status": "PASS",
        "source_revision": source_revision,
        "run_tag": run_tag,
        **recomputed,
    }


def verify_bundle(bundle_dir: Path) -> dict[str, object]:
    return recompute_bundle_evidence(
        Path(bundle_dir),
        verify_manifest=True,
        require_receipt=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_bundle(args.bundle_dir),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
