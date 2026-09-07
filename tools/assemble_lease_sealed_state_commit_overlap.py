#!/usr/bin/env python3
"""Assemble the immutable Stage-0 state-commit overlap evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

if __package__:
    from tools.lease_sealed_state_commit_overlap import (
        ACTIVE_TOKEN_GROUPS,
        HIDDEN_SIZE,
        LINEAR_LAYER_COUNT,
        MEASURED_PAIR_COUNT,
        STATE_BYTES_PER_TOKEN_PER_LAYER,
        WARMUP_PAIR_COUNT,
        WORLD_SIZE,
        classify_stage0,
        validate_measurement_row,
        validate_runtime_capabilities,
        validate_strict_clean_admission,
    )
else:
    from lease_sealed_state_commit_overlap import (
        ACTIVE_TOKEN_GROUPS,
        HIDDEN_SIZE,
        LINEAR_LAYER_COUNT,
        MEASURED_PAIR_COUNT,
        STATE_BYTES_PER_TOKEN_PER_LAYER,
        WARMUP_PAIR_COUNT,
        WORLD_SIZE,
        classify_stage0,
        validate_measurement_row,
        validate_runtime_capabilities,
        validate_strict_clean_admission,
    )


MANIFEST_SCHEMA = "lease-sealed-state-commit-overlap-manifest.v1"
PRODUCER_ARTIFACTS = frozenset(
    {
        "source_manifest.json",
        "environment_manifest.json",
        "gpu_rank_manifest.json",
        "workload_manifest.json",
        "admission.json",
        "paired_rows.jsonl",
        "correctness_rows.jsonl",
        "lifecycle_rows.jsonl",
        "memory_rows.jsonl",
        "overlap_rows.jsonl",
        "cleanup.json",
        "producer_result.json",
        "report.md",
        "manifest.sha256",
    }
)


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"JSON number must be finite: {value}")


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_duplicate_keys,
                parse_constant=_nonfinite,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(
                    json.loads(
                        line,
                        object_pairs_hook=_duplicate_keys,
                        parse_constant=_nonfinite,
                    )
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def _require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric evidence must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite(child)


def _write_json(path, payload):
    _require_finite(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl(path, rows):
    _require_finite(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root):
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(Path(root).iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json(
        Path(root) / "manifest.sha256",
        {
            "schema_version": MANIFEST_SCHEMA,
            "artifacts": artifacts,
        },
    )


def _is_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(source):
    return {
        "attempt": source["attempt"],
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
    }


def _validate_rank_rows(rows):
    return (
        isinstance(rows, list)
        and len(rows) == WORLD_SIZE
        and sorted(row.get("rank") for row in rows if isinstance(row, dict))
        == list(range(WORLD_SIZE))
    )


def _validate_source_identity(source):
    if (
        not isinstance(source, dict)
        or source.get("schema_version")
        != "lease-sealed-state-commit-overlap-source.v1"
        or not isinstance(source.get("attempt"), str)
        or not source["attempt"]
        or not _is_hex(source.get("source_revision"), 40)
        or not _is_hex(source.get("source_tree_sha256"), 64)
        or not isinstance(source.get("environment"), dict)
        or not _validate_rank_rows(source.get("gpu_rank_rows"))
    ):
        raise ValueError("source identity is invalid")
    try:
        source["admission"] = validate_strict_clean_admission(
            source.get("admission")
        )
    except ValueError as error:
        raise ValueError("source admission identity is invalid") from error
    source["environment"]["runtime_capabilities"] = (
        validate_runtime_capabilities(
            source["environment"].get("runtime_capabilities"),
            source["gpu_rank_rows"],
        )
    )
    return dict(source)


def _validate_identity(payload, identity, name):
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in identity.items()
    ):
        raise ValueError(f"{name} identity is invalid")


def _project(row, fields):
    return {field: row[field] for field in fields}


def _report(producer):
    lines = [
        "# Lease-Sealed State-Commit / AllReduce Overlap Stage-0",
        "",
        f"- Classification: `{producer['classification']}`",
        f"- Stage-1 authorized: `{str(producer['stage1_authorized']).lower()}`",
        f"- Measurement rows: `{producer['measurement_row_count']}`",
        "",
        "| Active tokens | Median speedup | P99 regression | Realized overlap | Host submission regression | Improving pairs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in producer["shape_summaries"]:
        lines.append(
            "| {active_tokens} | {median_speedup_ratio:.6f} | "
            "{p99_regression_ratio:.6f} | "
            "{median_realized_overlap_ratio:.6f} | "
            "{host_submission_regression_ratio:.6f} | "
            "{improving_pair_count} |".format(**row)
        )
    lines.extend([
        "",
        "Benefit and cost are reported together above. This Stage-0 result "
        "qualifies only the model-neutral mechanism and is not Qwen3.8 "
        "end-to-end performance evidence.",
        "",
    ])
    return "\n".join(lines)


def assemble_bundle(
    *,
    output_root,
    source_identity,
    rows,
    memory,
    lifecycle,
    cleanup,
):
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    source = _validate_source_identity(source_identity)
    _require_finite(
        {
            "source": source,
            "rows": rows,
            "memory": memory,
            "lifecycle": lifecycle,
            "cleanup": cleanup,
        }
    )
    identity = _identity(source)
    validated_rows = []
    for raw in rows:
        if any(raw.get(key) != value for key, value in identity.items()):
            raise ValueError("measurement identity is invalid")
        validated_rows.append(validate_measurement_row(raw))
    _validate_identity(lifecycle, identity, "lifecycle")
    _validate_identity(cleanup, identity, "cleanup")

    memory_rows = memory.get("rank_rows") if isinstance(memory, dict) else None
    if not _validate_rank_rows(memory_rows):
        raise ValueError("memory rank identity is invalid")
    lifecycle_rows = lifecycle.get("rank_rows")
    if (
        not isinstance(lifecycle_rows, list)
        or len(lifecycle_rows) != WORLD_SIZE * len(ACTIVE_TOKEN_GROUPS)
    ):
        raise ValueError("lifecycle rank identity is invalid")
    expected_lifecycle = {
        (rank, active_tokens)
        for rank in range(WORLD_SIZE)
        for active_tokens in ACTIVE_TOKEN_GROUPS
    }
    actual_lifecycle = {
        (row.get("rank"), row.get("active_tokens"))
        for row in lifecycle_rows
        if isinstance(row, dict)
    }
    if actual_lifecycle != expected_lifecycle:
        raise ValueError("lifecycle rank identity is invalid")
    cleanup_rows = cleanup.get("rank_rows")
    if (
        not _validate_rank_rows(cleanup_rows)
        or cleanup.get("owned_children_remaining") != []
        or cleanup.get("exact_tag_scans") != [[], [], []]
    ):
        raise ValueError("cleanup identity is invalid")

    classification = classify_stage0(validated_rows, memory, cleanup)
    source_manifest = {
        "schema_version": source["schema_version"],
        **identity,
    }
    environment_manifest = {
        "schema_version": (
            "lease-sealed-state-commit-overlap-environment.v1"
        ),
        **identity,
        **source["environment"],
    }
    gpu_rank_manifest = {
        "schema_version": (
            "lease-sealed-state-commit-overlap-gpu-ranks.v1"
        ),
        **identity,
        "rank_rows": source["gpu_rank_rows"],
    }
    workload_manifest = {
        "schema_version": (
            "lease-sealed-state-commit-overlap-workload.v1"
        ),
        **identity,
        "world_size": WORLD_SIZE,
        "active_token_groups": list(ACTIVE_TOKEN_GROUPS),
        "warmup_pair_count": WARMUP_PAIR_COUNT,
        "measured_pair_count": MEASURED_PAIR_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "state_bytes_per_token_per_layer": (
            STATE_BYTES_PER_TOKEN_PER_LAYER
        ),
        "linear_layer_count": LINEAR_LAYER_COUNT,
        "collective_dtype": "float32",
        "output_dtype": "bfloat16",
        "state_dtype": "bfloat16",
    }
    admission = {
        "schema_version": (
            "lease-sealed-state-commit-overlap-admission.v1"
        ),
        **identity,
        **source["admission"],
    }
    correctness_fields = (
        "attempt",
        "source_revision",
        "source_tree_sha256",
        "active_tokens",
        "pair_index",
        "rank",
        "reduced_output_exact",
        "final_output_exact",
        "shadow_payload_exact",
        "active_state_preserved_before_publish",
        "published_state_exact",
        "abort_preserved_old_state",
        "commit_identity_match",
        "finite_output",
        "timed_out",
    )
    overlap_fields = (
        "attempt",
        "source_revision",
        "source_tree_sha256",
        "active_tokens",
        "pair_index",
        "rank",
        "allreduce_interval_ns",
        "state_copy_interval_ns",
        "overlap_intersection_ns",
    )
    emitted_lifecycle = [{**identity, **row} for row in lifecycle_rows]
    emitted_memory = [{**identity, **row} for row in memory_rows]
    producer = {
        "schema_version": (
            "lease-sealed-state-commit-overlap-producer-result.v1"
        ),
        "classification": classification["classification"],
        "stage1_authorized": classification["stage1_authorized"],
        **identity,
        "measurement_row_count": classification["measurement_row_count"],
        "shape_summaries": classification["shape_summaries"],
    }

    _write_json(output_root / "source_manifest.json", source_manifest)
    _write_json(
        output_root / "environment_manifest.json",
        environment_manifest,
    )
    _write_json(output_root / "gpu_rank_manifest.json", gpu_rank_manifest)
    _write_json(output_root / "workload_manifest.json", workload_manifest)
    _write_json(output_root / "admission.json", admission)
    _write_jsonl(output_root / "paired_rows.jsonl", validated_rows)
    _write_jsonl(
        output_root / "correctness_rows.jsonl",
        [_project(row, correctness_fields) for row in validated_rows],
    )
    _write_jsonl(
        output_root / "lifecycle_rows.jsonl",
        emitted_lifecycle,
    )
    _write_jsonl(output_root / "memory_rows.jsonl", emitted_memory)
    _write_jsonl(
        output_root / "overlap_rows.jsonl",
        [_project(row, overlap_fields) for row in validated_rows],
    )
    _write_json(
        output_root / "cleanup.json",
        {
            "schema_version": (
                "lease-sealed-state-commit-overlap-cleanup.v1"
            ),
            **cleanup,
        },
    )
    _write_json(output_root / "producer_result.json", producer)
    _write_text(output_root / "report.md", _report(producer))
    _write_manifest(output_root)
    if {path.name for path in output_root.iterdir()} != PRODUCER_ARTIFACTS:
        raise RuntimeError("producer artifact inventory is incomplete")
    return producer


def assemble_raw_attempt(
    *,
    raw_root,
    source_identity_path,
    admission_path,
    output_root,
):
    raw_root = Path(raw_root).resolve()
    source = _load_json(source_identity_path)
    source["admission"] = _load_json(admission_path)
    capabilities = validate_runtime_capabilities(
        _load_json(raw_root / "runtime_capabilities.json"),
        source.get("gpu_rank_rows"),
    )
    source.setdefault("environment", {})["runtime_capabilities"] = (
        capabilities
    )
    lifecycle = _load_json(raw_root / "lifecycle.json")
    lifecycle.update(_identity(source))
    cleanup = _load_json(raw_root / "cleanup.json")
    cleanup.update(_identity(source))
    return assemble_bundle(
        output_root=output_root,
        source_identity=source,
        rows=_load_jsonl(raw_root / "measurement_rows.jsonl"),
        memory=_load_json(raw_root / "memory.json"),
        lifecycle=lifecycle,
        cleanup=cleanup,
    )


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--source-identity", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    result = assemble_raw_attempt(
        raw_root=args.raw_root,
        source_identity_path=args.source_identity,
        admission_path=args.admission,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
