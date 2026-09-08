#!/usr/bin/env python3
"""Independently verify a Stage-0 state-commit overlap evidence bundle."""

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
        DIAGNOSTIC_ITERATION_COUNT,
        HIDDEN_SIZE,
        LINEAR_LAYER_COUNT,
        MEASURED_PAIR_COUNT,
        STATE_BYTES_PER_TOKEN_PER_LAYER,
        WARMUP_PAIR_COUNT,
        WORLD_SIZE,
        classify_stage01,
        classify_stage0,
        validate_stage01_diagnostic_row,
        validate_stage01_measurement_row,
        validate_measurement_row,
        validate_runtime_capabilities,
        validate_strict_clean_admission,
    )
else:
    from lease_sealed_state_commit_overlap import (
        ACTIVE_TOKEN_GROUPS,
        DIAGNOSTIC_ITERATION_COUNT,
        HIDDEN_SIZE,
        LINEAR_LAYER_COUNT,
        MEASURED_PAIR_COUNT,
        STATE_BYTES_PER_TOKEN_PER_LAYER,
        WARMUP_PAIR_COUNT,
        WORLD_SIZE,
        classify_stage01,
        classify_stage0,
        validate_stage01_diagnostic_row,
        validate_stage01_measurement_row,
        validate_measurement_row,
        validate_runtime_capabilities,
        validate_strict_clean_admission,
    )


MANIFEST_SCHEMA = "lease-sealed-state-commit-overlap-manifest.v1"
STAGE01_MANIFEST_SCHEMA = "tp4-completion-owned-overlap-manifest.v2"
STAGE0_SOURCE_SCHEMA = "lease-sealed-state-commit-overlap-source.v1"
STAGE01_SOURCE_SCHEMA = "tp4-completion-owned-overlap-source.v2"
TERMINAL_MANIFEST_SCHEMA = (
    "lease-sealed-state-commit-overlap-terminal-manifest.v1"
)
LEGACY_RECEIPT_NAME = "independent_verification.json"
REMOTE_RECEIPT_NAME = "remote_independent_verification.json"
LOCAL_RECEIPT_NAME = "local_streaming_independent_verification.json"
TERMINAL_MANIFEST_NAME = "manifest.json"
PRODUCER_FILES = frozenset(
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
    }
)
STAGE01_PRODUCER_FILES = frozenset(
    set(PRODUCER_FILES) | {"diagnostic_rows.jsonl"}
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root):
    manifest = _load_json(root / "manifest.sha256")
    manifest_schema = (
        manifest.get("schema_version")
        if isinstance(manifest, dict)
        else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest_schema not in {MANIFEST_SCHEMA, STAGE01_MANIFEST_SCHEMA}
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise ValueError("manifest is invalid")
    producer_files = (
        STAGE01_PRODUCER_FILES
        if manifest_schema == STAGE01_MANIFEST_SCHEMA
        else PRODUCER_FILES
    )
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    }
    accepted = {
        producer_files,
        producer_files | {LEGACY_RECEIPT_NAME},
        producer_files | {REMOTE_RECEIPT_NAME},
        producer_files
        | {
            REMOTE_RECEIPT_NAME,
            LOCAL_RECEIPT_NAME,
            TERMINAL_MANIFEST_NAME,
        },
    }
    if actual not in accepted or set(manifest["artifacts"]) != actual:
        raise ValueError("manifest artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(root / name) != expected
        ):
            raise ValueError("manifest artifact hash mismatch")
    return manifest_schema


def _rewrite_manifest(root, manifest_schema):
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json(
        root / "manifest.sha256",
        {
            "schema_version": manifest_schema,
            "artifacts": artifacts,
        },
    )


def _is_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(payload):
    return {
        "attempt": payload["attempt"],
        "source_revision": payload["source_revision"],
        "source_tree_sha256": payload["source_tree_sha256"],
    }


def _validate_rank_rows(rows):
    return (
        isinstance(rows, list)
        and len(rows) == WORLD_SIZE
        and sorted(row.get("rank") for row in rows if isinstance(row, dict))
        == list(range(WORLD_SIZE))
    )


def _terminal_manifest_payload(root, identity, classification):
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file()
        and path.name not in {"manifest.sha256", TERMINAL_MANIFEST_NAME}
    }
    return {
        "schema_version": TERMINAL_MANIFEST_SCHEMA,
        **identity,
        "classification": classification,
        "artifact_hashes": artifact_hashes,
    }


def _validate_existing_receipts(root, expected):
    for name in (
        LEGACY_RECEIPT_NAME,
        REMOTE_RECEIPT_NAME,
        LOCAL_RECEIPT_NAME,
    ):
        path = root / name
        if path.is_file() and _load_json(path) != expected:
            raise ValueError(f"{name} disagrees with reconstruction")


def _validate_terminal_manifest(root, identity, classification):
    path = root / TERMINAL_MANIFEST_NAME
    if not path.is_file():
        return
    expected = _terminal_manifest_payload(root, identity, classification)
    if _load_json(path) != expected:
        raise ValueError("terminal manifest disagrees with bundle")


def _require_identity(payload, identity, name):
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in identity.items()
    ):
        raise ValueError(f"{name} identity mismatch")


def _project(row, fields):
    return {field: row[field] for field in fields}


def _verify_bundle_for_schema(
    root,
    source,
    *,
    manifest_schema,
    stage01,
    receipt_name=LEGACY_RECEIPT_NAME,
    seal_terminal=False,
):
    identity = _identity(source)
    environment = _load_json(root / "environment_manifest.json")
    gpu_ranks = _load_json(root / "gpu_rank_manifest.json")
    workload = _load_json(root / "workload_manifest.json")
    admission = _load_json(root / "admission.json")
    cleanup = _load_json(root / "cleanup.json")
    producer = _load_json(root / "producer_result.json")
    rows = _load_jsonl(root / "paired_rows.jsonl")
    diagnostic_rows = (
        _load_jsonl(root / "diagnostic_rows.jsonl") if stage01 else None
    )
    correctness_rows = _load_jsonl(root / "correctness_rows.jsonl")
    lifecycle_rows = _load_jsonl(root / "lifecycle_rows.jsonl")
    memory_rows = _load_jsonl(root / "memory_rows.jsonl")
    overlap_rows = _load_jsonl(root / "overlap_rows.jsonl")
    _require_finite(
        {
            "environment": environment,
            "gpu_ranks": gpu_ranks,
            "workload": workload,
            "admission": admission,
            "cleanup": cleanup,
            "producer": producer,
            "rows": rows,
            "diagnostics": diagnostic_rows,
            "correctness": correctness_rows,
            "lifecycle": lifecycle_rows,
            "memory": memory_rows,
            "overlap": overlap_rows,
        }
    )
    for name, payload in (
        ("environment", environment),
        ("gpu rank", gpu_ranks),
        ("workload", workload),
        ("admission", admission),
        ("cleanup", cleanup),
        ("producer", producer),
    ):
        _require_identity(payload, identity, name)
    if not _validate_rank_rows(gpu_ranks.get("rank_rows")):
        raise ValueError("gpu rank identity mismatch")
    validate_runtime_capabilities(
        environment.get("runtime_capabilities"),
        gpu_ranks["rank_rows"],
    )
    try:
        validate_strict_clean_admission({
            "classification": admission.get("classification"),
            "rank_rows": admission.get("rank_rows"),
        })
    except ValueError as error:
        raise ValueError("admission identity mismatch") from error
    expected_workload = {
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
    if stage01:
        expected_workload.update({
            "protocol": "completion-owned-stage01",
            "diagnostic_iteration_count": DIAGNOSTIC_ITERATION_COUNT,
            "formal_arms": ["baseline", "completion_owned"],
            "diagnostic_arms": [
                "baseline",
                "event_only",
                "completion_owned",
            ],
        })
    if any(workload.get(key) != value for key, value in expected_workload.items()):
        raise ValueError("workload identity mismatch")

    validated_rows = []
    for raw in rows:
        _require_identity(raw, identity, "measurement")
        validated_rows.append(
            (
                validate_stage01_measurement_row
                if stage01
                else validate_measurement_row
            )(raw)
        )
    validated_diagnostics = []
    if stage01:
        for raw in diagnostic_rows:
            _require_identity(raw, identity, "diagnostic")
            validated_diagnostics.append(
                validate_stage01_diagnostic_row(raw)
            )
    common_fields = (
        "attempt",
        "source_revision",
        "source_tree_sha256",
        "active_tokens",
        "pair_index",
        "rank",
    )
    correctness_fields = common_fields + (
        (
            "expected_reduced_exact",
            "baseline_reduced_exact",
            "candidate_reduced_exact",
            "baseline_final_exact",
            "candidate_final_exact",
            "baseline_candidate_exact",
            "shadow_payload_exact",
            "active_state_preserved_before_publish",
            "published_state_exact",
            "abort_preserved_old_state",
            "commit_identity_match",
            "collective_wait_invoked",
            "collective_dependency_transferred",
            "side_effect_dependency_joined",
            "finite_output",
            "timed_out",
        )
        if stage01
        else (
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
    )
    overlap_fields = common_fields + (
        (
            "collective_outstanding_window_ns",
            "side_effect_window_ns",
            "overlap_intersection_ns",
        )
        if stage01
        else (
            "allreduce_interval_ns",
            "state_copy_interval_ns",
            "overlap_intersection_ns",
        )
    )
    if correctness_rows != [
        _project(row, correctness_fields) for row in validated_rows
    ]:
        raise ValueError("correctness rows disagree with paired rows")
    if overlap_rows != [
        _project(row, overlap_fields) for row in validated_rows
    ]:
        raise ValueError("overlap rows disagree with paired rows")
    if (
        len(lifecycle_rows) != WORLD_SIZE * len(ACTIVE_TOKEN_GROUPS)
        or {
            (row.get("rank"), row.get("active_tokens"))
            for row in lifecycle_rows
        }
        != {
            (rank, active_tokens)
            for rank in range(WORLD_SIZE)
            for active_tokens in ACTIVE_TOKEN_GROUPS
        }
        or any(
            any(row.get(key) != value for key, value in identity.items())
            for row in lifecycle_rows
        )
    ):
        raise ValueError("lifecycle identity mismatch")
    if not _validate_rank_rows(memory_rows) or any(
        any(row.get(key) != value for key, value in identity.items())
        for row in memory_rows
    ):
        raise ValueError("memory identity mismatch")
    cleanup_for_classifier = {
        "classification": (
            "CLEAN"
            if cleanup.get("classification") == "CLEAN"
            and cleanup.get("owned_children_remaining") == []
            and cleanup.get("exact_tag_scans") == [[], [], []]
            and _validate_rank_rows(cleanup.get("rank_rows"))
            and all(
                row.get("streams_released") is True
                and row.get("events_released") is True
                and row.get("timed_out") is False
                and row.get("process_group_destroyed") is True
                for row in cleanup["rank_rows"]
            )
            else "DIRTY"
        )
    }
    reconstructed = (
        classify_stage01(
            validated_rows,
            validated_diagnostics,
            {"rank_rows": memory_rows},
            cleanup_for_classifier,
        )
        if stage01
        else classify_stage0(
            validated_rows,
            {"rank_rows": memory_rows},
            cleanup_for_classifier,
        )
    )
    expected_producer = {
        "schema_version": (
            "tp4-completion-owned-overlap-producer-result.v2"
            if stage01
            else "lease-sealed-state-commit-overlap-producer-result.v1"
        ),
        "classification": reconstructed["classification"],
        "stage1_authorized": reconstructed["stage1_authorized"],
        **identity,
        "measurement_row_count": reconstructed["measurement_row_count"],
        "shape_summaries": reconstructed["shape_summaries"],
    }
    if stage01:
        expected_producer["diagnostic_row_count"] = reconstructed[
            "diagnostic_row_count"
        ]
    if producer != expected_producer:
        raise ValueError("producer classification or summary disagreement")
    receipt = {
        "schema_version": (
            "tp4-completion-owned-overlap-independent-verification.v2"
            if stage01
            else (
                "lease-sealed-state-commit-overlap-"
                "independent-verification.v1"
            )
        ),
        "status": "PASS",
        "producer_classification": producer["classification"],
        "reconstructed_classification": reconstructed["classification"],
        "artifact_hashes_verified": True,
        "measurement_row_count": reconstructed["measurement_row_count"],
    }
    if stage01:
        receipt["diagnostic_row_count"] = reconstructed[
            "diagnostic_row_count"
        ]
    _validate_existing_receipts(root, receipt)
    _validate_terminal_manifest(
        root,
        identity,
        reconstructed["classification"],
    )
    if seal_terminal and not (root / REMOTE_RECEIPT_NAME).is_file():
        raise ValueError("remote independent verification is missing")
    if receipt_name is not None:
        _write_json(root / receipt_name, receipt)
    if seal_terminal:
        _write_json(
            root / TERMINAL_MANIFEST_NAME,
            _terminal_manifest_payload(
                root,
                identity,
                reconstructed["classification"],
            ),
        )
    if receipt_name is not None or seal_terminal:
        _rewrite_manifest(root, manifest_schema)
    return receipt


def _verify_stage0_bundle(
    root,
    source,
    *,
    receipt_name,
    seal_terminal,
):
    return _verify_bundle_for_schema(
        root,
        source,
        manifest_schema=MANIFEST_SCHEMA,
        stage01=False,
        receipt_name=receipt_name,
        seal_terminal=seal_terminal,
    )


def _verify_stage01_bundle(
    root,
    source,
    *,
    receipt_name,
    seal_terminal,
):
    return _verify_bundle_for_schema(
        root,
        source,
        manifest_schema=STAGE01_MANIFEST_SCHEMA,
        stage01=True,
        receipt_name=receipt_name,
        seal_terminal=seal_terminal,
    )


def _verify_by_schema(root, source, *, receipt_name, seal_terminal):
    if source["schema_version"] == STAGE0_SOURCE_SCHEMA:
        return _verify_stage0_bundle(
            root,
            source,
            receipt_name=receipt_name,
            seal_terminal=seal_terminal,
        )
    if source["schema_version"] == STAGE01_SOURCE_SCHEMA:
        return _verify_stage01_bundle(
            root,
            source,
            receipt_name=receipt_name,
            seal_terminal=seal_terminal,
        )
    raise ValueError("source schema is unsupported")


def verify_bundle(
    root,
    *,
    receipt_name=LEGACY_RECEIPT_NAME,
    seal_terminal=False,
):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root must be an existing directory")
    if receipt_name not in {
        None,
        LEGACY_RECEIPT_NAME,
        REMOTE_RECEIPT_NAME,
        LOCAL_RECEIPT_NAME,
    }:
        raise ValueError("independent verification receipt name is invalid")
    if seal_terminal and receipt_name != LOCAL_RECEIPT_NAME:
        raise ValueError("terminal sealing requires the local receipt")
    if (
        (root / TERMINAL_MANIFEST_NAME).is_file()
        and receipt_name is not None
    ):
        raise ValueError("sealed terminal bundle is read-only")
    manifest_schema = _verify_manifest(root)
    source = _load_json(root / "source_manifest.json")
    expected_source_schema = (
        STAGE01_SOURCE_SCHEMA
        if manifest_schema == STAGE01_MANIFEST_SCHEMA
        else STAGE0_SOURCE_SCHEMA
    )
    if (
        source.get("schema_version") != expected_source_schema
        or (
            expected_source_schema == STAGE01_SOURCE_SCHEMA
            and source.get("protocol") != "completion-owned-stage01"
        )
        or not isinstance(source.get("attempt"), str)
        or not source["attempt"]
        or not _is_hex(source.get("source_revision"), 40)
        or not _is_hex(source.get("source_tree_sha256"), 64)
    ):
        raise ValueError("source identity mismatch")
    return _verify_by_schema(
        root,
        source,
        receipt_name=receipt_name,
        seal_terminal=seal_terminal,
    )


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--receipt-name",
        choices=(
            LEGACY_RECEIPT_NAME,
            REMOTE_RECEIPT_NAME,
            LOCAL_RECEIPT_NAME,
        ),
        default=LEGACY_RECEIPT_NAME,
    )
    parser.add_argument("--seal-terminal", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.check_only and args.seal_terminal:
        raise ValueError("--check-only and --seal-terminal are incompatible")
    result = verify_bundle(
        args.root,
        receipt_name=None if args.check_only else args.receipt_name,
        seal_terminal=args.seal_terminal,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
