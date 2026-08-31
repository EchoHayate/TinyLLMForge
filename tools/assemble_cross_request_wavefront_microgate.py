#!/usr/bin/env python3
"""Assemble compact cross-request wavefront microgate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

if __package__:
    from tools.cross_request_wavefront_overlap import (
        ACTIVE_TOKEN_GROUPS,
        WORLD_SIZE,
        classify_wavefront_microgate,
    )
else:
    from cross_request_wavefront_overlap import (
        ACTIVE_TOKEN_GROUPS,
        WORLD_SIZE,
        classify_wavefront_microgate,
    )


MANIFEST_SCHEMA = "cross-request-wavefront-manifest.v1"
CLASSIFICATION_SCHEMA = "cross-request-wavefront-classification.v1"
SUMMARY_SCHEMA = "cross-request-wavefront-summary.v1"
PRODUCER_ARTIFACTS = (
    "source_identity.json",
    "runtime_capabilities.json",
    "cohort_policy.json",
    "microgate_rows.jsonl",
    "memory_summary.json",
    "cleanup.json",
    "microgate_summary.json",
    "classification.json",
    "manifest.sha256",
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


def _validate_source_identity(source):
    if (
        not isinstance(source, dict)
        or source.get("schema_version")
        != "cross-request-wavefront-source.v1"
        or not isinstance(source.get("attempt"), str)
        or not source["attempt"]
        or not _is_hex(source.get("source_revision"), 40)
        or not _is_hex(source.get("source_tree_sha256"), 64)
    ):
        raise ValueError("source identity is invalid")
    return dict(source)


def _validate_rank_rows(rows):
    return (
        isinstance(rows, list)
        and len(rows) == WORLD_SIZE
        and sorted(row.get("rank") for row in rows if isinstance(row, dict))
        == list(range(WORLD_SIZE))
    )


def _cleanup_for_classifier(cleanup):
    valid = (
        isinstance(cleanup, dict)
        and cleanup.get("classification") == "CLEAN"
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
    )
    return {"classification": "CLEAN" if valid else "DIRTY"}


def _validate_evidence_identity(
    source,
    runtime_capabilities,
    cohort_policy,
    rows,
):
    attempt = source["attempt"]
    revision = source["source_revision"]
    tree_hash = source["source_tree_sha256"]
    for name, payload in (
        ("runtime capabilities", runtime_capabilities),
        ("cohort policy", cohort_policy),
    ):
        if (
            not isinstance(payload, dict)
            or payload.get("attempt") != attempt
            or payload.get("source_revision") != revision
            or payload.get("source_tree_sha256") != tree_hash
        ):
            raise ValueError(f"{name} identity is invalid")
    if (
        runtime_capabilities.get("schema_version")
        != "cross-request-wavefront-runtime-capabilities.v1"
        or not _validate_rank_rows(runtime_capabilities.get("rank_rows"))
    ):
        raise ValueError("runtime capabilities identity is invalid")
    if (
        cohort_policy.get("schema_version")
        != "cross-request-wavefront-cohort-policy.v1"
        or cohort_policy.get("active_token_groups")
        != list(ACTIVE_TOKEN_GROUPS)
        or set(cohort_policy.get("cohort_digests", {}))
        != {str(value) for value in ACTIVE_TOKEN_GROUPS}
        or not _is_hex(cohort_policy.get("collective_order_digest"), 64)
    ):
        raise ValueError("cohort policy identity is invalid")
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("attempt") != attempt
            or row.get("source_revision") != revision
            or row.get("source_tree_sha256") != tree_hash
        ):
            raise ValueError("microgate row identity is invalid")
        expected_cohort = cohort_policy["cohort_digests"].get(
            str(row.get("active_tokens"))
        )
        if (
            row.get("cohort_digest") != expected_cohort
            or row.get("collective_order_digest")
            != cohort_policy["collective_order_digest"]
        ):
            raise ValueError("microgate row policy identity is invalid")


def assemble_bundle(
    *,
    output_root,
    source_identity,
    runtime_capabilities,
    cohort_policy,
    rows,
    memory,
    cleanup,
):
    source = _validate_source_identity(source_identity)
    _require_finite(
        {
            "source": source,
            "runtime_capabilities": runtime_capabilities,
            "cohort_policy": cohort_policy,
            "rows": rows,
            "memory": memory,
            "cleanup": cleanup,
        }
    )
    if not isinstance(rows, (list, tuple)):
        raise ValueError("microgate rows are invalid")
    _validate_evidence_identity(
        source,
        runtime_capabilities,
        cohort_policy,
        rows,
    )
    gate = classify_wavefront_microgate(
        rows,
        memory,
        _cleanup_for_classifier(cleanup),
    )
    classification = gate["classification"]
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "attempt": source["attempt"],
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "classification": classification,
        "runtime_integration_authorized": gate[
            "runtime_integration_authorized"
        ],
        "shape_summaries": gate["shape_summaries"],
        "measurement_row_count": len(rows),
        "timeout_count": sum(
            row.get("timed_out") is True
            for row in rows
            if isinstance(row, dict)
        ),
        "maximum_allocated_delta_bytes": memory.get(
            "maximum_allocated_delta_bytes"
        ),
        "maximum_reserved_delta_bytes": memory.get(
            "maximum_reserved_delta_bytes"
        ),
        "cleanup_classification": cleanup.get("classification"),
    }
    classification_payload = {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "runtime_integration_authorized": gate[
            "runtime_integration_authorized"
        ],
    }

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("bundle output directory must be empty")
    _write_json(root / "source_identity.json", source)
    _write_json(
        root / "runtime_capabilities.json",
        runtime_capabilities,
    )
    _write_json(root / "cohort_policy.json", cohort_policy)
    _write_jsonl(root / "microgate_rows.jsonl", rows)
    _write_json(root / "memory_summary.json", memory)
    _write_json(root / "cleanup.json", cleanup)
    _write_json(root / "microgate_summary.json", summary)
    _write_json(root / "classification.json", classification_payload)
    _write_manifest(root)
    return {
        "classification": classification,
        "bundle_root": str(root),
        "artifact_count": len(PRODUCER_ARTIFACTS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args(argv)
    attempt_root = args.attempt_root.resolve()
    raw = attempt_root / "raw"
    bundle = (
        args.bundle_root.resolve()
        if args.bundle_root is not None
        else attempt_root / "final_bundle"
    )
    result = assemble_bundle(
        output_root=bundle,
        source_identity=_load_json(
            attempt_root / "controller/source_identity.json"
        ),
        runtime_capabilities=_load_json(
            raw / "runtime_capabilities.json"
        ),
        cohort_policy=_load_json(
            attempt_root / "controller/cohort_policy.json"
        ),
        rows=_load_jsonl(raw / "microgate_rows.jsonl"),
        memory=_load_json(raw / "memory_summary.json"),
        cleanup=_load_json(raw / "cleanup.json"),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
