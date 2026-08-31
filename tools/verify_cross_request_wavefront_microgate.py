#!/usr/bin/env python3
"""Independently verify a cross-request wavefront microgate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
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
VERIFICATION_SCHEMA = (
    "cross-request-wavefront-independent-verification.v1"
)
PRODUCER_FILES = frozenset(
    {
        "source_identity.json",
        "runtime_capabilities.json",
        "cohort_policy.json",
        "microgate_rows.jsonl",
        "memory_summary.json",
        "cleanup.json",
        "microgate_summary.json",
        "classification.json",
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


def _write_json(path, payload):
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
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise ValueError("manifest is invalid")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    }
    accepted = {
        PRODUCER_FILES,
        PRODUCER_FILES | {"independent_verification.json"},
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


def _rewrite_manifest(root):
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json(
        root / "manifest.sha256",
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


def _validate_identity(source, capabilities, policy, rows):
    if (
        not isinstance(source, dict)
        or source.get("schema_version")
        != "cross-request-wavefront-source.v1"
        or not isinstance(source.get("attempt"), str)
        or not source["attempt"]
        or not _is_hex(source.get("source_revision"), 40)
        or not _is_hex(source.get("source_tree_sha256"), 64)
    ):
        raise ValueError("source identity mismatch")
    attempt = source["attempt"]
    revision = source["source_revision"]
    tree_hash = source["source_tree_sha256"]
    for payload in (capabilities, policy):
        if (
            not isinstance(payload, dict)
            or payload.get("attempt") != attempt
            or payload.get("source_revision") != revision
            or payload.get("source_tree_sha256") != tree_hash
        ):
            raise ValueError("evidence identity mismatch")
    if (
        capabilities.get("schema_version")
        != "cross-request-wavefront-runtime-capabilities.v1"
        or not _validate_rank_rows(capabilities.get("rank_rows"))
    ):
        raise ValueError("runtime capability identity mismatch")
    if (
        policy.get("schema_version")
        != "cross-request-wavefront-cohort-policy.v1"
        or policy.get("active_token_groups")
        != list(ACTIVE_TOKEN_GROUPS)
        or set(policy.get("cohort_digests", {}))
        != {str(value) for value in ACTIVE_TOKEN_GROUPS}
        or not _is_hex(policy.get("collective_order_digest"), 64)
    ):
        raise ValueError("cohort policy identity mismatch")
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("attempt") != attempt
            or row.get("source_revision") != revision
            or row.get("source_tree_sha256") != tree_hash
        ):
            raise ValueError("evidence identity mismatch")
        if (
            row.get("cohort_digest")
            != policy["cohort_digests"].get(
                str(row.get("active_tokens"))
            )
            or row.get("collective_order_digest")
            != policy["collective_order_digest"]
        ):
            raise ValueError("microgate row policy identity mismatch")


def verify_bundle(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root must be an existing directory")
    _verify_manifest(root)
    source = _load_json(root / "source_identity.json")
    capabilities = _load_json(root / "runtime_capabilities.json")
    policy = _load_json(root / "cohort_policy.json")
    rows = _load_jsonl(root / "microgate_rows.jsonl")
    memory = _load_json(root / "memory_summary.json")
    cleanup = _load_json(root / "cleanup.json")
    _validate_identity(source, capabilities, policy, rows)

    reconstructed = classify_wavefront_microgate(
        rows,
        memory,
        _cleanup_for_classifier(cleanup),
    )
    producer = _load_json(root / "classification.json")
    expected_producer = {
        "schema_version": "cross-request-wavefront-classification.v1",
        "classification": reconstructed["classification"],
        "runtime_integration_authorized": reconstructed[
            "runtime_integration_authorized"
        ],
    }
    if producer != expected_producer:
        raise ValueError("producer classification mismatch")

    expected_summary = {
        "schema_version": "cross-request-wavefront-summary.v1",
        "attempt": source["attempt"],
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "classification": reconstructed["classification"],
        "runtime_integration_authorized": reconstructed[
            "runtime_integration_authorized"
        ],
        "shape_summaries": reconstructed["shape_summaries"],
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
    if _load_json(root / "microgate_summary.json") != expected_summary:
        raise ValueError("microgate summary mismatch")

    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "attempt": source["attempt"],
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "measurement_row_count": len(rows),
        "artifact_hashes_verified": True,
        "producer_classification": reconstructed["classification"],
        "reconstructed_classification": reconstructed["classification"],
    }
    _write_json(root / "independent_verification.json", result)
    _rewrite_manifest(root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_bundle(args.bundle_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
