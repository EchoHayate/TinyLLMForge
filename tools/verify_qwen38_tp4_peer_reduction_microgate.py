#!/usr/bin/env python3
"""Independently verify a TP4 peer-reduction microgate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

if __package__:
    from tools.qwen38_tp4_peer_reduction import (
        classify_peer_microgate,
        validate_peer_topology,
    )
else:
    from qwen38_tp4_peer_reduction import (
        classify_peer_microgate,
        validate_peer_topology,
    )


MANIFEST_SCHEMA = "qwen38.tp4-peer-reduction-manifest.v1"
VERIFICATION_SCHEMA = (
    "qwen38.tp4-peer-reduction-independent-verification.v1"
)
PRODUCER_FILES = frozenset({
    "source_identity.json",
    "peer_access_matrix.json",
    "ipc_roundtrip.jsonl",
    "microgate_rows.jsonl",
    "memory_summary.json",
    "cleanup.json",
    "microgate_summary.json",
    "classification.json",
})


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
                rows.append(json.loads(
                    line,
                    object_pairs_hook=_duplicate_keys,
                    parse_constant=_nonfinite,
                ))
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
    allowed = PRODUCER_FILES | {"independent_verification.json"}
    if (
        actual not in {
            PRODUCER_FILES,
            PRODUCER_FILES | {"independent_verification.json"},
        }
        or set(manifest["artifacts"]) != actual
        or not actual.issubset(allowed)
    ):
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
    _write_json(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _cleanup_for_classifier(cleanup):
    valid = (
        isinstance(cleanup, dict)
        and cleanup.get("classification") == "CLEAN"
        and cleanup.get("owned_children_remaining", []) == []
        and cleanup.get("exact_tag_scans", [[], [], []])
        == [[], [], []]
        and isinstance(cleanup.get("rank_rows"), list)
        and len(cleanup["rank_rows"]) == 4
        and sorted(row.get("rank") for row in cleanup["rank_rows"])
        == [0, 1, 2, 3]
        and all(
            row.get("peer_group_closed") is True
            and row.get("timed_out") is False
            for row in cleanup["rank_rows"]
        )
    )
    return {"classification": "CLEAN" if valid else "DIRTY"}


def verify_bundle(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root must be an existing directory")
    _verify_manifest(root)
    source = _load_json(root / "source_identity.json")
    matrix = _load_json(root / "peer_access_matrix.json")
    ipc_rows = _load_jsonl(root / "ipc_roundtrip.jsonl")
    rows = _load_jsonl(root / "microgate_rows.jsonl")
    memory = _load_json(root / "memory_summary.json")
    cleanup = _load_json(root / "cleanup.json")

    if (
        source.get("schema_version")
        != "qwen38.tp4-peer-reduction-source.v1"
        or not isinstance(source.get("attempt"), str)
        or not isinstance(source.get("source_revision"), str)
        or len(source["source_revision"]) != 40
        or matrix.get("attempt") != source["attempt"]
        or matrix.get("source_revision") != source["source_revision"]
        or matrix.get("world_size") != 4
        or matrix.get("rows") != ipc_rows
        or any(
            row.get("attempt") != source["attempt"]
            or row.get("source_revision") != source["source_revision"]
            for row in rows
        )
    ):
        raise ValueError("evidence identity mismatch")

    try:
        topology = validate_peer_topology(matrix["rows"])
        validate_peer_topology(ipc_rows)
        reconstructed = classify_peer_microgate(
            rows,
            _cleanup_for_classifier(cleanup),
            memory,
        )
    except ValueError:
        topology = {
            "classification": "INELIGIBLE_TOPOLOGY",
            "world_size": 4,
            "directed_peer_edge_count": len(ipc_rows),
        }
        reconstructed = {
            "classification": "INELIGIBLE_TOPOLOGY",
            "shape_summaries": [],
        }

    producer = _load_json(root / "classification.json")
    expected_producer = {
        "schema_version": (
            "qwen38.tp4-peer-reduction-classification.v1"
        ),
        "classification": reconstructed["classification"],
        "runtime_integration_authorized": (
            reconstructed["classification"] == "PASS"
        ),
    }
    if producer != expected_producer:
        raise ValueError("producer classification mismatch")
    expected_summary = {
        "schema_version": "qwen38.tp4-peer-reduction-summary.v1",
        "attempt": source["attempt"],
        "source_revision": source["source_revision"],
        "classification": reconstructed["classification"],
        "topology": topology,
        "shape_summaries": reconstructed.get("shape_summaries", []),
        "measurement_row_count": len(rows),
        "timeout_count": sum(row.get("timed_out") is True for row in rows),
        "maximum_allocated_delta_bytes": memory.get(
            "maximum_allocated_delta_bytes"
        ),
        "cleanup_classification": cleanup.get("classification"),
    }
    if _load_json(root / "microgate_summary.json") != expected_summary:
        raise ValueError("microgate summary mismatch")

    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "directed_peer_edge_count": len(ipc_rows),
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
