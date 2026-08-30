#!/usr/bin/env python3
"""Assemble compact TP4 peer-reduction microgate evidence."""

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
CLASSIFICATION_SCHEMA = (
    "qwen38.tp4-peer-reduction-classification.v1"
)
PRODUCER_ARTIFACTS = (
    "source_identity.json",
    "peer_access_matrix.json",
    "ipc_roundtrip.jsonl",
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
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
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
    _write_json(Path(root) / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _validate_source_identity(source):
    if (
        not isinstance(source, dict)
        or source.get("schema_version")
        != "qwen38.tp4-peer-reduction-source.v1"
        or not isinstance(source.get("attempt"), str)
        or not source["attempt"]
        or not isinstance(source.get("source_revision"), str)
        or len(source["source_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source["source_revision"]
        )
        or not isinstance(source.get("source_tree_sha256"), str)
        or len(source["source_tree_sha256"]) != 64
    ):
        raise ValueError("source identity is invalid")
    return dict(source)


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


def _validate_evidence_identity(source, matrix, ipc_rows, rows):
    attempt = source["attempt"]
    revision = source["source_revision"]
    if (
        not isinstance(matrix, dict)
        or matrix.get("attempt") != attempt
        or matrix.get("source_revision") != revision
        or matrix.get("world_size") != 4
        or not isinstance(matrix.get("rows"), list)
    ):
        raise ValueError("peer topology identity is invalid")
    if any(
        not isinstance(row, dict)
        or row.get("attempt") != attempt
        or row.get("source_revision") != revision
        for row in rows
    ):
        raise ValueError("microgate row identity is invalid")


def assemble_bundle(
    *,
    output_root,
    source_identity,
    peer_access_matrix,
    ipc_roundtrip_rows,
    microgate_rows,
    memory_summary,
    cleanup,
):
    source = _validate_source_identity(source_identity)
    _require_finite({
        "source": source,
        "matrix": peer_access_matrix,
        "ipc": ipc_roundtrip_rows,
        "rows": microgate_rows,
        "memory": memory_summary,
        "cleanup": cleanup,
    })
    _validate_evidence_identity(
        source,
        peer_access_matrix,
        ipc_roundtrip_rows,
        microgate_rows,
    )
    try:
        topology = validate_peer_topology(peer_access_matrix["rows"])
        validate_peer_topology(ipc_roundtrip_rows)
        topology_classification = topology["classification"]
    except ValueError:
        topology = {
            "classification": "INELIGIBLE_TOPOLOGY",
            "world_size": 4,
            "directed_peer_edge_count": len(ipc_roundtrip_rows),
        }
        topology_classification = "INELIGIBLE_TOPOLOGY"

    if topology_classification == "PASS":
        gate = classify_peer_microgate(
            microgate_rows,
            _cleanup_for_classifier(cleanup),
            memory_summary,
        )
        classification = gate["classification"]
    else:
        gate = {
            "classification": topology_classification,
            "shape_summaries": [],
        }
        classification = topology_classification

    summary = {
        "schema_version": "qwen38.tp4-peer-reduction-summary.v1",
        "attempt": source["attempt"],
        "source_revision": source["source_revision"],
        "classification": classification,
        "topology": topology,
        "shape_summaries": gate.get("shape_summaries", []),
        "measurement_row_count": len(microgate_rows),
        "timeout_count": sum(
            row.get("timed_out") is True
            for row in microgate_rows
            if isinstance(row, dict)
        ),
        "maximum_allocated_delta_bytes": (
            memory_summary.get("maximum_allocated_delta_bytes")
            if isinstance(memory_summary, dict)
            else None
        ),
        "cleanup_classification": (
            cleanup.get("classification")
            if isinstance(cleanup, dict)
            else None
        ),
    }
    classification_payload = {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "runtime_integration_authorized": classification == "PASS",
    }

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("bundle output directory must be empty")
    _write_json(root / "source_identity.json", source)
    _write_json(root / "peer_access_matrix.json", peer_access_matrix)
    _write_jsonl(root / "ipc_roundtrip.jsonl", ipc_roundtrip_rows)
    _write_jsonl(root / "microgate_rows.jsonl", microgate_rows)
    _write_json(root / "memory_summary.json", memory_summary)
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
        peer_access_matrix=_load_json(
            raw / "peer_access_matrix.json"
        ),
        ipc_roundtrip_rows=_load_jsonl(
            raw / "ipc_roundtrip.jsonl"
        ),
        microgate_rows=_load_jsonl(raw / "microgate_rows.jsonl"),
        memory_summary=_load_json(raw / "memory_summary.json"),
        cleanup=_load_json(raw / "cleanup.json"),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
