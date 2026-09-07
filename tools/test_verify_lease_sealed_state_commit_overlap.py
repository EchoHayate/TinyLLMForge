from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from tools.assemble_lease_sealed_state_commit_overlap import (
    MANIFEST_SCHEMA,
    assemble_bundle,
)
from tools.test_assemble_lease_sealed_state_commit_overlap import (
    clone_inputs,
)
from tools.verify_lease_sealed_state_commit_overlap import verify_bundle
import tools.verify_lease_sealed_state_commit_overlap as verifier_module


def _write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def rewrite_manifest(root):
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
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


def test_verifier_reconstructs_go_without_importing_assembler(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    result = verify_bundle(tmp_path)

    assert result["status"] == "PASS"
    assert result["producer_classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert result["reconstructed_classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert result["measurement_row_count"] == 180
    assert result["artifact_hashes_verified"] is True
    source = inspect.getsource(verifier_module)
    assert "assemble_lease_sealed_state_commit_overlap" not in source


def test_verifier_rejects_hash_mutation_extra_file_and_missing_row(tmp_path):
    hashed = tmp_path / "hashed"
    assemble_bundle(output_root=hashed, **clone_inputs())
    (hashed / "memory_rows.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest artifact hash"):
        verify_bundle(hashed)

    extra = tmp_path / "extra"
    assemble_bundle(output_root=extra, **clone_inputs())
    (extra / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="artifact inventory"):
        verify_bundle(extra)

    missing = tmp_path / "missing"
    assemble_bundle(output_root=missing, **clone_inputs())
    for name in (
        "paired_rows.jsonl",
        "correctness_rows.jsonl",
        "overlap_rows.jsonl",
    ):
        rows_path = missing / name
        rows = rows_path.read_text().splitlines()
        rows_path.write_text("\n".join(rows[:-1]) + "\n")
    rewrite_manifest(missing)
    with pytest.raises(ValueError, match="producer classification"):
        verify_bundle(missing)


def test_verifier_rejects_producer_summary_disagreement(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    producer_path = tmp_path / "producer_result.json"
    producer = json.loads(producer_path.read_text())
    producer["classification"] = "NO_GO_PERFORMANCE"
    _write_json(producer_path, producer)
    rewrite_manifest(tmp_path)

    with pytest.raises(ValueError, match="producer classification"):
        verify_bundle(tmp_path)
