from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from tools.assemble_cross_request_wavefront_microgate import (
    MANIFEST_SCHEMA,
    assemble_bundle,
)
from tools.test_assemble_cross_request_wavefront_microgate import (
    clone_inputs,
)
from tools.verify_cross_request_wavefront_microgate import verify_bundle
import tools.verify_cross_request_wavefront_microgate as verifier_module


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


def _rewrite_manifest(root):
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


def test_verifier_independently_reconstructs_go_and_manifest(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())

    result = verify_bundle(tmp_path)

    assert result["status"] == "PASS"
    assert result["producer_classification"] == (
        "GO_WAVEFRONT_MICROGATE"
    )
    assert result["reconstructed_classification"] == (
        "GO_WAVEFRONT_MICROGATE"
    )
    assert result["measurement_row_count"] == 2400
    assert result["artifact_hashes_verified"] is True
    assert (tmp_path / "independent_verification.json").is_file()


def test_verifier_does_not_import_producer_assembler():
    source = inspect.getsource(verifier_module)

    assert "import assemble_cross_request_wavefront_microgate" not in source
    assert (
        "from tools.assemble_cross_request_wavefront_microgate"
        not in source
    )


def test_verifier_rejects_manifest_hash_mutation(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    path = tmp_path / "memory_summary.json"
    payload = json.loads(path.read_text())
    payload["maximum_allocated_delta_bytes"] += 1
    _write_json(path, payload)

    with pytest.raises(ValueError, match="manifest artifact hash"):
        verify_bundle(tmp_path)


def test_verifier_rejects_producer_classification_disagreement(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    path = tmp_path / "classification.json"
    payload = json.loads(path.read_text())
    payload["classification"] = "NO_GO_PERFORMANCE"
    _write_json(path, payload)
    _rewrite_manifest(tmp_path)

    with pytest.raises(ValueError, match="producer classification"):
        verify_bundle(tmp_path)


def test_verifier_rejects_summary_disagreement(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    path = tmp_path / "microgate_summary.json"
    payload = json.loads(path.read_text())
    payload["measurement_row_count"] -= 1
    _write_json(path, payload)
    _rewrite_manifest(tmp_path)

    with pytest.raises(ValueError, match="microgate summary"):
        verify_bundle(tmp_path)


def test_verifier_rejects_extra_file(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    (tmp_path / "unexpected.txt").write_text("unexpected")

    with pytest.raises(ValueError, match="manifest artifact inventory"):
        verify_bundle(tmp_path)


def test_verifier_rejects_missing_row(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    path = tmp_path / "microgate_rows.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    _rewrite_manifest(tmp_path)

    with pytest.raises(ValueError, match="producer classification"):
        verify_bundle(tmp_path)


def test_verifier_rejects_rank_digest_disagreement(tmp_path):
    assemble_bundle(output_root=tmp_path, **clone_inputs())
    path = tmp_path / "microgate_rows.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["cohort_digest"] = "0" * 64
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    _rewrite_manifest(tmp_path)

    with pytest.raises(ValueError, match="policy identity"):
        verify_bundle(tmp_path)
