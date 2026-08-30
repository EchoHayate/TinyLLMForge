from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from tools.assemble_qwen38_tp4_peer_reduction_microgate import (
    MANIFEST_SCHEMA,
    assemble_bundle,
)
from tools.test_assemble_qwen38_tp4_peer_reduction_microgate import (
    _inputs,
)
from tools.verify_qwen38_tp4_peer_reduction_microgate import (
    verify_bundle,
)
import tools.verify_qwen38_tp4_peer_reduction_microgate as verifier_module


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
    _write_json(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def test_verifier_independently_reconstructs_pass_and_manifest(tmp_path):
    assemble_bundle(output_root=tmp_path, **_inputs())

    result = verify_bundle(tmp_path)

    assert result["status"] == "PASS"
    assert result["producer_classification"] == "PASS"
    assert result["reconstructed_classification"] == "PASS"
    assert result["directed_peer_edge_count"] == 12
    assert result["measurement_row_count"] == 2400
    assert result["artifact_hashes_verified"] is True


def test_verifier_does_not_import_producer_assembler():
    source = inspect.getsource(verifier_module)

    assert "import assemble_qwen38_tp4_peer_reduction_microgate" not in source
    assert (
        "from tools.assemble_qwen38_tp4_peer_reduction_microgate"
        not in source
    )


def test_verifier_rejects_manifest_hash_mutation(tmp_path):
    assemble_bundle(output_root=tmp_path, **_inputs())
    path = tmp_path / "memory_summary.json"
    payload = json.loads(path.read_text())
    payload["maximum_allocated_delta_bytes"] += 1
    _write_json(path, payload)

    with pytest.raises(ValueError, match="manifest artifact hash"):
        verify_bundle(tmp_path)


def test_verifier_rejects_producer_classification_disagreement(tmp_path):
    assemble_bundle(output_root=tmp_path, **_inputs())
    path = tmp_path / "classification.json"
    payload = json.loads(path.read_text())
    payload["classification"] = "NO_GO_MICROGATE"
    _write_json(path, payload)
    _rewrite_manifest(tmp_path)

    with pytest.raises(ValueError, match="producer classification"):
        verify_bundle(tmp_path)
