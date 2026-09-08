from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.assemble_qwen38_topology_local_tp2_whole_model import (
    assemble_attempt,
)
from tools.test_assemble_qwen38_topology_local_tp2_whole_model import (
    passing_attempt,
)
from tools.verify_qwen38_topology_local_tp2_whole_model import (
    verify_bundle,
)
import tools.verify_qwen38_topology_local_tp2_whole_model as verifier


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _write_jsonl(path, rows):
    Path(path).write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))


def _rewrite_manifest(root):
    manifest_path = root / "manifest.json"
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
        and path.name not in {"manifest.json", "manifest.sha256"}
    }
    manifest = _read_json(manifest_path)
    manifest["files"] = files
    _write_json(manifest_path, manifest)
    (root / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n"
    )


def _bundle(tmp_path):
    attempt = passing_attempt(tmp_path / "attempt")
    output = tmp_path / "bundle"
    assemble_attempt(attempt, output)
    return output


def test_verifier_is_standard_library_only_and_reconstructs_go(tmp_path):
    root = _bundle(tmp_path)
    source = Path(verifier.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    result = verify_bundle(root)

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"
    )
    assert not any(
        name == "torch"
        or name == "tinyvllm"
        or (name or "").startswith("tinyvllm.")
        or (name or "").startswith(
            "tools.assemble_qwen38_topology_local_tp2_whole_model"
        )
        or (name or "").startswith(
            "tools.qwen38_topology_local_tp2_whole_model_worker"
        )
        for name in imported
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("source_sha", "source"),
        ("model_revision", "model"),
        ("epoch_arm", "epoch"),
        ("request_token", "token"),
        ("raw_gap", "classification"),
        ("candidate_hit", "candidate"),
        ("short_chunk", "short"),
        ("collective_bytes", "collective"),
        ("publication", "publication"),
        ("memory_peak", "classification"),
        ("cleanup", "cleanup"),
        ("service_authority", "service"),
        ("classification", "classification"),
    ),
)
def test_verifier_rejects_semantic_mutations(
    tmp_path,
    mutation,
    message,
):
    root = _bundle(tmp_path)
    if mutation == "source_sha":
        payload = _read_json(root / "source_manifest.json")
        payload["source_tree_sha256"] = "c" * 64
        _write_json(root / "source_manifest.json", payload)
    elif mutation == "model_revision":
        payload = _read_json(root / "model_manifest.json")
        payload["model_revision"] = "f" * 40
        _write_json(root / "model_manifest.json", payload)
    elif mutation == "epoch_arm":
        payload = _read_json(root / "campaign_epoch_manifest.json")
        payload["epochs"][1]["arm"] = "baseline"
        _write_json(root / "campaign_epoch_manifest.json", payload)
    elif mutation == "request_token":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["output_token_ids"][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "raw_gap":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["arm"] == "candidate":
                row["requests"][0]["token_gaps_ns"] = [200.0] * 127
                row["requests"][0]["tpot_ns"] = 200.0
        _write_jsonl(path, rows)
    elif mutation == "candidate_hit":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["tp2_decode_calls"] -= 1
        _write_jsonl(path, rows)
    elif mutation == "short_chunk":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["short_chunk_calls"] = 1
        _write_jsonl(path, rows)
    elif mutation == "collective_bytes":
        path = root / "collective_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["pair_local_bytes"] -= 1
        _write_jsonl(path, rows)
    elif mutation == "publication":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["migration_publications"] -= 1
        _write_jsonl(path, rows)
    elif mutation == "memory_peak":
        path = root / "memory_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["peak_allocated_bytes"] = 79 * 1024**3
        _write_jsonl(path, rows)
    elif mutation == "cleanup":
        payload = _read_json(root / "cleanup.json")
        payload["retained_process_groups"] = 1
        _write_json(root / "cleanup.json", payload)
    elif mutation == "service_authority":
        path = root / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["classification_authority"] = True
        _write_jsonl(path, rows)
    elif mutation == "classification":
        payload = _read_json(root / "classification.json")
        payload["classification"] = "NO_GO_PERFORMANCE"
        _write_json(root / "classification.json", payload)
    _rewrite_manifest(root)

    with pytest.raises((ValueError, RuntimeError), match=message):
        verify_bundle(root)


def test_verifier_rejects_manifest_digest_mutation(tmp_path):
    root = _bundle(tmp_path)
    (root / "manifest.sha256").write_text("0" * 64 + "\n")

    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(root)


def test_remote_and_local_receipts_are_byte_identical(tmp_path):
    root = _bundle(tmp_path)
    remote_path = tmp_path / "remote.json"
    local_path = tmp_path / "local.json"

    remote = verify_bundle(root, output_path=remote_path)
    local = verify_bundle(root, output_path=local_path)

    assert remote == local
    assert remote_path.read_bytes() == local_path.read_bytes()
