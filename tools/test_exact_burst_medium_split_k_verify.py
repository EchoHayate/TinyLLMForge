#!/usr/bin/env python3
"""Independent-verifier contracts for medium split-K artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.exact_burst_medium_split_k_gate import (
    GO_EXACT_BURST_MEDIUM_SPLIT_K,
    produce_gate,
)
from tools.exact_burst_medium_split_k_verify import (
    verify_artifact_directory,
)
from tools.test_exact_burst_medium_split_k_gate import (
    _fixture,
    _set_tpot,
    _write_json,
    _write_jsonl,
)


def _complete_artifact(path: Path) -> None:
    _fixture(path)
    result = produce_gate(path)
    assert result["classification"] == (
        GO_EXACT_BURST_MEDIUM_SPLIT_K
    )


def test_independent_verifier_reconstructs_go(
    tmp_path: Path,
) -> None:
    _complete_artifact(tmp_path)
    receipt = verify_artifact_directory(tmp_path)
    assert receipt["verified"] is True
    assert receipt["classification"] == (
        GO_EXACT_BURST_MEDIUM_SPLIT_K
    )
    assert receipt["manifest_verified"] is True
    assert receipt["raw_metrics_reconstructed"] is True


def test_verifier_rejects_manifest_and_summary_tampering(
    tmp_path: Path,
) -> None:
    _complete_artifact(tmp_path)
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text()
    )
    manifest["artifacts"]["summary.json"] = "0" * 64
    _write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest hash"):
        verify_artifact_directory(tmp_path)

    other = tmp_path / "summary"
    other.mkdir()
    _complete_artifact(other)
    summary = json.loads((other / "summary.json").read_text())
    summary["classification"] = "NO_GO_PERFORMANCE"
    _write_json(other / "summary.json", summary)
    manifest = json.loads((other / "manifest.json").read_text())
    from tools.profile_exact_burst_medium_split_k import sha256_file

    manifest["artifacts"]["summary.json"] = sha256_file(
        other / "summary.json"
    )
    _write_json(other / "manifest.json", manifest)
    with pytest.raises(
        ValueError,
        match="reconstructed classification",
    ):
        verify_artifact_directory(other)


def test_verifier_rejects_raw_sample_and_sidecar_tampering(
    tmp_path: Path,
) -> None:
    _complete_artifact(tmp_path)
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "performance_rows.jsonl"
        ).read_text().splitlines()
    ]
    candidate = next(
        row
        for row in rows
        if row["policy"] == "split12"
        and row["context_length"] == 2049
    )
    _set_tpot(candidate, 1_100_000)
    _write_jsonl(tmp_path / "performance_rows.jsonl", rows)
    with pytest.raises(ValueError, match="manifest hash"):
        verify_artifact_directory(tmp_path)

    other = tmp_path / "sidecar"
    other.mkdir()
    _complete_artifact(other)
    correctness = [
        json.loads(line)
        for line in (
            other / "correctness_rows.jsonl"
        ).read_text().splitlines()
    ]
    sidecar = other / correctness[0]["logits_path"]
    sidecar.write_bytes(sidecar.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sidecar"):
        verify_artifact_directory(other)


def test_verifier_does_not_import_producer_classification() -> None:
    source = Path(
        "tools/exact_burst_medium_split_k_verify.py"
    ).read_text(encoding="utf-8")
    assert "exact_burst_medium_split_k_gate" not in source
