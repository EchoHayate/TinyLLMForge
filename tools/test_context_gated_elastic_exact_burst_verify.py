from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import context_gated_elastic_exact_burst_gate as gate
from tools import context_gated_elastic_exact_burst_verify as verifier
from tools.test_context_gated_elastic_exact_burst_gate import (
    ROOT,
    _write_json,
    _write_jsonl,
    write_fixture_bundle,
)


def _produce(root: Path) -> Path:
    run_dir = write_fixture_bundle(root)
    gate.produce_artifacts(run_dir, source_root=ROOT)
    return run_dir


def _refresh_manifest_digest(run_dir: Path, relative: str) -> None:
    path = run_dir / "terminal_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_sha256"][relative] = verifier.sha256_file(
        run_dir / relative
    )
    _write_json(path, manifest)


def test_verifier_is_independent_and_reconstructs_raw_evidence(
    tmp_path: Path,
) -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "context_gated_elastic_exact_burst_gate import" not in source
    assert "profile_context_gated_elastic_exact_burst import" not in source
    run_dir = _produce(tmp_path)

    result = verifier.verify_artifact_directory(
        run_dir,
        source_root=ROOT,
    )

    assert result == {
        "schema_version": verifier.VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "run_tag": "20260825-context-gated-elastic-k16-fixture",
        "source_commit": "a" * 40,
        "classification": (
            "GO_CONTEXT_GATED_ELASTIC_EXACT_BURST"
        ),
        "performance_row_count": 40,
        "correctness_row_count": 32,
    }


@pytest.mark.parametrize(
    "relative",
    (
        "terminal_summary.json",
        "terminal_gate.json",
        "producer_receipt.json",
    ),
)
def test_verifier_rejects_derived_artifact_tamper(
    tmp_path: Path,
    relative: str,
) -> None:
    run_dir = _produce(tmp_path)
    payload = json.loads((run_dir / relative).read_text())
    payload["classification"] = "NO_GO_CORRECTNESS"
    _write_json(run_dir / relative, payload)
    with pytest.raises(ValueError, match="manifest digest"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )

def test_verifier_rejects_sidecar_and_row_inventory_tamper(
    tmp_path: Path,
) -> None:
    run_dir = _produce(tmp_path / "sidecar")
    sidecar = next((run_dir / "logits").glob("*.f32"))
    sidecar.write_bytes(sidecar.read_bytes() + b"x")
    with pytest.raises(ValueError, match="manifest digest"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )

    run_dir = _produce(tmp_path / "rows")
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ][:-1]
    _write_jsonl(path, rows)
    _refresh_manifest_digest(run_dir, path.name)
    with pytest.raises(ValueError, match="performance row inventory"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )


def test_verifier_rejects_negative_performance_measurement(
    tmp_path: Path,
) -> None:
    run_dir = _produce(tmp_path)
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    rows[0]["decode_host_ns"] = -1
    _write_jsonl(path, rows)
    _refresh_manifest_digest(run_dir, path.name)

    with pytest.raises(ValueError):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )


def test_verifier_rejects_non_integer_output_tokens_even_when_paired(
    tmp_path: Path,
) -> None:
    run_dir = _produce(tmp_path)
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["repetition"] == 0 and row["context_length"] == 256:
            row["output_token_ids"] = (
                [True] * gate.profile.GENERATED_TOKENS
            )
    _write_jsonl(path, rows)
    _refresh_manifest_digest(run_dir, path.name)

    with pytest.raises(ValueError, match="output inventory"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )


def test_verifier_rejects_fixed_k8_elastic_counter_tamper(
    tmp_path: Path,
) -> None:
    run_dir = _produce(tmp_path)
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    fixed = next(row for row in rows if row["policy"] == "fixed_k8")
    fixed["exact_greedy_decode_burst_summary"]["k16_attempts"] = 1
    _write_jsonl(path, rows)
    _refresh_manifest_digest(run_dir, path.name)

    with pytest.raises(ValueError, match="terminal summary"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )


def test_verifier_rejects_source_hash_and_producer_receipt_tamper(
    tmp_path: Path,
) -> None:
    run_dir = _produce(tmp_path / "source")
    path = run_dir / "terminal_source_manifest.json"
    payload = json.loads(path.read_text())
    relative = next(iter(payload["source_sha256"]))
    payload["source_sha256"][relative] = "0" * 64
    _write_json(path, payload)
    _refresh_manifest_digest(run_dir, path.name)
    with pytest.raises(ValueError, match="source hash"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )

    run_dir = _produce(tmp_path / "receipt")
    path = run_dir / "producer_receipt.json"
    payload = json.loads(path.read_text())
    payload["performance_row_count"] = 39
    _write_json(path, payload)
    _refresh_manifest_digest(run_dir, path.name)
    with pytest.raises(ValueError, match="producer receipt"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )


def test_verifier_rejects_workload_execution_contract_tamper(
    tmp_path: Path,
) -> None:
    run_dir = _produce(tmp_path)
    path = run_dir / "workload_manifest.json"
    payload = json.loads(path.read_text())
    payload["device"] = "cpu"
    _write_json(path, payload)
    _refresh_manifest_digest(run_dir, path.name)

    with pytest.raises(ValueError, match="workload manifest"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )
