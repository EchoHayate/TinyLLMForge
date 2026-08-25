from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

from tools.test_exact_burst_generation_sealed_lease_identity_gate import (
    _correctness_rows,
    _load_module as _load_gate,
    _performance_rows,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "exact_burst_generation_sealed_lease_identity_verify.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "exact_burst_generation_sealed_lease_identity_verify_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    )


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_digest(run_dir: Path, name: str) -> None:
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][name] = _sha256(run_dir / name)
    _write_json(manifest_path, manifest)


def write_fixture_bundle(root: Path) -> Path:
    gate = _load_gate()
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    performance = _performance_rows(gate)
    correctness = _correctness_rows(gate)
    summary = gate.summarize_evidence(performance, correctness)
    workload = gate.build_workload_manifest(
        run_tag="synthetic-r1",
        source_sha="a" * 40,
    )
    lifecycle = [
        {
            "policy": row["policy"],
            "context": row["context"],
            "repetition": row["repetition"],
            "lease_grant_ns": row["lease_grant_ns"],
            "scheduler_lifecycle_ns": row[
                "scheduler_lifecycle_ns"
            ],
        }
        for row in performance
    ]
    _write_json(run_dir / "workload_manifest.json", workload)
    _write_jsonl(run_dir / "performance_rows.jsonl", performance)
    _write_jsonl(run_dir / "correctness_rows.jsonl", correctness)
    _write_jsonl(run_dir / "lifecycle_samples.jsonl", lifecycle)
    _write_json(run_dir / "summary.json", summary)
    artifact_names = (
        "workload_manifest.json",
        "performance_rows.jsonl",
        "correctness_rows.jsonl",
        "lifecycle_samples.jsonl",
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema": gate.SOURCE_MANIFEST_SCHEMA,
            "run_tag": "synthetic-r1",
            "source_sha": "a" * 40,
            "source_patch_sha256": hashlib.sha256(b"").hexdigest(),
            "source_file_sha256": {
                relative: _sha256(ROOT / relative)
                for relative in gate.SOURCE_FILES
            },
            "artifact_sha256": {
                name: _sha256(run_dir / name)
                for name in artifact_names
            },
        },
    )
    _write_json(
        run_dir / "runner_receipt.json",
        {
            "schema": gate.RUNNER_RECEIPT_SCHEMA,
            "run_tag": "synthetic-r1",
            "source_sha": "a" * 40,
            "performance_rows": gate.PERFORMANCE_ROW_COUNT,
            "correctness_rows": gate.CORRECTNESS_ROW_COUNT,
            "classification": summary["classification"],
        },
    )
    return run_dir


def test_verifier_is_independent_and_reconstructs_gate(tmp_path):
    source = MODULE_PATH.read_text()
    assert (
        "exact_burst_generation_sealed_lease_identity_gate import"
        not in source
    )
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path)

    result = verifier.verify_artifact_directory(run_dir)

    assert result == {
        "schema": (
            "exact_burst_generation_sealed_lease_identity_verify_v1"
        ),
        "verified": True,
        "classification": (
            "GO_EXACT_BURST_GENERATION_SEALED_LEASE_IDENTITY"
        ),
        "performance_row_count": 60,
        "correctness_row_count": 24,
    }


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    (
        (
            "performance_rows.jsonl",
            lambda rows: rows[:-1],
            "performance row inventory",
        ),
        (
            "correctness_rows.jsonl",
            lambda rows: rows + [dict(rows[0])],
            "duplicate correctness row",
        ),
        (
            "performance_rows.jsonl",
            lambda rows: [
                {
                    **row,
                    "identity_seal_hot_reuses": 0
                    if row["policy"] == "generation_sealed"
                    and row["context"] == "8k"
                    and row["repetition"] == 0
                    else row["identity_seal_hot_reuses"],
                }
                for row in rows
            ],
            "summary mismatch",
        ),
        (
            "correctness_rows.jsonl",
            lambda rows: [
                {
                    **row,
                    "sampled_logits": [1.0, 2.0, 9.0]
                    if row["policy"] == "generation_sealed"
                    and row["context"] == "8k"
                    and row["sampling_point"] == "decode-final"
                    else row["sampled_logits"],
                }
                for row in rows
            ],
            "summary mismatch",
        ),
    ),
)
def test_verifier_rejects_tampered_rows(
    tmp_path,
    target,
    mutate,
    message,
):
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path)
    path = run_dir / target
    _write_jsonl(path, mutate(_rows(path)))
    _refresh_digest(run_dir, target)

    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_missing_non_finite_and_bad_order(tmp_path):
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path / "missing")
    (run_dir / "lifecycle_samples.jsonl").unlink()
    with pytest.raises(ValueError, match="required artifact"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "non-finite")
    path = run_dir / "performance_rows.jsonl"
    rows = _rows(path)
    rows[0]["ttft_ns"] = math.nan
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows)
    )
    _refresh_digest(run_dir, path.name)
    with pytest.raises(ValueError, match="non-finite"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "order")
    workload_path = run_dir / "workload_manifest.json"
    workload = json.loads(workload_path.read_text())
    workload["policy_order"]["0"]["2k"].reverse()
    _write_json(workload_path, workload)
    _refresh_digest(run_dir, workload_path.name)
    with pytest.raises(ValueError, match="policy order"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_authority_summary_and_source_tamper(
    tmp_path,
):
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path / "source")
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha"] = "b" * 40
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source SHA"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "summary")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"] = "NO_GO_PERFORMANCE"
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="summary mismatch"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "patch")
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_patch_sha256"] = "b" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source patch digest"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "drift")
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative = next(iter(manifest["source_file_sha256"]))
    manifest["source_file_sha256"][relative] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source file digest"):
        verifier.verify_artifact_directory(run_dir)
