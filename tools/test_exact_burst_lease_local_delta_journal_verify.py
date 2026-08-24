from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

from tools.test_exact_burst_lease_local_delta_journal_gate import (
    write_fixture_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "exact_burst_lease_local_delta_journal_verify.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "exact_burst_lease_local_delta_journal_verify_under_test",
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


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _refresh_digest(run_dir: Path, name: str) -> None:
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][name] = hashlib.sha256(
        (run_dir / name).read_bytes()
    ).hexdigest()
    _write_json(manifest_path, manifest)


def test_verifier_is_independent_and_reconstructs_gate(tmp_path):
    source = MODULE_PATH.read_text()
    assert (
        "exact_burst_lease_local_delta_journal_gate import"
        not in source
    )
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path)

    result = verifier.verify_artifact_directory(run_dir)

    assert result == {
        "schema": (
            "exact_burst_lease_local_delta_journal_verify_v1"
        ),
        "verified": True,
        "classification": (
            "GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL"
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
                    "output_tokens": [999]
                    if row["policy"] == "lease_local_delta"
                    and row["context"] == "short"
                    and row["repetition"] == 0
                    else row["output_tokens"],
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
                    "sampled_logits": [9.0]
                    if row["policy"] == "lease_local_delta"
                    and row["context"] == "short"
                    and row["sampling_point"] == "prefill-final"
                    else row["sampled_logits"],
                }
                for row in rows
            ],
            "summary mismatch",
        ),
        (
            "performance_rows.jsonl",
            lambda rows: [
                {
                    **row,
                    "delta_rollbacks": 1
                    if row["policy"] == "lease_local_delta"
                    and row["context"] == "short"
                    and row["repetition"] == 0
                    else row["delta_rollbacks"],
                }
                for row in rows
            ],
            "summary mismatch",
        ),
        (
            "performance_rows.jsonl",
            lambda rows: [
                {
                    **row,
                    "d2h_bytes": row["d2h_bytes"] + 1
                    if row["policy"] == "lease_local_delta"
                    and row["context"] == "short"
                    and row["repetition"] == 0
                    else row["d2h_bytes"],
                }
                for row in rows
            ],
            "summary mismatch",
        ),
        (
            "performance_rows.jsonl",
            lambda rows: [
                {
                    **row,
                    "delta_fallbacks": {"unknown_reason": 1}
                    if row["policy"] == "lease_local_delta"
                    and row["context"] == "short"
                    and row["repetition"] == 0
                    else row["delta_fallbacks"],
                }
                for row in rows
            ],
            "unknown fallback reason",
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


def test_verifier_rejects_source_and_summary_tamper(tmp_path):
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha"] = "c" * 40
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


def test_verifier_rejects_non_empty_source_patch_digest(tmp_path):
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_patch_sha256"] = "b" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="source patch digest"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_non_finite_value(tmp_path):
    verifier = _load_module()
    run_dir = write_fixture_bundle(tmp_path)
    path = run_dir / "performance_rows.jsonl"
    rows = _rows(path)
    rows[0]["ttft_ns"] = math.nan
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=True) + "\n"
            for row in rows
        )
    )
    _refresh_digest(run_dir, path.name)

    with pytest.raises(ValueError, match="non-finite"):
        verifier.verify_artifact_directory(run_dir)
