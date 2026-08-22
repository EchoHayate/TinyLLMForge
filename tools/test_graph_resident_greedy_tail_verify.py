#!/usr/bin/env python3
"""Independent-verifier tests for graph-resident greedy-tail evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.graph_resident_greedy_tail_gate import produce_gate
from tools.graph_resident_greedy_tail_verify import verify_bundle
from tools.test_graph_resident_greedy_tail_gate import (
    write_fixture_bundle,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _ready_bundle(root: Path) -> tuple[Path, Path]:
    run_dir, repo_root = write_fixture_bundle(root)
    produce_gate(run_dir, repo_root=repo_root)
    return run_dir, repo_root


def _refresh_manifest_entry(run_dir: Path, name: str) -> None:
    manifest_path = run_dir / "manifest.sha256"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["artifacts"][name] = hashlib.sha256(
        (run_dir / name).read_bytes()
    ).hexdigest()
    _write_json(manifest_path, manifest)


def test_independent_verifier_reconstructs_go_fixture() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        result = verify_bundle(
            run_dir,
            repo_root=repo_root,
        )
        assert result["status"] == "PASS"
        assert result["reconstructed_classification"] == (
            "GO_GRAPH_RESIDENT_GREEDY_TAIL"
        )
        assert len(result["comparison_sha256"]) == 64
        assert len(result["manifest_sha256"]) == 64


def test_independent_verifier_rejects_comparison_and_gate_drift() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        comparison = json.loads(
            (run_dir / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        comparison["aggregate"]["legacy_vs_graph"][
            "tpot_p95_improvement_fraction"
        ] = 0.99
        _write_json(run_dir / "comparison.json", comparison)
        _refresh_manifest_entry(run_dir, "comparison.json")
        with pytest.raises(ValueError, match="comparison drift"):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        gate = json.loads(
            (run_dir / "gate.json").read_text(encoding="utf-8")
        )
        gate["classification"] = "NO_GO_LEGACY_TPOT_MEDIAN"
        _write_json(run_dir / "gate.json", gate)
        _refresh_manifest_entry(run_dir, "gate.json")
        with pytest.raises(ValueError, match="classification drift"):
            verify_bundle(run_dir, repo_root=repo_root)


def test_independent_verifier_rejects_sidecar_tamper() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        correctness = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        (run_dir / correctness[0]["logits_path"]).unlink()
        with pytest.raises(ValueError, match="missing|manifest"):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        correctness = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        path = run_dir / correctness[0]["logits_path"]
        path.write_bytes(path.read_bytes() + b"\x00")
        with pytest.raises(
            ValueError,
            match="digest mismatch|byte length mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)


def test_independent_verifier_rejects_manifest_drift() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        manifest_path = run_dir / "manifest.sha256"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        sidecar = next(
            name
            for name in manifest["artifacts"]
            if name.startswith("logits/")
        )
        del manifest["artifacts"][sidecar]
        _write_json(manifest_path, manifest)
        with pytest.raises(
            ValueError,
            match="manifest file inventory mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        manifest_path = run_dir / "manifest.sha256"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["artifacts"]["case_rows.jsonl"] = "0" * 64
        _write_json(manifest_path, manifest)
        with pytest.raises(
            ValueError,
            match="manifest digest mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)


def test_independent_verifier_rejects_threshold_semantic_drift() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        comparison = json.loads(
            (run_dir / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        comparison["thresholds"][
            "host_aggregate_median_min_improvement_fraction"
        ] = 0.0
        _write_json(run_dir / "comparison.json", comparison)
        _refresh_manifest_entry(run_dir, "comparison.json")

        gate = json.loads(
            (run_dir / "gate.json").read_text(encoding="utf-8")
        )
        gate["comparison_sha256"] = hashlib.sha256(
            (run_dir / "comparison.json").read_bytes()
        ).hexdigest()
        _write_json(run_dir / "gate.json", gate)
        _refresh_manifest_entry(run_dir, "gate.json")
        with pytest.raises(ValueError, match="comparison drift"):
            verify_bundle(run_dir, repo_root=repo_root)


def main() -> None:
    test_independent_verifier_reconstructs_go_fixture()
    test_independent_verifier_rejects_comparison_and_gate_drift()
    test_independent_verifier_rejects_sidecar_tamper()
    test_independent_verifier_rejects_manifest_drift()
    test_independent_verifier_rejects_threshold_semantic_drift()
    print("graph-resident greedy-tail verifier tests passed")


if __name__ == "__main__":
    main()
