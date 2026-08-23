#!/usr/bin/env python3
"""Independent-verifier contracts for split-phase K8 evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.exact_burst_split_phase_gate import produce_gate
from tools.exact_burst_split_phase_verify import verify_bundle
from tools.test_exact_burst_split_phase_gate import write_fixture_bundle


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


def _refresh_manifest(run_dir: Path, relative: str) -> None:
    path = run_dir / "manifest.sha256"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"][relative] = hashlib.sha256(
        (run_dir / relative).read_bytes()
    ).hexdigest()
    _write_json(path, manifest)


def _assert_raises(message: str, callback) -> None:
    try:
        callback()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def _ready_bundle(root: Path) -> tuple[Path, Path]:
    run_dir, repo_root = write_fixture_bundle(root)
    produce_gate(run_dir, repo_root=repo_root)
    return run_dir, repo_root


def test_verifier_is_independent_and_reconstructs_every_metric() -> None:
    source = (
        REPO_ROOT / "tools/exact_burst_split_phase_verify.py"
    ).read_text(encoding="utf-8")
    assert "exact_burst_split_phase_gate import" not in source
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        result = verify_bundle(run_dir, repo_root=repo_root)
        assert result["status"] == "PASS"
        assert result["reconstructed_classification"] == (
            "GO_EXACT_BURST_SPLIT_PHASE"
        )
        assert result["reconstructed_selected_policy"] == (
            "decode_burst_k8_split_phase"
        )
        assert result["performance_row_count"] == 60
        assert result["correctness_row_count"] == 48
        assert result["maximum_metric_disagreement"] <= 1e-9
        assert (run_dir / "independent-verification.json").is_file()


def test_verifier_rejects_producer_disagreement_and_manifest_tamper() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        comparison_path = run_dir / "comparison.json"
        comparison = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
        comparison["candidate_evaluation"]["aggregate"][
            "k8_vs_split"
        ]["tpot_median_regression_fraction"] += 1e-8
        _write_json(comparison_path, comparison)
        _refresh_manifest(run_dir, "comparison.json")
        gate_path = run_dir / "gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["comparison_sha256"] = hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest()
        _write_json(gate_path, gate)
        _refresh_manifest(run_dir, "gate.json")
        _assert_raises(
            "metric disagreement",
            lambda: verify_bundle(run_dir, repo_root=repo_root),
        )

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        gate_path = run_dir / "gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["classification"] = (
            "NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE"
        )
        _write_json(gate_path, gate)
        _refresh_manifest(run_dir, "gate.json")
        _assert_raises(
            "classification drift",
            lambda: verify_bundle(run_dir, repo_root=repo_root),
        )

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        (run_dir / "source.patch").write_text(
            "tampered after manifest\n",
            encoding="utf-8",
        )
        _assert_raises(
            "manifest digest mismatch",
            lambda: verify_bundle(run_dir, repo_root=repo_root),
        )


def test_verifier_accepts_metric_rounding_within_one_e_minus_nine() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        comparison_path = run_dir / "comparison.json"
        comparison = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
        comparison["candidate_evaluation"]["aggregate"][
            "k8_vs_split"
        ]["tpot_median_regression_fraction"] += 5e-10
        _write_json(comparison_path, comparison)
        _refresh_manifest(run_dir, "comparison.json")
        gate_path = run_dir / "gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["comparison_sha256"] = hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest()
        _write_json(gate_path, gate)
        _refresh_manifest(run_dir, "gate.json")
        result = verify_bundle(run_dir, repo_root=repo_root)
        assert 0.0 < result["maximum_metric_disagreement"] <= 1e-9


def main() -> None:
    test_verifier_is_independent_and_reconstructs_every_metric()
    test_verifier_rejects_producer_disagreement_and_manifest_tamper()
    test_verifier_accepts_metric_rounding_within_one_e_minus_nine()
    print("exact burst split-phase verifier tests passed")


if __name__ == "__main__":
    main()
