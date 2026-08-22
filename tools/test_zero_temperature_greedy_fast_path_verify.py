#!/usr/bin/env python3
"""Independent-verifier tests for greedy fast-path evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.test_zero_temperature_greedy_fast_path_gate import (
    write_fixture_bundle,
)
from tools.zero_temperature_greedy_fast_path_gate import (
    produce_gate,
)
from tools.zero_temperature_greedy_fast_path_verify import (
    verify_bundle,
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


def test_independent_verifier_reconstructs_go_fixture() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        result = verify_bundle(
            run_dir,
            repo_root=repo_root,
        )
        assert result["status"] == "PASS"
        assert result["reconstructed_classification"] == (
            "GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH"
        )
        assert len(result["comparison_sha256"]) == 64
        assert len(result["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    "artifact",
    ("comparison.json", "gate.json"),
)
def test_independent_verifier_rejects_producer_drift(
    artifact,
) -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        payload = json.loads(
            (run_dir / artifact).read_text(encoding="utf-8")
        )
        if artifact == "comparison.json":
            payload["aggregate"][
                "tpot_p95_improvement_fraction"
            ] = 0.99
        else:
            payload["classification"] = "NO_GO_TPOT_MEDIAN"
        _write_json(run_dir / artifact, payload)
        with pytest.raises(
            ValueError,
            match="drift|manifest digest mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)


def test_independent_verifier_rejects_missing_or_stale_sidecar() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        correctness_rows = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        (run_dir / correctness_rows[0]["logits_path"]).unlink()
        with pytest.raises(
            ValueError,
            match="missing|manifest",
        ):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        correctness_rows = [
            json.loads(line)
            for line in (
                run_dir / "correctness_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        path = run_dir / correctness_rows[0]["logits_path"]
        path.write_bytes(path.read_bytes() + b"\x00")
        with pytest.raises(
            ValueError,
            match="digest mismatch|byte length mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)


def test_independent_verifier_rejects_manifest_inventory_drift() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        first_sidecar = next(
            name
            for name in manifest["artifacts"]
            if name.startswith("logits/")
        )
        del manifest["artifacts"][first_sidecar]
        _write_json(run_dir / "manifest.sha256", manifest)
        with pytest.raises(
            ValueError,
            match="manifest file inventory mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        manifest["artifacts"]["case_rows.jsonl"] = "0" * 64
        _write_json(run_dir / "manifest.sha256", manifest)
        with pytest.raises(
            ValueError,
            match="manifest digest mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)


def main() -> None:
    test_independent_verifier_reconstructs_go_fixture()
    for artifact in ("comparison.json", "gate.json"):
        test_independent_verifier_rejects_producer_drift(artifact)
    test_independent_verifier_rejects_missing_or_stale_sidecar()
    test_independent_verifier_rejects_manifest_inventory_drift()
    print("zero-temperature greedy verifier tests passed")


if __name__ == "__main__":
    main()
