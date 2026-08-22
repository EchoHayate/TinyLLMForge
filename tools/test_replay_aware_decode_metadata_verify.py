"""Independent-verifier tests for replay-aware metadata evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.replay_aware_decode_metadata_gate import (
    produce_gate,
)
from tools.replay_aware_decode_metadata_verify import (
    verify_bundle,
)
from tools.test_replay_aware_decode_metadata_gate import (
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


def test_independent_verifier_reconstructs_go_fixture():
    with TemporaryDirectory() as tmp:
        run_dir, repo_root = _ready_bundle(Path(tmp))

        result = verify_bundle(
            run_dir,
            repo_root=repo_root,
        )

        assert result["status"] == "PASS"
        assert result["reconstructed_classification"] == (
            "GO_REPLAY_AWARE_METADATA"
        )
        assert len(result["comparison_sha256"]) == 64
        assert len(result["manifest_sha256"]) == 64


def test_independent_verifier_rejects_comparison_drift():
    with TemporaryDirectory() as tmp:
        run_dir, repo_root = _ready_bundle(Path(tmp))
        comparison = json.loads(
            (run_dir / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        comparison["aggregate"][
            "tpot_p95_improvement_fraction"
        ] = 0.99
        _write_json(
            run_dir / "comparison.json",
            comparison,
        )

        with pytest.raises(
            ValueError,
            match="comparison drift|manifest digest mismatch",
        ):
            verify_bundle(
                run_dir,
                repo_root=repo_root,
            )


def test_independent_verifier_rejects_classification_drift():
    with TemporaryDirectory() as tmp:
        run_dir, repo_root = _ready_bundle(Path(tmp))
        gate = json.loads(
            (run_dir / "gate.json").read_text(
                encoding="utf-8"
            )
        )
        gate["classification"] = "NO_GO_TPOT_MEDIAN"
        _write_json(run_dir / "gate.json", gate)

        with pytest.raises(
            ValueError,
            match="classification drift|manifest digest mismatch",
        ):
            verify_bundle(
                run_dir,
                repo_root=repo_root,
            )


def test_independent_verifier_rejects_omitted_primary_artifact():
    with TemporaryDirectory() as tmp:
        run_dir, repo_root = _ready_bundle(Path(tmp))
        (run_dir / "summary.json").unlink()

        with pytest.raises(
            ValueError,
            match="primary artifact is missing",
        ):
            verify_bundle(
                run_dir,
                repo_root=repo_root,
            )


def test_independent_verifier_rejects_stale_manifest_digest():
    with TemporaryDirectory() as tmp:
        run_dir, repo_root = _ready_bundle(Path(tmp))
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        manifest["artifacts"]["case_rows.jsonl"] = "0" * 64
        _write_json(
            run_dir / "manifest.sha256",
            manifest,
        )

        with pytest.raises(
            ValueError,
            match="manifest digest mismatch",
        ):
            verify_bundle(
                run_dir,
                repo_root=repo_root,
            )


def main() -> None:
    test_independent_verifier_reconstructs_go_fixture()
    test_independent_verifier_rejects_comparison_drift()
    test_independent_verifier_rejects_classification_drift()
    test_independent_verifier_rejects_omitted_primary_artifact()
    test_independent_verifier_rejects_stale_manifest_digest()
    print("replay-aware metadata verifier tests passed")


if __name__ == "__main__":
    main()
