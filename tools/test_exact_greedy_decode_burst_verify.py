#!/usr/bin/env python3
"""Independent-verifier tests for exact greedy decode-burst evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

try:
    import pytest
except ModuleNotFoundError:
    class _Raises:
        def __init__(self, expected, *, match=None):
            self.expected = expected
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, _traceback):
            if exception_type is None:
                raise AssertionError(
                    f"did not raise {self.expected!r}"
                )
            if not issubclass(exception_type, self.expected):
                return False
            if (
                self.match is not None
                and re.search(self.match, str(exception)) is None
            ):
                raise AssertionError(
                    f"{exception!r} does not match {self.match!r}"
                )
            return True

    class _PytestCompat:
        @staticmethod
        def raises(expected, *, match=None):
            return _Raises(expected, match=match)

    pytest = _PytestCompat()


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.exact_greedy_decode_burst_gate import produce_gate
from tools.exact_greedy_decode_burst_verify import verify_bundle
from tools.test_exact_greedy_decode_burst_gate import (
    make_fixture_rows,
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


def test_verifier_is_independent_and_reconstructs_go() -> None:
    source = (
        REPO_ROOT / "tools/exact_greedy_decode_burst_verify.py"
    ).read_text(encoding="utf-8")
    assert "exact_greedy_decode_burst_gate import" not in source
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        result = verify_bundle(run_dir, repo_root=repo_root)
        assert result["status"] == "PASS"
        assert result["reconstructed_classification"] == (
            "GO_EXACT_GREEDY_DECODE_BURST"
        )
        assert result["reconstructed_selected_policy"] == (
            "decode_burst_k8"
        )
        assert len(result["comparison_sha256"]) == 64
        assert len(result["manifest_sha256"]) == 64
        assert result["performance_row_count"] == 60
        assert result["correctness_row_count"] == 48

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(
            Path(temporary),
            correctness_mutation=lambda bucket, policy, point, values: (
                [1.0, 2.0, 5.3, 3.0]
                if policy == "decode_burst_k8"
                and point == "decode-middle"
                else values
            ),
        )
        produce_gate(run_dir, repo_root=repo_root)
        result = verify_bundle(run_dir, repo_root=repo_root)
        assert result["reconstructed_classification"] == (
            "GO_EXACT_GREEDY_DECODE_BURST"
        )
        assert result["reconstructed_selected_policy"] == (
            "decode_burst_k4"
        )

    with TemporaryDirectory() as temporary:
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == "full_step_graph_k1":
                row["exact_greedy_decode_burst_summary"][
                    "graph_replays"
                ] = 126
        run_dir, repo_root = write_fixture_bundle(
            Path(temporary),
            performance_rows=rows,
        )
        produce_gate(run_dir, repo_root=repo_root)
        result = verify_bundle(run_dir, repo_root=repo_root)
        assert result["reconstructed_classification"] == (
            "NO_GO_REPLAY_INCOMPLETE"
        )


def test_verifier_rejects_comparison_gate_and_threshold_drift() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        comparison = json.loads(
            (run_dir / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        comparison["candidate_evaluations"][
            "decode_burst_k8"
        ]["aggregate"]["host_vs_candidate"][
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
        gate["classification"] = "NO_GO_HOST_TPOT_MEDIAN"
        _write_json(run_dir / "gate.json", gate)
        _refresh_manifest_entry(run_dir, "gate.json")
        with pytest.raises(ValueError, match="classification drift"):
            verify_bundle(run_dir, repo_root=repo_root)

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


def test_verifier_rejects_manifest_sidecar_and_source_tamper() -> None:
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
        sidecar = run_dir / correctness[0]["logits_path"]
        sidecar.write_bytes(sidecar.read_bytes() + b"\x00")
        with pytest.raises(
            ValueError,
            match="digest mismatch|byte length mismatch",
        ):
            verify_bundle(run_dir, repo_root=repo_root)

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
        source_file = next(repo_root.rglob("config.py"))
        source_file.write_text("tampered\n", encoding="utf-8")
        with pytest.raises(ValueError, match="source digest mismatch"):
            verify_bundle(run_dir, repo_root=repo_root)


def test_verifier_rejects_row_and_workload_tamper() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        rows_path = run_dir / "case_rows.jsonl"
        rows = rows_path.read_text(encoding="utf-8").splitlines()
        rows_path.write_text(
            "\n".join(rows[:-1]) + "\n",
            encoding="utf-8",
        )
        _refresh_manifest_entry(run_dir, "case_rows.jsonl")
        with pytest.raises(ValueError, match="60 measured rows"):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        workload_path = run_dir / "workload_manifest.json"
        workload = json.loads(
            workload_path.read_text(encoding="utf-8")
        )
        workload["warmup_repetitions"] = 1
        _write_json(workload_path, workload)
        _refresh_manifest_entry(run_dir, "workload_manifest.json")
        with pytest.raises(ValueError, match="workload manifest mismatch"):
            verify_bundle(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = _ready_bundle(Path(temporary))
        workload_path = run_dir / "workload_manifest.json"
        workload = json.loads(
            workload_path.read_text(encoding="utf-8")
        )
        workload["model"] = "/models/not-the-stage1-model"
        _write_json(workload_path, workload)
        _refresh_manifest_entry(run_dir, "workload_manifest.json")
        with pytest.raises(ValueError, match="workload manifest mismatch"):
            verify_bundle(run_dir, repo_root=repo_root)


def main() -> None:
    test_verifier_is_independent_and_reconstructs_go()
    test_verifier_rejects_comparison_gate_and_threshold_drift()
    test_verifier_rejects_manifest_sidecar_and_source_tamper()
    test_verifier_rejects_row_and_workload_tamper()
    print("exact greedy decode-burst verifier tests passed")


if __name__ == "__main__":
    main()
