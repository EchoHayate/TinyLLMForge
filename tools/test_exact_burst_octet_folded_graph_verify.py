from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import exact_burst_octet_folded_graph_ceiling as ceiling
from tools import profile_exact_burst_octet_folded_graph as profile
from tools import exact_burst_octet_folded_graph_verify as verifier
from tools.test_exact_burst_octet_folded_graph_ceiling import (
    _performance_rows,
)
from tools.test_profile_exact_burst_octet_folded_graph import (
    RUN_TAG,
    SOURCE_COMMIT,
    _correctness_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PATCH_SHA256 = hashlib.sha256(b"").hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_fixture_bundle(root: Path) -> Path:
    run_dir = root / RUN_TAG
    run_dir.mkdir(parents=True)
    (run_dir / "source.patch").write_bytes(b"")
    performance = _performance_rows()
    correctness = _correctness_rows(run_dir)
    for row in (*performance, *correctness):
        row["source_patch_sha256"] = PATCH_SHA256
    workload = profile.build_workload_manifest(
        model="/models/Qwen3-0.6B",
        device="cuda:0",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        source_patch_sha256=PATCH_SHA256,
        gpu_memory_utilization=0.5,
        environment={"fixture": True},
    )
    source = profile._source_manifest(
        repo_root=ROOT,
        source_commit=SOURCE_COMMIT,
        source_patch_sha256=PATCH_SHA256,
        run_tag=RUN_TAG,
    )
    summary = profile.summarize_rows(
        performance,
        expected_repetitions=profile.REPETITIONS,
    )
    summary["correctness_row_count"] = len(correctness)
    _write_json(run_dir / "workload_manifest.json", workload)
    _write_json(run_dir / "source_manifest.json", source)
    _write_jsonl(run_dir / "performance_rows.jsonl", performance)
    _write_jsonl(run_dir / "correctness_rows.jsonl", correctness)
    _write_json(run_dir / "profile_summary.json", summary)
    _write_json(
        run_dir / "ceiling.json",
        ceiling.summarize_evidence(performance, correctness),
    )
    return run_dir


def test_verifier_is_independent_and_reconstructs_raw_evidence(
    tmp_path: Path,
) -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "profile_exact_burst_octet_folded_graph import" not in source
    assert "exact_burst_octet_folded_graph_ceiling import" not in source
    run_dir = write_fixture_bundle(tmp_path)

    result = verifier.verify_artifact_directory(
        run_dir,
        source_root=ROOT,
    )

    assert result == {
        "schema_version":
            "exact-burst-octet-folded.verification.v1",
        "verified": True,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "source_patch_sha256": PATCH_SHA256,
        "classification": "GO_CEILING",
        "performance_row_count": 30,
        "correctness_row_count": 24,
    }


def test_verifier_rejects_row_and_sidecar_tamper(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path / "row")
    rows = [
        json.loads(line)
        for line in (run_dir / "performance_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows[0]["logical_replays"] -= 1
    _write_jsonl(run_dir / "performance_rows.jsonl", rows)
    with pytest.raises(ValueError, match="runtime inventory"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "sidecar")
    sidecar = next((run_dir / "logits").glob("*.f32"))
    sidecar.write_bytes(sidecar.read_bytes() + b"x")
    with pytest.raises(ValueError, match="logits"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_recorded_classification_tamper(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path)
    recorded = json.loads(
        (run_dir / "ceiling.json").read_text(encoding="utf-8")
    )
    recorded["classification"] = "NO_GO_CEILING"
    _write_json(run_dir / "ceiling.json", recorded)
    with pytest.raises(ValueError, match="ceiling"):
        verifier.verify_artifact_directory(run_dir)


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_verifier_rejects_incomplete_or_duplicate_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_dir = write_fixture_bundle(tmp_path)
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "missing":
        rows.pop()
    else:
        rows.append(dict(rows[-1]))
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="inventory"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_non_finite_metric_and_source_drift(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path / "non-finite")
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["tpot_median_ns"] = float("nan")
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "source-drift")
    source = json.loads(
        (run_dir / "source_manifest.json").read_text(encoding="utf-8")
    )
    source["source_sha256"][verifier.SOURCE_FILES[0]] = "0" * 64
    _write_json(run_dir / "source_manifest.json", source)
    with pytest.raises(ValueError, match="source hash"):
        verifier.verify_artifact_directory(
            run_dir,
            source_root=ROOT,
        )


def test_verifier_rejects_threshold_and_physical_counter_drift(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path / "threshold")
    recorded = json.loads(
        (run_dir / "ceiling.json").read_text(encoding="utf-8")
    )
    recorded["aggregate_median_tpot_improvement_pct"] += 0.001
    _write_json(run_dir / "ceiling.json", recorded)
    with pytest.raises(ValueError, match="ceiling"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "physical")
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["one_token_cuda_graph_launches"] -= 1
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="physical launch"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_frozen_workload_and_unconsumed_metric_drift(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path / "workload")
    workload = json.loads(
        (run_dir / "workload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    workload["temperature"] = 0.5
    _write_json(run_dir / "workload_manifest.json", workload)
    with pytest.raises(ValueError, match="workload"):
        verifier.verify_artifact_directory(run_dir)

    run_dir = write_fixture_bundle(tmp_path / "p99")
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["tpot_p99_ns"] = float("nan")
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_independently_enforces_tpot_p99_protection(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path)
    path = run_dir / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if (
            row["context_length"] == 8192
            and row["policy"] == "octet_folded_graph"
        ):
            row["tpot_p99_ns"] = 1_020_001.0
    _write_jsonl(path, rows)
    correctness = [
        json.loads(line)
        for line in (run_dir / "correctness_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    _write_json(
        run_dir / "ceiling.json",
        ceiling.summarize_evidence(rows, correctness),
    )

    result = verifier.verify_artifact_directory(
        run_dir,
        source_root=ROOT,
    )

    assert result["classification"] == "NO_GO_CEILING"
