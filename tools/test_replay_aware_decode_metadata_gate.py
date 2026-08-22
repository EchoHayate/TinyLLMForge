"""Producer-gate tests for replay-aware decode metadata landing."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
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

from tools.profile_replay_aware_decode_metadata import (
    SOURCE_FILES,
    summarize_rows,
)
from tools.replay_aware_decode_metadata_gate import (
    classify,
    produce_gate,
)


RUN_TAG = "20260822-qwen3-06b-replay-meta-fixture"
SOURCE_COMMIT = "a" * 40


def _landing(policy: str) -> dict:
    optimized = 127 if policy == "on" else 0
    return {
        "eligible_steps": optimized,
        "optimized_steps": optimized,
        "allocation_count": 0,
        "growth_count": 0,
        "staged_h2d_bytes": (
            optimized * 40 if policy == "on" else 0
        ),
        "avoided_temporary_cuda_tensors": (
            optimized * 5 if policy == "on" else 0
        ),
        "avoided_blanket_zero_bytes": (
            optimized * 8_000 if policy == "on" else 0
        ),
        "current_pinned_capacity_bytes": (
            1_792 if policy == "on" else 0
        ),
        "peak_pinned_capacity_bytes": (
            1_792 if policy == "on" else 0
        ),
        "fallback_counts": {},
    }


def make_fixture_rows(on_scale: float = 0.90) -> list[dict]:
    rows = []
    shapes = (
        ("short", 256, 1_000_000),
        ("medium", 2048, 1_200_000),
        ("long", 8192, 1_400_000),
    )
    for repetition in range(5):
        for bucket, prompt_tokens, base_tpot in shapes:
            output_ids = [
                10 + ((index + repetition) % 100)
                for index in range(128)
            ]
            text_digest = hashlib.sha256(
                json.dumps(output_ids).encode("utf-8")
            ).hexdigest()
            for policy in ("off", "on"):
                scale = 1.0 if policy == "off" else on_scale
                tpot = round(base_tpot * scale)
                e2e = 20_000_000 + tpot * 127
                rows.append({
                    "schema_version":
                        "replay-aware-decode-metadata.case.v1",
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "repetition": repetition,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": 128,
                    "output_token_ids": list(output_ids),
                    "output_text_sha256": text_digest,
                    "ttft_ns": 20_000_000,
                    "e2e_ns": e2e,
                    "tpot_samples_ns": [tpot] * 127,
                    "decode_host_ns": [
                        round(tpot * 0.85)
                    ] * 127,
                    "decode_cuda_ns": [
                        round(tpot * 0.70)
                    ] * 127,
                    "output_tokens_per_second": (
                        128 * 1_000_000_000 / e2e
                    ),
                    "cuda_peak_allocated_bytes":
                        1_000_000_000,
                    "cuda_peak_reserved_bytes":
                        1_200_000_000,
                    "landing_summary": _landing(policy),
                })
    return rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
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
        ),
        encoding="utf-8",
    )


def write_fixture_bundle(
    root: Path,
    rows: list[dict] | None = None,
) -> tuple[Path, Path]:
    rows = make_fixture_rows() if rows is None else rows
    run_dir = root / "run"
    repo_root = root / "repo"
    run_dir.mkdir()
    for relative in SOURCE_FILES:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"fixture source for {relative}\n",
            encoding="utf-8",
        )
    source_sha256 = {
        relative: hashlib.sha256(
            (repo_root / relative).read_bytes()
        ).hexdigest()
        for relative in SOURCE_FILES
    }
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version":
                "replay-aware-decode-metadata.source.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": source_sha256,
        },
    )
    _write_json(
        run_dir / "workload_manifest.json",
        {
            "schema_version":
                "replay-aware-decode-metadata.workload.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "model": "/models/Qwen3-0.6B",
            "context_cases": [
                {
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": 128,
                }
                for bucket, prompt_tokens in (
                    ("short", 256),
                    ("medium", 2048),
                    ("long", 8192),
                )
            ],
            "repetitions": 5,
            "warmup_repetitions": 2,
            "batch_size": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "gpu_memory_utilization": 0.5,
            "policy_order": {
                str(index): (
                    ["off", "on"]
                    if index % 2 == 0
                    else ["on", "off"]
                )
                for index in range(5)
            },
        },
    )
    _write_jsonl(run_dir / "case_rows.jsonl", rows)
    _write_json(
        run_dir / "summary.json",
        summarize_rows(rows),
    )
    return run_dir, repo_root


def test_classify_accepts_complete_go_fixture():
    result = classify(make_fixture_rows())
    assert result["classification"] == (
        "GO_REPLAY_AWARE_METADATA"
    )
    assert result["median_tpot_winning_bucket_count"] == 3
    assert result["aggregate"]["tpot_p95_improvement_fraction"] >= 0.05


def test_classify_rejects_token_mismatch():
    rows = make_fixture_rows()
    candidate = next(
        row for row in rows
        if row["policy"] == "on"
    )
    candidate["output_token_ids"][-1] += 1

    assert classify(rows)["classification"] == (
        "NO_GO_CORRECTNESS"
    )


def test_classify_rejects_incomplete_optimized_path():
    rows = make_fixture_rows()
    candidate = next(
        row for row in rows
        if row["policy"] == "on"
    )
    candidate["landing_summary"]["optimized_steps"] -= 1

    assert classify(rows)["classification"] == (
        "NO_GO_OPTIMIZED_PATH_INCOMPLETE"
    )


def test_classify_rejects_flat_tpot():
    assert classify(
        make_fixture_rows(on_scale=1.0)
    )["classification"] == "NO_GO_TPOT_MEDIAN"


def test_classify_rejects_p95_tail_regression():
    rows = make_fixture_rows()
    for row in rows:
        if (
            row["policy"] == "on"
            and row["context_bucket"] == "long"
        ):
            base = 1_400_000
            row["tpot_samples_ns"] = (
                [round(base * 0.90)] * 102
                + [round(base * 1.10)] * 25
            )

    assert classify(rows)["classification"] == (
        "NO_GO_TPOT_P95"
    )


def test_classify_rejects_ttft_regression():
    rows = make_fixture_rows()
    for row in rows:
        if row["policy"] == "on":
            row["ttft_ns"] = 20_800_001

    assert classify(rows)["classification"] == (
        "NO_GO_PROTECTED_REGRESSION"
    )


@pytest.mark.parametrize(
    "regression",
    [
        "single_bucket_tpot",
        "e2e",
        "throughput",
        "cuda_reserved",
        "pinned_capacity",
    ],
)
def test_classify_enforces_every_protected_regression(
    regression,
):
    rows = make_fixture_rows()
    for row in rows:
        if row["policy"] != "on":
            continue
        if (
            regression == "single_bucket_tpot"
            and row["context_bucket"] == "short"
        ):
            row["tpot_samples_ns"] = [1_040_001] * 127
        elif regression == "e2e":
            baseline = next(
                candidate
                for candidate in rows
                if candidate["policy"] == "off"
                and candidate["context_bucket"]
                == row["context_bucket"]
                and candidate["repetition"]
                == row["repetition"]
            )
            row["e2e_ns"] = round(
                baseline["e2e_ns"] * 1.031
            )
        elif regression == "throughput":
            baseline = next(
                candidate
                for candidate in rows
                if candidate["policy"] == "off"
                and candidate["context_bucket"]
                == row["context_bucket"]
                and candidate["repetition"]
                == row["repetition"]
            )
            row["output_tokens_per_second"] = (
                baseline["output_tokens_per_second"]
                * 0.979
            )
        elif regression == "cuda_reserved":
            row["cuda_peak_reserved_bytes"] = (
                1_213_200_001
            )
        elif regression == "pinned_capacity":
            row["landing_summary"][
                "peak_pinned_capacity_bytes"
            ] = 1_793

    assert classify(rows)["classification"] == (
        "NO_GO_PROTECTED_REGRESSION"
    )


def test_classify_rejects_text_hash_mismatch():
    rows = make_fixture_rows()
    candidate = next(
        row for row in rows
        if row["policy"] == "on"
    )
    candidate["output_text_sha256"] = "f" * 64

    assert classify(rows)["classification"] == (
        "NO_GO_CORRECTNESS"
    )


def test_classify_rejects_duplicate_and_nonfinite_rows():
    rows = make_fixture_rows()
    with pytest.raises(
        ValueError,
        match="exactly 30 measured rows",
    ):
        classify(rows + [deepcopy(rows[0])])

    rows = make_fixture_rows()
    rows[0]["output_tokens_per_second"] = math.inf
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        classify(rows)


def test_producer_hashes_primary_evidence_and_rejects_source_tamper():
    with TemporaryDirectory() as tmp:
        run_dir, repo_root = write_fixture_bundle(
            Path(tmp)
        )
        result = produce_gate(
            run_dir,
            repo_root=repo_root,
        )
        assert result["classification"] == (
            "GO_REPLAY_AWARE_METADATA"
        )
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        assert set(manifest["artifacts"]) == {
            "case_rows.jsonl",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
            "comparison.json",
            "gate.json",
        }

        source = json.loads(
            (run_dir / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source["source_sha256"][SOURCE_FILES[0]] = "0" * 64
        _write_json(
            run_dir / "source_manifest.json",
            source,
        )
        with pytest.raises(
            ValueError,
            match="source digest mismatch",
        ):
            produce_gate(
                run_dir,
                repo_root=repo_root,
            )


def main() -> None:
    test_classify_accepts_complete_go_fixture()
    test_classify_rejects_token_mismatch()
    test_classify_rejects_incomplete_optimized_path()
    test_classify_rejects_flat_tpot()
    test_classify_rejects_p95_tail_regression()
    test_classify_rejects_ttft_regression()
    for regression in (
        "single_bucket_tpot",
        "e2e",
        "throughput",
        "cuda_reserved",
        "pinned_capacity",
    ):
        test_classify_enforces_every_protected_regression(
            regression
        )
    test_classify_rejects_text_hash_mismatch()
    test_classify_rejects_duplicate_and_nonfinite_rows()
    test_producer_hashes_primary_evidence_and_rejects_source_tamper()
    print("replay-aware metadata gate tests passed")


if __name__ == "__main__":
    main()
