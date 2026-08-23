#!/usr/bin/env python3
"""Producer-gate contracts for split-phase K8 exact bursts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.exact_burst_split_phase_gate import classify, produce_gate
from tools.profile_exact_burst_split_phase import (
    CONTEXT_CASES,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    build_workload_manifest,
    summarize_rows,
    write_float32_sidecar,
)
from tools.test_profile_exact_burst_split_phase import (
    _case_row,
    _correctness_row,
)


RUN_TAG = "20260823-qwen3-06b-split-phase-gate-fixture"
SOURCE_COMMIT = "a" * 40
CANDIDATE = "decode_burst_k8_split_phase"


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


def _set_tpot(row: dict, values: list[int]) -> None:
    ordered = sorted(values)
    row["amortized_tpot_samples_ns"] = values
    row["amortized_tpot_median_ns"] = statistics.median(values)
    row["amortized_tpot_p95_ns"] = ordered[
        math.ceil(0.95 * len(ordered)) - 1
    ]
    row["amortized_tpot_p99_ns"] = ordered[
        math.ceil(0.99 * len(ordered)) - 1
    ]


def _set_gap(row: dict, maximum: int) -> None:
    summary = row["exact_greedy_decode_burst_summary"]
    if row["policy"] == CANDIDATE:
        gaps = [maximum - 1, maximum] * summary["commits"]
        row["split_phase_inventory"]["host_visible_gaps_ns"] = gaps
    elif POLICY_CONFIGS[row["policy"]]["enabled"]:
        gaps = [maximum] * summary["commits"]
    else:
        gaps = []
        maximum = 0
    row["host_visible_burst_gaps_ns"] = gaps
    row["maximum_host_visible_burst_gap_ns"] = maximum
    summary["maximum_host_visible_gap_ns"] = maximum


def make_fixture_rows() -> list[dict]:
    rows = []
    bucket_scale = {
        "short": 1.0,
        "medium": 1.2,
        "long": 1.4,
    }
    policy_scale = {
        "host_greedy": 1.20,
        "decode_burst_k4": 1.03,
        "decode_burst_k8": 1.00,
        CANDIDATE: 1.01,
    }
    for repetition in range(5):
        for bucket, prompt_tokens, generated_tokens in CONTEXT_CASES:
            output_ids = [
                10 + ((index + repetition) % 100)
                for index in range(generated_tokens)
            ]
            output_digest = hashlib.sha256(
                json.dumps(output_ids).encode("utf-8")
            ).hexdigest()
            for policy in POLICIES:
                row = deepcopy(_case_row(policy))
                row.update({
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "repetition": repetition,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "output_token_ids": output_ids,
                    "output_text_sha256": output_digest,
                })
                tpot = round(
                    1_000_000
                    * bucket_scale[bucket]
                    * policy_scale[policy]
                )
                _set_tpot(row, [tpot] * 127)
                if policy == "decode_burst_k8":
                    row.update({
                        "ttft_ns": 20_000_000,
                        "e2e_ns": 147_000_000,
                        "output_tokens_per_second": 1_000.0,
                        "cuda_peak_reserved_bytes": 1_000_000_000,
                    })
                    _set_gap(row, 10_000_000)
                elif policy == "decode_burst_k4":
                    row.update({
                        "ttft_ns": 20_000_000,
                        "e2e_ns": 150_000_000,
                        "output_tokens_per_second": 970.0,
                        "cuda_peak_reserved_bytes": 1_000_000_000,
                    })
                    _set_gap(row, 5_000_000)
                elif policy == CANDIDATE:
                    row.update({
                        "ttft_ns": 20_600_000,
                        "e2e_ns": 151_410_000,
                        "output_tokens_per_second": 980.0,
                        "cuda_peak_reserved_bytes": 1_030_000_000,
                    })
                    _set_gap(row, 5_150_000)
                else:
                    row.update({
                        "ttft_ns": 20_000_000,
                        "e2e_ns": 170_000_000,
                        "output_tokens_per_second": 850.0,
                        "cuda_peak_reserved_bytes": 990_000_000,
                    })
                    _set_gap(row, 0)
                rows.append(row)
    return rows


def _correctness_rows(
    run_dir: Path,
    *,
    candidate_values: tuple[float, ...] | None = None,
) -> list[dict]:
    rows = []
    for bucket, _prompt, _generated in CONTEXT_CASES:
        for policy in POLICIES:
            for point in SAMPLING_POINTS:
                row = _correctness_row(
                    run_dir=run_dir,
                    policy=policy,
                    bucket=bucket,
                    sampling_point=point,
                )
                row["run_tag"] = RUN_TAG
                row["source_commit"] = SOURCE_COMMIT
                if policy == CANDIDATE and candidate_values is not None:
                    sidecar = write_float32_sidecar(
                        run_dir,
                        row["logits_path"],
                        candidate_values,
                    )
                    row.update({
                        "logits_shape": [1, len(candidate_values)],
                        "logits_element_count":
                            sidecar["element_count"],
                        "logits_byte_length": sidecar["byte_length"],
                        "logits_sha256": sidecar["sha256"],
                    })
                rows.append(row)
    return rows


def write_fixture_bundle(
    root: Path,
    *,
    rows: list[dict] | None = None,
    candidate_values: tuple[float, ...] | None = None,
) -> tuple[Path, Path]:
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
    measured = make_fixture_rows() if rows is None else rows
    correctness = _correctness_rows(
        run_dir,
        candidate_values=candidate_values,
    )
    source_patch = run_dir / "source.patch"
    source_patch.write_bytes(b"")
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": "exact-burst-split-phase.source.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": {
                relative: hashlib.sha256(
                    (repo_root / relative).read_bytes()
                ).hexdigest()
                for relative in SOURCE_FILES
            },
        },
    )
    _write_json(
        run_dir / "workload_manifest.json",
        build_workload_manifest(
            model="/models/Qwen3-0.6B",
            run_tag=RUN_TAG,
            source_commit=SOURCE_COMMIT,
            gpu_memory_utilization=0.5,
            environment={
                "python_version": "3.12.0",
                "platform": "fixture",
                "python_executable": "/python",
                "torch_available": True,
                "torch_version": "2.8.0",
                "cuda_runtime_version": "12.8",
                "cuda_available": True,
                "cuda_device_name": "fixture-gpu",
            },
        ),
    )
    _write_jsonl(run_dir / "case_rows.jsonl", measured)
    _write_jsonl(run_dir / "correctness_rows.jsonl", correctness)
    summary = summarize_rows(measured, expected_repetitions=5)
    summary["correctness_row_count"] = 48
    _write_json(run_dir / "summary.json", summary)
    return run_dir, repo_root


def _load_correctness(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (
            run_dir / "correctness_rows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def _classification(
    run_dir: Path,
    rows: list[dict],
) -> str:
    return classify(
        rows,
        _load_correctness(run_dir),
        run_dir=run_dir,
    )["classification"]


def _mutate_candidate(rows: list[dict], mutation) -> None:
    for row in rows:
        if row["policy"] == CANDIDATE:
            mutation(row)


def _assert_raises(message: str, callback) -> None:
    try:
        callback()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_go_accepts_every_exact_frozen_boundary() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        result = classify(
            make_fixture_rows(),
            _load_correctness(run_dir),
            run_dir=run_dir,
        )
    assert result["classification"] == "GO_EXACT_BURST_SPLIT_PHASE"
    assert result["selected_policy"] == CANDIDATE
    evaluation = result["candidate_evaluation"]
    assert evaluation["aggregate"]["k8_vs_split"][
        "tpot_median_regression_fraction"
    ] == 0.01
    assert evaluation["aggregate"]["k8_vs_split"][
        "sample_count_per_policy"
    ] == 15
    assert evaluation["aggregate"]["k8_vs_split"][
        "throughput_regression_fraction"
    ] == 0.02
    assert evaluation["maximum_gap_ratio_vs_k8"] == 0.515
    assert evaluation["median_max_gap_regression_vs_k4"] == 0.03
    assert not evaluation["bucket_regressions"]
    assert not evaluation["latency_regressions"]
    assert evaluation["memory_regression"] is False


def test_go_accepts_remaining_tpot_and_visibility_boundaries() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] != CANDIDATE:
                continue
            baseline = {
                "short": 1_000_000,
                "medium": 1_200_000,
                "long": 1_400_000,
            }[row["context_bucket"]]
            _set_tpot(row, [round(baseline * 1.02)] * 127)
        assert _classification(run_dir, rows) == (
            "GO_EXACT_BURST_SPLIT_PHASE"
        )

    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == CANDIDATE:
                baseline = {
                    "short": 1_000_000,
                    "medium": 1_200_000,
                    "long": 1_400_000,
                }[row["context_bucket"]]
                scale = (
                    1.03
                    if row["context_bucket"] == "short"
                    else 0.99
                )
                _set_tpot(row, [round(baseline * scale)] * 127)
        target = next(
            row for row in rows
            if row["policy"] == CANDIDATE
            and row["context_bucket"] == "long"
            and row["repetition"] == 4
        )
        _set_gap(target, 6_000_000)
        result = classify(
            rows,
            _load_correctness(run_dir),
            run_dir=run_dir,
        )
        assert result["classification"] == (
            "GO_EXACT_BURST_SPLIT_PHASE"
        )
        evaluation = result["candidate_evaluation"]
        assert evaluation["maximum_gap_ratio_vs_k8"] == 0.6
        assert evaluation["by_bucket"]["short"][
            "tpot_median_regression_fraction"
        ] == 0.03
        assert evaluation["by_bucket"]["short"][
            "tpot_p95_regression_fraction"
        ] == 0.03


def test_correctness_precedes_performance_failure() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(
            Path(temporary),
            candidate_values=(1.0, 2.0, 3.3, 4.0),
        )
        rows = make_fixture_rows()
        _mutate_candidate(
            rows,
            lambda row: _set_tpot(row, [1_500_000] * 127),
        )
        assert _classification(run_dir, rows) == (
            "NO_GO_EXACT_BURST_SPLIT_PHASE_CORRECTNESS"
        )


def test_any_frozen_performance_limit_failure_is_no_go() -> None:
    mutations = (
        lambda row: _set_tpot(
            row,
            [
                round({
                    "short": 1_000_000,
                    "medium": 1_200_000,
                    "long": 1_400_000,
                }[row["context_bucket"]] * 1.020001)
            ] * 127,
        ),
        lambda row: row.__setitem__(
            "output_tokens_per_second", 979.999
        ),
        lambda row: row.__setitem__("ttft_ns", 20_600_001),
        lambda row: row.__setitem__("e2e_ns", 151_410_001),
        lambda row: row.__setitem__(
            "cuda_peak_reserved_bytes", 1_030_000_001
        ),
        lambda row: _set_gap(row, 6_000_001),
        lambda row: _set_gap(row, 5_150_001),
    )
    for mutation in mutations:
        with TemporaryDirectory() as temporary:
            run_dir, _repo_root = write_fixture_bundle(
                Path(temporary)
            )
            rows = make_fixture_rows()
            _mutate_candidate(rows, mutation)
            assert _classification(run_dir, rows) == (
                "NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE"
            )

    for field in ("median", "p95"):
        with TemporaryDirectory() as temporary:
            run_dir, _repo_root = write_fixture_bundle(
                Path(temporary)
            )
            rows = make_fixture_rows()
            for row in rows:
                if (
                    row["policy"] == CANDIDATE
                    and row["context_bucket"] == "short"
                ):
                    values = [1_000_000] * 127
                    if field == "median":
                        values = [1_030_001] * 127
                    else:
                        values = [1_000_000] * 120 + [1_030_001] * 7
                    _set_tpot(row, values)
            assert _classification(run_dir, rows) == (
                "NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE"
            )


def test_mixed_source_identity_is_incomplete_evidence() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        rows[0]["run_tag"] = "different-run"
        assert _classification(run_dir, rows) == (
            "INCOMPLETE_EXACT_BURST_SPLIT_PHASE_EVIDENCE"
        )


def test_producer_writes_all_artifacts_and_rejects_tamper() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        gate = produce_gate(run_dir, repo_root=repo_root)
        assert gate["classification"] == (
            "GO_EXACT_BURST_SPLIT_PHASE"
        )
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        assert {
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "source.patch",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
            "comparison.json",
            "gate.json",
        } <= set(manifest["artifacts"])
        assert len({
            name
            for name in manifest["artifacts"]
            if name.startswith("logits/")
        }) == 48

    def rejected(mutator, message: str) -> None:
        with TemporaryDirectory() as temporary:
            run_dir, repo_root = write_fixture_bundle(
                Path(temporary)
            )
            mutator(run_dir, repo_root)
            _assert_raises(
                message,
                lambda: produce_gate(run_dir, repo_root=repo_root),
            )

    def remove_row(run_dir: Path, _repo_root: Path) -> None:
        lines = (run_dir / "case_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        (run_dir / "case_rows.jsonl").write_text(
            "\n".join(lines[:-1]) + "\n",
            encoding="utf-8",
        )

    rejected(remove_row, "60 measured rows")

    def tamper_sidecar(run_dir: Path, _repo_root: Path) -> None:
        sidecar = next((run_dir / "logits").glob("*.f32"))
        sidecar.write_bytes(sidecar.read_bytes() + b"\0")

    rejected(tamper_sidecar, "byte length mismatch")

    def tamper_commit(run_dir: Path, _repo_root: Path) -> None:
        source = json.loads(
            (run_dir / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source["source_commit"] = "b" * 40
        _write_json(run_dir / "source_manifest.json", source)

    rejected(tamper_commit, "source-bound identity mismatch")

    def tamper_source_patch(run_dir: Path, _repo_root: Path) -> None:
        (run_dir / "source.patch").write_text(
            "unbound dirty patch\n",
            encoding="utf-8",
        )

    rejected(tamper_source_patch, "dirty source patch")

    def tamper_inventory(run_dir: Path, _repo_root: Path) -> None:
        rows = [
            json.loads(line)
            for line in (run_dir / "case_rows.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        target = next(row for row in rows if row["policy"] == CANDIDATE)
        target["split_phase_inventory"]["suffix_cleared_count"] -= 1
        _write_jsonl(run_dir / "case_rows.jsonl", rows)

    rejected(tamper_inventory, "split phase observation inventory mismatch")


def main() -> None:
    test_go_accepts_every_exact_frozen_boundary()
    test_go_accepts_remaining_tpot_and_visibility_boundaries()
    test_correctness_precedes_performance_failure()
    test_any_frozen_performance_limit_failure_is_no_go()
    test_mixed_source_identity_is_incomplete_evidence()
    test_producer_writes_all_artifacts_and_rejects_tamper()
    print("exact burst split-phase gate tests passed")


if __name__ == "__main__":
    main()
