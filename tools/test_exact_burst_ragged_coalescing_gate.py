#!/usr/bin/env python3
"""Producer-gate contracts for exact-burst ragged coalescing."""

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

from tools.exact_burst_ragged_coalescing_gate import (
    classify,
    produce_gate,
)
from tools.profile_exact_burst_ragged_coalescing import (
    CASE_SCHEMA_VERSION,
    CONTEXT_CASES,
    CORRECTNESS_SCHEMA_VERSION,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    _expected_selected_replay_ordinal,
    build_workload_manifest,
    correctness_point_uses_burst_trace,
    summarize_rows,
    write_float32_sidecar,
)
from tools.test_profile_exact_burst_ragged_coalescing import (
    _candidate_row,
)
from tools.test_profile_exact_burst_split_phase import (
    _case_row,
    _correctness_row,
)


RUN_TAG = "20260823-qwen3-06b-ragged-gate-fixture"
SOURCE_COMMIT = "a" * 40
CANDIDATE = "decode_burst_k8_split_phase_ragged"
SPLIT_BASELINE = "decode_burst_k8_split_phase"
K4_BASELINE = "decode_burst_k4"


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


def _set_tpot(
    row: dict,
    values: list[int],
    *,
    tail_seven_elapsed_ns: int | None = None,
) -> None:
    ordered = sorted(values)
    row["amortized_tpot_samples_ns"] = values
    row["amortized_tpot_median_ns"] = statistics.median(values)
    row["amortized_tpot_p95_ns"] = ordered[
        math.ceil(0.95 * len(ordered)) - 1
    ]
    row["amortized_tpot_p99_ns"] = ordered[
        math.ceil(0.99 * len(ordered)) - 1
    ]
    row["tail_seven_elapsed_ns"] = (
        sum(values[-7:])
        if tail_seven_elapsed_ns is None
        else tail_seven_elapsed_ns
    )


def _set_gap(row: dict, maximum: int) -> None:
    summary = row["exact_greedy_decode_burst_summary"]
    if row["policy"] == SPLIT_BASELINE:
        gaps = [maximum - 1, maximum] * summary["commits"]
        row["split_phase_inventory"]["host_visible_gaps_ns"] = gaps
    elif row["policy"] == CANDIDATE:
        split_gaps = [maximum - 1, maximum] * 15
        row["split_phase_inventory"]["host_visible_gaps_ns"] = split_gaps
        gaps = split_gaps + [maximum, maximum - 1]
    else:
        gaps = [maximum] * summary["commits"]
    row["host_visible_burst_gaps_ns"] = gaps
    row["maximum_host_visible_burst_gap_ns"] = maximum
    summary["maximum_host_visible_gap_ns"] = maximum


def _set_capture_retained(row: dict, retained: int) -> None:
    row["capture_retained_static_bytes"] = retained
    row["exact_greedy_decode_burst_summary"][
        "capture_receipts"
    ][0]["retained_static_bytes"] = retained


def _base_row(policy: str) -> dict:
    if policy == CANDIDATE:
        return _candidate_row()
    return deepcopy(_case_row(policy))


def make_fixture_rows() -> list[dict]:
    rows = []
    bucket_scale = {
        "short": 1.0,
        "medium": 1.2,
        "long": 1.4,
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
                row = _base_row(policy)
                row.update({
                    "schema_version": CASE_SCHEMA_VERSION,
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "repetition": repetition,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "output_token_ids": output_ids,
                    "output_text_sha256": output_digest,
                })
                baseline_tpot = round(
                    1_000_000 * bucket_scale[bucket]
                )
                if policy == K4_BASELINE:
                    _set_tpot(row, [baseline_tpot] * 127)
                    row.update({
                        "ttft_ns": 20_000_000,
                        "e2e_ns": 150_000_000,
                        "output_tokens_per_second": 980.0,
                        "cuda_peak_allocated_bytes": 900_000_000,
                        "cuda_peak_reserved_bytes": 1_000_000_000,
                    })
                    _set_gap(row, 5_000_000)
                elif policy == SPLIT_BASELINE:
                    _set_tpot(
                        row,
                        [baseline_tpot] * 120
                        + [2_000_000] * 7,
                    )
                    row.update({
                        "ttft_ns": 20_000_000,
                        "e2e_ns": 160_000_000,
                        "output_tokens_per_second": 1_000.0,
                        "cuda_peak_allocated_bytes": 900_000_000,
                        "cuda_peak_reserved_bytes": 1_000_000_000,
                    })
                    _set_gap(row, 5_000_000)
                else:
                    _set_tpot(
                        row,
                        [baseline_tpot] * 120
                        + [1_800_000] * 7,
                    )
                    row.update({
                        "ttft_ns": 20_600_000,
                        "e2e_ns": 163_200_000,
                        "output_tokens_per_second": 990.0,
                        "cuda_peak_allocated_bytes": 927_000_000,
                        "cuda_peak_reserved_bytes": 1_030_000_000,
                    })
                    _set_gap(row, 5_150_000)
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
                source_policy = (
                    SPLIT_BASELINE
                    if policy == CANDIDATE
                    else policy
                )
                row = _correctness_row(
                    run_dir=run_dir,
                    policy=source_policy,
                    bucket=bucket,
                    sampling_point=point,
                )
                row.update({
                    "schema_version": CORRECTNESS_SCHEMA_VERSION,
                    "policy": policy,
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "trace_identity": (
                        "gate-only-exact-burst-ragged-"
                        "coalescing-correctness-v1"
                    ),
                })
                if policy == CANDIDATE:
                    row["logits_path"] = (
                        f"logits/{bucket}-{policy}-{point}.f32"
                    )
                    row["exact_greedy_decode_burst_summary"] = (
                        deepcopy(
                            _candidate_row()[
                                "exact_greedy_decode_burst_summary"
                            ]
                        )
                    )
                    row["exact_greedy_decode_burst_summary"][
                        "capture_receipts"
                    ][0]["correctness_trace"] = True
                    row["exact_greedy_decode_burst_summary"][
                        "sampled_logit_d2h_calls"
                    ] = 3
                    burst_sample = correctness_point_uses_burst_trace(
                        policy,
                        point,
                    )
                    row["trace_graph_identity_sha256"] = (
                        "c" * 64 if burst_sample else None
                    )
                    row["selected_replay_ordinal"] = (
                        _expected_selected_replay_ordinal(
                            policy,
                            point,
                            _generated,
                        )
                        if burst_sample
                        else None
                    )
                    row["sampled_logit_d2h_calls"] = (
                        1 if burst_sample else 0
                    )
                    values = (
                        candidate_values
                        if candidate_values is not None
                        else (1.0, 2.0, 3.0, 4.0)
                    )
                    sidecar = write_float32_sidecar(
                        run_dir,
                        row["logits_path"],
                        values,
                    )
                    row.update({
                        "logits_shape": [1, len(values)],
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
    (run_dir / "source.patch").write_bytes(b"")
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version":
                "exact-burst-ragged-coalescing.source.v1",
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
    summary["correctness_row_count"] = 36
    _write_json(run_dir / "summary.json", summary)
    return run_dir, repo_root


def _load_correctness(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (
            run_dir / "correctness_rows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def _classification(run_dir: Path, rows: list[dict]) -> str:
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
        raise AssertionError(
            f"expected ValueError containing {message!r}"
        )


def test_go_accepts_all_exact_boundaries() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        result = classify(
            make_fixture_rows(),
            _load_correctness(run_dir),
            run_dir=run_dir,
        )
    assert result["classification"] == (
        "GO_EXACT_BURST_RAGGED_COALESCING"
    )
    evaluation = result["candidate_evaluation"]
    assert evaluation[
        "tail_seven_improvement_fraction"
    ] == 0.1
    assert evaluation["aggregate"][
        "split_vs_ragged"
    ]["throughput_regression_fraction"] == 0.01
    assert evaluation[
        "median_max_gap_regression_vs_k4"
    ] == 0.03
    assert evaluation[
        "maximum_gap_regression_vs_k4"
    ] == 0.03
    assert evaluation["lifecycle"]["complete"] is True


def test_correctness_precedes_performance_failure() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(
            Path(temporary),
            candidate_values=(1.0, 2.0, 3.3, 4.0),
        )
        rows = make_fixture_rows()
        _mutate_candidate(
            rows,
            lambda row: _set_tpot(
                row,
                [1_000_000] * 120 + [2_100_000] * 7,
            ),
        )
        assert _classification(run_dir, rows) == (
            "NO_GO_EXACT_BURST_RAGGED_COALESCING_CORRECTNESS"
        )


def test_each_performance_or_cost_limit_failure_is_no_go() -> None:
    mutations = (
        lambda row: _set_tpot(
            row,
            [
                round(
                    row["amortized_tpot_samples_ns"][0]
                    * 1.010001
                )
            ] * 120
            + [1_800_000] * 7,
        ),
        lambda row: row.__setitem__(
            "output_tokens_per_second", 989.999
        ),
        lambda row: row.__setitem__("ttft_ns", 20_600_001),
        lambda row: row.__setitem__("e2e_ns", 163_200_001),
        lambda row: row.__setitem__(
            "cuda_peak_allocated_bytes", 927_000_001
        ),
        lambda row: row.__setitem__(
            "cuda_peak_reserved_bytes", 1_030_000_001
        ),
        lambda row: _set_gap(row, 5_150_001),
        lambda row: _set_capture_retained(row, 904_001),
    )
    for mutation in mutations:
        with TemporaryDirectory() as temporary:
            run_dir, _repo_root = write_fixture_bundle(
                Path(temporary)
            )
            rows = make_fixture_rows()
            _mutate_candidate(rows, mutation)
            assert _classification(run_dir, rows) == (
                "NO_GO_EXACT_BURST_RAGGED_COALESCING_PERFORMANCE"
            )

    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        _mutate_candidate(
            rows,
            lambda row: _set_tpot(
                row,
                row["amortized_tpot_samples_ns"][:-7]
                + [1_800_001] * 7,
            ),
        )
        assert _classification(run_dir, rows) == (
            "NO_GO_EXACT_BURST_RAGGED_COALESCING_PERFORMANCE"
        )


def test_mixed_identity_or_lifecycle_is_incomplete() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        rows[0]["run_tag"] = "different-run"
        assert _classification(run_dir, rows) == (
            "INCOMPLETE_EXACT_BURST_RAGGED_COALESCING_EVIDENCE"
        )

    with TemporaryDirectory() as temporary:
        run_dir, _repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        target = next(
            row for row in rows if row["policy"] == CANDIDATE
        )
        target["exact_greedy_decode_burst_summary"][
            "requested_width_histogram"
        ] = {"4": 2, "8": 15}
        _assert_raises(
            "ragged coalescing fallback inventory mismatch",
            lambda: classify(
                rows,
                _load_correctness(run_dir),
                run_dir=run_dir,
            ),
        )


def test_producer_writes_manifest_and_rejects_tamper() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        gate = produce_gate(run_dir, repo_root=repo_root)
        assert gate["classification"] == (
            "GO_EXACT_BURST_RAGGED_COALESCING"
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
        }) == 36

    def rejected(mutator, message: str) -> None:
        with TemporaryDirectory() as temporary:
            run_dir, repo_root = write_fixture_bundle(
                Path(temporary)
            )
            mutator(run_dir, repo_root)
            _assert_raises(
                message,
                lambda: produce_gate(
                    run_dir,
                    repo_root=repo_root,
                ),
            )

    def remove_row(run_dir: Path, _repo_root: Path) -> None:
        lines = (run_dir / "case_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        (run_dir / "case_rows.jsonl").write_text(
            "\n".join(lines[:-1]) + "\n",
            encoding="utf-8",
        )

    rejected(remove_row, "45 measured rows")

    def duplicate_row(run_dir: Path, _repo_root: Path) -> None:
        path = run_dir / "case_rows.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(lines[:-1] + [lines[0]]) + "\n",
            encoding="utf-8",
        )

    rejected(duplicate_row, "measured case inventory")

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

    def tamper_patch(run_dir: Path, _repo_root: Path) -> None:
        (run_dir / "source.patch").write_text(
            "unbound dirty patch\n",
            encoding="utf-8",
        )

    rejected(tamper_patch, "dirty source patch")

    def non_finite(run_dir: Path, _repo_root: Path) -> None:
        path = run_dir / "case_rows.jsonl"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                '"ttft_ns":20000000',
                '"ttft_ns":NaN',
                1,
            ),
            encoding="utf-8",
        )

    rejected(non_finite, "non-finite JSON value")


def main() -> None:
    test_go_accepts_all_exact_boundaries()
    test_correctness_precedes_performance_failure()
    test_each_performance_or_cost_limit_failure_is_no_go()
    test_mixed_identity_or_lifecycle_is_incomplete()
    test_producer_writes_manifest_and_rejects_tamper()
    print("exact burst ragged-coalescing gate tests passed")


if __name__ == "__main__":
    main()
