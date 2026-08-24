#!/usr/bin/env python3
"""Producer-gate contracts for medium-context split-K evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

import pytest

from tools.exact_burst_medium_split_k_gate import (
    GO_EXACT_BURST_MEDIUM_SPLIT_K,
    NO_GO_CAPTURE_COST,
    NO_GO_CORRECTNESS,
    NO_GO_EVIDENCE_INCOMPLETE,
    NO_GO_GRAPH_SELECTION,
    NO_GO_MEMORY,
    NO_GO_PERFORMANCE,
    classify,
    produce_gate,
)
from tools.profile_exact_burst_medium_split_k import (
    CONTEXT_LENGTHS,
    CORRECTNESS_SCHEMA_VERSION,
    CORRECTNESS_TRACE_IDENTITY,
    GENERATED_TOKENS,
    POLICIES,
    REPETITIONS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    SOURCE_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
    build_workload_manifest,
    expected_flash_attn_num_splits,
    policy_order,
    sha256_file,
    write_float32_sidecar,
)
from tools.test_profile_exact_burst_medium_split_k import (
    RUN_TAG,
    SOURCE_COMMIT,
    _case_row,
    _summary,
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


def _set_tpot(row: dict, value: float) -> None:
    samples = [value] * (GENERATED_TOKENS - 1)
    row["amortized_tpot_samples_ns"] = samples
    row["amortized_tpot_median_ns"] = statistics.median(samples)
    row["amortized_tpot_p95_ns"] = samples[0]
    row["amortized_tpot_p99_ns"] = samples[0]


def _performance_rows() -> list[dict]:
    rows = []
    target_contexts = set(CONTEXT_LENGTHS) - {1025, 6145}
    for repetition in range(REPETITIONS):
        for context_index, context_length in enumerate(
            CONTEXT_LENGTHS
        ):
            output_ids = [
                10 + ((index + repetition) % 100)
                for index in range(GENERATED_TOKENS)
            ]
            output_digest = hashlib.sha256(
                json.dumps(output_ids).encode()
            ).hexdigest()
            for order_position, policy in enumerate(
                policy_order(repetition, context_index)
            ):
                row = _case_row(
                    policy,
                    context_length=context_length,
                    repetition=repetition,
                    order_position=order_position,
                )
                row["output_token_ids"] = output_ids
                row["output_text_sha256"] = output_digest
                baseline_tpot = 1_000_000.0 + context_index * 10_000
                candidate_factor = (
                    0.96 if context_length in target_contexts else 1.0
                )
                _set_tpot(
                    row,
                    baseline_tpot
                    * (candidate_factor if policy == "split12" else 1.0),
                )
                row["ttft_ns"] = (
                    10_000_000
                    * (1.005 if policy == "split12" else 1.0)
                )
                row["e2e_ns"] = (
                    140_000_000
                    * (0.97 if policy == "split12" else 1.0)
                )
                row["output_tokens_per_second"] = (
                    940.0 if policy == "split12" else 920.0
                )
                if policy == "split12":
                    row["cuda_peak_allocated_bytes"] += 1_000_000
                    row["cuda_peak_reserved_bytes"] += 2_000_000
                rows.append(row)
    return rows


def _correctness_rows(run_dir: Path) -> list[dict]:
    rows = []
    expected_ordinals = {
        "prefill-final": None,
        "decode-first": 0,
        "decode-middle": 7,
        "decode-final": 6,
    }
    for context_length in CONTEXT_LENGTHS:
        output_ids = list(range(GENERATED_TOKENS))
        for policy in POLICIES:
            selected_split = expected_flash_attn_num_splits(
                policy=policy,
                context_length=context_length,
            )
            selected_identity = (
                ("c" if selected_split == 0 else "d") * 64
            )
            summary = _summary(policy, context_length)
            summary["sampled_logit_d2h_calls"] = 3
            for receipt in summary["capture_receipts"]:
                receipt["correctness_trace"] = True
            for point in SAMPLING_POINTS:
                values = (1.0, 2.0, 4.0, 3.0)
                if policy == "split12":
                    values = (1.01, 2.01, 4.01, 3.01)
                sidecar = write_float32_sidecar(
                    run_dir,
                    (
                        f"logits/{context_length}-{policy}-"
                        f"{point}.f32"
                    ),
                    values,
                )
                burst_sample = point != "prefill-final"
                rows.append({
                    "schema_version": CORRECTNESS_SCHEMA_VERSION,
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "context_length": context_length,
                    "generated_tokens": GENERATED_TOKENS,
                    "sampling_point": point,
                    "prompt_sha256": "b" * 64,
                    "output_token_ids": output_ids,
                    "output_text_sha256": "e" * 64,
                    "logits_path": sidecar["path"],
                    "logits_shape": [1, 4],
                    "logits_element_count":
                        sidecar["element_count"],
                    "logits_byte_length":
                        sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "correctness_trace": True,
                    "trace_identity": CORRECTNESS_TRACE_IDENTITY,
                    "trace_graph_identity_sha256": (
                        selected_identity if burst_sample else None
                    ),
                    "trace_flash_attn_num_splits": (
                        selected_split if burst_sample else None
                    ),
                    "selected_replay_ordinal":
                        expected_ordinals[point],
                    "sampled_logit_d2h_calls":
                        1 if burst_sample else 0,
                    "exact_greedy_decode_burst_summary":
                        deepcopy(summary),
                })
    return rows


def _fixture(run_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    performance = _performance_rows()
    correctness = _correctness_rows(run_dir)
    _write_jsonl(
        run_dir / "performance_rows.jsonl",
        performance,
    )
    _write_jsonl(
        run_dir / "correctness_rows.jsonl",
        correctness,
    )
    _write_json(
        run_dir / "workload_manifest.json",
        build_workload_manifest(
            model="/models/Qwen3-0.6B",
            run_tag=RUN_TAG,
            source_commit=SOURCE_COMMIT,
            gpu_memory_utilization=0.5,
            environment={"fixture": True},
        ),
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": {
                relative: (
                    sha256_file(repo_root / relative)
                    if (repo_root / relative).is_file()
                    else "0" * 64
                )
                for relative in SOURCE_FILES
            },
        },
    )


def _rewrite_rows(path: Path, mutate) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mutate(rows)
    _write_jsonl(path, rows)


def test_complete_fixture_classifies_go_and_writes_bound_artifacts(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    result = produce_gate(tmp_path)
    assert result["classification"] == (
        GO_EXACT_BURST_MEDIUM_SPLIT_K
    )
    assert result["performance"]["target"][
        "tpot_median_improvement_fraction"
    ] >= 0.01
    assert result["correctness"]["all_token_ids_exact"] is True
    assert result["correctness"]["all_argmax_exact"] is True
    assert result["graph_selection"]["all_exact"] is True
    assert result["cost"]["added_retained_static_bytes"] <= 8 * 1024**2
    assert {
        "comparison.json",
        "summary.json",
        "manifest.json",
        "report.md",
    }.issubset(path.name for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda rows: _set_tpot(
                next(
                    row
                    for row in rows
                    if row["policy"] == "split12"
                    and row["context_length"] == 2049
                ),
                1_100_000,
            ),
            NO_GO_PERFORMANCE,
        ),
        (
            lambda rows: rows[0].update(
                replay_flash_attn_num_splits=12,
                replay_graph_identity_sha256="d" * 64,
            ),
            NO_GO_GRAPH_SELECTION,
        ),
        (
            lambda rows: rows[1].update(
                capture_retained_static_bytes=20 * 1024**2
            ),
            NO_GO_MEMORY,
        ),
        (
            lambda rows: rows[1].update(
                capture_duration_ns=6_000_000_000
            ),
            NO_GO_CAPTURE_COST,
        ),
    ],
)
def test_performance_selection_and_cost_thresholds(
    tmp_path: Path,
    mutation,
    expected: str,
) -> None:
    _fixture(tmp_path)
    path = tmp_path / "performance_rows.jsonl"
    _rewrite_rows(path, mutation)
    result = classify(tmp_path)
    assert result["classification"] == expected


def test_correctness_token_argmax_and_logit_limits(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    path = tmp_path / "correctness_rows.jsonl"

    def token_mismatch(rows):
        candidate = next(
            row for row in rows if row["policy"] == "split12"
        )
        candidate["output_token_ids"][0] += 1

    _rewrite_rows(path, token_mismatch)
    assert classify(tmp_path)["classification"] == NO_GO_CORRECTNESS

    other = tmp_path / "argmax"
    other.mkdir()
    _fixture(other)
    candidate = next(
        row
        for row in [
            json.loads(line)
            for line in (
                other / "correctness_rows.jsonl"
            ).read_text().splitlines()
        ]
        if row["policy"] == "split12"
    )
    sidecar = write_float32_sidecar(
        other,
        candidate["logits_path"],
        (9.0, 2.0, 4.0, 3.0),
    )
    candidate.update({
        "logits_element_count": sidecar["element_count"],
        "logits_byte_length": sidecar["byte_length"],
        "logits_sha256": sidecar["sha256"],
    })
    rows = [
        json.loads(line)
        for line in (
            other / "correctness_rows.jsonl"
        ).read_text().splitlines()
    ]
    rows[rows.index(next(row for row in rows if (
        row["policy"] == "split12"
        and row["context_length"] == candidate["context_length"]
        and row["sampling_point"] == candidate["sampling_point"]
    )))] = candidate
    _write_jsonl(other / "correctness_rows.jsonl", rows)
    assert classify(other)["classification"] == NO_GO_CORRECTNESS


def test_incomplete_duplicate_source_and_stale_summary_are_rejected(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    performance_path = tmp_path / "performance_rows.jsonl"
    rows = [
        json.loads(line)
        for line in performance_path.read_text().splitlines()
    ]
    _write_jsonl(performance_path, rows[:-1])
    assert classify(tmp_path)["classification"] == (
        NO_GO_EVIDENCE_INCOMPLETE
    )

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    _fixture(duplicate)
    rows = [
        json.loads(line)
        for line in (
            duplicate / "performance_rows.jsonl"
        ).read_text().splitlines()
    ]
    _write_jsonl(
        duplicate / "performance_rows.jsonl",
        rows + [deepcopy(rows[0])],
    )
    assert classify(duplicate)["classification"] == (
        NO_GO_EVIDENCE_INCOMPLETE
    )

    stale = tmp_path / "stale"
    stale.mkdir()
    _fixture(stale)
    _write_json(
        stale / "summary.json",
        {"classification": GO_EXACT_BURST_MEDIUM_SPLIT_K},
    )
    _rewrite_rows(
        stale / "performance_rows.jsonl",
        lambda values: _set_tpot(
            next(
                row
                for row in values
                if row["policy"] == "split12"
                and row["context_length"] == 2049
            ),
            1_100_000,
        ),
    )
    assert classify(stale)["classification"] == NO_GO_PERFORMANCE
