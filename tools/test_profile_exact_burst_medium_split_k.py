#!/usr/bin/env python3
"""Contracts for the medium-context split-K exact-burst profiler."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.profile_exact_burst_medium_split_k import (
    CONTEXT_LENGTHS,
    GENERATED_TOKENS,
    POLICIES,
    POLICY_CONFIGS,
    REPETITIONS,
    SOURCE_FILES,
    WARMUP_REPETITIONS,
    _construct_llm,
    build_workload_manifest,
    correctness_identities,
    expected_flash_attn_num_splits,
    performance_identities,
    policy_order,
    summarize_rows,
    validate_case_row,
    validate_correctness_rows,
    write_float32_sidecar,
)


RUN_TAG = "20260824-medium-split-k-fixture"
SOURCE_COMMIT = "a" * 40


def _receipt(*, split: int, correctness_trace: bool = False) -> dict:
    identity_character = "c" if split == 0 else "d"
    return {
        "graph_identity_sha256": identity_character * 64,
        "graph_generation": 7,
        "capture_duration_ns": 1_000_000 + split,
        "allocated_delta_bytes": 400_000 + split,
        "reserved_delta_bytes": 2_000_000 + split,
        "retained_static_bytes": 900_008 + split,
        "scratch_block_count": 1,
        "correctness_trace": correctness_trace,
        "flash_attn_num_splits": split,
    }


def _summary(policy: str, context_length: int) -> dict:
    selected_split = expected_flash_attn_num_splits(
        policy=policy,
        context_length=context_length,
    )
    receipts = [_receipt(split=0)]
    if policy == "split12":
        receipts.append(_receipt(split=12))
    return {
        "attempts": 16,
        "acceptances": 16,
        "target_model_forwards": 127,
        "graph_replays": 127,
        "intermediate_token_d2h_calls": 0,
        "final_token_d2h_calls": 16,
        "final_token_d2h_bytes": 1_016,
        "sampled_logit_d2h_calls": 0,
        "output_budget_clipped": 1,
        "block_boundary_clipped": 0,
        "commits": 16,
        "committed_tokens": 127,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
        "maximum_host_visible_gap_ns": 4_000_000,
        "requested_width_histogram": {"8": 16},
        "authorized_width_histogram": {"7": 1, "8": 15},
        "fallback_counts": {},
        "quarantine_reason": None,
        "capture_receipts": receipts,
        "selected_flash_attn_num_splits": selected_split,
    }


def _case_row(
    policy: str,
    *,
    context_length: int = 2049,
    repetition: int = 0,
    order_position: int | None = None,
) -> dict:
    selected_split = expected_flash_attn_num_splits(
        policy=policy,
        context_length=context_length,
    )
    identity_character = "c" if selected_split == 0 else "d"
    candidate = policy == "split12"
    if order_position is None:
        order_position = policy_order(
            repetition,
            CONTEXT_LENGTHS.index(context_length),
        ).index(policy)
    samples = [1_000_000.0] * (GENERATED_TOKENS - 1)
    return {
        "schema_version":
            "exact-burst-medium-split-k.case.v1",
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "repetition": repetition,
        "order_position": order_position,
        "context_length": context_length,
        "generated_tokens": GENERATED_TOKENS,
        "prompt_sha256": "b" * 64,
        "output_token_ids": list(range(GENERATED_TOKENS)),
        "output_text_sha256": "e" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "amortized_tpot_samples_ns": samples,
        "amortized_tpot_median_ns": 1_000_000.0,
        "amortized_tpot_p95_ns": 1_000_000.0,
        "amortized_tpot_p99_ns": 1_000_000.0,
        "decode_host_ns": [900_000] * 16,
        "decode_cuda_ns": [700_000] * 16,
        "output_tokens_per_second": 914.285714,
        "host_visible_burst_gaps_ns": [4_000_000] * 16,
        "maximum_host_visible_burst_gap_ns": 4_000_000,
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "capture_duration_ns":
            2_000_012 if candidate else 1_000_000,
        "capture_allocated_delta_bytes":
            800_012 if candidate else 400_000,
        "capture_reserved_delta_bytes":
            4_000_012 if candidate else 2_000_000,
        "capture_retained_static_bytes":
            1_800_028 if candidate else 900_008,
        "reserved_scratch_blocks": 1,
        "replay_graph_identity_sha256":
            identity_character * 64,
        "replay_graph_identity_counts": {
            identity_character * 64: GENERATED_TOKENS - 1,
        },
        "replay_flash_attn_num_splits": selected_split,
        "correctness_trace": False,
        "exact_greedy_decode_burst_summary":
            _summary(policy, context_length),
    }


def test_frozen_workload_and_alternating_order() -> None:
    assert POLICIES == ("auto", "split12")
    assert POLICY_CONFIGS == {
        "auto": {
            "exact_greedy_decode_burst_medium_split_k": False,
        },
        "split12": {
            "exact_greedy_decode_burst_medium_split_k": True,
        },
    }
    assert CONTEXT_LENGTHS == (
        1025,
        1537,
        2049,
        2561,
        3073,
        3585,
        4090,
        6145,
    )
    assert GENERATED_TOKENS == 128
    assert REPETITIONS == 5
    assert WARMUP_REPETITIONS == 2
    for repetition in range(REPETITIONS):
        for context_index in range(len(CONTEXT_LENGTHS)):
            expected = (
                POLICIES
                if (repetition + context_index) % 2 == 0
                else tuple(reversed(POLICIES))
            )
            assert policy_order(
                repetition,
                context_index,
            ) == expected


def test_manifest_binds_exact_inventory_and_source() -> None:
    manifest = build_workload_manifest(
        model="/models/Qwen3-0.6B",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment={"fixture": True},
    )
    assert manifest["contexts"] == list(CONTEXT_LENGTHS)
    assert manifest["policies"] == list(POLICIES)
    assert manifest["performance_row_count"] == 80
    assert manifest["correctness_row_count"] == 64
    assert manifest["repetitions"] == REPETITIONS
    assert manifest["warmup_repetitions"] == WARMUP_REPETITIONS
    assert manifest["generated_tokens"] == GENERATED_TOKENS
    assert manifest["source_commit"] == SOURCE_COMMIT
    identities = performance_identities(
        repetitions=REPETITIONS
    )
    assert len(identities) == 80
    assert len(set(identities)) == 80
    assert (
        "tools/profile_exact_burst_medium_split_k.py"
        in SOURCE_FILES
    )
    assert (
        "tools/exact_burst_medium_split_k_verify.py"
        in SOURCE_FILES
    )
    microgate = build_workload_manifest(
        model="/models/Qwen3-0.6B",
        run_tag=RUN_TAG + "-micro",
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment={"fixture": True},
        repetitions=3,
        warmup_repetitions=1,
    )
    assert microgate["repetitions"] == 3
    assert microgate["warmup_repetitions"] == 1
    assert microgate["performance_row_count"] == 48


def test_llm_arms_only_differ_by_medium_split_k_flag() -> None:
    calls = []

    def fake_llm(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace()

    with patch.dict(
        sys.modules,
        {"tinyvllm": SimpleNamespace(LLM=fake_llm)},
    ):
        for policy in POLICIES:
            _construct_llm(
                model="/models/qwen3",
                prompt_tokens=2049,
                generated_tokens=GENERATED_TOKENS,
                gpu_memory_utilization=0.5,
                policy=policy,
            )

    auto_kwargs = calls[0][1]
    candidate_kwargs = calls[1][1]
    differing = {
        key
        for key in set(auto_kwargs) | set(candidate_kwargs)
        if auto_kwargs.get(key) != candidate_kwargs.get(key)
    }
    assert differing == {
        "exact_greedy_decode_burst_medium_split_k"
    }
    assert auto_kwargs["exact_greedy_decode_burst"] is True
    assert auto_kwargs["exact_greedy_decode_burst_tokens"] == 8
    assert auto_kwargs[
        "exact_greedy_decode_burst_medium_split_k"
    ] is False
    assert candidate_kwargs[
        "exact_greedy_decode_burst_medium_split_k"
    ] is True


@pytest.mark.parametrize(
    ("policy", "context_length", "expected"),
    [
        ("auto", 2049, 0),
        ("split12", 1025, 0),
        ("split12", 1537, 12),
        ("split12", 2049, 12),
        ("split12", 4090, 12),
        ("split12", 6145, 0),
    ],
)
def test_expected_split_mapping(
    policy: str,
    context_length: int,
    expected: int,
) -> None:
    assert expected_flash_attn_num_splits(
        policy=policy,
        context_length=context_length,
    ) == expected


def test_case_row_binds_raw_metrics_receipts_and_selected_graph() -> None:
    for policy in POLICIES:
        for context_length in CONTEXT_LENGTHS:
            row = validate_case_row(
                _case_row(policy, context_length=context_length)
            )
            assert len(
                row["amortized_tpot_samples_ns"]
            ) == GENERATED_TOKENS - 1
            assert math.isfinite(
                row["output_tokens_per_second"]
            )
            assert row[
                "replay_flash_attn_num_splits"
            ] == expected_flash_attn_num_splits(
                policy=policy,
                context_length=context_length,
            )

    wrong_candidate = _case_row(
        "split12",
        context_length=2049,
    )
    wrong_candidate["replay_flash_attn_num_splits"] = 0
    wrong_candidate["replay_graph_identity_sha256"] = "c" * 64
    with pytest.raises(
        ValueError,
        match="selected split-K mapping",
    ):
        validate_case_row(wrong_candidate)

    wrong_control = _case_row(
        "split12",
        context_length=6145,
    )
    wrong_control["replay_flash_attn_num_splits"] = 12
    wrong_control["replay_graph_identity_sha256"] = "d" * 64
    with pytest.raises(
        ValueError,
        match="selected split-K mapping",
    ):
        validate_case_row(wrong_control)

    mixed_boundary = _case_row(
        "split12",
        context_length=4090,
    )
    mixed_boundary["replay_graph_identity_counts"] = {
        "d" * 64: 7,
        "c" * 64: GENERATED_TOKENS - 8,
    }
    validate_case_row(mixed_boundary)

    unknown_graph = deepcopy(mixed_boundary)
    unknown_graph["replay_graph_identity_counts"]["f" * 64] = 1
    with pytest.raises(
        ValueError,
        match="replay graph identity inventory",
    ):
        validate_case_row(unknown_graph)

    stale_percentile = _case_row("auto")
    stale_percentile["amortized_tpot_p95_ns"] = 2_000_000
    with pytest.raises(
        ValueError,
        match="TPOT summary",
    ):
        validate_case_row(stale_percentile)

    malformed_prompt = _case_row("auto")
    malformed_prompt["prompt_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="prompt digest"):
        validate_case_row(malformed_prompt)


def test_summary_rejects_missing_duplicate_and_mixed_source_rows() -> None:
    rows = [
        _case_row(
            policy,
            context_length=context_length,
            repetition=repetition,
            order_position=order_position,
        )
        for repetition in range(REPETITIONS)
        for context_index, context_length
        in enumerate(CONTEXT_LENGTHS)
        for order_position, policy in enumerate(
            policy_order(repetition, context_index)
        )
    ]
    summary = summarize_rows(
        rows,
        expected_repetitions=REPETITIONS,
    )
    assert summary["row_count"] == 80
    assert summary["comparison_set_count"] == 40
    assert summary["all_outputs_exact"] is True

    with pytest.raises(ValueError, match="row inventory"):
        summarize_rows(
            rows[:-1],
            expected_repetitions=REPETITIONS,
        )

    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows(
            rows + [deepcopy(rows[0])],
            expected_repetitions=REPETITIONS,
        )

    mixed = deepcopy(rows)
    mixed[-1]["source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="source identity"):
        summarize_rows(
            mixed,
            expected_repetitions=REPETITIONS,
        )


def test_invalid_policy_and_non_finite_metric_are_rejected() -> None:
    invalid_policy = _case_row("auto")
    invalid_policy["policy"] = "unknown"
    with pytest.raises(ValueError, match="policy"):
        validate_case_row(invalid_policy)

    non_finite = _case_row("auto")
    non_finite["ttft_ns"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_case_row(non_finite)


def test_correctness_rows_bind_sidecars_and_selected_graphs(
    tmp_path: Path,
) -> None:
    rows = []
    expected_ordinals = {
        "prefill-final": None,
        "decode-first": 0,
        "decode-middle": 7,
        "decode-final": 6,
    }
    for context_length in CONTEXT_LENGTHS:
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
            for point in (
                "prefill-final",
                "decode-first",
                "decode-middle",
                "decode-final",
            ):
                sidecar = write_float32_sidecar(
                    tmp_path,
                    (
                        f"logits/{context_length}-{policy}-"
                        f"{point}.f32"
                    ),
                    (1.0, 2.0, 4.0, 3.0),
                )
                burst_sample = point != "prefill-final"
                rows.append({
                    "schema_version": (
                        "exact-burst-medium-split-k."
                        "correctness.v1"
                    ),
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "context_length": context_length,
                    "generated_tokens": GENERATED_TOKENS,
                    "sampling_point": point,
                    "prompt_sha256": "b" * 64,
                    "output_token_ids": list(
                        range(GENERATED_TOKENS)
                    ),
                    "output_text_sha256": "e" * 64,
                    "logits_path": sidecar["path"],
                    "logits_shape": [1, 4],
                    "logits_element_count":
                        sidecar["element_count"],
                    "logits_byte_length":
                        sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "correctness_trace": True,
                    "trace_identity": (
                        "gate-only-exact-burst-medium-split-k-"
                        "correctness-v1"
                    ),
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

    assert len(correctness_identities()) == 64
    validated = validate_correctness_rows(
        rows,
        run_dir=tmp_path,
    )
    assert len(validated) == 64

    wrong_graph = deepcopy(rows)
    candidate = next(
        row
        for row in wrong_graph
        if row["policy"] == "split12"
        and row["context_length"] == 2049
        and row["sampling_point"] == "decode-first"
    )
    candidate["trace_flash_attn_num_splits"] = 0
    candidate["trace_graph_identity_sha256"] = "c" * 64
    with pytest.raises(
        ValueError,
        match="correctness selected split-K mapping",
    ):
        validate_correctness_rows(
            wrong_graph,
            run_dir=tmp_path,
        )

    with pytest.raises(
        ValueError,
        match="correctness row inventory",
    ):
        validate_correctness_rows(
            rows[:-1],
            run_dir=tmp_path,
        )
