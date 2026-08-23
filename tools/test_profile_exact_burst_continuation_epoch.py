#!/usr/bin/env python3
"""Contracts for the exact-burst continuation epoch profiler."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.profile_exact_burst_continuation_epoch import (
    CASE_SCHEMA_VERSION,
    CONTEXT_CASES,
    CORRECTNESS_SCHEMA_VERSION,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
    build_workload_manifest,
    correctness_identities,
    performance_identities,
    policy_order,
    validate_case_row,
)


RUN_TAG = "20260823-qwen3-06b-exact-burst-continuation-test"
SOURCE_COMMIT = "a" * 40


def _capture_receipt() -> dict:
    return {
        "graph_identity_sha256": "c" * 64,
        "graph_generation": 7,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_static_bytes": 902_000,
        "scratch_block_count": 1,
        "correctness_trace": False,
    }


def _burst_summary(policy: str) -> dict:
    enabled = POLICY_CONFIGS[policy]["enabled"]
    continuation = POLICY_CONFIGS[policy]["continuation"]
    width = POLICY_CONFIGS[policy]["width"]
    commits = math.ceil(127 / width) if enabled else 0
    hits = commits - 1 if continuation else 0
    return {
        "attempts": commits,
        "acceptances": commits,
        "target_model_forwards": 127 if enabled else 0,
        "graph_replays": 127 if enabled else 0,
        "intermediate_token_d2h_calls": 0,
        "final_token_d2h_calls": commits,
        "final_token_d2h_bytes": 127 * 8 if enabled else 0,
        "sampled_logit_d2h_calls": 0,
        "output_budget_clipped": int(enabled),
        "block_boundary_clipped": int(enabled),
        "commits": commits,
        "committed_tokens": 127 if enabled else 0,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
        "maximum_host_visible_gap_ns": (
            4_000_000 if enabled else 0
        ),
        "continuation_attempts": commits if continuation else 0,
        "continuation_hits": hits,
        "cold_binds": 1 if continuation else 0,
        "continuation_tokens": hits * width,
        "continuation_bursts": hits,
        "skipped_static_reset_operations": hits * 7,
        "skipped_scalar_bind_operations": hits * 5,
        "skipped_block_table_constructions": hits,
        "skipped_block_table_copy_calls": hits,
        "skipped_block_table_bytes": hits * 16,
        "requested_width_histogram": (
            {str(width): commits} if enabled else {}
        ),
        "authorized_width_histogram": (
            {str(width): commits} if enabled else {}
        ),
        "fallback_counts": {},
        "continuation_miss_counts": (
            {"receipt_missing": 1} if continuation else {}
        ),
        "continuation_invalidation_counts": {},
        "quarantine_reason": None,
        "capture_receipts": (
            [_capture_receipt()] if enabled else []
        ),
    }


def _case_row(policy: str) -> dict:
    prompt_tokens, generated_tokens = {
        name: (prompt, generated)
        for name, prompt, generated in CONTEXT_CASES
    }["short"]
    enabled = POLICY_CONFIGS[policy]["enabled"]
    width = POLICY_CONFIGS[policy]["width"]
    decode_steps = math.ceil(127 / width) if enabled else 127
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "selectable": POLICY_CONFIGS[policy]["selectable"],
        "burst_width": width,
        "repetition": 0,
        "context_bucket": "short",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "output_token_ids": list(range(generated_tokens)),
        "output_text_sha256": "b" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "amortized_tpot_samples_ns": [1_000_000] * 127,
        "amortized_tpot_median_ns": 1_000_000,
        "amortized_tpot_p95_ns": 1_000_000,
        "amortized_tpot_p99_ns": 1_000_000,
        "decode_host_ns": [900_000] * decode_steps,
        "decode_cuda_ns": [700_000] * decode_steps,
        "output_tokens_per_second": 914.285714,
        "host_visible_burst_gaps_ns": (
            [4_000_000] * decode_steps if enabled else []
        ),
        "maximum_host_visible_burst_gap_ns": (
            4_000_000 if enabled else 0
        ),
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "capture_duration_ns": 1_000_000 if enabled else 0,
        "capture_allocated_delta_bytes": (
            400_000 if enabled else 0
        ),
        "capture_reserved_delta_bytes": (
            2_000_000 if enabled else 0
        ),
        "capture_retained_static_bytes": (
            902_000 if enabled else 0
        ),
        "reserved_scratch_blocks": 1 if enabled else 0,
        "correctness_trace": False,
        "exact_greedy_decode_burst_summary":
            _burst_summary(policy),
    }


def test_frozen_schema_and_inventory() -> None:
    assert CASE_SCHEMA_VERSION == (
        "exact-burst-continuation-epoch.case.v1"
    )
    assert CORRECTNESS_SCHEMA_VERSION == (
        "exact-burst-continuation-epoch.correctness.v1"
    )
    assert SUMMARY_SCHEMA_VERSION == (
        "exact-burst-continuation-epoch.summary.v1"
    )
    assert SOURCE_SCHEMA_VERSION == (
        "exact-burst-continuation-epoch.source.v1"
    )
    assert WORKLOAD_SCHEMA_VERSION == (
        "exact-burst-continuation-epoch.workload.v1"
    )
    assert POLICIES == (
        "host_greedy",
        "decode_burst_k4",
        "decode_burst_k4_continuation",
        "decode_burst_k8",
    )
    performance = performance_identities(repetitions=5)
    correctness = correctness_identities()
    assert len(performance) == len(set(performance)) == 60
    assert len(correctness) == len(set(correctness)) == 48
    assert policy_order(0, 0) != policy_order(0, 1)
    assert set(policy_order(3, 2)) == set(POLICIES)


def test_only_continuation_arm_is_selectable() -> None:
    selectable = tuple(
        policy
        for policy in POLICIES
        if POLICY_CONFIGS[policy]["selectable"]
    )
    assert selectable == ("decode_burst_k4_continuation",)
    assert POLICY_CONFIGS[
        "decode_burst_k4_continuation"
    ]["continuation"] is True
    assert all(
        POLICY_CONFIGS[policy]["continuation"] is False
        for policy in POLICIES
        if policy != "decode_burst_k4_continuation"
    )


def test_workload_and_source_inventory_are_source_bound() -> None:
    manifest = build_workload_manifest(
        model="/models/qwen3",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment={"python_version": "3.10"},
    )
    assert manifest["performance_row_count"] == 60
    assert manifest["correctness_row_count"] == 48
    assert manifest["correctness_sampling_points"] == list(
        SAMPLING_POINTS
    )
    assert manifest["policy_configs"][
        "decode_burst_k4_continuation"
    ]["continuation"] is True
    assert (
        "tools/profile_exact_burst_continuation_epoch.py"
        in SOURCE_FILES
    )
    assert (
        "tools/exact_burst_continuation_epoch_gate.py"
        in SOURCE_FILES
    )


def test_case_schema_requires_exact_continuation_counters() -> None:
    row = validate_case_row(
        _case_row("decode_burst_k4_continuation")
    )
    summary = row["exact_greedy_decode_burst_summary"]
    assert summary["continuation_attempts"] == 32
    assert summary["continuation_hits"] == 31
    assert summary["cold_binds"] == 1
    assert summary["continuation_tokens"] == 124
    assert summary["skipped_block_table_constructions"] == 31

    invalid = _case_row("decode_burst_k4_continuation")
    del invalid["exact_greedy_decode_burst_summary"][
        "continuation_hits"
    ]
    try:
        validate_case_row(invalid)
    except ValueError as error:
        assert str(error) == (
            "exact burst continuation summary fields are missing"
        )
    else:
        raise AssertionError(
            "missing continuation counter was accepted"
        )


def main() -> None:
    test_frozen_schema_and_inventory()
    test_only_continuation_arm_is_selectable()
    test_workload_and_source_inventory_are_source_bound()
    test_case_schema_requires_exact_continuation_counters()
    print("exact burst continuation profile tests passed")


if __name__ == "__main__":
    main()
