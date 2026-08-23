#!/usr/bin/env python3
"""Contracts for the split-phase ragged-coalescing profiler."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.profile_exact_burst_ragged_coalescing import (
    CASE_SCHEMA_VERSION,
    CORRECTNESS_SCHEMA_VERSION,
    POLICIES,
    POLICY_CONFIGS,
    SOURCE_FILES,
    SUMMARY_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
    _construct_llm,
    _sampled_local_ordinals,
    correctness_trace_for_step,
    correctness_identities,
    performance_identities,
    validate_case_row,
)
from tools.test_profile_exact_burst_split_phase import (
    _case_row as _split_case_row,
)


def _candidate_row() -> dict:
    row = deepcopy(
        _split_case_row("decode_burst_k8_split_phase")
    )
    row.update(
        schema_version=CASE_SCHEMA_VERSION,
        policy="decode_burst_k8_split_phase_ragged",
        tail_seven_elapsed_ns=7_000_000,
    )
    row["decode_host_ns"] = [900_000] * 17
    row["decode_cuda_ns"] = [700_000] * 17
    row["host_visible_burst_gaps_ns"] = (
        row["split_phase_inventory"]["host_visible_gaps_ns"]
        + [3_000_000, 2_500_000]
    )
    summary = row["exact_greedy_decode_burst_summary"]
    summary.update({
        "attempts": 17,
        "acceptances": 17,
        "target_model_forwards": 127,
        "graph_replays": 127,
        "final_token_d2h_calls": 2,
        "final_token_d2h_bytes": 56,
        "output_budget_clipped": 0,
        "block_boundary_clipped": 0,
        "commits": 17,
        "committed_tokens": 127,
        "requested_width_histogram": {
            "3": 1,
            "4": 1,
            "8": 15,
        },
        "authorized_width_histogram": {
            "3": 1,
            "4": 1,
            "8": 15,
        },
        "fallback_counts": {},
        "maximum_host_visible_gap_ns": 3_000_000,
    })
    return row


def test_frozen_matrix_and_source_inventory() -> None:
    assert CASE_SCHEMA_VERSION == (
        "exact-burst-ragged-coalescing.case.v1"
    )
    assert CORRECTNESS_SCHEMA_VERSION == (
        "exact-burst-ragged-coalescing.correctness.v1"
    )
    assert SUMMARY_SCHEMA_VERSION == (
        "exact-burst-ragged-coalescing.summary.v1"
    )
    assert WORKLOAD_SCHEMA_VERSION == (
        "exact-burst-ragged-coalescing.workload.v1"
    )
    assert POLICIES == (
        "decode_burst_k4",
        "decode_burst_k8_split_phase",
        "decode_burst_k8_split_phase_ragged",
    )
    assert len(performance_identities(repetitions=5)) == 45
    assert len(correctness_identities()) == 36
    assert POLICY_CONFIGS[
        "decode_burst_k8_split_phase_ragged"
    ]["ragged_coalescing"] is True
    assert (
        "tools/profile_exact_burst_ragged_coalescing.py"
        in SOURCE_FILES
    )
    assert (
        "tools/exact_burst_ragged_coalescing_verify.py"
        in SOURCE_FILES
    )


def test_candidate_row_binds_k8_plus_k4_k3_inventory() -> None:
    row = validate_case_row(_candidate_row())
    summary = row["exact_greedy_decode_burst_summary"]
    assert summary["commits"] == 17
    assert summary["committed_tokens"] == 127
    assert summary["prefix_commits"] == 15
    assert summary["suffix_commits"] == 15
    assert summary["final_token_d2h_calls"] == 2
    assert summary["fallback_counts"] == {}
    assert summary["requested_width_histogram"] == {
        "3": 1,
        "4": 1,
        "8": 15,
    }
    assert len(row["decode_host_ns"]) == 17
    assert len(row["host_visible_burst_gaps_ns"]) == 32
    assert row["tail_seven_elapsed_ns"] == 7_000_000

    invalid = _candidate_row()
    invalid["exact_greedy_decode_burst_summary"][
        "fallback_counts"
    ] = {"split_phase_requires_k8": 1}
    try:
        validate_case_row(invalid)
    except ValueError as error:
        assert str(error) == (
            "ragged coalescing fallback inventory mismatch"
        )
    else:
        raise AssertionError("ragged fallback was accepted")


def test_candidate_runtime_and_correctness_trace_are_explicit() -> None:
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
                prompt_tokens=256,
                generated_tokens=128,
                gpu_memory_utilization=0.5,
                policy=policy,
            )

    by_policy = {
        policy: kwargs
        for policy, (_model, kwargs) in zip(POLICIES, calls)
    }
    assert by_policy["decode_burst_k4"][
        "exact_greedy_decode_burst_ragged_coalescing"
    ] is False
    assert by_policy["decode_burst_k8_split_phase"][
        "exact_greedy_decode_burst_ragged_coalescing"
    ] is False
    assert by_policy["decode_burst_k8_split_phase_ragged"][
        "exact_greedy_decode_burst_ragged_coalescing"
    ] is True
    assert correctness_trace_for_step(
        "decode_burst_k8_split_phase_ragged",
        emitted_total=125,
        generated_tokens=128,
    ) is True
    assert _sampled_local_ordinals(
        "decode_burst_k8_split_phase_ragged"
    ) == (0, 2, 7)


def main() -> None:
    test_frozen_matrix_and_source_inventory()
    test_candidate_row_binds_k8_plus_k4_k3_inventory()
    test_candidate_runtime_and_correctness_trace_are_explicit()
    print("exact burst ragged-coalescing profile tests passed")


if __name__ == "__main__":
    main()
