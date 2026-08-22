"""Dependency-light contracts for the replay-aware metadata benchmark worker."""

from __future__ import annotations

from copy import deepcopy
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.profile_replay_aware_decode_metadata import (
    context_cases,
    nearest_rank_percentile,
    policy_order,
    summarize_rows,
)


def _row(policy: str) -> dict:
    landing_summary = {
        "eligible_steps": 0,
        "optimized_steps": 0,
        "allocation_count": 0,
        "growth_count": 0,
        "staged_h2d_bytes": 0,
        "avoided_temporary_cuda_tensors": 0,
        "avoided_blanket_zero_bytes": 0,
        "current_pinned_capacity_bytes": 0,
        "peak_pinned_capacity_bytes": 0,
        "fallback_counts": {},
    }
    if policy == "on":
        landing_summary.update({
            "eligible_steps": 127,
            "optimized_steps": 127,
            "allocation_count": 5,
            "growth_count": 5,
            "staged_h2d_bytes": 5_080,
            "avoided_temporary_cuda_tensors": 635,
            "avoided_blanket_zero_bytes": 1_000_000,
            "current_pinned_capacity_bytes": 2_048,
            "peak_pinned_capacity_bytes": 2_048,
        })
    return {
        "schema_version":
            "replay-aware-decode-metadata.case.v1",
        "run_tag": "20260822-qwen3-06b-replay-meta-test",
        "source_commit": "a" * 40,
        "policy": policy,
        "repetition": 0,
        "context_bucket": "short",
        "prompt_tokens": 256,
        "generated_tokens": 128,
        "output_token_ids": list(range(128)),
        "output_text_sha256": "b" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "tpot_samples_ns": [1_000_000] * 127,
        "decode_host_ns": [900_000] * 127,
        "decode_cuda_ns": [700_000] * 127,
        "output_tokens_per_second": 914.285714,
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "landing_summary": landing_summary,
    }


def test_worker_contract_helpers_are_deterministic():
    assert context_cases() == (
        ("short", 256, 128),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    )
    assert policy_order(0) == ("off", "on")
    assert policy_order(1) == ("on", "off")
    assert nearest_rank_percentile(
        [1, 2, 3, 4, 5],
        0.95,
    ) == 5


def test_summarize_rows_accepts_exact_complete_pair():
    summary = summarize_rows([_row("off"), _row("on")])

    assert summary["schema_version"] == (
        "replay-aware-decode-metadata.summary.v1"
    )
    assert summary["row_count"] == 2
    assert summary["pair_count"] == 1
    assert summary["all_outputs_exact"] is True
    assert summary["all_on_steps_optimized"] is True
    assert summary["peak_pinned_capacity_bytes"] == 2_048


def test_summarize_rows_rejects_output_token_mismatch():
    off = _row("off")
    on = _row("on")
    on["output_token_ids"][-1] = 999

    with pytest.raises(
        ValueError,
        match="output token mismatch",
    ):
        summarize_rows([off, on])


def test_summarize_rows_rejects_missing_optimized_steps():
    off = _row("off")
    on = _row("on")
    on["landing_summary"]["optimized_steps"] = 126

    with pytest.raises(
        ValueError,
        match="optimized decode step inventory mismatch",
    ):
        summarize_rows([off, on])


def test_summarize_rows_rejects_missing_cost_field():
    off = _row("off")
    on = _row("on")
    del on["landing_summary"]["peak_pinned_capacity_bytes"]

    with pytest.raises(
        ValueError,
        match="landing summary field is missing",
    ):
        summarize_rows([off, on])


def test_summarize_rows_rejects_duplicate_policy_identity():
    first = _row("off")
    duplicate = deepcopy(first)

    with pytest.raises(
        ValueError,
        match="duplicate case identity",
    ):
        summarize_rows([first, duplicate])


def main() -> None:
    test_worker_contract_helpers_are_deterministic()
    test_summarize_rows_accepts_exact_complete_pair()
    test_summarize_rows_rejects_output_token_mismatch()
    test_summarize_rows_rejects_missing_optimized_steps()
    test_summarize_rows_rejects_missing_cost_field()
    test_summarize_rows_rejects_duplicate_policy_identity()
    print("replay-aware metadata profile tests passed")


if __name__ == "__main__":
    main()
