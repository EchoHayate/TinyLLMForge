"""Prefix-cache gate report tests.

Run: python3 tools/test_profile_prefix_cache.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.profile_prefix_cache import (
    decide_gate,
    expected_reusable_tokens,
    make_token_prompt,
    summarize_case_rows,
)


def _perf_case(prefix_tokens, cold_ms, warm_ms, correct=True):
    return {
        "shared_prefix_tokens": prefix_tokens,
        "cold": {"median_ttft_ms": cold_ms},
        "warm": {"median_ttft_ms": warm_ms},
        "all_correct": correct,
        "expected_reusable_tokens": prefix_tokens,
        "warm_median_cached_tokens": prefix_tokens,
        "warm_median_query_tokens": 300,
        "cold_median_query_tokens": prefix_tokens + 300,
    }


def test_expected_reusable_tokens_keeps_sampleable_suffix():
    assert expected_reusable_tokens(255, 256) == 0
    assert expected_reusable_tokens(256, 256) == 0
    assert expected_reusable_tokens(257, 256) == 256
    assert expected_reusable_tokens(512, 256) == 256
    assert expected_reusable_tokens(513, 256) == 512


def test_make_token_prompt_is_deterministic_and_offset_sensitive():
    assert make_token_prompt(8, 0) == make_token_prompt(8, 0)
    assert make_token_prompt(8, 0) != make_token_prompt(8, 11)
    assert len(make_token_prompt(257, 3)) == 257


def test_summarize_case_rows_reports_medians_and_correctness():
    rows = [
        {
            "state": "warm",
            "ttft_ms": 10.0,
            "query_tokens": 300,
            "cached_tokens": 1024,
            "correct": True,
        },
        {
            "state": "warm",
            "ttft_ms": 12.0,
            "query_tokens": 300,
            "cached_tokens": 1024,
            "correct": True,
        },
        {
            "state": "warm",
            "ttft_ms": 11.0,
            "query_tokens": 300,
            "cached_tokens": 1024,
            "correct": True,
        },
    ]
    summary = summarize_case_rows(rows)
    assert summary["median_ttft_ms"] == 11.0
    assert summary["median_query_tokens"] == 300
    assert summary["median_cached_tokens"] == 1024
    assert summary["all_correct"] is True


def test_decide_gate_requires_correctness_and_two_large_prefix_wins():
    correctness = [{"case": "boundary_256", "correct": True}]
    performance = [
        _perf_case(256, 10.0, 10.2),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    decision = decide_gate(correctness, performance)
    assert decision["decision"] == "GO"

    performance[2]["warm"]["median_ttft_ms"] = 35.0
    decision = decide_gate(correctness, performance)
    assert decision["decision"] == "NO_GO"
    assert "2048" in " ".join(decision["reasons"])


def test_decide_gate_rejects_any_correctness_failure_or_warm_regression():
    performance = [
        _perf_case(256, 10.0, 10.6),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    decision = decide_gate([{"case": "triple", "correct": False}], performance)
    assert decision["decision"] == "NO_GO"

    decision = decide_gate([{"case": "triple", "correct": True}], performance)
    assert decision["decision"] == "NO_GO"
    assert "regression" in " ".join(decision["reasons"]).lower()


def test_decide_gate_rejects_cached_or_query_token_mismatch():
    performance = [
        _perf_case(256, 10.0, 9.8),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    performance[1]["warm_median_cached_tokens"] = 768
    decision = decide_gate([{"case": "boundary", "correct": True}], performance)
    assert decision["decision"] == "NO_GO"
    assert "cached-token" in " ".join(decision["reasons"])


def main():
    test_expected_reusable_tokens_keeps_sampleable_suffix()
    test_make_token_prompt_is_deterministic_and_offset_sensitive()
    test_summarize_case_rows_reports_medians_and_correctness()
    test_decide_gate_requires_correctness_and_two_large_prefix_wins()
    test_decide_gate_rejects_any_correctness_failure_or_warm_regression()
    test_decide_gate_rejects_cached_or_query_token_mismatch()
    print("prefix cache profiler tests passed")


if __name__ == "__main__":
    main()
