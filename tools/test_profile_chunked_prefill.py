"""Chunked prefill latency profiler helper tests.

跑法：python3 tools/test_profile_chunked_prefill.py
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.profile_chunked_prefill import percentile, summarize_steps


def test_percentile_uses_nearest_rank_without_interpolation():
    assert percentile([], 0.5) == 0.0
    assert percentile([0.4, 0.1, 0.3, 0.2], 0.5) == 0.2
    assert percentile([0.4, 0.1, 0.3, 0.2], 0.95) == 0.4


def test_summarize_steps_splits_prefill_decode_and_decode_gap():
    records = [
        {"step": 0, "kind": "prefill", "tokens": 8, "dt_ms": 10.0, "outputs": 0},
        {"step": 1, "kind": "decode", "tokens": 2, "dt_ms": 2.0, "outputs": 0},
        {"step": 2, "kind": "prefill", "tokens": 4, "dt_ms": 6.0, "outputs": 0},
        {"step": 3, "kind": "decode", "tokens": 2, "dt_ms": 3.0, "outputs": 1},
        {"step": 4, "kind": "decode", "tokens": 1, "dt_ms": 4.0, "outputs": 2},
    ]

    summary = summarize_steps(records)

    assert summary["num_steps"] == 5
    assert summary["prefill"]["steps"] == 2
    assert summary["prefill"]["tokens"] == 12
    assert summary["prefill"]["mean_ms"] == 8.0
    assert summary["decode"]["steps"] == 3
    assert summary["decode"]["tokens"] == 5
    assert summary["decode"]["p50_ms"] == 3.0
    assert summary["decode_gap"]["max_steps_between_decode"] == 2
    assert summary["decode_gap"]["max_ms_between_decode"] == 9.0
    assert summary["first_output_step"] == 3
    assert summary["first_output_ms"] == 21.0


def main():
    test_percentile_uses_nearest_rank_without_interpolation()
    test_summarize_steps_splits_prefill_decode_and_decode_gap()
    print("chunked prefill profiler tests passed")


if __name__ == "__main__":
    main()
