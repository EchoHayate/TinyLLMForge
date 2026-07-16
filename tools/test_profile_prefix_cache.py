"""Prefix-cache gate report tests.

Run: python3 tools/test_profile_prefix_cache.py
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.profile_prefix_cache import (
    adjusted_ttft_ms,
    build_manifest,
    clone_logits_for_capture,
    compare_logits,
    decide_gate,
    expected_reusable_tokens,
    expected_shared_reusable_tokens,
    make_token_prompt,
    materialize_captured_logits,
    parse_int_list,
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


def test_expected_shared_reusable_tokens_requires_full_shared_blocks():
    assert expected_shared_reusable_tokens(255, 319, 256) == 0
    assert expected_shared_reusable_tokens(256, 320, 256) == 256
    assert expected_shared_reusable_tokens(300, 364, 256) == 256
    assert expected_shared_reusable_tokens(512, 512, 256) == 256


def test_make_token_prompt_is_deterministic_and_offset_sensitive():
    assert make_token_prompt(8, 0) == make_token_prompt(8, 0)
    assert make_token_prompt(8, 0) != make_token_prompt(8, 11)
    assert len(make_token_prompt(257, 3)) == 257
    prefix = make_token_prompt(256, 100)
    producer = prefix + make_token_prompt(64, 311)
    consumer = prefix + make_token_prompt(64, 623)
    assert producer[:256] == consumer[:256]
    assert producer[256:] != consumer[256:]


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("256,1024,2048") == [256, 1024, 2048]


def test_build_manifest_records_source_hashes():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.py"
        source.write_text("print('ok')\n")
        manifest = build_manifest(root, ["source.py"], {"model": "/tmp/model"})
        assert manifest["args"]["model"] == "/tmp/model"
        assert len(manifest["source_sha256"]["source.py"]) == 64


def test_compare_logits_requires_argmax_and_numeric_tolerance():
    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def __sub__(self, other):
            return FakeTensor(
                left - right for left, right in zip(self.values, other.values)
            )

        def abs(self):
            return FakeTensor(abs(value) for value in self.values)

        def max(self):
            return max(self.values)

        def mean(self):
            return sum(self.values) / len(self.values)

        def argmax(self):
            return max(range(len(self.values)), key=self.values.__getitem__)

    reference = FakeTensor([1.0, 3.0, 2.0])
    close = FakeTensor([1.05, 3.0, 1.95])
    comparison = compare_logits(reference, close)
    assert comparison["argmax_match"] is True
    assert comparison["within_tolerance"] is True

    changed_argmax = FakeTensor([1.0, 2.9, 3.1])
    comparison = compare_logits(reference, changed_argmax)
    assert comparison["argmax_match"] is False
    assert comparison["within_tolerance"] is False

    large_delta = FakeTensor([1.0, 3.0, 1.7])
    comparison = compare_logits(reference, large_delta)
    assert comparison["argmax_match"] is True
    assert comparison["within_tolerance"] is False


def test_logit_capture_defers_cpu_transfer_until_after_timing():
    calls = []

    class FakeTensor:
        def detach(self):
            calls.append("detach")
            return self

        def float(self):
            calls.append("float")
            return self

        def clone(self):
            calls.append("clone")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

    captured = clone_logits_for_capture(FakeTensor())
    assert calls == ["detach", "float", "clone"]

    materialized = materialize_captured_logits([captured])
    assert materialized is captured
    assert calls == ["detach", "float", "clone", "cpu"]


def test_adjusted_ttft_excludes_capture_instrumentation():
    assert adjusted_ttft_ms(12.5, 2.0) == 10.5
    assert adjusted_ttft_ms(1.0, 2.0) == 0.0


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
    test_expected_shared_reusable_tokens_requires_full_shared_blocks()
    test_make_token_prompt_is_deterministic_and_offset_sensitive()
    test_parse_int_list_accepts_comma_separated_values()
    test_build_manifest_records_source_hashes()
    test_compare_logits_requires_argmax_and_numeric_tolerance()
    test_logit_capture_defers_cpu_transfer_until_after_timing()
    test_adjusted_ttft_excludes_capture_instrumentation()
    test_summarize_case_rows_reports_medians_and_correctness()
    test_decide_gate_requires_correctness_and_two_large_prefix_wins()
    test_decide_gate_rejects_any_correctness_failure_or_warm_regression()
    test_decide_gate_rejects_cached_or_query_token_mismatch()
    print("prefix cache profiler tests passed")


if __name__ == "__main__":
    main()
