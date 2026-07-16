"""Prefix-cache gate report tests.

Run: python3 tools/test_profile_prefix_cache.py
"""

import os
import sys
from types import ModuleType
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.profile_prefix_cache import (
    adjusted_ttft_ms,
    audit_artifact_payloads,
    build_manifest,
    clone_logits_for_capture,
    compare_logits,
    decide_gate,
    expected_reusable_tokens,
    expected_shared_reusable_tokens,
    make_token_prompt,
    materialize_captured_logits,
    parse_int_list,
    schedule_and_run_prefill,
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


def test_schedule_and_run_prefill_records_metadata_and_instrumentation():
    import tools.profile_prefix_cache as profile_prefix_cache

    class FakeTensor:
        is_cuda = False

        def __init__(self, values):
            self.values = list(values)

        def detach(self):
            return self

        def float(self):
            return self

        def clone(self):
            return FakeTensor(self.values)

        def cpu(self):
            return self

    class FakeSampler:
        def forward(self, logits, temperatures):
            assert temperatures == [0.0]
            return [2]

    class FakeRunner:
        def __init__(self):
            self.sampler = FakeSampler()

        def call(self, method_name, seqs, is_prefill, do_sample, batch_kind):
            assert method_name == "run"
            assert is_prefill is True
            assert do_sample is True
            assert batch_kind is None
            return self.sampler.forward(FakeTensor([1.0, 2.0, 3.0]), [0.0])

    class FakeSequence:
        seq_id = 7
        num_cached_tokens = 256
        prefill_chunk_start = 256
        prefill_chunk_end = 320
        block_table = [4, 9]

        def __len__(self):
            return 320

    seq = FakeSequence()

    class FakeScheduler:
        def schedule(self):
            return [seq], True, True

        def postprocess(self, seqs, token_ids, is_prefill, do_sample, batch_kind):
            assert seqs == [seq]
            assert token_ids == [2]
            assert is_prefill is True
            assert do_sample is True
            assert batch_kind is None

    class FakeLLM:
        def __init__(self):
            self.scheduler = FakeScheduler()
            self.model_runner = FakeRunner()
            self.tokenizer = SimpleNamespace(
                decode=lambda token_ids: f"token-{token_ids[0]}"
            )
            self.prompts = []

        def add_request(self, prompt, params):
            self.prompts.append((prompt, params))

    original_sync = profile_prefix_cache.cuda_sync
    original_tinyvllm = sys.modules.get("tinyvllm")
    fake_tinyvllm = ModuleType("tinyvllm")
    fake_tinyvllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    sys.modules["tinyvllm"] = fake_tinyvllm
    profile_prefix_cache.cuda_sync = lambda: None
    try:
        result = schedule_and_run_prefill(FakeLLM(), [[1] * 320])
    finally:
        profile_prefix_cache.cuda_sync = original_sync
        if original_tinyvllm is None:
            del sys.modules["tinyvllm"]
        else:
            sys.modules["tinyvllm"] = original_tinyvllm

    assert result["metadata"] == [
        {
            "seq_id": 7,
            "prompt_tokens": 320,
            "cached_tokens": 256,
            "chunk_start": 256,
            "chunk_end": 320,
            "query_tokens": 64,
            "block_table": [4, 9],
        }
    ]
    assert result["token_ids"] == [2]
    assert result["decoded"] == ["token-2"]
    assert result["logits"].values == [1.0, 2.0, 3.0]
    assert result["raw_ttft_ms"] >= 0.0
    assert result["capture_overhead_ms"] == 0
    assert result["ttft_ms"] == result["raw_ttft_ms"]


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


def test_audit_artifact_payloads_recomputes_raw_performance_rows():
    correctness = [{"case": "boundary", "correct": True}]
    performance_rows = []
    performance_cases = []
    for prefix, cold_ms, warm_ms in (
        (256, 10.0, 9.8),
        (1024, 20.0, 15.0),
        (2048, 40.0, 28.0),
    ):
        rows = [
            {
                "state": state,
                "ttft_ms": ttft_ms,
                "query_tokens": query_tokens,
                "cached_tokens": cached_tokens,
                "correct": True,
                "shared_prefix_tokens": prefix,
                "suffix_tokens": 300,
            }
            for state, ttft_ms, query_tokens, cached_tokens in (
                ("cold", cold_ms, prefix + 300, 0),
                ("warm", warm_ms, 300, prefix),
                ("cache_cleared", cold_ms, prefix + 300, 0),
            )
        ]
        performance_rows.extend(rows)
        summaries = {
            state: summarize_case_rows(
                [row for row in rows if row["state"] == state]
            )
            for state in ("cold", "warm", "cache_cleared")
        }
        performance_cases.append(
            {
                "shared_prefix_tokens": prefix,
                "suffix_tokens": 300,
                "expected_reusable_tokens": prefix,
                "cold": summaries["cold"],
                "warm": summaries["warm"],
                "cache_cleared": summaries["cache_cleared"],
                "cold_median_query_tokens": prefix + 300,
                "warm_median_query_tokens": 300,
                "warm_median_cached_tokens": prefix,
                "all_correct": True,
            }
        )
    decision = decide_gate(correctness, performance_cases)
    summary = {
        "correctness_rows": correctness,
        "performance_cases": performance_cases,
        "decision": decision,
    }

    assert audit_artifact_payloads(
        correctness,
        performance_rows,
        summary,
        repetitions=1,
    ) == []

    summary["performance_cases"][1]["warm"]["median_ttft_ms"] = 14.0
    errors = audit_artifact_payloads(
        correctness,
        performance_rows,
        summary,
        repetitions=1,
    )
    assert any("raw rows" in error for error in errors)

    summary["performance_cases"][1]["warm"]["median_ttft_ms"] = 15.0
    summary["performance_cases"][1]["expected_reusable_tokens"] = 768
    errors = audit_artifact_payloads(
        correctness,
        performance_rows,
        summary,
        repetitions=1,
    )
    assert any("expected reusable tokens" in error for error in errors)


def main():
    test_expected_reusable_tokens_keeps_sampleable_suffix()
    test_expected_shared_reusable_tokens_requires_full_shared_blocks()
    test_make_token_prompt_is_deterministic_and_offset_sensitive()
    test_parse_int_list_accepts_comma_separated_values()
    test_build_manifest_records_source_hashes()
    test_compare_logits_requires_argmax_and_numeric_tolerance()
    test_logit_capture_defers_cpu_transfer_until_after_timing()
    test_adjusted_ttft_excludes_capture_instrumentation()
    test_schedule_and_run_prefill_records_metadata_and_instrumentation()
    test_summarize_case_rows_reports_medians_and_correctness()
    test_decide_gate_requires_correctness_and_two_large_prefix_wins()
    test_decide_gate_rejects_any_correctness_failure_or_warm_regression()
    test_decide_gate_rejects_cached_or_query_token_mismatch()
    test_audit_artifact_payloads_recomputes_raw_performance_rows()
    print("prefix cache profiler tests passed")


if __name__ == "__main__":
    main()
