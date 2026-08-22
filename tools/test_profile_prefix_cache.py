"""Prefix-cache gate report tests.

Run: python3 tools/test_profile_prefix_cache.py
"""

import json
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
    append_jsonl,
    audit_artifact_payloads,
    audit_batch_artifact_payloads,
    build_prefix_memory_rows,
    build_prefix_contract_bundle,
    build_manifest,
    clear_reusable_cache_observation,
    clone_logits_for_capture,
    compare_logits,
    decide_staged_prefix_gate,
    decide_gate,
    expected_reusable_tokens,
    expected_shared_reusable_tokens,
    make_token_prompt,
    materialize_captured_logits,
    parse_int_list,
    prefix_case_shape,
    prefix_cost_observation,
    prefix_source_files,
    render_report,
    summarize_batch_result,
    schedule_and_run_prefill,
    schedule_and_run_prefill_batches,
    summarize_batch_case_rows,
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


def _batch_perf_case(
    prefix_tokens,
    cold_ms,
    warm_ms,
    correct=True,
    batch_size=8,
):
    return {
        "shared_prefix_tokens": prefix_tokens,
        "suffix_tokens": 64,
        "batch_size": batch_size,
        "expected_reusable_tokens_per_request": prefix_tokens,
        "cold": {
            "median_batch_elapsed_ms": cold_ms,
            "median_model_batches": 2,
            "median_total_query_tokens": batch_size * (prefix_tokens + 64),
            "median_total_cached_tokens": 0,
            "median_requests": batch_size,
        },
        "warm": {
            "median_batch_elapsed_ms": warm_ms,
            "median_model_batches": 1,
            "median_total_query_tokens": batch_size * 64,
            "median_total_cached_tokens": batch_size * prefix_tokens,
            "median_requests": batch_size,
        },
        "cache_cleared": {
            "median_batch_elapsed_ms": cold_ms,
            "median_model_batches": 2,
            "median_total_query_tokens": batch_size * (prefix_tokens + 64),
            "median_total_cached_tokens": 0,
            "median_requests": batch_size,
        },
        "all_correct": correct,
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


def test_schedule_and_run_prefill_batches_drains_all_requests():
    import tools.profile_prefix_cache as profile_prefix_cache

    class FakeTensor:
        is_cuda = False

        def __init__(self, rows):
            self.rows = list(rows)

        def detach(self):
            return self

        def float(self):
            return self

        def clone(self):
            return FakeTensor(self.rows)

        def cpu(self):
            return self

        def __getitem__(self, index):
            return self.rows[index]

    class FakeSequence:
        def __init__(self, seq_id):
            self.seq_id = seq_id
            self.num_cached_tokens = 8
            self.prefill_chunk_start = 8
            self.prefill_chunk_end = 9
            self.block_table = [seq_id]

        def __len__(self):
            return 9

    seqs = [FakeSequence(index) for index in range(3)]

    class FakeSampler:
        def forward(self, logits, temperatures):
            return [int(row[-1]) for row in logits.rows]

    class FakeRunner:
        def __init__(self):
            self.sampler = FakeSampler()

        def call(self, method_name, batch, is_prefill, do_sample, batch_kind):
            rows = [[0.0, float(seq.seq_id + 1)] for seq in batch]
            temperatures = [0.0] * len(batch)
            return self.sampler.forward(FakeTensor(rows), temperatures)

    class FakeScheduler:
        def __init__(self):
            self.batches = [seqs[:2], seqs[2:]]
            self.clear_calls = 0
            self.block_manager = SimpleNamespace(
                clear_reusable_cache=self.clear_reusable_cache
            )

        def schedule(self):
            return self.batches.pop(0), True, True

        def clear_reusable_cache(self):
            self.clear_calls += 1

        def postprocess(
            self,
            batch,
            token_ids,
            is_prefill,
            do_sample,
            batch_kind,
        ):
            assert len(batch) == len(token_ids)

    class FakeLLM:
        def __init__(self):
            self.scheduler = FakeScheduler()
            self.model_runner = FakeRunner()
            self.tokenizer = SimpleNamespace(
                decode=lambda token_ids: f"token-{token_ids[0]}"
            )
            self.prompts = []

        def add_request(self, prompt, params):
            self.prompts.append(prompt)

    original_sync = profile_prefix_cache.cuda_sync
    original_tinyvllm = sys.modules.get("tinyvllm")
    fake_tinyvllm = ModuleType("tinyvllm")
    fake_tinyvllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    sys.modules["tinyvllm"] = fake_tinyvllm
    profile_prefix_cache.cuda_sync = lambda: None
    try:
        llm = FakeLLM()
        result = schedule_and_run_prefill_batches(
            llm,
            [[1] * 9 for _ in range(3)],
            clear_cache_between_batches=True,
        )
    finally:
        profile_prefix_cache.cuda_sync = original_sync
        if original_tinyvllm is None:
            del sys.modules["tinyvllm"]
        else:
            sys.modules["tinyvllm"] = original_tinyvllm

    assert result["model_batches"] == 2
    assert llm.scheduler.clear_calls == 1
    assert [row["seq_id"] for row in result["metadata"]] == [0, 1, 2]
    assert result["token_ids"] == [1, 2, 3]
    assert result["decoded"] == ["token-1", "token-2", "token-3"]
    assert result["logits"] == [
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ]


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


def test_summarize_batch_case_rows_reports_admission_and_accounting():
    rows = [
        {
            "batch_elapsed_ms": 20.0,
            "model_batches": 1,
            "total_query_tokens": 512,
            "total_cached_tokens": 8192,
            "requests": 8,
            "correct": True,
        },
        {
            "batch_elapsed_ms": 22.0,
            "model_batches": 1,
            "total_query_tokens": 512,
            "total_cached_tokens": 8192,
            "requests": 8,
            "correct": True,
        },
        {
            "batch_elapsed_ms": 21.0,
            "model_batches": 1,
            "total_query_tokens": 512,
            "total_cached_tokens": 8192,
            "requests": 8,
            "correct": True,
        },
    ]

    summary = summarize_batch_case_rows(rows)

    assert summary["median_batch_elapsed_ms"] == 21.0
    assert summary["median_model_batches"] == 1
    assert summary["median_total_query_tokens"] == 512
    assert summary["median_total_cached_tokens"] == 8192
    assert summary["median_requests"] == 8
    assert summary["all_correct"] is True


def test_summarize_case_rows_reports_p95_and_prefix_costs():
    rows = [
        {
            "ttft_ms": value,
            "query_tokens": 64,
            "cached_tokens": 1024,
            "correct": True,
            "retained_reusable_blocks": 4,
            "retained_logical_kv_bytes": 8192,
            "cuda_peak_allocated_bytes": 100 + index,
            "cuda_peak_reserved_bytes": 200 + index,
            "cache_clear_host_ns": 200_000,
        }
        for index, value in enumerate((10, 11, 12, 13, 20, 21, 22))
    ]

    summary = summarize_case_rows(rows)

    assert summary["median_ttft_ms"] == 13
    assert summary["p95_ttft_ms"] == 22
    assert summary["peak_retained_reusable_blocks"] == 4
    assert summary["peak_retained_logical_kv_bytes"] == 8192
    assert summary["peak_cuda_allocated_bytes"] == 106
    assert summary["peak_cuda_reserved_bytes"] == 206
    assert summary["median_cache_clear_host_ms"] == 0.2


def test_summarize_batch_rows_reports_p95_and_prefix_costs():
    rows = [
        {
            "batch_elapsed_ms": value,
            "model_batches": 1,
            "total_query_tokens": 512,
            "total_cached_tokens": 8192,
            "requests": 8,
            "correct": True,
            "retained_reusable_blocks": 32,
            "retained_logical_kv_bytes": 65536,
            "cuda_peak_allocated_bytes": 300 + index,
            "cuda_peak_reserved_bytes": 400 + index,
            "cache_clear_host_ns": 300_000,
        }
        for index, value in enumerate((30, 31, 32, 33, 40, 41, 42))
    ]

    summary = summarize_batch_case_rows(rows)

    assert summary["median_batch_elapsed_ms"] == 33
    assert summary["p95_batch_elapsed_ms"] == 42
    assert summary["peak_retained_reusable_blocks"] == 32
    assert summary["peak_retained_logical_kv_bytes"] == 65536
    assert summary["peak_cuda_allocated_bytes"] == 306
    assert summary["peak_cuda_reserved_bytes"] == 406
    assert summary["median_cache_clear_host_ms"] == 0.3


def test_prefix_cost_observation_uses_all_hashed_blocks():
    class FakeBlock:
        def __init__(self, block_hash):
            self.hash = block_hash

    class FakeBlockManager:
        block_size = 256
        blocks = [
            FakeBlock(101),
            FakeBlock(-1),
            FakeBlock(202),
            FakeBlock(-1),
        ]

    class FakeLLM:
        scheduler = SimpleNamespace(block_manager=FakeBlockManager())
        last_step_observation = {
            "memory": {
                "cuda_allocated_bytes": 100,
                "cuda_reserved_bytes": 200,
                "cuda_peak_allocated_bytes": 300,
                "cuda_peak_reserved_bytes": 400,
                "kv_capacity_bytes": 8000,
            }
        }

        def capacity_snapshot(self):
            return {
                "num_kvcache_blocks": 4,
                "block_size": 256,
            }

    observation = prefix_cost_observation(FakeLLM())

    assert observation == {
        "retained_reusable_blocks": 2,
        "retained_logical_kv_bytes": 4000,
        "cuda_allocated_bytes": 100,
        "cuda_reserved_bytes": 200,
        "cuda_peak_allocated_bytes": 300,
        "cuda_peak_reserved_bytes": 400,
        "kv_block_bytes": 2000,
    }


def test_prefix_cost_observation_supports_direct_model_runner_path():
    class FakeBlock:
        hash = 101

    memory = {
        "cuda_allocated_bytes": 10,
        "cuda_reserved_bytes": 20,
        "cuda_peak_allocated_bytes": 30,
        "cuda_peak_reserved_bytes": 40,
        "kv_capacity_bytes": 2000,
    }
    llm = SimpleNamespace(
        scheduler=SimpleNamespace(
            block_manager=SimpleNamespace(blocks=[FakeBlock()])
        ),
        last_step_observation=None,
        model_runner=SimpleNamespace(
            memory_snapshot=lambda: dict(memory)
        ),
        capacity_snapshot=lambda: {"num_kvcache_blocks": 2},
    )

    observation = prefix_cost_observation(llm)

    assert observation["retained_reusable_blocks"] == 1
    assert observation["retained_logical_kv_bytes"] == 1000
    assert observation["cuda_peak_reserved_bytes"] == 40


def test_clear_reusable_cache_observation_records_host_cost():
    calls = []

    class FakeBlockManager:
        def clear_reusable_cache(self):
            calls.append("clear")
            return 3

    ticks = iter((100, 160))
    result = clear_reusable_cache_observation(
        FakeBlockManager(),
        clock_ns=lambda: next(ticks),
    )

    assert calls == ["clear"]
    assert result == {
        "cleared_reusable_blocks": 3,
        "cache_clear_host_ns": 60,
    }


def test_append_jsonl_is_canonical_append_only_and_newline_terminated():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        append_jsonl(path, [{"b": 2, "a": 1}])
        append_jsonl(path, [{"c": 3}])

        assert path.read_text() == '{"a":1,"b":2}\n{"c":3}\n'
        assert [
            json.loads(line)
            for line in path.read_text().splitlines()
        ] == [{"a": 1, "b": 2}, {"c": 3}]


def test_prefix_source_files_bind_staged_classifier_source():
    assert (
        "tools/staged_inference_benchmark_contract.py"
        in prefix_source_files()
    )


def test_prefix_case_shape_uses_actual_batch_size():
    assert prefix_case_shape(1024, batch_size=1) == "single-1024"
    assert prefix_case_shape(1024, batch_size=4) == "batch4-1024"
    assert prefix_case_shape(2048, batch_size=8) == "batch8-2048"


def _staged_state(
    *,
    elapsed_ms,
    cached_tokens,
    query_tokens,
    model_batches,
    cuda_reserved=1000,
):
    return {
        "samples": 7,
        "median_elapsed_ms": elapsed_ms,
        "p95_elapsed_ms": elapsed_ms,
        "median_cached_prompt_tokens": cached_tokens,
        "median_executed_query_tokens": query_tokens,
        "median_model_batches": model_batches,
        "peak_cuda_reserved_bytes": cuda_reserved,
        "exact_outputs": True,
        "logit_argmax_match": True,
        "logit_max_abs": 0.0,
        "logit_mean_abs": 0.0,
    }


def _staged_case(prefix_tokens, batch_size):
    expected_cached = prefix_tokens * batch_size
    cold_query = (prefix_tokens + 64) * batch_size
    warm_query = 64 * batch_size
    return {
        "shared_prefix_tokens": prefix_tokens,
        "suffix_tokens": 64,
        "batch_size": batch_size,
        "expected_reusable_tokens": expected_cached,
        "cold": _staged_state(
            elapsed_ms=100.0,
            cached_tokens=0,
            query_tokens=cold_query,
            model_batches=1 if batch_size == 1 else 2,
        ),
        "warm": _staged_state(
            elapsed_ms=75.0,
            cached_tokens=expected_cached,
            query_tokens=warm_query,
            model_batches=1,
            cuda_reserved=1040,
        ),
        "cache_cleared": _staged_state(
            elapsed_ms=101.0,
            cached_tokens=0,
            query_tokens=cold_query,
            model_batches=1 if batch_size == 1 else 2,
        ),
        "retained_reusable_blocks": expected_cached // 256,
        "retained_logical_kv_bytes": expected_cached * 32,
        "median_cache_clear_host_ms": 0.2,
    }


def test_prefix_contract_bundle_classifies_complete_and_missing_costs():
    single_cases = [
        _staged_case(prefix_tokens, 1)
        for prefix_tokens in (256, 1024, 2048)
    ]
    batch_cases = [
        _staged_case(prefix_tokens, 8)
        for prefix_tokens in (1024, 2048)
    ]
    bundle = build_prefix_contract_bundle(
        single_cases,
        batch_cases,
        artifact_complete=True,
    )

    result = decide_staged_prefix_gate(bundle)

    assert result["classification"] == "PREFIX_CACHE_GO"
    assert bundle["single"]["1024"]["warm"][
        "median_cached_prompt_tokens"
    ] == 1024
    assert bundle["batch"]["2048"]["warm"][
        "median_cached_prompt_tokens"
    ] == 16384

    incorrect_bundle = build_prefix_contract_bundle(
        single_cases,
        batch_cases,
        artifact_complete=True,
        correctness_failures=[
            "repeat_513: targeted correctness check failed"
        ],
    )
    incorrect = decide_staged_prefix_gate(incorrect_bundle)
    assert incorrect["classification"] == (
        "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    )
    assert incorrect["structural_failures"] == []
    assert incorrect["correctness_failures"] == [
        "repeat_513: targeted correctness check failed"
    ]

    del single_cases[1]["warm"]["peak_cuda_reserved_bytes"]
    try:
        build_prefix_contract_bundle(
            single_cases,
            batch_cases,
            artifact_complete=True,
        )
    except ValueError as error:
        assert "peak_cuda_reserved_bytes" in str(error)
    else:
        raise AssertionError("missing Prefix cost evidence must fail closed")


def test_prefix_memory_rows_preserve_frozen_identity_fields():
    performance_row = {
        "schema_version": 2,
        "case_id": "single-256__cold__r0",
        "shape": "single-256",
        "state": "cold",
        "repetition": 0,
        "warmup": False,
        "cuda_allocated_bytes": 100,
        "cuda_reserved_bytes": 200,
        "cuda_peak_allocated_bytes": 300,
        "cuda_peak_reserved_bytes": 400,
        "retained_logical_kv_bytes": 500,
    }

    assert build_prefix_memory_rows([performance_row]) == [
        performance_row
    ]


def test_summarize_batch_result_compares_each_request_to_reference():
    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def __sub__(self, other):
            return FakeTensor(
                left - right
                for left, right in zip(self.values, other.values)
            )

        def abs(self):
            return FakeTensor(abs(value) for value in self.values)

        def max(self):
            return max(self.values)

        def mean(self):
            return sum(self.values) / len(self.values)

        def argmax(self):
            return max(
                range(len(self.values)),
                key=self.values.__getitem__,
            )

    reference = {
        "metadata": [
            {"cached_tokens": 0, "query_tokens": 1088},
            {"cached_tokens": 0, "query_tokens": 1088},
        ],
        "token_ids": [7, 8],
        "decoded": ["a", "b"],
        "logits": [
            FakeTensor([1.0, 3.0, 2.0]),
            FakeTensor([1.0, 2.0, 3.0]),
        ],
        "model_batches": 2,
        "ttft_ms": 20.0,
        "raw_ttft_ms": 21.0,
        "capture_overhead_ms": 1.0,
    }
    warm = {
        "metadata": [
            {"cached_tokens": 1024, "query_tokens": 64},
            {"cached_tokens": 1024, "query_tokens": 64},
        ],
        "token_ids": [7, 8],
        "decoded": ["a", "b"],
        "logits": [
            FakeTensor([1.0, 3.0, 2.0]),
            FakeTensor([1.0, 2.0, 3.0]),
        ],
        "model_batches": 1,
        "ttft_ms": 12.0,
        "raw_ttft_ms": 13.0,
        "capture_overhead_ms": 1.0,
    }

    row = summarize_batch_result(
        "warm",
        warm,
        reference,
        repetition=0,
    )

    assert row["requests"] == 2
    assert row["model_batches"] == 1
    assert row["total_cached_tokens"] == 2048
    assert row["total_query_tokens"] == 128
    assert row["correct"] is True


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


def test_decide_gate_requires_warm_batch_admission_and_speedup():
    correctness = [{"case": "boundary", "correct": True}]
    performance = [
        _perf_case(256, 10.0, 9.8),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    batch_performance = [
        _batch_perf_case(1024, 40.0, 28.0),
        _batch_perf_case(2048, 80.0, 48.0),
    ]

    decision = decide_gate(
        correctness,
        performance,
        batch_performance,
    )
    assert decision["decision"] == "GO"

    batch_performance[0]["warm"]["median_model_batches"] = 2
    decision = decide_gate(
        correctness,
        performance,
        batch_performance,
    )
    assert decision["decision"] == "NO_GO"
    assert "single model batch" in " ".join(decision["reasons"])


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


def test_audit_batch_artifact_payloads_recomputes_raw_rows():
    rows = []
    cases = []
    for prefix, cold_ms, warm_ms in (
        (1024, 40.0, 28.0),
        (2048, 80.0, 48.0),
    ):
        case_rows = [
            {
                "state": state,
                "batch_elapsed_ms": ttft_ms,
                "model_batches": model_batches,
                "total_query_tokens": query_tokens,
                "total_cached_tokens": cached_tokens,
                "requests": 8,
                "correct": True,
                "cached_tokens_per_request": (
                    [prefix] * 8 if state == "warm" else [0] * 8
                ),
                "query_tokens_per_request": (
                    [64] * 8
                    if state == "warm"
                    else [prefix + 64] * 8
                ),
                "shared_prefix_tokens": prefix,
                "suffix_tokens": 64,
                "batch_size": 8,
                "cache_isolation_between_batches": state != "warm",
            }
            for state, ttft_ms, model_batches, query_tokens, cached_tokens in (
                ("cold", cold_ms, 2, 8 * (prefix + 64), 0),
                ("warm", warm_ms, 1, 8 * 64, 8 * prefix),
                ("cache_cleared", cold_ms, 2, 8 * (prefix + 64), 0),
            )
        ]
        rows.extend(case_rows)
        summaries = {
            state: summarize_batch_case_rows(
                [row for row in case_rows if row["state"] == state]
            )
            for state in ("cold", "warm", "cache_cleared")
        }
        cases.append(
            {
                "shared_prefix_tokens": prefix,
                "suffix_tokens": 64,
                "batch_size": 8,
                "expected_reusable_tokens_per_request": prefix,
                "cold": summaries["cold"],
                "warm": summaries["warm"],
                "cache_cleared": summaries["cache_cleared"],
                "all_correct": True,
            }
        )
    summary = {
        "batch_performance_cases": cases,
        "decision": decide_gate(
            [{"case": "boundary", "correct": True}],
            [
                _perf_case(256, 10.0, 9.8),
                _perf_case(1024, 20.0, 15.0),
                _perf_case(2048, 40.0, 28.0),
            ],
            cases,
        ),
    }

    assert audit_batch_artifact_payloads(
        rows,
        summary,
        repetitions=1,
        correctness_rows=[{"case": "boundary", "correct": True}],
        performance_cases=[
            _perf_case(256, 10.0, 9.8),
            _perf_case(1024, 20.0, 15.0),
            _perf_case(2048, 40.0, 28.0),
        ],
    ) == []

    summary["batch_performance_cases"][0]["warm"][
        "median_model_batches"
    ] = 2
    errors = audit_batch_artifact_payloads(
        rows,
        summary,
        repetitions=1,
        correctness_rows=[{"case": "boundary", "correct": True}],
        performance_cases=[
            _perf_case(256, 10.0, 9.8),
            _perf_case(1024, 20.0, 15.0),
            _perf_case(2048, 40.0, 28.0),
        ],
    )
    assert any("raw rows" in error for error in errors)

    summary["batch_performance_cases"] = cases
    summary["decision"] = decide_gate(
        [{"case": "boundary", "correct": True}],
        [
            _perf_case(256, 10.0, 9.8),
            _perf_case(1024, 20.0, 15.0),
            _perf_case(2048, 40.0, 28.0),
        ],
        cases,
    )
    cold_row = next(
        row
        for row in rows
        if row["shared_prefix_tokens"] == 1024
        and row["state"] == "cold"
    )
    cold_row["cache_isolation_between_batches"] = False
    errors = audit_batch_artifact_payloads(
        rows,
        summary,
        repetitions=1,
        correctness_rows=[{"case": "boundary", "correct": True}],
        performance_cases=[
            _perf_case(256, 10.0, 9.8),
            _perf_case(1024, 20.0, 15.0),
            _perf_case(2048, 40.0, 28.0),
        ],
    )
    assert any("raw rows" in error for error in errors)
    cold_row["cache_isolation_between_batches"] = True

    warm_row = next(
        row
        for row in rows
        if row["shared_prefix_tokens"] == 1024
        and row["state"] == "warm"
    )
    warm_row["cache_isolation_between_batches"] = True
    errors = audit_batch_artifact_payloads(
        rows,
        summary,
        repetitions=1,
        correctness_rows=[{"case": "boundary", "correct": True}],
        performance_cases=[
            _perf_case(256, 10.0, 9.8),
            _perf_case(1024, 20.0, 15.0),
            _perf_case(2048, 40.0, 28.0),
        ],
    )
    assert any("raw rows" in error for error in errors)
    warm_row["cache_isolation_between_batches"] = False

    warm_row["cached_tokens_per_request"][0] = 0
    errors = audit_batch_artifact_payloads(
        rows,
        summary,
        repetitions=1,
        correctness_rows=[{"case": "boundary", "correct": True}],
        performance_cases=[
            _perf_case(256, 10.0, 9.8),
            _perf_case(1024, 20.0, 15.0),
            _perf_case(2048, 40.0, 28.0),
        ],
    )
    assert any("raw rows" in error for error in errors)


def test_render_report_includes_batch_admission_table():
    performance_case = _perf_case(1024, 20.0, 15.0)
    performance_case["cache_cleared"] = {"median_ttft_ms": 20.0}
    performance_case["warm_ttft_improvement_fraction"] = 0.25
    batch_case = _batch_perf_case(1024, 40.0, 28.0)
    batch_case["warm_ttft_improvement_fraction"] = 0.30
    report = render_report(
        {"source_sha256": {"source.py": "a" * 64}},
        [{"case": "boundary", "correct": True}],
        [performance_case],
        [batch_case],
        {"decision": "GO", "reasons": []},
    )

    assert "## Warm Batch Admission" in report
    assert "Cold Batches" in report
    assert "| 1024 | 8 | 2 | 1 |" in report


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
    test_schedule_and_run_prefill_batches_drains_all_requests()
    test_summarize_case_rows_reports_medians_and_correctness()
    test_summarize_batch_case_rows_reports_admission_and_accounting()
    test_summarize_case_rows_reports_p95_and_prefix_costs()
    test_summarize_batch_rows_reports_p95_and_prefix_costs()
    test_prefix_cost_observation_uses_all_hashed_blocks()
    test_prefix_cost_observation_supports_direct_model_runner_path()
    test_clear_reusable_cache_observation_records_host_cost()
    test_append_jsonl_is_canonical_append_only_and_newline_terminated()
    test_prefix_source_files_bind_staged_classifier_source()
    test_prefix_case_shape_uses_actual_batch_size()
    test_prefix_contract_bundle_classifies_complete_and_missing_costs()
    test_prefix_memory_rows_preserve_frozen_identity_fields()
    test_summarize_batch_result_compares_each_request_to_reference()
    test_decide_gate_requires_correctness_and_two_large_prefix_wins()
    test_decide_gate_rejects_any_correctness_failure_or_warm_regression()
    test_decide_gate_rejects_cached_or_query_token_mismatch()
    test_decide_gate_requires_warm_batch_admission_and_speedup()
    test_audit_artifact_payloads_recomputes_raw_performance_rows()
    test_audit_batch_artifact_payloads_recomputes_raw_rows()
    test_render_report_includes_batch_admission_table()
    print("prefix cache profiler tests passed")


if __name__ == "__main__":
    main()
