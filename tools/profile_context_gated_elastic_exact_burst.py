#!/usr/bin/env python3
"""Source-bound K8 versus context-gated elastic K16 profiler."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import time

from tools import profile_exact_greedy_decode_burst as _base
from tools.profile_exact_burst_medium_split_k import (
    _flatten_logits,
    _run_request,
)
from tools.profile_zero_temperature_greedy_fast_path import (
    _aggregate_memory,
    _make_prompt,
    _write_json,
    append_jsonl,
    read_float32_sidecar,
    sha256_file,
    sha256_text,
    write_float32_sidecar,
)


CASE_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.case.v1"
)
CORRECTNESS_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.workload.v1"
)
SOURCE_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.source.v1"
)

POLICIES = (
    "fixed_k8",
    "context_gated_elastic_k16",
)
POLICY_CONFIGS = {
    "fixed_k8": {
        "exact_greedy_decode_burst_elastic_k16": False,
    },
    "context_gated_elastic_k16": {
        "exact_greedy_decode_burst_elastic_k16": True,
    },
}
CONTEXT_LENGTHS = (256, 2048, 4096, 8192)
GENERATED_TOKENS = 128
REPETITIONS = 5
WARMUP_REPETITIONS = 2
SAMPLING_POINTS = _base.SAMPLING_POINTS
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_context_gated_elastic_exact_burst.py",
    "tools/test_profile_context_gated_elastic_exact_burst.py",
)

_COUNTER_FIELDS = (
    "attempts",
    "acceptances",
    "target_model_forwards",
    "graph_replays",
    "intermediate_token_d2h_calls",
    "final_token_d2h_calls",
    "final_token_d2h_bytes",
    "sampled_logit_d2h_calls",
    "output_budget_clipped",
    "block_boundary_clipped",
    "commits",
    "committed_tokens",
    "failures",
    "quarantines",
    "pending_leases",
    "k16_attempts",
    "k16_acceptances",
    "k8_fallbacks",
    "lease_local_delta_journal_attempts",
    "lease_local_delta_journal_captures",
    "lease_local_delta_journal_commits",
    "lease_local_delta_journal_rollbacks",
    "lease_local_delta_journal_one_phase_attempts",
    "lease_local_delta_journal_one_phase_captures",
    "lease_local_delta_journal_one_phase_commits",
    "lease_local_delta_journal_one_phase_rollbacks",
)
_MAP_FIELDS = (
    "requested_width_histogram",
    "authorized_width_histogram",
    "fallback_counts",
    "elastic_k16_fallback_counts",
    "per_width_commits",
    "lease_local_delta_journal_fallback_counts",
    "lease_local_delta_journal_one_phase_fallback_counts",
)
_CASE_REQUIRED_FIELDS = {
    "schema_version",
    "run_tag",
    "source_commit",
    "policy",
    "repetition",
    "order_position",
    "context_length",
    "prompt_tokens",
    "generated_tokens",
    "temperature",
    "ignore_eos",
    "tensor_parallel_size",
    "max_num_seqs",
    "completion_only",
    "prompt_sha256",
    "output_token_ids",
    "output_text_sha256",
    "ttft_ns",
    "e2e_ns",
    "amortized_tpot_samples_ns",
    "amortized_tpot_median_ns",
    "amortized_tpot_p95_ns",
    "amortized_tpot_p99_ns",
    "decode_host_ns",
    "decode_cuda_ns",
    "output_tokens_per_second",
    "host_visible_burst_gaps_ns",
    "maximum_host_visible_burst_gap_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "shared_capture_duration_ns",
    "shared_capture_allocated_delta_bytes",
    "shared_capture_reserved_delta_bytes",
    "shared_capture_retained_static_bytes",
    "elastic_incremental_allocated_bytes",
    "elastic_incremental_reserved_bytes",
    "elastic_incremental_retained_static_bytes",
    "correctness_trace",
    "exact_greedy_decode_burst_summary",
}


def _require_non_negative_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_finite_non_negative(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _validate_digest(
    value,
    name: str,
    *,
    lengths: tuple[int, ...] = (64,),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("nearest-rank input cannot be empty")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def policy_order(
    repetition: int,
    context_index: int,
) -> tuple[str, str]:
    _require_non_negative_int(repetition, "repetition")
    _require_non_negative_int(context_index, "context_index")
    if (repetition + context_index) % 2:
        return tuple(reversed(POLICIES))
    return POLICIES


def performance_identities(
    *,
    repetitions: int,
) -> tuple[tuple[int, int, str], ...]:
    _require_non_negative_int(repetitions, "repetitions")
    return tuple(
        (repetition, context_length, policy)
        for repetition in range(repetitions)
        for context_index, context_length in enumerate(CONTEXT_LENGTHS)
        for policy in policy_order(repetition, context_index)
    )


def correctness_identities() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (context_length, policy, point)
        for context_length in CONTEXT_LENGTHS
        for policy in POLICIES
        for point in SAMPLING_POINTS
    )


def runtime_environment_manifest() -> dict:
    return _base.runtime_environment_manifest()


def build_workload_manifest(
    *,
    model: str,
    device: str,
    run_tag: str,
    source_commit: str,
    gpu_memory_utilization: float,
    environment: dict,
    repetitions: int = REPETITIONS,
    warmup_repetitions: int = WARMUP_REPETITIONS,
) -> dict:
    if not isinstance(model, str) or not model:
        raise ValueError("model is invalid")
    if not isinstance(device, str) or not device:
        raise ValueError("device is invalid")
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run tag is invalid")
    _validate_digest(
        source_commit,
        "source commit",
        lengths=(40, 64),
    )
    if (
        isinstance(gpu_memory_utilization, bool)
        or not isinstance(gpu_memory_utilization, (int, float))
        or not 0.0 < float(gpu_memory_utilization) <= 1.0
    ):
        raise ValueError("gpu memory utilization must be in (0, 1]")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("environment manifest is invalid")
    if repetitions not in (3, REPETITIONS):
        raise ValueError("repetitions must equal 3 or 5")
    if warmup_repetitions not in (1, WARMUP_REPETITIONS):
        raise ValueError("warmup repetitions must equal 1 or 2")
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "model": os.fspath(Path(model).resolve()),
        "device": device,
        "contexts": list(CONTEXT_LENGTHS),
        "policies": list(POLICIES),
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "generated_tokens": GENERATED_TOKENS,
        "performance_row_count": (
            repetitions * len(CONTEXT_LENGTHS) * len(POLICIES)
        ),
        "correctness_row_count": (
            len(CONTEXT_LENGTHS)
            * len(POLICIES)
            * len(SAMPLING_POINTS)
        ),
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "completion_only": True,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "environment": dict(environment),
    }


def _validate_counter_map(value, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    normalized = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} key is invalid")
        normalized[key] = _require_non_negative_int(
            count,
            f"{name}[{key}]",
        )
    return normalized


def _validate_summary(
    value,
    *,
    policy: str,
    context_length: int,
) -> dict:
    required = set(_COUNTER_FIELDS) | set(_MAP_FIELDS) | {
        "maximum_host_visible_gap_ns",
        "quarantine_reason",
        "capture_receipts",
    }
    if not isinstance(value, dict) or required - set(value):
        raise ValueError("burst summary fields are missing")
    normalized = dict(value)
    for field in _COUNTER_FIELDS:
        normalized[field] = _require_non_negative_int(
            value[field],
            field,
        )
    normalized["maximum_host_visible_gap_ns"] = (
        _require_non_negative_int(
            value["maximum_host_visible_gap_ns"],
            "maximum_host_visible_gap_ns",
        )
    )
    for field in _MAP_FIELDS:
        normalized[field] = _validate_counter_map(
            value[field],
            field,
        )
    if value["quarantine_reason"] is not None:
        raise ValueError("profile row cannot be quarantined")
    receipts = value["capture_receipts"]
    if not isinstance(receipts, list) or len(receipts) != 1:
        raise ValueError("shared capture receipt inventory mismatch")
    receipt = receipts[0]
    if not isinstance(receipt, dict):
        raise ValueError("shared capture receipt is invalid")
    _validate_digest(
        receipt.get("graph_identity_sha256"),
        "shared graph identity",
    )
    for field in (
        "capture_duration_ns",
        "allocated_delta_bytes",
        "reserved_delta_bytes",
        "retained_static_bytes",
        "scratch_block_count",
    ):
        _require_non_negative_int(receipt.get(field), field)
    if normalized["target_model_forwards"] != normalized["graph_replays"]:
        raise ValueError("target forwards and graph replays differ")
    if normalized["graph_replays"] != normalized["committed_tokens"]:
        raise ValueError("graph replay and committed-token inventory differ")
    if normalized["intermediate_token_d2h_calls"] != 0:
        raise ValueError("intermediate token D2H must remain zero")
    if (
        normalized["final_token_d2h_calls"]
        != normalized["commits"]
    ):
        raise ValueError("final token D2H call inventory mismatch")
    if (
        normalized["final_token_d2h_bytes"]
        != normalized["committed_tokens"] * 8
    ):
        raise ValueError("final token D2H byte inventory mismatch")
    elastic = policy == "context_gated_elastic_k16"
    eligible = elastic and context_length <= 2048
    selected_k16 = (
        normalized["k16_acceptances"] > 0
        or normalized["authorized_width_histogram"].get("16", 0) > 0
        or normalized["per_width_commits"].get("16", 0) > 0
    )
    if eligible and not selected_k16:
        raise ValueError("eligible row did not record K16 selection")
    if not eligible and selected_k16:
        raise ValueError("K16 selection is forbidden for this row")
    if not elastic and (
        normalized["k16_attempts"]
        or normalized["k16_acceptances"]
        or normalized["k8_fallbacks"]
        or normalized["elastic_k16_fallback_counts"]
    ):
        raise ValueError("fixed K8 row reported elastic counters")
    return normalized


def validate_case_row(row) -> dict:
    if not isinstance(row, dict) or _CASE_REQUIRED_FIELDS - set(row):
        raise ValueError("case row fields are missing")
    if row["schema_version"] != CASE_SCHEMA_VERSION:
        raise ValueError("case row schema mismatch")
    if not isinstance(row["run_tag"], str) or not row["run_tag"]:
        raise ValueError("run tag is invalid")
    _validate_digest(
        row["source_commit"],
        "source commit",
        lengths=(40, 64),
    )
    policy = row["policy"]
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
    context_length = row["context_length"]
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError("context length is invalid")
    repetition = _require_non_negative_int(
        row["repetition"],
        "repetition",
    )
    order_position = _require_non_negative_int(
        row["order_position"],
        "order position",
    )
    order = policy_order(
        repetition,
        CONTEXT_LENGTHS.index(context_length),
    )
    if order_position >= len(order) or order[order_position] != policy:
        raise ValueError("policy order position mismatch")
    if (
        row["prompt_tokens"] != context_length
        or row["generated_tokens"] != GENERATED_TOKENS
        or row["temperature"] != 0.0
        or row["ignore_eos"] is not True
        or row["tensor_parallel_size"] != 1
        or row["max_num_seqs"] != 1
        or row["completion_only"] is not True
    ):
        raise ValueError("case execution contract mismatch")
    _validate_digest(row["prompt_sha256"], "prompt digest")
    _validate_digest(row["output_text_sha256"], "output text digest")
    output_ids = row["output_token_ids"]
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != GENERATED_TOKENS
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in output_ids
        )
    ):
        raise ValueError("output token inventory mismatch")
    samples = row["amortized_tpot_samples_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != GENERATED_TOKENS - 1
    ):
        raise ValueError("raw TPOT inventory mismatch")
    normalized_samples = [
        _require_finite_non_negative(sample, "raw TPOT sample")
        for sample in samples
    ]
    expected_tpot = {
        "amortized_tpot_median_ns":
            statistics.median(normalized_samples),
        "amortized_tpot_p95_ns":
            _nearest_rank(normalized_samples, 0.95),
        "amortized_tpot_p99_ns":
            _nearest_rank(normalized_samples, 0.99),
    }
    for field, expected in expected_tpot.items():
        if _require_finite_non_negative(row[field], field) != expected:
            raise ValueError(f"TPOT summary mismatch: {field}")
    for field in (
        "ttft_ns",
        "e2e_ns",
        "output_tokens_per_second",
        "maximum_host_visible_burst_gap_ns",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "shared_capture_duration_ns",
        "shared_capture_allocated_delta_bytes",
        "shared_capture_reserved_delta_bytes",
        "shared_capture_retained_static_bytes",
        "elastic_incremental_allocated_bytes",
        "elastic_incremental_reserved_bytes",
        "elastic_incremental_retained_static_bytes",
    ):
        _require_finite_non_negative(row[field], field)
    for field in (
        "decode_host_ns",
        "decode_cuda_ns",
        "host_visible_burst_gaps_ns",
    ):
        values = row[field]
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        for value in values:
            _require_finite_non_negative(value, field)
    if row["correctness_trace"] is not False:
        raise ValueError("performance row cannot enable correctness tracing")
    summary = _validate_summary(
        row["exact_greedy_decode_burst_summary"],
        policy=policy,
        context_length=context_length,
    )
    if row["maximum_host_visible_burst_gap_ns"] != max(
        row["host_visible_burst_gaps_ns"],
        default=0,
    ):
        raise ValueError("maximum host-visible gap mismatch")
    receipt = summary["capture_receipts"][0]
    expected_capture = {
        "shared_capture_duration_ns":
            receipt["capture_duration_ns"],
        "shared_capture_allocated_delta_bytes":
            receipt["allocated_delta_bytes"],
        "shared_capture_reserved_delta_bytes":
            receipt["reserved_delta_bytes"],
        "shared_capture_retained_static_bytes":
            receipt["retained_static_bytes"],
    }
    for field, expected in expected_capture.items():
        if row[field] != expected:
            raise ValueError(f"shared capture cost mismatch: {field}")
    if (
        row["elastic_incremental_allocated_bytes"] != 0
        or row["elastic_incremental_reserved_bytes"] != 0
        or row["elastic_incremental_retained_static_bytes"] != 0
    ):
        raise ValueError("elastic incremental capture cost must be zero")
    normalized = dict(row)
    normalized["amortized_tpot_samples_ns"] = normalized_samples
    normalized["exact_greedy_decode_burst_summary"] = summary
    return normalized


def summarize_rows(
    rows: list[dict],
    *,
    expected_repetitions: int = REPETITIONS,
) -> dict:
    validated = [validate_case_row(row) for row in rows]
    identities = [
        (
            row["repetition"],
            row["context_length"],
            row["policy"],
        )
        for row in validated
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate case identity")
    if set(identities) != set(
        performance_identities(repetitions=expected_repetitions)
    ):
        raise ValueError("performance row inventory is incomplete")
    run_tags = {row["run_tag"] for row in validated}
    source_commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(source_commits) != 1:
        raise ValueError("performance rows do not share source identity")
    comparisons = {}
    all_outputs_exact = True
    for row in validated:
        key = (row["repetition"], row["context_length"])
        comparisons.setdefault(key, {})[row["policy"]] = row
    for arms in comparisons.values():
        control = arms["fixed_k8"]
        candidate = arms["context_gated_elastic_k16"]
        if (
            control["output_token_ids"]
            != candidate["output_token_ids"]
            or control["output_text_sha256"]
            != candidate["output_text_sha256"]
        ):
            all_outputs_exact = False
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(run_tags)),
        "source_commit": next(iter(source_commits)),
        "row_count": len(validated),
        "comparison_set_count": len(comparisons),
        "all_outputs_exact": all_outputs_exact,
    }


def validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
) -> list[dict]:
    required = {
        "schema_version",
        "run_tag",
        "source_commit",
        "policy",
        "context_length",
        "generated_tokens",
        "sampling_point",
        "prompt_sha256",
        "output_token_ids",
        "output_text_sha256",
        "argmax_token_id",
        "logits_path",
        "logits_shape",
        "logits_element_count",
        "logits_byte_length",
        "logits_sha256",
        "correctness_trace",
        "exact_greedy_decode_burst_summary",
    }
    validated = []
    identities = []
    for row in rows:
        if not isinstance(row, dict) or required - set(row):
            raise ValueError("correctness row fields are missing")
        if row["schema_version"] != CORRECTNESS_SCHEMA_VERSION:
            raise ValueError("correctness row schema mismatch")
        policy = row["policy"]
        context_length = row["context_length"]
        point = row["sampling_point"]
        if policy not in POLICIES:
            raise ValueError("correctness policy is invalid")
        if context_length not in CONTEXT_LENGTHS:
            raise ValueError("correctness context is invalid")
        if point not in SAMPLING_POINTS:
            raise ValueError("correctness sampling point is invalid")
        identities.append((context_length, policy, point))
        _validate_digest(
            row["source_commit"],
            "correctness source commit",
            lengths=(40, 64),
        )
        _validate_digest(row["prompt_sha256"], "prompt digest")
        _validate_digest(
            row["output_text_sha256"],
            "output text digest",
        )
        output_ids = row["output_token_ids"]
        if (
            row["generated_tokens"] != GENERATED_TOKENS
            or not isinstance(output_ids, list)
            or len(output_ids) != GENERATED_TOKENS
        ):
            raise ValueError("correctness output inventory mismatch")
        values = read_float32_sidecar(
            run_dir,
            path=row["logits_path"],
            expected_element_count=row["logits_element_count"],
            expected_byte_length=row["logits_byte_length"],
            expected_sha256=row["logits_sha256"],
        )
        shape = row["logits_shape"]
        if (
            not isinstance(shape, list)
            or math.prod(shape) != len(values)
        ):
            raise ValueError("correctness logits shape mismatch")
        expected_argmax = max(
            range(len(values)),
            key=values.__getitem__,
        )
        if row["argmax_token_id"] != expected_argmax:
            raise ValueError("correctness argmax mismatch")
        if row["correctness_trace"] is not True:
            raise ValueError("correctness trace must be enabled")
        summary = _validate_summary(
            row["exact_greedy_decode_burst_summary"],
            policy=policy,
            context_length=context_length,
        )
        normalized = dict(row)
        normalized["exact_greedy_decode_burst_summary"] = summary
        validated.append(normalized)
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate correctness identity")
    if set(identities) != set(correctness_identities()):
        raise ValueError("correctness row inventory is incomplete")
    if (
        len({row["run_tag"] for row in validated}) != 1
        or len({row["source_commit"] for row in validated}) != 1
    ):
        raise ValueError(
            "correctness rows do not share source identity"
        )
    for context_length in CONTEXT_LENGTHS:
        by_policy = {
            policy: [
                row
                for row in validated
                if row["context_length"] == context_length
                and row["policy"] == policy
            ]
            for policy in POLICIES
        }
        for point in SAMPLING_POINTS:
            control = next(
                row
                for row in by_policy["fixed_k8"]
                if row["sampling_point"] == point
            )
            candidate = next(
                row
                for row in by_policy[
                    "context_gated_elastic_k16"
                ]
                if row["sampling_point"] == point
            )
            if (
                control["output_token_ids"]
                != candidate["output_token_ids"]
                or control["output_text_sha256"]
                != candidate["output_text_sha256"]
                or control["argmax_token_id"]
                != candidate["argmax_token_id"]
                or control["logits_sha256"]
                != candidate["logits_sha256"]
            ):
                raise ValueError("correctness policy mismatch")
    return validated


def _construct_llm(
    *,
    model: str,
    device: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
    policy: str,
):
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("device must identify a CUDA device")
    from tinyvllm import LLM

    return LLM(
        model,
        max_num_batched_tokens=prompt_tokens + generated_tokens,
        max_num_seqs=1,
        max_model_len=prompt_tokens + generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False,
        zero_temperature_greedy_fast_path=True,
        graph_resident_greedy_tail=False,
        exact_greedy_decode_burst=True,
        exact_greedy_decode_burst_continuation=False,
        exact_greedy_decode_burst_tokens=8,
        exact_greedy_decode_burst_split_phase=False,
        exact_greedy_decode_burst_ragged_coalescing=False,
        exact_greedy_decode_burst_lease_local_delta_journal=True,
        exact_greedy_decode_burst_generation_sealed_identity=True,
        exact_greedy_decode_burst_elastic_k16=(
            POLICY_CONFIGS[policy][
                "exact_greedy_decode_burst_elastic_k16"
            ]
        ),
    )


def _counter_delta(before: dict, after: dict) -> dict:
    result = {}
    for field in _COUNTER_FIELDS:
        difference = int(after.get(field, 0)) - int(
            before.get(field, 0)
        )
        if difference < 0:
            raise RuntimeError(f"counter decreased: {field}")
        result[field] = difference
    for field in _MAP_FIELDS:
        before_map = before.get(field, {})
        after_map = after.get(field, {})
        result[field] = {}
        for key in sorted(set(before_map) | set(after_map), key=str):
            difference = int(after_map.get(key, 0)) - int(
                before_map.get(key, 0)
            )
            if difference < 0:
                raise RuntimeError(f"map counter decreased: {field}")
            if difference:
                result[field][str(key)] = difference
    result["quarantine_reason"] = after.get("quarantine_reason")
    result["capture_receipts"] = list(
        after.get("capture_receipts", ())
    )
    result["maximum_host_visible_gap_ns"] = int(
        after.get("maximum_host_visible_gap_ns", 0)
    )
    return result


def _runner_summaries(llm) -> tuple[dict, dict]:
    return (
        llm.model_runner.exact_greedy_decode_burst_summary(),
        llm.scheduler.exact_greedy_decode_burst_summary(),
    )


def _combined_summary(
    llm,
    before: tuple[dict, dict],
    *,
    correctness_trace: bool = False,
) -> dict:
    runner = _counter_delta(
        before[0],
        llm.model_runner.exact_greedy_decode_burst_summary(),
    )
    scheduler = _counter_delta(
        before[1],
        llm.scheduler.exact_greedy_decode_burst_summary(),
    )
    result = {
        field: (
            runner[field]
            if field
            in {
                "target_model_forwards",
                "graph_replays",
                "intermediate_token_d2h_calls",
                "final_token_d2h_calls",
                "final_token_d2h_bytes",
                "sampled_logit_d2h_calls",
                "failures",
                "quarantines",
            }
            else scheduler[field]
        )
        for field in _COUNTER_FIELDS
    }
    for field in _MAP_FIELDS:
        result[field] = scheduler[field]
    result["maximum_host_visible_gap_ns"] = scheduler[
        "maximum_host_visible_gap_ns"
    ]
    result["quarantine_reason"] = runner["quarantine_reason"]
    result["capture_receipts"] = [
        receipt
        for receipt in runner["capture_receipts"]
        if bool(receipt.get("correctness_trace"))
        is correctness_trace
    ]
    return result


def _capture_cost(summary: dict) -> dict:
    receipt = summary["capture_receipts"][0]
    return {
        "shared_capture_duration_ns": receipt["capture_duration_ns"],
        "shared_capture_allocated_delta_bytes":
            receipt["allocated_delta_bytes"],
        "shared_capture_reserved_delta_bytes":
            receipt["reserved_delta_bytes"],
        "shared_capture_retained_static_bytes":
            receipt["retained_static_bytes"],
        "elastic_incremental_allocated_bytes": 0,
        "elastic_incremental_reserved_bytes": 0,
        "elastic_incremental_retained_static_bytes": 0,
    }


def run_case(
    *,
    model: str,
    device: str,
    run_tag: str,
    source_commit: str,
    policy: str,
    repetition: int,
    order_position: int,
    context_length: int,
    warmup_repetitions: int,
    gpu_memory_utilization: float,
) -> dict:
    llm = _construct_llm(
        model=model,
        device=device,
        prompt_tokens=context_length,
        generated_tokens=GENERATED_TOKENS,
        gpu_memory_utilization=gpu_memory_utilization,
        policy=policy,
    )
    try:
        for warmup_index in range(warmup_repetitions):
            _run_request(
                llm,
                prompt=_make_prompt(
                    context_length,
                    offset=50_000 + warmup_index * 2_003,
                ),
                generated_tokens=GENERATED_TOKENS,
                profile_label=None,
                require_graph_identity=False,
            )
            llm.clear_reusable_prefix_cache()
        before = _runner_summaries(llm)
        llm.reset_peak_memory_stats(timeout_s=60.0)
        prompt = _make_prompt(
            context_length,
            offset=repetition * 10_007,
        )
        measured = _run_request(
            llm,
            prompt=prompt,
            generated_tokens=GENERATED_TOKENS,
            profile_label=(
                f"{run_tag}/{context_length}/r{repetition}/{policy}"
            ),
            require_graph_identity=True,
        )
        memory = _aggregate_memory(
            llm.memory_snapshots(timeout_s=60.0)
        )
        summary = _combined_summary(llm, before)
        summary["maximum_host_visible_gap_ns"] = max(
            measured["host_visible_burst_gaps_ns"],
            default=0,
        )
        samples = measured["amortized_tpot_samples_ns"]
        e2e_seconds = measured["e2e_ns"] / 1_000_000_000
        row = {
            "schema_version": CASE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_commit": source_commit,
            "policy": policy,
            "repetition": repetition,
            "order_position": order_position,
            "context_length": context_length,
            "prompt_tokens": context_length,
            "generated_tokens": GENERATED_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "completion_only": True,
            "prompt_sha256": sha256_text(
                ",".join(str(token) for token in prompt)
            ),
            "output_token_ids": measured["output_token_ids"],
            "output_text_sha256": sha256_text(
                measured["output_text"]
            ),
            "ttft_ns": measured["ttft_ns"],
            "e2e_ns": measured["e2e_ns"],
            "amortized_tpot_samples_ns": samples,
            "amortized_tpot_median_ns": statistics.median(samples),
            "amortized_tpot_p95_ns": _nearest_rank(samples, 0.95),
            "amortized_tpot_p99_ns": _nearest_rank(samples, 0.99),
            "decode_host_ns": measured["decode_host_ns"],
            "decode_cuda_ns": measured["decode_cuda_ns"],
            "output_tokens_per_second":
                GENERATED_TOKENS / e2e_seconds,
            "host_visible_burst_gaps_ns":
                measured["host_visible_burst_gaps_ns"],
            "maximum_host_visible_burst_gap_ns": max(
                measured["host_visible_burst_gaps_ns"],
                default=0,
            ),
            **memory,
            **_capture_cost(summary),
            "correctness_trace": False,
            "exact_greedy_decode_burst_summary": summary,
        }
        return validate_case_row(row)
    finally:
        llm.exit()


def _sampling_output_index(point: str) -> int:
    try:
        return {
            "prefill-final": 0,
            "decode-first": 1,
            "decode-middle": GENERATED_TOKENS // 2,
            "decode-final": GENERATED_TOKENS - 1,
        }[point]
    except KeyError as error:
        raise ValueError("sampling point is invalid") from error


def run_correctness_probe(
    *,
    model: str,
    device: str,
    run_dir: Path,
    run_tag: str,
    source_commit: str,
    policy: str,
    context_length: int,
    gpu_memory_utilization: float,
) -> list[dict]:
    from tinyvllm import SamplingParams

    llm = _construct_llm(
        model=model,
        device=device,
        prompt_tokens=context_length,
        generated_tokens=GENERATED_TOKENS,
        gpu_memory_utilization=gpu_memory_utilization,
        policy=policy,
    )
    try:
        llm.model_runner.call(
            "capture_exact_greedy_decode_burst_correctness_graph",
            (0, 6, 7, 15),
        )
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        before = _runner_summaries(llm)
        prompt = _make_prompt(context_length, offset=90_001)
        llm.add_request(
            prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=GENERATED_TOKENS,
                ignore_eos=True,
            ),
        )
        captured = {}
        final_outputs = None
        emitted_total = 0
        while not llm.is_finished():
            trace_this_step = emitted_total > 0
            outputs, _num_tokens = llm.step(
                completion_only=True,
                exact_burst_correctness_trace=trace_this_step,
            )
            observation = llm.last_step_observation
            emitted = sum(
                len(tokens)
                for tokens in observation[
                    "new_completion_tokens_by_seq"
                ].values()
            )
            if emitted_total == 0:
                logits = (
                    llm.read_step_logits_authority()
                    .detach()
                    .to(dtype=__import__("torch").float32)
                    .contiguous()
                )
                captured["prefill-final"] = logits
                llm.enable_step_logits_authority_recording(
                    False,
                    timeout_s=60.0,
                )
            elif trace_this_step:
                sampled = observation.get(
                    "exact_greedy_decode_burst_sampled_logits",
                    (),
                )
                for local_ordinal, values in sampled:
                    output_index = emitted_total + int(local_ordinal)
                    for point in SAMPLING_POINTS[1:]:
                        if output_index == _sampling_output_index(point):
                            captured[point] = values
            emitted_total += emitted
            if outputs:
                final_outputs = outputs
        if set(captured) != set(SAMPLING_POINTS):
            raise RuntimeError("correctness sampling points are incomplete")
        if not isinstance(final_outputs, list) or len(final_outputs) != 1:
            raise RuntimeError("correctness output is incomplete")
        output_ids = list(final_outputs[0][1])
        if len(output_ids) != GENERATED_TOKENS:
            raise RuntimeError(
                "correctness output token inventory mismatch"
            )
        output_text_sha256 = sha256_text(
            llm.tokenizer.decode(output_ids)
        )
        summary = _combined_summary(
            llm,
            before,
            correctness_trace=True,
        )
        prompt_sha256 = sha256_text(
            ",".join(str(token) for token in prompt)
        )
        rows = []
        for point in SAMPLING_POINTS:
            shape, values = _flatten_logits(captured[point])
            sidecar = write_float32_sidecar(
                run_dir,
                f"logits/{context_length}-{policy}-{point}.f32",
                values,
            )
            rows.append({
                "schema_version": CORRECTNESS_SCHEMA_VERSION,
                "run_tag": run_tag,
                "source_commit": source_commit,
                "policy": policy,
                "context_length": context_length,
                "generated_tokens": GENERATED_TOKENS,
                "sampling_point": point,
                "prompt_sha256": prompt_sha256,
                "output_token_ids": output_ids,
                "output_text_sha256": output_text_sha256,
                "argmax_token_id": max(
                    range(len(values)),
                    key=values.__getitem__,
                ),
                "logits_path": sidecar["path"],
                "logits_shape": shape,
                "logits_element_count": sidecar["element_count"],
                "logits_byte_length": sidecar["byte_length"],
                "logits_sha256": sidecar["sha256"],
                "correctness_trace": True,
                "exact_greedy_decode_burst_summary": summary,
            })
        return rows
    finally:
        try:
            llm.enable_step_logits_authority_recording(
                False,
                timeout_s=60.0,
            )
        finally:
            llm.exit()


def _source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    run_tag: str,
) -> dict:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in SOURCE_FILES
        },
    }


def create_output_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=WARMUP_REPETITIONS,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    workload = build_workload_manifest(
        model=args.model,
        device=args.device,
        run_tag=args.run_tag,
        source_commit=args.source_commit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        environment=runtime_environment_manifest(),
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup_repetitions,
    )
    output_dir = create_output_directory(Path(args.output_dir))
    repo_root = Path(__file__).resolve().parents[1]
    _write_json(
        output_dir / "source_manifest.json",
        _source_manifest(
            repo_root=repo_root,
            source_commit=args.source_commit,
            run_tag=args.run_tag,
        ),
    )
    _write_json(output_dir / "workload_manifest.json", workload)
    rows = []
    row_path = output_dir / "performance_rows.jsonl"
    for repetition in range(args.repetitions):
        for context_index, context_length in enumerate(CONTEXT_LENGTHS):
            for order_position, policy in enumerate(
                policy_order(repetition, context_index)
            ):
                row = run_case(
                    model=args.model,
                    device=args.device,
                    run_tag=args.run_tag,
                    source_commit=args.source_commit,
                    policy=policy,
                    repetition=repetition,
                    order_position=order_position,
                    context_length=context_length,
                    warmup_repetitions=args.warmup_repetitions,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                append_jsonl(row_path, row)
                rows.append(row)
    correctness_rows = []
    correctness_path = output_dir / "correctness_rows.jsonl"
    for context_length in CONTEXT_LENGTHS:
        for policy in POLICIES:
            for row in run_correctness_probe(
                model=args.model,
                device=args.device,
                run_dir=output_dir,
                run_tag=args.run_tag,
                source_commit=args.source_commit,
                policy=policy,
                context_length=context_length,
                gpu_memory_utilization=args.gpu_memory_utilization,
            ):
                append_jsonl(correctness_path, row)
                correctness_rows.append(row)
    validate_correctness_rows(
        correctness_rows,
        run_dir=output_dir,
    )
    summary = summarize_rows(
        rows,
        expected_repetitions=args.repetitions,
    )
    summary["correctness_row_count"] = len(correctness_rows)
    _write_json(output_dir / "profile_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
