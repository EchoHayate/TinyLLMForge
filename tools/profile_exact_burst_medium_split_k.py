#!/usr/bin/env python3
"""Source-bound paired profiler for the medium-context split-K burst graph."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import time

from tools import profile_exact_greedy_decode_burst as _base
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


CASE_SCHEMA_VERSION = "exact-burst-medium-split-k.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-burst-medium-split-k.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = (
    "exact-burst-medium-split-k.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "exact-burst-medium-split-k.workload.v1"
)
SOURCE_SCHEMA_VERSION = (
    "exact-burst-medium-split-k.source.v1"
)
CORRECTNESS_TRACE_IDENTITY = (
    "gate-only-exact-burst-medium-split-k-correctness-v1"
)

POLICIES = ("auto", "split12")
POLICY_CONFIGS = {
    "auto": {
        "exact_greedy_decode_burst_medium_split_k": False,
    },
    "split12": {
        "exact_greedy_decode_burst_medium_split_k": True,
    },
}
CONTEXT_LENGTHS = (
    1025,
    1537,
    2049,
    2561,
    3073,
    3585,
    4090,
    6145,
)
GENERATED_TOKENS = 128
REPETITIONS = 5
WARMUP_REPETITIONS = 2
BURST_WIDTH = 8
SAMPLING_POINTS = _base.SAMPLING_POINTS
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/scheduler.py",
    "tools/profile_exact_burst_medium_split_k.py",
    "tools/test_profile_exact_burst_medium_split_k.py",
    "tools/exact_burst_medium_split_k_gate.py",
    "tools/test_exact_burst_medium_split_k_gate.py",
    "tools/exact_burst_medium_split_k_verify.py",
    "tools/test_exact_burst_medium_split_k_verify.py",
    "tools/run_exact_burst_medium_split_k_remote.py",
    "tools/test_run_exact_burst_medium_split_k_remote.py",
)

_BURST_COUNTER_FIELDS = _base.BURST_COUNTER_FIELDS
_CAPTURE_SUM_FIELDS = {
    "capture_duration_ns": "capture_duration_ns",
    "capture_allocated_delta_bytes": "allocated_delta_bytes",
    "capture_reserved_delta_bytes": "reserved_delta_bytes",
    "capture_retained_static_bytes": "retained_static_bytes",
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
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _context_bucket(context_length: int) -> str:
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError("context length is invalid")
    return f"context-{context_length}"


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


def correctness_identities(
) -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (context_length, policy, point)
        for context_length in CONTEXT_LENGTHS
        for policy in POLICIES
        for point in SAMPLING_POINTS
    )


def expected_flash_attn_num_splits(
    *,
    policy: str,
    context_length: int,
) -> int:
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError("context length is invalid")
    if (
        policy == "split12"
        and context_length >= 1537
        and context_length + BURST_WIDTH - 1 <= 4097
    ):
        return 12
    return 0


def runtime_environment_manifest() -> dict:
    return _base.runtime_environment_manifest()


def build_workload_manifest(
    *,
    model: str,
    run_tag: str,
    source_commit: str,
    gpu_memory_utilization: float,
    environment: dict,
    repetitions: int = REPETITIONS,
    warmup_repetitions: int = WARMUP_REPETITIONS,
) -> dict:
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
        "contexts": list(CONTEXT_LENGTHS),
        "policies": list(POLICIES),
        "policy_configs": {
            policy: dict(config)
            for policy, config in POLICY_CONFIGS.items()
        },
        "generated_tokens": GENERATED_TOKENS,
        "burst_width": BURST_WIDTH,
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "environment": dict(environment),
        "performance_row_count": (
            repetitions * len(CONTEXT_LENGTHS) * len(POLICIES)
        ),
        "correctness_row_count": (
            len(CONTEXT_LENGTHS)
            * len(POLICIES)
            * len(SAMPLING_POINTS)
        ),
        "policy_order": {
            str(repetition): {
                str(context_length): list(
                    policy_order(repetition, context_index)
                )
                for context_index, context_length
                in enumerate(CONTEXT_LENGTHS)
            }
            for repetition in range(repetitions)
        },
        "correctness_sampling_points": list(SAMPLING_POINTS),
        "correctness_trace_identity": CORRECTNESS_TRACE_IDENTITY,
    }


def _validate_capture_receipt(value) -> dict:
    required = {
        "graph_identity_sha256",
        "graph_generation",
        "capture_duration_ns",
        "allocated_delta_bytes",
        "reserved_delta_bytes",
        "retained_static_bytes",
        "scratch_block_count",
        "correctness_trace",
        "flash_attn_num_splits",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("capture receipt fields mismatch")
    _validate_digest(value["graph_identity_sha256"], "graph identity")
    for field in required - {
        "graph_identity_sha256",
        "correctness_trace",
    }:
        _require_non_negative_int(
            value[field],
            f"capture receipt {field}",
        )
    if not isinstance(value["correctness_trace"], bool):
        raise ValueError("capture correctness trace is invalid")
    if value["flash_attn_num_splits"] not in (0, 12):
        raise ValueError("capture split-K value is invalid")
    return dict(value)


def _validate_summary(
    value,
    *,
    policy: str,
    context_length: int,
    correctness_trace: bool,
) -> dict:
    required = set(_BURST_COUNTER_FIELDS) | {
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
        "quarantine_reason",
        "capture_receipts",
    }
    if not isinstance(value, dict) or required - set(value):
        raise ValueError("exact burst summary fields are missing")
    normalized = dict(value)
    for field in _BURST_COUNTER_FIELDS:
        normalized[field] = _require_non_negative_int(
            value[field],
            f"exact burst summary {field}",
        )
    for field in (
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
    ):
        if not isinstance(value[field], dict):
            raise ValueError(f"{field} is invalid")
    if value["quarantine_reason"] is not None:
        raise ValueError("burst quarantine reason is present")
    receipts = value["capture_receipts"]
    if not isinstance(receipts, list):
        raise ValueError("capture receipts must be a list")
    normalized_receipts = [
        _validate_capture_receipt(receipt) for receipt in receipts
    ]
    if any(
        receipt["correctness_trace"] is not correctness_trace
        for receipt in normalized_receipts
    ):
        raise ValueError("capture correctness trace mismatch")
    splits = [
        receipt["flash_attn_num_splits"]
        for receipt in normalized_receipts
    ]
    expected_splits = [0] if policy == "auto" else [0, 12]
    if sorted(splits) != expected_splits or len(set(splits)) != len(splits):
        raise ValueError("capture receipt split inventory mismatch")
    if (
        normalized["intermediate_token_d2h_calls"] != 0
        or normalized["failures"] != 0
        or normalized["quarantines"] != 0
        or normalized["pending_leases"] != 0
    ):
        raise ValueError("exact burst lifecycle mismatch")
    selected_split = expected_flash_attn_num_splits(
        policy=policy,
        context_length=context_length,
    )
    if "selected_flash_attn_num_splits" in value:
        if value["selected_flash_attn_num_splits"] != selected_split:
            raise ValueError("selected split-K mapping mismatch")
    normalized["selected_flash_attn_num_splits"] = selected_split
    normalized["capture_receipts"] = normalized_receipts
    return normalized


def _capture_cost(receipts: list[dict]) -> dict:
    result = {
        output: sum(receipt[source] for receipt in receipts)
        for output, source in _CAPTURE_SUM_FIELDS.items()
    }
    result["reserved_scratch_blocks"] = max(
        (
            receipt["scratch_block_count"]
            for receipt in receipts
        ),
        default=0,
    )
    return result


def validate_case_row(row) -> dict:
    required = {
        "schema_version",
        "run_tag",
        "source_commit",
        "policy",
        "repetition",
        "order_position",
        "context_length",
        "generated_tokens",
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
        *_CAPTURE_SUM_FIELDS,
        "reserved_scratch_blocks",
        "replay_graph_identity_sha256",
        "replay_graph_identity_counts",
        "replay_flash_attn_num_splits",
        "correctness_trace",
        "exact_greedy_decode_burst_summary",
    }
    if not isinstance(row, dict) or required - set(row):
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
    context_index = CONTEXT_LENGTHS.index(context_length)
    order = policy_order(repetition, context_index)
    if order_position >= len(order) or order[order_position] != policy:
        raise ValueError("policy order position mismatch")
    if row["generated_tokens"] != GENERATED_TOKENS:
        raise ValueError("generated token inventory mismatch")
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
        actual = _require_finite_non_negative(row[field], field)
        if actual != expected:
            raise ValueError(f"TPOT summary mismatch: {field}")
    for field in (
        "ttft_ns",
        "e2e_ns",
        "output_tokens_per_second",
        "maximum_host_visible_burst_gap_ns",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        *_CAPTURE_SUM_FIELDS,
        "reserved_scratch_blocks",
    ):
        _require_finite_non_negative(row[field], field)
    for field in (
        "decode_host_ns",
        "decode_cuda_ns",
        "host_visible_burst_gaps_ns",
    ):
        if not isinstance(row[field], list):
            raise ValueError(f"{field} must be a list")
        for value in row[field]:
            _require_finite_non_negative(value, field)
    if row["correctness_trace"] is not False:
        raise ValueError("performance row cannot enable correctness tracing")
    summary = _validate_summary(
        row["exact_greedy_decode_burst_summary"],
        policy=policy,
        context_length=context_length,
        correctness_trace=False,
    )
    expected_split = expected_flash_attn_num_splits(
        policy=policy,
        context_length=context_length,
    )
    if row["replay_flash_attn_num_splits"] != expected_split:
        raise ValueError("selected split-K mapping mismatch")
    identity = _validate_digest(
        row["replay_graph_identity_sha256"],
        "replay graph identity",
    )
    receipt_by_split = {
        receipt["flash_attn_num_splits"]: receipt
        for receipt in summary["capture_receipts"]
    }
    if (
        receipt_by_split[expected_split]["graph_identity_sha256"]
        != identity
    ):
        raise ValueError("replay graph identity mismatch")
    identity_counts = row["replay_graph_identity_counts"]
    if (
        not isinstance(identity_counts, dict)
        or not identity_counts
        or any(
            _validate_digest(
                graph_identity,
                "replay graph identity inventory",
            )
            not in {
                receipt["graph_identity_sha256"]
                for receipt in summary["capture_receipts"]
            }
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for graph_identity, count in identity_counts.items()
        )
        or sum(identity_counts.values()) != GENERATED_TOKENS - 1
        or identity not in identity_counts
    ):
        raise ValueError("replay graph identity inventory mismatch")
    expected_cost = _capture_cost(summary["capture_receipts"])
    for field, expected in expected_cost.items():
        if row[field] != expected:
            raise ValueError(f"capture cost mismatch: {field}")
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
        performance_identities(
            repetitions=expected_repetitions
        )
    ):
        raise ValueError("performance row inventory is incomplete")
    run_tags = {row["run_tag"] for row in validated}
    source_commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(source_commits) != 1:
        raise ValueError(
            "performance rows do not share source identity"
        )
    comparison_sets = {}
    all_outputs_exact = True
    for row in validated:
        key = (row["repetition"], row["context_length"])
        comparison_sets.setdefault(key, {})[row["policy"]] = row
    for arms in comparison_sets.values():
        if (
            arms["auto"]["output_token_ids"]
            != arms["split12"]["output_token_ids"]
        ):
            all_outputs_exact = False
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(run_tags)),
        "source_commit": next(iter(source_commits)),
        "row_count": len(validated),
        "comparison_set_count": len(comparison_sets),
        "all_outputs_exact": all_outputs_exact,
    }


def _sampling_output_index(
    point: str,
    generated_tokens: int,
) -> int:
    try:
        return {
            "prefill-final": 0,
            "decode-first": 1,
            "decode-middle": generated_tokens // 2,
            "decode-final": generated_tokens - 1,
        }[point]
    except KeyError as error:
        raise ValueError("sampling point is invalid") from error


def _expected_replay_ordinal(point: str) -> int | None:
    if point == "prefill-final":
        return None
    return (
        _sampling_output_index(point, GENERATED_TOKENS) - 1
    ) % BURST_WIDTH


def _correctness_trace_for_step(
    *,
    emitted_total: int,
) -> bool:
    if emitted_total <= 0 or emitted_total >= GENERATED_TOKENS:
        return False
    sampled_indices = {
        _sampling_output_index(point, GENERATED_TOKENS)
        for point in SAMPLING_POINTS[1:]
    }
    return any(
        emitted_total <= index
        < min(emitted_total + BURST_WIDTH, GENERATED_TOKENS)
        for index in sampled_indices
    )


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
        "logits_path",
        "logits_shape",
        "logits_element_count",
        "logits_byte_length",
        "logits_sha256",
        "correctness_trace",
        "trace_identity",
        "trace_graph_identity_sha256",
        "trace_flash_attn_num_splits",
        "selected_replay_ordinal",
        "sampled_logit_d2h_calls",
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
        if row["generated_tokens"] != GENERATED_TOKENS:
            raise ValueError(
                "correctness generated token inventory mismatch"
            )
        _validate_digest(
            row["source_commit"],
            "correctness source commit",
            lengths=(40, 64),
        )
        _validate_digest(
            row["prompt_sha256"],
            "correctness prompt digest",
        )
        _validate_digest(
            row["output_text_sha256"],
            "correctness output text digest",
        )
        if (
            row["correctness_trace"] is not True
            or row["trace_identity"]
            != CORRECTNESS_TRACE_IDENTITY
        ):
            raise ValueError("correctness trace identity mismatch")
        output_ids = row["output_token_ids"]
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != GENERATED_TOKENS
        ):
            raise ValueError(
                "correctness output token inventory mismatch"
            )
        shape = row["logits_shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or not isinstance(shape[1], int)
            or shape[1] <= 0
        ):
            raise ValueError("correctness logits shape is invalid")
        values = read_float32_sidecar(
            run_dir,
            path=row["logits_path"],
            expected_element_count=row["logits_element_count"],
            expected_byte_length=row["logits_byte_length"],
            expected_sha256=row["logits_sha256"],
        )
        if len(values) != shape[0] * shape[1]:
            raise ValueError(
                "correctness logits element count mismatch"
            )
        summary = _validate_summary(
            row["exact_greedy_decode_burst_summary"],
            policy=policy,
            context_length=context_length,
            correctness_trace=True,
        )
        burst_sample = point != "prefill-final"
        if burst_sample:
            expected_split = expected_flash_attn_num_splits(
                policy=policy,
                context_length=context_length,
            )
            if row["trace_flash_attn_num_splits"] != expected_split:
                raise ValueError(
                    "correctness selected split-K mapping mismatch"
                )
            identity = _validate_digest(
                row["trace_graph_identity_sha256"],
                "correctness graph identity",
            )
            receipt_by_split = {
                receipt["flash_attn_num_splits"]: receipt
                for receipt in summary["capture_receipts"]
            }
            if (
                receipt_by_split[expected_split][
                    "graph_identity_sha256"
                ]
                != identity
            ):
                raise ValueError(
                    "correctness graph identity mismatch"
                )
            if (
                row["selected_replay_ordinal"]
                != _expected_replay_ordinal(point)
            ):
                raise ValueError(
                    "correctness replay ordinal mismatch"
                )
            if row["sampled_logit_d2h_calls"] != 1:
                raise ValueError(
                    "correctness sampled-logit D2H mismatch"
                )
        elif any(
            value is not None
            for value in (
                row["trace_graph_identity_sha256"],
                row["trace_flash_attn_num_splits"],
                row["selected_replay_ordinal"],
            )
        ) or row["sampled_logit_d2h_calls"] != 0:
            raise ValueError(
                "prefill correctness row reported burst trace"
            )
        normalized = dict(row)
        normalized["exact_greedy_decode_burst_summary"] = summary
        validated.append(normalized)
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate correctness identity")
    if set(identities) != set(correctness_identities()):
        raise ValueError("correctness row inventory is incomplete")
    run_tags = {row["run_tag"] for row in validated}
    source_commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(source_commits) != 1:
        raise ValueError(
            "correctness rows do not share source identity"
        )
    return validated


def _construct_llm(
    *,
    model: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
    policy: str,
):
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
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
        exact_greedy_decode_burst_tokens=BURST_WIDTH,
        exact_greedy_decode_burst_medium_split_k=(
            POLICY_CONFIGS[policy][
                "exact_greedy_decode_burst_medium_split_k"
            ]
        ),
    )


def _run_request(
    llm,
    *,
    prompt: list[int],
    generated_tokens: int,
    profile_label: str | None,
) -> dict:
    from tinyvllm import SamplingParams

    if profile_label is not None:
        llm.configure_decode_internal_profile(
            True,
            profile_label,
            timeout_s=60.0,
        )
    llm.add_request(
        prompt,
        SamplingParams(
            temperature=0.0,
            max_tokens=generated_tokens,
            ignore_eos=True,
        ),
    )
    started_ns = time.perf_counter_ns()
    first_token_ns = None
    tpot_samples = []
    burst_gaps = []
    graph_identity_counts: dict[str, int] = {}
    final_outputs = None
    while not llm.is_finished():
        step_started_ns = time.perf_counter_ns()
        outputs, _num_tokens = llm.step(completion_only=True)
        step_finished_ns = time.perf_counter_ns()
        observation = llm.last_step_observation
        emitted = sum(
            len(tokens)
            for tokens in observation[
                "new_completion_tokens_by_seq"
            ].values()
        )
        if emitted:
            if first_token_ns is None:
                first_token_ns = step_finished_ns
            elif not observation["is_prefill"]:
                per_token_ns = (
                    step_finished_ns - step_started_ns
                ) / emitted
                tpot_samples.extend([per_token_ns] * emitted)
                gap_ns = observation[
                    "exact_greedy_decode_burst_host_visible_gap_ns"
                ]
                if gap_ns:
                    burst_gaps.append(int(gap_ns))
                identity = observation.get(
                    "exact_greedy_decode_burst_graph_identity_sha256"
                )
                if identity is not None:
                    graph_identity_counts[identity] = (
                        graph_identity_counts.get(identity, 0)
                        + emitted
                    )
        if outputs:
            final_outputs = outputs
    import torch

    torch.cuda.synchronize()
    finished_ns = time.perf_counter_ns()
    if first_token_ns is None:
        raise RuntimeError("request produced no first token")
    if not isinstance(final_outputs, list) or len(final_outputs) != 1:
        raise RuntimeError("request completion output is incomplete")
    output_ids = list(final_outputs[0][1])
    if len(output_ids) != generated_tokens:
        raise RuntimeError("generated token inventory mismatch")
    if len(tpot_samples) != generated_tokens - 1:
        raise RuntimeError("amortized TPOT inventory mismatch")
    if not graph_identity_counts:
        raise RuntimeError("replay graph identity inventory mismatch")
    decode_host_ns = []
    decode_cuda_ns = []
    if profile_label is not None:
        profile = llm.finalize_decode_internal_profile(
            already_synchronized=True,
            timeout_s=60.0,
        )
        rank_rows = profile.get("ranks", ())
        if (
            profile.get("rank_inventory") != [0]
            or len(rank_rows) != 1
        ):
            raise RuntimeError(
                "profile requires tensor parallel size one"
            )
        decode_steps = sorted(
            (
                row
                for row in rank_rows[0]["steps"]
                if row["is_decode"]
            ),
            key=lambda row: row["decode_ordinal"],
        )
        decode_host_ns = [
            int(row["wall_ns"]) for row in decode_steps
        ]
        decode_cuda_ns = [
            int(row["cuda_ns"]) for row in decode_steps
        ]
    return {
        "output_token_ids": output_ids,
        "output_text": llm.tokenizer.decode(output_ids),
        "ttft_ns": first_token_ns - started_ns,
        "e2e_ns": finished_ns - started_ns,
        "amortized_tpot_samples_ns": tpot_samples,
        "decode_host_ns": decode_host_ns,
        "decode_cuda_ns": decode_cuda_ns,
        "host_visible_burst_gaps_ns": burst_gaps,
        "replay_graph_identity_counts": graph_identity_counts,
    }


def run_case(
    *,
    model: str,
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
            )
            llm.clear_reusable_prefix_cache()
        before = _base._runner_summaries(llm)
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
                f"{run_tag}/{context_length}/"
                f"r{repetition}/{policy}"
            ),
        )
        memory = _aggregate_memory(
            llm.memory_snapshots(timeout_s=60.0)
        )
        summary = _base._combined_summary(llm, before)
        summary["maximum_host_visible_gap_ns"] = max(
            measured["host_visible_burst_gaps_ns"],
            default=0,
        )
        selected_split = expected_flash_attn_num_splits(
            policy=policy,
            context_length=context_length,
        )
        summary["selected_flash_attn_num_splits"] = selected_split
        samples = measured["amortized_tpot_samples_ns"]
        e2e_seconds = measured["e2e_ns"] / 1_000_000_000
        receipts = summary["capture_receipts"]
        receipt_by_split = {
            receipt["flash_attn_num_splits"]: receipt
            for receipt in receipts
        }
        selected_identity = receipt_by_split[selected_split][
            "graph_identity_sha256"
        ]
        row = {
            "schema_version": CASE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_commit": source_commit,
            "policy": policy,
            "repetition": repetition,
            "order_position": order_position,
            "context_length": context_length,
            "generated_tokens": GENERATED_TOKENS,
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
            "amortized_tpot_median_ns":
                statistics.median(samples),
            "amortized_tpot_p95_ns":
                _nearest_rank(samples, 0.95),
            "amortized_tpot_p99_ns":
                _nearest_rank(samples, 0.99),
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
            **_capture_cost(receipts),
            "replay_graph_identity_sha256": selected_identity,
            "replay_graph_identity_counts": measured[
                "replay_graph_identity_counts"
            ],
            "replay_flash_attn_num_splits": selected_split,
            "correctness_trace": False,
            "exact_greedy_decode_burst_summary": summary,
        }
        return validate_case_row(row)
    finally:
        llm.exit()


def run_correctness_probe(
    *,
    model: str,
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
        prompt_tokens=context_length,
        generated_tokens=GENERATED_TOKENS,
        gpu_memory_utilization=gpu_memory_utilization,
        policy=policy,
    )
    try:
        llm.model_runner.call(
            "capture_exact_greedy_decode_burst_correctness_graph",
            (0, 6, 7),
        )
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        before = _base._runner_summaries(llm)
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
            trace_this_step = _correctness_trace_for_step(
                emitted_total=emitted_total,
            )
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
                captured["prefill-final"] = {
                    "logits": logits,
                    "trace_graph_identity_sha256": None,
                    "trace_flash_attn_num_splits": None,
                    "selected_replay_ordinal": None,
                    "sampled_logit_d2h_calls": 0,
                }
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
                        if output_index == _sampling_output_index(
                            point,
                            GENERATED_TOKENS,
                        ):
                            captured[point] = {
                                "logits": values,
                                "trace_graph_identity_sha256":
                                    observation[
                                        "exact_greedy_decode_burst_graph_identity_sha256"
                                    ],
                                "trace_flash_attn_num_splits":
                                    expected_flash_attn_num_splits(
                                        policy=policy,
                                        context_length=context_length,
                                    ),
                                "selected_replay_ordinal": int(
                                    local_ordinal
                                ),
                                "sampled_logit_d2h_calls": int(
                                    observation[
                                        "exact_greedy_decode_burst_sampled_logit_d2h_calls"
                                    ]
                                ),
                            }
            emitted_total += emitted
            if outputs:
                final_outputs = outputs
        if set(captured) != set(SAMPLING_POINTS):
            raise RuntimeError(
                "correctness sampling points are incomplete"
            )
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
        summary = _base._combined_summary(
            llm,
            before,
            correctness_trace=True,
        )
        summary["selected_flash_attn_num_splits"] = (
            expected_flash_attn_num_splits(
                policy=policy,
                context_length=context_length,
            )
        )
        prompt_sha256 = sha256_text(
            ",".join(str(token) for token in prompt)
        )
        rows = []
        for point in SAMPLING_POINTS:
            sample = captured[point]
            logits = sample["logits"]
            shape = [int(value) for value in logits.shape]
            values = logits.view(-1).tolist()
            sidecar = write_float32_sidecar(
                run_dir,
                (
                    f"logits/{context_length}-{policy}-"
                    f"{point}.f32"
                ),
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
                "logits_path": sidecar["path"],
                "logits_shape": shape,
                "logits_element_count": sidecar["element_count"],
                "logits_byte_length": sidecar["byte_length"],
                "logits_sha256": sidecar["sha256"],
                "correctness_trace": True,
                "trace_identity": CORRECTNESS_TRACE_IDENTITY,
                "trace_graph_identity_sha256": sample[
                    "trace_graph_identity_sha256"
                ],
                "trace_flash_attn_num_splits": sample[
                    "trace_flash_attn_num_splits"
                ],
                "selected_replay_ordinal": sample[
                    "selected_replay_ordinal"
                ],
                "sampled_logit_d2h_calls": sample[
                    "sampled_logit_d2h_calls"
                ],
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
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
        "--context-lengths",
        default=",".join(str(value) for value in CONTEXT_LENGTHS),
    )
    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=GENERATED_TOKENS,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contexts = tuple(
        int(item.strip())
        for item in args.context_lengths.split(",")
        if item.strip()
    )
    if args.repetitions not in (3, REPETITIONS):
        raise ValueError("repetitions must equal 3 or 5")
    if args.warmup_repetitions not in (1, WARMUP_REPETITIONS):
        raise ValueError("warmup repetitions must equal 1 or 2")
    if contexts != CONTEXT_LENGTHS:
        raise ValueError("context lengths do not match frozen workload")
    if args.generated_tokens != GENERATED_TOKENS:
        raise ValueError(
            f"generated tokens must equal {GENERATED_TOKENS}"
        )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[1]
    _write_json(
        out_dir / "source_manifest.json",
        _source_manifest(
            repo_root=repo_root,
            source_commit=args.source_commit,
            run_tag=args.run_tag,
        ),
    )
    _write_json(
        out_dir / "workload_manifest.json",
        build_workload_manifest(
            model=args.model,
            run_tag=args.run_tag,
            source_commit=args.source_commit,
            gpu_memory_utilization=args.gpu_memory_utilization,
            environment=runtime_environment_manifest(),
            repetitions=args.repetitions,
            warmup_repetitions=args.warmup_repetitions,
        ),
    )
    rows = []
    row_path = out_dir / "performance_rows.jsonl"
    for repetition in range(args.repetitions):
        for context_index, context_length in enumerate(
            CONTEXT_LENGTHS
        ):
            for order_position, policy in enumerate(
                policy_order(repetition, context_index)
            ):
                row = run_case(
                    model=args.model,
                    run_tag=args.run_tag,
                    source_commit=args.source_commit,
                    policy=policy,
                    repetition=repetition,
                    order_position=order_position,
                    context_length=context_length,
                    warmup_repetitions=args.warmup_repetitions,
                    gpu_memory_utilization=(
                        args.gpu_memory_utilization
                    ),
                )
                append_jsonl(row_path, row)
                rows.append(row)
    correctness_rows = []
    correctness_path = out_dir / "correctness_rows.jsonl"
    for context_length in CONTEXT_LENGTHS:
        for policy in POLICIES:
            for row in run_correctness_probe(
                model=args.model,
                run_dir=out_dir,
                run_tag=args.run_tag,
                source_commit=args.source_commit,
                policy=policy,
                context_length=context_length,
                gpu_memory_utilization=(
                    args.gpu_memory_utilization
                ),
            ):
                append_jsonl(correctness_path, row)
                correctness_rows.append(row)
    validate_correctness_rows(
        correctness_rows,
        run_dir=out_dir,
    )
    summary = summarize_rows(
        rows,
        expected_repetitions=args.repetitions,
    )
    summary["correctness_row_count"] = len(correctness_rows)
    _write_json(out_dir / "profile_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
