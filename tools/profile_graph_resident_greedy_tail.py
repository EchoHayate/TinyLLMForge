#!/usr/bin/env python3
"""Source-bound three-arm benchmark for graph-resident greedy decode."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics

from tools.profile_zero_temperature_greedy_fast_path import (
    _aggregate_memory,
    _make_prompt,
    _run_request,
    _write_json,
    append_jsonl,
    read_float32_sidecar,
    sha256_file,
    sha256_text,
    write_float32_sidecar,
)


CASE_SCHEMA_VERSION = "graph-resident-greedy-tail.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "graph-resident-greedy-tail.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = "graph-resident-greedy-tail.summary.v1"
WORKLOAD_SCHEMA_VERSION = "graph-resident-greedy-tail.workload.v1"
SOURCE_SCHEMA_VERSION = "graph-resident-greedy-tail.source.v1"
POLICIES = ("legacy", "host_greedy", "graph_greedy")
POLICY_FLAGS = {
    "legacy": (False, False),
    "host_greedy": (True, False),
    "graph_greedy": (True, True),
}
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-final",
)
GREEDY_COUNTER_FIELDS = (
    "eligible_steps",
    "optimized_steps",
    "avoided_temperature_h2d_bytes",
    "avoided_softmax_calls",
    "avoided_gumbel_rng_calls",
    "avoided_stochastic_divisions",
    "avoided_stochastic_argmax_calls",
    "avoided_where_calls",
)
GRAPH_COUNTER_FIELDS = (
    "eligible_steps",
    "captured_graphs",
    "replayed_steps",
    "final_token_d2h_calls",
    "avoided_external_compute_logits_calls",
    "avoided_external_float32_conversions",
    "avoided_external_argmax_calls",
)
GRAPH_COST_FIELDS = (
    "graph_capture_duration_ns",
    "graph_allocated_delta_bytes",
    "graph_reserved_delta_bytes",
    "graph_retained_static_bytes",
)
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/greedy_sampling_fast_path.py",
    "tinyvllm/engine/graph_resident_greedy_tail.py",
    "tinyvllm/engine/model_runner.py",
    "tools/profile_zero_temperature_greedy_fast_path.py",
    "tools/profile_graph_resident_greedy_tail.py",
    "tools/test_profile_graph_resident_greedy_tail.py",
    "tools/graph_resident_greedy_tail_gate.py",
    "tools/test_graph_resident_greedy_tail_gate.py",
    "tools/graph_resident_greedy_tail_verify.py",
    "tools/test_graph_resident_greedy_tail_verify.py",
    "tools/run_graph_resident_greedy_tail_remote.py",
    "tools/test_run_graph_resident_greedy_tail_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)


def context_cases() -> tuple[tuple[str, int, int], ...]:
    return (
        ("short", 256, 128),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    )


def policy_order(repetition: int) -> tuple[str, str, str]:
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
    ):
        raise ValueError(
            "repetition must be a non-negative integer"
        )
    if repetition % 2 == 0:
        return POLICIES
    return tuple(reversed(POLICIES))


def _require_non_negative_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_finite_non_negative(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and non-negative"
        )
    return float(value)


def _validate_digest(value, name: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _validate_fallback_counts(value, name: str) -> dict[str, int]:
    if (
        not isinstance(value, dict)
        or any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for reason, count in value.items()
        )
    ):
        raise ValueError(f"{name} is invalid")
    return dict(sorted(value.items()))


def _validate_greedy_summary_shape(summary) -> dict:
    if not isinstance(summary, dict):
        raise ValueError(
            "greedy fast-path summary must be an object"
        )
    missing = (
        set(GREEDY_COUNTER_FIELDS) | {"fallback_counts"}
    ) - set(summary)
    if missing:
        raise ValueError(
            "greedy fast-path summary fields are missing: "
            f"{sorted(missing)}"
        )
    normalized = {
        field: _require_non_negative_int(
            summary[field],
            f"greedy_fast_path_summary.{field}",
        )
        for field in GREEDY_COUNTER_FIELDS
    }
    normalized["fallback_counts"] = _validate_fallback_counts(
        summary["fallback_counts"],
        "greedy fast-path fallback counts",
    )
    return normalized


def _validate_greedy_summary(
    summary,
    *,
    policy: str,
    generated_tokens: int,
) -> dict:
    normalized = _validate_greedy_summary_shape(summary)
    optimized = {
        "legacy": 0,
        "host_greedy": generated_tokens,
        "graph_greedy": 1,
    }[policy]
    if (
        normalized["eligible_steps"] != optimized
        or normalized["optimized_steps"] != optimized
    ):
        raise ValueError(
            "greedy fast-path step inventory mismatch"
        )
    expected_avoided = {
        "avoided_temperature_h2d_bytes": 4 * optimized,
        "avoided_softmax_calls": optimized,
        "avoided_gumbel_rng_calls": optimized,
        "avoided_stochastic_divisions": 2 * optimized,
        "avoided_stochastic_argmax_calls": optimized,
        "avoided_where_calls": optimized,
    }
    for field, expected in expected_avoided.items():
        if normalized[field] != expected:
            raise ValueError(
                f"greedy avoided work mismatch: {field}"
            )
    if policy == "legacy":
        if normalized["fallback_counts"] != {
            "disabled": generated_tokens
        }:
            raise ValueError(
                "legacy greedy fallback inventory mismatch"
            )
    elif normalized["fallback_counts"]:
        raise ValueError(
            "enabled greedy path unexpectedly fell back"
        )
    return normalized


def _validate_source_identity(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            "graph-tail source identity must be an object"
        )
    required = {
        "data_ptr",
        "shape",
        "stride",
        "storage_offset",
        "dtype",
        "device",
    }
    if set(value) != required:
        raise ValueError(
            "graph-tail source identity fields mismatch"
        )
    _require_non_negative_int(
        value["data_ptr"],
        "graph-tail source data pointer",
    )
    _require_non_negative_int(
        value["storage_offset"],
        "graph-tail source storage offset",
    )
    shape = value["shape"]
    stride = value["stride"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in shape
        )
    ):
        raise ValueError("graph-tail source shape is invalid")
    if (
        not isinstance(stride, list)
        or len(stride) != len(shape)
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            for item in stride
        )
    ):
        raise ValueError("graph-tail source stride is invalid")
    for field in ("dtype", "device"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(
                f"graph-tail source {field} is invalid"
            )
    return {
        "data_ptr": value["data_ptr"],
        "shape": list(shape),
        "stride": list(stride),
        "storage_offset": value["storage_offset"],
        "dtype": value["dtype"],
        "device": value["device"],
    }


def _validate_capture_receipt(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            "graph-tail capture receipt must be an object"
        )
    required = {
        "source_identity",
        "graph_generation",
        "rank",
        "capture_duration_ns",
        "allocated_delta_bytes",
        "reserved_delta_bytes",
        "retained_logits_bytes",
        "retained_float32_bytes",
        "retained_token_bytes",
        "retained_static_bytes",
    }
    if set(value) != required:
        raise ValueError(
            "graph-tail capture receipt fields mismatch"
        )
    normalized = {
        "source_identity": _validate_source_identity(
            value["source_identity"]
        )
    }
    for field in required - {"source_identity"}:
        normalized[field] = _require_non_negative_int(
            value[field],
            f"graph-tail capture receipt {field}",
        )
    if normalized["graph_generation"] <= 0:
        raise ValueError(
            "graph-tail graph generation must be positive"
        )
    retained = (
        normalized["retained_logits_bytes"]
        + normalized["retained_float32_bytes"]
        + normalized["retained_token_bytes"]
    )
    if normalized["retained_static_bytes"] != retained:
        raise ValueError(
            "graph-tail retained byte accounting mismatch"
        )
    return normalized


def _validate_graph_summary_shape(summary) -> dict:
    if not isinstance(summary, dict):
        raise ValueError(
            "graph-resident greedy-tail summary must be an object"
        )
    required = (
        set(GRAPH_COUNTER_FIELDS)
        | {
            "fallback_counts",
            "quarantine_reason",
            "capture_receipt",
        }
    )
    missing = required - set(summary)
    if missing:
        raise ValueError(
            "graph-tail summary fields are missing: "
            f"{sorted(missing)}"
        )
    normalized = {
        field: _require_non_negative_int(
            summary[field],
            f"graph_resident_greedy_tail_summary.{field}",
        )
        for field in GRAPH_COUNTER_FIELDS
    }
    normalized["fallback_counts"] = _validate_fallback_counts(
        summary["fallback_counts"],
        "graph-tail fallback counts",
    )
    quarantine_reason = summary["quarantine_reason"]
    if quarantine_reason is not None and (
        not isinstance(quarantine_reason, str)
        or not quarantine_reason
    ):
        raise ValueError(
            "graph-tail quarantine reason is invalid"
        )
    normalized["quarantine_reason"] = quarantine_reason
    receipt = summary["capture_receipt"]
    normalized["capture_receipt"] = (
        None
        if receipt is None
        else _validate_capture_receipt(receipt)
    )
    return normalized


def _validate_graph_summary(
    summary,
    *,
    policy: str,
    generated_tokens: int,
) -> dict:
    normalized = _validate_graph_summary_shape(summary)
    expected_steps = (
        generated_tokens - 1
        if policy == "graph_greedy"
        else 0
    )
    if normalized["eligible_steps"] != expected_steps:
        raise ValueError(
            "graph-tail eligible inventory mismatch"
        )
    if normalized["replayed_steps"] != expected_steps:
        raise ValueError("graph-tail replay inventory mismatch")
    if normalized["final_token_d2h_calls"] != expected_steps:
        raise ValueError("graph-tail token D2H inventory mismatch")
    for field in (
        "avoided_external_compute_logits_calls",
        "avoided_external_float32_conversions",
        "avoided_external_argmax_calls",
    ):
        if normalized[field] != expected_steps:
            raise ValueError(
                f"graph-tail avoided work mismatch: {field}"
            )
    expected_captures = 1 if policy == "graph_greedy" else 0
    if normalized["captured_graphs"] != expected_captures:
        raise ValueError(
            "graph-tail capture inventory mismatch"
        )
    if policy == "graph_greedy":
        if normalized["capture_receipt"] is None:
            raise ValueError(
                "graph-tail capture receipt is missing"
            )
        if normalized["fallback_counts"]:
            raise ValueError(
                "graph-tail measured request fell back"
            )
        if normalized["quarantine_reason"] is not None:
            raise ValueError(
                "graph-tail measured request was quarantined"
            )
    elif (
        normalized["capture_receipt"] is not None
        or normalized["fallback_counts"]
        or normalized["quarantine_reason"] is not None
    ):
        raise ValueError(
            "disabled graph-tail path reported activity"
        )
    return normalized


def _cost_from_graph_summary(summary: dict) -> dict[str, int]:
    receipt = summary["capture_receipt"]
    if receipt is None:
        return {
            field: 0
            for field in GRAPH_COST_FIELDS
        }
    return {
        "graph_capture_duration_ns":
            receipt["capture_duration_ns"],
        "graph_allocated_delta_bytes":
            receipt["allocated_delta_bytes"],
        "graph_reserved_delta_bytes":
            receipt["reserved_delta_bytes"],
        "graph_retained_static_bytes":
            receipt["retained_static_bytes"],
    }


def validate_case_row(
    row,
    *,
    require_complete_optimized_path: bool = True,
) -> dict:
    if not isinstance(row, dict):
        raise ValueError("case row must be an object")
    required = {
        "schema_version",
        "run_tag",
        "source_commit",
        "policy",
        "repetition",
        "context_bucket",
        "prompt_tokens",
        "generated_tokens",
        "output_token_ids",
        "output_text_sha256",
        "ttft_ns",
        "e2e_ns",
        "tpot_samples_ns",
        "decode_host_ns",
        "decode_cuda_ns",
        "output_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "greedy_fast_path_summary",
        "graph_resident_greedy_tail_summary",
        *GRAPH_COST_FIELDS,
    }
    missing = required - set(row)
    if missing:
        raise ValueError(
            f"case row fields are missing: {sorted(missing)}"
        )
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
    repetition = _require_non_negative_int(
        row["repetition"],
        "repetition",
    )
    cases = {
        bucket: (prompt_tokens, generated_tokens)
        for bucket, prompt_tokens, generated_tokens
        in context_cases()
    }
    bucket = row["context_bucket"]
    if bucket not in cases:
        raise ValueError("context bucket is invalid")
    prompt_tokens = _require_non_negative_int(
        row["prompt_tokens"],
        "prompt_tokens",
    )
    generated_tokens = _require_non_negative_int(
        row["generated_tokens"],
        "generated_tokens",
    )
    if (prompt_tokens, generated_tokens) != cases[bucket]:
        raise ValueError("context shape does not match bucket")
    output_ids = row["output_token_ids"]
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != generated_tokens
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in output_ids
        )
    ):
        raise ValueError("output token inventory is invalid")
    _validate_digest(
        row["output_text_sha256"],
        "output text digest",
    )
    expected_decode_steps = generated_tokens - 1
    for values, name in (
        (row["tpot_samples_ns"], "tpot_samples_ns"),
        (row["decode_host_ns"], "decode_host_ns"),
        (row["decode_cuda_ns"], "decode_cuda_ns"),
    ):
        if (
            not isinstance(values, list)
            or len(values) != expected_decode_steps
        ):
            raise ValueError(f"{name} inventory mismatch")
        for index, value in enumerate(values):
            _require_finite_non_negative(
                value,
                f"{name}[{index}]",
            )
    for field in (
        "ttft_ns",
        "e2e_ns",
        "output_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        *GRAPH_COST_FIELDS,
    ):
        _require_finite_non_negative(row[field], field)
    if require_complete_optimized_path:
        greedy_summary = _validate_greedy_summary(
            row["greedy_fast_path_summary"],
            policy=policy,
            generated_tokens=generated_tokens,
        )
        graph_summary = _validate_graph_summary(
            row["graph_resident_greedy_tail_summary"],
            policy=policy,
            generated_tokens=generated_tokens,
        )
    else:
        greedy_summary = _validate_greedy_summary_shape(
            row["greedy_fast_path_summary"]
        )
        graph_summary = _validate_graph_summary_shape(
            row["graph_resident_greedy_tail_summary"]
        )
    expected_cost = _cost_from_graph_summary(graph_summary)
    for field, expected in expected_cost.items():
        if row[field] != expected:
            raise ValueError(
                f"graph-tail cost mismatch: {field}"
            )
    normalized = dict(row)
    normalized["repetition"] = repetition
    normalized["greedy_fast_path_summary"] = greedy_summary
    normalized[
        "graph_resident_greedy_tail_summary"
    ] = graph_summary
    return normalized


def summarize_rows(
    rows: list[dict],
    *,
    expected_repetitions: int = 5,
) -> dict:
    if (
        isinstance(expected_repetitions, bool)
        or not isinstance(expected_repetitions, int)
        or expected_repetitions <= 0
    ):
        raise ValueError(
            "expected repetitions must be a positive integer"
        )
    validated = [validate_case_row(row) for row in rows]
    identities = {}
    for row in validated:
        identity = (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        )
        if identity in identities:
            raise ValueError(
                f"duplicate case identity: {identity}"
            )
        identities[identity] = row
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in context_cases()
        for repetition in range(expected_repetitions)
        for policy in POLICIES
    }
    if set(identities) != expected:
        raise ValueError("case row inventory is incomplete")
    triples = []
    for bucket, _prompt, _generated in context_cases():
        for repetition in range(expected_repetitions):
            triple = [
                identities[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            outputs = {
                tuple(row["output_token_ids"])
                for row in triple
            }
            text_hashes = {
                row["output_text_sha256"]
                for row in triple
            }
            if len(outputs) != 1:
                raise ValueError(
                    "output token mismatch in policy triple"
                )
            if len(text_hashes) != 1:
                raise ValueError(
                    "output text mismatch in policy triple"
                )
            triples.append({
                "context_bucket": bucket,
                "repetition": repetition,
                "legacy_tpot_median_ns": statistics.median(
                    triple[0]["tpot_samples_ns"]
                ),
                "host_greedy_tpot_median_ns": statistics.median(
                    triple[1]["tpot_samples_ns"]
                ),
                "graph_greedy_tpot_median_ns": statistics.median(
                    triple[2]["tpot_samples_ns"]
                ),
            })
    run_tags = {row["run_tag"] for row in validated}
    commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "case rows do not share source identity"
        )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(run_tags)),
        "source_commit": next(iter(commits)),
        "row_count": len(validated),
        "triple_count": len(triples),
        "all_outputs_exact": True,
        "all_graph_decode_steps_optimized": all(
            row[
                "graph_resident_greedy_tail_summary"
            ]["replayed_steps"] == row["generated_tokens"] - 1
            and row[
                "graph_resident_greedy_tail_summary"
            ]["final_token_d2h_calls"]
            == row["generated_tokens"] - 1
            for row in validated
            if row["policy"] == "graph_greedy"
        ),
        "triples": triples,
    }


def validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
    expected_buckets: tuple[str, ...] = (
        "short",
        "medium",
        "long",
    ),
) -> list[dict]:
    expected = {
        (bucket, policy, point)
        for bucket in expected_buckets
        for policy in POLICIES
        for point in SAMPLING_POINTS
    }
    identities = {}
    validated = []
    cases = {
        name: (prompt_tokens, generated_tokens)
        for name, prompt_tokens, generated_tokens
        in context_cases()
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "correctness row must be an object"
            )
        if row.get("schema_version") != CORRECTNESS_SCHEMA_VERSION:
            raise ValueError("correctness row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        if identity in identities:
            raise ValueError(
                f"duplicate correctness identity: {identity}"
            )
        if identity not in expected:
            raise ValueError(
                f"unexpected correctness identity: {identity}"
            )
        identities[identity] = row
        bucket, policy, _point = identity
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != cases[bucket]:
            raise ValueError(
                "correctness context shape mismatch"
            )
        generated_tokens = row["generated_tokens"]
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != generated_tokens
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output_ids
            )
        ):
            raise ValueError(
                "correctness output token inventory is invalid"
            )
        _validate_digest(
            row.get("output_text_sha256"),
            "correctness output text digest",
        )
        _validate_digest(
            row.get("source_commit"),
            "correctness source commit",
            lengths=(40, 64),
        )
        if not isinstance(row.get("run_tag"), str) or not row["run_tag"]:
            raise ValueError(
                "correctness run tag is invalid"
            )
        shape = row.get("logits_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or isinstance(shape[1], bool)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
        ):
            raise ValueError(
                "correctness logits shape is invalid"
            )
        values = read_float32_sidecar(
            run_dir,
            path=row.get("logits_path"),
            expected_element_count=row.get(
                "logits_element_count"
            ),
            expected_byte_length=row.get(
                "logits_byte_length"
            ),
            expected_sha256=row.get("logits_sha256"),
        )
        if len(values) != shape[0] * shape[1]:
            raise ValueError(
                "correctness logits element count mismatch"
            )
        normalized = dict(row)
        normalized["greedy_fast_path_summary"] = (
            _validate_greedy_summary(
                row.get("greedy_fast_path_summary"),
                policy=policy,
                generated_tokens=generated_tokens,
            )
        )
        normalized[
            "graph_resident_greedy_tail_summary"
        ] = _validate_graph_summary(
            row.get(
                "graph_resident_greedy_tail_summary"
            ),
            policy=policy,
            generated_tokens=generated_tokens,
        )
        validated.append(normalized)
    if set(identities) != expected:
        raise ValueError(
            "correctness row inventory is incomplete"
        )
    run_tags = {row.get("run_tag") for row in validated}
    commits = {row.get("source_commit") for row in validated}
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "correctness rows do not share source identity"
        )
    return validated


def _counter_delta(
    before: dict,
    after: dict,
    *,
    fields: tuple[str, ...],
    label: str,
) -> dict:
    result = {}
    for field in fields:
        before_value = _require_non_negative_int(
            before.get(field),
            f"{label} before.{field}",
        )
        after_value = _require_non_negative_int(
            after.get(field),
            f"{label} after.{field}",
        )
        if after_value < before_value:
            raise RuntimeError(
                f"{label} counter decreased: {field}"
            )
        result[field] = after_value - before_value
    before_fallbacks = before.get("fallback_counts")
    after_fallbacks = after.get("fallback_counts")
    if not isinstance(before_fallbacks, dict) or not isinstance(
        after_fallbacks,
        dict,
    ):
        raise RuntimeError(
            f"{label} fallback counters are unavailable"
        )
    result["fallback_counts"] = {}
    for reason in sorted(set(before_fallbacks) | set(after_fallbacks)):
        difference = int(after_fallbacks.get(reason, 0)) - int(
            before_fallbacks.get(reason, 0)
        )
        if difference < 0:
            raise RuntimeError(
                f"{label} fallback counter decreased"
            )
        if difference:
            result["fallback_counts"][reason] = difference
    return result


def _greedy_stats_delta(before: dict, after: dict) -> dict:
    return _counter_delta(
        before,
        after,
        fields=GREEDY_COUNTER_FIELDS,
        label="greedy fast-path",
    )


def _graph_stats_delta(before: dict, after: dict) -> dict:
    result = _counter_delta(
        before,
        after,
        fields=GRAPH_COUNTER_FIELDS,
        label="graph-tail",
    )
    receipt = after.get("capture_receipt")
    before_receipt = before.get("capture_receipt")
    if receipt != before_receipt:
        raise RuntimeError(
            "graph-tail capture receipt changed during request"
        )
    quarantine_reason = after.get("quarantine_reason")
    if before.get("quarantine_reason") != quarantine_reason:
        raise RuntimeError(
            "graph-tail quarantine changed during request"
        )
    result["captured_graphs"] = _require_non_negative_int(
        after.get("captured_graphs"),
        "graph-tail captured graphs",
    )
    result["capture_receipt"] = receipt
    result["quarantine_reason"] = quarantine_reason
    return result


def _construct_llm(
    *,
    model: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
    policy: str,
):
    from tinyvllm import LLM

    try:
        greedy_enabled, graph_enabled = POLICY_FLAGS[policy]
    except KeyError as error:
        raise ValueError("policy is invalid") from error
    return LLM(
        model,
        max_num_batched_tokens=prompt_tokens + generated_tokens,
        max_num_seqs=1,
        max_model_len=prompt_tokens + generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False,
        zero_temperature_greedy_fast_path=greedy_enabled,
        graph_resident_greedy_tail=graph_enabled,
    )


def _runner_summaries(llm) -> tuple[dict, dict]:
    return (
        llm.model_runner.zero_temperature_greedy_fast_path_summary(),
        llm.model_runner.graph_resident_greedy_tail_summary(),
    )


def run_case(
    *,
    model: str,
    run_tag: str,
    source_commit: str,
    policy: str,
    repetition: int,
    context_bucket: str,
    prompt_tokens: int,
    generated_tokens: int,
    warmup_repetitions: int,
    gpu_memory_utilization: float,
) -> dict:
    llm = _construct_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        policy=policy,
    )
    try:
        for warmup_index in range(warmup_repetitions):
            _run_request(
                llm,
                prompt=_make_prompt(
                    prompt_tokens,
                    offset=50_000 + warmup_index * 2_003,
                ),
                generated_tokens=generated_tokens,
                profile_label=None,
            )
            llm.clear_reusable_prefix_cache()
        greedy_before, graph_before = _runner_summaries(llm)
        llm.reset_peak_memory_stats(timeout_s=60.0)
        measured = _run_request(
            llm,
            prompt=_make_prompt(
                prompt_tokens,
                offset=repetition * 10_007,
            ),
            generated_tokens=generated_tokens,
            profile_label=(
                f"{run_tag}/{context_bucket}/"
                f"r{repetition}/{policy}"
            ),
        )
        memory = _aggregate_memory(
            llm.memory_snapshots(timeout_s=60.0)
        )
        greedy_after, graph_after = _runner_summaries(llm)
        greedy_summary = _greedy_stats_delta(
            greedy_before,
            greedy_after,
        )
        graph_summary = _graph_stats_delta(
            graph_before,
            graph_after,
        )
        graph_cost = _cost_from_graph_summary(graph_summary)
        e2e_seconds = measured["e2e_ns"] / 1_000_000_000
        return validate_case_row({
            "schema_version": CASE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_commit": source_commit,
            "policy": policy,
            "repetition": repetition,
            "context_bucket": context_bucket,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "output_token_ids": measured["output_token_ids"],
            "output_text_sha256": sha256_text(
                measured["output_text"]
            ),
            "ttft_ns": measured["ttft_ns"],
            "e2e_ns": measured["e2e_ns"],
            "tpot_samples_ns": measured["tpot_samples_ns"],
            "decode_host_ns": measured["decode_host_ns"],
            "decode_cuda_ns": measured["decode_cuda_ns"],
            "output_tokens_per_second": (
                generated_tokens / e2e_seconds
            ),
            **memory,
            **graph_cost,
            "greedy_fast_path_summary": greedy_summary,
            "graph_resident_greedy_tail_summary":
                graph_summary,
        })
    finally:
        llm.exit()


def run_correctness_probe(
    *,
    model: str,
    run_dir: Path,
    run_tag: str,
    source_commit: str,
    policy: str,
    context_bucket: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
) -> list[dict]:
    from tinyvllm import SamplingParams

    llm = _construct_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        policy=policy,
    )
    try:
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        greedy_before, graph_before = _runner_summaries(llm)
        llm.add_request(
            _make_prompt(prompt_tokens, offset=90_001),
            SamplingParams(
                temperature=0.0,
                max_tokens=generated_tokens,
                ignore_eos=True,
            ),
        )
        captured = {}
        final_outputs = None
        step_index = 0
        while not llm.is_finished():
            outputs, _num_tokens = llm.step()
            if step_index == 0:
                point = "prefill-final"
            elif step_index == 1:
                point = "decode-first"
            elif step_index == generated_tokens - 1:
                point = "decode-final"
            else:
                point = None
            if point is not None:
                logits = (
                    llm.read_step_logits_authority()
                    .detach()
                    .to(dtype=__import__("torch").float32)
                    .contiguous()
                )
                shape = [int(value) for value in logits.shape]
                sidecar = write_float32_sidecar(
                    run_dir,
                    (
                        f"logits/{context_bucket}-{policy}-"
                        f"{point}.f32"
                    ),
                    logits.view(-1).tolist(),
                )
                captured[point] = (shape, sidecar)
            if outputs:
                final_outputs = outputs
            step_index += 1
        if step_index != generated_tokens:
            raise RuntimeError(
                "correctness sampling step inventory mismatch"
            )
        if set(captured) != set(SAMPLING_POINTS):
            raise RuntimeError(
                "correctness sampling points are incomplete"
            )
        if not isinstance(final_outputs, list) or len(final_outputs) != 1:
            raise RuntimeError(
                "correctness output is incomplete"
            )
        output_ids = list(final_outputs[0][1])
        if len(output_ids) != generated_tokens:
            raise RuntimeError(
                "correctness output token inventory mismatch"
            )
        output_text_sha256 = sha256_text(
            llm.tokenizer.decode(output_ids)
        )
        greedy_after, graph_after = _runner_summaries(llm)
        greedy_summary = _greedy_stats_delta(
            greedy_before,
            greedy_after,
        )
        graph_summary = _graph_stats_delta(
            graph_before,
            graph_after,
        )
        rows = []
        for point in SAMPLING_POINTS:
            shape, sidecar = captured[point]
            rows.append({
                "schema_version": CORRECTNESS_SCHEMA_VERSION,
                "run_tag": run_tag,
                "source_commit": source_commit,
                "policy": policy,
                "context_bucket": context_bucket,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "sampling_point": point,
                "output_token_ids": output_ids,
                "output_text_sha256": output_text_sha256,
                "logits_path": sidecar["path"],
                "logits_shape": shape,
                "logits_element_count":
                    sidecar["element_count"],
                "logits_byte_length": sidecar["byte_length"],
                "logits_sha256": sidecar["sha256"],
                "greedy_fast_path_summary":
                    greedy_summary,
                "graph_resident_greedy_tail_summary":
                    graph_summary,
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


def _parse_prompt_lengths(raw: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in raw.split(",")
        if item.strip()
    )
    if (
        not values
        or any(value <= 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "prompt lengths must be unique positive integers"
        )
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--prompt-lengths",
        default="256,2048,8192",
    )
    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.repetitions != 5:
        raise ValueError(
            "Stage-1 repetitions must equal 5"
        )
    if args.warmup_repetitions != 2:
        raise ValueError(
            "Stage-1 warmup repetitions must equal 2"
        )
    if args.generated_tokens != 128:
        raise ValueError(
            "Stage-1 generated tokens must equal 128"
        )
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError(
            "gpu memory utilization must be in (0, 1]"
        )
    prompt_lengths = _parse_prompt_lengths(
        args.prompt_lengths
    )
    expected_lengths = tuple(
        prompt_tokens
        for _bucket, prompt_tokens, _generated
        in context_cases()
    )
    if prompt_lengths != expected_lengths:
        raise ValueError(
            "Stage-1 prompt lengths must be 256,2048,8192"
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
        {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "run_tag": args.run_tag,
            "source_commit": args.source_commit,
            "model": str(Path(args.model).resolve()),
            "context_cases": [
                {
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                }
                for bucket, prompt_tokens, generated_tokens
                in context_cases()
            ],
            "repetitions": args.repetitions,
            "warmup_repetitions":
                args.warmup_repetitions,
            "batch_size": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "gpu_memory_utilization":
                args.gpu_memory_utilization,
            "policy_flags": {
                policy: {
                    "zero_temperature_greedy_fast_path":
                        flags[0],
                    "graph_resident_greedy_tail":
                        flags[1],
                }
                for policy, flags in POLICY_FLAGS.items()
            },
            "policy_order": {
                str(repetition): list(
                    policy_order(repetition)
                )
                for repetition in range(args.repetitions)
            },
            "correctness_sampling_points": list(
                SAMPLING_POINTS
            ),
        },
    )
    case_rows = []
    case_path = out_dir / "case_rows.jsonl"
    for repetition in range(args.repetitions):
        for (
            bucket,
            prompt_tokens,
            generated_tokens,
        ) in context_cases():
            for policy in policy_order(repetition):
                row = run_case(
                    model=args.model,
                    run_tag=args.run_tag,
                    source_commit=args.source_commit,
                    policy=policy,
                    repetition=repetition,
                    context_bucket=bucket,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    warmup_repetitions=(
                        args.warmup_repetitions
                    ),
                    gpu_memory_utilization=(
                        args.gpu_memory_utilization
                    ),
                )
                append_jsonl(case_path, row)
                case_rows.append(row)
    correctness_rows = []
    correctness_path = out_dir / "correctness_rows.jsonl"
    for (
        bucket,
        prompt_tokens,
        generated_tokens,
    ) in context_cases():
        for policy in POLICIES:
            rows = run_correctness_probe(
                model=args.model,
                run_dir=out_dir,
                run_tag=args.run_tag,
                source_commit=args.source_commit,
                policy=policy,
                context_bucket=bucket,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                gpu_memory_utilization=(
                    args.gpu_memory_utilization
                ),
            )
            for row in rows:
                append_jsonl(correctness_path, row)
                correctness_rows.append(row)
    validate_correctness_rows(
        correctness_rows,
        run_dir=out_dir,
    )
    summary = summarize_rows(
        case_rows,
        expected_repetitions=args.repetitions,
    )
    summary["correctness_row_count"] = len(
        correctness_rows
    )
    _write_json(out_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
