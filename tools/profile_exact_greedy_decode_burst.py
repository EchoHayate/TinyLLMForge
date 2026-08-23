#!/usr/bin/env python3
"""Source-bound four-arm benchmark for exact greedy decode bursts."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

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


CASE_SCHEMA_VERSION = "exact-greedy-decode-burst.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-greedy-decode-burst.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = "exact-greedy-decode-burst.summary.v1"
WORKLOAD_SCHEMA_VERSION = (
    "exact-greedy-decode-burst.workload.v1"
)
SOURCE_SCHEMA_VERSION = "exact-greedy-decode-burst.source.v1"

POLICIES = (
    "host_greedy",
    "full_step_graph_k1",
    "decode_burst_k4",
    "decode_burst_k8",
)
CONTEXT_CASES = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
CORRECTNESS_TRACE_IDENTITY = (
    "gate-only-exact-burst-correctness-v1"
)
POLICY_CONFIGS = {
    "host_greedy": {
        "enabled": False,
        "width": 1,
        "selectable": False,
        "entrypoint": "ordinary",
    },
    "full_step_graph_k1": {
        "enabled": True,
        "width": 1,
        "selectable": False,
        "entrypoint": "gate_direct",
    },
    "decode_burst_k4": {
        "enabled": True,
        "width": 4,
        "selectable": True,
        "entrypoint": "production",
    },
    "decode_burst_k8": {
        "enabled": True,
        "width": 8,
        "selectable": True,
        "entrypoint": "production",
    },
}
BURST_COUNTER_FIELDS = (
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
    "maximum_host_visible_gap_ns",
)
CAPTURE_COST_FIELDS = (
    "capture_duration_ns",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "capture_retained_static_bytes",
    "reserved_scratch_blocks",
)
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/test_profile_exact_greedy_decode_burst.py",
    "tools/exact_greedy_decode_burst_gate.py",
    "tools/test_exact_greedy_decode_burst_gate.py",
    "tools/exact_greedy_decode_burst_verify.py",
    "tools/test_exact_greedy_decode_burst_verify.py",
    "tools/run_exact_greedy_decode_burst_remote.py",
    "tools/test_run_exact_greedy_decode_burst_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)


def context_cases() -> tuple[tuple[str, int, int], ...]:
    return CONTEXT_CASES


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


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("nearest-rank input cannot be empty")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def policy_order(
    repetition: int,
    context_index: int,
) -> tuple[str, str, str, str]:
    _require_non_negative_int(repetition, "repetition")
    _require_non_negative_int(context_index, "context_index")
    rotation = repetition % len(POLICIES)
    order = POLICIES[rotation:] + POLICIES[:rotation]
    return tuple(reversed(order)) if context_index % 2 else order


def performance_identities(
    *,
    repetitions: int,
) -> tuple[tuple[int, str, str], ...]:
    _require_non_negative_int(repetitions, "repetitions")
    return tuple(
        (repetition, bucket, policy)
        for repetition in range(repetitions)
        for context_index, (bucket, _prompt, _generated)
        in enumerate(CONTEXT_CASES)
        for policy in policy_order(repetition, context_index)
    )


def correctness_identities(
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (bucket, policy, point)
        for bucket, _prompt, _generated in CONTEXT_CASES
        for policy in POLICIES
        for point in SAMPLING_POINTS
    )


def correctness_uses_burst_trace(policy: str) -> bool:
    try:
        return bool(POLICY_CONFIGS[policy]["enabled"])
    except KeyError as error:
        raise ValueError("policy is invalid") from error


def correctness_point_uses_burst_trace(
    policy: str,
    sampling_point: str,
) -> bool:
    if sampling_point not in SAMPLING_POINTS:
        raise ValueError("sampling point is invalid")
    return (
        correctness_uses_burst_trace(policy)
        and sampling_point != "prefill-final"
        and sampling_point
        not in POLICY_CONFIGS[policy].get(
            "ordinary_tail_sampling_points",
            (),
        )
    )


def correctness_trace_for_step(
    policy: str,
    *,
    emitted_total: int,
    generated_tokens: int,
) -> bool:
    if policy not in POLICY_CONFIGS:
        raise ValueError("policy is invalid")
    _require_non_negative_int(emitted_total, "emitted_total")
    _require_non_negative_int(
        generated_tokens,
        "generated_tokens",
    )
    if (
        not correctness_uses_burst_trace(policy)
        or emitted_total <= 0
        or emitted_total >= generated_tokens
    ):
        return False
    width = min(
        POLICY_CONFIGS[policy]["width"],
        generated_tokens - emitted_total,
    )
    sampled_indices = {
        _sampling_output_index(point, generated_tokens)
        for point in SAMPLING_POINTS[1:]
    }
    return any(
        emitted_total <= output_index < emitted_total + width
        for output_index in sampled_indices
    )


def runtime_environment_manifest() -> dict:
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": os.fspath(Path(sys.executable).resolve()),
    }
    try:
        import torch
    except ModuleNotFoundError:
        environment.update({
            "torch_available": False,
            "torch_version": None,
            "cuda_runtime_version": None,
            "cuda_device_name": None,
        })
        return environment
    cuda_available = bool(torch.cuda.is_available())
    environment.update({
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None
            if torch.version.cuda is None
            else str(torch.version.cuda)
        ),
        "cuda_available": cuda_available,
        "cuda_device_name": (
            str(torch.cuda.get_device_name(0))
            if cuda_available
            else None
        ),
    })
    return environment


def build_workload_manifest(
    *,
    model: str,
    run_tag: str,
    source_commit: str,
    gpu_memory_utilization: float,
    environment: dict,
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
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "model": str(Path(model).resolve()),
        "context_cases": [
            {
                "context_bucket": bucket,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            }
            for bucket, prompt_tokens, generated_tokens
            in CONTEXT_CASES
        ],
        "generated_tokens": 128,
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(
            gpu_memory_utilization
        ),
        "environment": dict(environment),
        "performance_row_count": 60,
        "correctness_row_count": 48,
        "performance_correctness_trace": False,
        "correctness_trace_identity":
            CORRECTNESS_TRACE_IDENTITY,
        "correctness_sampling_points": list(SAMPLING_POINTS),
        "policy_configs": {
            policy: dict(values)
            for policy, values in POLICY_CONFIGS.items()
        },
        "policy_order": {
            str(repetition): {
                bucket: list(
                    policy_order(repetition, context_index)
                )
                for context_index, (
                    bucket,
                    _prompt,
                    _generated,
                ) in enumerate(CONTEXT_CASES)
            }
            for repetition in range(5)
        },
    }


def _validate_reason_counts(value, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in value.items()
    ):
        raise ValueError(f"{name} is invalid")
    return dict(sorted(value.items()))


def _validate_width_histogram(value, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is invalid")
    normalized = {}
    for width, count in value.items():
        if (
            not isinstance(width, str)
            or not width.isdigit()
            or int(width) <= 0
        ):
            raise ValueError(f"{name} is invalid")
        normalized[width] = _require_non_negative_int(
            count,
            f"{name}.{width}",
        )
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def _validate_capture_receipt(
    value,
    *,
    expected_correctness_trace: bool,
) -> dict:
    required = {
        "graph_identity_sha256",
        "graph_generation",
        "capture_duration_ns",
        "allocated_delta_bytes",
        "reserved_delta_bytes",
        "retained_static_bytes",
        "scratch_block_count",
        "correctness_trace",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("burst capture receipt fields mismatch")
    _validate_digest(
        value["graph_identity_sha256"],
        "graph identity",
    )
    _require_non_negative_int(
        value["graph_generation"],
        "graph generation",
    )
    for field in required - {
        "graph_identity_sha256",
        "correctness_trace",
    }:
        _require_non_negative_int(
            value[field],
            f"capture receipt {field}",
        )
    if value["correctness_trace"] is not expected_correctness_trace:
        raise ValueError("capture correctness trace mismatch")
    return dict(value)


def _validate_burst_summary(
    summary,
    *,
    policy: str,
    correctness_trace: bool,
) -> dict:
    required = set(BURST_COUNTER_FIELDS) | {
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
        "quarantine_reason",
        "capture_receipts",
    }
    if not isinstance(summary, dict) or required - set(summary):
        raise ValueError("exact burst summary fields are missing")
    normalized = {
        field: _require_non_negative_int(
            summary[field],
            f"exact burst summary {field}",
        )
        for field in BURST_COUNTER_FIELDS
    }
    normalized["requested_width_histogram"] = (
        _validate_width_histogram(
            summary["requested_width_histogram"],
            "requested width histogram",
        )
    )
    normalized["authorized_width_histogram"] = (
        _validate_width_histogram(
            summary["authorized_width_histogram"],
            "authorized width histogram",
        )
    )
    normalized["fallback_counts"] = _validate_reason_counts(
        summary["fallback_counts"],
        "fallback counts",
    )
    quarantine_reason = summary["quarantine_reason"]
    if quarantine_reason is not None and (
        not isinstance(quarantine_reason, str)
        or not quarantine_reason
    ):
        raise ValueError("quarantine reason is invalid")
    normalized["quarantine_reason"] = quarantine_reason
    receipts = summary["capture_receipts"]
    if not isinstance(receipts, list):
        raise ValueError("capture receipts must be a list")
    normalized["capture_receipts"] = [
        _validate_capture_receipt(
            receipt,
            expected_correctness_trace=correctness_trace,
        )
        for receipt in receipts
    ]
    enabled = POLICY_CONFIGS[policy]["enabled"]
    if normalized["intermediate_token_d2h_calls"] != 0:
        raise ValueError(
            "intermediate token D2H must remain zero"
        )
    if normalized["pending_leases"] != 0:
        raise ValueError("pending burst lease inventory is nonzero")
    if normalized["failures"] or normalized["quarantines"]:
        raise ValueError("burst failure inventory is nonzero")
    if normalized["quarantine_reason"] is not None:
        raise ValueError("burst quarantine reason is present")
    if not correctness_trace and (
        normalized["sampled_logit_d2h_calls"] != 0
    ):
        raise ValueError(
            "performance path transferred sampled logits"
        )
    if (
        correctness_trace
        and enabled
        and normalized["sampled_logit_d2h_calls"]
        != POLICY_CONFIGS[policy].get(
            "correctness_sampled_logit_d2h_calls",
            len(SAMPLING_POINTS) - 1,
        )
    ):
        raise ValueError(
            "correctness sampled-logit D2H inventory mismatch"
        )
    if enabled and len(normalized["capture_receipts"]) != 1:
        raise ValueError(
            "enabled burst policy requires exactly one capture receipt"
        )
    if not enabled and (
        normalized["graph_replays"]
        or normalized["final_token_d2h_calls"]
        or normalized["capture_receipts"]
    ):
        raise ValueError("host policy reported burst activity")
    return normalized


def _case_shape(bucket: str) -> tuple[int, int]:
    try:
        return {
            name: (prompt, generated)
            for name, prompt, generated in CONTEXT_CASES
        }[bucket]
    except KeyError as error:
        raise ValueError("context bucket is invalid") from error


def validate_case_row(row) -> dict:
    required = {
        "schema_version",
        "run_tag",
        "source_commit",
        "policy",
        "selectable",
        "burst_width",
        "repetition",
        "context_bucket",
        "prompt_tokens",
        "generated_tokens",
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
        *CAPTURE_COST_FIELDS,
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
    config = POLICY_CONFIGS[policy]
    if (
        row["selectable"] is not config["selectable"]
        or row["burst_width"] != config["width"]
    ):
        raise ValueError("case policy metadata mismatch")
    _require_non_negative_int(row["repetition"], "repetition")
    expected_shape = _case_shape(row["context_bucket"])
    if (
        row["prompt_tokens"],
        row["generated_tokens"],
    ) != expected_shape:
        raise ValueError("context shape does not match bucket")
    generated_tokens = row["generated_tokens"]
    output_ids = row["output_token_ids"]
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != generated_tokens
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in output_ids
        )
    ):
        raise ValueError("output token inventory is invalid")
    _validate_digest(
        row["output_text_sha256"],
        "output text digest",
    )
    if row["correctness_trace"] is not False:
        raise ValueError(
            "performance row cannot enable correctness tracing"
        )
    expected_decode_tokens = generated_tokens - 1
    samples = row["amortized_tpot_samples_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != expected_decode_tokens
    ):
        raise ValueError("amortized TPOT inventory mismatch")
    for index, value in enumerate(samples):
        _require_finite_non_negative(
            value,
            f"amortized_tpot_samples_ns[{index}]",
        )
    expected_statistics = {
        "amortized_tpot_median_ns": statistics.median(samples),
        "amortized_tpot_p95_ns": _nearest_rank(samples, 0.95),
        "amortized_tpot_p99_ns": _nearest_rank(samples, 0.99),
    }
    for field, expected in expected_statistics.items():
        actual = _require_finite_non_negative(row[field], field)
        if actual != expected:
            raise ValueError(f"{field} does not match samples")
    for field in (
        "ttft_ns",
        "e2e_ns",
        "output_tokens_per_second",
        "maximum_host_visible_burst_gap_ns",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        *CAPTURE_COST_FIELDS,
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
        for index, value in enumerate(values):
            _require_finite_non_negative(
                value,
                f"{field}[{index}]",
            )
    gaps = row["host_visible_burst_gaps_ns"]
    expected_max = max(gaps, default=0)
    if row["maximum_host_visible_burst_gap_ns"] != expected_max:
        raise ValueError("maximum host-visible gap mismatch")
    summary = _validate_burst_summary(
        row["exact_greedy_decode_burst_summary"],
        policy=policy,
        correctness_trace=False,
    )
    if (
        summary["maximum_host_visible_gap_ns"]
        != row["maximum_host_visible_burst_gap_ns"]
    ):
        raise ValueError(
            "summary host-visible gap does not match measured request"
        )
    expected_decode_profile_steps = (
        (
            summary["attempts"]
            + (
                expected_decode_tokens
                - summary["committed_tokens"]
            )
            if config.get(
                "profile_ordinary_tail_after_full_bursts",
                False,
            )
            else summary["commits"]
        )
        if config["enabled"]
        else expected_decode_tokens
    )
    if (
        len(row["decode_host_ns"])
        != expected_decode_profile_steps
        or len(row["decode_cuda_ns"])
        != expected_decode_profile_steps
    ):
        raise ValueError(
            "decode profile inventory mismatch: "
            f"policy={policy}, "
            f"expected_steps={expected_decode_profile_steps}, "
            f"host_steps={len(row['decode_host_ns'])}, "
            f"cuda_steps={len(row['decode_cuda_ns'])}, "
            f"commits={summary['commits']}, "
            f"attempts={summary['attempts']}, "
            f"fallbacks={summary['fallback_counts']}"
        )
    receipt = (
        summary["capture_receipts"][0]
        if summary["capture_receipts"]
        else None
    )
    expected_costs = {
        "capture_duration_ns": (
            receipt["capture_duration_ns"] if receipt else 0
        ),
        "capture_allocated_delta_bytes": (
            receipt["allocated_delta_bytes"] if receipt else 0
        ),
        "capture_reserved_delta_bytes": (
            receipt["reserved_delta_bytes"] if receipt else 0
        ),
        "capture_retained_static_bytes": (
            receipt["retained_static_bytes"] if receipt else 0
        ),
        "reserved_scratch_blocks": (
            receipt["scratch_block_count"] if receipt else 0
        ),
    }
    for field, expected in expected_costs.items():
        if row[field] != expected:
            raise ValueError(f"capture cost mismatch: {field}")
    normalized = dict(row)
    normalized["exact_greedy_decode_burst_summary"] = summary
    return normalized


def summarize_rows(
    rows: list[dict],
    *,
    expected_repetitions: int = 5,
) -> dict:
    _require_non_negative_int(
        expected_repetitions,
        "expected_repetitions",
    )
    validated = [validate_case_row(row) for row in rows]
    expected = set(
        performance_identities(
            repetitions=expected_repetitions
        )
    )
    identities = {}
    for row in validated:
        identity = (
            row["repetition"],
            row["context_bucket"],
            row["policy"],
        )
        if identity in identities:
            raise ValueError(
                f"duplicate case identity: {identity}"
            )
        identities[identity] = row
    if set(identities) != expected:
        raise ValueError("case row inventory is incomplete")
    run_tags = {row["run_tag"] for row in validated}
    commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "performance rows do not share source identity"
        )
    comparisons = []
    for repetition in range(expected_repetitions):
        for bucket, _prompt, _generated in CONTEXT_CASES:
            group = [
                identities[(repetition, bucket, policy)]
                for policy in POLICIES
            ]
            baseline = group[0]
            for candidate in group[1:]:
                if (
                    candidate["output_token_ids"]
                    != baseline["output_token_ids"]
                    or candidate["output_text_sha256"]
                    != baseline["output_text_sha256"]
                ):
                    raise ValueError(
                        "output mismatch in four-arm comparison"
                    )
            comparisons.append({
                "repetition": repetition,
                "context_bucket": bucket,
                "amortized_tpot_median_ns": {
                    row["policy"]:
                        row["amortized_tpot_median_ns"]
                    for row in group
                },
                "amortized_tpot_p95_ns": {
                    row["policy"]:
                        row["amortized_tpot_p95_ns"]
                    for row in group
                },
            })
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(run_tags)),
        "source_commit": next(iter(commits)),
        "row_count": len(validated),
        "comparison_set_count": len(comparisons),
        "all_outputs_exact": True,
        "comparisons": comparisons,
    }


def validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
) -> list[dict]:
    expected = set(correctness_identities())
    identities = {}
    validated = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("correctness row must be an object")
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
        ) != _case_shape(bucket):
            raise ValueError("correctness context shape mismatch")
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != row["generated_tokens"]
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
        if (
            not isinstance(row.get("run_tag"), str)
            or not row["run_tag"]
        ):
            raise ValueError("correctness run tag is invalid")
        if (
            row.get("correctness_trace") is not True
            or row.get("trace_identity")
            != CORRECTNESS_TRACE_IDENTITY
        ):
            raise ValueError(
                "correctness row must use gate-only correctness trace"
            )
        point = row["sampling_point"]
        burst_decode_sample = correctness_point_uses_burst_trace(
            policy,
            point,
        )
        graph_identity = row.get(
            "trace_graph_identity_sha256"
        )
        selected_ordinal = row.get(
            "selected_replay_ordinal"
        )
        sampled_d2h_calls = row.get(
            "sampled_logit_d2h_calls"
        )
        if burst_decode_sample:
            _validate_digest(
                graph_identity,
                "correctness graph identity",
            )
            expected_ordinal = (
                _sampling_output_index(
                    point,
                    row["generated_tokens"],
                )
                - 1
            )
            if not POLICY_CONFIGS[policy].get(
                "epoch_relative_sampling",
                False,
            ):
                expected_ordinal %= POLICY_CONFIGS[policy]["width"]
            if selected_ordinal != expected_ordinal:
                raise ValueError(
                    "selected replay ordinal mismatch"
                )
            if sampled_d2h_calls != 1:
                raise ValueError(
                    "sampled-logit D2H inventory mismatch"
                )
        elif (
            graph_identity is not None
            or selected_ordinal is not None
            or sampled_d2h_calls != 0
        ):
            raise ValueError(
                "non-burst sample reported sampled-logit D2H"
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
            raise ValueError("correctness logits shape is invalid")
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
        burst_summary = _validate_burst_summary(
                row.get(
                    "exact_greedy_decode_burst_summary"
                ),
                policy=policy,
                correctness_trace=correctness_uses_burst_trace(
                    policy
                ),
            )
        if burst_decode_sample:
            receipts = burst_summary["capture_receipts"]
            if (
                len(receipts) != 1
                or receipts[0]["graph_identity_sha256"]
                != graph_identity
            ):
                raise ValueError(
                    "correctness graph identity does not match "
                    "capture receipt"
                )
        normalized["exact_greedy_decode_burst_summary"] = (
            burst_summary
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


def _counter_delta(before: dict, after: dict) -> dict:
    result = {}
    for field in BURST_COUNTER_FIELDS:
        before_value = _require_non_negative_int(
            before.get(field, 0),
            f"before.{field}",
        )
        after_value = _require_non_negative_int(
            after.get(field, 0),
            f"after.{field}",
        )
        if field in {
            "pending_leases",
            "maximum_host_visible_gap_ns",
        }:
            result[field] = after_value
        else:
            if after_value < before_value:
                raise RuntimeError(
                    f"exact burst counter decreased: {field}"
                )
            result[field] = after_value - before_value
    for field in (
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
    ):
        before_map = before.get(field, {})
        after_map = after.get(field, {})
        result[field] = {}
        for key in sorted(
            set(before_map) | set(after_map),
            key=str,
        ):
            difference = int(after_map.get(key, 0)) - int(
                before_map.get(key, 0)
            )
            if difference < 0:
                raise RuntimeError(
                    f"exact burst map counter decreased: {field}"
                )
            if difference:
                result[field][str(key)] = difference
    result["quarantine_reason"] = after.get(
        "quarantine_reason"
    )
    result["capture_receipts"] = list(
        after.get("capture_receipts", ())
    )
    return result


def _combined_summary(
    llm,
    before: tuple[dict, dict],
    *,
    correctness_trace: bool = False,
) -> dict:
    runner_after = (
        llm.model_runner.exact_greedy_decode_burst_summary()
    )
    scheduler_after = (
        llm.scheduler.exact_greedy_decode_burst_summary()
    )
    runner = _counter_delta(before[0], runner_after)
    scheduler = _counter_delta(before[1], scheduler_after)
    return {
        "attempts": scheduler["attempts"],
        "acceptances": scheduler["acceptances"],
        "target_model_forwards": runner[
            "target_model_forwards"
        ],
        "graph_replays": runner["graph_replays"],
        "intermediate_token_d2h_calls": runner[
            "intermediate_token_d2h_calls"
        ],
        "final_token_d2h_calls": runner[
            "final_token_d2h_calls"
        ],
        "final_token_d2h_bytes": runner[
            "final_token_d2h_bytes"
        ],
        "sampled_logit_d2h_calls": runner[
            "sampled_logit_d2h_calls"
        ],
        "output_budget_clipped": scheduler[
            "output_budget_clipped"
        ],
        "block_boundary_clipped": scheduler[
            "block_boundary_clipped"
        ],
        "commits": scheduler["commits"],
        "committed_tokens": scheduler["committed_tokens"],
        "failures": runner["failures"] + scheduler["failures"],
        "quarantines": runner["quarantines"],
        "pending_leases": scheduler["pending_leases"],
        "maximum_host_visible_gap_ns": scheduler[
            "maximum_host_visible_gap_ns"
        ],
        "requested_width_histogram": scheduler[
            "requested_width_histogram"
        ],
        "authorized_width_histogram": scheduler[
            "authorized_width_histogram"
        ],
        "fallback_counts": scheduler["fallback_counts"],
        "quarantine_reason": runner["quarantine_reason"],
        "capture_receipts": [
            receipt
            for receipt in runner["capture_receipts"]
            if bool(receipt.get("correctness_trace"))
            is correctness_trace
        ],
    }


def _construct_llm(
    *,
    model: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
    policy: str,
):
    from tinyvllm import LLM

    config = POLICY_CONFIGS[policy]
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
        exact_greedy_decode_burst=config["enabled"],
        exact_greedy_decode_burst_continuation=bool(
            config.get("continuation", False)
        ),
        exact_greedy_decode_burst_tokens=max(2, config["width"]),
    )


def _runner_summaries(llm) -> tuple[dict, dict]:
    return (
        llm.model_runner.exact_greedy_decode_burst_summary(),
        llm.scheduler.exact_greedy_decode_burst_summary(),
    )


def _step_kwargs(
    policy: str,
    *,
    correctness_trace: bool,
) -> dict:
    config = POLICY_CONFIGS[policy]
    return {
        "completion_only": True,
        "exact_burst_gate_width": (
            config["width"]
            if config["entrypoint"] == "gate_direct"
            else None
        ),
        "exact_burst_correctness_trace": correctness_trace,
    }


def _run_request(
    llm,
    *,
    prompt: list[int],
    generated_tokens: int,
    policy: str,
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
    amortized_tpot = []
    burst_gaps = []
    final_outputs = None
    while not llm.is_finished():
        step_started_ns = time.perf_counter_ns()
        outputs, _num_tokens = llm.step(
            **_step_kwargs(
                policy,
                correctness_trace=False,
            )
        )
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
                per_token = (
                    step_finished_ns - step_started_ns
                ) / emitted
                amortized_tpot.extend([per_token] * emitted)
                gap = observation[
                    "exact_greedy_decode_burst_host_visible_gap_ns"
                ]
                if gap:
                    burst_gaps.append(int(gap))
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
    if len(amortized_tpot) != generated_tokens - 1:
        raise RuntimeError("amortized TPOT inventory mismatch")
    decode_host_ns = []
    decode_cuda_ns = []
    if profile_label is not None:
        profile = (
            llm.finalize_decode_internal_profile(
                already_synchronized=True,
                timeout_s=60.0,
            )
        )
        rank_rows = profile.get("ranks", ())
        if (
            profile.get("rank_inventory") != [0]
            or len(rank_rows) != 1
        ):
            raise RuntimeError(
                "Stage-1 worker requires tensor parallel size one"
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
        "amortized_tpot_samples_ns": amortized_tpot,
        "decode_host_ns": decode_host_ns,
        "decode_cuda_ns": decode_cuda_ns,
        "host_visible_burst_gaps_ns": burst_gaps,
    }


def _capture_cost(summary: dict) -> dict:
    receipt = (
        summary["capture_receipts"][0]
        if summary["capture_receipts"]
        else None
    )
    return {
        "capture_duration_ns": (
            receipt["capture_duration_ns"] if receipt else 0
        ),
        "capture_allocated_delta_bytes": (
            receipt["allocated_delta_bytes"] if receipt else 0
        ),
        "capture_reserved_delta_bytes": (
            receipt["reserved_delta_bytes"] if receipt else 0
        ),
        "capture_retained_static_bytes": (
            receipt["retained_static_bytes"] if receipt else 0
        ),
        "reserved_scratch_blocks": (
            receipt["scratch_block_count"] if receipt else 0
        ),
    }


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
                policy=policy,
                profile_label=None,
            )
            llm.clear_reusable_prefix_cache()
        before = _runner_summaries(llm)
        llm.reset_peak_memory_stats(timeout_s=60.0)
        measured = _run_request(
            llm,
            prompt=_make_prompt(
                prompt_tokens,
                offset=repetition * 10_007,
            ),
            generated_tokens=generated_tokens,
            policy=policy,
            profile_label=(
                f"{run_tag}/{context_bucket}/"
                f"r{repetition}/{policy}"
            ),
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
            "selectable": POLICY_CONFIGS[policy]["selectable"],
            "burst_width": POLICY_CONFIGS[policy]["width"],
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
            "amortized_tpot_samples_ns": samples,
            "amortized_tpot_median_ns":
                statistics.median(samples),
            "amortized_tpot_p95_ns":
                _nearest_rank(samples, 0.95),
            "amortized_tpot_p99_ns":
                _nearest_rank(samples, 0.99),
            "decode_host_ns": measured["decode_host_ns"],
            "decode_cuda_ns": measured["decode_cuda_ns"],
            "output_tokens_per_second": (
                generated_tokens / e2e_seconds
            ),
            "host_visible_burst_gaps_ns": measured[
                "host_visible_burst_gaps_ns"
            ],
            "maximum_host_visible_burst_gap_ns": max(
                measured["host_visible_burst_gaps_ns"],
                default=0,
            ),
            **memory,
            **_capture_cost(summary),
            "correctness_trace": False,
            "exact_greedy_decode_burst_summary": summary,
        }
        if "split_phase_inventory" in measured:
            row["split_phase_inventory"] = measured[
                "split_phase_inventory"
            ]
        return validate_case_row(row)
    finally:
        llm.exit()


def _sampling_output_index(point: str, generated_tokens: int) -> int:
    return {
        "prefill-final": 0,
        "decode-first": 1,
        "decode-middle": generated_tokens // 2,
        "decode-final": generated_tokens - 1,
    }[point]


def _sampled_local_ordinals(policy: str) -> tuple[int, ...]:
    width = POLICY_CONFIGS[policy]["width"]
    if not POLICY_CONFIGS[policy]["enabled"]:
        return ()
    decode_ordinals = (0, 63, 126)
    return tuple(sorted({
        ordinal % width for ordinal in decode_ordinals
    }))


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
        if correctness_uses_burst_trace(policy):
            llm.model_runner.call(
                "capture_exact_greedy_decode_burst_correctness_graph",
                _sampled_local_ordinals(policy),
            )
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        before = _runner_summaries(llm)
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
        emitted_total = 0
        while not llm.is_finished():
            trace_this_step = correctness_trace_for_step(
                policy,
                emitted_total=emitted_total,
                generated_tokens=generated_tokens,
            )
            outputs, _num_tokens = llm.step(
                **_step_kwargs(
                    policy,
                    correctness_trace=trace_this_step,
                )
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
                    "selected_replay_ordinal": None,
                    "sampled_logit_d2h_calls": 0,
                }
                if correctness_uses_burst_trace(policy):
                    llm.enable_step_logits_authority_recording(
                        False,
                        timeout_s=60.0,
                    )
            elif correctness_uses_burst_trace(policy):
                sampled = observation.get(
                    "exact_greedy_decode_burst_sampled_logits",
                    (),
                )
                for local_ordinal, values in sampled:
                    output_index = (
                        1 + int(local_ordinal)
                        if POLICY_CONFIGS[policy].get(
                            "epoch_relative_sampling",
                            False,
                        )
                        else emitted_total + int(local_ordinal)
                    )
                    for point in SAMPLING_POINTS[1:]:
                        if output_index == _sampling_output_index(
                            point,
                            generated_tokens,
                        ):
                            captured[point] = {
                                "logits": values,
                                "trace_graph_identity_sha256":
                                    observation[
                                        "exact_greedy_decode_burst_graph_identity_sha256"
                                    ],
                                "selected_replay_ordinal": int(
                                    local_ordinal
                                ),
                                "sampled_logit_d2h_calls": int(
                                    observation[
                                        "exact_greedy_decode_burst_sampled_logit_d2h_calls"
                                    ]
                                ),
                            }
            else:
                output_index = emitted_total
                if output_index in {
                    _sampling_output_index(point, generated_tokens)
                    for point in SAMPLING_POINTS[1:]
                }:
                    logits = (
                        llm.read_step_logits_authority()
                        .detach()
                        .to(dtype=__import__("torch").float32)
                        .contiguous()
                    )
                    for point in SAMPLING_POINTS[1:]:
                        if output_index == _sampling_output_index(
                            point,
                            generated_tokens,
                        ):
                            captured[point] = {
                                "logits": logits,
                                "trace_graph_identity_sha256": None,
                                "selected_replay_ordinal": None,
                                "sampled_logit_d2h_calls": 0,
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
        output_text_sha256 = sha256_text(
            llm.tokenizer.decode(output_ids)
        )
        summary = _combined_summary(
            llm,
            before,
            correctness_trace=correctness_uses_burst_trace(
                policy
            ),
        )
        rows = []
        for point in SAMPLING_POINTS:
            sample = captured[point]
            logits = sample["logits"]
            if hasattr(logits, "view"):
                shape = [int(value) for value in logits.shape]
                values = logits.view(-1).tolist()
            else:
                values = list(logits)
                shape = [1, len(values)]
            sidecar = write_float32_sidecar(
                run_dir,
                f"logits/{context_bucket}-{policy}-{point}.f32",
                values,
            )
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
                "correctness_trace": True,
                "trace_identity": CORRECTNESS_TRACE_IDENTITY,
                "trace_graph_identity_sha256": sample[
                    "trace_graph_identity_sha256"
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
        raise ValueError("Stage-1 repetitions must equal 5")
    if args.warmup_repetitions != 2:
        raise ValueError(
            "Stage-1 warmup repetitions must equal 2"
        )
    if args.generated_tokens != 128:
        raise ValueError(
            "Stage-1 generated tokens must equal 128"
        )
    prompt_lengths = tuple(
        int(item.strip())
        for item in args.prompt_lengths.split(",")
        if item.strip()
    )
    if prompt_lengths != (256, 2048, 8192):
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
    workload = build_workload_manifest(
        model=args.model,
        run_tag=args.run_tag,
        source_commit=args.source_commit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        environment=runtime_environment_manifest(),
    )
    _write_json(out_dir / "workload_manifest.json", workload)
    case_rows = []
    case_path = out_dir / "case_rows.jsonl"
    for repetition in range(5):
        for context_index, (
            bucket,
            prompt_tokens,
            generated_tokens,
        ) in enumerate(CONTEXT_CASES):
            for policy in policy_order(
                repetition,
                context_index,
            ):
                row = run_case(
                    model=args.model,
                    run_tag=args.run_tag,
                    source_commit=args.source_commit,
                    policy=policy,
                    repetition=repetition,
                    context_bucket=bucket,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    warmup_repetitions=2,
                    gpu_memory_utilization=(
                        args.gpu_memory_utilization
                    ),
                )
                append_jsonl(case_path, row)
                case_rows.append(row)
    correctness_rows = []
    correctness_path = out_dir / "correctness_rows.jsonl"
    for bucket, prompt_tokens, generated_tokens in CONTEXT_CASES:
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
        expected_repetitions=5,
    )
    summary["correctness_row_count"] = len(correctness_rows)
    _write_json(out_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
