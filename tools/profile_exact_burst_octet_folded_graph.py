#!/usr/bin/env python3
"""Source-bound one-token versus octet-folded exact-burst profiler."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

from tools import profile_context_gated_elastic_exact_burst as _base
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


CASE_SCHEMA_VERSION = "exact-burst-octet-folded.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-burst-octet-folded.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = "exact-burst-octet-folded.summary.v1"
WORKLOAD_SCHEMA_VERSION = "exact-burst-octet-folded.workload.v1"
SOURCE_SCHEMA_VERSION = "exact-burst-octet-folded.source.v1"

POLICIES = ("one_token_graph", "octet_folded_graph")
POLICY_CONFIGS = {
    "one_token_graph": {
        "exact_greedy_decode_burst_octet_folded_graph": False,
    },
    "octet_folded_graph": {
        "exact_greedy_decode_burst_octet_folded_graph": True,
    },
}
CONTEXT_LENGTHS = (256, 2048, 8192)
GENERATED_TOKENS = 128
REPETITIONS = 5
WARMUP_REPETITIONS = 2
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
CORRECTNESS_BLOCK_SIZE = 256
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_burst_octet_folded_graph.py",
    "tools/test_profile_exact_burst_octet_folded_graph.py",
    "tools/exact_burst_octet_folded_graph_ceiling.py",
    "tools/test_exact_burst_octet_folded_graph_ceiling.py",
)

_CASE_REQUIRED_FIELDS = {
    "schema_version",
    "run_tag",
    "source_commit",
    "source_patch_sha256",
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
    "one_token_graph_identity_sha256",
    "folded_graph_identity_sha256",
    "logical_forwards",
    "logical_replays",
    "one_token_cuda_graph_launches",
    "folded_cuda_graph_launches",
    "token_d2h_calls",
    "token_d2h_bytes",
    "capture_duration_ns",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "capture_retained_static_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "ttft_ns",
    "e2e_ns",
    "tpot_samples_ns",
    "tpot_median_ns",
    "tpot_p95_ns",
    "tpot_p99_ns",
    "output_tokens_per_second",
    "host_visible_burst_gaps_ns",
    "maximum_host_visible_burst_gap_ns",
    "fallback_count",
    "rollback_count",
    "quarantine_reason",
}
_NON_NEGATIVE_INTEGER_FIELDS = {
    "repetition",
    "order_position",
    "context_length",
    "prompt_tokens",
    "generated_tokens",
    "tensor_parallel_size",
    "max_num_seqs",
    "logical_forwards",
    "logical_replays",
    "one_token_cuda_graph_launches",
    "folded_cuda_graph_launches",
    "token_d2h_calls",
    "token_d2h_bytes",
    "capture_duration_ns",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "capture_retained_static_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "ttft_ns",
    "e2e_ns",
    "maximum_host_visible_burst_gap_ns",
    "fallback_count",
    "rollback_count",
}
_FINITE_NON_NEGATIVE_FIELDS = {
    "tpot_median_ns",
    "tpot_p95_ns",
    "tpot_p99_ns",
    "output_tokens_per_second",
}
_CORRECTNESS_REQUIRED_FIELDS = {
    "schema_version",
    "run_tag",
    "source_commit",
    "source_patch_sha256",
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
    "one_token_graph_identity_sha256",
    "folded_graph_identity_sha256",
    "logical_forwards",
    "logical_replays",
    "one_token_cuda_graph_launches",
    "folded_cuda_graph_launches",
    "token_d2h_calls",
    "token_d2h_bytes",
    "fallback_count",
    "rollback_count",
    "quarantine_reason",
    "correctness_trace",
}
_COUNTER_FIELDS = (
    "target_model_forwards",
    "graph_replays",
    "one_token_cuda_graph_launches",
    "folded_cuda_graph_launches",
    "final_token_d2h_calls",
    "final_token_d2h_bytes",
    "failures",
    "quarantines",
    "lease_local_delta_journal_rollbacks",
    "lease_local_delta_journal_one_phase_rollbacks",
)
_MAP_FIELDS = ("fallback_counts", "folded_fallback_counts")


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
    optional: bool = False,
):
    if optional and value is None:
        return None
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
    ordered = sorted(
        _require_finite_non_negative(value, "TPOT sample")
        for value in values
    )
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


def correctness_identities(
) -> tuple[tuple[int, str, str], ...]:
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
    source_patch_sha256: str,
    gpu_memory_utilization: float,
    environment: dict,
    repetitions: int = REPETITIONS,
    warmup_repetitions: int = WARMUP_REPETITIONS,
) -> dict:
    if not isinstance(model, str) or not model:
        raise ValueError("model is invalid")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("device must identify a CUDA device")
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run tag is invalid")
    _validate_digest(source_commit, "source commit", lengths=(40, 64))
    _validate_digest(source_patch_sha256, "source patch sha256")
    _require_finite_non_negative(
        gpu_memory_utilization,
        "gpu memory utilization",
    )
    if not 0.0 < float(gpu_memory_utilization) <= 1.0:
        raise ValueError("gpu memory utilization must be in (0, 1]")
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")
    _require_non_negative_int(repetitions, "repetitions")
    _require_non_negative_int(
        warmup_repetitions,
        "warmup repetitions",
    )
    if (
        repetitions != REPETITIONS
        or warmup_repetitions != WARMUP_REPETITIONS
    ):
        raise ValueError(
            "ceiling repetitions and warmups are frozen"
        )
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "model": model,
        "device": device,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_patch_sha256": source_patch_sha256,
        "contexts": list(CONTEXT_LENGTHS),
        "policies": list(POLICIES),
        "generated_tokens": GENERATED_TOKENS,
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "performance_row_count": (
            repetitions * len(CONTEXT_LENGTHS) * len(POLICIES)
        ),
        "correctness_row_count": len(correctness_identities()),
        "execution_order": [
            list(policy_order(0, index))
            for index in range(len(CONTEXT_LENGTHS))
        ],
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "completion_only": True,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "environment": environment,
    }


def validate_case_row(row) -> dict:
    if not isinstance(row, dict):
        raise ValueError("case row must be an object")
    if set(row) != _CASE_REQUIRED_FIELDS:
        raise ValueError("case row fields are invalid")
    if row["schema_version"] != CASE_SCHEMA_VERSION:
        raise ValueError("case schema version is invalid")
    if row["policy"] not in POLICIES:
        raise ValueError("policy is invalid")
    if row["context_length"] not in CONTEXT_LENGTHS:
        raise ValueError("context length is invalid")
    expected_order_position = policy_order(
        row["repetition"],
        CONTEXT_LENGTHS.index(row["context_length"]),
    ).index(row["policy"])
    if row["order_position"] != expected_order_position:
        raise ValueError("execution order position is invalid")
    _validate_digest(
        row["source_commit"],
        "source commit",
        lengths=(40, 64),
    )
    _validate_digest(
        row["source_patch_sha256"],
        "source patch sha256",
    )
    _validate_digest(row["prompt_sha256"], "prompt sha256")
    _validate_digest(
        row["output_text_sha256"],
        "output text sha256",
    )
    _validate_digest(
        row["one_token_graph_identity_sha256"],
        "one-token graph identity",
    )
    _validate_digest(
        row["folded_graph_identity_sha256"],
        "folded graph identity",
        optional=True,
    )
    for field in _NON_NEGATIVE_INTEGER_FIELDS:
        _require_non_negative_int(row[field], field)
    for field in _FINITE_NON_NEGATIVE_FIELDS:
        _require_finite_non_negative(row[field], field)
    for field in ("temperature",):
        _require_finite_non_negative(row[field], field)
    for field in (
        "ignore_eos",
        "completion_only",
    ):
        if not isinstance(row[field], bool):
            raise ValueError(f"{field} must be a bool")
    if (
        not isinstance(row["output_token_ids"], list)
        or len(row["output_token_ids"]) != GENERATED_TOKENS
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in row["output_token_ids"]
        )
    ):
        raise ValueError("output token inventory is invalid")
    samples = row["tpot_samples_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != GENERATED_TOKENS - 1
    ):
        raise ValueError("TPOT sample inventory is invalid")
    for value in samples:
        _require_finite_non_negative(value, "TPOT sample")
    gaps = row["host_visible_burst_gaps_ns"]
    if not isinstance(gaps, list):
        raise ValueError("host-visible gap inventory is invalid")
    for value in gaps:
        _require_non_negative_int(value, "host-visible gap")
    _validate_runtime_inventory(row)
    return row


def _validate_runtime_inventory(row: dict) -> None:
    if row["logical_forwards"] != row["logical_replays"]:
        raise ValueError("logical forward/replay counts differ")
    if row["logical_replays"] != GENERATED_TOKENS - 1:
        raise ValueError("logical replay count is invalid")
    expected_bursts = math.ceil((GENERATED_TOKENS - 1) / 8)
    if row["token_d2h_calls"] != expected_bursts:
        raise ValueError("token D2H count is invalid")
    if row["token_d2h_bytes"] != (
        (GENERATED_TOKENS - 1) * 8
    ):
        raise ValueError("token D2H byte count is invalid")
    gaps = row.get("host_visible_burst_gaps_ns")
    if gaps is not None and len(gaps) != expected_bursts:
        raise ValueError("host-visible gap inventory is invalid")
    if row["policy"] == "one_token_graph":
        if (
            row["one_token_cuda_graph_launches"]
            != GENERATED_TOKENS - 1
            or row["folded_cuda_graph_launches"] != 0
            or row["folded_graph_identity_sha256"] is not None
        ):
            raise ValueError("physical launch inventory is invalid")
    elif (
        row["one_token_cuda_graph_launches"]
        != (GENERATED_TOKENS - 1) % 8
        or row["folded_cuda_graph_launches"]
        != (GENERATED_TOKENS - 1) // 8
        or row["folded_graph_identity_sha256"] is None
    ):
        raise ValueError("physical launch inventory is invalid")
    if row["fallback_count"] or row["rollback_count"]:
        raise ValueError("runtime anomaly count is non-zero")
    if row["quarantine_reason"] is not None:
        raise ValueError("runtime is quarantined")


def validate_correctness_row(
    row,
    *,
    run_dir: Path | None = None,
) -> dict:
    if not isinstance(row, dict):
        raise ValueError("correctness row must be an object")
    if set(row) != _CORRECTNESS_REQUIRED_FIELDS:
        raise ValueError("correctness row fields are invalid")
    if row["schema_version"] != CORRECTNESS_SCHEMA_VERSION:
        raise ValueError("correctness row schema is invalid")
    if row["policy"] not in POLICIES:
        raise ValueError("correctness policy is invalid")
    if row["context_length"] not in CONTEXT_LENGTHS:
        raise ValueError("correctness context is invalid")
    if row["sampling_point"] not in SAMPLING_POINTS:
        raise ValueError("correctness sampling point is invalid")
    _validate_digest(
        row["source_commit"],
        "correctness source commit",
        lengths=(40, 64),
    )
    _validate_digest(
        row["source_patch_sha256"],
        "correctness source patch sha256",
    )
    _validate_digest(
        row["prompt_sha256"],
        "correctness prompt sha256",
    )
    _validate_digest(
        row["output_text_sha256"],
        "correctness output text sha256",
    )
    _validate_digest(
        row["logits_sha256"],
        "correctness logits sha256",
    )
    _validate_digest(
        row["one_token_graph_identity_sha256"],
        "correctness one-token graph identity",
    )
    _validate_digest(
        row["folded_graph_identity_sha256"],
        "correctness folded graph identity",
        optional=True,
    )
    for field in (
        "context_length",
        "generated_tokens",
        "argmax_token_id",
        "logits_element_count",
        "logits_byte_length",
        "logical_forwards",
        "logical_replays",
        "one_token_cuda_graph_launches",
        "folded_cuda_graph_launches",
        "token_d2h_calls",
        "token_d2h_bytes",
        "fallback_count",
        "rollback_count",
    ):
        _require_non_negative_int(row[field], field)
    output_ids = row["output_token_ids"]
    if (
        row["generated_tokens"] != GENERATED_TOKENS
        or not isinstance(output_ids, list)
        or len(output_ids) != GENERATED_TOKENS
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in output_ids
        )
    ):
        raise ValueError("correctness output inventory is invalid")
    shape = row["logits_shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
        or math.prod(shape) != row["logits_element_count"]
    ):
        raise ValueError("correctness logits shape is invalid")
    if row["logits_byte_length"] != (
        row["logits_element_count"] * 4
    ):
        raise ValueError("correctness logits byte length is invalid")
    if row["correctness_trace"] is not True:
        raise ValueError("correctness trace must be enabled")
    _validate_runtime_inventory(row)
    if run_dir is not None:
        values = read_float32_sidecar(
            run_dir,
            path=row["logits_path"],
            expected_element_count=row["logits_element_count"],
            expected_byte_length=row["logits_byte_length"],
            expected_sha256=row["logits_sha256"],
        )
        expected_argmax = max(
            range(len(values)),
            key=values.__getitem__,
        )
        if row["argmax_token_id"] != expected_argmax:
            raise ValueError("correctness argmax mismatch")
    return row


def validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path | None = None,
) -> list[dict]:
    validated = [
        validate_correctness_row(row, run_dir=run_dir)
        for row in rows
    ]
    identities = [
        (
            row["context_length"],
            row["policy"],
            row["sampling_point"],
        )
        for row in validated
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate correctness row identity")
    if set(identities) != set(correctness_identities()):
        raise ValueError("correctness row inventory is incomplete")
    if (
        len({row["run_tag"] for row in validated}) != 1
        or len({row["source_commit"] for row in validated}) != 1
        or len({
            row["source_patch_sha256"] for row in validated
        }) != 1
    ):
        raise ValueError(
            "correctness rows do not share source identity"
        )
    indexed = {
        (
            row["context_length"],
            row["sampling_point"],
            row["policy"],
        ): row
        for row in validated
    }
    for context_length in CONTEXT_LENGTHS:
        for point in SAMPLING_POINTS:
            control = indexed[
                (context_length, point, "one_token_graph")
            ]
            candidate = indexed[
                (context_length, point, "octet_folded_graph")
            ]
            if (
                control["prompt_sha256"]
                != candidate["prompt_sha256"]
                or control["output_token_ids"]
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


def summarize_rows(
    rows: list[dict],
    *,
    expected_repetitions: int,
) -> dict:
    validated = [validate_case_row(row) for row in rows]
    expected = set(
        performance_identities(repetitions=expected_repetitions)
    )
    observed = {
        (
            row["repetition"],
            row["context_length"],
            row["policy"],
        )
        for row in validated
    }
    if len(observed) != len(validated):
        raise ValueError("duplicate performance row identity")
    if observed != expected:
        raise ValueError("performance row inventory is incomplete")
    by_policy = {
        policy: [
            row for row in validated if row["policy"] == policy
        ]
        for policy in POLICIES
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "performance_row_count": len(validated),
        "all_outputs_exact": all(
            control["output_token_ids"] == candidate["output_token_ids"]
            and control["output_text_sha256"]
            == candidate["output_text_sha256"]
            for control, candidate in zip(
                sorted(
                    by_policy["one_token_graph"],
                    key=lambda row: (
                        row["repetition"],
                        row["context_length"],
                    ),
                ),
                sorted(
                    by_policy["octet_folded_graph"],
                    key=lambda row: (
                        row["repetition"],
                        row["context_length"],
                    ),
                ),
            )
        ),
    }


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
        kvcache_block_size=CORRECTNESS_BLOCK_SIZE,
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
        exact_greedy_decode_burst_elastic_k16=False,
        exact_greedy_decode_burst_octet_folded_graph=(
            POLICY_CONFIGS[policy][
                "exact_greedy_decode_burst_octet_folded_graph"
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
        result[field] = {
            str(key): int(after_map.get(key, 0))
            - int(before_map.get(key, 0))
            for key in set(before_map) | set(after_map)
            if int(after_map.get(key, 0))
            - int(before_map.get(key, 0))
        }
    result["quarantine_reason"] = after.get(
        "folded_quarantine_reason"
    ) or after.get("quarantine_reason")
    result["capture_receipts"] = list(
        after.get("capture_receipts", ())
    )
    return result


def _capture_receipt(
    summary: dict,
    *,
    folded: bool,
    correctness_trace: bool = False,
) -> dict:
    steps = 8 if folded else 1
    matches = [
        receipt
        for receipt in summary["capture_receipts"]
        if bool(receipt.get("correctness_trace"))
        is correctness_trace
        and int(receipt.get("steps_per_launch", 1)) == steps
        and int(receipt.get("flash_attn_num_splits", 0)) == 0
    ]
    if len(matches) != 1:
        raise RuntimeError("capture receipt inventory is invalid")
    return matches[0]


def _summary(llm, before: dict) -> dict:
    return _counter_delta(
        before,
        llm.model_runner.exact_greedy_decode_burst_summary(),
    )


def run_case(
    *,
    model: str,
    device: str,
    run_tag: str,
    source_commit: str,
    source_patch_sha256: str,
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
        before = llm.model_runner.exact_greedy_decode_burst_summary()
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
        summary = _summary(llm, before)
        one_token_receipt = _capture_receipt(
            summary,
            folded=False,
        )
        folded = policy == "octet_folded_graph"
        selected_receipt = (
            _capture_receipt(summary, folded=True)
            if folded
            else one_token_receipt
        )
        samples = measured["amortized_tpot_samples_ns"]
        e2e_seconds = measured["e2e_ns"] / 1_000_000_000
        row = {
            "schema_version": CASE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_commit": source_commit,
            "source_patch_sha256": source_patch_sha256,
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
            "one_token_graph_identity_sha256": one_token_receipt[
                "graph_identity_sha256"
            ],
            "folded_graph_identity_sha256": (
                selected_receipt["graph_identity_sha256"]
                if folded
                else None
            ),
            "logical_forwards": summary["target_model_forwards"],
            "logical_replays": summary["graph_replays"],
            "one_token_cuda_graph_launches": summary[
                "one_token_cuda_graph_launches"
            ],
            "folded_cuda_graph_launches": summary[
                "folded_cuda_graph_launches"
            ],
            "token_d2h_calls": summary["final_token_d2h_calls"],
            "token_d2h_bytes": summary["final_token_d2h_bytes"],
            "capture_duration_ns": selected_receipt[
                "capture_duration_ns"
            ],
            "capture_allocated_delta_bytes": selected_receipt[
                "allocated_delta_bytes"
            ],
            "capture_reserved_delta_bytes": selected_receipt[
                "reserved_delta_bytes"
            ],
            "capture_retained_static_bytes": selected_receipt[
                "retained_static_bytes"
            ],
            **memory,
            "ttft_ns": measured["ttft_ns"],
            "e2e_ns": measured["e2e_ns"],
            "tpot_samples_ns": samples,
            "tpot_median_ns": statistics.median(samples),
            "tpot_p95_ns": _nearest_rank(samples, 0.95),
            "tpot_p99_ns": _nearest_rank(samples, 0.99),
            "output_tokens_per_second": (
                GENERATED_TOKENS / e2e_seconds
            ),
            "host_visible_burst_gaps_ns": measured[
                "host_visible_burst_gaps_ns"
            ],
            "maximum_host_visible_burst_gap_ns": max(
                measured["host_visible_burst_gaps_ns"],
                default=0,
            ),
            "fallback_count": sum(
                summary["fallback_counts"].values()
            )
            + sum(summary["folded_fallback_counts"].values()),
            "rollback_count": (
                summary["lease_local_delta_journal_rollbacks"]
                + summary[
                    "lease_local_delta_journal_one_phase_rollbacks"
                ]
            ),
            "quarantine_reason": summary["quarantine_reason"],
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


def _correctness_selected_width(
    *,
    context_length: int,
    emitted_total: int,
) -> int:
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError("context length is invalid")
    _require_non_negative_int(emitted_total, "emitted_total")
    if emitted_total <= 0 or emitted_total >= GENERATED_TOKENS:
        return 0
    first_write_position = context_length + emitted_total - 1
    write_block_capacity = (
        CORRECTNESS_BLOCK_SIZE
        - (first_write_position % CORRECTNESS_BLOCK_SIZE)
    )
    return min(
        8,
        GENERATED_TOKENS - emitted_total,
        write_block_capacity,
    )


def _correctness_sampled_logit_ordinals(
    *,
    context_length: int,
) -> tuple[int, ...]:
    sampled_indices = {
        _sampling_output_index(point)
        for point in SAMPLING_POINTS[1:]
    }
    sampled_ordinals = set()
    emitted_total = 1
    while emitted_total < GENERATED_TOKENS:
        width = _correctness_selected_width(
            context_length=context_length,
            emitted_total=emitted_total,
        )
        for output_index in sampled_indices:
            if emitted_total <= output_index < emitted_total + width:
                sampled_ordinals.add(output_index - emitted_total)
        emitted_total += width
    return tuple(sorted(sampled_ordinals))


def _correctness_trace_for_step(
    *,
    context_length: int,
    emitted_total: int,
) -> bool:
    width = _correctness_selected_width(
        context_length=context_length,
        emitted_total=emitted_total,
    )
    if width == 0:
        return False
    return any(
        emitted_total <= _sampling_output_index(point)
        < emitted_total + width
        for point in SAMPLING_POINTS[1:]
    )


def run_correctness_probe(
    *,
    model: str,
    device: str,
    run_dir: Path,
    run_tag: str,
    source_commit: str,
    source_patch_sha256: str,
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
            _correctness_sampled_logit_ordinals(
                context_length=context_length,
            ),
        )
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        before = llm.model_runner.exact_greedy_decode_burst_summary()
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
                context_length=context_length,
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
                captured["prefill-final"] = (
                    llm.read_step_logits_authority()
                    .detach()
                    .to(dtype=__import__("torch").float32)
                    .contiguous()
                )
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
                            point
                        ):
                            captured[point] = values
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
        summary = _summary(llm, before)
        one_token_receipt = _capture_receipt(
            summary,
            folded=False,
            correctness_trace=True,
        )
        folded = policy == "octet_folded_graph"
        folded_receipt = (
            _capture_receipt(
                summary,
                folded=True,
                correctness_trace=True,
            )
            if folded
            else None
        )
        prompt_sha256 = sha256_text(
            ",".join(str(token) for token in prompt)
        )
        runtime_fields = {
            "one_token_graph_identity_sha256": one_token_receipt[
                "graph_identity_sha256"
            ],
            "folded_graph_identity_sha256": (
                folded_receipt["graph_identity_sha256"]
                if folded_receipt is not None
                else None
            ),
            "logical_forwards": summary["target_model_forwards"],
            "logical_replays": summary["graph_replays"],
            "one_token_cuda_graph_launches": summary[
                "one_token_cuda_graph_launches"
            ],
            "folded_cuda_graph_launches": summary[
                "folded_cuda_graph_launches"
            ],
            "token_d2h_calls": summary["final_token_d2h_calls"],
            "token_d2h_bytes": summary["final_token_d2h_bytes"],
            "fallback_count": sum(
                summary["fallback_counts"].values()
            )
            + sum(summary["folded_fallback_counts"].values()),
            "rollback_count": (
                summary["lease_local_delta_journal_rollbacks"]
                + summary[
                    "lease_local_delta_journal_one_phase_rollbacks"
                ]
            ),
            "quarantine_reason": summary["quarantine_reason"],
        }
        rows = []
        for point in SAMPLING_POINTS:
            shape, values = _flatten_logits(captured[point])
            sidecar = write_float32_sidecar(
                run_dir,
                f"logits/{context_length}-{policy}-{point}.f32",
                values,
            )
            row = {
                "schema_version": CORRECTNESS_SCHEMA_VERSION,
                "run_tag": run_tag,
                "source_commit": source_commit,
                "source_patch_sha256": source_patch_sha256,
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
                **runtime_fields,
                "correctness_trace": True,
            }
            rows.append(validate_correctness_row(row))
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
    source_patch_sha256: str,
    run_tag: str,
) -> dict:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_patch_sha256": source_patch_sha256,
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-patch-sha256", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
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
    output_dir = create_output_directory(Path(args.output_dir))
    workload = build_workload_manifest(
        model=args.model,
        device=args.device,
        run_tag=args.run_tag,
        source_commit=args.source_commit,
        source_patch_sha256=args.source_patch_sha256,
        gpu_memory_utilization=args.gpu_memory_utilization,
        environment=runtime_environment_manifest(),
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup_repetitions,
    )
    repo_root = Path(__file__).resolve().parents[1]
    _write_json(
        output_dir / "source_manifest.json",
        _source_manifest(
            repo_root=repo_root,
            source_commit=args.source_commit,
            source_patch_sha256=args.source_patch_sha256,
            run_tag=args.run_tag,
        ),
    )
    _write_json(output_dir / "workload_manifest.json", workload)
    rows = []
    row_path = output_dir / "performance_rows.jsonl"
    for repetition in range(args.repetitions):
        for context_index, context_length in enumerate(
            CONTEXT_LENGTHS
        ):
            for order_position, policy in enumerate(
                policy_order(repetition, context_index)
            ):
                row = run_case(
                    model=args.model,
                    device=args.device,
                    run_tag=args.run_tag,
                    source_commit=args.source_commit,
                    source_patch_sha256=args.source_patch_sha256,
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
                source_patch_sha256=args.source_patch_sha256,
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
        run_dir=output_dir,
    )
    summary = summarize_rows(
        rows,
        expected_repetitions=args.repetitions,
    )
    summary["correctness_row_count"] = len(correctness_rows)
    _write_json(
        output_dir / "profile_summary.json",
        summary,
    )
    from tools import exact_burst_octet_folded_graph_ceiling

    _write_json(
        output_dir / "ceiling.json",
        exact_burst_octet_folded_graph_ceiling.summarize_evidence(
            rows,
            correctness_rows,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
