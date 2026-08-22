"""Frozen contracts for staged Prefix Cache and Chunked Prefill gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics


PREFIX_STATES = ("cold", "warm", "cache_cleared")
PREFIX_TOKENS = (256, 1024, 2048)
PREFIX_BATCH_TOKENS = (1024, 2048)
PREFIX_SUFFIX_TOKENS = 64
PREFIX_BATCH_SIZE = 8
PREFIX_WARMUP_REPETITIONS = 2
PREFIX_MEASURED_REPETITIONS = 7

CHUNKED_ENGINE_CONFIG = {
    "max_model_len": 4352,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
}
CHUNKED_POLICIES = {
    "OFF": {
        **CHUNKED_ENGINE_CONFIG,
        "max_num_prefill_tokens_per_step": 0,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    },
    "FAIR_CHUNKED": {
        **CHUNKED_ENGINE_CONFIG,
        "max_num_prefill_tokens_per_step": 128,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    },
}
CHUNKED_MEASURED_REPETITIONS = 5


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_model_tier(model_tier: str) -> str:
    if model_tier not in {"qwen3-0.6b", "qwen3-8b"}:
        raise ValueError(f"unsupported model tier: {model_tier!r}")
    return model_tier


def build_prefix_case_matrix(*, model_tier: str) -> list[dict]:
    tier = _validate_model_tier(model_tier)
    rows = []
    for prefix_tokens in PREFIX_TOKENS:
        for state in PREFIX_STATES:
            rows.append({
                "case_id": f"single-{prefix_tokens}__{state}",
                "gate": "prefix",
                "model_tier": tier,
                "shape": f"single-{prefix_tokens}",
                "state": state,
                "prefix_tokens": prefix_tokens,
                "suffix_tokens": PREFIX_SUFFIX_TOKENS,
                "batch_size": 1,
                "warmup_repetitions": PREFIX_WARMUP_REPETITIONS,
                "measured_repetitions": PREFIX_MEASURED_REPETITIONS,
                "enforce_eager": True,
            })
    for prefix_tokens in PREFIX_BATCH_TOKENS:
        for state in PREFIX_STATES:
            rows.append({
                "case_id": f"batch8-{prefix_tokens}__{state}",
                "gate": "prefix",
                "model_tier": tier,
                "shape": f"batch8-{prefix_tokens}",
                "state": state,
                "prefix_tokens": prefix_tokens,
                "suffix_tokens": PREFIX_SUFFIX_TOKENS,
                "batch_size": PREFIX_BATCH_SIZE,
                "warmup_repetitions": PREFIX_WARMUP_REPETITIONS,
                "measured_repetitions": PREFIX_MEASURED_REPETITIONS,
                "enforce_eager": True,
            })
    return rows


def _balanced_output_tokens(
    prompt_tokens: int,
    ordinal_by_prompt: dict[int, int],
) -> int:
    ordinal = ordinal_by_prompt.get(prompt_tokens, 0)
    ordinal_by_prompt[prompt_tokens] = ordinal + 1
    return 16 if ordinal % 2 == 0 else 64


def build_chunked_workload(*, seed: int = 20260821) -> list[dict]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    measured_shapes = [64] * 58 + [512] * 24 + [4096] * 14
    rng.shuffle(measured_shapes)
    prompt_ordinals: dict[int, int] = {}
    rows = []
    arrival_offset_ns = 0
    for index in range(104):
        warmup = index < 8
        prompt_tokens = (
            (64, 512, 4096)[index % 3]
            if warmup
            else measured_shapes[index - 8]
        )
        output_tokens = _balanced_output_tokens(
            prompt_tokens,
            prompt_ordinals,
        )
        if index < 40:
            phase = "steady"
            arrival_offset_ns += 25_000_000
        elif index < 72:
            phase = "burst"
            arrival_offset_ns += 5_000_000
        else:
            phase = "long_injection"
            arrival_offset_ns += (
                40_000_000 if prompt_tokens == 4096 else 15_000_000
            )
        rows.append({
            "schema_version": 2,
            "request_id": (
                f"{'warmup' if warmup else 'measured'}-{index:03d}"
            ),
            "warmup": warmup,
            "phase": phase,
            "arrival_offset_ns": arrival_offset_ns,
            "prompt_tokens": prompt_tokens,
            "requested_output_tokens": output_tokens,
            "service_time_bucket": (
                f"{_prompt_class(prompt_tokens)}__"
                f"{'short' if output_tokens == 16 else 'long'}"
            ),
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": output_tokens,
            },
            "starvation_deadline_ns": 30_000_000_000,
            "drain_timeout_ns": 180_000_000_000,
        })
    _validate_chunked_workload(rows)
    return rows


def _prompt_class(prompt_tokens: int) -> str:
    return {
        64: "short",
        512: "medium",
        4096: "long",
    }[prompt_tokens]


def _validate_chunked_workload(rows: list[dict]) -> None:
    warmup = [row for row in rows if row.get("warmup") is True]
    measured = [row for row in rows if row.get("warmup") is False]
    if len(warmup) != 8 or len(measured) != 96:
        raise ValueError("chunked workload request count mismatch")
    counts = {
        prompt_tokens: sum(
            row["prompt_tokens"] == prompt_tokens for row in measured
        )
        for prompt_tokens in (64, 512, 4096)
    }
    if counts != {64: 58, 512: 24, 4096: 14}:
        raise ValueError("chunked workload prompt mix mismatch")
    request_ids = [row.get("request_id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in request_ids)
        or len(request_ids) != len(set(request_ids))
    ):
        raise ValueError("chunked workload request ids are invalid")
    offsets = [row.get("arrival_offset_ns") for row in rows]
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in offsets
        )
        or offsets != sorted(offsets)
    ):
        raise ValueError("chunked workload arrival offsets are invalid")
    for row in rows:
        prompt_tokens = row.get("prompt_tokens")
        output_tokens = row.get("requested_output_tokens")
        if (
            prompt_tokens not in (64, 512, 4096)
            or output_tokens not in (16, 64)
            or prompt_tokens + output_tokens > 4352
        ):
            raise ValueError("chunked workload shape is invalid")


def build_chunked_case_matrix(*, model_tier: str) -> list[dict]:
    tier = _validate_model_tier(model_tier)
    rows = []
    for repetition in range(CHUNKED_MEASURED_REPETITIONS):
        policy_order = (
            ("OFF", "FAIR_CHUNKED")
            if repetition % 2 == 0
            else ("FAIR_CHUNKED", "OFF")
        )
        for order, policy in enumerate(policy_order):
            rows.append({
                "case_id": f"{policy.lower()}__r{repetition}",
                "gate": "chunked",
                "model_tier": tier,
                "policy": policy,
                "repetition": repetition,
                "policy_order": order,
                "engine_config": dict(CHUNKED_POLICIES[policy]),
                "workload_sha256": canonical_json_sha256(
                    build_chunked_workload()
                ),
            })
    return rows


def _finite_number(value, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def _ratio(numerator, denominator, label: str) -> float:
    top = _finite_number(numerator, f"{label} numerator")
    bottom = _finite_number(denominator, f"{label} denominator")
    if bottom <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return top / bottom


def _prefix_shapes(raw: dict) -> list[tuple[str, dict]]:
    shapes = []
    for family, expected in (
        ("single", ("256", "1024", "2048")),
        ("batch", ("1024", "2048")),
    ):
        family_rows = raw.get(family)
        if not isinstance(family_rows, dict) or tuple(
            sorted(family_rows, key=int)
        ) != expected:
            raise ValueError(f"invalid prefix {family} shapes")
        shapes.extend(
            (f"{family}-{prefix}", family_rows[prefix])
            for prefix in expected
        )
    return shapes


def classify_prefix_bundle(raw: dict) -> dict:
    structural_failures = []
    correctness_failures = []
    performance_failures = []
    if not isinstance(raw, dict) or raw.get("artifact_complete") is not True:
        structural_failures.append("prefix artifacts are incomplete")
    targeted_failures = raw.get("correctness_failures", [])
    if (
        not isinstance(targeted_failures, list)
        or any(
            not isinstance(failure, str) or not failure
            for failure in targeted_failures
        )
        or len(set(targeted_failures)) != len(targeted_failures)
    ):
        structural_failures.append(
            "prefix targeted correctness failures are malformed"
        )
    else:
        correctness_failures.extend(targeted_failures)
    try:
        shapes = _prefix_shapes(raw)
    except (TypeError, ValueError) as error:
        structural_failures.append(str(error))
        shapes = []

    single_primary_improvements = []
    batch_improvements = []
    cuda_regressions = []
    protected_regressions = []
    retained_blocks = 0
    retained_bytes = 0
    for shape_name, shape in shapes:
        prefix_tokens = int(shape_name.rsplit("-", 1)[1])
        batch_size = 1 if shape_name.startswith("single-") else 8
        expected_reusable = prefix_tokens * batch_size
        expected_cold_query = (prefix_tokens + 64) * batch_size
        expected_warm_query = 64 * batch_size
        if (
            shape.get("prefix_tokens") != prefix_tokens
            or shape.get("suffix_tokens") != 64
            or shape.get("batch_size") != batch_size
            or shape.get("expected_reusable_tokens") != expected_reusable
        ):
            structural_failures.append(
                f"{shape_name}: shape identity mismatch"
            )
            continue
        retained_blocks = max(
            retained_blocks,
            int(shape.get("retained_reusable_blocks", 0)),
        )
        retained_bytes = max(
            retained_bytes,
            int(shape.get("retained_logical_kv_bytes", 0)),
        )
        states = {}
        for state in PREFIX_STATES:
            row = shape.get(state)
            if not isinstance(row, dict) or row.get("samples") != 7:
                structural_failures.append(
                    f"{shape_name} {state}: missing seven samples"
                )
                continue
            states[state] = row
            if (
                row.get("exact_outputs") is not True
                or row.get("logit_argmax_match") is not True
                or _finite_number(
                    row.get("logit_max_abs"),
                    f"{shape_name} {state} logit_max_abs",
                ) > 0.25
                or _finite_number(
                    row.get("logit_mean_abs"),
                    f"{shape_name} {state} logit_mean_abs",
                ) > 0.05
            ):
                correctness_failures.append(
                    f"{shape_name} {state}: output or logit mismatch"
                )
        if set(states) != set(PREFIX_STATES):
            continue
        cold = states["cold"]
        warm = states["warm"]
        cleared = states["cache_cleared"]
        accounting = (
            cold.get("median_cached_prompt_tokens") == 0
            and cold.get("median_executed_query_tokens")
            == expected_cold_query
            and warm.get("median_cached_prompt_tokens")
            == expected_reusable
            and warm.get("median_executed_query_tokens")
            == expected_warm_query
            and cleared.get("median_cached_prompt_tokens") == 0
            and cleared.get("median_executed_query_tokens")
            == expected_cold_query
        )
        if not accounting:
            correctness_failures.append(
                f"{shape_name}: cached or query token accounting mismatch"
            )
        cold_elapsed = _finite_number(
            cold.get("median_elapsed_ms"),
            f"{shape_name} cold elapsed",
        )
        warm_ratio = _ratio(
            warm.get("median_elapsed_ms"),
            cold_elapsed,
            f"{shape_name} warm elapsed",
        )
        improvement = 1.0 - warm_ratio
        cleared_ratio = _ratio(
            cleared.get("median_elapsed_ms"),
            cold_elapsed,
            f"{shape_name} cache-cleared elapsed",
        )
        cuda_ratio = _ratio(
            warm.get("peak_cuda_reserved_bytes"),
            cold.get("peak_cuda_reserved_bytes"),
            f"{shape_name} CUDA reserved",
        )
        cuda_regressions.append(cuda_ratio - 1.0)
        protected_regressions.extend((
            cleared_ratio - 1.0,
            cuda_ratio - 1.0,
        ))
        if cleared_ratio > 1.05:
            performance_failures.append(
                f"{shape_name}: cache-cleared regression exceeds 5%"
            )
        if cuda_ratio > 1.05:
            performance_failures.append(
                f"{shape_name}: CUDA reserved regression exceeds 5%"
            )
        if shape_name in ("single-1024", "single-2048"):
            single_primary_improvements.append(improvement)
            if improvement < 0.20:
                performance_failures.append(
                    f"{shape_name}: warm median TTFT improvement below 20%"
                )
        if shape_name.startswith("batch-"):
            batch_improvements.append(improvement)
            if warm.get("median_model_batches") != 1:
                performance_failures.append(
                    f"{shape_name}: warm consumers require multiple batches"
                )
            if improvement < 0.15:
                performance_failures.append(
                    f"{shape_name}: warm batch elapsed improvement below 15%"
                )

    if structural_failures or correctness_failures:
        classification = "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    elif performance_failures:
        classification = "PREFIX_CACHE_NO_GO"
    else:
        classification = "PREFIX_CACHE_GO"
    return {
        "classification": classification,
        "structural_failures": sorted(set(structural_failures)),
        "correctness_failures": sorted(set(correctness_failures)),
        "performance_failures": sorted(set(performance_failures)),
        "benefit": {
            "minimum_primary_improvement_fraction": round(
                min(single_primary_improvements, default=0.0),
                12,
            ),
            "minimum_batch_improvement_fraction": round(
                min(batch_improvements, default=0.0),
                12,
            ),
        },
        "cost": {
            "worst_protected_metric_regression_fraction": round(
                max(protected_regressions, default=0.0),
                12,
            ),
            "maximum_cuda_reserved_regression_fraction": round(
                max(cuda_regressions, default=0.0),
                12,
            ),
            "maximum_retained_reusable_blocks": retained_blocks,
            "maximum_retained_logical_kv_bytes": retained_bytes,
        },
    }


def _chunked_policy_row(
    repetition: dict,
    policy: str,
) -> dict:
    row = repetition.get(policy)
    if not isinstance(row, dict):
        raise ValueError(f"missing {policy} repetition row")
    required = (
        "short_p99_ttft_ns",
        "short_p99_itl_ns",
        "maximum_decode_gap_ns",
        "long_p95_completion_ns",
        "request_throughput_rps",
        "output_token_throughput_tps",
        "peak_cuda_reserved_bytes",
    )
    for field in required:
        _finite_number(row.get(field), f"{policy} {field}")
    buckets = row.get("service_class_p95_completion_ns")
    if not isinstance(buckets, dict) or set(buckets) != {
        "short__short",
        "short__long",
        "medium__short",
        "medium__long",
        "long__short",
        "long__long",
    }:
        raise ValueError(f"invalid {policy} service classes")
    for bucket, value in buckets.items():
        _finite_number(value, f"{policy} {bucket}")
    return row


def classify_chunked_bundle(raw: dict) -> dict:
    structural_failures = []
    correctness_failures = []
    performance_failures = []
    repetitions = raw.get("repetitions") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or raw.get("artifact_complete") is not True:
        structural_failures.append("chunked artifacts are incomplete")
    if (
        not isinstance(repetitions, list)
        or len(repetitions) != CHUNKED_MEASURED_REPETITIONS
        or {
            row.get("repetition")
            for row in repetitions
            if isinstance(row, dict)
        } != set(range(CHUNKED_MEASURED_REPETITIONS))
    ):
        structural_failures.append("chunked repetition matrix is invalid")
        repetitions = []

    ttft_improvements = []
    favorable_repetitions = 0
    protected_regressions = []
    cuda_regressions = []
    for repetition in repetitions:
        repetition_id = repetition["repetition"]
        try:
            baseline = _chunked_policy_row(repetition, "OFF")
            candidate = _chunked_policy_row(repetition, "FAIR_CHUNKED")
        except ValueError as error:
            structural_failures.append(f"r{repetition_id}: {error}")
            continue
        for policy, row in (
            ("OFF", baseline),
            ("FAIR_CHUNKED", candidate),
        ):
            if (
                row.get("exact_outputs") is not True
                or row.get("complete_lifecycle") is not True
                or any(
                    row.get(field) != 0
                    for field in (
                        "dropped_requests",
                        "rejected_requests",
                        "truncated_requests",
                        "unfinished_requests",
                        "starved_requests",
                    )
                )
            ):
                correctness_failures.append(
                    f"r{repetition_id} {policy}: lifecycle or output failure"
                )
        ttft_ratio = _ratio(
            candidate["short_p99_ttft_ns"],
            baseline["short_p99_ttft_ns"],
            f"r{repetition_id} short p99 TTFT",
        )
        ttft_improvement = 1.0 - ttft_ratio
        ttft_improvements.append(ttft_improvement)
        if ttft_improvement >= 0.10:
            favorable_repetitions += 1
        itl_ratio = _ratio(
            candidate["short_p99_itl_ns"],
            baseline["short_p99_itl_ns"],
            f"r{repetition_id} short p99 ITL",
        )
        gap_ratio = _ratio(
            candidate["maximum_decode_gap_ns"],
            baseline["maximum_decode_gap_ns"],
            f"r{repetition_id} maximum decode gap",
        )
        long_ratio = _ratio(
            candidate["long_p95_completion_ns"],
            baseline["long_p95_completion_ns"],
            f"r{repetition_id} long p95 completion",
        )
        request_throughput_ratio = _ratio(
            candidate["request_throughput_rps"],
            baseline["request_throughput_rps"],
            f"r{repetition_id} request throughput",
        )
        token_throughput_ratio = _ratio(
            candidate["output_token_throughput_tps"],
            baseline["output_token_throughput_tps"],
            f"r{repetition_id} token throughput",
        )
        cuda_ratio = _ratio(
            candidate["peak_cuda_reserved_bytes"],
            baseline["peak_cuda_reserved_bytes"],
            f"r{repetition_id} CUDA reserved",
        )
        bucket_ratios = {
            bucket: _ratio(
                candidate["service_class_p95_completion_ns"][bucket],
                baseline["service_class_p95_completion_ns"][bucket],
                f"r{repetition_id} {bucket} p95 completion",
            )
            for bucket in baseline["service_class_p95_completion_ns"]
        }
        protected_regressions.extend((
            itl_ratio - 1.0,
            gap_ratio - 1.0,
            long_ratio - 1.0,
            1.0 - request_throughput_ratio,
            1.0 - token_throughput_ratio,
            cuda_ratio - 1.0,
            *(ratio - 1.0 for ratio in bucket_ratios.values()),
        ))
        cuda_regressions.append(cuda_ratio - 1.0)
        if itl_ratio > 1.05:
            performance_failures.append(
                f"r{repetition_id}: short p99 ITL regression exceeds 5%"
            )
        if gap_ratio > 1.10:
            performance_failures.append(
                f"r{repetition_id}: maximum decode gap regression exceeds 10%"
            )
        if long_ratio > 1.10:
            performance_failures.append(
                f"r{repetition_id}: long p95 completion regression exceeds 10%"
            )
        for bucket, ratio in bucket_ratios.items():
            if ratio > 1.10:
                performance_failures.append(
                    f"r{repetition_id} {bucket}: p95 completion regression exceeds 10%"
                )
        if request_throughput_ratio < 0.97:
            performance_failures.append(
                f"r{repetition_id}: request throughput regression exceeds 3%"
            )
        if token_throughput_ratio < 0.97:
            performance_failures.append(
                f"r{repetition_id}: token throughput regression exceeds 3%"
            )
        if cuda_ratio > 1.05:
            performance_failures.append(
                f"r{repetition_id}: CUDA reserved regression exceeds 5%"
            )
    if repetitions and favorable_repetitions < 4:
        performance_failures.append(
            "short p99 TTFT benefit direction is absent in four of five repetitions"
        )
    if structural_failures or correctness_failures:
        classification = "FAIR_CHUNKED_INCOMPLETE"
    elif performance_failures:
        classification = "FAIR_CHUNKED_NO_GO"
    else:
        classification = "FAIR_CHUNKED_GO"
    return {
        "classification": classification,
        "structural_failures": sorted(set(structural_failures)),
        "correctness_failures": sorted(set(correctness_failures)),
        "performance_failures": sorted(set(performance_failures)),
        "benefit": {
            "short_p99_ttft_improvement_fraction": round(
                statistics.median(ttft_improvements)
                if ttft_improvements
                else 0.0,
                12,
            ),
            "favorable_repetitions": favorable_repetitions,
        },
        "cost": {
            "worst_protected_metric_regression_fraction": round(
                max(protected_regressions, default=0.0),
                12,
            ),
            "maximum_cuda_reserved_regression_fraction": round(
                max(cuda_regressions, default=0.0),
                12,
            ),
        },
    }


def select_stage2_winner(prefix: dict, chunked: dict) -> dict:
    prefix_eligible = (
        isinstance(prefix, dict)
        and prefix.get("classification") == "PREFIX_CACHE_GO"
    )
    chunked_eligible = (
        isinstance(chunked, dict)
        and chunked.get("classification") == "FAIR_CHUNKED_GO"
    )
    if not prefix_eligible and not chunked_eligible:
        return {
            "winner": None,
            "reason": "no Stage-1 gate is eligible",
        }
    if prefix_eligible and not chunked_eligible:
        return {
            "winner": "prefix",
            "reason": "prefix is the only eligible Stage-1 gate",
        }
    if chunked_eligible and not prefix_eligible:
        return {
            "winner": "chunked",
            "reason": "chunked is the only eligible Stage-1 gate",
        }

    prefix_benefit = _finite_number(
        prefix["benefit"]["minimum_primary_improvement_fraction"],
        "prefix primary benefit",
    )
    chunked_benefit = _finite_number(
        chunked["benefit"]["short_p99_ttft_improvement_fraction"],
        "chunked primary benefit",
    )
    if prefix_benefit != chunked_benefit:
        return {
            "winner": (
                "prefix" if prefix_benefit > chunked_benefit else "chunked"
            ),
            "reason": "larger normalized primary benefit",
        }
    prefix_worst = _finite_number(
        prefix["cost"]["worst_protected_metric_regression_fraction"],
        "prefix worst protected regression",
    )
    chunked_worst = _finite_number(
        chunked["cost"]["worst_protected_metric_regression_fraction"],
        "chunked worst protected regression",
    )
    if prefix_worst != chunked_worst:
        return {
            "winner": "prefix" if prefix_worst < chunked_worst else "chunked",
            "reason": "smaller worst protected-metric regression",
        }
    prefix_cuda = _finite_number(
        prefix["cost"]["maximum_cuda_reserved_regression_fraction"],
        "prefix CUDA reserved regression",
    )
    chunked_cuda = _finite_number(
        chunked["cost"]["maximum_cuda_reserved_regression_fraction"],
        "chunked CUDA reserved regression",
    )
    if prefix_cuda != chunked_cuda:
        return {
            "winner": "prefix" if prefix_cuda < chunked_cuda else "chunked",
            "reason": "smaller peak CUDA reserved regression",
        }
    return {
        "winner": "prefix",
        "reason": "exact tie favors lower-occupancy prefix gate",
    }
