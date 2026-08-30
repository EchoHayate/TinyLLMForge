from __future__ import annotations

import hashlib
import math
import statistics
from typing import Mapping, Sequence


WORKLOAD_SCHEMA_VERSION = "cross-engine-k8.workload.v1"
NOT_EXPOSED = "NOT_EXPOSED"
REQUIRED_ARMS = (
    "tinyllmforge_host_greedy",
    "tinyllmforge_exact_k8",
    "vllm_default_greedy",
)
OPTIONAL_ARM = "vllm_public_multi_step"
CONTEXTS = (
    ("short", 256),
    ("medium", 2048),
    ("long", 8192),
)
OUTPUT_TOKENS = 128
WARMUPS = 2
MEASURED_REPETITIONS = 7
_PROMPT_STRATEGY = "periodic_natural_sentence_model_config_bos"
_PROMPT_BOS_TOKEN_ID = 151643
_PROMPT_PATTERN_TEXT = " The quick brown fox jumps over the lazy dog."
_PROMPT_PATTERN_TOKEN_IDS = (
    576,
    3974,
    13876,
    38835,
    34208,
    916,
    279,
    15678,
    5562,
    13,
)
_PROTECTED_RATIO_FIELDS = (
    "ttft_ratio",
    "e2e_ratio",
    "p95_tpot_ratio",
    "p99_tpot_ratio",
    "peak_gpu_memory_ratio",
    "peak_rss_ratio",
)


def _validate_digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _prompt_token_ids(context: str, length: int) -> list[int]:
    del context
    body_length = length - 1
    repeats = (
        body_length + len(_PROMPT_PATTERN_TOKEN_IDS) - 1
    ) // len(_PROMPT_PATTERN_TOKEN_IDS)
    return [
        _PROMPT_BOS_TOKEN_ID,
        *(_PROMPT_PATTERN_TOKEN_IDS * repeats)[:body_length],
    ]


def build_workload_manifest(model_inventory_sha256: str) -> dict:
    _validate_digest(model_inventory_sha256, "model_inventory_sha256")
    cases = [
        {
            "context": context,
            "prompt_tokens": prompt_tokens,
            "prompt_token_ids": _prompt_token_ids(
                context,
                prompt_tokens,
            ),
            "output_tokens": OUTPUT_TOKENS,
        }
        for context, prompt_tokens in CONTEXTS
    ]
    manifest = {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "model": "Qwen3-0.6B",
        "model_inventory_sha256": model_inventory_sha256,
        "precision": "bfloat16",
        "tensor_parallel": 1,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "prompt_strategy": _PROMPT_STRATEGY,
        "prompt_bos_token_id": _PROMPT_BOS_TOKEN_ID,
        "prompt_pattern_text": _PROMPT_PATTERN_TEXT,
        "prompt_pattern_token_ids": list(_PROMPT_PATTERN_TOKEN_IDS),
        "prompt_lengths": [
            prompt_tokens for _context, prompt_tokens in CONTEXTS
        ],
        "output_tokens": OUTPUT_TOKENS,
        "warmups": WARMUPS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "cases": cases,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        repr(manifest).encode("utf-8")
    ).hexdigest()
    return manifest


def _validate_eligible_arms(
    eligible_arms: Sequence[str],
) -> tuple[str, ...]:
    arms = tuple(eligible_arms)
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("eligible_arms must be non-empty and unique")
    missing = set(REQUIRED_ARMS) - set(arms)
    if missing:
        raise ValueError("eligible_arms are missing required arms")
    unexpected = set(arms) - set(REQUIRED_ARMS) - {OPTIONAL_ARM}
    if unexpected:
        raise ValueError("eligible_arms contain unsupported arms")
    return arms


def arm_order(
    repetition: int,
    eligible_arms: Sequence[str],
) -> tuple[str, ...]:
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
    ):
        raise ValueError("repetition must be a non-negative integer")
    arms = _validate_eligible_arms(eligible_arms)
    rotation = repetition % len(arms)
    return arms[rotation:] + arms[:rotation]


def expected_case_identities(
    manifest: Mapping,
    eligible_arms: Sequence[str],
) -> tuple[tuple[int, str, str], ...]:
    arms = _validate_eligible_arms(eligible_arms)
    repetitions = manifest.get("measured_repetitions")
    cases = manifest.get("cases")
    if repetitions != MEASURED_REPETITIONS or not isinstance(cases, list):
        raise ValueError("workload manifest is not frozen")
    identities = []
    for repetition in range(repetitions):
        order = arm_order(repetition, arms)
        for case in cases:
            context = case.get("context")
            if context not in {item[0] for item in CONTEXTS}:
                raise ValueError("workload context is invalid")
            for arm in order:
                identities.append((repetition, context, arm))
    return tuple(identities)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile input cannot be empty")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def reconstruct_metrics(
    *,
    request_start_ns: int,
    token_timestamps_ns: Sequence[int],
    request_end_ns: int,
    output_tokens: int,
) -> dict:
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
        or len(token_timestamps_ns) != output_tokens
    ):
        raise ValueError("output token count does not match timestamps")
    timeline = [request_start_ns, *token_timestamps_ns, request_end_ns]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in timeline
    ):
        raise ValueError("timestamps must be integers")
    if any(left > right for left, right in zip(timeline, timeline[1:])):
        if token_timestamps_ns[-1] > request_end_ns:
            raise ValueError("request_end_ns precedes the final token")
        raise ValueError("token timestamps must be monotonic")
    if request_end_ns < token_timestamps_ns[-1]:
        raise ValueError("request_end_ns precedes the final token")
    tpot_samples = [
        right - left
        for left, right in zip(
            token_timestamps_ns,
            token_timestamps_ns[1:],
        )
    ]
    if not tpot_samples:
        tpot_samples = [request_end_ns - token_timestamps_ns[0]]
    e2e_ns = request_end_ns - request_start_ns
    if e2e_ns <= 0:
        raise ValueError("request duration must be positive")
    return {
        "ttft_ns": token_timestamps_ns[0] - request_start_ns,
        "tpot_samples_ns": tpot_samples,
        "median_tpot_ns": statistics.median(tpot_samples),
        "p95_tpot_ns": _nearest_rank(tpot_samples, 0.95),
        "p99_tpot_ns": _nearest_rank(tpot_samples, 0.99),
        "e2e_ns": e2e_ns,
        "output_tokens_per_second": output_tokens / (e2e_ns / 1e9),
    }


def reconcile_correctness(
    rows: Sequence[Mapping],
    *,
    expected_tokens: Mapping[str, Sequence[int]],
    eligible_arms: Sequence[str],
) -> dict:
    arms = _validate_eligible_arms(eligible_arms)
    expected_pairs = {
        (context, arm)
        for context in expected_tokens
        for arm in arms
    }
    seen = set()
    mismatches = []
    invalid_arms = set()
    for row in rows:
        identity = (row.get("context"), row.get("arm"))
        if identity not in expected_pairs or identity in seen:
            mismatches.append({
                "context": row.get("context"),
                "arm": row.get("arm"),
                "reason": "unexpected_or_duplicate",
            })
            if row.get("arm") in arms:
                invalid_arms.add(row["arm"])
            continue
        seen.add(identity)
        expected = list(expected_tokens[identity[0]])
        actual = row.get("token_ids")
        if (
            actual != expected
            or row.get("output_tokens") != len(expected)
        ):
            mismatches.append({
                "context": identity[0],
                "arm": identity[1],
                "reason": "token_mismatch",
            })
            invalid_arms.add(identity[1])
    for context, arm in sorted(expected_pairs - seen):
        mismatches.append({
            "context": context,
            "arm": arm,
            "reason": "missing",
        })
        invalid_arms.add(arm)
    return {
        "valid": not mismatches,
        "eligible_arms": [
            arm for arm in arms if arm not in invalid_arms
        ],
        "mismatches": mismatches,
    }


def _median(values: Sequence[float]):
    if not values:
        raise ValueError("metric rows cannot be empty")
    if any(value == NOT_EXPOSED for value in values):
        return NOT_EXPOSED
    numeric = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError("metric value is invalid")
        numeric.append(float(value))
    return float(statistics.median(numeric))


def aggregate_case_rows(rows: Sequence[Mapping]) -> dict:
    metrics = (
        "median_tpot_ns",
        "p95_tpot_ns",
        "p99_tpot_ns",
        "ttft_ns",
        "e2e_ns",
        "output_tokens_per_second",
        "peak_gpu_memory_bytes",
        "peak_rss_bytes",
    )
    grouped: dict[str, dict[str, list[Mapping]]] = {}
    for row in rows:
        if not row.get("performance_eligible"):
            continue
        grouped.setdefault(row["arm"], {}).setdefault(
            row["context"],
            [],
        ).append(row)
    result = {}
    for arm, contexts in grouped.items():
        context_results = {}
        for context, context_rows in contexts.items():
            context_results[context] = {
                metric: _median([row[metric] for row in context_rows])
                for metric in metrics
            }
        result[arm] = {
            "contexts": context_results,
            "aggregate": {
                metric: _median([
                    values[metric]
                    for values in context_results.values()
                ])
                for metric in metrics
            },
        }
    return result


def classify_comparison(comparison: Mapping) -> dict:
    evidence_fields = (
        "complete",
        "correctness_valid",
        "storage_valid",
        "terminal_receipts_valid",
        "verifiers_agree",
    )
    missing = [
        field for field in evidence_fields if comparison.get(field) is not True
    ]
    if missing:
        return {
            "classification": "INCOMPLETE",
            "reasons": missing,
        }
    aggregate = comparison["aggregate"]
    contexts = comparison["contexts"]
    unavailable = [
        field
        for field in (
            "median_tpot_ratio",
            "throughput_ratio",
            *_PROTECTED_RATIO_FIELDS,
        )
        if (
            isinstance(aggregate.get(field), bool)
            or not isinstance(aggregate.get(field), (int, float))
            or not math.isfinite(float(aggregate[field]))
        )
    ]
    unavailable.extend(
        f"{context}.median_tpot_ratio"
        for context, values in contexts.items()
        if (
            isinstance(values.get("median_tpot_ratio"), bool)
            or not isinstance(
                values.get("median_tpot_ratio"),
                (int, float),
            )
            or not math.isfinite(float(values["median_tpot_ratio"]))
        )
    )
    if unavailable:
        return {
            "classification": "INCOMPLETE",
            "reasons": [
                f"metric_unavailable:{field}"
                for field in unavailable
            ],
        }
    protected = all(
        float(aggregate[field]) <= 1.02
        for field in _PROTECTED_RATIO_FIELDS
    )
    no_bucket_tpot_regression = all(
        float(values["median_tpot_ratio"]) <= 1.0
        for values in contexts.values()
    )
    go = (
        float(aggregate["median_tpot_ratio"]) <= 0.95
        and float(aggregate["throughput_ratio"]) >= 1.05
        and no_bucket_tpot_regression
        and protected
    )
    if go:
        return {
            "classification": "GO_CROSS_ENGINE_ADVANTAGE",
            "reasons": [],
        }
    parity = (
        0.95 <= float(aggregate["median_tpot_ratio"]) <= 1.05
        and 0.95 <= float(aggregate["throughput_ratio"]) <= 1.05
        and protected
    )
    if parity:
        return {
            "classification": "CROSS_ENGINE_PARITY",
            "reasons": [],
        }
    return {
        "classification": "NO_CROSS_ENGINE_ADVANTAGE",
        "reasons": [],
    }
