"""Production arrival-load gate contracts and offline orchestration helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1

COMMON_ENGINE_CONFIG = {
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
    "max_num_prefill_tokens_per_step": 128,
    "enforce_eager": False,
}

POLICY_FIELDS = (
    "chunked_prefill_decode_first",
    "chunked_prefill_max_consecutive_chunks",
    "chunked_prefill_mixed_batch",
    "chunked_prefill_mixed_min_prompt_tokens",
)

POLICY_OVERRIDES = {
    "P0": {},
    "P1": {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
    "P2": {
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
    "P3": {
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": True,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
}

PROMPT_CLASS_TARGET_TOKENS = {
    "short": 64,
    "medium": 512,
    "long": 1536,
}
OUTPUT_CLASS_TOKENS = {
    "short": 16,
    "long": 64,
}

CALIBRATION_INITIAL_RATE_RPS = 0.5
CALIBRATION_MAX_DOUBLINGS = 8
CALIBRATION_BISECTION_STEPS = 3
CALIBRATION_REQUESTS_PER_RATE = 24
CALIBRATION_DRAIN_TIMEOUT_NS = 120_000_000_000

CANONICAL_WARMUP_REQUESTS = 8
CANONICAL_MEASURED_REQUESTS = 64
FAIRNESS_REQUESTS_PER_BUCKET = 20
CANONICAL_DRAIN_TIMEOUT_NS = 120_000_000_000
STARVATION_DEADLINE_NS = 5_000_000_000
MEASURED_REPETITIONS = 3

CANONICAL_SCENARIOS = (
    "steady_moderate",
    "near_saturation",
    "overload",
    "burst",
    "long_prompt_pressure",
    "mixed_service_fairness",
)

ARRIVAL_SEEDS = {
    "steady_moderate": 601,
    "near_saturation": 901,
    "overload": 1201,
    "burst": 1701,
    "long_prompt_pressure": 1901,
    "mixed_service_fairness": 2301,
}

SCENARIO_RATE_MULTIPLIERS = {
    "steady_moderate": 0.6,
    "near_saturation": 0.9,
    "overload": 1.2,
    "burst": 0.9,
    "long_prompt_pressure": 0.9,
    "mixed_service_fairness": 0.9,
}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("nearest_rank requires finite samples")
    if not math.isfinite(percentile) or not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be finite and in (0, 1]")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("nearest_rank requires finite samples")
    normalized.sort()
    index = math.ceil(len(normalized) * percentile) - 1
    return normalized[index]


def _prompt_record(
    prompt_id: str,
    prompt_class: str,
    prompt: str,
    prompt_token_ids: list[int],
) -> dict:
    record = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "prompt_token_count": len(prompt_token_ids),
        "prompt_class": prompt_class,
    }
    record["prompt_sha256"] = canonical_json_sha256({
        "prompt": record["prompt"],
        "prompt_token_ids": record["prompt_token_ids"],
    })
    return record


def build_prompt_bank(tokenizer, *, model_id: str) -> dict:
    prompts = []
    seed_text = (
        "TinyLLMForge deterministic arrival load prompt token "
        "scheduling fairness latency throughput memory evidence "
    )
    for prompt_class, target_tokens in PROMPT_CLASS_TARGET_TOKENS.items():
        repetitions = 1
        token_ids = []
        prompt = ""
        while len(token_ids) < target_tokens:
            prompt = (seed_text * repetitions).strip()
            token_ids = list(tokenizer.encode(prompt))
            repetitions *= 2
        token_ids = token_ids[:target_tokens]
        prompts.append(_prompt_record(
            f"{prompt_class}-0",
            prompt_class,
            prompt,
            token_ids,
        ))
    bank = {
        "schema_version": SCHEMA_VERSION,
        "model_id": str(model_id),
        "prompts": sorted(
            prompts,
            key=lambda record: record["prompt_id"],
        ),
    }
    bank["prompt_bank_sha256"] = canonical_json_sha256(bank)
    return bank


def validate_prompt_bank(prompt_bank: dict) -> None:
    if prompt_bank.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported prompt bank schema")
    prompts = prompt_bank.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt bank requires prompts")
    prompt_ids = []
    for record in prompts:
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("invalid prompt id")
        prompt_ids.append(prompt_id)
        token_ids = record.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("invalid prompt token ids")
        if record.get("prompt_token_count") != len(token_ids):
            raise ValueError("prompt token count mismatch")
        expected_hash = canonical_json_sha256({
            "prompt": record.get("prompt"),
            "prompt_token_ids": token_ids,
        })
        if record.get("prompt_sha256") != expected_hash:
            raise ValueError(f"prompt hash mismatch: {prompt_id}")
    if prompt_ids != sorted(prompt_ids):
        raise ValueError("prompt records must be sorted")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("duplicate prompt id")
    without_hash = {
        key: value
        for key, value in prompt_bank.items()
        if key != "prompt_bank_sha256"
    }
    if (
        prompt_bank.get("prompt_bank_sha256")
        != canonical_json_sha256(without_hash)
    ):
        raise ValueError("prompt bank hash mismatch")


def resolve_policy_config(policy_name: str, defaults: dict) -> dict:
    if policy_name not in POLICY_OVERRIDES:
        raise ValueError(f"unknown policy: {policy_name}")
    missing = [field for field in POLICY_FIELDS if field not in defaults]
    if missing:
        raise ValueError(
            "missing policy defaults: " + ", ".join(missing)
        )
    return {
        **COMMON_ENGINE_CONFIG,
        **{field: defaults[field] for field in POLICY_FIELDS},
        **POLICY_OVERRIDES[policy_name],
    }


def policy_identity(resolved_config: dict) -> str:
    return canonical_json_sha256(resolved_config)


def deduplicate_policies(resolved: dict[str, dict]) -> dict:
    expected_names = ("P0", "P1", "P2", "P3")
    if tuple(resolved) != expected_names:
        raise ValueError("policies must be ordered P0, P1, P2, P3")
    identity_by_name = {
        name: policy_identity(resolved[name])
        for name in expected_names
    }
    names_by_identity: dict[str, list[str]] = {}
    for name in expected_names:
        names_by_identity.setdefault(
            identity_by_name[name],
            [],
        ).append(name)
    for names in names_by_identity.values():
        if len(names) > 1 and names != ["P0", "P1"]:
            raise ValueError(
                "unexpected policy identity collision: "
                + ", ".join(names)
            )
    canonical_policy_by_name = {}
    for name in expected_names:
        aliases = names_by_identity[identity_by_name[name]]
        canonical_policy_by_name[name] = aliases[0]
    return {
        "identity_by_name": identity_by_name,
        "canonical_policy_by_name": canonical_policy_by_name,
        "aliases_by_canonical_policy": {
            names[0]: names
            for names in names_by_identity.values()
        },
    }


def build_calibration_manifest(prompt_bank: dict) -> list[dict]:
    validate_prompt_bank(prompt_bank)
    prompt_hash = prompt_bank["prompt_bank_sha256"]
    rows = []
    for rate_index in range(CALIBRATION_MAX_DOUBLINGS + 1):
        requested_rate_rps = (
            CALIBRATION_INITIAL_RATE_RPS * (2 ** rate_index)
        )
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "calibration_id": f"p0-rate-{rate_index:02d}",
            "policy": "P0",
            "requested_rate_rps": requested_rate_rps,
            "request_count": CALIBRATION_REQUESTS_PER_RATE,
            "seed": 4000 + rate_index,
            "prompt_bank_sha256": prompt_hash,
            "drain_timeout_ns": CALIBRATION_DRAIN_TIMEOUT_NS,
        })
    return rows


def _prompt_by_class(prompt_bank: dict) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for prompt in prompt_bank["prompts"]:
        grouped.setdefault(prompt["prompt_class"], []).append(prompt)
    missing = set(PROMPT_CLASS_TARGET_TOKENS) - set(grouped)
    if missing:
        raise ValueError(
            "prompt bank missing classes: " + ", ".join(sorted(missing))
        )
    return {
        prompt_class: sorted(
            records,
            key=lambda record: record["prompt_id"],
        )[0]
        for prompt_class, records in grouped.items()
    }


def _exponential_offsets(
    count: int,
    requested_rate_rps: float,
    seed: int,
) -> list[int]:
    generator = random.Random(seed)
    elapsed_s = 0.0
    offsets = []
    for _ in range(count):
        elapsed_s += generator.expovariate(requested_rate_rps)
        offsets.append(round(elapsed_s * 1_000_000_000))
    return offsets


def _burst_offsets(count: int, seed: int) -> list[int]:
    if count != CANONICAL_MEASURED_REQUESTS:
        raise ValueError("burst workload requires 64 measured requests")
    generator = random.Random(seed)
    offsets = []
    for burst_index in range(4):
        burst_start_ns = burst_index * 2_250_000_000
        for _ in range(16):
            offsets.append(
                burst_start_ns
                + generator.randrange(0, 250_000_001)
            )
    return sorted(offsets)


def _balanced_classes(index: int) -> tuple[str, str]:
    buckets = (
        ("short", "short"),
        ("short", "long"),
        ("medium", "short"),
        ("medium", "long"),
        ("long", "short"),
        ("long", "long"),
    )
    return buckets[index % len(buckets)]


def _long_pressure_classes(index: int) -> tuple[str, str]:
    prompt_cycle = (
        "long",
        "long",
        "long",
        "long",
        "long",
        "long",
        "medium",
        "medium",
        "short",
        "short",
    )
    return prompt_cycle[index % len(prompt_cycle)], (
        "short" if index % 2 == 0 else "long"
    )


def _fairness_classes(index: int) -> tuple[str, str]:
    buckets = (
        ("short", "short"),
        ("short", "long"),
        ("medium", "short"),
        ("medium", "long"),
        ("long", "short"),
        ("long", "long"),
    )
    return buckets[index // FAIRNESS_REQUESTS_PER_BUCKET]


def _request_row(
    *,
    scenario: str,
    index: int,
    warmup: bool,
    arrival_offset_ns: int,
    requested_rate_rps: float,
    prompt_class: str,
    output_class: str,
    prompt: dict,
) -> dict:
    requested_output_tokens = OUTPUT_CLASS_TOKENS[output_class]
    phase = "warmup" if warmup else "measured"
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "scenario": scenario,
        "request_id": f"{scenario}-{phase}-{index:04d}",
        "warmup": warmup,
        "arrival_offset_ns": int(arrival_offset_ns),
        "requested_rate_rps": float(requested_rate_rps),
        "seed": ARRIVAL_SEEDS[scenario],
        "prompt_id": prompt["prompt_id"],
        "prompt": prompt["prompt"],
        "prompt_sha256": prompt["prompt_sha256"],
        "prompt_token_ids": list(prompt["prompt_token_ids"]),
        "prompt_token_count": int(prompt["prompt_token_count"]),
        "prompt_class": prompt_class,
        "output_class": output_class,
        "service_time_bucket": (
            f"{prompt_class}__{output_class}"
        ),
        "requested_output_tokens": requested_output_tokens,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": requested_output_tokens,
        },
        "drain_timeout_ns": CANONICAL_DRAIN_TIMEOUT_NS,
        "starvation_deadline_ns": STARVATION_DEADLINE_NS,
    }


def build_canonical_workload(
    *,
    lambda_ref: float,
    prompt_bank: dict,
) -> list[dict]:
    if not math.isfinite(lambda_ref) or lambda_ref <= 0.0:
        raise ValueError("lambda_ref must be finite and positive")
    validate_prompt_bank(prompt_bank)
    prompts = _prompt_by_class(prompt_bank)
    rows = []
    for scenario in CANONICAL_SCENARIOS:
        requested_rate_rps = (
            lambda_ref * SCENARIO_RATE_MULTIPLIERS[scenario]
        )
        measured_count = (
            FAIRNESS_REQUESTS_PER_BUCKET * 6
            if scenario == "mixed_service_fairness"
            else CANONICAL_MEASURED_REQUESTS
        )
        warmup_offsets = _exponential_offsets(
            CANONICAL_WARMUP_REQUESTS,
            requested_rate_rps,
            ARRIVAL_SEEDS[scenario] + 10_000,
        )
        warmup_end_ns = warmup_offsets[-1] if warmup_offsets else 0
        for index, arrival_offset_ns in enumerate(warmup_offsets):
            prompt_class, output_class = _balanced_classes(index)
            rows.append(_request_row(
                scenario=scenario,
                index=index,
                warmup=True,
                arrival_offset_ns=arrival_offset_ns,
                requested_rate_rps=requested_rate_rps,
                prompt_class=prompt_class,
                output_class=output_class,
                prompt=prompts[prompt_class],
            ))

        if scenario == "burst":
            relative_offsets = _burst_offsets(
                measured_count,
                ARRIVAL_SEEDS[scenario],
            )
        else:
            relative_offsets = _exponential_offsets(
                measured_count,
                requested_rate_rps,
                ARRIVAL_SEEDS[scenario],
            )
        measured_start_ns = warmup_end_ns + 1_000_000_000
        for index, relative_offset_ns in enumerate(relative_offsets):
            if scenario == "long_prompt_pressure":
                prompt_class, output_class = (
                    _long_pressure_classes(index)
                )
            elif scenario == "mixed_service_fairness":
                prompt_class, output_class = _fairness_classes(index)
            else:
                prompt_class, output_class = _balanced_classes(index)
            rows.append(_request_row(
                scenario=scenario,
                index=index,
                warmup=False,
                arrival_offset_ns=(
                    measured_start_ns + relative_offset_ns
                ),
                requested_rate_rps=requested_rate_rps,
                prompt_class=prompt_class,
                output_class=output_class,
                prompt=prompts[prompt_class],
            ))

    scenario_order = {
        name: index
        for index, name in enumerate(CANONICAL_SCENARIOS)
    }
    rows.sort(key=lambda row: (
        scenario_order[row["scenario"]],
        row["arrival_offset_ns"],
        row["request_id"],
    ))
    counts = Counter(row["request_id"] for row in rows)
    duplicates = [
        request_id
        for request_id, count in counts.items()
        if count != 1
    ]
    if duplicates:
        raise ValueError(
            "duplicate workload request ids: "
            + ", ".join(sorted(duplicates))
        )
    return rows


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def reconstruct_request_metrics(
    workload_rows: list[dict],
    timeline_rows: list[dict],
    scheduler_rows: list[dict],
) -> list[dict]:
    del scheduler_rows
    workload_by_id = {}
    for row in workload_rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("invalid workload request id")
        if request_id in workload_by_id:
            raise ValueError(f"duplicate workload request: {request_id}")
        workload_by_id[request_id] = row

    timeline_by_id = {}
    seq_ids = set()
    for row in timeline_rows:
        request_id = row.get("request_id")
        if request_id in timeline_by_id:
            raise ValueError(f"duplicate timeline request: {request_id}")
        if request_id not in workload_by_id:
            raise ValueError(f"unexpected timeline request: {request_id}")
        seq_id = row.get("seq_id")
        if not isinstance(seq_id, int) or seq_id < 0:
            raise ValueError(f"invalid seq_id for {request_id}")
        if seq_id in seq_ids:
            raise ValueError(f"duplicate sequence binding: {seq_id}")
        seq_ids.add(seq_id)
        timeline_by_id[request_id] = row

    if set(timeline_by_id) != set(workload_by_id):
        missing = sorted(set(workload_by_id) - set(timeline_by_id))
        raise ValueError(
            "missing timeline requests: " + ", ".join(missing)
        )

    metrics = []
    for request_id, workload in workload_by_id.items():
        timeline = timeline_by_id[request_id]
        timestamp_names = (
            "scheduled_arrival_ns",
            "actual_arrival_ns",
            "first_scheduled_ns",
            "first_token_ns",
            "completion_ns",
        )
        timestamps = {
            name: _finite_number(timeline.get(name), name)
            for name in timestamp_names
        }
        if not (
            timestamps["scheduled_arrival_ns"]
            <= timestamps["actual_arrival_ns"]
            <= timestamps["first_scheduled_ns"]
            <= timestamps["first_token_ns"]
            <= timestamps["completion_ns"]
        ):
            raise ValueError(
                f"invalid timestamp ordering for {request_id}"
            )
        token_timestamps = [
            _finite_number(value, "token timestamp")
            for value in timeline.get("token_timestamps_ns", [])
        ]
        output_token_ids = timeline.get("output_token_ids")
        if not isinstance(output_token_ids, list):
            raise ValueError(
                f"invalid output token ids for {request_id}"
            )
        if len(token_timestamps) != len(output_token_ids):
            raise ValueError(
                f"token timestamp count mismatch for {request_id}"
            )
        if (
            len(output_token_ids)
            != workload.get("requested_output_tokens")
        ):
            raise ValueError(
                f"output token count mismatch for {request_id}"
            )
        if not token_timestamps:
            raise ValueError(f"request has no output tokens: {request_id}")
        if token_timestamps[0] != timestamps["first_token_ns"]:
            raise ValueError(
                f"first token timestamp mismatch for {request_id}"
            )
        if token_timestamps[-1] > timestamps["completion_ns"]:
            raise ValueError(
                f"token after completion for {request_id}"
            )
        if any(
            current < previous
            for previous, current in zip(
                token_timestamps,
                token_timestamps[1:],
            )
        ):
            raise ValueError(
                f"non-monotonic token timestamps for {request_id}"
            )
        if timeline.get("error") is not None:
            raise ValueError(f"request error for {request_id}")
        if timeline.get("finish_reason") != "length":
            raise ValueError(
                f"unexpected finish reason for {request_id}"
            )
        itl_ns = [
            current - previous
            for previous, current in zip(
                token_timestamps,
                token_timestamps[1:],
            )
        ]
        metrics.append({
            **workload,
            "seq_id": timeline["seq_id"],
            "output_token_ids": list(output_token_ids),
            "finish_reason": timeline["finish_reason"],
            "scheduled_arrival_ns": timestamps[
                "scheduled_arrival_ns"
            ],
            "actual_arrival_ns": timestamps["actual_arrival_ns"],
            "first_scheduled_ns": timestamps["first_scheduled_ns"],
            "first_token_ns": timestamps["first_token_ns"],
            "completion_ns": timestamps["completion_ns"],
            "injection_lag_ns": (
                timestamps["actual_arrival_ns"]
                - timestamps["scheduled_arrival_ns"]
            ),
            "queue_delay_ns": (
                timestamps["first_scheduled_ns"]
                - timestamps["actual_arrival_ns"]
            ),
            "ttft_ns": (
                timestamps["first_token_ns"]
                - timestamps["scheduled_arrival_ns"]
            ),
            "e2e_ns": (
                timestamps["completion_ns"]
                - timestamps["scheduled_arrival_ns"]
            ),
            "itl_ns": itl_ns,
            "maximum_decode_gap_ns": (
                max(itl_ns) if itl_ns else None
            ),
        })
    return metrics


def _percentile_metrics(
    rows: list[dict],
    field: str,
) -> dict[str, float]:
    samples = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    if not samples:
        return {}
    return {
        f"p{percentile}_{field}": nearest_rank(
            samples,
            percentile / 100.0,
        )
        for percentile in (50, 95, 99)
    }


def _jain_index(values: list[float]) -> float:
    if not values or any(value < 0.0 for value in values):
        raise ValueError("invalid Jain index samples")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0.0:
        return 0.0
    return (sum(values) ** 2) / denominator


def summarize_repetition(
    case: dict,
    request_metrics: list[dict],
    memory_rows: list[dict],
) -> dict:
    measured = [
        row for row in request_metrics
        if not row.get("warmup", False)
    ]
    if not measured:
        raise ValueError("repetition has no measured requests")
    start_ns = _finite_number(
        case.get("measurement_start_ns"),
        "measurement_start_ns",
    )
    end_ns = _finite_number(
        case.get("measurement_end_ns"),
        "measurement_end_ns",
    )
    if end_ns <= start_ns:
        raise ValueError("invalid measurement interval")
    duration_s = (end_ns - start_ns) / 1_000_000_000.0

    metrics = {
        "request_throughput_rps": len(measured) / duration_s,
        "output_token_throughput_tps": sum(
            len(row["output_token_ids"]) for row in measured
        ) / duration_s,
        "maximum_injection_lag_ns": max(
            row["injection_lag_ns"] for row in measured
        ),
        "maximum_decode_gap_ns": max(
            (
                row["maximum_decode_gap_ns"]
                for row in measured
                if row["maximum_decode_gap_ns"] is not None
            ),
            default=None,
        ),
    }
    for field in (
        "injection_lag_ns",
        "queue_delay_ns",
        "ttft_ns",
        "e2e_ns",
    ):
        metrics.update(_percentile_metrics(measured, field))
    itl_samples = [
        {"itl_ns": value}
        for row in measured
        for value in row["itl_ns"]
    ]
    metrics.update(_percentile_metrics(itl_samples, "itl_ns"))

    service_buckets = {}
    service_rates = []
    for bucket in case.get("required_service_buckets", []):
        bucket_rows = [
            row for row in measured
            if row["service_time_bucket"] == bucket
        ]
        if not bucket_rows:
            raise ValueError(f"missing service bucket: {bucket}")
        bucket_metrics = {
            "completed_requests": len(bucket_rows),
            "request_throughput_rps": len(bucket_rows) / duration_s,
            "worst_e2e_ns": max(row["e2e_ns"] for row in bucket_rows),
        }
        bucket_metrics.update(
            _percentile_metrics(bucket_rows, "e2e_ns")
        )
        service_buckets[bucket] = bucket_metrics
        service_rates.append(bucket_metrics["request_throughput_rps"])
    metrics["service_buckets"] = service_buckets
    metrics["jain_service_rate_index"] = _jain_index(service_rates)

    if not memory_rows:
        raise ValueError("repetition has no memory rows")
    for row in memory_rows:
        for field in (
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "used_kv_blocks",
            "kv_block_bytes",
        ):
            _finite_number(row.get(field), field)
    metrics["peak_cuda_allocated_bytes"] = int(max(
        row["cuda_allocated_bytes"] for row in memory_rows
    ))
    metrics["peak_cuda_reserved_bytes"] = int(max(
        row["cuda_reserved_bytes"] for row in memory_rows
    ))
    metrics["peak_used_kv_blocks"] = int(max(
        row["used_kv_blocks"] for row in memory_rows
    ))
    metrics["peak_kv_bytes"] = int(max(
        row["used_kv_blocks"] * row["kv_block_bytes"]
        for row in memory_rows
    ))

    return {
        "policy": case["policy"],
        "scenario": case["scenario"],
        "repetition": case["repetition"],
        "status": "PASS",
        "correctness": {
            "exact_outputs": True,
            "complete_requests": True,
            "no_starvation": True,
            "valid_lifecycle": True,
            "stable_p0_outputs": True,
        },
        "metrics": metrics,
    }


def aggregate_case_repetitions(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate empty repetitions")
    if len({
        (row.get("policy"), row.get("scenario"))
        for row in rows
    }) != 1:
        raise ValueError("case aggregation requires one policy/scenario")
    repetition_ids = [row.get("repetition") for row in rows]
    if (
        any(not isinstance(value, int) for value in repetition_ids)
        or len(repetition_ids) != len(set(repetition_ids))
    ):
        raise ValueError("case repetitions must be unique integers")
    metric_names = (
        "request_throughput_rps",
        "output_token_throughput_tps",
        "p95_ttft_ns",
        "p95_itl_ns",
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
        "peak_cuda_reserved_bytes",
        "peak_kv_bytes",
    )
    medians = {}
    for metric_name in metric_names:
        values = [
            _finite_number(
                row["metrics"].get(metric_name),
                metric_name,
            )
            for row in rows
            if row["metrics"].get(metric_name) is not None
        ]
        if values:
            medians[metric_name] = statistics.median(values)
    worst_repetition = min(
        rows,
        key=lambda row: (
            _finite_number(
                row["metrics"].get("request_throughput_rps"),
                "request_throughput_rps",
            ),
            -_finite_number(
                row["metrics"].get("p95_ttft_ns"),
                "p95_ttft_ns",
            ),
            -_finite_number(
                row["metrics"].get("p95_itl_ns"),
                "p95_itl_ns",
            ),
        ),
    )
    return {
        "policy": rows[0]["policy"],
        "scenario": rows[0]["scenario"],
        "repetitions": len(rows),
        "median_metrics": medians,
        "worst_repetition": worst_repetition,
    }


def _ratio(candidate: dict, baseline: dict, metric: str) -> float:
    candidate_value = _finite_number(
        candidate["metrics"].get(metric),
        f"candidate {metric}",
    )
    baseline_value = _finite_number(
        baseline["metrics"].get(metric),
        f"baseline {metric}",
    )
    if baseline_value <= 0.0:
        raise ValueError(f"baseline {metric} must be positive")
    return candidate_value / baseline_value


def _candidate_classification(
    policy: str,
    paired_rows: list[tuple[dict, dict]],
) -> dict:
    ratios = {
        metric: [
            _ratio(candidate, baseline, metric)
            for baseline, candidate in paired_rows
        ]
        for metric in (
            "request_throughput_rps",
            "p95_ttft_ns",
            "p95_itl_ns",
            "p99_ttft_ns",
            "p99_itl_ns",
            "p99_e2e_ns",
            "maximum_decode_gap_ns",
            "peak_cuda_reserved_bytes",
            "peak_kv_bytes",
        )
    }
    bucket_ratios = []
    for baseline, candidate in paired_rows:
        baseline_buckets = baseline["metrics"]["service_buckets"]
        candidate_buckets = candidate["metrics"]["service_buckets"]
        if set(baseline_buckets) != set(candidate_buckets):
            raise ValueError("service bucket set mismatch")
        for bucket in sorted(baseline_buckets):
            baseline_bucket = _finite_number(
                baseline_buckets[bucket].get("p95_e2e_ns"),
                "baseline bucket p95",
            )
            candidate_bucket = _finite_number(
                candidate_buckets[bucket].get("p95_e2e_ns"),
                "candidate bucket p95",
            )
            if baseline_bucket <= 0.0:
                raise ValueError("baseline bucket p95 must be positive")
            bucket_ratios.append(candidate_bucket / baseline_bucket)

    median = {
        metric: statistics.median(values)
        for metric, values in ratios.items()
    }
    worst = {
        "request_throughput_rps": min(
            ratios["request_throughput_rps"]
        ),
        **{
            metric: max(values)
            for metric, values in ratios.items()
            if metric != "request_throughput_rps"
        },
    }
    guard_failures = []
    for metric in (
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
    ):
        if max(ratios[metric]) > 1.10:
            guard_failures.append(f"{metric} regression exceeds 10%")
    if bucket_ratios and max(bucket_ratios) > 1.10:
        guard_failures.append(
            "service bucket p95 E2E regression exceeds 10%"
        )

    median_paths = {
        "throughput": (
            median["request_throughput_rps"] >= 1.05
            and median["p95_ttft_ns"] <= 1.05
            and median["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                median["p95_ttft_ns"] <= 0.90
                and median["p95_itl_ns"] <= 1.05
            )
            or (
                median["p95_itl_ns"] <= 0.90
                and median["p95_ttft_ns"] <= 1.05
            )
        ) and median["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                median["peak_cuda_reserved_bytes"],
                median["peak_kv_bytes"],
            ) <= 0.95
            and median["request_throughput_rps"] >= 0.98
            and median["p95_ttft_ns"] <= 1.02
            and median["p95_itl_ns"] <= 1.02
        ),
    }
    worst_paths = {
        "throughput": (
            worst["request_throughput_rps"] >= 1.05
            and worst["p95_ttft_ns"] <= 1.05
            and worst["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                worst["p95_ttft_ns"] <= 0.90
                and worst["p95_itl_ns"] <= 1.05
            )
            or (
                worst["p95_itl_ns"] <= 0.90
                and worst["p95_ttft_ns"] <= 1.05
            )
        ) and worst["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                worst["peak_cuda_reserved_bytes"],
                worst["peak_kv_bytes"],
            ) <= 0.95
            and worst["request_throughput_rps"] >= 0.98
            and worst["p95_ttft_ns"] <= 1.02
            and worst["p95_itl_ns"] <= 1.02
        ),
    }
    benefit_path = next(
        (
            path
            for path in ("throughput", "latency", "memory")
            if median_paths[path] and worst_paths[path]
        ),
        None,
    )
    favorable_direction = (
        median["request_throughput_rps"] > 1.0
        or median["p95_ttft_ns"] < 1.0
        or median["p95_itl_ns"] < 1.0
        or median["peak_cuda_reserved_bytes"] < 1.0
        or median["peak_kv_bytes"] < 1.0
    )
    if guard_failures:
        classification = "NO_GO"
    elif benefit_path is not None:
        classification = "GO"
    elif favorable_direction:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "policy": policy,
        "classification": classification,
        "benefit_path": benefit_path,
        "median_ratios": median,
        "worst_repetition_ratios": worst,
        "guard_failures": guard_failures,
    }


def classify_gate(
    run_manifest: dict,
    case_rows: list[dict],
) -> dict:
    structural_failures = []
    correctness_failures = []
    required_scenarios = run_manifest.get("required_scenarios")
    repetitions = run_manifest.get("measured_repetitions")
    canonical_by_name = run_manifest.get(
        "canonical_policy_by_name",
        {},
    )
    identities = run_manifest.get("policy_identity_by_name", {})
    if (
        not isinstance(required_scenarios, list)
        or not required_scenarios
        or not isinstance(repetitions, int)
        or repetitions < 3
    ):
        structural_failures.append("invalid required case matrix")
    if set(canonical_by_name) != {"P0", "P1", "P2", "P3"}:
        structural_failures.append("invalid policy alias map")
    if set(identities) != {"P0", "P1", "P2", "P3"}:
        structural_failures.append("invalid policy identity map")
    if (
        canonical_by_name.get("P1") == "P0"
        and identities.get("P1") != identities.get("P0")
    ):
        structural_failures.append("P1 alias identity mismatch")
    if identities.get("P2") in {
        identities.get("P0"),
        identities.get("P3"),
    }:
        structural_failures.append("unexpected P2 identity collision")
    if identities.get("P3") == identities.get("P0"):
        structural_failures.append("unexpected P3 identity collision")

    canonical_policies = []
    for name in ("P0", "P1", "P2", "P3"):
        canonical = canonical_by_name.get(name)
        if canonical == name and name not in canonical_policies:
            canonical_policies.append(name)
    expected_keys = {
        (policy, scenario, repetition)
        for policy in canonical_policies
        for scenario in required_scenarios or []
        for repetition in range(repetitions or 0)
    }
    observed_keys = []
    rows_by_key = {}
    for row in case_rows:
        key = (
            row.get("policy"),
            row.get("scenario"),
            row.get("repetition"),
        )
        observed_keys.append(key)
        rows_by_key[key] = row
        if row.get("status") != "PASS":
            structural_failures.append(
                f"incomplete case row: {key}"
            )
        metrics = row.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if (
                isinstance(metric_value, (int, float))
                and not isinstance(metric_value, bool)
                and not math.isfinite(float(metric_value))
            ):
                structural_failures.append(
                    f"non-finite metric {metric_name}: {key}"
                )
        correctness = row.get("correctness", {})
        for field in (
            "exact_outputs",
            "complete_requests",
            "no_starvation",
            "valid_lifecycle",
            "stable_p0_outputs",
        ):
            if correctness.get(field) is not True:
                correctness_failures.append(
                    f"{key} failed {field}"
                )
    if len(observed_keys) != len(set(observed_keys)):
        structural_failures.append("duplicate case rows")
    if set(observed_keys) != expected_keys:
        structural_failures.append("missing or unexpected case rows")

    if structural_failures:
        return {
            "classification": "INCOMPLETE",
            "structural_failures": sorted(set(structural_failures)),
            "correctness_failures": sorted(set(correctness_failures)),
            "candidate_results": {},
        }
    if correctness_failures:
        return {
            "classification": "NO_GO",
            "structural_failures": [],
            "correctness_failures": sorted(set(correctness_failures)),
            "candidate_results": {},
        }

    candidate_results = {}
    for policy in canonical_policies:
        if policy == "P0":
            continue
        paired_rows = []
        for scenario in required_scenarios:
            for repetition in range(repetitions):
                paired_rows.append((
                    rows_by_key[("P0", scenario, repetition)],
                    rows_by_key[(policy, scenario, repetition)],
                ))
        try:
            candidate_results[policy] = _candidate_classification(
                policy,
                paired_rows,
            )
        except ValueError as exc:
            structural_failures.append(f"{policy}: {exc}")
    if structural_failures:
        return {
            "classification": "INCOMPLETE",
            "structural_failures": sorted(set(structural_failures)),
            "correctness_failures": [],
            "candidate_results": candidate_results,
        }
    classifications = {
        result["classification"]
        for result in candidate_results.values()
    }
    if "GO" in classifications:
        classification = "GO"
    elif "PROMISING_NOT_PROVEN" in classifications:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "classification": classification,
        "structural_failures": [],
        "correctness_failures": [],
        "candidate_results": candidate_results,
    }


def render_report(run_manifest: dict, summary: dict) -> str:
    lines = [
        "# Production Arrival-Load Gate",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        "## Policies",
        "",
    ]
    for policy, canonical in sorted(
        run_manifest.get("canonical_policy_by_name", {}).items()
    ):
        lines.append(f"- `{policy}` -> `{canonical}`")
    if summary.get("structural_failures"):
        lines.extend(["", "## Structural Failures", ""])
        lines.extend(
            f"- {failure}"
            for failure in summary["structural_failures"]
        )
    if summary.get("correctness_failures"):
        lines.extend(["", "## Correctness Failures", ""])
        lines.extend(
            f"- {failure}"
            for failure in summary["correctness_failures"]
        )
    return "\n".join(lines) + "\n"
