"""Production arrival-load gate contracts and offline orchestration helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
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
