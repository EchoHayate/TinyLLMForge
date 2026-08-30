#!/usr/bin/env python3
"""Frozen contract for the paired Phase-Stitch profile gate."""

from __future__ import annotations

import hashlib
import json
import math


ARMS = ("instrumentation_off", "instrumentation_on")
PROMPT_TOKEN_COUNTS = (256, 2048)
ROUNDS = 2
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 5
GENERATED_TOKENS = 128

MEDIAN_GAP_MINIMUM_NS = 150_000
MEDIAN_GAP_E2E_FRACTION_MINIMUM = 0.03
P95_GAP_MINIMUM_NS = 500_000
PROFILE_E2E_OVERHEAD_LIMIT = 0.01

ROW_SCHEMA_VERSION = "phase-stitch-profile.row.v1"
RESULT_SCHEMA_VERSION = "phase-stitch-profile.result.v1"
GATE_SCHEMA_VERSION = "phase-stitch-profile.gate.v1"
MANIFEST_SCHEMA_VERSION = "phase-stitch-profile.manifest.v1"
RUN_SCHEMA_VERSION = "phase-stitch-profile.run.v1"

PHASE_STITCH_EVENTS = (
    "prefill_dispatch_finished",
    "first_token_host_available",
    "prefill_scheduler_commit_finished",
    "next_schedule_started",
    "next_schedule_finished",
    "k8_lease_prepare_finished",
    "first_k8_dispatch_started",
)

SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/phase_stitch_profile.py",
    "tinyvllm/engine/exact_prefill_cuda_graph.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tools/phase_stitch_profile_contract.py",
    "tools/phase_stitch_profile_worker.py",
    "tools/phase_stitch_profile_gate.py",
    "tools/phase_stitch_profile_verify.py",
    "tools/run_phase_stitch_profile_remote.py",
)

ENGINE_CONFIG = {
    "tensor_parallel_size": 1,
    "enforce_eager": False,
    "max_model_len": 2304,
    "max_num_batched_tokens": 2304,
    "max_num_seqs": 1,
    "gpu_memory_utilization": 0.5,
    "prefill_cuda_graphs": True,
    "prefill_cuda_graph_token_allowlist": [256, 2048],
    "exact_greedy_decode_burst": True,
    "exact_greedy_decode_burst_tokens": 8,
}
SAMPLING = {
    "temperature": 0.0,
    "max_tokens": GENERATED_TOKENS,
    "ignore_eos": True,
}


def canonical_json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_case_matrix():
    cases = []
    for round_index in range(ROUNDS):
        arm_order = (
            ARMS if round_index % 2 == 0 else tuple(reversed(ARMS))
        )
        for prompt_tokens in PROMPT_TOKEN_COUNTS:
            for order_position, arm in enumerate(arm_order):
                engine_config = dict(ENGINE_CONFIG)
                engine_config["prefill_cuda_graph_token_allowlist"] = list(
                    ENGINE_CONFIG[
                        "prefill_cuda_graph_token_allowlist"
                    ]
                )
                engine_config["phase_stitch_profile"] = (
                    arm == "instrumentation_on"
                )
                cases.append({
                    "case_id": (
                        f"r{round_index}-p{prompt_tokens}-"
                        f"{arm.replace('_', '-')}"
                    ),
                    "round": round_index,
                    "order_position": order_position,
                    "arm": arm,
                    "prompt_tokens": prompt_tokens,
                    "warmup_repetitions": WARMUP_REPETITIONS,
                    "measured_repetitions": MEASURED_REPETITIONS,
                    "engine_config": engine_config,
                    "sampling": dict(SAMPLING),
                })
    return cases


def expected_case_ids():
    return tuple(case["case_id"] for case in build_case_matrix())


def contract_sha256():
    return canonical_json_sha256({
        "arms": ARMS,
        "prompt_token_counts": PROMPT_TOKEN_COUNTS,
        "rounds": ROUNDS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "generated_tokens": GENERATED_TOKENS,
        "engine_config": ENGINE_CONFIG,
        "sampling": SAMPLING,
        "thresholds": {
            "median_gap_minimum_ns": MEDIAN_GAP_MINIMUM_NS,
            "median_gap_e2e_fraction_minimum": (
                MEDIAN_GAP_E2E_FRACTION_MINIMUM
            ),
            "p95_gap_minimum_ns": P95_GAP_MINIMUM_NS,
            "profile_e2e_overhead_limit": (
                PROFILE_E2E_OVERHEAD_LIMIT
            ),
        },
        "events": PHASE_STITCH_EVENTS,
        "case_matrix": build_case_matrix(),
        "source_files": SOURCE_FILES,
    })


def validate_case_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("case spec must be an object")
    expected = {
        case["case_id"]: case for case in build_case_matrix()
    }.get(spec.get("case_id"))
    if expected is None or spec != expected:
        raise ValueError("case spec does not match frozen contract")
    return json.loads(json.dumps(spec))


def _finite_number(value, label, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier} numeric")
    return float(value)


def _nonnegative_int(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _validate_profile_row(profile, output_token_ids, prompt_tokens):
    if not isinstance(profile, dict):
        raise ValueError("phase stitch profile row is missing")
    if profile.get("status") != "complete":
        raise ValueError("phase stitch profile row is not complete")
    if profile.get("events") != list(PHASE_STITCH_EVENTS):
        raise ValueError("phase stitch profile event inventory is invalid")
    if profile.get("event_coverage_complete") is not True:
        raise ValueError("phase stitch profile event coverage is incomplete")
    if profile.get("prompt_tokens") != prompt_tokens:
        raise ValueError("phase stitch profile prompt token count drifted")
    timestamps = []
    for event in PHASE_STITCH_EVENTS:
        timestamp = profile.get(f"{event}_ns")
        _nonnegative_int(timestamp, f"profile.{event}_ns")
        timestamps.append(timestamp)
    if timestamps != sorted(timestamps):
        raise ValueError("phase stitch profile timestamps are non-monotonic")
    expected_gap = (
        profile["first_k8_dispatch_started_ns"]
        - profile["first_token_host_available_ns"]
    )
    if profile.get("removable_host_gap_ns") != expected_gap:
        raise ValueError("phase stitch profile gap does not reconstruct")
    if profile.get("output_token_ids_sha256") != canonical_json_sha256(
        output_token_ids
    ):
        raise ValueError("phase stitch profile output hash drifted")


def validate_case_result(result):
    if not isinstance(result, dict):
        raise ValueError("case result must be an object")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("case result schema version is invalid")
    case = validate_case_spec(result.get("case"))
    if not isinstance(result.get("model"), str) or not result["model"]:
        raise ValueError("case result model is invalid")
    rows = result.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != MEASURED_REPETITIONS
    ):
        raise ValueError("case result row inventory is invalid")
    observed_sample_indices = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("case result row must be an object")
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError("case result row schema version is invalid")
        for key in (
            "case_id",
            "round",
            "order_position",
            "arm",
            "prompt_tokens",
        ):
            if row.get(key) != case[key]:
                raise ValueError(f"case result row {key} drifted")
        sample_index = _nonnegative_int(
            row.get("sample_index"),
            "sample_index",
        )
        if sample_index in observed_sample_indices:
            raise ValueError("case result has duplicate samples")
        observed_sample_indices.add(sample_index)
        if row.get("generated_tokens") != GENERATED_TOKENS:
            raise ValueError("generated token count is invalid")
        output_token_ids = row.get("output_token_ids")
        if (
            not isinstance(output_token_ids, list)
            or len(output_token_ids) != GENERATED_TOKENS
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output_token_ids
            )
        ):
            raise ValueError("output token inventory is invalid")
        _sha256(row.get("prompt_sha256"), "prompt_sha256")
        _sha256(
            row.get("output_token_ids_sha256"),
            "output_token_ids_sha256",
        )
        _sha256(row.get("output_text_sha256"), "output_text_sha256")
        if row["output_token_ids_sha256"] != canonical_json_sha256(
            output_token_ids
        ):
            raise ValueError("output token hash drifted")
        _finite_number(row.get("ttft_ns"), "ttft_ns", positive=True)
        _finite_number(row.get("e2e_ns"), "e2e_ns", positive=True)
        _finite_number(
            row.get("output_tokens_per_second"),
            "output_tokens_per_second",
            positive=True,
        )
        tpot_samples = row.get("tpot_samples_ns")
        if (
            not isinstance(tpot_samples, list)
            or len(tpot_samples) != GENERATED_TOKENS - 1
        ):
            raise ValueError("TPOT sample inventory is invalid")
        for value in tpot_samples:
            _finite_number(value, "tpot sample", positive=True)
        _finite_number(
            row.get("tpot_median_ns"),
            "tpot_median_ns",
            positive=True,
        )
        if _nonnegative_int(
            row.get("prefill_graph_replay_delta"),
            "prefill_graph_replay_delta",
        ) < 1:
            raise ValueError("prefill graph replay evidence is absent")
        if _nonnegative_int(
            row.get("exact_burst_replay_delta"),
            "exact_burst_replay_delta",
        ) < 1:
            raise ValueError("exact burst replay evidence is absent")
        if _nonnegative_int(
            row.get("exact_burst_acceptance_delta"),
            "exact_burst_acceptance_delta",
        ) < 1:
            raise ValueError("exact burst acceptance evidence is absent")
        _nonnegative_int(
            row.get("cuda_peak_allocated_bytes"),
            "cuda_peak_allocated_bytes",
        )
        _nonnegative_int(
            row.get("cuda_peak_reserved_bytes"),
            "cuda_peak_reserved_bytes",
        )
        if case["arm"] == "instrumentation_on":
            _validate_profile_row(
                row.get("phase_stitch_profile"),
                output_token_ids,
                case["prompt_tokens"],
            )
        elif row.get("phase_stitch_profile") is not None:
            raise ValueError(
                "instrumentation-off row contains a profile"
            )
    if observed_sample_indices != set(range(MEASURED_REPETITIONS)):
        raise ValueError("case result sample inventory is incomplete")
    if not isinstance(result.get("prefill_graph_summary"), dict):
        raise ValueError("prefill graph summary is missing")
    if not isinstance(result.get("exact_burst_summary"), dict):
        raise ValueError("exact burst summary is missing")
    return json.loads(json.dumps(result))
