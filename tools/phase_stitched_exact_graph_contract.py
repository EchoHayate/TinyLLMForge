#!/usr/bin/env python3
"""Frozen contract for the phase-stitched exact-graph gate."""

from __future__ import annotations

import hashlib
import json
import math


ARMS = (
    "eager",
    "prefill_only",
    "independent_composition",
    "stitched_composition",
)
PROMPT_TOKEN_COUNTS = (256, 2048)
ROUNDS = 2
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 5
GENERATED_TOKENS = 128

E2E_SHAPE_IMPROVEMENT_MINIMUM = 0.03
E2E_AGGREGATE_IMPROVEMENT_MINIMUM = 0.02
TOKEN_0_TO_1_GAP_IMPROVEMENT_MINIMUM = 0.10
TTFT_REGRESSION_LIMIT = 0.02
E2E_TAIL_REGRESSION_LIMIT = 0.02
PEAK_RESERVED_MEMORY_REGRESSION_LIMIT = 0.03

ROW_SCHEMA_VERSION = "phase-stitched-exact-graph.row.v1"
RESULT_SCHEMA_VERSION = "phase-stitched-exact-graph.result.v1"
SUMMARY_SCHEMA_VERSION = "phase-stitched-exact-graph.summary.v1"
GATE_SCHEMA_VERSION = "phase-stitched-exact-graph.gate.v1"
MANIFEST_SCHEMA_VERSION = "phase-stitched-exact-graph.manifest.v1"
RUN_SCHEMA_VERSION = "phase-stitched-exact-graph.run.v1"
RECEIPT_SCHEMA_VERSION = "phase-stitched-exact-graph.receipt.v1"

SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_prefill_cuda_graph.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/phase_stitched_exact_graph.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/phase_stitched_exact_graph_contract.py",
    "tools/phase_stitched_exact_graph_worker.py",
    "tools/phase_stitched_exact_graph_gate.py",
    "tools/phase_stitched_exact_graph_verify.py",
    "tools/run_phase_stitched_exact_graph_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/run_zero_temperature_greedy_fast_path_remote.py",
)

BASE_ENGINE_CONFIG = {
    "tensor_parallel_size": 1,
    "max_model_len": 2304,
    "max_num_batched_tokens": 2304,
    "max_num_seqs": 1,
    "gpu_memory_utilization": 0.5,
    "prefill_cuda_graph_token_allowlist": [256, 2048],
    "exact_greedy_decode_burst_tokens": 8,
    "exact_greedy_decode_burst_split_phase": False,
    "exact_greedy_decode_burst_ragged_coalescing": False,
    "exact_greedy_decode_burst_elastic_k16": False,
    "kv_offload_mvp0": False,
    "cpu_offload": False,
    "kv_quant_bits": 0,
    "quest_top_k_blocks": -1,
    "am_compact_blocks": 0,
}
SAMPLING = {
    "temperature": 0.0,
    "max_tokens": GENERATED_TOKENS,
    "ignore_eos": True,
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


def _engine_config(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown benchmark arm")
    config = dict(BASE_ENGINE_CONFIG)
    config.update({
        "enforce_eager": arm == "eager",
        "prefill_cuda_graphs": arm != "eager",
        "exact_greedy_decode_burst": arm in (
            "independent_composition",
            "stitched_composition",
        ),
        "phase_stitched_exact_graph_runtime": (
            arm == "stitched_composition"
        ),
    })
    return config


def build_case_matrix() -> list[dict]:
    rows = []
    for round_index in range(ROUNDS):
        arm_order = (
            ARMS
            if round_index == 0
            else tuple(reversed(ARMS))
        )
        for prompt_tokens in PROMPT_TOKEN_COUNTS:
            for order_position, arm in enumerate(arm_order):
                rows.append({
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
                    "precision": "bfloat16",
                    "completion_only": True,
                    "engine_config": _engine_config(arm),
                    "sampling": dict(SAMPLING),
                })
    return rows


def expected_case_ids() -> tuple[str, ...]:
    return tuple(row["case_id"] for row in build_case_matrix())


def contract_sha256() -> str:
    return canonical_json_sha256({
        "arms": ARMS,
        "prompt_token_counts": PROMPT_TOKEN_COUNTS,
        "rounds": ROUNDS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "generated_tokens": GENERATED_TOKENS,
        "base_engine_config": BASE_ENGINE_CONFIG,
        "sampling": SAMPLING,
        "thresholds": {
            "e2e_shape_improvement_minimum":
                E2E_SHAPE_IMPROVEMENT_MINIMUM,
            "e2e_aggregate_improvement_minimum":
                E2E_AGGREGATE_IMPROVEMENT_MINIMUM,
            "token_0_to_1_gap_improvement_minimum":
                TOKEN_0_TO_1_GAP_IMPROVEMENT_MINIMUM,
            "ttft_regression_limit": TTFT_REGRESSION_LIMIT,
            "e2e_tail_regression_limit":
                E2E_TAIL_REGRESSION_LIMIT,
            "peak_reserved_memory_regression_limit":
                PEAK_RESERVED_MEMORY_REGRESSION_LIMIT,
        },
        "case_matrix": build_case_matrix(),
        "source_files": SOURCE_FILES,
    })


def validate_case_spec(spec: object) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("case spec must be an object")
    expected = {
        row["case_id"]: row for row in build_case_matrix()
    }.get(spec.get("case_id"))
    if expected is None or spec != expected:
        raise ValueError("case spec does not match frozen contract")
    return json.loads(json.dumps(spec))


def finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def nonnegative_int(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def positive_number(value: object, label: str) -> float:
    result = finite_number(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
