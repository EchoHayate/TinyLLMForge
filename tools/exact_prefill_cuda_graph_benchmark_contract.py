#!/usr/bin/env python3
"""Frozen contract for the exact-prefill CUDA Graph paired gate."""

from __future__ import annotations

import hashlib
import json
import math


ARMS = ("eager", "exact_prefill_graph")
PROMPT_TOKEN_COUNTS = (256, 2048)
ROUNDS = 2
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 5
GENERATED_TOKENS = 16

TTFT_256_IMPROVEMENT_MINIMUM = 0.25
TTFT_2048_REGRESSION_LIMIT = 0.02
TPOT_REGRESSION_LIMIT = 0.02
E2E_REGRESSION_LIMIT = 0.02

ROW_SCHEMA_VERSION = "exact-prefill-cuda-graph.row.v1"
RESULT_SCHEMA_VERSION = "exact-prefill-cuda-graph.result.v1"
COMPARISON_SCHEMA_VERSION = "exact-prefill-cuda-graph.comparison.v1"
GATE_SCHEMA_VERSION = "exact-prefill-cuda-graph.gate.v1"
MANIFEST_SCHEMA_VERSION = "exact-prefill-cuda-graph.manifest.v1"
RUN_SCHEMA_VERSION = "exact-prefill-cuda-graph.run.v1"

SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_prefill_cuda_graph.py",
    "tinyvllm/engine/model_runner.py",
    "tools/exact_prefill_cuda_graph_benchmark_contract.py",
    "tools/exact_prefill_cuda_graph_benchmark_worker.py",
    "tools/exact_prefill_cuda_graph_gate.py",
    "tools/exact_prefill_cuda_graph_verify.py",
)

ENGINE_CONFIG = {
    "tensor_parallel_size": 1,
    "enforce_eager": False,
    "max_model_len": 2304,
    "max_num_batched_tokens": 2304,
    "max_num_seqs": 1,
    "gpu_memory_utilization": 0.5,
    "prefill_cuda_graph_token_allowlist": [256, 2048],
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


def build_case_matrix() -> list[dict]:
    rows = []
    for round_index in range(ROUNDS):
        arm_order = (
            ARMS if round_index % 2 == 0 else tuple(reversed(ARMS))
        )
        for prompt_tokens in PROMPT_TOKEN_COUNTS:
            for order_position, arm in enumerate(arm_order):
                engine_config = dict(ENGINE_CONFIG)
                engine_config["prefill_cuda_graphs"] = (
                    arm == "exact_prefill_graph"
                )
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
                    "engine_config": engine_config,
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
        "engine_config": ENGINE_CONFIG,
        "sampling": SAMPLING,
        "thresholds": {
            "ttft_256_improvement_minimum":
                TTFT_256_IMPROVEMENT_MINIMUM,
            "ttft_2048_regression_limit":
                TTFT_2048_REGRESSION_LIMIT,
            "tpot_regression_limit": TPOT_REGRESSION_LIMIT,
            "e2e_regression_limit": E2E_REGRESSION_LIMIT,
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


def positive_int(value: object, label: str) -> int:
    result = nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result
