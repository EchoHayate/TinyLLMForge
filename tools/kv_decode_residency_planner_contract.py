"""Frozen contract for the KV decode residency planner evidence gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


SCHEMA_VERSION = 1

STAGING_SHAPES = (
    (2, 1),
    (2, 2),
    (4, 1),
    (4, 2),
)

WORKLOADS = (
    "single_long_context",
    "multi_prompt_thrash",
)

POLICIES = ("baseline", "candidate")
WARMUP_REPETITIONS = 1
CORRECTNESS_REPETITIONS = 1
MEASURED_REPETITIONS = 5
LOGIT_RTOL = 1e-3
LOGIT_ATOL = 1e-2

THRESHOLDS = {
    "movement_improvement": 0.05,
    "other_movement_max_regression": 0.01,
    "decode_latency_max_regression": 0.02,
}

CASE_ROW_FIELDS = (
    "row_id",
    "case_id",
    "policy",
    "workload",
    "gpu_blocks",
    "blockwise_blocks",
    "repetition",
    "phase",
    "warmup",
    "source_sha256",
    "worker_pid",
    "tinyvllm_dist_port",
    "master_port",
    "cuda_visible_devices",
    "model_path",
    "python_path",
    "prompt_sha256",
    "decoded_token_ids",
    "decode_logits_path",
    "decode_logits_sha256",
    "decode_logits_shape",
    "decode_step_ms",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "peak_resident_blocks",
    "kv_offload",
    "planner",
    "complete",
)

REQUIRED_FILES = (
    "manifest.json",
    "environment.json",
    "source_manifest.json",
    "worker_logs_manifest.json",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "independent_verification.json",
)

KV_COUNTER_FIELDS = (
    "h2d_copies",
    "h2d_bytes",
    "d2h_copies",
    "d2h_bytes",
    "evictions",
    "copy_waits",
    "prefetch_plans",
    "evict_dirty",
)

PLANNER_COUNTER_FIELDS = (
    "decode_plan_builds",
    "decode_plan_cache_hits",
    "decode_plan_identity_invalidations",
    "decode_windows_with_spare_capacity",
    "decode_cross_layer_hint_blocks",
    "decode_cross_layer_hint_resident",
    "decode_cross_layer_hint_retained",
)

PASS_BOOLEAN_FIELDS = (
    "pair_regressions_pass",
    "copy_waits_pass",
    "prefetch_plans_pass",
    "d2h_copies_pass",
    "d2h_bytes_pass",
    "evict_dirty_pass",
    "peak_resident_blocks_pass",
    "peak_cuda_allocated_bytes_pass",
    "peak_cuda_reserved_bytes_pass",
    "decode_latency_pass",
)


@dataclass(frozen=True)
class GateCase:
    workload: str
    policy: str
    gpu_blocks: int
    blockwise_blocks: int
    repetition: int
    phase: str
    warmup: bool

    @property
    def pair_id(self):
        return (
            f"{self.workload}__g{self.gpu_blocks}"
            f"__w{self.blockwise_blocks}"
            f"__{self.phase}__r{self.repetition}"
        )

    @property
    def case_id(self):
        return f"{self.pair_id}__{self.policy}"


def build_case_matrix() -> tuple[GateCase, ...]:
    cases = []
    phases = (
        ("warmup", WARMUP_REPETITIONS),
        ("correctness", CORRECTNESS_REPETITIONS),
        ("measured", MEASURED_REPETITIONS),
    )
    for workload in WORKLOADS:
        for gpu_blocks, blockwise_blocks in STAGING_SHAPES:
            for phase, repetitions in phases:
                for repetition in range(repetitions):
                    for policy in POLICIES:
                        cases.append(GateCase(
                            workload=workload,
                            policy=policy,
                            gpu_blocks=gpu_blocks,
                            blockwise_blocks=blockwise_blocks,
                            repetition=repetition,
                            phase=phase,
                            warmup=phase == "warmup",
                        ))
    return tuple(cases)


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def movement_improvement(baseline: int | float, candidate: int | float) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline < 0 or candidate < 0:
        raise ValueError("movement counters must be non-negative")
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline


def movement_regression(baseline: int | float, candidate: int | float) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline < 0 or candidate < 0:
        raise ValueError("movement counters must be non-negative")
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf
    return max(0.0, (candidate - baseline) / baseline)


def classify_ratios(ratios: dict) -> str:
    if ratios.get("valid") is False:
        return "INVALID"

    movement_threshold = THRESHOLDS["movement_improvement"]
    movement_pass = (
        float(ratios.get("h2d_improvement", 0.0)) >= movement_threshold
        or float(ratios.get("eviction_improvement", 0.0))
        >= movement_threshold
    )
    other_movement_pass = (
        float(ratios.get("h2d_regression", math.inf))
        <= THRESHOLDS["other_movement_max_regression"]
        and float(ratios.get("eviction_regression", math.inf))
        <= THRESHOLDS["other_movement_max_regression"]
    )
    low_capacity_pass = (
        float(ratios.get("low_capacity_movement_improvement", 0.0))
        >= movement_threshold
    )
    multi_prompt_pass = (
        float(ratios.get("multi_prompt_movement_improvement", 0.0))
        >= movement_threshold
    )
    booleans_pass = all(
        ratios.get(field) is True
        for field in PASS_BOOLEAN_FIELDS
    )
    if (
        movement_pass
        and other_movement_pass
        and low_capacity_pass
        and multi_prompt_pass
        and booleans_pass
    ):
        return "GO"
    return "NO_GO"
