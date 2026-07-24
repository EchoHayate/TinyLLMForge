"""Frozen contract for the Qwen3.5 hybrid-state compatibility gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = 2
MODEL_REPOSITORY = "Qwen/Qwen3.5-2B"
EXPECTED_NUM_HIDDEN_LAYERS = 24
EXPECTED_LINEAR_LAYERS = 18
EXPECTED_FULL_ATTENTION_LAYERS = 6
EXPECTED_FULL_ATTENTION_INTERVAL = 4
EXPECTED_LINEAR_NUM_KEY_HEADS = 16
EXPECTED_LINEAR_NUM_VALUE_HEADS = 16
EXPECTED_LINEAR_KEY_HEAD_DIM = 128
EXPECTED_LINEAR_VALUE_HEAD_DIM = 128
EXPECTED_LINEAR_CONV_KERNEL_DIM = 4
EXPECTED_MAMBA_SSM_DTYPE = "float32"

PROMPT_LENGTHS = (17, 65, 257, 1025)
DECODE_STEPS = 8
SAME_PATH_REPEATS = 2
CHUNK_TEMPLATES = (
    (1,),
    (3, 5),
    (31, 34),
    (64,),
)
MULTI_REQUEST_LENGTHS = (17, 65, 257)
SLOT_REUSE_PROMPT_LENGTH = 33
STATE_ROLES = (
    "full_attention_key",
    "full_attention_value",
    "linear_recurrent_state",
    "linear_convolution_state",
    "position_or_sequence_metadata",
    "other_persistent_state",
)
UPDATE_KINDS = (
    "created",
    "unchanged",
    "replaced",
    "grown",
    "mutated_in_place",
    "released",
)
FINAL_CLASSIFICATIONS = ("GO", "NO_GO", "INCOMPLETE")
DECISION_TOPK = 20
FP32_ATOL = 2e-5
FP32_RTOL = 1e-5
FP32_MEAN_ABS_CAP = 3e-6
EXECUTION_DTYPES = ("bfloat16", "float32", "metadata_only")
COMPARISON_POLICIES = (
    "bf16_decision_preserving",
    "fp32_elementwise",
    "none",
)
ABS_DIFF_PERCENTILE_FIELDS = ("p50", "p95", "p99", "p99_9")
FP32_CONTROL_CASE_ID = (
    "fp32_path_control__cached_vs_one_shot__p17__r0__c17"
)

REQUIRED_PHASES = (
    "environment_preflight",
    "architecture_verification",
    "same_path_repeatability",
    "fp32_path_control",
    "one_shot_vs_cached",
    "one_shot_vs_chunked",
    "state_export_import",
    "interleaved_multi_request",
    "completion_release_slot_reuse",
    "state_memory_ledger",
    "post_run_audit",
)

CASE_ROW_FIELDS = (
    "row_id",
    "case_id",
    "phase",
    "execution_mode",
    "prompt_length",
    "chunk_schedule",
    "request_count",
    "decode_steps",
    "repeat_index",
    "request_ids",
    "request_generations",
    "decoded_token_ids",
    "logit_records",
    "state_snapshot_ids",
    "memory_snapshot_ids",
    "complete",
    "failure_kind",
    "failure_detail",
    "execution_dtype",
    "comparison_policy",
)

LOGIT_RECORD_FIELDS = (
    "request_id",
    "request_generation",
    "step_index",
    "full_logit_sha256",
    "topk_token_ids",
    "topk_logits",
    "max_abs_diff",
    "mean_abs_diff",
    "max_rel_diff",
    "mean_rel_diff",
    "sequence_length",
    "position_metadata",
    "actual_topk_token_ids",
    "actual_topk_logits",
    "oracle_topk_token_ids",
    "oracle_topk_logits",
    "topk_intersection_size",
    "oracle_topk_recall",
    "actual_winner_token_id",
    "oracle_winner_token_id",
    "actual_runner_up_token_id",
    "oracle_runner_up_token_id",
    "actual_winner_logit",
    "oracle_winner_logit",
    "actual_runner_up_logit",
    "oracle_runner_up_logit",
    "actual_winner_margin",
    "oracle_winner_margin",
    "winner_logit_abs_diff",
    "runner_up_logit_abs_diff",
    "winner_margin_abs_diff",
    "abs_diff_percentiles",
    "cosine_similarity",
    "allclose_violation_count",
    "max_allclose_scaled_error",
)

GO_GUARDS = (
    "architecture_pass",
    "same_path_repeatability_pass",
    "cached_continuation_pass",
    "chunked_continuation_pass",
    "state_export_import_pass",
    "state_roles_explained_pass",
    "state_growth_pass",
    "request_isolation_pass",
    "release_pass",
    "slot_reuse_pass",
    "storage_ledger_pass",
    "post_run_audit_pass",
)

DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}

MAX_LOGIT_ATOL = 1e-3
MAX_LOGIT_RTOL = 1e-4
MIN_LOGIT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GateCase:
    phase: str
    case_id: str
    execution_mode: str
    prompt_length: int
    chunk_schedule: tuple[int, ...]
    request_count: int
    decode_steps: int
    repeat_index: int
    expected_state_snapshots: int
    execution_dtype: str = "bfloat16"
    comparison_policy: str = "bf16_decision_preserving"

    def __post_init__(self) -> None:
        if self.phase not in REQUIRED_PHASES:
            raise ValueError(f"unsupported phase: {self.phase}")
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.execution_mode:
            raise ValueError("execution_mode must not be empty")
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if not self.chunk_schedule:
            raise ValueError("chunk_schedule must not be empty")
        if any(chunk <= 0 for chunk in self.chunk_schedule):
            raise ValueError("chunk_schedule entries must be positive")
        if sum(self.chunk_schedule) != self.prompt_length:
            raise ValueError("chunk_schedule must sum to prompt_length")
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        if self.decode_steps < 0:
            raise ValueError("decode_steps must not be negative")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must not be negative")
        if self.expected_state_snapshots < 0:
            raise ValueError(
                "expected_state_snapshots must not be negative"
            )
        if self.execution_dtype not in EXECUTION_DTYPES:
            raise ValueError(
                f"unsupported execution dtype: {self.execution_dtype}"
            )
        if self.comparison_policy not in COMPARISON_POLICIES:
            raise ValueError(
                f"unsupported comparison policy: {self.comparison_policy}"
            )
        valid_pairs = {
            ("bfloat16", "bf16_decision_preserving"),
            ("float32", "fp32_elementwise"),
            ("metadata_only", "none"),
        }
        if (
            self.execution_dtype,
            self.comparison_policy,
        ) not in valid_pairs:
            raise ValueError(
                "execution dtype and comparison policy are inconsistent"
            )


@dataclass(frozen=True)
class StateComponent:
    request_id: str
    request_generation: int
    layer_index: int
    declared_layer_type: str
    state_role: str
    tensor_path: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    logical_numel: int
    logical_bytes: int
    storage_data_ptr: int
    storage_offset: int
    storage_nbytes: int
    storage_identity: str
    lifetime_epoch: int
    sequence_length: int
    update_kind: str
    content_sha256: str


def build_chunk_schedule(
    prompt_length: int,
    prefix_chunks: tuple[int, ...],
) -> tuple[int, ...]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if any(chunk <= 0 for chunk in prefix_chunks):
        raise ValueError("prefix chunks must be positive")
    consumed = sum(prefix_chunks)
    if consumed > prompt_length:
        raise ValueError("prefix chunks exceed prompt_length")
    remainder = prompt_length - consumed
    schedule = prefix_chunks + ((remainder,) if remainder else ())
    if not schedule or sum(schedule) != prompt_length:
        raise ValueError("chunk schedule does not cover prompt_length")
    return schedule


def validate_ranked_topk(
    token_ids: list[int],
    logits: list[float],
    *,
    expected_count: int = DECISION_TOPK,
) -> None:
    if len(token_ids) != expected_count or len(logits) != expected_count:
        raise ValueError("top-k length mismatch")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise ValueError("top-k token IDs must be integers")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("top-k token IDs must be unique")
    if any(
        not isinstance(logit, (int, float))
        or isinstance(logit, bool)
        or not math.isfinite(float(logit))
        for logit in logits
    ):
        raise ValueError("top-k logits must be finite numbers")
    if any(
        float(left) < float(right)
        for left, right in zip(logits, logits[1:])
    ):
        raise ValueError("top-k logits must be non-increasing")
    if float(logits[0]) <= float(logits[1]):
        raise ValueError("top-k winner must have a strict positive margin")


def winner_margin(
    token_ids: list[int],
    logits: list[float],
) -> dict[str, int | float]:
    validate_ranked_topk(token_ids, logits)
    winner_logit = float(logits[0])
    runner_up_logit = float(logits[1])
    return {
        "winner_token_id": token_ids[0],
        "runner_up_token_id": token_ids[1],
        "winner_logit": winner_logit,
        "runner_up_logit": runner_up_logit,
        "winner_margin": winner_logit - runner_up_logit,
    }


def deterministic_token_ids(
    *,
    length: int,
    vocab_size: int,
    seed: int,
    forbidden_ids: set[int],
) -> tuple[int, ...]:
    if length < 0:
        raise ValueError("length must not be negative")
    if vocab_size <= 256:
        raise ValueError("vocab_size must be greater than 256")
    modulus = vocab_size - 256
    available = {
        token_id
        for token_id in range(128, vocab_size)
        if token_id not in forbidden_ids
    }
    if length and not available:
        raise ValueError("no allowed token IDs remain")
    values = []
    cursor = seed
    attempts = 0
    max_attempts = max(1, length) * max(vocab_size, 1024)
    while len(values) < length:
        candidate = 128 + ((cursor * 1103515245 + 12345) % modulus)
        cursor += 1
        attempts += 1
        if candidate < vocab_size and candidate not in forbidden_ids:
            values.append(candidate)
        if attempts > max_attempts:
            raise ValueError("unable to generate allowed token IDs")
    return tuple(values)


def _single_request_case(
    *,
    phase: str,
    execution_mode: str,
    prompt_length: int,
    repeat_index: int = 0,
    chunk_schedule: tuple[int, ...] | None = None,
) -> GateCase:
    schedule = chunk_schedule or (prompt_length,)
    return GateCase(
        phase=phase,
        case_id=(
            f"{phase}__{execution_mode}__p{prompt_length}"
            f"__r{repeat_index}__c{'-'.join(map(str, schedule))}"
        ),
        execution_mode=execution_mode,
        prompt_length=prompt_length,
        chunk_schedule=schedule,
        request_count=1,
        decode_steps=DECODE_STEPS,
        repeat_index=repeat_index,
        expected_state_snapshots=len(schedule) + DECODE_STEPS + 1,
    )


def build_case_matrix() -> tuple[GateCase, ...]:
    cases = [
        GateCase(
            phase="environment_preflight",
            case_id="environment_preflight",
            execution_mode="preflight",
            prompt_length=PROMPT_LENGTHS[0],
            chunk_schedule=(PROMPT_LENGTHS[0],),
            request_count=1,
            decode_steps=0,
            repeat_index=0,
            expected_state_snapshots=0,
            execution_dtype="metadata_only",
            comparison_policy="none",
        ),
        GateCase(
            phase="architecture_verification",
            case_id="architecture_verification",
            execution_mode="inspect_model",
            prompt_length=PROMPT_LENGTHS[0],
            chunk_schedule=(PROMPT_LENGTHS[0],),
            request_count=1,
            decode_steps=0,
            repeat_index=0,
            expected_state_snapshots=0,
            execution_dtype="metadata_only",
            comparison_policy="none",
        ),
    ]
    for prompt_length in PROMPT_LENGTHS:
        for repeat_index in range(SAME_PATH_REPEATS):
            cases.append(_single_request_case(
                phase="same_path_repeatability",
                execution_mode="cached_repeatability",
                prompt_length=prompt_length,
                repeat_index=repeat_index,
            ))
        cases.append(_single_request_case(
            phase="one_shot_vs_cached",
            execution_mode="one_shot_vs_cached",
            prompt_length=prompt_length,
        ))
    cases.append(GateCase(
        phase="fp32_path_control",
        case_id=FP32_CONTROL_CASE_ID,
        execution_mode="cached_vs_one_shot",
        prompt_length=PROMPT_LENGTHS[0],
        chunk_schedule=(PROMPT_LENGTHS[0],),
        request_count=1,
        decode_steps=DECODE_STEPS,
        repeat_index=0,
        expected_state_snapshots=DECODE_STEPS + 2,
        execution_dtype="float32",
        comparison_policy="fp32_elementwise",
    ))
    for prompt_length in PROMPT_LENGTHS[1:]:
        for template in CHUNK_TEMPLATES:
            if sum(template) <= prompt_length:
                cases.append(_single_request_case(
                    phase="one_shot_vs_chunked",
                    execution_mode="one_shot_vs_chunked",
                    prompt_length=prompt_length,
                    chunk_schedule=build_chunk_schedule(
                        prompt_length,
                        template,
                    ),
                ))
    for prompt_length in PROMPT_LENGTHS:
        cases.append(_single_request_case(
            phase="state_export_import",
            execution_mode="state_export_import",
            prompt_length=prompt_length,
        ))
    cases.extend([
        GateCase(
            phase="interleaved_multi_request",
            case_id="interleaved_multi_request__p17-65-257",
            execution_mode="interleaved_multi_request",
            prompt_length=max(MULTI_REQUEST_LENGTHS),
            chunk_schedule=(max(MULTI_REQUEST_LENGTHS),),
            request_count=len(MULTI_REQUEST_LENGTHS),
            decode_steps=DECODE_STEPS,
            repeat_index=0,
            expected_state_snapshots=(
                len(MULTI_REQUEST_LENGTHS) * (DECODE_STEPS + 2)
            ),
        ),
        GateCase(
            phase="completion_release_slot_reuse",
            case_id="completion_release_slot_reuse__p33",
            execution_mode="completion_release_slot_reuse",
            prompt_length=SLOT_REUSE_PROMPT_LENGTH,
            chunk_schedule=(SLOT_REUSE_PROMPT_LENGTH,),
            request_count=len(MULTI_REQUEST_LENGTHS) + 1,
            decode_steps=DECODE_STEPS,
            repeat_index=0,
            expected_state_snapshots=34,
        ),
        GateCase(
            phase="state_memory_ledger",
            case_id="state_memory_ledger",
            execution_mode="state_memory_ledger",
            prompt_length=PROMPT_LENGTHS[-1],
            chunk_schedule=(PROMPT_LENGTHS[-1],),
            request_count=1,
            decode_steps=DECODE_STEPS,
            repeat_index=0,
            expected_state_snapshots=DECODE_STEPS + 2,
            execution_dtype="metadata_only",
            comparison_policy="none",
        ),
        GateCase(
            phase="post_run_audit",
            case_id="post_run_audit",
            execution_mode="post_run_audit",
            prompt_length=PROMPT_LENGTHS[0],
            chunk_schedule=(PROMPT_LENGTHS[0],),
            request_count=1,
            decode_steps=0,
            repeat_index=0,
            expected_state_snapshots=0,
            execution_dtype="metadata_only",
            comparison_policy="none",
        ),
    ])
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs in case matrix")
    if {case.phase for case in cases} != set(REQUIRED_PHASES):
        raise ValueError("case matrix does not cover REQUIRED_PHASES")
    return tuple(cases)


def logical_bytes(shape: tuple[int, ...], dtype: str) -> int:
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"unknown dtype: {dtype}")
    if any(not isinstance(dimension, int) or dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions must be non-negative integers")
    return math.prod(shape) * DTYPE_SIZES[dtype]


def unique_storage_bytes(components: list[dict]) -> int:
    storage_sizes: dict[tuple[str, str], int] = {}
    for component in components:
        device = component.get("device")
        identity = component.get("storage_identity")
        storage_nbytes = component.get("storage_nbytes")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        if not isinstance(identity, str) or not identity:
            raise ValueError("storage_identity must be a non-empty string")
        if (
            not isinstance(storage_nbytes, int)
            or isinstance(storage_nbytes, bool)
            or storage_nbytes < 0
        ):
            raise ValueError("storage_nbytes must be a non-negative integer")
        key = (device, identity)
        previous = storage_sizes.setdefault(key, storage_nbytes)
        if previous != storage_nbytes:
            raise ValueError(
                f"conflicting storage sizes for {device}:{identity}"
            )
    return sum(storage_sizes.values())


def derive_logit_tolerance(
    repeatability_rows: list[dict],
) -> dict[str, float]:
    if not repeatability_rows:
        raise ValueError("repeatability rows are required")
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for row in repeatability_rows:
        absolute = float(row["max_abs_diff"])
        relative = float(row["max_rel_diff"])
        if (
            not math.isfinite(absolute)
            or not math.isfinite(relative)
            or absolute < 0
            or relative < 0
        ):
            raise ValueError("repeatability differences must be finite")
        max_abs_diff = max(max_abs_diff, absolute)
        max_rel_diff = max(max_rel_diff, relative)
    atol = max(MIN_LOGIT_TOLERANCE, 4 * max_abs_diff)
    rtol = max(MIN_LOGIT_TOLERANCE, 4 * max_rel_diff)
    if atol > MAX_LOGIT_ATOL or rtol > MAX_LOGIT_RTOL:
        raise ValueError(
            "INCOMPLETE_NUMERICAL_INSTABILITY: repeatability exceeds caps"
        )
    return {"atol": atol, "rtol": rtol}


def classify_evidence(
    guards: dict[str, bool],
    failure_kind: str | None,
) -> str:
    expected = set(GO_GUARDS)
    actual = set(guards)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"guard domain mismatch: missing={missing}, unknown={unknown}"
        )
    if any(type(value) is not bool for value in guards.values()):
        raise ValueError("guards must be exactly True or False")
    if failure_kind is not None:
        if not isinstance(failure_kind, str) or not failure_kind:
            raise ValueError("failure_kind must be a non-empty string")
        if failure_kind.startswith("INCOMPLETE_"):
            return "INCOMPLETE"
        if failure_kind != "semantic_failure":
            return "INCOMPLETE"
    false_guards = [name for name, value in guards.items() if not value]
    if false_guards:
        return "NO_GO"
    if failure_kind == "semantic_failure":
        return "INCOMPLETE"
    return "GO"


def _canonical_value(value):
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical_value(item) for item in value)
    return value


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_sha256(value) -> str:
    return canonical_json_sha256(value)
