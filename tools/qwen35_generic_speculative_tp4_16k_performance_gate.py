from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

SCHEMA_VERSION = "qwen35.generic-speculative-tp4-16k-performance.v1"
CLASSIFICATION = "SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED"
CLAIM_SCOPE = "second_model_tp4_16k_performance_only"
LIMITATIONS = (
    "phase1_not_promotable",
    "context_32k_performance_not_established",
    "learned_drafter_not_established",
    "kv_quantization_not_established",
    "five_runs_do_not_establish_statistical_significance",
    "production_readiness_not_established",
)
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
WORLD_SIZE = 4
PROMPT_TOKENS = 16384
MAX_OUTPUT_TOKENS = 64
BATCH_SIZES = (1, 4)
POLICIES = ("baseline", "ngram")
WARMUP_RUNS = 1
PARITY_RUNS = 1
MEASURED_RUNS = 5
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
MAX_MODEL_LEN = 33024
MAX_NUM_BATCHED_TOKENS = 132096
MAX_NUM_PREFILL_TOKENS_PER_STEP = 1024
KV_OFFLOAD_GPU_BLOCKS = 48
KV_OFFLOAD_LOGICAL_BLOCKS = 640
KV_OFFLOAD_BLOCKWISE_BLOCKS = 8
REAL_MOVEMENT_KEYS = (
    "h2d_copies",
    "h2d_bytes",
    "d2h_copies",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/engine/speculative_side_state.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/speculative/ngram_adapter.py",
    "tinyvllm/layers/qwen35_full_attention.py",
    "tools/speculative_runtime_performance_gate.py",
    "tools/qwen35_generic_speculative_tp4_gate.py",
    "tools/qwen35_generic_speculative_tp4_worker.py",
    "tools/qwen35_generic_speculative_tp4_16k_gate.py",
    "tools/qwen35_generic_speculative_tp4_16k_worker.py",
    "tools/qwen35_generic_speculative_tp4_16k_performance_gate.py",
    "tools/qwen35_generic_speculative_tp4_16k_performance_worker.py",
    "tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_frozen = _load_module(
    "_qwen35_tp4_16k_frozen_performance_helpers",
    TOOLS / "speculative_runtime_performance_gate.py",
)

subtract_counter_summaries = _frozen.subtract_counter_summaries
build_run_metrics = _frozen.build_run_metrics
aggregate_measurements = _frozen.aggregate_measurements
classify_batch_direction = _frozen.classify_batch_direction


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    return f"{policy}:b{batch_size}"


def _non_negative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_movement(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("movement must be a mapping")
    ranks = value.get("ranks")
    totals = value.get("totals")
    if not isinstance(ranks, list) or len(ranks) != WORLD_SIZE:
        raise ValueError("movement must contain exactly four ranks")
    if not isinstance(totals, dict):
        raise ValueError("movement totals must be a mapping")
    recomputed = {key: 0 for key in REAL_MOVEMENT_KEYS}
    normalized_ranks = []
    for rank, row in enumerate(ranks):
        if not isinstance(row, dict) or row.get("rank") != rank:
            raise ValueError("movement rank inventory mismatch")
        normalized = {"rank": rank}
        for key in REAL_MOVEMENT_KEYS:
            normalized[key] = _non_negative_integer(
                row.get(key),
                f"movement rank {rank} {key}",
            )
            recomputed[key] += normalized[key]
        normalized_ranks.append(normalized)
    normalized_totals = {
        key: _non_negative_integer(
            totals.get(key),
            f"movement total {key}",
        )
        for key in REAL_MOVEMENT_KEYS
    }
    if normalized_totals != recomputed:
        raise ValueError("movement totals do not match rank sums")
    if (
        normalized_totals[
            "speculative_residency_rejected_d2h_copies"
        ]
        != 0
    ):
        raise ValueError(
            "rejected speculative D2H copies must remain zero"
        )
    return {
        "ranks": normalized_ranks,
        "totals": normalized_totals,
    }


def validate_memory(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("memory must be a mapping")
    ranks = value.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != WORLD_SIZE:
        raise ValueError("memory must contain exactly four ranks")
    normalized_ranks = []
    peak_allocated = 0
    peak_reserved = 0
    peak_allocated_delta = 0
    peak_reserved_delta = 0
    for rank, row in enumerate(ranks):
        if not isinstance(row, dict) or row.get("rank") != rank:
            raise ValueError("memory rank inventory mismatch")
        reset = row.get("reset")
        final = row.get("final")
        if not isinstance(reset, dict) or not isinstance(final, dict):
            raise ValueError("memory reset/final rows are incomplete")
        normalized_snapshots = {}
        for snapshot_name, snapshot in (
            ("reset", reset),
            ("final", final),
        ):
            if snapshot.get("rank") != rank:
                raise ValueError("memory reset/final rank mismatch")
            normalized_snapshot = {"rank": rank}
            for key in (
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "kv_capacity_bytes",
            ):
                normalized_snapshot[key] = _non_negative_integer(
                    snapshot.get(key),
                    f"{snapshot_name} memory {key}",
                )
            normalized_snapshots[snapshot_name] = normalized_snapshot
        allocated_delta = max(
            0,
            normalized_snapshots["final"][
                "cuda_peak_allocated_bytes"
            ]
            - normalized_snapshots["reset"]["cuda_allocated_bytes"],
        )
        reserved_delta = max(
            0,
            normalized_snapshots["final"][
                "cuda_peak_reserved_bytes"
            ]
            - normalized_snapshots["reset"]["cuda_reserved_bytes"],
        )
        if (
            row.get("peak_allocated_delta_bytes") != allocated_delta
            or row.get("peak_reserved_delta_bytes") != reserved_delta
        ):
            raise ValueError("memory peak delta mismatch")
        normalized_ranks.append({
            "rank": rank,
            "reset": normalized_snapshots["reset"],
            "final": normalized_snapshots["final"],
            "peak_allocated_delta_bytes": allocated_delta,
            "peak_reserved_delta_bytes": reserved_delta,
        })
        peak_allocated = max(
            peak_allocated,
            normalized_snapshots["final"][
                "cuda_peak_allocated_bytes"
            ],
        )
        peak_reserved = max(
            peak_reserved,
            normalized_snapshots["final"][
                "cuda_peak_reserved_bytes"
            ],
        )
        peak_allocated_delta = max(
            peak_allocated_delta,
            allocated_delta,
        )
        peak_reserved_delta = max(
            peak_reserved_delta,
            reserved_delta,
        )
    expected = {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated_delta,
        "peak_reserved_delta_bytes": peak_reserved_delta,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"memory {key} mismatch")
    return {
        "ranks": normalized_ranks,
        **expected,
    }


def validate_runtime(value: object, *, policy: str) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if not isinstance(value, dict):
        raise ValueError("runtime summary must be a mapping")
    integer_keys = (
        "engine_steps",
        "prefill_steps",
        "decode_steps",
        "selected_rows",
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "first_target_callbacks",
        "tail_callbacks",
        "speculative_output_tokens",
        "target_callbacks",
    )
    normalized = {
        key: _non_negative_integer(
            value.get(key),
            f"runtime {key}",
        )
        for key in integer_keys
    }
    speculative_keys = (
        "selected_rows",
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "first_target_callbacks",
        "tail_callbacks",
        "speculative_output_tokens",
        "target_callbacks",
    )
    if policy == "baseline" and any(
        normalized[key] != 0 for key in speculative_keys
    ):
        raise ValueError(
            "baseline speculative evidence must remain zero"
        )
    proposed = normalized["proposed_tokens"]
    accepted = normalized["accepted_draft_tokens"]
    if accepted > proposed:
        raise ValueError(
            "runtime accepted tokens exceed proposed tokens"
        )
    expected_rate = accepted / proposed if proposed else 0.0
    acceptance_rate = _finite_number(
        value.get("acceptance_rate"),
        "runtime acceptance rate",
    )
    if not math.isclose(
        acceptance_rate,
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "runtime accepted-token acceptance rate mismatch"
        )
    if normalized["target_callbacks"] != (
        normalized["first_target_callbacks"]
        + normalized["tail_callbacks"]
    ):
        raise ValueError("runtime target callback mismatch")
    if policy == "ngram" and (
        normalized["selected_rows"] <= 0
        or normalized["proposal_rows"] <= 0
        or proposed <= 0
        or accepted <= 0
        or normalized["first_target_callbacks"] <= 0
        or normalized["tail_callbacks"] <= 0
    ):
        raise ValueError(
            "candidate accepted proposal evidence is incomplete"
        )
    timing_ms = value.get("timing_ms")
    if not isinstance(timing_ms, dict):
        raise ValueError("runtime timing must be a mapping")
    normalized["acceptance_rate"] = acceptance_rate
    normalized["timing_ms"] = {
        key: _finite_number(
            timing_value,
            f"runtime timing {key}",
        )
        for key, timing_value in timing_ms.items()
    }
    return normalized


def validate_cleanup_receipt(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cleanup receipt must be a mapping")
    if value.get("process_group_destroyed") is not True:
        raise ValueError("cleanup process group was not destroyed")
    if value.get("rank_exit_codes") != [0, 0, 0, 0]:
        raise ValueError("cleanup rank exit codes are invalid")
    if value.get("owned_children_remaining") != []:
        raise ValueError("cleanup owned children remain")
    rows = value.get("rank_cleanup_receipts")
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        raise ValueError("cleanup rank inventory mismatch")
    normalized_rows = []
    for rank, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("rank") != rank:
            raise ValueError("cleanup rank inventory mismatch")
        if (
            row.get("worker_exit_code") != 0
            or row.get("process_group_initialized") is not False
            or row.get("engine_exit_called") is not True
            or row.get("live_lease_count") != 0
            or row.get("prepared_transaction_count") != 0
            or row.get("runtime_poisoned") is not False
        ):
            raise ValueError("cleanup rank receipt is incomplete")
        normalized_rows.append(copy.deepcopy(row))
    return {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0, 0, 0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": normalized_rows,
    }


def validate_run(
    value: object,
    *,
    policy: str,
    batch_size: int,
) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    if not isinstance(value, dict):
        raise ValueError("run must be a mapping")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise ValueError(f"run must contain {batch_size} outputs")
    for row in outputs:
        if (
            not isinstance(row, list)
            or len(row) != MAX_OUTPUT_TOKENS
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                for token_id in row
            )
        ):
            raise ValueError(
                f"each output must contain exactly "
                f"{MAX_OUTPUT_TOKENS} tokens"
            )
    timing = value.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("run timing must be a mapping")
    if timing.get("request_count") != batch_size:
        raise ValueError("run request count mismatch")
    if (
        timing.get("total_output_tokens")
        != batch_size * MAX_OUTPUT_TOKENS
    ):
        raise ValueError("run output token total mismatch")
    per_request = timing.get("per_request")
    if (
        not isinstance(per_request, list)
        or len(per_request) != batch_size
    ):
        raise ValueError("per-request timing is incomplete")
    sequence_ids = [
        row.get("sequence_id")
        if isinstance(row, dict)
        else None
        for row in per_request
    ]
    if (
        any(
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
            for sequence_id in sequence_ids
        )
        or sequence_ids
        != list(
            range(
                sequence_ids[0],
                sequence_ids[0] + batch_size,
            )
        )
    ):
        raise ValueError(
            "per-request sequence IDs must be non-negative "
            "and contiguous within a run"
        )
    normalized_per_request = []
    for sequence_id, row in zip(sequence_ids, per_request):
        if (
            not isinstance(row, dict)
            or row.get("output_tokens") != MAX_OUTPUT_TOKENS
        ):
            raise ValueError("per-request token timing mismatch")
        normalized_row = {
            "sequence_id": sequence_id,
            "output_tokens": MAX_OUTPUT_TOKENS,
        }
        for key in (
            "ttft_s",
            "tpot_s",
            "completion_latency_s",
        ):
            normalized_row[key] = _finite_number(
                row.get(key),
                f"per-request {key}",
            )
            if normalized_row[key] < 0.0:
                raise ValueError(
                    f"per-request {key} must be non-negative"
                )
        expected_tpot = (
            normalized_row["completion_latency_s"]
            - normalized_row["ttft_s"]
        ) / (MAX_OUTPUT_TOKENS - 1)
        if (
            expected_tpot < 0.0
            or not math.isclose(
                normalized_row["tpot_s"],
                expected_tpot,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "per-request TPOT does not match TTFT/completion"
            )
        normalized_per_request.append(normalized_row)
    normalized_timing = {
        "request_count": batch_size,
        "total_output_tokens": batch_size * MAX_OUTPUT_TOKENS,
        "per_request": normalized_per_request,
    }
    for key in (
        "batch_elapsed_s",
        "batch_token_throughput_tps",
        "request_throughput_rps",
    ):
        normalized_timing[key] = _finite_number(
            timing.get(key),
            f"run timing {key}",
        )
        if normalized_timing[key] <= 0.0:
            raise ValueError(f"run timing {key} must be positive")
    expected_token_throughput = (
        normalized_timing["total_output_tokens"]
        / normalized_timing["batch_elapsed_s"]
    )
    expected_request_throughput = (
        normalized_timing["request_count"]
        / normalized_timing["batch_elapsed_s"]
    )
    if not math.isclose(
        normalized_timing["batch_token_throughput_tps"],
        expected_token_throughput,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "run token throughput does not match batch elapsed"
        )
    if not math.isclose(
        normalized_timing["request_throughput_rps"],
        expected_request_throughput,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "run request throughput does not match batch elapsed"
        )
    return {
        "outputs": copy.deepcopy(outputs),
        "timing": normalized_timing,
        "movement": validate_movement(value.get("movement")),
        "memory": validate_memory(value.get("memory")),
        "runtime": validate_runtime(
            value.get("runtime"),
            policy=policy,
        ),
        "observations": copy.deepcopy(
            value.get("observations", [])
        ),
    }


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_prompt_rows(
    value: object,
    *,
    batch_size: int,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError("prompt row inventory mismatch")
    normalized = []
    for prompt_index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError("prompt row must be a mapping")
        token_ids = row.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != PROMPT_TOKENS
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError(
                f"prompt {prompt_index} must contain exactly "
                f"{PROMPT_TOKENS} token IDs"
            )
        if row.get("prompt_index") != prompt_index:
            raise ValueError("prompt index mismatch")
        if row.get("token_count") != PROMPT_TOKENS:
            raise ValueError("prompt token count mismatch")
        if row.get("sha256") != _json_sha256(token_ids):
            raise ValueError("prompt digest mismatch")
        normalized.append({
            "prompt_index": prompt_index,
            "token_count": PROMPT_TOKENS,
            "token_ids": list(token_ids),
            "sha256": row["sha256"],
        })
    return normalized


def validate_worker_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("worker result must be a mapping")
    policy = value.get("policy")
    batch_size = value.get("batch_size")
    cell_key(policy, batch_size)
    run_groups = {}
    for group_name, expected_count in (
        ("warmup_runs", WARMUP_RUNS),
        ("parity_runs", PARITY_RUNS),
        ("measured_runs", MEASURED_RUNS),
    ):
        runs = value.get(group_name)
        if not isinstance(runs, list):
            raise ValueError(f"{group_name} must be a list")
        if len(runs) != expected_count:
            count_name = "five" if expected_count == 5 else "one"
            raise ValueError(
                f"{group_name} must contain exactly {count_name} runs"
            )
        run_groups[group_name] = [
            validate_run(
                run,
                policy=policy,
                batch_size=batch_size,
            )
            for run in runs
        ]
    tokenizer_identifier = value.get("tokenizer_identifier")
    dtype = value.get("dtype")
    if (
        not isinstance(tokenizer_identifier, str)
        or not tokenizer_identifier
        or not isinstance(dtype, str)
        or not dtype
    ):
        raise ValueError(
            "worker tokenizer/dtype identity is incomplete"
        )
    normalized = {
        "policy": policy,
        "batch_size": batch_size,
        "prompt_rows": _validate_prompt_rows(
            value.get("prompt_rows"),
            batch_size=batch_size,
        ),
        **run_groups,
        "tokenizer_identifier": tokenizer_identifier,
        "dtype": dtype,
    }
    normalized["cleanup_receipt"] = validate_cleanup_receipt(
        value.get("cleanup_receipt")
    )
    return normalized


def aggregate_worker(worker: dict) -> dict:
    runs = worker["measured_runs"]
    ttft = []
    tpot = []
    completion = []
    throughput = []
    request_throughput = []
    peak_allocated = []
    peak_reserved = []
    movement = {key: 0 for key in REAL_MOVEMENT_KEYS}
    runtime_keys = (
        "selected_rows",
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "first_target_callbacks",
        "tail_callbacks",
        "speculative_output_tokens",
        "target_callbacks",
    )
    runtime = {key: 0 for key in runtime_keys}
    for run in runs:
        timing = run["timing"]
        ttft.append(statistics.median(
            row["ttft_s"]
            for row in timing["per_request"]
        ))
        tpot.append(statistics.median(
            row["tpot_s"]
            for row in timing["per_request"]
        ))
        completion.append(statistics.median(
            row["completion_latency_s"]
            for row in timing["per_request"]
        ))
        throughput.append(
            timing["batch_token_throughput_tps"]
        )
        request_throughput.append(
            timing["request_throughput_rps"]
        )
        peak_allocated.append(
            run["memory"]["peak_allocated_bytes"]
        )
        peak_reserved.append(
            run["memory"]["peak_reserved_bytes"]
        )
        for key in REAL_MOVEMENT_KEYS:
            movement[key] += run["movement"]["totals"][key]
        for key in runtime_keys:
            runtime[key] += run["runtime"][key]
    proposed = runtime["proposed_tokens"]
    runtime["acceptance_rate"] = (
        runtime["accepted_draft_tokens"] / proposed
        if proposed
        else 0.0
    )
    return {
        "ttft_s": aggregate_measurements(ttft),
        "tpot_s": aggregate_measurements(tpot),
        "completion_latency_s": aggregate_measurements(
            completion
        ),
        "batch_token_throughput_tps": aggregate_measurements(
            throughput
        ),
        "request_throughput_rps": aggregate_measurements(
            request_throughput
        ),
        "peak_allocated_bytes": aggregate_measurements(
            peak_allocated
        ),
        "peak_reserved_bytes": aggregate_measurements(
            peak_reserved
        ),
        "movement_totals": movement,
        "runtime_totals": runtime,
    }


def _positive_ratio(
    candidate: float | int,
    baseline: float | int,
    name: str,
) -> float:
    baseline_value = _finite_number(baseline, f"baseline {name}")
    candidate_value = _finite_number(candidate, f"candidate {name}")
    if baseline_value <= 0.0:
        raise ValueError(f"baseline {name} must be positive")
    if candidate_value < 0.0:
        raise ValueError(f"candidate {name} must be non-negative")
    return candidate_value / baseline_value


def derive_comparison(cells: dict, batch_size: int) -> dict:
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    try:
        baseline = cells[cell_key("baseline", batch_size)][
            "aggregate"
        ]
        candidate = cells[cell_key("ngram", batch_size)][
            "aggregate"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("comparison cell inventory is incomplete") from error
    tpot_ratio = _positive_ratio(
        candidate["tpot_s"]["median"],
        baseline["tpot_s"]["median"],
        "median TPOT",
    )
    throughput_ratio = _positive_ratio(
        candidate["batch_token_throughput_tps"]["median"],
        baseline["batch_token_throughput_tps"]["median"],
        "median batch token throughput",
    )
    return {
        "direction": classify_batch_direction(
            baseline,
            candidate,
        ),
        "tpot_ratio": tpot_ratio,
        "tpot_percent_delta": (tpot_ratio - 1.0) * 100.0,
        "throughput_ratio": throughput_ratio,
        "throughput_percent_delta": (
            throughput_ratio - 1.0
        ) * 100.0,
        "ttft_ratio": _positive_ratio(
            candidate["ttft_s"]["median"],
            baseline["ttft_s"]["median"],
            "median TTFT",
        ),
        "completion_latency_ratio": _positive_ratio(
            candidate["completion_latency_s"]["median"],
            baseline["completion_latency_s"]["median"],
            "median completion latency",
        ),
        "request_throughput_ratio": _positive_ratio(
            candidate["request_throughput_rps"]["median"],
            baseline["request_throughput_rps"]["median"],
            "median request throughput",
        ),
        "peak_allocated_ratio": _positive_ratio(
            candidate["peak_allocated_bytes"]["median"],
            baseline["peak_allocated_bytes"]["median"],
            "median peak allocated bytes",
        ),
        "peak_reserved_ratio": _positive_ratio(
            candidate["peak_reserved_bytes"]["median"],
            baseline["peak_reserved_bytes"]["median"],
            "median peak reserved bytes",
        ),
        "h2d_bytes_ratio": _positive_ratio(
            candidate["movement_totals"]["h2d_bytes"],
            baseline["movement_totals"]["h2d_bytes"],
            "H2D bytes",
        ),
        "d2h_bytes_ratio": _positive_ratio(
            candidate["movement_totals"]["d2h_bytes"],
            baseline["movement_totals"]["d2h_bytes"],
            "D2H bytes",
        ),
    }


def derive_artifact(worker_results: list[dict]) -> dict:
    if (
        not isinstance(worker_results, list)
        or len(worker_results) != 4
    ):
        raise ValueError(
            "artifact requires exactly four worker cells"
        )
    cells = {}
    tokenizer_identifier = None
    dtype = None
    for raw_worker in worker_results:
        worker = validate_worker_result(raw_worker)
        key = cell_key(
            worker["policy"],
            worker["batch_size"],
        )
        if key in cells:
            raise ValueError("duplicate worker cell")
        if tokenizer_identifier is None:
            tokenizer_identifier = worker["tokenizer_identifier"]
            dtype = worker["dtype"]
        elif (
            worker["tokenizer_identifier"]
            != tokenizer_identifier
            or worker["dtype"] != dtype
        ):
            raise ValueError(
                "worker tokenizer/dtype identities differ"
            )
        cells[key] = {
            **worker,
            "aggregate": aggregate_worker(worker),
        }
    expected_cells = {
        cell_key(policy, batch_size)
        for policy in POLICIES
        for batch_size in BATCH_SIZES
    }
    if set(cells) != expected_cells:
        raise ValueError("worker cell inventory mismatch")
    for batch_size in BATCH_SIZES:
        baseline = cells[cell_key("baseline", batch_size)]
        candidate = cells[cell_key("ngram", batch_size)]
        if baseline["prompt_rows"] != candidate["prompt_rows"]:
            raise ValueError(
                f"batch {batch_size} prompt parity failed"
            )
        reference_outputs = baseline["parity_runs"][0][
            "outputs"
        ]
        for cell in (baseline, candidate):
            for run in (
                cell["parity_runs"]
                + cell["measured_runs"]
            ):
                if run["outputs"] != reference_outputs:
                    raise ValueError(
                        f"batch {batch_size} exact token parity failed"
                    )
    for policy in POLICIES:
        aggregate = cells[cell_key(policy, 4)]["aggregate"]
        if (
            aggregate["movement_totals"]["h2d_copies"] <= 0
            or aggregate["movement_totals"]["h2d_bytes"] <= 0
            or aggregate["movement_totals"]["d2h_copies"] <= 0
            or aggregate["movement_totals"]["d2h_bytes"] <= 0
        ):
            raise ValueError(
                f"batch-4 {policy} lacks bidirectional real movement"
            )
    comparisons = {
        str(batch_size): derive_comparison(cells, batch_size)
        for batch_size in BATCH_SIZES
    }
    directions = [
        comparison["direction"]
        for comparison in comparisons.values()
    ]
    if all(direction == "IMPROVED" for direction in directions):
        campaign_direction = "POSITIVE"
    elif any(direction == "REGRESSED" for direction in directions):
        campaign_direction = "NEGATIVE"
    else:
        campaign_direction = "MIXED"
    return {
        "cells": cells,
        "comparisons": comparisons,
        "campaign_direction": campaign_direction,
        "tokenizer_identifier": tokenizer_identifier,
        "dtype": dtype,
    }


def _artifact_values_equivalent(
    stored: object,
    derived: object,
) -> bool:
    if isinstance(stored, bool) or isinstance(derived, bool):
        return stored is derived
    if (
        isinstance(stored, (int, float))
        or isinstance(derived, (int, float))
    ):
        if (
            isinstance(stored, bool)
            or isinstance(derived, bool)
            or not isinstance(stored, (int, float))
            or not isinstance(derived, (int, float))
        ):
            return False
        return math.isclose(
            float(stored),
            float(derived),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if isinstance(stored, dict) or isinstance(derived, dict):
        if not isinstance(stored, dict) or not isinstance(
            derived,
            dict,
        ):
            return False
        return (
            set(stored) == set(derived)
            and all(
                _artifact_values_equivalent(
                    stored[key],
                    derived[key],
                )
                for key in stored
            )
        )
    if isinstance(stored, list) or isinstance(derived, list):
        if not isinstance(stored, list) or not isinstance(
            derived,
            list,
        ):
            return False
        return (
            len(stored) == len(derived)
            and all(
                _artifact_values_equivalent(left, right)
                for left, right in zip(stored, derived)
            )
        )
    return stored == derived


def _campaign_contract() -> dict:
    return {
        "tensor_parallel_size": WORLD_SIZE,
        "prompt_tokens": PROMPT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "batch_sizes": list(BATCH_SIZES),
        "policies": list(POLICIES),
        "temperature": 0.0,
        "ignore_eos": True,
        "warmup_runs": WARMUP_RUNS,
        "parity_runs": PARITY_RUNS,
        "measured_runs": MEASURED_RUNS,
        "ngram_size": NGRAM_SIZE,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
    }


def _engine_contract() -> dict:
    return {
        "tensor_parallel_size": WORLD_SIZE,
        "enforce_eager": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_prefill_tokens_per_step": (
            MAX_NUM_PREFILL_TOKENS_PER_STEP
        ),
        "chunked_prefill_decode_first": False,
        "chunked_prefill_mixed_batch": False,
        "kv_offload_mvp0": True,
        "kv_offload_gpu_blocks": KV_OFFLOAD_GPU_BLOCKS,
        "kv_offload_logical_blocks": KV_OFFLOAD_LOGICAL_BLOCKS,
        "kv_offload_blockwise_decode": True,
        "kv_offload_blockwise_prefill": True,
        "kv_offload_blockwise_blocks": (
            KV_OFFLOAD_BLOCKWISE_BLOCKS
        ),
    }


def build_performance_artifact(
    *,
    worker_results: list[dict],
    environment: dict,
    gpu_indices: tuple[int, ...],
    source_files: dict[str, str],
    source_tree_sha256: str,
    model_manifest_sha256: str,
) -> dict:
    if (
        not isinstance(gpu_indices, tuple)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("artifact GPU inventory mismatch")
    if not isinstance(environment, dict):
        raise ValueError("artifact environment must be a mapping")
    gpu_inventory = environment.get("gpu_inventory")
    if (
        not isinstance(gpu_inventory, dict)
        or gpu_inventory.get("selected_physical_indices")
        != list(gpu_indices)
    ):
        raise ValueError(
            "environment selected GPU inventory mismatch"
        )
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source files must be a non-empty mapping")
    normalized_sources = {}
    for name, digest in source_files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("source path must be safe and relative")
        normalized_sources[name] = _sha256(
            digest,
            f"source hash {name}",
        )
    source_digest = _sha256(
        source_tree_sha256,
        "source tree",
    )
    model_digest = _sha256(
        model_manifest_sha256,
        "model manifest",
    )
    if model_digest != MODEL_MANIFEST_SHA256:
        raise ValueError(
            "model manifest does not match approved checkpoint"
        )
    derived = derive_artifact(worker_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "claim_scope": CLAIM_SCOPE,
        "limitations": list(LIMITATIONS),
        "source_tree_sha256": source_digest,
        "model_manifest_sha256": model_digest,
        "source_files": normalized_sources,
        "world_size": WORLD_SIZE,
        "gpu_indices": list(gpu_indices),
        "campaign": _campaign_contract(),
        "engine_config": _engine_contract(),
        "environment": copy.deepcopy(environment),
        "cells": derived["cells"],
        "comparisons": derived["comparisons"],
        "campaign_direction": derived[
            "campaign_direction"
        ],
    }


def validate_performance_artifact(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("artifact must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("artifact schema version mismatch")
    if value.get("status") != "PASS":
        raise ValueError("artifact status must be PASS")
    if value.get("classification") != CLASSIFICATION:
        raise ValueError("artifact classification mismatch")
    if value.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("artifact claim scope mismatch")
    if value.get("limitations") != list(LIMITATIONS):
        raise ValueError("artifact limitations mismatch")
    if value.get("world_size") != WORLD_SIZE:
        raise ValueError("artifact world size mismatch")
    gpu_indices = value.get("gpu_indices")
    if (
        not isinstance(gpu_indices, list)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("artifact GPU inventory mismatch")
    if value.get("campaign") != _campaign_contract():
        raise ValueError("artifact campaign contract mismatch")
    if value.get("engine_config") != _engine_contract():
        raise ValueError("artifact engine contract mismatch")
    environment = value.get("environment")
    if (
        not isinstance(environment, dict)
        or not isinstance(
            environment.get("gpu_inventory"),
            dict,
        )
        or environment["gpu_inventory"].get(
            "selected_physical_indices"
        )
        != gpu_indices
    ):
        raise ValueError("artifact environment mismatch")
    source_files = value.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("artifact source inventory is incomplete")
    for name, digest in source_files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("artifact source path is invalid")
        _sha256(digest, f"artifact source hash {name}")
    _sha256(
        value.get("source_tree_sha256"),
        "artifact source tree",
    )
    model_digest = _sha256(
        value.get("model_manifest_sha256"),
        "artifact model manifest",
    )
    if model_digest != MODEL_MANIFEST_SHA256:
        raise ValueError(
            "artifact model manifest does not match authority"
        )
    cells = value.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("artifact cells must be a mapping")
    worker_results = []
    for key in sorted(cells):
        cell = cells[key]
        if not isinstance(cell, dict):
            raise ValueError("artifact cell must be a mapping")
        try:
            worker_results.append({
                field: copy.deepcopy(cell[field])
                for field in (
                    "policy",
                    "batch_size",
                    "prompt_rows",
                    "warmup_runs",
                    "parity_runs",
                    "measured_runs",
                    "tokenizer_identifier",
                    "dtype",
                    "cleanup_receipt",
                )
            })
        except KeyError as error:
            raise ValueError(
                "artifact cell raw evidence is incomplete"
            ) from error
    derived = derive_artifact(worker_results)
    if set(cells) != set(derived["cells"]):
        raise ValueError("artifact cell inventory mismatch")
    for key, derived_cell in derived["cells"].items():
        if not _artifact_values_equivalent(
            cells[key],
            derived_cell,
        ):
            raise ValueError(
                f"artifact aggregate mismatch for {key}"
            )
    if not _artifact_values_equivalent(
        value.get("comparisons"),
        derived["comparisons"],
    ):
        raise ValueError("artifact comparison mismatch")
    if (
        value.get("campaign_direction")
        != derived["campaign_direction"]
    ):
        raise ValueError("artifact campaign direction mismatch")
    return {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "campaign_direction": derived[
            "campaign_direction"
        ],
        "comparisons": derived["comparisons"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_files(
    repo_root: Path,
    source_files: tuple[str, ...],
) -> dict[str, str]:
    repo_root = Path(repo_root)
    if not isinstance(source_files, tuple) or not source_files:
        raise ValueError(
            "source_files must be a non-empty tuple"
        )
    result = {}
    for name in source_files:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("source path must be safe and relative")
        path = repo_root / name
        if not path.is_file():
            raise ValueError(f"source file is missing: {name}")
        result[name] = sha256_file(path)
    return result


def source_tree_sha256(
    repo_root: Path,
    source_files: tuple[str, ...],
) -> str:
    repo_root = Path(repo_root)
    hash_source_files(repo_root, source_files)
    digest = hashlib.sha256()
    for name in sorted(source_files):
        payload = (repo_root / name).read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def model_manifest_sha256(model_path: str) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(
            "model path must be a checkpoint directory"
        )
    manifest = root.parent / "model_manifest.json"
    if not manifest.is_file():
        raise ValueError("approved model manifest is missing")
    return sha256_file(manifest)


def write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _subprocess_worker_runner(
    command,
    *,
    log_path,
    cwd,
) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _default_environment(
    *,
    gpu_indices: tuple[int, ...],
) -> dict:
    try:
        import torch

        torch_version = str(torch.__version__)
        device_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "unavailable"
        )
    except Exception:
        torch_version = "unavailable"
        device_name = "unavailable"
    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "device_name": device_name,
        "gpu_inventory": {
            "selected_physical_indices": list(gpu_indices),
            "campaign_start": [],
            "pre_cells": {},
            "post_cells": {},
        },
    }


def _load_default_verifier():
    path = (
        TOOLS
        / "verify_qwen35_generic_speculative_tp4_16k_performance_gate.py"
    )
    return _load_module(
        "verify_qwen35_generic_speculative_tp4_16k_performance_gate",
        path,
    ).verify_run


def run_campaign(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    output_dir: Path,
    dist_port_base: int,
    master_port_base: int,
    repo_root: Path | None = None,
    worker_script: Path | None = None,
    worker_runner=_subprocess_worker_runner,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    environment: dict | None = None,
    verifier=None,
) -> dict:
    if (
        not isinstance(gpu_indices, tuple)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
    ):
        raise ValueError("campaign GPU inventory mismatch")
    if (
        isinstance(dist_port_base, bool)
        or not isinstance(dist_port_base, int)
        or dist_port_base <= 0
        or isinstance(master_port_base, bool)
        or not isinstance(master_port_base, int)
        or master_port_base <= 0
    ):
        raise ValueError("campaign port bases must be positive")
    repo_root = (
        ROOT if repo_root is None else Path(repo_root)
    )
    worker_script = (
        TOOLS
        / "qwen35_generic_speculative_tp4_16k_performance_worker.py"
        if worker_script is None
        else Path(worker_script)
    )
    output_dir = Path(output_dir)
    failed_dir = output_dir.with_name(
        f"{output_dir.name}.failed"
    )
    if output_dir.exists() or failed_dir.exists():
        raise ValueError(
            "campaign output directory already exists"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        cell_dir = temporary_dir / "cells"
        cell_dir.mkdir()
        worker_results = []
        commands = []
        cell_index = 0
        for batch_size in BATCH_SIZES:
            for policy in POLICIES:
                key = cell_key(policy, batch_size)
                output_path = cell_dir / f"{key}.json"
                log_path = cell_dir / f"{key}.log"
                command = [
                    python_executable,
                    str(worker_script),
                    "--model",
                    model_path,
                    "--gpu-indices",
                    ",".join(str(index) for index in gpu_indices),
                    "--policy",
                    policy,
                    "--batch-size",
                    str(batch_size),
                    "--dist-port",
                    str(dist_port_base + cell_index),
                    "--master-port",
                    str(master_port_base + cell_index),
                    "--out",
                    str(output_path),
                ]
                commands.append(command)
                status = worker_runner(
                    command,
                    log_path=log_path,
                    cwd=repo_root,
                )
                if status != 0:
                    raise RuntimeError(
                        f"TP4 performance worker failed: {key}; "
                        f"status={status}; log={log_path}"
                    )
                if not output_path.is_file():
                    raise RuntimeError(
                        f"TP4 performance worker output is missing: {key}"
                    )
                try:
                    raw_worker = json.loads(
                        output_path.read_text(
                            encoding="utf-8"
                        )
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as error:
                    raise RuntimeError(
                        f"TP4 performance worker output is invalid: {key}"
                    ) from error
                worker_results.append(
                    validate_worker_result(raw_worker)
                )
                cell_index += 1
        source_hashes = hash_source_files(
            repo_root,
            source_files,
        )
        source_digest = source_tree_sha256(
            repo_root,
            source_files,
        )
        model_digest = model_manifest_sha256(model_path)
        if model_digest != MODEL_MANIFEST_SHA256:
            raise RuntimeError(
                "model manifest does not match approved checkpoint"
            )
        command_environment = (
            _default_environment(gpu_indices=gpu_indices)
            if environment is None
            else copy.deepcopy(environment)
        )
        command_environment["worker_commands"] = commands
        artifact = build_performance_artifact(
            worker_results=worker_results,
            environment=command_environment,
            gpu_indices=gpu_indices,
            source_files=source_hashes,
            source_tree_sha256=source_digest,
            model_manifest_sha256=model_digest,
        )
        validate_performance_artifact(artifact)
        result_path = temporary_dir / "result.json"
        write_json_atomic(result_path, artifact)
        write_json_atomic(
            temporary_dir / "source_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_tree_sha256": source_digest,
                "model_manifest_sha256": model_digest,
                "source_files": source_hashes,
                "artifacts": {
                    "result.json": sha256_file(result_path),
                },
            },
        )
        verify = (
            _load_default_verifier()
            if verifier is None
            else verifier
        )
        verification = verify(temporary_dir, repo_root)
        write_json_atomic(
            temporary_dir / "verify.json",
            verification,
        )
        if (
            verification.get("classification") != "PASS"
            or verification.get("failures") != []
        ):
            raise RuntimeError(
                "independent verification failed"
            )
        os.replace(temporary_dir, output_dir)
        return artifact
    except Exception as error:
        write_json_atomic(
            temporary_dir / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "FAIL",
                "classification": CLASSIFICATION,
                "error": str(error),
            },
        )
        if temporary_dir.exists():
            os.replace(temporary_dir, failed_dir)
        raise RuntimeError(
            f"{error}; failed_artifacts={failed_dir}"
        ) from error
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from error
    if (
        len(indices) != WORLD_SIZE
        or len(set(indices)) != WORLD_SIZE
        or any(index < 0 for index in indices)
    ):
        raise argparse.ArgumentTypeError(
            "exactly four distinct GPU indices are required"
        )
    return indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu-indices",
        required=True,
        type=_gpu_indices,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dist-port-base",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--master-port-base",
        required=True,
        type=int,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_campaign(
        model_path=args.model,
        gpu_indices=args.gpu_indices,
        output_dir=Path(args.output_dir),
        dist_port_base=args.dist_port_base,
        master_port_base=args.master_port_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
