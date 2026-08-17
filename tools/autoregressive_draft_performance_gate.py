from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from speculative_runtime_performance_gate import aggregate_measurements


SCHEMA_VERSION = 3
CLASSIFICATION = "PILOT_ONLY"
POLICIES = ("target", "learned")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 256
MAX_OUTPUT_TOKENS = 16
WARMUP_RUNS = 1
MEASURED_RUNS = 3
TENSOR_PARALLEL_SIZE = 4
MAX_PROPOSAL_TOKENS = 4
PROPOSAL_KV_COUNTER_KEYS = (
    "h2d_entries",
    "h2d_bytes",
    "d2h_entries",
    "d2h_bytes",
)
RUNTIME_STAGE_TIMING_KEYS = (
    "first_target_batch_ms",
    "draft_proposal_ms",
    "reserve_blocks_ms",
    "tail_batch_ms",
    "kv_materialize_ms",
    "accept_sample_ms",
    "commit_metadata_ms",
)
DRAFT_EXECUTOR_TIMING_KEYS = (
    "prompt_bootstrap",
    "proposal_forward",
    "proposal_finalize",
)
PROPOSAL_FORWARD_DETAIL_KEYS = (
    "setup",
    "backend_submit",
    "selection_collective",
    "decode_authority",
    "token_readback",
    "materialize_register",
)
PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS = 1e-9
LIMITATIONS = (
    "classification remains PILOT_ONLY",
    "256-token prompts do not establish 4K, 16K, or 32K performance",
    "direct Proposal-KV allocation does not establish offload benefit",
    "no second model structure evidence",
    "three measured runs do not establish statistical significance",
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/autoregressive_draft_executor.py",
    "tinyvllm/engine/qwen3_draft_backend.py",
    "tinyvllm/engine/qwen3_draft_proposal_kv.py",
    "tinyvllm/engine/proposal_kv_allocator.py",
    "tinyvllm/engine/proposal_kv_cache.py",
    "tinyvllm/engine/proposal_kv_lifecycle.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/layers/attention.py",
    "tools/autoregressive_draft_tp4_engine_gate.py",
    "tools/autoregressive_draft_performance_gate.py",
    "tools/autoregressive_draft_performance_worker.py",
    "tools/verify_autoregressive_draft_performance_gate.py",
)


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def proposal_slot_capacity_for_batch(batch_size: int) -> int:
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return batch_size * (
        PROMPT_TOKENS
        + MAX_OUTPUT_TOKENS
        + MAX_PROPOSAL_TOKENS
    )


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


def _non_negative_number(value: object, name: str) -> float:
    normalized = _finite_number(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _validate_prompt_rows(
    rows: object,
    *,
    batch_size: int,
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != batch_size:
        raise ValueError("prompt row inventory must match batch size")
    for prompt_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("prompt row must be a mapping")
        if row.get("prompt_index") != prompt_index:
            raise ValueError("prompt indices must be contiguous")
        if row.get("token_count") != PROMPT_TOKENS:
            raise ValueError("prompt rows must contain 256 tokens")
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
            raise ValueError("prompt token IDs are invalid")
        expected_hash = hashlib.sha256(
            json.dumps(
                token_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if row.get("sha256") != expected_hash:
            raise ValueError("prompt token hash mismatch")
    return rows


def _validate_rank_rows(
    rows: object,
    *,
    name: str,
    keys: tuple[str, ...],
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != TENSOR_PARALLEL_SIZE:
        raise ValueError(
            f"{name} requires four distributed rank rows"
        )
    ranks = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{name} row must be a mapping")
        rank = _non_negative_integer(
            row.get("rank"),
            f"{name} rank",
        )
        ranks.add(rank)
        for key in keys:
            _non_negative_integer(
                row.get(key),
                f"{name} {key}",
            )
    if ranks != set(range(TENSOR_PARALLEL_SIZE)):
        raise ValueError(f"{name} rank inventory mismatch")
    return rows


def _validate_stage_timing(
    value: object,
    *,
    policy: str,
    measured: bool,
    proposed_tokens: int,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError("stage timing must be a mapping")
    steps = value.get("steps")
    if not isinstance(steps, list):
        raise ValueError("stage timing steps must be a list")
    if value.get("step_count") != len(steps):
        raise ValueError("stage timing step count mismatch")
    totals = value.get("totals_ms")
    if (
        not isinstance(totals, dict)
        or set(totals) != set(RUNTIME_STAGE_TIMING_KEYS)
    ):
        raise ValueError("stage timing totals inventory mismatch")
    expected_totals = {
        key: 0.0 for key in RUNTIME_STAGE_TIMING_KEYS
    }
    for step_index, row in enumerate(steps):
        if not isinstance(row, dict):
            raise ValueError("stage timing row must be a mapping")
        if row.get("step_index") != step_index:
            raise ValueError("stage timing indices must be contiguous")
        timing_ms = row.get("timing_ms")
        if (
            not isinstance(timing_ms, dict)
            or set(timing_ms) != set(RUNTIME_STAGE_TIMING_KEYS)
        ):
            raise ValueError("stage timing inventory mismatch")
        for key in RUNTIME_STAGE_TIMING_KEYS:
            expected_totals[key] += _non_negative_number(
                timing_ms.get(key),
                f"stage timing {key}",
            )
    for key in RUNTIME_STAGE_TIMING_KEYS:
        if not math.isclose(
            _non_negative_number(
                totals.get(key),
                f"stage timing total {key}",
            ),
            expected_totals[key],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("stage timing total mismatch")
    if policy == "target" and steps:
        raise ValueError("target run contains stage timing evidence")
    if (
        policy == "learned"
        and measured
        and proposed_tokens > 0
        and not steps
    ):
        raise ValueError("learned run lacks stage timing evidence")
    return value


def _validate_draft_executor_timing(
    value: object,
    *,
    policy: str,
    measured: bool,
    proposed_tokens: int,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError("executor timing must be a mapping")
    rows = value.get("ranks")
    if (
        not isinstance(rows, list)
        or len(rows) != TENSOR_PARALLEL_SIZE
    ):
        raise ValueError("executor timing requires four rank rows")
    ranks = set()
    expected_max = {
        key: 0.0 for key in DRAFT_EXECUTOR_TIMING_KEYS
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("executor timing row must be a mapping")
        rank = _non_negative_integer(
            row.get("rank"),
            "executor timing rank",
        )
        ranks.add(rank)
        if set(row) != {
            "rank",
            *DRAFT_EXECUTOR_TIMING_KEYS,
        }:
            raise ValueError("executor timing inventory mismatch")
        for key in DRAFT_EXECUTOR_TIMING_KEYS:
            normalized = _non_negative_number(
                row.get(key),
                f"executor timing {key}",
            )
            expected_max[key] = max(
                expected_max[key],
                normalized,
            )
    if ranks != set(range(TENSOR_PARALLEL_SIZE)):
        raise ValueError("executor timing rank inventory mismatch")
    max_rank_ms = value.get("max_rank_ms")
    if (
        not isinstance(max_rank_ms, dict)
        or set(max_rank_ms) != set(DRAFT_EXECUTOR_TIMING_KEYS)
    ):
        raise ValueError("executor timing max inventory mismatch")
    for key in DRAFT_EXECUTOR_TIMING_KEYS:
        if not math.isclose(
            _non_negative_number(
                max_rank_ms.get(key),
                f"executor timing max {key}",
            ),
            expected_max[key],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("executor timing max mismatch")
    if policy == "target" and any(
        expected_max[key] != 0.0
        for key in DRAFT_EXECUTOR_TIMING_KEYS
    ):
        raise ValueError("target run contains executor timing evidence")
    if (
        policy == "learned"
        and measured
        and proposed_tokens > 0
        and expected_max["proposal_forward"] <= 0.0
    ):
        raise ValueError("learned run lacks executor timing evidence")
    return value


def _validate_draft_executor_proposal_detail(
    value: object,
    *,
    policy: str,
    measured: bool,
    proposed_tokens: int,
    executor_timing: dict,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError("executor proposal detail must be a mapping")
    rows = value.get("ranks")
    if (
        not isinstance(rows, list)
        or len(rows) != TENSOR_PARALLEL_SIZE
    ):
        raise ValueError(
            "executor proposal detail requires four rank rows"
        )
    ranks = set()
    rows_by_rank = {}
    expected_max = {
        key: 0.0 for key in PROPOSAL_FORWARD_DETAIL_KEYS
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "executor proposal detail row must be a mapping"
            )
        rank = _non_negative_integer(
            row.get("rank"),
            "executor proposal detail rank",
        )
        ranks.add(rank)
        rows_by_rank[rank] = row
        if set(row) != {
            "rank",
            *PROPOSAL_FORWARD_DETAIL_KEYS,
        }:
            raise ValueError(
                "executor proposal detail inventory mismatch"
            )
        for key in PROPOSAL_FORWARD_DETAIL_KEYS:
            normalized = _non_negative_number(
                row.get(key),
                f"executor proposal detail {key}",
            )
            expected_max[key] = max(
                expected_max[key],
                normalized,
            )
    if ranks != set(range(TENSOR_PARALLEL_SIZE)):
        raise ValueError(
            "executor proposal detail rank inventory mismatch"
        )
    max_rank_ms = value.get("max_rank_ms")
    if (
        not isinstance(max_rank_ms, dict)
        or set(max_rank_ms) != set(PROPOSAL_FORWARD_DETAIL_KEYS)
    ):
        raise ValueError(
            "executor proposal detail max inventory mismatch"
        )
    for key in PROPOSAL_FORWARD_DETAIL_KEYS:
        if not math.isclose(
            _non_negative_number(
                max_rank_ms.get(key),
                f"executor proposal detail max {key}",
            ),
            expected_max[key],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "executor proposal detail max mismatch"
            )
    timing_rows = executor_timing["ranks"]
    critical_timing_row = max(
        timing_rows,
        key=lambda row: row["proposal_forward"],
    )
    expected_critical_rank = critical_timing_row["rank"]
    if value.get("critical_rank") != expected_critical_rank:
        raise ValueError(
            "executor proposal detail critical rank mismatch"
        )
    critical_rank_ms = value.get("critical_rank_ms")
    if (
        not isinstance(critical_rank_ms, dict)
        or set(critical_rank_ms) != set(PROPOSAL_FORWARD_DETAIL_KEYS)
    ):
        raise ValueError(
            "executor proposal detail critical inventory mismatch"
        )
    expected_critical = {
        key: rows_by_rank[expected_critical_rank][key]
        for key in PROPOSAL_FORWARD_DETAIL_KEYS
    }
    for key in PROPOSAL_FORWARD_DETAIL_KEYS:
        if not math.isclose(
            _non_negative_number(
                critical_rank_ms.get(key),
                f"executor proposal detail critical {key}",
            ),
            expected_critical[key],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "executor proposal detail critical mismatch"
            )
    expected_sum = sum(expected_critical.values())
    if not math.isclose(
        _non_negative_number(
            value.get("detail_sum_ms"),
            "executor proposal detail sum",
        ),
        expected_sum,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("executor proposal detail sum mismatch")
    raw_residual = (
        critical_timing_row["proposal_forward"] - expected_sum
    )
    if raw_residual < -PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS:
        raise ValueError(
            "executor proposal detail residual is negative"
        )
    expected_residual = max(0.0, raw_residual)
    residual = _finite_number(
        value.get("residual_ms"),
        "executor proposal detail residual",
    )
    if residual < -PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS:
        raise ValueError(
            "executor proposal detail residual is negative"
        )
    if not math.isclose(
        max(0.0, residual),
        expected_residual,
        rel_tol=1e-12,
        abs_tol=PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS,
    ):
        raise ValueError(
            "executor proposal detail residual mismatch"
        )
    if policy == "target" and any(
        expected_max[key] != 0.0
        for key in PROPOSAL_FORWARD_DETAIL_KEYS
    ):
        raise ValueError(
            "target run contains executor proposal detail evidence"
        )
    if (
        policy == "learned"
        and measured
        and proposed_tokens > 0
        and not any(
            expected_max[key] > 0.0
            for key in (
                "backend_submit",
                "selection_collective",
                "decode_authority",
                "token_readback",
            )
        )
    ):
        raise ValueError(
            "learned run lacks executor proposal detail evidence"
        )
    return value


def _validate_run(
    run: object,
    *,
    policy: str,
    batch_size: int,
    measured: bool,
) -> dict:
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    if measured:
        _non_negative_integer(run.get("repeat"), "repeat")
    outputs = run.get("outputs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != batch_size
        or any(
            not isinstance(tokens, list)
            or len(tokens) != MAX_OUTPUT_TOKENS
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in tokens
            )
            for tokens in outputs
        )
    ):
        raise ValueError("run outputs must contain exactly 16 tokens")

    timing = run.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("run timing must be a mapping")
    if timing.get("request_count") != batch_size:
        raise ValueError("timing request count mismatch")
    if timing.get("total_output_tokens") != (
        batch_size * MAX_OUTPUT_TOKENS
    ):
        raise ValueError("timing output token count mismatch")
    _finite_number(timing.get("batch_elapsed_s"), "batch elapsed")
    _finite_number(
        timing.get("batch_token_throughput_tps"),
        "output throughput",
    )
    per_request = timing.get("per_request")
    if not isinstance(per_request, list) or len(per_request) != batch_size:
        raise ValueError("timing per-request inventory mismatch")
    for row in per_request:
        if not isinstance(row, dict):
            raise ValueError("timing row must be a mapping")
        if row.get("output_tokens") != MAX_OUTPUT_TOKENS:
            raise ValueError("timing row output count mismatch")
        for key in ("ttft_s", "tpot_s", "completion_latency_s"):
            _finite_number(row.get(key), key)

    runtime = run.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime evidence must be a mapping")
    proposed = _non_negative_integer(
        runtime.get("proposed_tokens"),
        "proposed tokens",
    )
    accepted = _non_negative_integer(
        runtime.get("accepted_draft_tokens"),
        "accepted draft tokens",
    )
    if accepted > proposed:
        raise ValueError("accepted draft tokens exceed proposals")
    if policy == "learned" and measured and accepted <= 0:
        raise ValueError(
            "learned measured run lacks accepted draft tokens"
        )
    if policy == "target" and (proposed != 0 or accepted != 0):
        raise ValueError("target run contains proposal evidence")
    expected_rate = accepted / proposed if proposed else 0.0
    if not math.isclose(
        _finite_number(
            runtime.get("acceptance_rate"),
            "acceptance rate",
        ),
        expected_rate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("acceptance rate mismatch")
    _validate_stage_timing(
        runtime.get("stage_timing"),
        policy=policy,
        measured=measured,
        proposed_tokens=proposed,
    )
    executor_timing = _validate_draft_executor_timing(
        runtime.get("draft_executor_timing"),
        policy=policy,
        measured=measured,
        proposed_tokens=proposed,
    )
    _validate_draft_executor_proposal_detail(
        runtime.get("draft_executor_proposal_detail"),
        policy=policy,
        measured=measured,
        proposed_tokens=proposed,
        executor_timing=executor_timing,
    )

    memory = run.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("memory evidence must be a mapping")
    memory_rows = _validate_rank_rows(
        memory.get("ranks"),
        name="memory",
        keys=(
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ),
    )
    expected_peak_allocated = max(
        row["peak_allocated_bytes"] for row in memory_rows
    )
    expected_peak_reserved = max(
        row["peak_reserved_bytes"] for row in memory_rows
    )
    if (
        memory.get("peak_allocated_bytes")
        != expected_peak_allocated
        or memory.get("peak_reserved_bytes")
        != expected_peak_reserved
    ):
        raise ValueError("memory peak summary mismatch")

    proposal_kv = run.get("proposal_kv")
    if not isinstance(proposal_kv, dict):
        raise ValueError("Proposal-KV evidence must be a mapping")
    proposal_rows = _validate_rank_rows(
        proposal_kv.get("ranks"),
        name="Proposal-KV",
        keys=PROPOSAL_KV_COUNTER_KEYS,
    )
    totals = proposal_kv.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("Proposal-KV totals must be a mapping")
    for key in PROPOSAL_KV_COUNTER_KEYS:
        expected_total = sum(row[key] for row in proposal_rows)
        if totals.get(key) != expected_total:
            raise ValueError(f"Proposal-KV {key} total mismatch")
    return run


def validate_worker_result(
    worker_result: object,
    *,
    expected_warmup_runs: int = WARMUP_RUNS,
    expected_measured_runs: int = MEASURED_RUNS,
) -> dict:
    _positive_integer(expected_warmup_runs, "warmup runs")
    _positive_integer(expected_measured_runs, "measured runs")
    if not isinstance(worker_result, dict):
        raise ValueError("worker result must be a mapping")
    policy = worker_result.get("policy")
    if policy not in POLICIES:
        raise ValueError("worker policy is invalid")
    batch_size = worker_result.get("batch_size")
    if batch_size not in BATCH_SIZES:
        raise ValueError("worker batch size is invalid")
    if worker_result.get("tensor_parallel_size") != TENSOR_PARALLEL_SIZE:
        raise ValueError("worker tensor parallel size must be four")
    if worker_result.get("proposal_kv_allocator") != "direct":
        raise ValueError("worker Proposal-KV allocator must be direct")
    expected_slot_capacity = proposal_slot_capacity_for_batch(
        batch_size
    )
    if (
        worker_result.get("proposal_slot_capacity")
        != expected_slot_capacity
    ):
        raise ValueError(
            "worker Proposal-KV slot capacity must match the "
            "workload-derived bound"
        )
    _validate_prompt_rows(
        worker_result.get("prompt_rows"),
        batch_size=batch_size,
    )
    warmup_runs = worker_result.get("warmup_runs")
    if (
        not isinstance(warmup_runs, list)
        or len(warmup_runs) != expected_warmup_runs
    ):
        raise ValueError(
            "worker requires one warmup run"
            if expected_warmup_runs == WARMUP_RUNS
            else "worker warmup run count mismatch"
        )
    measured_runs = worker_result.get("measured_runs")
    if (
        not isinstance(measured_runs, list)
        or len(measured_runs) != expected_measured_runs
    ):
        raise ValueError(
            "worker requires exactly three measured runs"
            if expected_measured_runs == MEASURED_RUNS
            else "worker measured run count mismatch"
        )
    for run in warmup_runs:
        _validate_run(
            run,
            policy=policy,
            batch_size=batch_size,
            measured=False,
        )
    for repeat, run in enumerate(measured_runs):
        _validate_run(
            run,
            policy=policy,
            batch_size=batch_size,
            measured=True,
        )
        if run["repeat"] != repeat:
            raise ValueError("measured repeat indices must be contiguous")
    for key in (
        "target_checkpoint_identifier",
        "tokenizer_identifier",
        "dtype",
    ):
        if (
            not isinstance(worker_result.get(key), str)
            or not worker_result[key]
        ):
            raise ValueError(f"worker {key} is invalid")
    draft_identifier = worker_result.get("draft_checkpoint_identifier")
    if policy == "learned":
        if not isinstance(draft_identifier, str) or not draft_identifier:
            raise ValueError("learned worker draft checkpoint is missing")
    elif draft_identifier is not None:
        raise ValueError("target worker must not load a draft checkpoint")
    return copy.deepcopy(worker_result)


def _aggregate_worker(worker: dict) -> dict:
    values = {
        "ttft_s": [],
        "tpot_s": [],
        "e2e_s": [],
        "output_throughput_tps": [],
        "peak_allocated_bytes": [],
        "peak_reserved_bytes": [],
        "proposal_kv_h2d_bytes": [],
        "proposal_kv_d2h_bytes": [],
        "proposed_tokens": [],
        "accepted_draft_tokens": [],
        "acceptance_rate": [],
        **{
            f"stage_{key}": []
            for key in RUNTIME_STAGE_TIMING_KEYS
        },
        **{
            f"executor_{key}_ms": []
            for key in DRAFT_EXECUTOR_TIMING_KEYS
        },
        **{
            f"executor_detail_{key}_ms": []
            for key in PROPOSAL_FORWARD_DETAIL_KEYS
        },
        "executor_detail_sum_ms": [],
        "executor_detail_residual_ms": [],
    }
    for run in worker["measured_runs"]:
        per_request = run["timing"]["per_request"]
        values["ttft_s"].append(statistics.median(
            row["ttft_s"] for row in per_request
        ))
        values["tpot_s"].append(statistics.median(
            row["tpot_s"] for row in per_request
        ))
        values["e2e_s"].append(statistics.median(
            row["completion_latency_s"] for row in per_request
        ))
        values["output_throughput_tps"].append(
            run["timing"]["batch_token_throughput_tps"]
        )
        values["peak_allocated_bytes"].append(
            run["memory"]["peak_allocated_bytes"]
        )
        values["peak_reserved_bytes"].append(
            run["memory"]["peak_reserved_bytes"]
        )
        values["proposal_kv_h2d_bytes"].append(
            run["proposal_kv"]["totals"]["h2d_bytes"]
        )
        values["proposal_kv_d2h_bytes"].append(
            run["proposal_kv"]["totals"]["d2h_bytes"]
        )
        values["proposed_tokens"].append(
            run["runtime"]["proposed_tokens"]
        )
        values["accepted_draft_tokens"].append(
            run["runtime"]["accepted_draft_tokens"]
        )
        values["acceptance_rate"].append(
            run["runtime"]["acceptance_rate"]
        )
        for key in RUNTIME_STAGE_TIMING_KEYS:
            values[f"stage_{key}"].append(
                run["runtime"]["stage_timing"][
                    "totals_ms"
                ][key]
            )
        for key in DRAFT_EXECUTOR_TIMING_KEYS:
            values[f"executor_{key}_ms"].append(
                run["runtime"]["draft_executor_timing"][
                    "max_rank_ms"
                ][key]
            )
        proposal_detail = run["runtime"][
            "draft_executor_proposal_detail"
        ]
        for key in PROPOSAL_FORWARD_DETAIL_KEYS:
            values[
                f"executor_detail_{key}_ms"
            ].append(proposal_detail["critical_rank_ms"][key])
        values["executor_detail_sum_ms"].append(
            proposal_detail["detail_sum_ms"]
        )
        values["executor_detail_residual_ms"].append(
            proposal_detail["residual_ms"]
        )
    return {
        key: aggregate_measurements(measurements)
        for key, measurements in values.items()
    }


def _derive_cells(worker_results: object) -> dict[str, dict]:
    if (
        not isinstance(worker_results, list)
        or len(worker_results) != 4
    ):
        raise ValueError("artifact requires exactly four worker cells")
    cells = {}
    identity = None
    for raw_worker in worker_results:
        worker = validate_worker_result(raw_worker)
        cell_key = f"{worker['policy']}:b{worker['batch_size']}"
        if cell_key in cells:
            raise ValueError("duplicate worker cell")
        current_identity = (
            worker["target_checkpoint_identifier"],
            worker["tokenizer_identifier"],
            worker["dtype"],
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("worker target identities differ")
        cells[cell_key] = {
            **worker,
            "aggregate": _aggregate_worker(worker),
        }
    expected = {
        f"{policy}:b{batch_size}"
        for policy in POLICIES
        for batch_size in BATCH_SIZES
    }
    if set(cells) != expected:
        raise ValueError("worker cell inventory mismatch")
    for batch_size in BATCH_SIZES:
        target = cells[f"target:b{batch_size}"]
        learned = cells[f"learned:b{batch_size}"]
        if target["prompt_rows"] != learned["prompt_rows"]:
            raise ValueError(f"batch {batch_size} prompt parity failed")
        for repeat in range(MEASURED_RUNS):
            if (
                target["measured_runs"][repeat]["outputs"]
                != learned["measured_runs"][repeat]["outputs"]
            ):
                raise ValueError(
                    f"batch {batch_size} exact repeat parity failed"
                )
    return cells


def _derive_directions(
    cells: dict[str, dict],
) -> tuple[dict[str, str], str]:
    batch_directions = {}
    for batch_size in BATCH_SIZES:
        target = cells[f"target:b{batch_size}"]["aggregate"]
        learned = cells[f"learned:b{batch_size}"]["aggregate"]
        improved = (
            learned["tpot_s"]["median"]
            < target["tpot_s"]["median"]
            and learned["e2e_s"]["median"]
            < target["e2e_s"]["median"]
            and learned["output_throughput_tps"]["median"]
            > target["output_throughput_tps"]["median"]
        )
        regressed = (
            learned["tpot_s"]["median"]
            > target["tpot_s"]["median"]
            and learned["e2e_s"]["median"]
            > target["e2e_s"]["median"]
            and learned["output_throughput_tps"]["median"]
            < target["output_throughput_tps"]["median"]
        )
        batch_directions[str(batch_size)] = (
            "IMPROVED"
            if improved
            else "REGRESSED"
            if regressed
            else "MIXED"
        )
    if all(
        direction == "IMPROVED"
        for direction in batch_directions.values()
    ):
        direction = "POSITIVE"
    elif any(
        direction == "REGRESSED"
        for direction in batch_directions.values()
    ):
        direction = "NEGATIVE"
    else:
        direction = "MIXED"
    return batch_directions, direction


def build_performance_artifact(
    *,
    worker_results: list[dict],
    environment: dict,
    source_files: dict[str, str],
) -> dict:
    if not isinstance(environment, dict):
        raise ValueError("environment must be a mapping")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source files must be a non-empty mapping")
    normalized_sources = {
        path: _sha256(digest, f"source hash {path}")
        for path, digest in source_files.items()
        if isinstance(path, str) and path
    }
    if len(normalized_sources) != len(source_files):
        raise ValueError("source path must be non-empty")
    cells = _derive_cells(worker_results)
    batch_directions, direction = _derive_directions(cells)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "direction": direction,
        "batch_directions": batch_directions,
        "campaign": {
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "prompt_tokens": PROMPT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "batch_sizes": list(BATCH_SIZES),
            "policies": list(POLICIES),
            "temperature": 0.0,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
            "proposal_kv_allocator": "direct",
            "proposal_slot_capacity_by_batch": {
                str(batch_size): (
                    proposal_slot_capacity_for_batch(batch_size)
                )
                for batch_size in BATCH_SIZES
            },
        },
        "environment": copy.deepcopy(environment),
        "cells": cells,
        "source_files": normalized_sources,
        "limitations": list(LIMITATIONS),
    }


def _equivalent(stored: object, derived: object) -> bool:
    if isinstance(stored, bool) or isinstance(derived, bool):
        return stored is derived
    if isinstance(stored, (int, float)) or isinstance(
        derived,
        (int, float),
    ):
        if not isinstance(stored, (int, float)) or not isinstance(
            derived,
            (int, float),
        ):
            return False
        return math.isclose(
            float(stored),
            float(derived),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if isinstance(stored, dict) or isinstance(derived, dict):
        return (
            isinstance(stored, dict)
            and isinstance(derived, dict)
            and set(stored) == set(derived)
            and all(
                _equivalent(stored[key], derived[key])
                for key in stored
            )
        )
    if isinstance(stored, list) or isinstance(derived, list):
        return (
            isinstance(stored, list)
            and isinstance(derived, list)
            and len(stored) == len(derived)
            and all(
                _equivalent(left, right)
                for left, right in zip(stored, derived)
            )
        )
    return stored == derived


def validate_performance_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a mapping")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("artifact schema version mismatch")
    if artifact.get("status") != "PASS":
        raise ValueError("artifact status must be PASS")
    if artifact.get("classification") != CLASSIFICATION:
        raise ValueError(
            "artifact classification must remain PILOT_ONLY"
        )
    cells = artifact.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("artifact cells must be a mapping")
    worker_results = []
    for cell in cells.values():
        if not isinstance(cell, dict):
            raise ValueError("artifact cell must be a mapping")
        worker_results.append({
            key: copy.deepcopy(cell[key])
            for key in (
                "policy",
                "batch_size",
                "prompt_rows",
                "warmup_runs",
                "measured_runs",
                "target_checkpoint_identifier",
                "draft_checkpoint_identifier",
                "tokenizer_identifier",
                "dtype",
                "tensor_parallel_size",
                "proposal_kv_allocator",
                "proposal_slot_capacity",
            )
        })
    derived_cells = _derive_cells(worker_results)
    for key, derived in derived_cells.items():
        if key not in cells or not _equivalent(
            cells[key].get("aggregate"),
            derived["aggregate"],
        ):
            raise ValueError(f"artifact aggregate mismatch for {key}")
    batch_directions, direction = _derive_directions(derived_cells)
    if artifact.get("batch_directions") != batch_directions:
        raise ValueError("artifact batch direction mismatch")
    if artifact.get("direction") != direction:
        raise ValueError("artifact direction mismatch")
    source_files = artifact.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("artifact source files must be non-empty")
    for path, digest in source_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("artifact source path is invalid")
        _sha256(digest, f"artifact source hash {path}")
    return {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "direction": direction,
        "batch_directions": batch_directions,
    }


def hash_source_files(
    *,
    repo_root: Path,
    source_files: tuple[str, ...],
) -> dict[str, str]:
    repo_root = Path(repo_root)
    if (
        not isinstance(source_files, tuple)
        or not source_files
    ):
        raise ValueError("source_files must be a non-empty tuple")
    result = {}
    for relative_path in source_files:
        path = Path(relative_path)
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("source path must be safe and relative")
        source_path = repo_root / path
        if not source_path.is_file():
            raise ValueError(f"source file is missing: {relative_path}")
        result[relative_path] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    return result


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _subprocess_worker_runner(
    command,
    *,
    log_path,
    cwd,
) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _default_environment(
    *,
    target_model: str,
    draft_model: str,
    command: list[str],
) -> dict:
    try:
        import torch

        torch_version = str(torch.__version__)
        device_names = (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        )
    except Exception:
        torch_version = "unavailable"
        device_names = []
    return {
        "target_model_path": str(Path(target_model).resolve()),
        "draft_model_path": str(Path(draft_model).resolve()),
        "device_names": device_names,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "command": list(command),
    }


def run_performance_gate(
    *,
    target_model: str,
    draft_model: str,
    output_path: Path,
    repo_root: Path,
    worker_script: Path,
    worker_runner=_subprocess_worker_runner,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    environment: dict | None = None,
) -> dict:
    repo_root = Path(repo_root)
    output_path = Path(output_path)
    worker_script = Path(worker_script)
    worker_root = output_path.parent / "workers"
    log_root = output_path.parent / "logs"
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    worker_results = []
    command_prefix = [
        python_executable,
        str(worker_script),
        "--target-model",
        target_model,
        "--draft-model",
        draft_model,
    ]
    for policy in POLICIES:
        for batch_size in BATCH_SIZES:
            cell_key = f"{policy}:b{batch_size}"
            worker_output = (
                worker_root / f"{policy}-b{batch_size}.json"
            )
            worker_log = log_root / f"{policy}-b{batch_size}.log"
            command = [
                *command_prefix,
                "--policy",
                policy,
                "--batch-size",
                str(batch_size),
                "--out",
                str(worker_output),
            ]
            return_code = worker_runner(
                command,
                log_path=worker_log,
                cwd=repo_root,
            )
            if return_code != 0:
                raise RuntimeError(
                    f"worker {cell_key} failed with exit "
                    f"code {return_code}"
                )
            if not worker_output.is_file():
                raise RuntimeError(
                    f"worker {cell_key} did not produce JSON"
                )
            try:
                worker_result = json.loads(
                    worker_output.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"worker {cell_key} JSON is invalid"
                ) from error
            worker_results.append(worker_result)
    if environment is None:
        environment = _default_environment(
            target_model=target_model,
            draft_model=draft_model,
            command=command_prefix,
        )
    artifact = build_performance_artifact(
        worker_results=worker_results,
        environment=environment,
        source_files=hash_source_files(
            repo_root=repo_root,
            source_files=source_files,
        ),
    )
    validate_performance_artifact(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
    )
    parser.add_argument(
        "--worker-script",
        default=str(
            TOOLS_ROOT
            / "autoregressive_draft_performance_worker.py"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_performance_gate(
        target_model=args.target_model,
        draft_model=args.draft_model,
        output_path=Path(args.out),
        repo_root=Path(args.repo_root),
        worker_script=Path(args.worker_script),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
