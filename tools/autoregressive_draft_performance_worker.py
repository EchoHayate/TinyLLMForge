from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
for search_path in (TOOLS_ROOT, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from autoregressive_draft_performance_gate import (
    BATCH_SIZES,
    DRAFT_EXECUTOR_TIMING_KEYS,
    MAX_OUTPUT_TOKENS,
    MAX_PROPOSAL_TOKENS,
    MEASURED_RUNS,
    POLICIES,
    PROMPT_TOKENS,
    PROPOSAL_FORWARD_DETAIL_KEYS,
    PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS,
    PROPOSAL_KV_COUNTER_KEYS,
    RUNTIME_STAGE_TIMING_KEYS,
    TENSOR_PARALLEL_SIZE,
    WARMUP_RUNS,
    proposal_slot_capacity_for_batch,
    write_json_atomic,
)
from speculative_runtime_performance_gate import build_run_metrics


DEFAULT_PROMPT_SEEDS = (
    "A deterministic systems trace repeats alpha beta gamma delta. ",
    "The controlled workload cycles north east south west. ",
    "For exact reproducibility repeat one two three four five. ",
    "The benchmark sequence is red green blue amber violet. ",
)
COMMAND_TIMELINE_MAX_ROWS = 8192


def build_prompt_token_batches(
    tokenizer,
    *,
    batch_size: int,
    prompt_tokens: int = PROMPT_TOKENS,
) -> list[dict]:
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    rows = []
    for prompt_index, seed in enumerate(
        DEFAULT_PROMPT_SEEDS[:batch_size]
    ):
        encoded = tokenizer.encode(
            seed,
            add_special_tokens=False,
        )
        if (
            not isinstance(encoded, (list, tuple))
            or not encoded
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in encoded
            )
        ):
            raise ValueError("prompt seed produced invalid token IDs")
        repeats = (prompt_tokens + len(encoded) - 1) // len(encoded)
        token_ids = (list(encoded) * repeats)[:prompt_tokens]
        rows.append({
            "prompt_index": prompt_index,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "sha256": hashlib.sha256(
                json.dumps(
                    token_ids,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        })
    return rows


def _rank_rows(rows, *, name: str) -> tuple[dict, ...]:
    if (
        not isinstance(rows, tuple)
        or len(rows) != TENSOR_PARALLEL_SIZE
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(f"{name} requires four rank rows")
    normalized = tuple(dict(row) for row in rows)
    if tuple(row.get("rank") for row in normalized) != tuple(range(4)):
        raise ValueError(f"{name} rank inventory mismatch")
    return normalized


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _request_set_sha256(prompt_rows: list[dict]) -> str:
    return _canonical_sha256([
        row["token_ids"]
        for row in prompt_rows
    ])


def _identity_nonnegative_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} is malformed")
    return value


def _identity_sha256(value, name, *, optional=False):
    if optional and value is None:
        return value
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is malformed")
    return value


def _zero_evidence_counter(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} is malformed")
    return value == 0


def _command_transport_identity(
    row,
    *,
    expected_rank,
    expected_repeat_index,
    expected_request_set_sha256,
):
    required = (
        "rank",
        "command_id",
        "method_name",
        "requires_ack",
        "engine_step_id",
        "repeat_index",
        "request_set_sha256",
        "batch_kind",
        "speculative_selected_sequence_ids_sha256",
        "dispatch_started_monotonic_ns",
        "dispatch_published_monotonic_ns",
    )
    if (
        not isinstance(row, dict)
        or any(name not in row for name in required)
    ):
        raise ValueError("command identity is malformed")
    rank = _identity_nonnegative_int(
        row["rank"],
        "command identity",
    )
    if rank != expected_rank:
        raise ValueError("command identity is malformed")
    command_id = _identity_nonnegative_int(
        row["command_id"],
        "command identity",
    )
    engine_step_id = _identity_nonnegative_int(
        row["engine_step_id"],
        "command identity",
    )
    if not isinstance(row["method_name"], str) or not row["method_name"]:
        raise ValueError("command identity is malformed")
    if not isinstance(row["requires_ack"], bool):
        raise ValueError("command identity is malformed")
    if not isinstance(row["batch_kind"], str) or not row["batch_kind"]:
        raise ValueError("command identity is malformed")
    repeat_index = _identity_nonnegative_int(
        row["repeat_index"],
        "command identity",
    )
    if repeat_index != expected_repeat_index:
        raise ValueError("command timeline repeat identity mismatch")
    request_set_sha256 = _identity_sha256(
        row["request_set_sha256"],
        "command request digest",
    )
    if request_set_sha256 != expected_request_set_sha256:
        raise ValueError("command timeline request digest mismatch")
    selected_sha256 = _identity_sha256(
        row["speculative_selected_sequence_ids_sha256"],
        "command selected-sequence digest",
        optional=True,
    )
    dispatch_started = _identity_nonnegative_int(
        row["dispatch_started_monotonic_ns"],
        "command identity",
    )
    dispatch_published = _identity_nonnegative_int(
        row["dispatch_published_monotonic_ns"],
        "command identity",
    )
    if dispatch_published < dispatch_started:
        raise ValueError("command identity is malformed")
    return (
        command_id,
        row["method_name"],
        row["requires_ack"],
        engine_step_id,
        repeat_index,
        request_set_sha256,
        row["batch_kind"],
        selected_sha256,
        dispatch_started,
        dispatch_published,
    )


def _timeline_row_identity(
    row,
    *,
    kind,
    expected_rank,
    expected_repeat_index,
    expected_request_set_sha256,
):
    required = (
        "rank",
        "command_id",
        "engine_step_id",
        "repeat_index",
        "request_set_sha256",
        "speculative_selected_sequence_ids_sha256",
    )
    if (
        not isinstance(row, dict)
        or any(name not in row for name in required)
    ):
        raise ValueError(f"{kind} identity is malformed")
    rank = _identity_nonnegative_int(
        row["rank"],
        f"{kind} identity",
    )
    if rank != expected_rank:
        raise ValueError(f"{kind} identity is malformed")
    command_id = _identity_nonnegative_int(
        row["command_id"],
        f"{kind} identity",
    )
    engine_step_id = _identity_nonnegative_int(
        row["engine_step_id"],
        f"{kind} identity",
    )
    repeat_index = _identity_nonnegative_int(
        row["repeat_index"],
        f"{kind} identity",
    )
    if repeat_index != expected_repeat_index:
        raise ValueError("command timeline repeat identity mismatch")
    request_set_sha256 = _identity_sha256(
        row["request_set_sha256"],
        f"{kind} request digest",
    )
    if request_set_sha256 != expected_request_set_sha256:
        raise ValueError("command timeline request digest mismatch")
    selected_sha256 = _identity_sha256(
        row["speculative_selected_sequence_ids_sha256"],
        f"{kind} selected-sequence digest",
        optional=True,
    )
    return (
        command_id,
        engine_step_id,
        selected_sha256,
    )


def _engine_step_identity(
    row,
    *,
    expected_repeat_index,
    expected_request_set_sha256,
):
    required = (
        "engine_step_id",
        "repeat_index",
        "request_set_sha256",
        "batch_kind",
        "speculative_selected_sequence_ids_sha256",
    )
    if (
        not isinstance(row, dict)
        or any(name not in row for name in required)
    ):
        raise ValueError("engine step identity is malformed")
    engine_step_id = _identity_nonnegative_int(
        row["engine_step_id"],
        "engine step identity",
    )
    repeat_index = _identity_nonnegative_int(
        row["repeat_index"],
        "engine step identity",
    )
    if repeat_index != expected_repeat_index:
        raise ValueError("command timeline repeat identity mismatch")
    request_set_sha256 = _identity_sha256(
        row["request_set_sha256"],
        "engine step request digest",
    )
    if request_set_sha256 != expected_request_set_sha256:
        raise ValueError("command timeline request digest mismatch")
    if not isinstance(row["batch_kind"], str) or not row["batch_kind"]:
        raise ValueError("engine step identity is malformed")
    selected_sha256 = _identity_sha256(
        row["speculative_selected_sequence_ids_sha256"],
        "engine step selected-sequence digest",
        optional=True,
    )
    return engine_step_id, selected_sha256


def _validate_command_timeline_run(
    run: dict,
    *,
    expected_public_repeat: int,
    expected_repeat_index: int,
    expected_request_set_sha256: str,
) -> None:
    try:
        repeat = run["repeat"]
        timeline = run["runtime"]["command_timeline"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "command timeline evidence is missing"
        ) from error
    if timeline.get("schema_version") != 1:
        raise ValueError("command timeline schema is invalid")
    _identity_nonnegative_int(
        expected_repeat_index,
        "expected repeat identity",
    )
    if (
        isinstance(expected_public_repeat, bool)
        or not isinstance(expected_public_repeat, int)
    ):
        raise ValueError("expected public repeat is malformed")
    _identity_sha256(
        expected_request_set_sha256,
        "expected request digest",
    )
    for name in ("rank_snapshots", "cuda_rank_snapshots"):
        rows = timeline.get(name)
        if not isinstance(rows, list) or len(rows) != TENSOR_PARALLEL_SIZE:
            raise ValueError(
                f"command timeline {name} rank inventory is invalid"
            )
        try:
            ranks = [
                _identity_nonnegative_int(
                    row.get("rank") if isinstance(row, dict) else None,
                    f"{name} rank",
                )
                for row in rows
            ]
        except ValueError as error:
            raise ValueError(
                f"command timeline {name} rank inventory is invalid"
            ) from error
        if ranks != list(range(TENSOR_PARALLEL_SIZE)):
            raise ValueError(
                f"command timeline {name} rank inventory is invalid"
            )
    command_inventories = []
    command_rows_by_rank = []
    for rank, snapshot in enumerate(timeline["rank_snapshots"]):
        rows = snapshot.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                "command timeline command row evidence is missing"
            )
        if not _zero_evidence_counter(
            snapshot.get("dropped_rows"),
            "command dropped_rows",
        ):
            raise ValueError("command timeline command rows were dropped")
        inventory = [
            _command_transport_identity(
                row,
                expected_rank=rank,
                expected_repeat_index=expected_repeat_index,
                expected_request_set_sha256=(
                    expected_request_set_sha256
                ),
            )
            for row in rows
        ]
        command_ids = [identity[0] for identity in inventory]
        if len(set(command_ids)) != len(command_ids):
            raise ValueError("command identity is malformed")
        command_inventories.append(inventory)
        command_rows_by_rank.append({
            identity[0]: identity
            for identity in inventory
        })
    if any(
        inventory != command_inventories[0]
        for inventory in command_inventories[1:]
    ):
        raise ValueError(
            "command timeline command inventories differ across ranks"
        )
    cuda_rows = timeline["cuda_rank_snapshots"]
    if any(
        not isinstance(row.get("steps"), list)
        or not row["steps"]
        for row in cuda_rows
    ):
        raise ValueError("command timeline CUDA step evidence is missing")
    for snapshot in cuda_rows:
        if (
            not _zero_evidence_counter(
                snapshot.get("dropped_steps"),
                "CUDA dropped_steps",
            )
            or not _zero_evidence_counter(
                snapshot.get("dropped_collectives"),
                "CUDA dropped_collectives",
            )
        ):
            raise ValueError("command timeline CUDA evidence was dropped")
        if not isinstance(snapshot.get("collectives"), list):
            raise ValueError(
                "command timeline CUDA collective evidence is malformed"
            )
    engine_steps = timeline.get("engine_steps")
    if not isinstance(engine_steps, list) or not engine_steps:
        raise ValueError("command timeline engine step evidence is missing")
    if not _zero_evidence_counter(
        timeline.get("engine_dropped_steps"),
        "engine dropped_steps",
    ):
        raise ValueError(
            "command timeline engine step evidence was dropped"
        )
    engine_step_rows = {}
    for row in engine_steps:
        engine_step_id, selected_sha256 = _engine_step_identity(
            row,
            expected_repeat_index=expected_repeat_index,
            expected_request_set_sha256=expected_request_set_sha256,
        )
        if engine_step_id in engine_step_rows:
            raise ValueError("engine step identity is malformed")
        engine_step_rows[engine_step_id] = selected_sha256
    for inventory in command_inventories:
        for command_identity in inventory:
            if command_identity[3] not in engine_step_rows:
                raise ValueError(
                    "command timeline unknown engine step identity"
                )
            selected_digests = {
                value
                for value in (
                    command_identity[7],
                    engine_step_rows[command_identity[3]],
                )
                if value is not None
            }
            if len(selected_digests) > 1:
                raise ValueError(
                    "command timeline selected-sequence digest mismatch"
                )
    for rank, snapshot in enumerate(cuda_rows):
        command_rows = command_rows_by_rank[rank]
        for kind, rows in (
            ("CUDA step", snapshot["steps"]),
            ("CUDA collective", snapshot["collectives"]),
        ):
            for row in rows:
                (
                    command_id,
                    engine_step_id,
                    selected_sha256,
                ) = _timeline_row_identity(
                    row,
                    kind=kind,
                    expected_rank=rank,
                    expected_repeat_index=expected_repeat_index,
                    expected_request_set_sha256=(
                        expected_request_set_sha256
                    ),
                )
                command_identity = command_rows.get(command_id)
                if command_identity is None:
                    raise ValueError(
                        "command timeline unknown command identity"
                    )
                if engine_step_id not in engine_step_rows:
                    raise ValueError(
                        "command timeline unknown engine step identity"
                    )
                if command_identity[3] != engine_step_id:
                    raise ValueError(
                        "command timeline command/step identity mismatch"
                    )
                selected_digests = {
                    value
                    for value in (
                        command_identity[7],
                        selected_sha256,
                        engine_step_rows[engine_step_id],
                    )
                    if value is not None
                }
                if len(selected_digests) > 1:
                    raise ValueError(
                        "command timeline selected-sequence digest mismatch"
                    )
    if not isinstance(repeat, int) or isinstance(repeat, bool):
        raise ValueError("command timeline repeat is invalid")
    if repeat != expected_public_repeat:
        raise ValueError("command timeline public repeat mismatch")


def _validate_command_timeline_graph_lifecycle(
    *,
    warmup_results: list[dict],
    measured_results: list[dict],
    cuda_graph_mode: str,
    expected_request_set_sha256: str,
) -> None:
    if len(warmup_results) != 1 or len(measured_results) != 5:
        raise ValueError(
            "command timeline requires one warmup and five measured runs"
        )
    _validate_command_timeline_run(
        warmup_results[0],
        expected_public_repeat=-1,
        expected_repeat_index=0,
        expected_request_set_sha256=expected_request_set_sha256,
    )
    for expected_public_repeat, run in enumerate(measured_results):
        _validate_command_timeline_run(
            run,
            expected_public_repeat=expected_public_repeat,
            expected_repeat_index=expected_public_repeat + 1,
            expected_request_set_sha256=expected_request_set_sha256,
        )
    runs = warmup_results + measured_results
    for rank in range(TENSOR_PARALLEL_SIZE):
        counter_rows = [
            run["correctness"]["rank_graph_counters"][rank]
            for run in runs
        ]
        resource_rows = [
            run["correctness"]["rank_graph_resources"][rank]
            for run in runs
        ]
        if cuda_graph_mode == "eager":
            if any(
                any(
                    row.get(name) != 0
                    for name in (
                        "capture_attempts",
                        "captures",
                        "replays",
                        "quarantines",
                        "fallback_pre_replay",
                    )
                )
                for row in counter_rows
            ) or any(
                any(
                    row.get(name) != 0
                    for name in (
                        "ready_entry_count",
                        "static_bytes",
                        "reserved_bytes",
                        "total_capture_ns",
                    )
                )
                for row in resource_rows
            ):
                raise ValueError(
                    "eager command timeline has CUDA graph activity"
                )
            continue
        warmup_counter = counter_rows[0]
        if (
            warmup_counter.get("capture_attempts") != 1
            or warmup_counter.get("captures") != 1
            or warmup_counter.get("replays") != 1
        ):
            raise ValueError(
                "graph warmup counters are invalid: "
                f"rank={rank} "
                "counters="
                f"{json.dumps(warmup_counter, sort_keys=True, separators=(',', ':'))} "
                "resources="
                f"{json.dumps(resource_rows[0], sort_keys=True, separators=(',', ':'))}"
            )
        if (
            warmup_counter.get("quarantines") != 0
            or warmup_counter.get("fallback_pre_replay") != 0
        ):
            raise ValueError(
                "graph warmup contains quarantine or fallback"
            )
        warmup_resource = resource_rows[0]
        if warmup_resource.get("ready_entry_count") != 1:
            raise ValueError("graph warmup ready entry count is invalid")
        previous_replays = 1
        for counter, resources in zip(
            counter_rows[1:],
            resource_rows[1:],
        ):
            if (
                counter.get("capture_attempts") != 1
                or counter.get("captures") != 1
            ):
                raise ValueError(
                    "graph capture counters changed"
                )
            if counter.get("replays") != previous_replays + 1:
                raise ValueError(
                    "graph replay counters did not grow by one"
                )
            if (
                counter.get("quarantines") != 0
                or counter.get("fallback_pre_replay") != 0
            ):
                raise ValueError(
                    "graph measured run contains quarantine or fallback"
                )
            if resources != warmup_resource:
                raise ValueError(
                    "graph retained resources changed"
                )
            previous_replays = counter["replays"]


def _allocator_snapshot(rank_snapshot: dict) -> dict:
    try:
        allocator = rank_snapshot["executor"]["backend"][
            "proposal_kv_cache"
        ]["entry_allocator"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Proposal-KV allocator snapshot is missing"
        ) from error
    if (
        not isinstance(allocator, dict)
        or allocator.get("allocator_mode") != "direct"
    ):
        raise ValueError(
            "Proposal-KV allocator snapshot is not direct"
        )
    return allocator


def _executor_timing_snapshot(rank_snapshot: dict) -> dict:
    try:
        timing_ms = rank_snapshot["executor"]["timing_ms"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "draft executor timing snapshot is missing"
        ) from error
    if not isinstance(timing_ms, dict):
        raise ValueError(
            "draft executor timing snapshot is invalid"
        )
    normalized = {}
    for key in DRAFT_EXECUTOR_TIMING_KEYS:
        value = timing_ms.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"draft executor timing {key} is invalid"
            )
        normalized[key] = float(value)
    return normalized


def _executor_proposal_detail_snapshot(
    rank_snapshot: dict,
) -> dict:
    try:
        detail_ms = rank_snapshot["executor"][
            "proposal_forward_detail_ms"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "draft executor proposal detail snapshot is missing"
        ) from error
    if not isinstance(detail_ms, dict):
        raise ValueError(
            "draft executor proposal detail snapshot is invalid"
        )
    normalized = {}
    for key in PROPOSAL_FORWARD_DETAIL_KEYS:
        value = detail_ms.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"draft executor proposal detail {key} is invalid"
            )
        normalized[key] = float(value)
    return normalized


def _proposal_kv_delta(
    before_rows: tuple[dict, ...],
    after_rows: tuple[dict, ...],
) -> dict:
    before_rows = _rank_rows(
        before_rows,
        name="before Proposal-KV authority",
    )
    after_rows = _rank_rows(
        after_rows,
        name="after Proposal-KV authority",
    )
    rank_rows = []
    totals = {key: 0 for key in PROPOSAL_KV_COUNTER_KEYS}
    source_keys = {
        "h2d_entries": "h2d_entry_count",
        "h2d_bytes": "h2d_bytes",
        "d2h_entries": "d2h_entry_count",
        "d2h_bytes": "d2h_bytes",
    }
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        before_allocator = _allocator_snapshot(before)
        after_allocator = _allocator_snapshot(after)
        row = {"rank": rank}
        for output_key, source_key in source_keys.items():
            before_value = before_allocator.get(source_key)
            after_value = after_allocator.get(source_key)
            if (
                isinstance(before_value, bool)
                or not isinstance(before_value, int)
                or before_value < 0
                or isinstance(after_value, bool)
                or not isinstance(after_value, int)
                or after_value < before_value
            ):
                raise ValueError(
                    f"Proposal-KV counter {source_key} is invalid"
                )
            row[output_key] = after_value - before_value
            totals[output_key] += row[output_key]
        rank_rows.append(row)
    return {"ranks": rank_rows, "totals": totals}


def _draft_executor_timing_delta(
    before_rows: tuple[dict, ...],
    after_rows: tuple[dict, ...],
) -> dict:
    before_rows = _rank_rows(
        before_rows,
        name="before draft executor timing",
    )
    after_rows = _rank_rows(
        after_rows,
        name="after draft executor timing",
    )
    rank_rows = []
    max_rank_ms = {
        key: 0.0 for key in DRAFT_EXECUTOR_TIMING_KEYS
    }
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        before_timing = _executor_timing_snapshot(before)
        after_timing = _executor_timing_snapshot(after)
        row = {"rank": rank}
        for key in DRAFT_EXECUTOR_TIMING_KEYS:
            if after_timing[key] < before_timing[key]:
                raise ValueError(
                    f"draft executor timing {key} regressed"
                )
            row[key] = after_timing[key] - before_timing[key]
            max_rank_ms[key] = max(
                max_rank_ms[key],
                row[key],
            )
        rank_rows.append(row)
    return {
        "ranks": rank_rows,
        "max_rank_ms": max_rank_ms,
    }


def _draft_executor_proposal_detail_delta(
    before_rows: tuple[dict, ...],
    after_rows: tuple[dict, ...],
    *,
    draft_executor_timing: dict,
) -> dict:
    before_rows = _rank_rows(
        before_rows,
        name="before draft executor proposal detail",
    )
    after_rows = _rank_rows(
        after_rows,
        name="after draft executor proposal detail",
    )
    rank_rows = []
    max_rank_ms = {
        key: 0.0 for key in PROPOSAL_FORWARD_DETAIL_KEYS
    }
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        before_detail = _executor_proposal_detail_snapshot(before)
        after_detail = _executor_proposal_detail_snapshot(after)
        row = {"rank": rank}
        for key in PROPOSAL_FORWARD_DETAIL_KEYS:
            delta = after_detail[key] - before_detail[key]
            if delta < -PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS:
                raise ValueError(
                    f"draft executor proposal detail {key} regressed"
                )
            row[key] = max(0.0, delta)
            max_rank_ms[key] = max(
                max_rank_ms[key],
                row[key],
            )
        rank_rows.append(row)
    timing_rows = draft_executor_timing.get("ranks")
    if (
        not isinstance(timing_rows, list)
        or len(timing_rows) != TENSOR_PARALLEL_SIZE
        or any(
            row.get("rank") != rank
            for rank, row in enumerate(timing_rows)
        )
    ):
        raise ValueError(
            "draft executor timing rank rows are invalid"
        )
    critical_rank = max(
        range(TENSOR_PARALLEL_SIZE),
        key=lambda rank: timing_rows[rank]["proposal_forward"],
    )
    critical_rank_ms = {
        key: rank_rows[critical_rank][key]
        for key in PROPOSAL_FORWARD_DETAIL_KEYS
    }
    detail_sum_ms = sum(critical_rank_ms.values())
    proposal_forward_ms = timing_rows[critical_rank][
        "proposal_forward"
    ]
    residual_ms = proposal_forward_ms - detail_sum_ms
    if residual_ms < -PROPOSAL_FORWARD_RESIDUAL_TOLERANCE_MS:
        raise ValueError(
            "draft executor proposal detail residual is negative"
        )
    return {
        "ranks": rank_rows,
        "max_rank_ms": max_rank_ms,
        "critical_rank": critical_rank,
        "critical_rank_ms": critical_rank_ms,
        "detail_sum_ms": detail_sum_ms,
        "residual_ms": max(0.0, residual_ms),
    }


def _zero_proposal_kv() -> dict:
    rows = [
        {
            "rank": rank,
            **{
                key: 0
                for key in PROPOSAL_KV_COUNTER_KEYS
            },
        }
        for rank in range(TENSOR_PARALLEL_SIZE)
    ]
    return {
        "ranks": rows,
        "totals": {
            key: 0 for key in PROPOSAL_KV_COUNTER_KEYS
        },
    }


def _zero_draft_executor_timing() -> dict:
    rows = [
        {
            "rank": rank,
            **{
                key: 0.0
                for key in DRAFT_EXECUTOR_TIMING_KEYS
            },
        }
        for rank in range(TENSOR_PARALLEL_SIZE)
    ]
    return {
        "ranks": rows,
        "max_rank_ms": {
            key: 0.0 for key in DRAFT_EXECUTOR_TIMING_KEYS
        },
    }


def _zero_draft_executor_proposal_detail() -> dict:
    rows = [
        {
            "rank": rank,
            **{
                key: 0.0
                for key in PROPOSAL_FORWARD_DETAIL_KEYS
            },
        }
        for rank in range(TENSOR_PARALLEL_SIZE)
    ]
    return {
        "ranks": rows,
        "max_rank_ms": {
            key: 0.0 for key in PROPOSAL_FORWARD_DETAIL_KEYS
        },
        "critical_rank": 0,
        "critical_rank_ms": {
            key: 0.0 for key in PROPOSAL_FORWARD_DETAIL_KEYS
        },
        "detail_sum_ms": 0.0,
        "residual_ms": 0.0,
    }


def _memory_result(rows: tuple[dict, ...]) -> dict:
    rows = _rank_rows(rows, name="memory snapshots")
    result_rows = []
    for rank, row in enumerate(rows):
        allocated = row.get("cuda_peak_allocated_bytes")
        reserved = row.get("cuda_peak_reserved_bytes")
        for name, value in (
            ("peak allocated", allocated),
            ("peak reserved", reserved),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"memory {name} is invalid")
        result_rows.append({
            "rank": rank,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        })
    return {
        "ranks": result_rows,
        "peak_allocated_bytes": max(
            row["peak_allocated_bytes"]
            for row in result_rows
        ),
        "peak_reserved_bytes": max(
            row["peak_reserved_bytes"]
            for row in result_rows
        ),
    }


def _stage_timing_result(
    observations: list[dict],
    *,
    policy: str,
) -> dict:
    steps = []
    totals_ms = {
        key: 0.0 for key in RUNTIME_STAGE_TIMING_KEYS
    }
    for observation in observations:
        timing_ms = observation.get(
            "speculative_runtime_timing_ms",
            {},
        )
        if not isinstance(timing_ms, dict):
            raise ValueError(
                "speculative runtime timing row is invalid"
            )
        if not timing_ms:
            continue
        if policy != "learned":
            raise ValueError(
                "target observation contains speculative timing"
            )
        if set(timing_ms) != set(RUNTIME_STAGE_TIMING_KEYS):
            raise ValueError(
                "speculative runtime timing inventory mismatch"
            )
        normalized = {}
        for key in RUNTIME_STAGE_TIMING_KEYS:
            value = timing_ms.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"speculative runtime timing {key} is invalid"
                )
            normalized[key] = float(value)
            totals_ms[key] += normalized[key]
        steps.append({
            "step_index": len(steps),
            "timing_ms": normalized,
        })
    return {
        "step_count": len(steps),
        "steps": steps,
        "totals_ms": totals_ms,
    }


def _runtime_result(
    observations: list[dict],
    *,
    policy: str,
    draft_executor_timing: dict,
    draft_executor_proposal_detail: dict,
) -> dict:
    proposed_tokens = 0
    accepted_draft_tokens = 0
    for observation in observations:
        proposal_rows = observation.get(
            "speculative_proposal_token_ids_by_seq",
            {},
        )
        accepted_rows = observation.get(
            "speculative_accepted_draft_token_counts",
            {},
        )
        if not isinstance(proposal_rows, dict) or not isinstance(
            accepted_rows,
            dict,
        ):
            raise ValueError("speculative runtime rows are invalid")
        proposed_tokens += sum(
            len(token_ids)
            for token_ids in proposal_rows.values()
        )
        accepted_draft_tokens += sum(
            int(count)
            for count in accepted_rows.values()
        )
    return {
        "proposed_tokens": proposed_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "acceptance_rate": (
            accepted_draft_tokens / proposed_tokens
            if proposed_tokens
            else 0.0
        ),
        "stage_timing": _stage_timing_result(
            observations,
            policy=policy,
        ),
        "draft_executor_timing": draft_executor_timing,
        "draft_executor_proposal_detail": (
            draft_executor_proposal_detail
        ),
    }


def _correctness_result(
    observations: list[dict],
    after_authority: tuple[dict, ...],
) -> dict:
    proposal_token_rows = []
    accepted_prefix_counts = []
    for observation in observations:
        proposal_rows = observation.get(
            "speculative_proposal_token_ids_by_seq",
            {},
        )
        accepted_rows = observation.get(
            "speculative_accepted_draft_token_counts",
            {},
        )
        if not isinstance(proposal_rows, dict) or not isinstance(
            accepted_rows,
            dict,
        ):
            raise ValueError(
                "correctness observation rows are invalid"
            )
        if not proposal_rows:
            if accepted_rows:
                raise ValueError(
                    "accepted rows exist without proposal rows"
                )
            continue
        if set(proposal_rows) != set(accepted_rows):
            raise ValueError(
                "proposal and accepted row inventories differ"
            )
        proposal_token_rows.append([
            list(proposal_rows[sequence_id])
            for sequence_id in sorted(proposal_rows)
        ])
        accepted_prefix_counts.extend(
            int(accepted_rows[sequence_id])
            for sequence_id in sorted(accepted_rows)
        )
    authority_rows = _rank_rows(
        after_authority,
        name="correctness Proposal-KV authority",
    )
    digests = []
    active_transaction_counts = []
    rank_graph_counters = []
    rank_graph_quarantined = []
    rank_graph_quarantine_details = []
    rank_graph_resources = []
    counter_names = (
        "capture_attempts",
        "captures",
        "replays",
        "quarantines",
        "fallback_pre_replay",
    )
    for rank, row in enumerate(authority_rows):
        executor = row.get("executor")
        if not isinstance(executor, dict):
            raise ValueError(
                "correctness executor authority is missing"
            )
        digest = executor.get(
            "last_logical_authority_sha256"
        )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                "correctness transaction digest is invalid"
            )
        digests.append(digest)
        lifecycle = executor.get("proposal_kv_lifecycle")
        if not isinstance(lifecycle, dict):
            raise ValueError(
                "correctness Proposal-KV lifecycle is missing"
            )
        active = lifecycle.get("active_transaction_count")
        if (
            isinstance(active, bool)
            or not isinstance(active, int)
            or active < 0
        ):
            raise ValueError(
                "correctness active transaction count is invalid"
            )
        active_transaction_counts.append(active)
        graph = executor.get("cuda_graph")
        if graph is None:
            graph = {name: 0 for name in counter_names}
        if not isinstance(graph, dict):
            raise ValueError(
                "correctness graph summary is invalid"
            )
        graph_row = {"rank": rank}
        for name in counter_names:
            value = graph.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"correctness graph counter {name} is invalid"
                )
            graph_row[name] = value
        rank_graph_counters.append(graph_row)
        quarantined = graph.get("quarantined", {})
        if (
            not isinstance(quarantined, dict)
            or any(
                not isinstance(identity_sha256, str)
                or not isinstance(reason, str)
                for identity_sha256, reason in quarantined.items()
            )
        ):
            raise ValueError(
                "correctness graph quarantined inventory is invalid"
            )
        rank_graph_quarantined.append({
            "rank": rank,
            "quarantined": dict(quarantined),
        })
        quarantine_details = graph.get(
            "quarantine_details",
            {},
        )
        if not isinstance(quarantine_details, dict):
            raise ValueError(
                "correctness graph quarantine details are invalid"
            )
        rank_graph_quarantine_details.append({
            "rank": rank,
            "quarantine_details": quarantine_details,
        })
        ready_entries = graph.get("ready_entries", ())
        if not isinstance(ready_entries, (list, tuple)):
            raise ValueError(
                "correctness graph ready entries are invalid"
            )
        resource_row = {
            "rank": rank,
            "ready_entry_count": len(ready_entries),
        }
        for name in (
            "static_bytes",
            "reserved_bytes",
            "total_capture_ns",
        ):
            value = graph.get(name, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"correctness graph resource {name} is invalid"
                )
            resource_row[name] = value
        rank_graph_resources.append(resource_row)
    if len(set(digests)) != 1:
        raise ValueError(
            "correctness transaction digests differ across ranks"
        )
    return {
        "proposal_token_rows": proposal_token_rows,
        "accepted_prefix_counts": accepted_prefix_counts,
        "transaction_digest": digests[0],
        "active_transaction_count": sum(
            active_transaction_counts
        ),
        "rank_graph_counters": rank_graph_counters,
        "rank_graph_quarantined": rank_graph_quarantined,
        "rank_graph_resources": rank_graph_resources,
        "rank_graph_quarantine_details": (
            rank_graph_quarantine_details
        ),
    }


def run_request_batch(
    *,
    engine,
    policy: str,
    prompt_rows: list[dict],
    sampling_params,
    expected_output_tokens: int,
    synchronize,
    clock_ns,
    repeat: int,
    command_timeline: bool = False,
    command_timeline_repeat_index: int | None = None,
) -> dict:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if not engine.is_finished():
        raise RuntimeError("engine must be idle before a run")
    engine.clear_reusable_prefix_cache()
    before_authority = None
    if policy == "learned":
        before_authority = _rank_rows(
            engine.autoregressive_draft_authority_snapshots(
                timeout_s=60.0
            ),
            name="before Proposal-KV authority",
        )
    _rank_rows(
        engine.reset_peak_memory_stats(timeout_s=60.0),
        name="peak memory reset",
    )
    synchronize()
    timeline_evidence = None
    if command_timeline:
        if command_timeline_repeat_index is None:
            command_timeline_repeat_index = repeat
        if (
            isinstance(command_timeline_repeat_index, bool)
            or not isinstance(command_timeline_repeat_index, int)
            or command_timeline_repeat_index < 0
        ):
            raise ValueError(
                "command timeline repeat index must be "
                "a non-negative integer"
            )
        engine.reset_command_timeline(60.0)
        engine.begin_command_timeline_repeat(
            command_timeline_repeat_index,
            request_set_sha256=_request_set_sha256(prompt_rows),
        )
        engine.reset_decode_internal_profile(timeout_s=60.0)
    request_start_ns = clock_ns()
    for prompt_row in prompt_rows:
        token_ids = prompt_row.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != PROMPT_TOKENS
        ):
            raise ValueError("worker prompt must contain 256 tokens")
        engine.add_request(token_ids, sampling_params)

    token_events = {}
    finished_at_ns = {}
    outputs_by_id = {}
    observations = []
    request_finish_ns = request_start_ns
    while not engine.is_finished():
        output_rows, _ = engine.step()
        synchronize()
        request_finish_ns = clock_ns()
        observation = getattr(
            engine,
            "last_step_observation",
            None,
        )
        if not isinstance(observation, dict):
            raise RuntimeError("engine step observation is unavailable")
        observations.append(copy.deepcopy(observation))
        deltas = observation.get(
            "new_completion_tokens_by_seq",
            {},
        )
        if not isinstance(deltas, dict):
            raise ValueError("completion token deltas are invalid")
        for sequence_id, token_ids in deltas.items():
            if token_ids:
                token_events.setdefault(
                    int(sequence_id),
                    [],
                ).append((request_finish_ns, len(token_ids)))
        for sequence_id in observation.get("finished_seq_ids", ()):
            finished_at_ns[int(sequence_id)] = request_finish_ns
        for sequence_id, token_ids in output_rows:
            sequence_id = int(sequence_id)
            outputs_by_id[sequence_id] = list(token_ids)
            finished_at_ns.setdefault(
                sequence_id,
                request_finish_ns,
            )

    if command_timeline:
        engine.end_command_timeline_repeat()
        command_rows = engine.command_timeline_snapshots(
            timeout_s=60.0
        )
        cuda_result = engine.finalize_decode_internal_profile(
            already_synchronized_rank=0,
            timeout_s=60.0,
        )
        step_rows = engine.engine_step_timeline_snapshot()
        timeline_evidence = {
            "schema_version": 1,
            "rank_snapshots": list(command_rows),
            "cuda_rank_snapshots": list(cuda_result["ranks"]),
            "engine_steps": list(step_rows["steps"]),
            "engine_dropped_steps": step_rows["dropped_steps"],
        }

    if policy == "learned":
        engine.flush_pending_hybrid_state_releases(timeout_s=60.0)
        after_authority = _rank_rows(
            engine.autoregressive_draft_authority_snapshots(
                timeout_s=60.0
            ),
            name="after Proposal-KV authority",
        )
        proposal_kv = _proposal_kv_delta(
            before_authority,
            after_authority,
        )
        draft_executor_timing = _draft_executor_timing_delta(
            before_authority,
            after_authority,
        )
        draft_executor_proposal_detail = (
            _draft_executor_proposal_detail_delta(
                before_authority,
                after_authority,
                draft_executor_timing=draft_executor_timing,
            )
        )
        correctness = _correctness_result(
            observations,
            after_authority,
        )
    else:
        proposal_kv = _zero_proposal_kv()
        draft_executor_timing = (
            _zero_draft_executor_timing()
        )
        draft_executor_proposal_detail = (
            _zero_draft_executor_proposal_detail()
        )
        correctness = None
    memory = _memory_result(
        engine.memory_snapshots(timeout_s=60.0)
    )
    outputs = [
        outputs_by_id[sequence_id]
        for sequence_id in sorted(outputs_by_id)
    ]
    if len(outputs) != len(prompt_rows):
        raise RuntimeError("engine did not return one output per prompt")
    if any(
        len(token_ids) != expected_output_tokens
        for token_ids in outputs
    ):
        raise RuntimeError("engine output token count is incorrect")
    runtime = _runtime_result(
        observations,
        policy=policy,
        draft_executor_timing=draft_executor_timing,
        draft_executor_proposal_detail=(
            draft_executor_proposal_detail
        ),
    )
    if timeline_evidence is not None:
        runtime["command_timeline"] = timeline_evidence
    return {
        "repeat": repeat,
        "outputs": outputs,
        "timing": build_run_metrics(
            request_start_ns=request_start_ns,
            request_finish_ns=request_finish_ns,
            token_events=token_events,
            finished_at_ns=finished_at_ns,
            expected_output_tokens=expected_output_tokens,
        ),
        "runtime": runtime,
        "memory": memory,
        "proposal_kv": proposal_kv,
        "correctness": correctness,
    }


def run_policy_campaign(
    *,
    target_model: str,
    draft_model: str,
    policy: str,
    batch_size: int,
    engine_factory,
    sampling_params_type,
    synchronize,
    clock_ns,
    wall_clock_ns=time.time_ns,
    monotonic_clock_ns=time.monotonic_ns,
    run_batch_fn=run_request_batch,
    warmup_runs: int = WARMUP_RUNS,
    measured_runs: int = MEASURED_RUNS,
    cuda_graph_mode: str = "eager",
    command_timeline: bool = False,
    cuda_graph_max_reserved_bytes: int | None = None,
    cuda_graph_max_total_capture_ns: int | None = None,
    cuda_graph_max_single_capture_ns: int | None = None,
) -> dict:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    if cuda_graph_mode not in ("eager", "graph"):
        raise ValueError("CUDA graph mode is invalid")
    if policy != "learned" and cuda_graph_mode != "eager":
        raise ValueError(
            "target policy cannot enable draft CUDA graphs"
        )
    if not isinstance(command_timeline, bool):
        raise ValueError("command_timeline must be a bool")
    graph_budget_overrides = {
        "cuda_graph_max_reserved_bytes": (
            cuda_graph_max_reserved_bytes
        ),
        "cuda_graph_max_total_capture_ns": (
            cuda_graph_max_total_capture_ns
        ),
        "cuda_graph_max_single_capture_ns": (
            cuda_graph_max_single_capture_ns
        ),
    }
    for name, value in graph_budget_overrides.items():
        if value is None:
            continue
        if (
            cuda_graph_mode != "graph"
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{name} requires graph mode and a positive integer"
            )
    for name, value, minimum in (
        ("warmup runs", warmup_runs, 0),
        ("measured runs", measured_runs, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer >= {minimum}"
            )
    diagnostic_counts = warmup_runs == 1 and measured_runs == 5
    if diagnostic_counts and not command_timeline:
        raise ValueError(
            "one warmup and five measured runs require command timeline"
        )
    if command_timeline and (
        policy != "learned"
        or batch_size != 4
        or not diagnostic_counts
        or TENSOR_PARALLEL_SIZE != 4
        or MAX_PROPOSAL_TOKENS != 4
    ):
        raise ValueError(
            "command timeline requires learned TP4/B4/Q4 with "
            "one warmup and five measured runs"
        )
    proposal_slot_capacity = proposal_slot_capacity_for_batch(
        batch_size
    )
    adapter = engine_factory(
        policy,
        target_model=target_model,
        draft_model=draft_model,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=batch_size,
        max_model_len=512,
        max_num_batched_tokens=2048,
        proposal_slot_capacity=proposal_slot_capacity,
        learned_enabled=policy == "learned",
        cuda_graph_enabled=cuda_graph_mode == "graph",
        **graph_budget_overrides,
    )
    try:
        engine = adapter.engine
        prompt_rows = build_prompt_token_batches(
            engine.tokenizer,
            batch_size=batch_size,
        )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
            ignore_eos=True,
        )
        if command_timeline:
            engine.configure_command_timeline(
                True,
                COMMAND_TIMELINE_MAX_ROWS,
                60.0,
            )
            engine.configure_decode_internal_profile(
                True,
                (
                    "autoregressive-draft-command-timeline/"
                    f"{cuda_graph_mode}"
                ),
                timeout_s=60.0,
            )

        def run_once(
            repeat: int,
            command_timeline_repeat_index: int,
        ):
            started_at_unix_ns = wall_clock_ns()
            started_at_monotonic_ns = (
                monotonic_clock_ns() if command_timeline else None
            )
            run_batch_kwargs = {
                "engine": engine,
                "policy": policy,
                "prompt_rows": prompt_rows,
                "sampling_params": sampling_params,
                "expected_output_tokens": MAX_OUTPUT_TOKENS,
                "synchronize": synchronize,
                "clock_ns": clock_ns,
                "repeat": repeat,
            }
            if command_timeline:
                run_batch_kwargs.update({
                    "command_timeline": True,
                    "command_timeline_repeat_index": (
                        command_timeline_repeat_index
                    ),
                })
            result = run_batch_fn(
                **run_batch_kwargs,
            )
            finished_at_monotonic_ns = (
                monotonic_clock_ns() if command_timeline else None
            )
            finished_at_unix_ns = wall_clock_ns()
            if (
                isinstance(started_at_unix_ns, bool)
                or not isinstance(started_at_unix_ns, int)
                or isinstance(finished_at_unix_ns, bool)
                or not isinstance(finished_at_unix_ns, int)
                or started_at_unix_ns <= 0
                or finished_at_unix_ns <= started_at_unix_ns
            ):
                raise ValueError("campaign interval is invalid")
            if command_timeline and (
                isinstance(started_at_monotonic_ns, bool)
                or not isinstance(started_at_monotonic_ns, int)
                or isinstance(finished_at_monotonic_ns, bool)
                or not isinstance(finished_at_monotonic_ns, int)
                or started_at_monotonic_ns < 0
                or finished_at_monotonic_ns
                <= started_at_monotonic_ns
            ):
                raise ValueError("campaign monotonic interval is invalid")
            campaign_interval = {
                "started_at_unix_ns": started_at_unix_ns,
                "finished_at_unix_ns": finished_at_unix_ns,
            }
            if command_timeline:
                campaign_interval.update({
                    "started_at_monotonic_ns": (
                        started_at_monotonic_ns
                    ),
                    "finished_at_monotonic_ns": (
                        finished_at_monotonic_ns
                    ),
                })
            completed = {
                **result,
                "campaign_interval": campaign_interval,
            }
            if command_timeline:
                completed["command_timeline_repeat_index"] = (
                    command_timeline_repeat_index
                )
            return completed

        warmup_results = [
            run_once(repeat, repeat + warmup_runs)
            for repeat in range(-warmup_runs, 0)
        ]
        measured_results = [
            run_once(repeat, warmup_runs + repeat)
            for repeat in range(measured_runs)
        ]
        if command_timeline:
            _validate_command_timeline_graph_lifecycle(
                warmup_results=warmup_results,
                measured_results=measured_results,
                cuda_graph_mode=cuda_graph_mode,
                expected_request_set_sha256=_request_set_sha256(
                    prompt_rows
                ),
            )
        config = getattr(engine, "config", None)
        return {
            "policy": policy,
            "batch_size": batch_size,
            "prompt_rows": prompt_rows,
            "warmup_runs": warmup_results,
            "measured_runs": measured_results,
            "target_checkpoint_identifier": Path(
                target_model
            ).name,
            "draft_checkpoint_identifier": (
                Path(draft_model).name
                if policy == "learned"
                else None
            ),
            "tokenizer_identifier": str(
                getattr(
                    engine.tokenizer,
                    "name_or_path",
                    type(engine.tokenizer).__name__,
                )
            ),
            "dtype": str(getattr(config, "dtype", "unknown")),
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "proposal_kv_allocator": "direct",
            "proposal_slot_capacity": proposal_slot_capacity,
            "cuda_graph_mode": cuda_graph_mode,
            "cuda_graph_budget_overrides": {
                name: value
                for name, value in graph_budget_overrides.items()
                if value is not None
            },
        }
    finally:
        adapter.close()


def _default_dependencies():
    import torch

    from autoregressive_draft_tp4_engine_gate import (
        _TinyVLLMTP4EngineAdapter,
    )
    from tinyvllm import SamplingParams

    return {
        "engine_factory": _TinyVLLMTP4EngineAdapter,
        "sampling_params_type": SamplingParams,
        "synchronize": torch.cuda.synchronize,
        "clock_ns": time.perf_counter_ns,
        "wall_clock_ns": time.time_ns,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument(
        "--policy",
        required=True,
        choices=POLICIES,
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        choices=BATCH_SIZES,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=WARMUP_RUNS,
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=MEASURED_RUNS,
    )
    parser.add_argument(
        "--cuda-graph-mode",
        choices=("eager", "graph"),
        default="eager",
    )
    parser.add_argument(
        "--command-timeline",
        action="store_true",
    )
    parser.add_argument(
        "--cuda-graph-max-reserved-bytes",
        type=int,
    )
    parser.add_argument(
        "--cuda-graph-max-total-capture-ns",
        type=int,
    )
    parser.add_argument(
        "--cuda-graph-max-single-capture-ns",
        type=int,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_campaign(
        target_model=args.target_model,
        draft_model=args.draft_model,
        policy=args.policy,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        cuda_graph_mode=args.cuda_graph_mode,
        command_timeline=args.command_timeline,
        cuda_graph_max_reserved_bytes=(
            args.cuda_graph_max_reserved_bytes
        ),
        cuda_graph_max_total_capture_ns=(
            args.cuda_graph_max_total_capture_ns
        ),
        cuda_graph_max_single_capture_ns=(
            args.cuda_graph_max_single_capture_ns
        ),
        **_default_dependencies(),
    )
    write_json_atomic(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
