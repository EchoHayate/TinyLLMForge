#!/usr/bin/env python3
"""Baseline attribution for the SLO-aware cohort-burst ceiling gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import slo_cohort_burst_ceiling as ceiling


PROFILE_ROW_SCHEMA_VERSION = "slo-cohort-burst.ceiling-profile-row.v1"
FROZEN_LOADS = ("low", "medium", "high")
_RAW_COMPONENT_FIELDS = (
    ("target_cuda_ns", "target_cuda"),
    ("graph_launch_gap_ns", "graph_launch_gap"),
    ("scheduler_ns", "scheduler"),
    ("token_d2h_publication_ns", "token_d2h_publication"),
    ("batch_binding_ns", "batch_binding"),
)
_AMORTIZABLE_COMPONENTS = (
    "graph_launch_gap",
    "token_d2h_publication",
    "batch_binding",
)
_ENGINE_STEP_PHASES = (
    "scheduler_schedule",
    "partition_and_step_setup",
    "ordinary_or_first_target_dispatch",
    "speculative_prepare",
    "scheduler_prepare_postprocess",
    "proposal_kv_prepare_commit",
    "proposal_lifecycle_finalize_prepare",
    "scheduler_commit_postprocess",
    "proposal_lifecycle_finalize_commit",
    "side_state_seal",
    "residency_precommit_or_seal",
    "ordinary_scheduler_postprocess",
)
_SCHEDULER_PHASES = (
    "scheduler_schedule",
    "scheduler_prepare_postprocess",
    "proposal_kv_prepare_commit",
    "proposal_lifecycle_finalize_prepare",
    "scheduler_commit_postprocess",
    "proposal_lifecycle_finalize_commit",
    "side_state_seal",
    "residency_precommit_or_seal",
    "ordinary_scheduler_postprocess",
)
_LOAD_ARRIVAL_GAP_NS = {
    "low": 4_000_000,
    "medium": 1_000_000,
    "high": 0,
}
_BATCH_SIZES = (1, 2, 4, 8)
_CONTEXT_BUCKETS = (512, 4096, 16384)
_DEFAULT_REQUESTED_OUTPUT_TOKENS = 64
_DEFAULT_WARMUP_STEPS = 4
_DEFAULT_MEASURED_STEPS = 16
_ENGINE_CONFIG = {
    "enforce_eager": False,
    "max_num_seqs": 8,
    "max_model_len": 32768,
    "max_num_batched_tokens": (
        max(_BATCH_SIZES) * max(_CONTEXT_BUCKETS)
    ),
    "max_num_prefill_tokens_per_step": 0,
    "autoregressive_draft_command_timeline": True,
    "autoregressive_draft_command_timeline_max_rows": 4096,
}


@dataclass(frozen=True)
class CeilingProfileCase:
    case_id: str
    load: str
    batch_size: int
    context_bucket: int
    burst_width: int
    source_commit: str
    arrival_offsets_ns: tuple[int, ...]
    requested_output_tokens: int
    warmup_steps: int
    measured_steps: int

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        if self.load not in FROZEN_LOADS:
            raise ValueError("load is outside the frozen inventory")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.context_bucket, "context_bucket")
        if self.burst_width != 1:
            raise ValueError("baseline burst width must be one")
        _source_commit(self.source_commit)
        if (
            not isinstance(self.arrival_offsets_ns, tuple)
            or len(self.arrival_offsets_ns) != self.batch_size
            or tuple(sorted(self.arrival_offsets_ns))
            != self.arrival_offsets_ns
        ):
            raise ValueError("arrival offsets are invalid")
        for offset in self.arrival_offsets_ns:
            _non_negative_int(offset, "arrival_offset_ns")
        _positive_int(
            self.requested_output_tokens,
            "requested_output_tokens",
        )
        _non_negative_int(self.warmup_steps, "warmup_steps")
        _positive_int(self.measured_steps, "measured_steps")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _source_commit(value: object) -> str:
    text = _text(value, "source_commit")
    if len(text) != 40 or any(
        character not in "0123456789abcdef"
        for character in text
    ):
        raise ValueError("source_commit must be a lowercase Git SHA")
    return text


def build_frozen_case_inventory(
    source_commit: str,
) -> tuple[CeilingProfileCase, ...]:
    commit = _source_commit(source_commit)
    return tuple(
        CeilingProfileCase(
            case_id=(
                f"{load}-b{batch_size}"
                f"-c{context_bucket}-r0"
            ),
            load=load,
            batch_size=batch_size,
            context_bucket=context_bucket,
            burst_width=1,
            source_commit=commit,
            arrival_offsets_ns=tuple(
                index * _LOAD_ARRIVAL_GAP_NS[load]
                for index in range(batch_size)
            ),
            requested_output_tokens=(
                _DEFAULT_REQUESTED_OUTPUT_TOKENS
            ),
            warmup_steps=_DEFAULT_WARMUP_STEPS,
            measured_steps=_DEFAULT_MEASURED_STEPS,
        )
        for load in FROZEN_LOADS
        for batch_size in _BATCH_SIZES
        for context_bucket in _CONTEXT_BUCKETS
    )


def profile_baseline_case(
    engine,
    case,
    *,
    clock_ns,
) -> dict[str, object]:
    measured = engine.profile_baseline_case(case, clock_ns=clock_ns)
    if not isinstance(measured, Mapping):
        raise ValueError("profile result must be a mapping")
    wall_ns = _positive_int(measured.get("wall_ns"), "wall_ns")
    component_ns = {}
    for raw_name, public_name in _RAW_COMPONENT_FIELDS:
        component_ns[public_name] = _non_negative_int(
            measured.get(raw_name),
            raw_name,
        )
    attributed_ns = sum(component_ns.values())
    if attributed_ns > wall_ns:
        raise ValueError("profile components exceed wall time")
    component_ns["unattributed"] = wall_ns - attributed_ns

    load = _text(case.load, "load")
    if load not in FROZEN_LOADS:
        raise ValueError("load is outside the frozen inventory")
    key = ceiling.SLOCohortCostKey(
        batch_size=case.batch_size,
        context_bucket=case.context_bucket,
        burst_width=case.burst_width,
    )
    return {
        "schema_version": PROFILE_ROW_SCHEMA_VERSION,
        "case_id": _text(case.case_id, "case_id"),
        "load": load,
        "source_commit": _text(
            case.source_commit,
            "source_commit",
        ),
        "batch_size": key.batch_size,
        "context_bucket": key.context_bucket,
        "burst_width": key.burst_width,
        "component_ns": component_ns,
        "wall_ns": wall_ns,
        "committed_tokens": _positive_int(
            measured.get("committed_tokens"),
            "committed_tokens",
        ),
        "cuda_reserved_bytes": _non_negative_int(
            measured.get("cuda_reserved_bytes"),
            "cuda_reserved_bytes",
        ),
    }


def components_from_timeline_step(
    step: Mapping[str, object],
    *,
    target_cuda_ns: int | None = None,
) -> dict[str, int]:
    if not isinstance(step, Mapping):
        raise ValueError("timeline step must be a mapping")
    wall_ns = _positive_int(step.get("step_wall_ns"), "step_wall_ns")
    phases = step.get("phases")
    if (
        not isinstance(phases, Mapping)
        or set(phases) != set(_ENGINE_STEP_PHASES)
    ):
        raise ValueError("timeline phase inventory is invalid")
    durations = {}
    for name in _ENGINE_STEP_PHASES:
        phase = phases[name]
        if not isinstance(phase, Mapping):
            raise ValueError("timeline phase row is invalid")
        durations[name] = _non_negative_int(
            phase.get("duration_ns"),
            f"{name}.duration_ns",
        )
    measured_cuda_ns = (
        durations["ordinary_or_first_target_dispatch"]
        if target_cuda_ns is None
        else _non_negative_int(target_cuda_ns, "target_cuda_ns")
    )
    if measured_cuda_ns > wall_ns:
        raise ValueError("target CUDA exceeds step wall time")
    host_budget_ns = wall_ns - measured_cuda_ns
    scheduler_ns = min(
        sum(durations[name] for name in _SCHEDULER_PHASES),
        host_budget_ns,
    )
    remaining_host_ns = host_budget_ns - scheduler_ns
    batch_binding_ns = min(
        durations["partition_and_step_setup"],
        remaining_host_ns,
    )
    remaining_host_ns -= batch_binding_ns
    return {
        "target_cuda": measured_cuda_ns,
        "graph_launch_gap": remaining_host_ns,
        "scheduler": scheduler_ns,
        "token_d2h_publication": 0,
        "batch_binding": batch_binding_ns,
        "unattributed": 0,
    }


def _prompt_tokens(case: CeilingProfileCase, request_index: int) -> list[int]:
    return [
        100 + ((request_index * 997 + token_index) % 30_000)
        for token_index in range(case.context_bucket)
    ]


def _case_request_set_sha256(case: CeilingProfileCase) -> str:
    payload = {
        "case_id": case.case_id,
        "prompts": [
            _prompt_tokens(case, request_index)
            for request_index in range(case.batch_size)
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _memory_reserved_bytes(observation: Mapping[str, object]) -> int:
    memory = observation.get("memory")
    if not isinstance(memory, Mapping):
        raise ValueError("step observation is missing memory")
    return _non_negative_int(
        memory.get("cuda_reserved_bytes"),
        "cuda_reserved_bytes",
    )


def run_profile_case(
    engine,
    case: CeilingProfileCase,
    *,
    sampling_params_factory,
    step_timer,
    clock_ns=time.monotonic_ns,
    sleep=time.sleep,
) -> list[dict[str, object]]:
    if not isinstance(case, CeilingProfileCase):
        raise ValueError("case must be a CeilingProfileCase")
    if not callable(sampling_params_factory):
        raise ValueError("sampling_params_factory must be callable")
    if not callable(clock_ns) or not callable(sleep):
        raise ValueError("clock and sleep must be callable")
    if not callable(getattr(step_timer, "measure", None)):
        raise ValueError("step timer must provide measure")

    start_ns = clock_ns()
    for request_index, offset_ns in enumerate(
        case.arrival_offsets_ns
    ):
        delay_ns = max(
            0,
            start_ns + offset_ns - clock_ns(),
        )
        if delay_ns:
            sleep(delay_ns / 1_000_000_000.0)
        engine.add_request(
            _prompt_tokens(case, request_index),
            sampling_params_factory(
                temperature=0.0,
                max_tokens=case.requested_output_tokens,
                ignore_eos=True,
            ),
        )
    target_batch_steps = 0
    measured_rows = []
    step_index = 0
    guard = (
        case.batch_size * case.requested_output_tokens * 4
        + case.batch_size
        + 128
    )
    while not engine.is_finished():
        if step_index > guard:
            raise RuntimeError("profile case exceeded the step guard")

        step_started_ns = clock_ns()
        (_outputs, num_tokens), target_cuda_ns = step_timer.measure(
            engine.step
        )
        step_finished_ns = clock_ns()
        step_index += 1
        if (
            isinstance(num_tokens, bool)
            or not isinstance(num_tokens, int)
        ):
            raise ValueError("engine token count must be an integer")
        if num_tokens >= 0 or -num_tokens != case.batch_size:
            continue

        target_batch_steps += 1
        if (
            target_batch_steps <= case.warmup_steps
            or len(measured_rows) >= case.measured_steps
        ):
            continue
        observation = engine.last_step_observation
        if not isinstance(observation, Mapping):
            raise ValueError("engine step observation is missing")
        timeline = observation.get("command_timeline_step")
        if not isinstance(timeline, Mapping):
            raise ValueError("command timeline step is missing")
        timeline_with_wall = dict(timeline)
        timeline_with_wall["step_wall_ns"] = _positive_int(
            step_finished_ns - step_started_ns,
            "step_wall_ns",
        )
        component_ns = components_from_timeline_step(
            timeline_with_wall,
            target_cuda_ns=_positive_int(
                int(target_cuda_ns),
                "target_cuda_ns",
            ),
        )
        measured_rows.append({
            "schema_version": PROFILE_ROW_SCHEMA_VERSION,
            "case_id": f"{case.case_id}-s{step_index}",
            "load": case.load,
            "source_commit": case.source_commit,
            "batch_size": case.batch_size,
            "context_bucket": case.context_bucket,
            "burst_width": case.burst_width,
            "offered_arrival_offsets_ns": list(
                case.arrival_offsets_ns
            ),
            "component_ns": component_ns,
            "wall_ns": timeline_with_wall["step_wall_ns"],
            "committed_tokens": case.batch_size,
            "cuda_reserved_bytes": _memory_reserved_bytes(
                observation
            ),
        })
    if len(measured_rows) != case.measured_steps:
        raise RuntimeError(
            f"{case.case_id} produced {len(measured_rows)} of "
            f"{case.measured_steps} required samples"
        )
    return measured_rows


class CudaTargetStepTimer:
    def __init__(self, engine, *, torch_module=None):
        self._runner = engine.model_runner
        if torch_module is None:
            import torch

            torch_module = torch
        self._torch = torch_module

    def measure(self, operation):
        original = self._runner.run_model
        events = []

        def timed_run_model(*args, **kwargs):
            started = self._torch.cuda.Event(enable_timing=True)
            finished = self._torch.cuda.Event(enable_timing=True)
            started.record()
            result = original(*args, **kwargs)
            finished.record()
            events.append((started, finished))
            return result

        self._runner.run_model = timed_run_model
        try:
            result = operation()
            self._torch.cuda.synchronize()
        finally:
            self._runner.run_model = original
        if len(events) != 1:
            raise RuntimeError(
                "ordinary baseline step must execute one target forward"
            )
        elapsed_ns = int(
            round(events[0][0].elapsed_time(events[0][1]) * 1_000_000)
        )
        return result, max(1, elapsed_ns)


def _validate_profile_row(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("profile row must be a mapping")
    if row.get("schema_version") != PROFILE_ROW_SCHEMA_VERSION:
        raise ValueError("profile row schema version is invalid")
    load = _text(row.get("load"), "load")
    if load not in FROZEN_LOADS:
        raise ValueError("load is outside the frozen inventory")
    components = row.get("component_ns")
    if not isinstance(components, Mapping):
        raise ValueError("component_ns must be a mapping")
    expected_components = {
        "target_cuda",
        "graph_launch_gap",
        "scheduler",
        "token_d2h_publication",
        "batch_binding",
        "unattributed",
    }
    if set(components) != expected_components:
        raise ValueError("profile component inventory is incomplete")
    normalized_components = {
        name: _non_negative_int(components[name], name)
        for name in sorted(expected_components)
    }
    wall_ns = _positive_int(row.get("wall_ns"), "wall_ns")
    if sum(normalized_components.values()) != wall_ns:
        raise ValueError("profile components do not equal wall time")
    normalized = {
        "schema_version": PROFILE_ROW_SCHEMA_VERSION,
        "case_id": _text(row.get("case_id"), "case_id"),
        "load": load,
        "source_commit": _text(
            row.get("source_commit"),
            "source_commit",
        ),
        "batch_size": _positive_int(
            row.get("batch_size"),
            "batch_size",
        ),
        "context_bucket": _positive_int(
            row.get("context_bucket"),
            "context_bucket",
        ),
        "burst_width": _positive_int(
            row.get("burst_width"),
            "burst_width",
        ),
        "component_ns": normalized_components,
        "wall_ns": wall_ns,
        "committed_tokens": _positive_int(
            row.get("committed_tokens"),
            "committed_tokens",
        ),
        "cuda_reserved_bytes": _non_negative_int(
            row.get("cuda_reserved_bytes"),
            "cuda_reserved_bytes",
        ),
    }
    if "offered_arrival_offsets_ns" in row:
        offsets = row["offered_arrival_offsets_ns"]
        if (
            not isinstance(offsets, (list, tuple))
            or len(offsets) != normalized["batch_size"]
        ):
            raise ValueError("offered arrival offsets are invalid")
        normalized_offsets = [
            _non_negative_int(value, "arrival_offset_ns")
            for value in offsets
        ]
        if normalized_offsets != sorted(normalized_offsets):
            raise ValueError("offered arrival offsets are invalid")
        normalized["offered_arrival_offsets_ns"] = normalized_offsets
    return normalized


def _optimistic_headroom_ratio(row: Mapping[str, object]) -> float:
    components = row["component_ns"]
    removable_ns = sum(
        components[name] for name in _AMORTIZABLE_COMPONENTS
    )
    irreducible_ns = row["wall_ns"] - removable_ns
    if irreducible_ns <= 0:
        raise ValueError("optimistic irreducible time must be positive")
    value = row["wall_ns"] / irreducible_ns - 1.0
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("optimistic headroom is invalid")
    return value


def build_ceiling_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("profile rows must be non-empty")
    normalized = [_validate_profile_row(row) for row in rows]
    case_ids = [row["case_id"] for row in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case ID")
    observed_loads = {row["load"] for row in normalized}
    if observed_loads != set(FROZEN_LOADS):
        raise ValueError("load inventory is incomplete")
    source_commits = {row["source_commit"] for row in normalized}
    source_exact = (
        len(source_commits) == 1
        and len(next(iter(source_commits))) == 40
        and all(
            character in "0123456789abcdef"
            for character in next(iter(source_commits))
        )
    )
    by_load = {}
    for load in FROZEN_LOADS:
        values = [
            _optimistic_headroom_ratio(row)
            for row in normalized
            if row["load"] == load
        ]
        by_load[load] = statistics.median(values)
    summary = {
        "schema_version": ceiling.CEILING_SUMMARY_SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": source_exact,
        "row_count": len(normalized),
        "low_headroom_ratio": by_load["low"],
        "medium_headroom_ratio": by_load["medium"],
        "high_headroom_ratio": by_load["high"],
    }
    summary["classification"] = ceiling.classify_ceiling(summary)
    return summary


def validate_frozen_profile_inventory(
    rows: Sequence[Mapping[str, object]],
    *,
    source_commit: str,
) -> list[dict[str, object]]:
    commit = _source_commit(source_commit)
    normalized = [_validate_profile_row(row) for row in rows]
    cases = build_frozen_case_inventory(commit)
    expected = {
        (
            case.load,
            case.batch_size,
            case.context_bucket,
            case.burst_width,
        ): case
        for case in cases
    }
    observed: dict[tuple[str, int, int, int], list[dict[str, object]]] = {
        key: [] for key in expected
    }
    for row in normalized:
        key = (
            row["load"],
            row["batch_size"],
            row["context_bucket"],
            row["burst_width"],
        )
        case = expected.get(key)
        if (
            case is None
            or row["source_commit"] != commit
            or row["committed_tokens"] != case.batch_size
            or row.get("offered_arrival_offsets_ns")
            != list(case.arrival_offsets_ns)
            or not row["case_id"].startswith(case.case_id + "-s")
        ):
            raise ValueError("frozen profile inventory is invalid")
        observed[key].append(row)
    if any(
        len(observed[key]) != case.measured_steps
        for key, case in expected.items()
    ):
        raise ValueError("frozen profile inventory is incomplete")
    return normalized


def build_optimistic_cost_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("profile rows must be non-empty")
    normalized = [_validate_profile_row(row) for row in rows]
    result = []
    for row in normalized:
        components = row["component_ns"]
        amortized_once_ns = sum(
            components[name] for name in _AMORTIZABLE_COMPONENTS
        )
        irreducible_per_step_ns = (
            row["wall_ns"] - amortized_once_ns
        )
        for width in ceiling.SUPPORTED_BURST_WIDTHS:
            result.append({
                "schema_version": ceiling.COST_SAMPLE_SCHEMA_VERSION,
                "sample_id": f"{row['case_id']}-k{width}",
                "batch_size": row["batch_size"],
                "context_bucket": row["context_bucket"],
                "burst_width": width,
                "duration_ns": (
                    irreducible_per_step_ns * width
                    + amortized_once_ns
                ),
            })
    return result


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    destination = Path(path)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ValueError(
            f"artifact already exists: {destination.name}"
        ) from error


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_ceiling_bundle(
    *,
    output_dir: Path,
    profile_rows: Sequence[Mapping[str, object]],
    cost_rows: Sequence[Mapping[str, object]],
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("artifact destination is not empty")
    else:
        destination.mkdir(parents=True)

    normalized_rows = [
        _validate_profile_row(row) for row in profile_rows
    ]
    summary = build_ceiling_summary(normalized_rows)
    table = ceiling.build_frozen_cost_table(
        cost_rows,
        source_identity,
    )
    artifact = {
        "schema_version": ceiling.ARTIFACT_SCHEMA_VERSION,
        "source_identity": dict(source_identity),
        "cost_rows": [dict(row) for row in cost_rows],
        "cost_table": table,
        "ceiling_summary": summary,
    }
    verification = ceiling.verify_ceiling_artifact(artifact)

    _write_bytes_exclusive(
        destination / "raw_rows.jsonl",
        b"".join(
            json.dumps(
                row,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
            for row in normalized_rows
        ),
    )
    _write_bytes_exclusive(
        destination / "cost_table.json",
        _canonical_json_bytes(table),
    )
    _write_bytes_exclusive(
        destination / "ceiling_summary.json",
        _canonical_json_bytes(summary),
    )
    _write_bytes_exclusive(
        destination / "source_manifest.json",
        _canonical_json_bytes(dict(source_identity)),
    )
    _write_bytes_exclusive(
        destination / "remote_verify.json",
        _canonical_json_bytes(verification),
    )
    return verification


def run_profile_inventory(
    *,
    model: str,
    cases: Sequence[CeilingProfileCase],
    output_dir: Path,
    source_identity: Mapping[str, object],
    engine_factory,
    case_runner=run_profile_case,
    sampling_params_factory,
    step_timer_factory,
) -> dict[str, object]:
    _text(model, "model")
    if not isinstance(cases, Sequence) or not cases:
        raise ValueError("case inventory must be non-empty")
    normalized_cases = tuple(cases)
    if any(
        not isinstance(case, CeilingProfileCase)
        for case in normalized_cases
    ):
        raise ValueError("case inventory is invalid")
    if not isinstance(source_identity, Mapping):
        raise ValueError("source identity must be a mapping")
    bound_source_identity = dict(source_identity)
    source_commit = _source_commit(
        bound_source_identity.get("source_commit")
    )
    if any(
        case.source_commit != source_commit
        for case in normalized_cases
    ):
        raise ValueError("case source commit does not match manifest")
    engine = engine_factory(model, **dict(_ENGINE_CONFIG))
    rows = []
    try:
        hf_config = getattr(
            getattr(
                getattr(engine, "model_runner", None),
                "config",
                None,
            ),
            "hf_config",
            None,
        )
        runtime_dtype = getattr(hf_config, "torch_dtype", None)
        if runtime_dtype is None:
            runtime_dtype = getattr(hf_config, "dtype", None)
        runtime_dtype = _text(str(runtime_dtype), "runtime_dtype")
        declared_dtype = bound_source_identity.get("dtype")
        if (
            declared_dtype is not None
            and declared_dtype != runtime_dtype
        ):
            raise ValueError(
                "source manifest dtype does not match runtime dtype"
            )
        bound_source_identity["dtype"] = runtime_dtype
        step_timer = step_timer_factory(engine)
        begin_repeat = getattr(
            engine,
            "begin_command_timeline_repeat",
            None,
        )
        end_repeat = getattr(
            engine,
            "end_command_timeline_repeat",
            None,
        )
        if not callable(begin_repeat) or not callable(end_repeat):
            raise ValueError(
                "engine command timeline repeat controls are unavailable"
            )
        for repeat_index, case in enumerate(normalized_cases):
            begin_repeat(
                repeat_index,
                request_set_sha256=_case_request_set_sha256(case),
            )
            try:
                rows.extend(case_runner(
                    engine,
                    case,
                    sampling_params_factory=sampling_params_factory,
                    step_timer=step_timer,
                ))
            finally:
                end_repeat()
    finally:
        engine.exit()
    rows = validate_frozen_profile_inventory(
        rows,
        source_commit=source_commit,
    )
    cost_rows = build_optimistic_cost_rows(rows)
    return write_ceiling_bundle(
        output_dir=Path(output_dir),
        profile_rows=rows,
        cost_rows=cost_rows,
        source_identity=bound_source_identity,
    )


def _sha256_files(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _checkpoint_sha256(model: Path) -> str:
    root = Path(model)
    if not root.is_dir():
        raise ValueError("model path must be a directory")
    files = [
        path for path in root.rglob("*")
        if path.is_file()
    ]
    if not files:
        raise ValueError("model path contains no checkpoint files")
    return _sha256_files(root, files)


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        path
        for path in (root / "tinyvllm").rglob("*.py")
        if path.is_file()
    ]
    paths.extend((
        root / "tools" / "slo_cohort_burst_ceiling.py",
        root / "tools" / "profile_slo_cohort_burst_ceiling.py",
        root / "tools" / "run_slo_cohort_burst_remote.py",
    ))
    if any(not path.is_file() for path in paths):
        raise ValueError("source snapshot is incomplete")
    return _sha256_files(root, paths)


def _gpu_identity() -> tuple[str, str]:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not isinstance(visible_device, str)
        or not visible_device
        or "," in visible_device
    ):
        raise RuntimeError(
            "exactly one CUDA_VISIBLE_DEVICES selector is required"
        )
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            visible_device,
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi GPU identity query failed: "
            + result.stderr.strip()
        )
    rows = [
        row.strip() for row in result.stdout.splitlines()
        if row.strip()
    ]
    if len(rows) != 1 or "," not in rows[0]:
        raise RuntimeError("exactly one visible GPU is required")
    uuid, name = (part.strip() for part in rows[0].split(",", 1))
    return _text(uuid, "gpu_uuid"), _text(name, "gpu_name")


def _config_sha256(cases: Sequence[CeilingProfileCase]) -> str:
    payload = {
        "engine_config": _ENGINE_CONFIG,
        "cases": [
            {
                "case_id": case.case_id,
                "load": case.load,
                "batch_size": case.batch_size,
                "context_bucket": case.context_bucket,
                "burst_width": case.burst_width,
                "arrival_offsets_ns": list(case.arrival_offsets_ns),
                "requested_output_tokens": (
                    case.requested_output_tokens
                ),
                "warmup_steps": case.warmup_steps,
                "measured_steps": case.measured_steps,
            }
            for case in cases
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object"
                )
            rows.append(value)
    if not rows:
        raise ValueError("raw profile rows are empty")
    return rows


def verify_ceiling_bundle(artifact_dir: Path) -> dict[str, object]:
    root = Path(artifact_dir)
    rows = _load_jsonl(root / "raw_rows.jsonl")
    source_identity = _load_json(root / "source_manifest.json")
    if not isinstance(source_identity, Mapping):
        raise ValueError("source manifest must be a mapping")
    source_commit = _source_commit(source_identity.get("source_commit"))
    normalized_rows = validate_frozen_profile_inventory(
        rows,
        source_commit=source_commit,
    )
    cost_rows = build_optimistic_cost_rows(normalized_rows)
    cost_table = _load_json(root / "cost_table.json")
    summary = _load_json(root / "ceiling_summary.json")
    rebuilt_summary = build_ceiling_summary(normalized_rows)
    if summary != rebuilt_summary:
        raise ValueError(
            "ceiling summary does not match raw profile rows"
        )
    artifact = {
        "schema_version": ceiling.ARTIFACT_SCHEMA_VERSION,
        "source_identity": source_identity,
        "cost_rows": cost_rows,
        "cost_table": cost_table,
        "ceiling_summary": rebuilt_summary,
    }
    verification = ceiling.verify_ceiling_artifact(artifact)
    recorded_verification = _load_json(root / "remote_verify.json")
    if recorded_verification != verification:
        raise ValueError(
            "recorded verification does not reconstruct"
        )
    return verification


def run_cli(args) -> int:
    cases = build_frozen_case_inventory(args.source_commit)
    gpu_uuid, gpu_name = _gpu_identity()
    source_identity = {
        "source_commit": args.source_commit,
        "source_patch_sha256": _source_tree_sha256(),
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": _checkpoint_sha256(Path(args.model)),
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "tensor_parallel_size": 1,
        "config_sha256": _config_sha256(cases),
    }
    from tinyvllm.engine.llm_engine import LLMEngine
    from tinyvllm.sampling_params import SamplingParams

    verification = run_profile_inventory(
        model=args.model,
        cases=cases,
        output_dir=Path(args.output_dir),
        source_identity=source_identity,
        engine_factory=LLMEngine,
        sampling_params_factory=SamplingParams,
        step_timer_factory=CudaTargetStepTimer,
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


def verify_cli(args) -> int:
    artifact_dir = Path(args.artifact_dir)
    verification = verify_ceiling_bundle(artifact_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_exclusive(
        output,
        _canonical_json_bytes(verification),
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile the SLO cohort-burst Stage-0 ceiling",
    )
    parser.add_argument("--mode", required=True, choices=("run", "verify"))
    parser.add_argument("--model")
    parser.add_argument("--run-tag")
    parser.add_argument("--source-commit")
    parser.add_argument("--output-dir")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.mode == "run":
        for name in (
            "model",
            "run_tag",
            "source_commit",
            "output_dir",
        ):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required")
        _source_commit(args.source_commit)
    else:
        for name in ("artifact_dir", "output"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode == "run":
        return run_cli(args)
    return verify_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
