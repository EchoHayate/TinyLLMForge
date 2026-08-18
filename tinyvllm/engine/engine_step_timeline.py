from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import asdict, dataclass, replace
from itertools import count
import time


SCHEMA_VERSION = 1
MAX_DETAIL_BYTES = 4096
ABSOLUTE_TOLERANCE_NS = 2_000_000
ENGINE_STEP_PHASES = (
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
COMMAND_BEARING_PHASES = frozenset({
    "ordinary_or_first_target_dispatch",
    "speculative_prepare",
    "proposal_kv_prepare_commit",
    "proposal_lifecycle_finalize_prepare",
    "proposal_lifecycle_finalize_commit",
    "side_state_seal",
    "residency_precommit_or_seal",
})
_ACTIVE_TRACE = ContextVar(
    "tinyvllm_engine_step_trace",
    default=None,
)


def _nonnegative_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value, name):
    if value is not None:
        _nonnegative_int(value, name)
    return value


def _optional_sha256(value, name):
    if value is None:
        return value
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 or None")
    return value


def _bounded_detail(value):
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_DETAIL_BYTES:
        return text
    return encoded[:MAX_DETAIL_BYTES].decode(
        "utf-8",
        errors="ignore",
    )


def _invalid_conservation(detail):
    return {
        "serial_phase_sum_ns": None,
        "command_critical_path_ns": None,
        "acknowledged_wait_ns": None,
        "attributed_ns": None,
        "residual_ns": None,
        "tolerance_ns": None,
        "status": "invalid",
        "detail": _bounded_detail(detail),
        "passed": False,
    }


def _interval_union(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, finish in ordered[1:]:
        previous = merged[-1]
        if start <= previous[1]:
            previous[1] = max(previous[1], finish)
        else:
            merged.append([start, finish])
    return [tuple(interval) for interval in merged]


def _interval_total(intervals):
    return sum(finish - start for start, finish in intervals)


def _require_non_overlapping(intervals, name):
    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"{name} intervals overlap")
    return ordered


def _subtract_intervals(intervals, exclusions):
    exclusions = _interval_union(exclusions)
    remaining = []
    for start, finish in intervals:
        fragments = [(start, finish)]
        for excluded_start, excluded_finish in exclusions:
            next_fragments = []
            for fragment_start, fragment_finish in fragments:
                if (
                    excluded_finish <= fragment_start
                    or excluded_start >= fragment_finish
                ):
                    next_fragments.append(
                        (fragment_start, fragment_finish)
                    )
                    continue
                if fragment_start < excluded_start:
                    next_fragments.append(
                        (fragment_start, excluded_start)
                    )
                if excluded_finish < fragment_finish:
                    next_fragments.append(
                        (excluded_finish, fragment_finish)
                    )
            fragments = next_fragments
        remaining.extend(fragments)
    return remaining


def _phase_rows(
    step,
    *,
    require_intervals,
    require_inventory=False,
):
    phases = step.get("phases")
    if not isinstance(phases, dict):
        raise ValueError("step phases must be a dict")
    if require_inventory and tuple(phases) != ENGINE_STEP_PHASES:
        raise ValueError("step phase inventory is invalid")
    durations = []
    intervals = []
    executed_names = set()
    for phase_name, phase in phases.items():
        if not isinstance(phase_name, str) or not isinstance(phase, dict):
            raise ValueError("step phase rows are malformed")
        duration = phase.get("duration_ns")
        _nonnegative_int(duration, f"{phase_name}.duration_ns")
        durations.append(duration)
        if not require_intervals:
            continue
        executed = phase.get("executed")
        if not isinstance(executed, bool):
            raise ValueError(
                f"{phase_name}.executed must be a bool"
            )
        started = phase.get("started_monotonic_ns")
        finished = phase.get("finished_monotonic_ns")
        if not executed:
            if (
                started is not None
                or finished is not None
                or duration != 0
            ):
                raise ValueError(
                    f"{phase_name} skipped phase row is malformed"
                )
            continue
        executed_names.add(phase_name)
        _nonnegative_int(
            started,
            f"{phase_name}.started_monotonic_ns",
        )
        _nonnegative_int(
            finished,
            f"{phase_name}.finished_monotonic_ns",
        )
        if finished < started or finished - started != duration:
            raise ValueError(
                f"{phase_name} phase timestamps are malformed"
            )
        intervals.append((started, finished))
    _require_non_overlapping(intervals, "step phase")
    return sum(durations), intervals, frozenset(executed_names)


def _validate_interval(interval, name, containing):
    start, finish = interval
    _nonnegative_int(start, f"{name}.start")
    _nonnegative_int(finish, f"{name}.finish")
    if finish < start:
        raise ValueError(f"{name} duration is negative")
    if containing is not None:
        containing_start, containing_finish = containing
        if start < containing_start or finish > containing_finish:
            raise ValueError(
                f"{name} lies outside the engine step"
            )
    return interval


def _command_components(
    step,
    command_rows,
    phase_intervals,
    *,
    command_data_required,
):
    if not isinstance(command_rows, (list, tuple)):
        raise ValueError("command rows must be a list or tuple")
    engine_step_id = step.get("engine_step_id")
    _nonnegative_int(engine_step_id, "engine_step_id")
    step_start = step.get("started_monotonic_ns")
    step_finish = step.get("finished_monotonic_ns")
    _nonnegative_int(step_start, "started_monotonic_ns")
    _nonnegative_int(step_finish, "finished_monotonic_ns")
    if step_finish < step_start:
        raise ValueError("step timestamps are malformed")
    containing = (step_start, step_finish)

    command_intervals = []
    ack_intervals = []
    matching_rows = 0
    for index, row in enumerate(command_rows):
        if not isinstance(row, dict):
            raise ValueError(f"command row {index} is malformed")
        if row.get("rank") != 0:
            continue
        if row.get("engine_step_id") != engine_step_id:
            continue
        matching_rows += 1
        local_start = row.get("local_method_started_monotonic_ns")
        local_finish = row.get("local_method_finished_monotonic_ns")
        if local_start is None or local_finish is None:
            raise ValueError(
                f"command row {index} is missing local method timestamps"
            )
        command_intervals.append(
            _validate_interval(
                (local_start, local_finish),
                f"command row {index} local method",
                containing,
            )
        )
        ack_start = row.get("ack_wait_started_monotonic_ns")
        ack_finish = row.get("ack_wait_finished_monotonic_ns")
        if (ack_start is None) != (ack_finish is None):
            raise ValueError(
                f"command row {index} ack wait is partially missing"
            )
        if ack_start is None:
            continue
        post_local_start = max(ack_start, local_finish)
        ack_intervals.append(
            _validate_interval(
                (post_local_start, ack_finish),
                f"command row {index} post-local ack wait",
                containing,
            )
        )

    if command_data_required and matching_rows == 0:
        raise ValueError(
            "matching rank-zero command rows are required"
        )
    command_intervals = _require_non_overlapping(
        command_intervals,
        "rank-zero command",
    )
    ack_intervals = _require_non_overlapping(
        ack_intervals,
        "post-local ack wait",
    )
    _require_non_overlapping(
        command_intervals + ack_intervals,
        "command and acknowledged-wait",
    )
    ack_intervals = _interval_union(
        _subtract_intervals(ack_intervals, command_intervals)
    )
    attributed_command_intervals = (
        command_intervals + ack_intervals
    )
    serial_phase_intervals = _subtract_intervals(
        phase_intervals,
        attributed_command_intervals,
    )
    return (
        _interval_total(serial_phase_intervals),
        _interval_total(command_intervals),
        _interval_total(ack_intervals),
    )


def compute_step_conservation(
    step,
    command_rows=None,
    *,
    command_critical_path_ns=None,
    acknowledged_wait_ns=None,
):
    try:
        if not isinstance(step, dict):
            raise ValueError("step must be a dict")
        step_wall_ns = step.get("step_wall_ns")
        _nonnegative_int(step_wall_ns, "step_wall_ns")
        explicit = (
            command_critical_path_ns is not None
            or acknowledged_wait_ns is not None
        )
        if explicit:
            if command_rows is not None:
                raise ValueError(
                    "command rows and explicit command totals are "
                    "mutually exclusive"
                )
            if (
                command_critical_path_ns is None
                or acknowledged_wait_ns is None
            ):
                raise ValueError(
                    "both explicit command totals are required"
                )
            _nonnegative_int(
                command_critical_path_ns,
                "command_critical_path_ns",
            )
            _nonnegative_int(
                acknowledged_wait_ns,
                "acknowledged_wait_ns",
            )
            serial_phase_sum_ns, _, _ = _phase_rows(
                step,
                require_intervals=False,
            )
        else:
            if command_rows is None:
                raise ValueError("command rows are required")
            _, phase_intervals, executed_phases = _phase_rows(
                step,
                require_intervals=True,
                require_inventory=True,
            )
            (
                serial_phase_sum_ns,
                command_critical_path_ns,
                acknowledged_wait_ns,
            ) = _command_components(
                step,
                command_rows,
                phase_intervals,
                command_data_required=bool(
                    executed_phases & COMMAND_BEARING_PHASES
                ),
            )
        attributed_ns = (
            serial_phase_sum_ns
            + command_critical_path_ns
            + acknowledged_wait_ns
        )
        residual_ns = step_wall_ns - attributed_ns
        tolerance_ns = max(
            ABSOLUTE_TOLERANCE_NS,
            (step_wall_ns + 99) // 100,
        )
        if residual_ns < 0:
            raise ValueError(
                "step timing is over-attributed with negative residual"
            )
        return {
            "serial_phase_sum_ns": serial_phase_sum_ns,
            "command_critical_path_ns": command_critical_path_ns,
            "acknowledged_wait_ns": acknowledged_wait_ns,
            "attributed_ns": attributed_ns,
            "residual_ns": residual_ns,
            "tolerance_ns": tolerance_ns,
            "status": "ok",
            "detail": "",
            "passed": residual_ns <= tolerance_ns,
        }
    except (TypeError, ValueError) as error:
        return _invalid_conservation(error)


@dataclass(frozen=True)
class EngineStepTraceIdentity:
    engine_step_id: int
    repeat_index: int
    request_set_sha256: str | None
    batch_kind: str
    speculative_selected_sequence_ids_sha256: str | None

    def __post_init__(self):
        _nonnegative_int(self.engine_step_id, "engine_step_id")
        _nonnegative_int(self.repeat_index, "repeat_index")
        _optional_sha256(
            self.request_set_sha256,
            "request_set_sha256",
        )
        if not isinstance(self.batch_kind, str) or not self.batch_kind:
            raise ValueError("batch_kind must be non-empty")
        _optional_sha256(
            self.speculative_selected_sequence_ids_sha256,
            "speculative_selected_sequence_ids_sha256",
        )


def active_engine_step_trace():
    return _ACTIVE_TRACE.get()


@contextmanager
def engine_step_trace_scope(identity):
    if identity is not None and not isinstance(
        identity,
        EngineStepTraceIdentity,
    ):
        raise ValueError(
            "engine step trace identity must be "
            "EngineStepTraceIdentity or None"
        )
    token = _ACTIVE_TRACE.set(identity)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


class EngineStepTimelineRecorder:

    def __init__(
        self,
        *,
        enabled,
        max_steps=8192,
        clock_ns=time.monotonic_ns,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        if not callable(clock_ns):
            raise ValueError("clock_ns must be callable")
        self._enabled = enabled
        if enabled:
            if (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or max_steps <= 0
            ):
                raise ValueError("max_steps must be a positive integer")
            self._max_steps = max_steps
        else:
            self._max_steps = 0
        self._clock_ns = clock_ns
        self._step_ids = count()
        self._steps = []
        self._dropped_steps = 0
        self._active_identity = None
        self._active_row = None
        self._active_phase = None
        self._active_context_token = None

    @classmethod
    def disabled(cls):
        return cls(enabled=False)

    @property
    def enabled(self):
        return self._enabled

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def active(self):
        return self._active_identity is not None

    def begin_step(
        self,
        *,
        repeat_index,
        request_set_sha256,
        batch_kind,
        speculative_selected_sequence_ids_sha256,
    ):
        if not self._enabled:
            return None
        if self._active_identity is not None:
            raise RuntimeError("engine step recorder already has an active step")
        identity = EngineStepTraceIdentity(
            engine_step_id=next(self._step_ids),
            repeat_index=repeat_index,
            request_set_sha256=request_set_sha256,
            batch_kind=batch_kind,
            speculative_selected_sequence_ids_sha256=(
                speculative_selected_sequence_ids_sha256
            ),
        )
        started_ns = self._clock_ns()
        row = {
            **asdict(identity),
            "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": None,
            "step_wall_ns": None,
            "phases": {
                phase: {
                    "executed": False,
                    "started_monotonic_ns": None,
                    "finished_monotonic_ns": None,
                    "duration_ns": 0,
                }
                for phase in ENGINE_STEP_PHASES
            },
            "serial_phase_sum_ns": None,
            "command_critical_path_ns": None,
            "acknowledged_wait_ns": None,
            "step_residual_ns": None,
            "conservation_tolerance_ns": None,
            "conservation_status": "pending",
            "conservation_detail": "",
            "conservation_passed": False,
            "status": "active",
            "error_type": "",
            "detail": "",
        }
        self._active_identity = identity
        self._active_row = row
        self._active_context_token = _ACTIVE_TRACE.set(identity)
        return identity

    def bind_step_identity(
        self,
        identity,
        *,
        batch_kind,
        speculative_selected_sequence_ids_sha256,
    ):
        if not self._enabled:
            return None
        if identity != self._active_identity:
            raise ValueError(
                "identity does not match active step identity"
            )
        updated = replace(
            identity,
            batch_kind=batch_kind,
            speculative_selected_sequence_ids_sha256=(
                speculative_selected_sequence_ids_sha256
            ),
        )
        _ACTIVE_TRACE.reset(self._active_context_token)
        self._active_context_token = _ACTIVE_TRACE.set(updated)
        self._active_identity = updated
        self._active_row.update(asdict(updated))
        return updated

    @contextmanager
    def phase(self, phase_name):
        if not self._enabled:
            yield
            return
        if self._active_identity is None:
            raise RuntimeError("engine step recorder has no active step")
        if phase_name not in ENGINE_STEP_PHASES:
            raise ValueError(f"unknown engine step phase {phase_name}")
        if self._active_phase is not None:
            raise RuntimeError(
                "engine step recorder already has active phase "
                f"{self._active_phase}"
            )
        phase = self._active_row["phases"][phase_name]
        if phase["executed"]:
            raise RuntimeError(
                f"engine step phase {phase_name} already executed"
            )
        started_ns = self._clock_ns()
        self._active_phase = phase_name
        try:
            yield
        finally:
            finished_ns = self._clock_ns()
            self._active_phase = None
            if finished_ns < started_ns:
                raise ValueError(
                    f"engine step phase {phase_name} duration is negative"
                )
            phase.update({
                "executed": True,
                "started_monotonic_ns": started_ns,
                "finished_monotonic_ns": finished_ns,
                "duration_ns": finished_ns - started_ns,
            })

    def finish_step(
        self,
        identity,
        *,
        error=None,
        command_rows=None,
    ):
        if not self._enabled:
            return None
        if self._active_identity is None:
            raise RuntimeError("engine step recorder has no active step")
        if identity != self._active_identity:
            raise ValueError(
                "identity does not match active step identity"
            )
        if self._active_phase is not None:
            raise RuntimeError(
                "cannot finish engine step with an active phase"
            )
        try:
            finished_ns = self._clock_ns()
            row = self._active_row
            if finished_ns < row["started_monotonic_ns"]:
                raise ValueError("engine step duration is negative")
            row["finished_monotonic_ns"] = finished_ns
            row["step_wall_ns"] = (
                finished_ns - row["started_monotonic_ns"]
            )
            row["status"] = "ok" if error is None else "error"
            if error is not None:
                row["error_type"] = type(error).__name__[:128]
                row["detail"] = _bounded_detail(error)
            if command_rows is None:
                conservation = _invalid_conservation(
                    "command rows are required"
                )
            else:
                conservation = compute_step_conservation(
                    row,
                    command_rows,
                )
            row.update({
                "serial_phase_sum_ns": conservation[
                    "serial_phase_sum_ns"
                ],
                "command_critical_path_ns": conservation[
                    "command_critical_path_ns"
                ],
                "acknowledged_wait_ns": conservation[
                    "acknowledged_wait_ns"
                ],
                "step_residual_ns": conservation["residual_ns"],
                "conservation_tolerance_ns": conservation[
                    "tolerance_ns"
                ],
                "conservation_status": conservation["status"],
                "conservation_detail": conservation["detail"],
                "conservation_passed": conservation["passed"],
            })
            if len(self._steps) < self._max_steps:
                self._steps.append(copy.deepcopy(row))
            else:
                self._dropped_steps += 1
            return copy.deepcopy(row)
        finally:
            _ACTIVE_TRACE.reset(self._active_context_token)
            self._active_identity = None
            self._active_row = None
            self._active_phase = None
            self._active_context_token = None

    def snapshot(self):
        return copy.deepcopy({
            "schema_version": SCHEMA_VERSION,
            "enabled": self._enabled,
            "max_steps": self._max_steps,
            "dropped_steps": self._dropped_steps,
            "steps": self._steps,
        })
