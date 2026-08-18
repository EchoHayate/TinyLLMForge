from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time


SCHEMA_VERSION = 1
MAX_ERROR_TYPE_LENGTH = 128
_ACTIVE_TRACE = ContextVar(
    "tinyvllm_model_runner_command_trace",
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


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class CommandClockIdentity:
    boot_id: str
    implementation: str
    resolution_s: float
    monotonic: bool
    adjustable: bool
    captured_at_unix_ns: int

    def __post_init__(self):
        if not isinstance(self.boot_id, str) or not self.boot_id:
            raise ValueError("boot_id must be non-empty")
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("clock implementation must be non-empty")
        if (
            not isinstance(self.resolution_s, (int, float))
            or isinstance(self.resolution_s, bool)
            or not math.isfinite(float(self.resolution_s))
            or float(self.resolution_s) <= 0
        ):
            raise ValueError("clock resolution must be positive and finite")
        if self.monotonic is not True or not isinstance(
            self.adjustable,
            bool,
        ):
            raise ValueError(
                "clock must be monotonic with boolean adjustability"
            )
        _nonnegative_int(
            self.captured_at_unix_ns,
            "captured_at_unix_ns",
        )


def read_command_clock_identity():
    info = time.get_clock_info("monotonic")
    return CommandClockIdentity(
        boot_id=Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip(),
        implementation=str(info.implementation),
        resolution_s=float(info.resolution),
        monotonic=bool(info.monotonic),
        adjustable=bool(info.adjustable),
        captured_at_unix_ns=time.time_ns(),
    )


@dataclass(frozen=True)
class CommandTraceIdentity:
    command_id: int
    method_name: str
    requires_ack: bool
    engine_step_id: int | None
    repeat_index: int | None
    request_set_sha256: str | None
    batch_kind: str | None
    speculative_selected_sequence_ids_sha256: str | None
    dispatch_started_monotonic_ns: int
    dispatch_published_monotonic_ns: int

    def __post_init__(self):
        _nonnegative_int(self.command_id, "command_id")
        if not isinstance(self.method_name, str) or not self.method_name:
            raise ValueError("method_name must be non-empty")
        if not isinstance(self.requires_ack, bool):
            raise ValueError("requires_ack must be a bool")
        _optional_nonnegative_int(
            self.engine_step_id,
            "engine_step_id",
        )
        _optional_nonnegative_int(
            self.repeat_index,
            "repeat_index",
        )
        for name in (
            "request_set_sha256",
            "speculative_selected_sequence_ids_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if self.batch_kind is not None and (
            not isinstance(self.batch_kind, str)
            or not self.batch_kind
        ):
            raise ValueError("batch_kind must be non-empty when set")
        _nonnegative_int(
            self.dispatch_started_monotonic_ns,
            "dispatch_started_monotonic_ns",
        )
        _nonnegative_int(
            self.dispatch_published_monotonic_ns,
            "dispatch_published_monotonic_ns",
        )
        if (
            self.dispatch_published_monotonic_ns
            < self.dispatch_started_monotonic_ns
        ):
            raise ValueError("dispatch timestamps are invalid")


def active_model_runner_command_trace():
    return _ACTIVE_TRACE.get()


@contextmanager
def command_trace_scope(identity):
    if identity is not None and not isinstance(
        identity,
        CommandTraceIdentity,
    ):
        raise ValueError(
            "command trace identity must be CommandTraceIdentity or None"
        )
    token = _ACTIVE_TRACE.set(identity)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


class ModelRunnerCommandTimelineRecorder:

    def __init__(
        self,
        *,
        rank,
        max_rows,
        clock_identity,
    ):
        self._rank = _nonnegative_int(rank, "rank")
        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or max_rows <= 0
        ):
            raise ValueError("max_rows must be a positive integer")
        if not isinstance(clock_identity, CommandClockIdentity):
            raise ValueError(
                "clock_identity must be CommandClockIdentity"
            )
        self._enabled = True
        self._max_rows = max_rows
        self._clock_identity = clock_identity
        self._rows = []
        self._rows_by_command_id = {}
        self._active_phases = {}
        self._last_command_id = None
        self._dropped_from_command_id = None
        self._dropped_rows = 0

    @classmethod
    def disabled(cls, rank):
        recorder = cls.__new__(cls)
        recorder._rank = _nonnegative_int(rank, "rank")
        recorder._enabled = False
        recorder._max_rows = 0
        recorder._clock_identity = None
        recorder._rows = []
        recorder._rows_by_command_id = {}
        recorder._active_phases = {}
        recorder._last_command_id = None
        recorder._dropped_from_command_id = None
        recorder._dropped_rows = 0
        return recorder

    @property
    def enabled(self):
        return self._enabled

    def _begin_row(self, identity, phase):
        if not self._enabled:
            return None
        if not isinstance(identity, CommandTraceIdentity):
            raise ValueError("identity must be CommandTraceIdentity")
        command_id = identity.command_id
        if (
            self._last_command_id is not None
            and command_id <= self._last_command_id
        ):
            raise ValueError(
                "command IDs must be strictly increasing"
            )
        self._last_command_id = command_id
        if len(self._rows) >= self._max_rows:
            if self._dropped_from_command_id is None:
                self._dropped_from_command_id = command_id
            self._dropped_rows += 1
            return None
        row = {
            "rank": self._rank,
            **asdict(identity),
            "event_woken_monotonic_ns": None,
            "envelope_read_monotonic_ns": None,
            "method_started_monotonic_ns": None,
            "method_finished_monotonic_ns": None,
            "local_method_started_monotonic_ns": None,
            "local_method_finished_monotonic_ns": None,
            "ack_send_started_monotonic_ns": None,
            "ack_send_finished_monotonic_ns": None,
            "ack_wait_started_monotonic_ns": None,
            "ack_wait_finished_monotonic_ns": None,
            "status": "pending",
            "error_type": "",
        }
        self._rows.append(row)
        self._rows_by_command_id[command_id] = row
        self._active_phases[command_id] = phase
        return row

    def _row(self, command_id):
        if not self._enabled:
            return None
        _nonnegative_int(command_id, "command_id")
        row = self._rows_by_command_id.get(command_id)
        if row is not None:
            return row
        if (
            self._dropped_from_command_id is not None
            and command_id >= self._dropped_from_command_id
        ):
            return None
        raise ValueError(f"unknown command_id {command_id}")

    def _start_phase(self, command_id, phase):
        current = self._active_phases.get(command_id)
        if current not in (None, "dispatched", "received"):
            raise ValueError(
                f"command {command_id} already has active phase {current}"
            )
        self._active_phases[command_id] = phase

    def _finish_phase(self, command_id, phase):
        if self._active_phases.get(command_id) != phase:
            raise ValueError(
                f"command {command_id} is not in {phase} phase"
            )
        self._active_phases.pop(command_id, None)

    def _transition_phase(self, command_id, current, next_phase):
        if self._active_phases.get(command_id) != current:
            raise ValueError(
                f"command {command_id} is not in {current} phase"
            )
        self._active_phases[command_id] = next_phase

    def record_dispatch(self, identity):
        if not self._enabled:
            return
        if self._rank != 0:
            raise ValueError("dispatch rows may only be recorded on rank 0")
        self._begin_row(identity, "dispatched")

    def record_worker_receive(
        self,
        identity,
        *,
        event_woken_monotonic_ns,
        envelope_read_monotonic_ns,
    ):
        if not self._enabled:
            return
        event_woken_monotonic_ns = _nonnegative_int(
            event_woken_monotonic_ns,
            "event_woken_monotonic_ns",
        )
        envelope_read_monotonic_ns = _nonnegative_int(
            envelope_read_monotonic_ns,
            "envelope_read_monotonic_ns",
        )
        if (
            event_woken_monotonic_ns
            < identity.dispatch_published_monotonic_ns
            or envelope_read_monotonic_ns
            < event_woken_monotonic_ns
        ):
            raise ValueError("worker receive timestamps are invalid")
        row = self._begin_row(identity, "received")
        if row is not None:
            row["event_woken_monotonic_ns"] = (
                event_woken_monotonic_ns
            )
            row["envelope_read_monotonic_ns"] = (
                envelope_read_monotonic_ns
            )

    def record_method_start(self, command_id, *, started_ns):
        row = self._row(command_id)
        if row is None:
            return
        started_ns = _nonnegative_int(started_ns, "started_ns")
        minimum_ns = row["dispatch_published_monotonic_ns"]
        if self._rank != 0:
            minimum_ns = row["envelope_read_monotonic_ns"]
        if started_ns < minimum_ns:
            raise ValueError("method start timestamp is invalid")
        self._start_phase(command_id, "method")
        key = (
            "local_method_started_monotonic_ns"
            if self._rank == 0
            else "method_started_monotonic_ns"
        )
        row[key] = started_ns

    def record_method_end(
        self,
        command_id,
        *,
        finished_ns,
        status="ok",
        error_type="",
    ):
        row = self._row(command_id)
        if row is None:
            return
        finished_ns = _nonnegative_int(finished_ns, "finished_ns")
        if status not in ("ok", "error"):
            raise ValueError("status must be 'ok' or 'error'")
        if not isinstance(error_type, str):
            raise ValueError("error_type must be a string")
        started_key = (
            "local_method_started_monotonic_ns"
            if self._rank == 0
            else "method_started_monotonic_ns"
        )
        finished_key = (
            "local_method_finished_monotonic_ns"
            if self._rank == 0
            else "method_finished_monotonic_ns"
        )
        if finished_ns < row[started_key]:
            raise ValueError("method finish timestamp is invalid")
        if row["requires_ack"]:
            self._transition_phase(
                command_id,
                "method",
                "awaiting_ack",
            )
        else:
            self._finish_phase(command_id, "method")
        row[finished_key] = finished_ns
        row["status"] = status
        row["error_type"] = error_type[:MAX_ERROR_TYPE_LENGTH]

    def record_ack_send_start(self, command_id, *, started_ns):
        row = self._row(command_id)
        if row is None:
            return
        if not row["requires_ack"]:
            raise ValueError("ack send requires an acknowledged command")
        started_ns = _nonnegative_int(started_ns, "started_ns")
        if started_ns < row["method_finished_monotonic_ns"]:
            raise ValueError("ack send start timestamp is invalid")
        self._transition_phase(
            command_id,
            "awaiting_ack",
            "ack_send",
        )
        row["ack_send_started_monotonic_ns"] = started_ns

    def record_ack_send_end(self, command_id, *, finished_ns):
        row = self._row(command_id)
        if row is None:
            return
        finished_ns = _nonnegative_int(finished_ns, "finished_ns")
        if finished_ns < row["ack_send_started_monotonic_ns"]:
            raise ValueError("ack send finish timestamp is invalid")
        self._finish_phase(command_id, "ack_send")
        row["ack_send_finished_monotonic_ns"] = finished_ns

    def record_ack_wait_start(self, command_id, *, started_ns):
        row = self._row(command_id)
        if row is None:
            return
        if self._rank != 0 or not row["requires_ack"]:
            raise ValueError(
                "ack wait requires a rank-zero acknowledged command"
            )
        started_ns = _nonnegative_int(started_ns, "started_ns")
        if started_ns < row["dispatch_published_monotonic_ns"]:
            raise ValueError("ack wait start timestamp is invalid")
        self._transition_phase(
            command_id,
            "awaiting_ack",
            "ack_wait",
        )
        row["ack_wait_started_monotonic_ns"] = started_ns

    def record_ack_wait_end(self, command_id, *, finished_ns):
        row = self._row(command_id)
        if row is None:
            return
        finished_ns = _nonnegative_int(finished_ns, "finished_ns")
        if finished_ns < row["ack_wait_started_monotonic_ns"]:
            raise ValueError("ack wait finish timestamp is invalid")
        self._finish_phase(command_id, "ack_wait")
        row["ack_wait_finished_monotonic_ns"] = finished_ns

    def record_ack_wait(self, command_id, *, started_ns, finished_ns):
        self.record_ack_wait_start(command_id, started_ns=started_ns)
        self.record_ack_wait_end(command_id, finished_ns=finished_ns)

    def snapshot(self):
        if self._active_phases:
            raise ValueError(
                "cannot snapshot command timeline with unfinished rows"
            )
        return copy.deepcopy(
            {
                "schema_version": SCHEMA_VERSION,
                "rank": self._rank,
                "enabled": self._enabled,
                "clock": (
                    asdict(self._clock_identity)
                    if self._clock_identity is not None
                    else None
                ),
                "rows": self._rows,
                "dropped_rows": self._dropped_rows,
            }
        )


def _duration(started_ns, finished_ns, name):
    started_ns = _nonnegative_int(started_ns, f"{name} start")
    finished_ns = _nonnegative_int(finished_ns, f"{name} finish")
    if finished_ns < started_ns:
        raise ValueError(f"{name} duration must be non-negative")
    return finished_ns - started_ns


def _method_timestamps(row):
    if (
        row.get("method_started_monotonic_ns") is not None
        or row.get("method_finished_monotonic_ns") is not None
    ):
        return (
            row.get("method_started_monotonic_ns"),
            row.get("method_finished_monotonic_ns"),
        )
    return (
        row.get("local_method_started_monotonic_ns"),
        row.get("local_method_finished_monotonic_ns"),
    )


def compute_command_decomposition(rows):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple")
    copied_rows = [copy.deepcopy(row) for row in rows]
    previous_input_command = {}
    for row in copied_rows:
        if not isinstance(row, dict):
            raise ValueError("each command row must be a dict")
        rank = _nonnegative_int(row.get("rank"), "rank")
        command_id = _nonnegative_int(
            row.get("command_id"),
            "command_id",
        )
        previous = previous_input_command.get(rank)
        if previous is not None:
            if command_id <= previous:
                raise ValueError(
                    "rank-local command order must be monotonic"
                )
            if command_id != previous + 1:
                raise ValueError(
                    f"command {command_id} is missing predecessor "
                    f"{previous + 1} on rank {rank}"
                )
        previous_input_command[rank] = command_id

    result = []
    previous_finished_by_rank = {}
    for row in sorted(
        copied_rows,
        key=lambda item: (item["rank"], item["command_id"]),
    ):
        rank = row["rank"]
        dispatch_published_ns = _nonnegative_int(
            row.get("dispatch_published_monotonic_ns"),
            "dispatch_published_monotonic_ns",
        )
        method_started_ns, method_finished_ns = _method_timestamps(row)
        worker_method_wall_ns = _duration(
            method_started_ns,
            method_finished_ns,
            "worker method",
        )
        worker_queue_wait_ns = _duration(
            dispatch_published_ns,
            method_started_ns,
            "worker queue wait",
        )
        previous_finished_ns = previous_finished_by_rank.get(rank)
        queued_behind_prior_command_ns = (
            0
            if previous_finished_ns is None
            else max(
                0,
                previous_finished_ns - dispatch_published_ns,
            )
        )
        worker_ready_delay_ns = (
            worker_queue_wait_ns - queued_behind_prior_command_ns
        )
        if worker_ready_delay_ns < 0:
            raise ValueError(
                "prior command overlap exceeds worker queue wait"
            )
        cuda_ns = _nonnegative_int(row.get("cuda_ns"), "cuda_ns")
        if cuda_ns > worker_method_wall_ns:
            raise ValueError(
                "cuda_ns may not exceed worker_method_wall_ns"
            )
        row["worker_method_wall_ns"] = worker_method_wall_ns
        row["worker_queue_wait_ns"] = worker_queue_wait_ns
        row["queued_behind_prior_command_ns"] = (
            queued_behind_prior_command_ns
        )
        row["worker_ready_delay_ns"] = worker_ready_delay_ns
        row["worker_non_cuda_upper_bound_ns"] = (
            worker_method_wall_ns - cuda_ns
        )

        ack_started_ns = row.get("ack_wait_started_monotonic_ns")
        ack_finished_ns = row.get("ack_wait_finished_monotonic_ns")
        if ack_started_ns is not None or ack_finished_ns is not None:
            if ack_started_ns is None or ack_finished_ns is None:
                raise ValueError("ack wait timestamps must be paired")
            row["ack_wait_ns"] = _duration(
                ack_started_ns,
                ack_finished_ns,
                "ack wait",
            )
            local_finished_ns = row.get(
                "local_method_finished_monotonic_ns"
            )
            if local_finished_ns is None:
                raise ValueError(
                    "ack wait requires local method finish timestamp"
                )
            local_finished_ns = _nonnegative_int(
                local_finished_ns,
                "local_method_finished_monotonic_ns",
            )
            if ack_finished_ns < max(
                ack_started_ns,
                local_finished_ns,
            ):
                raise ValueError(
                    "post-local ack wait duration must be non-negative"
                )
            row["post_local_ack_wait_ns"] = (
                ack_finished_ns
                - max(ack_started_ns, local_finished_ns)
            )
        else:
            row["ack_wait_ns"] = None
            row["post_local_ack_wait_ns"] = None

        previous_finished_by_rank[rank] = method_finished_ns
        result.append(row)
    return result
