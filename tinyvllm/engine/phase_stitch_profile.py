from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable


SCHEMA_VERSION = 1
PHASE_STITCH_EVENTS = (
    "prefill_dispatch_finished",
    "first_token_host_available",
    "prefill_scheduler_commit_finished",
    "next_schedule_started",
    "next_schedule_finished",
    "k8_lease_prepare_finished",
    "first_k8_dispatch_started",
)


def _nonnegative_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value, name):
    _nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _token_ids_sha256(output_token_ids):
    if not isinstance(output_token_ids, tuple):
        raise ValueError("output_token_ids must be a tuple of integers")
    for token_id in output_token_ids:
        _nonnegative_int(token_id, "output_token_ids item")
    payload = json.dumps(
        output_token_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PhaseStitchProfileRecorder:

    def __init__(
        self,
        *,
        enabled,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        if not callable(clock_ns):
            raise ValueError("clock_ns must be callable")
        self.enabled = enabled
        self._clock_ns = clock_ns
        self._active = {}
        self._finalized_sequence_ids = set()
        self._rows = []

    def begin_request(self, sequence_id, prompt_tokens):
        if not self.enabled:
            return
        _nonnegative_int(sequence_id, "sequence_id")
        _positive_int(prompt_tokens, "prompt_tokens")
        if sequence_id in self._finalized_sequence_ids:
            raise ValueError(f"sequence_id {sequence_id} is finalized")
        if sequence_id in self._active:
            raise ValueError(f"sequence_id {sequence_id} is already active")
        self._active[sequence_id] = {
            "sequence_id": sequence_id,
            "prompt_tokens": prompt_tokens,
            "events": [],
            "timestamps_ns": {},
        }

    def mark(self, sequence_id, event):
        if not self.enabled:
            return
        _nonnegative_int(sequence_id, "sequence_id")
        if sequence_id in self._finalized_sequence_ids:
            raise ValueError(f"sequence_id {sequence_id} is finalized")
        request = self._active.get(sequence_id)
        if request is None:
            raise ValueError(
                f"sequence_id {sequence_id} has no active request"
            )
        event_index = len(request["events"])
        if (
            event_index >= len(PHASE_STITCH_EVENTS)
            or event != PHASE_STITCH_EVENTS[event_index]
        ):
            expected = (
                PHASE_STITCH_EVENTS[event_index]
                if event_index < len(PHASE_STITCH_EVENTS)
                else None
            )
            raise ValueError(
                "phase-stitch event order violation: "
                f"expected {expected!r}, received {event!r}"
            )
        timestamp_ns = self._clock_ns()
        _nonnegative_int(timestamp_ns, "clock timestamp")
        if request["events"]:
            previous_event = request["events"][-1]
            previous_timestamp_ns = request["timestamps_ns"][
                previous_event
            ]
            if timestamp_ns < previous_timestamp_ns:
                raise ValueError(
                    "phase-stitch timestamps must be monotonic"
                )
        request["events"].append(event)
        request["timestamps_ns"][event] = timestamp_ns

    def finish_request(self, sequence_id, output_token_ids):
        if not self.enabled:
            return {}
        _nonnegative_int(sequence_id, "sequence_id")
        if sequence_id in self._finalized_sequence_ids:
            raise ValueError(f"sequence_id {sequence_id} is finalized")
        request = self._active.get(sequence_id)
        if request is None:
            raise ValueError(
                f"sequence_id {sequence_id} has no active request"
            )
        if tuple(request["events"]) != PHASE_STITCH_EVENTS:
            raise ValueError(
                "phase-stitch event coverage is incomplete"
            )
        output_sha256 = _token_ids_sha256(output_token_ids)
        timestamps = request["timestamps_ns"]
        adjacent_intervals = {}
        for previous_event, current_event in zip(
            PHASE_STITCH_EVENTS,
            PHASE_STITCH_EVENTS[1:],
        ):
            adjacent_intervals[
                f"{previous_event}_to_{current_event}"
            ] = timestamps[current_event] - timestamps[previous_event]
        row = {
            "sequence_id": sequence_id,
            "prompt_tokens": request["prompt_tokens"],
            "status": "complete",
            "events": list(request["events"]),
            **{
                f"{event}_ns": timestamps[event]
                for event in PHASE_STITCH_EVENTS
            },
            "adjacent_intervals_ns": adjacent_intervals,
            "removable_host_gap_ns": (
                timestamps["first_k8_dispatch_started"]
                - timestamps["first_token_host_available"]
            ),
            "output_token_ids_sha256": output_sha256,
            "event_coverage_complete": True,
        }
        del self._active[sequence_id]
        self._finalized_sequence_ids.add(sequence_id)
        self._rows.append(row)
        return copy.deepcopy(row)

    def snapshot(self):
        if not self.enabled:
            return {
                "schema_version": SCHEMA_VERSION,
                "enabled": False,
                "active": [],
                "rows": [],
            }
        active = []
        for sequence_id in sorted(self._active):
            request = self._active[sequence_id]
            active.append({
                "sequence_id": sequence_id,
                "prompt_tokens": request["prompt_tokens"],
                "events": list(request["events"]),
                "timestamps_ns": dict(request["timestamps_ns"]),
            })
        return copy.deepcopy({
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "active": active,
            "rows": self._rows,
        })
