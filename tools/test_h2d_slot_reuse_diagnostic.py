from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_DIAGNOSTIC_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "engine",
    "h2d_slot_reuse_diagnostic.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "h2d_slot_reuse_diagnostic_under_test",
    _DIAGNOSTIC_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_DIAGNOSTIC = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DIAGNOSTIC
_SPEC.loader.exec_module(_DIAGNOSTIC)

H2D_SLOT_REUSE_SCHEMA = _DIAGNOSTIC.H2D_SLOT_REUSE_SCHEMA
H2DSlotReuseDiagnostic = _DIAGNOSTIC.H2DSlotReuseDiagnostic
SlotOccupancy = _DIAGNOSTIC.SlotOccupancy
classify_read_h2d_ordering = _DIAGNOSTIC.classify_read_h2d_ordering


class _DummyEvent:
    def __init__(self, ordinal, elapsed=None):
        self.ordinal = ordinal
        self.elapsed = {} if elapsed is None else dict(elapsed)
        self.recorded_streams = []

    def record(self, stream=None):
        self.recorded_streams.append(stream)

    def elapsed_time(self, other):
        return float(self.elapsed[other.ordinal])


class _DummyStream:
    def __init__(self, stream_id):
        self.stream_id = int(stream_id)
        self.waited = []
        self.operations = []

    def wait_event(self, event):
        self.waited.append(event.ordinal)
        self.operations.append(("wait", event.ordinal))


class _EventFactory:
    def __init__(self):
        self.next_ordinal = 1
        self.events = {}

    def __call__(self):
        event = _DummyEvent(self.next_ordinal)
        self.events[event.ordinal] = event
        self.next_ordinal += 1
        return event


def _diagnostic(slot_count=2, **kwargs):
    events = _EventFactory()
    diagnostic = H2DSlotReuseDiagnostic(
        rank=0,
        slot_count=slot_count,
        event_factory=events,
        stream_id=lambda stream: stream.stream_id,
        **kwargs,
    )
    return diagnostic, events


def _replace_after_read(mode="observe"):
    diagnostic, events = _diagnostic(slot_count=1)
    current = _DummyStream(3)
    copy = _DummyStream(5)
    diagnostic.configure(mode)
    diagnostic.set_context(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.record_read_window(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
        logical_blocks=(1,),
        physical_slots=(0,),
        current_stream=current,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=2,
        bound_generation=0,
    )
    handle = diagnostic.begin_h2d_span(
        copy_batch_ordinal=0,
        copy_span_ordinal=0,
        pairs=((2, 0),),
        copy_stream=copy,
    )
    diagnostic.end_h2d_span(handle, copy_stream=copy)
    return diagnostic, events, current, copy


def test_schema_and_modes_are_frozen():
    assert H2D_SLOT_REUSE_SCHEMA == (
        "qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1"
    )
    diagnostic, _ = _diagnostic()
    assert diagnostic.configure("off") == {
        "rank": 0,
        "mode": "off",
    }
    with pytest.raises(ValueError, match="mode must be one of"):
        diagnostic.configure("serialize")


def test_slot_generation_increments_for_every_assignment():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    first = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=10,
        bound_generation=2,
    )
    second = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=11,
        bound_generation=7,
    )
    third = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=10,
        bound_generation=3,
    )
    assert first.occupancy_generation == 1
    assert second.occupancy_generation == 2
    assert third.occupancy_generation == 3


def test_replacement_occupancy_does_not_inherit_stale_read_event():
    diagnostic, _ = _diagnostic(slot_count=1)
    current = _DummyStream(101)
    diagnostic.configure("observe")
    first = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=4,
        bound_generation=1,
    )
    diagnostic.record_read_window(
        engine_step=3,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=2,
        logical_blocks=(4,),
        physical_slots=(0,),
        current_stream=current,
    )
    second = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=5,
        bound_generation=1,
    )
    assert second.occupancy_generation == 2
    assert diagnostic.predecessor_event_ordinals(first) == (1,)
    assert diagnostic.predecessor_event_ordinals(second) == ()


def test_latest_read_event_supersedes_same_stream():
    diagnostic, _ = _diagnostic(slot_count=1)
    stream = _DummyStream(7)
    diagnostic.configure("observe")
    occupancy = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    for window in (0, 1):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=window,
            logical_blocks=(1,),
            physical_slots=(0,),
            current_stream=stream,
        )
    assert diagnostic.predecessor_event_ordinals(occupancy) == (2,)


def test_distinct_read_streams_are_all_retained():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    occupancy = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    for stream_id in (7, 9):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=stream_id,
            logical_blocks=(1,),
            physical_slots=(0,),
            current_stream=_DummyStream(stream_id),
        )
    assert diagnostic.predecessor_event_ordinals(occupancy) == (1, 2)


@pytest.mark.parametrize(
    ("mode", "expected_waits"),
    (("observe", []), ("control", [1])),
)
def test_h2d_span_waits_only_in_control(mode, expected_waits):
    diagnostic, _, _, copy = _replace_after_read(mode)
    assert copy.waited == expected_waits
    assert diagnostic.retained_event_count == 3


def test_timing_classification_uses_fixed_epsilon():
    assert classify_read_h2d_ordering(0.21, 0.20) == (
        "UNSAFE_OVERLAP_OBSERVED"
    )
    assert classify_read_h2d_ordering(0.20, 0.20) == (
        "ORDERING_AMBIGUOUS"
    )
    assert classify_read_h2d_ordering(-0.21, 0.20) == (
        "READ_COMPLETED_BEFORE_H2D"
    )


def test_drain_returns_frozen_tensor_free_rows_and_releases_events():
    diagnostic, events, _, _ = _replace_after_read()
    events.events[2].elapsed[1] = 0.5
    drained = diagnostic.drain(
        synchronize=lambda: None,
        timing_epsilon_ms=0.2,
    )
    assert drained.overwrite_rows[0].timing_status == (
        "UNSAFE_OVERLAP_OBSERVED"
    )
    assert drained.as_dict()["overwrite_rows"][0][
        "read_done_after_h2d_start_ms"
    ] == 0.5
    with pytest.raises(FrozenInstanceError):
        drained.overwrite_rows[0].control_wait_count = 9
    assert diagnostic.retained_event_count == 0
    assert diagnostic.mode == "observe"


def test_disable_clears_undrained_state():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.configure("off")
    assert diagnostic.retained_event_count == 0
    assert diagnostic.active_occupancies == ()


def test_empty_slot_replacement_is_explicit():
    diagnostic, events = _diagnostic(slot_count=1)
    copy = _DummyStream(5)
    diagnostic.configure("observe")
    diagnostic.set_context(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=2,
        bound_generation=0,
    )
    handle = diagnostic.begin_h2d_span(
        copy_batch_ordinal=0,
        copy_span_ordinal=0,
        pairs=((2, 0),),
        copy_stream=copy,
    )
    diagnostic.end_h2d_span(handle, copy_stream=copy)
    drained = diagnostic.drain(
        synchronize=lambda: None,
        timing_epsilon_ms=0.2,
    )
    assert drained.overwrite_rows[0].timing_status == "NO_PRIOR_OCCUPANCY"
    assert events.next_ordinal == 3


def test_occupied_slot_without_read_is_explicit():
    diagnostic, _ = _diagnostic(slot_count=1)
    copy = _DummyStream(5)
    diagnostic.configure("observe")
    diagnostic.set_context(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=2,
        bound_generation=0,
    )
    handle = diagnostic.begin_h2d_span(
        copy_batch_ordinal=0,
        copy_span_ordinal=0,
        pairs=((2, 0),),
        copy_stream=copy,
    )
    diagnostic.end_h2d_span(handle, copy_stream=copy)
    drained = diagnostic.drain(
        synchronize=lambda: None,
        timing_epsilon_ms=0.2,
    )
    assert drained.overwrite_rows[0].timing_status == "NO_PRIOR_READ"


def test_missing_event_timing_is_a_hard_error():
    diagnostic, _, _, _ = _replace_after_read()
    with pytest.raises(RuntimeError, match="timing"):
        diagnostic.drain(
            synchronize=lambda: None,
            timing_epsilon_ms=0.2,
        )


def test_duplicate_event_ordinal_is_a_hard_error():
    events = _EventFactory()
    diagnostic = H2DSlotReuseDiagnostic(
        rank=0,
        slot_count=1,
        event_factory=lambda: _DummyEvent(1),
        stream_id=lambda stream: stream.stream_id,
    )
    diagnostic.configure("observe")
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.record_read_window(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
        logical_blocks=(1,),
        physical_slots=(0,),
        current_stream=_DummyStream(3),
    )
    with pytest.raises(RuntimeError, match="duplicate event ordinal"):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=1,
            logical_blocks=(1,),
            physical_slots=(0,),
            current_stream=_DummyStream(3),
        )
    assert events.next_ordinal == 1


def test_stale_occupancy_and_generation_mismatch_are_hard_errors():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=2,
        bound_generation=1,
    )
    with pytest.raises(RuntimeError, match="stale occupancy"):
        diagnostic.predecessor_event_ordinals(
            SlotOccupancy(
                physical_slot=0,
                occupancy_generation=99,
                logical_block=1,
                bound_generation=0,
            )
        )
    with pytest.raises(RuntimeError, match="active occupancy"):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=0,
            logical_blocks=(2,),
            physical_slots=(0,),
            bound_generations=(2,),
            current_stream=_DummyStream(3),
        )


def test_enabled_buffer_capacity_overflow_raises():
    diagnostic, _ = _diagnostic(slot_count=1, max_read_rows=1)
    diagnostic.configure("observe")
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.record_read_window(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
        logical_blocks=(1,),
        physical_slots=(0,),
        current_stream=_DummyStream(3),
    )
    with pytest.raises(RuntimeError, match="read row capacity"):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=1,
            logical_blocks=(1,),
            physical_slots=(0,),
            current_stream=_DummyStream(3),
        )


def test_drain_clears_rows_but_preserves_mode():
    diagnostic, events, _, _ = _replace_after_read()
    events.events[2].elapsed[1] = -0.5
    drained = diagnostic.drain(
        synchronize=lambda: None,
        timing_epsilon_ms=0.2,
    )
    assert drained.mode == "observe"
    assert diagnostic.mode == "observe"
    assert diagnostic.retained_event_count == 0
    assert diagnostic.drain(
        synchronize=lambda: None,
        timing_epsilon_ms=0.2,
    ).overwrite_rows == ()
