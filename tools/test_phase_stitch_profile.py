from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tinyvllm/engine/phase_stitch_profile.py"


def _load():
    name = "tinyvllm.engine.phase_stitch_profile"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile_module = _load()
PHASE_STITCH_EVENTS = profile_module.PHASE_STITCH_EVENTS
PhaseStitchProfileRecorder = profile_module.PhaseStitchProfileRecorder


def test_profile_reconstructs_one_prefill_to_k8_handoff():
    timestamps = (100, 120, 150, 190, 230, 260, 300)
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter(timestamps).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    for event in PHASE_STITCH_EVENTS:
        recorder.mark(7, event)

    row = recorder.finish_request(7, (11,) * 128)

    assert row["sequence_id"] == 7
    assert row["prompt_tokens"] == 256
    assert row["status"] == "complete"
    assert row["events"] == list(PHASE_STITCH_EVENTS)
    for event, timestamp in zip(PHASE_STITCH_EVENTS, timestamps):
        assert row[f"{event}_ns"] == timestamp
    assert row["adjacent_intervals_ns"] == {
        "prefill_dispatch_finished_to_first_token_host_available": 20,
        "first_token_host_available_to_prefill_scheduler_commit_finished": 30,
        "prefill_scheduler_commit_finished_to_next_schedule_started": 40,
        "next_schedule_started_to_next_schedule_finished": 40,
        "next_schedule_finished_to_k8_lease_prepare_finished": 30,
        "k8_lease_prepare_finished_to_first_k8_dispatch_started": 40,
    }
    assert row["removable_host_gap_ns"] == 180
    assert len(row["output_token_ids_sha256"]) == 64
    assert row["event_coverage_complete"] is True
    assert recorder.snapshot()["rows"] == [row]


def test_profile_rejects_duplicate_or_out_of_order_events():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter((100, 120, 130)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    recorder.mark(7, "prefill_dispatch_finished")

    with pytest.raises(ValueError, match="event order"):
        recorder.mark(7, "prefill_dispatch_finished")

    with pytest.raises(ValueError, match="event order"):
        recorder.mark(7, "next_schedule_started")


def test_profile_rejects_unknown_sequence_and_missing_events():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter(range(100, 900, 100)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)

    with pytest.raises(ValueError, match="active request"):
        recorder.mark(8, "prefill_dispatch_finished")
    with pytest.raises(ValueError, match="event coverage"):
        recorder.finish_request(7, (11,) * 128)


def test_profile_rejects_non_monotonic_timestamps():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter((100, 99)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    recorder.mark(7, "prefill_dispatch_finished")

    with pytest.raises(ValueError, match="monotonic"):
        recorder.mark(7, "first_token_host_available")


def test_profile_rejects_post_finalization_events():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter(range(100, 800, 100)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    for event in PHASE_STITCH_EVENTS:
        recorder.mark(7, event)
    recorder.finish_request(7, (11,) * 128)

    with pytest.raises(ValueError, match="finalized"):
        recorder.mark(7, "prefill_dispatch_finished")
    with pytest.raises(ValueError, match="finalized"):
        recorder.begin_request(sequence_id=7, prompt_tokens=256)


def test_profile_validates_identity_and_token_inputs():
    recorder = PhaseStitchProfileRecorder(enabled=True, clock_ns=lambda: 1)

    with pytest.raises(ValueError, match="sequence_id"):
        recorder.begin_request(sequence_id=True, prompt_tokens=256)
    with pytest.raises(ValueError, match="prompt_tokens"):
        recorder.begin_request(sequence_id=7, prompt_tokens=0)

    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    for event in PHASE_STITCH_EVENTS:
        recorder.mark(7, event)
    with pytest.raises(ValueError, match="output_token_ids"):
        recorder.finish_request(7, (11, True))


def test_snapshot_is_detached_from_recorder_state():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter(range(100, 800, 100)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    recorder.mark(7, "prefill_dispatch_finished")

    snapshot = recorder.snapshot()
    snapshot["active"][0]["events"].append("mutated")

    assert recorder.snapshot()["active"][0]["events"] == [
        "prefill_dispatch_finished"
    ]


def test_disabled_profile_is_a_noop():
    recorder = PhaseStitchProfileRecorder(
        enabled=False,
        clock_ns=lambda: (_ for _ in ()).throw(
            AssertionError("disabled recorder read the clock")
        ),
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    recorder.mark(7, "prefill_dispatch_finished")
    assert recorder.finish_request(7, (11,) * 128) == {}
    assert recorder.snapshot() == {
        "schema_version": 1,
        "enabled": False,
        "active": [],
        "rows": [],
    }
