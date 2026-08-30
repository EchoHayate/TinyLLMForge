from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from types import MethodType

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tinyvllm/engine/phase_stitch_profile.py"
ENGINE_PATH = ROOT / "tinyvllm/engine/llm_engine.py"
CONFIG_PATH = ROOT / "tinyvllm/config.py"
EXACT_BURST_ENGINE_TEST_PATH = (
    ROOT / "tools/test_llm_engine_exact_greedy_decode_burst.py"
)


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


def _load_config_class():
    module_name = "phase_stitch_config_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:

        @staticmethod
        def from_pretrained(model):
            del model
            return types.SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        module = types.ModuleType(module_name)
        module.__file__ = os.fspath(CONFIG_PATH)
        sys.modules[module_name] = module
        source = CONFIG_PATH.read_text(encoding="utf-8")
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                os.fspath(CONFIG_PATH),
                "exec",
            ),
            module.__dict__,
        )
        return module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def _load_exact_burst_engine_test_support():
    name = "phase_stitch_exact_burst_engine_test_support"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        name,
        EXACT_BURST_ENGINE_TEST_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_engine_method(name):
    import ast

    tree = ast.parse(
        ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(ENGINE_PATH),
    )
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method = next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    namespace = {
        "PHASE_STITCH_EVENTS": PHASE_STITCH_EVENTS,
        "PhaseStitchProfileRecorder": PhaseStitchProfileRecorder,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _make_profiled_fake_engine(*, profile_enabled=True, with_k8=True):
    support = _load_exact_burst_engine_test_support()
    lease = support._lease(width=8) if with_k8 else None
    outcome = (
        support._result(
            lease,
            tokens=(102, 103, 104, 105, 106, 107, 108, 109),
        )
        if lease is not None
        else None
    )
    engine, sequence, scheduler, _model_runner, step = support._engine(
        outcome,
        lease=None,
        enabled=True,
        is_prefill=True,
        do_sample=True,
        configured_width=8,
        no_lease_reason="waiting_present",
    )
    step.__globals__["PHASE_STITCH_EVENTS"] = PHASE_STITCH_EVENTS
    sequence.prefill_chunk_start = 0
    sequence.prefill_chunk_end = sequence.num_prompt_tokens
    sequence.prefill_chunk_final = True
    sequence.max_tokens = 9
    engine.phase_stitch_profile = PhaseStitchProfileRecorder(
        enabled=profile_enabled,
        clock_ns=iter(range(100, 1000, 100)).__next__,
    )
    engine.configure_phase_stitch_profile = MethodType(
        _load_engine_method("configure_phase_stitch_profile"),
        engine,
    )
    engine.phase_stitch_profile_snapshot = MethodType(
        _load_engine_method("phase_stitch_profile_snapshot"),
        engine,
    )
    step(engine, completion_only=True)
    scheduler.is_prefill = False
    sequence.step_is_decode = True
    scheduler.lease = lease
    scheduler.pending_leases = int(lease is not None)
    return engine, sequence, scheduler, step


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


def test_phase_stitch_profile_config_defaults_off_and_requires_bool():
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        assert Config(model=model).phase_stitch_profile is False
        assert (
            Config(model=model, phase_stitch_profile=True)
            .phase_stitch_profile
            is True
        )
        with pytest.raises(ValueError, match="phase_stitch_profile"):
            Config(model=model, phase_stitch_profile=1)


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


def test_profile_can_finish_ineligible_without_fake_gap():
    recorder = PhaseStitchProfileRecorder(
        enabled=True,
        clock_ns=iter((100, 120, 150, 190, 230)).__next__,
    )
    recorder.begin_request(sequence_id=7, prompt_tokens=256)
    for event in PHASE_STITCH_EVENTS[:5]:
        recorder.mark(7, event)

    row = recorder.finish_ineligible(
        7,
        reason="exact_k8_lease_unavailable",
        output_token_ids=(11, 12),
    )

    assert row["status"] == "ineligible"
    assert row["ineligible_reason"] == "exact_k8_lease_unavailable"
    assert row["event_coverage_complete"] is False
    assert row["removable_host_gap_ns"] is None
    assert row["k8_lease_prepare_finished_ns"] is None
    assert row["first_k8_dispatch_started_ns"] is None


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


def test_engine_marks_final_prefill_and_first_k8_boundaries():
    engine, sequence, _scheduler, step = _make_profiled_fake_engine()

    step(engine, completion_only=True)

    snapshot = engine.phase_stitch_profile_snapshot()
    assert snapshot["active"] == []
    assert len(snapshot["rows"]) == 1
    row = snapshot["rows"][0]
    assert row["sequence_id"] == sequence.seq_id
    assert row["events"] == list(PHASE_STITCH_EVENTS)
    assert row["status"] == "complete"
    assert row["removable_host_gap_ns"] == 500


def test_engine_does_not_touch_profile_clock_when_disabled():
    engine, _sequence, _scheduler, step = _make_profiled_fake_engine(
        profile_enabled=False,
    )
    engine.phase_stitch_profile._clock_ns = lambda: (
        (_ for _ in ()).throw(
            AssertionError("disabled profile read the clock")
        )
    )

    step(engine, completion_only=True)

    assert engine.phase_stitch_profile_snapshot()["rows"] == []


def test_non_k8_followup_finishes_as_ineligible_without_fake_gap():
    engine, _sequence, _scheduler, step = _make_profiled_fake_engine(
        with_k8=False,
    )

    step(engine, completion_only=True)

    row = engine.phase_stitch_profile_snapshot()["rows"][0]
    assert row["status"] == "ineligible"
    assert row["ineligible_reason"] == "exact_k8_lease_unavailable"
    assert row["removable_host_gap_ns"] is None


def test_engine_profile_configuration_replaces_state():
    engine, _sequence, _scheduler, _step = _make_profiled_fake_engine(
        profile_enabled=False,
    )

    receipt = engine.configure_phase_stitch_profile(True)

    assert receipt == {"enabled": True, "schema_version": 1}
    assert engine.phase_stitch_profile_snapshot()["enabled"] is True
    with pytest.raises(ValueError, match="enabled"):
        engine.configure_phase_stitch_profile(1)
