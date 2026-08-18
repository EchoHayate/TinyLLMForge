from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tinyvllm/engine/decode_internal_profiler.py"
)
COMMAND_TIMELINE_MODULE_PATH = (
    ROOT / "tinyvllm/engine/model_runner_command_timeline.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "decode_internal_profiler",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profiler_module = _load()
DecodeInternalProfiler = profiler_module.DecodeInternalProfiler
profile_collective = profiler_module.profile_collective
run_profiled_step = profiler_module.run_profiled_step


def _load_command_timeline():
    spec = importlib.util.spec_from_file_location(
        "decode_internal_profiler_command_timeline_test_module",
        COMMAND_TIMELINE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClock:

    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 100
        return self.value


class FakeEvent:

    def __init__(self, elapsed_ms):
        self.elapsed_ms = elapsed_ms
        self.record_count = 0

    def record(self):
        self.record_count += 1

    def elapsed_time(self, other):
        return other.elapsed_ms - self.elapsed_ms


class FakeEvents:

    def __init__(self, elapsed_values):
        self.values = iter(elapsed_values)
        self.created = []

    def __call__(self):
        event = FakeEvent(next(self.values))
        self.created.append(event)
        return event


class FakeTensor:
    shape = (4, 2048)
    dtype = "torch.bfloat16"


class FakeNvtxRange:

    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __enter__(self):
        self.events.append(("enter", self.label))

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append(("exit", self.label))


class FakeNvtxRanges:

    def __init__(self):
        self.events = []

    def __call__(self, label):
        return FakeNvtxRange(label, self.events)


def _profiler(
    elapsed_values=None,
    *,
    nvtx_ranges=None,
    active_command_trace=None,
):
    synchronizations = []
    profiler_kwargs = {}
    if active_command_trace is not None:
        profiler_kwargs["active_command_trace"] = (
            active_command_trace
        )
    return (
        DecodeInternalProfiler(
            rank=2,
            clock_ns=FakeClock(),
            event_factory=FakeEvents(
                elapsed_values
                or [0.0, 0.25, 0.75, 1.0]
            ),
            synchronize=lambda: synchronizations.append(True),
            nvtx_range_factory=nvtx_ranges,
            profile_label="policy=exact_restore/case=test-case",
            **profiler_kwargs,
        ),
        synchronizations,
    )


def _command_identity(module):
    return module.CommandTraceIdentity(
        command_id=9,
        method_name="run",
        requires_ack=False,
        engine_step_id=4,
        repeat_index=2,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
        dispatch_started_monotonic_ns=100,
        dispatch_published_monotonic_ns=110,
    )


def test_nvtx_ranges_distinguish_prefill_first_steady_and_collective():
    nvtx_ranges = FakeNvtxRanges()
    profiler, _ = _profiler(
        [float(value) for value in range(8)],
        nvtx_ranges=nvtx_ranges,
    )

    profiler.begin_step(
        batch_kind="prefill",
        is_decode=False,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    profiler.end_step()
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    with profiler.collective("row_parallel_all_reduce", FakeTensor()):
        pass
    profiler.end_step()
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    profiler.end_step()

    assert nvtx_ranges.events == [
        (
            "enter",
            "decode_internal/policy=exact_restore/case=test-case/prefill",
        ),
        (
            "exit",
            "decode_internal/policy=exact_restore/case=test-case/prefill",
        ),
        (
            "enter",
            "decode_internal/policy=exact_restore/case=test-case/"
            "decode_first",
        ),
        (
            "enter",
            "decode_internal/policy=exact_restore/case=test-case/"
            "collective/row_parallel_all_reduce",
        ),
        (
            "exit",
            "decode_internal/policy=exact_restore/case=test-case/"
            "collective/row_parallel_all_reduce",
        ),
        (
            "exit",
            "decode_internal/policy=exact_restore/case=test-case/"
            "decode_first",
        ),
        (
            "enter",
            "decode_internal/policy=exact_restore/case=test-case/"
            "decode_steady",
        ),
        (
            "exit",
            "decode_internal/policy=exact_restore/case=test-case/"
            "decode_steady",
        ),
    ]


def test_records_step_and_collective_and_synchronizes_once():
    profiler, synchronizations = _profiler()

    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=4,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    with profiler.collective(
        "row_parallel_all_reduce",
        FakeTensor(),
    ):
        pass
    profiler.end_step()
    snapshot = profiler.finalize()

    assert synchronizations == [True]
    assert snapshot["rank"] == 2
    assert snapshot["finalization_status"] == "complete"
    assert snapshot["steps"] == [{
        "rank": 2,
        "step_index": 0,
        "batch_kind": "decode",
        "is_decode": True,
        "decode_ordinal": 0,
        "active_sequence_count": 4,
        "request_set_sha256": "a" * 64,
        "command_id": None,
        "engine_step_id": None,
        "repeat_index": None,
        "wall_ns": 300,
        "cuda_ns": 1_000_000,
        "non_cuda_upper_bound_ns": 0,
        "dispatch": "eager",
    }]
    assert snapshot["collectives"] == [{
        "rank": 2,
        "step_index": 0,
        "decode_ordinal": 0,
        "command_id": None,
        "engine_step_id": None,
        "repeat_index": None,
        "operation": "row_parallel_all_reduce",
        "tensor_shape": [4, 2048],
        "tensor_dtype": "torch.bfloat16",
        "wall_ns": 100,
        "cuda_ns": 500_000,
    }]


def test_finalize_already_synchronized_reuses_existing_fence():
    profiler, synchronizations = _profiler([0.0, 1.0])
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=4,
        request_set_sha256="a" * 64,
        dispatch="graph",
    )
    profiler.end_step()

    snapshot = profiler.finalize(already_synchronized=True)

    assert synchronizations == []
    assert snapshot["steps"][0]["cuda_ns"] == 1_000_000


@pytest.mark.parametrize("value", (None, 0, 1, "yes"))
def test_finalize_rejects_non_boolean_already_synchronized(value):
    profiler, _ = _profiler([0.0, 1.0])
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=4,
        request_set_sha256="a" * 64,
        dispatch="graph",
    )
    profiler.end_step()

    with pytest.raises(
        ValueError,
        match="already_synchronized must be a bool",
    ):
        profiler.finalize(already_synchronized=value)


def test_profile_rows_bind_only_the_active_command_identity():
    command_module = _load_command_timeline()
    profiler, _ = _profiler(
        [0.0, 0.25, 0.75, 1.0, 1.0, 2.0],
        active_command_trace=(
            command_module.active_model_runner_command_trace
        ),
    )
    identity = _command_identity(command_module)

    with command_module.command_trace_scope(identity):
        profiler.begin_step(
            batch_kind="decode",
            is_decode=True,
            active_sequence_count=4,
            request_set_sha256="a" * 64,
            dispatch="graph",
        )
        with profiler.collective(
            "row_parallel_all_reduce",
            FakeTensor(),
        ):
            pass
        profiler.end_step()

    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=4,
        request_set_sha256="a" * 64,
        dispatch="graph",
    )
    profiler.end_step()

    snapshot = profiler.finalize()
    identity_fields = (
        "command_id",
        "engine_step_id",
        "repeat_index",
    )
    assert tuple(
        snapshot["steps"][0][name]
        for name in identity_fields
    ) == (9, 4, 2)
    assert tuple(
        snapshot["collectives"][0][name]
        for name in identity_fields
    ) == (9, 4, 2)
    assert tuple(
        snapshot["steps"][1][name]
        for name in identity_fields
    ) == (None, None, None)


def test_prefill_has_no_decode_ordinal_and_decode_ordinals_increment():
    profiler, _ = _profiler(
        [0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    )
    for is_decode in (False, True, True):
        profiler.begin_step(
            batch_kind="decode" if is_decode else "prefill",
            is_decode=is_decode,
            active_sequence_count=4,
            request_set_sha256="a" * 64,
            dispatch="eager",
        )
        profiler.end_step()

    snapshot = profiler.finalize()

    assert [
        row["decode_ordinal"] for row in snapshot["steps"]
    ] == [None, 0, 1]


def test_prefill_collective_is_not_recorded():
    profiler, _ = _profiler([0.0, 0.25, 0.75, 1.0])
    profiler.begin_step(
        batch_kind="prefill",
        is_decode=False,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    with profiler.collective(
        "row_parallel_all_reduce",
        FakeTensor(),
    ):
        pass
    profiler.end_step()

    snapshot = profiler.finalize()

    assert len(snapshot["steps"]) == 1
    assert snapshot["collectives"] == []


def test_disabled_profiler_is_noop():
    profiler = DecodeInternalProfiler.disabled(rank=1)

    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="graph",
    )
    with profiler.collective("all_reduce", FakeTensor()):
        pass
    profiler.end_step()

    assert profiler.finalize() == {
        "rank": 1,
        "enabled": False,
        "finalization_status": "complete",
        "steps": [],
        "collectives": [],
    }


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (
            lambda profiler: (
                profiler.begin_step(
                    batch_kind="decode",
                    is_decode=True,
                    active_sequence_count=1,
                    request_set_sha256="a" * 64,
                    dispatch="eager",
                ),
                profiler.begin_step(
                    batch_kind="decode",
                    is_decode=True,
                    active_sequence_count=1,
                    request_set_sha256="a" * 64,
                    dispatch="eager",
                ),
            ),
            "active",
        ),
        (
            lambda profiler: profiler.end_step(),
            "active",
        ),
        (
            lambda profiler: (
                profiler.begin_step(
                    batch_kind="decode",
                    is_decode=True,
                    active_sequence_count=1,
                    request_set_sha256="a" * 64,
                    dispatch="eager",
                ),
                profiler.finalize(),
            ),
            "active",
        ),
    ],
)
def test_lifecycle_rejects_invalid_transitions(action, message):
    profiler, _ = _profiler()

    with pytest.raises(RuntimeError, match=message):
        action(profiler)


def test_finalize_is_idempotent_and_blocks_new_events():
    profiler, synchronizations = _profiler([0.0, 1.0])
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    profiler.end_step()

    first = profiler.finalize()
    second = profiler.finalize()

    assert first == second
    assert synchronizations == [True]
    with pytest.raises(RuntimeError, match="finalized"):
        profiler.begin_step(
            batch_kind="decode",
            is_decode=True,
            active_sequence_count=1,
            request_set_sha256="a" * 64,
            dispatch="eager",
        )


def test_collective_exception_still_records_event():
    profiler, _ = _profiler()
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )
    with pytest.raises(ValueError, match="boom"):
        with profiler.collective("all_reduce", FakeTensor()):
            raise ValueError("boom")
    profiler.end_step()

    snapshot = profiler.finalize()

    assert len(snapshot["collectives"]) == 1


def test_profile_collective_uses_active_process_local_profiler():
    profiler, _ = _profiler()
    tensor = FakeTensor()
    calls = []
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=1,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )

    result = profile_collective(
        "row_parallel_all_reduce",
        tensor,
        lambda value: calls.append(value) or "done",
    )
    profiler.end_step()
    snapshot = profiler.finalize()

    assert result == "done"
    assert calls == [tensor]
    assert len(snapshot["collectives"]) == 1


def test_profile_collective_without_active_profiler_calls_operation_once():
    tensor = FakeTensor()
    calls = []

    result = profile_collective(
        "row_parallel_all_reduce",
        tensor,
        lambda value: calls.append(value) or value,
    )

    assert result is tensor
    assert calls == [tensor]


def test_run_profiled_step_closes_step_on_success_and_failure():
    successful, _ = _profiler([0.0, 1.0])
    result = run_profiled_step(
        successful,
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=2,
        request_set_sha256="a" * 64,
        dispatch="eager",
        call=lambda: "ok",
    )

    assert result == "ok"
    assert len(successful.finalize()["steps"]) == 1

    failing, _ = _profiler([0.0, 1.0])
    with pytest.raises(ValueError, match="boom"):
        run_profiled_step(
            failing,
            batch_kind="decode",
            is_decode=True,
            active_sequence_count=2,
            request_set_sha256="a" * 64,
            dispatch="eager",
            call=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )
    assert len(failing.finalize()["steps"]) == 1
