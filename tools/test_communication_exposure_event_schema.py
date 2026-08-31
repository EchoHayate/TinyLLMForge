from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tinyvllm/engine/decode_internal_profiler.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "communication_exposure_profiler_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profiler_module = _load()
DecodeInternalProfiler = profiler_module.DecodeInternalProfiler
profile_collective = profiler_module.profile_collective
profile_operation = profiler_module.profile_operation


def test_generic_layer_roles_and_operation_classes_are_frozen():
    assert profiler_module.LAYER_ROLES == frozenset({
        "linear_attention",
        "full_attention",
        "mlp",
        "normalization",
        "residual",
        "embedding",
        "output_head",
    })
    assert profiler_module.OPERATION_CLASSES == frozenset({
        "gemm",
        "attention",
        "recurrent",
        "collective",
        "memory",
        "other_compute",
    })


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

    def __init__(self):
        self.next_value = 0.0
        self.created = []

    def __call__(self):
        event = FakeEvent(self.next_value)
        self.next_value += 0.25
        self.created.append(event)
        return event


class FakeTensor:
    shape = (4, 5120)
    dtype = "torch.bfloat16"


def _profiler():
    synchronizations = []
    events = FakeEvents()
    profiler = DecodeInternalProfiler(
        rank=2,
        clock_ns=FakeClock(),
        event_factory=events,
        synchronize=lambda: synchronizations.append(True),
        stream_resolver=lambda: "cuda:2:stream:7",
        profile_label="attempt=a/workload=Q1/repetition=3",
    )
    return profiler, events, synchronizations


def _begin_decode(profiler):
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=4,
        request_set_sha256="a" * 64,
        dispatch="eager",
    )


def test_layer_and_operation_scopes_emit_generic_identity_and_events():
    profiler, events, synchronizations = _profiler()
    _begin_decode(profiler)

    with profiler.layer(7, "full_attention"):
        with profiler.operation(
            "gemm",
            "qkv_projection",
            tensor=FakeTensor(),
        ):
            pass
    profiler.end_step()

    assert synchronizations == []
    assert len(events.created) == 6
    assert all(event.record_count == 1 for event in events.created)

    snapshot = profiler.finalize()

    assert synchronizations == [True]
    assert snapshot["layers"] == [{
        "rank": 2,
        "step_index": 0,
        "decode_ordinal": 0,
        "command_id": None,
        "engine_step_id": None,
        "repeat_index": None,
        "request_set_sha256": "a" * 64,
        "speculative_selected_sequence_ids_sha256": None,
        "attempt": "a",
        "workload": "Q1",
        "repetition": 3,
        "layer_index": 7,
        "layer_role": "full_attention",
        "wall_start_ns": 200,
        "wall_end_ns": 500,
        "wall_ns": 300,
        "cuda_ns": 750_000,
    }]
    assert snapshot["operations"] == [{
        "rank": 2,
        "step_index": 0,
        "decode_ordinal": 0,
        "command_id": None,
        "engine_step_id": None,
        "repeat_index": None,
        "request_set_sha256": "a" * 64,
        "speculative_selected_sequence_ids_sha256": None,
        "attempt": "a",
        "workload": "Q1",
        "repetition": 3,
        "layer_index": 7,
        "layer_role": "full_attention",
        "operation_ordinal": 0,
        "operation_class": "gemm",
        "operation_name": "qkv_projection",
        "tensor_shape": [4, 5120],
        "tensor_dtype": "torch.bfloat16",
        "source_stream": "cuda:2:stream:7",
        "completion_stream": "cuda:2:stream:7",
        "wall_start_ns": 300,
        "wall_end_ns": 400,
        "wall_ns": 100,
        "cuda_ns": 250_000,
    }]


def test_cpu_enqueue_bounds_enclose_cuda_event_records():
    ordering = []

    class OrderedEvent:

        def __init__(self, event_index):
            self.event_index = event_index

        def record(self):
            ordering.append(f"event:{self.event_index}")

        def elapsed_time(self, other):
            return float(other.event_index - self.event_index)

    next_event = 0

    def event_factory():
        nonlocal next_event
        event = OrderedEvent(next_event)
        next_event += 1
        return event

    def clock_ns():
        ordering.append("clock")
        return len(ordering)

    profiler = DecodeInternalProfiler(
        rank=0,
        clock_ns=clock_ns,
        event_factory=event_factory,
        synchronize=lambda: None,
        stream_resolver=lambda: "cuda:0:stream:0",
    )
    _begin_decode(profiler)
    ordering.clear()

    with profiler.layer(0, "full_attention"):
        with profiler.operation("gemm", "projection"):
            ordering.append("body")

    assert ordering == [
        "clock",
        "event:1",
        "clock",
        "event:2",
        "body",
        "event:3",
        "clock",
        "event:4",
        "clock",
    ]


def test_operation_ordinals_are_monotonic_within_a_step():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)
    with profiler.layer(3, "linear_attention"):
        with profiler.operation("recurrent", "delta_rule"):
            pass
        with profiler.operation("gemm", "output_projection"):
            pass
    profiler.end_step()

    snapshot = profiler.finalize()

    assert [
        row["operation_ordinal"] for row in snapshot["operations"]
    ] == [0, 1]
    assert [
        row["operation_name"] for row in snapshot["operations"]
    ] == ["delta_rule", "output_projection"]


def test_operation_ordinals_reset_for_each_step():
    profiler, _, _ = _profiler()
    for layer_index in (3, 4):
        _begin_decode(profiler)
        with profiler.layer(layer_index, "linear_attention"):
            with profiler.operation("gemm", "projection"):
                pass
        profiler.end_step()

    snapshot = profiler.finalize()

    assert [
        row["operation_ordinal"] for row in snapshot["operations"]
    ] == [0, 0]


def test_collective_records_explicit_synchronous_stream_metadata():
    profiler, _, _ = _profiler()
    tensor = FakeTensor()
    _begin_decode(profiler)
    with profiler.layer(9, "mlp"):
        result = profile_collective(
            "row_parallel_all_reduce",
            tensor,
            lambda value: value,
            collective_kind="all_reduce",
            process_group="tensor_parallel",
            async_mode=False,
            source_stream="cuda:2:stream:7",
            completion_stream="cuda:2:stream:7",
        )
    profiler.end_step()

    snapshot = profiler.finalize()

    assert result is tensor
    assert snapshot["collectives"][0] | {
        "wall_ns": 0,
        "cuda_ns": 0,
    } == {
        "rank": 2,
        "step_index": 0,
        "decode_ordinal": 0,
        "command_id": None,
        "engine_step_id": None,
        "repeat_index": None,
        "request_set_sha256": "a" * 64,
        "speculative_selected_sequence_ids_sha256": None,
        "attempt": "a",
        "workload": "Q1",
        "repetition": 3,
        "layer_index": 9,
        "layer_role": "mlp",
        "operation_ordinal": 0,
        "operation": "row_parallel_all_reduce",
        "operation_class": "collective",
        "collective_kind": "all_reduce",
        "process_group": "tensor_parallel",
        "async_mode": False,
        "source_stream": "cuda:2:stream:7",
        "completion_stream": "cuda:2:stream:7",
        "tensor_shape": [4, 5120],
        "tensor_dtype": "torch.bfloat16",
        "wall_start_ns": 300,
        "wall_end_ns": 400,
        "wall_ns": 0,
        "cuda_ns": 0,
    }


@pytest.mark.parametrize(
    "action,match",
    (
        (
            lambda profiler: profiler.layer(0, "full_attention").__enter__(),
            "active step",
        ),
        (
            lambda profiler: (
                _begin_decode(profiler),
                profiler.layer(0, "unsupported").__enter__(),
            ),
            "layer role",
        ),
        (
            lambda profiler: (
                _begin_decode(profiler),
                profiler.layer(0, "full_attention").__enter__(),
                profiler.operation("unsupported", "x").__enter__(),
            ),
            "operation class",
        ),
    ),
)
def test_invalid_scope_lifecycle_or_enum_fails_closed(action, match):
    profiler, _, _ = _profiler()

    with pytest.raises((ValueError, RuntimeError), match=match):
        action(profiler)


def test_finalize_rejects_open_layer_or_operation_scope():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)
    layer = profiler.layer(1, "full_attention")
    layer.__enter__()

    with pytest.raises(RuntimeError, match="open profiler scope"):
        profiler.finalize()


def test_operation_requires_active_layer_and_scopes_cannot_nest():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)

    with pytest.raises(RuntimeError, match="active layer"):
        profiler.operation("gemm", "orphan").__enter__()

    layer = profiler.layer(1, "full_attention")
    layer.__enter__()
    with pytest.raises(RuntimeError, match="layer scope is active"):
        profiler.layer(2, "mlp").__enter__()
    operation = profiler.operation("gemm", "qkv_projection")
    operation.__enter__()
    with pytest.raises(RuntimeError, match="operation scope is active"):
        profiler.operation("attention", "flash_attention").__enter__()
    operation.__exit__(None, None, None)
    layer.__exit__(None, None, None)
    profiler.end_step()


def test_step_end_rejects_open_scope():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)
    layer = profiler.layer(1, "full_attention")
    layer.__enter__()

    with pytest.raises(RuntimeError, match="open profiler scope"):
        profiler.end_step()


def test_duplicate_layer_identity_in_one_step_fails_closed():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)
    with profiler.layer(1, "full_attention"):
        pass

    with pytest.raises(RuntimeError, match="duplicate layer identity"):
        profiler.layer(1, "full_attention").__enter__()


def test_suspension_excludes_internal_capture_and_restores_outer_step():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)
    with profiler.layer(1, "full_attention"):
        pass

    suspend = getattr(
        profiler_module,
        "suspend_decode_internal_profiler",
    )
    with suspend():
        with profiler_module.profile_layer(1, "full_attention"):
            pass
        with profiler_module.profile_layer(1, "full_attention"):
            pass

    with profiler.layer(2, "mlp"):
        pass
    profiler.end_step()

    rows = profiler.finalize()["layers"]
    assert [
        (row["layer_index"], row["layer_role"])
        for row in rows
    ] == [
        (1, "full_attention"),
        (2, "mlp"),
    ]


def test_collective_rejects_async_execution_in_observation_only_phase():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)

    with profiler.layer(1, "full_attention"):
        with pytest.raises(ValueError, match="async_mode must be False"):
            profile_collective(
                "row_parallel_all_reduce",
                FakeTensor(),
                lambda tensor: tensor,
                collective_kind="all_reduce",
                process_group="tensor_parallel",
                async_mode=True,
            )
    profiler.end_step()


def test_collective_rejects_async_execution_without_active_profiler():
    calls = []

    with pytest.raises(ValueError, match="async_mode must be False"):
        profile_collective(
            "row_parallel_all_reduce",
            FakeTensor(),
            lambda tensor: calls.append(tensor),
            collective_kind="all_reduce",
            process_group="tensor_parallel",
            async_mode=True,
        )

    assert calls == []


def test_generic_operation_helper_is_noop_outside_profiled_layer():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)

    with profile_operation("gemm", "outside_layer"):
        pass
    profiler.end_step()

    assert profiler.finalize()["operations"] == []


def test_finalized_profiler_rejects_new_scopes():
    profiler, _, _ = _profiler()
    _begin_decode(profiler)
    profiler.end_step()
    profiler.finalize()

    with pytest.raises(RuntimeError, match="finalized"):
        with profiler.layer(1, "full_attention"):
            pass
