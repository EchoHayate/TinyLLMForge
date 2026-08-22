#!/usr/bin/env python3
"""Dependency-light tests for graph-resident greedy-tail contracts."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "graph_resident_greedy_tail.py"
)
SPEC = importlib.util.spec_from_file_location(
    "graph_resident_greedy_tail_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
GraphResidentGreedyTailCaptureReceipt = (
    module.GraphResidentGreedyTailCaptureReceipt
)
GraphResidentGreedyTail = module.GraphResidentGreedyTail
GraphResidentGreedyTailReplay = (
    module.GraphResidentGreedyTailReplay
)
GraphResidentGreedyTailStats = (
    module.GraphResidentGreedyTailStats
)
decide_graph_resident_greedy_tail = (
    module.decide_graph_resident_greedy_tail
)
tensor_identity = module.tensor_identity


def _eligible_kwargs() -> dict:
    return {
        "enabled": True,
        "rank": 0,
        "tensor_parallel_size": 1,
        "is_prefill": False,
        "enforce_eager": False,
        "batch_kind": None,
        "active_batch_size": 1,
        "selected_graph_batch_size": 1,
        "do_sample": True,
        "temperatures": (0.0,),
        "input_embeds_present": False,
        "return_hidden": False,
        "incompatible_modes": (),
        "capture_available": True,
        "quarantined": False,
        "source_matches": True,
    }


def test_exact_ordinary_batch_one_greedy_decode_is_eligible() -> None:
    decision = decide_graph_resident_greedy_tail(
        **_eligible_kwargs()
    )

    assert decision.optimized is True
    assert decision.fallback_reason is None


def test_ineligible_cases_fail_closed_in_stable_order() -> None:
    cases = (
        ("enabled", False, "disabled"),
        ("rank", 1, "non_root_rank"),
        (
            "tensor_parallel_size",
            2,
            "tensor_parallel_unsupported",
        ),
        ("is_prefill", True, "prefill_unsupported"),
        ("enforce_eager", True, "eager_unsupported"),
        ("batch_kind", "mixed", "mixed_batch_unsupported"),
        (
            "active_batch_size",
            2,
            "batch_size_unsupported",
        ),
        (
            "selected_graph_batch_size",
            2,
            "selected_graph_batch_unsupported",
        ),
        ("do_sample", False, "sampling_disabled"),
        ("temperatures", ("0",), "temperature_invalid"),
        ("temperatures", (0.7,), "nonzero_temperature"),
        (
            "input_embeds_present",
            True,
            "input_embeds_unsupported",
        ),
        ("return_hidden", True, "return_hidden_unsupported"),
        (
            "incompatible_modes",
            ("kv_offload",),
            "incompatible_mode:kv_offload",
        ),
        (
            "capture_available",
            False,
            "capture_unavailable",
        ),
        ("quarantined", True, "quarantined"),
        (
            "source_matches",
            False,
            "source_identity_drift",
        ),
    )

    for field, value, expected_reason in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        decision = decide_graph_resident_greedy_tail(**kwargs)
        assert decision.optimized is False
        assert decision.fallback_reason == expected_reason

    kwargs = _eligible_kwargs()
    kwargs.update(
        enabled=False,
        rank=1,
        tensor_parallel_size=2,
        is_prefill=True,
        capture_available=False,
    )
    decision = decide_graph_resident_greedy_tail(**kwargs)
    assert decision.fallback_reason == "disabled"


def test_invalid_control_values_raise_exact_messages() -> None:
    cases = (
        ("enabled", 1, "enabled must be a bool"),
        ("rank", True, "rank must be a non-negative integer"),
        (
            "tensor_parallel_size",
            0,
            "tensor_parallel_size must be a positive integer",
        ),
        ("is_prefill", 0, "is_prefill must be a bool"),
        ("enforce_eager", 0, "enforce_eager must be a bool"),
        (
            "batch_kind",
            1,
            "batch_kind must be a string or None",
        ),
        (
            "active_batch_size",
            True,
            "active_batch_size must be a non-negative integer",
        ),
        (
            "selected_graph_batch_size",
            0,
            "selected_graph_batch_size must be a positive integer",
        ),
        ("do_sample", 1, "do_sample must be a bool"),
        (
            "temperatures",
            [0.0],
            "temperatures must be a tuple",
        ),
        (
            "input_embeds_present",
            0,
            "input_embeds_present must be a bool",
        ),
        (
            "return_hidden",
            0,
            "return_hidden must be a bool",
        ),
        (
            "incompatible_modes",
            ["kv_offload"],
            "incompatible_modes must be a tuple",
        ),
        (
            "capture_available",
            1,
            "capture_available must be a bool",
        ),
        ("quarantined", 0, "quarantined must be a bool"),
        (
            "source_matches",
            1,
            "source_matches must be a bool",
        ),
    )

    for field, value, expected_message in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        try:
            decide_graph_resident_greedy_tail(**kwargs)
        except ValueError as error:
            assert str(error) == expected_message
        else:
            raise AssertionError(
                f"{field} validation did not fail"
            )


class _FakeTensor:
    def __init__(
        self,
        *,
        data_ptr: int = 123_456,
        shape=(1, 1024),
        stride=(1024, 1),
        storage_offset: int = 8,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        element_size: int = 2,
        events: list | None = None,
        label: str = "hidden",
    ):
        self._data_ptr = data_ptr
        self.shape = shape
        self._stride = stride
        self._storage_offset = storage_offset
        self.dtype = dtype
        self.device = device
        self._element_size = element_size
        self.events = [] if events is None else events
        self.label = label

    def data_ptr(self) -> int:
        return self._data_ptr

    def stride(self) -> tuple[int, int]:
        return self._stride

    def storage_offset(self) -> int:
        return self._storage_offset

    def numel(self) -> int:
        total = 1
        for value in self.shape:
            total *= value
        return total

    def element_size(self) -> int:
        return self._element_size

    def __getitem__(self, index):
        self.events.append((self.label, "slice", index))
        if index != slice(None, 1, None):
            raise AssertionError(f"unexpected slice: {index!r}")
        return _FakeTensor(
            data_ptr=self._data_ptr,
            shape=(1, self.shape[-1]),
            stride=self._stride,
            storage_offset=self._storage_offset,
            dtype=self.dtype,
            device=self.device,
            element_size=self._element_size,
            events=self.events,
            label=self.label,
        )


class _FakeValue(_FakeTensor):
    def to(self, dtype):
        self.events.append((self.label, "to", dtype))
        return _FakeValue(
            data_ptr=self._data_ptr + 1_000,
            shape=self.shape,
            stride=self._stride,
            storage_offset=0,
            dtype=str(dtype),
            device=self.device,
            element_size=4,
            events=self.events,
            label="float_logits",
        )

    def argmax(self, *, dim):
        self.events.append((self.label, "argmax", dim))
        return _FakeTensor(
            data_ptr=self._data_ptr + 2_000,
            shape=(1,),
            stride=(1,),
            storage_offset=0,
            dtype="int64",
            device=self.device,
            element_size=8,
            events=self.events,
            label="token_ids",
        )


class _FakeGraph:
    def __init__(self):
        self.replay_calls = 0
        self.replay_error = None

    def replay(self):
        self.replay_calls += 1
        if self.replay_error is not None:
            raise self.replay_error


def _capture_fixture(*, fail_during_capture=False):
    events = []
    hidden = _FakeTensor(events=events)
    graphs = []
    stats = GraphResidentGreedyTailStats()
    memory_samples = iter(((1_000, 2_000), (1_256, 2_512)))
    clock_samples = iter((10_000, 11_000))

    def compute_logits(value):
        events.append(("compute_logits", value.data_ptr()))
        if (
            fail_during_capture
            and events.count(("capture", "enter")) == 1
        ):
            raise RuntimeError("capture failed")
        return _FakeValue(
            data_ptr=200_000,
            shape=(1, 8),
            stride=(8, 1),
            storage_offset=0,
            dtype="bfloat16",
            device="cuda:0",
            element_size=2,
            events=events,
            label="logits",
        )

    def graph_factory():
        graph = _FakeGraph()
        graphs.append(graph)
        events.append(("graph", "created"))
        return graph

    @contextmanager
    def capture_context_factory(graph):
        assert graph is graphs[-1]
        events.append(("capture", "enter"))
        try:
            yield
        finally:
            events.append(("capture", "exit"))

    def synchronize():
        events.append(("cuda", "synchronize"))

    return {
        "events": events,
        "hidden": hidden,
        "graphs": graphs,
        "stats": stats,
        "kwargs": {
            "static_hidden": hidden,
            "compute_logits": compute_logits,
            "float32_dtype": "float32",
            "graph_generation": 7,
            "rank": 0,
            "graph_factory": graph_factory,
            "capture_context_factory": capture_context_factory,
            "synchronize": synchronize,
            "memory_snapshot": lambda: next(memory_samples),
            "clock_ns": lambda: next(clock_samples),
            "stats": stats,
        },
    }


def test_tensor_identity_uses_storage_geometry_not_python_id() -> None:
    assert tensor_identity(_FakeTensor()) == (
        123_456,
        (1, 1024),
        (1024, 1),
        8,
        "bfloat16",
        "cuda:0",
    )


def test_capture_receipt_and_replay_are_immutable_records() -> None:
    receipt = GraphResidentGreedyTailCaptureReceipt(
        source_identity=tensor_identity(_FakeTensor()),
        graph_generation=7,
        rank=0,
        capture_duration_ns=1_000,
        allocated_delta_bytes=256,
        reserved_delta_bytes=512,
        retained_logits_bytes=16,
        retained_float32_bytes=32,
        retained_token_bytes=8,
    )
    logits = object()
    token_ids = object()
    replay = GraphResidentGreedyTailReplay(
        logits=logits,
        token_ids=token_ids,
    )

    assert receipt.graph_generation == 7
    assert replay.logits is logits
    assert replay.token_ids is token_ids
    try:
        receipt.rank = 1
    except Exception:
        pass
    else:
        raise AssertionError("capture receipt is mutable")


def test_stats_account_exact_graph_work_and_cost() -> None:
    stats = GraphResidentGreedyTailStats()
    receipt = GraphResidentGreedyTailCaptureReceipt(
        source_identity=tensor_identity(_FakeTensor()),
        graph_generation=7,
        rank=0,
        capture_duration_ns=1_000,
        allocated_delta_bytes=256,
        reserved_delta_bytes=512,
        retained_logits_bytes=16,
        retained_float32_bytes=32,
        retained_token_bytes=8,
    )
    stats.record_capture(receipt)
    stats.record_fallback("disabled")
    stats.record_fallback("disabled")
    stats.record_replay()
    stats.record_replay()
    stats.record_token_d2h()
    stats.record_token_d2h()

    assert stats.summary() == {
        "eligible_steps": 2,
        "captured_graphs": 1,
        "replayed_steps": 2,
        "final_token_d2h_calls": 2,
        "avoided_external_compute_logits_calls": 2,
        "avoided_external_float32_conversions": 2,
        "avoided_external_argmax_calls": 2,
        "fallback_counts": {"disabled": 2},
        "quarantine_reason": None,
        "capture_receipt": {
            "source_identity": {
                "data_ptr": 123_456,
                "shape": [1, 1024],
                "stride": [1024, 1],
                "storage_offset": 8,
                "dtype": "bfloat16",
                "device": "cuda:0",
            },
            "graph_generation": 7,
            "rank": 0,
            "capture_duration_ns": 1_000,
            "allocated_delta_bytes": 256,
            "reserved_delta_bytes": 512,
            "retained_logits_bytes": 16,
            "retained_float32_bytes": 32,
            "retained_token_bytes": 8,
            "retained_static_bytes": 56,
        },
    }


def test_stats_reject_invalid_updates_and_keep_first_quarantine() -> None:
    stats = GraphResidentGreedyTailStats()
    for value in ("", 1, None):
        try:
            stats.record_fallback(value)
        except ValueError as error:
            assert str(error) == (
                "fallback reason must be a non-empty string"
            )
        else:
            raise AssertionError("invalid fallback reason was accepted")

    for value in ("", 1, None):
        try:
            stats.quarantine(value)
        except ValueError as error:
            assert str(error) == (
                "quarantine reason must be a non-empty string"
            )
        else:
            raise AssertionError(
                "invalid quarantine reason was accepted"
            )

    stats.quarantine("replay_failure:RuntimeError")
    stats.quarantine("replay_failure:ValueError")
    assert stats.summary()["quarantine_reason"] == (
        "replay_failure:RuntimeError"
    )


def test_capture_runs_exact_warmup_and_graph_body() -> None:
    fixture = _capture_fixture()

    tail = GraphResidentGreedyTail.capture(**fixture["kwargs"])

    events = fixture["events"]
    assert len(fixture["graphs"]) == 1
    assert events.count(("compute_logits", 123_456)) == 2
    assert events.count(("logits", "to", "float32")) == 2
    assert events.count(("float_logits", "argmax", -1)) == 2
    assert events.count(("hidden", "slice", slice(None, 1))) == 2
    assert ("capture", "enter") in events
    assert ("capture", "exit") in events
    assert not any(
        event[1] in {"clone", "copy", "copy_"}
        for event in events
        if len(event) > 1
    )
    summary = tail.summary()
    assert summary["captured_graphs"] == 1
    assert summary["capture_receipt"]["capture_duration_ns"] == 1_000
    assert summary["capture_receipt"]["allocated_delta_bytes"] == 256
    assert summary["capture_receipt"]["reserved_delta_bytes"] == 512
    assert summary["capture_receipt"]["retained_logits_bytes"] == 16
    assert summary["capture_receipt"]["retained_float32_bytes"] == 32
    assert summary["capture_receipt"]["retained_token_bytes"] == 8


def test_replay_accepts_same_storage_view_and_accounts_once() -> None:
    fixture = _capture_fixture()
    tail = GraphResidentGreedyTail.capture(**fixture["kwargs"])
    same_storage_view = _FakeTensor(events=fixture["events"])

    assert tail.matches(
        static_hidden=same_storage_view,
        graph_generation=7,
        rank=0,
    )
    replay = tail.replay(
        static_hidden=same_storage_view,
        graph_generation=7,
        rank=0,
    )
    assert isinstance(replay, GraphResidentGreedyTailReplay)
    assert replay.logits is tail.logits
    assert replay.token_ids is tail.token_ids
    assert fixture["graphs"][0].replay_calls == 1
    tail.mark_token_d2h()
    assert tail.summary()["replayed_steps"] == 1
    assert tail.summary()["final_token_d2h_calls"] == 1
    try:
        tail.mark_token_d2h()
    except RuntimeError as error:
        assert str(error) == "no replay token D2H is pending"
    else:
        raise AssertionError("duplicate token D2H was accepted")


def test_drift_is_rejected_before_graph_replay() -> None:
    fixture = _capture_fixture()
    tail = GraphResidentGreedyTail.capture(**fixture["kwargs"])
    graph = fixture["graphs"][0]
    cases = (
        (
            _FakeTensor(data_ptr=654_321),
            7,
            0,
            "source identity drift",
        ),
        (
            _FakeTensor(shape=(1, 2048), stride=(2048, 1)),
            7,
            0,
            "source identity drift",
        ),
        (
            _FakeTensor(dtype="float16"),
            7,
            0,
            "source identity drift",
        ),
        (
            _FakeTensor(device="cuda:1"),
            7,
            0,
            "source identity drift",
        ),
        (_FakeTensor(), 8, 0, "graph generation drift"),
        (_FakeTensor(), 7, 1, "rank drift"),
    )
    for hidden, generation, rank, expected in cases:
        try:
            tail.replay(
                static_hidden=hidden,
                graph_generation=generation,
                rank=rank,
            )
        except RuntimeError as error:
            assert str(error) == expected
        else:
            raise AssertionError("drifted replay was accepted")
    assert graph.replay_calls == 0


def test_replay_failure_quarantines_without_retry() -> None:
    fixture = _capture_fixture()
    tail = GraphResidentGreedyTail.capture(**fixture["kwargs"])
    graph = fixture["graphs"][0]
    graph.replay_error = RuntimeError("boom")

    try:
        tail.replay(
            static_hidden=_FakeTensor(),
            graph_generation=7,
            rank=0,
        )
    except RuntimeError as error:
        assert str(error) == "boom"
    else:
        raise AssertionError("replay failure did not propagate")
    assert graph.replay_calls == 1
    assert tail.summary()["quarantine_reason"] == (
        "replay_failure:RuntimeError"
    )

    graph.replay_error = None
    try:
        tail.replay(
            static_hidden=_FakeTensor(),
            graph_generation=7,
            rank=0,
        )
    except RuntimeError as error:
        assert str(error) == (
            "graph-resident greedy tail is quarantined: "
            "replay_failure:RuntimeError"
        )
    else:
        raise AssertionError("quarantined replay was accepted")
    assert graph.replay_calls == 1


def test_capture_failure_produces_no_receipt() -> None:
    fixture = _capture_fixture(fail_during_capture=True)

    try:
        GraphResidentGreedyTail.capture(**fixture["kwargs"])
    except RuntimeError as error:
        assert str(error) == "capture failed"
    else:
        raise AssertionError("capture failure did not propagate")
    assert fixture["stats"].summary()["captured_graphs"] == 0
    assert fixture["stats"].summary()["capture_receipt"] is None


def main() -> None:
    test_exact_ordinary_batch_one_greedy_decode_is_eligible()
    test_ineligible_cases_fail_closed_in_stable_order()
    test_invalid_control_values_raise_exact_messages()
    test_tensor_identity_uses_storage_geometry_not_python_id()
    test_capture_receipt_and_replay_are_immutable_records()
    test_stats_account_exact_graph_work_and_cost()
    test_stats_reject_invalid_updates_and_keep_first_quarantine()
    test_capture_runs_exact_warmup_and_graph_body()
    test_replay_accepts_same_storage_view_and_accounts_once()
    test_drift_is_rejected_before_graph_replay()
    test_replay_failure_quarantines_without_retry()
    test_capture_failure_produces_no_receipt()
    print("graph-resident greedy tail tests passed")


if __name__ == "__main__":
    main()
