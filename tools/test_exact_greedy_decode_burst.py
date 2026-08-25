#!/usr/bin/env python3
"""Dependency-light tests for exact greedy decode burst contracts."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import pickle
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(REPO_ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


xxhash_module = types.ModuleType("xxhash")
xxhash_module.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_module)
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_decode_burst.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exact_greedy_decode_burst_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

SPLIT_MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_decode_burst_split_phase.py"
)
SPLIT_SPEC = importlib.util.spec_from_file_location(
    "exact_greedy_decode_burst_split_phase_for_burst_test",
    SPLIT_MODULE_PATH,
)
split_module = importlib.util.module_from_spec(SPLIT_SPEC)
sys.modules[SPLIT_SPEC.name] = split_module
SPLIT_SPEC.loader.exec_module(split_module)

ExactGreedyDecodeBurstCaptureReceipt = (
    module.ExactGreedyDecodeBurstCaptureReceipt
)
ExactGreedyDecodeBurstContinuationReceipt = (
    module.ExactGreedyDecodeBurstContinuationReceipt
)
ExactGreedyDecodeBurstFallback = (
    module.ExactGreedyDecodeBurstFallback
)
ExactGreedyDecodeBurstGraph = module.ExactGreedyDecodeBurstGraph
ExactGreedyDecodeBurstLease = module.ExactGreedyDecodeBurstLease
ElasticBurstWidthHealth = getattr(
    module,
    "ElasticBurstWidthHealth",
    None,
)
BlockTableIdentitySeal = module.BlockTableIdentitySeal
ExactGreedyDecodeBurstResult = module.ExactGreedyDecodeBurstResult
ExactGreedyDecodeBurstStats = module.ExactGreedyDecodeBurstStats
build_exact_greedy_decode_burst_decision = (
    module.build_exact_greedy_decode_burst_decision
)
build_exact_greedy_decode_burst_lease = (
    module.build_exact_greedy_decode_burst_lease
)
decide_exact_greedy_decode_burst_continuation = (
    module.decide_exact_greedy_decode_burst_continuation
)
exact_greedy_decode_burst_flash_attn_num_splits = (
    getattr(
        module,
        "exact_greedy_decode_burst_flash_attn_num_splits",
        None,
    )
)
select_exact_greedy_decode_burst_width = (
    module.select_exact_greedy_decode_burst_width
)
select_context_gated_elastic_exact_burst_width = getattr(
    module,
    "select_context_gated_elastic_exact_burst_width",
    None,
)
validate_exact_greedy_decode_burst_result = (
    module.validate_exact_greedy_decode_burst_result
)


def test_medium_split_k_selector_requires_complete_burst_in_range():
    select = exact_greedy_decode_burst_flash_attn_num_splits
    assert callable(select)
    cases = (
        (False, 1537, 8, 0),
        (True, 1536, 1, 0),
        (True, 1537, 1, 12),
        (True, 1537, 8, 12),
        (True, 4090, 8, 12),
        (True, 4091, 8, 0),
        (True, 4097, 1, 12),
        (True, 4098, 1, 0),
    )
    for enabled, initial, count, expected in cases:
        assert select(
            enabled=enabled,
            initial_sequence_length=initial,
            authorized_token_count=count,
        ) == expected


def test_medium_split_k_selector_rejects_invalid_inputs():
    select = exact_greedy_decode_burst_flash_attn_num_splits
    for kwargs, message in (
        (
            {
                "enabled": 1,
                "initial_sequence_length": 1537,
                "authorized_token_count": 8,
            },
            "enabled must be a bool",
        ),
        (
            {
                "enabled": True,
                "initial_sequence_length": 0,
                "authorized_token_count": 8,
            },
            "initial_sequence_length must be a positive integer",
        ),
        (
            {
                "enabled": True,
                "initial_sequence_length": 1537,
                "authorized_token_count": 0,
            },
            "authorized_token_count must be a positive integer",
        ),
    ):
        _assert_raises(
            ValueError,
            message,
            lambda kwargs=kwargs: select(**kwargs),
        )


class _BurstTensor:
    _next_ptr = 10_000

    def __init__(
        self,
        values,
        *,
        label,
        events,
        dtype="int64",
        device="cuda:0",
        element_size=8,
    ):
        self.values = values
        self.label = label
        self.events = events
        self.dtype = dtype
        self.device = device
        self._element_size = element_size
        self._data_ptr = _BurstTensor._next_ptr
        _BurstTensor._next_ptr += 1_000
        self.fail_tolist = False
        self.fail_copy = False
        self.tolist_calls = 0

    @property
    def shape(self):
        if isinstance(self.values, list):
            if self.values and isinstance(self.values[0], list):
                return (len(self.values), len(self.values[0]))
            return (len(self.values),)
        return ()

    def stride(self):
        if len(self.shape) == 2:
            return (self.shape[1], 1)
        if len(self.shape) == 1:
            return (1,)
        return ()

    def storage_offset(self):
        return 0

    def data_ptr(self):
        return self._data_ptr

    def numel(self):
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total

    def element_size(self):
        return self._element_size

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def fill_(self, value):
        self.events.append((self.label, "fill_", value))
        if isinstance(self.values, list):
            if self.values and isinstance(self.values[0], list):
                self.values = [
                    [value for _ in row]
                    for row in self.values
                ]
            else:
                self.values = [value for _ in self.values]
        else:
            self.values = value
        return self

    def zero_(self):
        return self.fill_(0)

    def copy_(self, other):
        if self.fail_copy:
            raise RuntimeError("copy failed")
        value = getattr(other, "values", other)
        if isinstance(value, list):
            value = [
                list(row) if isinstance(row, list) else row
                for row in value
            ]
        self.values = value
        self.events.append((self.label, "copy_", value))
        return self

    def add_(self, value):
        observed = getattr(value, "values", value)
        self.events.append((self.label, "add_", observed))
        if isinstance(self.values, list):
            if self.values and isinstance(self.values[0], list):
                self.values = [
                    [
                        item + increment
                        for item, increment in zip(row, delta)
                    ]
                    for row, delta in zip(self.values, observed)
                ]
            else:
                self.values = [
                    item + value for item in self.values
                ]
        else:
            self.values += value
        return self

    def eq(self, other):
        scalar = (
            other.values[0]
            if isinstance(other.values, list)
            else other.values
        )
        self.events.append((self.label, "eq", scalar))
        return _BurstTensor(
            [int(value == scalar) for value in self.values],
            label="sample_mask",
            events=self.events,
            dtype="bool",
            element_size=1,
        )

    def __mul__(self, other):
        left = self.values
        right = other.values
        if left and isinstance(left[0], list):
            left_rows = left
        else:
            left_rows = [[value] for value in left]
        if right and isinstance(right[0], list):
            right_row = right[0]
        else:
            right_row = right
        values = [
            [row[0] * value for value in right_row]
            for row in left_rows
        ]
        self.events.append((self.label, "mul", other.label))
        return _BurstTensor(
            values,
            label="masked_logits",
            events=self.events,
            dtype=other.dtype,
            element_size=other.element_size(),
        )

    def view(self, *shape):
        self.events.append((self.label, "view", shape))
        return self

    def index_copy_(self, dim, index, source):
        self.events.append(
            (
                self.label,
                "index_copy_",
                dim,
                getattr(index, "values", index),
                getattr(source, "values", source),
            )
        )
        destination = int(
            index.values[0]
            if isinstance(index.values, list)
            else index.values
        )
        source_value = (
            source.values[0]
            if isinstance(source.values, list)
            else source.values
        )
        self.values[destination] = source_value
        return self

    def to(self, dtype):
        self.events.append((self.label, "to", dtype))
        return _BurstTensor(
            (
                [list(row) for row in self.values]
                if self.values
                and isinstance(self.values[0], list)
                else list(self.values)
            ),
            label="float_logits",
            events=self.events,
            dtype=str(dtype),
            element_size=4,
        )

    def argmax(self, *, dim):
        self.events.append((self.label, "argmax", dim))
        rows = self.values
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        values = [
            max(range(len(row)), key=row.__getitem__)
            for row in rows
        ]
        return _BurstTensor(
            values,
            label="next_token",
            events=self.events,
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = _BurstTensor(
                self.values[index],
                label=self.label + "_slice",
                events=self.events,
                dtype=self.dtype,
                device=self.device,
                element_size=self._element_size,
            )
            result.fail_tolist = self.fail_tolist
            result._source = self
            return result
        return self.values[index]

    def tolist(self):
        source = getattr(self, "_source", self)
        source.tolist_calls += 1
        if source.fail_tolist:
            raise RuntimeError("final D2H failed")
        if isinstance(self.values, list):
            return [
                list(row) if isinstance(row, list) else row
                for row in self.values
            ]
        return self.values


class _BurstGraph:
    def __init__(self):
        self.replay_calls = 0
        self.replay_error_at = None
        self.on_replay = None

    def replay(self):
        self.replay_calls += 1
        if self.replay_error_at == self.replay_calls:
            raise RuntimeError(f"replay {self.replay_calls} failed")
        if self.on_replay is not None:
            self.on_replay(self.replay_calls - 1)

    def pool(self):
        return "dedicated-pool"


def _graph_fixture(
    *,
    correctness_trace=False,
    live_kv_changed=False,
    history_capacity=8,
    sampled_logit_ordinals=None,
    flash_attn_num_splits=0,
):
    events = []
    tensors = {
        "input_token": _BurstTensor(
            [0], label="input_token", events=events
        ),
        "position": _BurstTensor(
            [0], label="position", events=events
        ),
        "context_length": _BurstTensor(
            [0], label="context_length", events=events
        ),
        "slot_mapping": _BurstTensor(
            [0],
            label="slot_mapping",
            events=events,
            dtype="int32",
            element_size=4,
        ),
        "block_table": _BurstTensor(
            [[-1, -1]],
            label="block_table",
            events=events,
            dtype="int32",
            element_size=4,
        ),
        "token_history": _BurstTensor(
            [-1] * history_capacity,
            label="token_history",
            events=events,
        ),
        "history_index": _BurstTensor(
            [0], label="history_index", events=events
        ),
    }
    if correctness_trace:
        tensors["sampled_logits"] = _BurstTensor(
            [[0.0] * 5 for _ in range(3)],
            label="sampled_logits",
            events=events,
            dtype="float32",
            element_size=4,
        )
        tensors["sample_ordinals"] = _BurstTensor(
            [0, 2, -1],
            label="sample_ordinals",
            events=events,
        )
    graphs = []
    phase = {"value": "outside"}
    next_tokens = iter((1, 2))

    def model(input_token, position):
        events.append(
            (
                phase["value"],
                "model",
                tuple(input_token.values),
                tuple(position.values),
            )
        )
        return _BurstTensor(
            [[7.0]],
            label="hidden",
            events=events,
            dtype="bfloat16",
            element_size=2,
        )

    def compute_logits(hidden):
        del hidden
        token = next(next_tokens)
        events.append((phase["value"], "compute_logits"))
        row = [0.0] * 5
        row[token] = 9.0
        return _BurstTensor(
            [row],
            label="logits",
            events=events,
            dtype="bfloat16",
            element_size=2,
        )

    def graph_factory():
        graph = _BurstGraph()
        graphs.append(graph)
        return graph

    @contextmanager
    def capture_context_factory(graph, pool):
        assert graph is graphs[-1]
        assert pool == "pool-17"
        phase["value"] = "capture"
        events.append(("capture", "enter"))
        try:
            yield
        finally:
            events.append(("capture", "exit"))
            phase["value"] = "outside"

    context_slots = []

    def set_decode_context(**kwargs):
        context_slots.append(
            tuple(kwargs["slot_mapping"].values)
        )
        events.append(
            (
                phase["value"],
                "set_context",
                kwargs["flash_attn_num_splits"],
            )
        )

    live_snapshots = iter(
        (
            b"live-kv",
            b"mutated-live-kv"
            if live_kv_changed
            else b"live-kv",
        )
    )
    memory_samples = iter(((100, 200), (140, 260)))
    clock_samples = iter((1_000, 1_900))
    stats = ExactGreedyDecodeBurstStats()

    graph = ExactGreedyDecodeBurstGraph.capture(
        tensors=tensors,
        model=model,
        compute_logits=compute_logits,
        float32_dtype="float32",
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
        scratch_block_id=9,
        block_size=256,
        graph_pool="pool-17",
        graph_factory=graph_factory,
        capture_context_factory=capture_context_factory,
        synchronize=lambda: events.append(("cuda", "sync")),
        memory_snapshot=lambda: next(memory_samples),
        clock_ns=lambda: next(clock_samples),
        set_decode_context=set_decode_context,
        reset_context=lambda: events.append(
            (phase["value"], "reset_context")
        ),
        live_kv_snapshot=lambda: next(live_snapshots),
        correctness_trace=correctness_trace,
        sampled_logit_ordinals=(
            (
                (0, 2)
                if sampled_logit_ordinals is None
                else sampled_logit_ordinals
            )
            if correctness_trace
            else ()
        ),
        flash_attn_num_splits=flash_attn_num_splits,
        stats=stats,
    )
    return graph, tensors, graphs[0], events, context_slots


def _assert_raises(error_type, message, callback):
    try:
        callback()
    except error_type as error:
        assert str(error) == message, (str(error), message)
    else:
        raise AssertionError(
            f"expected {error_type.__name__}: {message}"
        )


def _eligible_kwargs() -> dict:
    return {
        "enabled": True,
        "configured_width": 8,
        "remaining_output_tokens": 6,
        "initial_sequence_length": 251,
        "block_size": 256,
        "sequence_count": 1,
        "waiting_count": 0,
        "prefilling_count": 0,
        "is_prefill": False,
        "do_sample": True,
        "batch_kind": None,
        "temperatures": (0.0,),
        "ignore_eos": (True,),
        "completion_only": True,
        "tensor_parallel_size": 1,
        "rank": 0,
        "graph_available": True,
        "incompatible_modes": (),
        "pending_lease": False,
        "quarantined": False,
    }


def test_ragged_width_selector_caps_output_tail_at_k4() -> None:
    selected = {}
    for capacity in range(9):
        selected[capacity] = select_exact_greedy_decode_burst_width(
            configured_width=8,
            remaining_output_tokens=capacity,
            initial_sequence_length=241,
            block_size=256,
            split_phase_enabled=True,
            ragged_coalescing_enabled=True,
        )
    assert selected == {
        0: 8,
        1: 8,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 4,
        7: 4,
        8: 8,
    }


def test_ragged_width_selector_caps_block_edge_at_k4() -> None:
    selected = {}
    for capacity in range(1, 9):
        selected[capacity] = select_exact_greedy_decode_burst_width(
            configured_width=8,
            remaining_output_tokens=8,
            initial_sequence_length=257 - capacity,
            block_size=256,
            split_phase_enabled=True,
            ragged_coalescing_enabled=True,
        )
    assert selected == {
        1: 8,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 4,
        7: 4,
        8: 8,
    }


def test_ragged_width_selector_is_strictly_opt_in() -> None:
    common = {
        "configured_width": 8,
        "remaining_output_tokens": 7,
        "initial_sequence_length": 241,
        "block_size": 256,
    }
    assert select_exact_greedy_decode_burst_width(
        **common,
        split_phase_enabled=True,
        ragged_coalescing_enabled=False,
    ) == 8
    assert select_exact_greedy_decode_burst_width(
        **common,
        split_phase_enabled=False,
        ragged_coalescing_enabled=True,
    ) == 8
    assert select_exact_greedy_decode_burst_width(
        **{
            **common,
            "configured_width": 4,
        },
        split_phase_enabled=True,
        ragged_coalescing_enabled=True,
    ) == 4


def test_context_gated_elastic_width_selector_is_deterministic() -> None:
    select = select_context_gated_elastic_exact_burst_width
    assert callable(select)
    assert module.ElasticExactBurstWidthDecision.__dataclass_params__.frozen
    cases = (
        # enabled, context, remaining, writable, healthy, width, reason
        (False, 256, 128, 128, True, 8, "disabled"),
        (True, 2048, 16, 16, True, 16, None),
        (
            True,
            2049,
            128,
            128,
            True,
            8,
            "context_above_2048",
        ),
        (
            True,
            256,
            15,
            128,
            True,
            8,
            "output_budget_below_16",
        ),
        (
            True,
            256,
            128,
            15,
            True,
            8,
            "write_block_capacity_below_16",
        ),
        (
            True,
            256,
            128,
            128,
            False,
            8,
            "k16_graph_unavailable",
        ),
        (
            True,
            4096,
            128,
            128,
            True,
            8,
            "context_above_2048",
        ),
        (
            True,
            8192,
            128,
            128,
            True,
            8,
            "context_above_2048",
        ),
    )
    for (
        enabled,
        context,
        remaining,
        writable,
        healthy,
        expected_width,
        expected_reason,
    ) in cases:
        decision = select(
            enabled=enabled,
            base_width=8,
            initial_sequence_length=context,
            remaining_output_tokens=remaining,
            write_block_capacity=writable,
            shared_graph_available=healthy,
            k16_health_quarantined=False,
            incompatible_modes=(),
        )
        assert decision.requested_width == (
            16 if enabled else 8
        )
        assert decision.selected_width == expected_width
        assert decision.k16_eligible is (
            enabled and expected_width == 16
        )
        assert decision.k16_fallback_reason == expected_reason
        if not enabled:
            assert decision.selected_width == (
                select_exact_greedy_decode_burst_width(
                    configured_width=8,
                    remaining_output_tokens=remaining,
                    initial_sequence_length=context,
                    block_size=256,
                    split_phase_enabled=False,
                    ragged_coalescing_enabled=False,
                )
            )


def test_context_gated_elastic_width_selector_honors_k16_health() -> None:
    select = select_context_gated_elastic_exact_burst_width
    assert callable(select)
    decision = select(
        enabled=True,
        base_width=8,
        initial_sequence_length=256,
        remaining_output_tokens=128,
        write_block_capacity=128,
        shared_graph_available=True,
        k16_health_quarantined=True,
        incompatible_modes=(),
    )
    assert decision.requested_width == 16
    assert decision.selected_width == 8
    assert decision.k16_eligible is False
    assert decision.k16_fallback_reason == "k16_health_quarantined"


def test_context_gated_elastic_width_selector_validates_inputs() -> None:
    select = select_context_gated_elastic_exact_burst_width
    assert callable(select)
    valid = {
        "enabled": True,
        "base_width": 8,
        "initial_sequence_length": 256,
        "remaining_output_tokens": 128,
        "write_block_capacity": 128,
        "shared_graph_available": True,
        "k16_health_quarantined": False,
        "incompatible_modes": (),
    }
    invalid_cases = (
        ("enabled", 1),
        ("base_width", True),
        ("initial_sequence_length", True),
        ("remaining_output_tokens", True),
        ("write_block_capacity", True),
        ("shared_graph_available", 1),
        ("k16_health_quarantined", 0),
        ("initial_sequence_length", 0),
        ("write_block_capacity", 0),
        ("remaining_output_tokens", -1),
        ("base_width", 4),
        ("incompatible_modes", ["split_phase"]),
        ("incompatible_modes", ("",)),
        ("incompatible_modes", (1,)),
    )
    for name, value in invalid_cases:
        kwargs = dict(valid)
        kwargs[name] = value
        _assert_raises(
            ValueError,
            {
                "enabled": "enabled must be a bool",
                "base_width": "base_width must be exactly 8",
                "initial_sequence_length": (
                    "initial_sequence_length must be a positive integer"
                ),
                "remaining_output_tokens": (
                    "remaining_output_tokens must be a "
                    "non-negative integer"
                ),
                "write_block_capacity": (
                    "write_block_capacity must be a positive integer"
                ),
                "shared_graph_available": (
                    "shared_graph_available must be a bool"
                ),
                "k16_health_quarantined": (
                    "k16_health_quarantined must be a bool"
                ),
                "incompatible_modes": (
                    "incompatible_modes must be a tuple of "
                    "non-empty strings"
                ),
            }[name],
            lambda kwargs=kwargs: select(**kwargs),
        )


def test_context_gated_elastic_width_selector_checks_modes_first() -> None:
    select = select_context_gated_elastic_exact_burst_width
    assert callable(select)
    decision = select(
        enabled=True,
        base_width=8,
        initial_sequence_length=4096,
        remaining_output_tokens=0,
        write_block_capacity=1,
        shared_graph_available=False,
        k16_health_quarantined=True,
        incompatible_modes=("split_phase",),
    )
    assert decision.requested_width == 16
    assert decision.selected_width == 8
    assert decision.k16_eligible is False
    assert decision.k16_fallback_reason == "incompatible_mode"


def test_policy_clips_to_budget_and_current_block() -> None:
    decision = build_exact_greedy_decode_burst_decision(
        **_eligible_kwargs()
    )
    assert decision.optimized is True
    assert decision.authorized_token_count == 6
    assert decision.first_write_position == 250
    assert decision.last_write_position == 255
    assert decision.fallback_reason is None

    kwargs = _eligible_kwargs()
    kwargs.update(
        configured_width=8,
        remaining_output_tokens=3,
        initial_sequence_length=101,
    )
    decision = build_exact_greedy_decode_burst_decision(
        **kwargs
    )
    assert decision.authorized_token_count == 3
    assert decision.output_budget_clipped is True
    assert decision.block_boundary_clipped is False

    kwargs = _eligible_kwargs()
    kwargs.update(
        configured_width=8,
        remaining_output_tokens=8,
        initial_sequence_length=253,
    )
    decision = build_exact_greedy_decode_burst_decision(
        **kwargs
    )
    assert decision.authorized_token_count == 4
    assert decision.output_budget_clipped is False
    assert decision.block_boundary_clipped is True


def test_boundary_width_one_falls_back_before_replay() -> None:
    kwargs = _eligible_kwargs()
    kwargs.update(
        configured_width=4,
        remaining_output_tokens=4,
        initial_sequence_length=256,
    )
    decision = build_exact_greedy_decode_burst_decision(
        **kwargs
    )
    assert decision.optimized is False
    assert decision.authorized_token_count == 1
    assert decision.first_write_position == 255
    assert decision.last_write_position == 255
    assert decision.fallback_reason == "authorized_width_below_two"


def test_gate_only_width_one_is_explicit_and_never_implicit() -> None:
    kwargs = _eligible_kwargs()
    kwargs["configured_width"] = 1
    _assert_raises(
        ValueError,
        "configured_width must be an integer in [2, 8]",
        lambda: build_exact_greedy_decode_burst_decision(**kwargs),
    )

    decision = build_exact_greedy_decode_burst_decision(
        **kwargs,
        allow_single_token_gate=True,
    )
    assert decision.optimized is True
    assert decision.authorized_token_count == 1
    assert decision.fallback_reason is None


def test_fallback_reasons_have_stable_precedence() -> None:
    cases = (
        ("enabled", False, "disabled"),
        ("sequence_count", 2, "sequence_count_unsupported"),
        ("waiting_count", 1, "waiting_present"),
        ("prefilling_count", 1, "prefilling_present"),
        ("is_prefill", True, "prefill_unsupported"),
        ("do_sample", False, "sampling_disabled"),
        ("batch_kind", "mixed", "mixed_batch_unsupported"),
        ("temperatures", (0.5,), "nonzero_temperature"),
        ("ignore_eos", (False,), "eos_sensitive"),
        ("completion_only", False, "visibility_unsupported"),
        (
            "tensor_parallel_size",
            2,
            "tensor_parallel_unsupported",
        ),
        ("rank", 1, "non_root_rank"),
        ("graph_available", False, "graph_unavailable"),
        (
            "incompatible_modes",
            ("kv_offload",),
            "incompatible_mode:kv_offload",
        ),
        ("pending_lease", True, "lease_pending"),
        ("quarantined", True, "quarantined"),
    )
    for field, value, expected_reason in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        decision = build_exact_greedy_decode_burst_decision(
            **kwargs
        )
        assert decision.optimized is False
        assert decision.fallback_reason == expected_reason

    kwargs = _eligible_kwargs()
    kwargs.update(
        enabled=False,
        sequence_count=2,
        graph_available=False,
    )
    assert (
        build_exact_greedy_decode_burst_decision(
            **kwargs
        ).fallback_reason
        == "disabled"
    )


def test_invalid_policy_inputs_fail_closed() -> None:
    cases = (
        (
            "enabled",
            1,
            "enabled must be a bool",
        ),
        (
            "configured_width",
            True,
            "configured_width must be an integer in [2, 8]",
        ),
        (
            "configured_width",
            1,
            "configured_width must be an integer in [2, 8]",
        ),
        (
            "configured_width",
            9,
            "configured_width must be an integer in [2, 8]",
        ),
        (
            "remaining_output_tokens",
            -1,
            "remaining_output_tokens must be a non-negative integer",
        ),
        (
            "initial_sequence_length",
            0,
            "initial_sequence_length must be a positive integer",
        ),
        (
            "block_size",
            0,
            "block_size must be a positive integer",
        ),
        (
            "temperatures",
            ("0",),
            "temperatures must contain finite numbers",
        ),
        (
            "ignore_eos",
            [True],
            "ignore_eos must be a tuple",
        ),
        (
            "incompatible_modes",
            ["kv_offload"],
            "incompatible_modes must be a tuple",
        ),
    )
    for field, value, message in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        _assert_raises(
            ValueError,
            message,
            lambda kwargs=kwargs:
                build_exact_greedy_decode_burst_decision(
                    **kwargs
                ),
        )


def _lease() -> ExactGreedyDecodeBurstLease:
    return build_exact_greedy_decode_burst_lease(
        sequence_id=17,
        schedule_generation=9,
        graph_generation=4,
        requested_token_count=8,
        authorized_token_count=4,
        initial_completion_count=3,
        initial_sequence_length=253,
        block_table_identity=((7, 2),),
        write_block_id=7,
        write_block_generation=2,
        first_write_position=252,
        last_write_position=255,
        first_physical_slot=2044,
        last_physical_slot=2047,
        remaining_output_tokens=9,
        completion_only=True,
    )


def _k8_lease() -> ExactGreedyDecodeBurstLease:
    return build_exact_greedy_decode_burst_lease(
        sequence_id=17,
        schedule_generation=9,
        graph_generation=4,
        requested_token_count=8,
        authorized_token_count=8,
        initial_completion_count=3,
        initial_sequence_length=249,
        block_table_identity=((7, 2),),
        write_block_id=7,
        write_block_generation=2,
        first_write_position=248,
        last_write_position=255,
        first_physical_slot=2040,
        last_physical_slot=2047,
        remaining_output_tokens=9,
        completion_only=True,
    )


class _SplitCompletion:
    def synchronize(self):
        return None


class _SplitMailbox:
    def __init__(self, values):
        self.values = tuple(values)

    def tolist(self):
        return list(self.values)


class _SplitBackend:
    def __init__(self, fake_graph, *, fail_phase=None):
        self._transfer_type = split_module.ExactBurstPhaseTransfer
        self._result_type = (
            split_module.ExactGreedyDecodeBurstSplitResult
        )
        self.fake_graph = fake_graph
        self.fail_phase = fail_phase
        self.calls = []
        self.aborted = []

    def begin_transaction(self):
        self.calls.append(("begin", self.fake_graph.replay_calls))
        return 7

    def build_tickets(self, **kwargs):
        return split_module.build_exact_burst_publication_tickets(
            **kwargs
        )

    def enqueue_phase(
        self,
        *,
        ticket,
        token_slice,
        mailbox_generation,
    ):
        self.calls.append(
            (
                ticket.phase,
                self.fake_graph.replay_calls,
                tuple(token_slice.values),
                mailbox_generation,
            )
        )
        if ticket.phase == self.fail_phase:
            raise RuntimeError(f"{ticket.phase} copy failed")
        return self._transfer_type(
            ticket=ticket,
            mailbox_generation=mailbox_generation,
            token_count=ticket.phase_token_count,
            byte_count=ticket.phase_token_count * 8,
            completion=_SplitCompletion(),
            mailbox=_SplitMailbox(token_slice.values),
        )

    def abort_transaction(self, mailbox_generation):
        self.aborted.append(mailbox_generation)

    def build_result(
        self,
        *,
        parent_lease_identity_sha256,
        graph_identity_sha256,
        replay_count,
        prefix,
        suffix,
        sampled_logit_d2h_calls=0,
        sampled_logits=(),
        correctness_trace=False,
    ):
        assert isinstance(correctness_trace, bool)
        assert not sampled_logits or correctness_trace
        return self._result_type(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            graph_identity_sha256=graph_identity_sha256,
            replay_count=replay_count,
            prefix=prefix,
            suffix=suffix,
            sampled_logit_d2h_calls=sampled_logit_d2h_calls,
            sampled_logits=sampled_logits,
        )


def _k16_lease(
    *,
    width_health_generation=1,
) -> ExactGreedyDecodeBurstLease:
    return build_exact_greedy_decode_burst_lease(
        sequence_id=17,
        schedule_generation=9,
        graph_generation=4,
        requested_token_count=16,
        authorized_token_count=16,
        initial_completion_count=3,
        initial_sequence_length=241,
        block_table_identity=((7, 2),),
        write_block_id=7,
        write_block_generation=2,
        first_write_position=240,
        last_write_position=255,
        first_physical_slot=2032,
        last_physical_slot=2047,
        remaining_output_tokens=16,
        completion_only=True,
        width_health_generation=width_health_generation,
    )


def test_elastic_width_health_is_monotonic_and_fail_closed() -> None:
    assert callable(ElasticBurstWidthHealth)
    health = ElasticBurstWidthHealth()
    assert health.generation == 1
    assert health.k16_quarantine_reason is None

    health.quarantine_k16("host_visible_gap_exceeded")
    assert health.generation == 2
    assert (
        health.k16_quarantine_reason
        == "host_visible_gap_exceeded"
    )

    health.quarantine_k16("host_visible_gap_exceeded")
    health.quarantine_k16("different_failure")
    assert health.generation == 2
    assert (
        health.k16_quarantine_reason
        == "host_visible_gap_exceeded"
    )

    for invalid in ("", None, 1):
        _assert_raises(
            ValueError,
            (
                "K16 quarantine reason must be a "
                "non-empty string"
            ),
            lambda invalid=invalid: ElasticBurstWidthHealth(
            ).quarantine_k16(invalid),
        )

    maximum = module.MAX_ELASTIC_BURST_WIDTH_HEALTH_GENERATION
    exhausted = ElasticBurstWidthHealth(generation=maximum)
    _assert_raises(
        OverflowError,
        "K16 width-health generation exhausted",
        lambda: exhausted.quarantine_k16("failure"),
    )
    assert exhausted.generation == maximum
    assert exhausted.k16_quarantine_reason is None


def test_elastic_width_health_capability_reuses_shared_graph() -> None:
    assert callable(ElasticBurstWidthHealth)
    shared = {
        "available": True,
        "fallback_reason": None,
        "graph_identity_sha256": "a" * 64,
        "graph_generation": 7,
        "history_capacity": 16,
    }
    original = dict(shared)
    health = ElasticBurstWidthHealth()

    capability = health.capability(shared)

    assert shared == original
    assert capability == {
        "k8_available": True,
        "k16_available": True,
        "k16_fallback_reason": None,
        "k16_health_generation": 1,
        "shared_graph_identity_sha256": "a" * 64,
        "shared_graph_generation": 7,
        "shared_graph_history_capacity": 16,
        "incremental_retained_static_bytes": 0,
    }

    health.quarantine_k16("host_visible_gap_exceeded")
    quarantined = health.capability(shared)
    assert quarantined["k8_available"] is True
    assert quarantined["k16_available"] is False
    assert quarantined["k16_fallback_reason"] == (
        "k16_quarantined:host_visible_gap_exceeded"
    )

    unavailable = health.capability(
        {
            **shared,
            "available": False,
            "fallback_reason": "quarantined",
        }
    )
    assert unavailable["k8_available"] is False
    assert unavailable["k16_available"] is False
    assert unavailable["k16_fallback_reason"] == (
        "shared_graph_unavailable:quarantined"
    )

    undersized = ElasticBurstWidthHealth().capability(
        {**shared, "history_capacity": 8}
    )
    assert undersized["k8_available"] is True
    assert undersized["k16_available"] is False
    assert undersized["k16_fallback_reason"] == (
        "k16_history_capacity_below_16"
    )


def test_k16_lease_rejects_stale_width_health_before_mutation() -> None:
    graph, _tensors, fake_graph, events, _slots = _graph_fixture(
        history_capacity=16,
    )
    events.clear()
    replay_calls_before = fake_graph.replay_calls

    result = graph.replay(
        lease=_k16_lease(width_health_generation=3),
        initial_token=19,
        block_table=_BurstTensor(
            [[7, -1]],
            label="live_block_table",
            events=events,
            dtype="int32",
            element_size=4,
        ),
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
        expected_width_health_generation=4,
    )

    assert isinstance(result, ExactGreedyDecodeBurstFallback)
    assert (
        result.fallback_reason
        == "lease_width_health_generation_drift"
    )
    assert fake_graph.replay_calls == replay_calls_before
    assert events == []


def test_k16_post_replay_failure_quarantines_only_width_health() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture(
        history_capacity=16,
    )
    health = ElasticBurstWidthHealth()

    def replay_step(ordinal):
        token = 20 + ordinal
        tensors["token_history"].values[ordinal] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    tensors["token_history"].fail_tolist = True
    _assert_raises(
        RuntimeError,
        "final D2H failed",
        lambda: graph.replay(
            lease=_k16_lease(width_health_generation=1),
            initial_token=19,
            block_table=_BurstTensor(
                [[7, -1]],
                label="live_block_table",
                events=[],
                dtype="int32",
                element_size=4,
            ),
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
            expected_width_health_generation=1,
            width_health=health,
        ),
    )
    assert graph.summary()["quarantine_reason"] is None
    assert health.generation == 2
    assert health.k16_quarantine_reason == (
        "final_token_d2h_failure:RuntimeError"
    )


def test_k16_replay_failure_quarantines_the_shared_graph() -> None:
    graph, _tensors, fake_graph, _events, _slots = _graph_fixture(
        history_capacity=16,
    )
    health = ElasticBurstWidthHealth()
    fake_graph.replay_error_at = fake_graph.replay_calls + 1

    _assert_raises(
        RuntimeError,
        f"replay {fake_graph.replay_error_at} failed",
        lambda: graph.replay(
            lease=_k16_lease(width_health_generation=1),
            initial_token=19,
            block_table=_BurstTensor(
                [[7, -1]],
                label="live_block_table",
                events=[],
                dtype="int32",
                element_size=4,
            ),
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
            expected_width_health_generation=1,
            width_health=health,
        ),
    )
    assert graph.summary()["quarantine_reason"] == (
        "replay_failure:RuntimeError"
    )
    assert health.generation == 1
    assert health.k16_quarantine_reason is None


def _continuation_receipt(
    **overrides,
) -> ExactGreedyDecodeBurstContinuationReceipt:
    values = {
        "sequence_id": 7,
        "graph_generation": 3,
        "block_table_identity": ((11, 4), (12, 1)),
        "write_block_id": 12,
        "write_block_generation": 1,
        "next_input_token": 99,
        "next_position": 260,
        "next_context_length": 261,
        "next_physical_slot": 12 * 256 + 4,
        "history_cursor": 4,
    }
    values.update(overrides)
    return ExactGreedyDecodeBurstContinuationReceipt(**values)


def _continuation_lease(
    **overrides,
) -> ExactGreedyDecodeBurstLease:
    values = {
        "sequence_id": 7,
        "schedule_generation": 10,
        "graph_generation": 3,
        "requested_token_count": 4,
        "authorized_token_count": 4,
        "initial_completion_count": 8,
        "initial_sequence_length": 261,
        "block_table_identity": ((11, 4), (12, 1)),
        "write_block_id": 12,
        "write_block_generation": 1,
        "first_write_position": 260,
        "last_write_position": 263,
        "first_physical_slot": 12 * 256 + 4,
        "last_physical_slot": 12 * 256 + 7,
        "remaining_output_tokens": 120,
        "completion_only": True,
    }
    values.update(overrides)
    return build_exact_greedy_decode_burst_lease(**values)


def _epoch_replay_lease(
    *,
    first_write_position: int,
    first_physical_slot: int,
) -> ExactGreedyDecodeBurstLease:
    return build_exact_greedy_decode_burst_lease(
        sequence_id=7,
        schedule_generation=10,
        graph_generation=4,
        requested_token_count=4,
        authorized_token_count=4,
        initial_completion_count=(
            first_write_position - 252
        ),
        initial_sequence_length=first_write_position + 1,
        block_table_identity=((11, 4), (12, 1)),
        write_block_id=12,
        write_block_generation=1,
        first_write_position=first_write_position,
        last_write_position=first_write_position + 3,
        first_physical_slot=first_physical_slot,
        last_physical_slot=first_physical_slot + 3,
        remaining_output_tokens=120,
        completion_only=True,
    )


def _continuation_decision(
    *,
    receipt=None,
    lease=None,
    enabled=True,
    initial_token=99,
    graph_generation=3,
    history_capacity=256,
    block_size=256,
):
    return decide_exact_greedy_decode_burst_continuation(
        enabled=enabled,
        receipt=(
            _continuation_receipt()
            if receipt is None
            else receipt
        ),
        lease=_continuation_lease() if lease is None else lease,
        initial_token=initial_token,
        graph_generation=graph_generation,
        history_capacity=history_capacity,
        block_size=block_size,
    )


def test_generation_sealed_lease_and_continuation_receipt_contract():
    baseline = _continuation_lease()
    assert baseline.block_table_identity
    assert baseline.block_table_identity_seal is None
    assert baseline.identity_sha256 == (
        "92e6396f17264a52420a96d3e5f69053"
        "be3461ac028f704a6c57979480ba7622"
    )

    seal = BlockTableIdentitySeal(
        sequence_id=7,
        table_revision=3,
        ownership_generation=11,
        block_count=2,
        write_block_index=1,
        write_block_id=12,
        write_block_generation=1,
        predecessor_block_id=11,
        predecessor_block_generation=4,
        identity_sha256="a" * 64,
    )
    sealed = _continuation_lease(
        block_table_identity=(),
        block_table_identity_seal=seal,
    )
    assert sealed.block_table_identity == ()
    assert sealed.block_table_identity_seal == seal
    assert pickle.loads(pickle.dumps(sealed)) == sealed
    assert sealed.identity_sha256 != baseline.identity_sha256
    assert _continuation_lease(
        block_table_identity=(),
        block_table_identity_seal=seal,
    ).identity_sha256 == sealed.identity_sha256
    assert _continuation_lease(
        block_table_identity=(),
        block_table_identity_seal=replace(
            seal,
            ownership_generation=12,
        ),
    ).identity_sha256 != sealed.identity_sha256
    for changed_seal, lease_overrides in (
        (replace(seal, sequence_id=8), {"sequence_id": 8}),
        (replace(seal, table_revision=4), {}),
        (replace(seal, ownership_generation=12), {}),
        (replace(seal, block_count=3), {}),
        (replace(seal, write_block_index=0), {}),
        (
            replace(seal, write_block_id=13),
            {"write_block_id": 13},
        ),
        (
            replace(seal, write_block_generation=2),
            {"write_block_generation": 2},
        ),
        (replace(seal, predecessor_block_id=10), {}),
        (
            replace(seal, predecessor_block_generation=5),
            {},
        ),
        (replace(seal, identity_sha256="b" * 64), {}),
    ):
        assert _continuation_lease(
            block_table_identity=(),
            block_table_identity_seal=changed_seal,
            **lease_overrides,
        ).identity_sha256 != sealed.identity_sha256

    baseline_receipt = _continuation_receipt()
    sealed_receipt = _continuation_receipt(
        block_table_identity=(),
        block_table_identity_seal=seal,
    )
    assert (
        baseline_receipt.block_table_identity
        == baseline.block_table_identity
    )
    assert baseline_receipt.block_table_identity_seal is None
    assert sealed_receipt.block_table_identity == ()
    assert sealed_receipt.block_table_identity_seal == seal

    for kwargs in (
        {
            "block_table_identity": ((11, 4), (12, 1)),
            "block_table_identity_seal": seal,
        },
        {
            "block_table_identity": (),
            "block_table_identity_seal": None,
        },
    ):
        _assert_raises(
            ValueError,
            "exactly one block-table identity mode is required",
            lambda kwargs=kwargs: _continuation_receipt(**kwargs),
        )


def test_generation_sealed_lease_digest_avoids_recursive_dataclass_copy():
    seal = BlockTableIdentitySeal(
        sequence_id=7,
        table_revision=3,
        ownership_generation=11,
        block_count=2,
        write_block_index=1,
        write_block_id=12,
        write_block_generation=1,
        predecessor_block_id=11,
        predecessor_block_generation=4,
        identity_sha256="a" * 64,
    )
    original_asdict = module.asdict

    def reject_recursive_copy(_value):
        raise AssertionError(
            "sealed lease digest must not recursively copy the seal"
        )

    module.asdict = reject_recursive_copy
    try:
        sealed = _continuation_lease(
            block_table_identity=(),
            block_table_identity_seal=seal,
        )
    finally:
        module.asdict = original_asdict

    assert sealed.block_table_identity_seal is seal
    assert sealed.identity_sha256 == (
        "707ee58e07b1a534d5550ef2c65db108"
        "dad58c5c4809852162a42f65e821cc27"
    )


def test_generation_sealed_lease_digest_avoids_generic_json_encoder():
    seal = BlockTableIdentitySeal(
        sequence_id=7,
        table_revision=3,
        ownership_generation=11,
        block_count=2,
        write_block_index=1,
        write_block_id=12,
        write_block_generation=1,
        predecessor_block_id=11,
        predecessor_block_generation=4,
        identity_sha256="a" * 64,
    )
    original_dumps = module.json.dumps

    def reject_generic_json(*_args, **_kwargs):
        raise AssertionError(
            "sealed lease digest must use its canonical fast encoder"
        )

    module.json.dumps = reject_generic_json
    try:
        first = _continuation_lease(
            block_table_identity=(),
            block_table_identity_seal=seal,
        )
        second = _continuation_lease(
            block_table_identity=(),
            block_table_identity_seal=seal,
        )
    finally:
        module.json.dumps = original_dumps

    assert first.identity_sha256 == second.identity_sha256
    assert len(first.identity_sha256) == 64


def test_continuation_requires_an_exact_receipt_match() -> None:
    decision = _continuation_decision()
    assert decision.continue_from_resident_state is True
    assert decision.history_start == 4
    assert decision.miss_reason is None

    missing = decide_exact_greedy_decode_burst_continuation(
        enabled=True,
        receipt=None,
        lease=_continuation_lease(),
        initial_token=99,
        graph_generation=3,
        history_capacity=256,
        block_size=256,
    )
    cases = (
        (
            "disabled",
            _continuation_decision(enabled=False),
        ),
        ("receipt_missing", missing),
        (
            "sequence_identity_drift",
            _continuation_decision(
                lease=replace(
                    _continuation_lease(),
                    sequence_id=8,
                )
            ),
        ),
        (
            "graph_generation_drift",
            _continuation_decision(graph_generation=4),
        ),
        (
            "block_table_identity_drift",
            _continuation_decision(
                lease=replace(
                    _continuation_lease(),
                    block_table_identity=((10, 4), (12, 1)),
                )
            ),
        ),
        (
            "write_block_identity_drift",
            _continuation_decision(
                receipt=_continuation_receipt(
                    write_block_generation=2,
                )
            ),
        ),
        (
            "initial_token_drift",
            _continuation_decision(initial_token=100),
        ),
        (
            "position_drift",
            _continuation_decision(
                lease=replace(
                    _continuation_lease(),
                    first_write_position=261,
                    last_write_position=264,
                )
            ),
        ),
        (
            "context_length_drift",
            _continuation_decision(
                lease=replace(
                    _continuation_lease(),
                    initial_sequence_length=262,
                )
            ),
        ),
        (
            "physical_slot_drift",
            _continuation_decision(
                lease=replace(
                    _continuation_lease(),
                    first_physical_slot=12 * 256 + 5,
                    last_physical_slot=12 * 256 + 8,
                )
            ),
        ),
        (
            "physical_block_boundary_crossed",
            _continuation_decision(
                receipt=_continuation_receipt(
                    next_physical_slot=12 * 256 + 254,
                ),
                lease=replace(
                    _continuation_lease(),
                    first_physical_slot=12 * 256 + 254,
                    last_physical_slot=13 * 256 + 1,
                ),
            ),
        ),
        (
            "history_capacity_exceeded",
            _continuation_decision(
                receipt=_continuation_receipt(
                    history_cursor=254,
                )
            ),
        ),
    )
    for expected_reason, actual in cases:
        assert actual.continue_from_resident_state is False
        assert actual.history_start == 0
        assert actual.miss_reason == expected_reason


def test_continuation_rejects_invalid_scalar_contracts() -> None:
    for field, value, message in (
        (
            "sequence_id",
            True,
            "sequence_id must be a non-negative integer",
        ),
        (
            "history_cursor",
            -1,
            "history_cursor must be a non-negative integer",
        ),
    ):
        _assert_raises(
            ValueError,
            message,
            lambda field=field, value=value:
                _continuation_receipt(**{field: value}),
        )
    for field, value, message in (
        ("enabled", 1, "enabled must be a bool"),
        (
            "initial_token",
            True,
            "initial_token must be a non-negative integer",
        ),
        (
            "graph_generation",
            -1,
            "graph_generation must be a non-negative integer",
        ),
        (
            "history_capacity",
            0,
            "history_capacity must be a positive integer",
        ),
        (
            "block_size",
            False,
            "block_size must be a positive integer",
        ),
    ):
        kwargs = {field: value}
        _assert_raises(
            ValueError,
            message,
            lambda kwargs=kwargs: _continuation_decision(**kwargs),
        )


def _result(lease) -> ExactGreedyDecodeBurstResult:
    return ExactGreedyDecodeBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        tokens=(11, 12, 13, 14),
        replay_count=4,
        final_input_token=14,
        final_position=256,
        final_context_length=257,
        final_physical_slot=2048,
        graph_identity_sha256="a" * 64,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=0,
    )


def test_lease_identity_is_canonical_and_result_is_exact() -> None:
    first = _lease()
    second = _lease()
    assert first == second
    assert len(first.identity_sha256) == 64
    int(first.identity_sha256, 16)
    assert validate_exact_greedy_decode_burst_result(
        first,
        _result(first),
    ) == _result(first)

    mismatched = ExactGreedyDecodeBurstResult(
        **{
            **_result(first).__dict__,
            "replay_count": 3,
        }
    )
    _assert_raises(
        ValueError,
        "burst result replay count does not match lease",
        lambda: validate_exact_greedy_decode_burst_result(
            first,
            mismatched,
        ),
    )


def test_correctness_trace_is_bounded_and_ordered() -> None:
    lease = _lease()
    result = ExactGreedyDecodeBurstResult(
        **{
            **_result(lease).__dict__,
            "sampled_logit_d2h_calls": 1,
            "sampled_logits": (
                (0, (1.0, 2.0)),
                (2, (3.0, 4.0)),
            ),
        }
    )
    validate_exact_greedy_decode_burst_result(
        lease,
        result,
        correctness_trace=True,
    )
    duplicate = ExactGreedyDecodeBurstResult(
        **{
            **result.__dict__,
            "sampled_logits": (
                (0, (1.0,)),
                (0, (2.0,)),
            ),
        }
    )
    _assert_raises(
        ValueError,
        "sampled logit ordinals must be strictly increasing",
        lambda: validate_exact_greedy_decode_burst_result(
            lease,
            duplicate,
            correctness_trace=True,
        ),
    )
    _assert_raises(
        ValueError,
        "production burst cannot return sampled logits",
        lambda: validate_exact_greedy_decode_burst_result(
            lease,
            result,
            correctness_trace=False,
        ),
    )


def test_stats_track_benefit_cost_and_terminal_state() -> None:
    receipt = ExactGreedyDecodeBurstCaptureReceipt(
        graph_identity_sha256="b" * 64,
        graph_generation=4,
        capture_duration_ns=123,
        allocated_delta_bytes=456,
        reserved_delta_bytes=789,
        retained_static_bytes=321,
        scratch_block_count=1,
        correctness_trace=False,
        flash_attn_num_splits=0,
    )
    stats = ExactGreedyDecodeBurstStats()
    stats.record_attempt()
    stats.record_acceptance(
        requested_token_count=8,
        authorized_token_count=4,
        output_budget_clipped=False,
        block_boundary_clipped=True,
    )
    stats.record_capture(receipt)
    stats.record_replays(4)
    stats.record_final_token_d2h(token_count=4, byte_count=32)
    stats.record_commit(token_count=4, host_visible_gap_ns=12_000_000)

    summary = stats.summary()
    json.dumps(summary, allow_nan=False)
    assert summary["attempts"] == 1
    assert summary["acceptances"] == 1
    assert summary["target_model_forwards"] == 4
    assert summary["graph_replays"] == 4
    assert summary["intermediate_token_d2h_calls"] == 0
    assert summary["final_token_d2h_calls"] == 1
    assert summary["final_token_d2h_bytes"] == 32
    assert summary["committed_tokens"] == 4
    assert summary["pending_leases"] == 0
    assert summary["maximum_host_visible_gap_ns"] == 12_000_000
    assert summary["block_boundary_clipped"] == 1
    assert summary["capture_receipts"][0][
        "scratch_block_count"
    ] == 1


def test_stats_track_elastic_k16_selection_and_per_width_commits() -> None:
    stats = ExactGreedyDecodeBurstStats()
    stats.record_elastic_width_decision(
        requested_width=16,
        selected_width=16,
        fallback_reason=None,
    )
    stats.record_acceptance(
        requested_token_count=16,
        authorized_token_count=16,
        output_budget_clipped=False,
        block_boundary_clipped=False,
    )
    stats.record_commit(
        token_count=16,
        host_visible_gap_ns=10,
        authorized_width=16,
    )
    stats.record_elastic_width_decision(
        requested_width=16,
        selected_width=8,
        fallback_reason="context_above_2048",
    )
    stats.record_acceptance(
        requested_token_count=8,
        authorized_token_count=8,
        output_budget_clipped=False,
        block_boundary_clipped=False,
    )
    stats.record_commit(
        token_count=8,
        host_visible_gap_ns=20,
        authorized_width=8,
    )

    summary = stats.summary()
    assert summary["k16_attempts"] == 2
    assert summary["k16_acceptances"] == 1
    assert summary["k8_fallbacks"] == 1
    assert summary["elastic_k16_fallback_counts"] == {
        "context_above_2048": 1,
    }
    assert summary["per_width_commits"] == {"8": 1, "16": 1}
    assert summary["fallback_counts"] == {}


def test_k16_selection_is_not_acceptance_until_lease_is_issued() -> None:
    stats = ExactGreedyDecodeBurstStats()
    stats.record_elastic_width_decision(
        requested_width=16,
        selected_width=16,
        fallback_reason=None,
    )

    summary = stats.summary()
    assert summary["k16_attempts"] == 1
    assert summary["k16_acceptances"] == 0
    assert summary["pending_leases"] == 0


def test_stats_track_split_phase_benefit_cost_and_failures() -> None:
    stats = ExactGreedyDecodeBurstStats()
    stats.record_acceptance(
        requested_token_count=8,
        authorized_token_count=8,
        output_budget_clipped=False,
        block_boundary_clipped=False,
    )
    stats.record_split_phase_inventory(
        prefix_byte_count=32,
        suffix_byte_count=32,
        replay_count=8,
    )
    stats.record_split_phase_wait("prefix")
    stats.record_split_phase_commit(
        phase="prefix",
        token_count=4,
        parent_token_count=8,
        host_visible_gap_ns=12_000_000,
    )
    stats.record_split_phase_wait("suffix")
    stats.record_split_phase_drain()
    stats.record_split_phase_failure("suffix_wait_failure")
    stats.record_split_phase_commit(
        phase="suffix",
        token_count=4,
        parent_token_count=8,
        host_visible_gap_ns=7_000_000,
    )

    summary = stats.summary()
    assert summary["graph_replays"] == 8
    assert summary["prefix_publication_tickets"] == 1
    assert summary["suffix_publication_tickets"] == 1
    assert summary["prefix_token_d2h_calls"] == 1
    assert summary["suffix_token_d2h_calls"] == 1
    assert summary["prefix_token_d2h_bytes"] == 32
    assert summary["suffix_token_d2h_bytes"] == 32
    assert summary["prefix_phase_waits"] == 1
    assert summary["suffix_phase_waits"] == 1
    assert summary["suffix_drains"] == 1
    assert summary["prefix_commits"] == 1
    assert summary["suffix_commits"] == 1
    assert summary["prefix_committed_tokens"] == 4
    assert summary["suffix_committed_tokens"] == 4
    assert summary["commits"] == 1
    assert summary["committed_tokens"] == 8
    assert summary["pending_leases"] == 0
    assert summary["maximum_host_visible_gap_ns"] == 12_000_000
    assert summary["split_phase_failure_counts"] == {
        "suffix_wait_failure": 1,
    }


def test_stats_track_lease_local_delta_journal_lifecycle() -> None:
    stats = ExactGreedyDecodeBurstStats()
    stats.record_lease_local_delta_journal_attempt()
    stats.record_lease_local_delta_journal_capture()
    stats.record_lease_local_delta_journal_commit(
        published_blocks=1,
    )
    stats.record_lease_local_delta_journal_rollback()
    stats.record_lease_local_delta_journal_fallback(
        "terminal_suffix",
    )
    stats.record_lease_local_delta_journal_one_phase_attempt()
    stats.record_lease_local_delta_journal_one_phase_capture()
    stats.record_lease_local_delta_journal_one_phase_commit(
        published_blocks=1,
    )
    stats.record_lease_local_delta_journal_one_phase_rollback()
    stats.record_lease_local_delta_journal_one_phase_fallback(
        "terminal_one_phase",
    )

    summary = stats.summary()
    assert summary["lease_local_delta_journal_attempts"] == 1
    assert summary["lease_local_delta_journal_captures"] == 1
    assert summary["lease_local_delta_journal_commits"] == 1
    assert summary["lease_local_delta_journal_rollbacks"] == 1
    assert (
        summary["lease_local_delta_journal_published_blocks"]
        == 1
    )
    assert summary[
        "lease_local_delta_journal_fallback_counts"
    ] == {"terminal_suffix": 1}
    assert summary[
        "lease_local_delta_journal_one_phase_attempts"
    ] == 1
    assert summary[
        "lease_local_delta_journal_one_phase_captures"
    ] == 1
    assert summary[
        "lease_local_delta_journal_one_phase_commits"
    ] == 1
    assert summary[
        "lease_local_delta_journal_one_phase_rollbacks"
    ] == 1
    assert summary[
        "lease_local_delta_journal_one_phase_published_blocks"
    ] == 1
    assert summary[
        "lease_local_delta_journal_one_phase_fallback_counts"
    ] == {"terminal_one_phase": 1}

    try:
        stats.record_lease_local_delta_journal_commit(
            published_blocks=2,
        )
    except ValueError as error:
        assert str(error) == "published_blocks must be at most one"
    else:
        raise AssertionError(
            "published_blocks greater than one was accepted"
        )
    try:
        stats.record_lease_local_delta_journal_fallback("")
    except ValueError as error:
        assert str(error) == (
            "lease-local delta journal fallback reason "
            "must be a non-empty string"
        )
    else:
        raise AssertionError(
            "empty delta journal fallback reason was accepted"
        )


def test_stats_track_continuation_benefit_and_cost() -> None:
    stats = ExactGreedyDecodeBurstStats()
    stats.record_identity_seal_capture(hot_reuse=False)
    stats.record_identity_seal_capture(hot_reuse=True)
    stats.record_identity_seal_validation()
    for _ in range(3):
        stats.record_continuation_attempt()
    stats.record_cold_bind()
    stats.record_continuation_hit(
        token_count=4,
        skipped_block_table_bytes=32,
    )
    stats.record_continuation_hit(
        token_count=4,
        skipped_block_table_bytes=32,
    )
    stats.record_continuation_miss("position_drift")
    stats.record_continuation_invalidation(
        "engine_failure:RuntimeError"
    )

    summary = stats.summary()
    assert summary["identity_seal_cold_captures"] == 1
    assert summary["identity_seal_hot_reuses"] == 1
    assert summary["identity_seal_validations"] == 1
    assert summary["identity_seal_fallback_counts"] == {}
    assert summary["continuation_attempts"] == 3
    assert summary["continuation_hits"] == 2
    assert summary["cold_binds"] == 1
    assert summary["continuation_miss_counts"] == {
        "position_drift": 1,
    }
    assert summary["continuation_invalidation_counts"] == {
        "engine_failure:RuntimeError": 1,
    }
    assert summary["continuation_tokens"] == 8
    assert summary["continuation_bursts"] == 2
    assert summary["skipped_static_reset_operations"] == 14
    assert summary["skipped_scalar_bind_operations"] == 10
    assert summary["skipped_block_table_constructions"] == 2
    assert summary["skipped_block_table_copy_calls"] == 2
    assert summary["skipped_block_table_bytes"] == 64

    for callback, message in (
        (
            lambda: stats.record_continuation_miss(""),
            "continuation miss reason must be a non-empty string",
        ),
        (
            lambda: stats.record_continuation_invalidation(""),
            (
                "continuation invalidation reason must be a "
                "non-empty string"
            ),
        ),
        (
            lambda: stats.record_continuation_hit(
                token_count=True,
                skipped_block_table_bytes=0,
            ),
            "token_count must be a positive integer",
        ),
        (
            lambda: stats.record_continuation_hit(
                token_count=1,
                skipped_block_table_bytes=-1,
            ),
            (
                "skipped_block_table_bytes must be a "
                "non-negative integer"
            ),
        ),
    ):
        _assert_raises(ValueError, message, callback)

    stats.quarantine("replay_failure:RuntimeError")
    stats.record_continuation_invalidation(
        "replay_failure:RuntimeError"
    )
    assert stats.summary()[
        "continuation_invalidation_counts"
    ] == {
        "engine_failure:RuntimeError": 1,
    }


def test_stats_reject_unknown_identity_seal_fallback_reason() -> None:
    stats = ExactGreedyDecodeBurstStats()
    stats.record_identity_seal_fallback("untracked_block_table")
    assert stats.summary()["identity_seal_fallback_counts"] == {
        "untracked_block_table": 1,
    }

    stats.identity_seal_fallback_counts["unknown_reason"] = 1
    _assert_raises(
        ValueError,
        "unknown identity seal fallback reason: unknown_reason",
        stats.summary,
    )


def test_contract_is_model_agnostic_and_supports_second_caller() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "Qwen",
        "checkpoint",
        "tokenizer",
        "prompt",
        "A100",
        '"short"',
        '"medium"',
        '"long"',
    ):
        assert forbidden not in source

    lease = build_exact_greedy_decode_burst_lease(
        sequence_id=3,
        schedule_generation=2,
        graph_generation=5,
        requested_token_count=2,
        authorized_token_count=2,
        initial_completion_count=1,
        initial_sequence_length=9,
        block_table_identity=((2, 6),),
        write_block_id=2,
        write_block_generation=6,
        first_write_position=8,
        last_write_position=9,
        first_physical_slot=40,
        last_physical_slot=41,
        remaining_output_tokens=2,
        completion_only=True,
    )
    result = ExactGreedyDecodeBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        tokens=(101, 102),
        replay_count=2,
        final_input_token=102,
        final_position=10,
        final_context_length=11,
        final_physical_slot=42,
        graph_identity_sha256="c" * 64,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=0,
    )
    assert validate_exact_greedy_decode_burst_result(
        lease,
        result,
    ).tokens == (101, 102)


def test_complete_step_capture_orders_body_and_uses_private_scratch() -> None:
    graph, tensors, _fake_graph, events, context_slots = (
        _graph_fixture()
    )

    expected_body = [
        "model",
        "compute_logits",
        "to",
        "argmax",
        "index_copy_",
        "copy_",
        "add_",
        "add_",
        "add_",
        "add_",
    ]
    body = [
        event[1]
        for event in events
        if len(event) > 1
        and event[1] in set(expected_body)
    ]
    assert body == expected_body + expected_body

    assert context_slots == [(9 * 256,), (9 * 256,)]
    assert tensors["input_token"].values == [-1]
    assert tensors["position"].values == [-1]
    assert tensors["context_length"].values == [-1]
    assert tensors["slot_mapping"].values == [-1]
    assert tensors["block_table"].values == [[-1, -1]]
    assert tensors["token_history"].values == [-1] * 8
    assert tensors["history_index"].values == [0]
    summary = graph.summary()
    assert summary["capture_receipts"][0][
        "capture_duration_ns"
    ] == 900
    assert summary["capture_receipts"][0][
        "allocated_delta_bytes"
    ] == 40
    assert summary["capture_receipts"][0][
        "reserved_delta_bytes"
    ] == 60
    assert summary["capture_receipts"][0][
        "scratch_block_count"
    ] == 1


def test_capture_binds_flash_attention_split_to_identity_and_receipt() -> None:
    auto, _tensors, _graph, auto_events, _slots = _graph_fixture()
    split, _tensors, _graph, split_events, _slots = _graph_fixture(
        flash_attn_num_splits=12,
    )

    assert auto.receipt.flash_attn_num_splits == 0
    assert split.receipt.flash_attn_num_splits == 12
    assert (
        auto.receipt.graph_identity_sha256
        != split.receipt.graph_identity_sha256
    )
    assert auto.capability()["flash_attn_num_splits"] == 0
    assert split.capability()["flash_attn_num_splits"] == 12
    assert [
        event[2]
        for event in auto_events
        if event[1] == "set_context"
    ] == [0, 0]
    assert [
        event[2]
        for event in split_events
        if event[1] == "set_context"
    ] == [12, 12]


def test_capture_rejects_invalid_flash_attention_split() -> None:
    for invalid in (-1, True, 1.5, "12"):
        _assert_raises(
            ValueError,
            (
                "flash_attn_num_splits must be a "
                "non-negative integer"
            ),
            lambda invalid=invalid: _graph_fixture(
                flash_attn_num_splits=invalid,
            ),
        )


def test_replay_runs_exact_count_then_one_token_d2h() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture()
    lease = _lease()

    def replay_step(ordinal):
        token = 20 + ordinal
        tensors["token_history"].values[ordinal] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    block_table = _BurstTensor(
        [[7, -1]],
        label="live_block_table",
        events=[],
        dtype="int32",
        element_size=4,
    )

    result = graph.replay(
        lease=lease,
        initial_token=19,
        block_table=block_table,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )

    assert result.tokens == (20, 21, 22, 23)
    assert result.replay_count == 4
    assert result.final_input_token == 23
    assert result.final_position == 256
    assert result.final_context_length == 257
    assert result.final_physical_slot == 2048
    assert fake_graph.replay_calls == 4
    assert tensors["token_history"].tolist_calls == 1
    assert result.token_d2h_calls == 1
    assert result.sampled_logit_d2h_calls == 0
    summary = graph.summary()
    assert summary["target_model_forwards"] == 4
    assert summary["graph_replays"] == 4
    assert summary["intermediate_token_d2h_calls"] == 0
    assert summary["final_token_d2h_calls"] == 1


def test_replay_cold_binds_then_continues_resident_state() -> None:
    graph, tensors, fake_graph, events, _slots = _graph_fixture()
    events.clear()
    factory_calls = []

    def materialize_block_table():
        factory_calls.append("called")
        return _BurstTensor(
            [[11, 12]],
            label="live_block_table",
            events=events,
            dtype="int32",
            element_size=4,
        )

    def replay_step(_ordinal):
        history_index = tensors["history_index"].values[0]
        token = 100 + history_index
        tensors["token_history"].values[history_index] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    first = graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=260,
            first_physical_slot=12 * 256 + 4,
        ),
        initial_token=99,
        block_table=None,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    first_setup_events = tuple(
        event
        for event in events
        if event[1] in {"fill_", "copy_"}
    )
    assert first.tokens == (100, 101, 102, 103)
    assert factory_calls == ["called"]
    assert first_setup_events

    events.clear()
    second = graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=264,
            first_physical_slot=12 * 256 + 8,
        ),
        initial_token=103,
        block_table=None,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert second.tokens == (104, 105, 106, 107)
    assert factory_calls == ["called"]
    assert not [
        event
        for event in events
        if event[1] in {"fill_", "copy_"}
    ]
    summary = graph.summary()
    assert summary["continuation_attempts"] == 2
    assert summary["continuation_hits"] == 1
    assert summary["cold_binds"] == 1
    assert summary["continuation_miss_counts"] == {
        "receipt_missing": 1,
    }
    assert summary["skipped_block_table_bytes"] == 8

    events.clear()
    mismatched = graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=269,
            first_physical_slot=12 * 256 + 12,
        ),
        initial_token=107,
        block_table=None,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert isinstance(mismatched, ExactGreedyDecodeBurstResult)
    assert factory_calls == ["called", "called"]
    assert any(
        event[1] in {"fill_", "copy_"} for event in events
    )
    summary = graph.summary()
    assert summary["continuation_miss_counts"] == {
        "position_drift": 1,
        "receipt_missing": 1,
    }
    assert summary["cold_binds"] == 2


def test_continuation_factory_failure_precedes_replay() -> None:
    graph, _tensors, fake_graph, _events, _slots = _graph_fixture()

    def fail_factory():
        raise RuntimeError("materialization failed")

    fallback = graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=260,
            first_physical_slot=12 * 256 + 4,
        ),
        initial_token=99,
        block_table=None,
        block_table_factory=fail_factory,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert isinstance(fallback, ExactGreedyDecodeBurstFallback)
    assert (
        fallback.fallback_reason
        == "block_table_materialization_failure"
    )
    assert fake_graph.replay_calls == 0


def test_explicit_continuation_invalidation_forces_cold_bind() -> None:
    graph, tensors, fake_graph, events, _slots = _graph_fixture()
    factory_calls = []

    def materialize_block_table():
        factory_calls.append("called")
        return _BurstTensor(
            [[11, 12]],
            label="live_block_table",
            events=events,
            dtype="int32",
            element_size=4,
        )

    def replay_step(_ordinal):
        history_index = tensors["history_index"].values[0]
        token = 200 + history_index
        tensors["token_history"].values[history_index] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=260,
            first_physical_slot=12 * 256 + 4,
        ),
        initial_token=199,
        block_table=None,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    graph.invalidate_continuation(
        "engine_failure:RuntimeError"
    )
    graph.invalidate_continuation(
        "engine_failure:RuntimeError"
    )
    graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=264,
            first_physical_slot=12 * 256 + 8,
        ),
        initial_token=203,
        block_table=None,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert factory_calls == ["called", "called"]
    assert graph.summary()[
        "continuation_invalidation_counts"
    ] == {
        "engine_failure:RuntimeError": 1,
    }


def test_disabled_continuation_preserves_existing_event_order() -> None:
    observed = []
    for explicit_flag in (False, True):
        graph, tensors, fake_graph, events, _slots = _graph_fixture()
        events.clear()

        def replay_step(_ordinal):
            history_index = tensors["history_index"].values[0]
            token = 300 + history_index
            tensors["token_history"].values[history_index] = token
            tensors["input_token"].values = [token]
            tensors["position"].values[0] += 1
            tensors["context_length"].values[0] += 1
            tensors["slot_mapping"].values[0] += 1
            tensors["history_index"].values[0] += 1

        fake_graph.on_replay = replay_step
        kwargs = {
            "lease": _lease(),
            "initial_token": 299,
            "block_table": _BurstTensor(
                [[7, -1]],
                label="live_block_table",
                events=events,
                dtype="int32",
                element_size=4,
            ),
            "graph_generation": 4,
            "rank": 0,
            "tensor_parallel_size": 1,
        }
        if explicit_flag:
            kwargs["continuation_enabled"] = False
        result = graph.replay(**kwargs)
        summary = graph.summary()
        assert summary["continuation_attempts"] == 0
        assert summary["continuation_hits"] == 0
        assert summary["cold_binds"] == 0
        assert summary["continuation_miss_counts"] == {}
        observed.append((result.tokens, tuple(events)))

    assert observed[0] == observed[1]


def test_continuation_replay_failure_invalidates_once() -> None:
    graph, tensors, fake_graph, events, _slots = _graph_fixture()

    def materialize_block_table():
        return _BurstTensor(
            [[11, 12]],
            label="live_block_table",
            events=events,
            dtype="int32",
            element_size=4,
        )

    def replay_step(_ordinal):
        history_index = tensors["history_index"].values[0]
        token = 400 + history_index
        tensors["token_history"].values[history_index] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=260,
            first_physical_slot=12 * 256 + 4,
        ),
        initial_token=399,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    fake_graph.replay_error_at = 5
    _assert_raises(
        RuntimeError,
        "replay 5 failed",
        lambda: graph.replay(
            lease=_epoch_replay_lease(
                first_write_position=264,
                first_physical_slot=12 * 256 + 8,
            ),
            initial_token=403,
            block_table_factory=materialize_block_table,
            continuation_enabled=True,
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
        ),
    )
    graph.invalidate_continuation("replay_failure:RuntimeError")
    assert graph.summary()["quarantine_reason"] == (
        "replay_failure:RuntimeError"
    )
    assert graph.summary()[
        "continuation_invalidation_counts"
    ] == {
        "replay_failure:RuntimeError": 1,
    }


def test_pre_replay_drift_returns_typed_fallback_without_replay() -> None:
    graph, _tensors, fake_graph, _events, _slots = _graph_fixture()
    fallback = graph.replay(
        lease=_lease(),
        initial_token=19,
        block_table=_BurstTensor(
            [[7, -1]],
            label="live_block_table",
            events=[],
            dtype="int32",
            element_size=4,
        ),
        graph_generation=5,
        rank=0,
        tensor_parallel_size=1,
    )

    assert isinstance(fallback, ExactGreedyDecodeBurstFallback)
    assert fallback.fallback_reason == "graph_generation_drift"
    assert fallback.replay_count == 0
    assert fake_graph.replay_calls == 0


def test_pre_replay_capacity_and_block_boundary_fail_closed() -> None:
    graph, _tensors, fake_graph, _events, _slots = _graph_fixture()
    oversized = build_exact_greedy_decode_burst_lease(
        sequence_id=17,
        schedule_generation=9,
        graph_generation=4,
        requested_token_count=9,
        authorized_token_count=9,
        initial_completion_count=3,
        initial_sequence_length=1,
        block_table_identity=((7, 2),),
        write_block_id=7,
        write_block_generation=2,
        first_write_position=0,
        last_write_position=8,
        first_physical_slot=1792,
        last_physical_slot=1800,
        remaining_output_tokens=9,
        completion_only=True,
    )
    crossing = build_exact_greedy_decode_burst_lease(
        sequence_id=17,
        schedule_generation=9,
        graph_generation=4,
        requested_token_count=4,
        authorized_token_count=4,
        initial_completion_count=3,
        initial_sequence_length=15,
        block_table_identity=((7, 2),),
        write_block_id=7,
        write_block_generation=2,
        first_write_position=14,
        last_write_position=17,
        first_physical_slot=2046,
        last_physical_slot=2049,
        remaining_output_tokens=4,
        completion_only=True,
    )
    block_table = _BurstTensor(
        [[7, -1]],
        label="live_block_table",
        events=[],
        dtype="int32",
        element_size=4,
    )

    assert graph.replay(
        lease=oversized,
        initial_token=19,
        block_table=block_table,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    ).fallback_reason == "history_capacity_exceeded"
    assert graph.replay(
        lease=crossing,
        initial_token=19,
        block_table=block_table,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    ).fallback_reason == "physical_block_boundary_crossed"
    assert fake_graph.replay_calls == 0


def test_pre_replay_block_table_bind_failure_is_specific() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture()
    tensors["block_table"].fail_copy = True

    fallback = graph.replay(
        lease=_lease(),
        initial_token=19,
        block_table=_BurstTensor(
            [[7, -1]],
            label="live_block_table",
            events=[],
            dtype="int32",
            element_size=4,
        ),
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )

    assert isinstance(fallback, ExactGreedyDecodeBurstFallback)
    assert fallback.fallback_reason == "block_table_bind_failure"
    assert fake_graph.replay_calls == 0


def test_post_replay_failures_quarantine_and_never_retry() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture()
    fake_graph.replay_error_at = 2
    block_table = _BurstTensor(
        [[7, -1]],
        label="live_block_table",
        events=[],
        dtype="int32",
        element_size=4,
    )

    _assert_raises(
        RuntimeError,
        "replay 2 failed",
        lambda: graph.replay(
            lease=_lease(),
            initial_token=19,
            block_table=block_table,
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
        ),
    )
    assert fake_graph.replay_calls == 2
    assert graph.summary()["quarantine_reason"] == (
        "replay_failure:RuntimeError"
    )
    assert graph.summary()["graph_replays"] == 1

    fake_graph.replay_error_at = None
    fallback = graph.replay(
        lease=_lease(),
        initial_token=19,
        block_table=block_table,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert isinstance(fallback, ExactGreedyDecodeBurstFallback)
    assert fallback.fallback_reason == "quarantined"
    assert fake_graph.replay_calls == 2
    assert tensors["token_history"].tolist_calls == 0


def test_final_d2h_failure_quarantines_after_all_replays() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture()

    def replay_step(ordinal):
        tensors["token_history"].values[ordinal] = ordinal + 30
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    tensors["token_history"].fail_tolist = True

    _assert_raises(
        RuntimeError,
        "final D2H failed",
        lambda: graph.replay(
            lease=_lease(),
            initial_token=29,
            block_table=_BurstTensor(
                [[7, -1]],
                label="live_block_table",
                events=[],
                dtype="int32",
                element_size=4,
            ),
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
        ),
    )
    assert fake_graph.replay_calls == 4
    assert graph.summary()["quarantine_reason"] == (
        "final_token_d2h_failure:RuntimeError"
    )
    assert graph.summary()["final_token_d2h_calls"] == 0


def test_correctness_graph_samples_declared_logits_with_one_d2h() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture(
        correctness_trace=True
    )

    def replay_step(ordinal):
        token = 40 + ordinal
        tensors["token_history"].values[ordinal] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1
        if ordinal == 0:
            tensors["sampled_logits"].values[0] = [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        elif ordinal == 2:
            tensors["sampled_logits"].values[1] = [
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ]

    fake_graph.on_replay = replay_step
    result = graph.replay(
        lease=_lease(),
        initial_token=39,
        block_table=_BurstTensor(
            [[7, -1]],
            label="live_block_table",
            events=[],
            dtype="int32",
            element_size=4,
        ),
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )

    assert result.sampled_logits == (
        (0, (1.0, 2.0, 3.0, 4.0, 5.0)),
        (2, (5.0, 4.0, 3.0, 2.0, 1.0)),
    )
    assert result.sampled_logit_d2h_calls == 1
    assert tensors["sampled_logits"].tolist_calls == 1
    assert graph.capability()["correctness_trace"] is True
    assert graph.capability()["sampled_logit_ordinals"] == [0, 2]


def test_correctness_sampling_is_epoch_relative_across_k4_hits() -> None:
    ordinals = (0, 63, 126)
    graph, tensors, fake_graph, events, _slots = _graph_fixture(
        correctness_trace=True,
        history_capacity=128,
        sampled_logit_ordinals=ordinals,
    )
    events.clear()

    def materialize_block_table():
        return _BurstTensor(
            [[11, 12]],
            label="live_block_table",
            events=events,
            dtype="int32",
            element_size=4,
        )

    def replay_step(_ordinal):
        history_index = tensors["history_index"].values[0]
        token = 500 + history_index
        tensors["token_history"].values[history_index] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1
        if history_index in ordinals:
            row = ordinals.index(history_index)
            tensors["sampled_logits"].values[row] = [
                float(history_index + column)
                for column in range(5)
            ]

    fake_graph.on_replay = replay_step
    sampled = []
    initial_token = 499
    for burst_index in range(32):
        result = graph.replay(
            lease=_epoch_replay_lease(
                first_write_position=260 + burst_index * 4,
                first_physical_slot=(
                    12 * 256 + 4 + burst_index * 4
                ),
            ),
            initial_token=initial_token,
            block_table_factory=materialize_block_table,
            continuation_enabled=True,
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
        )
        sampled.extend(result.sampled_logits)
        initial_token = result.final_input_token

    assert tuple(ordinal for ordinal, _row in sampled) == ordinals
    assert sampled[0][1][0] == 0.0
    assert sampled[1][1][0] == 63.0
    assert sampled[2][1][0] == 126.0
    sampled_resets = [
        event
        for event in events
        if event[:3] == ("sampled_logits", "fill_", 0)
    ]
    assert len(sampled_resets) == 1
    assert graph.summary()["continuation_hits"] == 31


def test_correctness_sampling_resets_only_on_cold_bind() -> None:
    graph, tensors, fake_graph, events, _slots = _graph_fixture(
        correctness_trace=True,
        history_capacity=128,
        sampled_logit_ordinals=(0, 63, 126),
    )
    events.clear()

    def materialize_block_table():
        return _BurstTensor(
            [[11, 12]],
            label="live_block_table",
            events=events,
            dtype="int32",
            element_size=4,
        )

    def replay_step(_ordinal):
        history_index = tensors["history_index"].values[0]
        token = 600 + history_index
        tensors["token_history"].values[history_index] = token
        tensors["input_token"].values = [token]
        tensors["position"].values[0] += 1
        tensors["context_length"].values[0] += 1
        tensors["slot_mapping"].values[0] += 1
        tensors["history_index"].values[0] += 1
        if history_index == 0:
            tensors["sampled_logits"].values[0] = [1.0] * 5

    fake_graph.on_replay = replay_step
    first = graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=260,
            first_physical_slot=12 * 256 + 4,
        ),
        initial_token=599,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    tensors["sampled_logits"].values[1] = [7.0] * 5
    second = graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=264,
            first_physical_slot=12 * 256 + 8,
        ),
        initial_token=first.final_input_token,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert second.tokens == (604, 605, 606, 607)
    assert tensors["sampled_logits"].values[1] == [7.0] * 5

    graph.replay(
        lease=_epoch_replay_lease(
            first_write_position=269,
            first_physical_slot=12 * 256 + 12,
        ),
        initial_token=second.final_input_token,
        block_table_factory=materialize_block_table,
        continuation_enabled=True,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )
    assert tensors["sampled_logits"].values[1] == [0] * 5


def test_correctness_sampling_rejects_invalid_epoch_ordinals() -> None:
    for ordinals, message in (
        (
            (0, 0),
            "sampled_logit_ordinals must be strictly increasing",
        ),
        (
            (2, 1),
            "sampled_logit_ordinals must be strictly increasing",
        ),
        (
            (-1,),
            (
                "sampled_logit_ordinals must contain "
                "non-negative integers"
            ),
        ),
        (
            (128,),
            (
                "sampled_logit_ordinals must be below "
                "history capacity"
            ),
        ),
    ):
        _assert_raises(
            ValueError,
            message,
            lambda ordinals=ordinals: _graph_fixture(
                correctness_trace=True,
                history_capacity=128,
                sampled_logit_ordinals=ordinals,
            ),
        )


def test_capture_rejects_any_live_kv_mutation() -> None:
    _assert_raises(
        RuntimeError,
        "exact burst capture mutated live KV",
        lambda: _graph_fixture(live_kv_changed=True),
    )


def test_result_construction_failure_quarantines_original_error() -> None:
    graph, tensors, fake_graph, _events, _slots = _graph_fixture()

    def replay_step(ordinal):
        tensors["token_history"].values[ordinal] = ordinal + 50
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    original = module.ExactGreedyDecodeBurstResult

    def fail_result(**_kwargs):
        raise LookupError("result construction failed")

    module.ExactGreedyDecodeBurstResult = fail_result
    try:
        _assert_raises(
            LookupError,
            "result construction failed",
            lambda: graph.replay(
                lease=_lease(),
                initial_token=49,
                block_table=_BurstTensor(
                    [[7, -1]],
                    label="live_block_table",
                    events=[],
                    dtype="int32",
                    element_size=4,
                ),
                graph_generation=4,
                rank=0,
                tensor_parallel_size=1,
            ),
        )
    finally:
        module.ExactGreedyDecodeBurstResult = original
    assert graph.summary()["quarantine_reason"] == (
        "result_construction_failure:LookupError"
    )
    assert fake_graph.replay_calls == 4


def test_split_phase_replay_enqueues_prefix_after_four_and_suffix_after_eight():
    graph, tensors, fake_graph, _events, _slots = _graph_fixture(
        history_capacity=8
    )

    def replay_step(ordinal):
        tensors["token_history"].values[ordinal] = ordinal + 50
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    backend = _SplitBackend(fake_graph)
    result = graph.replay_split_phase(
        lease=_k8_lease(),
        initial_token=49,
        block_table=_BurstTensor(
            [[7, -1]],
            label="live_block_table",
            events=[],
            dtype="int32",
            element_size=4,
        ),
        mailbox_backend=backend,
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )

    assert fake_graph.replay_calls == 8
    assert backend.calls == [
        ("begin", 0),
        ("prefix", 4, (50, 51, 52, 53), 7),
        ("suffix", 8, (54, 55, 56, 57), 7),
    ]
    assert result.replay_count == 8
    assert result.prefix.wait_tokens() == (50, 51, 52, 53)
    assert result.suffix.wait_tokens() == (54, 55, 56, 57)
    assert tensors["token_history"].tolist_calls == 0


def test_split_phase_correctness_trace_returns_only_sampled_logits():
    graph, tensors, fake_graph, _events, _slots = _graph_fixture(
        correctness_trace=True,
        history_capacity=8,
        sampled_logit_ordinals=(0, 7),
    )

    def replay_step(ordinal):
        tensors["token_history"].values[ordinal] = ordinal + 70
        tensors["history_index"].values[0] += 1
        if ordinal == 0:
            tensors["sampled_logits"].values[0] = [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        if ordinal == 7:
            tensors["sampled_logits"].values[1] = [
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ]

    fake_graph.on_replay = replay_step
    result = graph.replay_split_phase(
        lease=_k8_lease(),
        initial_token=69,
        block_table=_BurstTensor(
            [[7, -1]],
            label="live_block_table",
            events=[],
            dtype="int32",
            element_size=4,
        ),
        mailbox_backend=_SplitBackend(fake_graph),
        graph_generation=4,
        rank=0,
        tensor_parallel_size=1,
    )

    assert result.sampled_logits == (
        (0, (1.0, 2.0, 3.0, 4.0, 5.0)),
        (7, (5.0, 4.0, 3.0, 2.0, 1.0)),
    )
    assert result.sampled_logit_d2h_calls == 1
    assert tensors["sampled_logits"].tolist_calls == 1
    assert tensors["token_history"].tolist_calls == 0


def test_split_phase_copy_failure_aborts_and_quarantines():
    graph, tensors, fake_graph, _events, _slots = _graph_fixture(
        history_capacity=8
    )

    def replay_step(ordinal):
        tensors["token_history"].values[ordinal] = ordinal + 60
        tensors["history_index"].values[0] += 1

    fake_graph.on_replay = replay_step
    backend = _SplitBackend(fake_graph, fail_phase="prefix")
    _assert_raises(
        RuntimeError,
        "prefix copy failed",
        lambda: graph.replay_split_phase(
            lease=_k8_lease(),
            initial_token=59,
            block_table=_BurstTensor(
                [[7, -1]],
                label="live_block_table",
                events=[],
                dtype="int32",
                element_size=4,
            ),
            mailbox_backend=backend,
            graph_generation=4,
            rank=0,
            tensor_parallel_size=1,
        ),
    )
    assert fake_graph.replay_calls == 4
    assert backend.aborted == [7]
    assert graph.summary()["quarantine_reason"] == (
        "split_phase_failure:RuntimeError"
    )


def main() -> None:
    test_ragged_width_selector_caps_output_tail_at_k4()
    test_ragged_width_selector_caps_block_edge_at_k4()
    test_ragged_width_selector_is_strictly_opt_in()
    test_policy_clips_to_budget_and_current_block()
    test_boundary_width_one_falls_back_before_replay()
    test_gate_only_width_one_is_explicit_and_never_implicit()
    test_fallback_reasons_have_stable_precedence()
    test_invalid_policy_inputs_fail_closed()
    test_lease_identity_is_canonical_and_result_is_exact()
    test_continuation_requires_an_exact_receipt_match()
    test_continuation_rejects_invalid_scalar_contracts()
    test_correctness_trace_is_bounded_and_ordered()
    test_stats_track_benefit_cost_and_terminal_state()
    test_stats_track_split_phase_benefit_cost_and_failures()
    test_stats_track_continuation_benefit_and_cost()
    test_contract_is_model_agnostic_and_supports_second_caller()
    test_complete_step_capture_orders_body_and_uses_private_scratch()
    test_replay_runs_exact_count_then_one_token_d2h()
    test_replay_cold_binds_then_continues_resident_state()
    test_continuation_factory_failure_precedes_replay()
    test_explicit_continuation_invalidation_forces_cold_bind()
    test_disabled_continuation_preserves_existing_event_order()
    test_continuation_replay_failure_invalidates_once()
    test_pre_replay_drift_returns_typed_fallback_without_replay()
    test_pre_replay_capacity_and_block_boundary_fail_closed()
    test_pre_replay_block_table_bind_failure_is_specific()
    test_post_replay_failures_quarantine_and_never_retry()
    test_final_d2h_failure_quarantines_after_all_replays()
    test_correctness_graph_samples_declared_logits_with_one_d2h()
    test_correctness_sampling_is_epoch_relative_across_k4_hits()
    test_correctness_sampling_resets_only_on_cold_bind()
    test_correctness_sampling_rejects_invalid_epoch_ordinals()
    test_capture_rejects_any_live_kv_mutation()
    test_result_construction_failure_quarantines_original_error()
    test_split_phase_replay_enqueues_prefix_after_four_and_suffix_after_eight()
    test_split_phase_copy_failure_aborts_and_quarantines()
    print("exact greedy decode burst tests passed")


if __name__ == "__main__":
    main()
