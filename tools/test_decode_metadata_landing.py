"""Dependency-light tests for replay-aware decode metadata landing.

Run with:
    python3 tools/test_decode_metadata_landing.py
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "decode_metadata_landing.py"
)
SPEC = importlib.util.spec_from_file_location(
    "decode_metadata_landing_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
build_decode_metadata_plan = module.build_decode_metadata_plan
ReplayAwareDecodeMetadataArena = (
    module.ReplayAwareDecodeMetadataArena
)


@dataclass
class FakeSequence:
    last_token: int
    token_count: int
    block_table: list[int]
    block_size: int

    def __len__(self) -> int:
        return self.token_count

    @property
    def last_block_num_tokens(self) -> int:
        remainder = self.token_count % self.block_size
        return remainder or self.block_size


class FakeHostView:
    def __init__(self, tensor, start, length, shape=None):
        self.tensor = tensor
        self.start = start
        self.length = length
        self.shape = shape or (length,)

    def view(self, *shape):
        assert self.length == _numel(shape)
        return FakeHostView(
            self.tensor,
            self.start,
            self.length,
            tuple(shape),
        )

    def tolist(self):
        values = self.tensor.values[
            self.start:self.start + self.length
        ]
        if len(self.shape) == 2:
            rows, columns = self.shape
            return [
                values[
                    row * columns:(row + 1) * columns
                ]
                for row in range(rows)
            ]
        return list(values)


class FakeHostTensor:
    def __init__(self, size, dtype):
        self.values = [None] * size
        self.dtype = dtype

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 8 if self.dtype == "int64" else 4

    def __setitem__(self, index, value):
        self.values[index] = value

    def __getitem__(self, index):
        assert isinstance(index, slice)
        start = index.start or 0
        stop = index.stop
        return FakeHostView(
            self,
            start,
            stop - start,
        )


def _numel(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class FakeDestinationView:
    def __init__(self, tensor, index):
        self.tensor = tensor
        self.index = index

    def copy_(self, source, non_blocking=False):
        self.tensor.writes.append(
            (
                self.index,
                source.tolist(),
                non_blocking,
            )
        )
        return self


class FakeDeviceTensor:
    def __init__(self, shape, element_size):
        self.shape = tuple(shape)
        self._element_size = element_size
        self.writes = []
        self.zero_calls = 0

    def size(self, dimension=None):
        if dimension is None:
            return self.shape
        return self.shape[dimension]

    def numel(self):
        return _numel(self.shape)

    def element_size(self):
        return self._element_size

    def __getitem__(self, index):
        return FakeDestinationView(self, index)

    def zero_(self):
        self.zero_calls += 1
        return self


class FakeTorch:
    int64 = "int64"
    int32 = "int32"

    def __init__(self):
        self.empty_calls = []
        self.tensor_calls = 0

    def empty(
        self,
        size,
        *,
        dtype,
        device,
        pin_memory,
    ):
        self.empty_calls.append(
            (size, dtype, device, pin_memory)
        )
        return FakeHostTensor(size, dtype)

    def tensor(self, *_args, **_kwargs):
        self.tensor_calls += 1
        raise AssertionError(
            "optimized staging must not create pageable tensors"
        )


def make_graph_vars(block_table_width=8):
    return {
        "input_ids": FakeDeviceTensor((1,), 8),
        "positions": FakeDeviceTensor((1,), 8),
        "slot_mapping": FakeDeviceTensor((1,), 4),
        "context_lens": FakeDeviceTensor((1,), 4),
        "block_tables": FakeDeviceTensor(
            (1, block_table_width),
            4,
        ),
        "outputs": FakeDeviceTensor((1, 32), 2),
    }


def test_build_decode_metadata_plan_preserves_readable_rows():
    sequence = FakeSequence(
        last_token=17,
        token_count=513,
        block_table=[4, 8, 15],
        block_size=256,
    )

    plan = build_decode_metadata_plan([sequence], 256)

    assert plan.input_ids == (17,)
    assert plan.positions == (512,)
    assert plan.slot_mapping == (15 * 256,)
    assert plan.context_lens == (513,)
    assert plan.block_table_rows == ((4, 8, 15),)
    assert plan.active_batch_size == 1
    assert plan.readable_page_table_width == 3


def test_build_decode_metadata_plan_pads_rows_deterministically():
    first = FakeSequence(
        last_token=3,
        token_count=257,
        block_table=[1, 2],
        block_size=256,
    )
    second = FakeSequence(
        last_token=5,
        token_count=1,
        block_table=[7],
        block_size=256,
    )

    plan = build_decode_metadata_plan([first, second], 256)

    assert plan.block_table_rows == ((1, 2), (7, -1))
    assert plan.active_batch_size == 2
    assert plan.readable_page_table_width == 2


def test_arena_lands_only_readable_batch_one_metadata():
    torch_module = FakeTorch()
    arena = ReplayAwareDecodeMetadataArena(torch_module)
    plan = build_decode_metadata_plan(
        [
            FakeSequence(
                last_token=17,
                token_count=513,
                block_table=[4, 8, 15],
                block_size=256,
            )
        ],
        256,
    )
    graph_vars = make_graph_vars()

    result = arena.land(
        plan,
        graph_vars,
        graph_batch_size=1,
    )

    assert result.optimized is True
    assert result.fallback_reason is None
    assert graph_vars["input_ids"].writes == [
        (slice(0, 1), [17], True)
    ]
    assert graph_vars["positions"].writes == [
        (slice(0, 1), [512], True)
    ]
    assert graph_vars["slot_mapping"].writes == [
        (slice(0, 1), [15 * 256], True)
    ]
    assert graph_vars["context_lens"].writes == [
        (slice(0, 1), [513], True)
    ]
    assert graph_vars["block_tables"].writes == [
        (
            (slice(0, 1), slice(0, 3)),
            [[4, 8, 15]],
            True,
        )
    ]
    assert all(
        tensor.zero_calls == 0
        for tensor in graph_vars.values()
    )
    assert torch_module.tensor_calls == 0
    assert result.input_ids is not None
    assert result.positions is not None
    assert result.slot_mapping is not None
    assert result.context_lens is not None
    assert result.block_tables is not None


def test_arena_reuses_capacity_and_accounts_exact_cost():
    torch_module = FakeTorch()
    arena = ReplayAwareDecodeMetadataArena(torch_module)
    graph_vars = make_graph_vars()
    short = build_decode_metadata_plan(
        [FakeSequence(17, 513, [4, 8, 15], 256)],
        256,
    )
    long = build_decode_metadata_plan(
        [FakeSequence(19, 1025, [4, 8, 15, 16, 23], 256)],
        256,
    )

    arena.land(short, graph_vars, graph_batch_size=1)
    allocation_count = len(torch_module.empty_calls)
    arena.land(long, graph_vars, graph_batch_size=1)

    assert len(torch_module.empty_calls) == allocation_count
    summary = arena.summary()
    assert summary["optimized_steps"] == 2
    assert summary["fallback_counts"] == {}
    assert summary["allocation_count"] == 5
    assert summary["growth_count"] == 5
    assert summary["staged_h2d_bytes"] == 80
    assert summary[
        "avoided_temporary_cuda_tensors"
    ] == 10
    assert summary["avoided_blanket_zero_bytes"] == 112
    assert summary["current_pinned_capacity_bytes"] == 1792
    assert summary["peak_pinned_capacity_bytes"] == 1792


def test_arena_falls_back_before_writing_for_inexact_graph():
    torch_module = FakeTorch()
    arena = ReplayAwareDecodeMetadataArena(torch_module)
    plan = build_decode_metadata_plan(
        [FakeSequence(17, 513, [4, 8, 15], 256)],
        256,
    )
    graph_vars = make_graph_vars()

    result = arena.land(
        plan,
        graph_vars,
        graph_batch_size=2,
    )

    assert result.optimized is False
    assert result.fallback_reason == "graph_batch_size_mismatch"
    assert all(
        tensor.writes == []
        for tensor in graph_vars.values()
    )
    assert arena.summary()["fallback_counts"] == {
        "graph_batch_size_mismatch": 1
    }


def test_arena_falls_back_before_writing_for_small_block_table():
    torch_module = FakeTorch()
    arena = ReplayAwareDecodeMetadataArena(torch_module)
    plan = build_decode_metadata_plan(
        [FakeSequence(19, 1025, [4, 8, 15, 16, 23], 256)],
        256,
    )
    graph_vars = make_graph_vars(block_table_width=4)

    result = arena.land(
        plan,
        graph_vars,
        graph_batch_size=1,
    )

    assert result.optimized is False
    assert result.fallback_reason == "graph_capacity_mismatch"
    assert all(
        tensor.writes == []
        for tensor in graph_vars.values()
    )


def main() -> None:
    test_build_decode_metadata_plan_preserves_readable_rows()
    test_build_decode_metadata_plan_pads_rows_deterministically()
    test_arena_lands_only_readable_batch_one_metadata()
    test_arena_reuses_capacity_and_accounts_exact_cost()
    test_arena_falls_back_before_writing_for_inexact_graph()
    test_arena_falls_back_before_writing_for_small_block_table()
    print("decode metadata landing tests passed")


if __name__ == "__main__":
    main()
