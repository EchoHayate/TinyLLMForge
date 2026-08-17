import importlib.util
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.layers"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

_load_module(
    "tinyvllm.layers.quantization",
    "tinyvllm/layers/quantization.py",
)
linear = _load_module(
    "tinyvllm.layers.linear",
    "tinyvllm/layers/linear.py",
)
HeadPairedColumnParallelLinear = linear.HeadPairedColumnParallelLinear


class _DistLayout:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.old_get_rank = linear.dist.get_rank
        self.old_get_world_size = linear.dist.get_world_size

    def __enter__(self):
        linear.dist.get_rank = lambda: self.rank
        linear.dist.get_world_size = lambda: self.world_size

    def __exit__(self, exc_type, exc_value, traceback):
        linear.dist.get_rank = self.old_get_rank
        linear.dist.get_world_size = self.old_get_world_size


def _head_paired_source(
    num_heads: int,
    head_dim: int,
    input_size: int,
) -> torch.Tensor:
    rows = []
    for head in range(num_heads):
        for half in range(2):
            for lane in range(head_dim):
                code = 1000 * head + 100 * half + lane
                rows.append(
                    torch.arange(input_size, dtype=torch.float32) + code * 10
                )
    return torch.stack(rows)


def _expected_contiguous_rank(
    source: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    local_rows = source.shape[0] // world_size
    return source.narrow(0, rank * local_rows, local_rows)


def _incorrect_global_segment_rank(
    source: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    by_head = source.view(num_heads, 2, head_dim, source.shape[1])
    all_query = by_head[:, 0].reshape(num_heads * head_dim, source.shape[1])
    all_gate = by_head[:, 1].reshape(num_heads * head_dim, source.shape[1])
    local_rows = num_heads * head_dim // world_size
    return torch.cat(
        (
            all_query.narrow(0, rank * local_rows, local_rows),
            all_gate.narrow(0, rank * local_rows, local_rows),
        ),
        dim=0,
    )


def test_tp_1_2_4_loads_contiguous_complete_head_pairs() -> None:
    input_size = 3
    head_dim = 2
    for num_heads, world_size in ((3, 1), (4, 2), (8, 4)):
        source = _head_paired_source(num_heads, head_dim, input_size)
        for rank in range(world_size):
            with _DistLayout(rank, world_size):
                layer = HeadPairedColumnParallelLinear(
                    input_size=input_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    bias=False,
                )
            layer.weight.weight_loader(layer.weight, source)
            expected = _expected_contiguous_rank(source, rank, world_size)
            torch.testing.assert_close(layer.weight, expected)
            assert layer.local_num_heads == num_heads // world_size
            assert layer.weight.shape[0] == (
                layer.local_num_heads * 2 * head_dim
            )


def test_official_head_pair_rows_differ_from_global_segment_sharding() -> None:
    input_size = 2
    head_dim = 2
    for num_heads, world_size in ((4, 2), (8, 4)):
        source = _head_paired_source(num_heads, head_dim, input_size)
        for rank in range(world_size):
            official = _expected_contiguous_rank(source, rank, world_size)
            incorrect = _incorrect_global_segment_rank(
                source,
                num_heads,
                head_dim,
                rank,
                world_size,
            )
            assert not torch.equal(official, incorrect)


def test_split_query_gate_chunks_inside_each_local_head() -> None:
    with _DistLayout(rank=1, world_size=2):
        layer = HeadPairedColumnParallelLinear(
            input_size=3,
            num_heads=4,
            head_dim=2,
            bias=False,
        )
    projected = torch.tensor(
        [
            [
                2000.0,
                2001.0,
                2100.0,
                2101.0,
                3000.0,
                3001.0,
                3100.0,
                3101.0,
            ],
            [
                4000.0,
                4001.0,
                4100.0,
                4101.0,
                5000.0,
                5001.0,
                5100.0,
                5101.0,
            ],
        ]
    )
    query, gate = layer.split_query_gate(projected)
    expected_query = torch.tensor(
        [
            [[2000.0, 2001.0], [3000.0, 3001.0]],
            [[4000.0, 4001.0], [5000.0, 5001.0]],
        ]
    )
    expected_gate = torch.tensor(
        [
            [2100.0, 2101.0, 3100.0, 3101.0],
            [4100.0, 4101.0, 5100.0, 5101.0],
        ]
    )
    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(gate, expected_gate)


def test_split_query_gate_accepts_noncontiguous_projection_output() -> None:
    with _DistLayout(rank=0, world_size=2):
        layer = HeadPairedColumnParallelLinear(
            input_size=3,
            num_heads=4,
            head_dim=2,
            bias=False,
        )
    projected = torch.arange(16, dtype=torch.float32).reshape(8, 2).t()
    assert not projected.is_contiguous()
    query, gate = layer.split_query_gate(projected)
    paired = projected.reshape(2, 2, 4)
    torch.testing.assert_close(query, paired[..., :2])
    torch.testing.assert_close(gate, paired[..., 2:].reshape(2, 4))


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_constructor_and_split_fail_closed() -> None:
    for name, value in (
        ("num_heads", True),
        ("num_heads", 0),
        ("num_heads", 2.5),
        ("head_dim", True),
        ("head_dim", 0),
        ("head_dim", 2.5),
    ):
        kwargs = dict(input_size=3, num_heads=4, head_dim=2, bias=False)
        kwargs[name] = value
        with _DistLayout(rank=0, world_size=2):
            _expect_value_error(
                lambda kwargs=kwargs: HeadPairedColumnParallelLinear(**kwargs),
                name,
            )

    with _DistLayout(rank=0, world_size=4):
        _expect_value_error(
            lambda: HeadPairedColumnParallelLinear(
                input_size=3,
                num_heads=6,
                head_dim=2,
                bias=False,
            ),
            "divisible",
        )

    with _DistLayout(rank=0, world_size=2):
        layer = HeadPairedColumnParallelLinear(
            input_size=3,
            num_heads=4,
            head_dim=2,
            bias=False,
        )
    for projected, message in (
        (torch.ones(2, 2, 4), "rank two"),
        (torch.ones(2, 7), "feature"),
        (torch.ones(2, 8, dtype=torch.int64), "floating point"),
    ):
        _expect_value_error(
            lambda projected=projected: layer.split_query_gate(projected),
            message,
        )


def main() -> None:
    test_tp_1_2_4_loads_contiguous_complete_head_pairs()
    test_official_head_pair_rows_differ_from_global_segment_sharding()
    test_split_query_gate_chunks_inside_each_local_head()
    test_split_query_gate_accepts_noncontiguous_projection_output()
    test_constructor_and_split_fail_closed()
    print("qwen35 head paired projection tests passed")


if __name__ == "__main__":
    main()
