import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch.nn import functional as F

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
SegmentedColumnParallelLinear = linear.SegmentedColumnParallelLinear


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


def _row_coded_weight(output_sizes: tuple[int, ...], input_size: int):
    rows = []
    global_offset = 0
    for segment_id, segment_size in enumerate(output_sizes):
        for segment_row in range(segment_size):
            row_code = 1000 * segment_id + global_offset + segment_row
            rows.append(
                torch.arange(input_size, dtype=torch.float32) + row_code * 10
            )
        global_offset += segment_size
    return torch.stack(rows)


def _expected_local(
    source: torch.Tensor,
    output_sizes: tuple[int, ...],
    rank: int,
    world_size: int,
) -> torch.Tensor:
    shards = []
    global_offset = 0
    for segment_size in output_sizes:
        local_size = segment_size // world_size
        segment = source.narrow(0, global_offset, segment_size)
        shards.append(segment.narrow(0, rank * local_size, local_size))
        global_offset += segment_size
    return torch.cat(shards, dim=0)


def test_fused_weight_loading_tp_1_2_4() -> None:
    input_size = 3
    cases = (
        ((3, 5, 7), 1),
        ((4, 6, 8), 2),
        ((8, 12, 16), 4),
    )
    for output_sizes, world_size in cases:
        source = _row_coded_weight(output_sizes, input_size)
        for rank in range(world_size):
            with _DistLayout(rank, world_size):
                layer = SegmentedColumnParallelLinear(
                    input_size, output_sizes, bias=False
                )
            layer.weight.weight_loader(layer.weight, source)
            expected = _expected_local(
                source, output_sizes, rank, world_size
            )
            torch.testing.assert_close(layer.weight, expected)
            assert layer.weight.shape == expected.shape


def test_fused_bias_and_inherited_forward() -> None:
    output_sizes = (4, 6, 8)
    input_size = 3
    world_size = 2
    rank = 1
    source_weight = _row_coded_weight(output_sizes, input_size)
    source_bias = torch.arange(sum(output_sizes), dtype=torch.float32) + 0.5
    with _DistLayout(rank, world_size):
        layer = SegmentedColumnParallelLinear(
            input_size, output_sizes, bias=True
        )
    layer.weight.weight_loader(layer.weight, source_weight)
    layer.bias.weight_loader(layer.bias, source_bias)
    expected_weight = _expected_local(
        source_weight, output_sizes, rank, world_size
    )
    expected_bias = _expected_local(
        source_bias, output_sizes, rank, world_size
    )
    input_tensor = torch.tensor(
        [[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]]
    )
    torch.testing.assert_close(layer.weight, expected_weight)
    torch.testing.assert_close(layer.bias, expected_bias)
    torch.testing.assert_close(
        layer(input_tensor),
        F.linear(input_tensor, expected_weight, expected_bias),
    )


def test_separate_segment_loading_matches_fused_source() -> None:
    output_sizes = (8, 12, 16)
    input_size = 3
    world_size = 4
    source_weight = _row_coded_weight(output_sizes, input_size)
    source_bias = torch.arange(sum(output_sizes), dtype=torch.float32) + 0.25
    global_offsets = (
        0,
        output_sizes[0],
        output_sizes[0] + output_sizes[1],
    )
    for rank in range(world_size):
        with _DistLayout(rank, world_size):
            fused_layer = SegmentedColumnParallelLinear(
                input_size, output_sizes, bias=True
            )
            separate_layer = SegmentedColumnParallelLinear(
                input_size, output_sizes, bias=True
            )
        fused_layer.weight.weight_loader(fused_layer.weight, source_weight)
        fused_layer.bias.weight_loader(fused_layer.bias, source_bias)
        with torch.no_grad():
            separate_layer.weight.fill_(-999)
            separate_layer.bias.fill_(-999)
        for segment_id in (2, 0, 1):
            segment_size = output_sizes[segment_id]
            offset = global_offsets[segment_id]
            separate_layer.weight.weight_loader(
                separate_layer.weight,
                source_weight.narrow(0, offset, segment_size),
                segment_id,
            )
            separate_layer.bias.weight_loader(
                separate_layer.bias,
                source_bias.narrow(0, offset, segment_size),
                segment_id,
            )
        torch.testing.assert_close(separate_layer.weight, fused_layer.weight)
        torch.testing.assert_close(separate_layer.bias, fused_layer.bias)


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def _make_layer(
    output_sizes=(4, 6, 8),
    *,
    input_size=3,
    rank=0,
    world_size=2,
    bias=False,
):
    with _DistLayout(rank, world_size):
        return SegmentedColumnParallelLinear(
            input_size, output_sizes, bias=bias
        )


def test_constructor_rejects_invalid_segment_sizes() -> None:
    invalid_cases = (
        ((), "at least one"),
        ((4, True), "positive integer"),
        ((4, 2.5), "positive integer"),
        ((4, 0), "positive integer"),
        ((4, -2), "positive integer"),
    )
    for output_sizes, message in invalid_cases:
        _expect_value_error(
            lambda output_sizes=output_sizes: _make_layer(output_sizes),
            message,
        )
    _expect_value_error(
        lambda: _make_layer((4, 5), world_size=2),
        "divisible",
    )


def test_loader_rejects_invalid_segment_ids_and_shapes() -> None:
    layer = _make_layer()
    source = _row_coded_weight((4, 6, 8), 3)
    for segment_id in (True, -1, 3, 1.5):
        _expect_value_error(
            lambda segment_id=segment_id: layer.weight.weight_loader(
                layer.weight, source[:4], segment_id
            ),
            "valid segment",
        )
    _expect_value_error(
        lambda: layer.weight.weight_loader(layer.weight, source[:-1]),
        "fused loaded output",
    )
    _expect_value_error(
        lambda: layer.weight.weight_loader(
            layer.weight, torch.ones(5, 3), 1
        ),
        "segment 1 loaded output",
    )
    _expect_value_error(
        lambda: layer.weight.weight_loader(
            layer.weight, torch.ones(sum((4, 6, 8)), 4)
        ),
        "input dimension",
    )
    _expect_value_error(
        lambda: layer.weight.weight_loader(
            layer.weight, source.unsqueeze(0)
        ),
        "rank",
    )


def test_loader_rejects_dtype_and_device_mismatch() -> None:
    layer = _make_layer()
    source = _row_coded_weight((4, 6, 8), 3)
    _expect_value_error(
        lambda: layer.weight.weight_loader(
            layer.weight, source.to(torch.float64)
        ),
        "dtype",
    )
    meta_source = torch.empty(source.shape, device="meta")
    _expect_value_error(
        lambda: layer.weight.weight_loader(layer.weight, meta_source),
        "device",
    )


def test_failed_fused_validation_does_not_mutate_destination() -> None:
    layer = _make_layer(bias=True)
    with torch.no_grad():
        layer.weight.fill_(123.0)
        layer.bias.fill_(456.0)
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()
    source = _row_coded_weight((4, 6, 8), 3)

    _expect_value_error(
        lambda: layer.weight.weight_loader(layer.weight, source[:-1]),
        "fused loaded output",
    )
    _expect_value_error(
        lambda: layer.bias.weight_loader(
            layer.bias,
            torch.arange(sum((4, 6, 8)) - 1, dtype=torch.float32),
        ),
        "fused loaded output",
    )
    torch.testing.assert_close(layer.weight, original_weight)
    torch.testing.assert_close(layer.bias, original_bias)


def main() -> None:
    test_fused_weight_loading_tp_1_2_4()
    test_fused_bias_and_inherited_forward()
    test_separate_segment_loading_matches_fused_source()
    test_constructor_rejects_invalid_segment_sizes()
    test_loader_rejects_invalid_segment_ids_and_shapes()
    test_loader_rejects_dtype_and_device_mismatch()
    test_failed_fused_validation_does_not_mutate_destination()
    print("segmented column parallel linear tests passed")


if __name__ == "__main__":
    main()
