from contextlib import contextmanager
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

from tinyvllm.layers.linear import RowParallelLinear
import tinyvllm.layers.linear as linear_module


@contextmanager
def _tp_layout(rank: int, world_size: int):
    original_rank = torch.distributed.get_rank
    original_world_size = torch.distributed.get_world_size
    torch.distributed.get_rank = lambda: rank
    torch.distributed.get_world_size = lambda: world_size
    try:
        yield
    finally:
        torch.distributed.get_rank = original_rank
        torch.distributed.get_world_size = original_world_size


def test_qwen35_output_projection_shards_checkpoint_columns_and_sums():
    world_size = 4
    input_size = 16
    output_size = 8
    full_input = (
        torch.arange(
            3 * input_size,
            dtype=torch.float32,
        )
        .reshape(3, input_size)
        .div(11)
    )
    full_weight = (
        torch.arange(
            output_size * input_size,
            dtype=torch.float32,
        )
        .reshape(output_size, input_size)
        .sub(17)
        .div(13)
    )

    partial_outputs = []
    for rank in range(world_size):
        with _tp_layout(rank, world_size):
            projection = RowParallelLinear(
                input_size,
                output_size,
                bias=False,
            )
        projection.weight.weight_loader(
            projection.weight,
            full_weight,
        )
        local_width = input_size // world_size
        local_input = full_input.narrow(
            1,
            rank * local_width,
            local_width,
        )
        expected_weight = full_weight.narrow(
            1,
            rank * local_width,
            local_width,
        )
        torch.testing.assert_close(
            projection.weight,
            expected_weight,
        )
        partial_outputs.append(
            F.linear(local_input, projection.weight)
        )

    torch.testing.assert_close(
        torch.stack(partial_outputs).sum(dim=0),
        F.linear(full_input, full_weight),
        rtol=1e-5,
        atol=1e-5,
    )


def test_qwen35_output_projection_can_accumulate_partials_in_fp32(
    monkeypatch,
):
    full_input = torch.tensor(
        [[
            -3.046875,
            -1.5,
            -1.3046875,
            -3.21875,
            -0.2001953125,
            -1.21875,
            -1.9609375,
            -3.21875,
            -1.421875,
            0.609375,
            -1.5546875,
            -0.50390625,
            -0.4453125,
            3.375,
            0.45703125,
            0.93359375,
        ]],
        dtype=torch.bfloat16,
    )
    full_weight = torch.tensor(
        [[
            -1.390625,
            -2.328125,
            1.3984375,
            0.3984375,
            1.734375,
            0.48828125,
            -1.328125,
            1.6171875,
            2.203125,
            -0.3515625,
            -4.5,
            -2.890625,
            0.1220703125,
            -1.234375,
            -1.59375,
            -0.263671875,
        ]],
        dtype=torch.bfloat16,
    )
    with _tp_layout(rank=0, world_size=4):
        projection = RowParallelLinear(
            input_size=16,
            output_size=1,
            bias=False,
            accumulation_dtype=torch.float32,
        ).to(dtype=torch.bfloat16)
    projection.weight.weight_loader(
        projection.weight,
        full_weight,
    )
    assert projection.accumulation_weight.dtype is torch.float32
    torch.testing.assert_close(
        projection.accumulation_weight,
        full_weight[:, :4].float(),
        rtol=0.0,
        atol=0.0,
    )
    other_partials = [
        F.linear(
            full_input[:, start:start + 4].float(),
            full_weight[:, start:start + 4].float(),
        )
        for start in (4, 8, 12)
    ]

    def reduce_partials(name, output, collective):
        assert name == "row_parallel_all_reduce"
        assert collective is torch.distributed.all_reduce
        assert output.dtype is torch.float32
        output.add_(torch.stack(other_partials).sum(dim=0))

    monkeypatch.setattr(
        linear_module,
        "profile_collective",
        reduce_partials,
    )
    actual = projection(full_input[:, :4])
    expected = F.linear(full_input, full_weight)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_qwen35_output_projection_preserves_dense_prefill_math(
    monkeypatch,
):
    world_size = 4
    input_size = 16
    output_size = 8
    full_input = (
        torch.arange(
            3 * input_size,
            dtype=torch.bfloat16,
        )
        .reshape(3, input_size)
        .div(11)
    )
    full_weight = (
        torch.arange(
            output_size * input_size,
            dtype=torch.bfloat16,
        )
        .reshape(output_size, input_size)
        .sub(17)
        .div(13)
    )
    with _tp_layout(rank=0, world_size=world_size):
        projection = RowParallelLinear(
            input_size,
            output_size,
            bias=False,
            accumulation_dtype=torch.float32,
            preserve_dense_prefill=True,
        ).to(dtype=torch.bfloat16)
    projection.weight.weight_loader(
        projection.weight,
        full_weight,
    )
    assert projection.prefill_weight.dtype is torch.bfloat16
    torch.testing.assert_close(
        projection.prefill_weight,
        full_weight,
        rtol=0.0,
        atol=0.0,
    )
    local_width = input_size // world_size
    input_shards = [
        full_input.narrow(
            1,
            rank * local_width,
            local_width,
        )
        for rank in range(world_size)
    ]

    def gather_inputs(outputs, tensor):
        assert tensor is input_shards[0]
        for destination, source in zip(outputs, input_shards):
            destination.copy_(source)

    def profile_gather(name, tensor, collective):
        assert name == "row_parallel_prefill_all_gather"
        assert callable(collective)
        return collective(tensor)

    monkeypatch.setattr(
        linear_module,
        "profile_collective",
        profile_gather,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        gather_inputs,
    )
    actual = projection.forward_prefill(input_shards[0])
    expected = F.linear(full_input, full_weight)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
