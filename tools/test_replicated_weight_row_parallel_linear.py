from contextlib import contextmanager

import torch
import torch.nn.functional as F

from tinyvllm.layers import linear as linear_module


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


def test_gathers_local_inputs_and_executes_one_full_gemm():
    projection_type = getattr(
        linear_module,
        "ReplicatedWeightRowParallelLinear",
        None,
    )
    assert projection_type is not None

    local_inputs = [
        (
            torch.arange(3 * 2, dtype=torch.float32)
            .reshape(3, 2)
            .add(rank * 10)
            .to(torch.bfloat16)
        )
        for rank in range(4)
    ]
    full_weight = (
        torch.arange(5 * 8, dtype=torch.float32)
        .reshape(5, 8)
        .sub(10)
        .div(8)
        .to(torch.bfloat16)
    )
    original_all_gather = torch.distributed.all_gather
    original_all_reduce = torch.distributed.all_reduce
    calls = []

    def all_gather(outputs, tensor):
        calls.append(tensor.detach().clone())
        for output, value in zip(outputs, local_inputs):
            output.copy_(value)

    def reject_all_reduce(*args, **kwargs):
        raise AssertionError("forward must not all-reduce partial outputs")

    torch.distributed.all_gather = all_gather
    torch.distributed.all_reduce = reject_all_reduce
    try:
        with _tp_layout(rank=2, world_size=4):
            projection = projection_type(
                input_size=8,
                output_size=5,
                bias=False,
            ).to(dtype=torch.bfloat16)
        with torch.no_grad():
            projection.weight.copy_(full_weight)

        output = projection(local_inputs[2])
    finally:
        torch.distributed.all_gather = original_all_gather
        torch.distributed.all_reduce = original_all_reduce

    expected = F.linear(
        torch.cat(local_inputs, dim=-1).unsqueeze(0),
        full_weight,
    ).squeeze(0)
    assert projection.weight.shape == (5, 8)
    assert len(calls) == 1
    torch.testing.assert_close(
        output,
        expected,
        rtol=0.0,
        atol=0.0,
    )
