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


def test_full_gemm_then_returns_rank_local_output_without_collective():
    projection_type = getattr(
        linear_module,
        "ReplicatedLocalOutputLinear",
        None,
    )
    assert projection_type is not None

    collective_names = (
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "gather",
        "reduce_scatter",
        "reduce_scatter_tensor",
    )
    originals = {
        name: getattr(torch.distributed, name)
        for name in collective_names
        if hasattr(torch.distributed, name)
    }

    def reject_collective(*args, **kwargs):
        raise AssertionError("forward must not call a distributed collective")

    for name in originals:
        setattr(torch.distributed, name, reject_collective)
    try:
        for rank in range(4):
            with _tp_layout(rank, 4):
                projection = projection_type(
                    input_size=8,
                    output_size=16,
                    bias=False,
                ).to(dtype=torch.bfloat16)
            weight = (
                torch.arange(16 * 8, dtype=torch.float32)
                .reshape(16, 8)
                .div(32)
                .to(torch.bfloat16)
            )
            hidden_states = (
                torch.arange(5 * 8, dtype=torch.float32)
                .reshape(5, 8)
                .sub(16)
                .div(8)
                .to(torch.bfloat16)
            )
            with torch.no_grad():
                projection.weight.copy_(weight)

            output = projection(hidden_states)
            full_output = F.linear(
                hidden_states.unsqueeze(0),
                weight,
            ).squeeze(0)
            expected = full_output.narrow(1, rank * 4, 4)

            assert projection.weight.shape == (16, 8)
            assert output.shape == (5, 4)
            torch.testing.assert_close(
                output,
                expected,
                rtol=0.0,
                atol=0.0,
            )
    finally:
        for name, function in originals.items():
            setattr(torch.distributed, name, function)
