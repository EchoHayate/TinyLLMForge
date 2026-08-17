import torch

from tinyvllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    configure_linear_execution_rows,
)


def test_bfloat16_projection_matches_scheduler_partition() -> None:
    original_get_rank = torch.distributed.get_rank
    original_get_world_size = torch.distributed.get_world_size
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    try:
        layer = ReplicatedLinear(
            input_size=2048,
            output_size=1536,
            bias=False,
        ).to(device="cuda", dtype=torch.bfloat16)
    finally:
        torch.distributed.get_rank = original_get_rank
        torch.distributed.get_world_size = original_get_world_size

    torch.manual_seed(17)
    with torch.no_grad():
        layer.weight.normal_()
    hidden_states = torch.randn(
        1088,
        2048,
        device="cuda",
        dtype=torch.bfloat16,
    )
    configure_linear_execution_rows(layer, 1024)

    one_shot = layer(hidden_states)
    partitioned = torch.cat((
        layer(hidden_states[:1024]),
        layer(hidden_states[1024:]),
    ))

    torch.testing.assert_close(
        one_shot,
        partitioned,
        rtol=0.0,
        atol=0.0,
    )


def test_full_sequence_projection_ignores_global_row_partition() -> None:
    original_get_rank = torch.distributed.get_rank
    original_get_world_size = torch.distributed.get_world_size
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    try:
        layer = ColumnParallelLinear(
            input_size=8,
            output_size=8,
            bias=False,
        )
    finally:
        torch.distributed.get_rank = original_get_rank
        torch.distributed.get_world_size = original_get_world_size

    layer.requires_unpartitioned_linear_execution = True
    configure_linear_execution_rows(layer, 4)

    assert layer.linear_execution_rows == 0
