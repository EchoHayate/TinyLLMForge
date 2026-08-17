import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F


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


def _required_component(name: str):
    component = getattr(linear, name, None)
    assert component is not None, f"missing {name}"
    return component


def test_query_projection_runs_full_gemm_then_selects_local_head_pairs():
    projection_type = _required_component(
        "ReplicatedHeadPairedColumnParallelLinear"
    )
    input_size = 16
    num_heads = 8
    head_dim = 4
    world_size = 4
    source = torch.linspace(
        -1.0,
        1.0,
        steps=num_heads * 2 * head_dim * input_size,
        dtype=torch.float32,
    ).reshape(num_heads * 2 * head_dim, input_size).to(torch.bfloat16)
    hidden_states = torch.linspace(
        -0.75,
        0.75,
        steps=7 * input_size,
        dtype=torch.float32,
    ).reshape(7, input_size).to(torch.bfloat16)
    full_output = F.linear(hidden_states, source)
    local_width = num_heads * 2 * head_dim // world_size

    for rank in range(world_size):
        with _DistLayout(rank, world_size):
            projection = projection_type(
                input_size=input_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=False,
            ).to(dtype=torch.bfloat16)
        projection.weight.weight_loader(projection.weight, source)
        projection.linear_execution_rows = 2

        assert projection.weight.shape == source.shape
        assert projection.requires_unpartitioned_linear_execution is True
        torch.testing.assert_close(
            projection(hidden_states),
            full_output.narrow(-1, rank * local_width, local_width),
            rtol=0,
            atol=0,
        )


def test_kv_projection_runs_full_gemm_then_selects_replicated_head():
    projection_type = _required_component(
        "ReplicatedKVHeadParallelLinear"
    )
    input_size = 16
    total_num_kv_heads = 2
    head_dim = 4
    world_size = 4
    source = torch.linspace(
        -0.5,
        0.5,
        steps=total_num_kv_heads * head_dim * input_size,
        dtype=torch.float32,
    ).reshape(
        total_num_kv_heads * head_dim,
        input_size,
    ).to(torch.bfloat16)
    hidden_states = torch.linspace(
        -1.0,
        1.0,
        steps=9 * input_size,
        dtype=torch.float32,
    ).reshape(9, input_size).to(torch.bfloat16)
    full_output = F.linear(hidden_states, source)

    for rank in range(world_size):
        with _DistLayout(rank, world_size):
            projection = projection_type(
                input_size=input_size,
                total_num_kv_heads=total_num_kv_heads,
                head_dim=head_dim,
                bias=False,
            ).to(dtype=torch.bfloat16)
        projection.weight.weight_loader(projection.weight, source)
        projection.linear_execution_rows = 3
        source_kv_rank = rank // 2

        assert projection.weight.shape == source.shape
        assert projection.local_num_kv_heads == 1
        assert projection.num_kv_head_replicas == 2
        assert projection.source_kv_rank == source_kv_rank
        assert projection.requires_unpartitioned_linear_execution is True
        torch.testing.assert_close(
            projection(hidden_states),
            full_output.narrow(-1, source_kv_rank * head_dim, head_dim),
            rtol=0,
            atol=0,
        )


def test_segmented_projection_runs_full_gemm_then_selects_each_local_segment():
    projection_type = _required_component(
        "ReplicatedSegmentedColumnParallelLinear"
    )
    input_size = 16
    output_sizes = (12, 8, 16)
    world_size = 4
    source = torch.linspace(
        -0.875,
        0.875,
        steps=sum(output_sizes) * input_size,
        dtype=torch.float32,
    ).reshape(sum(output_sizes), input_size).to(torch.bfloat16)
    hidden_states = torch.linspace(
        -0.625,
        0.625,
        steps=11 * input_size,
        dtype=torch.float32,
    ).reshape(11, input_size).to(torch.bfloat16)
    full_output = F.linear(hidden_states, source)

    for rank in range(world_size):
        with _DistLayout(rank, world_size):
            projection = projection_type(
                input_size=input_size,
                output_sizes=output_sizes,
                bias=False,
            ).to(dtype=torch.bfloat16)
        projection.weight.weight_loader(projection.weight, source)
        expected = []
        global_offset = 0
        for output_size in output_sizes:
            local_size = output_size // world_size
            expected.append(
                full_output.narrow(
                    -1,
                    global_offset + rank * local_size,
                    local_size,
                )
            )
            global_offset += output_size

        assert projection.weight.shape == source.shape
        torch.testing.assert_close(
            projection(hidden_states),
            torch.cat(expected, dim=-1),
            rtol=0,
            atol=0,
        )


def test_replicated_column_projection_runs_full_gemm_then_selects_local_slice():
    projection_type = _required_component(
        "ReplicatedColumnParallelLinear"
    )
    input_size = 16
    output_size = 32
    world_size = 4
    source = torch.linspace(
        -0.9375,
        0.9375,
        steps=output_size * input_size,
        dtype=torch.float32,
    ).reshape(output_size, input_size).to(torch.bfloat16)
    hidden_states = torch.linspace(
        -0.6875,
        0.6875,
        steps=13 * input_size,
        dtype=torch.float32,
    ).reshape(13, input_size).to(torch.bfloat16)
    full_output = F.linear(hidden_states, source)
    local_size = output_size // world_size

    for rank in range(world_size):
        with _DistLayout(rank, world_size):
            projection = projection_type(
                input_size=input_size,
                output_size=output_size,
                bias=False,
            ).to(dtype=torch.bfloat16)
        projection.weight.weight_loader(projection.weight, source)

        assert projection.weight.shape == source.shape
        torch.testing.assert_close(
            projection(hidden_states),
            full_output.narrow(-1, rank * local_size, local_size),
            rtol=0,
            atol=0,
        )


def main():
    test_query_projection_runs_full_gemm_then_selects_local_head_pairs()
    test_kv_projection_runs_full_gemm_then_selects_replicated_head()
    test_segmented_projection_runs_full_gemm_then_selects_each_local_segment()
    test_replicated_column_projection_runs_full_gemm_then_selects_local_slice()
    print("qwen35 replicated head projection tests passed")


if __name__ == "__main__":
    main()
