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
KVHeadParallelLinear = linear.KVHeadParallelLinear


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


def _source(total_heads: int, head_dim: int, input_size: int):
    rows = []
    for head in range(total_heads):
        for lane in range(head_dim):
            code = 1000 * head + lane
            rows.append(
                torch.arange(input_size, dtype=torch.float32) + code * 10
            )
    return torch.stack(rows)


def _expect_value_error(function, message: str):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {message!r}"
        )


def test_tp4_replicates_two_complete_kv_heads():
    source = _source(total_heads=2, head_dim=3, input_size=2)
    observed = []
    for rank in range(4):
        with _DistLayout(rank, 4):
            layer = KVHeadParallelLinear(
                input_size=2,
                total_num_kv_heads=2,
                head_dim=3,
                bias=False,
            )
        layer.weight.weight_loader(layer.weight, source)
        observed.append(layer.weight.detach().clone())
        assert layer.local_num_kv_heads == 1
        assert layer.num_kv_head_replicas == 2
        assert layer.source_kv_rank == rank // 2
        assert layer.weight.shape == (3, 2)

    torch.testing.assert_close(observed[0], observed[1])
    torch.testing.assert_close(observed[2], observed[3])
    assert not torch.equal(observed[0], observed[2])
    torch.testing.assert_close(observed[0], source[:3])
    torch.testing.assert_close(observed[2], source[3:])


def test_tp1_tp2_preserve_normal_complete_head_sharding():
    source = _source(total_heads=4, head_dim=2, input_size=3)
    for world_size in (1, 2):
        local_heads = 4 // world_size
        local_rows = local_heads * 2
        for rank in range(world_size):
            with _DistLayout(rank, world_size):
                layer = KVHeadParallelLinear(
                    input_size=3,
                    total_num_kv_heads=4,
                    head_dim=2,
                    bias=False,
                )
            layer.weight.weight_loader(layer.weight, source)
            assert layer.local_num_kv_heads == local_heads
            assert layer.num_kv_head_replicas == 1
            assert layer.source_kv_rank == rank
            torch.testing.assert_close(
                layer.weight,
                source.narrow(0, rank * local_rows, local_rows),
            )


def test_constructor_rejects_partial_or_uneven_kv_replication():
    with _DistLayout(rank=0, world_size=4):
        _expect_value_error(
            lambda: KVHeadParallelLinear(
                input_size=2,
                total_num_kv_heads=3,
                head_dim=2,
                bias=False,
            ),
            "replication",
        )
    with _DistLayout(rank=0, world_size=3):
        _expect_value_error(
            lambda: KVHeadParallelLinear(
                input_size=2,
                total_num_kv_heads=4,
                head_dim=2,
                bias=False,
            ),
            "sharding",
        )


def main():
    test_tp4_replicates_two_complete_kv_heads()
    test_tp1_tp2_preserve_normal_complete_head_sharding()
    test_constructor_rejects_partial_or_uneven_kv_replication()
    print("qwen35 KV-head parallel linear tests passed")


if __name__ == "__main__":
    main()
