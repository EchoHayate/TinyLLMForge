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


for package_name in (
    "tinyvllm",
    "tinyvllm.layers",
    "tinyvllm.models",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

transformers = types.ModuleType("transformers")
transformers.Qwen3Config = object
sys.modules["transformers"] = transformers

linear = _load_module("tinyvllm.layers.linear", "tinyvllm/layers/linear.py")
_load_module("tinyvllm.layers.activation", "tinyvllm/layers/activation.py")


class _Unused:

    def __init__(self, *args, **kwargs):
        raise AssertionError("unrelated Qwen3 model module was constructed")


for module_name, attributes in {
    "tinyvllm.layers.attention": {"Attention": _Unused},
    "tinyvllm.layers.layernorm": {"RMSNorm": _Unused},
    "tinyvllm.layers.rotary_embedding": {"get_rope": _Unused},
    "tinyvllm.layers.embed_head": {
        "VocabParallelEmbedding": _Unused,
        "ParallelLMHead": _Unused,
    },
}.items():
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules[module_name] = module

qwen3 = _load_module("tinyvllm.models.qwen3", "tinyvllm/models/qwen3.py")
Qwen3MLP = qwen3.Qwen3MLP


class _DistLayout:

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.old_get_rank = linear.dist.get_rank
        self.old_get_world_size = linear.dist.get_world_size
        self.old_all_reduce = linear.dist.all_reduce

    def __enter__(self):
        linear.dist.get_rank = lambda: self.rank
        linear.dist.get_world_size = lambda: self.world_size
        linear.dist.all_reduce = lambda tensor: tensor

    def __exit__(self, exc_type, exc_value, traceback):
        linear.dist.get_rank = self.old_get_rank
        linear.dist.get_world_size = self.old_get_world_size
        linear.dist.all_reduce = self.old_all_reduce


def _weights(
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
) -> tuple:
    gate = (
        torch.arange(
            intermediate_size * hidden_size,
            dtype=torch.float32,
        ).reshape(intermediate_size, hidden_size)
        / 17
        - 1.5
    ).to(dtype)
    up = (
        torch.arange(
            intermediate_size * hidden_size,
            dtype=torch.float32,
        ).reshape(intermediate_size, hidden_size)
        .flip(0)
        / 13
        - 0.75
    ).to(dtype)
    down = (
        torch.arange(
            hidden_size * intermediate_size,
            dtype=torch.float32,
        ).reshape(hidden_size, intermediate_size)
        / 19
        - 1.0
    ).to(dtype)
    return gate, up, down


def _official_oracle(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)


def _run_rank(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    swap_gate_up: bool = False,
) -> tuple:
    with _DistLayout(rank, world_size):
        mlp = Qwen3MLP(
            hidden_size=hidden_states.shape[1],
            intermediate_size=gate_weight.shape[0],
            hidden_act="silu",
        ).to(hidden_states.dtype)
        first = up_weight if swap_gate_up else gate_weight
        second = gate_weight if swap_gate_up else up_weight
        mlp.gate_up_proj.weight.weight_loader(
            mlp.gate_up_proj.weight,
            first,
            0,
        )
        mlp.gate_up_proj.weight.weight_loader(
            mlp.gate_up_proj.weight,
            second,
            1,
        )
        mlp.down_proj.weight.weight_loader(
            mlp.down_proj.weight,
            down_weight,
        )
        local_width = gate_weight.shape[0] // world_size
        expected_fused = torch.cat(
            (
                first.narrow(0, rank * local_width, local_width),
                second.narrow(0, rank * local_width, local_width),
            ),
            dim=0,
        )
        torch.testing.assert_close(
            mlp.gate_up_proj.weight,
            expected_fused,
        )
        expected_down = down_weight.narrow(
            1,
            rank * local_width,
            local_width,
        )
        torch.testing.assert_close(mlp.down_proj.weight, expected_down)
        return mlp(hidden_states), mlp


def test_tp_1_2_4_real_loaders_and_forward_match_official_formula() -> None:
    hidden_size = 4
    intermediate_size = 8
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]
    )
    original = hidden_states.clone()
    gate_weight, up_weight, down_weight = _weights(
        hidden_size,
        intermediate_size,
        torch.float32,
    )
    expected = _official_oracle(
        hidden_states,
        gate_weight,
        up_weight,
        down_weight,
    )
    for world_size in (1, 2, 4):
        partials = []
        for rank in range(world_size):
            partial, _ = _run_rank(
                hidden_states,
                gate_weight,
                up_weight,
                down_weight,
                rank,
                world_size,
            )
            partials.append(partial)
        actual = torch.stack(partials).sum(dim=0)
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden_states, original)


def test_reversed_up_gate_local_order_is_not_official_formula() -> None:
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]
    )
    gate_weight, up_weight, down_weight = _weights(4, 8, torch.float32)
    expected = _official_oracle(
        hidden_states,
        gate_weight,
        up_weight,
        down_weight,
    )
    partials = []
    for rank in range(2):
        partial, _ = _run_rank(
            hidden_states,
            gate_weight,
            up_weight,
            down_weight,
            rank,
            2,
            swap_gate_up=True,
        )
        partials.append(partial)
    incorrect = torch.stack(partials).sum(dim=0)
    assert not torch.allclose(incorrect, expected)


def test_bfloat16_preserves_dtype_and_matches_official_formula() -> None:
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]],
        dtype=torch.bfloat16,
    )
    gate_weight, up_weight, down_weight = _weights(4, 8, torch.bfloat16)
    expected = _official_oracle(
        hidden_states,
        gate_weight,
        up_weight,
        down_weight,
    )
    partials = []
    for rank in range(2):
        partial, _ = _run_rank(
            hidden_states,
            gate_weight,
            up_weight,
            down_weight,
            rank,
            2,
        )
        partials.append(partial)
    actual = torch.stack(partials).sum(dim=0)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=4e-2,
        atol=2e-2,
    )


def test_qwen35_replicated_mlp_matches_official_bfloat16_topology() -> None:
    replicated_gate_up_type = getattr(
        linear,
        "ReplicatedMergedColumnParallelLinear",
        None,
    )
    assert replicated_gate_up_type is not None
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]],
        dtype=torch.bfloat16,
    )
    gate_weight, up_weight, down_weight = _weights(
        4,
        8,
        torch.bfloat16,
    )
    expected = _official_oracle(
        hidden_states,
        gate_weight,
        up_weight,
        down_weight,
    )
    for rank in range(4):
        with _DistLayout(rank=rank, world_size=4):
            gate_up = replicated_gate_up_type(
                4,
                [8, 8],
                bias=False,
            ).to(torch.bfloat16)
            down = linear.ReplicatedLinear(
                8,
                4,
                bias=False,
            ).to(torch.bfloat16)
            gate_up.weight.weight_loader(
                gate_up.weight,
                gate_weight,
                0,
            )
            gate_up.weight.weight_loader(
                gate_up.weight,
                up_weight,
                1,
            )
            down.weight.weight_loader(down.weight, down_weight)
            gate, up = gate_up(hidden_states).chunk(2, dim=-1)
            actual = down(F.silu(gate) * up)
            assert gate_up.weight.shape == (16, 4)
            assert down.weight.shape == (4, 8)
            assert torch.equal(actual, expected)


def test_existing_constructor_rejects_non_silu_activation() -> None:
    with _DistLayout(rank=0, world_size=1):
        try:
            Qwen3MLP(
                hidden_size=4,
                intermediate_size=8,
                hidden_act="gelu",
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("Qwen3MLP must reject non-SiLU activation")


def main() -> None:
    test_tp_1_2_4_real_loaders_and_forward_match_official_formula()
    test_reversed_up_gate_local_order_is_not_official_formula()
    test_bfloat16_preserves_dtype_and_matches_official_formula()
    test_qwen35_replicated_mlp_matches_official_bfloat16_topology()
    test_existing_constructor_rejects_non_silu_activation()
    print("qwen35 mlp reuse compatibility tests passed")


if __name__ == "__main__":
    main()
