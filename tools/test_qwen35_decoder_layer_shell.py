import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn

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

decoder_layer = _load_module(
    "tinyvllm.layers.qwen35_decoder_layer",
    "tinyvllm/layers/qwen35_decoder_layer.py",
)
Qwen35DecoderLayerShell = decoder_layer.Qwen35DecoderLayerShell


class _Affine(nn.Module):

    def __init__(
        self,
        name: str,
        events: list,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.name = name
        self.events = events
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.events.append(self.name)
        return (
            tensor * self.scale.to(tensor.dtype)
            + self.bias.to(tensor.dtype)
        )


class _FullAttention(nn.Module):

    def __init__(
        self,
        events: list,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.events = events
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self.events.append("full_attention")
        positions = (
            position_ids[0]
            if position_ids.ndim == 2
            else position_ids
        )
        position_term = positions.to(hidden_states.dtype).unsqueeze(-1)
        return (
            hidden_states * self.scale.to(hidden_states.dtype)
            + self.bias.to(hidden_states.dtype)
            + position_term
        )


class _LinearAttention(nn.Module):

    def __init__(
        self,
        events: list,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.events = events
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.events.append("linear_attention")
        return (
            hidden_states * self.scale.to(hidden_states.dtype)
            + self.bias.to(hidden_states.dtype)
        )


class _Forbidden(nn.Module):

    def forward(self, *args):
        raise AssertionError("unselected token mixer must not be called")


def _parameters(dtype: torch.dtype) -> dict:
    values = {
        "input_scale": torch.tensor([1.5, -0.5, 0.25, 2.0]),
        "input_bias": torch.tensor([0.25, -0.75, 1.0, -0.5]),
        "full_scale": torch.tensor([0.5, 1.25, -1.0, 0.75]),
        "full_bias": torch.tensor([-0.5, 0.25, 0.75, -1.0]),
        "linear_scale": torch.tensor([1.0, -0.25, 0.5, 1.5]),
        "linear_bias": torch.tensor([0.0, 0.5, -0.75, 0.25]),
        "post_scale": torch.tensor([-0.5, 2.0, 0.75, 0.25]),
        "post_bias": torch.tensor([1.0, -0.25, 0.5, -1.5]),
        "mlp_scale": torch.tensor([0.25, -1.0, 1.5, 0.5]),
        "mlp_bias": torch.tensor([-0.5, 0.75, 0.25, 1.0]),
    }
    return {
        name: tensor.to(dtype)
        for name, tensor in values.items()
    }


def _new_full_shell(
    events: list,
    dtype: torch.dtype = torch.float32,
) -> Qwen35DecoderLayerShell:
    values = _parameters(dtype)
    return Qwen35DecoderLayerShell(
        block_type="full_attention",
        input_layernorm=_Affine(
            "input_layernorm",
            events,
            values["input_scale"],
            values["input_bias"],
        ),
        post_attention_layernorm=_Affine(
            "post_attention_layernorm",
            events,
            values["post_scale"],
            values["post_bias"],
        ),
        mlp=_Affine(
            "mlp",
            events,
            values["mlp_scale"],
            values["mlp_bias"],
        ),
        full_attention=_FullAttention(
            events,
            values["full_scale"],
            values["full_bias"],
        ),
        linear_attention=_Forbidden(),
    )


def _new_linear_shell(
    events: list,
    dtype: torch.dtype = torch.float32,
) -> Qwen35DecoderLayerShell:
    values = _parameters(dtype)
    return Qwen35DecoderLayerShell(
        block_type="linear_attention",
        input_layernorm=_Affine(
            "input_layernorm",
            events,
            values["input_scale"],
            values["input_bias"],
        ),
        post_attention_layernorm=_Affine(
            "post_attention_layernorm",
            events,
            values["post_scale"],
            values["post_bias"],
        ),
        mlp=_Affine(
            "mlp",
            events,
            values["mlp_scale"],
            values["mlp_bias"],
        ),
        full_attention=_Forbidden(),
        linear_attention=_LinearAttention(
            events,
            values["linear_scale"],
            values["linear_bias"],
        ),
    )


def _manual_full(
    position_ids: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    values = _parameters(hidden_states.dtype)
    normalized = (
        hidden_states * values["input_scale"]
        + values["input_bias"]
    )
    positions = position_ids[0] if position_ids.ndim == 2 else position_ids
    mixed = (
        normalized * values["full_scale"]
        + values["full_bias"]
        + positions.to(hidden_states.dtype).unsqueeze(-1)
    )
    after_mixer = hidden_states + mixed
    normalized = (
        after_mixer * values["post_scale"]
        + values["post_bias"]
    )
    mlp_output = (
        normalized * values["mlp_scale"]
        + values["mlp_bias"]
    )
    return after_mixer + mlp_output


def _manual_linear(hidden_states: torch.Tensor) -> torch.Tensor:
    values = _parameters(hidden_states.dtype)
    normalized = (
        hidden_states * values["input_scale"]
        + values["input_bias"]
    )
    mixed = (
        normalized * values["linear_scale"]
        + values["linear_bias"]
    )
    after_mixer = hidden_states + mixed
    normalized = (
        after_mixer * values["post_scale"]
        + values["post_bias"]
    )
    mlp_output = (
        normalized * values["mlp_scale"]
        + values["mlp_bias"]
    )
    return after_mixer + mlp_output


def test_full_attention_order_residuals_and_numerical_oracle() -> None:
    events = []
    shell = _new_full_shell(events)
    positions = torch.tensor(
        [[0, 2], [4, 6], [8, 10]],
        dtype=torch.int64,
    )
    hidden_states = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]]
    )
    original_positions = positions.clone()
    original_hidden = hidden_states.clone()
    actual = shell(positions, hidden_states)
    expected = _manual_full(positions, hidden_states)
    assert events == [
        "input_layernorm",
        "full_attention",
        "post_attention_layernorm",
        "mlp",
    ]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(positions, original_positions)
    torch.testing.assert_close(hidden_states, original_hidden)


def test_linear_attention_order_dispatch_and_numerical_oracle() -> None:
    events = []
    shell = _new_linear_shell(events)
    positions = torch.tensor([3, 7], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[0.5, -1.0, 2.0, 0.25], [3.0, -0.5, 1.0, -2.0]]
    )
    actual = shell(positions, hidden_states)
    expected = _manual_linear(hidden_states)
    assert events == [
        "input_layernorm",
        "linear_attention",
        "post_attention_layernorm",
        "mlp",
    ]
    torch.testing.assert_close(actual, expected)


def test_bfloat16_and_noncontiguous_inputs_match_oracles() -> None:
    events = []
    full_shell = _new_full_shell(events, dtype=torch.bfloat16)
    hidden_states = (
        torch.arange(8, dtype=torch.float32)
        .reshape(4, 2)
        .t()
        .to(torch.bfloat16)
    )
    assert not hidden_states.is_contiguous()
    positions = torch.tensor([1, 4], dtype=torch.int64)
    full_output = full_shell(positions, hidden_states)
    full_expected = _manual_full(positions, hidden_states)
    assert full_output.dtype == torch.bfloat16
    torch.testing.assert_close(full_output.float(), full_expected.float())

    events.clear()
    linear_shell = _new_linear_shell(events, dtype=torch.bfloat16)
    linear_output = linear_shell(positions, hidden_states)
    linear_expected = _manual_linear(hidden_states)
    assert linear_output.dtype == torch.bfloat16
    torch.testing.assert_close(linear_output.float(), linear_expected.float())


def test_noncontiguous_component_outputs_follow_shape_contract() -> None:
    hidden_states = torch.ones(2, 4)
    positions = torch.tensor([0, 1])
    noncontiguous = torch.arange(
        8,
        dtype=torch.float32,
    ).reshape(4, 2).t()
    assert not noncontiguous.is_contiguous()
    shell = _boundary_shell(
        input_layernorm=_ReturnModule(noncontiguous),
        full_attention=_ReturnModule(noncontiguous),
        post_attention_layernorm=_ReturnModule(noncontiguous),
        mlp=_ReturnModule(noncontiguous),
    )
    actual = shell(positions, hidden_states)
    expected_after_mixer = hidden_states + noncontiguous
    expected = expected_after_mixer + noncontiguous
    torch.testing.assert_close(actual, expected)


class _ReturnModule(nn.Module):

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args):
        return self.output


class _Identity(nn.Module):

    def forward(self, tensor):
        return tensor


class _IdentityFull(nn.Module):

    def forward(self, positions, hidden_states):
        return hidden_states


def _boundary_shell(
    block_type: str = "full_attention",
    **overrides
) -> Qwen35DecoderLayerShell:
    modules = {
        "input_layernorm": _Identity(),
        "post_attention_layernorm": _Identity(),
        "mlp": _Identity(),
        "full_attention": _IdentityFull(),
        "linear_attention": _Identity(),
    }
    modules.update(overrides)
    return Qwen35DecoderLayerShell(
        block_type=block_type,
        **modules,
    )


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_constructor_and_input_fail_closed() -> None:
    _expect_value_error(
        lambda: _boundary_shell(block_type="attention"),
        "block_type",
    )
    _expect_value_error(
        lambda: Qwen35DecoderLayerShell(
            block_type="full_attention",
            input_layernorm=_Identity(),
            post_attention_layernorm=_Identity(),
            mlp=_Identity(),
            full_attention=None,
        ),
        "full_attention",
    )
    _expect_value_error(
        lambda: Qwen35DecoderLayerShell(
            block_type="linear_attention",
            input_layernorm=_Identity(),
            post_attention_layernorm=_Identity(),
            mlp=_Identity(),
            linear_attention=None,
        ),
        "linear_attention",
    )
    shell = _boundary_shell()
    _expect_value_error(
        lambda: shell(torch.tensor([0, 1]), torch.ones(2, 2, 2)),
        "rank two",
    )
    _expect_value_error(
        lambda: shell(
            torch.tensor([0, 1]),
            torch.ones(2, 4, dtype=torch.int64),
        ),
        "floating point",
    )


def test_component_boundaries_fail_closed() -> None:
    positions = torch.tensor([0, 1])
    hidden_states = torch.ones(2, 4)
    cases = (
        (
            "input_layernorm",
            _ReturnModule(torch.ones(2, 3)),
            "input_layernorm shape",
        ),
        (
            "input_layernorm",
            _ReturnModule(torch.ones(2, 4, dtype=torch.float64)),
            "input_layernorm dtype",
        ),
        (
            "input_layernorm",
            _ReturnModule(torch.ones(2, 4, device="meta")),
            "input_layernorm device",
        ),
        (
            "input_layernorm",
            _ReturnModule(torch.ones(2, 4, dtype=torch.int64)),
            "input_layernorm output must use a floating point dtype",
        ),
        (
            "input_layernorm",
            _ReturnModule("not a tensor"),
            "input_layernorm output must be a tensor",
        ),
        (
            "full_attention",
            _ReturnModule(torch.ones(2, 3)),
            "full_attention shape",
        ),
        (
            "full_attention",
            _ReturnModule(torch.ones(2, 4, dtype=torch.float64)),
            "full_attention dtype",
        ),
        (
            "full_attention",
            _ReturnModule(torch.ones(2, 4, device="meta")),
            "full_attention device",
        ),
        (
            "full_attention",
            _ReturnModule(torch.ones(2, 4, dtype=torch.int64)),
            "full_attention output must use a floating point dtype",
        ),
        (
            "full_attention",
            _ReturnModule("not a tensor"),
            "full_attention output must be a tensor",
        ),
        (
            "post_attention_layernorm",
            _ReturnModule(torch.ones(1, 4)),
            "post_attention_layernorm shape",
        ),
        (
            "post_attention_layernorm",
            _ReturnModule(torch.ones(2, 4, dtype=torch.float64)),
            "post_attention_layernorm dtype",
        ),
        (
            "post_attention_layernorm",
            _ReturnModule(torch.ones(2, 4, device="meta")),
            "post_attention_layernorm device",
        ),
        (
            "post_attention_layernorm",
            _ReturnModule(torch.ones(2, 4, dtype=torch.int64)),
            "post_attention_layernorm output must use a floating point dtype",
        ),
        (
            "post_attention_layernorm",
            _ReturnModule("not a tensor"),
            "post_attention_layernorm output must be a tensor",
        ),
        ("mlp", _ReturnModule(torch.ones(2, 5)), "mlp shape"),
        (
            "mlp",
            _ReturnModule(torch.ones(2, 4, dtype=torch.float64)),
            "mlp dtype",
        ),
        (
            "mlp",
            _ReturnModule(torch.ones(2, 4, device="meta")),
            "mlp device",
        ),
        (
            "mlp",
            _ReturnModule(torch.ones(2, 4, dtype=torch.int64)),
            "mlp output must use a floating point dtype",
        ),
        (
            "mlp",
            _ReturnModule("not a tensor"),
            "mlp output must be a tensor",
        ),
    )
    for name, module, message in cases:
        shell = _boundary_shell(**{name: module})
        _expect_value_error(
            lambda shell=shell: shell(positions, hidden_states),
            message,
        )

    for output, message in (
        (torch.ones(2, 3), "linear_attention shape"),
        (
            torch.ones(2, 4, dtype=torch.float64),
            "linear_attention dtype",
        ),
        (
            torch.ones(2, 4, device="meta"),
            "linear_attention device",
        ),
        (
            torch.ones(2, 4, dtype=torch.int64),
            "linear_attention output must use a floating point dtype",
        ),
        ("not a tensor", "linear_attention output must be a tensor"),
    ):
        shell = _boundary_shell(
            block_type="linear_attention",
            linear_attention=_ReturnModule(output),
        )
        _expect_value_error(
            lambda shell=shell: shell(positions, hidden_states),
            message,
        )


def main() -> None:
    test_full_attention_order_residuals_and_numerical_oracle()
    test_linear_attention_order_dispatch_and_numerical_oracle()
    test_bfloat16_and_noncontiguous_inputs_match_oracles()
    test_noncontiguous_component_outputs_follow_shape_contract()
    test_constructor_and_input_fail_closed()
    test_component_boundaries_fail_closed()
    print("qwen35 decoder layer shell tests passed")


if __name__ == "__main__":
    main()
