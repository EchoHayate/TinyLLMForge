import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.layers"):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
        sys.modules[package_name] = package

decoder_module = _load_module(
    "tinyvllm.layers.qwen35_decoder_layer",
    "tinyvllm/layers/qwen35_decoder_layer.py",
)
packed_module = _load_module(
    "tinyvllm.layers.qwen35_packed_full_decoder_layer",
    "tinyvllm/layers/qwen35_packed_full_decoder_layer.py",
)
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35PackedFullDecoderLayer = packed_module.Qwen35PackedFullDecoderLayer


class _Affine(nn.Module):

    def __init__(self, name, events, scale, bias):
        super().__init__()
        self.name = name
        self.events = events
        self.scale = scale
        self.bias = bias

    def forward(self, tensor):
        self.events.append((self.name, tensor.shape[0]))
        return tensor * self.scale + self.bias


class _SegmentAttention(nn.Module):

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.call_index = 0
        self.fail_on_call = None
        self.positions = []

    def forward(self, position_ids, hidden_states):
        call_index = self.call_index
        self.call_index += 1
        self.events.append(("full_attention", hidden_states.shape[0]))
        self.positions.append(position_ids.clone())
        if call_index == self.fail_on_call:
            raise RuntimeError("injected full request failure")
        segment_mean = hidden_states.mean(dim=0, keepdim=True)
        positions = (
            position_ids[0]
            if position_ids.ndim == 2
            else position_ids
        )
        return hidden_states + segment_mean + positions.to(
            hidden_states.dtype
        ).unsqueeze(-1)


class _ForbiddenLinear(nn.Module):

    def forward(self, *args):
        raise AssertionError("linear attention must not be called")


def _fixture():
    events = []
    attention = _SegmentAttention(events)
    decoder = Qwen35DecoderLayerShell(
        block_type="full_attention",
        input_layernorm=_Affine("input_layernorm", events, 2.0, 1.0),
        post_attention_layernorm=_Affine(
            "post_attention_layernorm",
            events,
            1.0,
            -0.25,
        ),
        mlp=_Affine("mlp", events, 0.2, 0.0),
        full_attention=attention,
        linear_attention=_ForbiddenLinear(),
    )
    return events, attention, Qwen35PackedFullDecoderLayer(decoder)


def _hidden(dtype=torch.float32):
    return torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [-1.0, 0.25, 2.0, -0.5],
            [0.5, 1.5, -1.0, 2.0],
            [2.0, 0.0, -0.5, 1.0],
            [-2.0, 1.0, 0.25, 0.75],
            [1.25, -0.75, 2.5, -1.5],
        ],
        dtype=dtype,
    )


def _manual(hidden_states, token_counts, position_ids):
    outputs = []
    offset = 0
    positions = position_ids[0] if position_ids.ndim == 2 else position_ids
    for token_count in token_counts:
        segment = hidden_states[offset:offset + token_count]
        normalized = segment * 2.0 + 1.0
        mixed = (
            normalized
            + normalized.mean(dim=0, keepdim=True)
            + positions[offset:offset + token_count]
            .to(hidden_states.dtype)
            .unsqueeze(-1)
        )
        after_mixer = segment + mixed
        post_normalized = after_mixer - 0.25
        outputs.append(after_mixer + post_normalized * 0.2)
        offset += token_count
    return torch.cat(outputs)


def test_segmented_full_attention_matches_oracle_and_call_lengths() -> None:
    events, attention, wrapper = _fixture()
    hidden_states = _hidden()
    position_ids = torch.arange(6)
    original_hidden = hidden_states.clone()
    original_positions = position_ids.clone()
    actual = wrapper((2, 1, 3), position_ids, hidden_states)
    expected = _manual(hidden_states, (2, 1, 3), position_ids)
    torch.testing.assert_close(actual, expected)
    assert [
        length for name, length in events if name == "full_attention"
    ] == [2, 1, 3]
    assert [tensor.tolist() for tensor in attention.positions] == [
        [0, 1],
        [2],
        [3, 4, 5],
    ]
    torch.testing.assert_close(hidden_states, original_hidden)
    torch.testing.assert_close(position_ids, original_positions)


def test_other_request_changes_do_not_affect_first_request() -> None:
    _, _, wrapper = _fixture()
    hidden_states = _hidden()
    positions = torch.arange(6)
    first = wrapper((2, 1, 3), positions, hidden_states)
    changed = hidden_states.clone()
    changed[2:].add_(10000)
    _, _, second_wrapper = _fixture()
    second = second_wrapper((2, 1, 3), positions, changed)
    torch.testing.assert_close(first[:2], second[:2])


def test_position_row_shapes_and_bfloat16_noncontiguous() -> None:
    for position_ids in (
        torch.arange(6),
        torch.arange(6).unsqueeze(0),
        torch.stack((
            torch.arange(6),
            torch.arange(6) + 10,
            torch.arange(6) + 20,
        )),
    ):
        _, attention, wrapper = _fixture()
        hidden_states = (
            torch.arange(24, dtype=torch.float32)
            .reshape(4, 6)
            .t()
            .to(torch.bfloat16)
        )
        assert not hidden_states.is_contiguous()
        actual = wrapper((2, 1, 3), position_ids, hidden_states)
        expected = _manual(hidden_states, (2, 1, 3), position_ids)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected.float())
        expected_shapes = (
            [(2,), (1,), (3,)]
            if position_ids.ndim == 1
            else [
                (position_ids.shape[0], 2),
                (position_ids.shape[0], 1),
                (position_ids.shape[0], 3),
            ]
        )
        assert [
            tuple(tensor.shape) for tensor in attention.positions
        ] == expected_shapes


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected {error_type.__name__} containing {message!r}"
        )


def test_constructor_metadata_and_later_failure_boundaries() -> None:
    linear_decoder = Qwen35DecoderLayerShell(
        block_type="linear_attention",
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        mlp=nn.Identity(),
        linear_attention=nn.Identity(),
    )
    _expect_error(
        lambda: Qwen35PackedFullDecoderLayer(linear_decoder),
        ValueError,
        "full_attention",
    )

    for counts, positions, hidden, message in (
        ([], torch.arange(6), _hidden(), "token_counts"),
        ((2, 0, 4), torch.arange(6), _hidden(), "positive integers"),
        ((2, True, 3), torch.arange(6), _hidden(), "positive integers"),
        ((2, 1, 2), torch.arange(6), _hidden(), "sum"),
        ((2, 1, 3), torch.arange(5), _hidden(), "position_ids"),
        (
            (2, 1, 3),
            torch.zeros(2, 6, dtype=torch.int64),
            _hidden(),
            "one or three rows",
        ),
        (
            (2, 1, 3),
            torch.arange(6, dtype=torch.float32),
            _hidden(),
            "integer dtype",
        ),
        (
            (2, 1, 3),
            torch.arange(6),
            torch.ones(2, 3, 4),
            "rank two",
        ),
    ):
        _, _, wrapper = _fixture()
        _expect_error(
            lambda counts=counts, positions=positions,
            hidden=hidden: wrapper(counts, positions, hidden),
            ValueError,
            message,
        )

    _, attention, wrapper = _fixture()
    attention.fail_on_call = 2
    hidden_states = _hidden()
    original_hidden = hidden_states.clone()
    _expect_error(
        lambda: wrapper((2, 1, 3), torch.arange(6), hidden_states),
        RuntimeError,
        "full request failure",
    )
    torch.testing.assert_close(hidden_states, original_hidden)


def test_public_full_attention_builder_is_exported_from_components() -> None:
    components_path = (
        ROOT / "tinyvllm/models/qwen35_components.py"
    )
    source = components_path.read_text()
    assert "def build_qwen35_full_attention_decoder_layer(" in source
    assert source.count(
        "def build_qwen35_full_attention_decoder_layer("
    ) == 1


def main():
    test_segmented_full_attention_matches_oracle_and_call_lengths()
    test_other_request_changes_do_not_affect_first_request()
    test_position_row_shapes_and_bfloat16_noncontiguous()
    test_constructor_metadata_and_later_failure_boundaries()
    test_public_full_attention_builder_is_exported_from_components()
    print("qwen35 packed full decoder layer tests passed")


if __name__ == "__main__":
    main()
