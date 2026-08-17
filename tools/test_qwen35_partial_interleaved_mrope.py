import importlib.util
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tinyvllm/layers/qwen35_rotary_embedding.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_rotary_embedding_test_target", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rotary = _load_module()
Qwen35PartialInterleavedRotaryEmbedding = (
    rotary.Qwen35PartialInterleavedRotaryEmbedding
)


def _manual_selected_frequencies(
    position_ids: torch.Tensor,
    rotary_dim: int,
    base: float,
    mrope_section: tuple[int, int, int],
) -> torch.Tensor:
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).expand(3, -1)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32)
            / rotary_dim
        )
    )
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    selected = frequencies[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        length = mrope_section[axis] * 3
        selected[:, offset:length:3] = frequencies[
            axis, :, offset:length:3
        ]
    return selected


def _manual_apply(
    tensor: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    base: float,
    mrope_section: tuple[int, int, int],
) -> torch.Tensor:
    tokens = tensor.shape[0]
    heads = tensor.shape[1] // head_dim
    by_head = tensor.view(tokens, heads, head_dim)
    prefix = by_head[..., :rotary_dim].float()
    suffix = by_head[..., rotary_dim:]
    frequencies = _manual_selected_frequencies(
        position_ids,
        rotary_dim,
        base,
        mrope_section,
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().unsqueeze(1)
    sin = embedding.sin().unsqueeze(1)
    first, second = prefix.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    rotated = prefix * cos + rotated_half * sin
    return torch.cat((rotated.to(tensor.dtype), suffix), dim=-1).reshape(
        tensor.shape
    )


def _manual_apply_reference_dtype(
    tensor: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    base: float,
    mrope_section: tuple[int, int, int],
) -> torch.Tensor:
    tokens = tensor.shape[0]
    heads = tensor.shape[1] // head_dim
    by_head = tensor.view(tokens, heads, head_dim)
    prefix = by_head[..., :rotary_dim]
    suffix = by_head[..., rotary_dim:]
    frequencies = _manual_selected_frequencies(
        position_ids,
        rotary_dim,
        base,
        mrope_section,
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(tensor.dtype).unsqueeze(1)
    sin = embedding.sin().to(tensor.dtype).unsqueeze(1)
    first, second = prefix.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    rotated = prefix * cos + rotated_half * sin
    return torch.cat((rotated, suffix), dim=-1).reshape(tensor.shape)


def _new_layer() -> Qwen35PartialInterleavedRotaryEmbedding:
    return Qwen35PartialInterleavedRotaryEmbedding(
        head_dim=12,
        rotary_dim=8,
        base=100.0,
        mrope_section=(2, 1, 1),
    )


def test_explicit_axes_match_manual_interleaved_partial_oracle() -> None:
    position_ids = torch.tensor(
        [[1, 2], [10, 20], [100, 200]],
        dtype=torch.int64,
    )
    query = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 7
    key = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 5
    layer = _new_layer()
    query_original = query.clone()
    key_original = key.clone()
    position_original = position_ids.clone()

    actual_query, actual_key = layer(position_ids, query, key)
    expected_query = _manual_apply(
        query, position_ids, 12, 8, 100.0, (2, 1, 1)
    )
    expected_key = _manual_apply(
        key, position_ids, 12, 8, 100.0, (2, 1, 1)
    )
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    torch.testing.assert_close(query, query_original)
    torch.testing.assert_close(key, key_original)
    torch.testing.assert_close(position_ids, position_original)
    torch.testing.assert_close(actual_query[:, 8:12], query[:, 8:12])
    torch.testing.assert_close(actual_query[:, 20:24], query[:, 20:24])
    torch.testing.assert_close(actual_key[:, 8:12], key[:, 8:12])

    selected = _manual_selected_frequencies(
        position_ids, 8, 100.0, (2, 1, 1)
    )
    all_frequencies = (
        position_ids.float().unsqueeze(-1)
        * layer.inv_freq.float().view(1, 1, -1)
    )
    torch.testing.assert_close(selected[:, 0], all_frequencies[0, :, 0])
    torch.testing.assert_close(selected[:, 1], all_frequencies[1, :, 1])
    torch.testing.assert_close(selected[:, 2], all_frequencies[2, :, 2])
    torch.testing.assert_close(selected[:, 3], all_frequencies[0, :, 3])


def test_text_positions_equal_replicated_three_axis_positions() -> None:
    positions = torch.tensor([0, 3, 7], dtype=torch.int64)
    explicit = positions.unsqueeze(0).expand(3, -1).clone()
    query = torch.arange(72, dtype=torch.float32).reshape(3, 24) / 13
    key = torch.arange(36, dtype=torch.float32).reshape(3, 12) / 11
    layer = _new_layer()
    text_query, text_key = layer(positions, query, key)
    explicit_query, explicit_key = layer(explicit, query, key)
    torch.testing.assert_close(text_query, explicit_query)
    torch.testing.assert_close(text_key, explicit_key)


def test_zero_position_is_identity_and_supports_different_head_counts() -> None:
    positions = torch.zeros(4, dtype=torch.int64)
    query = torch.randn(4, 36)
    key = torch.randn(4, 24)
    actual_query, actual_key = _new_layer()(positions, query, key)
    torch.testing.assert_close(actual_query, query)
    torch.testing.assert_close(actual_key, key)


def test_bfloat16_matches_official_reference_dtype_arithmetic() -> None:
    positions = torch.tensor(
        [[2, 257], [3, 509], [7, 1021]],
        dtype=torch.int64,
    )
    query = (
        torch.arange(48, dtype=torch.float32).reshape(2, 24) / 7
    ).to(torch.bfloat16)
    key = (
        torch.arange(24, dtype=torch.float32).reshape(2, 12) / 5
    ).to(torch.bfloat16)
    actual_query, actual_key = _new_layer()(positions, query, key)
    expected_query = _manual_apply_reference_dtype(
        query, positions, 12, 8, 100.0, (2, 1, 1)
    )
    expected_key = _manual_apply_reference_dtype(
        key, positions, 12, 8, 100.0, (2, 1, 1)
    )
    assert actual_query.dtype == torch.bfloat16
    assert actual_key.dtype == torch.bfloat16
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_constructor_rejects_invalid_contracts() -> None:
    valid = dict(
        head_dim=12,
        rotary_dim=8,
        base=100.0,
        mrope_section=(2, 1, 1),
    )
    for name, values in {
        "head_dim": (True, 0, -1, 12.5),
        "rotary_dim": (True, 0, -2, 7, 14, 8.5),
    }.items():
        for value in values:
            kwargs = dict(valid)
            kwargs[name] = value
            _expect_value_error(
                lambda kwargs=kwargs: Qwen35PartialInterleavedRotaryEmbedding(
                    **kwargs
                ),
                name,
            )
    for base in (1.0, 0.0, -2.0, float("inf"), float("nan")):
        kwargs = dict(valid)
        kwargs["base"] = base
        _expect_value_error(
            lambda kwargs=kwargs: Qwen35PartialInterleavedRotaryEmbedding(
                **kwargs
            ),
            "base",
        )
    for section in (
        (2, 1),
        (2, 1, 1, 1),
        (2, 0, 2),
        (2, -1, 3),
        (2, True, 1),
        (2, 1.5, 1),
        (1, 1, 1),
    ):
        kwargs = dict(valid)
        kwargs["mrope_section"] = section
        _expect_value_error(
            lambda kwargs=kwargs: Qwen35PartialInterleavedRotaryEmbedding(
                **kwargs
            ),
            "mrope_section",
        )


def test_forward_rejects_invalid_contracts() -> None:
    layer = _new_layer()
    valid_positions = torch.tensor([0, 1], dtype=torch.int64)
    valid_query = torch.ones(2, 24)
    valid_key = torch.ones(2, 12)
    cases = (
        (
            torch.ones(1, 2, 1, dtype=torch.int64),
            valid_query,
            valid_key,
            "position_ids",
        ),
        (
            torch.ones(2, 2, dtype=torch.int64),
            valid_query,
            valid_key,
            "position_ids",
        ),
        (
            torch.ones(3, 3, dtype=torch.int64),
            valid_query,
            valid_key,
            "token",
        ),
        (
            torch.ones(2, dtype=torch.float32),
            valid_query,
            valid_key,
            "integer",
        ),
        (
            valid_positions,
            torch.ones(2, 2, 12),
            valid_key,
            "rank two",
        ),
        (
            valid_positions,
            valid_query,
            torch.ones(3, 12),
            "token",
        ),
        (
            valid_positions,
            torch.ones(2, 25),
            valid_key,
            "multiple",
        ),
        (
            valid_positions,
            torch.ones(2, 24, dtype=torch.int64),
            valid_key,
            "floating point",
        ),
        (
            valid_positions,
            valid_query,
            valid_key.to(torch.float64),
            "dtype",
        ),
    )
    for position_ids, query, key, message in cases:
        _expect_value_error(
            lambda position_ids=position_ids, query=query, key=key: layer(
                position_ids, query, key
            ),
            message,
        )

    _expect_value_error(
        lambda: layer(
            valid_positions.to("meta"),
            valid_query,
            valid_key,
        ),
        "device",
    )
    _expect_value_error(
        lambda: layer(
            valid_positions,
            valid_query.to("meta"),
            valid_key,
        ),
        "device",
    )


def test_registered_inverse_frequency_matches_formula() -> None:
    layer = _new_layer()
    expected = 1.0 / (
        100.0
        ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8)
    )
    torch.testing.assert_close(layer.inv_freq, expected)
    assert layer.inv_freq.dtype == torch.float32
    assert math.isclose(layer.base, 100.0)


def test_module_to_moves_registered_inverse_frequency() -> None:
    layer = _new_layer()

    moved = layer.to("meta")

    assert moved is layer
    assert layer.inv_freq.device.type == "meta"
    assert layer.inv_freq.shape == (4,)
    assert layer.inv_freq.dtype == torch.float32


def main() -> None:
    test_explicit_axes_match_manual_interleaved_partial_oracle()
    test_text_positions_equal_replicated_three_axis_positions()
    test_zero_position_is_identity_and_supports_different_head_counts()
    test_bfloat16_matches_official_reference_dtype_arithmetic()
    test_constructor_rejects_invalid_contracts()
    test_forward_rejects_invalid_contracts()
    test_registered_inverse_frequency_matches_formula()
    test_module_to_moves_registered_inverse_frequency()
    print("qwen35 partial interleaved mrope tests passed")


if __name__ == "__main__":
    main()
