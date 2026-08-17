from __future__ import annotations

import torch

from tinyvllm.engine.hybrid_state import (
    HybridStateComponentSpec,
    HybridStateLayout,
)


_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}
_SUPPORTED_LAYER_TYPES = {
    "linear_attention",
    "full_attention",
}


def build_qwen35_hybrid_state_layout(
    hf_config,
    *,
    tensor_parallel_size: int,
    dtype: torch.dtype,
    recurrent_dtype: torch.dtype | None = None,
    speculative_tokens: int = 1,
) -> HybridStateLayout:
    config = getattr(hf_config, "text_config", hf_config)
    tensor_parallel_size = _positive_integer(
        tensor_parallel_size,
        "tensor_parallel_size",
    )
    if tensor_parallel_size > 8:
        raise ValueError("tensor_parallel_size must not exceed 8")
    speculative_tokens = _positive_integer(
        speculative_tokens,
        "speculative_tokens",
    )
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported Qwen3.5 hybrid state dtype: {dtype}")
    if recurrent_dtype is None:
        recurrent_dtype = dtype
    if recurrent_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "unsupported Qwen3.5 recurrent state dtype: "
            f"{recurrent_dtype}"
        )

    num_hidden_layers = _config_integer(
        config,
        "num_hidden_layers",
    )
    layer_types = _layer_types(config, num_hidden_layers)
    key_heads = _config_integer(config, "linear_num_key_heads")
    value_heads = _config_integer(config, "linear_num_value_heads")
    key_head_dim = _config_integer(config, "linear_key_head_dim")
    value_head_dim = _config_integer(config, "linear_value_head_dim")
    conv_kernel_dim = _config_integer(config, "linear_conv_kernel_dim")

    if key_heads % tensor_parallel_size != 0:
        raise ValueError(
            "linear_num_key_heads must be divisible by tensor_parallel_size"
        )
    if value_heads % tensor_parallel_size != 0:
        raise ValueError(
            "linear_num_value_heads must be divisible by tensor_parallel_size"
        )
    conv_channels = (
        key_head_dim * key_heads * 2
        + value_head_dim * value_heads
    )
    if conv_channels % tensor_parallel_size != 0:
        raise ValueError(
            "linear convolution channels must be divisible by "
            "tensor_parallel_size"
        )

    convolution_shape = (
        conv_channels // tensor_parallel_size,
        conv_kernel_dim - 1 + speculative_tokens,
    )
    recurrent_shape = (
        value_heads // tensor_parallel_size,
        value_head_dim,
        key_head_dim,
    )
    components = []
    for layer_index, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        components.extend((
            HybridStateComponentSpec(
                layer_index=layer_index,
                role="linear_convolution",
                shape=convolution_shape,
                dtype=dtype,
            ),
            HybridStateComponentSpec(
                layer_index=layer_index,
                role="linear_recurrent",
                shape=recurrent_shape,
                dtype=recurrent_dtype,
            ),
        ))
    if not components:
        raise ValueError(
            "Qwen3.5 hybrid state layout requires linear-attention layers"
        )
    return HybridStateLayout(tuple(components))


def _config_integer(config, field_name: str) -> int:
    if not hasattr(config, field_name):
        raise ValueError(f"missing Qwen3.5 config field: {field_name}")
    return _positive_integer(getattr(config, field_name), field_name)


def _positive_integer(value, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _layer_types(config, num_hidden_layers: int) -> tuple[str, ...]:
    if not hasattr(config, "layer_types"):
        raise ValueError("missing Qwen3.5 config field: layer_types")
    values = getattr(config, "layer_types")
    if not isinstance(values, (list, tuple)):
        raise ValueError("layer_types must be a list or tuple")
    if len(values) != num_hidden_layers:
        raise ValueError(
            "layer_types length must match num_hidden_layers"
        )
    normalized = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("layer_types entries must be strings")
        layer_type = value.strip().lower()
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            raise ValueError(
                f"unsupported Qwen3.5 layer type: {value}"
            )
        normalized.append(layer_type)
    return tuple(normalized)
