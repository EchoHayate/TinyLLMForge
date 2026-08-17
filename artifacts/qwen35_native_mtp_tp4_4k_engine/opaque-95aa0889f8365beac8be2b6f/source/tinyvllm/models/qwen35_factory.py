from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.engine.qwen35_hybrid_state import _layer_types
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell
from tinyvllm.layers.qwen35_packed_layer_stack import (
    Qwen35PackedHeterogeneousLayerStack,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


@dataclass(frozen=True)
class Qwen35PackedModelAssembly:
    model: Qwen35PackedForCausalLM
    layer_stack: Qwen35PackedHeterogeneousLayerStack
    state_transaction: Qwen35CrossLayerStateTransaction
    adapters: tuple[Qwen35LayerStateAdapter, ...]
    pool: HybridStateTensorPool


def _config_integer(config, field_name: str) -> int:
    if not hasattr(config, field_name):
        raise ValueError(f"missing Qwen3.5 config field: {field_name}")
    value = getattr(config, field_name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _pool_linear_layer_indices(
    pool: HybridStateTensorPool,
) -> tuple[int, ...]:
    roles_by_layer = {}
    for component in pool.layout.components:
        roles_by_layer.setdefault(component.layer_index, set()).add(
            component.role
        )
    expected_roles = {
        "linear_convolution",
        "linear_recurrent",
    }
    if any(
        roles != expected_roles
        for roles in roles_by_layer.values()
    ):
        raise ValueError(
            "pool linear layers must contain convolution and recurrent state"
        )
    return tuple(sorted(roles_by_layer))


def assemble_qwen35_packed_model(
    hf_config,
    *,
    pool: HybridStateTensorPool,
    embed_tokens: nn.Module,
    final_norm: nn.Module,
    lm_head: nn.Module,
    build_decoder_layer: Callable,
) -> Qwen35PackedModelAssembly:
    if type(pool) is not HybridStateTensorPool:
        raise ValueError("pool must be an exact HybridStateTensorPool")
    for name, module in (
        ("embed_tokens", embed_tokens),
        ("final_norm", final_norm),
        ("lm_head", lm_head),
    ):
        if not isinstance(module, nn.Module):
            raise ValueError(f"{name} must be a module")
    if not callable(build_decoder_layer):
        raise ValueError("build_decoder_layer must be callable")

    config = getattr(hf_config, "text_config", hf_config)
    num_hidden_layers = _config_integer(
        config,
        "num_hidden_layers",
    )
    layer_types = _layer_types(config, num_hidden_layers)
    linear_indices = tuple(
        layer_index
        for layer_index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    )
    if _pool_linear_layer_indices(pool) != linear_indices:
        raise ValueError(
            "pool linear layer indices must match config layer types"
        )

    adapters = tuple(
        Qwen35LayerStateAdapter(pool, layer_index)
        for layer_index in linear_indices
    )
    adapter_by_layer = {
        adapter.layer_index: adapter
        for adapter in adapters
    }
    layers = []
    for layer_index, layer_type in enumerate(layer_types):
        adapter = adapter_by_layer.get(layer_index)
        layer = build_decoder_layer(
            layer_index,
            layer_type,
            adapter,
        )
        if type(layer) is not Qwen35DecoderLayerShell:
            raise ValueError(
                "build_decoder_layer must return an exact "
                "Qwen35DecoderLayerShell"
            )
        if layer.block_type != layer_type:
            raise ValueError(
                "decoder layer block type must match config"
            )
        layers.append(layer)

    transaction = Qwen35CrossLayerStateTransaction(adapters)
    layer_stack = Qwen35PackedHeterogeneousLayerStack(
        tuple(layers),
        transaction,
    )
    model = Qwen35PackedForCausalLM(
        embed_tokens,
        layer_stack,
        final_norm,
        lm_head,
    )
    return Qwen35PackedModelAssembly(
        model=model,
        layer_stack=layer_stack,
        state_transaction=transaction,
        adapters=adapters,
        pool=pool,
    )
