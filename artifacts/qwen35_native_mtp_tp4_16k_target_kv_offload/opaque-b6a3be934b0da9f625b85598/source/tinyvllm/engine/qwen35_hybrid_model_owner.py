from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.hybrid_state import (
    HybridStateRuntimeBridge,
    HybridStateTensorPool,
)
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.layers.qwen35_packed_layer_stack import (
    Qwen35PackedHeterogeneousLayerStack,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


@dataclass(frozen=True)
class Qwen35HybridModelOwner:
    model: Qwen35PackedForCausalLM
    layer_stack: Qwen35PackedHeterogeneousLayerStack
    state_transaction: Qwen35CrossLayerStateTransaction
    pool: HybridStateTensorPool
    runtime_bridge: HybridStateRuntimeBridge


def build_qwen35_hybrid_model_owner(model):
    if type(model) is not Qwen35PackedForCausalLM:
        raise ValueError(
            "model must be an exact packed Qwen3.5 root model"
        )
    layer_stack = model.layer_stack
    if type(layer_stack) is not Qwen35PackedHeterogeneousLayerStack:
        raise ValueError(
            "packed Qwen3.5 root layer stack is invalid"
        )
    transaction = layer_stack.state_transaction
    if type(transaction) is not Qwen35CrossLayerStateTransaction:
        raise ValueError(
            "packed Qwen3.5 model state transaction is invalid"
        )
    adapter_indices = tuple(
        adapter.layer_index
        for adapter in transaction.adapters
    )
    if adapter_indices != layer_stack.linear_indices:
        raise ValueError(
            "packed Qwen3.5 model transaction is misaligned"
        )
    pool = transaction.pool
    if not isinstance(pool, HybridStateTensorPool):
        raise ValueError(
            "packed Qwen3.5 model state pool is invalid"
        )
    if any(adapter.pool is not pool for adapter in transaction.adapters):
        raise ValueError(
            "packed Qwen3.5 model adapters do not share one pool"
        )
    runtime_bridge = HybridStateRuntimeBridge(pool)
    return Qwen35HybridModelOwner(
        model=model,
        layer_stack=layer_stack,
        state_transaction=transaction,
        pool=pool,
        runtime_bridge=runtime_bridge,
    )
