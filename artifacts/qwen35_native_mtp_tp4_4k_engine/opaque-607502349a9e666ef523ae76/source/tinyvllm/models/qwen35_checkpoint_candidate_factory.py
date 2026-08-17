from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.engine.qwen35_hybrid_state import (
    build_qwen35_hybrid_state_layout,
)
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointTensorPlan,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    build_qwen35_checkpoint_binding_plan,
)
from tinyvllm.models.qwen35_components import (
    Qwen35ConcreteComponentAssembly,
    build_qwen35_concrete_component_assembly,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


@dataclass
class Qwen35PreparedCheckpointCandidateTarget:
    assembly: Qwen35ConcreteComponentAssembly
    binding_plan: Qwen35CheckpointBindingPlan
    pool: HybridStateTensorPool
    _consumed: bool = field(default=False, init=False, repr=False)

    def take(
        self,
    ) -> tuple[Qwen35PackedForCausalLM, Qwen35CheckpointBindingPlan]:
        if self._consumed:
            raise RuntimeError(
                "Qwen3.5 checkpoint candidate target already consumed"
            )
        self._consumed = True
        return self.assembly.packed.model, self.binding_plan


def _snapshot_pool(pool: HybridStateTensorPool):
    return (
        id(pool.layout),
        pool.capacity,
        pool.device,
        tuple(pool._bindings.items()),
        {
            key: (
                id(tensor),
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tensor._version,
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
            )
            for key, tensor in pool._tensors.items()
        },
    )


def _require_pool_unchanged(
    pool: HybridStateTensorPool,
    snapshot,
) -> None:
    (
        layout_id,
        capacity,
        device,
        bindings,
        tensors,
    ) = snapshot
    if (
        id(pool.layout) != layout_id
        or pool.capacity != capacity
        or pool.device != device
        or tuple(pool._bindings.items()) != bindings
        or set(pool._tensors) != set(tensors)
    ):
        raise RuntimeError(
            "checkpoint candidate preparation mutated the state pool"
        )
    for key, tensor in pool._tensors.items():
        (
            object_id,
            pointer,
            storage_offset,
            version,
            shape,
            dtype,
            tensor_device,
        ) = tensors[key]
        if (
            id(tensor) != object_id
            or tensor.untyped_storage().data_ptr() != pointer
            or tensor.storage_offset() != storage_offset
            or tensor._version != version
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != tensor_device
        ):
            raise RuntimeError(
                "checkpoint candidate preparation mutated state storage"
            )


def _require_exact_composition(
    assembly: Qwen35ConcreteComponentAssembly,
    binding_plan: Qwen35CheckpointBindingPlan,
    pool: HybridStateTensorPool,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> None:
    if assembly.packed.pool is not pool:
        raise RuntimeError(
            "checkpoint candidate assembly must retain the supplied pool"
        )
    if (
        assembly.tensor_parallel_size != tensor_parallel_size
        or assembly.tensor_parallel_rank != tensor_parallel_rank
    ):
        raise RuntimeError(
            "checkpoint candidate assembly TP context mismatch"
        )
    if (
        binding_plan.tensor_parallel_size != tensor_parallel_size
        or binding_plan.tensor_parallel_rank != tensor_parallel_rank
    ):
        raise RuntimeError(
            "checkpoint candidate binding TP context mismatch"
        )

    model = assembly.packed.model
    registered = dict(
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    )
    for binding in binding_plan.bindings:
        if registered.get(binding.destination_name) is not (
            binding.destination
        ):
            raise RuntimeError(
                "checkpoint binding destination is not model-registered"
            )
        if binding.destination.device != assembly.parameter_device:
            raise RuntimeError(
                "checkpoint binding destination device mismatch"
            )


def _expected_pool_layout(
    hf_config,
    pool: HybridStateTensorPool,
    tensor_parallel_size: int,
):
    config = getattr(hf_config, "text_config", hf_config)
    dtypes_by_role = {}
    for component in pool.layout.components:
        previous = dtypes_by_role.setdefault(
            component.role,
            component.dtype,
        )
        if previous != component.dtype:
            raise ValueError(
                "pool layout must use one dtype per state role"
            )
    convolution_widths = {
        component.shape[-1]
        for component in pool.layout.components
        if component.role == "linear_convolution"
    }
    if len(convolution_widths) != 1:
        raise ValueError(
            "pool layout must have one linear convolution state width"
        )
    if not hasattr(config, "linear_conv_kernel_dim"):
        raise ValueError(
            "missing Qwen3.5 config field: linear_conv_kernel_dim"
        )
    kernel_dim = config.linear_conv_kernel_dim
    if (
        isinstance(kernel_dim, bool)
        or not isinstance(kernel_dim, int)
        or kernel_dim <= 0
    ):
        raise ValueError(
            "linear_conv_kernel_dim must be a positive integer"
        )
    speculative_tokens = next(iter(convolution_widths)) - kernel_dim + 1
    return build_qwen35_hybrid_state_layout(
        hf_config,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtypes_by_role["linear_convolution"],
        recurrent_dtype=dtypes_by_role["linear_recurrent"],
        speculative_tokens=speculative_tokens,
    )


def prepare_qwen35_checkpoint_candidate_target(
    hf_config,
    tensor_plan: Qwen35CheckpointTensorPlan,
    *,
    pool: HybridStateTensorPool,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    build_attention_backend,
    parameter_device: str | torch.device = "meta",
) -> Qwen35PreparedCheckpointCandidateTarget:
    if type(pool) is not HybridStateTensorPool:
        raise ValueError("pool must be an exact HybridStateTensorPool")
    if type(tensor_plan) is not Qwen35CheckpointTensorPlan:
        raise ValueError(
            "tensor_plan must be an exact Qwen35CheckpointTensorPlan"
        )
    pool_snapshot = _snapshot_pool(pool)
    expected_layout = _expected_pool_layout(
        hf_config,
        pool,
        tensor_parallel_size,
    )
    if pool.layout.fingerprint != expected_layout.fingerprint:
        raise ValueError(
            "pool layout must match Qwen3.5 config and TP context"
        )
    assembly = build_qwen35_concrete_component_assembly(
        hf_config,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        build_attention_backend=build_attention_backend,
        parameter_device=parameter_device,
    )
    binding_plan = build_qwen35_checkpoint_binding_plan(
        assembly.packed.model,
        tensor_plan,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
    )
    _require_exact_composition(
        assembly,
        binding_plan,
        pool,
        tensor_parallel_size,
        tensor_parallel_rank,
    )
    _require_pool_unchanged(pool, pool_snapshot)
    return Qwen35PreparedCheckpointCandidateTarget(
        assembly=assembly,
        binding_plan=binding_plan,
        pool=pool,
    )
