from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    Qwen35CheckpointTensorBinding,
)


@dataclass(frozen=True)
class Qwen35CheckpointAssignmentResult:
    assigned_bindings: int
    unique_destinations: int
    source_tensors: int


@dataclass(frozen=True)
class _AssignmentOperation:
    binding: Qwen35CheckpointTensorBinding
    source: torch.Tensor
    transformed: torch.Tensor
    local_tensor: torch.Tensor | None


_CHECKPOINT_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def _validate_tp_context(
    tensor_parallel_size,
    tensor_parallel_rank,
) -> tuple[int, int]:
    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size <= 0
    ):
        raise ValueError(
            "tensor_parallel_size must be a positive integer"
        )
    if (
        isinstance(tensor_parallel_rank, bool)
        or not isinstance(tensor_parallel_rank, int)
        or tensor_parallel_rank < 0
        or tensor_parallel_rank >= tensor_parallel_size
    ):
        raise ValueError(
            "tensor_parallel_rank must be in "
            "[0, tensor_parallel_size)"
        )
    return tensor_parallel_size, tensor_parallel_rank


def _transform_source(
    binding: Qwen35CheckpointTensorBinding,
    source: torch.Tensor,
) -> torch.Tensor:
    transform = binding.load.transform
    if transform == "identity":
        return source
    if transform == "squeeze_conv_channel":
        if source.ndim != 3 or source.shape[1] != 1:
            raise ValueError(
                "squeeze_conv_channel requires "
                "[channels, 1, kernel] source shape"
            )
        return source.squeeze(1)
    raise ValueError(
        f"unsupported checkpoint transform: {transform}"
    )


def _direct_buffer_local(
    binding: Qwen35CheckpointTensorBinding,
    transformed: torch.Tensor,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> torch.Tensor:
    target = binding.load.weight.target
    if target.endswith("linear_attention.norm_weight"):
        local = transformed
    elif binding.source_segments is not None:
        source_segments = binding.source_segments
        if (
            not source_segments
            or any(
                isinstance(segment, bool)
                or not isinstance(segment, int)
                or segment <= 0
                or segment % tensor_parallel_size != 0
                for segment in source_segments
            )
            or sum(source_segments) != transformed.shape[0]
        ):
            raise ValueError(
                f"{target} source segments are invalid"
            )
        local_segments = []
        global_offset = 0
        for global_rows in source_segments:
            local_rows = global_rows // tensor_parallel_size
            local_segments.append(
                transformed.narrow(
                    0,
                    global_offset
                    + tensor_parallel_rank * local_rows,
                    local_rows,
                )
            )
            global_offset += global_rows
        local = torch.cat(local_segments, dim=0)
    else:
        if transformed.ndim == 0:
            raise ValueError(
                f"{target} direct buffer source must have rank"
            )
        if transformed.shape[0] % tensor_parallel_size != 0:
            raise ValueError(
                f"{target} direct buffer source must be TP-divisible"
            )
        local_rows = transformed.shape[0] // tensor_parallel_size
        local = transformed.narrow(
            0,
            tensor_parallel_rank * local_rows,
            local_rows,
        )
    if tuple(local.shape) != binding.local_shape:
        raise ValueError(
            f"{target} direct buffer local shape "
            f"{tuple(local.shape)} must match {binding.local_shape}"
        )
    return local


def _validate_source(
    binding: Qwen35CheckpointTensorBinding,
    source,
) -> torch.Tensor:
    source_name = binding.load.weight.source.name
    if not isinstance(source, torch.Tensor):
        raise ValueError(
            f"checkpoint source must be a tensor: {source_name}"
        )
    if source.device.type != "cpu":
        raise ValueError(
            f"checkpoint source must be a CPU tensor: {source_name}"
        )
    expected_dtype = _CHECKPOINT_DTYPES.get(
        binding.load.metadata.dtype
    )
    if expected_dtype is None:
        raise ValueError(
            "unsupported checkpoint assignment dtype: "
            f"{binding.load.metadata.dtype}"
        )
    if source.dtype != expected_dtype:
        raise ValueError(
            f"checkpoint source dtype must match metadata: {source_name}"
        )
    if tuple(source.shape) != binding.load.metadata.shape:
        raise ValueError(
            f"checkpoint source shape must match metadata: {source_name}"
        )
    return source


def _prepare_operation(
    binding: Qwen35CheckpointTensorBinding,
    source,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> _AssignmentOperation:
    if type(binding) is not Qwen35CheckpointTensorBinding:
        raise ValueError(
            "binding plan entries must be exact "
            "Qwen35CheckpointTensorBinding values"
        )
    destination = binding.destination
    target = binding.load.weight.target
    if not isinstance(destination, torch.Tensor):
        raise ValueError(f"{target} destination must be a tensor")
    if destination.device.type != "cpu":
        raise ValueError(
            f"{target} destination must be a CPU tensor"
        )
    source = _validate_source(binding, source)
    transformed = _transform_source(binding, source)
    if target.endswith("linear_attention.norm_weight"):
        transformed = transformed.to(destination.dtype)
    local_tensor = None

    if binding.loader_kind == "custom_parameter_loader":
        loader = getattr(destination, "weight_loader", None)
        if not callable(loader):
            raise ValueError(
                f"{target} requires a callable custom loader"
            )
        if target.endswith("mlp.gate_up_proj.weight"):
            if binding.load.weight.packed_slot not in (0, 1):
                raise ValueError(
                    f"{target} requires packed slot 0 or 1"
                )
        elif binding.load.weight.packed_slot is not None:
            raise ValueError(
                f"{target} must not define a packed slot"
            )
        if tuple(transformed.shape) != tuple(
            binding.load.metadata.shape
        ):
            raise ValueError(
                f"{target} transformed source shape is invalid"
            )
    elif binding.loader_kind == "default_parameter_copy":
        if tuple(transformed.shape) != tuple(destination.shape):
            raise ValueError(
                f"{target} default copy shape must match destination"
            )
    elif binding.loader_kind == "direct_buffer_copy":
        local_tensor = _direct_buffer_local(
            binding,
            transformed,
            tensor_parallel_size,
            tensor_parallel_rank,
        )
        if tuple(local_tensor.shape) != tuple(destination.shape):
            raise ValueError(
                f"{target} buffer copy shape must match destination"
            )
    else:
        raise ValueError(
            f"unsupported loader kind: {binding.loader_kind}"
        )

    if transformed.dtype != destination.dtype:
        raise ValueError(
            f"{target} transformed dtype must match destination"
        )
    return _AssignmentOperation(
        binding=binding,
        source=source,
        transformed=transformed,
        local_tensor=local_tensor,
    )


def _execute_operations(
    operations: tuple[_AssignmentOperation, ...],
) -> int:
    current = None
    try:
        with torch.no_grad():
            for operation in operations:
                current = operation
                binding = operation.binding
                destination = binding.destination
                if binding.loader_kind == "custom_parameter_loader":
                    loader = destination.weight_loader
                    packed_slot = binding.load.weight.packed_slot
                    if packed_slot is None:
                        loader(destination, operation.transformed)
                    else:
                        loader(
                            destination,
                            operation.transformed,
                            packed_slot,
                        )
                elif binding.loader_kind == "default_parameter_copy":
                    destination.copy_(operation.transformed)
                else:
                    destination.copy_(operation.local_tensor)
    except Exception as error:
        source_name = current.binding.load.weight.source.name
        target = current.binding.load.weight.target
        raise RuntimeError(
            "Qwen3.5 checkpoint source assignment failed for "
            f"{source_name} -> {target}: {error}"
        ) from error
    return len(operations)


def _assign_qwen35_checkpoint_source_bindings(
    bindings: tuple[Qwen35CheckpointTensorBinding, ...],
    source: torch.Tensor,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> int:
    if not isinstance(bindings, tuple) or not bindings:
        raise ValueError("bindings must be a non-empty tuple")
    tensor_parallel_size, tensor_parallel_rank = (
        _validate_tp_context(
            tensor_parallel_size,
            tensor_parallel_rank,
        )
    )
    first = bindings[0]
    if type(first) is not Qwen35CheckpointTensorBinding:
        raise ValueError(
            "bindings must contain exact "
            "Qwen35CheckpointTensorBinding values"
        )
    source_name = first.load.weight.source.name
    metadata = first.load.metadata
    for binding in bindings:
        if type(binding) is not Qwen35CheckpointTensorBinding:
            raise ValueError(
                "bindings must contain exact "
                "Qwen35CheckpointTensorBinding values"
            )
        if (
            binding.load.weight.source.name != source_name
            or binding.load.metadata != metadata
        ):
            raise ValueError(
                "bindings must describe one checkpoint source"
            )
    operations = tuple(
        _prepare_operation(
            binding,
            source,
            tensor_parallel_size,
            tensor_parallel_rank,
        )
        for binding in bindings
    )
    return _execute_operations(operations)


def _prepare_operations(
    binding_plan: Qwen35CheckpointBindingPlan,
    source_tensors: Mapping[str, torch.Tensor],
) -> tuple[_AssignmentOperation, ...]:
    expected_sources = {
        binding.load.weight.source.name
        for binding in binding_plan.bindings
    }
    if set(source_tensors) != expected_sources:
        raise ValueError(
            "checkpoint source coverage must exactly match binding plan"
        )
    return tuple(
        _prepare_operation(
            binding,
            source_tensors[binding.load.weight.source.name],
            binding_plan.tensor_parallel_size,
            binding_plan.tensor_parallel_rank,
        )
        for binding in binding_plan.bindings
    )


def _restore_destinations(
    snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    first_error = None
    with torch.no_grad():
        for destination, snapshot in snapshots.values():
            try:
                destination.copy_(snapshot)
            except Exception as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise RuntimeError(
            "Qwen3.5 checkpoint assignment rollback failed"
        ) from first_error


def assign_qwen35_checkpoint_tensors(
    binding_plan: Qwen35CheckpointBindingPlan,
    source_tensors: Mapping[str, torch.Tensor],
) -> Qwen35CheckpointAssignmentResult:
    if type(binding_plan) is not Qwen35CheckpointBindingPlan:
        raise ValueError(
            "binding_plan must be an exact Qwen35CheckpointBindingPlan"
        )
    if not isinstance(source_tensors, Mapping):
        raise ValueError("source_tensors must be a mapping")
    operations = _prepare_operations(binding_plan, source_tensors)

    snapshots = {}
    for operation in operations:
        destination = operation.binding.destination
        snapshots.setdefault(
            id(destination),
            (destination, destination.detach().clone()),
        )

    try:
        _execute_operations(operations)
    except Exception as error:
        try:
            _restore_destinations(snapshots)
        except Exception as rollback_error:
            raise RuntimeError(
                "Qwen3.5 checkpoint assignment failed and rollback failed"
            ) from rollback_error
        raise RuntimeError(
            f"Qwen3.5 checkpoint assignment failed: {error}"
        ) from error

    return Qwen35CheckpointAssignmentResult(
        assigned_bindings=len(operations),
        unique_destinations=len(snapshots),
        source_tensors=len(source_tensors),
    )
