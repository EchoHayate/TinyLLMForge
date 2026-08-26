from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tinyvllm.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tinyvllm.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedColumnParallelLinear,
    ReplicatedHeadPairedColumnParallelLinear,
    ReplicatedKVHeadParallelLinear,
    ReplicatedLocalOutputLinear,
    ReplicatedLinear,
    ReplicatedMergedColumnParallelLinear,
    ReplicatedSegmentedColumnParallelLinear,
    RowParallelLinear,
)
from tinyvllm.layers.qwen35_full_attention import Qwen35FullAttentionShell
from tinyvllm.layers.qwen35_linear_attention import (
    Qwen35LinearAttentionShell,
)
from tinyvllm.layers.qwen35_primitives import Qwen35OffsetRMSNorm
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointTensorLoad,
    Qwen35CheckpointTensorPlan,
)
from tinyvllm.models.qwen35_packed import Qwen35PackedForCausalLM


@dataclass(frozen=True)
class Qwen35CheckpointTensorBinding:
    load: Qwen35CheckpointTensorLoad
    destination_name: str
    destination: torch.Tensor
    destination_kind: str
    loader_kind: str
    local_shape: tuple[int, ...]
    destination_slice: tuple[int, int] | None
    source_segments: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Qwen35CheckpointBindingPlan:
    bindings: tuple[Qwen35CheckpointTensorBinding, ...]
    tensor_parallel_size: int
    tensor_parallel_rank: int


_CHECKPOINT_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}
_ROOT_TARGETS = {
    "embed_tokens.weight": "embed_tokens.weight",
    "final_norm.weight": "final_norm.weight",
    "lm_head.weight": "lm_head.weight",
}
_REPLICATED_TARGET_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "full_attention.q_norm.weight",
    "full_attention.k_norm.weight",
    "linear_attention.norm_weight",
    "linear_attention.in_proj_b.weight",
    "linear_attention.in_proj_a.weight",
    "linear_attention.in_proj_qkv.weight",
    "linear_attention.in_proj_z.weight",
    "full_attention.q_projection.weight",
    "full_attention.k_projection.weight",
    "full_attention.v_projection.weight",
    "mlp.down_proj.weight",
)
_OFFSET_NORM_TARGET_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "full_attention.q_norm.weight",
    "full_attention.k_norm.weight",
)
_AXIS_ZERO_TARGET_SUFFIXES = (
    "linear_attention.conv_weight",
    "linear_attention.A_log",
    "linear_attention.dt_bias",
)
_AXIS_ONE_TARGET_SUFFIXES = (
    "mlp.down_proj.weight",
    "linear_attention.out_proj.weight",
    "full_attention.output_projection.weight",
)


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_tp_context(
    tensor_parallel_size,
    tensor_parallel_rank,
) -> tuple[int, int]:
    size = _positive_integer(
        tensor_parallel_size,
        "tensor_parallel_size",
    )
    if (
        isinstance(tensor_parallel_rank, bool)
        or not isinstance(tensor_parallel_rank, int)
        or tensor_parallel_rank < 0
        or tensor_parallel_rank >= size
    ):
        raise ValueError(
            "tensor_parallel_rank must be in "
            "[0, tensor_parallel_size)"
        )
    return size, tensor_parallel_rank


def _resolve_target(
    model: Qwen35PackedForCausalLM,
    target: str,
) -> tuple[str, nn.Module, str, torch.Tensor, str]:
    if not isinstance(target, str) or not target:
        raise ValueError("checkpoint target must be a non-empty string")
    if target in _ROOT_TARGETS:
        destination_name = _ROOT_TARGETS[target]
        parts = destination_name.split(".")
        parent = model
    elif target.startswith("layers."):
        parts = target.split(".")
        if len(parts) < 4:
            raise ValueError(f"malformed layer checkpoint target: {target}")
        try:
            layer_index = int(parts[1])
        except ValueError as error:
            raise ValueError(
                f"malformed layer checkpoint target: {target}"
            ) from error
        if (
            layer_index < 0
            or layer_index >= len(model.layer_stack.layers)
        ):
            raise ValueError(
                f"checkpoint target layer index is out of range: {target}"
            )
        parent = model.layer_stack.layers[layer_index]
        destination_name = "layer_stack." + target
        parts = parts[2:]
    else:
        raise ValueError(f"unsupported checkpoint target: {target}")

    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise ValueError(f"missing checkpoint destination: {target}")
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise ValueError(
                f"checkpoint destination parent must be a module: {target}"
            )
    leaf = parts[-1]
    parameter = parent._parameters.get(leaf)
    buffer = parent._buffers.get(leaf)
    if parameter is not None and buffer is not None:
        raise ValueError(f"ambiguous checkpoint destination: {target}")
    if parameter is not None:
        return destination_name, parent, leaf, parameter, "parameter"
    if buffer is not None:
        return destination_name, parent, leaf, buffer, "buffer"
    raise ValueError(
        f"missing registered parameter or buffer destination: {target}"
    )


def _require_exact_component(
    target: str,
    parent: nn.Module,
    destination_kind: str,
) -> None:
    if target == "embed_tokens.weight":
        expected_type = VocabParallelEmbedding
    elif target == "lm_head.weight":
        expected_type = ParallelLMHead
    elif (
        target == "final_norm.weight"
        or target.endswith(_OFFSET_NORM_TARGET_SUFFIXES)
    ):
        expected_type = Qwen35OffsetRMSNorm
    elif target.endswith("mlp.gate_up_proj.weight"):
        expected_type = ReplicatedMergedColumnParallelLinear
    elif target.endswith("mlp.down_proj.weight"):
        expected_type = ReplicatedLinear
    elif target.endswith("linear_attention.out_proj.weight"):
        expected_type = RowParallelLinear
    elif target.endswith("full_attention.output_projection.weight"):
        expected_type = RowParallelLinear
    elif target.endswith(_AXIS_ONE_TARGET_SUFFIXES):
        expected_type = RowParallelLinear
    elif target.endswith("linear_attention.in_proj_qkv.weight"):
        expected_type = ReplicatedSegmentedColumnParallelLinear
    elif target.endswith("full_attention.q_projection.weight"):
        expected_type = ReplicatedHeadPairedColumnParallelLinear
    elif target.endswith("linear_attention.in_proj_z.weight"):
        expected_type = ReplicatedColumnParallelLinear
    elif target.endswith((
        "linear_attention.in_proj_b.weight",
        "linear_attention.in_proj_a.weight",
    )):
        expected_type = ReplicatedLocalOutputLinear
    elif target.endswith((
        "full_attention.k_projection.weight",
        "full_attention.v_projection.weight",
    )):
        expected_type = ReplicatedKVHeadParallelLinear
    elif target.endswith((
        "linear_attention.conv_weight",
        "linear_attention.A_log",
        "linear_attention.dt_bias",
        "linear_attention.norm_weight",
    )):
        expected_type = Qwen35LinearAttentionShell
    else:
        raise ValueError(f"unsupported checkpoint binding target: {target}")

    if type(parent) is not expected_type:
        raise ValueError(
            f"{target} parent must be exact {expected_type.__name__}"
        )
    expected_kind = (
        "buffer"
        if expected_type is Qwen35LinearAttentionShell
        else "parameter"
    )
    if destination_kind != expected_kind:
        raise ValueError(
            f"{target} must resolve to a registered {expected_kind}"
        )


def _validate_layer_block_type(
    model: Qwen35PackedForCausalLM,
    target: str,
) -> None:
    if not target.startswith("layers."):
        return
    parts = target.split(".")
    layer = model.layer_stack.layers[int(parts[1])]
    expected = None
    if ".linear_attention." in target:
        expected = "linear_attention"
    elif ".full_attention." in target:
        expected = "full_attention"
    if expected is not None and layer.block_type != expected:
        raise ValueError(
            f"checkpoint target block type must be {expected}: {target}"
        )


def _transformed_shape(load: Qwen35CheckpointTensorLoad) -> tuple[int, ...]:
    shape = tuple(load.metadata.shape)
    if load.transform == "identity":
        return shape
    if load.transform == "squeeze_conv_channel":
        if len(shape) != 3 or shape[1] != 1:
            raise ValueError(
                "squeeze_conv_channel requires [channels, 1, kernel] shape"
            )
        return (shape[0], shape[2])
    raise ValueError(f"unsupported checkpoint transform: {load.transform}")


def _shard_shape(
    shape: tuple[int, ...],
    axis: int,
    tensor_parallel_size: int,
    target: str,
) -> tuple[int, ...]:
    if shape[axis] % tensor_parallel_size != 0:
        raise ValueError(
            f"{target} sharded dimension must be divisible by "
            "tensor_parallel_size"
        )
    local = list(shape)
    local[axis] //= tensor_parallel_size
    return tuple(local)


def _validate_component_tp(
    parent: nn.Module,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    target: str,
) -> None:
    if hasattr(parent, "tp_size"):
        if parent.tp_size != tensor_parallel_size:
            raise ValueError(f"{target} component tp_size must match")
    if hasattr(parent, "tp_rank"):
        if parent.tp_rank != tensor_parallel_rank:
            raise ValueError(f"{target} component tp_rank must match")


def _local_contract(
    load: Qwen35CheckpointTensorLoad,
    parent: nn.Module,
    tensor_parallel_size: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, int] | None,
]:
    target = load.weight.target
    shape = _transformed_shape(load)
    if target in ("embed_tokens.weight", "lm_head.weight"):
        local_shape = _shard_shape(
            shape,
            0,
            tensor_parallel_size,
            target,
        )
        return local_shape, local_shape, None
    if target.endswith("mlp.gate_up_proj.weight"):
        slot = load.weight.packed_slot
        if slot not in (0, 1):
            raise ValueError(
                f"{target} packed slot must be 0 or 1"
            )
        if (
            len(parent.output_sizes) != 2
            or parent.output_sizes[slot] != shape[0]
        ):
            raise ValueError(
                f"{target} packed source shape must match output_sizes"
            )
        local_shape = shape
        local_lengths = tuple(parent.output_sizes)
        destination_shape = (
            sum(local_lengths),
            shape[1],
        )
        offset = sum(local_lengths[:slot])
        return (
            local_shape,
            destination_shape,
            (offset, local_shape[0]),
        )
    if (
        target == "final_norm.weight"
        or target.endswith(_REPLICATED_TARGET_SUFFIXES)
    ):
        return shape, shape, None
    if target.endswith(_AXIS_ZERO_TARGET_SUFFIXES):
        local_shape = _shard_shape(
            shape,
            0,
            tensor_parallel_size,
            target,
        )
        return local_shape, local_shape, None
    if target.endswith(_AXIS_ONE_TARGET_SUFFIXES):
        local_shape = _shard_shape(
            shape,
            1,
            tensor_parallel_size,
            target,
        )
        return local_shape, local_shape, None
    raise ValueError(f"unsupported checkpoint local-shape target: {target}")


def _validate_specialized_component(
    load: Qwen35CheckpointTensorLoad,
    parent: nn.Module,
    tensor_parallel_size: int,
) -> None:
    target = load.weight.target
    shape = _transformed_shape(load)
    if type(parent) is ReplicatedSegmentedColumnParallelLinear:
        if sum(parent.output_sizes) != shape[0]:
            raise ValueError(
                f"{target} segmented output sizes must match source shape"
            )
        if any(
            output_size % tensor_parallel_size != 0
            for output_size in parent.output_sizes
        ):
            raise ValueError(
                f"{target} segmented outputs must be TP-divisible"
            )
    if type(parent) is ReplicatedHeadPairedColumnParallelLinear:
        if parent.num_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"{target} query heads must be TP-divisible"
            )
        if shape[0] != parent.num_heads * 2 * parent.head_dim:
            raise ValueError(
                f"{target} source rows must contain complete head pairs"
            )
    if type(parent) is ReplicatedKVHeadParallelLinear:
        if shape != (
            parent.total_num_kv_heads * parent.head_dim,
            parent.input_size,
        ):
            raise ValueError(
                f"{target} source shape must contain complete KV heads"
            )
    if type(parent) is ReplicatedLocalOutputLinear:
        if shape != (parent.output_size, parent.input_size):
            raise ValueError(
                f"{target} source shape must match replicated projection"
            )


def _source_segments(
    load: Qwen35CheckpointTensorLoad,
    parent: nn.Module,
    tensor_parallel_size: int,
) -> tuple[int, ...] | None:
    target = load.weight.target
    if not target.endswith("linear_attention.conv_weight"):
        return None
    if type(parent) is not Qwen35LinearAttentionShell:
        raise ValueError(
            f"{target} parent must be exact Qwen35LinearAttentionShell"
        )
    segments = (
        parent.local_key_heads
        * tensor_parallel_size
        * parent.key_head_dim,
        parent.local_key_heads
        * tensor_parallel_size
        * parent.key_head_dim,
        parent.local_value_heads
        * tensor_parallel_size
        * parent.value_head_dim,
    )
    if sum(segments) != _transformed_shape(load)[0]:
        raise ValueError(
            f"{target} channel segments must match source shape"
        )
    return segments


def _loader_kind(
    destination: torch.Tensor,
    destination_kind: str,
    target: str,
) -> str:
    if destination_kind == "buffer":
        return "direct_buffer_copy"
    if (
        target == "final_norm.weight"
        or target.endswith((
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "full_attention.q_norm.weight",
            "full_attention.k_norm.weight",
        ))
    ):
        return "default_parameter_copy"
    loader = getattr(destination, "weight_loader", None)
    if not callable(loader):
        raise ValueError(f"{target} requires a callable weight_loader")
    return "custom_parameter_loader"


def _validate_output_head_contract(
    model: Qwen35PackedForCausalLM,
    *,
    tie_word_embeddings: bool,
) -> None:
    if type(model.lm_head) is not ParallelLMHead:
        raise ValueError("lm_head must be an exact ParallelLMHead")
    embedding = model.embed_tokens.weight
    lm_head = model.lm_head.weight
    if embedding.shape != lm_head.shape or embedding.dtype != lm_head.dtype:
        raise ValueError(
            "embed_tokens and lm_head local shape/dtype must match"
        )
    if embedding.device.type == "meta" or lm_head.device.type == "meta":
        aliases = embedding is lm_head
    else:
        aliases = (
            embedding.untyped_storage().data_ptr()
            == lm_head.untyped_storage().data_ptr()
            and embedding.storage_offset() == lm_head.storage_offset()
        )
    if tie_word_embeddings:
        if not aliases:
            raise ValueError(
                "embed_tokens and lm_head must share storage"
            )
    elif aliases:
        raise ValueError(
            "untied embed_tokens and lm_head must not share storage"
        )


def build_qwen35_checkpoint_binding_plan(
    model: Qwen35PackedForCausalLM,
    tensor_plan: Qwen35CheckpointTensorPlan,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> Qwen35CheckpointBindingPlan:
    if type(model) is not Qwen35PackedForCausalLM:
        raise ValueError("model must be an exact Qwen35PackedForCausalLM")
    if type(tensor_plan) is not Qwen35CheckpointTensorPlan:
        raise ValueError(
            "tensor_plan must be an exact Qwen35CheckpointTensorPlan"
        )
    tensor_parallel_size, tensor_parallel_rank = _validate_tp_context(
        tensor_parallel_size,
        tensor_parallel_rank,
    )
    for load in tensor_plan.loads:
        if type(load) is not Qwen35CheckpointTensorLoad:
            raise ValueError(
                "tensor plan loads must be "
                "Qwen35CheckpointTensorLoad values"
            )
    lm_head_load_count = sum(
        load.weight.target == "lm_head.weight"
        for load in tensor_plan.loads
    )
    if lm_head_load_count > 1:
        raise ValueError("duplicate checkpoint binding target: lm_head.weight")
    _validate_output_head_contract(
        model,
        tie_word_embeddings=lm_head_load_count == 0,
    )

    bindings = []
    binding_keys = set()
    for load in tensor_plan.loads:
        if type(load) is not Qwen35CheckpointTensorLoad:
            raise ValueError(
                "tensor plan loads must be Qwen35CheckpointTensorLoad values"
            )
        target = load.weight.target
        key = (target, load.weight.packed_slot)
        if key in binding_keys:
            raise ValueError(
                f"duplicate checkpoint binding target: {target} {key[1]}"
            )
        binding_keys.add(key)
        _validate_layer_block_type(model, target)
        (
            destination_name,
            parent,
            _,
            destination,
            destination_kind,
        ) = _resolve_target(model, target)
        _require_exact_component(target, parent, destination_kind)
        _validate_component_tp(
            parent,
            tensor_parallel_size,
            tensor_parallel_rank,
            target,
        )
        _validate_specialized_component(
            load,
            parent,
            tensor_parallel_size,
        )
        (
            local_shape,
            destination_shape,
            destination_slice,
        ) = _local_contract(load, parent, tensor_parallel_size)
        if tuple(destination.shape) != destination_shape:
            raise ValueError(
                f"{target} destination local shape "
                f"{tuple(destination.shape)} must match {destination_shape}"
            )
        expected_dtype = _CHECKPOINT_DTYPES.get(load.metadata.dtype)
        if expected_dtype is None:
            raise ValueError(
                f"unsupported checkpoint binding dtype: "
                f"{load.metadata.dtype}"
            )
        allows_runtime_cast = (
            (
                target.endswith("linear_attention.norm_weight")
                and expected_dtype == torch.float32
                and destination.dtype == torch.bfloat16
            )
            or (
                target.endswith("linear_attention.A_log")
                and expected_dtype == torch.bfloat16
                and destination.dtype == torch.float32
            )
        )
        if destination.dtype != expected_dtype and not allows_runtime_cast:
            raise ValueError(
                f"{target} destination dtype {destination.dtype} "
                f"must match {expected_dtype}"
            )
        bindings.append(Qwen35CheckpointTensorBinding(
            load=load,
            destination_name=destination_name,
            destination=destination,
            destination_kind=destination_kind,
            loader_kind=_loader_kind(
                destination,
                destination_kind,
                target,
            ),
            local_shape=local_shape,
            destination_slice=destination_slice,
            source_segments=_source_segments(
                load,
                parent,
                tensor_parallel_size,
            ),
        ))

    return Qwen35CheckpointBindingPlan(
        bindings=tuple(bindings),
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
    )
