from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from tinyvllm.engine.hybrid_state import HybridStateTensorPool
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
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell
from tinyvllm.layers.qwen35_full_attention import Qwen35FullAttentionShell
from tinyvllm.layers.qwen35_linear_attention import (
    Qwen35LinearAttentionShell,
)
from tinyvllm.layers.qwen35_primitives import Qwen35OffsetRMSNorm
from tinyvllm.layers.qwen35_rotary_embedding import (
    Qwen35PartialInterleavedRotaryEmbedding,
)
from tinyvllm.models.qwen35_factory import (
    Qwen35PackedModelAssembly,
    assemble_qwen35_packed_model,
)
from tinyvllm.utils.context import get_context


@dataclass(frozen=True)
class Qwen35ConcreteComponentAssembly:
    packed: Qwen35PackedModelAssembly
    tensor_parallel_size: int
    tensor_parallel_rank: int
    parameter_device: torch.device
    compute_dtype: torch.dtype
    stable_dtype: torch.dtype


def _field(container, name: str):
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"missing Qwen3.5 config field: {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"missing Qwen3.5 config field: {name}")
    return getattr(container, name)


def _positive_integer(container, name: str) -> int:
    value = _field(container, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(container, name: str) -> float:
    value = _field(container, name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _tp_context(size, rank) -> tuple[int, int]:
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError("tensor_parallel_size must be a positive integer")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= size
    ):
        raise ValueError(
            "tensor_parallel_rank must be in "
            "[0, tensor_parallel_size)"
        )
    return size, rank


def _require_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor != 0:
        raise ValueError(
            f"{name} must be divisible by tensor_parallel_size"
        )


def _local_kv_heads(
    total_num_kv_heads: int,
    tensor_parallel_size: int,
) -> int:
    if total_num_kv_heads >= tensor_parallel_size:
        _require_divisible(
            total_num_kv_heads,
            tensor_parallel_size,
            "num_key_value_heads",
        )
        return total_num_kv_heads // tensor_parallel_size
    if tensor_parallel_size % total_num_kv_heads != 0:
        raise ValueError(
            "num_key_value_heads replication requires "
            "tensor_parallel_size to be divisible by "
            "num_key_value_heads"
        )
    return 1


def _parameter_device(value) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("parameter_device must be meta or cpu") from error
    if device.type not in ("meta", "cpu"):
        raise ValueError("parameter_device must be meta or cpu")
    return device


@contextmanager
def _distributed_construction_context(size: int, rank: int):
    original_rank = torch.distributed.get_rank
    original_world_size = torch.distributed.get_world_size
    torch.distributed.get_rank = lambda: rank
    torch.distributed.get_world_size = lambda: size
    try:
        yield
    finally:
        torch.distributed.get_rank = original_rank
        torch.distributed.get_world_size = original_world_size


class _Qwen35MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = ReplicatedMergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        ).to(dtype=torch.bfloat16)
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        ).to(dtype=torch.bfloat16)
        self.down_proj.requires_unpartitioned_linear_execution = True

    def _project_down(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if not context.is_prefill:
            return self.down_proj(hidden_states)
        cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
        cu_seqlens_k = getattr(context, "cu_seqlens_k", None)
        if (
            isinstance(cu_seqlens_q, torch.Tensor)
            and isinstance(cu_seqlens_k, torch.Tensor)
            and cu_seqlens_q.numel() == cu_seqlens_k.numel()
            and cu_seqlens_q.numel() > 1
        ):
            q_offsets = cu_seqlens_q.tolist()
            k_offsets = cu_seqlens_k.tolist()
            outputs = []
            for index in range(len(q_offsets) - 1):
                q_start = int(q_offsets[index])
                q_end = int(q_offsets[index + 1])
                q_length = q_end - q_start
                k_length = int(k_offsets[index + 1]) - int(k_offsets[index])
                segment = hidden_states[q_start:q_end]
                if k_length > q_length:
                    segment = torch.cat(
                        (
                            segment.new_zeros(
                                k_length - q_length,
                                segment.shape[1],
                            ),
                            segment,
                        ),
                        dim=0,
                    )
                outputs.append(self.down_proj(segment)[-q_length:])
            return torch.cat(outputs, dim=0)
        target_rows = int(getattr(context, "max_seqlen_k", 0))
        if target_rows <= hidden_states.shape[0]:
            return self.down_proj(hidden_states)
        padded = torch.cat(
            (
                hidden_states.new_zeros(
                    target_rows - hidden_states.shape[0],
                    hidden_states.shape[1],
                ),
                hidden_states,
            ),
            dim=0,
        )
        return self.down_proj(padded)[-hidden_states.shape[0]:]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self._project_down(F.silu(gate) * up)


def _compute_norm(hidden_size: int, eps: float) -> Qwen35OffsetRMSNorm:
    return Qwen35OffsetRMSNorm(hidden_size, eps).to(
        dtype=torch.bfloat16
    )


def build_qwen35_full_attention_decoder_layer(
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    norm_eps: float,
    rotary_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, int, int],
    build_attention_backend,
) -> Qwen35DecoderLayerShell:
    if not callable(build_attention_backend):
        raise ValueError("build_attention_backend must be callable")
    tensor_parallel_size = torch.distributed.get_world_size()
    _require_divisible(
        query_heads,
        tensor_parallel_size,
        "num_attention_heads",
    )
    local_query_heads = query_heads // tensor_parallel_size
    local_kv_heads = _local_kv_heads(
        kv_heads,
        tensor_parallel_size,
    )
    backend = build_attention_backend(
        local_query_heads,
        local_kv_heads,
        head_dim,
    )
    if not isinstance(backend, nn.Module):
        raise ValueError("attention backend must be a module")
    full_attention = Qwen35FullAttentionShell(
        head_dim=head_dim,
        local_query_heads=local_query_heads,
        local_kv_heads=local_kv_heads,
        q_projection=ReplicatedHeadPairedColumnParallelLinear(
            hidden_size,
            query_heads,
            head_dim,
            bias=False,
        ).to(dtype=torch.bfloat16),
        k_projection=ReplicatedKVHeadParallelLinear(
            hidden_size,
            kv_heads,
            head_dim,
            bias=False,
        ).to(dtype=torch.bfloat16),
        v_projection=ReplicatedKVHeadParallelLinear(
            hidden_size,
            kv_heads,
            head_dim,
            bias=False,
        ).to(dtype=torch.bfloat16),
        q_norm=_compute_norm(head_dim, norm_eps),
        k_norm=_compute_norm(head_dim, norm_eps),
        rotary=Qwen35PartialInterleavedRotaryEmbedding(
            head_dim,
            rotary_dim,
            rope_theta,
            mrope_section,
        ),
        attention_backend=backend,
        output_projection=RowParallelLinear(
            query_heads * head_dim,
            hidden_size,
            bias=False,
            accumulation_dtype=torch.float32,
            preserve_dense_prefill=True,
        ).to(dtype=torch.bfloat16),
    )
    return Qwen35DecoderLayerShell(
        block_type="full_attention",
        input_layernorm=_compute_norm(hidden_size, norm_eps),
        post_attention_layernorm=_compute_norm(
            hidden_size,
            norm_eps,
        ),
        mlp=_Qwen35MLP(hidden_size, intermediate_size),
        full_attention=full_attention,
    )


def build_qwen35_concrete_component_assembly(
    hf_config,
    *,
    pool: HybridStateTensorPool,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    build_attention_backend: Callable[
        [int, int, int, int],
        nn.Module,
    ],
    parameter_device: str | torch.device = "meta",
) -> Qwen35ConcreteComponentAssembly:
    if type(pool) is not HybridStateTensorPool:
        raise ValueError("pool must be an exact HybridStateTensorPool")
    if not callable(build_attention_backend):
        raise ValueError("build_attention_backend must be callable")
    tensor_parallel_size, tensor_parallel_rank = _tp_context(
        tensor_parallel_size,
        tensor_parallel_rank,
    )
    device = _parameter_device(parameter_device)
    config = getattr(hf_config, "text_config", hf_config)

    if _field(config, "dtype") != "bfloat16":
        raise ValueError("dtype must be bfloat16")
    if _field(config, "hidden_act") != "silu":
        raise ValueError("hidden_act must be silu")
    tie_word_embeddings = _field(config, "tie_word_embeddings")
    if type(tie_word_embeddings) is not bool:
        raise ValueError("tie_word_embeddings must be a bool")

    hidden_size = _positive_integer(config, "hidden_size")
    intermediate_size = _positive_integer(config, "intermediate_size")
    vocab_size = _positive_integer(config, "vocab_size")
    num_hidden_layers = _positive_integer(config, "num_hidden_layers")
    linear_key_heads = _positive_integer(
        config,
        "linear_num_key_heads",
    )
    linear_value_heads = _positive_integer(
        config,
        "linear_num_value_heads",
    )
    linear_key_head_dim = _positive_integer(
        config,
        "linear_key_head_dim",
    )
    linear_value_head_dim = _positive_integer(
        config,
        "linear_value_head_dim",
    )
    linear_conv_kernel = _positive_integer(
        config,
        "linear_conv_kernel_dim",
    )
    query_heads = _positive_integer(config, "num_attention_heads")
    kv_heads = _positive_integer(config, "num_key_value_heads")
    head_dim = _positive_integer(config, "head_dim")
    norm_eps = _positive_finite(config, "rms_norm_eps")

    layer_types_value = _field(config, "layer_types")
    if not isinstance(layer_types_value, (tuple, list)):
        raise ValueError("layer_types must be a tuple or list")
    layer_types = tuple(layer_types_value)
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "layer_types length must match num_hidden_layers"
        )
    if any(
        layer_type not in ("linear_attention", "full_attention")
        for layer_type in layer_types
    ):
        raise ValueError("unsupported Qwen3.5 layer type")

    for value, name in (
        (vocab_size, "vocab_size"),
        (intermediate_size, "intermediate_size"),
        (linear_key_heads, "linear_num_key_heads"),
        (linear_value_heads, "linear_num_value_heads"),
        (query_heads, "num_attention_heads"),
    ):
        _require_divisible(value, tensor_parallel_size, name)
    local_kv_heads = _local_kv_heads(
        kv_heads,
        tensor_parallel_size,
    )

    rope_parameters = _field(config, "rope_parameters")
    rope_theta = _positive_finite(rope_parameters, "rope_theta")
    partial_rotary_factor = _positive_finite(
        rope_parameters,
        "partial_rotary_factor",
    )
    rotary_dim = int(head_dim * partial_rotary_factor)
    mrope_value = _field(rope_parameters, "mrope_section")
    if not isinstance(mrope_value, (tuple, list)):
        raise ValueError("mrope_section must be a tuple or list")
    mrope_section = tuple(mrope_value)
    if (
        rotary_dim <= 0
        or rotary_dim % 2 != 0
        or rotary_dim > head_dim
        or len(mrope_section) != 3
        or any(
            isinstance(section, bool)
            or not isinstance(section, int)
            or section <= 0
            for section in mrope_section
        )
        or sum(mrope_section) != rotary_dim // 2
    ):
        raise ValueError(
            "mrope_section must contain three positive integers "
            "summing to rotary_dim / 2"
        )

    global_key_width = linear_key_heads * linear_key_head_dim
    global_value_width = linear_value_heads * linear_value_head_dim
    global_conv_width = 2 * global_key_width + global_value_width
    local_key_heads = linear_key_heads // tensor_parallel_size
    local_value_heads = linear_value_heads // tensor_parallel_size
    local_query_heads = query_heads // tensor_parallel_size
    local_conv_width = global_conv_width // tensor_parallel_size

    with _distributed_construction_context(
        tensor_parallel_size,
        tensor_parallel_rank,
    ), torch.device(device):
        embed_tokens = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
        ).to(dtype=torch.bfloat16)
        lm_head = ParallelLMHead(
            vocab_size,
            hidden_size,
            exact_full_vocab=True,
        ).to(dtype=torch.bfloat16)
        if tie_word_embeddings:
            lm_head.weight = embed_tokens.weight
        final_norm = _compute_norm(hidden_size, norm_eps)

        def build_decoder_layer(layer_index, block_type, _adapter):
            shared = {
                "block_type": block_type,
                "input_layernorm": _compute_norm(
                    hidden_size,
                    norm_eps,
                ),
                "post_attention_layernorm": _compute_norm(
                    hidden_size,
                    norm_eps,
                ),
                "mlp": _Qwen35MLP(
                    hidden_size,
                    intermediate_size,
                ),
            }
            if block_type == "linear_attention":
                linear_attention = Qwen35LinearAttentionShell(
                    local_key_heads=local_key_heads,
                    local_value_heads=local_value_heads,
                    key_head_dim=linear_key_head_dim,
                    value_head_dim=linear_value_head_dim,
                    norm_eps=norm_eps,
                    in_proj_qkv=ReplicatedSegmentedColumnParallelLinear(
                        hidden_size,
                        (
                            global_key_width,
                            global_key_width,
                            global_value_width,
                        ),
                        bias=False,
                    ).to(dtype=torch.bfloat16),
                    in_proj_z=ReplicatedColumnParallelLinear(
                        hidden_size,
                        global_value_width,
                        bias=False,
                    ).to(dtype=torch.bfloat16),
                    in_proj_b=ReplicatedLocalOutputLinear(
                        hidden_size,
                        linear_value_heads,
                        bias=False,
                    ).to(dtype=torch.bfloat16),
                    in_proj_a=ReplicatedLocalOutputLinear(
                        hidden_size,
                        linear_value_heads,
                        bias=False,
                    ).to(dtype=torch.bfloat16),
                    out_proj=RowParallelLinear(
                        global_value_width,
                        hidden_size,
                        bias=False,
                        accumulation_dtype=torch.float32,
                        preserve_dense_prefill=True,
                    ).to(dtype=torch.bfloat16),
                    conv_weight=torch.empty(
                        local_conv_width,
                        linear_conv_kernel,
                        dtype=torch.bfloat16,
                    ),
                    A_log=torch.empty(
                        local_value_heads,
                        dtype=torch.float32,
                    ),
                    dt_bias=torch.empty(
                        local_value_heads,
                        dtype=torch.bfloat16,
                    ),
                    norm_weight=torch.empty(
                        linear_value_head_dim,
                        dtype=torch.bfloat16,
                    ),
                )
                linear_attention.in_proj_qkv.requires_unpartitioned_linear_execution = True
                linear_attention.in_proj_z.requires_unpartitioned_linear_execution = True
                return Qwen35DecoderLayerShell(
                    **shared,
                    linear_attention=linear_attention,
                )

            return build_qwen35_full_attention_decoder_layer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                norm_eps=norm_eps,
                rotary_dim=rotary_dim,
                rope_theta=rope_theta,
                mrope_section=mrope_section,
                build_attention_backend=(
                    lambda local_queries, local_keys, dimension: (
                        build_attention_backend(
                            layer_index,
                            local_queries,
                            local_keys,
                            dimension,
                        )
                    )
                ),
            )

        packed = assemble_qwen35_packed_model(
            hf_config,
            pool=pool,
            embed_tokens=embed_tokens,
            final_norm=final_norm,
            lm_head=lm_head,
            build_decoder_layer=build_decoder_layer,
        )

    return Qwen35ConcreteComponentAssembly(
        packed=packed,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        parameter_device=device,
        compute_dtype=torch.bfloat16,
        stable_dtype=torch.float32,
    )
