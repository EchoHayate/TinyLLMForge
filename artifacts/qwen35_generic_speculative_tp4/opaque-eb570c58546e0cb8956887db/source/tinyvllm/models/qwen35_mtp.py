from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from tinyvllm.layers.linear import ReplicatedLinear
from tinyvllm.layers.qwen35_packed_full_decoder_layer import (
    Qwen35PackedFullDecoderLayer,
)
from tinyvllm.layers.qwen35_primitives import Qwen35OffsetRMSNorm
from tinyvllm.models.qwen35_components import (
    _distributed_construction_context,
    _parameter_device,
    build_qwen35_full_attention_decoder_layer,
)


def _field(container, name: str):
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"missing Qwen3.5 MTP config field: {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"missing Qwen3.5 MTP config field: {name}")
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


def _mtp_norm(hidden_size: int, eps: float) -> Qwen35OffsetRMSNorm:
    return Qwen35OffsetRMSNorm(hidden_size, eps).to(
        dtype=torch.bfloat16
    )


class Qwen35NativeMTP(nn.Module):

    def __init__(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        fc: nn.Module,
        layer: Qwen35PackedFullDecoderLayer,
        norm: nn.Module,
        pre_fc_norm_embedding: nn.Module,
        pre_fc_norm_hidden: nn.Module,
    ):
        super().__init__()
        for name, module in (
            ("embed_tokens", embed_tokens),
            ("lm_head", lm_head),
            ("fc", fc),
            ("norm", norm),
            ("pre_fc_norm_embedding", pre_fc_norm_embedding),
            ("pre_fc_norm_hidden", pre_fc_norm_hidden),
        ):
            if not isinstance(module, nn.Module):
                raise ValueError(f"{name} must be a module")
        if not isinstance(layer, Qwen35PackedFullDecoderLayer):
            raise ValueError(
                "layer must be a Qwen35PackedFullDecoderLayer"
            )
        weight = getattr(fc, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.shape[1] != 2 * weight.shape[0]
        ):
            raise ValueError(
                "fc weight must have shape "
                "(hidden_size, 2 * hidden_size)"
            )
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.fc = fc
        self.layer = layer
        self.norm = norm
        self.pre_fc_norm_embedding = pre_fc_norm_embedding
        self.pre_fc_norm_hidden = pre_fc_norm_hidden
        self.hidden_size = int(weight.shape[0])

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a tensor")
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be rank one")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use an integer dtype")
        if not isinstance(positions, torch.Tensor):
            raise ValueError("positions must be a tensor")
        if positions.ndim not in (1, 2):
            raise ValueError("positions must be rank one or two")
        if positions.ndim == 2 and positions.shape[0] not in (1, 3):
            raise ValueError("positions must have one or three rows")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use an integer dtype")
        if positions.shape[-1] != input_ids.shape[0]:
            raise ValueError(
                "positions token count must match input_ids"
            )
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("hidden_states must be a tensor")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be rank two")
        if not hidden_states.is_floating_point():
            raise ValueError(
                "hidden_states must use a floating point dtype"
            )
        if hidden_states.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "hidden_states token count must match input_ids"
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                "hidden_states hidden size must match MTP hidden size"
            )
        if (
            input_ids.device != hidden_states.device
            or positions.device != hidden_states.device
        ):
            raise ValueError(
                "input_ids, positions, and hidden_states "
                "must use the same device"
            )

    def forward_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(input_ids, positions, hidden_states)
        embedded = self.pre_fc_norm_embedding(
            self.embed_tokens(input_ids)
        )
        hidden = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(torch.cat((embedded, hidden), dim=-1))
        decoded = self.layer(
            (int(input_ids.shape[0]),),
            positions,
            fused,
        )
        normalized = self.norm(decoded)
        return normalized, self.lm_head(normalized)


def build_qwen35_native_mtp(
    hf_config,
    *,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    build_attention_backend,
    parameter_device: str | torch.device = "meta",
) -> Qwen35NativeMTP:
    if tensor_parallel_size != 1 or tensor_parallel_rank != 0:
        raise ValueError("Qwen3.5 native MTP first slice requires TP1")
    if not isinstance(embed_tokens, nn.Module):
        raise ValueError("embed_tokens must be a module")
    if not isinstance(lm_head, nn.Module):
        raise ValueError("lm_head must be a module")
    if not callable(build_attention_backend):
        raise ValueError("build_attention_backend must be callable")
    embedding_weight = getattr(embed_tokens, "weight", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if embedding_weight is None or lm_head_weight is None:
        raise ValueError("embed_tokens and lm_head must expose weight")
    if embedding_weight is not lm_head_weight:
        raise ValueError(
            "embed_tokens and lm_head must share the same weight"
        )

    config = getattr(hf_config, "text_config", hf_config)
    if _field(config, "dtype") != "bfloat16":
        raise ValueError("dtype must be bfloat16")
    if _field(config, "hidden_act") != "silu":
        raise ValueError("hidden_act must be silu")
    if _field(config, "mtp_num_hidden_layers") != 1:
        raise ValueError("mtp_num_hidden_layers must equal one")
    if _field(config, "mtp_use_dedicated_embeddings") is not False:
        raise ValueError(
            "mtp_use_dedicated_embeddings must be false"
        )
    if _field(config, "tie_word_embeddings") is not True:
        raise ValueError("tie_word_embeddings must be true")

    hidden_size = _positive_integer(config, "hidden_size")
    intermediate_size = _positive_integer(config, "intermediate_size")
    query_heads = _positive_integer(config, "num_attention_heads")
    kv_heads = _positive_integer(config, "num_key_value_heads")
    head_dim = _positive_integer(config, "head_dim")
    norm_eps = _positive_finite(config, "rms_norm_eps")
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
    device = _parameter_device(parameter_device)

    with _distributed_construction_context(1, 0), torch.device(device):
        fc = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
        ).to(dtype=torch.bfloat16)
        decoder_layer = build_qwen35_full_attention_decoder_layer(
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
                        0,
                        local_queries,
                        local_keys,
                        dimension,
                    )
                )
            ),
        )
        return Qwen35NativeMTP(
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            fc=fc,
            layer=Qwen35PackedFullDecoderLayer(decoder_layer),
            norm=_mtp_norm(hidden_size, norm_eps),
            pre_fc_norm_embedding=_mtp_norm(
                hidden_size,
                norm_eps,
            ),
            pre_fc_norm_hidden=_mtp_norm(hidden_size, norm_eps),
        )
