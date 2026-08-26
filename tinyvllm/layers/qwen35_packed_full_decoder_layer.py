import torch
from torch import nn

from tinyvllm.engine.decode_internal_profiler import profile_layer
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell


class Qwen35PackedFullDecoderLayer(nn.Module):

    def __init__(
        self,
        decoder_layer: Qwen35DecoderLayerShell,
        layer_index: int = 0,
    ):
        super().__init__()
        if not isinstance(decoder_layer, Qwen35DecoderLayerShell):
            raise ValueError(
                "decoder_layer must be a Qwen35DecoderLayerShell"
            )
        if decoder_layer.block_type != "full_attention":
            raise ValueError(
                "decoder_layer must use the full_attention block type"
            )
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError(
                "layer_index must be a non-negative integer"
            )
        self.decoder_layer = decoder_layer
        self.layer_index = layer_index

    @staticmethod
    def _validate_inputs(
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if not isinstance(token_counts, tuple) or not token_counts:
            raise ValueError("token_counts must be a non-empty tuple")
        if any(
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            for token_count in token_counts
        ):
            raise ValueError(
                "token_counts must contain positive integers"
            )
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("hidden_states must be a tensor")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be rank two")
        if not hidden_states.is_floating_point():
            raise ValueError(
                "hidden_states must use a floating point dtype"
            )
        token_count = hidden_states.shape[0]
        if sum(token_counts) != token_count:
            raise ValueError(
                "token_counts sum must match hidden_states token count"
            )
        if not isinstance(position_ids, torch.Tensor):
            raise ValueError("position_ids must be a tensor")
        if position_ids.ndim not in (1, 2):
            raise ValueError("position_ids must be rank one or two")
        if position_ids.ndim == 2 and position_ids.shape[0] not in (1, 3):
            raise ValueError("position_ids must have one or three rows")
        if position_ids.shape[-1] != token_count:
            raise ValueError(
                "position_ids token count must match hidden_states"
            )
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must use an integer dtype")
        if position_ids.device != hidden_states.device:
            raise ValueError(
                "position_ids device must match hidden_states"
            )

    def _forward_unprofiled(
        self,
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(token_counts, position_ids, hidden_states)
        outputs = []
        offset = 0
        for token_count in token_counts:
            end = offset + token_count
            segment_positions = (
                position_ids[offset:end]
                if position_ids.ndim == 1
                else position_ids[:, offset:end]
            )
            outputs.append(self.decoder_layer(
                segment_positions,
                hidden_states[offset:end],
            ))
            offset = end
        return torch.cat(outputs)

    def forward(
        self,
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        with profile_layer(self.layer_index, "full_attention"):
            return self._forward_unprofiled(
                token_counts,
                position_ids,
                hidden_states,
            )
