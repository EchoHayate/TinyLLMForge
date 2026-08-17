import torch
from torch import nn

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell


class Qwen35PackedStatefulLinearDecoderLayer(nn.Module):

    def __init__(
        self,
        decoder_layer: Qwen35DecoderLayerShell,
        state_adapter: Qwen35LayerStateAdapter,
    ):
        super().__init__()
        if not isinstance(decoder_layer, Qwen35DecoderLayerShell):
            raise ValueError(
                "decoder_layer must be a Qwen35DecoderLayerShell"
            )
        if decoder_layer.block_type != "linear_attention":
            raise ValueError(
                "decoder_layer must use the linear_attention block type"
            )
        if not isinstance(state_adapter, Qwen35LayerStateAdapter):
            raise ValueError(
                "state_adapter must be a Qwen35LayerStateAdapter"
            )
        self.decoder_layer = decoder_layer
        self.state_adapter = state_adapter

    @staticmethod
    def _validate_inputs(
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if not isinstance(leases, tuple) or not leases:
            raise ValueError("leases must be a non-empty tuple")
        if any(not isinstance(lease, HybridStateLease) for lease in leases):
            raise ValueError(
                "leases must contain only HybridStateLease values"
            )
        if not isinstance(token_counts, tuple) or not token_counts:
            raise ValueError("token_counts must be a non-empty tuple")
        if len(token_counts) != len(leases):
            raise ValueError(
                "leases and token_counts batch size must match"
            )
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
        if position_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("position_ids must use an integer dtype")
        if position_ids.device != hidden_states.device:
            raise ValueError(
                "position_ids device must match hidden_states"
            )

    def forward(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            leases,
            token_counts,
            position_ids,
            hidden_states,
        )
        convolution_states, recurrent_states = (
            self.state_adapter.gather_batch(leases)
        )

        outputs = []
        candidate_convolution = []
        candidate_recurrent = []
        offset = 0
        for request_index, token_count in enumerate(token_counts):
            segment = hidden_states[offset:offset + token_count]
            first_residual = segment
            normalized = self.decoder_layer.input_layernorm(segment)
            self.decoder_layer._validate_component_output(
                normalized,
                segment,
                "input_layernorm",
            )
            mixer_output = self.decoder_layer.linear_attention(
                normalized,
                convolution_states[request_index],
                recurrent_states[request_index],
            )
            if (
                not isinstance(mixer_output, tuple)
                or len(mixer_output) != 3
            ):
                raise ValueError(
                    "linear_attention output must be a three-item tuple"
                )
            mixed, next_convolution, next_recurrent = mixer_output
            self.decoder_layer._validate_component_output(
                mixed,
                segment,
                "linear_attention",
            )
            self.state_adapter._validate_candidate(
                next_convolution,
                convolution_states[request_index],
                "convolution_state",
            )
            self.state_adapter._validate_candidate(
                next_recurrent,
                recurrent_states[request_index],
                "recurrent_state",
            )
            segment = first_residual + mixed

            second_residual = segment
            normalized = self.decoder_layer.post_attention_layernorm(segment)
            self.decoder_layer._validate_component_output(
                normalized,
                segment,
                "post_attention_layernorm",
            )
            mlp_output = self.decoder_layer.mlp(normalized)
            self.decoder_layer._validate_component_output(
                mlp_output,
                segment,
                "mlp",
            )
            outputs.append(second_residual + mlp_output)
            candidate_convolution.append(next_convolution)
            candidate_recurrent.append(next_recurrent)
            offset += token_count

        self.state_adapter.commit_batch(
            leases,
            torch.stack(candidate_convolution),
            torch.stack(candidate_recurrent),
        )
        return torch.cat(outputs)
