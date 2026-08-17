import torch
from torch import nn

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell


class Qwen35StatefulLinearDecoderLayer(nn.Module):

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

    def forward(
        self,
        lease: HybridStateLease,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("hidden_states must be a tensor")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be rank two")
        if not hidden_states.is_floating_point():
            raise ValueError(
                "hidden_states must use a floating point dtype"
            )

        convolution_state, recurrent_state = (
            self.state_adapter.gather(lease)
        )

        first_residual = hidden_states
        normalized = self.decoder_layer.input_layernorm(hidden_states)
        self.decoder_layer._validate_component_output(
            normalized,
            hidden_states,
            "input_layernorm",
        )

        mixer_output = self.decoder_layer.linear_attention(
            normalized,
            convolution_state,
            recurrent_state,
        )
        if (
            not isinstance(mixer_output, tuple)
            or len(mixer_output) != 3
        ):
            raise ValueError(
                "linear_attention output must be a three-item tuple"
            )
        mixed, candidate_convolution, candidate_recurrent = mixer_output
        self.decoder_layer._validate_component_output(
            mixed,
            hidden_states,
            "linear_attention",
        )
        hidden_states = first_residual + mixed

        second_residual = hidden_states
        normalized = self.decoder_layer.post_attention_layernorm(
            hidden_states
        )
        self.decoder_layer._validate_component_output(
            normalized,
            hidden_states,
            "post_attention_layernorm",
        )
        mlp_output = self.decoder_layer.mlp(normalized)
        self.decoder_layer._validate_component_output(
            mlp_output,
            hidden_states,
            "mlp",
        )
        output = second_residual + mlp_output

        self.state_adapter.commit(
            lease,
            candidate_convolution,
            candidate_recurrent,
        )
        return output
