from typing import Optional

import torch
from torch import nn


class Qwen35DecoderLayerShell(nn.Module):

    def __init__(
        self,
        *,
        block_type: str,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        mlp: nn.Module,
        full_attention: Optional[nn.Module] = None,
        linear_attention: Optional[nn.Module] = None,
    ):
        super().__init__()
        if block_type not in ("full_attention", "linear_attention"):
            raise ValueError(
                "block_type must be 'full_attention' or "
                "'linear_attention'"
            )
        if block_type == "full_attention" and full_attention is None:
            raise ValueError(
                "full_attention must be provided for full_attention blocks"
            )
        if block_type == "linear_attention" and linear_attention is None:
            raise ValueError(
                "linear_attention must be provided for linear_attention blocks"
            )
        self.block_type = block_type
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.full_attention = full_attention
        self.linear_attention = linear_attention

    @staticmethod
    def _validate_component_output(
        output: torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"{name} output must be a tensor")
        if not output.is_floating_point():
            raise ValueError(
                f"{name} output must use a floating point dtype"
            )
        if output.shape != reference.shape:
            raise ValueError(f"{name} shape must remain unchanged")
        if output.dtype != reference.dtype:
            raise ValueError(f"{name} dtype must remain unchanged")
        if output.device != reference.device:
            raise ValueError(f"{name} device must remain unchanged")

    def forward(
        self,
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

        first_residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        self._validate_component_output(
            normalized,
            hidden_states,
            "input_layernorm",
        )

        if self.block_type == "full_attention":
            mixed = self.full_attention(position_ids, normalized)
            mixer_name = "full_attention"
        else:
            mixed = self.linear_attention(normalized)
            mixer_name = "linear_attention"
        self._validate_component_output(mixed, hidden_states, mixer_name)
        hidden_states = first_residual + mixed

        second_residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        self._validate_component_output(
            normalized,
            hidden_states,
            "post_attention_layernorm",
        )
        mlp_output = self.mlp(normalized)
        self._validate_component_output(mlp_output, hidden_states, "mlp")
        return second_residual + mlp_output
