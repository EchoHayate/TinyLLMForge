import math

import torch
from torch import nn


def qwen35_apply_query_gate(
    attention_output: torch.Tensor,
    query_gate: torch.Tensor,
) -> torch.Tensor:
    if not attention_output.is_floating_point():
        raise ValueError("attention_output must use a floating point dtype")
    if not query_gate.is_floating_point():
        raise ValueError("query_gate must use a floating point dtype")
    if attention_output.dtype != query_gate.dtype:
        raise ValueError("attention_output and query_gate dtype must match")
    if attention_output.shape != query_gate.shape:
        raise ValueError("attention_output and query_gate shapes must exactly match")
    return attention_output * torch.sigmoid(query_gate)


class Qwen35OffsetRMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            raise ValueError("hidden_size must be a positive integer")
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            raise ValueError("tensor must use a floating point dtype")
        if tensor.ndim == 0 or tensor.shape[-1] != self.hidden_size:
            raise ValueError(
                f"tensor last dimension must match hidden_size {self.hidden_size}"
            )
        tensor_fp32 = tensor.float()
        normalized = tensor_fp32 * torch.rsqrt(
            tensor_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        output = normalized * (1.0 + self.weight.float())
        return output.to(tensor.dtype)
