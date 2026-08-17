import math

import torch
from torch import nn


class Qwen35PartialInterleavedRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        base: float,
        mrope_section: tuple[int, int, int],
    ):
        super().__init__()
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")
        if (
            isinstance(rotary_dim, bool)
            or not isinstance(rotary_dim, int)
            or rotary_dim <= 0
            or rotary_dim % 2 != 0
            or rotary_dim > head_dim
        ):
            raise ValueError(
                "rotary_dim must be a positive even integer no larger than head_dim"
            )
        if (
            isinstance(base, bool)
            or not isinstance(base, (int, float))
            or not math.isfinite(base)
            or base <= 1
        ):
            raise ValueError("base must be a finite number greater than one")
        if (
            not isinstance(mrope_section, tuple)
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

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = float(base)
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _validate_inputs(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> int:
        if position_ids.ndim == 1:
            token_count = position_ids.shape[0]
        elif position_ids.ndim == 2 and position_ids.shape[0] == 3:
            token_count = position_ids.shape[1]
        else:
            raise ValueError(
                "position_ids must have shape [tokens] or [3, tokens]"
            )
        if position_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("position_ids must use an integer dtype")
        if query.ndim != 2 or key.ndim != 2:
            raise ValueError("query and key must be rank two tensors")
        if not query.is_floating_point() or not key.is_floating_point():
            raise ValueError("query and key must use a floating point dtype")
        if query.shape[0] != token_count or key.shape[0] != token_count:
            raise ValueError(
                "position_ids, query, and key token counts must match"
            )
        if (
            query.shape[1] <= 0
            or key.shape[1] <= 0
            or query.shape[1] % self.head_dim != 0
            or key.shape[1] % self.head_dim != 0
        ):
            raise ValueError(
                "query and key feature dimensions must be positive "
                "multiples of head_dim"
            )
        if query.dtype != key.dtype:
            raise ValueError("query and key dtype must match")
        if query.device != key.device or position_ids.device != query.device:
            raise ValueError("position_ids, query, and key device must match")
        return token_count

    def _selected_frequencies(
        self,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(3, -1)
        frequencies = (
            position_ids.float().unsqueeze(-1)
            * self.inv_freq.float().view(1, 1, -1)
        )
        selected = frequencies[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[axis] * 3
            selected[:, offset:length:3] = frequencies[
                axis, :, offset:length:3
            ]
        return selected

    def _apply_rotary(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        shape = tensor.shape
        by_head = tensor.view(token_count, -1, self.head_dim)
        prefix = by_head[..., : self.rotary_dim]
        suffix = by_head[..., self.rotary_dim :]
        cos = cos.to(tensor.dtype)
        sin = sin.to(tensor.dtype)
        first, second = prefix.chunk(2, dim=-1)
        rotated_half = torch.cat((-second, first), dim=-1)
        rotated = prefix * cos + rotated_half * sin
        return torch.cat((rotated, suffix), dim=-1).reshape(shape)

    def forward(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = self._validate_inputs(position_ids, query, key)
        frequencies = self._selected_frequencies(position_ids)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().unsqueeze(1)
        sin = embedding.sin().unsqueeze(1)
        return (
            self._apply_rotary(query, cos, sin, token_count),
            self._apply_rotary(key, cos, sin, token_count),
        )
