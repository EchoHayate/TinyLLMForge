from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(
    x: torch.Tensor,       # [batch_size * seq_len, num_head, head_size], num_tokens = batch_size * seq_len
    cos: torch.Tensor,     # [batch_size * seq_len, 1, head_size/2]
    sin: torch.Tensor,     # [batch_size * seq_len, 1, head_size/2]
) -> torch.Tensor:         # [batch_size * seq_len, num_head, head_size]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = cos * x1 - sin * x2
    y2 = cos * x2 + sin * x1
    return torch.cat((y1, y2), dim = -1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    
    def __init__(
        self, 
        head_size: int, 
        rotary_dim: int, 
        max_position_embedding: int, 
        base: float                     # rope_theta = 10000
    ):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position_embedding, dtype = torch.float32)
        # 计算外积，变成矩阵
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim = -1)
        # persistent决定该buffer会不会被保存到模型中
        self.register_buffer("cos_sin_cache", cache, persistent= False)

    @torch.compile
    def forward(self,
        positions: torch.Tensor,    # [batch_size * seq_len]
        query: torch.Tensor,        # [batch_size * seq_len, num_heads * head_size] = [16384, 16 * 128]
        key: torch.Tensor,          # [batch_size * seq_len, num_kv_heads * head_size] = [16384, 16 * 128]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, -1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)                 
        query = apply_rotary_emb(query, cos, sin).view(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int, 
    max_position: int,
    base: float,
    rope_scaling: dict | None = None, 
) -> RotaryEmbedding:
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
    