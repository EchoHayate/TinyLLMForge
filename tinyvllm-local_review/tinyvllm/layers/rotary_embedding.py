from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(              
    x: torch.Tensor,       # [batch_size * seq_len, num_head, head_size], num_tokens = batch_size * seq_len
    cos: torch.Tensor,     # [batch_size * seq_len, 1, head_size/2]
    sin: torch.Tensor,     # [batch_size * seq_len, 1, head_size/2]
) -> torch.Tensor:         # [batch_size * seq_len, num_head, head_size]
    cos = cos.unsqueeze(-2) #[40960, 1, 64]
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)    #x1=x2=[40960, 16, 64]      若原始维度是[x0, x1, x2, x3]； 拆分后x1 = [x0, x2]（每对的 “第一个元素”），x2 = [x1, x3]（每对的 “第二个元素”）。
    y1 = cos * x1 - sin * x2             #公式可参考 根据ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING p7 eq34之间的公式
    y2 = cos * x2 + sin * x1
    return torch.cat((y1, y2), dim = -1).to(x.dtype)
# 公式中旋转后 Q 的维度顺序是 [q0, q1, q2, q3]（每组的两个元素相邻）；
# 代码中旋转后 Q 的维度顺序是 [q0, q2, q1, q3] （先所有组的 y1，再所有组的 y2）。  后面还要经过点积处理 所以顺序不满足公式没关系

class RotaryEmbedding(nn.Module):
    
    def __init__(
        self, 
        head_size: int, 
        rotary_dim: int, 
        max_position_embedding: int, 
        base: float                     # rope_theta = 10000
    ):
        super().__init__()
        self.head_size = head_size      #for qwen  head_size =[128]
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))       #[64]   根据ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING p5 eq15-eq16之间的公式
        t = torch.arange(max_position_embedding, dtype = torch.float32)             #[40960]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)      # 计算外积，变成矩阵  freqs =  [40960,64]
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim = -1)     #cache = [40960,128]
        # persistent决定该buffer会不会被保存到模型stat_dict中
        # 这里不保存到stat_dict是为了用一点计算时间换取显存空间
        self.register_buffer("cos_sin_cache", cache, persistent= False)

    @torch.compile
    def forward(self,
        positions: torch.Tensor,    # [batch_size * seq_len]    positions表示当前输入序列的具体位置
        query: torch.Tensor,        # [batch_size * seq_len, num_heads * head_size] = [40960, 16 * 128]
        key: torch.Tensor,          # [batch_size * seq_len, num_kv_heads * head_size] = [40960, 16 * 128]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)  #总tokens   
        cos_sin = self.cos_sin_cache[positions] #按位置索引获取缓存的旋转参数   
        cos, sin = cos_sin.chunk(2, -1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)         #[batch_size*seq_len,num_heads*head_size] => [batch_size*seq_len,num_heads, head_size]     
        query = apply_rotary_emb(query, cos, sin).view(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1) #缓存cos_sin_cache
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
    