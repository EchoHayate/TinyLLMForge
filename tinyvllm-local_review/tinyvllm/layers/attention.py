import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from tinyvllm.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
    key_ptr: torch.Tensor,
    key_stride: int,   
    value_ptr: torch.Tensor,
    value_stride: int,
    k_cache_ptr: torch.Tensor, 
    v_cache_ptr: torch.Tensor,
    slot_mapping_ptr: torch.Tensor,         # 1一个token对应的 一行 kv cache, 因此需要一个slot去定位当前 token 的kv cache位置
    D: tl.constexpr                         # 单个 token 的 Key/Value 数据长度
):
    pid = tl.program_id(axis = 0)
    key_offsets = pid * key_stride + tl.arange(0, D)
    value_offsets = pid * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)    #tl.load(内存地址)：从 GPU 内存读取数据到 GPU 寄存器（“加载”）
    value = tl.load(value_ptr + value_offsets)

    slot = tl.load(slot_mapping_ptr + pid)  #当前 token 在 KV Cache 中的 “起始位置索引”
    offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + offsets, key)    #key 和 value 是要存入这个 slot 的 “具体内容”
    tl.store(v_cache_ptr + offsets, value)  #tl.store(内存地址, 数据)：从 GPU 寄存器写入数据到 GPU 内存（“存储”）。


def store_kvcache(
    key: torch.Tensor,                       # 当前步计算的key张量  [batch_size * seq_len, num_heads, head_dim]
    value: torch.Tensor,                     # 当前步计算的value张量 [batch_size * seq_len, num_heads, head_dim]
    k_cache: torch.Tensor,                   # key缓存 [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,                   # value缓存  [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping: torch.Tensor,              # [N], num_kvcache_blocks, slot_mapping[i] 里面存的是 block_id * block_size, 即，token_id 在kv_cache中的位置 
):
    # N = batch_size * seq_len
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1  #    确保连续
    # 确保逻辑视图和物理内存视图一致 key = [N, num_heads, head_dim]
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N, )](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


#简化版本
def store_kvcache_simplified(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache:torch.Tensor,
    v_cache:torch.Tensor,
    slot_mapping: torch.Tensor
):
    N,num_heads,head_dim= key.shape

    flat_key = key.view(N,-1)
    flat_value = value.view(N,-1)

    for i in range(N):
        slot = slot_mapping[i].item()
        k_cache[slot] = flat_key[i]
        v_cache[slot] = flat_value[i]



class Attention(nn.Module):

    def __init__(
        self, 
        num_heads: int, 
        head_dim: int,
        scale: float, 
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.Tensor([]) 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # prefill传入的 q = [batch_size, seq_len, num_heads, head_dim]
            # 经过 view变成 q = [batch_size * seq_len, num_heads, head_dim]
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v, 
                                       cu_seqlens_q = context.cu_seqlens_q, cu_seqlens_k = context.cu_seqlens_k, 
                                       max_seqlen_q = context.max_seqlen_q, max_seqlen_k = context.max_seqlen_k, 
                                        softmax_scale = self.scale, causal = True, block_table = context.block_tables
                                       )
        else:
            # decode阶段传入的 q = [batch_size, num_heads, head_dim]
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens = context.context_lens,
                                        block_table = context.block_tables, softmax_scale = self.scale, causal = True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
