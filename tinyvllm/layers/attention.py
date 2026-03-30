import torch
from torch import nn
import triton
import triton.language as tl

from tinyvllm.kernels import flash_attn2_fwd, flash_decoding_fwd, reduction



from tinyvllm.utils.context import get_context




@triton.jit
def store_kvcache_int8_kernel(
    key_ptr: torch.Tensor,
    key_stride: int,   
    value_ptr: torch.Tensor,
    value_stride: int,
    k_cache_ptr: torch.Tensor, 
    v_cache_ptr: torch.Tensor,
    k_scale_ptr: torch.Tensor,
    v_scale_ptr: torch.Tensor,
    slot_mapping_ptr: torch.Tensor,         
    num_kv_heads: int,
    head_dim: tl.constexpr                         
):
    pid = tl.program_id(axis = 0)
    slot = tl.load(slot_mapping_ptr + pid)

    for head_idx in range(num_kv_heads):
        key_offsets = pid * key_stride + head_idx * head_dim + tl.arange(0, head_dim)
        value_offsets = pid * value_stride + head_idx * head_dim + tl.arange(0, head_dim)

        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)

        k_max = tl.max(tl.abs(key))
        v_max = tl.max(tl.abs(value))
        
        k_scale = k_max / 127.0
        v_scale = v_max / 127.0
        
        k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
        v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
        
        key_int8 = (key / k_scale).to(tl.int8)
        value_int8 = (value / v_scale).to(tl.int8)
        
        out_offsets = slot * num_kv_heads * head_dim + head_idx * head_dim + tl.arange(0, head_dim)
        
        tl.store(k_cache_ptr + out_offsets, key_int8)
        tl.store(v_cache_ptr + out_offsets, value_int8)
        
        scale_offset = slot * num_kv_heads + head_idx
        tl.store(k_scale_ptr + scale_offset, k_scale)
        tl.store(v_scale_ptr + scale_offset, v_scale)

@triton.jit
def store_kvcache_kernel(
    key_ptr: torch.Tensor,
    key_stride: int,   
    value_ptr: torch.Tensor,
    value_stride: int,
    k_cache_ptr: torch.Tensor, 
    v_cache_ptr: torch.Tensor,
    slot_mapping_ptr: torch.Tensor,         
    D: tl.constexpr                         
):
    pid = tl.program_id(axis = 0)
    key_offsets = pid * key_stride + tl.arange(0, D)
    value_offsets = pid * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)    
    value = tl.load(value_ptr + value_offsets)

    slot = tl.load(slot_mapping_ptr + pid)  
    offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + offsets, key)    
    tl.store(v_cache_ptr + offsets, value)  

def store_kvcache(
    key: torch.Tensor,                       
    value: torch.Tensor,                     
    k_cache: torch.Tensor,                   
    v_cache: torch.Tensor,                   
    slot_mapping: torch.Tensor,              
    k_cache_scale: torch.Tensor = None,
    v_cache_scale: torch.Tensor = None
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert slot_mapping.numel() == N
    
    if k_cache.dtype == torch.int8:
        store_kvcache_int8_kernel[(N, )](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, k_cache_scale, v_cache_scale,
            slot_mapping, num_heads, head_dim
        )
    else:
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        store_kvcache_kernel[(N, )](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

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
        k_cache_scale = getattr(self, "k_cache_scale", None)
        v_cache_scale = getattr(self, "v_cache_scale", None)
        
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, k_cache_scale, v_cache_scale)
            
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn2_fwd(q, k, v, 
                                       cu_seqlens_q = context.cu_seqlens_q, cu_seqlens_k = context.cu_seqlens_k, 
                                       max_seqlen_q = context.max_seqlen_q, max_seqlen_k = context.max_seqlen_k, 
                                        softmax_scale = self.scale, causal = True
                                       )
        else:
            mid_o, mid_l = flash_decoding_fwd(q, k_cache, v_cache, context.block_tables, context.context_lens, 
                                        context.max_seqlen_k, softmax_scale=self.scale, 
                                        k_scale=k_cache_scale, v_scale=v_cache_scale)
            o = reduction(mid_o, mid_l)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
