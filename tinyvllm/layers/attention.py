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


def update_block_summary(
    key: torch.Tensor,           # [N, num_kv_heads, head_dim]
    slot_mapping: torch.Tensor,  # [N] int32, slot = block_id * block_size + intra
    k_min: torch.Tensor,         # [num_blocks, num_kv_heads, head_dim]
    k_max: torch.Tensor,         # [num_blocks, num_kv_heads, head_dim]
    block_size: int,
):
    """Quest：对刚写入的 token，按其所在 block_id 维护 per-channel min/max。

    用 index_reduce_ 做 segment-style amin/amax；初始值为 +inf / -inf，include_self=True
    保证多次调用累计正确。
    """
    block_ids = (slot_mapping.to(torch.long) // block_size)  # [N]
    # index_reduce 在 fp16 上有支持，但部分版本 amin/amax 仅 fp32 稳定 —— 先转 fp32 再写回
    # 为减少开销，直接用原 dtype；如遇精度问题再加 fp32 fallback
    k_min.index_reduce_(0, block_ids, key, reduce="amin", include_self=True)
    k_max.index_reduce_(0, block_ids, key, reduce="amax", include_self=True)


def quest_select_blocks(
    q: torch.Tensor,                 # [B, num_q_heads, head_dim]，已 RoPE
    block_tables: torch.Tensor,      # [B, max_blocks] int32
    context_lens: torch.Tensor,      # [B] int32
    k_min_layer: torch.Tensor,       # [num_blocks_total, num_kv_heads, head_dim]
    k_max_layer: torch.Tensor,       # [num_blocks_total, num_kv_heads, head_dim]
    block_size: int,
    top_k: int,
    min_seq_len: int,
):
    """Quest 选块：对每个 batch row，估计每个 block 的 max-inner-product 上界，取 top-k。

    返回 (sparse_block_tables, sparse_context_lens)：
      - 仅当某 row 的 num_blocks > top_k 才稀疏化；否则原样
      - 全 batch 都不需要稀疏化时，返回 (None, None)
    """
    B, max_blocks = block_tables.shape
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k_min_layer.shape[1]
    n_groups = num_q_heads // num_kv_heads  # GQA group

    # 每行实际有效 block 数：ceil(context_len / block_size)
    num_blocks_in_seq = (context_lens.to(torch.long) + block_size - 1) // block_size  # [B]

    # 任何一行 seq_len < min_seq_len 都不稀疏（保稳）；
    # 且要求 batch 内所有 row 的 num_blocks > top_k，避免 mixed batch 边界 + 重复块带来的 logits 漂移
    if (context_lens.min().item() < min_seq_len
            or num_blocks_in_seq.min().item() <= top_k):
        return None, None

    valid_mask = block_tables >= 0                      # [B, max_blocks]
    safe_idx = block_tables.clamp_min(0).to(torch.long) # [B, max_blocks]

    # gather 每个 (b, block) 对应的 k_min/k_max
    k_min_b = k_min_layer[safe_idx]                     # [B, max_blocks, kv_h, d]
    k_max_b = k_max_layer[safe_idx]

    # GQA: 把 q head 按 kv head 分组，组内取 max（保激进估计）
    q_grouped = q.view(B, num_kv_heads, n_groups, head_dim)
    q_repr = q_grouped.amax(dim=2)                      # [B, kv_h, d]

    # criticality = sum_d max(q*k_min, q*k_max) per kv_head, 再跨 head 求和
    qm_min = q_repr.unsqueeze(1) * k_min_b               # [B, max_blocks, kv_h, d]
    qm_max = q_repr.unsqueeze(1) * k_max_b
    qm = torch.maximum(qm_min, qm_max)
    criticality = qm.sum(dim=(-1, -2)).to(torch.float32) # [B, max_blocks]

    # 屏蔽 invalid block
    criticality = criticality.masked_fill(~valid_mask, float("-inf"))

    # 强制保留：首 block (attention sink) + 末 block (recency / partial)
    arange = torch.arange(max_blocks, device=block_tables.device).unsqueeze(0)  # [1, max_blocks]
    is_first = (arange == 0)
    last_idx = (num_blocks_in_seq - 1).clamp_min(0).unsqueeze(1)
    is_last = (arange == last_idx)
    must_keep = (is_first | is_last) & valid_mask
    criticality = criticality.masked_fill(must_keep, float("inf"))

    # top-k：因前置条件保证所有 row 的 num_blocks > top_k，topk 必然落在 valid 位置
    actual_k = top_k
    _, topk_idx = criticality.topk(actual_k, dim=-1)    # [B, actual_k]
    topk_idx, _ = topk_idx.sort(dim=-1)                  # 保持原顺序，最后一位 = 原最后 block
    sparse_bt = torch.gather(block_tables, 1, topk_idx).to(torch.int32)  # [B, actual_k]

    # sparse_context_lens = (actual_k - 1) * block_size + last_partial
    last_partial = ((context_lens.to(torch.long) - 1) % block_size) + 1  # [B]
    sparse_context_lens = ((actual_k - 1) * block_size + last_partial).to(torch.int32)

    return sparse_bt.contiguous(), sparse_context_lens.contiguous()


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
        # Quest per-block summary（model_runner 在 allocate_kv_cache 里挂入；未启用为 None）
        self.k_min = self.k_max = None
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            # Quest：写入 KV 后同步维护 per-block summary
            if self.k_min is not None and self.k_max is not None:
                update_block_summary(k, context.slot_mapping, self.k_min, self.k_max, k_cache.shape[1])
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
            block_tables = context.block_tables
            cache_seqlens = context.context_lens
            # Quest：动态选 top-k block，重写 block_table 与 cache_seqlens
            if (context.quest_top_k_blocks > 0 and self.k_min is not None
                    and block_tables is not None and cache_seqlens is not None):
                sparse_bt, sparse_lens = quest_select_blocks(
                    q, block_tables, cache_seqlens,
                    self.k_min, self.k_max,
                    k_cache.shape[1],
                    context.quest_top_k_blocks,
                    context.quest_min_seq_len,
                )
                if sparse_bt is not None:
                    block_tables = sparse_bt
                    cache_seqlens = sparse_lens
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens = cache_seqlens,
                                        block_table = block_tables, softmax_scale = self.scale, causal = True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
