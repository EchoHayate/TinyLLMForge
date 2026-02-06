
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_decoding_fwd_kernel(
    Q, K_cache, V_cache, Block_tables, Context_lens,
    Mid_O, Mid_L, # Intermediate buffers for Split-K
    sm_scale,
    stride_qt, stride_qh, stride_qd,
    stride_k_b, stride_k_bs, stride_k_h, stride_k_d,
    stride_v_b, stride_v_bs, stride_v_h, stride_v_d,
    stride_bt_b, stride_bt_s,
    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_d,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_s,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr # GQA
):
    # Grid: (batch_size, num_heads, split_k)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    # GQA: Map Head (0..31) to KV Head (0..7) if GROUP_SIZE=4
    pid_kv_h = pid_h // GROUP_SIZE

    # Load Q: [1, HEAD_DIM] for this batch and head
    # Q shape: [batch, heads, dim]
    q_ptr = Q + pid_b * stride_qt + pid_h * stride_qh
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + offs_d * stride_qd)
    
    # Context length for this sequence
    context_len = tl.load(Context_lens + pid_b)
    
    # Total number of blocks for this sequence
    # Each block has BLOCK_SIZE_KV tokens
    # num_blocks = (context_len + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    # But we iterate based on splits.
    
    # We want to divide the work (context_len) among SPLIT_K blocks
    total_kv_blocks = (context_len + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    import math
    # blocks_per_split = (total_kv_blocks + SPLIT_K - 1) // SPLIT_K
    blocks_per_split = triton.cdiv(total_kv_blocks, SPLIT_K)
    
    start_block_idx = pid_s * blocks_per_split
    end_block_idx = min((pid_s + 1) * blocks_per_split, total_kv_blocks)
    
    
    # Initialize accumulators
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    qk_scale = sm_scale * 1.44269504
    
    offs_n = tl.arange(0, BLOCK_N)
    
    # Loop over blocks assigned to this split
    for block_idx in range(start_block_idx, end_block_idx):
        # Get physical block index from block table
        # block_table shape: [batch, max_blocks]
        # offset = pid_b * stride_bt_b + block_idx * stride_bt_s
        physical_block_id = tl.load(Block_tables + pid_b * stride_bt_b + block_idx * stride_bt_s)
        
        # Base pointers for K and V in this block
        # Use pid_kv_h instead of pid_h
        k_base = K_cache + physical_block_id * stride_k_b + pid_kv_h * stride_k_h
        v_base = V_cache + physical_block_id * stride_v_b + pid_kv_h * stride_v_h
        
        # Iterate over tokens in the block (BLOCK_SIZE_KV usually 16 or 32)
        # Assuming BLOCK_N matches BLOCK_SIZE_KV or we loop inside
        # Use BLOCK_N for loading. If BLOCK_SIZE_KV > BLOCK_N, loop.
        # If BLOCK_SIZE_KV < BLOCK_N, need masking.
        # Usually BLOCK_SIZE_KV is small (16), BLOCK_N in Triton is >= 16.
        # Let's assume BLOCK_N == BLOCK_SIZE_KV for simplicity or handle it.
        
        # For this example, let's strictly load BLOCK_SIZE_KV tokens.
        # Assuming BLOCK_SIZE_KV is power of 2 and <= 128
        
        # Load K, V
        # K shape: [num_blocks, block_size, heads, dim]
        # Ptr: k_base + offs_n * stride_k_bs + offs_d * stride_k_d
        
        # Mask for valid tokens in the last block
        # context_len might not align with block boundary
        is_last_block = (block_idx == total_kv_blocks - 1)
        valid_tokens = BLOCK_SIZE_KV
        if is_last_block:
             valid_tokens = context_len % BLOCK_SIZE_KV
             if valid_tokens == 0: valid_tokens = BLOCK_SIZE_KV
        
        mask = offs_n < valid_tokens
        
        k = tl.load(k_base + offs_n[:, None] * stride_k_bs + offs_d[None, :] * stride_k_d, mask=mask[:, None], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_v_bs + offs_d[None, :] * stride_v_d, mask=mask[:, None], other=0.0)
        
        # QK
        # q: [HEAD_DIM], k: [BLOCK_N, HEAD_DIM]
        # qk: [BLOCK_N]
        qk = tl.sum(q[None, :] * k, 1)
        qk *= qk_scale
        
        # Mask
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 0)
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, 0)
        
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # acc update
        # acc: [HEAD_DIM]
        # p: [BLOCK_N]
        # v: [BLOCK_N, HEAD_DIM]
        # p * v -> sum -> [HEAD_DIM]
        pv = tl.sum(p[:, None] * v, 0)
        
        acc = acc * alpha + pv * beta
        l_i = l_i * alpha + l_ij * beta
        m_i = m_new

    # Store partial results
    # Mid_O: [batch, heads, split, dim]
    # Mid_L: [batch, heads, split]
    
    off_mid_o = pid_b * stride_mid_o_b + pid_h * stride_mid_o_h + pid_s * stride_mid_o_s
    off_mid_l = pid_b * stride_mid_l_b + pid_h * stride_mid_l_h + pid_s * stride_mid_l_s
    
    tl.store(Mid_O + off_mid_o + offs_d * stride_mid_o_d, acc)
    tl.store(Mid_L + off_mid_l, m_i + tl.log(l_i)) # Store logsumexp for stability in reduction

def flash_decoding_fwd(q, k_cache, v_cache, block_tables, context_lens, max_seq_len, softmax_scale=None):
    # q: [batch, heads, dim]
    # k_cache: [num_blocks, block_size, heads, dim]
    
    batch_size, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, kv_head_dim = k_cache.shape
    
    # assert num_heads == num_kv_heads, "GQA/MQA not fully implemented in simpler kernel yet"
    assert num_heads % num_kv_heads == 0, f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
    group_size = num_heads // num_kv_heads
    
    SPLIT_K = 4 # Tuning parameter

    
    mid_o = torch.zeros((batch_size, num_heads, SPLIT_K, head_dim), device=q.device, dtype=torch.float32)
    mid_l = torch.zeros((batch_size, num_heads, SPLIT_K), device=q.device, dtype=torch.float32)
    
    grid = (batch_size, num_heads, SPLIT_K)
    
    _flash_decoding_fwd_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens,
        mid_o, mid_l,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
        BLOCK_SIZE_KV=block_size, 
        HEAD_DIM=head_dim,
        BLOCK_N=block_size, # Assuming BLOCK_N matches block_size for simplicity in this kernel optimized for PagedAttention
        SPLIT_K=SPLIT_K,
        GROUP_SIZE=group_size
    )
    
    # Reduction Buffer
    # We can do reduction in Python for simplicity if SPLIT_K is small, 
    # or write a reduction kernel.
    # For high performance, reduction kernel is better.
    
    # Let's implement a simple reduction kernel in reduce_kernel.py
    # For now, return mid_o, mid_l to be reduced.
    return mid_o, mid_l

