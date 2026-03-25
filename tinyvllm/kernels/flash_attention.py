
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    cu_seqlens_q, cu_seqlens_k,
    num_seqlens,
    max_seqlen_q, max_seqlen_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr  # GQA Group Size
):
    # Grid: (num_m_blocks, batch_size, num_heads)
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # GQA: Map Head (0..31) to KV Head (0..7) if GROUP_SIZE=4
    pid_kv_h = pid_h // GROUP_SIZE

    # Get start/end index for this sequence
    # cu_seqlens_q is pointer to int32 tensor
    start_seq_q = tl.load(cu_seqlens_q + pid_b)
    end_seq_q = tl.load(cu_seqlens_q + pid_b + 1)
    len_seq_q = end_seq_q - start_seq_q
    
    start_seq_k = tl.load(cu_seqlens_k + pid_b)
    end_seq_k = tl.load(cu_seqlens_k + pid_b + 1)
    len_seq_k = end_seq_k - start_seq_k

    # Current Block start index in the sequence
    start_m = pid_m * BLOCK_M
    
    # If this block is completely outside the sequence, exit
    if start_m >= len_seq_q:
        return

    # Offsets in Q, K, V (flattened)
    # Q uses pid_h
    q_offset = (start_seq_q + start_m) * stride_qm + pid_h * stride_qh
    # K, V use pid_kv_h
    k_offset = (start_seq_k) * stride_kn + pid_kv_h * stride_kh
    v_offset = (start_seq_k) * stride_vn + pid_kv_h * stride_vh
    o_offset = (start_seq_q + start_m) * stride_om + pid_h * stride_oh

    qs_ptr = Q + q_offset
    ks_ptr = K + k_offset
    vs_ptr = V + v_offset
    os_ptr = Out + o_offset

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q
    # Mask check: start_m + offs_m < len_seq_q
    q_ptrs = qs_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    mask_m = (start_m + offs_m) < len_seq_q
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Init
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    # Causal loop
    # We attend to k in range [0, start_m + BLOCK_M] (roughly) for causal
    # But strictly: token i usually attends to 0..i
    
    # Range of N blocks
    low = 0
    # For causal, max N we need to check is roughly (start_m + 1) * BLOCK_M
    # But we also need to respect len_seq_k
    high = len_seq_k
    if high > (start_m + BLOCK_M): # Causal optimization
         high = (start_m + BLOCK_M)
    # Round up to BLOCK_N
    high = ((high + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    for start_n in range(low, high, BLOCK_N):
        # Load K, V
        k_ptrs = ks_ptr + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        v_ptrs = vs_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        
        mask_n = (start_n + offs_n) < len_seq_k
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # QK
        qk = tl.dot(q, k)
        qk *= qk_scale
        
        # Causal Mask
        # global_m = start_m + offs_m
        # global_n = start_n + offs_n
        # mask = global_n <= global_m
        mask_causal = (start_n + offs_n[None, :]) <= (start_m + offs_m[:, None])
        qk = tl.where(mask_causal, qk, float("-inf"))
        
        mask_n_bool = (start_n + offs_n[None, :]) < len_seq_k
        qk = tl.where(mask_n_bool, qk, float("-inf"))

        # Softmax stats
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v) * beta[:, None]
        l_i = l_i * alpha + l_ij * beta
        m_i = m_new

    # Store
    acc = acc / l_i[:, None]
    out_ptrs = os_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None])


def flash_attn2_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True, softmax_scale=None):
    # wrapper
    # q, k, v are 3D: [total_tokens, heads, head_dim]
    
    total_tokens, num_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape # Get KV heads
    
    # Validation 
    assert num_heads % num_kv_heads == 0, f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
    group_size = num_heads // num_kv_heads

    # num_seqlens = batch_size + 1
    num_seqlens = cu_seqlens_q.numel()
    batch_size = num_seqlens - 1
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Grid
    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size, num_heads)
    
    o = torch.empty_like(q)
    
    # Expect strides
    _flash_attn_fwd_kernel[grid](
        q, k, v, softmax_scale,
        None, o, # L is not needed for inference usually
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        cu_seqlens_q, cu_seqlens_k,
        num_seqlens,
        max_seqlen_q, max_seqlen_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size
    )
    return o


