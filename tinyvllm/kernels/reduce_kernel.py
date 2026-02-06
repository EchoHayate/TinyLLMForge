
import torch
import triton
import triton.language as tl

@triton.jit
def _reduce_kernel(
    Mid_O, Mid_L, Out,
    stride_mid_o_b, stride_mid_o_h, stride_mid_o_s, stride_mid_o_d,
    stride_mid_l_b, stride_mid_l_h, stride_mid_l_s,
    stride_out_b, stride_out_h, stride_out_d,
    SPLIT_K: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # Grid: (batch, heads)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Offsets
    mid_o_ptr = Mid_O + pid_b * stride_mid_o_b + pid_h * stride_mid_o_h
    mid_l_ptr = Mid_L + pid_b * stride_mid_l_b + pid_h * stride_mid_l_h
    out_ptr = Out + pid_b * stride_out_b + pid_h * stride_out_h
    
    # Load all splits
    # We need to compute global max L
    # l_global = logsumexp(l_i)
    # But Mid_L stores lse_i (logsumexp) or just max + log...
    # In flash decoding, usually we store (m, l) or just lse.
    # Logic:
    # L_final = logsumexp(L_i for i in splits)
    # O_final = sum(O_i * exp(L_i - L_final))
    
    # Let's iterate over splits
    
    # Find max_l first
    max_l = float("-inf")
    for s in range(SPLIT_K):
        l_s = tl.load(mid_l_ptr + s * stride_mid_l_s)
        if l_s > max_l:
            max_l = l_s
            
    # Compute sum_exp
    sum_exp = 0.0
    for s in range(SPLIT_K):
        l_s = tl.load(mid_l_ptr + s * stride_mid_l_s)
        sum_exp += tl.exp(l_s - max_l)
        
    l_final = max_l + tl.log(sum_exp)
    
    # Compute weighted sum
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offs_d = tl.arange(0, HEAD_DIM)
    
    for s in range(SPLIT_K):
        l_s = tl.load(mid_l_ptr + s * stride_mid_l_s)
        o_s = tl.load(mid_o_ptr + s * stride_mid_o_s + offs_d * stride_mid_o_d)
        
        weight = tl.exp(l_s - l_final)
        acc += o_s * weight
        
    # Store
    tl.store(out_ptr + offs_d * stride_out_d, acc.to(tl.float16))

def reduction(mid_o, mid_l):
    batch, heads, split_k, dim = mid_o.shape
    out = torch.empty((batch, heads, dim), device=mid_o.device, dtype=torch.float16)
    
    grid = (batch, heads)
    _reduce_kernel[grid](
        mid_o, mid_l, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        SPLIT_K=split_k,
        HEAD_DIM=dim
    )
    return out
