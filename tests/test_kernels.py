
import sys
from unittest.mock import MagicMock
sys.modules["transformers"] = MagicMock()

import torch
import torch.nn.functional as F
from tinyvllm.kernels import flash_attn2_fwd, flash_decoding_fwd, reduction

def naive_attention(q, k, v, scale):
    # q: [B, H, D] or [B, S, H, D]
    # k, v: [B, S, H, D]
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out

def test_flash_attn2():
    print("Testing Flash Attention 2 Prefill...")
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 128
    num_heads = 4
    head_dim = 64
    dtype = torch.float16
    device = "cuda"

    # Create simplified inputs
    # flatten: [batch * seq, heads, dim]
    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    
    # cu_seqlens
    cu_seqlens_q = torch.tensor([0, seq_len, 2*seq_len], device=device, dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()
    max_seqlen_q = seq_len
    max_seqlen_k = seq_len
    
    out = flash_attn2_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, softmax_scale=1.0)
    
    # Reference
    # Reshape to [B, S, H, D]
    q_ref = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # [B, H, S, D]
    k_ref = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v_ref = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    ref_out = naive_attention(q_ref, k_ref, v_ref, 1.0) # [B, H, S, D]
    ref_out = ref_out.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
    
    # Check
    # Note: Triton fp16 might have some precision diffs, but should be small
    diff = (out - ref_out).abs().max()
    print(f"Max Diff: {diff}")
    if diff < 1e-2:
        print("PASS")
    else:
        print("FAIL")

def test_flash_decoding():
    print("Testing Flash Decoding...")
    torch.manual_seed(0)
    batch_size = 2
    num_heads = 4
    head_dim = 64
    block_size = 16
    max_blocks = 8
    dtype = torch.float16
    device = "cuda"

    # Queries: [B, H, D]
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # KV Cache setup
    num_blocks = 100
    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # Block tables
    # Seq 0: uses blocks [0, 1] -> len 32
    # Seq 1: uses blocks [2, 3, 4] -> len 48
    block_tables = torch.full((batch_size, max_blocks), -1, device=device, dtype=torch.int32)
    block_tables[0, 0] = 0
    block_tables[0, 1] = 1
    block_tables[1, 0] = 2
    block_tables[1, 1] = 3
    block_tables[1, 2] = 4
    
    context_lens = torch.tensor([32, 48], device=device, dtype=torch.int32)
    max_seq_len = 48
    
    mid_o, mid_l = flash_decoding_fwd(q, k_cache, v_cache, block_tables, context_lens, max_seq_len, softmax_scale=1.0)
    out = reduction(mid_o, mid_l)
    
    # Reference
    # Construct K, V for each seq from cache
    ref_outs = []
    
    for b in range(batch_size):
        c_len = context_lens[b].item()
        blocks = block_tables[b]
        # Gather K, V
        # naive implementation of gathering
        cur_k = []
        cur_v = []
        for i in range(max_blocks):
             blk_id = blocks[i].item()
             if blk_id == -1: break
             cur_k.append(k_cache[blk_id]) # [block_size, H, D]
             cur_v.append(v_cache[blk_id])
        
        cur_k = torch.cat(cur_k, dim=0)[:c_len] # [L, H, D]
        cur_v = torch.cat(cur_v, dim=0)[:c_len]
        
        # Q: [1, H, D]
        cur_q = q[b].unsqueeze(0)
        
        # Attention
        # [1, H, D] x [L, H, D].T -> [1, H, L]
        # Transpose K to [H, D, L]
        # cur_k: [L, H, D] -> [H, L, D]
        cur_k = cur_k.transpose(0, 1)
        cur_v = cur_v.transpose(0, 1)
        cur_q = cur_q.transpose(0, 1) # [H, 1, D]
        
        attn = torch.matmul(cur_q, cur_k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        cur_out = torch.matmul(attn, cur_v) # [H, 1, D]
        
        ref_outs.append(cur_out.transpose(0, 1)) # [1, H, D]
        
    ref_out_batch = torch.cat(ref_outs, dim=0)
    
    diff = (out - ref_out_batch).abs().max()
    print(f"Max Diff Decode: {diff}")
    if diff < 1e-2:
        print("PASS")
    else:
        print("FAIL")


def test_gqa_decoding():
    print("Testing GQA Flash Decoding...")
    torch.manual_seed(0)
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2 # GQA 2:1
    head_dim = 64
    block_size = 16
    max_blocks = 8
    dtype = torch.float16
    device = "cuda"

    # Queries: [B, H, D]
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    
    # KV Cache setup
    num_blocks = 100
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    block_tables = torch.full((batch_size, max_blocks), -1, device=device, dtype=torch.int32)
    block_tables[0, 0] = 0
    block_tables[0, 1] = 1
    block_tables[1, 0] = 2
    block_tables[1, 1] = 3
    block_tables[1, 2] = 4
    
    context_lens = torch.tensor([32, 48], device=device, dtype=torch.int32)
    max_seq_len = 48
    
    mid_o, mid_l = flash_decoding_fwd(q, k_cache, v_cache, block_tables, context_lens, max_seq_len, softmax_scale=1.0)
    out = reduction(mid_o, mid_l)
    
    # Reference
    ref_outs = []
    
    for b in range(batch_size):
        c_len = context_lens[b].item()
        blocks = block_tables[b]
        
        cur_k = []
        cur_v = []
        for i in range(max_blocks):
             blk_id = blocks[i].item()
             if blk_id == -1: break
             cur_k.append(k_cache[blk_id]) # [block_size, H_kv, D]
             cur_v.append(v_cache[blk_id])
        
        cur_k = torch.cat(cur_k, dim=0)[:c_len] # [L, H_kv, D]
        cur_v = torch.cat(cur_v, dim=0)[:c_len]
        
        # Repeat/Interleave KV for GQA
        # simple repeat: [L, H_kv, D] -> [L, H_q, D]
        # We need to map h_q to h_kv
        # h_kv = h_q // group_size
        
        group_size = num_heads // num_kv_heads
        cur_k_expanded = cur_k.repeat_interleave(group_size, dim=1) # [L, H_q, D]
        cur_v_expanded = cur_v.repeat_interleave(group_size, dim=1)
        
        # Q: [1, H, D]
        cur_q = q[b].unsqueeze(0)
        
        cur_k_expanded = cur_k_expanded.transpose(0, 1) # [H, L, D]
        cur_v_expanded = cur_v_expanded.transpose(0, 1)
        cur_q = cur_q.transpose(0, 1) # [H, 1, D]
        
        attn = torch.matmul(cur_q, cur_k_expanded.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        cur_out = torch.matmul(attn, cur_v_expanded) # [H, 1, D]
        
        ref_outs.append(cur_out.transpose(0, 1)) # [1, H, D]
        
    ref_out_batch = torch.cat(ref_outs, dim=0)
    
    diff = (out - ref_out_batch).abs().max()
    print(f"Max Diff GQA Decode: {diff}")
    if diff < 1e-2:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_flash_attn2()
    test_flash_decoding()
    test_gqa_decoding()
