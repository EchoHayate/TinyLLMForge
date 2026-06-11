import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from tinyvllm.utils.context import get_context
from tinyvllm.engine.attention_matching import attention_matching_decode

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


# ============================================================================
# C4 / KV cache 4-bit 量化路径
# ============================================================================
# group-wise 对称量化：每 group 个 fp16/bf16 数 -> 1 个 fp scale + group_size 个 int4
# 对称量化的反量化：x_fp = x_int4 * scale，量化范围 [-8, 7]
# 4-bit pack：把 group_size 个 int4（实际存为 int8 寄存器）打成 group_size/2 个 int8
#              低 4 位 = 偶数索引，高 4 位 = 奇数索引（注意 sign-extension 在反量化时处理）
#
# 设计上 head_dim 必须能被 group_size 整除；group_size 必须为偶数。
# 实现上把 [num_kv_heads, head_dim] 拍平成 [num_kv_heads * head_dim] 一条 1D 处理。
@triton.jit
def store_kvcache_q4_kernel(
    key_ptr,                # [N, num_kv_heads, head_dim]，dtype = fp16/bf16
    key_stride_n,           # = num_kv_heads * head_dim
    value_ptr,
    value_stride_n,
    k_cache_ptr,            # int8 packed，按 token-major 存：slot * (D//2)
    v_cache_ptr,
    k_scale_ptr,            # fp，按 slot * num_groups 存
    v_scale_ptr,
    slot_mapping_ptr,       # [N] int32
    D: tl.constexpr,        # = num_kv_heads * head_dim
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,     # = GROUP_SIZE // 2
    NUM_GROUPS: tl.constexpr,  # = D // GROUP_SIZE
):
    pid = tl.program_id(0)         # token id (= 0..N-1)
    gid = tl.program_id(1)         # group id (= 0..NUM_GROUPS-1)

    base_in_k = pid * key_stride_n + gid * GROUP_SIZE
    base_in_v = pid * value_stride_n + gid * GROUP_SIZE

    # 一次性 load 整 group 求 amax；再分两次（偶/奇 stride=2）load 用于 pack
    full_offs = tl.arange(0, GROUP_SIZE)
    k_full = tl.load(key_ptr + base_in_k + full_offs).to(tl.float32)
    v_full = tl.load(value_ptr + base_in_v + full_offs).to(tl.float32)

    # symmetric int4：scale = max(|x|) / 7（[-8,7] 区间，对称用 7）
    k_scale = tl.max(tl.abs(k_full), axis=0) / 7.0 + 1e-8
    v_scale = tl.max(tl.abs(v_full), axis=0) / 7.0 + 1e-8

    # 偶/奇 lane 分别量化，pack 进 int8（低 4 位 = 偶，高 4 位 = 奇）
    even_offs = tl.arange(0, HALF) * 2
    odd_offs = even_offs + 1

    k_lo_f = tl.load(key_ptr + base_in_k + even_offs).to(tl.float32)
    k_hi_f = tl.load(key_ptr + base_in_k + odd_offs).to(tl.float32)
    v_lo_f = tl.load(value_ptr + base_in_v + even_offs).to(tl.float32)
    v_hi_f = tl.load(value_ptr + base_in_v + odd_offs).to(tl.float32)

    k_lo = tl.minimum(tl.maximum(libdevice.rint(k_lo_f / k_scale), -8.0), 7.0).to(tl.int32) & 0xF
    k_hi = tl.minimum(tl.maximum(libdevice.rint(k_hi_f / k_scale), -8.0), 7.0).to(tl.int32) & 0xF
    v_lo = tl.minimum(tl.maximum(libdevice.rint(v_lo_f / v_scale), -8.0), 7.0).to(tl.int32) & 0xF
    v_hi = tl.minimum(tl.maximum(libdevice.rint(v_hi_f / v_scale), -8.0), 7.0).to(tl.int32) & 0xF

    k_packed = (k_lo | (k_hi << 4)).to(tl.int8)
    v_packed = (v_lo | (v_hi << 4)).to(tl.int8)

    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    out_offs = slot * (D // 2) + gid * HALF + tl.arange(0, HALF)
    tl.store(k_cache_ptr + out_offs, k_packed)
    tl.store(v_cache_ptr + out_offs, v_packed)

    # scale：按 slot * NUM_GROUPS 存
    tl.store(k_scale_ptr + slot * NUM_GROUPS + gid, k_scale)
    tl.store(v_scale_ptr + slot * NUM_GROUPS + gid, v_scale)


def store_kvcache_q4(
    key: torch.Tensor,            # [N, num_kv_heads, head_dim] fp
    value: torch.Tensor,
    k_cache: torch.Tensor,        # int8 packed [num_blocks, block_size, num_kv_heads, head_dim/2]
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,        # fp [num_blocks, block_size, num_kv_heads, num_groups]
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    group_size: int,
):
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    num_groups_total = D // group_size  # 注意这里把 num_kv_heads 也展平了
    assert head_dim % group_size == 0
    assert group_size % 2 == 0
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    # 调度：每个 (token, group) 一个 program
    store_kvcache_q4_kernel[(N, num_groups_total)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        k_scale, v_scale,
        slot_mapping,
        D=D,
        GROUP_SIZE=group_size,
        HALF=group_size // 2,
        NUM_GROUPS=num_groups_total,
    )


@triton.jit
def store_kvcache_q8_kernel(
    key_ptr,                # [N, num_kv_heads, head_dim] fp16/bf16
    key_stride_n,           # = num_kv_heads * head_dim
    value_ptr,
    value_stride_n,
    k_cache_ptr,            # int8 [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,
    k_scale_ptr,            # fp，按 slot * num_groups 存
    v_scale_ptr,
    slot_mapping_ptr,       # [N] int32
    D: tl.constexpr,        # = num_kv_heads * head_dim
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,  # = D // GROUP_SIZE
):
    pid = tl.program_id(0)         # token id
    gid = tl.program_id(1)         # group id

    base_in_k = pid * key_stride_n + gid * GROUP_SIZE
    base_in_v = pid * value_stride_n + gid * GROUP_SIZE
    offs = tl.arange(0, GROUP_SIZE)
    k_full = tl.load(key_ptr + base_in_k + offs).to(tl.float32)
    v_full = tl.load(value_ptr + base_in_v + offs).to(tl.float32)

    # symmetric int8：scale = max(|x|) / 127（[-128,127] 区间，对称用 127）
    k_scale = tl.max(tl.abs(k_full), axis=0) / 127.0 + 1e-8
    v_scale = tl.max(tl.abs(v_full), axis=0) / 127.0 + 1e-8

    k_q = tl.minimum(tl.maximum(libdevice.rint(k_full / k_scale), -128.0), 127.0).to(tl.int8)
    v_q = tl.minimum(tl.maximum(libdevice.rint(v_full / v_scale), -128.0), 127.0).to(tl.int8)

    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    out_offs = slot * D + gid * GROUP_SIZE + offs
    tl.store(k_cache_ptr + out_offs, k_q)
    tl.store(v_cache_ptr + out_offs, v_q)

    tl.store(k_scale_ptr + slot * NUM_GROUPS + gid, k_scale)
    tl.store(v_scale_ptr + slot * NUM_GROUPS + gid, v_scale)


def store_kvcache_q8(
    key: torch.Tensor,            # [N, num_kv_heads, head_dim] fp
    value: torch.Tensor,
    k_cache: torch.Tensor,        # int8 [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,        # fp [num_blocks, block_size, num_kv_heads, num_groups]
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    group_size: int,
):
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    num_groups_total = D // group_size
    assert head_dim % group_size == 0
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    store_kvcache_q8_kernel[(N, num_groups_total)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        k_scale, v_scale,
        slot_mapping,
        D=D,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups_total,
    )


def dequant_kv_blocks_q8(
    cache_q: torch.Tensor,        # int8 [num_blocks_total, block_size, num_kv_heads, head_dim]
    cache_scale: torch.Tensor,    # fp   [num_blocks_total, block_size, num_kv_heads, num_groups]
    block_tables: torch.Tensor,   # int32 [B, max_blocks], -1 padded
    group_size: int,
    out_dtype: torch.dtype,
):
    """C8 反量化：gather batch 命中块（int8），按 group scale 展开成 fp。

    与 dequant_kv_blocks（int4）等价语义，但 int8 不需要 nibble 解包/符号扩展。
    """
    B, max_blocks = block_tables.shape
    nb_total, block_size, num_kv_heads, head_dim = cache_q.shape
    num_groups = cache_scale.shape[-1]
    assert num_groups * group_size == head_dim, "head_dim 必须 = num_groups * group_size"

    valid_mask = block_tables >= 0
    safe_idx = block_tables.clamp_min(0).to(torch.long)

    q_b = cache_q[safe_idx]                                  # [B, max_blocks, block_size, kv_h, head_dim] int8
    scale_b = cache_scale[safe_idx]                          # [..., num_groups]
    scale_exp = scale_b.repeat_interleave(group_size, dim=-1)  # [..., head_dim]
    cache_fp = q_b.to(out_dtype) * scale_exp.to(out_dtype)

    nb_active = B * max_blocks
    cache_fp = cache_fp.reshape(nb_active, block_size, num_kv_heads, head_dim).contiguous()
    new_table = (torch.arange(nb_active, device=block_tables.device, dtype=torch.int32)
                 .view(B, max_blocks))
    new_table = torch.where(valid_mask, new_table, torch.full_like(new_table, -1))
    return cache_fp, new_table


def dequant_kv_blocks(
    cache_packed: torch.Tensor,   # int8 [num_blocks_total, block_size, num_kv_heads, head_dim/2]
    cache_scale: torch.Tensor,    # fp   [num_blocks_total, block_size, num_kv_heads, num_groups]
    block_tables: torch.Tensor,   # int32 [B, max_blocks], -1 padded
    group_size: int,
    out_dtype: torch.dtype,
):
    """A-2 朴素反量化：把 batch 实际访问的所有 block (int4 packed) gather 出来并展开成 fp16。

    返回:
      cache_fp:    [B*max_blocks, block_size, num_kv_heads, head_dim] (out_dtype)
      new_table:   [B, max_blocks] int32，identity 编号；padding 位置保持 -1
    备注:
      - padding 位（block_tables == -1）会被 clamp 到 0 然后产生 garbage 数据，
        但 flash-attn 的 cache_seqlens 已经限定了实际读取的 token 数，所以无害。
      - int4 sign-extension：用 int32 算术左/右移；torch 的 int32 `>>` 是算术右移。
      - 内存：B*max_blocks*block_size*num_kv_heads*head_dim*sizeof(fp)。
        15k 上下文 batch=16 时约 ~480MB/层 K + 480MB/层 V = ~1GB 瞬态，A-3 会优化。
    """
    B, max_blocks = block_tables.shape
    nb_total, block_size, num_kv_heads, half = cache_packed.shape
    head_dim = half * 2
    num_groups = cache_scale.shape[-1]
    assert num_groups * group_size == head_dim, "head_dim 必须 = num_groups * group_size"

    valid_mask = block_tables >= 0
    safe_idx = block_tables.clamp_min(0).to(torch.long)  # [B, max_blocks]

    # gather： [B, max_blocks, block_size, num_kv_heads, head_dim/2]   int8
    pack_b = cache_packed[safe_idx]
    scale_b = cache_scale[safe_idx]                        # [B, max_blocks, block_size, num_kv_heads, num_groups]

    # int4 nibble 解包到 int32（带符号），然后转 fp
    p32 = pack_b.to(torch.int32)
    low = (p32 << 28) >> 28          # 低 4 位 sign-extend
    high = (p32 << 24) >> 28         # 高 4 位 sign-extend
    # 交错 low/high → [..., head_dim]，与 store 时 even=low/odd=high 对齐
    nibbles = torch.stack([low, high], dim=-1).flatten(-2)  # [..., head_dim] int32 in [-8,7]

    # 把 scale 沿 head_dim 复制 group_size 次
    scale_exp = scale_b.repeat_interleave(group_size, dim=-1)  # [..., head_dim]
    # 反量化
    cache_fp = nibbles.to(out_dtype) * scale_exp.to(out_dtype)

    # 展平 (B, max_blocks) -> 单一 block 轴，给 flash-attn block_table 用
    nb_active = B * max_blocks
    cache_fp = cache_fp.reshape(nb_active, block_size, num_kv_heads, head_dim).contiguous()
    new_table = (torch.arange(nb_active, device=block_tables.device, dtype=torch.int32)
                 .view(B, max_blocks))
    new_table = torch.where(valid_mask, new_table, torch.full_like(new_table, -1))
    return cache_fp, new_table




def update_block_summary(
    key: torch.Tensor,           # [N, num_kv_heads, head_dim]
    slot_mapping: torch.Tensor,  # [N] int32, slot = block_id * block_size + intra
    k_min: torch.Tensor,         # [num_blocks, num_kv_heads, head_dim] fp32
    k_max: torch.Tensor,         # [num_blocks, num_kv_heads, head_dim] fp32
    block_size: int,
):
    """Quest：对刚写入的 token，按其所在 block_id 维护 per-channel min/max。

    走 fused triton kernel：一个 program 处理一个 token 的整 (kv_h, head_dim) 向量，
    triton 的 `atomic_min/atomic_max` 直接打到 fp32 buffer 上。
    与原 index_reduce_(amin/amax)×2 等价但只走一次 HBM、且只 launch 一次 kernel。
    """
    N, num_kv_heads, head_dim = key.shape
    KVHD = num_kv_heads * head_dim
    assert k_min.dtype == torch.float32 and k_max.dtype == torch.float32, \
        "k_min/k_max must be fp32 for atomic_min/atomic_max"
    update_block_summary_kernel[(N,)](
        key, slot_mapping, k_min, k_max,
        BLOCK_SIZE=block_size,
        KVHD=KVHD,
        BLOCK=_next_pow2(KVHD),
    )


@triton.jit
def update_block_summary_kernel(
    key_ptr,             # [N, KVHD] fp16/bf16
    slot_ptr,            # [N] int32, slot = block_id * block_size + intra
    k_min_ptr,           # [num_blocks, KVHD] fp32
    k_max_ptr,           # [num_blocks, KVHD] fp32
    BLOCK_SIZE: tl.constexpr,
    KVHD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # 一个 token 一个 program
    slot = tl.load(slot_ptr + pid)
    block_id = (slot // BLOCK_SIZE).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < KVHD
    k = tl.load(key_ptr + pid * KVHD + offs, mask=mask, other=0.0).to(tl.float32)
    base = block_id * KVHD + offs
    # atomic_min/atomic_max 直接打到 fp32 summary buffer
    tl.atomic_min(k_min_ptr + base, k, mask=mask)
    tl.atomic_max(k_max_ptr + base, k, mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def quest_score_kernel(
    q_repr_ptr,         # [B, KV_H, D] fp16/bf16, contig
    k_min_ptr,          # [N_BLOCKS, KV_H, D]
    k_max_ptr,
    block_tables_ptr,   # [B, MAX_BLOCKS] int32
    out_ptr,            # [B, MAX_BLOCKS] fp32
    MAX_BLOCKS: tl.constexpr,
    KVHD: tl.constexpr,     # = num_kv_heads * head_dim
    BLOCK: tl.constexpr,    # next pow2 >= KVHD
):
    """每个 program 算一个 (b, n) 的 criticality：
       score = sum_{h,d} max(q_repr[b,h,d] * k_min[block_id,h,d],
                              q_repr[b,h,d] * k_max[block_id,h,d])
       gather + 双 einsum + max + sum 在一个 kernel 内完成，省掉 [B,N,kv_h,d] 中间 tensor。
       reduce 在 fp32 上做（比原 einsum-fp16 reduce 更稳）。
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    bt_off = pid_b * MAX_BLOCKS + pid_n
    block_id = tl.load(block_tables_ptr + bt_off)
    valid = block_id >= 0
    safe_id = tl.where(valid, block_id, 0).to(tl.int64)

    offs = tl.arange(0, BLOCK)
    mask = offs < KVHD
    q = tl.load(q_repr_ptr + pid_b * KVHD + offs, mask=mask, other=0.0).to(tl.float32)
    kmin = tl.load(k_min_ptr + safe_id * KVHD + offs, mask=mask, other=0.0).to(tl.float32)
    kmax = tl.load(k_max_ptr + safe_id * KVHD + offs, mask=mask, other=0.0).to(tl.float32)
    score = tl.sum(tl.maximum(q * kmin, q * kmax), axis=0)
    score = tl.where(valid, score, float("-inf"))
    tl.store(out_ptr + bt_off, score)


def quest_select_blocks(
    q: torch.Tensor,                 # [B, num_q_heads, head_dim]，已 RoPE
    block_tables: torch.Tensor,      # [B, max_blocks] int32
    context_lens: torch.Tensor,      # [B] int32
    k_min_layer: torch.Tensor,       # [num_blocks_total, num_kv_heads, head_dim]
    k_max_layer: torch.Tensor,       # [num_blocks_total, num_kv_heads, head_dim]
    block_size: int,
    top_k: int,
):
    """Quest 选块：对每个 batch row，估计每个 block 的 max-inner-product 上界，取 top-k。

    前置条件（在 prepare_decode host 端已校验）：
      - batch 内所有 row 都满足 num_blocks > top_k
      - 不需要再做 .item() / .min() 等 sync 判断
    """
    B, max_blocks = block_tables.shape
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k_min_layer.shape[1]
    n_groups = num_q_heads // num_kv_heads  # GQA group

    valid_mask = block_tables >= 0                      # [B, max_blocks]

    # GQA: 把 q head 按 kv head 分组，组内取 max（保激进估计）
    q_grouped = q.view(B, num_kv_heads, n_groups, head_dim)
    q_repr = q_grouped.amax(dim=2).contiguous()         # [B, kv_h, d]

    # 融合 kernel：gather + max(q·kmin, q·kmax) + sum 一气呵成
    KVHD = num_kv_heads * head_dim
    criticality = torch.empty(B, max_blocks, device=q.device, dtype=torch.float32)
    quest_score_kernel[(B, max_blocks)](
        q_repr,
        k_min_layer,
        k_max_layer,
        block_tables,
        criticality,
        MAX_BLOCKS=max_blocks,
        KVHD=KVHD,
        BLOCK=_next_pow2(KVHD),
    )

    # 强制保留：首 block (attention sink) + 末 block (recency / partial)
    num_blocks_in_seq = (context_lens.to(torch.long) + block_size - 1) // block_size  # [B]
    arange = torch.arange(max_blocks, device=block_tables.device).unsqueeze(0)  # [1, max_blocks]
    last_idx = (num_blocks_in_seq - 1).clamp_min(0).unsqueeze(1)
    must_keep = ((arange == 0) | (arange == last_idx)) & valid_mask
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


def gather_kv_cache_dense(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather block-table KV cache into dense `[B, max_tokens, kv_h, head_dim]` tensors."""
    safe_idx = block_tables.clamp_min(0).to(torch.long)
    B, max_blocks = block_tables.shape
    block_size = k_cache.shape[1]
    k_dense = k_cache[safe_idx].reshape(B, max_blocks * block_size, k_cache.shape[2], k_cache.shape[3])
    v_dense = v_cache[safe_idx].reshape(B, max_blocks * block_size, v_cache.shape[2], v_cache.shape[3])
    return k_dense.contiguous(), v_dense.contiguous()


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
        # C4：KV 量化辅助张量 + 标志位（model_runner 注入；未启用为 None / 0）
        self.k_scale = self.v_scale = None
        self.k_zero = self.v_zero = None
        self.kv_quant_bits = 0
        self.kv_quant_group_size = 128
        self.kv_quant_symmetric = True
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.kv_quant_bits == 4:
                # C4 写入路径：4-bit pack + group-wise scale
                # 注意 k_cache 此时是 int8 packed，shape [num_blocks, block_size, num_kv_heads, head_dim/2]
                store_kvcache_q4(
                    k, v,
                    k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.slot_mapping,
                    self.kv_quant_group_size,
                )
                # Quest summary：现在 K 已经被量化存了，summary 用反量化前的 k 维护更准
                if self.k_min is not None and self.k_max is not None:
                    update_block_summary(k, context.slot_mapping, self.k_min, self.k_max, k_cache.shape[1])
            elif self.kv_quant_bits == 8:
                # C8 写入路径：8-bit 对称 group 量化（不 pack）
                store_kvcache_q8(
                    k, v,
                    k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.slot_mapping,
                    self.kv_quant_group_size,
                )
                if self.k_min is not None and self.k_max is not None:
                    update_block_summary(k, context.slot_mapping, self.k_min, self.k_max, k_cache.shape[1])
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
                # Quest：写入 KV 后同步维护 per-block summary
                if self.k_min is not None and self.k_max is not None:
                    update_block_summary(k, context.slot_mapping, self.k_min, self.k_max, k_cache.shape[1])
        if context.is_prefill:
            # prefill传入的 q = [batch_size, seq_len, num_heads, head_dim]
            # 经过 view变成 q = [batch_size * seq_len, num_heads, head_dim]
            if self.kv_quant_bits in (4, 8):
                # C4/C8 prefill：
                #   - 无 prefix-cache：cache 里只有当前刚写入的 token；为避免再读一次量化后掉精度，
                #     直接用反量化前的 fp k/v 算 attention（等价语义，且省一次 dequant）。
                #   - 有 prefix-cache：必须把命中块从 int4/int8 反量化出来再喂 flash-attn。
                _dequant = dequant_kv_blocks if self.kv_quant_bits == 4 else dequant_kv_blocks_q8
                if context.block_tables is None:
                    o = flash_attn_varlen_func(
                        q, k, v,
                        cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
                        max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                        softmax_scale=self.scale, causal=True, block_table=None)
                else:
                    k_fp, new_bt = _dequant(
                        k_cache, self.k_scale, context.block_tables,
                        self.kv_quant_group_size, q.dtype)
                    v_fp, _ = _dequant(
                        v_cache, self.v_scale, context.block_tables,
                        self.kv_quant_group_size, q.dtype)
                    o = flash_attn_varlen_func(
                        q, k_fp, v_fp,
                        cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
                        max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                        softmax_scale=self.scale, causal=True, block_table=new_bt)
            else:
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

            # ---- C4 + Quest 叠加路径（β3）----
            # 单独 C4：要把命中块全部 dequant 成 fp16；瞬态 buffer ~ B*max_blocks*block_size*..
            #         A-3 评测显示长 ctx 下这层瞬态 buffer 把"省下的 KV 带宽"全吃了，反慢 5.4×
            # 叠加 Quest：先用 k_min/k_max summary 选 top_k 块，再"只对选中块"做 dequant，
            #             瞬态 buffer 缩到 B*top_k，长 ctx 下 max_blocks 远 > top_k，理论上能翻盘
            quest_active = (context.quest_top_k_blocks > 0
                            and self.k_min is not None
                            and block_tables is not None
                            and cache_seqlens is not None)
            am_active = (context.am_compact_blocks > 0
                         and block_tables is not None
                         and cache_seqlens is not None)

            if self.kv_quant_bits in (4, 8):
                _dequant = dequant_kv_blocks if self.kv_quant_bits == 4 else dequant_kv_blocks_q8
                if am_active:
                    assert self.kv_quant_bits == 8, "Attention Matching compact decode v0 仅支持 KV8 / fp16 KV"
                    k_fp, new_bt = _dequant(
                        k_cache, self.k_scale, block_tables,
                        self.kv_quant_group_size, q.dtype)
                    v_fp, _ = _dequant(
                        v_cache, self.v_scale, block_tables,
                        self.kv_quant_group_size, q.dtype)
                    B, max_blocks = block_tables.shape
                    block_size = k_fp.shape[1]
                    k_dense = k_fp.reshape(B, max_blocks * block_size, self.num_kv_heads, self.head_dim)
                    v_dense = v_fp.reshape(B, max_blocks * block_size, self.num_kv_heads, self.head_dim)
                    o = attention_matching_decode(
                        q,
                        k_dense,
                        v_dense,
                        cache_seqlens,
                        budget=context.am_compact_blocks,
                        selector=context.am_compact_selector,
                        score_method=context.am_compact_score_method,
                        beta_bound=context.am_compact_beta_bound,
                        ridge_lambda=context.am_compact_ridge_lambda,
                        omp_candidate_pool_size=context.am_omp_candidate_pool_size,
                    )
                elif quest_active:
                    # 1) 用未量化的 k_min/k_max（store 时维护，反量化前）算 criticality 选 top-k
                    sparse_bt, sparse_cs = quest_select_blocks(
                        q, block_tables, cache_seqlens,
                        self.k_min, self.k_max,
                        k_cache.shape[1],
                        context.quest_top_k_blocks,
                    )
                    # 2) 仅对选中的 top-k 块做反量化；buffer = [B*top_k, block_size, kv_h, hd]
                    k_fp, new_bt = _dequant(
                        k_cache, self.k_scale, sparse_bt,
                        self.kv_quant_group_size, q.dtype)
                    v_fp, _ = _dequant(
                        v_cache, self.v_scale, sparse_bt,
                        self.kv_quant_group_size, q.dtype)
                    o = flash_attn_with_kvcache(
                        q.unsqueeze(1), k_fp, v_fp,
                        cache_seqlens=sparse_cs,
                        block_table=new_bt, softmax_scale=self.scale, causal=True)
                else:
                    # C4/C8 单独路径（α）：把全部命中块 dequant，长 ctx 下慢，留作回退/对照
                    k_fp, new_bt = _dequant(
                        k_cache, self.k_scale, block_tables,
                        self.kv_quant_group_size, q.dtype)
                    v_fp, _ = _dequant(
                        v_cache, self.v_scale, block_tables,
                        self.kv_quant_group_size, q.dtype)
                    o = flash_attn_with_kvcache(
                        q.unsqueeze(1), k_fp, v_fp,
                        cache_seqlens=cache_seqlens,
                        block_table=new_bt, softmax_scale=self.scale, causal=True)
                o = o.view(-1, self.num_heads * self.head_dim)
                return o
            if am_active:
                k_dense, v_dense = gather_kv_cache_dense(k_cache, v_cache, block_tables)
                o = attention_matching_decode(
                    q,
                    k_dense,
                    v_dense,
                    cache_seqlens,
                    budget=context.am_compact_blocks,
                    selector=context.am_compact_selector,
                    score_method=context.am_compact_score_method,
                    beta_bound=context.am_compact_beta_bound,
                    ridge_lambda=context.am_compact_ridge_lambda,
                    omp_candidate_pool_size=context.am_omp_candidate_pool_size,
                )
                o = o.view(-1, self.num_heads * self.head_dim)
                return o
            # Quest（无 C4）：动态选 top-k block，重写 block_table 与 cache_seqlens
            if quest_active:
                block_tables, cache_seqlens = quest_select_blocks(
                    q, block_tables, cache_seqlens,
                    self.k_min, self.k_max,
                    k_cache.shape[1],
                    context.quest_top_k_blocks,
                )
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens = cache_seqlens,
                                        block_table = block_tables, softmax_scale = self.scale, causal = True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
