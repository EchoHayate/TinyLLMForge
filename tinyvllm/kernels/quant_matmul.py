import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def w8a16_gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, scales_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_scale_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Computes: C = A @ diag(scales) @ B^T
    A: [M, K] in FP16 (Activations)
    B: [N, K] in INT8 (Weights)
    scales: [N] in FP16 (Weight scales)
    C: [M, N] in FP16 (Output)
    
    This kernel dequantizes the INT8 weights dynamically to FP16 before multiplying with the activations.
    """
    
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is transposed when loading, so B shape is [N, K]
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Load B (INT8)
        b_int8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        
        # Dequantize B to Float16 directly.
        # Note: B was [N, K], we loaded it as [BLOCK_K, BLOCK_N]
        b_fp16 = b_int8.to(tl.float16)
        
        # We compute along the K dimension.
        accumulator += tl.dot(a, b_fp16)
        
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # -----------------------------------------------------------
    # Epilogue
    # 1. Convert accumulator to FP16
    c = accumulator.to(tl.float16)

    # 2. Load weight scales and multiply
    # scales shape is [N]
    scales_ptrs = scales_ptr + offs_bn * stride_scale_n
    scales = tl.load(scales_ptrs, mask=offs_bn < N, other=0.0)
    
    # Apply scales: c = c * scales
    # c is [BLOCK_M, BLOCK_N], scales is [BLOCK_N]
    c = c * scales[None, :]

    # 3. Write back to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Mask C
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def w8a16_gemm_fwd(a: torch.Tensor, b_int8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Perform W8A16 Matrix Multiplication.
    
    Args:
        a: Activation tensor of shape [..., K] (FP16/BF16)
        b_int8: Weight tensor of shape [N, K] (INT8)
        scales: Weight scales of shape [N] (FP16/BF16)
        
    Returns:
        c: Output tensor of shape [..., N] (FP16/BF16)
    """
    # Flatten A to [M, K]
    a_shape = a.shape
    M = a.numel() // a_shape[-1]
    K = a_shape[-1]
    N = b_int8.shape[0]
    
    assert b_int8.shape[1] == K, "Inner dimensions must match"
    assert scales.shape[0] == N, "Scales dimension must match N"
    
    a_2d = a.view(M, K)
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    w8a16_gemm_kernel[grid](
        a_2d, b_int8, c, scales,
        M, N, K,
        a_2d.stride(0), a_2d.stride(1),
        b_int8.stride(0), b_int8.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0),
    )
    
    # Reshape back to [..., N]
    out_shape = list(a_shape[:-1]) + [N]
    return c.view(*out_shape)
