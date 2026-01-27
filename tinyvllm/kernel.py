from torch import sigmoid
from torch.sparse import DimOrDims


@triton.jit
def swiglu_kernel(
    x_ptr, output_ptr,
    batch_stride, seq_stride, dim_stride,
    batch_size, seq_len, dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    SWiGLU Triton Kernel
    - 输入：x (batch_size, seq_len, dim)，dim必须是偶数
    - 输出：output (batch_size, seq_len, dim//2)
    """
    # 1. 计算当前thread处理的索引
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    dim_offs = tl.arange(0, BLOCK_SIZE)
    
    # 2. 加载x的切片（拆分为x1和x2）
    x_ptr = x_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs
    x = tl.load(x_ptr, mask=dim_offs < dim, other=0.0)

    half_dim = dim //2 
    x1 = tl.where(dim_offs <half_dim,x,0.0)
    x2 = tl.where(dim_offs >=half_dim,x,0.0)
    x2 = x2[half_dim:]

    x2 =tl.reshape(x2[:half_dim],(BLOCK_SIZE//2,))

    sigmoid_x2 = tl.sigmoid(x2)
    output = x1[:half_dim] *sigmoid_x2

    output_ptr = output_ptr + batch_idx*batch_stride+seq_idx*seq_stride+dim_offs[:half_dim]
    tl.load(output_ptr,output,mask = dim_offs[:half_dim]<half_dim)


@triton.jit
def skip_conn_kernel(
    x_ptr, residual_ptr, output_ptr,
    batch_stride, seq_stride, dim_stride,
    batch_size, seq_len, dim,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    dim_offs = tl.arange(0, BLOCK_SIZE)

    x_ptr = x_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs
    residual_ptr = residual_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs

    x = tl.load(x_ptr, mask=dim_offs < dim, other=0.0)
    residual = tl.load(residual_ptr, mask=dim_offs < dim, other=0.0)

    output = x + residual

    output_ptr = output_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs
    tl.store(output_ptr, output, mask=dim_offs < dim)





@triton.jit
def fused_rmsnorm_swiglu_skipconn_kernel(
    x_ptr, gamma_ptr, residual_ptr, output_ptr,
    batch_stride, seq_stride, dim_stride,
    batch_size, seq_len, dim,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    dim_offs = tl.arange(0, BLOCK_SIZE)

    x_ptr = x_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs
    x = tl.load(x_ptr, mask=dim_offs < dim, other=0.0)

    x_sq = x * x
    mean_x_sq = tl.sum(x_sq, axis=0) / dim
    rms = tl.sqrt(mean_x_sq + eps)
    gamma = tl.load(gamma_ptr + dim_offs, mask=dim_offs < dim, other=1.0)
    x_norm = (x / rms) * gamma
    
    # 3. SWiGLU计算（拆分x_norm为x1和x2）
    half_dim = dim // 2
    x1 = tl.where(dim_offs < half_dim, x_norm, 0.0)
    x2 = tl.where(dim_offs >= half_dim, x_norm, 0.0)
    x2 = x2[:half_dim]  # 对齐到half_dim
    swiglu_out = x1[:half_dim] * tl.sigmoid(x2)
    
    # 4. Skip Connection（加载residual并相加）
    residual_ptr = residual_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs[:half_dim]
    residual = tl.load(residual_ptr, mask=dim_offs[:half_dim] < half_dim, other=0.0)
    output = swiglu_out + residual
    
    # 5. 写入输出
    output_ptr = output_ptr + batch_idx * batch_stride + seq_idx * seq_stride + dim_offs[:half_dim]
    tl.store(output_ptr, output, mask=dim_offs[:half_dim] < half_dim)