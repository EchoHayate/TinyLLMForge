"""
权重量化工具：支持 int8 / int2 分组对称量化（per-output-row, group-wise）。

存储约定：
    - weight 形状原本 [out, in]
    - 量化后保存：
        qweight: 整数张量 [out, in]，dtype = int8 (int8 量化) 或者 uint8 (int2 packed)
                 对于 int2，每 4 个权重 pack 成 1 个 uint8 字节，因此实际形状为 [out, in/4]
        scales:  [out, in/group_size]  (float32)
    - 反量化：w = qweight * scales (按组广播)

使用方式：
    Linear 层在 forward 时调用 dequantize_weight() 临时还原 fp 权重，再做矩阵乘。
    这是 weight-only 量化（W8A16 / W2A16），实现简单、对推理精度影响较小。
"""

import torch


def _percentile_along_last(x: torch.Tensor, q: float) -> torch.Tensor:
    """沿最后一维取 q 分位（0<q<1）。比 torch.quantile 快得多：用 kthvalue O(n)。

    返回 shape = x.shape[:-1] + (1,)
    """
    n = x.shape[-1]
    k = max(1, min(n, int(round(q * n))))
    # kthvalue 返回升序第 k 小的值；q 分位 = 升序第 ceil(q*n) 个
    val = torch.kthvalue(x, k, dim=-1, keepdim=True).values
    return val


def quantize_int8(weight: torch.Tensor, group_size: int = 128):
    """对称分组量化 -> int8。

    Args:
        weight: [out, in] fp tensor
        group_size: 分组大小（沿 in 维度）

    Returns:
        qweight: [out, in] int8
        scales:  [out, num_groups] float32
    """
    assert weight.dim() == 2
    out, in_dim = weight.shape
    assert in_dim % group_size == 0, f"int8 量化要求 in_dim={in_dim} 能被 group_size={group_size} 整除"
    num_groups = in_dim // group_size

    w = weight.detach().float().reshape(out, num_groups, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = max_abs / 127.0                                        # [out, num_groups, 1]
    qweight = (w / scales).round().clamp(-128, 127).to(torch.int8)
    qweight = qweight.reshape(out, in_dim)
    scales = scales.reshape(out, num_groups).to(torch.float32)
    return qweight, scales


def dequantize_int8(qweight: torch.Tensor, scales: torch.Tensor,
                    group_size: int, dtype: torch.dtype) -> torch.Tensor:
    out, in_dim = qweight.shape
    num_groups = scales.shape[1]
    g = in_dim // num_groups
    # [out, num_groups, g] * [out, num_groups, 1]
    w = qweight.to(torch.float32).reshape(out, num_groups, g) * scales.unsqueeze(-1)
    return w.reshape(out, in_dim).to(dtype)


def quantize_int4(weight: torch.Tensor, group_size: int = 128):
    """对称分组量化 -> int4，每 2 个 int4 pack 成 1 个 uint8。

    int4 取值集合 {-8, -7, ..., 6, 7}（对称 16 level，clip 到 [-7, 7] 保证 zero 落在中心）。
    存储时偏移 +8 -> [1, 15]，再每 2 个 pack 进 1 字节（low nibble 在前）。

    精度优化沿用 int2 的策略：
      1. 99.5% 分位异常值保护
      2. 1D scale 搜索最小化重建 MSE
    """
    assert weight.dim() == 2
    out, in_dim = weight.shape
    assert in_dim % group_size == 0, f"int4 量化要求 in_dim={in_dim} 能被 group_size={group_size} 整除"
    assert in_dim % 2 == 0, "int4 pack 要求 in_dim 能被 2 整除"
    num_groups = in_dim // group_size

    w = weight.detach().float().reshape(out, num_groups, group_size)

    abs_w = w.abs()
    p995 = _percentile_along_last(abs_w, 0.995).clamp_min(1e-8)
    abs_max = abs_w.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    clip_val = torch.minimum(abs_max, p995 * 1.5)

    # 对 int4，scale 让 ±7*s ≈ clip_val
    n_grid = 13
    ks = torch.linspace(0.7, 1.1, n_grid, device=w.device, dtype=w.dtype)
    best_mse = None
    best_scales = None
    for k in ks.tolist():
        s = (clip_val * k) / 7.0
        q = (w / s).round().clamp(-7, 7)
        recon = q * s
        mse = (recon - w).pow(2).mean(dim=-1, keepdim=True)
        if best_mse is None:
            best_mse = mse
            best_scales = s
        else:
            mask = mse < best_mse
            best_mse = torch.where(mask, mse, best_mse)
            best_scales = torch.where(mask, s, best_scales)
    scales = best_scales
    q = (w / scales).round().clamp(-7, 7).to(torch.int8)
    q = q.reshape(out, in_dim)

    # pack：每 2 个连续元素 -> 1 字节
    q_unsigned = (q + 8).to(torch.uint8)                            # [0, 15] 实际 [1, 15]
    q_unsigned = q_unsigned.reshape(out, in_dim // 2, 2)
    packed = (q_unsigned[..., 0]
              | (q_unsigned[..., 1] << 4)).to(torch.uint8)          # [out, in_dim/2]
    scales = scales.reshape(out, num_groups).to(torch.float32)
    return packed, scales


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor,
                    group_size: int, dtype: torch.dtype) -> torch.Tensor:
    out, packed_in = packed.shape
    in_dim = packed_in * 2
    num_groups = scales.shape[1]
    g = in_dim // num_groups

    p = packed.to(torch.int16)
    q0 = (p & 0xF)
    q1 = (p >> 4) & 0xF
    q = torch.stack([q0, q1], dim=-1).reshape(out, in_dim)
    q = q.to(torch.float32) - 8.0

    w = q.reshape(out, num_groups, g) * scales.unsqueeze(-1)
    return w.reshape(out, in_dim).to(dtype)


def fake_quantize_act_int8(x: torch.Tensor) -> torch.Tensor:
    """per-token 对称 int8 dynamic fake-quant：round + clamp + 立即 dequant 回 fp。

    用于 W4A8 naive 路径：在 GEMM 前对 activation 做"假量化"以模拟 int8 GEMM 的舍入行为。
    输入形状任意，最后一维是 hidden / channel；以最后一维为 token，沿其余维度共享 scale 不合理，
    所以这里"per-token" = 沿最后一维之外的所有维度逐位置算 scale，最后一维（channel）共享。

    具体：x [..., C] → reshape [N, C]，每行算 max(|x|)/127 当 scale。
    """
    if x.numel() == 0:
        return x
    orig_dtype = x.dtype
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    # 用 fp32 算 scale 避免 amax 在 fp16/bf16 上的精度问题
    x_fp = x2d.float()
    s = x_fp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0     # [N, 1]
    q = (x_fp / s).round().clamp(-128, 127)
    deq = (q * s).to(orig_dtype)
    return deq.reshape(orig_shape)


def quantize_int2(weight: torch.Tensor, group_size: int = 128):
    """对称分组量化 -> int2，每 4 个 int2 pack 进 1 个 uint8。

    int2 取值集合 {-2, -1, 0, 1}（非对称，4 个 level 中的 3 个负值 +1 个正值）。
    存储时偏移 +2 -> [0, 3]，再 4 个 pack 成 1 字节。

    精度优化：
      1. 异常值保护：先按 99.5 分位 clip，避免少量极端值把整个 group 的 scale 拉爆
      2. scale 通过 1D 搜索最小化重建 MSE（在合理范围内扫 N 个候选），
         比固定 max/1.5 更接近最优；只在量化时一次性扫，推理无开销
    """
    assert weight.dim() == 2
    out, in_dim = weight.shape
    assert in_dim % group_size == 0, f"int2 量化要求 in_dim={in_dim} 能被 group_size={group_size} 整除"
    assert in_dim % 4 == 0, "int2 量化要求 in_dim 能被 4 整除"
    num_groups = in_dim // group_size

    w = weight.detach().float().reshape(out, num_groups, group_size)

    # ---- 异常值保护：每组单独按 99.5% absmax 做 soft clip ----
    abs_w = w.abs()
    p995 = _percentile_along_last(abs_w, 0.995).clamp_min(1e-8)               # [out, ng, 1]
    abs_max = abs_w.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    clip_val = torch.minimum(abs_max, p995 * 1.5)                              # 保留少量大值

    # ---- 1D scale 搜索：候选 = clip_val * k, k in [0.6 .. 1.2] N=13 ----
    n_grid = 13
    ks = torch.linspace(0.6, 1.2, n_grid, device=w.device, dtype=w.dtype)      # [N]
    # 对每个 k 计算重建 mse，选最小
    best_mse = None
    best_scales = None
    for k in ks.tolist():
        s = (clip_val * k) / 1.5                                               # 让 ±1.5*s 大致覆盖大部分数据
        q = (w / s).round().clamp(-2, 1)
        recon = q * s
        mse = (recon - w).pow(2).mean(dim=-1, keepdim=True)
        if best_mse is None:
            best_mse = mse
            best_scales = s
        else:
            mask = mse < best_mse
            best_mse = torch.where(mask, mse, best_mse)
            best_scales = torch.where(mask, s, best_scales)
    scales = best_scales                                                       # [out, ng, 1]
    q = (w / scales).round().clamp(-2, 1).to(torch.int8)
    q = q.reshape(out, in_dim)

    # pack：每 4 个连续元素 -> 1 字节  (low bits 在前)
    q_unsigned = (q + 2).to(torch.uint8)                            # [0, 3]
    q_unsigned = q_unsigned.reshape(out, in_dim // 4, 4)
    packed = (q_unsigned[..., 0]
              | (q_unsigned[..., 1] << 2)
              | (q_unsigned[..., 2] << 4)
              | (q_unsigned[..., 3] << 6)).to(torch.uint8)          # [out, in_dim/4]
    scales = scales.reshape(out, num_groups).to(torch.float32)
    return packed, scales


def dequantize_int2(packed: torch.Tensor, scales: torch.Tensor,
                    group_size: int, dtype: torch.dtype) -> torch.Tensor:
    out, packed_in = packed.shape
    in_dim = packed_in * 4
    num_groups = scales.shape[1]
    g = in_dim // num_groups

    p = packed.to(torch.int16)                                       # 避免移位时溢出
    q0 = (p & 0x3)
    q1 = (p >> 2) & 0x3
    q2 = (p >> 4) & 0x3
    q3 = (p >> 6) & 0x3
    # 重新组合 [out, packed_in, 4] -> [out, in_dim]
    q = torch.stack([q0, q1, q2, q3], dim=-1).reshape(out, in_dim)
    q = q.to(torch.float32) - 2.0                                    # 还原符号

    w = q.reshape(out, num_groups, g) * scales.unsqueeze(-1)
    return w.reshape(out, in_dim).to(dtype)


def quantize_int8_per_row(weight: torch.Tensor):
    """per-output-row 对称量化（用于 bnb fused W8A16 GEMM）。

    Returns:
        qweight: [out, in]  int8
        scales:  [out]      float32
    """
    assert weight.dim() == 2
    w = weight.detach().float()
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)    # [out, 1]
    scales = (max_abs / 127.0).reshape(-1).to(torch.float32)         # [out]
    qweight = (w / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
    return qweight, scales


def dequantize_int8_per_row(qweight: torch.Tensor, scales: torch.Tensor,
                            dtype: torch.dtype) -> torch.Tensor:
    return (qweight.to(torch.float32) * scales.unsqueeze(-1)).to(dtype)


def quantize_weight(weight: torch.Tensor, method: str, group_size: int):
    if method == "int8":
        return quantize_int8(weight, group_size)
    if method == "int8_bnb":
        return quantize_int8_per_row(weight)
    if method == "int4":
        return quantize_int4(weight, group_size)
    if method == "int2":
        return quantize_int2(weight, group_size)
    raise ValueError(f"unsupported quantization method: {method}")


def dequantize_weight(qweight: torch.Tensor, scales: torch.Tensor,
                      method: str, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    if method == "int8":
        return dequantize_int8(qweight, scales, group_size, dtype)
    if method == "int8_bnb":
        return dequantize_int8_per_row(qweight, scales, dtype)
    if method == "int4":
        return dequantize_int4(qweight, scales, group_size, dtype)
    if method == "int2":
        return dequantize_int2(qweight, scales, group_size, dtype)
    raise ValueError(f"unsupported quantization method: {method}")
