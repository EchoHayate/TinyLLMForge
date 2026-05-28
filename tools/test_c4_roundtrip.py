"""C4 / KV-cache int4 量化 round-trip 数值正确性测试。

两个层级：
  1) 纯 numpy 模拟：验证 pack/unpack/scale 的语义本身没问题（无需 GPU）
  2) GPU 端到端：调实际 triton kernel + dequant_kv_blocks，验证两路实现一致

A-3 评测前的护栏：如果 round-trip MSE 异常大，就不用浪费时间跑 needle 了。
"""

import os
import sys
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# 1) 纯 numpy 参考实现（与 triton kernel 语义对齐）
# ---------------------------------------------------------------------------

def quant_pack_ref(x: np.ndarray, group_size: int):
    """对称 int4 量化 + nibble pack（参考实现）。

    x: shape [..., head_dim] (head_dim % group_size == 0, group_size 偶数)
    返回:
      packed: int8 [..., head_dim//2]，低 4 位 = 偶数索引，高 4 位 = 奇数索引
      scale:  fp32 [..., head_dim//group_size]
    """
    assert x.shape[-1] % group_size == 0
    assert group_size % 2 == 0
    head_dim = x.shape[-1]
    num_groups = head_dim // group_size

    grouped = x.reshape(*x.shape[:-1], num_groups, group_size)
    amax = np.max(np.abs(grouped), axis=-1)             # [..., num_groups]
    scale = amax / 7.0 + 1e-8

    q = np.rint(grouped / scale[..., None]).astype(np.int32)
    q = np.clip(q, -8, 7)                               # [-8, 7]

    q_flat = q.reshape(*x.shape[:-1], head_dim)         # [..., head_dim]
    even = q_flat[..., 0::2] & 0xF                      # 低 4 位
    odd = q_flat[..., 1::2] & 0xF                       # 高 4 位
    packed = (even | (odd << 4)).astype(np.int8)
    return packed, scale.astype(np.float32)


def dequant_unpack_ref(packed: np.ndarray, scale: np.ndarray, group_size: int) -> np.ndarray:
    """nibble unpack + 反量化（参考实现，对应 attention.py::dequant_kv_blocks）。"""
    half = packed.shape[-1]
    head_dim = half * 2
    p32 = packed.astype(np.int32)
    # int4 sign-extend：取低/高 nibble 后映射回 [-8, 7]
    low = ((p32 & 0xF) ^ 0x8) - 0x8
    high = (((p32 >> 4) & 0xF) ^ 0x8) - 0x8
    nibble = np.empty(packed.shape[:-1] + (head_dim,), dtype=np.int32)
    nibble[..., 0::2] = low
    nibble[..., 1::2] = high

    # scale 沿 head_dim 复制 group_size 次
    num_groups = scale.shape[-1]
    assert num_groups * group_size == head_dim
    scale_exp = np.repeat(scale, group_size, axis=-1)
    return nibble.astype(np.float32) * scale_exp


def test_numpy_roundtrip(verbose=True):
    rng = np.random.default_rng(0)
    # 模拟 [N=128 tokens, num_kv_heads=8, head_dim=128]
    x = rng.standard_normal((128, 8, 128)).astype(np.float32)
    # 加点 outlier，模拟 attention 真实数据
    x[5, 2, :8] *= 20

    for group_size in (32, 64, 128):
        packed, scale = quant_pack_ref(x, group_size)
        x_hat = dequant_unpack_ref(packed, scale, group_size)

        # 误差量级：scale = max/7，单值最大误差 = scale/2 ≈ amax/14
        amax_per_group = np.max(np.abs(x.reshape(*x.shape[:-1], -1, group_size)), axis=-1)
        max_allowed = (amax_per_group / 14.0 + 1e-6).max()

        abs_err = np.abs(x - x_hat)
        max_err = abs_err.max()
        mse = (abs_err ** 2).mean()

        if verbose:
            print(f"[numpy] group_size={group_size:>3}  "
                  f"max_err={max_err:.4f}  bound={max_allowed:.4f}  mse={mse:.4e}  "
                  f"x_amax={np.abs(x).max():.2f}")

        # bound 留 10% 余量给浮点 round 边界
        assert max_err <= max_allowed * 1.1 + 1e-5, \
            f"group_size={group_size} 误差超出对称量化理论上界"

        # 反量化值落在 [-amax, amax] 内
        assert (np.abs(x_hat) <= np.abs(x).max() + 1e-3).all()


# ---------------------------------------------------------------------------
# 2) GPU 端到端测试（需要 torch + cuda + triton）
# ---------------------------------------------------------------------------

def test_gpu_kernel(verbose=True):
    import torch
    if not torch.cuda.is_available():
        print("[gpu] cuda not available, 跳过 kernel 测试")
        return

    from tinyvllm.layers.attention import store_kvcache_q4, dequant_kv_blocks

    torch.manual_seed(0)
    N, num_kv_heads, head_dim = 64, 8, 128
    block_size = 256
    num_blocks = 4
    group_size = 128

    # 初始化 packed cache 与 scale buffer
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim // 2,
                          dtype=torch.int8, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    num_groups = head_dim // group_size
    k_scale = torch.zeros(num_blocks, block_size, num_kv_heads, num_groups,
                          dtype=torch.float16, device="cuda")
    v_scale = torch.zeros_like(k_scale)

    # 构造 N 个 token 的 K/V 与 slot_mapping
    k = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    # outlier
    k[3, 1, :8] *= 15.0
    # slot_mapping：把 N=64 个 token 写到 block 0 的前 64 个 slot
    slot_mapping = torch.arange(N, dtype=torch.int32, device="cuda")

    store_kvcache_q4(k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping, group_size)

    # 用 dequant_kv_blocks 把 block 0 整块拉回来
    bt = torch.tensor([[0]], dtype=torch.int32, device="cuda")  # B=1, max_blocks=1
    k_fp, _ = dequant_kv_blocks(k_cache, k_scale, bt, group_size, torch.float16)
    v_fp, _ = dequant_kv_blocks(v_cache, v_scale, bt, group_size, torch.float16)
    # k_fp shape: [1*1, block_size, num_kv_heads, head_dim]
    k_fp = k_fp[0, :N]   # 取实际写入的 N 个 token
    v_fp = v_fp[0, :N]

    abs_err_k = (k.to(torch.float32) - k_fp.to(torch.float32)).abs()
    abs_err_v = (v.to(torch.float32) - v_fp.to(torch.float32)).abs()

    # 上界（按 group_size=head_dim 算 per-(token, head)）
    amax_k = k.to(torch.float32).abs().reshape(N, num_kv_heads, num_groups, group_size).amax(-1)
    bound_k = amax_k.max().item() / 14.0 + 1e-3
    amax_v = v.to(torch.float32).abs().reshape(N, num_kv_heads, num_groups, group_size).amax(-1)
    bound_v = amax_v.max().item() / 14.0 + 1e-3

    if verbose:
        print(f"[gpu] N={N} group_size={group_size}")
        print(f"      K max_err={abs_err_k.max().item():.4f}  bound={bound_k:.4f}  "
              f"mse={abs_err_k.pow(2).mean().item():.4e}")
        print(f"      V max_err={abs_err_v.max().item():.4f}  bound={bound_v:.4f}  "
              f"mse={abs_err_v.pow(2).mean().item():.4e}")

    assert abs_err_k.max().item() <= bound_k * 1.2, "K round-trip 超过对称量化上界"
    assert abs_err_v.max().item() <= bound_v * 1.2, "V round-trip 超过对称量化上界"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true", help="跑 GPU kernel 端到端测试")
    p.add_argument("-v", "--verbose", action="store_true", default=True)
    args = p.parse_args()

    print("=== numpy round-trip ===")
    test_numpy_roundtrip(verbose=args.verbose)
    print("ok")

    if args.gpu:
        print("\n=== GPU kernel round-trip ===")
        test_gpu_kernel(verbose=args.verbose)
        print("ok")


if __name__ == "__main__":
    main()
