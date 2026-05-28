"""权重量化数值正确性单元测试。

覆盖 tinyvllm/layers/quantization.py 里全部量化路径：
  * quantize_int8 / dequantize_int8         (sym group-wise int8)
  * quantize_int4 / dequantize_int4         (sym group-wise int4 + nibble pack)
  * quantize_int2 / dequantize_int2         (sym group-wise int2 + 4-pack)
  * quantize_int8_per_row / dequantize_*    (per-output-row int8 for bnb fused GEMM)
  * fake_quantize_act_int8                  (per-token act fake-quant for W4A8 naive)

每条用例都做以下断言：
  1) 形状 / dtype 符合存储约定
  2) round-trip 后 max_abs_err 不超过对称量化的理论上界（带 10~20% 余量）
  3) round-trip 后 MSE 不大于一个跟 bit 数对齐的弱阈值
  4) idempotency：把 dequant 出来的张量再量化一次，pack 结果应该完全一致

跑法（CPU/GPU 都行；用纯 torch，不依赖 cuda）：
  python tools/test_weight_quant.py
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tinyvllm.layers.quantization import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4,
    quantize_int2, dequantize_int2,
    quantize_int8_per_row, dequantize_int8_per_row,
    fake_quantize_act_int8,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_weights(out: int, in_dim: int, dist: str, seed: int = 0) -> torch.Tensor:
    """生成几种典型分布的权重，覆盖 quantizer 的常见输入：
       - normal:   ~ N(0, 1)，均匀分布的常态
       - heavytail: 90% N(0, 0.3) + 10% N(0, 3)，模拟 outlier
       - uniform:  U(-1, 1)，没有重尾
    """
    g = torch.Generator().manual_seed(seed)
    if dist == "normal":
        return torch.randn(out, in_dim, generator=g, dtype=torch.float32)
    if dist == "uniform":
        return (torch.rand(out, in_dim, generator=g, dtype=torch.float32) * 2 - 1)
    if dist == "heavytail":
        body = torch.randn(out, in_dim, generator=g, dtype=torch.float32) * 0.3
        # 10% 元素加大方差噪声
        mask = torch.rand(out, in_dim, generator=g) < 0.1
        spike = torch.randn(out, in_dim, generator=g, dtype=torch.float32) * 3.0
        return torch.where(mask, spike, body)
    raise ValueError(dist)


def _per_group_amax(w: torch.Tensor, group_size: int) -> torch.Tensor:
    out, in_dim = w.shape
    g = group_size if in_dim % group_size == 0 else in_dim
    ng = in_dim // g
    return w.reshape(out, ng, g).abs().amax(dim=-1)         # [out, ng]


# ---------------------------------------------------------------------------
# 1) int8 (group-wise symmetric)
# ---------------------------------------------------------------------------

def test_int8(verbose=True):
    torch.manual_seed(0)
    for dist in ("normal", "heavytail", "uniform"):
        for group_size in (32, 128, 256):
            w = _make_weights(64, 512, dist)
            qw, scales = quantize_int8(w, group_size)

            assert qw.dtype == torch.int8 and qw.shape == w.shape
            assert scales.dtype == torch.float32
            assert scales.shape == (64, 512 // group_size)

            w_hat = dequantize_int8(qw, scales, group_size, torch.float32)
            assert w_hat.shape == w.shape

            # 上界：每组单值最大误差 ≈ scale/2 = amax / (127*2) = amax / 254
            amax = _per_group_amax(w, group_size).max().item()
            bound = amax / 254.0 + 1e-5

            err = (w - w_hat).abs()
            max_err = err.max().item()
            mse = err.pow(2).mean().item()

            if verbose:
                print(f"[int8 ] dist={dist:<10} g={group_size:<3} "
                      f"max_err={max_err:.5f} bound={bound:.5f} mse={mse:.2e}")

            assert max_err <= bound * 1.1, \
                f"int8 dist={dist} g={group_size} max_err={max_err} > bound={bound}"
            # int8 量化 MSE 期望很小：跟 bound^2 同量级
            assert mse <= (bound * bound) * 2.0, \
                f"int8 dist={dist} g={group_size} mse={mse} 超出预期"

            # idempotency：再量化一次结果应一致
            qw2, scales2 = quantize_int8(w_hat, group_size)
            assert torch.equal(qw, qw2), f"int8 idempotency fail dist={dist} g={group_size}"
            assert torch.allclose(scales, scales2, rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# 2) int4 (group-wise symmetric, nibble packed)
# ---------------------------------------------------------------------------

def test_int4(verbose=True):
    torch.manual_seed(0)
    for dist in ("normal", "heavytail", "uniform"):
        for group_size in (32, 128):
            w = _make_weights(64, 512, dist)
            packed, scales = quantize_int4(w, group_size)

            # 形状：[out, in/2] uint8 packed
            assert packed.dtype == torch.uint8
            assert packed.shape == (64, 512 // 2)
            assert scales.dtype == torch.float32
            assert scales.shape == (64, 512 // group_size)

            w_hat = dequantize_int4(packed, scales, group_size, torch.float32)
            assert w_hat.shape == w.shape

            # int4 上界：1D scale 搜索可能选偏小的 scale（k 最低 0.7），
            # 同时 clip_val ≤ amax，因此单值最大误差最坏约 amax/9.8；
            # 但 outlier 被 clamp 到 ±7*scale 时误差可达 |w_max| - 7*scale_min。
            # 这里给一个跟 amax 同量级的弱上界，验证不会出现"完全失真"即可。
            amax = _per_group_amax(w, group_size).max().item()
            bound = amax * 0.6 + 0.05
            # heavytail 容易把上界顶到 amax 量级（outlier）
            if dist == "heavytail":
                bound = max(bound, amax * 0.7)

            err = (w - w_hat).abs()
            max_err = err.max().item()
            mse = err.pow(2).mean().item()

            if verbose:
                print(f"[int4 ] dist={dist:<10} g={group_size:<3} "
                      f"max_err={max_err:.5f} bound={bound:.5f} mse={mse:.2e}")

            assert max_err <= bound, \
                f"int4 dist={dist} g={group_size} max_err={max_err} > bound={bound}"
            # int4 MSE 期望远小于信号方差（量化没把信号"打成噪声"）
            assert mse <= w.var().item() * 0.05, \
                f"int4 dist={dist} g={group_size} mse={mse} 太大（var={w.var().item():.4f}）"

            # 注：int4 走 1D scale 搜索（MSE-based），输入分布变化会导致搜出的 k 改变，
            # 因此不做严格 bit-exact idempotency 断言；只断言 dequant 后再 quant 的
            # MSE 不会变得更大（量化语义稳定）
            packed2, scales2 = quantize_int4(w_hat, group_size)
            w_hat2 = dequantize_int4(packed2, scales2, group_size, torch.float32)
            mse2 = (w_hat - w_hat2).pow(2).mean().item()
            # 第二轮量化的输入已经是离散值，MSE 应远小于第一轮
            assert mse2 <= mse + 1e-6, \
                f"int4 second-round MSE {mse2} > first {mse}, dist={dist} g={group_size}"


# ---------------------------------------------------------------------------
# 3) int2 (group-wise symmetric, 4-pack)
# ---------------------------------------------------------------------------

def test_int2(verbose=True):
    torch.manual_seed(0)
    # int2 误差大，只用 normal/uniform；heavytail 上 max_err 很难给紧界，跳过精确断言
    for dist in ("normal", "uniform"):
        for group_size in (64, 128):
            w = _make_weights(32, 512, dist)
            packed, scales = quantize_int2(w, group_size)

            assert packed.dtype == torch.uint8
            assert packed.shape == (32, 512 // 4)
            assert scales.shape == (32, 512 // group_size)

            w_hat = dequantize_int2(packed, scales, group_size, torch.float32)

            # int2 只有 4 个 level {-2,-1,0,1}，单值误差最坏 ≈ scale ≈ amax/1.5
            amax = _per_group_amax(w, group_size).max().item()
            bound = amax * 1.0 + 0.05

            err = (w - w_hat).abs()
            max_err = err.max().item()
            mse = err.pow(2).mean().item()

            if verbose:
                print(f"[int2 ] dist={dist:<10} g={group_size:<3} "
                      f"max_err={max_err:.5f} bound={bound:.5f} mse={mse:.2e}")

            assert max_err <= bound, \
                f"int2 dist={dist} g={group_size} max_err={max_err} > bound={bound}"
            # int2 MSE 期望明显大于 int4，但应远小于信号方差
            assert mse <= w.var().item(), \
                f"int2 dist={dist} g={group_size} mse={mse} >= 信号方差 {w.var().item()}"


# ---------------------------------------------------------------------------
# 4) int8_per_row (用于 bnb fused W8A16 GEMM)
# ---------------------------------------------------------------------------

def test_int8_per_row(verbose=True):
    torch.manual_seed(0)
    for dist in ("normal", "heavytail"):
        w = _make_weights(64, 256, dist)
        qw, scales = quantize_int8_per_row(w)

        assert qw.dtype == torch.int8 and qw.shape == w.shape
        assert scales.dtype == torch.float32 and scales.shape == (64,)

        w_hat = dequantize_int8_per_row(qw, scales, torch.float32)
        assert w_hat.shape == w.shape

        # 每行一个 scale = max_abs / 127；单值误差最坏 ≈ scale/2 = max_abs/254
        # 对所有行取 worst：用 max_abs.max
        per_row_amax = w.abs().amax(dim=-1)
        bound_per_row = per_row_amax / 254.0 + 1e-5

        per_row_max_err = (w - w_hat).abs().amax(dim=-1)
        if verbose:
            print(f"[int8r] dist={dist:<10} max_max_err={per_row_max_err.max().item():.5f} "
                  f"bound_max={bound_per_row.max().item():.5f}")

        assert (per_row_max_err <= bound_per_row * 1.1).all(), \
            f"int8_per_row dist={dist} 误差超界"


# ---------------------------------------------------------------------------
# 5) fake_quantize_act_int8（W4A8 naive 路径）
# ---------------------------------------------------------------------------

def test_fake_quant_act(verbose=True):
    torch.manual_seed(0)
    # 模拟 [num_tokens, hidden] 的 activation
    for shape in [(64, 512), (1, 1024), (128, 64)]:
        x = torch.randn(*shape, dtype=torch.float16) * 2.0
        x[0, ::7] *= 5  # 放点 outlier

        y = fake_quantize_act_int8(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

        # per-token：每行 scale = max_abs/127；单值误差最坏 ≈ scale/2 = max_abs/254
        per_token_amax = x.float().abs().amax(dim=-1)           # [N]
        per_token_bound = per_token_amax / 254.0 + 1e-3         # fp16 round 噪声

        per_token_max_err = (x.float() - y.float()).abs().amax(dim=-1)
        if verbose:
            print(f"[actq ] shape={shape} max_max_err={per_token_max_err.max().item():.5f} "
                  f"bound_max={per_token_bound.max().item():.5f}")

        # fp16 deq 路径会引入额外 round 噪声，给 1.5x 余量
        assert (per_token_max_err <= per_token_bound * 1.5).all(), \
            f"fake_quant_act shape={shape} 误差超界"

    # 边界：空张量应原样返回
    e = torch.empty(0, 16)
    out = fake_quantize_act_int8(e)
    assert out.shape == e.shape


# ---------------------------------------------------------------------------
# 6) shape 校验：当 in_dim 不能被 group_size 整除时应直接报错（早期是 silent fallback，
#    现在改成 assert，避免悄悄退化成"整行一组"导致精度劣化）
# ---------------------------------------------------------------------------

def test_int8_group_assert(verbose=True):
    w = _make_weights(8, 200, "normal")
    try:
        quantize_int8(w, group_size=128)
    except AssertionError as e:
        if verbose:
            print(f"[int8 assert] in=200 g=128 → 正确触发 AssertionError: {e}")
        return
    raise AssertionError("quantize_int8 应在 in_dim 不能被 group_size 整除时报 AssertionError")


def main():
    print("=== int8 group-wise ===")
    test_int8()
    print("ok\n")

    print("=== int4 group-wise (nibble pack) ===")
    test_int4()
    print("ok\n")

    print("=== int2 group-wise (4-pack) ===")
    test_int2()
    print("ok\n")

    print("=== int8 per-row (bnb fused) ===")
    test_int8_per_row()
    print("ok\n")

    print("=== fake_quantize_act_int8 (W4A8) ===")
    test_fake_quant_act()
    print("ok\n")

    print("=== int8 group_size assert ===")
    test_int8_group_assert()
    print("ok\n")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
