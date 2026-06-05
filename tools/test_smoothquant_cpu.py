"""SmoothQuant 注入逻辑 CPU 单测（无需 GPU）。

只测 `_apply_smoothquant_scales` 本身的正确性 + 错误路径，不涉及真实模型权重 / 量化。

覆盖点：
  1) ColumnParallelLinear（input 维全量）：W' == W*s，forward(x) ≈ (x/s) @ W^T （即与原 W·x 等价）
  2) RowParallelLinear（input 维分片）：单卡场景下 narrow(0,0,in_local) == 全量，等价性同 #1
  3) NaN scale → ValueError
  4) shape mismatch → ValueError
  5) 没有任何 key 命中 → ValueError
  6) buffer 已注入：finalize_quantization 把 self.weight 设 None 后 smooth_scale 仍在

跑法：python3 tools/test_smoothquant_cpu.py
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist
from torch import nn


def _init_dist_single():
    """单进程 gloo 初始化，让 LinearBase 的 dist.get_rank/world_size 可用。"""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


_init_dist_single()

# 绕过 tinyvllm/__init__.py 链式 import（会拉 triton / Qwen3 等仅 GPU 依赖）。
# 用 importlib 直接加载我们要测的两个模块文件。
import importlib.util


def _load_module(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1) tinyvllm.layers.quantization（被 linear.py 引用）。先建好父级占位。
import types
for pkg in ("tinyvllm", "tinyvllm.layers", "tinyvllm.utils"):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_REPO_ROOT, *pkg.split(".")[1:])]
        sys.modules[pkg] = m

_load_module("tinyvllm.layers.quantization",
             os.path.join(_REPO_ROOT, "tinyvllm/layers/quantization.py"))
linear_mod = _load_module("tinyvllm.layers.linear",
                          os.path.join(_REPO_ROOT, "tinyvllm/layers/linear.py"))
loader_mod = _load_module("tinyvllm.utils.loader",
                          os.path.join(_REPO_ROOT, "tinyvllm/utils/loader.py"))

ColumnParallelLinear = linear_mod.ColumnParallelLinear
RowParallelLinear = linear_mod.RowParallelLinear
set_quant_config = linear_mod.set_quant_config
_apply_smoothquant_scales = loader_mod._apply_smoothquant_scales


# ---------------------------------------------------------------------------
# 工具：构造一个 mini 模型（顶层包一个 nn.Module，named_modules 才能给出层名）
# ---------------------------------------------------------------------------

class _Mini(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.col = ColumnParallelLinear(in_dim, out_dim, bias=False)
        self.row = RowParallelLinear(in_dim, out_dim, bias=False)


def _ok(msg):
    print(f"  [ok] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 测试 1：数值等价 —— y = x · W^T == (x/s) · (W*s)^T
# ---------------------------------------------------------------------------

def test_numerical_equivalence():
    print("[test_numerical_equivalence]", flush=True)
    torch.manual_seed(0)
    in_dim, out_dim = 8, 4
    m = _Mini(in_dim, out_dim)
    # 随机化 weight；默认 nn.Parameter(torch.empty(...)) 是未初始化内存
    with torch.no_grad():
        m.col.weight.data.normal_(0, 0.1)
        m.row.weight.data.normal_(0, 0.1)
    W_col_orig = m.col.weight.data.clone()
    W_row_orig = m.row.weight.data.clone()

    # 构造 per-input-channel scale
    s = torch.linspace(0.5, 2.0, in_dim, dtype=torch.float32)
    scales = {"col": s.clone(), "row": s.clone()}

    x = torch.randn(3, in_dim)
    y_ref_col = x @ W_col_orig.t()
    y_ref_row = x @ W_row_orig.t()  # tp=1 时 row 的 weight 也是 [out, in]

    _apply_smoothquant_scales(m, scales)

    # W' == W * s
    assert torch.allclose(m.col.weight.data, W_col_orig * s.view(1, -1), atol=1e-6)
    assert torch.allclose(m.row.weight.data, W_row_orig * s.view(1, -1), atol=1e-6)
    _ok("W' == W * s for both Column and Row")

    # buffer 注册
    assert hasattr(m.col, "smooth_scale") and m.col.smooth_scale.shape == (in_dim,)
    assert hasattr(m.row, "smooth_scale") and m.row.smooth_scale.shape == (in_dim,)
    _ok("smooth_scale buffer registered with correct shape")

    # forward 等价（tp=1，row 的 all_reduce 是 no-op）
    y_col = m.col(x)
    y_row = m.row(x)
    assert torch.allclose(y_col, y_ref_col, atol=1e-5), f"col diff {(y_col - y_ref_col).abs().max()}"
    assert torch.allclose(y_row, y_ref_row, atol=1e-5), f"row diff {(y_row - y_ref_row).abs().max()}"
    _ok("forward(x) numerically equivalent to original y = x @ W^T")


# ---------------------------------------------------------------------------
# 测试 2：错误路径 —— NaN / 形状错配 / 全部 miss
# ---------------------------------------------------------------------------

def test_nan_scale_rejected():
    print("[test_nan_scale_rejected]", flush=True)
    m = _Mini(4, 2)
    bad = torch.tensor([1.0, float("nan"), 1.0, 1.0])
    try:
        _apply_smoothquant_scales(m, {"col": bad})
    except ValueError as e:
        assert "NaN/Inf" in str(e), str(e)
        _ok(f"NaN scale rejected: {e}")
        return
    raise AssertionError("expected ValueError for NaN scale")


def test_shape_mismatch_rejected():
    print("[test_shape_mismatch_rejected]", flush=True)
    m = _Mini(4, 2)
    wrong = torch.ones(7)  # in_dim=4 但传 7
    try:
        _apply_smoothquant_scales(m, {"col": wrong})
    except ValueError as e:
        assert "scale len" in str(e), str(e)
        _ok(f"shape mismatch rejected: {e}")
        return
    raise AssertionError("expected ValueError for shape mismatch")


def test_no_match_rejected():
    print("[test_no_match_rejected]", flush=True)
    m = _Mini(4, 2)
    try:
        _apply_smoothquant_scales(m, {"nonexistent.module": torch.ones(4)})
    except ValueError as e:
        assert "no scales matched" in str(e), str(e)
        _ok(f"all-miss rejected: {e}")
        return
    raise AssertionError("expected ValueError when no key matches")


# ---------------------------------------------------------------------------
# 测试 3：smooth_scale 不会被 finalize_quantization 清掉
# ---------------------------------------------------------------------------

def test_buffer_survives_finalize():
    print("[test_buffer_survives_finalize]", flush=True)
    set_quant_config("int4", group_size=4, act_bits=0)
    try:
        m = _Mini(8, 4)
        with torch.no_grad():
            m.col.weight.data.normal_(0, 0.1)
            m.row.weight.data.normal_(0, 0.1)
        s = torch.linspace(0.5, 2.0, 8, dtype=torch.float32)
        _apply_smoothquant_scales(m, {"col": s.clone(), "row": s.clone()})
        assert m.col.smooth_scale is not None
        # 触发 finalize：会把 self.weight 设 None
        m.col.finalize_quantization()
        m.row.finalize_quantization()
        assert m.col.weight is None and m.row.weight is None
        # 关键断言：buffer 仍在
        assert hasattr(m.col, "smooth_scale") and m.col.smooth_scale is not None
        assert hasattr(m.row, "smooth_scale") and m.row.smooth_scale is not None
        _ok("smooth_scale survives finalize_quantization (weight set to None)")
    finally:
        set_quant_config(None)


# ---------------------------------------------------------------------------
# 测试 4：partial 命中（只有部分模块有 scale）—— 应该成功 + 报告 skipped 数
# ---------------------------------------------------------------------------

def test_partial_match_ok():
    print("[test_partial_match_ok]", flush=True)
    m = _Mini(4, 2)
    s = torch.ones(4)
    # 只给 col 提供 scale；row 跳过
    _apply_smoothquant_scales(m, {"col": s})
    assert hasattr(m.col, "smooth_scale")
    assert not hasattr(m.row, "smooth_scale"), "row 不该被注入"
    _ok("partial match: only matched modules get smooth_scale")


# ---------------------------------------------------------------------------
# 测试 5：核心数学动机 —— SQ 应当显著降低「带 outlier 激活 + A8 fake quant」的 MSE
# ---------------------------------------------------------------------------

def test_sq_suppresses_outlier_a8_error():
    """
    构造场景：x 在少数通道上有 ~50× 的 outlier；W 是平稳分布。
    走 act_quant_bits=8 的 fake-quant 路径，比较：
      (a) 不开 SQ：per-token p999 clip 会把 outlier 通道压住，但其他通道的 scale 仍偏大 → 误差大
      (b) 开 SQ：s 把 outlier 迁到 weight，x/s 后激活分布平坦 → A8 量化误差显著降低
    断言：MSE_SQ < MSE_NoSQ * 0.5（保守阈值，防止环境抖动 flaky）
    """
    print("[test_sq_suppresses_outlier_a8_error]", flush=True)
    set_quant_config(None, group_size=128, act_bits=8)  # 仅启用 A8 fake quant，不做 W 量化
    try:
        torch.manual_seed(42)
        in_dim, out_dim = 64, 32
        n_tokens = 128

        # —— 激活：少数通道 outlier，幅值约 50×
        x = torch.randn(n_tokens, in_dim) * 0.1
        outlier_channels = [3, 17, 41, 55]  # ~6% 通道当 outlier
        for c in outlier_channels:
            x[:, c] = x[:, c] * 50.0

        # —— 参考输出（fp32，不量化）
        m_ref = ColumnParallelLinear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            m_ref.weight.data.normal_(0, 0.05)
        W_orig = m_ref.weight.data.clone()
        # 强行禁用 A8（_QuantMixin._maybe_init_quant 已设 act_quant_bits=8，临时手动改）
        m_ref.act_quant_bits = 0
        y_ref = m_ref(x)

        # —— 路径 a：A8 但不开 SQ
        m_a = ColumnParallelLinear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            m_a.weight.data.copy_(W_orig)
        assert m_a.act_quant_bits == 8
        y_a = m_a(x)
        mse_a = (y_a - y_ref).pow(2).mean().item()

        # —— 路径 b：A8 + SQ；用经典 alpha=0.5 的 s 公式
        a_max = x.abs().amax(dim=0).clamp_min(1e-5)               # [in_dim]
        w_max = W_orig.abs().amax(dim=0).clamp_min(1e-5)          # [in_dim]
        alpha = 0.5
        s = (a_max.pow(alpha) / w_max.pow(1 - alpha)).clamp_(1e-3, 1e3)

        m_b = ColumnParallelLinear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            m_b.weight.data.copy_(W_orig)
        assert m_b.act_quant_bits == 8
        # 包到容器里，让 named_modules 给出 "col" key（_apply_smoothquant_scales 按 name 匹配）
        container = nn.Module()
        container.col = m_b
        _apply_smoothquant_scales(container, {"col": s})
        y_b = m_b(x)
        mse_b = (y_b - y_ref).pow(2).mean().item()

        print(f"    MSE no-SQ  : {mse_a:.6e}")
        print(f"    MSE with-SQ: {mse_b:.6e}")
        print(f"    ratio b/a  : {mse_b / max(mse_a, 1e-12):.3f}")
        assert mse_b < mse_a * 0.5, (
            f"SQ should reduce A8 outlier error by >2x; "
            f"got mse_a={mse_a:.4e}, mse_b={mse_b:.4e}"
        )
        _ok(f"SQ reduces A8 MSE under outlier activation (ratio={mse_b/mse_a:.3f} < 0.5)")
    finally:
        set_quant_config(None, group_size=128, act_bits=0)


# ---------------------------------------------------------------------------
# 测试 6：calibrate_smoothquant.py 的 hook → state → s 聚合 dry-run
# 不依赖真实 LLM；用真实的 LinearBase 子类（ColumnParallel）+ 复刻一份脚本里的
# hook/聚合代码，验证整条管线在 CPU 上行为正确。
# ---------------------------------------------------------------------------

def test_calibrate_hook_pipeline_dryrun():
    """
    覆盖点：
      1) forward_pre_hook 在 forward(x) 之前拿到原始 x（在 SQ 注入之前）
      2) 多个 batch 的 act_max 用 torch.maximum 聚合（不是覆盖、不是累加）
      3) s = a_max^α / w_max^(1-α) + clamp_(lo, hi) 行为符合预期
      4) NaN/Inf 在 clamp 后被替换为 1.0
      5) 跳过没有捕获到激活的模块（state 不全时不应崩）
    """
    print("[test_calibrate_hook_pipeline_dryrun]", flush=True)

    torch.manual_seed(7)
    in_dim, out_dim = 16, 8
    container = nn.Module()
    container.col0 = ColumnParallelLinear(in_dim, out_dim, bias=False)
    container.col1 = ColumnParallelLinear(in_dim, out_dim, bias=False)  # 这层故意不喂数据
    with torch.no_grad():
        container.col0.weight.data.normal_(0, 0.05)
        container.col1.weight.data.normal_(0, 0.05)

    # —— 复刻 calibrate_smoothquant.py::main 里的 hook 管线 ——
    state: dict = {}

    def make_hook(name: str):
        def _hook(_mod, inputs):
            x = inputs[0]
            if x.numel() == 0:
                return
            x_flat = x.detach().reshape(-1, x.shape[-1]).float()
            cur = x_flat.abs().amax(dim=0)
            prev = state.get(name)
            state[name] = cur if prev is None else torch.maximum(prev, cur)
        return _hook

    handles = []
    for name, mod in container.named_modules():
        # 与脚本一致：只挂 LinearBase
        if isinstance(mod, linear_mod.LinearBase):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    # —— 多个 batch，每个 batch 在不同通道上有 outlier，验证 max 聚合 ——
    # batch1：通道 2 是 outlier
    x1 = torch.randn(4, in_dim) * 0.1
    x1[:, 2] = 5.0
    container.col0(x1)
    # batch2：通道 11 是 outlier；通道 2 仍有正常幅值（不能覆盖掉之前 max）
    x2 = torch.randn(4, in_dim) * 0.1
    x2[:, 11] = 8.0
    container.col0(x2)
    # batch3：再喂一遍，确认幂等 / max 单调
    container.col0(x2 * 0.1)

    for h in handles:
        h.remove()

    # 断言 hook 只对 col0 累计
    assert "col0" in state and "col1" not in state
    a_max = state["col0"]
    # 通道 2 应保留 batch1 的 5.0，通道 11 保留 batch2 的 8.0
    assert a_max[2].item() >= 5.0 - 1e-4, f"channel 2 max degraded: {a_max[2].item()}"
    assert a_max[11].item() >= 8.0 - 1e-4, f"channel 11 max degraded: {a_max[11].item()}"
    # 其它通道远小于 outlier
    non_outlier = torch.cat([a_max[:2], a_max[3:11], a_max[12:]])
    assert non_outlier.max().item() < 2.0, f"non-outlier leaked: {non_outlier.max().item()}"
    _ok(f"hook accumulates max across batches "
        f"(a_max[2]={a_max[2]:.3f}, a_max[11]={a_max[11]:.3f}, others<{non_outlier.max():.3f})")

    # —— 复刻脚本的聚合逻辑（含 clamp + NaN 兜底）——
    alpha = 0.5
    clamp_min, clamp_max = 1e-3, 1e3
    w = container.col0.weight.data.detach().float()
    w_max = w.abs().amax(dim=0).clamp_min(1e-5)
    a_max_clamped = a_max.clamp_min(1e-5)
    s = (a_max_clamped.pow(alpha) / w_max.pow(1.0 - alpha)).to(torch.float32)
    n_lo = int((s < clamp_min).sum().item())
    n_hi = int((s > clamp_max).sum().item())
    s = s.clamp_(clamp_min, clamp_max)
    assert torch.isfinite(s).all() and (s >= clamp_min).all() and (s <= clamp_max).all()
    _ok(f"aggregation s clamped to [{clamp_min},{clamp_max}] "
        f"(lo={n_lo}, hi={n_hi}, all finite)")

    # —— NaN/Inf 兜底分支 ——
    s_bad = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 2.0])
    s_bad = s_bad.clamp_(clamp_min, clamp_max)
    # NaN 不会被 clamp 掉；走脚本里 torch.where 替换分支
    if not torch.isfinite(s_bad).all():
        s_bad = torch.where(torch.isfinite(s_bad), s_bad, torch.ones_like(s_bad))
    assert torch.isfinite(s_bad).all()
    assert s_bad[1].item() == 1.0  # NaN -> 1.0
    _ok("NaN/Inf in s replaced with 1.0 (matches calibrate script fallback)")

    # —— 验证整条链路：把聚合出的 s 注入到原层，forward 前后等价（无 A8 时）——
    container2 = nn.Module()
    container2.col0 = ColumnParallelLinear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        container2.col0.weight.data.copy_(container.col0.weight.data)
    W_before = container2.col0.weight.data.clone()
    _apply_smoothquant_scales(container2, {"col0": s})
    # W *= s
    assert torch.allclose(container2.col0.weight.data, W_before * s.view(1, -1), atol=1e-6)
    # forward(x) == x @ W_before^T
    x_test = torch.randn(2, in_dim)
    y = container2.col0(x_test)
    y_ref = x_test @ W_before.t()
    assert torch.allclose(y, y_ref, atol=1e-5)
    _ok("end-to-end: hook → aggregate → inject → forward matches fp reference")


def test_tp_smoke_requires_scale_path_for_sq_configs():
    """w4a8_sq_* 配置名必须真的加载 SQ scale，不能静默退化成普通 W4A8。"""
    print("[test_tp_smoke_requires_scale_path_for_sq_configs]", flush=True)
    tp_smoke = _load_module("tp_smoke_under_test",
                            os.path.join(_REPO_ROOT, "tools/tp_smoke.py"))

    try:
        tp_smoke.smoothquant_extra_cfg_for_config("w4a8_sq_g32", None)
    except ValueError as e:
        assert "--smoothquant-scale-path" in str(e), str(e)
        _ok(f"missing SQ scale path rejected: {e}")
    else:
        raise AssertionError("expected ValueError for w4a8_sq config without scale path")

    assert tp_smoke.smoothquant_extra_cfg_for_config("w4a8_g32", None) == {}
    assert tp_smoke.smoothquant_extra_cfg_for_config("w4a8_sq_g32", "/tmp/sq.pt") == {
        "smoothquant_scale_path": "/tmp/sq.pt",
    }
    _ok("tp_smoke SQ config helper returns expected extra config")


def test_tp_smoke_act_quant_skip_extra_cfg():
    """tp_smoke 需要能把 A8 skip 参数透传给 LLM，用于验证 W4A8+SQ 稳态配置。"""
    print("[test_tp_smoke_act_quant_skip_extra_cfg]", flush=True)
    tp_smoke = _load_module("tp_smoke_under_test_skip",
                            os.path.join(_REPO_ROOT, "tools/tp_smoke.py"))

    assert tp_smoke.act_quant_skip_extra_cfg(0, 0, None) == {}
    assert tp_smoke.act_quant_skip_extra_cfg(2, 2, None) == {
        "act_quant_skip_first": 2,
        "act_quant_skip_last": 2,
    }
    assert tp_smoke.act_quant_skip_extra_cfg(1, 0, "6,31,35") == {
        "act_quant_skip_first": 1,
        "act_quant_skip_layers": [6, 31, 35],
    }
    _ok("tp_smoke act quant skip helper returns expected extra config")


def test_tp_smoke_summary_path_creates_nested_parent(tmp_root=None):
    """tp_smoke --out-file 支持带子目录的路径，避免汇总写文件阶段失败。"""
    print("[test_tp_smoke_summary_path_creates_nested_parent]", flush=True)
    import tempfile

    tp_smoke = _load_module("tp_smoke_under_test_summary_path",
                            os.path.join(_REPO_ROOT, "tools/tp_smoke.py"))
    root = tmp_root or tempfile.mkdtemp(prefix="tp_smoke_summary_path_")
    summary_path = tp_smoke.prepare_summary_path(root, "nested/result.json")

    assert summary_path == os.path.join(root, "nested/result.json")
    assert os.path.isdir(os.path.join(root, "nested"))
    with open(summary_path, "w") as f:
        f.write("[]")
    _ok("tp_smoke summary path creates nested parent directories")


def main():
    print("=" * 60)
    print("SmoothQuant CPU unit tests")
    print("=" * 60)
    test_numerical_equivalence()
    test_nan_scale_rejected()
    test_shape_mismatch_rejected()
    test_no_match_rejected()
    test_partial_match_ok()
    test_buffer_survives_finalize()
    test_sq_suppresses_outlier_a8_error()
    test_calibrate_hook_pipeline_dryrun()
    test_tp_smoke_requires_scale_path_for_sq_configs()
    test_tp_smoke_act_quant_skip_extra_cfg()
    test_tp_smoke_summary_path_creates_nested_parent()
    print()
    print("ALL PASS")


if __name__ == "__main__":
    main()
