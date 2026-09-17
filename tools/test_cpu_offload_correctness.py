"""Tests for the CPU offload correctness harness - the parts that can be wrong silently.

Run on any machine: everything here is CPU-only and model-free.

The incremental min/max maintenance gets most of the attention, because it is the one piece
whose failure does not look like a failure. A wrong summary does not crash and does not produce
NaNs; it produces a *slightly different selection*, which then looks like "sparse attention
loses a bit of quality" - the most expensive kind of bug in this project, since the whole
fidelity gate is measured in exactly those units.
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cpu_offload_correctness import (  # noqa: E402
    CpuKvMirror,
    CpuOffloadController,
    cpu_sparse_attention,
    verdict,
)
from tools.e2e_sparse_attention import (  # noqa: E402
    _repeat_kv,
    unit_minmax,
    units_to_token_index,
)

GRAN = 8
KVH = 2
DIM = 4


def _mirror(gran: int = GRAN) -> CpuKvMirror:
    return CpuKvMirror(gran, device="cpu", pin=False)


def _kv(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, KVH, n, DIM, generator=g, dtype=torch.float32)
    v = torch.randn(1, KVH, n, DIM, generator=g, dtype=torch.float32)
    return k, v


def test_summary_matches_recompute_after_bulk_prefill():
    m = _mirror()
    k, v = _kv(20)
    m.append(0, k, v, capacity_hint=40)
    assert m.drift_vs_recompute(0, k) == 0.0


def test_summary_matches_recompute_across_unit_boundaries():
    """The interesting case: appending one token at a time through a unit boundary.

    A unit that is starting must not inherit the +-inf sentinels in a way that survives, and a
    unit that is half full must not be widened by tokens it does not hold yet.
    """
    m = _mirror()
    k_all, v_all = _kv(23)
    m.append(0, k_all[:, :, :5, :], v_all[:, :, :5, :], capacity_hint=64)
    for n in range(6, 24):
        m.append(0, k_all[:, :, :n, :], v_all[:, :, :n, :])
        drift = m.drift_vs_recompute(0, k_all[:, :, :n, :])
        assert drift == 0.0, f"summary drifted at length {n}: {drift}"


def test_partial_last_unit_is_not_widened_by_absent_tokens():
    """A half-filled unit's bound must come only from the tokens present."""
    m = _mirror(gran=8)
    k_all, v_all = _kv(12, seed=3)
    m.append(0, k_all[:, :, :12, :], v_all[:, :, :12, :], capacity_hint=32)
    kmin, kmax = m.summaries(0)
    # unit 1 holds tokens 8..11 only
    ref_min = k_all[0, :, 8:12, :].amin(dim=1)      # [KVH, DIM]
    ref_max = k_all[0, :, 8:12, :].amax(dim=1)
    assert torch.equal(kmin[0, 1], ref_min)
    assert torch.equal(kmax[0, 1], ref_max)


def test_summaries_are_trimmed_to_live_units():
    m = _mirror(gran=8)
    k, v = _kv(9)
    m.append(0, k, v, capacity_hint=64)
    kmin, kmax = m.summaries(0)
    assert kmin.shape[1] == 2 and kmax.shape[1] == 2, "9 tokens at gran 8 is 2 units"
    assert torch.isfinite(kmin).all() and torch.isfinite(kmax).all(), \
        "a live unit must not still hold the +-inf sentinel"


def test_mirror_holds_the_same_bytes_as_the_source():
    m = _mirror()
    k_all, v_all = _kv(17, seed=7)
    m.append(0, k_all[:, :, :10, :], v_all[:, :, :10, :], capacity_hint=32)
    m.append(0, k_all, v_all)
    assert torch.equal(m.k[0][:, :, :17, :], k_all)
    assert torch.equal(m.v[0][:, :, :17, :], v_all)


def test_only_the_tail_is_copied():
    """The write path must be one token per step, not a bulk recopy of the history."""
    m = _mirror()
    k_all, v_all = _kv(40)
    m.append(0, k_all[:, :, :32, :], v_all[:, :, :32, :], capacity_hint=64)
    after_prefill = m.d2h_bytes
    m.append(0, k_all[:, :, :33, :], v_all[:, :, :33, :])
    one_token = m.d2h_bytes - after_prefill
    expected = 1 * KVH * DIM * k_all.element_size() * 2      # K and V
    assert one_token == expected, f"copied {one_token} bytes for one token, want {expected}"


def test_append_rejects_shrinking_and_overflow():
    m = _mirror()
    k_all, v_all = _kv(20)
    m.append(0, k_all[:, :, :10, :], v_all[:, :, :10, :], capacity_hint=12)
    with pytest.raises(ValueError):
        m.append(0, k_all[:, :, :5, :], v_all[:, :, :5, :])
    with pytest.raises(ValueError):
        m.append(0, k_all[:, :, :20, :], v_all[:, :, :20, :])


def test_gather_returns_the_selected_tokens_in_order():
    m = _mirror()
    k_all, v_all = _kv(24, seed=11)
    m.append(0, k_all, v_all, capacity_hint=24)
    idx = torch.tensor([0, 3, 17, 23], dtype=torch.long)
    k_sel, v_sel = m.gather(0, idx)
    assert torch.equal(k_sel, k_all[:, :, idx, :])
    assert torch.equal(v_sel, v_all[:, :, idx, :])


def test_cpu_attention_matches_a_dense_reference_over_the_same_tokens():
    """The CPU kernel must agree with torch's own attention over the gathered set."""
    g = torch.Generator().manual_seed(5)
    q = torch.randn(1, KVH * 3, 1, DIM, generator=g)
    k = torch.randn(1, KVH, 7, DIM, generator=g)
    v = torch.randn(1, KVH, 7, DIM, generator=g)
    scaling = DIM ** -0.5
    got = cpu_sparse_attention(q, k, v, scaling, 3)
    want = F.scaled_dot_product_attention(q, _repeat_kv(k, 3), _repeat_kv(v, 3),
                                          attn_mask=None, scale=scaling)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_cpu_attention_is_gqa_aware():
    """Repeating kv heads the wrong way is a plausible bug that still runs."""
    g = torch.Generator().manual_seed(6)
    q = torch.randn(1, 4, 1, DIM, generator=g)
    k = torch.randn(1, 2, 5, DIM, generator=g)
    v = torch.randn(1, 2, 5, DIM, generator=g)
    out = cpu_sparse_attention(q, k, v, DIM ** -0.5, 2)
    assert out.shape == (1, 4, 1, DIM)
    # head 0 and head 1 share kv head 0, so with identical queries they must agree
    q2 = q.clone()
    q2[:, 1] = q2[:, 0]
    out2 = cpu_sparse_attention(q2, k, v, DIM ** -0.5, 2)
    assert torch.allclose(out2[:, 0], out2[:, 1], atol=1e-6)


def test_units_to_token_index_stays_inside_the_sequence():
    idx = units_to_token_index(torch.tensor([0, 2]), 8, seq_len=20)
    assert int(idx.max()) < 20
    assert idx.numel() == 8 + 4, "the last unit is short and must not be padded out"


def test_summary_dtype_is_fp32_even_for_bf16_kv():
    """bf16 K with fp32 summaries: the bound must be exact for the values it saw."""
    m = _mirror(gran=4)
    k = torch.randn(1, KVH, 4, DIM).to(torch.bfloat16)
    v = torch.randn(1, KVH, 4, DIM).to(torch.bfloat16)
    m.append(0, k, v, capacity_hint=8)
    kmin, kmax = m.summaries(0)
    assert kmin.dtype == torch.float32 and kmax.dtype == torch.float32
    ref_min, ref_max = unit_minmax(k.to(torch.float32), 4)
    assert torch.equal(kmin, ref_min) and torch.equal(kmax, ref_max)


def test_verdict_fails_loudly_on_each_clause():
    good = {"summary_drift": {"max": 0.0}, "index_mismatches": 0,
            "rel_err_vs_gpu_fp32": {"max": 1e-7}}
    assert verdict(good, 1.0, 0.9, True)["passed"]

    drifted = dict(good, summary_drift={"max": 1e-9})
    assert not verdict(drifted, 1.0, 0.9, True)["passed"], \
        "any drift at all is a maintenance bug, not noise"

    mismatched = dict(good, index_mismatches=1)
    assert not verdict(mismatched, 1.0, 0.9, True)["passed"]

    inaccurate = dict(good, rel_err_vs_gpu_fp32={"max": 1e-3})
    assert not verdict(inaccurate, 1.0, 0.9, True)["passed"]

    # a different continuation from the *dense* arm is allowed; from the GPU sparse arm is not
    assert verdict(good, 1.0, 0.1, False)["passed"]
    assert not verdict(good, 0.5, 1.0, True)["passed"]


def test_controller_reset_clears_the_mirror_and_counters():
    ctl = CpuOffloadController(GRAN, 0.5, capacity=32, device="cpu", pin=False)
    k, v = _kv(16)
    ctl.mirror.append(0, k, v, capacity_hint=32)
    ctl.err_fp32.append(1.0)
    ctl.reset("cpu")
    assert ctl.mirror.length == {} and ctl.err_fp32 == []
    assert ctl.sparse_calls == 0 and ctl.idx_mismatches == 0
