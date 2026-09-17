"""Tests for the three new gates' pure logic.

The measurements need a GPU, a compiled kernel and 64 cores, so what is testable here is the
arithmetic that turns samples into verdicts, plus the selector maths - the latter matters
most, because if the clamp trick is not equal to the literal Quest bound then the selector
cost gate is pricing a kernel nobody would ship.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from async_handshake_gate import VARIANTS, host_syncs_per_step
from microbatch_pipeline_prototype import analyse
from selector_cost_gate import quest_scores_shared, reference_scores_shared, verify_bound_math

torch = pytest.importorskip("torch")


# --- handshake bookkeeping ---------------------------------------------------

def test_lookahead_halves_the_host_syncs():
    # the whole point of the variant: the output no longer needs a host block
    assert host_syncs_per_step("device_sync", 36) == 72
    assert host_syncs_per_step("lookahead", 36) == 36


def test_fused_is_two_syncs_regardless_of_depth():
    assert host_syncs_per_step("fused", 36) == 2
    assert host_syncs_per_step("fused", 80) == 2


def test_every_variant_declares_a_sync_count():
    for v in VARIANTS:
        assert host_syncs_per_step(v, 36) > 0


def test_unknown_variant_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError):
        host_syncs_per_step("wishful", 36)


# --- selector maths ---------------------------------------------------------

def test_clamp_trick_equals_the_literal_quest_bound():
    assert verify_bound_math()["ok"]


def test_bound_math_holds_when_all_queries_are_negative():
    # the sign split is where a clamp implementation usually breaks
    torch.manual_seed(1)
    q = -torch.rand(4, 8).abs() - 0.1
    kmin = torch.randn(9, 4, 8)
    kmax = kmin + torch.rand_like(kmin)
    a = quest_scores_shared(q, kmin, kmax)
    b = reference_scores_shared(q, kmin, kmax)
    assert torch.allclose(a, b, atol=1e-5)


def test_bound_is_an_upper_bound_on_the_real_dot_products():
    # Quest's guarantee: the bound must never underestimate any key inside the unit
    torch.manual_seed(2)
    dim, heads, unit = 8, 2, 16
    keys = torch.randn(unit, heads, dim)
    kmin, kmax = keys.min(dim=0).values, keys.max(dim=0).values
    q = torch.randn(heads, dim)
    bound = quest_scores_shared(q, kmin.unsqueeze(0), kmax.unsqueeze(0))[0]
    real = (keys * q).sum(dim=(1, 2))
    assert bound >= real.max() - 1e-5


# --- pipeline analysis -----------------------------------------------------

def _res(gpu, cpu, serial, pipe):
    return {"gpu_only": {"median_ms": gpu}, "cpu_only": {"median_ms": cpu},
            "serial": {"median_ms": serial}, "pipelined": {"median_ms": pipe}}


def test_perfect_overlap_reports_efficiency_one_and_hides_all_cpu_time():
    a = analyse(_res(gpu=12.0, cpu=2.0, serial=14.0, pipe=12.0))
    assert a["overlap_efficiency"] == 1.0
    assert a["cost_of_imperfect_overlap_ms"] == 0.0
    assert a["cpu_time_hidden_pct"] == 100.0


def test_no_overlap_at_all_reports_zero_hidden():
    a = analyse(_res(gpu=12.0, cpu=2.0, serial=14.0, pipe=14.0))
    assert a["cpu_time_hidden_pct"] == 0.0
    assert a["overlap_efficiency"] < 0.9


def test_cpu_bound_case_uses_the_cpu_as_the_ideal():
    # when the CPU is slower than the GPU, max() must pick the CPU, otherwise a CPU-bound
    # schedule would be scored as if it were doing great
    a = analyse(_res(gpu=5.0, cpu=20.0, serial=25.0, pipe=21.0))
    assert a["ideal_ms"] == 20.0
    assert a["overlap_efficiency"] == pytest.approx(20.0 / 21.0, abs=1e-3)
