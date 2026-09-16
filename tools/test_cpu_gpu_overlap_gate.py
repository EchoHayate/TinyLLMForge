"""Tests for the overlap gate accounting.

The measurement functions need a GPU and a compiled benchmark, so what is tested here is
the part that turns numbers into a decision - which is where a wrong sign or a forgotten
term would silently produce an optimistic verdict.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpu_gpu_overlap_gate import C0_MS, summarize


def test_overhead_charges_both_inflation_and_transfers():
    s = summarize(gpu_alone_ms=12.0, gpu_under_load_ms=13.0, cpu_attn_ms=2.0,
                  pcie_per_layer_ms=1.5, window_ms=12.113)
    assert s["gpu_inflation_ms"] == 1.0
    # the transfers and the GPU slowdown are paid whether or not the overlap works
    assert s["unconditional_overhead_ms"] == 2.5
    assert s["headroom_ms"] == round(12.113 - 2.5 - 2.0, 3)
    assert s["verdict"] == "HIDEABLE"


def test_verdict_flips_when_the_cpu_attention_alone_exceeds_the_window():
    s = summarize(gpu_alone_ms=12.0, gpu_under_load_ms=12.0, cpu_attn_ms=20.1,
                  pcie_per_layer_ms=0.0, window_ms=C0_MS)
    assert s["verdict"] == "NOT_HIDEABLE"
    assert s["headroom_ms"] < 0


def test_transfers_alone_can_kill_the_schedule():
    # cheap attention, but a per-layer round trip that eats the entire window
    s = summarize(gpu_alone_ms=12.0, gpu_under_load_ms=12.0, cpu_attn_ms=0.42,
                  pcie_per_layer_ms=13.0, window_ms=C0_MS)
    assert s["verdict"] == "NOT_HIDEABLE"


def test_inflation_percentage_is_relative_to_the_undisturbed_gpu():
    s = summarize(gpu_alone_ms=10.0, gpu_under_load_ms=11.5, cpu_attn_ms=1.0,
                  pcie_per_layer_ms=0.5, window_ms=C0_MS)
    assert s["gpu_inflation_pct"] == 15.0


def test_summary_keeps_the_critical_path_caveat_visible():
    s = summarize(12.0, 12.1, 0.4, 0.3)
    assert "critical" in s["note"]
