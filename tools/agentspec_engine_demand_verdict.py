#!/usr/bin/env python3
"""Recompute Stage 0 break-even verdicts from measured serving-path cost.

Stage 0 froze a scenario matrix around a *declared* actor demand of
0.080 s. Stage 1a measured 1.20 to 3.42 s with an eager Hugging Face
loop, which moved every threshold, and Stage 1a-bis re-measured 0.49 to
2.04 s on the tinyvllm serving path. Because the profitability
thresholds are functions of D, the honest move is to re-run the same
cost model at the measured operating points rather than to reuse the
pre-registered tau bound.

This script consumes an ``agentspec_engine_demand`` artifact and reports,
per context length and per drafter arm, how the frozen scenario grid
resolves at the measured ``(D, tau)``. It measures nothing itself and
claims no match probability: every match probability in the grid is a
hypothesis, and the useful output is the *minimum* match probability
each operating point would require.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types

# The cost model is imported the same way the Stage 0 gate imports it:
# ``tinyvllm/__init__.py`` pulls in torch and transformers, which are
# not present on a laptop, so the parent packages are stubbed and only
# the dependency-free submodule is executed. This keeps the verdict
# recomputation runnable anywhere the artifact can be read.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cost_model():
    package_root = os.path.join(_REPO_ROOT, "tinyvllm")
    if "tinyvllm" not in sys.modules:
        parent = types.ModuleType("tinyvllm")
        parent.__path__ = [package_root]
        sys.modules["tinyvllm"] = parent
    if "tinyvllm.agentspec" not in sys.modules:
        child = types.ModuleType("tinyvllm.agentspec")
        child.__path__ = [os.path.join(package_root, "agentspec")]
        sys.modules["tinyvllm.agentspec"] = child
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return importlib.import_module("tinyvllm.agentspec.cost_model")


_COST_MODEL = _load_cost_model()
build_cost_inputs = _COST_MODEL.build_cost_inputs
evaluate = _COST_MODEL.evaluate

ARMS = ("text_drafter", "code_drafter", "code_drafter_ckv")
TOOL_SECONDS = (0.2, 1.0, 5.0)
BASELINE_UTILIZATION = (0.0, 0.3, 0.6, 0.8, 0.9)
MATCH_PROBABILITY = (0.55, 0.75, 0.90)
ROLLBACK_SECONDS = (0.0, 0.5)

# The reference point is the one Stage 1b would actually be built for:
# a shared engine that is already busy, a tool call that is slower than
# a single actor step, a drafter that is right more often than not, and
# a rollback that is not free.
REFERENCE = {
    "tool_seconds": 1.0,
    "baseline_utilization": 0.6,
    "match_probability": 0.75,
    "rollback_seconds": 0.5,
}


def _rows(payload, mode):
    modes = payload["modes"]
    if mode not in modes:
        raise SystemExit(
            "mode %r not in artifact; available: %s"
            % (mode, ", ".join(sorted(modes)))
        )
    return modes[mode]["rows"]


def _grid_summary(demand, tax):
    """Resolve the frozen scenario grid at one measured (D, tax)."""

    counts = {}
    total = 0
    for tool_seconds in TOOL_SECONDS:
        for utilization in BASELINE_UTILIZATION:
            for probability in MATCH_PROBABILITY:
                for rollback in ROLLBACK_SECONDS:
                    result = evaluate(
                        build_cost_inputs(
                            actor_gpu_seconds=demand,
                            draft_gpu_tax=tax,
                            tool_seconds=tool_seconds,
                            baseline_utilization=utilization,
                            match_probability=probability,
                            rollback_seconds=rollback,
                        )
                    )
                    counts[result.verdict] = (
                        counts.get(result.verdict, 0) + 1
                    )
                    total += 1
    return counts, total


def _reference_point(demand, tax):
    return evaluate(
        build_cost_inputs(
            actor_gpu_seconds=demand,
            draft_gpu_tax=tax,
            **REFERENCE,
        )
    )


def analyse(payload, mode):
    lines = []
    rows = _rows(payload, mode)
    lines.append("serving path   %s" % payload["serving_path"])
    lines.append("mode           %s" % mode)
    lines.append("device         %s" % payload["cuda_device_name"])
    lines.append("actor          %s" % payload["actor_identity"]["path"])
    lines.append(
        "drafter        %s" % payload["drafter_identity"]["path"]
    )
    lines.append("")
    lines.append(
        "reference point: tool %.1fs, rho %.2f, p %.2f, rollback %.1fs"
        % (
            REFERENCE["tool_seconds"],
            REFERENCE["baseline_utilization"],
            REFERENCE["match_probability"],
            REFERENCE["rollback_seconds"],
        )
    )
    lines.append("")
    header = (
        "context      D_s  arm                 tau   verdict"
        "            speedup   min_p  crit_tax  crit_rho"
    )
    lines.append(header)
    for row in rows:
        demand = row["actor_demand_seconds"]
        for arm in ARMS:
            tax = row["measured_draft_gpu_tax"][arm]
            result = _reference_point(demand, tax)
            lines.append(
                "%7d  %7.4f  %-16s  %6.4f  %-18s %7s  %6s  %8s  %8s"
                % (
                    row["context_length"],
                    demand,
                    arm,
                    tax,
                    result.verdict,
                    "n/a"
                    if result.speedup is None
                    else "%.4f" % result.speedup,
                    "n/a"
                    if result.minimum_match_probability is None
                    else "%.3f" % result.minimum_match_probability,
                    "n/a"
                    if result.critical_draft_tax is None
                    else "%.4f" % result.critical_draft_tax,
                    "n/a"
                    if result.critical_utilization is None
                    else "%.4f" % result.critical_utilization,
                )
            )
    lines.append("")
    lines.append("frozen scenario grid, %d points per arm" % (
        len(TOOL_SECONDS)
        * len(BASELINE_UTILIZATION)
        * len(MATCH_PROBABILITY)
        * len(ROLLBACK_SECONDS)
    ))
    lines.append("")
    lines.append(
        "context  arm                 net_positive  net_negative"
        "  no_benefit  unstable"
    )
    for row in rows:
        demand = row["actor_demand_seconds"]
        for arm in ARMS:
            tax = row["measured_draft_gpu_tax"][arm]
            counts, _total = _grid_summary(demand, tax)
            lines.append(
                "%7d  %-16s  %12d  %12d  %10d  %8d"
                % (
                    row["context_length"],
                    arm,
                    counts.get("net_positive", 0),
                    counts.get("net_negative", 0),
                    counts.get("infeasible_no_match_benefit", 0),
                    counts.get("unstable_capacity", 0),
                )
            )
    return "\n".join(lines)


def compare_modes(payload, fast_mode, slow_mode):
    """Show how much of Stage 1a's D was harness overhead."""

    fast = {
        row["context_length"]: row for row in _rows(payload, fast_mode)
    }
    slow = {
        row["context_length"]: row for row in _rows(payload, slow_mode)
    }
    shared = sorted(set(fast) & set(slow))
    lines = [
        "",
        "harness attribution: %s versus %s" % (slow_mode, fast_mode),
        "context   D_%s   D_%s   ratio   step_ratio_%s  step_ratio_%s"
        % (slow_mode, fast_mode, slow_mode, fast_mode),
    ]
    for context_length in shared:
        slow_row = slow[context_length]
        fast_row = fast[context_length]
        lines.append(
            "%7d  %8.4f  %8.4f  %6.3f  %13.3f  %13.3f"
            % (
                context_length,
                slow_row["actor_demand_seconds"],
                fast_row["actor_demand_seconds"],
                slow_row["actor_demand_seconds"]
                / fast_row["actor_demand_seconds"],
                slow_row["decode_step_ratio_drafter_over_actor"],
                fast_row["decode_step_ratio_drafter_over_actor"],
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute Stage 0 verdicts at measured cost",
    )
    parser.add_argument("artifact")
    parser.add_argument("--mode", default="cuda_graph")
    parser.add_argument("--compare-mode", default="eager")
    args = parser.parse_args(argv)

    with open(args.artifact, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("worker") != "agentspec_engine_demand":
        raise SystemExit(
            "artifact is not an agentspec_engine_demand payload"
        )
    print(analyse(payload, args.mode))
    if args.compare_mode in payload["modes"]:
        print(compare_modes(payload, args.mode, args.compare_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
