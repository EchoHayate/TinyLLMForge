#!/usr/bin/env python3
"""Turn measured context growth into a verdict on the speculation line.

The erratum left one number able to reverse the Stage 1b NO-GO: how much
fresh context a real agent turn re-prefills. If it is large, warm demand
stays large, the profitability floor stays low, and the training-free
n-gram drafter can still pay. ``agentspec_context_growth.py`` measured
it. This script does the arithmetic that follows.

Two derived quantities matter and neither is in the growth artifact:

``D threshold``
    The actor demand at which the cost model's minimum match probability
    falls to exactly what the best training-free predictor achieves. Any
    demand below it is unprofitable no matter how cheap the drafter is,
    because the drafter here is already priced at zero.

``required prefix-cache miss rate``
    Real serving is neither fully warm nor fully cold: under memory
    pressure an idle agent's blocks get recycled while it waits on its
    tool. Expected demand is a mixture, so the honest question is not
    "warm or cold" but "how often must the cache miss before speculation
    pays". That is solvable in closed form from the measured warm and
    cold means, and it is the number a serving team can actually check.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cost_model():
    parent = types.ModuleType("tinyvllm")
    parent.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
    sys.modules.setdefault("tinyvllm", parent)
    child = types.ModuleType("tinyvllm.agentspec")
    child.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "agentspec")]
    sys.modules.setdefault("tinyvllm.agentspec", child)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return importlib.import_module("tinyvllm.agentspec.cost_model")


def _minimum_p(cost_model, demand, inputs):
    verdict = cost_model.evaluate(
        cost_model.build_cost_inputs(
            actor_gpu_seconds=demand,
            draft_gpu_tax=0.0,
            tool_seconds=inputs["tool_seconds"],
            baseline_utilization=inputs["rho"],
            match_probability=0.75,
            rollback_seconds=inputs["rollback_seconds"],
        )
    )
    return verdict.minimum_match_probability


def _solve_demand_threshold(cost_model, target_p, inputs, hi=60.0):
    """Smallest demand whose minimum required p drops to ``target_p``.

    ``minimum_p`` falls monotonically as demand grows: a longer actor
    turn is more worth hiding. Bisection is therefore safe and needs no
    derivative of the cost model.
    """
    lo = 1e-4
    if _minimum_p(cost_model, hi, inputs) > target_p:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _minimum_p(cost_model, mid, inputs) > target_p:
            lo = mid
        else:
            hi = mid
    return hi


def analyse(growth, cost_model):
    inputs = growth["cost_inputs"]
    trigram_p = growth["speculation_gate"]["trigram_p"]
    warm = growth["demand_seconds"]["warm"]
    cold = growth["demand_seconds"]["cold"]
    threshold = _solve_demand_threshold(cost_model, trigram_p, inputs)

    miss_rate = None
    if threshold is not None and cold["mean"] > warm["mean"]:
        raw = (threshold - warm["mean"]) / (cold["mean"] - warm["mean"])
        miss_rate = raw
    return {
        "corpus": growth["corpus"],
        "turns_scored": growth["turns_scored"],
        "trigram_p": trigram_p,
        "prefill_tokens_per_turn": {
            key: growth["tokens_per_turn"]["prefill"][key]
            for key in ("mean", "p50", "p90", "p99", "max")
        },
        "observation_tokens_p50": growth["tokens_per_turn"]["observation"]["p50"],
        "demand_warm_mean": warm["mean"],
        "demand_cold_mean": cold["mean"],
        "minimum_p_at_warm_mean": _minimum_p(
            cost_model, warm["mean"], inputs
        ),
        "minimum_p_at_cold_mean": _minimum_p(
            cost_model, cold["mean"], inputs
        ),
        "demand_threshold_seconds": threshold,
        "prefill_tokens_needed_estimate": 4000,
        "required_prefix_cache_miss_rate": miss_rate,
        "fraction_turns_clearing_warm": growth["speculation_gate"][
            "fraction_clearing_floor_warm"
        ],
        "fraction_turns_clearing_cold": growth["speculation_gate"][
            "fraction_clearing_floor_cold"
        ],
        "gate": (
            "FAIL"
            if growth["speculation_gate"]["fraction_clearing_floor_warm"] < 0.5
            else "PASS"
        ),
        "cost_inputs": inputs,
        "growth_payload_sha256": growth["payload_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("growth", nargs="+", help="context growth artifacts")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    cost_model = _load_cost_model()
    results = []
    for path in args.growth:
        with open(path, "r", encoding="utf-8") as handle:
            results.append(analyse(json.load(handle), cost_model))

    first = results[0]["cost_inputs"]
    print(
        "tool_seconds %.1f  rho %.2f  rollback %.2f  drafter tax 0"
        % (first["tool_seconds"], first["rho"], first["rollback_seconds"])
    )
    print()
    header = (
        "%-10s %8s %8s %8s %9s %9s %9s %7s %6s"
        % (
            "corpus",
            "pre_p50",
            "pre_p99",
            "D_warm",
            "min_p@w",
            "trigram",
            "D_thresh",
            "miss%",
            "gate",
        )
    )
    print(header)
    for row in results:
        print(
            "%-10s %8.0f %8.0f %8.4f %9.4f %9.4f %9.4f %7s %6s"
            % (
                row["corpus"],
                row["prefill_tokens_per_turn"]["p50"],
                row["prefill_tokens_per_turn"]["p99"],
                row["demand_warm_mean"],
                row["minimum_p_at_warm_mean"],
                row["trigram_p"],
                row["demand_threshold_seconds"] or float("nan"),
                (
                    "%.0f%%" % (100 * row["required_prefix_cache_miss_rate"])
                    if row["required_prefix_cache_miss_rate"] is not None
                    else "n/a"
                ),
                row["gate"],
            )
        )
    print()
    for row in results:
        print(
            "%-10s turns clearing floor: warm %.4f  cold %.4f"
            % (
                row["corpus"],
                row["fraction_turns_clearing_warm"],
                row["fraction_turns_clearing_cold"],
            )
        )

    if args.output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": 1, "results": results},
                handle,
                indent=1,
                sort_keys=True,
            )
            handle.write("\n")
        print()
        print("artifact %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
