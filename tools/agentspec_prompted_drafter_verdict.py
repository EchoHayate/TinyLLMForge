#!/usr/bin/env python3
"""Judge the prompted drafter against the step 0b pre-registration.

The pre-registration fixed the thresholds before the sweep ran, so the
only job here is arithmetic: recompute the minimum match probability
the cost model demands at each context length under the measured
drafter cost, and compare it with what the sweep actually produced.

Two match rates are carried through rather than one, because they
bracket the truth. ``p_speculated`` is the cost model's `p` and assumes
every eligible step is spoken on. ``p_effective`` charges every
abstention as a miss, which overcharges, because an abstention never
pays rollback. A design that fails on both is not a borderline call.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
import types


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_SECONDS = 5.0
BASELINE_UTILIZATION = 0.6
ROLLBACK_SECONDS = 0.5


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


def required_probability(row, tokens, prefill_key):
    demand = row["actor_demand_seconds"]
    prefill = row[prefill_key]
    step = row["drafter_decode_step_seconds"]
    tax = (prefill + tokens * step) / demand
    result = _COST_MODEL.evaluate(
        _COST_MODEL.build_cost_inputs(
            actor_gpu_seconds=demand,
            draft_gpu_tax=tax,
            tool_seconds=TOOL_SECONDS,
            baseline_utilization=BASELINE_UTILIZATION,
            match_probability=0.75,
            rollback_seconds=ROLLBACK_SECONDS,
        )
    )
    return tax, result.minimum_match_probability, result.verdict


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Judge prompted drafter runs against the gate",
    )
    parser.add_argument("engine_demand")
    parser.add_argument("match_dir")
    parser.add_argument("--mode", default="cuda_graph")
    parser.add_argument("--prefill", default="compressed_prefill_seconds")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    with open(args.engine_demand, "r", encoding="utf-8") as handle:
        engine = json.load(handle)
    rows = engine["modes"][args.mode]["rows"]

    runs = []
    for path in sorted(glob.glob(os.path.join(args.match_dir, "match_*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            runs.append((os.path.basename(path)[6:-5], json.load(handle)))
    if not runs:
        raise SystemExit("no match_*.json under %s" % args.match_dir)

    verdicts = []
    for name, run in runs:
        for row in rows:
            tax, minimum, verdict = required_probability(
                row, run["token_cap"], args.prefill
            )
            if minimum is None:
                passed = False
            else:
                passed = run["p_speculated"] >= minimum
            verdicts.append(
                {
                    "run": name,
                    "corpus": run["corpus"],
                    "variant": run["variant"],
                    "prompt_style": run["prompt_style"],
                    "context_length": row["context_length"],
                    "token_cap": run["token_cap"],
                    "draft_gpu_tax": tax,
                    "cost_verdict": verdict,
                    "required_p": minimum,
                    "p_speculated": run["p_speculated"],
                    "p_effective": run["p_effective"],
                    "coverage": run["coverage"],
                    "tool_accuracy": run["tool_accuracy_speculated"],
                    "gate": "PASS" if passed else "FAIL",
                    "shortfall": (
                        None
                        if minimum in (None, 0)
                        else run["p_speculated"] / minimum
                    ),
                }
            )

    print(
        "tool_seconds %.1f  rho %.1f  rollback %.1f  prefill %s"
        % (
            TOOL_SECONDS,
            BASELINE_UTILIZATION,
            ROLLBACK_SECONDS,
            args.prefill,
        )
    )
    print("")
    print(
        "%-34s %7s %6s %9s %9s %9s %7s %6s"
        % (
            "run",
            "context",
            "cap",
            "required_p",
            "p_spec",
            "p_eff",
            "of_req",
            "gate",
        )
    )
    for row in verdicts:
        print(
            "%-34s %7d %6d %9s %9.4f %9.4f %7s %6s"
            % (
                row["run"],
                row["context_length"],
                row["token_cap"],
                "%.4f" % row["required_p"]
                if row["required_p"] is not None
                else "n/a",
                row["p_speculated"],
                row["p_effective"],
                "%.3fx" % row["shortfall"]
                if row["shortfall"] is not None
                else "n/a",
                row["gate"],
            )
        )

    passed = sum(1 for row in verdicts if row["gate"] == "PASS")
    print("")
    print("PASS %d of %d" % (passed, len(verdicts)))
    if args.output:
        payload = {
            "worker": "agentspec_prompted_drafter_verdict",
            "tool_seconds": TOOL_SECONDS,
            "baseline_utilization": BASELINE_UTILIZATION,
            "rollback_seconds": ROLLBACK_SECONDS,
            "prefill_key": args.prefill,
            "source_engine_artifact": engine["payload_sha256"],
            "rows": verdicts,
            "passed": passed,
            "evaluated": len(verdicts),
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("artifact %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
