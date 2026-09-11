#!/usr/bin/env python3
"""How many tokens may the action drafter emit before it stops paying?

Stage 1b step 0 falsified the fixed 4096-entry code head. The instinctive
repair is to train a pointer head, which is a lot of work for a line that
has not yet earned it. Before accepting that cost, it is worth asking what
the cost model actually demands.

Stage 1a-bis priced three arms, and the code head was never the point. The
point was *few output tokens*: a drafter's cost is one prefill over the
compressed context plus one decode step per emitted token. The head was
simply the extreme case, k = 1 projection. So the real constraint is a
token budget, and the useful question is how large that budget is.

If the budget is around ten tokens, a prompted small model cannot emit a
tool call in text and a compact learned output space is unavoidable. If
the budget is forty tokens, a *zero-training* prompted drafter that writes
the tool call as ordinary text is admissible, and the whole pointer-head
programme can be skipped until something cheaper has been falsified.

This script inverts the Stage 0 cost model over measured serving-path
costs. It trains nothing, measures nothing, and adds no assumption that
is not already in the artifact it reads.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_TOKENS_SCANNED = 512


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

TOOL_SECONDS = (0.2, 1.0, 5.0, 20.0)
BASELINE_UTILIZATION = (0.0, 0.6, 0.8)
MATCH_PROBABILITY = 0.75
ROLLBACK_SECONDS = 0.5
CURVE_TOKENS = (1, 2, 4, 8, 11, 16, 24, 32, 48)


def max_output_tokens(
    demand,
    prefill_seconds,
    step_seconds,
    tool_seconds,
    baseline_utilization,
    match_probability,
    rollback_seconds,
):
    """Largest k whose draft cost still yields a net_positive verdict.

    Scanned rather than solved. The verdict is not guaranteed monotone in
    the tax by inspection alone, and a scan cannot be wrong about the
    model it is scanning.
    """

    admissible = 0
    for tokens in range(0, MAX_TOKENS_SCANNED + 1):
        tax = (prefill_seconds + tokens * step_seconds) / demand
        try:
            result = _COST_MODEL.evaluate(
                _COST_MODEL.build_cost_inputs(
                    actor_gpu_seconds=demand,
                    draft_gpu_tax=tax,
                    tool_seconds=tool_seconds,
                    baseline_utilization=baseline_utilization,
                    match_probability=match_probability,
                    rollback_seconds=rollback_seconds,
                )
            )
        except ValueError:
            break
        if result.verdict == "net_positive":
            admissible = tokens
        elif tokens > 0 and admissible:
            # Past the boundary; the first non-positive verdict after an
            # admissible run is the end of the feasible region.
            break
    return admissible


def latency_saving(
    demand,
    prefill_seconds,
    step_seconds,
    tokens,
    tool_seconds,
    baseline_utilization,
    match_probability,
    rollback_seconds,
):
    """Absolute and relative saving for a drafter that emits `tokens`.

    The break-even scan above reports where the saving reaches zero,
    which is precisely the operating point nobody wants. The saving
    itself is the quantity a design decision should be made on, because
    it decays with every emitted token and the decay is not gentle.
    """

    tax = (prefill_seconds + tokens * step_seconds) / demand
    result = _COST_MODEL.evaluate(
        _COST_MODEL.build_cost_inputs(
            actor_gpu_seconds=demand,
            draft_gpu_tax=tax,
            tool_seconds=tool_seconds,
            baseline_utilization=baseline_utilization,
            match_probability=match_probability,
            rollback_seconds=rollback_seconds,
        )
    )
    if result.speculative_latency_seconds is None:
        return {
            "tokens": tokens,
            "draft_gpu_tax": tax,
            "verdict": result.verdict,
            "saving_seconds": None,
            "speedup": None,
        }
    return {
        "tokens": tokens,
        "draft_gpu_tax": tax,
        "verdict": result.verdict,
        "saving_seconds": (
            result.baseline_latency_seconds
            - result.speculative_latency_seconds
        ),
        "speedup": result.speedup,
    }


def build_payload(artifact_path, mode, compressed_budget):
    with open(artifact_path, "r", encoding="utf-8") as handle:
        engine = json.load(handle)
    if engine.get("worker") != "agentspec_engine_demand":
        raise SystemExit("not an agentspec_engine_demand artifact")
    rows = engine["modes"][mode]["rows"]
    payload = {
        "worker": "agentspec_output_token_budget",
        "source_artifact_sha256": engine["payload_sha256"],
        "mode": mode,
        "match_probability": MATCH_PROBABILITY,
        "rollback_seconds": ROLLBACK_SECONDS,
        "compressed_prefill_context": compressed_budget,
        "rows": [],
    }
    for row in rows:
        demand = row["actor_demand_seconds"]
        step = row["drafter_decode_step_seconds"]
        for label, prefill in (
            ("compressed", row["compressed_prefill_seconds"]),
            ("full_context", row["drafter_prefill_seconds"]),
        ):
            for tool_seconds in TOOL_SECONDS:
                for utilization in BASELINE_UTILIZATION:
                    payload["rows"].append(
                        {
                            "context_length": row["context_length"],
                            "drafter_context": label,
                            "tool_seconds": tool_seconds,
                            "baseline_utilization": utilization,
                            "actor_demand_seconds": demand,
                            "drafter_prefill_seconds": prefill,
                            "drafter_step_seconds": step,
                            "max_output_tokens": max_output_tokens(
                                demand,
                                prefill,
                                step,
                                tool_seconds,
                                utilization,
                                MATCH_PROBABILITY,
                                ROLLBACK_SECONDS,
                            ),
                            "token_curve": [
                                latency_saving(
                                    demand,
                                    prefill,
                                    step,
                                    tokens,
                                    tool_seconds,
                                    utilization,
                                    MATCH_PROBABILITY,
                                    ROLLBACK_SECONDS,
                                )
                                for tokens in CURVE_TOKENS
                            ],
                        }
                    )
    return payload


def render(payload):
    lines = [
        "source artifact sha256 %s"
        % payload["source_artifact_sha256"],
        "mode                   %s" % payload["mode"],
        "match probability      %.2f" % payload["match_probability"],
        "rollback               %.1f s" % payload["rollback_seconds"],
        "",
        "max output tokens a drafter may emit and still be net_positive",
        "",
    ]
    contexts = sorted(
        {row["context_length"] for row in payload["rows"]}
    )
    for drafter_context in ("compressed", "full_context"):
        lines.append("drafter context: %s" % drafter_context)
        lines.append(
            "%-10s %8s %8s %8s %8s"
            % ("context", "tool.2s", "tool1s", "tool5s", "tool20s")
        )
        for context_length in contexts:
            for utilization in sorted(
                {row["baseline_utilization"] for row in payload["rows"]}
            ):
                cells = []
                for tool_seconds in TOOL_SECONDS:
                    match = [
                        row["max_output_tokens"]
                        for row in payload["rows"]
                        if row["context_length"] == context_length
                        and row["drafter_context"] == drafter_context
                        and row["tool_seconds"] == tool_seconds
                        and row["baseline_utilization"] == utilization
                    ]
                    cells.append(match[0] if match else 0)
                lines.append(
                    "%-10s %8d %8d %8d %8d   rho=%.1f"
                    % ((context_length,) + tuple(cells) + (utilization,))
                )
        lines.append("")
    lines.extend(
        [
            "latency saving per action as the drafter gets more verbose",
            "compressed drafter context, rho=0.6, negative means loss",
            "",
            "%-8s %-8s %s"
            % (
                "context",
                "tool_s",
                " ".join("%8s" % ("k=%d" % k) for k in CURVE_TOKENS),
            ),
        ]
    )
    for context_length in contexts:
        for tool_seconds in TOOL_SECONDS:
            match = [
                row
                for row in payload["rows"]
                if row["context_length"] == context_length
                and row["drafter_context"] == "compressed"
                and row["tool_seconds"] == tool_seconds
                and row["baseline_utilization"] == 0.6
            ]
            if not match:
                continue
            cells = []
            for point in match[0]["token_curve"]:
                saving = point["saving_seconds"]
                cells.append(
                    "%8.4f" % saving if saving is not None else "%8s" % "n/a"
                )
            lines.append(
                "%-8s %-8s %s"
                % (context_length, tool_seconds, " ".join(cells))
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Invert the cost model into an output token budget",
    )
    parser.add_argument("artifact")
    parser.add_argument("--mode", default="cuda_graph")
    parser.add_argument("--compressed-budget", type=int, default=512)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = build_payload(
        args.artifact, args.mode, args.compressed_budget
    )
    print(render(payload))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("artifact %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
