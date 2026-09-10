#!/usr/bin/env python3
"""Stage 0 gate for the latent action speculation line.

The gate is analytic and dependency-light. It imports no third-party
package, loads no checkpoint, and runs no model. It evaluates the
frozen scenario matrix in ``tinyvllm/agentspec/cost_model.py``,
applies the fail-closed router, checks a fixed set of falsifiable
invariants, and writes a deterministic JSON artifact.

Explicit claim boundary: a PASS here means the analytic contract and
its invariants hold. It is not evidence of any measured speedup.

Usage::

    python3 tools/agentspec_breakeven_gate.py --print-summary
    python3 tools/agentspec_breakeven_gate.py --output-dir OUT
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
GATE_NAME = "agentspec_breakeven"
GATE_SCHEMA_VERSION = 1
ROUNDING_DECIMALS = 6


def load_agentspec():
    """Load ``tinyvllm.agentspec`` without executing ``tinyvllm``.

    The real ``tinyvllm/__init__.py`` imports torch and transformers.
    Stage 0 must stay runnable on a laptop, so stub parent packages
    are registered before the submodules are imported. Production code
    keeps ordinary absolute imports.
    """

    package_root = REPO_ROOT / "tinyvllm"
    if "tinyvllm" not in sys.modules:
        parent = types.ModuleType("tinyvllm")
        parent.__path__ = [os.fspath(package_root)]
        sys.modules["tinyvllm"] = parent
    if "tinyvllm.agentspec" not in sys.modules:
        child = types.ModuleType("tinyvllm.agentspec")
        child.__path__ = [os.fspath(package_root / "agentspec")]
        sys.modules["tinyvllm.agentspec"] = child
    action = importlib.import_module("tinyvllm.agentspec.action")
    cost_model = importlib.import_module(
        "tinyvllm.agentspec.cost_model"
    )
    latent_adapter = importlib.import_module(
        "tinyvllm.agentspec.latent_adapter"
    )
    router = importlib.import_module("tinyvllm.agentspec.router")
    return action, cost_model, latent_adapter, router


ACTION, COST_MODEL, LATENT_ADAPTER, ROUTER = load_agentspec()


# Frozen scenario matrix. Values are declared operating points, not
# measurements. Drafter taxes name the representation they stand for:
# a discrete-code drafter is assumed cheapest because it emits code
# indices rather than argument tokens; a small text drafter is assumed
# to cost about the same as the actor step it shadows.
ACTOR_GPU_SECONDS = 0.080
TOOL_SECONDS = (0.2, 1.0, 5.0)
BASELINE_UTILIZATIONS = (0.0, 0.3, 0.6, 0.8, 0.9)
DRAFTER_TAXES = (
    ("discrete_code", 0.10),
    ("continuous_latent", 0.25),
    ("text_small_model", 1.00),
)
MATCH_PROBABILITIES = (0.55, 0.75, 0.90)
ROLLBACK_SECONDS = (0.0, 0.5)

# Published single-branch action match rate used as the reference
# point for the headline finding.
REFERENCE_MATCH_PROBABILITY = 0.55

BRANCH_CONFIDENCES = (0.55, 0.18, 0.09, 0.05)
BRANCH_COUNTS = (1, 2, 3, 4)


def _round(value):
    if value is None:
        return None
    return round(float(value), ROUNDING_DECIMALS)


def _build_reference_tool_contracts():
    build = ACTION.build_tool_contract
    return (
        build(
            tool_name="search",
            side_effect_class="read_only",
            rollback_seconds=0.0,
        ),
        build(
            tool_name="write_file",
            side_effect_class="sandboxable",
            rollback_seconds=0.5,
            sandbox_available=True,
        ),
        build(
            tool_name="send_payment",
            side_effect_class="irreversible",
            rollback_seconds=0.0,
        ),
    )


def build_matrix_rows():
    rows = []
    for tool_seconds in TOOL_SECONDS:
        for utilization in BASELINE_UTILIZATIONS:
            for drafter_kind, tax in DRAFTER_TAXES:
                for probability in MATCH_PROBABILITIES:
                    for rollback in ROLLBACK_SECONDS:
                        inputs = COST_MODEL.build_cost_inputs(
                            actor_gpu_seconds=ACTOR_GPU_SECONDS,
                            draft_gpu_tax=tax,
                            tool_seconds=tool_seconds,
                            baseline_utilization=utilization,
                            match_probability=probability,
                            rollback_seconds=rollback,
                        )
                        result = COST_MODEL.evaluate(inputs)
                        route = (
                            ROUTER.choose_action_speculation_route(
                                proposal_available=True,
                                eligible_candidate_count=1,
                                branch_count=1,
                                aggregate_match_probability=(
                                    probability
                                ),
                                break_even=result,
                            )
                        )
                        rows.append(
                            {
                                "drafter_kind": drafter_kind,
                                "draft_gpu_tax": _round(tax),
                                "tool_seconds": _round(tool_seconds),
                                "baseline_utilization": _round(
                                    utilization
                                ),
                                "match_probability": _round(
                                    probability
                                ),
                                "rollback_seconds": _round(rollback),
                                "stable": bool(result.stable),
                                "verdict": result.verdict,
                                "route": route.name,
                                "baseline_latency_seconds": _round(
                                    result.baseline_latency_seconds
                                ),
                                "speculative_latency_seconds": _round(
                                    result.speculative_latency_seconds
                                ),
                                "hit_latency_seconds": _round(
                                    result.hit_latency_seconds
                                ),
                                "speedup": _round(result.speedup),
                                "capacity_ratio": _round(
                                    result.capacity_ratio
                                ),
                                "wasted_gpu_fraction": _round(
                                    result.wasted_gpu_fraction
                                ),
                                "minimum_match_probability": _round(
                                    result.minimum_match_probability
                                ),
                                "critical_utilization": _round(
                                    result.critical_utilization
                                ),
                                "critical_draft_tax": _round(
                                    result.critical_draft_tax
                                ),
                            }
                        )
    return tuple(rows)


def build_branch_rows():
    """Show branch widening under a single latent draft pass."""

    capabilities = LATENT_ADAPTER.build_capabilities(
        source_type="latent_code_action_drafter",
        representation="discrete_code",
        requires_target_hidden=True,
        requires_compressed_kv=True,
        max_candidate_actions=len(BRANCH_CONFIDENCES),
        max_horizon_actions=1,
        emits_match_confidence=True,
    )
    candidates = []
    for index, confidence in enumerate(BRANCH_CONFIDENCES):
        signature = ACTION.build_action_signature(
            tool_name="search",
            arguments={"query": "candidate-%d" % index},
        )
        candidates.append(
            LATENT_ADAPTER.LatentActionCandidate(
                signature,
                confidence,
            )
        )
    proposal = LATENT_ADAPTER.LatentActionDraftProposal(
        1,
        0,
        tuple(candidates),
        "latent_code_action_drafter",
        0.008,
    )
    LATENT_ADAPTER.validate_proposal(capabilities, proposal)
    contracts = _build_reference_tool_contracts()
    eligible = LATENT_ADAPTER.eligible_candidates(
        proposal,
        contracts,
    )
    rows = []
    for branch_count in BRANCH_COUNTS:
        probability = LATENT_ADAPTER.aggregate_match_probability(
            eligible,
            branch_count,
        )
        inputs = COST_MODEL.build_cost_inputs(
            actor_gpu_seconds=ACTOR_GPU_SECONDS,
            draft_gpu_tax=0.10,
            tool_seconds=1.0,
            baseline_utilization=0.6,
            match_probability=probability,
            rollback_seconds=0.0,
        )
        result = COST_MODEL.evaluate(inputs)
        route = ROUTER.choose_action_speculation_route(
            proposal_available=True,
            eligible_candidate_count=len(eligible),
            branch_count=branch_count,
            aggregate_match_probability=probability,
            break_even=result,
        )
        rows.append(
            {
                "branch_count": branch_count,
                "aggregate_match_probability": _round(probability),
                "draft_gpu_tax": _round(0.10),
                "speculative_tool_calls": route.speculative_tool_calls,
                "verdict": result.verdict,
                "route": route.name,
                "speedup": _round(result.speedup),
                "critical_utilization": _round(
                    result.critical_utilization
                ),
            }
        )
    return tuple(rows)


def build_guard_rows():
    """Prove the safety guards dominate the profitability guard."""

    contracts = _build_reference_tool_contracts()
    irreversible = ACTION.build_action_signature(
        tool_name="send_payment",
        arguments={"amount": 100},
    )
    capabilities = LATENT_ADAPTER.build_capabilities(
        source_type="latent_code_action_drafter",
        representation="discrete_code",
        requires_target_hidden=True,
        requires_compressed_kv=False,
        max_candidate_actions=2,
        max_horizon_actions=1,
        emits_match_confidence=True,
    )
    proposal = LATENT_ADAPTER.LatentActionDraftProposal(
        7,
        3,
        (
            LATENT_ADAPTER.LatentActionCandidate(irreversible, 0.95),
        ),
        "latent_code_action_drafter",
        0.008,
    )
    LATENT_ADAPTER.validate_proposal(capabilities, proposal)
    eligible = LATENT_ADAPTER.eligible_candidates(
        proposal,
        contracts,
    )
    favourable = COST_MODEL.evaluate(
        COST_MODEL.build_cost_inputs(
            actor_gpu_seconds=ACTOR_GPU_SECONDS,
            draft_gpu_tax=0.10,
            tool_seconds=5.0,
            baseline_utilization=0.0,
            match_probability=0.95,
            rollback_seconds=0.0,
        )
    )
    side_effect_route = ROUTER.choose_action_speculation_route(
        proposal_available=True,
        eligible_candidate_count=len(eligible),
        branch_count=1,
        aggregate_match_probability=0.95,
        break_even=favourable,
    )
    unstable = COST_MODEL.evaluate(
        COST_MODEL.build_cost_inputs(
            actor_gpu_seconds=ACTOR_GPU_SECONDS,
            draft_gpu_tax=1.00,
            tool_seconds=5.0,
            baseline_utilization=0.9,
            match_probability=0.95,
            rollback_seconds=0.0,
        )
    )
    capacity_route = ROUTER.choose_action_speculation_route(
        proposal_available=True,
        eligible_candidate_count=1,
        branch_count=1,
        aggregate_match_probability=0.95,
        break_even=unstable,
    )
    return (
        {
            "guard": "side_effect_dominates_profitability",
            "eligible_candidate_count": len(eligible),
            "cost_model_verdict": favourable.verdict,
            "route": side_effect_route.name,
        },
        {
            "guard": "capacity_dominates_profitability",
            "speculative_utilization": _round(
                unstable.speculative_utilization
            ),
            "cost_model_verdict": unstable.verdict,
            "route": capacity_route.name,
        },
    )


def check_invariants(matrix_rows, branch_rows, guard_rows):
    """Return a list of invariant records with pass or fail status."""

    findings = []

    def record(name, passed, detail):
        findings.append(
            {
                "invariant": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    unstable_rows = [
        row for row in matrix_rows if not row["stable"]
    ]
    record(
        "unstable_rows_route_to_capacity_guard",
        all(
            row["route"] == "baseline_capacity_guard"
            for row in unstable_rows
        ),
        "unstable_rows=%d" % len(unstable_rows),
    )
    record(
        "unstable_rows_have_no_speedup",
        all(
            row["speedup"] is None for row in unstable_rows
        ),
        "unstable_rows=%d" % len(unstable_rows),
    )
    positive_rows = [
        row
        for row in matrix_rows
        if row["verdict"] == "net_positive"
    ]
    record(
        "net_positive_rows_are_speculative",
        all(
            row["route"] == "speculative_commit_on_match"
            for row in positive_rows
        ),
        "net_positive_rows=%d" % len(positive_rows),
    )
    record(
        "net_positive_rows_have_speedup_above_one",
        all(row["speedup"] > 1.0 for row in positive_rows),
        "net_positive_rows=%d" % len(positive_rows),
    )
    record(
        "minimum_match_probability_within_unit_interval",
        all(
            row["minimum_match_probability"] is None
            or 0.0 <= row["minimum_match_probability"] <= 1.0
            for row in matrix_rows
        ),
        "rows=%d" % len(matrix_rows),
    )
    record(
        "capacity_ratio_below_one_when_taxed",
        all(
            row["capacity_ratio"] < 1.0
            for row in matrix_rows
            if row["draft_gpu_tax"] > 0.0
        ),
        "rows=%d" % len(matrix_rows),
    )

    # Speedup must not increase when the draft tax increases, all
    # other declared inputs held fixed.
    monotonic = True
    grouped = {}
    for row in matrix_rows:
        key = (
            row["tool_seconds"],
            row["baseline_utilization"],
            row["match_probability"],
            row["rollback_seconds"],
        )
        grouped.setdefault(key, []).append(row)
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: item["draft_gpu_tax"])
        previous = None
        for row in ordered:
            current = row["speedup"]
            if current is None:
                current = 0.0
            if previous is not None and current > previous + 1e-9:
                monotonic = False
                break
            previous = current
    record(
        "speedup_non_increasing_in_draft_tax",
        monotonic,
        "groups=%d" % len(grouped),
    )

    # At zero utilization with zero rollback the model reduces to
    # hit < base iff draft_gpu_tax < 1, because the drafter is
    # scheduled first and consumes tau / (1 + tau) of a step that is
    # itself (1 + tau) times longer. A drafter as expensive as the
    # actor therefore finishes exactly when the actor would have, so
    # accuracy becomes irrelevant. Both directions are asserted.
    idle_rows = [
        row
        for row in matrix_rows
        if row["baseline_utilization"] == 0.0
        and row["rollback_seconds"] == 0.0
    ]
    cheap_idle_rows = [
        row for row in idle_rows if row["draft_gpu_tax"] < 1.0
    ]
    unit_tax_idle_rows = [
        row for row in idle_rows if row["draft_gpu_tax"] >= 1.0
    ]
    record(
        "idle_capacity_rows_profitable_below_unit_tax",
        all(
            row["verdict"] == "net_positive"
            for row in cheap_idle_rows
        ),
        "rows=%d" % len(cheap_idle_rows),
    )
    record(
        "unit_tax_drafter_has_no_headroom",
        bool(unit_tax_idle_rows)
        and all(
            row["verdict"] == "infeasible_no_match_benefit"
            and row["minimum_match_probability"] is None
            for row in unit_tax_idle_rows
        ),
        "rows=%d" % len(unit_tax_idle_rows),
    )

    # The saving on a hit is bounded by the actor GPU sojourn, not
    # by tool latency: overlapping one tool call can hide at most one
    # think. Relative speedup therefore decays once tool latency
    # dominates, which is the quantitative argument for speculating a
    # multi-action horizon rather than a single next action.
    saving_capped = True
    for row in matrix_rows:
        if row["hit_latency_seconds"] is None:
            continue
        sojourn = (
            row["baseline_latency_seconds"] - row["tool_seconds"]
        )
        saving = (
            row["baseline_latency_seconds"]
            - row["hit_latency_seconds"]
        )
        if saving > sojourn + 1e-9:
            saving_capped = False
            break
    record(
        "hit_saving_capped_by_actor_sojourn",
        saving_capped,
        "rows=%d" % len(matrix_rows),
    )

    decaying = True
    compared = 0
    by_key = {}
    for row in matrix_rows:
        key = (
            row["drafter_kind"],
            row["baseline_utilization"],
            row["match_probability"],
            row["rollback_seconds"],
        )
        by_key.setdefault(key, {})[row["tool_seconds"]] = row
    for key, bucket in by_key.items():
        near = bucket.get(1.0)
        far = bucket.get(5.0)
        if near is None or far is None:
            continue
        if near["speedup"] is None or far["speedup"] is None:
            continue
        if near["verdict"] != "net_positive":
            # Below break-even the ratio approaches one from the
            # wrong side as tool latency grows, so decay is only
            # asserted where speculation actually wins.
            continue
        compared += 1
        if far["speedup"] > near["speedup"] + 1e-9:
            decaying = False
            break
    record(
        "relative_speedup_decays_when_tool_dominates",
        decaying and compared > 0,
        "compared=%d" % compared,
    )

    branch_probabilities = [
        row["aggregate_match_probability"] for row in branch_rows
    ]
    record(
        "branch_widening_raises_match_probability",
        all(
            branch_probabilities[index]
            < branch_probabilities[index + 1] + 1e-12
            for index in range(len(branch_probabilities) - 1)
        ),
        "branch_rows=%d" % len(branch_rows),
    )
    record(
        "branch_widening_holds_draft_tax_constant",
        len({row["draft_gpu_tax"] for row in branch_rows}) == 1,
        "branch_rows=%d" % len(branch_rows),
    )
    record(
        "branch_widening_raises_tool_calls",
        all(
            row["speculative_tool_calls"] == row["branch_count"]
            for row in branch_rows
            if row["route"] == "speculative_commit_on_match"
        ),
        "branch_rows=%d" % len(branch_rows),
    )
    record(
        "side_effect_guard_dominates",
        guard_rows[0]["route"] == "baseline_side_effect_guard",
        guard_rows[0]["cost_model_verdict"],
    )
    record(
        "capacity_guard_dominates",
        guard_rows[1]["route"] == "baseline_capacity_guard",
        guard_rows[1]["cost_model_verdict"],
    )
    return findings


def build_headline(matrix_rows):
    """Summarise the reference operating point."""

    reference = [
        row
        for row in matrix_rows
        if row["match_probability"] == REFERENCE_MATCH_PROBABILITY
        and row["rollback_seconds"] == 0.0
        and row["tool_seconds"] == 1.0
    ]
    per_drafter = {}
    for row in reference:
        kind = row["drafter_kind"]
        bucket = per_drafter.setdefault(
            kind,
            {
                "draft_gpu_tax": row["draft_gpu_tax"],
                "profitable_utilizations": [],
                "critical_utilization": None,
            },
        )
        if row["verdict"] == "net_positive":
            bucket["profitable_utilizations"].append(
                row["baseline_utilization"]
            )
            if bucket["critical_utilization"] is None:
                bucket["critical_utilization"] = row[
                    "critical_utilization"
                ]
    return {
        "reference_match_probability": REFERENCE_MATCH_PROBABILITY,
        "reference_tool_seconds": 1.0,
        "reference_rollback_seconds": 0.0,
        "per_drafter": per_drafter,
    }


def build_report():
    matrix_rows = build_matrix_rows()
    branch_rows = build_branch_rows()
    guard_rows = build_guard_rows()
    findings = check_invariants(matrix_rows, branch_rows, guard_rows)
    failed = [item for item in findings if item["status"] != "PASS"]
    payload = {
        "gate": GATE_NAME,
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "FAIL" if failed else "PASS",
        "claim_boundary": (
            "analytic contract only; no measured speedup is claimed"
        ),
        "actor_gpu_seconds": _round(ACTOR_GPU_SECONDS),
        "matrix_row_count": len(matrix_rows),
        "headline": build_headline(matrix_rows),
        "invariants": findings,
        "branch_rows": list(branch_rows),
        "guard_rows": list(guard_rows),
        "matrix_rows": list(matrix_rows),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    payload["payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def summarise(report):
    lines = [
        "gate            %s" % report["gate"],
        "status          %s" % report["status"],
        "matrix rows     %d" % report["matrix_row_count"],
        "payload sha256  %s" % report["payload_sha256"],
        "claim boundary  %s" % report["claim_boundary"],
        "",
        "invariants:",
    ]
    for item in report["invariants"]:
        lines.append(
            "  %-4s %s (%s)"
            % (item["status"], item["invariant"], item["detail"])
        )
    lines.append("")
    lines.append(
        "reference point p=%s, tool=%ss, rollback=0s:"
        % (
            report["headline"]["reference_match_probability"],
            report["headline"]["reference_tool_seconds"],
        )
    )
    for kind, bucket in sorted(
        report["headline"]["per_drafter"].items()
    ):
        lines.append(
            "  %-18s tax=%-5s profitable_rho=%s critical_rho=%s"
            % (
                kind,
                bucket["draft_gpu_tax"],
                bucket["profitable_utilizations"],
                bucket["critical_utilization"],
            )
        )
    lines.append("")
    lines.append("branch widening at rho=0.6, tool=1.0s, tax=0.10:")
    for row in report["branch_rows"]:
        lines.append(
            "  b=%d p=%-8s calls=%d %-12s speedup=%s"
            % (
                row["branch_count"],
                row["aggregate_match_probability"],
                row["speculative_tool_calls"],
                row["verdict"],
                row["speedup"],
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 0 analytic gate for action speculation",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)
    report = build_report()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "agentspec_breakeven_report.json"
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.print_summary or args.output_dir is None:
        print(summarise(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
