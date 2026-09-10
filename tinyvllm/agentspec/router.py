"""Fail-closed route selection for action-level speculation.

Semantics mirror ``tinyvllm.speculative.router``: the route set is
fixed, every non-speculative route is named, and an unknown or unsafe
condition never silently degrades into speculation.

Guard order is safety first, then capacity, then profitability:

1. no proposal available;
2. no candidate whose tool is speculation eligible;
3. shared engine would become unstable at the declared draft tax;
4. the analytic model says the operating point is net negative;
5. otherwise speculate under commit-on-match.

Guard 3 exists because an action drafter consumes GPU capacity even
when the actor is waiting on a tool. On a shared engine that capacity
is not free, and a drafter can move ``rho * (1 + tau)`` past one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from tinyvllm.agentspec.cost_model import SpeculationBreakEven


ActionRouteName = Literal[
    "baseline_no_proposal",
    "baseline_side_effect_guard",
    "baseline_capacity_guard",
    "baseline_unprofitable",
    "speculative_commit_on_match",
]


@dataclass(frozen=True)
class ActionSpeculationRoute:
    name: ActionRouteName
    branch_count: int
    aggregate_match_probability: float
    speculative_tool_calls: int
    fallback_reason: str | None = None


def choose_action_speculation_route(
    *,
    proposal_available: bool,
    eligible_candidate_count: int,
    branch_count: int,
    aggregate_match_probability: float,
    break_even: SpeculationBreakEven,
    allow_unprofitable: bool = False,
) -> ActionSpeculationRoute:
    if not isinstance(proposal_available, bool):
        raise ValueError("proposal_available must be a bool")
    if not isinstance(break_even, SpeculationBreakEven):
        raise ValueError("break_even must be SpeculationBreakEven")
    if isinstance(branch_count, bool) or not isinstance(
        branch_count,
        int,
    ):
        raise ValueError("branch_count must be an integer")
    if isinstance(eligible_candidate_count, bool) or not isinstance(
        eligible_candidate_count,
        int,
    ):
        raise ValueError("eligible_candidate_count must be an integer")
    if branch_count < 0 or eligible_candidate_count < 0:
        raise ValueError("counts must be >= 0")
    probability = float(aggregate_match_probability)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            "aggregate_match_probability must be within [0, 1]"
        )
    if not proposal_available or branch_count == 0:
        return ActionSpeculationRoute(
            "baseline_no_proposal",
            0,
            probability,
            0,
            "no usable action proposal",
        )
    if eligible_candidate_count == 0:
        return ActionSpeculationRoute(
            "baseline_side_effect_guard",
            0,
            probability,
            0,
            "no candidate tool is speculation eligible",
        )
    effective_branches = min(branch_count, eligible_candidate_count)
    if not break_even.stable:
        return ActionSpeculationRoute(
            "baseline_capacity_guard",
            0,
            probability,
            0,
            "draft tax would drive shared utilization to one",
        )
    if break_even.verdict != "net_positive":
        if not allow_unprofitable:
            return ActionSpeculationRoute(
                "baseline_unprofitable",
                0,
                probability,
                0,
                f"cost model verdict {break_even.verdict}",
            )
        return ActionSpeculationRoute(
            "speculative_commit_on_match",
            effective_branches,
            probability,
            effective_branches,
            "explicitly allowed despite net-negative model",
        )
    return ActionSpeculationRoute(
        "speculative_commit_on_match",
        effective_branches,
        probability,
        effective_branches,
    )


def route_to_dict(route: ActionSpeculationRoute) -> dict:
    return asdict(route)
