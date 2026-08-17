from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RouteName = Literal[
    "baseline_short_draft",
    "baseline_finished",
    "baseline_output_budget",
    "baseline_incompatible",
    "native_multi_token",
]


@dataclass(frozen=True)
class SpeculationRoute:
    name: RouteName
    draft_len: int
    native_compatible: bool
    fallback_reason: str | None = None


def choose_speculation_route(
    *,
    draft_len: int,
    finished: bool,
    remaining_output_budget: int,
    native_compatible: bool,
    compatibility_reason: str | None = None,
    allow_incompatible_fallback: bool = False,
) -> SpeculationRoute:
    draft_len = int(draft_len)
    remaining_output_budget = int(remaining_output_budget)
    if draft_len < 0:
        raise ValueError("draft_len must be >= 0")
    if finished:
        return SpeculationRoute(
            "baseline_finished",
            draft_len,
            bool(native_compatible),
        )
    if remaining_output_budget <= 0:
        return SpeculationRoute(
            "baseline_output_budget",
            draft_len,
            bool(native_compatible),
        )
    if draft_len <= 1:
        return SpeculationRoute(
            "baseline_short_draft",
            draft_len,
            bool(native_compatible),
        )
    if not native_compatible:
        reason = compatibility_reason or "native verifier incompatible"
        if not allow_incompatible_fallback:
            raise ValueError(reason)
        return SpeculationRoute(
            "baseline_incompatible",
            draft_len,
            False,
            reason,
        )
    return SpeculationRoute(
        "native_multi_token",
        draft_len,
        True,
    )


def route_to_dict(route: SpeculationRoute) -> dict[str, object]:
    return asdict(route)
