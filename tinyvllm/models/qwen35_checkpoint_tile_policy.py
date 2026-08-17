from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
)
from tinyvllm.models.qwen35_checkpoint_tiles import (
    Qwen35CheckpointTilePlan,
    build_qwen35_checkpoint_tile_plan,
)


@dataclass(frozen=True)
class Qwen35CheckpointTileBudgetEvaluation:
    max_tile_bytes: int
    tile_count: int
    peak_tile_bytes: int


@dataclass(frozen=True)
class Qwen35CheckpointTileBudgetDecision:
    tile_plan: Qwen35CheckpointTilePlan
    selected_max_tile_bytes: int
    max_tile_count: int
    evaluations: tuple[Qwen35CheckpointTileBudgetEvaluation, ...]


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _candidate_budgets(max_tile_bytes: int) -> tuple[int, ...]:
    budgets = []
    value = 1
    while value <= max_tile_bytes:
        budgets.append(value)
        value *= 2
    if budgets[-1] != max_tile_bytes:
        budgets.append(max_tile_bytes)
    return tuple(budgets)


def select_qwen35_checkpoint_tile_budget(
    binding_plan: Qwen35CheckpointBindingPlan,
    *,
    max_tile_bytes: int,
    max_tile_count: int,
) -> Qwen35CheckpointTileBudgetDecision:
    if type(binding_plan) is not Qwen35CheckpointBindingPlan:
        raise ValueError(
            "binding_plan must be an exact Qwen35CheckpointBindingPlan"
        )
    byte_cap = _positive_integer(
        max_tile_bytes,
        "max_tile_bytes",
    )
    count_cap = _positive_integer(
        max_tile_count,
        "max_tile_count",
    )

    evaluations = []
    last_plan = None
    for budget in _candidate_budgets(byte_cap):
        try:
            plan = build_qwen35_checkpoint_tile_plan(
                binding_plan,
                max_tile_bytes=budget,
            )
        except ValueError as error:
            if "indivisible tile unit" not in str(error):
                raise
            continue
        last_plan = plan
        evaluation = Qwen35CheckpointTileBudgetEvaluation(
            max_tile_bytes=budget,
            tile_count=len(plan.tiles),
            peak_tile_bytes=plan.peak_tile_bytes,
        )
        evaluations.append(evaluation)
        if evaluation.tile_count <= count_cap:
            return Qwen35CheckpointTileBudgetDecision(
                tile_plan=plan,
                selected_max_tile_bytes=budget,
                max_tile_count=count_cap,
                evaluations=tuple(evaluations),
            )

    if last_plan is None:
        raise ValueError(
            "no feasible tile budget within max_tile_bytes "
            f"{byte_cap}"
        )
    raise ValueError(
        "Qwen3.5 tile policy cannot satisfy max_tile_count "
        f"{count_cap} within max_tile_bytes {byte_cap}; "
        f"final tile_count {len(last_plan.tiles)}"
    )
