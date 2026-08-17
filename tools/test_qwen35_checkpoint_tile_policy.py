from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assignment_helper = _load_helper(
    "qwen35_checkpoint_assignment_policy_helper",
    "tools/test_qwen35_checkpoint_assignment.py",
)

from tinyvllm.models.qwen35_checkpoint_tile_policy import (
    Qwen35CheckpointTileBudgetDecision,
    Qwen35CheckpointTileBudgetEvaluation,
    select_qwen35_checkpoint_tile_budget,
)
from tinyvllm.models.qwen35_checkpoint_tiles import (
    build_qwen35_checkpoint_tile_plan,
)


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _snapshot(binding_plan):
    return {
        id(destination): destination.detach().clone()
        for destination in {
            id(binding.destination): binding.destination
            for binding in binding_plan.bindings
        }.values()
    }


def _assert_snapshot(binding_plan, snapshot):
    destinations = {
        id(binding.destination): binding.destination
        for binding in binding_plan.bindings
    }
    assert set(destinations) == set(snapshot)
    for object_id, destination in destinations.items():
        torch.testing.assert_close(destination, snapshot[object_id])


def test_selects_smallest_satisfying_power_of_two_budget():
    for world_size in (1, 2):
        _, _, _, binding_plan = assignment_helper._fixture(
            0,
            world_size,
        )
        snapshot = _snapshot(binding_plan)
        max_count = 40
        max_bytes = 256
        result = select_qwen35_checkpoint_tile_budget(
            binding_plan,
            max_tile_bytes=max_bytes,
            max_tile_count=max_count,
        )

        candidates = []
        budget = 1
        while budget <= max_bytes:
            try:
                plan = build_qwen35_checkpoint_tile_plan(
                    binding_plan,
                    max_tile_bytes=budget,
                )
            except ValueError:
                pass
            else:
                candidates.append((budget, plan))
            budget *= 2
        expected_budget, expected_plan = next(
            (budget, plan)
            for budget, plan in candidates
            if len(plan.tiles) <= max_count
        )

        assert type(result) is Qwen35CheckpointTileBudgetDecision
        assert result.selected_max_tile_bytes == expected_budget
        assert len(result.tile_plan.tiles) == len(expected_plan.tiles)
        assert len(result.tile_plan.tiles) <= max_count
        assert result.max_tile_count == max_count
        assert tuple(
            evaluation.max_tile_bytes
            for evaluation in result.evaluations
        ) == tuple(
            budget
            for budget, _ in candidates
            if budget <= expected_budget
        )
        assert all(
            type(evaluation)
            is Qwen35CheckpointTileBudgetEvaluation
            for evaluation in result.evaluations
        )
        assert result.evaluations[-1].tile_count == len(
            result.tile_plan.tiles
        )
        _assert_snapshot(binding_plan, snapshot)


def test_uses_exact_non_power_of_two_final_cap():
    _, _, _, binding_plan = assignment_helper._fixture(0, 1)
    result = select_qwen35_checkpoint_tile_budget(
        binding_plan,
        max_tile_bytes=96,
        max_tile_count=41,
    )
    assert result.selected_max_tile_bytes == 96
    assert len(result.tile_plan.tiles) == 41
    assert tuple(
        evaluation.max_tile_bytes
        for evaluation in result.evaluations
    ) == (32, 64, 96)


def test_policy_validation_failure_and_determinism():
    _, _, _, binding_plan = assignment_helper._fixture(0, 1)
    cases = (
        (object(), 256, 40, "exact Qwen35CheckpointBindingPlan"),
        (binding_plan, True, 40, "max_tile_bytes"),
        (binding_plan, 0, 40, "max_tile_bytes"),
        (binding_plan, 256, True, "max_tile_count"),
        (binding_plan, 256, 0, "max_tile_count"),
        (binding_plan, 23, 1000, "no feasible tile budget"),
        (binding_plan, 64, 40, "cannot satisfy max_tile_count"),
    )
    for plan, max_bytes, max_count, message in cases:
        _expect_error(
            lambda plan=plan,
            max_bytes=max_bytes,
            max_count=max_count: (
                select_qwen35_checkpoint_tile_budget(
                    plan,
                    max_tile_bytes=max_bytes,
                    max_tile_count=max_count,
                )
            ),
            message,
        )

    first = select_qwen35_checkpoint_tile_budget(
        binding_plan,
        max_tile_bytes=256,
        max_tile_count=40,
    )
    second = select_qwen35_checkpoint_tile_budget(
        binding_plan,
        max_tile_bytes=256,
        max_tile_count=40,
    )
    assert first.selected_max_tile_bytes == (
        second.selected_max_tile_bytes
    )
    assert first.evaluations == second.evaluations
    assert tuple(
        (
            tile.binding_index,
            tile.source_slices,
            tile.destination_slices,
        )
        for tile in first.tile_plan.tiles
    ) == tuple(
        (
            tile.binding_index,
            tile.source_slices,
            tile.destination_slices,
        )
        for tile in second.tile_plan.tiles
    )


def main():
    test_selects_smallest_satisfying_power_of_two_budget()
    test_uses_exact_non_power_of_two_final_cap()
    test_policy_validation_failure_and_determinism()
    print("qwen35 checkpoint tile policy tests passed (3 tests)")


if __name__ == "__main__":
    main()
