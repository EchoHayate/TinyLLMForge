#!/usr/bin/env python3
"""Dependency-light tests for zero-temperature greedy fast-path policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "greedy_sampling_fast_path.py"
)
SPEC = importlib.util.spec_from_file_location(
    "greedy_sampling_fast_path_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
GreedySamplingFastPathStats = module.GreedySamplingFastPathStats
decide_greedy_sampling_fast_path = (
    module.decide_greedy_sampling_fast_path
)


def test_exact_batch_one_zero_temperature_is_eligible() -> None:
    decision = decide_greedy_sampling_fast_path(
        enabled=True,
        rank=0,
        temperatures=(0.0,),
        batch_kind=None,
        logits_shape=(1, 151_936),
    )

    assert decision.optimized is True
    assert decision.fallback_reason is None


def test_ineligible_cases_fail_closed_with_stable_reasons() -> None:
    cases = (
        (
            {
                "enabled": False,
                "rank": 0,
                "temperatures": (0.0,),
                "batch_kind": None,
                "logits_shape": (1, 151_936),
            },
            "disabled",
        ),
        (
            {
                "enabled": True,
                "rank": 1,
                "temperatures": (0.0,),
                "batch_kind": None,
                "logits_shape": (1, 151_936),
            },
            "non_root_rank",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": (),
                "batch_kind": None,
                "logits_shape": (0, 151_936),
            },
            "batch_size_unsupported",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": (0.0, 0.0),
                "batch_kind": None,
                "logits_shape": (2, 151_936),
            },
            "batch_size_unsupported",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": (0.0,),
                "batch_kind": "mixed",
                "logits_shape": (1, 151_936),
            },
            "mixed_batch_unsupported",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": (0.7,),
                "batch_kind": None,
                "logits_shape": (1, 151_936),
            },
            "nonzero_temperature",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": ("0",),
                "batch_kind": None,
                "logits_shape": (1, 151_936),
            },
            "temperature_invalid",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": (0.0,),
                "batch_kind": None,
                "logits_shape": (151_936,),
            },
            "logits_shape_unsupported",
        ),
        (
            {
                "enabled": True,
                "rank": 0,
                "temperatures": (0.0,),
                "batch_kind": None,
                "logits_shape": (1, 0),
            },
            "logits_shape_unsupported",
        ),
    )

    for kwargs, expected_reason in cases:
        decision = decide_greedy_sampling_fast_path(**kwargs)
        assert decision.optimized is False
        assert decision.fallback_reason == expected_reason


def test_invalid_control_values_raise() -> None:
    common = {
        "enabled": True,
        "rank": 0,
        "temperatures": (0.0,),
        "batch_kind": None,
        "logits_shape": (1, 151_936),
    }
    for name, value, message in (
        ("enabled", 1, "enabled must be a bool"),
        ("rank", True, "rank must be a non-negative integer"),
        (
            "batch_kind",
            1,
            "batch_kind must be a string or None",
        ),
        (
            "logits_shape",
            [1, 151_936],
            "logits_shape must be a tuple",
        ),
    ):
        kwargs = dict(common)
        kwargs[name] = value
        try:
            decide_greedy_sampling_fast_path(**kwargs)
        except ValueError as error:
            assert str(error) == message
        else:
            raise AssertionError(
                f"{name} validation did not fail"
            )


def test_stats_account_exact_avoided_work() -> None:
    stats = GreedySamplingFastPathStats()
    stats.record_fallback("disabled")
    stats.record_fallback("disabled")
    stats.record_optimized(batch_size=1)
    stats.record_optimized(batch_size=1)

    assert stats.summary() == {
        "eligible_steps": 2,
        "optimized_steps": 2,
        "avoided_temperature_h2d_bytes": 8,
        "avoided_softmax_calls": 2,
        "avoided_gumbel_rng_calls": 2,
        "avoided_stochastic_divisions": 4,
        "avoided_stochastic_argmax_calls": 2,
        "avoided_where_calls": 2,
        "fallback_counts": {"disabled": 2},
    }


def test_stats_reject_invalid_batch_size_and_reason() -> None:
    stats = GreedySamplingFastPathStats()
    for value in (True, 0, -1, 1.5):
        try:
            stats.record_optimized(value)
        except ValueError as error:
            assert str(error) == (
                "batch_size must be a positive integer"
            )
        else:
            raise AssertionError(
                "invalid batch size was accepted"
            )
    for value in ("", 1, None):
        try:
            stats.record_fallback(value)
        except ValueError as error:
            assert str(error) == (
                "fallback reason must be a non-empty string"
            )
        else:
            raise AssertionError(
                "invalid fallback reason was accepted"
            )


def main() -> None:
    test_exact_batch_one_zero_temperature_is_eligible()
    test_ineligible_cases_fail_closed_with_stable_reasons()
    test_invalid_control_values_raise()
    test_stats_account_exact_avoided_work()
    test_stats_reject_invalid_batch_size_and_reason()
    print("greedy sampling fast path tests passed")


if __name__ == "__main__":
    main()
