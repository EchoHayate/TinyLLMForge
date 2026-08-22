"""Eligibility and accounting for zero-temperature greedy sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Optional


@dataclass(frozen=True)
class GreedySamplingFastPathDecision:
    optimized: bool
    fallback_reason: Optional[str]


@dataclass
class GreedySamplingFastPathStats:
    eligible_steps: int = 0
    optimized_steps: int = 0
    avoided_temperature_h2d_bytes: int = 0
    avoided_softmax_calls: int = 0
    avoided_gumbel_rng_calls: int = 0
    avoided_stochastic_divisions: int = 0
    avoided_stochastic_argmax_calls: int = 0
    avoided_where_calls: int = 0
    fallback_counts: dict[str, int] = field(default_factory=dict)

    def record_optimized(self, batch_size: int) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a positive integer"
            )
        self.eligible_steps += 1
        self.optimized_steps += 1
        self.avoided_temperature_h2d_bytes += 4 * batch_size
        self.avoided_softmax_calls += 1
        self.avoided_gumbel_rng_calls += 1
        self.avoided_stochastic_divisions += 2
        self.avoided_stochastic_argmax_calls += 1
        self.avoided_where_calls += 1

    def record_fallback(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                "fallback reason must be a non-empty string"
            )
        self.fallback_counts[reason] = (
            self.fallback_counts.get(reason, 0) + 1
        )

    def summary(self) -> dict[str, object]:
        return {
            "eligible_steps": self.eligible_steps,
            "optimized_steps": self.optimized_steps,
            "avoided_temperature_h2d_bytes":
                self.avoided_temperature_h2d_bytes,
            "avoided_softmax_calls": self.avoided_softmax_calls,
            "avoided_gumbel_rng_calls":
                self.avoided_gumbel_rng_calls,
            "avoided_stochastic_divisions":
                self.avoided_stochastic_divisions,
            "avoided_stochastic_argmax_calls":
                self.avoided_stochastic_argmax_calls,
            "avoided_where_calls": self.avoided_where_calls,
            "fallback_counts": dict(self.fallback_counts),
        }


def decide_greedy_sampling_fast_path(
    *,
    enabled: bool,
    rank: int,
    temperatures: tuple[object, ...],
    batch_kind: Optional[str],
    logits_shape: tuple[int, ...],
) -> GreedySamplingFastPathDecision:
    """Return whether the exact batch-1 greedy shortcut is safe."""

    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a bool")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
    ):
        raise ValueError("rank must be a non-negative integer")
    if batch_kind is not None and not isinstance(batch_kind, str):
        raise ValueError("batch_kind must be a string or None")
    if not isinstance(logits_shape, tuple):
        raise ValueError("logits_shape must be a tuple")

    if not enabled:
        return GreedySamplingFastPathDecision(False, "disabled")
    if rank != 0:
        return GreedySamplingFastPathDecision(
            False,
            "non_root_rank",
        )
    if len(temperatures) != 1:
        return GreedySamplingFastPathDecision(
            False,
            "batch_size_unsupported",
        )
    if batch_kind == "mixed":
        return GreedySamplingFastPathDecision(
            False,
            "mixed_batch_unsupported",
        )

    temperature = temperatures[0]
    if isinstance(temperature, bool) or not isinstance(
        temperature,
        Real,
    ):
        return GreedySamplingFastPathDecision(
            False,
            "temperature_invalid",
        )
    if temperature != 0.0:
        return GreedySamplingFastPathDecision(
            False,
            "nonzero_temperature",
        )
    if (
        len(logits_shape) != 2
        or logits_shape[0] != 1
        or isinstance(logits_shape[1], bool)
        or not isinstance(logits_shape[1], int)
        or logits_shape[1] <= 0
    ):
        return GreedySamplingFastPathDecision(
            False,
            "logits_shape_unsupported",
        )
    return GreedySamplingFastPathDecision(True, None)
