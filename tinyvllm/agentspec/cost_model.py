"""Analytic break-even model for action-level speculation.

Published action-level speculation results are reported on an
otherwise idle actor: the drafter runs while the actor waits for a
tool, so its GPU cost is treated as free. That assumption does not
hold on a shared serving engine, where the tool wait is not GPU idle
time -- the GPU is serving another request.

This module makes the shared-engine cost explicit and produces a
falsifiable break-even boundary. It is an analytic model over
declared inputs. It measures nothing and must never be cited as
evidence of a real speedup.

Model
-----

One agent step is modelled over two resources.

GPU (shared, capacity limited), as an M/M/1 queue with mean sojourn
time ``W = D / (1 - rho)``:

- ``D``    actor GPU service demand to emit one action, seconds;
- ``tau``  draft GPU tax, ``G_draft / D``, dimensionless;
- ``rho``  baseline GPU utilization without speculation.

Tool / environment (modelled as pure delay with ample parallelism):

- ``T``    tool latency, seconds.

Baseline, strictly sequential ``think -> act``::

    W_base = D / (1 - rho)
    T_base = W_base + T

Speculative, commit-on-match. The drafter always runs, so per-request
GPU demand becomes ``D * (1 + tau)`` and utilization rises to
``rho * (1 + tau)`` at unchanged arrival rate::

    rho_spec = rho * (1 + tau)
    W_spec   = D * (1 + tau) / (1 - rho_spec)
    t_draft  = alpha * W_spec
    T_hit    = max(W_spec, t_draft + T)
    T_miss   = W_spec + T + R
    T_spec   = p * T_hit + (1 - p) * T_miss

with ``p`` the action exact-match probability, ``R`` the rollback cost
on mismatch, and ``alpha`` the fraction of the step elapsed before the
speculated action is available. ``alpha`` defaults to
``tau / (1 + tau)``, i.e. the drafter is scheduled first and its share
of step GPU work determines when the speculative tool call launches.

Three consequences are worth stating separately because they are
absent from the action-level speculation literature this design
builds on:

1. ``rho * (1 + tau) < 1`` is a hard stability constraint. A drafter
   can push a stable engine into overload, which is a capacity
   failure, not a latency regression.
2. Peak capacity falls by ``1 / (1 + tau)`` whether or not proposals
   are accepted, because the draft always runs.
3. There exists a critical utilization above which speculation is net
   negative even at a fixed, good match probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


Verdict = Literal[
    "unstable_capacity",
    "infeasible_no_match_benefit",
    "net_negative",
    "net_positive",
]

_BISECTION_TOLERANCE = 1e-9
_BISECTION_MAX_STEPS = 200
_SCAN_POINTS = 512


@dataclass(frozen=True)
class SharedEngineCostInputs:
    actor_gpu_seconds: float
    draft_gpu_tax: float
    tool_seconds: float
    baseline_utilization: float
    match_probability: float
    rollback_seconds: float
    draft_ready_fraction: float | None = None

    @property
    def effective_draft_ready_fraction(self) -> float:
        if self.draft_ready_fraction is not None:
            return float(self.draft_ready_fraction)
        tax = self.draft_gpu_tax
        if tax <= 0.0:
            return 0.0
        return tax / (1.0 + tax)


@dataclass(frozen=True)
class SpeculationBreakEven:
    inputs: SharedEngineCostInputs
    stable: bool
    speculative_utilization: float
    stability_tax_bound: float
    baseline_latency_seconds: float
    speculative_latency_seconds: float | None
    hit_latency_seconds: float | None
    miss_latency_seconds: float | None
    speedup: float | None
    capacity_ratio: float
    wasted_gpu_fraction: float
    minimum_match_probability: float | None
    critical_utilization: float | None
    critical_draft_tax: float | None
    verdict: Verdict


def _require_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _require_positive(value: object, name: str) -> float:
    normalized = _require_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return normalized


def _require_non_negative(value: object, name: str) -> float:
    normalized = _require_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


def _require_unit_interval(value: object, name: str) -> float:
    normalized = _require_real(value, name)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return normalized


def _require_half_open_unit(value: object, name: str) -> float:
    normalized = _require_real(value, name)
    if normalized < 0.0 or normalized >= 1.0:
        raise ValueError(f"{name} must be within [0, 1)")
    return normalized


def build_cost_inputs(
    *,
    actor_gpu_seconds: object,
    draft_gpu_tax: object,
    tool_seconds: object,
    baseline_utilization: object,
    match_probability: object,
    rollback_seconds: object,
    draft_ready_fraction: object = None,
) -> SharedEngineCostInputs:
    ready_fraction = None
    if draft_ready_fraction is not None:
        ready_fraction = _require_unit_interval(
            draft_ready_fraction,
            "draft_ready_fraction",
        )
    return SharedEngineCostInputs(
        _require_positive(actor_gpu_seconds, "actor_gpu_seconds"),
        _require_non_negative(draft_gpu_tax, "draft_gpu_tax"),
        _require_non_negative(tool_seconds, "tool_seconds"),
        _require_half_open_unit(
            baseline_utilization,
            "baseline_utilization",
        ),
        _require_unit_interval(
            match_probability,
            "match_probability",
        ),
        _require_non_negative(rollback_seconds, "rollback_seconds"),
        ready_fraction,
    )


def _replace(
    inputs: SharedEngineCostInputs,
    **overrides: float,
) -> SharedEngineCostInputs:
    payload = {
        "actor_gpu_seconds": inputs.actor_gpu_seconds,
        "draft_gpu_tax": inputs.draft_gpu_tax,
        "tool_seconds": inputs.tool_seconds,
        "baseline_utilization": inputs.baseline_utilization,
        "match_probability": inputs.match_probability,
        "rollback_seconds": inputs.rollback_seconds,
        "draft_ready_fraction": inputs.draft_ready_fraction,
    }
    payload.update(overrides)
    return SharedEngineCostInputs(**payload)


def stability_tax_bound(baseline_utilization: float) -> float:
    """Largest draft tax that keeps the shared engine stable."""

    rho = _require_half_open_unit(
        baseline_utilization,
        "baseline_utilization",
    )
    if rho == 0.0:
        return math.inf
    return (1.0 - rho) / rho


def baseline_latency(inputs: SharedEngineCostInputs) -> float:
    demand = inputs.actor_gpu_seconds
    rho = inputs.baseline_utilization
    return demand / (1.0 - rho) + inputs.tool_seconds


def _speculative_terms(inputs: SharedEngineCostInputs):
    """Return ``(hit, miss, expected)`` or ``None`` when unstable."""

    tax = inputs.draft_gpu_tax
    rho_spec = inputs.baseline_utilization * (1.0 + tax)
    if rho_spec >= 1.0:
        return None
    sojourn = (
        inputs.actor_gpu_seconds * (1.0 + tax) / (1.0 - rho_spec)
    )
    draft_ready = inputs.effective_draft_ready_fraction * sojourn
    hit = max(sojourn, draft_ready + inputs.tool_seconds)
    miss = sojourn + inputs.tool_seconds + inputs.rollback_seconds
    probability = inputs.match_probability
    expected = probability * hit + (1.0 - probability) * miss
    return hit, miss, expected


def _is_profitable(inputs: SharedEngineCostInputs) -> bool:
    terms = _speculative_terms(inputs)
    if terms is None:
        return False
    _, _, expected = terms
    return expected < baseline_latency(inputs)


def _bisect_profitable_boundary(
    inputs: SharedEngineCostInputs,
    field: str,
    lower: float,
    upper: float,
) -> float | None:
    """Find the boundary between profitable and unprofitable.

    ``lower`` must be profitable and ``upper`` must not be. The
    returned value is the largest scanned value that is still
    profitable, resolved to ``_BISECTION_TOLERANCE``.
    """

    low = lower
    high = upper
    for _ in range(_BISECTION_MAX_STEPS):
        if high - low <= _BISECTION_TOLERANCE:
            break
        middle = 0.5 * (low + high)
        if _is_profitable(_replace(inputs, **{field: middle})):
            low = middle
        else:
            high = middle
    return low


def _scan_boundary(
    inputs: SharedEngineCostInputs,
    field: str,
    domain_upper: float,
) -> float | None:
    """Locate the first profitable-to-unprofitable crossing."""

    if not math.isfinite(domain_upper) or domain_upper <= 0.0:
        return None
    start = _replace(inputs, **{field: 0.0})
    if not _is_profitable(start):
        return None
    previous = 0.0
    for index in range(1, _SCAN_POINTS + 1):
        candidate = domain_upper * index / (_SCAN_POINTS + 1)
        if _is_profitable(_replace(inputs, **{field: candidate})):
            previous = candidate
            continue
        return _bisect_profitable_boundary(
            inputs,
            field,
            previous,
            candidate,
        )
    return None


def evaluate(inputs: SharedEngineCostInputs) -> SpeculationBreakEven:
    """Evaluate one declared operating point.

    ``critical_utilization`` and ``critical_draft_tax`` are reported
    as ``None`` when the whole admissible domain is profitable or when
    the operating point is not profitable at the origin of that
    domain.
    """

    if not isinstance(inputs, SharedEngineCostInputs):
        raise ValueError("inputs must be SharedEngineCostInputs")
    tax = inputs.draft_gpu_tax
    rho_spec = inputs.baseline_utilization * (1.0 + tax)
    bound = stability_tax_bound(inputs.baseline_utilization)
    base = baseline_latency(inputs)
    capacity_ratio = 1.0 / (1.0 + tax)
    wasted = (1.0 - inputs.match_probability) * tax / (1.0 + tax)
    terms = _speculative_terms(inputs)
    if terms is None:
        return SpeculationBreakEven(
            inputs,
            False,
            rho_spec,
            bound,
            base,
            None,
            None,
            None,
            None,
            capacity_ratio,
            wasted,
            None,
            None,
            None,
            "unstable_capacity",
        )
    hit, miss, expected = terms
    if hit >= base:
        return SpeculationBreakEven(
            inputs,
            True,
            rho_spec,
            bound,
            base,
            expected,
            hit,
            miss,
            base / expected,
            capacity_ratio,
            wasted,
            None,
            None,
            None,
            "infeasible_no_match_benefit",
        )
    if miss <= base:
        minimum_probability = 0.0
    else:
        minimum_probability = (miss - base) / (miss - hit)
    profitable = expected < base
    verdict = "net_positive" if profitable else "net_negative"
    critical_rho = None
    critical_tax = None
    if profitable:
        critical_rho = _scan_boundary(
            inputs,
            "baseline_utilization",
            1.0 - _BISECTION_TOLERANCE,
        )
        critical_tax = _scan_boundary(
            inputs,
            "draft_gpu_tax",
            bound if math.isfinite(bound) else 64.0,
        )
    return SpeculationBreakEven(
        inputs,
        True,
        rho_spec,
        bound,
        base,
        expected,
        hit,
        miss,
        base / expected,
        capacity_ratio,
        wasted,
        minimum_probability,
        critical_rho,
        critical_tax,
        verdict,
    )


def break_even_to_dict(result: SpeculationBreakEven) -> dict:
    payload = asdict(result)
    payload["inputs"]["effective_draft_ready_fraction"] = (
        result.inputs.effective_draft_ready_fraction
    )
    if not math.isfinite(result.stability_tax_bound):
        payload["stability_tax_bound"] = None
    return payload
