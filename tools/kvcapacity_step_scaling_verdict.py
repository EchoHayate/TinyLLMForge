#!/usr/bin/env python3
"""GATE A verdict: does the measured decode step obey the Stage 0 model?

Stage 0 of the latent KV capacity line assumes::

    step_ms(L, B) = c0 + c1 * L * B                              (model M1)

Every capacity number in the Stage 0 artifact depends on that form, but the two
constants were fit from batch-1 data only, so the batch term has never been
observed. This tool consumes the GATE A measurement and decides whether the
assumption survives.

It deliberately fits two less convenient models as well::

    step_ms(L, B) = c0 + a * B + c1 * L * B                       (model M2)
    step_ms(L, B) = c0 + c1 * L * B + c2 * (L * B)^2              (model M3)

M2 exists because there is a concrete physical reason to expect it. A decode step
does two different kinds of work. Reading the KV cache scales with the number of
resident tokens, `L * B`, which is M1's only term. But the weight-stationary
matrix multiplies, the sampler, and the per-sequence bookkeeping scale with `B`
alone, independent of `L`. M1 has nowhere to put that work, so it is forced to
smear it into either the constant or the KV slope.

The distinction is not academic, and it does not cut in the flattering direction.
If `a` is materially positive, then M1 understates the cost of a large batch, the
Stage 0 `N_demand` ceiling is optimistic, and the capacity gains attributed to KV
compression are partly an artifact of the fit. The previous research line in this
repository was closed because a favourable ratio turned out to be a property of
its denominator. Fitting M2 is how that failure mode is checked for here rather
than discovered later.

M3 exists because the two obvious tests for M1 are both blind to curvature. A
coefficient of determination is a poor curvature detector: a quadratic sampled
over a monotone range still fits a straight line with an R^2 above 0.99, so the
threshold below can be satisfied by data that plainly violates the model. The
equal-product check is blind for a stronger reason. Any function of `L * B`
alone, quadratic included, makes equal-product cells cost the same, so that check
constrains the *shape* dependence and says nothing about the *form*. If the step
is superlinear in resident tokens, the Stage 0 capacity ceilings are again
optimistic, and neither of the first two checks would notice. M3 is the check
that would.

The tool reports the same quantities for all three models and lets the
pre-registered thresholds decide. It has no preference for the outcome.
"""

import argparse
import hashlib
import json
import os

# Pre-registered before looking at any measurement.
MIN_R_SQUARED = 0.98
MAX_COLLISION_SPREAD = 0.10
MAX_BATCH_TERM_SHARE = 0.20
MAX_CURVATURE_SHARE = 0.10
MAX_DRIFT_DEVIATION = 0.05
# Above this, batch 1 and batch 2 are not the same execution path and must not be
# fitted together.
MAX_REGIME_STEP_RATIO = 1.30
# A cell whose spread dwarfs its own median contains something other than steady
# decoding, even when the median survives.
MAX_DISPERSION_RATIO = 0.25

# Frozen Stage 0 constants, fit from batch-1 data in the erratum artifact.
STAGE0_C0_MS = 13.05
STAGE0_C1_US_PER_TOKEN = 0.151


def _solve(matrix, vector):
    """Solve a small dense linear system by Gaussian elimination.

    Written out rather than imported so this tool stays dependency-free and can
    run in the same environment as the Stage 0 gate.
    """
    size = len(vector)
    rows = [list(matrix[index]) + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-15:
            raise ValueError("design matrix is singular; the grid cannot identify this model")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for other in range(size):
            if other == column:
                continue
            factor = rows[other][column]
            if factor:
                rows[other] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[other], rows[column])
                ]
    return [rows[index][size] for index in range(size)]


def least_squares(designs, observations):
    """Ordinary least squares via the normal equations."""
    if len(designs) != len(observations):
        raise ValueError("design and observation counts differ")
    width = len(designs[0])
    if len(designs) < width:
        raise ValueError("not enough measured cells to identify the model")
    gram = [
        [sum(row[i] * row[j] for row in designs) for j in range(width)]
        for i in range(width)
    ]
    moment = [sum(row[i] * value for row, value in zip(designs, observations))
              for i in range(width)]
    return _solve(gram, moment)


def r_squared(designs, observations, coefficients):
    predictions = [
        sum(coefficient * value for coefficient, value in zip(coefficients, row))
        for row in designs
    ]
    mean = sum(observations) / len(observations)
    residual = sum((actual - predicted) ** 2
                   for actual, predicted in zip(observations, predictions))
    total = sum((actual - mean) ** 2 for actual in observations)
    if total <= 0.0:
        return 0.0, predictions, residual
    return 1.0 - residual / total, predictions, residual


# The grid as first registered. It is retained because amending a pre-registered
# plan after seeing data is exactly the move that invalidates a result, so the
# original must stay visible next to the reason it changed.
#
# It is unrunnable on the target model. Qwen3-8B declares
# max_position_embeddings = 40960, so the engine clamps max_model_len to 40960 and
# rejects any prompt beyond it. The 65536 and 131072 rows could never have been
# measured without RoPE scaling, which would change the model rather than measure
# it. This was a planning error in Stage 0, which swept contexts up to 131072 for a
# model that cannot reach them.
PREREGISTERED_CELLS_V1 = (
    (16384, 1), (16384, 2), (16384, 4), (16384, 8), (16384, 16),
    (32768, 1), (32768, 2), (32768, 4), (32768, 8),
    (65536, 1), (65536, 2), (65536, 4),
    (131072, 1), (131072, 2),
)

# The amended grid. GATE A asks only whether the step is affine in L * B, and the
# V1 run answered that at four separate equal-product groups whose members agree
# within 5.3%: at fixed L * B, shape does not matter. Resident token count can
# therefore be extended through batch instead of through context, which keeps the
# whole L * B range that Stage 0 depends on, up to 262144, inside the positional
# limit the model actually has.
#
# Two properties of this amendment are worth stating plainly, because the
# alternative reading is that the grid was moved to obtain a nicer answer. The
# range of the quantity being modelled is unchanged, and the checks had already
# passed on the V1 cells that were measurable, so nothing here rescues a failure.
PREREGISTERED_CELLS = (
    (8192, 1), (8192, 2), (8192, 4), (8192, 8), (8192, 16), (8192, 32),
    (16384, 1), (16384, 2), (16384, 4), (16384, 8), (16384, 16),
    (32768, 1), (32768, 2), (32768, 4), (32768, 8),
    # 40448, not 40960: a request needs room for prompt plus generated tokens, and
    # asking for the full positional limit leaves none.
    (40448, 1), (40448, 2), (40448, 4),
)


def merge_payloads(payloads):
    """Combine per-context worker artifacts into one measurement.

    One artifact per context length is not a convenience. The engine initialises
    a `torch.distributed` process group on construction and refuses to do it
    twice in one process, so a single process cannot build an engine per context.
    Each context length therefore runs in its own process, and the grid is
    reassembled here.

    Pre-registration is re-derived from the union of cells attempted across all
    artifacts, not copied from any single one, since no individual per-context run
    covers the pre-registered grid on its own.
    """
    if not payloads:
        raise ValueError("no payload was provided")
    rows = []
    engines = []
    attempted = set()
    measured = set()
    seen = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            key = (row.get("context_length"), row.get("batch"))
            if key in seen:
                raise ValueError(f"cell {key} appears in more than one artifact")
            seen.add(key)
            attempted.add(key)
            if row.get("measured") and row.get("step"):
                measured.add(key)
            rows.append(row)
        engines.extend(payload.get("engines", []))
    # Coverage is judged on cells that produced a number, not on cells that were
    # attempted. The first full run made the difference matter: five of fourteen
    # cells failed on a positional limit, the remaining nine fit well, and the gate
    # returned PASS while a third of the grid had silently disappeared.
    target = set(PREREGISTERED_CELLS)
    merged = {
        "rows": rows,
        "engines": engines,
        "payload_sha256": ",".join(
            str(payload.get("payload_sha256")) for payload in payloads
        ),
        "grid_is_preregistered": measured == target,
        "grid_spec": ";".join(
            f"{context}:{batch}" for context, batch in sorted(measured)
        ),
        "attempted_cells": sorted(attempted),
        "measured_cells": sorted(measured),
        "missing_preregistered_cells": sorted(target - measured),
        "attempted_but_unmeasured_cells": sorted(attempted - measured),
        "extra_cells": sorted(measured - target),
        "source_artifact_count": len(payloads),
    }
    return merged


def extract_points(payload):
    """Pull the measured cells out of a worker artifact.

    Cells the worker refused to measure are skipped. Cells whose prefill token
    accounting did not match the expectation are also skipped, because a
    shortfall there means prefix caching merged KV across sequences and the batch
    is not what it claims to be.
    """
    points = []
    rejected = []
    for row in payload.get("rows", []):
        context_length = row.get("context_length")
        batch = row.get("batch")
        if not row.get("measured") or not row.get("step"):
            rejected.append(
                {
                    "context_length": context_length,
                    "batch": batch,
                    "reason": row.get("skipped_reason") or "not measured",
                }
            )
            continue
        if row.get("prefill_tokens_match") is False:
            rejected.append(
                {
                    "context_length": context_length,
                    "batch": batch,
                    "reason": (
                        "prefill token accounting mismatch, so KV may be shared "
                        f"across sequences: {row.get('prefill_tokens_total')} "
                        f"observed against {row.get('prefill_tokens_expected')} expected"
                    ),
                }
            )
            continue
        points.append(
            {
                "context_length": int(context_length),
                "batch": int(batch),
                "kv_tokens": int(context_length) * int(batch),
                "step_ms": float(row["step"]["median_ms"]),
                "step_stdev_ms": float(row["step"].get("stdev_ms") or 0.0),
                "sample_count": int(row["step"].get("count") or 0),
                "dispersion_ratio": (
                    float(row["step"]["stdev_ms"]) / float(row["step"]["median_ms"])
                    if row["step"].get("stdev_ms") is not None
                    and float(row["step"]["median_ms"]) > 0
                    else None
                ),
                "drift_ratio": (
                    float(row["drift"]["ratio"]) if row.get("drift") else None
                ),
            }
        )
    return points, rejected


def regime_discontinuity(points):
    """Compare batch 1 with batch 2 at each context that measured both.

    Decode CUDA graphs on this engine are captured for batches 1, 2, 4 and 8, but
    the first full run showed they only take effect at batch 1: at L=16384 the step
    was 15.3 ms at batch 1 and 43.3 ms at batch 2, then rose smoothly. The eager
    path shows no such jump, 40.4 ms against 42.0 ms, which confirms the cause is
    graph replay rather than compute.

    This matters more than a curve shape. Stage 0 fit `c0 = 13.05 ms` from
    batch-1 data, so its constant was measured on the graph fast path, and then
    applied at every batch. Stage 0's entire argument is that KV compression pays
    by raising the reachable decode batch, which lives entirely in the regime where
    that constant does not hold.

    Fitting the two regimes together would average two different execution paths
    and describe neither, so the discontinuity is measured explicitly.
    """
    by_context = {}
    for point in points:
        by_context.setdefault(point["context_length"], {})[point["batch"]] = point
    comparisons = []
    for context_length, batches in sorted(by_context.items()):
        if 1 in batches and 2 in batches:
            single = batches[1]["step_ms"]
            double = batches[2]["step_ms"]
            if single > 0:
                comparisons.append(
                    {
                        "context_length": context_length,
                        "batch_one_ms": single,
                        "batch_two_ms": double,
                        "ratio": double / single,
                    }
                )
    if not comparisons:
        return None
    worst = max(comparisons, key=lambda item: item["ratio"])
    return {
        "comparisons": comparisons,
        "max_ratio": worst["ratio"],
        "regimes_differ": worst["ratio"] > MAX_REGIME_STEP_RATIO,
    }


def fit_models(points):
    """Fit M1 and M2 and describe both without preferring either."""
    observations = [point["step_ms"] for point in points]

    m1_design = [[1.0, float(point["kv_tokens"])] for point in points]
    m1_coefficients = least_squares(m1_design, observations)
    m1_r2, m1_predictions, m1_residual = r_squared(m1_design, observations, m1_coefficients)

    m1 = {
        "form": "step_ms = c0 + c1 * L * B",
        "c0_ms": m1_coefficients[0],
        "c1_us_per_token": m1_coefficients[1] * 1000.0,
        "r_squared": m1_r2,
        "residual_sum_squares": m1_residual,
        "parameters": 2,
    }

    batches = {point["batch"] for point in points}
    if len(batches) < 2:
        return m1, None
    m2_design = [
        [1.0, float(point["batch"]), float(point["kv_tokens"])] for point in points
    ]
    try:
        m2_coefficients = least_squares(m2_design, observations)
    except ValueError as error:
        return m1, {"error": str(error)}
    m2_r2, _predictions, m2_residual = r_squared(m2_design, observations, m2_coefficients)
    m2 = {
        "form": "step_ms = c0 + a * B + c1 * L * B",
        "c0_ms": m2_coefficients[0],
        "a_ms_per_sequence": m2_coefficients[1],
        "c1_us_per_token": m2_coefficients[2] * 1000.0,
        "r_squared": m2_r2,
        "residual_sum_squares": m2_residual,
        "parameters": 3,
        "residual_reduction_vs_m1": (
            1.0 - m2_residual / m1_residual if m1_residual > 0 else 0.0
        ),
    }
    return m1, m2


def fit_curvature_model(points):
    """Fit M3, which allows the step to bend with resident token count.

    The regressors are scaled before solving. Resident token counts reach the
    hundreds of thousands, so an unscaled quadratic design would put terms of
    order 1e22 into the normal equations and lose most of the available
    precision. The coefficients are converted back to natural units afterwards,
    so the reported numbers mean what they say.
    """
    products = {point["kv_tokens"] for point in points}
    if len(products) < 3:
        return None
    scale = 1.0e5
    design = []
    for point in points:
        scaled = point["kv_tokens"] / scale
        design.append([1.0, scaled, scaled * scaled])
    observations = [point["step_ms"] for point in points]
    try:
        coefficients = least_squares(design, observations)
    except ValueError as error:
        return {"error": str(error)}
    score, _predictions, residual = r_squared(design, observations, coefficients)
    return {
        "form": "step_ms = c0 + c1 * L * B + c2 * (L * B)^2",
        "c0_ms": coefficients[0],
        "c1_us_per_token": (coefficients[1] / scale) * 1000.0,
        "c2_ms_per_token_squared": coefficients[2] / (scale * scale),
        "r_squared": score,
        "residual_sum_squares": residual,
        "parameters": 3,
        "regressor_scale": scale,
    }


def curvature_share(m3, points):
    """How much of the step the quadratic term explains, at the largest product.

    Reported at the most demanding measured cell for the same reason as the batch
    term: that is where the Stage 0 capacity claims lean hardest on linearity.
    """
    if not m3 or "c2_ms_per_token_squared" not in m3:
        return None
    worst = max(points, key=lambda point: point["kv_tokens"])
    tokens = worst["kv_tokens"]
    quadratic = m3["c2_ms_per_token_squared"] * tokens * tokens
    predicted = (
        m3["c0_ms"] + (m3["c1_us_per_token"] / 1000.0) * tokens + quadratic
    )
    if predicted <= 0:
        return None
    return {
        "context_length": worst["context_length"],
        "batch": worst["batch"],
        "kv_tokens": tokens,
        "predicted_step_ms": predicted,
        "curvature_term_ms": quadratic,
        "share": abs(quadratic) / predicted,
    }


def batch_term_share(m2, points):
    """How much of the step the pure batch term explains, at the largest batch.

    This is the quantity that decides whether ignoring the term is harmless. It
    is reported at the most demanding measured cell, because that is where the
    Stage 0 capacity claims are most load-bearing.
    """
    if not m2 or "a_ms_per_sequence" not in m2:
        return None
    worst = max(points, key=lambda point: point["batch"])
    predicted = (
        m2["c0_ms"]
        + m2["a_ms_per_sequence"] * worst["batch"]
        + (m2["c1_us_per_token"] / 1000.0) * worst["kv_tokens"]
    )
    if predicted <= 0:
        return None
    return {
        "context_length": worst["context_length"],
        "batch": worst["batch"],
        "predicted_step_ms": predicted,
        "batch_term_ms": m2["a_ms_per_sequence"] * worst["batch"],
        "share": (m2["a_ms_per_sequence"] * worst["batch"]) / predicted,
    }


def collision_consistency(points):
    """Compare cells that share the same `L * B` but differ in shape.

    Under M1 these cells must cost the same. This is the cleanest available test,
    because it needs no fitted parameters at all.
    """
    groups = {}
    for point in points:
        groups.setdefault(point["kv_tokens"], []).append(point)
    report = []
    for kv_tokens, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        values = [member["step_ms"] for member in members]
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / mean if mean > 0 else 0.0
        report.append(
            {
                "kv_tokens": kv_tokens,
                "shapes": [[member["context_length"], member["batch"]] for member in members],
                "step_ms": values,
                "mean_ms": mean,
                "relative_spread": spread,
                "within_threshold": spread <= MAX_COLLISION_SPREAD,
            }
        )
    return report


def stage0_comparison(m1):
    """Contrast the measured constants with the frozen Stage 0 inputs."""
    return {
        "stage0_c0_ms": STAGE0_C0_MS,
        "measured_c0_ms": m1["c0_ms"],
        "c0_ratio": m1["c0_ms"] / STAGE0_C0_MS if STAGE0_C0_MS else None,
        "stage0_c1_us_per_token": STAGE0_C1_US_PER_TOKEN,
        "measured_c1_us_per_token": m1["c1_us_per_token"],
        "c1_ratio": (
            m1["c1_us_per_token"] / STAGE0_C1_US_PER_TOKEN
            if STAGE0_C1_US_PER_TOKEN
            else None
        ),
    }


def decide(m1, m2, m3, collisions, share, curvature, points, coverage_points=None):
    """Apply the pre-registered thresholds."""
    checks = []
    coverage_points = coverage_points if coverage_points is not None else points

    checks.append(
        {
            "name": "m1_r_squared",
            "detail": f"M1 R^2 {m1['r_squared']:.4f} against threshold {MIN_R_SQUARED}",
            "passed": m1["r_squared"] >= MIN_R_SQUARED,
        }
    )

    failing = [group for group in collisions if not group["within_threshold"]]
    checks.append(
        {
            "name": "collision_consistency",
            "detail": (
                f"{len(collisions) - len(failing)}/{len(collisions)} equal-product groups "
                f"agree within {MAX_COLLISION_SPREAD:.0%}"
            )
            if collisions
            else "no equal-product group was measured, so this check is vacuous",
            "passed": not failing,
            "vacuous": not collisions,
        }
    )

    if share is None:
        checks.append(
            {
                "name": "batch_term_share",
                "detail": "M2 could not be identified from the measured grid",
                "passed": True,
                "vacuous": True,
            }
        )
    else:
        checks.append(
            {
                "name": "batch_term_share",
                "detail": (
                    f"the pure batch term explains {share['share']:.1%} of the step at "
                    f"L={share['context_length']}, B={share['batch']}, against a "
                    f"tolerance of {MAX_BATCH_TERM_SHARE:.0%}"
                ),
                "passed": share["share"] <= MAX_BATCH_TERM_SHARE,
            }
        )

    if curvature is None:
        checks.append(
            {
                "name": "curvature_share",
                "detail": (
                    "fewer than three distinct resident-token counts were measured, "
                    "so linearity in L*B cannot be tested"
                ),
                "passed": True,
                "vacuous": True,
            }
        )
    else:
        checks.append(
            {
                "name": "curvature_share",
                "detail": (
                    f"the quadratic term explains {curvature['share']:.1%} of the step at "
                    f"L={curvature['context_length']}, B={curvature['batch']}, against a "
                    f"tolerance of {MAX_CURVATURE_SHARE:.0%}"
                ),
                "passed": curvature["share"] <= MAX_CURVATURE_SHARE,
            }
        )

    drifting = [
        point
        for point in points
        if point.get("drift_ratio") is not None
        and abs(point["drift_ratio"] - 1.0) > MAX_DRIFT_DEVIATION
    ]
    rated = [point for point in points if point.get("drift_ratio") is not None]
    checks.append(
        {
            "name": "window_stability",
            "detail": (
                f"{len(rated) - len(drifting)}/{len(rated)} cells held steady within "
                f"{MAX_DRIFT_DEVIATION:.0%} across their measured window"
                + (
                    "; drifting: "
                    + ", ".join(
                        f"L={point['context_length']} B={point['batch']} "
                        f"ratio {point['drift_ratio']:.3f}"
                        for point in drifting[:6]
                    )
                    if drifting
                    else ""
                )
            )
            if rated
            else "no cell reported a drift ratio",
            "passed": not drifting,
            "vacuous": not rated,
        }
    )

    # A median can be robust to an outlier and still sit inside a cell that was not
    # doing steady decoding. One cell in the second full run reported a median of
    # 60.6 ms with a standard deviation of 109.7 ms, meaning at least one step took
    # hundreds of milliseconds. The median was believable and the cell was not, and
    # nothing in the gate could see it.
    dispersed = [
        point
        for point in coverage_points
        if point.get("dispersion_ratio") is not None
        and point["dispersion_ratio"] > MAX_DISPERSION_RATIO
    ]
    scored = [
        point for point in coverage_points if point.get("dispersion_ratio") is not None
    ]
    checks.append(
        {
            "name": "sample_dispersion",
            "detail": (
                f"{len(scored) - len(dispersed)}/{len(scored)} cells kept their spread "
                f"under {MAX_DISPERSION_RATIO:.0%} of their median"
                + (
                    "; dispersed: "
                    + ", ".join(
                        f"L={point['context_length']} B={point['batch']} "
                        f"stdev/median {point['dispersion_ratio']:.2f}"
                        for point in dispersed[:6]
                    )
                    if dispersed
                    else ""
                )
            )
            if scored
            else "no cell reported a dispersion ratio",
            "passed": not dispersed,
            "vacuous": not scored,
        }
    )

    measured_cells = {
        (point["context_length"], point["batch"]) for point in coverage_points
    }
    missing = sorted(set(PREREGISTERED_CELLS) - measured_cells)
    checks.append(
        {
            "name": "grid_coverage",
            "detail": (
                f"{len(measured_cells & set(PREREGISTERED_CELLS))}/"
                f"{len(PREREGISTERED_CELLS)} pre-registered cells produced a "
                "measurement"
                + (
                    "; missing: "
                    + ", ".join(f"({cell[0]},{cell[1]})" for cell in missing[:8])
                    + (" ..." if len(missing) > 8 else "")
                    if missing
                    else ""
                )
            ),
            "passed": not missing,
        }
    )

    coverage_ok = len({point["batch"] for point in points}) >= 3
    checks.append(
        {
            "name": "batch_coverage",
            "detail": (
                f"{len({point['batch'] for point in points})} distinct batches measured; "
                "at least 3 are needed to say anything about batch scaling"
            ),
            "passed": coverage_ok,
        }
    )

    passed = all(check["passed"] for check in checks)
    if passed:
        verdict = "PASS"
        consequence = (
            "The Stage 0 decode-step model survives on the serving path. "
            "Proceed to GATE B, the head-slicing phi probe."
        )
    elif dispersed:
        verdict = "INCONCLUSIVE"
        consequence = (
            f"{len(dispersed)} cells had a spread larger than "
            f"{MAX_DISPERSION_RATIO:.0%} of their own median, so something other "
            "than steady decoding happened inside them. The median can survive that "
            "while the cell remains untrustworthy. Re-run those cells before "
            "concluding anything."
        )
    elif drifting:
        verdict = "INCONCLUSIVE"
        consequence = (
            "At least one cell was still warming up while it was being timed, so "
            "the numbers describe compilation as much as decoding. This refutes the "
            "measurement, not the Stage 0 model. Raise the warmup and re-run before "
            "concluding anything."
        )
    elif missing:
        verdict = "INCONCLUSIVE"
        consequence = (
            f"{len(missing)} pre-registered cells produced no measurement, so the "
            "grid the thresholds were registered against was not actually covered. "
            "A missing cell is a gap in the measurement, not evidence against the "
            "Stage 0 model. Recover those cells or re-register the grid with the "
            "reason, then re-run."
        )
    elif not coverage_ok:
        verdict = "INCONCLUSIVE"
        consequence = (
            "Too few batches were measured to test the batch term. Fix the "
            "measurement before drawing any conclusion; do not proceed to GATE B."
        )
    else:
        verdict = "FAIL"
        consequence = (
            "The Stage 0 decode-step model does not describe the serving path. "
            "The capacity numbers in the Stage 0 artifact must be refit with the "
            "measured form before the latent KV line continues. Do not proceed to "
            "GATE B on the old constants."
        )
    return verdict, consequence, checks


def build_report(payload):
    points, rejected = extract_points(payload)
    if len(points) < 3:
        return {
            "schema": "kvcapacity-gate-a-verdict/1",
            "verdict": "INCONCLUSIVE",
            "consequence": (
                "Fewer than three cells were measured cleanly, so no fit is "
                "meaningful."
            ),
            "measured_points": points,
            "rejected_cells": rejected,
        }
    if len({point["kv_tokens"] for point in points}) < 2:
        # This is reachable, not theoretical: the four largest cells in the
        # pre-registered grid all share L*B = 262144, so a run that only managed
        # the largest batch at each context would land here. Refuse to fit rather
        # than crash inside the solver.
        return {
            "schema": "kvcapacity-gate-a-verdict/1",
            "source_payload_sha256": payload.get("payload_sha256"),
            "verdict": "INCONCLUSIVE",
            "consequence": (
                "Every measured cell shares the same resident token count, so no "
                "slope can be identified. Widen the grid before concluding anything."
            ),
            "measured_points": points,
            "rejected_cells": rejected,
        }
    discontinuity = regime_discontinuity(points)
    fit_points = points
    fit_scope = "all measured batches"
    if discontinuity and discontinuity["regimes_differ"]:
        multi = [point for point in points if point["batch"] >= 2]
        if len(multi) >= 3 and len({point["kv_tokens"] for point in multi}) >= 2:
            fit_points = multi
            fit_scope = (
                "batch >= 2 only, because batch 1 runs a different execution path"
            )
    points_all = points
    points = fit_points
    m1, m2 = fit_models(points)
    preregistered = payload.get("grid_is_preregistered")
    m3 = fit_curvature_model(points)
    collisions = collision_consistency(points)
    share = batch_term_share(m2, points)
    curvature = curvature_share(m3, points)
    verdict, consequence, checks = decide(
        m1, m2, m3, collisions, share, curvature, points,
        coverage_points=points_all,
    )
    if preregistered is False:
        # A narrowed grid cannot decide the gate in either direction, so every
        # verdict collapses to INCONCLUSIVE and the checks are kept as commentary.
        #
        # Both directions matter. A narrow grid can fit beautifully and be
        # presented as though it had answered GATE A. It can equally fail for a
        # reason that is purely an artifact of its own narrowness: the smoke grid
        # runs a 0.6B model at contexts up to 4096, where the KV term is only a
        # couple of percent of the step, so the constant dominates, R^2 collapses,
        # and the run reports FAIL while establishing nothing. Emitting that FAIL
        # would train the reader to discount a FAIL on the real grid, which is the
        # one verdict this whole gate exists to be able to deliver.
        previous = verdict
        verdict = "INCONCLUSIVE"
        consequence = (
            "This run used a narrowed grid rather than the pre-registered one, so "
            "it can only demonstrate that the measurement works and cannot decide "
            f"GATE A in either direction. The checks would have returned {previous} "
            "on this data, which is recorded as commentary only: " + consequence
        )
    report = {
        "schema": "kvcapacity-gate-a-verdict/1",
        "source_payload_sha256": payload.get("payload_sha256"),
        "grid_is_preregistered": preregistered,
        "grid_spec": payload.get("grid_spec"),
        "missing_preregistered_cells": payload.get("missing_preregistered_cells"),
        "attempted_but_unmeasured_cells": payload.get("attempted_but_unmeasured_cells"),
        "preregistered_cells_v1": [list(cell) for cell in PREREGISTERED_CELLS_V1],
        "preregistered_cells": [list(cell) for cell in PREREGISTERED_CELLS],
        "thresholds": {
            "min_r_squared": MIN_R_SQUARED,
            "max_collision_spread": MAX_COLLISION_SPREAD,
            "max_batch_term_share": MAX_BATCH_TERM_SHARE,
            "max_curvature_share": MAX_CURVATURE_SHARE,
            "max_drift_deviation": MAX_DRIFT_DEVIATION,
            "max_dispersion_ratio": MAX_DISPERSION_RATIO,
        },
        "measured_points": points_all,
        "fitted_points": points,
        "fit_scope": fit_scope,
        "regime_discontinuity": discontinuity,
        "rejected_cells": rejected,
        "model_m1": m1,
        "model_m2": m2,
        "model_m3": m3,
        "batch_term_at_largest_batch": share,
        "curvature_at_largest_product": curvature,
        "collision_groups": collisions,
        "stage0_comparison": stage0_comparison(m1),
        "checks": checks,
        "verdict": verdict,
        "consequence": consequence,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def render(report):
    lines = []
    lines.append("GATE A: decode step scaling against the Stage 0 model")
    lines.append("=" * 72)
    verdict = report["verdict"]
    lines.append(f"verdict {verdict}")
    lines.append(f"  {report['consequence']}")
    if report.get("grid_is_preregistered") is False:
        lines.append(f"  grid used: {report.get('grid_spec')} (NOT pre-registered)")
        missing = report.get("missing_preregistered_cells") or []
        if missing:
            lines.append(
                "  missing pre-registered cells: "
                + ", ".join(f"({cell[0]},{cell[1]})" for cell in missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
    lines.append("")

    points = report.get("measured_points") or []
    if points:
        lines.append("measured cells")
        lines.append(f"  {'L':>8}  {'B':>3}  {'L*B':>9}  {'step_ms':>9}  {'stdev':>7}  {'n':>3}")
        for point in sorted(points, key=lambda item: (item["context_length"], item["batch"])):
            lines.append(
                f"  {point['context_length']:>8}  {point['batch']:>3}  "
                f"{point['kv_tokens']:>9}  {point['step_ms']:>9.3f}  "
                f"{point['step_stdev_ms']:>7.3f}  {point['sample_count']:>3}"
            )
        lines.append("")

    rejected = report.get("rejected_cells") or []
    if rejected:
        lines.append("cells excluded from the fit")
        for entry in rejected:
            lines.append(
                f"  L={entry['context_length']} B={entry['batch']}: {entry['reason']}"
            )
        lines.append("")

    discontinuity = report.get("regime_discontinuity")
    if discontinuity:
        lines.append("batch 1 against batch 2, the CUDA graph regime boundary")
        for item in discontinuity["comparisons"]:
            lines.append(
                f"  L={item['context_length']}: {item['batch_one_ms']:.3f} ms at B=1 "
                f"against {item['batch_two_ms']:.3f} ms at B=2, ratio {item['ratio']:.2f}"
            )
        if discontinuity["regimes_differ"]:
            lines.append(
                "  these are different execution paths; Stage 0's c0 was fit at "
                "batch 1 and does not describe the multi-sequence regime its own "
                "capacity argument depends on"
            )
        lines.append("")
    if report.get("fit_scope"):
        lines.append(f"fitted on: {report['fit_scope']}")
        lines.append("")

    m1 = report.get("model_m1")
    if m1:
        lines.append("M1, the Stage 0 form")
        lines.append(f"  {m1['form']}")
        lines.append(
            f"  c0 = {m1['c0_ms']:.3f} ms, c1 = {m1['c1_us_per_token']:.4f} us/token, "
            f"R^2 = {m1['r_squared']:.4f}"
        )
        lines.append("")

    m2 = report.get("model_m2")
    if m2 and "a_ms_per_sequence" in m2:
        lines.append("M2, with a pure batch term")
        lines.append(f"  {m2['form']}")
        lines.append(
            f"  c0 = {m2['c0_ms']:.3f} ms, a = {m2['a_ms_per_sequence']:.4f} ms/seq, "
            f"c1 = {m2['c1_us_per_token']:.4f} us/token, R^2 = {m2['r_squared']:.4f}"
        )
        lines.append(
            f"  residual reduction against M1: {m2['residual_reduction_vs_m1']:.1%}"
        )
        share = report.get("batch_term_at_largest_batch")
        if share:
            lines.append(
                f"  at L={share['context_length']} B={share['batch']}, the batch term is "
                f"{share['batch_term_ms']:.3f} ms of {share['predicted_step_ms']:.3f} ms, "
                f"or {share['share']:.1%}"
            )
        lines.append("")

    m3 = report.get("model_m3")
    if m3 and "c2_ms_per_token_squared" in m3:
        lines.append("M3, allowing the step to bend with resident tokens")
        lines.append(f"  {m3['form']}")
        lines.append(
            f"  c0 = {m3['c0_ms']:.3f} ms, c1 = {m3['c1_us_per_token']:.4f} us/token, "
            f"c2 = {m3['c2_ms_per_token_squared']:.3e} ms/token^2, "
            f"R^2 = {m3['r_squared']:.4f}"
        )
        curvature = report.get("curvature_at_largest_product")
        if curvature:
            lines.append(
                f"  at L*B={curvature['kv_tokens']}, the quadratic term is "
                f"{curvature['curvature_term_ms']:.3f} ms of "
                f"{curvature['predicted_step_ms']:.3f} ms, or {curvature['share']:.1%}"
            )
        lines.append("")

    collisions = report.get("collision_groups") or []
    if collisions:
        lines.append("equal-product cells, which M1 requires to cost the same")
        for group in collisions:
            shapes = ", ".join(f"({shape[0]},{shape[1]})" for shape in group["shapes"])
            marker = "ok" if group["within_threshold"] else "MISMATCH"
            lines.append(
                f"  L*B={group['kv_tokens']}: {shapes} -> "
                + ", ".join(f"{value:.3f} ms" for value in group["step_ms"])
                + f"  spread {group['relative_spread']:.1%}  {marker}"
            )
        lines.append("")

    comparison = report.get("stage0_comparison")
    if comparison:
        lines.append("measured against the frozen Stage 0 inputs")
        lines.append(
            f"  c0: {comparison['measured_c0_ms']:.3f} ms measured against "
            f"{comparison['stage0_c0_ms']} ms assumed"
            + (f", ratio {comparison['c0_ratio']:.3f}" if comparison["c0_ratio"] else "")
        )
        lines.append(
            f"  c1: {comparison['measured_c1_us_per_token']:.4f} us/token measured against "
            f"{comparison['stage0_c1_us_per_token']} us/token assumed"
            + (f", ratio {comparison['c1_ratio']:.3f}" if comparison["c1_ratio"] else "")
        )
        lines.append("")

    checks = report.get("checks") or []
    if checks:
        lines.append("pre-registered checks")
        for check in checks:
            status = "PASS" if check["passed"] else "FAIL"
            if check.get("vacuous"):
                status += " (vacuous)"
            lines.append(f"  [{status}] {check['name']}: {check['detail']}")
        lines.append("")

    if report.get("report_sha256"):
        lines.append(f"report_sha256 {report['report_sha256']}")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        required=True,
        action="append",
        help=(
            "worker artifact JSON; repeat once per context length, since each "
            "context length must run in its own process"
        ),
    )
    parser.add_argument("--out", help="where to write the verdict JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payloads = []
    for path in args.payload:
        with open(path, encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    report = build_report(merge_payloads(payloads))
    print(render(report))
    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
