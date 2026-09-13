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
            }
        )
    return points, rejected


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


def decide(m1, m2, m3, collisions, share, curvature, points):
    """Apply the pre-registered thresholds."""
    checks = []

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
    m1, m2 = fit_models(points)
    preregistered = payload.get("grid_is_preregistered")
    m3 = fit_curvature_model(points)
    collisions = collision_consistency(points)
    share = batch_term_share(m2, points)
    curvature = curvature_share(m3, points)
    verdict, consequence, checks = decide(
        m1, m2, m3, collisions, share, curvature, points
    )
    if preregistered is False:
        # A narrowed grid can still be internally consistent, so a PASS here
        # would be technically true and materially misleading. Downgrade it.
        if verdict == "PASS":
            verdict = "INCONCLUSIVE"
        consequence = (
            "This run used a narrowed grid rather than the pre-registered one, so "
            "it can only demonstrate that the measurement works. It cannot decide "
            "GATE A. Consequence recorded from the checks: " + consequence
        )
    report = {
        "schema": "kvcapacity-gate-a-verdict/1",
        "source_payload_sha256": payload.get("payload_sha256"),
        "grid_is_preregistered": preregistered,
        "grid_spec": payload.get("grid_spec"),
        "thresholds": {
            "min_r_squared": MIN_R_SQUARED,
            "max_collision_spread": MAX_COLLISION_SPREAD,
            "max_batch_term_share": MAX_BATCH_TERM_SHARE,
            "max_curvature_share": MAX_CURVATURE_SHARE,
        },
        "measured_points": points,
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
    parser.add_argument("--payload", required=True, help="worker artifact JSON")
    parser.add_argument("--out", help="where to write the verdict JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.payload, encoding="utf-8") as handle:
        payload = json.load(handle)
    report = build_report(payload)
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
