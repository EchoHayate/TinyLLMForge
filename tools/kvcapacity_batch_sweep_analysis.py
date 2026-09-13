#!/usr/bin/env python3
"""Analyse a fixed-context, high-concurrency batch sweep.

GATE A falsified the Stage 0 decode model on the latency axis: the step is
dominated by a ~40 ms context-independent constant, so compressing resident KV
can only move a minority of the step. That result did not settle the capacity
axis, which is the only remaining reason to build GATE B: if KV compression lets
a box hold more concurrent sequences, the payoff is throughput, not per-step
latency.

GATE A also measured the two terms most likely to kill that argument, but only up
to batch 32:

    a  = 0.176 ms/seq   (eager pure batch term)
    c2 = 1.363e-10 ms/token^2 at L*B up to 262144

This tool reads a sweep that holds L fixed and pushes B until the box refuses,
and answers one question:

    does decode throughput keep rising with concurrency, or does it saturate?

If throughput saturates before the KV budget is reached, then raising the
sequence count by compressing KV buys nothing, and the capacity argument is dead
regardless of how good the compression is. If throughput is still rising at the
KV wall, the capacity argument survives and GATE B is worth building.

The tool is deliberately separate from the GATE A verdict. A sweep is not the
pre-registered grid and must never be presentable as GATE A; conversely GATE A's
pass/fail thresholds say nothing about throughput saturation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvcapacity_step_scaling_verdict import (  # noqa: E402
    extract_points,
    least_squares,
    merge_payloads,
    r_squared,
)

# A sweep is called saturating when the last measured concurrency step buys less
# than this fraction of the throughput that the first one bought. Pre-registered
# here rather than chosen after looking at the numbers.
SATURATION_MARGINAL_FRACTION = 0.25

# Throughput is called flat once an extra sequence adds less than this fraction
# of a proportional gain, i.e. doubling B yields less than this much more work.
MIN_USEFUL_THROUGHPUT_GAIN = 0.10


def load_payload(paths):
    payloads = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    if len(payloads) == 1:
        return payloads[0]
    return merge_payloads(payloads)


def group_by_context(points):
    """Group measured points by context length, ordered by batch."""

    grouped = {}
    for point in points:
        grouped.setdefault(point["context_length"], []).append(point)
    return {
        context: sorted(rows, key=lambda row: row["batch"])
        for context, rows in sorted(grouped.items())
    }


def throughput_rows(rows):
    """Per-batch decode throughput in sequence-steps per second."""

    out = []
    previous = None
    for row in rows:
        step_ms = row["step_ms"]
        batch = row["batch"]
        throughput = batch / step_ms * 1000.0
        marginal_ms = None
        marginal_throughput = None
        if previous is not None:
            added_seqs = batch - previous["batch"]
            if added_seqs > 0:
                marginal_ms = (step_ms - previous["step_ms"]) / added_seqs
                marginal_throughput = (
                    (throughput - previous["throughput_seq_per_s"])
                    / added_seqs
                )
        entry = {
            "context_length": row["context_length"],
            "batch": batch,
            "step_ms": step_ms,
            "step_stdev_ms": row.get("step_stdev_ms"),
            "throughput_seq_per_s": throughput,
            "marginal_ms_per_seq": marginal_ms,
            "marginal_throughput_per_seq": marginal_throughput,
            "drift_ratio": row.get("drift_ratio"),
            "dispersion_ratio": row.get("dispersion_ratio"),
        }
        out.append(entry)
        previous = entry
    return out


def fit_batch_terms(rows):
    """Fit step_ms against B and against B plus B^2 at one fixed context."""

    if len(rows) < 3:
        return None
    batches = [float(row["batch"]) for row in rows]
    observations = [row["step_ms"] for row in rows]
    if len(set(batches)) < 3:
        return None

    linear = least_squares([[1.0, b] for b in batches], observations)
    quadratic = least_squares(
        [[1.0, b, b * b] for b in batches], observations
    )
    if linear is None or quadratic is None:
        return None

    max_batch = max(batches)
    quad_term = quadratic[2] * max_batch * max_batch
    predicted = (
        quadratic[0] + quadratic[1] * max_batch + quad_term
    )
    return {
        "linear": {
            "c0_ms": linear[0],
            "a_ms_per_seq": linear[1],
            "r_squared": r_squared(
                [[1.0, b] for b in batches], observations, linear
            )[0],
        },
        "quadratic": {
            "c0_ms": quadratic[0],
            "a_ms_per_seq": quadratic[1],
            "b_ms_per_seq2": quadratic[2],
            "r_squared": r_squared(
                [[1.0, b, b * b] for b in batches], observations, quadratic
            )[0],
            "term_at_max_batch_ms": quad_term,
            "predicted_at_max_batch_ms": predicted,
            "share_at_max_batch": (
                abs(quad_term) / abs(predicted) if predicted else None
            ),
        },
        "max_batch": int(max_batch),
    }


def saturation(rows):
    """Decide whether throughput has stopped rising with concurrency."""

    scored = [
        row for row in rows if row["marginal_throughput_per_seq"] is not None
    ]
    if len(scored) < 2:
        return None
    first = scored[0]["marginal_throughput_per_seq"]
    last = scored[-1]["marginal_throughput_per_seq"]
    fraction = last / first if first else None
    peak = max(rows, key=lambda row: row["throughput_seq_per_s"])
    proportional_gain = None
    if len(rows) >= 2:
        base, top = rows[0], rows[-1]
        batch_ratio = top["batch"] / base["batch"]
        throughput_ratio = (
            top["throughput_seq_per_s"] / base["throughput_seq_per_s"]
        )
        if batch_ratio > 1:
            proportional_gain = throughput_ratio / batch_ratio
    return {
        "first_marginal_throughput_per_seq": first,
        "last_marginal_throughput_per_seq": last,
        "retained_fraction": fraction,
        "saturating": (
            fraction is not None and fraction < SATURATION_MARGINAL_FRACTION
        ),
        "throughput_still_rising": last > 0.0,
        "peak_batch": peak["batch"],
        "peak_throughput_seq_per_s": peak["throughput_seq_per_s"],
        "peak_is_largest_batch": peak["batch"] == rows[-1]["batch"],
        "scaling_efficiency": proportional_gain,
    }


def capacity_reading(contexts):
    """Turn per-context saturation into one statement about the capacity axis."""

    verdicts = []
    for context, rows in contexts.items():
        sat = saturation(rows)
        if sat is None:
            continue
        verdicts.append((context, sat))
    if not verdicts:
        return {
            "reading": "INCONCLUSIVE",
            "detail": "no context length carried enough batches to judge",
        }
    rising = [item for item in verdicts if item[1]["throughput_still_rising"]]
    if not rising:
        return {
            "reading": "CAPACITY DEAD",
            "detail": (
                "throughput fell as concurrency grew, so holding more "
                "sequences cannot pay even if KV were free"
            ),
        }
    saturating = [item for item in verdicts if item[1]["saturating"]]
    if len(saturating) == len(verdicts):
        return {
            "reading": "CAPACITY WEAK",
            "detail": (
                "throughput still rises but the last sequences added far less "
                "than the first; compressing KV to hold more sequences buys a "
                "shrinking return"
            ),
        }
    return {
        "reading": "CAPACITY OPEN",
        "detail": (
            "throughput was still rising usefully at the largest measured "
            "concurrency, so the capacity argument is not dead and GATE B "
            "remains worth building"
        ),
    }


def build_report(payload):
    points, rejected = extract_points(payload)
    contexts = {
        context: throughput_rows(rows)
        for context, rows in group_by_context(points).items()
    }
    fits = {
        context: fit_batch_terms(rows) for context, rows in contexts.items()
    }
    sats = {context: saturation(rows) for context, rows in contexts.items()}
    return {
        "kind": "kvcapacity_batch_sweep_analysis",
        "note": (
            "This is an exploratory sweep, not GATE A. It cannot pass or fail "
            "the pre-registered GATE A grid."
        ),
        "source_payload_sha256": payload.get("payload_sha256"),
        "grid_spec": payload.get("grid_spec"),
        "grid_is_preregistered": payload.get("grid_is_preregistered"),
        "contexts": contexts,
        "batch_fits": fits,
        "saturation": sats,
        "capacity_reading": capacity_reading(contexts),
        "rejected_cells": rejected,
    }


def render(report):
    lines = []
    lines.append("batch sweep: does concurrency still buy throughput?")
    lines.append("=" * 72)
    reading = report["capacity_reading"]
    lines.append(f"reading {reading['reading']}")
    lines.append(f"  {reading['detail']}")
    lines.append("")
    if report.get("grid_is_preregistered") is False:
        lines.append(
            "this is a sweep, not GATE A; it says nothing about the "
            "pre-registered grid"
        )
        lines.append("")

    for context, rows in report["contexts"].items():
        lines.append(f"L={context}")
        lines.append(
            "        B    step_ms   stdev   seq/s   marg_ms/seq  marg_seq/s"
        )
        for row in rows:
            marginal_ms = row["marginal_ms_per_seq"]
            marginal_tp = row["marginal_throughput_per_seq"]
            lines.append(
                "  %7d %10.3f %7.3f %7.2f %13s %11s"
                % (
                    row["batch"],
                    row["step_ms"],
                    row["step_stdev_ms"] or 0.0,
                    row["throughput_seq_per_s"],
                    "-" if marginal_ms is None else f"{marginal_ms:.3f}",
                    "-" if marginal_tp is None else f"{marginal_tp:.3f}",
                )
            )
        sat = report["saturation"].get(context)
        if sat:
            lines.append(
                "  peak throughput %.2f seq/s at B=%d%s"
                % (
                    sat["peak_throughput_seq_per_s"],
                    sat["peak_batch"],
                    "" if sat["peak_is_largest_batch"] else " (before the wall)",
                )
            )
            if sat["retained_fraction"] is not None:
                lines.append(
                    "  the last sequences added %.1f%% of what the first ones "
                    "added" % (sat["retained_fraction"] * 100.0)
                )
            if sat["scaling_efficiency"] is not None:
                lines.append(
                    "  scaling efficiency across the sweep: %.2f of "
                    "proportional" % sat["scaling_efficiency"]
                )
        fit = report["batch_fits"].get(context)
        if fit:
            linear = fit["linear"]
            quad = fit["quadratic"]
            lines.append(
                "  step_ms = %.3f + %.3f * B                 R^2 = %.4f"
                % (linear["c0_ms"], linear["a_ms_per_seq"], linear["r_squared"])
            )
            lines.append(
                "  step_ms = %.3f + %.3f * B + %.3e * B^2    R^2 = %.4f"
                % (
                    quad["c0_ms"],
                    quad["a_ms_per_seq"],
                    quad["b_ms_per_seq2"],
                    quad["r_squared"],
                )
            )
            if quad["share_at_max_batch"] is not None:
                lines.append(
                    "  at B=%d the B^2 term is %.3f ms of %.3f ms, or %.1f%%"
                    % (
                        fit["max_batch"],
                        quad["term_at_max_batch_ms"],
                        quad["predicted_at_max_batch_ms"],
                        quad["share_at_max_batch"] * 100.0,
                    )
                )
        lines.append("")

    rejected = report.get("rejected_cells") or []
    if rejected:
        lines.append("cells that produced no usable measurement")
        for cell in rejected[:12]:
            lines.append(
                "  L=%s B=%s: %s"
                % (cell.get("context_length"), cell.get("batch"), cell.get("reason"))
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        action="append",
        required=True,
        help="worker artifact; repeat once per context-length process",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    payload = load_payload(args.payload)
    report = build_report(payload)
    sys.stdout.write(render(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
