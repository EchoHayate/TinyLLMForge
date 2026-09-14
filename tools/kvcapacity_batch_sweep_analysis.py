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

# Concurrency is called useful only if the best throughput in the sweep beats the
# smallest batch by at least this much. Below it, the extra sequences were paid
# for and returned nothing.
MIN_USEFUL_THROUGHPUT_GAIN = 0.10

# Batch 1 runs a CUDA graph fast path on the default execution path: GATE A
# measured 15-18 ms at B=1 against 43-50 ms at B=2. Comparing across that boundary
# produces a fake collapse in marginal throughput, so a sweep whose B=1 cell is
# this much cheaper than its B=2 cell is analysed from B=2 upwards and reports
# B=1 separately. Same threshold as the GATE A verdict, deliberately.
REGIME_STEP_RATIO = 1.30

# Cells whose spread exceeds this fraction of their median are not steady
# decoding. GATE A rejects them; a sweep flags them and keeps them out of the fits
# rather than letting one contaminated cell set the shape of the curve.
MAX_DISPERSION_RATIO = 0.25


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


def regime_boundary(rows):
    """Detect the batch-1 CUDA graph fast path at one context length."""

    by_batch = {row["batch"]: row for row in rows}
    first, second = by_batch.get(1), by_batch.get(2)
    if not first or not second:
        return None
    ratio = second["step_ms"] / first["step_ms"] if first["step_ms"] else None
    if ratio is None or ratio < REGIME_STEP_RATIO:
        return None
    return {
        "batch1_step_ms": first["step_ms"],
        "batch2_step_ms": second["step_ms"],
        "ratio": ratio,
        "note": (
            "batch 1 runs a different execution path, so the sweep is read from "
            "batch 2 upwards"
        ),
    }


def analysable_rows(rows):
    """Drop cells that cannot carry an argument, and say which and why."""

    kept, dropped = [], []
    for row in rows:
        ratio = row.get("dispersion_ratio")
        if ratio is not None and ratio > MAX_DISPERSION_RATIO:
            dropped.append(
                {
                    "batch": row["batch"],
                    "reason": (
                        "spread %.0f%% of the median, not steady decoding"
                        % (ratio * 100.0)
                    ),
                }
            )
            continue
        kept.append(row)
    boundary = regime_boundary(kept)
    if boundary is not None:
        for row in list(kept):
            if row["batch"] == 1:
                kept.remove(row)
                dropped.append(
                    {
                        "batch": 1,
                        "reason": (
                            "CUDA graph fast path, %.3f ms against %.3f ms at "
                            "batch 2" % (boundary["batch1_step_ms"],
                                         boundary["batch2_step_ms"])
                        ),
                    }
                )
    return kept, dropped, boundary


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
    smallest = rows[0]["throughput_seq_per_s"]
    peak_gain = (
        peak["throughput_seq_per_s"] / smallest if smallest else None
    )
    return {
        "peak_gain_over_smallest_batch": peak_gain,
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


MAX_FOREIGN_DEVICE_SHARE = 0.05


def contamination(payload):
    """Report engines whose KV pool was carved out of a shared card.

    A wall sweep answers "how much concurrency fits", and the answer is only
    about the model when the card is idle. Two consecutive sweeps on the same
    host produced ctx8192 capacities of 343808 and 89856 tokens because a
    neighbour took 38 GiB in between; the second one refused every cell and
    reported INCONCLUSIVE, which reads like the grid was too small. It was not:
    the device was not the device under test any more. Contamination therefore
    gets its own reading instead of being laundered into a grid complaint.
    """
    offenders = []
    for engine in payload.get("engines") or []:
        identity = engine.get("identity") or {}
        share = identity.get("device_foreign_share")
        pinned = identity.get("kv_blocks_requested")
        if pinned:
            # A pinned pool is either served in full or construction fails, so a
            # busy neighbour cannot quietly move the wall.
            continue
        if share is None or float(share) > MAX_FOREIGN_DEVICE_SHARE:
            offenders.append(
                {
                    "context_length": engine.get("context_length"),
                    "device_foreign_share": share,
                    "kv_capacity_tokens": identity.get("kv_capacity_tokens"),
                }
            )
    return offenders


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
    # Dead means concurrency bought essentially nothing anywhere: the best
    # throughput in the sweep is no better than the smallest batch already gave.
    # A sweep that climbed and then fell is not dead, it is bounded, and the two
    # lead to different decisions.
    useful = [
        item
        for item in verdicts
        if item[1]["peak_gain_over_smallest_batch"] is not None
        and item[1]["peak_gain_over_smallest_batch"] > 1.0 + MIN_USEFUL_THROUGHPUT_GAIN
    ]
    if not useful:
        return {
            "reading": "CAPACITY DEAD",
            "detail": (
                "the best throughput in the sweep was no better than the "
                "smallest batch already gave, so holding more sequences cannot "
                "pay even if KV were free"
            ),
        }
    # A peak strictly inside the sweep is the most important shape a sweep can
    # find: it means concurrency has an optimum and pushing past it costs
    # throughput, so the capacity payoff is bounded by that peak rather than by
    # the KV budget. The wall sweep hit exactly this at L=2048, where throughput
    # rose to 2015 seq/s at B=128 and fell to 1950 seq/s at B=144.
    interior_peak = [
        item for item in verdicts if not item[1]["peak_is_largest_batch"]
    ]
    if interior_peak:
        peaks = ", ".join(
            "L=%s peaks at B=%d with %.0f seq/s"
            % (
                context,
                sat["peak_batch"],
                sat["peak_throughput_seq_per_s"],
            )
            for context, sat in interior_peak
        )
        return {
            "reading": "CAPACITY BOUNDED",
            "detail": (
                "throughput has an interior optimum, so trading KV bytes for "
                "concurrency pays only up to that peak: " + peaks
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
    contexts = {}
    excluded = {}
    boundaries = {}
    for context, rows in group_by_context(points).items():
        priced = throughput_rows(rows)
        kept, dropped, boundary = analysable_rows(priced)
        contexts[context] = throughput_rows(
            [
                {
                    "context_length": row["context_length"],
                    "batch": row["batch"],
                    "step_ms": row["step_ms"],
                    "step_stdev_ms": row["step_stdev_ms"],
                    "drift_ratio": row["drift_ratio"],
                    "dispersion_ratio": row["dispersion_ratio"],
                }
                for row in kept
            ]
        )
        excluded[context] = dropped
        boundaries[context] = boundary
    fits = {
        context: fit_batch_terms(rows) for context, rows in contexts.items()
    }
    sats = {context: saturation(rows) for context, rows in contexts.items()}
    dirty = contamination(payload)
    reading = capacity_reading(contexts)
    if dirty:
        shares = ", ".join(
            f"L={item['context_length']} foreign_share="
            f"{'unknown' if item['device_foreign_share'] is None else item['device_foreign_share']}"
            f" capacity={item['kv_capacity_tokens']} tokens"
            for item in dirty
        )
        reading = {
            "reading": "CONTAMINATED",
            "detail": (
                "the KV pool was sized against a card somebody else was already "
                f"using, so the wall is not a property of the model: {shares}. "
                "Rerun on an idle device or pin the pool with --kv-blocks."
            ),
            "suppressed_reading": reading,
        }
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
        "excluded_cells": excluded,
        "regime_boundary": boundaries,
        "batch_fits": fits,
        "saturation": sats,
        "capacity_reading": reading,
        "contaminated_engines": dirty,
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
        for dropped in (report.get("excluded_cells") or {}).get(context, []):
            lines.append(
                "  excluded B=%s: %s" % (dropped["batch"], dropped["reason"])
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
