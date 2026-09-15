"""Price any latent / KV-compression proposal against the measured decode model.

The point of this tool is to stop measuring proposals one at a time. Every
compression idea moves at most three things, so once the decode model is fitted
they can all be priced from the same parameters:

    step_ms = c0 + (a + c1_eff * L_eff) * B

    c0      per-step fixed cost, mostly weight reads, amortised by B
    a       per-sequence cost independent of L  <-- compression cannot touch it
    c1_eff  proportional to bytes per token
    L_eff   tokens actually attended
    B       concurrency, bounded by the KV budget

`a` therefore sets the ceiling: drive `c1_eff * L_eff` to zero and what remains is
`c0 + a*B`. Two fits of the same engine disagree about `a` by 5.7x, which is the
difference between a 8.7x ceiling and a 49.7x one, so `a` is exposed as a required
argument rather than hidden in a default. See
docs/superpowers/plans/2026-09-15-latent-pricing-and-the-a-term.md.

Measured inputs (Qwen3-8B, A100 80GB, graph path, pinned 640-block pool, bf16):
    L=2048  step_ms = 12.839 + 0.383*B   R^2=0.9959   wall B=70, 1743.54 seq/s
    L=8192  step_ms = 12.113 + 1.407*B   R^2=0.9977   wall B=19,  482.74 seq/s
"""

from __future__ import annotations

import argparse
import json

# Measured, not assumed. Source artifacts:
# experiments/kvcapacity_step_scaling/step-scaling-baseline-pool640-20260914-233549
#
# c0 is anchored on the L=8192 fit. The L=2048 fit puts it at 12.839 instead, so a
# single c0 cannot honour both: extrapolating the L=8192 anchor to shorter contexts
# is optimistic by about 3% (it predicts 38.93 ms at L=2048 B=70 against 40.148 ms
# measured). That bias runs in favour of the token-count axis, which is exactly the
# axis this tool is used to argue for, so it is stated rather than tuned away.
C0_MS = 12.113
C0_MS_AT_2048 = 12.839
# Fixed denominator for every gain. Using the model to recompute its own baseline
# would let model error and the unresolved `a` leak into the ratios; the measured
# wall throughput is a fact and does not move when `a` is re-estimated.
MEASURED_BASELINE_SEQ_PER_S = 482.74
C1_US_PER_TOKEN = 0.1667
KV_BYTES_PER_TOKEN = 147456
KVCACHE_BLOCK_SIZE = 256
POOL_BLOCKS = 640
SCHEDULER_RESERVE_BLOCKS = 8
GENERATED_TOKENS = 34
A100_HBM_PEAK_GB_S = 2039.0

# The two disagreeing estimates of `a`, both from experiments that were not
# designed to identify it. Neither is trusted; both are carried so the spread is
# visible in every report until a calibration run settles it.
A_CANDIDATES = {
    "two_point_extrapolation": 0.0417,
    "gate_a_m2_fit": 0.238,
}


def blocks_per_sequence(context_length, *, generated_tokens=GENERATED_TOKENS,
                        block_size=KVCACHE_BLOCK_SIZE):
    """Whole KV blocks a sequence pins, the way the allocator charges it."""
    return -(-(int(context_length) + int(generated_tokens)) // int(block_size))


def wall_batch(context_length, *, byte_fraction=1.0, pool_blocks=POOL_BLOCKS,
               reserve_blocks=SCHEDULER_RESERVE_BLOCKS):
    """Largest batch the byte budget admits.

    Removing bytes per token lets the same *byte* budget hold more blocks, which
    is how a byte-axis compression converts into concurrency. Removing tokens
    instead shrinks blocks per sequence. A proposal that does both compounds,
    which is the whole argument for latent representations over quantisation.
    """
    usable = pool_blocks / float(byte_fraction) - reserve_blocks
    return max(1, int(usable // blocks_per_sequence(context_length)))


def step_ms(context_length, batch, *, a_ms_per_seq, byte_fraction=1.0,
            c0_ms=C0_MS, c1_us_per_token=C1_US_PER_TOKEN):
    slope = a_ms_per_seq + (c1_us_per_token / 1000.0) * float(byte_fraction) * context_length
    return c0_ms + slope * batch


def throughput_seq_per_s(context_length, batch, **kwargs):
    return batch / step_ms(context_length, batch, **kwargs) * 1000.0


def ceiling_seq_per_s(a_ms_per_seq):
    """Throughput as B -> infinity with the per-token term driven to zero."""
    return 1000.0 / float(a_ms_per_seq)


def effective_kv_bandwidth_gb_s(c1_us_per_token=C1_US_PER_TOKEN,
                                bytes_per_token=KV_BYTES_PER_TOKEN):
    return bytes_per_token / (c1_us_per_token * 1e-6) / 1e9


def price(proposals, *, a_ms_per_seq, baseline_seq_per_s=MEASURED_BASELINE_SEQ_PER_S):
    base_tp = float(baseline_seq_per_s)
    priced = []
    for label, context_length, byte_fraction in proposals:
        batch = wall_batch(context_length, byte_fraction=byte_fraction)
        tp = throughput_seq_per_s(
            context_length, batch,
            a_ms_per_seq=a_ms_per_seq, byte_fraction=byte_fraction,
        )
        priced.append(
            {
                "label": label,
                "context_length": context_length,
                "byte_fraction": byte_fraction,
                "wall_batch": batch,
                "throughput_seq_per_s": round(tp, 2),
                "gain_over_baseline": round(tp / base_tp, 3),
            }
        )
    return {
        "a_ms_per_seq": a_ms_per_seq,
        "ceiling_seq_per_s": round(ceiling_seq_per_s(a_ms_per_seq), 1),
        "baseline_throughput_seq_per_s": round(base_tp, 2),
        "ceiling_gain_over_baseline": round(ceiling_seq_per_s(a_ms_per_seq) / base_tp, 2),
        "proposals": priced,
    }


DEFAULT_PROPOSALS = [
    ("baseline L=8192 bf16", 8192, 1.0),
    ("2x fewer bytes/token, same L", 8192, 0.5),
    ("4x fewer tokens (L->2048), same bytes", 2048, 1.0),
    ("4x fewer tokens AND 2x fewer bytes", 2048, 0.5),
]


def render(report):
    lines = []
    lines.append("pricing latent proposals against the measured decode model")
    lines.append("=" * 72)
    lines.append(
        f"a = {report['a_ms_per_seq']} ms/seq  ->  ceiling "
        f"{report['ceiling_seq_per_s']:.0f} seq/s "
        f"= {report['ceiling_gain_over_baseline']:.1f}x over baseline"
    )
    lines.append("")
    lines.append("  proposal                                    B_wall    seq/s    gain")
    for item in report["proposals"]:
        lines.append(
            "  %-42s %6d %8.0f %6.2fx"
            % (
                item["label"],
                item["wall_batch"],
                item["throughput_seq_per_s"],
                item["gain_over_baseline"],
            )
        )
    lines.append("")
    bw = effective_kv_bandwidth_gb_s()
    lines.append(
        f"effective KV read bandwidth {bw:.0f} GB/s = "
        f"{100 * bw / A100_HBM_PEAK_GB_S:.0f}% of A100 HBM peak"
    )
    lines.append(
        "  so removing bytes does not remove time one-for-one; there is roughly "
        "2x of kernel-efficiency slack ahead of any compression win"
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a-ms-per-seq",
        type=float,
        help=(
            "the per-sequence term compression cannot touch; if omitted, every "
            "candidate estimate is reported so the unresolved spread stays visible"
        ),
    )
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    if args.a_ms_per_seq is not None:
        reports = {"provided": price(DEFAULT_PROPOSALS, a_ms_per_seq=args.a_ms_per_seq)}
    else:
        reports = {
            name: price(DEFAULT_PROPOSALS, a_ms_per_seq=value)
            for name, value in A_CANDIDATES.items()
        }

    for name, report in reports.items():
        print(f"=== a estimate: {name}")
        print(render(report))
        print()
    if len(reports) > 1:
        gains = [r["ceiling_gain_over_baseline"] for r in reports.values()]
        print(
            f"UNRESOLVED: the ceiling is somewhere between {min(gains):.1f}x and "
            f"{max(gains):.1f}x. Calibrate `a` before committing to this line."
        )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
