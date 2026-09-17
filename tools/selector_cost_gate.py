"""What does the selector itself cost at gran=32?

Every gate so far has priced the *consequences* of the selector (how much KV has to be
gathered, how much attention has to be computed) and none has priced the selector. That is
a real hole: moving from gran=256 to gran=32 multiplies the number of units by 8, and the
end-to-end gate made gran=32 mandatory. If Quest scoring costs more than the attention it
saves, the plan is circular.

Two costs are measured separately, because they land in different places:

  summary maintenance  after each decode step the new key must be folded into its unit's
                       running min/max. O(kv_heads * dim) per layer per step, independent of
                       context length - cheap by construction, but measured rather than
                       assumed.
  scoring + top-k      for every layer and unit, the Quest bound
                       sum_d max(q_d * kmin_d, q_d * kmax_d), then a top-k over units.
                       This is the term that scales with U = seq / granularity.

The scoring is the shared-head formulation the end-to-end gate selected: the per-head bounds
are summed over kv heads into one ranking, and the query representative is the group-wise
amax. Both placements are timed - on the GPU (where Q already lives) and on the CPU (where
the KV summaries would live next to the KV) - because that placement decision is exactly
what this measurement is for.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time


def quest_scores_shared(q_rep, kmin, kmax):
    """Shared-head Quest bound.

    q_rep: [kv_heads, dim] group-wise amax of the query
    kmin/kmax: [units, kv_heads, dim]
    returns: [units]

    max(q*kmin, q*kmax) is q_pos*kmax + q_neg*kmin, which avoids materialising both products
    and is what a real kernel would do.
    """
    import torch
    q_pos = q_rep.clamp(min=0)
    q_neg = q_rep.clamp(max=0)
    return (kmax * q_pos + kmin * q_neg).sum(dim=(1, 2))


def reference_scores_shared(q_rep, kmin, kmax):
    """Literal max(q*kmin, q*kmax) sum, used only to verify the clamp trick."""
    import torch
    prod = torch.stack([q_rep * kmin, q_rep * kmax])
    return prod.max(dim=0).values.sum(dim=(1, 2))


def _sync(device: str):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def _time(fn, device: str, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {"median_ms": round(statistics.median(samples), 4),
            "mean_ms": round(statistics.mean(samples), 4),
            "p90_ms": round(samples[min(len(samples) - 1, int(0.9 * len(samples)))], 4)}


def measure_placement(device: str, layers: int, seq: int, kv_heads: int, group_size: int,
                      dim: int, granularity: int, k_frac: float, iters: int, warmup: int,
                      cpu_threads: int | None = None, summary_dtype: str = "float32") -> dict:
    import torch

    if device == "cpu" and cpu_threads:
        torch.set_num_threads(cpu_threads)

    units = (seq + granularity - 1) // granularity
    k = max(2, int(round(k_frac * units)))
    dev = torch.device(device)
    dtype = getattr(torch, summary_dtype)

    # summaries for the whole model, laid out [layers, units, kv_heads, dim].
    # The dtype is swept because the first laptop smoke showed the scoring running at ~2.7
    # GFLOP/s, i.e. bound by reading these summaries rather than by arithmetic: at gran=32
    # they are 2 x layers x units x kv_heads x dim x 4 B = 75 MB per sequence per step in
    # fp32, which is real DRAM traffic on top of the KV itself.
    kmin = torch.randn(layers, units, kv_heads, dim, device=dev, dtype=dtype)
    kmax = kmin + torch.rand_like(kmin)
    q = torch.randn(layers, kv_heads, group_size, dim, device=dev, dtype=dtype)
    # pre-allocated so the maintenance timing measures the fold, not an allocation
    new_key = torch.randn(layers, kv_heads, dim, device=dev, dtype=dtype)

    def score_all_layers():
        q_rep = q.amax(dim=2)                                   # [layers, kv_heads, dim]
        q_pos = q_rep.clamp(min=0).unsqueeze(1)
        q_neg = q_rep.clamp(max=0).unsqueeze(1)
        scores = (kmax * q_pos + kmin * q_neg).sum(dim=(2, 3))  # [layers, units]
        return torch.topk(scores, k, dim=-1).indices

    idx = units - 1

    def maintain_summaries():
        # fold one new key per layer into its unit's running min/max
        torch.minimum(kmin[:, idx], new_key, out=kmin[:, idx])
        torch.maximum(kmax[:, idx], new_key, out=kmax[:, idx])

    summary_bytes = kmin.numel() * kmin.element_size() * 2
    out = {
        "device": device,
        "cpu_threads": cpu_threads,
        "summary_dtype": summary_dtype,
        "units": units,
        "top_k": k,
        "granularity": granularity,
        "summary_bytes": summary_bytes,
        "scoring_ms": _time(score_all_layers, device, iters, warmup),
        "summary_maintenance_ms": _time(maintain_summaries, device, iters, warmup),
    }
    out["selector_total_ms"] = round(
        out["scoring_ms"]["median_ms"] + out["summary_maintenance_ms"]["median_ms"], 4)
    # the scoring must read every summary once, so this is the bandwidth it achieves; if it
    # is close to the machine's streaming peak then the selector is a memory problem and no
    # amount of arithmetic tuning will help
    ms = out["scoring_ms"]["median_ms"]
    out["scoring_gb_s"] = round(summary_bytes / (ms * 1e-3) / 1e9, 2) if ms else None
    return out


def verify_bound_math(device: str = "cpu") -> dict:
    """The clamp trick must equal the literal max, or the whole measurement prices the
    wrong kernel."""
    import torch
    torch.manual_seed(0)
    q_rep = torch.randn(4, 8, device=device)
    kmin = torch.randn(16, 4, 8, device=device)
    kmax = kmin + torch.rand_like(kmin)
    a = quest_scores_shared(q_rep, kmin, kmax)
    b = reference_scores_shared(q_rep, kmin, kmax)
    err = float((a - b).abs().max())
    return {"max_abs_err": err, "ok": err < 1e-4}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--granularities", type=int, nargs="+", default=[32, 256])
    p.add_argument("--k-frac", type=float, default=0.111,
                   help="trajectory-level budget from the end-to-end gate")
    p.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 8, 64])
    p.add_argument("--summary-dtypes", nargs="+", default=["float32", "bfloat16"],
                   help="bf16 halves the summary traffic, which is the suspected bottleneck")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--skip-cuda", action="store_true")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    payload = {"shape": {"layers": args.layers, "seq": args.seq, "kv_heads": args.kv_heads,
                         "group_size": args.group_size, "dim": args.dim,
                         "k_frac": args.k_frac},
               "bound_math_check": verify_bound_math(),
               "measurements": []}
    assert payload["bound_math_check"]["ok"], payload["bound_math_check"]

    def report(label, r):
        print(f"gran={r['granularity']:4d} {label:18s} units={r['units']:5d} "
              f"score={r['scoring_ms']['median_ms']:8.4f} ms "
              f"maint={r['summary_maintenance_ms']['median_ms']:.4f} ms "
              f"total={r['selector_total_ms']:8.4f} ms "
              f"summary={r['summary_bytes'] / (1 << 20):6.1f} MiB "
              f"{r['scoring_gb_s']:7.2f} GB/s", flush=True)

    for gran in args.granularities:
        for dt in args.summary_dtypes:
            if not args.skip_cuda:
                r = measure_placement("cuda", args.layers, args.seq, args.kv_heads,
                                      args.group_size, args.dim, gran, args.k_frac,
                                      args.iters, args.warmup, summary_dtype=dt)
                payload["measurements"].append(r)
                report(f"cuda/{dt}", r)
            for threads in args.cpu_threads:
                r = measure_placement("cpu", args.layers, args.seq, args.kv_heads,
                                      args.group_size, args.dim, gran, args.k_frac,
                                      args.iters, args.warmup, cpu_threads=threads,
                                      summary_dtype=dt)
                payload["measurements"].append(r)
                report(f"cpu{threads}/{dt}", r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
