"""Can the per-layer handshake stop costing 1.355 ms?

The overlap gate measured a synchronous per-layer handshake at 1.355 ms/step (37.6 us x 36
layers) while the same bytes fused into one transfer pair cost 0.068 ms. That 20x gap is
pure per-handshake latency, so it should be recoverable by removing host synchronisations
rather than by moving fewer bytes. This tool measures how much is actually recoverable.

It also fixes a direction error in the earlier gate: Q is produced on the GPU and consumed
by the CPU, so it travels D2H, and the attention output travels H2D. The earlier gate had
them swapped. The total is similar because both directions appear either way (D2H+sync
11.8 us, H2D+sync 14.3 us measured), but the arms below use the correct direction.

Variants, in increasing sophistication:

  device_sync    torch.cuda.synchronize() after each copy - the earlier baseline
  event_sync     copies on a side stream, host waits on a cuda event instead of the device
  stream_sync    copies on a side stream, host waits on the stream
  lookahead      Q for layer i+1 is issued during layer i, so the host waits on an event
                 that has usually already completed, and the output H2D is made visible to
                 the compute stream with a device-side wait_event instead of a host sync
  fused          all layers in one pair - forbidden by the data dependency, kept as floor

What is *not* measured here: the CPU attention itself (that is the CPU gate) and the GPU
layer work (that is the pipeline prototype). This isolates the coordination cost.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

VARIANTS = ("device_sync", "event_sync", "stream_sync", "lookahead", "fused")


def host_syncs_per_step(variant: str, n_layers: int) -> int:
    """How many times the host blocks per step. This is the quantity being optimised."""
    if variant == "device_sync":
        return 2 * n_layers          # one after Q down, one after output up
    if variant in ("event_sync", "stream_sync"):
        return 2 * n_layers          # same count, cheaper primitive
    if variant == "lookahead":
        return n_layers              # only the Q arrival; the output is a device-side wait
    if variant == "fused":
        return 2
    raise ValueError(variant)


def _make_buffers(n_layers: int, q_elems: int, out_elems: int, device: str):
    import torch
    return {
        # Q lives on the GPU and must reach the CPU: device -> pinned host
        "q_dev": torch.randn(n_layers, q_elems, dtype=torch.float32, device=device),
        "q_host": torch.empty(n_layers, q_elems, dtype=torch.float32).pin_memory(),
        # the attention output is produced on the CPU and must reach the GPU
        "out_host": torch.randn(n_layers, out_elems, dtype=torch.float32).pin_memory(),
        "out_dev": torch.empty(n_layers, out_elems, dtype=torch.float32, device=device),
    }


def _run_variant(variant: str, buf: dict, n_layers: int, device: str):
    import torch

    q_dev, q_host = buf["q_dev"], buf["q_host"]
    out_host, out_dev = buf["out_host"], buf["out_dev"]

    if variant == "device_sync":
        def step():
            for l in range(n_layers):
                q_host[l].copy_(q_dev[l], non_blocking=True)
                torch.cuda.synchronize()
                out_dev[l].copy_(out_host[l], non_blocking=True)
                torch.cuda.synchronize()
        return step

    if variant == "fused":
        def step():
            q_host.copy_(q_dev, non_blocking=True)
            torch.cuda.synchronize()
            out_dev.copy_(out_host, non_blocking=True)
            torch.cuda.synchronize()
        return step

    copy_stream = torch.cuda.Stream()

    if variant == "event_sync":
        events = [torch.cuda.Event() for _ in range(2 * n_layers)]

        def step():
            for l in range(n_layers):
                with torch.cuda.stream(copy_stream):
                    q_host[l].copy_(q_dev[l], non_blocking=True)
                    events[2 * l].record(copy_stream)
                events[2 * l].synchronize()
                with torch.cuda.stream(copy_stream):
                    out_dev[l].copy_(out_host[l], non_blocking=True)
                    events[2 * l + 1].record(copy_stream)
                events[2 * l + 1].synchronize()
        return step

    if variant == "stream_sync":
        def step():
            for l in range(n_layers):
                with torch.cuda.stream(copy_stream):
                    q_host[l].copy_(q_dev[l], non_blocking=True)
                copy_stream.synchronize()
                with torch.cuda.stream(copy_stream):
                    out_dev[l].copy_(out_host[l], non_blocking=True)
                copy_stream.synchronize()
        return step

    if variant == "lookahead":
        q_ready = [torch.cuda.Event() for _ in range(n_layers)]
        out_ready = [torch.cuda.Event() for _ in range(n_layers)]
        compute_stream = torch.cuda.current_stream()

        def step():
            # prime the pipeline: layer 0's Q goes down before the loop starts, which in the
            # engine means it is issued at the end of the previous step
            with torch.cuda.stream(copy_stream):
                q_host[0].copy_(q_dev[0], non_blocking=True)
                q_ready[0].record(copy_stream)
            for l in range(n_layers):
                if l + 1 < n_layers:
                    # issue the next layer's Q while this layer is still being handled
                    with torch.cuda.stream(copy_stream):
                        q_host[l + 1].copy_(q_dev[l + 1], non_blocking=True)
                        q_ready[l + 1].record(copy_stream)
                # the only place the host has to block: the CPU cannot start without Q
                q_ready[l].synchronize()
                # ... CPU attention would run here; its cost is measured by the CPU gate
                with torch.cuda.stream(copy_stream):
                    out_dev[l].copy_(out_host[l], non_blocking=True)
                    out_ready[l].record(copy_stream)
                # device-side wait: the consumer kernel is ordered after the copy without
                # the host ever blocking
                compute_stream.wait_event(out_ready[l])
            torch.cuda.synchronize()
        return step

    raise ValueError(variant)


def measure(n_layers: int, q_heads: int, dim: int, device: str, iters: int,
            warmup: int) -> dict:
    import torch

    q_elems = q_heads * dim
    out_elems = q_heads * dim
    buf = _make_buffers(n_layers, q_elems, out_elems, device)

    results = {}
    for variant in VARIANTS:
        step = _run_variant(variant, buf, n_layers, device)
        for _ in range(warmup):
            step()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            step()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        results[variant] = {
            "mean_ms": round(statistics.mean(samples), 4),
            "median_ms": round(statistics.median(samples), 4),
            "p90_ms": round(samples[min(len(samples) - 1, int(0.9 * len(samples)))], 4),
            "per_layer_us": round(statistics.median(samples) * 1e3 / n_layers, 2),
            "host_syncs_per_step": host_syncs_per_step(variant, n_layers),
        }
    base = results["device_sync"]["median_ms"]
    floor = results["fused"]["median_ms"]
    for variant, r in results.items():
        r["speedup_vs_device_sync"] = round(base / r["median_ms"], 2)
        # how much of the theoretically removable cost this variant actually removed
        removable = base - floor
        r["removable_fraction_captured"] = (
            round((base - r["median_ms"]) / removable, 3) if removable > 0 else None
        )
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    results = measure(args.layers, args.q_heads, args.dim, args.device,
                      args.iters, args.warmup)
    payload = {"layers": args.layers, "q_heads": args.q_heads, "dim": args.dim,
               "bytes_per_step": 2 * args.layers * args.q_heads * args.dim * 4,
               "variants": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    for variant in VARIANTS:
        r = results[variant]
        print(f"{variant:14s} {r['median_ms']:8.4f} ms  {r['per_layer_us']:7.2f} us/layer  "
              f"syncs={r['host_syncs_per_step']:3d}  x{r['speedup_vs_device_sync']}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
