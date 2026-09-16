"""Overlap gate: the CPU can compute it, but can it compute it *while* the GPU works?

Why this file exists
--------------------
`tools/cpu_sparse_attention_bench.c` showed the per-step CPU attention costs 0.42 ms
(answer-level budget) to 2.05 ms (trajectory-level) against a 12.113 ms GPU weight-read
window. That is a comparison of two numbers measured on an idle-ish counterpart, and it
quietly assumes three things that have to be measured instead:

1. that a busy CPU does not slow the GPU down. The CUDA launch thread lives on the same
   host, and 64 OpenMP threads spinning on DRAM is exactly the neighbour that starves it.
2. that the Q/output round trip is free. Per layer the GPU must ship Q down and read an
   attention output back, so a step contains 2 x n_layers small transfers *and* the
   synchronisation between them. Latency, not bandwidth, is the risk: 72 round trips at
   20 us would eat 1.4 ms of the window before any attention is computed.
3. that there is something to overlap with at all. Within one sequence, layer i's
   attention sits between qkv_proj and o_proj, so it is on the critical path - the only
   available overlap is against *other* work (another micro-batch, or the next layer's
   weight read). This tool measures the ingredients; it does not pretend the naive
   single-sequence schedule hides anything.

Modes:
  gpu-baseline      decode steps on the GPU alone, dense attention
  gpu-under-load    the same, while the CPU attention benchmark saturates the cores
  pcie              per-layer Q down / output up round trips with synchronisation,
                    plus a fused variant as the unreachable floor
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

C0_MS = 12.113          # GPU weight-read window, from the latent pricing model


def summarize(gpu_alone_ms: float, gpu_under_load_ms: float, cpu_attn_ms: float,
              pcie_per_layer_ms: float, window_ms: float = C0_MS) -> dict:
    """Turn four measurements into a verdict about the schedule.

    The budget is not `cpu_attn <= window`: the CPU time only helps if it is spent while
    the GPU is doing something else, and the transfers and the GPU slowdown are charged
    unconditionally. So the honest accounting is

        overhead = (gpu_under_load - gpu_alone) + pcie_round_trips
        headroom = window - overhead - cpu_attn

    A negative headroom means the offload cannot be hidden even with perfect overlap.
    """
    inflation_ms = gpu_under_load_ms - gpu_alone_ms
    overhead_ms = inflation_ms + pcie_per_layer_ms
    headroom_ms = window_ms - overhead_ms - cpu_attn_ms
    return {
        "window_ms": window_ms,
        "gpu_alone_ms": round(gpu_alone_ms, 3),
        "gpu_under_cpu_load_ms": round(gpu_under_load_ms, 3),
        "gpu_inflation_ms": round(inflation_ms, 3),
        "gpu_inflation_pct": round(100.0 * inflation_ms / gpu_alone_ms, 2) if gpu_alone_ms else None,
        "pcie_round_trip_ms": round(pcie_per_layer_ms, 3),
        "cpu_attention_ms": round(cpu_attn_ms, 3),
        "unconditional_overhead_ms": round(overhead_ms, 3),
        "headroom_ms": round(headroom_ms, 3),
        "verdict": "HIDEABLE" if headroom_ms > 0 else "NOT_HIDEABLE",
        "note": ("headroom assumes perfect overlap of the CPU attention with independent "
                 "GPU work; within a single sequence layer i's attention is on the critical "
                 "path and this headroom is not available"),
    }


# ---------------------------------------------------------------------------
# GPU decode loop
# ---------------------------------------------------------------------------

def measure_gpu_decode(model_path: str, seq_len: int, steps: int, device: str,
                       warmup: int = 5) -> dict:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # this function is called twice in one process; without this the caching allocator
    # keeps the first model's 16 GB reserved and the second load can OOM on a shared box
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    ids = torch.randint(1000, 20000, (1, seq_len), device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    cur = out.logits[0, -1].argmax().view(1, 1)
    pos = seq_len

    per_step = []
    with torch.inference_mode():
        for i in range(warmup + steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = model(input_ids=cur, past_key_values=past, use_cache=True,
                        cache_position=torch.tensor([pos], device=device))
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            past = res.past_key_values
            cur = res.logits[0, -1].argmax().view(1, 1)
            pos += 1
            if i >= warmup:
                per_step.append(dt)
    per_step.sort()
    n = len(per_step)
    result = {
        "steps": n,
        "mean_ms": sum(per_step) / n,
        "median_ms": per_step[n // 2],
        "p10_ms": per_step[max(0, int(0.1 * n))],
        "p90_ms": per_step[min(n - 1, int(0.9 * n))],
        "n_layers": model.config.num_hidden_layers,
        "tokenizer_vocab": len(tokenizer),
    }
    del past, cur, out, res, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# PCIe round trip
# ---------------------------------------------------------------------------

def measure_pcie(n_layers: int, q_heads: int, dim: int, device: str, iters: int = 50,
                 warmup: int = 5) -> dict:
    """Per-layer Q down / output up, with the synchronisation a real handoff needs.

    Two variants are timed:
      per_layer  what the schedule actually requires: 2 * n_layers transfers plus a sync
                 per layer, because layer i cannot proceed until its output is back
      fused      all layers in one transfer pair, which the data dependency forbids;
                 measured only to separate latency cost from bandwidth cost
    """
    import torch

    elems = q_heads * dim
    q_host = torch.empty(n_layers, elems, dtype=torch.float32).pin_memory()
    out_host = torch.empty(n_layers, elems, dtype=torch.float32).pin_memory()
    q_dev = torch.empty(n_layers, elems, dtype=torch.float32, device=device)
    out_dev = torch.empty(n_layers, elems, dtype=torch.float32, device=device)

    def per_layer_step():
        for l in range(n_layers):
            q_dev[l].copy_(q_host[l], non_blocking=True)
            torch.cuda.synchronize()
            out_host[l].copy_(out_dev[l], non_blocking=True)
            torch.cuda.synchronize()

    def fused_step():
        q_dev.copy_(q_host, non_blocking=True)
        torch.cuda.synchronize()
        out_host.copy_(out_dev, non_blocking=True)
        torch.cuda.synchronize()

    results = {}
    for name, fn in (("per_layer", per_layer_step), ("fused", fused_step)):
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        results[name] = {
            "mean_ms": sum(samples) / len(samples),
            "median_ms": samples[len(samples) // 2],
            "bytes_per_step": 2 * n_layers * elems * 4,
        }
    results["per_layer"]["per_round_trip_us"] = \
        results["per_layer"]["mean_ms"] * 1e3 / n_layers
    return results


# ---------------------------------------------------------------------------
# CPU load
# ---------------------------------------------------------------------------

class CpuLoad:
    """Runs the compiled CPU attention benchmark for the duration of a measurement."""

    def __init__(self, binary: str, args: list, enabled: bool = True):
        self.binary = binary
        self.args = args
        self.enabled = enabled
        self.proc = None

    def __enter__(self):
        if self.enabled:
            self.proc = subprocess.Popen([self.binary] + self.args,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            time.sleep(2.0)   # let the pool allocation and first-touch finish
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-bench-binary", required=True)
    p.add_argument("--cpu-threads", type=int, default=64)
    p.add_argument("--cpu-tokens-per-step", type=int, default=912)
    p.add_argument("--cpu-attn-ms", type=float, required=True,
                   help="measured CPU attention ms/step at this budget (from the CPU gate)")
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    payload = {"model": args.model, "seq_len": args.seq_len, "steps": args.steps,
               "cpu_threads": args.cpu_threads,
               "cpu_tokens_per_step": args.cpu_tokens_per_step}

    # PCIe first, in a clean process: measured after two 16 GB model loads the fused
    # variant came out 16x slower than it does here, so the transfer numbers are only
    # trustworthy before the allocator has been churned
    print("[1/4] PCIe round trips", flush=True)
    pcie = measure_pcie(36, args.q_heads, args.dim, args.device)
    payload["pcie"] = pcie
    print(f"      per_layer mean={pcie['per_layer']['mean_ms']:.3f} ms "
          f"({pcie['per_layer']['per_round_trip_us']:.1f} us per round trip), "
          f"fused mean={pcie['fused']['mean_ms']:.3f} ms", flush=True)

    print("[2/4] GPU decode alone", flush=True)
    alone = measure_gpu_decode(args.model, args.seq_len, args.steps, args.device)
    payload["gpu_alone"] = alone
    print(f"      mean={alone['mean_ms']:.3f} ms median={alone['median_ms']:.3f}", flush=True)

    n_layers = alone["n_layers"]
    payload["pcie_n_layers_assumed"] = 36
    payload["model_n_layers"] = n_layers

    print("[3/4] GPU decode while the CPU attention benchmark saturates the cores", flush=True)
    load_args = ["--layers", str(n_layers), "--seq", str(args.seq_len),
                 "--tokens-per-step", str(args.cpu_tokens_per_step),
                 "--threads", str(args.cpu_threads), "--iters", "100000", "--warmup", "0"]
    with CpuLoad(args.cpu_bench_binary, load_args):
        under = measure_gpu_decode(args.model, args.seq_len, args.steps, args.device)
    payload["gpu_under_cpu_load"] = under
    print(f"      mean={under['mean_ms']:.3f} ms median={under['median_ms']:.3f}", flush=True)

    print("[4/4] verdict", flush=True)
    payload["summary"] = summarize(alone["median_ms"], under["median_ms"],
                                   args.cpu_attn_ms, pcie["per_layer"]["mean_ms"])
    payload["summary"]["inflation_source"] = (
        "HF decode loop, which is launch-bound at ~34 ms/step and therefore more "
        "sensitive to host contention than the engine; the engine's own inflation is "
        "measured by tools/engine_step_cpu_interference.py and should replace this term")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
