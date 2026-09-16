"""Does 64 busy CPU cores slow down the *engine's* decode step?

`cpu_gpu_overlap_gate.py` measured this on an HF python decode loop, which turned out to
cost 33.8 ms/step - nearly 3x the 12.113 ms window the cost model is built on. That loop
is launch-bound, so its sensitivity to a busy host is not the engine's sensitivity, and an
absolute inflation measured there cannot be transplanted onto the engine's window.

This tool measures the same interference on tinyvllm itself, with CUDA graphs on, so the
number can be subtracted from c0 honestly.

Prefill is removed by slope rather than by guessing: the same prompt length is decoded for
n_short and n_long tokens and the per-step cost is the slope between them, so whatever the
prefill and the fixed generate() overhead cost, they cancel. Prompts are freshly randomised
per measurement so prefix caching cannot make the long run's prefill cheaper than the short
run's and corrupt the slope.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from random import randint, seed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cpu_gpu_overlap_gate import C0_MS, CpuLoad   # noqa: E402


def _fresh_prompt(seq_len: int) -> list:
    return [randint(0, 10000) for _ in range(seq_len)]


def measure_engine_step(llm, sampling_cls, seq_len: int, n_short: int, n_long: int,
                        repeats: int) -> dict:
    slopes, short_ms, long_ms = [], [], []
    for _ in range(repeats):
        timings = {}
        for tag, n_tok in (("short", n_short), ("long", n_long)):
            prompt = _fresh_prompt(seq_len)
            sp = sampling_cls(temperature=0.0, ignore_eos=True, max_tokens=n_tok)
            t0 = time.perf_counter()
            llm.generate([prompt], [sp], use_tqdm=False)
            timings[tag] = (time.perf_counter() - t0) * 1e3
        slopes.append((timings["long"] - timings["short"]) / (n_long - n_short))
        short_ms.append(timings["short"])
        long_ms.append(timings["long"])
    return {
        "step_ms_median": statistics.median(slopes),
        "step_ms_all": [round(s, 3) for s in slopes],
        "short_total_ms": [round(v, 1) for v in short_ms],
        "long_total_ms": [round(v, 1) for v in long_ms],
        "n_short": n_short,
        "n_long": n_long,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--n-short", type=int, default=16)
    p.add_argument("--n-long", type=int, default=80)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-bench-binary", required=True)
    p.add_argument("--cpu-threads", type=int, default=64)
    p.add_argument("--cpu-tokens-per-step", type=int, default=912)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    seed(0)
    from tinyvllm import LLM, SamplingParams

    llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=1,
              max_model_len=args.seq_len + args.n_long + 64)
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)

    load_args = ["--layers", str(args.layers), "--seq", str(args.seq_len),
                 "--tokens-per-step", str(args.cpu_tokens_per_step),
                 "--threads", str(args.cpu_threads), "--iters", "100000", "--warmup", "0"]

    print("[1/2] engine decode alone", flush=True)
    alone = measure_engine_step(llm, SamplingParams, args.seq_len,
                                args.n_short, args.n_long, args.repeats)
    print(f"      step={alone['step_ms_median']:.3f} ms {alone['step_ms_all']}", flush=True)

    print("[2/2] engine decode while the CPU attention benchmark saturates the cores",
          flush=True)
    with CpuLoad(args.cpu_bench_binary, load_args):
        under = measure_engine_step(llm, SamplingParams, args.seq_len,
                                    args.n_short, args.n_long, args.repeats)
    print(f"      step={under['step_ms_median']:.3f} ms {under['step_ms_all']}", flush=True)

    infl = under["step_ms_median"] - alone["step_ms_median"]
    payload = {
        "model": args.model,
        "seq_len": args.seq_len,
        "cpu_threads": args.cpu_threads,
        "cpu_tokens_per_step": args.cpu_tokens_per_step,
        "engine_alone": alone,
        "engine_under_cpu_load": under,
        "inflation_ms": round(infl, 3),
        "inflation_pct": round(100.0 * infl / alone["step_ms_median"], 2),
        "c0_reference_ms": C0_MS,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: payload[k] for k in
                      ("inflation_ms", "inflation_pct", "c0_reference_ms")}, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
