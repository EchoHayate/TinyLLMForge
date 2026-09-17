"""Does cross-microbatch pipelining turn the headroom into a real saving?

The overlap gate showed 7.18 ms of nominal headroom but also said plainly that inside one
sequence the attention of layer i sits between qkv_proj and o_proj, so nothing is hidden by
wishing. The only schedule that can collect the headroom is the one FastDecode and NEO use:
while the GPU runs microbatch A's layers, the CPU computes microbatch B's attention.

This prototype measures whether that holds on this machine, with:

  - a synthetic GPU workload calibrated to the weight-read window (a GEMV per layer over a
    weight slab, i.e. the same bandwidth-bound shape a decode step has)
  - the *real* CPU attention kernel, linked as a shared library from the same translation
    unit that produced the measured 2.051 ms/step, submitted asynchronously *from C*
    (csa_submit/csa_wait, pthread + condvar). The first version of this prototype drove the
    kernel from a Python thread and measured pipelining as *worse* than serial; at a ~2 ms
    effect size, Python thread scheduling is the finding, not the mechanism. The Python-thread
    arm is kept as a control (`pipelined_pythread`) precisely because it is a false negative.

Arms:
  gpu_only    the GPU pass alone
  cpu_only    the CPU attention step alone
  serial      GPU pass, then CPU step - what a naive per-layer synchronous offload pays
  pipelined   CPU step for microbatch B concurrent with the GPU pass for microbatch A (C async)
  pipelined_pythread  the same, orchestrated from a Python thread - kept as a false-negative control

The second trap lives in --gpu-wait. torch.cuda.synchronize() *spins*, and in graph mode that
spin is the only thing the host does for ~11 ms, with no preemption point, beside a 64-thread
OpenMP team whose parallel-for ends in a barrier. Measured on a loaded host: graph+spin hides
-2.3% of the CPU time (i.e. exactly serial), graph+blocking-event hides 97.5%. Same kernel,
same replay, same threads. An engine that waits by polling cannot pipeline a CPU offload.

The claim under test is pipelined ~= max(gpu_only, cpu_only). The gap between that ideal and
the measurement is interference the schedule cannot avoid, and it decides whether this
mechanism is worth building at all.

Also swept: how many concurrent sequences the CPU can serve, since the CPU cost is
per-sequence while the GPU's weight read is amortised over the batch. That is the ceiling
that matters, because raising the batch is the entire point of offloading KV.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import threading
import time


def load_kernel(so_path: str):
    lib = ctypes.CDLL(so_path)
    lib.csa_create.restype = ctypes.c_void_p
    lib.csa_create.argtypes = [ctypes.c_int] * 7
    lib.csa_step.restype = ctypes.c_double
    lib.csa_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.csa_destroy.argtypes = [ctypes.c_void_p]
    # async submission, implemented in C so the overlap does not depend on the GIL
    lib.csa_submit.restype = ctypes.c_int
    lib.csa_submit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.csa_wait.restype = ctypes.c_double
    lib.csa_wait.argtypes = [ctypes.c_void_p]
    return lib


class GpuWork:
    """A bandwidth-bound stand-in for one decode step's weight reads.

    Two modes, because the first run of this prototype showed pipelining being *worse* than
    serial and that result is only trustworthy if the GPU arm is not itself starved for host
    CPU. In eager mode the 36 GEMVs are dispatched from Python one at a time, so the driver
    thread needs the host constantly - exactly the condition under which the overlap gate saw
    an HF loop inflate by 47% while the CUDA-graph engine inflated by 11%. Graph mode replays
    the whole pass as one launch, which is what the engine actually does.
    """

    def __init__(self, n_layers: int, weight_gb: float, device: str, mode: str = "eager",
                 wait: str = "spin"):
        import torch
        self.torch = torch
        self.device = device
        self.mode = mode
        # How the host waits for the GPU matters more than expected: torch.cuda.synchronize
        # spins, and in graph mode that spin is the only thing the host does for 11 ms, next
        # to a 64-thread OpenMP team. A blocking event lets the host sleep instead.
        self.wait = wait
        self._done = None
        bytes_per_layer = weight_gb * (1 << 30) / n_layers
        cols = 4096
        rows = max(1, int(bytes_per_layer / (2 * cols)))
        self.weights = [torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
                        for _ in range(n_layers)]
        self.x = torch.randn(cols, dtype=torch.bfloat16, device=device)
        self.actual_gb = sum(w.numel() * 2 for w in self.weights) / (1 << 30)
        self.graph = None
        if mode == "graph":
            self._capture()

    def _pass(self):
        for w in self.weights:
            self.torch.mv(w, self.x)

    def _capture(self):
        torch = self.torch
        # warm the allocator and cublas handles on a side stream before capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._pass()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._pass()

    def run_and_sync(self):
        torch = self.torch
        if self.graph is not None:
            self.graph.replay()
        else:
            self._pass()
        if self.wait == "blocking_event":
            if self._done is None:
                self._done = torch.cuda.Event(blocking=True)
            self._done.record()
            self._done.synchronize()
        else:
            torch.cuda.synchronize()


def _stat(samples):
    samples = sorted(samples)
    return {"median_ms": round(statistics.median(samples), 3),
            "mean_ms": round(statistics.mean(samples), 3),
            "p90_ms": round(samples[min(len(samples) - 1, int(0.9 * len(samples)))], 3)}


def analyse(res: dict, pipe_key: str = "pipelined") -> dict:
    """The three questions the arms answer, kept in one place so the report cannot drift."""
    ideal = max(res["gpu_only"]["median_ms"], res["cpu_only"]["median_ms"])
    pipe = res[pipe_key]["median_ms"]
    serial = res["serial"]["median_ms"]
    cpu = res["cpu_only"]["median_ms"]
    return {
        "ideal_ms": round(ideal, 3),
        "pipelined_ms": pipe,
        # 1.0 means the CPU work became free; 0.5 means half of it still shows up
        "overlap_efficiency": round(ideal / pipe, 3),
        "cost_of_imperfect_overlap_ms": round(pipe - ideal, 3),
        "saving_vs_serial_ms": round(serial - pipe, 3),
        "cpu_time_hidden_pct": round(100.0 * (serial - pipe) / cpu, 1) if cpu else None,
    }


def measure_batch_ceiling(lib, handle, tokens_per_step: int, threads: int,
                          batches: list, iters: int) -> list:
    """CPU cost when it must serve several sequences per step.

    The GPU amortises its weight read over the batch; the CPU does not, because every
    sequence has its own KV and its own selected tokens. So this is where the offload's own
    purpose - a bigger batch - runs into its own cost.
    """
    out = []
    for b in batches:
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for _ in range(b):
                lib.csa_step(handle, tokens_per_step, threads)
            samples.append((time.perf_counter() - t0) * 1e3)
        s = _stat(samples)
        s["batch"] = b
        s["per_seq_ms"] = round(s["median_ms"] / b, 3)
        out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--so", required=True, help="libcpu_sparse_attn.so")
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--granularity", type=int, default=32)
    p.add_argument("--tokens-per-step", type=int, default=912)
    p.add_argument("--cpu-threads", type=int, default=64)
    p.add_argument("--weight-gb", type=float, default=16.0,
                   help="weight bytes read per GPU step; 16 GB ~ Qwen3-8B in bf16")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-mode", default="eager", choices=["eager", "graph"],
                   help="graph replays the whole GPU pass as one launch, like the engine")
    p.add_argument("--gpu-wait", default="spin", choices=["spin", "blocking_event"],
                   help="spin = torch.cuda.synchronize; blocking_event sleeps instead")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    lib = load_kernel(args.so)
    raw = lib.csa_create(args.layers, args.seq, args.kv_heads, args.group_size,
                         args.dim, args.granularity, args.cpu_threads)
    if not raw:
        print("csa_create failed (allocation)")
        return 1
    handle = ctypes.c_void_p(raw)

    gpu = GpuWork(args.layers, args.weight_gb, args.device, mode=args.gpu_mode,
                  wait=args.gpu_wait)
    print(f"gpu workload: {gpu.actual_gb:.2f} GB read per step, mode={args.gpu_mode}, "
          f"wait={args.gpu_wait}", flush=True)

    def cpu_step():
        lib.csa_step(handle, args.tokens_per_step, args.cpu_threads)

    for _ in range(args.warmup):
        gpu.run_and_sync()
        cpu_step()

    gpu_only, cpu_only, serial, pipelined, pipelined_pythread = [], [], [], [], []
    for _ in range(args.iters):
        t0 = time.perf_counter(); gpu.run_and_sync()
        gpu_only.append((time.perf_counter() - t0) * 1e3)
    for _ in range(args.iters):
        t0 = time.perf_counter(); cpu_step()
        cpu_only.append((time.perf_counter() - t0) * 1e3)
    for _ in range(args.iters):
        t0 = time.perf_counter(); gpu.run_and_sync(); cpu_step()
        serial.append((time.perf_counter() - t0) * 1e3)

    # the real arm: submit is a non-blocking C call, the work runs on a detached pthread that
    # never takes the GIL, and wait blocks on a condition variable
    lib.csa_submit(handle, args.tokens_per_step, args.cpu_threads)
    lib.csa_wait(handle)
    for _ in range(args.iters):
        t0 = time.perf_counter()
        lib.csa_submit(handle, args.tokens_per_step, args.cpu_threads)
        gpu.run_and_sync()
        lib.csa_wait(handle)
        pipelined.append((time.perf_counter() - t0) * 1e3)

    # kept as a control, because this is the version that came out worse than serial and the
    # difference between the two arms is the cost of orchestrating from Python
    for _ in range(args.iters):
        t0 = time.perf_counter()
        th = threading.Thread(target=cpu_step)
        th.start()
        gpu.run_and_sync()
        th.join()
        pipelined_pythread.append((time.perf_counter() - t0) * 1e3)

    res = {"gpu_only": _stat(gpu_only), "cpu_only": _stat(cpu_only),
           "serial": _stat(serial), "pipelined": _stat(pipelined),
           "pipelined_pythread": _stat(pipelined_pythread)}
    res["analysis"] = analyse(res)
    res["analysis_pythread"] = analyse(res, pipe_key="pipelined_pythread")

    print("batch ceiling sweep", flush=True)
    res["batch_sweep"] = measure_batch_ceiling(lib, handle, args.tokens_per_step,
                                               args.cpu_threads, args.batches,
                                               max(5, args.iters // 2))

    payload = {"config": dict(vars(args)),
               "gpu_weight_gb_actual": round(gpu.actual_gb, 3), "results": res}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    for k in ("gpu_only", "cpu_only", "serial", "pipelined", "pipelined_pythread"):
        print(f"{k:20s} {res[k]['median_ms']:8.3f} ms")
    a = res["analysis"]
    print(f"ideal max(gpu,cpu)={a['ideal_ms']:.3f} ms  pipelined={a['pipelined_ms']:.3f} ms  "
          f"efficiency={a['overlap_efficiency']}")
    print(f"hides {a['cpu_time_hidden_pct']}% of the CPU time versus serial")
    ap = res["analysis_pythread"]
    print(f"python-thread control: {ap['pipelined_ms']:.3f} ms  "
          f"efficiency={ap['overlap_efficiency']}  hides {ap['cpu_time_hidden_pct']}%")
    for s in res["batch_sweep"]:
        print(f"batch={s['batch']:2d}  cpu={s['median_ms']:8.3f} ms  "
              f"per_seq={s['per_seq_ms']:.3f} ms")
    lib.csa_destroy(handle)
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
