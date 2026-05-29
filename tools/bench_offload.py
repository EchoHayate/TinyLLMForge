"""cpu_offload 吞吐基线测量。

跑 baseline FP16（无 offload）vs cpu_offload，分别记录 prefill TPS / decode TPS / peak GPU mem。
覆盖多段 ctx，给出"什么 ctx 下 offload 才合算"的结论。

每个 (mode, ctx_len) 组合 spawn 一个独立子进程跑，避免 cpu_offload 的 state（独立 stream / 异步
H2D buffer）污染下一轮，也避免 cuda 上下文残留导致 peak mem 不准。

用法：
    python tools/bench_offload.py --model /path/to/Qwen3-0.6B \
        --num-seqs 16 --ctx-list 4096,15000 --max-output-len 64

子进程模式（内部使用，不需要手动调）：
    python tools/bench_offload.py ... --mode baseline --ctx 4096 --result-file /tmp/x.json
"""

import os
import sys
import json
import time
import argparse
import subprocess
from random import randint, seed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--ctx-list", type=str, default="4096,15000",
                   help="逗号分隔的 ctx 长度列表，每段都跑 baseline + offload")
    p.add_argument("--max-output-len", type=int, default=64,
                   help="decode 阶段每条 seq 生成 token 数")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--cpu-offload-num-layers", type=int, default=-1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    # 子进程内部参数
    p.add_argument("--mode", type=str, default=None,
                   choices=["baseline", "offload"],
                   help="internal: run single mode in subprocess")
    p.add_argument("--ctx", type=int, default=None,
                   help="internal: 子进程跑的单个 ctx 长度")
    p.add_argument("--result-file", type=str, default=None,
                   help="internal: 子进程写结果 json 的路径")
    return p.parse_args()


def build_inputs(num_seqs: int, ctx_len: int, max_output_len: int):
    seed(0)
    # 让每条 prompt 长度都接近 ctx_len（[ctx_len//2, ctx_len]），制造长 ctx 压力
    prompts = [
        [randint(0, 10000) for _ in range(randint(ctx_len // 2, ctx_len))]
        for _ in range(num_seqs)
    ]
    from tinyvllm import SamplingParams
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_output_len)
        for _ in range(num_seqs)
    ]
    return prompts, sps


def run_single(args):
    """子进程：跑单个 (mode, ctx) 组合，结果写到 --result-file。"""
    import torch
    from tinyvllm import LLM, SamplingParams

    mode = args.mode
    ctx_len = args.ctx
    use_offload = (mode == "offload")

    print(f"\n========== bench: mode={mode} ctx={ctx_len} ==========", flush=True)

    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload=use_offload,
        cpu_offload_num_layers=args.cpu_offload_num_layers if use_offload else -1,
    )

    # warmup：让 cuda graph capture / cpu_offload prefetch buffer 都到位
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    prompts, sps = build_inputs(args.num_seqs, ctx_len, args.max_output_len)

    # 手动 add_request + step，自己计时拆分 prefill / decode
    for prompt, sp in zip(prompts, sps):
        llm.add_request(prompt, sp)

    prefill_time = 0.0
    decode_time = 0.0
    prefill_tokens = 0
    decode_tokens = 0

    torch.cuda.synchronize()
    while not llm.is_finished():
        t0 = time.perf_counter()
        output, num_tokens = llm.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if num_tokens > 0:  # prefill step
            prefill_time += dt
            prefill_tokens += num_tokens
        else:               # decode step
            decode_time += dt
            decode_tokens += -num_tokens

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    prefill_tps = prefill_tokens / prefill_time if prefill_time > 0 else 0.0
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0

    result = {
        "mode": mode,
        "ctx": ctx_len,
        "num_seqs": args.num_seqs,
        "prefill_tokens": prefill_tokens,
        "prefill_time_s": round(prefill_time, 3),
        "prefill_tps": round(prefill_tps, 2),
        "decode_tokens": decode_tokens,
        "decode_time_s": round(decode_time, 3),
        "decode_tps": round(decode_tps, 2),
        "peak_mem_gb": round(peak_mem_gb, 3),
    }
    print(json.dumps(result, indent=2), flush=True)

    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(result, f)


def main():
    args = parse_args()

    if args.mode is not None:
        # 子进程模式
        if args.ctx is None:
            print("ERROR: --mode 需要配 --ctx", file=sys.stderr)
            sys.exit(2)
        run_single(args)
        return

    # 主进程模式：spawn 每个 (mode, ctx) 子进程
    ctxs = [int(x) for x in args.ctx_list.split(",") if x.strip()]
    modes = ["baseline", "offload"]

    script = os.path.abspath(__file__)
    base_cmd = [
        sys.executable, script,
        "--model", args.model,
        "--num-seqs", str(args.num_seqs),
        "--max-output-len", str(args.max_output_len),
        "--max-model-len", str(args.max_model_len),
        "--cpu-offload-num-layers", str(args.cpu_offload_num_layers),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    out_dir = os.path.join(os.getcwd(), "bench_offload_out")
    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    for ctx in ctxs:
        for mode in modes:
            tag = f"{mode}_ctx{ctx}"
            result_file = os.path.join(out_dir, f"{tag}.json")
            cmd = base_cmd + [
                "--mode", mode, "--ctx", str(ctx),
                "--result-file", result_file,
            ]
            print(f"\n>>> spawning: {tag}", flush=True)
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"FAIL: {tag} exited with {ret.returncode}")
                all_results.append({"mode": mode, "ctx": ctx, "fail": True})
                continue
            with open(result_file) as f:
                all_results.append(json.load(f))

    # 汇总输出
    print("\n\n========== SUMMARY ==========")
    header = f"{'mode':>10} | {'ctx':>6} | {'prefill_tps':>12} | {'decode_tps':>11} | {'peak_mem_gb':>12}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r.get("fail"):
            print(f"{r['mode']:>10} | {r['ctx']:>6} | {'FAIL':>12} | {'FAIL':>11} | {'-':>12}")
            continue
        print(f"{r['mode']:>10} | {r['ctx']:>6} | "
              f"{r['prefill_tps']:>12.2f} | {r['decode_tps']:>11.2f} | "
              f"{r['peak_mem_gb']:>12.3f}")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsummary saved to {summary_path}")


if __name__ == "__main__":
    main()
