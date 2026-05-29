"""Quest decode 阶段 torch.profiler 定位瓶颈 kernel。

按照 bench.py 同样配置跑一次 prefill + 若干步 decode，分别 profile baseline / Quest。
最后打印两份 top-N kernel 表 + diff，定位 Quest 引入的额外开销。

用法：
    python tools/profile_quest.py --model /path/to/Qwen3-0.6B \
        --num-seqs 16 --max-input-len 14000 --max-output-len 32 \
        --max-model-len 16384 --quest-top-k 16 --warmup-steps 8 --profile-steps 16
"""

import os
import sys
import time
import argparse
from random import randint, seed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from tinyvllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--max-input-len", type=int, default=14000)
    p.add_argument("--max-output-len", type=int, default=64,
                   help="profile 时只生成少量 token，避免 trace 文件过大")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--quest-top-k", type=int, default=16)
    p.add_argument("--quest-min-seq-len", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--profile-steps", type=int, default=16)
    p.add_argument("--out-dir", type=str, default="profile_out")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--mode", type=str, default=None,
                   choices=["baseline", "quest", "c4_only", "c4_quest"],
                   help="internal: run single mode in subprocess. "
                        "baseline = FP16 KV no Quest; quest = FP16 KV + Quest; "
                        "c4_only = C4 KV no Quest (α 路线); c4_quest = C4 KV + Quest (β3)")
    p.add_argument("--kv-quant-group-size", type=int, default=32,
                   help="C4 路径下的 group_size，仅 c4_only / c4_quest 模式生效")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--compare", type=str, nargs=2, default=["c4_only", "c4_quest"],
                   metavar=("MODE_A", "MODE_B"),
                   help="主进程要对比的两个 mode")
    return p.parse_args()


def build_inputs(args):
    seed(0)
    prompts = [
        [randint(0, 10000) for _ in range(randint(args.max_input_len // 2, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sps = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_output_len)
        for _ in range(args.num_seqs)
    ]
    return prompts, sps


def run_with_profiler(label, llm, prompts, sps, args):
    """跑一次 generate，外层套 torch.profiler。

    schedule:  wait + warmup + active —— 只 trace active 段，避免 prefill 和 cold-start。
    """
    # 让 step counter 推进的回调
    out_dir = os.path.join(args.out_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    sched = schedule(
        wait=0,
        warmup=args.warmup_steps,
        active=args.profile_steps,
        repeat=1,
    )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(f"\n========== profiling: {label} ==========", flush=True)

    # 进入 profiler 后跑 generate；profiler.step() 由我们 monkey-patch 进 LLMEngine.step
    # ——但 LLMEngine.step 不易 hook，简单粗暴：profiler 跨整个 generate，不分段；
    # 不过这样会把 prefill 也算进 active；为了仍能聚焦 decode，先 prefill 一遍，再 profile decode。

    # 1) 先 prefill：跑 1 步生成 1 token，把 KV cache 充满
    short_sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1) for _ in prompts]
    llm.generate(prompts, short_sps, use_tqdm=False)

    # 2) 第二轮：用 add_request + step，让 prompts 都进入 decode 阶段（命中 prefix cache）
    #    然后我们手动 step + profiler.step 以精确控制
    llm.scheduler.waiting.clear()
    llm.scheduler.running.clear()
    for prompt, sp in zip(prompts, sps):
        llm.add_request(prompt, sp)

    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        steps_done = 0
        total_steps = args.warmup_steps + args.profile_steps
        t0 = time.time()
        # 先把 batch prefill 完
        while not llm.is_finished() and steps_done < total_steps:
            llm.step()
            prof.step()
            steps_done += 1
        dt = time.time() - t0
    print(f"[{label}] {steps_done} steps, {dt:.2f}s", flush=True)

    # 打印 top-N
    print(f"\n[{label}] top {args.top_n} CUDA kernels by cuda_time_total:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=args.top_n,
    ))

    # 同时存 chrome trace 备用
    trace_path = os.path.join(out_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"[{label}] trace saved to {trace_path}")

    # 返回 key_averages 用于 diff
    return prof.key_averages()


def diff_kernels(base_avg, quest_avg, top_n: int = 20):
    """打印 Quest vs baseline 的 cuda_time_total 差异，定位新增/变慢的 kernel。"""

    def to_dict(avg):
        d = {}
        for ev in avg:
            # ev.key 唯一标识 op；不区分 input shape
            d[ev.key] = d.get(ev.key, 0) + ev.cuda_time_total
        return d

    b = to_dict(base_avg)
    q = to_dict(quest_avg)
    keys = set(b.keys()) | set(q.keys())
    rows = []
    for k in keys:
        bb = b.get(k, 0)
        qq = q.get(k, 0)
        rows.append((qq - bb, qq, bb, k))
    rows.sort(key=lambda r: -r[0])

    print("\n========== Quest - baseline (cuda_time_total, μs) ==========")
    print(f"{'delta':>14} | {'quest':>14} | {'base':>14} | op")
    print("-" * 80)
    for delta, qq, bb, k in rows[:top_n]:
        if delta < 1:  # 0~1 μs 噪声忽略
            break
        name = k if len(k) < 50 else k[:47] + "..."
        print(f"{delta/1e3:>14.2f} | {qq/1e3:>14.2f} | {bb/1e3:>14.2f} | {name}")


def run_single(mode: str, args):
    """Run profiling for a single mode (baseline / quest / c4_only / c4_quest) in current process."""
    prompts, sps = build_inputs(args)

    if mode == "baseline":
        kv_quant_bits = 0
        quest_top_k = -1
    elif mode == "quest":
        kv_quant_bits = 0
        quest_top_k = args.quest_top_k
    elif mode == "c4_only":
        kv_quant_bits = 4
        quest_top_k = -1
    elif mode == "c4_quest":
        kv_quant_bits = 4
        quest_top_k = args.quest_top_k
    else:
        raise ValueError(f"unknown mode {mode}")

    print(f"init {mode} LLM (kv_quant_bits={kv_quant_bits}, quest_top_k={quest_top_k})...")
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        quest_top_k_blocks=quest_top_k,
        quest_min_seq_len=args.quest_min_seq_len,
        kv_quant_bits=kv_quant_bits,
        kv_quant_group_size=args.kv_quant_group_size if kv_quant_bits == 4 else 128,
    )
    run_with_profiler(mode, llm, prompts, sps, args)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if hasattr(args, "mode") and args.mode:
        # 子进程模式：只跑单边
        run_single(args.mode, args)
        return

    # 主进程模式：分别 spawn 两个子进程，再 diff
    import subprocess, json, pickle

    script = os.path.abspath(__file__)
    base_cmd = [
        sys.executable, script,
        "--model", args.model,
        "--num-seqs", str(args.num_seqs),
        "--max-input-len", str(args.max_input_len),
        "--max-output-len", str(args.max_output_len),
        "--max-model-len", str(args.max_model_len),
        "--quest-top-k", str(args.quest_top_k),
        "--quest-min-seq-len", str(args.quest_min_seq_len),
        "--kv-quant-group-size", str(args.kv_quant_group_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs),
        "--warmup-steps", str(args.warmup_steps),
        "--profile-steps", str(args.profile_steps),
        "--out-dir", args.out_dir,
        "--top-n", str(args.top_n),
    ]

    mode_a, mode_b = args.compare
    for mode in (mode_a, mode_b):
        cmd = base_cmd + ["--mode", mode]
        print(f"\n>>> spawning subprocess: {mode}", flush=True)
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"ERROR: {mode} subprocess exited with {ret.returncode}")
            return

    # 3) 加载两份 trace 做 diff（从 chrome trace JSON 解析 kernel 时间）
    print(f"\n\n========== DIFF (from chrome traces) {mode_b} - {mode_a} ==========")
    try:
        a_kernels = parse_trace_kernels(os.path.join(args.out_dir, mode_a, "trace.json"))
        b_kernels = parse_trace_kernels(os.path.join(args.out_dir, mode_b, "trace.json"))
        diff_from_traces(a_kernels, b_kernels, top_n=args.top_n)
    except Exception as e:
        print(f"diff failed: {e}")


def parse_trace_kernels(trace_path: str) -> dict[str, float]:
    """从 chrome trace JSON 提取 CUDA kernel 的累积时间 (μs)。"""
    import json
    with open(trace_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    kernels: dict[str, float] = {}
    for ev in events:
        if ev.get("cat") == "kernel":
            name = ev.get("name", "?")
            dur = ev.get("dur", 0)  # μs
            kernels[name] = kernels.get(name, 0) + dur
    return kernels


def diff_from_traces(base: dict, quest: dict, top_n: int = 20):
    keys = set(base.keys()) | set(quest.keys())
    rows = []
    for k in keys:
        bb = base.get(k, 0)
        qq = quest.get(k, 0)
        rows.append((qq - bb, qq, bb, k))
    rows.sort(key=lambda r: -r[0])

    print(f"{'delta_ms':>12} | {'quest_ms':>12} | {'base_ms':>12} | kernel")
    print("-" * 90)
    for delta, qq, bb, k in rows[:top_n]:
        name = k if len(k) < 55 else k[:52] + "..."
        print(f"{delta/1e3:>12.3f} | {qq/1e3:>12.3f} | {bb/1e3:>12.3f} | {name}")


if __name__ == "__main__":
    main()
