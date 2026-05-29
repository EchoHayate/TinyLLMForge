"""TP=2 多卡路径兼容性回归。

类比 cuda_graph_smoke.py：spawn 子进程跑每个配置，避免 NCCL 进程组 / SharedMemory
跨配置污染。每条 config 单独占用一组 GPU，串行跑（端口 2333 + shm name "tinyvllm"
都硬编码，不能并发）。

每条 config 输出：
  - init_ok      : LLM 构造是否成功
  - gen_ok       : 一次 generate 是否能出非空文本
  - decode_tps   : 粗吞吐（warmup 后跑 16 条 prompt，max_tokens=32）
  - text_sample  : 首条输出前 60 字符
  - peak_mem_gb  : rank0 峰值显存（多卡时不代表总占用，但能判明有无异常增长）
  - error        : 失败时的简短堆栈

跑法（A100 上）：
    # 选两张空闲卡
    rm -f /dev/shm/tinyvllm
    CUDA_VISIBLE_DEVICES=1,3 python tools/tp_smoke.py \
        --model /path/to/Qwen3-0.6B --tp-size 2

    # 跑单条（debug）
    CUDA_VISIBLE_DEVICES=1,3 python tools/tp_smoke.py \
        --model /path/to/Qwen3-0.6B --tp-size 2 --filter c4_only
"""

import os
import sys
import json
import time
import argparse
import subprocess
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============== config 定义 ==============
# 全部 enforce_eager=True；cuda_graph 单独作为第 8 条
CONFIGS = [
    ("baseline",   dict(enforce_eager=True)),
    ("quest",      dict(enforce_eager=True, quest_top_k_blocks=8, quest_min_seq_len=512)),
    ("c4_only",    dict(enforce_eager=True, kv_quant_bits=4, kv_quant_group_size=128)),
    ("c4_quest",   dict(enforce_eager=True, kv_quant_bits=4, kv_quant_group_size=128,
                        quest_top_k_blocks=8, quest_min_seq_len=512)),
    ("w4_g128",    dict(enforce_eager=True, quantization="int4", quant_group_size=128)),
    ("w4a8_g128",  dict(enforce_eager=True, quantization="int4", quant_group_size=128,
                        act_quant_bits=8)),
    ("cpu_offload", dict(enforce_eager=True, cpu_offload=True, cpu_offload_num_layers=-1)),
    # 最后一条：放开 cuda graph（baseline，最稳的路径），看 NCCL × graph capture 行不行
    ("cuda_graph_baseline", dict(enforce_eager=False)),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-output-len", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--filter", type=str, default=None,
                   help="只跑名字精确匹配的 config（多个用逗号分隔）；不设则跑全部")
    p.add_argument("--out-file", type=str, default="tp_smoke.json")

    # 子进程参数
    p.add_argument("--mode", type=str, default=None,
                   help="internal: 子进程跑的 config 名")
    p.add_argument("--result-file", type=str, default=None)
    return p.parse_args()


def build_inputs(num_seqs: int, max_input_len: int, max_output_len: int):
    from random import randint, seed
    from tinyvllm import SamplingParams
    seed(0)
    prompts = [
        [randint(0, 10000) for _ in range(randint(max_input_len // 2, max_input_len))]
        for _ in range(num_seqs)
    ]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_output_len)
           for _ in range(num_seqs)]
    return prompts, sps


def run_single(args):
    """子进程：跑一条 config，结果写到 --result-file。"""
    import torch
    from tinyvllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    name = args.mode
    cfg = dict(c for c in CONFIGS if c[0] == name).get(name) if False else None
    cfg = next((c[1] for c in CONFIGS if c[0] == name), None)
    if cfg is None:
        print(f"unknown mode: {name}", file=sys.stderr)
        sys.exit(2)

    print(f"\n========== TP smoke: {name} (tp={args.tp_size}) ==========", flush=True)
    result = {"name": name, "tp_size": args.tp_size,
              "init_ok": False, "gen_ok": False,
              "decode_tps": 0.0, "text_sample": "",
              "peak_mem_gb": 0.0, "error": ""}

    try:
        llm = LLM(
            args.model,
            tensor_parallel_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **cfg,
        )
        result["init_ok"] = True

        # warmup
        llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        prompts, sps = build_inputs(args.num_seqs, args.max_input_len, args.max_output_len)

        for prompt, sp in zip(prompts, sps):
            llm.add_request(prompt, sp)

        decode_time = 0.0
        decode_tokens = 0
        outputs_collected = {}

        torch.cuda.synchronize()
        while not llm.is_finished():
            t0 = time.perf_counter()
            output, num_tokens = llm.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if num_tokens < 0:
                decode_time += dt
                decode_tokens += -num_tokens
            for seq_id, token_ids in output:
                outputs_collected[seq_id] = token_ids

        result["decode_tps"] = round(decode_tokens / decode_time, 2) if decode_time > 0 else 0.0
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

        # decode 出来的 token id list 是数字 prompt 的输出，也能体现"非空 + 非乱"
        if outputs_collected:
            tok_ids = next(iter(outputs_collected.values()))
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            text = tokenizer.decode(tok_ids[:32], skip_special_tokens=True)
            result["text_sample"] = text[:60].replace("\n", " ")
            result["gen_ok"] = len(tok_ids) > 0
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(result, f, ensure_ascii=False)


def main():
    args = parse_args()

    if args.mode is not None:
        run_single(args)
        return

    # 主进程：spawn 每条 config
    cfgs = CONFIGS
    if args.filter:
        wanted = set(s.strip() for s in args.filter.split(",") if s.strip())
        cfgs = [c for c in cfgs if c[0] in wanted]
        if not cfgs:
            print(f"no config matches --filter {args.filter}")
            sys.exit(1)

    out_dir = os.path.join(os.getcwd(), "tp_smoke_out")
    os.makedirs(out_dir, exist_ok=True)

    script = os.path.abspath(__file__)
    base_cmd = [
        sys.executable, script,
        "--model", args.model,
        "--tp-size", str(args.tp_size),
        "--max-model-len", str(args.max_model_len),
        "--num-seqs", str(args.num_seqs),
        "--max-input-len", str(args.max_input_len),
        "--max-output-len", str(args.max_output_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    all_results = []
    for name, _ in cfgs:
        # 跑前清理可能残留的 SharedMemory
        for p in ("/dev/shm/tinyvllm",):
            try: os.unlink(p)
            except FileNotFoundError: pass

        result_file = os.path.join(out_dir, f"{name}.json")
        cmd = base_cmd + ["--mode", name, "--result-file", result_file]
        print(f"\n>>> spawning: {name}", flush=True)
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"FAIL: {name} subprocess exited with {ret.returncode}")
            all_results.append({"name": name, "tp_size": args.tp_size,
                                "init_ok": False, "gen_ok": False,
                                "error": f"subprocess exit {ret.returncode}"})
            continue
        try:
            with open(result_file) as f:
                all_results.append(json.load(f))
        except Exception as e:
            all_results.append({"name": name, "init_ok": False,
                                "error": f"load result failed: {e}"})

    # 汇总
    print("\n\n========== TP=%d SMOKE SUMMARY ==========" % args.tp_size)
    header = f"{'config':>22} | {'init':>4} | {'gen':>3} | {'decode_tps':>10} | {'peak_gb':>7} | text_sample"
    print(header)
    print("-" * len(header))
    for r in all_results:
        init_str = "ok" if r.get("init_ok") else "FAIL"
        gen_str = "ok" if r.get("gen_ok") else "-"
        dtps = r.get("decode_tps", 0)
        peak = r.get("peak_mem_gb", 0)
        sample = r.get("text_sample", "") or r.get("error", "")[:60]
        print(f"{r['name']:>22} | {init_str:>4} | {gen_str:>3} | "
              f"{dtps:>10.2f} | {peak:>7.3f} | {sample}")

    summary_path = os.path.join(out_dir, args.out_file)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nsummary saved to {summary_path}")


if __name__ == "__main__":
    main()
