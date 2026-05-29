"""CUDA Graph 兼容性回归 smoke test。

对一组 (路径名, LLM kwargs) 配置跑短 prompt，记录：
  - LLM 初始化是否成功（cuda graph capture 走在 init 阶段）
  - generate 一发是否能跑出合理输出（eager 路径 + replay 路径都覆盖）
  - 输出是否是连贯文本（粗略 sanity check，避免 graph replay 出乱码）

支持单进程批量跑：每条路径开一个独立子进程，避免显存 / 静态状态污染。

用法：
    python tools/cuda_graph_smoke.py --model /path/to/Qwen3-0.6B
"""

import os
import sys
import time
import json
import argparse
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# (label, 初始化 kwargs)。enforce_eager 全部默认 False，专门测 cuda graph 兼容性
CONFIGS = [
    ("baseline",     dict()),
    ("quest",        dict(quest_top_k_blocks=4, quest_min_seq_len=128)),
    ("c4_only",      dict(kv_quant_bits=4, kv_quant_group_size=32)),
    ("c4_quest",     dict(kv_quant_bits=4, kv_quant_group_size=32,
                           quest_top_k_blocks=4, quest_min_seq_len=128)),
    ("w4_g128",      dict(quantization="int4", quant_group_size=128)),
    ("w4a8_g32",     dict(quantization="int4", quant_group_size=32, act_quant_bits=8)),
    ("cpu_offload",  dict(cpu_offload=True, cpu_offload_num_layers=2)),  # 只 offload 2 层快一点
]


def run_single(args, label, kwargs):
    """子进程内：init LLM + generate，输出 JSON 结果。"""
    from tinyvllm import LLM, SamplingParams

    result = dict(label=label, kwargs=kwargs, init_ok=False, gen_ok=False,
                  err_phase=None, err_msg=None, sample_text=None,
                  init_time_s=0.0, gen_time_s=0.0)

    t0 = time.time()
    try:
        llm = LLM(
            args.model,
            enforce_eager=False,            # 关键：开 cuda graph
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **kwargs,
        )
    except Exception as e:
        result["err_phase"] = "init"
        result["err_msg"] = f"{type(e).__name__}: {e}"
        result["init_time_s"] = time.time() - t0
        return result
    result["init_ok"] = True
    result["init_time_s"] = time.time() - t0

    # 跑两个 prompt，长度不同——一条短（eager 路径）+一条 prompt 后多生 16 token（让 decode replay 触发）
    prompts = ["The quick brown fox", "Hello world, please describe what artificial intelligence is in detail:"]
    sps = [SamplingParams(temperature=0.0, max_tokens=16) for _ in prompts]

    t1 = time.time()
    try:
        outputs = llm.generate(prompts, sps, use_tqdm=False)
    except Exception as e:
        result["err_phase"] = "generate"
        result["err_msg"] = f"{type(e).__name__}: {e}"
        result["gen_time_s"] = time.time() - t1
        return result
    result["gen_time_s"] = time.time() - t1
    result["gen_ok"] = True
    result["sample_text"] = outputs[0]["text"][:80]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--mode", type=str, default=None,
                   help="internal: run a single config in subprocess")
    p.add_argument("--out-json", type=str, default="cuda_graph_smoke.json")
    p.add_argument("--filter", type=str, default=None,
                   help="only run configs whose label matches this substring")
    args = p.parse_args()

    if args.mode:
        # 子进程：跑单条
        cfg_map = {label: kwargs for label, kwargs in CONFIGS}
        if args.mode not in cfg_map:
            print(json.dumps({"err": f"unknown mode {args.mode}"}))
            sys.exit(2)
        result = run_single(args, args.mode, cfg_map[args.mode])
        # 把结果写到约定路径
        with open(f"/tmp/cuda_graph_smoke_{args.mode}.json", "w") as f:
            json.dump(result, f)
        return

    # 主进程：spawn 每条路径
    results = []
    script = os.path.abspath(__file__)
    for label, kwargs in CONFIGS:
        if args.filter and args.filter not in label:
            continue
        print(f"\n>>> {label}  kwargs={kwargs}", flush=True)
        cmd = [sys.executable, script,
               "--model", args.model,
               "--max-model-len", str(args.max_model_len),
               "--max-num-seqs", str(args.max_num_seqs),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--mode", label]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        result_path = f"/tmp/cuda_graph_smoke_{label}.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            os.remove(result_path)
        else:
            # 子进程崩了（可能 import / segfault）
            r = dict(label=label, kwargs=kwargs, init_ok=False, gen_ok=False,
                     err_phase="subprocess_crash",
                     err_msg=f"rc={ret.returncode}; stderr_tail={ret.stderr[-300:]}")

        # 简短打印
        if r.get("gen_ok"):
            print(f"   PASS  init={r['init_time_s']:.1f}s gen={r['gen_time_s']:.1f}s  "
                  f"text={r['sample_text']!r}")
        else:
            print(f"   FAIL  phase={r.get('err_phase')} err={r.get('err_msg')}")
        results.append(r)

    # 打印汇总
    print("\n========== CUDA Graph 兼容性回归汇总 ==========")
    print(f"{'label':<14} | {'init':>5} | {'gen':>4} | {'init_s':>7} | {'gen_s':>6} | err")
    print("-" * 80)
    for r in results:
        init = "ok" if r["init_ok"] else "FAIL"
        gen = "ok" if r["gen_ok"] else "FAIL"
        init_s = f"{r.get('init_time_s', 0):.1f}"
        gen_s = f"{r.get('gen_time_s', 0):.1f}"
        err = ""
        if not r["gen_ok"]:
            err = f"{r.get('err_phase')}: {(r.get('err_msg') or '')[:60]}"
        print(f"{r['label']:<14} | {init:>5} | {gen:>4} | {init_s:>7} | {gen_s:>6} | {err}")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetails saved to {args.out_json}")


if __name__ == "__main__":
    main()
