"""
needle-in-haystack accuracy + throughput 评测

合成 N 段填充文本 + 在指定深度处插入 "the magic number is XXXXX"
让模型回答数字，对比 baseline 与 quest top-k 不同稀疏度。

用法示例：
    python tools/eval_needle.py --model /path/to/Qwen3-0.6B \
        --context-lens 4096 8192 15000 \
        --depths 0.0 0.25 0.5 0.75 1.0 \
        --top-k-blocks-list -1 4 8 16 \
        --num-trials 3
"""

import os
import sys
import re
import time
import json
import random
import argparse

# 允许从仓库根直接 `python tools/eval_needle.py` 运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer
from tinyvllm import LLM, SamplingParams


HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)

QUESTION = (
    "\n\nWhat is the magic number? Answer with only the digits, nothing else."
)

NEEDLE_TEMPLATE = "The magic number is {num}. Remember it. "


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--context-lens", type=int, nargs="+", default=[4096, 8192, 15000])
    p.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument(
        "--top-k-blocks-list",
        type=int,
        nargs="+",
        default=[-1, 4, 8, 16],
        help="-1 表示 baseline；其余值是 quest top-k",
    )
    p.add_argument("--quest-min-seq-len", type=int, default=512)
    p.add_argument("--num-trials", type=int, default=3, help="每个 (ctx_len, depth) 重复多少次")
    p.add_argument("--max-output-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-prompts", action="store_true", default=False,
                   help="所有 top-k setting 复用同一批 prompt/magic，用于质量归因；setting 间会清空 prefix cache")
    p.add_argument("--out-json", type=str, default="needle_results.json")
    # C4 / KV cache 量化（与 quest 不可同时开，互斥逻辑在 setting loop 里处理）
    p.add_argument("--kv-quant-bits", type=int, default=0, choices=[0, 4, 8],
                   help="C4 模式：KV cache 量化位宽。设为 4 后忽略 --top-k-blocks-list，单跑 C4 vs baseline")
    p.add_argument("--kv-quant-group-size", type=int, default=128)
    # W4A8：weight 量化 + activation 假量化（不与 KV 量化互斥，但通常单跑做对照）
    p.add_argument("--quantization", type=str, default=None,
                   choices=[None, "int8", "int4", "int2"])
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--act-quant-bits", type=int, default=0, choices=[0, 8])
    p.add_argument("--smoothquant-scale-path", type=str, default=None,
                   help="SmoothQuant 校准产物（.pt）。仅在 W4A8 场景下加载，把 per-input-channel scale 注入 weight。")
    p.add_argument("--act-quant-skip-first", type=int, default=0,
                   help="前 N 层不做 A8（W4A8+SQ 长文塌方根因——首尾层 outlier 极端，留 fp 可显著修复召回）")
    p.add_argument("--act-quant-skip-last", type=int, default=0,
                   help="后 N 层不做 A8（同上）")
    p.add_argument("--act-quant-skip-layers", type=int, nargs="+", default=None,
                   help="显式指定不做 A8 的层（按 outlier 强度精准 skip，见 tools/diag_layer_outlier.py）")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="KV pool 占显存比例。长 ctx + C4 全 dequant 路径下需要给瞬态 buffer 留空间，建议 0.6-0.7")
    p.add_argument("--max-num-seqs", type=int, default=512)
    return p.parse_args()


def build_prompt(tokenizer, ctx_len_tok: int, depth: float, magic_num: int) -> str:
    """构造一条 prompt：填充文本中按比例 depth 插入 needle，再加问题。

    ctx_len_tok 是目标 prompt token 数（含 needle 和问题）。
    """
    needle = NEEDLE_TEMPLATE.format(num=magic_num)
    question = QUESTION

    # 估计填充用的句子需要重复多少次：先按字符 → token 的近似比 4:1 给个上限
    target_filler_tok = max(64, ctx_len_tok - 64)
    repeat_n = max(1, target_filler_tok // 4)
    haystack = HAYSTACK_SENTENCE * repeat_n

    haystack_ids = tokenizer.encode(haystack, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)

    budget = ctx_len_tok - len(needle_ids) - len(question_ids)
    budget = max(64, budget)
    haystack_ids = haystack_ids[:budget]

    insert_pos = int(depth * len(haystack_ids))
    full_ids = haystack_ids[:insert_pos] + needle_ids + haystack_ids[insert_pos:] + question_ids
    return tokenizer.decode(full_ids)


def extract_answer(text: str) -> str | None:
    m = re.search(r"\d{4,6}", text)
    return m.group(0) if m else None


def _setting_seed(args, top_k: int) -> int:
    if getattr(args, "fixed_prompts", False):
        return args.seed
    # 每个 setting 用不同的 seed，避免后续 setting 命中前一个 setting 写入的 prefix cache，
    # 把吞吐数字虚高。fixed-prompts 模式会关闭这个 offset，用于逐条质量归因。
    seed_offset = (top_k if top_k > 0 else 0) * 7919 + (101 if args.kv_quant_bits == 4 else 0)
    return args.seed + seed_offset


def build_eval_batch(tokenizer, args, top_k: int):
    rng = random.Random(_setting_seed(args, top_k))
    prompts = []
    metas = []
    for ctx_len in args.context_lens:
        for depth in args.depths:
            for trial in range(args.num_trials):
                magic = rng.randint(10000, 99999)
                prompt = build_prompt(tokenizer, ctx_len, depth, magic)
                prompts.append(prompt)
                metas.append(dict(ctx_len=ctx_len, depth=depth, trial=trial, magic=magic))
    return prompts, metas


def clear_prefix_cache(llm) -> int:
    """清空跨 setting 复用的 prefix-cache 索引，返回被清掉元数据的空闲 block 数。

    fixed-prompts 模式需要复用同一批 prompt 做逐条质量归因；如果不清这里，第二个及
    后续 setting 会复用前一个 setting 留下的完整 block KV，吞吐会虚高，也会让归因日志
    难以解释。只清 ref_count==0 的空闲 block 元数据，避免误伤正在运行的序列。
    """
    block_manager = llm.scheduler.block_manager
    block_manager.hash_to_block_id.clear()
    cleared = 0
    for block in block_manager.blocks:
        if block.ref_count != 0:
            continue
        if block.hash != -1 or block.token_ids:
            block.hash = -1
            block.token_ids = []
            cleared += 1
    return cleared


def run_one_setting(llm, tokenizer, args, top_k: int):
    is_c4 = args.kv_quant_bits == 4
    if is_c4:
        label = "C4" if top_k <= 0 else f"C4+top_k={top_k}"
    else:
        label = f"top_k={top_k}" if top_k > 0 else "baseline"
    print(f"\n=== {label} ===", flush=True)

    # 热改 config：quest_top_k_blocks 仅在 prepare_decode 里被读，可运行时切换
    # C4 + Quest 叠加（β3）：attention.forward 里"先选 top-k 再 dequant"已支持
    llm.model_runner.config.quest_top_k_blocks = top_k
    llm.model_runner.config.quest_min_seq_len = args.quest_min_seq_len

    # 先组装所有 prompt（一次 generate 跑完）
    prompts, metas = build_eval_batch(tokenizer, args, top_k)

    sps = [
        SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=args.max_output_len)
        for _ in prompts
    ]

    # warmup
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)

    t0 = time.time()
    outputs = llm.generate(prompts, sps, use_tqdm=False)
    dt = time.time() - t0
    total_out_tok = sum(len(o["token_ids"]) for o in outputs)
    throughput = total_out_tok / dt if dt > 0 else 0.0

    results = []
    for meta, out in zip(metas, outputs):
        ans = extract_answer(out["text"])
        hit = ans is not None and ans == str(meta["magic"])
        results.append({**meta, "answer": ans, "hit": hit, "raw": out["text"][:128]})

    # 聚合
    by_ctx_depth = {}
    for r in results:
        key = (r["ctx_len"], r["depth"])
        by_ctx_depth.setdefault(key, []).append(r["hit"])

    summary = []
    for (ctx_len, depth), hits in sorted(by_ctx_depth.items()):
        acc = sum(hits) / len(hits)
        summary.append(dict(ctx_len=ctx_len, depth=depth, accuracy=acc, n=len(hits)))
        print(f"  ctx={ctx_len:>5} depth={depth:.2f} acc={acc*100:5.1f}% (n={len(hits)})", flush=True)

    # 整体准确率
    overall = sum(r["hit"] for r in results) / max(1, len(results))
    print(f"  overall_acc={overall*100:.1f}%  throughput={throughput:.2f} tok/s", flush=True)

    out = dict(
        top_k=top_k,
        overall_accuracy=overall,
        throughput_tok_s=throughput,
        time_s=dt,
        per_setting=summary,
        details=results,
    )

    return out


def main():
    args = parse_args()

    is_c4 = args.kv_quant_bits == 4
    if is_c4:
        # C4 模式：KV cache 物理布局变了；但 quest_top_k_blocks 仍可热切，
        # 所以同进程内可以跑 "C4 only" + "C4+top_k=k1" + "C4+top_k=k2"
        top_k_list = args.top_k_blocks_list
        # 用 list 中的最大 top_k 初始化（确保 kv_summary 被分配）；
        # 若 list 全是 <=0 则初始化时关闭 Quest（不分配 summary）
        positive = [k for k in top_k_list if k > 0]
        init_top_k = max(positive) if positive else -1
    else:
        # baseline + Quest 热切模式
        # 用 list 中的最大 top_k 初始化（确保 kv_summary 被分配）；后续每个 setting
        # 通过热改 self.config.quest_top_k_blocks 切换 baseline / 不同稀疏度。
        top_k_list = args.top_k_blocks_list
        init_top_k = max([k for k in args.top_k_blocks_list if k > 0] + [1])

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        quest_top_k_blocks=init_top_k,
        quest_min_seq_len=args.quest_min_seq_len,
        kv_quant_bits=args.kv_quant_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        quantization=args.quantization,
        quant_group_size=args.quant_group_size,
        act_quant_bits=args.act_quant_bits,
        smoothquant_scale_path=args.smoothquant_scale_path,
        act_quant_skip_first=args.act_quant_skip_first,
        act_quant_skip_last=args.act_quant_skip_last,
        act_quant_skip_layers=args.act_quant_skip_layers,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # 给 LLM 预热一次，避免第一个 setting 测的是 cold-start
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)

    all_results = []
    for top_k in top_k_list:
        if args.fixed_prompts:
            cleared = clear_prefix_cache(llm)
            if cleared:
                print(f"[fixed-prompts] cleared prefix cache metadata for {cleared} free blocks", flush=True)
        r = run_one_setting(llm, tokenizer, args, top_k)
        all_results.append(r)

    print("\n========== SUMMARY ==========")
    print(f"{'setting':>10} | {'acc':>7} | {'tok/s':>9}")
    print("-" * 36)
    for r in all_results:
        if is_c4:
            label = "C4"
        else:
            label = "baseline" if r["top_k"] < 0 else f"top_k={r['top_k']}"
        print(f"{label:>10} | {r['overall_accuracy']*100:6.1f}% | {r['throughput_tok_s']:9.2f}")

    with open(args.out_json, "w") as f:
        json.dump(dict(args=vars(args), results=all_results), f, indent=2)
    print(f"\nDetails saved to {args.out_json}")


if __name__ == "__main__":
    main()
