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

from __future__ import annotations

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
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--tp-size", type=int, default=1,
                   help="Tensor parallel size. 默认 1；TP=2/多卡用于验证 SQ scale 切片和通信路径。")
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
    p.add_argument("--kv-cartridge-blocks", type=int, default=0,
                   help="KV-Cartridge v0：decode 时保留的 uniform KV block 数；0 表示关闭。与 Quest 分开评测。")
    p.add_argument("--kv-cartridge-min-seq-len", type=int, default=1024,
                   help="KV-Cartridge 只在序列长度达到该阈值时启用。")
    p.add_argument("--kv-cartridge-mode", type=str, default="uniform", choices=["uniform"],
                   help="KV-Cartridge v0 压缩策略：保首尾，中间均匀抽样。")
    p.add_argument("--am-compact-blocks", type=int, default=0,
                   help="Attention Matching compact decode：每个 KV head 保留的 compact KV 数；0 表示关闭。")
    p.add_argument("--am-compact-selector", type=str, default="highest", choices=["highest", "omp"],
                   help="Attention Matching compact decode 的 key selector：highest 或 omp。")
    p.add_argument("--am-compact-min-seq-len", type=int, default=1024,
                   help="Attention Matching compact decode 只在序列长度达到该阈值时启用。")
    p.add_argument("--am-compact-score-method", type=str, default="rms", choices=["rms", "mean", "max"],
                   help="AM-HighestAttnKeys 的 key score 聚合方式。")
    p.add_argument("--am-compact-beta-bound", type=float, default=3.0,
                   help="Attention Matching beta box bound，实际范围为 [-bound, bound]。")
    p.add_argument("--am-compact-ridge-lambda", type=float, default=1e-6,
                   help="Attention Matching C_v least-squares ridge lambda。")
    p.add_argument("--am-omp-candidate-pool-size", type=int, default=0,
                   help="AM-OMP 候选池大小；0 表示按 max(2*b, b+4) 自动选择。")
    p.add_argument("--am-compact-cache-refresh-interval", type=int, default=0,
                   help="AM compact cache 复用步数；0 表示关闭，每个 decode step 重新拟合。")
    p.add_argument("--am-prefill-cache-ref-query-stride", type=int, default=8,
                   help="prefill 构建 AM compact cache 时参考 query 的采样 stride。")
    p.add_argument("--am-compact-num-clusters", type=int, default=1,
                   help="prefill 持久 AM compact cache 的 query cluster 数；1 表示单 compact bank。")
    p.add_argument("--am-compact-route-top-k", type=int, default=1,
                   help="decode 时 ensemble 最近的 N 个 AM compact clusters；1 表示硬路由。")
    p.add_argument("--am-compact-num-key-spans", type=int, default=1,
                   help="prefill 按连续 key span 构建局部 AM compact bank 数；1 表示关闭 span-local bank。")
    p.add_argument("--am-compact-decode-refit", action="store_true", default=False,
                   help="decode 时复用 cached indices，但用当前 query 重拟合 beta/C_v。")
    p.add_argument("--am-compact-decode-refit-mode", type=str, default="full", choices=["full", "direct", "beta", "anchor"],
                   help="decode refit 的 C_v 策略：full=用 full target 重拟合；direct=直接用原始 V[selected]；beta=只重拟合 beta；anchor=用少量位置 anchor 近似 target。")
    p.add_argument("--am-compact-decode-refit-interval", type=int, default=1,
                   help="decode refit 后复用多少个 decode step；1 表示每步 refit。")
    p.add_argument("--am-compact-skip-first-layers", type=int, default=0,
                   help="层级 AM 开关：前 N 层不启用 AM，走 baseline FlashAttention。")
    p.add_argument("--am-compact-skip-last-layers", type=int, default=0,
                   help="层级 AM 开关：后 N 层不启用 AM，走 baseline FlashAttention。")
    p.add_argument("--am-compact-enable-layers", type=int, nargs="+", default=None,
                   help="层级 AM 开关：显式指定启用 AM 的层号；设置后覆盖 skip/stride 规则。")
    p.add_argument("--am-compact-layer-stride", type=int, default=1,
                   help="层级 AM 开关：每隔 N 层启用 AM；1 表示所有未 skip 的层启用。")
    p.add_argument("--am-layer-sweep", type=str, nargs="*", default=None,
                   help=("同进程 sweep 多组 AM 层级开关，格式：auto / off / all / stride:N / skip:F:L / layers:i,j,k；"
                         "默认自动前置 off baseline。例：--am-layer-sweep auto"))
    p.add_argument("--am-layer-sweep-print-only", action="store_true", default=False,
                   help="只展开并打印 --am-layer-sweep 对应的 setting 列表，不加载模型、不运行评测。")
    p.add_argument("--am-layer-sweep-no-auto-off", action="store_true", default=False,
                   help="默认 --am-layer-sweep 会自动前置 off baseline；设置该 flag 后不自动加入 off。")
    p.add_argument("--num-trials", type=int, default=3, help="每个 (ctx_len, depth) 重复多少次")
    p.add_argument("--max-output-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-prompts", action="store_true", default=False,
                   help="所有 top-k setting 复用同一批 prompt/magic，用于质量归因；setting 间会清空 prefix cache")
    p.add_argument("--needle-style", type=str, default="original", choices=["original", "newline"],
                   help="needle 插入格式。original 保持历史口径；newline 用空行包住 needle，避免和 haystack 单词粘连")
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


def _format_needle(magic_num: int, needle_style: str = "original") -> str:
    if needle_style == "original":
        return NEEDLE_TEMPLATE.format(num=magic_num)
    if needle_style == "newline":
        return "\n\n" + NEEDLE_TEMPLATE.format(num=magic_num).rstrip() + "\n\n"
    raise ValueError(f"unknown needle_style={needle_style!r}")


def build_prompt(tokenizer, ctx_len_tok: int, depth: float, magic_num: int, needle_style: str = "original") -> str:
    """构造一条 prompt：填充文本中按比例 depth 插入 needle，再加问题。

    ctx_len_tok 是目标 prompt token 数（含 needle 和问题）。
    """
    needle = _format_needle(magic_num, needle_style)
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
                prompt = build_prompt(tokenizer, ctx_len, depth, magic, getattr(args, "needle_style", "original"))
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


def _format_layer_list(layers: tuple[int, ...], max_items: int = 16) -> str:
    if len(layers) <= max_items:
        return "[" + ",".join(str(x) for x in layers) + "]"
    head_n = max_items // 2
    tail_n = max_items - head_n
    head = ",".join(str(x) for x in layers[:head_n])
    tail = ",".join(str(x) for x in layers[-tail_n:])
    return f"[{head},...,{tail}]"


def _am_compact_enabled_layers_for_context(context, num_hidden_layers: int) -> tuple[int, ...]:
    def layer_enabled(layer_idx: int) -> bool:
        if context.am_compact_blocks <= 0:
            return False
        enable_layers = context.am_compact_enable_layers
        if enable_layers is not None:
            return layer_idx in enable_layers
        skip_first = int(context.am_compact_skip_first_layers)
        if layer_idx < skip_first:
            return False
        skip_last = int(context.am_compact_skip_last_layers)
        if skip_last > 0 and layer_idx >= int(num_hidden_layers) - skip_last:
            return False
        stride = max(1, int(context.am_compact_layer_stride))
        return ((layer_idx - skip_first) % stride) == 0

    return tuple(layer_idx for layer_idx in range(int(num_hidden_layers)) if layer_enabled(layer_idx))


def _parse_am_layer_spec(spec: str) -> dict:
    spec = spec.strip()
    if spec == "off":
        return dict(label="off", am_blocks=0, skip_first=0, skip_last=0, enable_layers=None, layer_stride=1)
    if spec == "all":
        return dict(label="all", am_blocks=None, skip_first=0, skip_last=0, enable_layers=None, layer_stride=1)
    if spec.startswith("stride:"):
        stride = int(spec.split(":", 1)[1])
        if stride <= 0:
            raise ValueError(f"invalid AM layer sweep spec {spec!r}: stride 必须 > 0")
        return dict(label=f"stride{stride}", am_blocks=None, skip_first=0, skip_last=0, enable_layers=None, layer_stride=stride)
    if spec.startswith("skip:"):
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid AM layer sweep spec {spec!r}: skip 格式应为 skip:F:L")
        skip_first, skip_last = int(parts[1]), int(parts[2])
        if skip_first < 0 or skip_last < 0:
            raise ValueError(f"invalid AM layer sweep spec {spec!r}: skip 必须 >= 0")
        return dict(label=f"skip{skip_first}_{skip_last}", am_blocks=None, skip_first=skip_first, skip_last=skip_last,
                    enable_layers=None, layer_stride=1)
    if spec.startswith("layers:"):
        raw = spec.split(":", 1)[1]
        layers = tuple(int(x) for x in raw.split(",") if x.strip() != "")
        if not layers:
            raise ValueError(f"invalid AM layer sweep spec {spec!r}: layers 不能为空")
        if any(x < 0 for x in layers):
            raise ValueError(f"invalid AM layer sweep spec {spec!r}: layer index 必须 >= 0")
        layers = tuple(sorted(set(layers)))
        label = "L" + ",".join(str(x) for x in layers)
        return dict(label=label, am_blocks=None, skip_first=0, skip_last=0, enable_layers=layers, layer_stride=1)
    raise ValueError(f"unknown AM layer sweep spec {spec!r}; 支持 off / all / stride:N / skip:F:L / layers:i,j,k")


def _current_am_layer_setting(args) -> dict:
    if args.am_compact_enable_layers is not None:
        layers = tuple(sorted(set(int(x) for x in args.am_compact_enable_layers)))
        label = "L" + ",".join(str(x) for x in layers)
    else:
        layers = None
        label_parts = []
        if args.am_compact_skip_first_layers > 0:
            label_parts.append(f"sf{args.am_compact_skip_first_layers}")
        if args.am_compact_skip_last_layers > 0:
            label_parts.append(f"sl{args.am_compact_skip_last_layers}")
        if args.am_compact_layer_stride > 1:
            label_parts.append(f"st{args.am_compact_layer_stride}")
        label = "_".join(label_parts) if label_parts else "all"
    return dict(
        label=label,
        am_blocks=None,
        skip_first=args.am_compact_skip_first_layers,
        skip_last=args.am_compact_skip_last_layers,
        enable_layers=layers,
        layer_stride=args.am_compact_layer_stride,
    )


def _auto_am_layer_specs(num_layers: int) -> list[str]:
    """Build a coarse, model-size-aware AM layer sweep preset."""
    num_layers = int(num_layers)
    specs = ["all"]
    for stride in (2, 4, 8):
        if stride < num_layers:
            specs.append(f"stride:{stride}")

    quarter = max(1, num_layers // 4)
    three_eighths = max(1, (num_layers * 3) // 8)
    for skip in (quarter, three_eighths):
        if skip * 2 < num_layers:
            specs.append(f"skip:{skip}:{skip}")

    start = quarter
    end = num_layers - quarter
    step = max(1, quarter // 2)
    layers = tuple(range(start, end + 1, step))
    if layers:
        specs.append("layers:" + ",".join(str(x) for x in layers))
    return _dedupe_specs(specs)


def _dedupe_specs(specs: list[str]) -> list[str]:
    out = []
    seen = set()
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        out.append(spec)
    return out


def _build_am_layer_settings(args, num_layers: int) -> list[dict]:
    if args.am_layer_sweep is not None and len(args.am_layer_sweep) > 0:
        raw_specs = list(args.am_layer_sweep)
        if not args.am_layer_sweep_no_auto_off and "off" not in raw_specs:
            raw_specs = ["off"] + raw_specs
        specs = []
        for spec in raw_specs:
            if spec == "auto":
                specs.extend(_auto_am_layer_specs(num_layers))
            else:
                specs.append(spec)
        specs = _dedupe_specs(specs)
        return [_parse_am_layer_spec(spec) for spec in specs]
    return [_current_am_layer_setting(args)]


def _print_am_layer_settings(settings: list[dict], num_layers: int):
    print(f"[AM layer sweep] num_layers={num_layers} settings={len(settings)}", flush=True)
    for setting in settings:
        context_like = type("AMLayerSweepContext", (), {})()
        context_like.am_compact_blocks = 0 if setting.get("am_blocks") == 0 else 1
        context_like.am_compact_skip_first_layers = setting["skip_first"]
        context_like.am_compact_skip_last_layers = setting["skip_last"]
        context_like.am_compact_enable_layers = setting["enable_layers"]
        context_like.am_compact_layer_stride = setting["layer_stride"]
        enabled = _am_compact_enabled_layers_for_context(context_like, num_layers)
        print(
            f"  {setting['label']:>16}: enabled={len(enabled):>2}/{num_layers} "
            f"layers={_format_layer_list(enabled)}",
            flush=True,
        )


def _apply_am_layer_setting(args, llm, setting: dict, num_layers: int, base_am_blocks: int) -> tuple[int, ...]:
    args.am_compact_blocks = base_am_blocks if setting.get("am_blocks") is None else int(setting["am_blocks"])
    args.am_compact_skip_first_layers = setting["skip_first"]
    args.am_compact_skip_last_layers = setting["skip_last"]
    args.am_compact_enable_layers = list(setting["enable_layers"]) if setting["enable_layers"] is not None else None
    args.am_compact_layer_stride = setting["layer_stride"]

    cfg = llm.model_runner.config
    cfg.am_compact_blocks = args.am_compact_blocks
    cfg.am_compact_skip_first_layers = args.am_compact_skip_first_layers
    cfg.am_compact_skip_last_layers = args.am_compact_skip_last_layers
    cfg.am_compact_enable_layers = tuple(args.am_compact_enable_layers) if args.am_compact_enable_layers is not None else None
    cfg.am_compact_layer_stride = args.am_compact_layer_stride

    if cfg.am_compact_enable_layers is not None and any(x >= num_layers for x in cfg.am_compact_enable_layers):
        raise ValueError(f"AM layer setting {setting['label']!r} 包含超过模型层数的 layer index")
    enabled_layers = _am_compact_enabled_layers_for_context(cfg, num_layers)
    if args.am_compact_blocks > 0 and not enabled_layers:
        raise ValueError(f"AM layer setting {setting['label']!r} 没有启用任何层")
    if args.am_compact_blocks <= 0:
        print(f"[AM layers:{setting['label']}] disabled; using baseline FlashAttention", flush=True)
    else:
        print(
            f"[AM layers:{setting['label']}] enabled={len(enabled_layers)}/{num_layers} "
            f"layers={_format_layer_list(enabled_layers)}",
            flush=True,
        )
    return enabled_layers


def _summary_label(result: dict, cartridge_enabled: bool, is_c4: bool, args) -> str:
    if "am_layer_setting" in result:
        setting = result["am_layer_setting"]
        enabled_count = result.get("enabled_am_layer_count", 0)
        if setting["label"] == "off":
            return "off/baseline"
        prefix = "am_highest" if args.am_compact_selector == "highest" else "am_omp"
        return f"{prefix}_{setting['label']}_L{enabled_count}"
    label = result.get("label")
    if label is not None:
        return label
    if cartridge_enabled:
        return f"cartridge_b{args.kv_cartridge_blocks}"
    if is_c4:
        return "C4"
    return "baseline" if result["top_k"] < 0 else f"top_k={result['top_k']}"


def _print_best_am_layer_summary(results: list[dict], cartridge_enabled: bool, is_c4: bool, args):
    if not any("am_layer_setting" in r for r in results):
        return
    off = next((r for r in results if r.get("am_layer_setting", {}).get("label") == "off"), None)
    all_layer = next((r for r in results if r.get("am_layer_setting", {}).get("label") == "all"), None)
    perfect = [r for r in results if r.get("overall_accuracy", 0.0) >= 0.999999]
    if not perfect:
        print("\n[AM sweep] 没有 100% recall 的 setting。", flush=True)
        return
    best = max(perfect, key=lambda r: r.get("throughput_tok_s", 0.0))
    best_label = _summary_label(best, cartridge_enabled, is_c4, args)
    best_tps = best.get("throughput_tok_s", 0.0)
    parts = [f"global_best_100={best_label}", f"tok/s={best_tps:.2f}"]
    if off is not None and off.get("throughput_tok_s", 0.0) > 0:
        parts.append(f"vs_off={best_tps / off['throughput_tok_s']:.3f}x")
    if all_layer is not None and all_layer.get("throughput_tok_s", 0.0) > 0:
        parts.append(f"vs_all={best_tps / all_layer['throughput_tok_s']:.3f}x")
    print("\n[AM sweep] " + "  ".join(parts), flush=True)

    am_perfect = [
        r for r in perfect
        if r.get("am_layer_setting", {}).get("label") not in (None, "off")
    ]
    if not am_perfect:
        print("[AM sweep] 没有 AM setting 达到 100% recall；若 off/baseline 已满足质量，应直接关闭 AM。", flush=True)
        return
    best_am = max(am_perfect, key=lambda r: r.get("throughput_tok_s", 0.0))
    best_am_label = _summary_label(best_am, cartridge_enabled, is_c4, args)
    best_am_tps = best_am.get("throughput_tok_s", 0.0)
    am_parts = [f"am_best_100={best_am_label}", f"tok/s={best_am_tps:.2f}"]
    if off is not None and off.get("throughput_tok_s", 0.0) > 0:
        am_parts.append(f"vs_off={best_am_tps / off['throughput_tok_s']:.3f}x")
    if all_layer is not None and all_layer.get("throughput_tok_s", 0.0) > 0:
        am_parts.append(f"vs_all={best_am_tps / all_layer['throughput_tok_s']:.3f}x")
    print("[AM sweep] " + "  ".join(am_parts), flush=True)


def run_one_setting(llm, tokenizer, args, top_k: int):
    is_c4 = args.kv_quant_bits == 4
    if args.am_compact_blocks > 0:
        am_name = "AM-HighestAttnKeys" if args.am_compact_selector == "highest" else "AM-OMP"
        cluster_suffix = f" c={args.am_compact_num_clusters}" if args.am_compact_num_clusters > 1 else ""
        span_suffix = f" s={args.am_compact_num_key_spans}" if args.am_compact_num_key_spans > 1 else ""
        route_suffix = f" rtop={args.am_compact_route_top_k}" if args.am_compact_route_top_k > 1 else ""
        interval_suffix = (f"/{args.am_compact_decode_refit_interval}"
                           if args.am_compact_decode_refit and args.am_compact_decode_refit_interval > 1 else "")
        refit_suffix = f" refit={args.am_compact_decode_refit_mode}{interval_suffix}" if args.am_compact_decode_refit else ""
        layer_parts = []
        if args.am_compact_enable_layers is not None:
            layer_parts.append("layers=" + ",".join(str(x) for x in args.am_compact_enable_layers))
        else:
            if args.am_compact_skip_first_layers > 0:
                layer_parts.append(f"skip_first={args.am_compact_skip_first_layers}")
            if args.am_compact_skip_last_layers > 0:
                layer_parts.append(f"skip_last={args.am_compact_skip_last_layers}")
            if args.am_compact_layer_stride > 1:
                layer_parts.append(f"stride={args.am_compact_layer_stride}")
        layer_suffix = (" " + " ".join(layer_parts)) if layer_parts else ""
        label = f"{am_name} b={args.am_compact_blocks}{cluster_suffix}{span_suffix}{route_suffix}{refit_suffix}{layer_suffix}"
    elif args.kv_cartridge_blocks > 0:
        label = f"KV-Cartridge b={args.kv_cartridge_blocks}"
    elif is_c4:
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
        label=label,
        top_k=top_k,
        overall_accuracy=overall,
        throughput_tok_s=throughput,
        time_s=dt,
        per_setting=summary,
        details=results,
    )

    return out


def build_llm_kwargs(args, init_top_k: int) -> dict:
    return dict(
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        quest_top_k_blocks=init_top_k,
        quest_min_seq_len=args.quest_min_seq_len,
        kv_quant_bits=args.kv_quant_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        kv_cartridge_blocks=args.kv_cartridge_blocks,
        kv_cartridge_min_seq_len=args.kv_cartridge_min_seq_len,
        kv_cartridge_mode=args.kv_cartridge_mode,
        am_compact_blocks=args.am_compact_blocks,
        am_compact_selector=args.am_compact_selector,
        am_compact_min_seq_len=args.am_compact_min_seq_len,
        am_compact_score_method=args.am_compact_score_method,
        am_compact_beta_bound=args.am_compact_beta_bound,
        am_compact_ridge_lambda=args.am_compact_ridge_lambda,
        am_omp_candidate_pool_size=args.am_omp_candidate_pool_size,
        am_compact_cache_refresh_interval=args.am_compact_cache_refresh_interval,
        am_prefill_cache_ref_query_stride=args.am_prefill_cache_ref_query_stride,
        am_compact_num_clusters=args.am_compact_num_clusters,
        am_compact_route_top_k=args.am_compact_route_top_k,
        am_compact_num_key_spans=args.am_compact_num_key_spans,
        am_compact_decode_refit=args.am_compact_decode_refit,
        am_compact_decode_refit_mode=args.am_compact_decode_refit_mode,
        am_compact_decode_refit_interval=args.am_compact_decode_refit_interval,
        am_compact_skip_first_layers=args.am_compact_skip_first_layers,
        am_compact_skip_last_layers=args.am_compact_skip_last_layers,
        am_compact_enable_layers=args.am_compact_enable_layers,
        am_compact_layer_stride=args.am_compact_layer_stride,
        quantization=args.quantization,
        quant_group_size=args.quant_group_size,
        act_quant_bits=args.act_quant_bits,
        smoothquant_scale_path=args.smoothquant_scale_path,
        act_quant_skip_first=args.act_quant_skip_first,
        act_quant_skip_last=args.act_quant_skip_last,
        act_quant_skip_layers=args.act_quant_skip_layers,
    )


def main():
    args = parse_args()
    args_for_json = vars(args).copy()
    base_am_compact_blocks = args.am_compact_blocks

    if args.am_layer_sweep_print_only:
        # 不加载模型时无法读取真实层数；Qwen3-8B 为 32 层，足够用于检查 auto 展开。
        num_layers = 32
        settings = _build_am_layer_settings(args, num_layers)
        _print_am_layer_settings(settings, num_layers)
        return

    if args.model is None:
        raise ValueError("--model is required unless --am-layer-sweep-print-only is set")

    global AutoTokenizer, LLM, SamplingParams
    from transformers import AutoTokenizer
    from tinyvllm import LLM, SamplingParams

    is_c4 = args.kv_quant_bits == 4
    cartridge_enabled = args.kv_cartridge_blocks > 0
    am_enabled = args.am_compact_blocks > 0
    if am_enabled:
        # Attention Matching 是 Quest / KV-Cartridge 的替代 decode compaction 策略。
        top_k_list = [-1]
        init_top_k = -1
    elif cartridge_enabled:
        # KV-Cartridge v0 是 Quest 的替代稀疏策略：一次运行只评测一个 cartridge budget。
        top_k_list = [-1]
        init_top_k = -1
    elif is_c4:
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
        **build_llm_kwargs(args, init_top_k),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    am_layer_settings = []
    num_layers = 0
    if am_enabled:
        num_layers = llm.model_runner.config.hf_config.num_hidden_layers
        am_layer_settings = _build_am_layer_settings(args, num_layers)

    # 给 LLM 预热一次，避免第一个 setting 测的是 cold-start
    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)

    all_results = []
    if am_enabled:
        for idx, layer_setting in enumerate(am_layer_settings):
            enabled_layers = _apply_am_layer_setting(args, llm, layer_setting, num_layers, base_am_compact_blocks)
            # AM layer sweep 使用同一批 prompt 做质量归因；每个 setting 前清 prefix cache，避免吞吐虚高。
            if args.fixed_prompts or len(am_layer_settings) > 1 or idx > 0:
                cleared = clear_prefix_cache(llm)
                if cleared:
                    print(f"[prefix-cache] cleared metadata for {cleared} free blocks", flush=True)
            r = run_one_setting(llm, tokenizer, args, -1)
            r["am_layer_setting"] = layer_setting
            r["enabled_am_layers"] = list(enabled_layers)
            r["enabled_am_layer_count"] = len(enabled_layers)
            r["label"] = _summary_label(r, cartridge_enabled, is_c4, args)
            all_results.append(r)
    else:
        for top_k in top_k_list:
            if args.fixed_prompts:
                cleared = clear_prefix_cache(llm)
                if cleared:
                    print(f"[fixed-prompts] cleared prefix cache metadata for {cleared} free blocks", flush=True)
            r = run_one_setting(llm, tokenizer, args, top_k)
            all_results.append(r)

    print("\n========== SUMMARY ==========")
    off_result = next((r for r in all_results if r.get("am_layer_setting", {}).get("label") == "off"), None)
    off_tps = off_result.get("throughput_tok_s", 0.0) if off_result is not None else 0.0
    print(f"{'setting':>28} | {'acc':>7} | {'tok/s':>9} | {'vs_off':>7}")
    print("-" * 67)
    for r in all_results:
        label = _summary_label(r, cartridge_enabled, is_c4, args)
        vs_off = (r["throughput_tok_s"] / off_tps) if off_tps > 0 else None
        vs_off_s = f"{vs_off:.3f}x" if vs_off is not None else "-"
        print(f"{label:>28} | {r['overall_accuracy']*100:6.1f}% | {r['throughput_tok_s']:9.2f} | {vs_off_s:>7}")
    _print_best_am_layer_summary(all_results, cartridge_enabled, is_c4, args)

    with open(args.out_json, "w") as f:
        json.dump(dict(args=args_for_json, results=all_results), f, indent=2)
    print(f"\nDetails saved to {args.out_json}")


if __name__ == "__main__":
    main()
