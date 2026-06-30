"""Profile Chunked Prefill v0 latency behavior.

This tool manually drives LLM.step() and records per-step latency. It is meant to
answer whether a long prompt insertion blocks decode steps less under chunked
prefill settings.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, math.ceil(len(values) * q) - 1))
    return values[idx]


def _kind_summary(records: list[dict], kind: str) -> dict[str, float | int]:
    rows = [r for r in records if r["kind"] == kind]
    lat = [float(r["dt_ms"]) for r in rows]
    total_ms = sum(lat)
    return {
        "steps": len(rows),
        "tokens": sum(int(r["tokens"]) for r in rows),
        "total_ms": total_ms,
        "mean_ms": total_ms / len(rows) if rows else 0.0,
        "p50_ms": percentile(lat, 0.50),
        "p95_ms": percentile(lat, 0.95),
        "max_ms": max(lat) if lat else 0.0,
    }


def summarize_steps(records: list[dict]) -> dict:
    decode_indices = [i for i, r in enumerate(records) if r["kind"] in ("decode", "mixed")]
    max_gap_steps = 0
    max_gap_ms = 0.0
    prev = None
    for idx in decode_indices:
        if prev is not None:
            gap_records = records[prev + 1:idx + 1]
            max_gap_steps = max(max_gap_steps, idx - prev)
            max_gap_ms = max(max_gap_ms, sum(float(r["dt_ms"]) for r in gap_records))
        prev = idx

    elapsed = 0.0
    first_output_step = -1
    first_output_ms = 0.0
    for r in records:
        elapsed += float(r["dt_ms"])
        if int(r.get("outputs", 0)) > 0:
            first_output_step = int(r["step"])
            first_output_ms = elapsed
            break

    return {
        "num_steps": len(records),
        "total_ms": sum(float(r["dt_ms"]) for r in records),
        "prefill": _kind_summary(records, "prefill"),
        "mixed": _kind_summary(records, "mixed"),
        "decode": _kind_summary(records, "decode"),
        "decode_gap": {
            "max_steps_between_decode": max_gap_steps,
            "max_ms_between_decode": max_gap_ms,
        },
        "first_output_step": first_output_step,
        "first_output_ms": first_output_ms,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--mode", choices=["default", "chunked", "mixed"], default="default")
    p.add_argument("--num-decode-seqs", type=int, default=4)
    p.add_argument("--decode-prompt-tokens", type=int, default=64)
    p.add_argument("--long-prompt-tokens", type=int, default=1024)
    p.add_argument("--short-insert-prompt-tokens", type=int, default=0)
    p.add_argument("--max-output-len", type=int, default=32)
    p.add_argument("--inject-long-after-decode-steps", type=int, default=2)
    p.add_argument("--inject-short-after-decode-steps", type=int, default=2)
    p.add_argument("--max-num-prefill-tokens-per-step", type=int, default=256)
    p.add_argument("--chunked-decode-first", action="store_true", default=False)
    p.add_argument("--max-consecutive-prefill-chunks", type=int, default=0)
    p.add_argument("--mixed-min-prompt-tokens", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--enforce-eager", action="store_true", default=False)
    p.add_argument("--skip-warmup", action="store_true", default=False)
    p.add_argument("--out-json", type=str, default=None)
    return p.parse_args()


def make_token_prompt(length: int, offset: int = 0) -> list[int]:
    # Keep ids in a conservative low range used by Qwen tokenizers.
    return [100 + ((i + offset) % 1000) for i in range(length)]


def cuda_sync_if_available():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def run_profile(args) -> dict:
    from tinyvllm import LLM, SamplingParams

    engine_kwargs = dict(
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.mode in ("chunked", "mixed"):
        engine_kwargs.update(
            max_num_prefill_tokens_per_step=args.max_num_prefill_tokens_per_step,
            chunked_prefill_decode_first=False if args.mode == "mixed" else args.chunked_decode_first,
            chunked_prefill_max_consecutive_chunks=args.max_consecutive_prefill_chunks,
            chunked_prefill_mixed_batch=(args.mode == "mixed"),
            chunked_prefill_mixed_min_prompt_tokens=args.mixed_min_prompt_tokens,
        )

    llm = LLM(args.model, **engine_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_output_len, ignore_eos=True)
    if not args.skip_warmup:
        warmup_sp = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
        warmup_prompts = [
            make_token_prompt(min(args.decode_prompt_tokens, 64), 9000 + i * 31)
            for i in range(max(1, args.num_decode_seqs))
        ]
        llm.generate(warmup_prompts, warmup_sp, use_tqdm=False)
        llm.generate([make_token_prompt(min(args.decode_prompt_tokens, 64), 9500)], warmup_sp, use_tqdm=False)
        cuda_sync_if_available()

    for i in range(args.num_decode_seqs):
        llm.add_request(make_token_prompt(args.decode_prompt_tokens, i * 17), sp)

    long_added = False
    short_added = False
    decode_steps_seen = 0
    outputs = {}
    records = []
    step_idx = 0

    cuda_sync_if_available()
    while not llm.is_finished():
        if (not long_added) and decode_steps_seen >= args.inject_long_after_decode_steps:
            llm.add_request(make_token_prompt(args.long_prompt_tokens, 777), sp)
            long_added = True
        if (
            args.short_insert_prompt_tokens > 0
            and (not short_added)
            and decode_steps_seen >= args.inject_short_after_decode_steps
        ):
            llm.add_request(make_token_prompt(args.short_insert_prompt_tokens, 1777), sp)
            short_added = True

        t0 = time.perf_counter()
        out, num_tokens = llm.step()
        cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        kind = llm.last_batch_kind if getattr(llm, "last_batch_kind", None) == "mixed" else ("prefill" if num_tokens > 0 else "decode")
        if kind in ("decode", "mixed"):
            decode_steps_seen += 1
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids

        records.append({
            "step": step_idx,
            "kind": kind,
            "tokens": abs(num_tokens),
            "dt_ms": dt_ms,
            "outputs": len(out),
        })
        step_idx += 1

    previews = []
    for seq_id in sorted(outputs)[:3]:
        previews.append(llm.tokenizer.decode(outputs[seq_id][:32], skip_special_tokens=True)[:120])
    return {
        "args": vars(args),
        "summary": summarize_steps(records),
        "records": records,
        "outputs": len(outputs),
        "text_previews": previews,
    }


def main():
    args = parse_args()
    result = run_profile(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
