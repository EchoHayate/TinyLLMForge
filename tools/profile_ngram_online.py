"""Online dry-run profiler for n-gram speculative decoding opportunities.

This tool drives the real TinyLLM decode loop but does not change generated
outputs or mutate KV state beyond normal generation. It observes scheduled decode
rows, proposes n-gram drafts from each sequence history, and checks subsequent
real tokens against the pending draft prefix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_NGRAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "ngram.py")
_NGRAM_SPEC = importlib.util.spec_from_file_location("ngram_profile", _NGRAM_PATH)
ngram = importlib.util.module_from_spec(_NGRAM_SPEC)
sys.modules["ngram_profile"] = ngram
_NGRAM_SPEC.loader.exec_module(ngram)

NGramOnlineDryRunState = ngram.NGramOnlineDryRunState
NGramOnlineDryRunTotals = ngram.NGramOnlineDryRunTotals
ngram_online_dry_run_step = ngram.ngram_online_dry_run_step
summarize_online_dry_run_totals = ngram.summarize_online_dry_run_totals


DEFAULT_PROMPTS = [
    "Repeat the following phrase five times: alpha beta gamma alpha beta gamma.",
    "Write a short Python function and then explain each line briefly.",
    "The grass is green. The sky is blue. The sun is yellow. Here we go. " * 32
    + "What color is the sky?",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompt", action="append", default=None,
                   help="Prompt to profile. Can be passed multiple times. Defaults to a small built-in set.")
    p.add_argument("--max-output-len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--ngram-size", type=int, default=3)
    p.add_argument("--max-draft-tokens", type=int, default=4)
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--max-events", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--quantization", type=str, default=None, choices=[None, "int8", "int4", "int2"])
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--act-quant-bits", type=int, default=0, choices=[0, 8])
    p.add_argument("--smoothquant-scale-path", type=str, default=None)
    p.add_argument("--act-quant-skip-first", type=int, default=0)
    p.add_argument("--act-quant-skip-last", type=int, default=0)
    p.add_argument("--kv-quant-bits", type=int, default=0, choices=[0, 4, 8])
    p.add_argument("--kv-quant-group-size", type=int, default=128)
    p.add_argument("--quest-top-k-blocks", type=int, default=-1)
    p.add_argument("--quest-min-seq-len", type=int, default=512)
    return p.parse_args()


def cuda_sync_if_available():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _scheduled_sampled_rows(llm, before: dict[int, list[int]]) -> list[tuple[object, list[int], int]]:
    """Return scheduled rows that consumed one sampled output token.

    Mixed scheduling resets ``step_is_decode`` during postprocess, so the robust
    online signal is whether a scheduled sequence's token stream grew by one
    compared with the snapshot captured immediately before ``llm.step()``.
    """
    rows = []
    for seq in getattr(llm, "last_scheduled_seqs", []) or []:
        history_before = before.get(seq.seq_id)
        if history_before is None or len(seq.token_ids) <= len(history_before):
            continue
        rows.append((seq, history_before, seq.token_ids[len(history_before)]))
    return rows


def run_profile(args) -> dict:
    from tinyvllm import LLM, SamplingParams

    prompts = args.prompt or DEFAULT_PROMPTS
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        quantization=args.quantization,
        quant_group_size=args.quant_group_size,
        act_quant_bits=args.act_quant_bits,
        smoothquant_scale_path=args.smoothquant_scale_path,
        act_quant_skip_first=args.act_quant_skip_first,
        act_quant_skip_last=args.act_quant_skip_last,
        kv_quant_bits=args.kv_quant_bits,
        kv_quant_group_size=args.kv_quant_group_size,
        quest_top_k_blocks=args.quest_top_k_blocks,
        quest_min_seq_len=args.quest_min_seq_len,
    )
    sp = SamplingParams(temperature=args.temperature, ignore_eos=False, max_tokens=args.max_output_len)
    prompt_lens_by_seq_id = {}
    for prompt in prompts:
        prompt_len = len(llm.tokenizer.encode(prompt))
        llm.add_request(prompt, sp)
        seq = llm.scheduler.waiting[-1]
        prompt_lens_by_seq_id[seq.seq_id] = prompt_len

    states: dict[int, NGramOnlineDryRunState] = {}
    totals = NGramOnlineDryRunTotals()
    per_seq: dict[int, NGramOnlineDryRunTotals] = {}
    per_seq_states: dict[int, NGramOnlineDryRunState] = {}
    events = []
    outputs = {}
    step_records = []
    step_idx = 0
    t_start = time.perf_counter()
    cuda_sync_if_available()
    while not llm.is_finished():
        before = {
            seq.seq_id: list(seq.token_ids)
            for seq in list(llm.scheduler.waiting) + list(llm.scheduler.running)
        }
        t0 = time.perf_counter()
        out, num_tokens = llm.step()
        cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        batch_kind = getattr(llm, "last_batch_kind", None)
        sampled_rows = _scheduled_sampled_rows(llm, before)
        for seq, history_before, actual_next in sampled_rows:
            state = states.setdefault(seq.seq_id, NGramOnlineDryRunState(pending_tokens=[]))
            seq_totals = per_seq.setdefault(seq.seq_id, NGramOnlineDryRunTotals())
            event = ngram_online_dry_run_step(
                history_before,
                actual_next,
                state,
                totals,
                args.ngram_size,
                args.max_draft_tokens,
            )
            seq_state = per_seq_states.setdefault(seq.seq_id, NGramOnlineDryRunState(pending_tokens=[]))
            ngram_online_dry_run_step(
                history_before,
                actual_next,
                seq_state,
                seq_totals,
                args.ngram_size,
                args.max_draft_tokens,
            )
            if len(events) < args.max_events and (event["proposed"] or event["accepted"] or event["rejected"]):
                events.append({
                    "step": step_idx,
                    "seq_id": seq.seq_id,
                    "history_len": len(history_before),
                    **event,
                })
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids
        step_records.append({
            "step": step_idx,
            "batch_kind": batch_kind or ("prefill" if num_tokens > 0 else "decode"),
            "num_tokens": num_tokens,
            "dt_ms": dt_ms,
            "decode_rows": len(sampled_rows),
            "outputs": len(out),
        })
        step_idx += 1

    elapsed_s = time.perf_counter() - t_start
    summary = summarize_online_dry_run_totals(totals)
    summary["generated_outputs"] = len(outputs)
    summary["elapsed_s"] = elapsed_s
    summary["decode_steps"] = sum(1 for r in step_records if r["decode_rows"] > 0)
    summary["prefill_steps"] = sum(1 for r in step_records if r["batch_kind"] == "prefill")
    per_sequence = []
    for seq_id, stats in sorted(per_seq.items()):
        row = summarize_online_dry_run_totals(stats)
        row["seq_id"] = seq_id
        row["prompt_tokens"] = prompt_lens_by_seq_id.get(seq_id)
        row["output_tokens"] = len(outputs.get(seq_id, []))
        per_sequence.append(row)
    return {
        "args": vars(args),
        "summary": summary,
        "per_sequence": per_sequence,
        "events": events,
        "step_records": step_records,
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
