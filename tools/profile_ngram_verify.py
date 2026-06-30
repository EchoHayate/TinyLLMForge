"""S2 profiler: KV-safe target verification for n-gram drafts.

This tool runs normal TinyLLM generation first, then replays the resulting token
stream. Whenever the online n-gram dry-run state proposes a draft, it asks the
same target model to verify the full draft in one temporary prefill forward.

The temporary verifier uses scratch KV blocks only. It does not attach speculative
KV to the live sequence, publish prefix-cache hashes, or change generated output.
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
_NGRAM_SPEC = importlib.util.spec_from_file_location("ngram_verify_profile", _NGRAM_PATH)
ngram = importlib.util.module_from_spec(_NGRAM_SPEC)
sys.modules["ngram_verify_profile"] = ngram
_NGRAM_SPEC.loader.exec_module(ngram)

NGramOnlineDryRunState = ngram.NGramOnlineDryRunState
NGramOnlineDryRunTotals = ngram.NGramOnlineDryRunTotals
NGramTargetVerifyStats = ngram.NGramTargetVerifyStats
count_accepted_prefix = ngram.count_accepted_prefix
ngram_online_dry_run_step = ngram.ngram_online_dry_run_step
summarize_online_dry_run_totals = ngram.summarize_online_dry_run_totals
summarize_target_verify_stats = ngram.summarize_target_verify_stats


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
    p.add_argument("--temperature", type=float, default=0.0,
                   help="S2 verifier currently requires greedy decoding, so this must be 0.0.")
    p.add_argument("--ngram-size", type=int, default=3)
    p.add_argument("--max-draft-tokens", type=int, default=4)
    p.add_argument("--max-verifications", type=int, default=128)
    p.add_argument("--max-events", type=int, default=64)
    p.add_argument("--out-json", type=str, default=None)
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


def _target_verify_draft_greedy(llm, history: list[int], draft_tokens: list[int]) -> list[int]:
    """Return target greedy predictions for each draft position using scratch KV.

    For draft ``[d0, d1, ...]`` and history ``h``, the verifier runs one prefill
    over ``h + [d0, d1, ... d{n-2}]``. Logits at ``h[-1]`` predict ``d0``;
    logits at ``d0`` predict ``d1``; and so on.
    """
    if not history:
        raise ValueError("history must be non-empty")
    if not draft_tokens:
        return []

    from tinyvllm.engine.sequence import Sequence
    from tinyvllm.utils.context import reset_context

    input_tokens = list(history) + list(draft_tokens[:-1])
    seq = Sequence(input_tokens)
    seq.prefill_chunk_start = 0
    seq.prefill_chunk_end = len(seq)
    seq.prefill_chunk_final = True
    block_manager = llm.scheduler.block_manager
    block_manager.allocate_ephemeral(seq)

    try:
        import torch
        from tinyvllm.utils.context import get_context

        input_ids, positions = llm.model_runner.prepare_prefill([seq])
        start = len(history) - 1
        end = start + len(draft_tokens)
        # ParallelLMHead normally keeps only the final prefill logit. For target
        # verification we need logits at h[-1], d0, ..., d{n-2} in one pass.
        get_context().logits_indices = llm.model_runner._list_to_cuda(
            list(range(start, end)), "verify_logits_indices", torch.int64)
        logits = llm.model_runner.run_model(input_ids, positions, is_prefill=True)
        predicted = logits.argmax(dim=-1).tolist()
        return [int(token_id) for token_id in predicted]
    finally:
        reset_context()
        if seq.block_table:
            block_manager.deallocate(seq)


def _combine_online_totals(items: list[NGramOnlineDryRunTotals]) -> NGramOnlineDryRunTotals:
    total = NGramOnlineDryRunTotals()
    for item in items:
        total.decode_positions += item.decode_positions
        total.draft_events += item.draft_events
        total.drafted_tokens += item.drafted_tokens
        total.accepted_tokens += item.accepted_tokens
        total.rejected_events += item.rejected_events
        total.completed_drafts += item.completed_drafts
        total.no_draft_positions += item.no_draft_positions
    return total


def _combine_verify_stats(items: list[NGramTargetVerifyStats]) -> NGramTargetVerifyStats:
    total = NGramTargetVerifyStats()
    for item in items:
        total.verify_events += item.verify_events
        total.verified_tokens += item.verified_tokens
        total.target_accepted_tokens += item.target_accepted_tokens
        total.replay_accepted_tokens += item.replay_accepted_tokens
        total.mismatched_events += item.mismatched_events
        total.truncated_future_events += item.truncated_future_events
    return total


def _replay_and_verify_one(llm, token_ids: list[int], prompt_len: int, args, events: list[dict]) -> tuple[NGramOnlineDryRunTotals, NGramTargetVerifyStats]:
    state = NGramOnlineDryRunState(pending_tokens=[])
    online_totals = NGramOnlineDryRunTotals()
    verify_stats = NGramTargetVerifyStats()

    for pos in range(prompt_len, len(token_ids)):
        history = token_ids[:pos]
        actual_next = token_ids[pos]
        event = ngram_online_dry_run_step(
            history,
            actual_next,
            state,
            online_totals,
            args.ngram_size,
            args.max_draft_tokens,
        )
        if not event["proposed"]:
            continue
        if verify_stats.verify_events >= args.max_verifications:
            continue

        draft_tokens = list(event["draft_tokens"])
        target_tokens = _target_verify_draft_greedy(llm, history, draft_tokens)
        future_tokens = token_ids[pos:pos + len(draft_tokens)]
        target_accepted = count_accepted_prefix(draft_tokens, target_tokens)
        replay_accepted = count_accepted_prefix(draft_tokens, future_tokens)
        comparable_target_accepted = min(target_accepted, len(future_tokens))
        mismatch = comparable_target_accepted != replay_accepted

        verify_stats.verify_events += 1
        verify_stats.verified_tokens += len(draft_tokens)
        verify_stats.target_accepted_tokens += target_accepted
        verify_stats.replay_accepted_tokens += replay_accepted
        if len(future_tokens) < len(draft_tokens):
            verify_stats.truncated_future_events += 1
        if mismatch:
            verify_stats.mismatched_events += 1

        if len(events) < args.max_events:
            events.append({
                "pos": pos,
                "history_len": len(history),
                "draft_tokens": draft_tokens,
                "target_tokens": target_tokens,
                "future_tokens": future_tokens,
                "target_accepted": target_accepted,
                "replay_accepted": replay_accepted,
                "mismatch": mismatch,
            })

    return online_totals, verify_stats


def run_profile(args) -> dict:
    if args.temperature != 0.0:
        raise ValueError("S2 target verification currently supports greedy decoding only (--temperature 0.0)")

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
    sps = [SamplingParams(temperature=args.temperature, ignore_eos=False, max_tokens=args.max_output_len)
           for _ in prompts]

    cuda_sync_if_available()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sps, use_tqdm=False)
    cuda_sync_if_available()
    generation_elapsed_s = time.perf_counter() - t0

    per_prompt = []
    online_items = []
    verify_items = []
    events = []
    t1 = time.perf_counter()
    for idx, (prompt, out) in enumerate(zip(prompts, outputs)):
        prompt_ids = llm.tokenizer.encode(prompt)
        token_ids = prompt_ids + out["token_ids"]
        online_totals, verify_stats = _replay_and_verify_one(llm, token_ids, len(prompt_ids), args, events)
        online_items.append(online_totals)
        verify_items.append(verify_stats)
        row = {
            "prompt_index": idx,
            "prompt_chars": len(prompt),
            "prompt_tokens": len(prompt_ids),
            "output_tokens": len(out["token_ids"]),
            "text_preview": out["text"][:160],
            "online": summarize_online_dry_run_totals(online_totals),
            "target_verify": summarize_target_verify_stats(verify_stats),
        }
        per_prompt.append(row)
    cuda_sync_if_available()
    verify_elapsed_s = time.perf_counter() - t1

    online_total = _combine_online_totals(online_items)
    verify_total = _combine_verify_stats(verify_items)
    return {
        "args": vars(args),
        "summary": {
            "online": summarize_online_dry_run_totals(online_total),
            "target_verify": summarize_target_verify_stats(verify_total),
            "generation_elapsed_s": generation_elapsed_s,
            "verify_elapsed_s": verify_elapsed_s,
        },
        "per_prompt": per_prompt,
        "events": events,
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
