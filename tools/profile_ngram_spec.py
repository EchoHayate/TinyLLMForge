"""Profile offline n-gram speculative decoding acceptance potential.

This tool does not change the TinyLLM decode loop. It runs normal generation,
then replays the final token stream to estimate how often a prompt/history based
n-gram drafter would have proposed tokens and how many would have matched.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tinyvllm import LLM, SamplingParams
from tinyvllm.speculative.ngram import NGramReplayStats, replay_ngram_acceptance, summarize_replay_stats


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
    # Pass-through engine knobs used by existing experiments.
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


def combine_stats(stats: list[NGramReplayStats]) -> NGramReplayStats:
    return NGramReplayStats(
        positions=sum(s.positions for s in stats),
        drafted_tokens=sum(s.drafted_tokens for s in stats),
        accepted_tokens=sum(s.accepted_tokens for s in stats),
        draft_events=sum(s.draft_events for s in stats),
    )


def main():
    args = parse_args()
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
    outputs = llm.generate(prompts, sps, use_tqdm=False)

    per_prompt = []
    stats_list = []
    for prompt, out in zip(prompts, outputs):
        prompt_ids = llm.tokenizer.encode(prompt)
        token_ids = prompt_ids + out["token_ids"]
        stats = replay_ngram_acceptance(
            token_ids,
            prompt_len=len(prompt_ids),
            ngram_size=args.ngram_size,
            max_draft_tokens=args.max_draft_tokens,
        )
        stats_list.append(stats)
        per_prompt.append({
            "prompt_chars": len(prompt),
            "prompt_tokens": len(prompt_ids),
            "output_tokens": len(out["token_ids"]),
            "text_preview": out["text"][:160],
            **summarize_replay_stats(stats),
        })

    total = combine_stats(stats_list)
    result = {
        "args": vars(args),
        "summary": summarize_replay_stats(total),
        "per_prompt": per_prompt,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
