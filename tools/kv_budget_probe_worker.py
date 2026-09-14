#!/usr/bin/env python3
"""Why does the graph path get less KV cache than the eager path?

The wall sweep on the graph path refused nearly every cell: the batch-2048 engine
reported a KV capacity of 173056 tokens, where the earlier eager wall sweep had
resident 2048x144 = 294912 tokens. Either the graph path really costs KV capacity,
or the sweep is now measuring a different engine for a reason unrelated to graphs.

Guessing at `allocate_kv_cache()` is how the last three mistakes happened, so this
prints the actual budget breakdown for one context, once per path. It reports the
scratch blocks the multi-sequence feature carves out, the auto block count the
memory arithmetic arrived at, and the visible blocks left for sequences.
"""

import argparse
import json
from pathlib import Path


def probe(*, model_path, context_length, batches, path_mode, gpu_memory_utilization,
          generated_tokens):
    from tinyvllm import LLM

    max_model_len = context_length + generated_tokens
    max_num_seqs = max(8, max(batches) + 4)
    extra = {}
    if path_mode == "msgraph":
        allowlist = tuple(sorted({int(b) for b in batches if int(b) > 1}))
        extra = {
            "multi_sequence_cuda_graphs": True,
            "multi_sequence_cuda_graph_batch_allowlist": allowlist,
            "multi_sequence_cuda_graph_max_entries": max(8, len(allowlist) * 2),
            "multi_sequence_cuda_graph_max_single_capture_ns": 120_000_000_000,
            "multi_sequence_cuda_graph_max_total_capture_ns": 900_000_000_000,
            "multi_sequence_cuda_graph_max_static_bytes": 1024 * 1024 * 1024,
            "multi_sequence_cuda_graph_max_reserved_bytes": 4 * 1024 * 1024 * 1024,
        }
    engine = LLM(
        model=model_path,
        enforce_eager=path_mode == "eager",
        max_model_len=max_model_len,
        max_num_batched_tokens=max(16384, max_model_len),
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        **extra,
    )
    runner = engine.model_runner
    config = runner.config
    block_size = int(config.kvcache_block_size)
    visible = int(config.num_kvcache_blocks)
    physical = int(getattr(runner, "_physical_num_kvcache_blocks", visible) or visible)
    scratch = len(getattr(runner, "_exact_graph_scratch_block_ids", ()) or ())
    burst_scratch = len(
        getattr(runner, "_exact_greedy_burst_scratch_block_ids", ()) or ()
    )
    spec_scratch = len(
        getattr(runner, "_spec_verify_capture_scratch_block_ids", ()) or ()
    )
    return {
        "path_mode": path_mode,
        "context_length": context_length,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "block_size": block_size,
        "visible_blocks": visible,
        "physical_blocks": physical,
        "decode_scratch_blocks": scratch,
        "burst_scratch_blocks": burst_scratch,
        "spec_verify_scratch_blocks": spec_scratch,
        "visible_tokens": visible * block_size,
        "weight_mem_bytes": int(getattr(runner, "weight_mem_bytes", 0) or 0),
        "max_resident_batch_at_context": (visible * block_size) // max(1, context_length),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--path-mode", choices=("eager", "msgraph"), required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batches", required=True, help="comma separated")
    parser.add_argument("--generated-tokens", type=int, default=34)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = probe(
        model_path=args.model_path,
        context_length=args.context_length,
        batches=[int(part) for part in args.batches.split(",") if part],
        path_mode=args.path_mode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generated_tokens=args.generated_tokens,
    )
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
