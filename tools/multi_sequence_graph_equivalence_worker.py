#!/usr/bin/env python3
"""Check that a multi-sequence decode CUDA Graph produces the same tokens as eager.

Why this exists
---------------
`model_runner.py` used to fail closed to eager for every decode batch above one,
with this comment:

    FlashAttention decode replay is only correctness-validated for one sequence.
    Multi-sequence captured graphs can corrupt rows after the first one.

Enabling `multi_sequence_cuda_graphs` and fixing the capture cut the measured
decode step at batch 4 from about 33 ms to about 5 ms, and cut the GATE A
constant from 40 ms to 12 ms. A faster path that returns different tokens is not
a faster path, it is a broken one, and every number measured on it would be
worthless. Timing alone cannot tell the two apart: the step times scale with
resident tokens and equal-product cells agree within 2%, which is consistent with
a correct forward but does not prove one.

So this worker decodes the same prompts twice, once per execution path, greedily,
and writes the produced token ids. A separate comparison step diffs them. Greedy
sampling is used because it makes the comparison exact: any divergence is either
numerics reordering inside the graph or a genuinely corrupted row, and both are
findings.

The engine builds a torch.distributed process group on construction and refuses
to do it twice in one process, so each path runs as its own process and this
worker only ever loads one engine.
"""

import argparse
import json
import random
from pathlib import Path


def build_prompts(*, prompt_length, batch, vocab_size, seed):
    """Independent random prompts, so no two sequences share a prefix.

    Shared prefixes would let the block manager serve later sequences from cached
    blocks, which changes what the decode batch actually reads and would make an
    agreement between the two paths mean less than it appears to.
    """
    rng = random.Random(seed)
    return [
        [rng.randrange(8, vocab_size - 8) for _ in range(prompt_length)]
        for _ in range(batch)
    ]


def multi_sequence_graph_kwargs(batch):
    """Mirror the settings the GATE A worker measures under.

    The allowlist has to contain the batch or the engine quietly serves it eager,
    and the capture budgets have to be lifted or the first capture in the process
    overruns the 2 s default while it pays torch.compile for the shape.
    """
    return {
        "multi_sequence_cuda_graphs": True,
        "multi_sequence_cuda_graph_batch_allowlist": (int(batch),),
        "multi_sequence_cuda_graph_max_entries": 8,
        "multi_sequence_cuda_graph_max_single_capture_ns": 120_000_000_000,
        "multi_sequence_cuda_graph_max_total_capture_ns": 900_000_000_000,
        "multi_sequence_cuda_graph_max_static_bytes": 1024 * 1024 * 1024,
        "multi_sequence_cuda_graph_max_reserved_bytes": 4 * 1024 * 1024 * 1024,
    }


def resolve_config(engine):
    """Find the object that actually carries `config`.

    `LLM` does not expose `config` on itself in this build, and a lookup that
    gives up at the first AttributeError is how the earlier sweep artifacts ended
    up with every engine identity field recorded as null.
    """
    for path in ("", "llm_engine", "engine", "model_runner"):
        target = engine
        if path:
            try:
                for attribute in path.split("."):
                    target = getattr(target, attribute)
            except AttributeError:
                continue
        config = getattr(target, "config", None)
        if config is not None:
            return config
    raise AttributeError("engine exposes no config")


def dispatch_label(event):
    if event is None:
        return "unobserved"
    dispatch = str(event.get("dispatch") or "unknown")
    if dispatch == "graph":
        return "graph"
    reason = event.get("fallback_reason") or event.get("cache_state") or "unspecified"
    return f"eager:{reason}"


def run(*, model_path, path_mode, prompt_length, batch, max_tokens, seed,
        gpu_memory_utilization):
    from tinyvllm import LLM
    from tinyvllm.sampling_params import SamplingParams

    extra = {} if path_mode == "eager" else multi_sequence_graph_kwargs(batch)
    engine = LLM(
        model=model_path,
        enforce_eager=path_mode == "eager",
        max_model_len=prompt_length + max_tokens + 16,
        max_num_batched_tokens=max(16384, prompt_length + max_tokens + 16),
        max_num_seqs=batch + 4,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        **extra,
    )
    vocab_size = int(resolve_config(engine).hf_config.vocab_size)
    prompts = build_prompts(
        prompt_length=prompt_length,
        batch=batch,
        vocab_size=vocab_size,
        seed=seed,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    for prompt in prompts:
        engine.add_request(prompt, params)

    runner = getattr(engine, "model_runner", None)
    labels = []
    last_step_id = None
    completions = {}
    while not engine.is_finished():
        outputs, num_tokens = engine.step()
        if num_tokens <= 0 and runner is not None:
            reader = getattr(runner, "cuda_graph_dispatch_observation", None)
            event = reader() if callable(reader) else None
            step_id = None if event is None else event.get("step_id")
            if event is None:
                labels.append("unobserved")
            elif step_id is not None and step_id == last_step_id:
                labels.append("unpublished")
            else:
                last_step_id = step_id
                labels.append(dispatch_label(event))
        for sequence_id, token_ids in outputs or []:
            completions[int(sequence_id)] = [int(token) for token in token_ids]

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {
        "path_mode": path_mode,
        "model_path": model_path,
        "prompt_length": prompt_length,
        "batch": batch,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt_sha_first_tokens": [prompt[:4] for prompt in prompts],
        "decode_dispatch_counts": dict(sorted(counts.items())),
        "decode_steps": len(labels),
        "completions": {str(key): value for key, value in sorted(completions.items())},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--path-mode", choices=("eager", "msgraph"), required=True)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = run(
        model_path=args.model_path,
        path_mode=args.path_mode,
        prompt_length=args.prompt_length,
        batch=args.batch,
        max_tokens=args.max_tokens,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "%s: %d sequences, %d decode steps, dispatch %s"
        % (
            args.path_mode,
            len(payload["completions"]),
            payload["decode_steps"],
            payload["decode_dispatch_counts"],
        )
    )


if __name__ == "__main__":
    main()
