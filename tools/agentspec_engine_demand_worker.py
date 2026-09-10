#!/usr/bin/env python3
"""Stage 1a-bis worker: measure actor demand D on the serving path.

Stage 1a measured the drafter tax with an eager Hugging Face decode
loop. The decomposition it reported showed that loop was overhead
bound: the 0.6B decode step cost 0.60 to 0.78 of the 8B step where
compute predicts roughly an order of magnitude. Every Stage 0
threshold is a function of D, so a D inflated by harness overhead
invalidates the thresholds, not just the precision.

This worker re-measures on ``tinyvllm``'s own engine, which is the
serving path this repository actually ships: paged KV cache, batched
scheduler, and CUDA graph replay for decode. It drives the engine one
scheduler step at a time so prefill and decode are timed separately,
and it runs both ``enforce_eager=False`` and ``enforce_eager=True`` so
the CUDA graph contribution is attributable rather than assumed.

Per agent step, with trajectory length L and A action tokens. The
engine emits the first action token out of the prefill step itself, so
producing A tokens costs one prefill plus A-1 decode steps:

    D            = actor_prefill(L)   + (A - 1) * actor_step
    G_text       = drafter_prefill(L) + (A - 1) * drafter_step
    G_code       = drafter_prefill(L) + head
    G_code_ckv   = drafter_prefill(B) + head

where B is the compressed context budget and ``head`` is one linear
projection over an action code vocabulary. The code head is randomly
initialised on purpose: head weights move accuracy, not cost. This
worker measures cost only and claims no match probability.

A fresh random prompt is used for every repetition because the engine
carries prefix caching, and reusing a prompt would make prefill look
free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time


SCHEMA_VERSION = 1
DEFAULT_ACTION_TOKENS = 32
DEFAULT_CONTEXT_LENGTHS = (1024, 4096, 16384)
DEFAULT_COMPRESSED_BUDGET = 512
DEFAULT_CODE_VOCABULARY = 4096
DEFAULT_REPETITIONS = 5
DEFAULT_WARMUP = 2
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


def _percentile(samples, fraction):
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("no samples")
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _summarise(samples):
    return {
        "count": len(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "p90_seconds": _percentile(samples, 0.9),
    }


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _model_identity(path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return {
        "path": path,
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_hidden_layers": int(
            getattr(config, "num_hidden_layers", 0)
        ),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "config_digest": _digest(config.to_json_string()),
    }


def _measure_head(hidden_size, code_vocabulary, repetitions, warmup):
    """Time one action-code projection over the last hidden state."""

    import torch

    device = torch.device("cuda:0")
    head = torch.nn.Linear(hidden_size, code_vocabulary)
    head.eval()
    head.to(device=device, dtype=torch.bfloat16)
    hidden = torch.randn(
        1,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        for _ in range(warmup):
            head(hidden)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            head(hidden)
            stop.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(stop) / 1000.0)
    return _summarise(samples)


def _random_prompt(length, vocab_size, rng):
    return [rng.randrange(16, max(32, vocab_size - 16)) for _ in range(length)]


def _time_one_request(engine, prompt_ids, action_tokens):
    """Drive one request step by step, timing prefill and decode.

    The engine reports a positive token count for a prefill step and a
    negative one for a decode step, which is how the two phases are
    separated without reaching into engine internals.
    """

    import torch
    from tinyvllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=action_tokens,
        ignore_eos=True,
    )
    engine.add_request(prompt_ids, params)
    prefill_seconds = None
    decode_seconds = []
    while not engine.is_finished():
        torch.cuda.synchronize()
        started = time.perf_counter()
        _output, num_tokens = engine.step(completion_only=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if num_tokens > 0:
            if prefill_seconds is None:
                prefill_seconds = elapsed
            else:
                # A chunked or re-scheduled prefill would break the
                # decomposition, so fail loudly instead of averaging.
                raise RuntimeError(
                    "unexpected second prefill step; prompt was "
                    "chunked and the decomposition is invalid"
                )
        else:
            decode_seconds.append(elapsed)
    if prefill_seconds is None:
        raise RuntimeError("no prefill step observed")
    # The prefill step already emits the first sampled token, so a
    # request for A tokens shows A-1 decode steps. Anything else means
    # the phase split is not what this decomposition assumes.
    expected_decode_steps = max(action_tokens - 1, 0)
    if len(decode_seconds) != expected_decode_steps:
        raise RuntimeError(
            "expected %d decode steps, saw %d"
            % (expected_decode_steps, len(decode_seconds))
        )
    return prefill_seconds, decode_seconds


def _measure_model(
    model_path,
    identity,
    context_lengths,
    action_tokens,
    repetitions,
    warmup,
    enforce_eager,
    gpu_memory_utilization,
    seed,
):
    import random

    from tinyvllm import LLM

    max_context = max(context_lengths)
    max_model_len = max_context + action_tokens + 64
    engine = LLM(
        model=model_path,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(16384, max_model_len),
        max_num_seqs=8,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    rng = random.Random(seed)
    rows = {}
    try:
        for context_length in context_lengths:
            prefill_samples = []
            decode_samples = []
            for index in range(warmup + repetitions):
                prompt = _random_prompt(
                    context_length,
                    identity["vocab_size"],
                    rng,
                )
                prefill, decodes = _time_one_request(
                    engine,
                    prompt,
                    action_tokens,
                )
                if index < warmup:
                    continue
                prefill_samples.append(prefill)
                decode_samples.extend(decodes)
            rows[context_length] = {
                "prefill": _summarise(prefill_samples),
                "decode_step": _summarise(decode_samples),
            }
    finally:
        engine.exit()
    return rows


def build_payload(args):
    import torch

    actor_identity = _model_identity(args.actor_model)
    drafter_identity = _model_identity(args.drafter_model)
    head_row = _measure_head(
        drafter_identity["hidden_size"],
        args.code_vocabulary,
        max(args.repetitions * 4, 20),
        args.warmup * 4,
    )
    contexts = sorted(set(args.context_lengths))
    drafter_contexts = sorted(
        set(contexts) | {min(args.compressed_budget, min(contexts))}
    )

    modes = {}
    for mode_name, enforce_eager in (
        ("cuda_graph", False),
        ("eager", True),
    ):
        if mode_name == "eager" and args.skip_eager:
            continue
        actor_rows = _measure_model(
            args.actor_model,
            actor_identity,
            contexts,
            args.action_tokens,
            args.repetitions,
            args.warmup,
            enforce_eager,
            args.gpu_memory_utilization,
            args.seed,
        )
        drafter_rows = _measure_model(
            args.drafter_model,
            drafter_identity,
            drafter_contexts,
            args.action_tokens,
            args.repetitions,
            args.warmup,
            enforce_eager,
            args.gpu_memory_utilization,
            args.seed + 1,
        )
        compressed = min(args.compressed_budget, min(contexts))
        head_seconds = head_row["median_seconds"]
        decode_steps = max(args.action_tokens - 1, 0)
        rows = []
        for context_length in contexts:
            actor = actor_rows[context_length]
            drafter = drafter_rows[context_length]
            actor_prefill = actor["prefill"]["median_seconds"]
            actor_step = actor["decode_step"]["median_seconds"]
            drafter_prefill = drafter["prefill"]["median_seconds"]
            drafter_step = drafter["decode_step"]["median_seconds"]
            compressed_prefill = drafter_rows[compressed][
                "prefill"
            ]["median_seconds"]
            demand = actor_prefill + decode_steps * actor_step
            text_cost = drafter_prefill + decode_steps * drafter_step
            code_cost = drafter_prefill + head_seconds
            ckv_cost = compressed_prefill + head_seconds
            rows.append(
                {
                    "context_length": context_length,
                    "compressed_context_length": compressed,
                    "action_tokens": args.action_tokens,
                    "decode_steps_per_action": decode_steps,
                    "actor_prefill_seconds": actor_prefill,
                    "actor_decode_step_seconds": actor_step,
                    "drafter_prefill_seconds": drafter_prefill,
                    "drafter_decode_step_seconds": drafter_step,
                    "compressed_prefill_seconds": compressed_prefill,
                    "actor_demand_seconds": demand,
                    "decode_share_of_demand": (
                        decode_steps * actor_step / demand
                    ),
                    "decode_step_ratio_drafter_over_actor": (
                        drafter_step / actor_step
                    ),
                    "arm_gpu_seconds": {
                        "text_drafter": text_cost,
                        "code_drafter": code_cost,
                        "code_drafter_ckv": ckv_cost,
                    },
                    "measured_draft_gpu_tax": {
                        "text_drafter": text_cost / demand,
                        "code_drafter": code_cost / demand,
                        "code_drafter_ckv": ckv_cost / demand,
                    },
                    "raw": {
                        "actor": actor,
                        "drafter": drafter,
                    },
                }
            )
        modes[mode_name] = {
            "enforce_eager": enforce_eager,
            "rows": rows,
        }

    payload = {
        "worker": "agentspec_engine_demand",
        "schema_version": SCHEMA_VERSION,
        "evidence_valid_for_gate": True,
        "claim_boundary": (
            "cost measurement on the tinyvllm serving path; the code "
            "head is randomly initialised and no match probability "
            "is measured"
        ),
        "serving_path": "tinyvllm.LLM",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "code_vocabulary": args.code_vocabulary,
        "code_head": head_row,
        "actor_identity": actor_identity,
        "drafter_identity": drafter_identity,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seed": args.seed,
        "modes": modes,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    payload["payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def summarise(payload):
    lines = [
        "worker          %s" % payload["worker"],
        "serving path    %s" % payload["serving_path"],
        "device          %s" % payload["cuda_device_name"],
        "actor           %s" % payload["actor_identity"]["path"],
        "drafter         %s" % payload["drafter_identity"]["path"],
        "code head       %.6f s" % payload["code_head"][
            "median_seconds"
        ],
    ]
    for mode_name, mode in sorted(payload["modes"].items()):
        lines.append("")
        lines.append("mode %s" % mode_name)
        lines.append(
            "context      D_s  a_pre_s  a_step_ms  d_step_ms"
            "  ratio  tau_text  tau_code  tau_ckv"
        )
        for row in mode["rows"]:
            tax = row["measured_draft_gpu_tax"]
            lines.append(
                "%7d  %7.4f  %7.4f  %9.3f  %9.3f  %5.3f"
                "  %8.4f  %8.4f  %7.4f"
                % (
                    row["context_length"],
                    row["actor_demand_seconds"],
                    row["actor_prefill_seconds"],
                    row["actor_decode_step_seconds"] * 1000.0,
                    row["drafter_decode_step_seconds"] * 1000.0,
                    row["decode_step_ratio_drafter_over_actor"],
                    tax["text_drafter"],
                    tax["code_drafter"],
                    tax["code_drafter_ckv"],
                )
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure actor demand on the tinyvllm serving path",
    )
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument(
        "--action-tokens",
        type=int,
        default=DEFAULT_ACTION_TOKENS,
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
    )
    parser.add_argument(
        "--compressed-budget",
        type=int,
        default=DEFAULT_COMPRESSED_BUDGET,
    )
    parser.add_argument(
        "--code-vocabulary",
        type=int,
        default=DEFAULT_CODE_VOCABULARY,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = build_payload(args)
    if args.output is not None:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
    print(summarise(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
