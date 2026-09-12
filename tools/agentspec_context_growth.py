#!/usr/bin/env python3
"""Stage 1b step 2: measure how fast an agent's context actually grows.

Why this exists
---------------
Stage 1a-bis measured the actor's demand ``D`` with prefix caching
deliberately defeated: every repetition used a fresh random prompt, so
every turn paid a full cold prefill over the whole context. That is the
right way to price *one prefill*. It is the wrong way to price *one
agent turn*, because an agent's context grows by appending, and
``tinyvllm/engine/block_manager.py`` keeps block hashes alive across
deallocation, so a returning agent re-prefills only what is new.

The erratum showed that this substitution decided both surviving
conclusions of the line. It also showed the one number that could
reverse the erratum: if a real agent appends enough fresh tokens per
turn, the warm demand stays large, and speculation can still pay. The
break-even was roughly 4000 fresh tokens per turn. Nobody had measured
the real figure. This script measures it.

What is counted, and why it is not the naive delta
--------------------------------------------------
Three different quantities are easy to confuse:

``assistant_tokens``
    Tokens the actor generated itself. These are **already resident**:
    decoding writes KV. They grow the context but they are not
    re-prefilled.

``observation_tokens``
    Tokens appended by the environment since the actor last ran: tool
    output, user turns. This is genuinely new text.

``prefill_tokens``
    What the engine must actually recompute, which is neither of the
    above. The block manager caches at ``block_size`` granularity, so
    the trailing partial block of the previous turn is not reusable::

        resident = floor((ctx_before + assistant) / block) * block
        prefill  = ctx_after - resident
                 = observation + wrapper + ((ctx_before + assistant) mod block)

    The modulus term averages half a block and is not negligible: at
    ``block_size=256`` it is comparable to a small tool observation.

``prefill_tokens`` is the quantity that enters ``D``, so it is the one
the cost model is fed.

No corpus text is emitted. The payload carries token counts,
percentiles and derived costs only.

Usage::

    python3 tools/agentspec_context_growth.py \
        --corpus swe_agent --input swe_agent_shard0.parquet \
        --tokenizer /tmp/qwen3_tokenizer.json --output growth.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import types


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Measured on one A100 80GB PCIe, cuda_graph mode, Qwen3-8B actor.
# experiments/agentspec_engine_demand/engine-demand-measure-a100-20260910-2258
_MEASURED_PREFILL = {1024: 0.0814409558661282, 4096: 0.31958, 16384: 1.55734}
_MEASURED_DECODE = {1024: 0.01320198504254222, 4096: 0.013843, 16384: 0.015521}
_ACTION_DECODE_STEPS = 31

# Step 0 training-free predictors, exact action match over eligible steps.
_TRIGRAM_P = {"apigen": 0.2078, "swe_agent": 0.1129}


def _load_module(name, filename):
    """Import a dependency-free helper without importing torch."""
    path = os.path.join(_REPO_ROOT, "tools", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_cost_model():
    parent = types.ModuleType("tinyvllm")
    parent.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
    sys.modules.setdefault("tinyvllm", parent)
    child = types.ModuleType("tinyvllm.agentspec")
    child.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "agentspec")]
    sys.modules.setdefault("tinyvllm.agentspec", child)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return importlib.import_module("tinyvllm.agentspec.cost_model")


def _fit_prefill():
    """Least-squares fit of prefill(L) = a*L + b*L^2 over the measured points.

    Prefill is superlinear because attention is quadratic in the prompt,
    so a straight line understates long contexts badly. Two terms are
    enough for three points; the residuals are reported so the reader
    can see how much to trust the extrapolation.
    """
    xs = sorted(_MEASURED_PREFILL)
    rows = [(x, x * x) for x in xs]
    ys = [_MEASURED_PREFILL[x] for x in xs]
    s11 = sum(r[0] * r[0] for r in rows)
    s12 = sum(r[0] * r[1] for r in rows)
    s22 = sum(r[1] * r[1] for r in rows)
    t1 = sum(r[0] * y for r, y in zip(rows, ys))
    t2 = sum(r[1] * y for r, y in zip(rows, ys))
    det = s11 * s22 - s12 * s12
    a = (t1 * s22 - t2 * s12) / det
    b = (t2 * s11 - t1 * s12) / det
    residuals = {
        x: (a * x + b * x * x) - _MEASURED_PREFILL[x] for x in xs
    }
    return a, b, residuals


def _fit_decode():
    """Linear fit of decode_step(L) = base + slope*L.

    Everything in a decode step except attention over history is weight
    traffic and does not move with context, so a line is the right
    shape here.
    """
    xs = sorted(_MEASURED_DECODE)
    lo, hi = xs[0], xs[-1]
    slope = (_MEASURED_DECODE[hi] - _MEASURED_DECODE[lo]) / (hi - lo)
    base = _MEASURED_DECODE[lo] - slope * lo
    residuals = {
        x: (base + slope * x) - _MEASURED_DECODE[x] for x in xs
    }
    return base, slope, residuals


class _Percentiles:
    """Streaming-free percentile holder; the corpora fit in memory."""

    def __init__(self):
        self.values = []

    def add(self, value):
        self.values.append(value)

    def summary(self):
        if not self.values:
            return None
        ordered = sorted(self.values)
        n = len(ordered)

        def pct(q):
            if n == 1:
                return float(ordered[0])
            idx = min(int(math.ceil(q * n)) - 1, n - 1)
            return float(ordered[max(idx, 0)])

        return {
            "count": n,
            "mean": sum(ordered) / n,
            "min": float(ordered[0]),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": float(ordered[-1]),
        }


def _render(role, content):
    """Qwen3 chat wrapper, so wrapper tokens are counted, not guessed."""
    return "<|im_start|>%s\n%s<|im_end|>\n" % (role, content or "")


def _iter_apigen(path, limit):
    """Yield (trace_id, [(role, text), ...]) for APIGen-MT traces.

    ``function_call`` and ``gpt`` are both assistant turns and both are
    separate model invocations in the reconstructed loop: one replies to
    the user, the other emits a call. ``observation`` and ``human`` are
    both environment input as far as the KV cache is concerned.
    """
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    role_map = {
        "human": "user",
        "observation": "user",
        "gpt": "assistant",
        "function_call": "assistant",
    }
    for index, record in enumerate(records):
        if limit and index >= limit:
            return
        messages = []
        system = record.get("system") or ""
        tools = record.get("tools")
        if tools:
            system = "%s\n%s" % (
                system,
                tools if isinstance(tools, str) else json.dumps(tools),
            )
        messages.append(("system", system))
        for turn in record.get("conversations", []):
            role = role_map.get(turn.get("from"))
            if role is None:
                continue
            messages.append((role, turn.get("value") or ""))
        yield "apigen-%d" % index, messages


def _iter_swe_agent(path, limit):
    """Yield (trace_id, [(role, text), ...]) for SWE-agent trajectories."""
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    emitted = 0
    role_map = {"system": "system", "user": "user", "ai": "assistant"}
    for group in range(reader.metadata.num_row_groups):
        rows = reader.read_row_group(group).to_pylist()
        for row in rows:
            if limit and emitted >= limit:
                return
            messages = []
            for message in row.get("trajectory", []):
                role = role_map.get(message.get("role"))
                if role is None:
                    continue
                if role == "system":
                    text = message.get("system_prompt") or ""
                else:
                    text = message.get("text") or ""
                    if text == "None":
                        text = ""
                messages.append((role, text))
            if not messages:
                continue
            emitted += 1
            yield "swe-%s" % row.get("instance_id"), messages


_ITERATORS = {"apigen": _iter_apigen, "swe_agent": _iter_swe_agent}


def walk_trace(messages, encode, block_size):
    """Replay one trace and yield the per-turn KV accounting.

    ``ctx_before`` is the prompt length the actor sees on this turn.
    ``prefill`` is what the engine must recompute given block-granular
    prefix reuse from the previous turn.
    """
    resident_hashed = 0
    ctx = 0
    pending_observation = 0
    first_turn = True
    for role, text in messages:
        length = encode(_render(role, text))
        if role != "assistant":
            pending_observation += length
            ctx += length
            continue
        ctx_before = ctx
        prefill = ctx_before - resident_hashed
        yield {
            "ctx_before": ctx_before,
            "observation": pending_observation,
            "prefill": max(prefill, 0),
            "assistant": length,
            "first_turn": first_turn,
        }
        first_turn = False
        ctx += length
        resident_hashed = (ctx // block_size) * block_size
        pending_observation = 0


def build_payload(corpus, path, tokenizer_path, limit, block_size, args):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    cache = {}

    def encode(text):
        # Traces repeat system prompts and identical observations often
        # enough that memoising pays for itself several times over.
        hit = cache.get(text)
        if hit is None:
            hit = len(tokenizer.encode(text, add_special_tokens=False).ids)
            if len(cache) < 200000:
                cache[text] = hit
        return hit

    cost_model = _load_cost_model()
    pa, pb, p_res = _fit_prefill()
    dbase, dslope, d_res = _fit_decode()

    def prefill_seconds(tokens):
        if tokens <= 0:
            return 0.0
        return pa * tokens + pb * tokens * tokens

    def decode_step_seconds(ctx):
        return dbase + dslope * max(ctx, 0)

    buckets = [(0, 2048), (2048, 8192), (8192, 32768), (32768, 1 << 30)]
    bucket_names = ["ctx<2k", "2k-8k", "8k-32k", "ctx>=32k"]

    overall = {
        "observation": _Percentiles(),
        "prefill": _Percentiles(),
        "assistant": _Percentiles(),
        "ctx_before": _Percentiles(),
        "d_warm": _Percentiles(),
        "d_cold": _Percentiles(),
    }
    by_bucket = {
        name: {
            "observation": _Percentiles(),
            "prefill": _Percentiles(),
            "ctx_before": _Percentiles(),
            "d_warm": _Percentiles(),
        }
        for name in bucket_names
    }

    trigram_p = _TRIGRAM_P[corpus]
    turns = 0
    traces = 0
    turns_clearing = 0
    turns_clearing_cold = 0
    prefill_over_4000 = 0
    partial_block_tokens = 0
    observation_tokens = 0

    for trace_id, messages in _ITERATORS[corpus](path, limit):
        traces += 1
        for turn in walk_trace(messages, encode, block_size):
            if turn["first_turn"]:
                # The opening turn is a genuine cold prefill of the task
                # statement. It is real cost but it happens once per
                # trace and is not what the steady-state loop pays, so
                # it is excluded from the per-turn distributions.
                continue
            turns += 1
            ctx_before = turn["ctx_before"]
            step = decode_step_seconds(ctx_before)
            decode = _ACTION_DECODE_STEPS * step
            d_warm = prefill_seconds(turn["prefill"]) + decode
            d_cold = prefill_seconds(ctx_before) + decode

            overall["observation"].add(turn["observation"])
            overall["prefill"].add(turn["prefill"])
            overall["assistant"].add(turn["assistant"])
            overall["ctx_before"].add(ctx_before)
            overall["d_warm"].add(d_warm)
            overall["d_cold"].add(d_cold)

            observation_tokens += turn["observation"]
            partial_block_tokens += max(
                turn["prefill"] - turn["observation"], 0
            )
            if turn["prefill"] >= 4000:
                prefill_over_4000 += 1

            for (lo, hi), name in zip(buckets, bucket_names):
                if lo <= ctx_before < hi:
                    by_bucket[name]["observation"].add(turn["observation"])
                    by_bucket[name]["prefill"].add(turn["prefill"])
                    by_bucket[name]["ctx_before"].add(ctx_before)
                    by_bucket[name]["d_warm"].add(d_warm)
                    break

            for demand, counter in (
                (d_warm, "warm"),
                (d_cold, "cold"),
            ):
                verdict = cost_model.evaluate(
                    cost_model.build_cost_inputs(
                        actor_gpu_seconds=demand,
                        draft_gpu_tax=0.0,
                        tool_seconds=args.tool_seconds,
                        baseline_utilization=args.rho,
                        match_probability=0.75,
                        rollback_seconds=args.rollback,
                    )
                )
                if trigram_p >= verdict.minimum_match_probability:
                    if counter == "warm":
                        turns_clearing += 1
                    else:
                        turns_clearing_cold += 1

    payload = {
        "schema_version": 1,
        "corpus": corpus,
        "input_sha256": _sha256_file(path),
        "tokenizer": os.path.basename(tokenizer_path),
        "block_size": block_size,
        "traces": traces,
        "turns_scored": turns,
        "action_decode_steps": _ACTION_DECODE_STEPS,
        "cost_inputs": {
            "tool_seconds": args.tool_seconds,
            "rho": args.rho,
            "rollback_seconds": args.rollback,
            "drafter_tax": 0.0,
            "note": "drafter priced at zero, so this is the friendliest "
                    "possible floor for any drafter whatsoever",
        },
        "fits": {
            "prefill_seconds": {
                "form": "a*L + b*L^2",
                "a": pa,
                "b": pb,
                "residuals_seconds": {str(k): v for k, v in p_res.items()},
            },
            "decode_step_seconds": {
                "form": "base + slope*L",
                "base": dbase,
                "slope": dslope,
                "residuals_seconds": {str(k): v for k, v in d_res.items()},
            },
            "source": "experiments/agentspec_engine_demand/"
                      "engine-demand-measure-a100-20260910-2258/"
                      "engine_demand.json",
        },
        "tokens_per_turn": {
            key: overall[key].summary()
            for key in ("observation", "prefill", "assistant", "ctx_before")
        },
        "demand_seconds": {
            "warm": overall["d_warm"].summary(),
            "cold": overall["d_cold"].summary(),
        },
        "by_context_bucket": {
            name: {
                key: by_bucket[name][key].summary()
                for key in ("observation", "prefill", "ctx_before", "d_warm")
            }
            for name in bucket_names
        },
        "block_granularity_waste": {
            "observation_tokens_total": observation_tokens,
            "partial_block_tokens_total": partial_block_tokens,
            "waste_fraction_of_prefill": (
                partial_block_tokens
                / (observation_tokens + partial_block_tokens)
                if observation_tokens + partial_block_tokens
                else 0.0
            ),
        },
        "speculation_gate": {
            "trigram_p": trigram_p,
            "turns_clearing_floor_warm": turns_clearing,
            "fraction_clearing_floor_warm": (
                turns_clearing / turns if turns else 0.0
            ),
            "turns_clearing_floor_cold": turns_clearing_cold,
            "fraction_clearing_floor_cold": (
                turns_clearing_cold / turns if turns else 0.0
            ),
            "turns_prefill_over_4000": prefill_over_4000,
            "fraction_prefill_over_4000": (
                prefill_over_4000 / turns if turns else 0.0
            ),
        },
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(summary):
    if not summary:
        return "n/a"
    return (
        "mean %8.1f  p50 %7.0f  p75 %7.0f  p90 %7.0f  p99 %8.0f  max %8.0f"
        % (
            summary["mean"],
            summary["p50"],
            summary["p75"],
            summary["p90"],
            summary["p99"],
            summary["max"],
        )
    )


def report(payload):
    lines = []
    lines.append(
        "corpus %s  traces %d  turns %d  block_size %d"
        % (
            payload["corpus"],
            payload["traces"],
            payload["turns_scored"],
            payload["block_size"],
        )
    )
    lines.append("")
    lines.append("tokens per steady-state turn")
    for key in ("observation", "prefill", "assistant", "ctx_before"):
        lines.append(
            "  %-12s %s" % (key, _fmt(payload["tokens_per_turn"][key]))
        )
    waste = payload["block_granularity_waste"]
    lines.append(
        "  partial-block waste is %.1f%% of everything re-prefilled"
        % (100 * waste["waste_fraction_of_prefill"])
    )
    lines.append("")
    lines.append("actor demand D (seconds), 31 action tokens")
    for name in ("warm", "cold"):
        summary = payload["demand_seconds"][name]
        lines.append(
            "  %-5s mean %.4f  p50 %.4f  p90 %.4f  p99 %.4f"
            % (
                name,
                summary["mean"],
                summary["p50"],
                summary["p90"],
                summary["p99"],
            )
        )
    lines.append("")
    lines.append("prefill tokens per turn, by context length")
    for name, block in payload["by_context_bucket"].items():
        summary = block["prefill"]
        if not summary:
            lines.append("  %-9s (no turns)" % name)
            continue
        lines.append(
            "  %-9s n=%6d  %s" % (name, summary["count"], _fmt(summary))
        )
    lines.append("")
    gate = payload["speculation_gate"]
    lines.append(
        "speculation floor, drafter priced at zero, trigram p=%.4f"
        % gate["trigram_p"]
    )
    lines.append(
        "  turns clearing floor, warm cache  %.4f  (%d)"
        % (gate["fraction_clearing_floor_warm"], gate["turns_clearing_floor_warm"])
    )
    lines.append(
        "  turns clearing floor, cold cache  %.4f  (%d)"
        % (gate["fraction_clearing_floor_cold"], gate["turns_clearing_floor_cold"])
    )
    lines.append(
        "  turns re-prefilling >= 4000 tok   %.4f  (%d)"
        % (gate["fraction_prefill_over_4000"], gate["turns_prefill_over_4000"])
    )
    lines.append("")
    lines.append("payload sha256 %s" % payload["payload_sha256"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=sorted(_ITERATORS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--tool-seconds", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--rollback", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    payload = build_payload(
        args.corpus,
        args.input,
        args.tokenizer,
        args.limit,
        args.block_size,
        args,
    )
    print(report(payload))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print("artifact %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
