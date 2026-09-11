#!/usr/bin/env python3
"""Stage 1b step 1: build the prompted-drafter evaluation set.

Step 0b pre-registered the only remaining unknown on this line: how
often a *prompted, untrained* Qwen3-0.6B writes exactly the action the
actor was about to take, given 512 tokens of compressed context and a
hard cap of 11 emitted tokens.

This script builds the evaluation set for that sweep on CPU, so that
the GPU job is pure inference with no parsing, no corpus handling and
no licence-bearing text left on the box beyond what it needs.

Two decisions here are the ones that could quietly rig the result.

The first is what "compressed context" means. Stage 1a-bis priced a
512-token drafter prefill and said nothing about what fills it. The
honest cheapest answer, and the one used here, is right truncation:
the drafter sees the *tail* of the rendered trace and nothing else. A
learned compressor would presumably do better; a learned compressor is
also exactly the training project this stage exists to avoid. Anything
better than truncation must be earned later, not assumed now.

The second is eligibility. It is imported from the step 0 normaliser
rather than re-implemented, so the fail-closed side-effect rules are
byte-identical to the ones the baseline was measured under.

Two prompt variants are emitted for the same steps, so that the sweep
reports an ablation rather than a single number:

``tail``
    Tail of the conversation only.

``tail_tools``
    The same tail, prefixed by the distinct tool names already used
    earlier in this trace. That inventory is free at run time because
    it is derived from context the drafter already has, and it costs a
    few tokens of the budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_INVENTORY_LIMIT = 24


def _load_normaliser():
    path = os.path.join(_HERE, "agentspec_trace_normalize.py")
    spec = importlib.util.spec_from_file_location(
        "agentspec_trace_normalize", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["agentspec_trace_normalize"] = module
    spec.loader.exec_module(module)
    return module


_NORM = _load_normaliser()


def _render(role, text):
    return "%s: %s" % (role, text)


def apigen_steps(path, limit):
    """Yield (trace_id, prefix_lines, tools_seen, gold_text, eligible)."""

    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if limit:
        records = records[:limit]
    for trace_index, record in enumerate(records):
        lines = []
        tools_seen = []
        for turn in record.get("conversations", []):
            origin = turn.get("from")
            value = turn.get("value")
            if origin == "function_call":
                try:
                    call = json.loads(value)
                    tool_name = str(call["name"])
                    arguments = call.get("arguments", {})
                except Exception:
                    continue
                gold = json.dumps(
                    {"name": tool_name, "arguments": arguments},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                side_effect = _NORM._classify_apigen(tool_name)
                yield (
                    "apigen-%d" % trace_index,
                    list(lines),
                    list(tools_seen),
                    gold,
                    tool_name,
                    side_effect,
                )
                lines.append(_render("CALL", gold))
                if tool_name not in tools_seen:
                    tools_seen.append(tool_name)
            elif origin == "observation":
                lines.append(_render("OBSERVATION", value))
            elif origin == "human":
                lines.append(_render("USER", value))
            elif origin == "gpt":
                lines.append(_render("ASSISTANT", value))


def swe_agent_steps(path, limit):
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    emitted = 0
    for group in range(reader.metadata.num_row_groups):
        for row in reader.read_row_group(group).to_pylist():
            if limit and emitted >= limit:
                return
            emitted += 1
            lines = []
            tools_seen = []
            trace_id = str(row.get("instance_id") or "swe-%d" % emitted)
            for message in row.get("trajectory", []):
                role = message.get("role")
                text = message.get("text") or ""
                if role == "user":
                    lines.append(_render("OBSERVATION", text))
                    continue
                if role != "ai":
                    continue
                blocks = _NORM._FENCE.findall(text)
                if not blocks:
                    lines.append(_render("ASSISTANT", text))
                    continue
                command = " ".join(blocks[-1].split())
                head = command.split(" ", 1)[0]
                side_effect = _NORM._classify_swe(head)
                yield (
                    trace_id,
                    list(lines),
                    list(tools_seen),
                    command,
                    head,
                    side_effect,
                )
                lines.append(_render("ACTION", command))
                if head not in tools_seen:
                    tools_seen.append(head)


def _tail_tokens(tokenizer, text, budget):
    """Right-truncate to `budget` tokens, keeping the most recent text."""

    ids = tokenizer.encode(text).ids
    if len(ids) <= budget:
        return text, len(ids)
    return tokenizer.decode(ids[-budget:]), budget


def build(args):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    scanner = (
        apigen_steps(args.input, args.trace_limit)
        if args.corpus == "apigen"
        else swe_agent_steps(args.input, args.trace_limit)
    )
    rng = random.Random(args.seed)
    kept = []
    seen_steps = 0
    skipped_ineligible = 0
    skipped_empty_prefix = 0

    for (
        trace_id,
        lines,
        tools_seen,
        gold_text,
        tool_name,
        side_effect,
    ) in scanner:
        seen_steps += 1
        if side_effect not in ("read_only", "sandboxable"):
            skipped_ineligible += 1
            continue
        if not lines:
            skipped_empty_prefix += 1
            continue
        if rng.random() > args.sample_rate:
            continue
        body = "\n".join(lines)
        tail, tail_tokens = _tail_tokens(
            tokenizer, body, args.context_tokens
        )
        inventory = ", ".join(tools_seen[:TOOL_INVENTORY_LIMIT])
        gold_tokens = len(tokenizer.encode(gold_text).ids)
        kept.append(
            {
                "trace_id": trace_id,
                "corpus": args.corpus,
                "tool_name": tool_name,
                "side_effect_class": side_effect,
                "gold_text": gold_text,
                "gold_tokens": gold_tokens,
                "gold_within_cap": gold_tokens <= args.token_cap,
                "context_tail": tail,
                "context_tail_tokens": tail_tokens,
                "tools_seen": inventory,
                "prefix_actions": len(tools_seen),
            }
        )
        if args.max_steps and len(kept) >= args.max_steps:
            break

    payload_header = {
        "worker": "agentspec_prompted_drafter_evalset",
        "corpus": args.corpus,
        "context_tokens": args.context_tokens,
        "token_cap": args.token_cap,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "steps_scanned": seen_steps,
        "steps_skipped_ineligible": skipped_ineligible,
        "steps_skipped_empty_prefix": skipped_empty_prefix,
        "steps_kept": len(kept),
        "gold_within_cap_fraction": (
            sum(1 for row in kept if row["gold_within_cap"]) / len(kept)
            if kept
            else 0.0
        ),
    }
    digest = hashlib.sha256()
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload_header, sort_keys=True) + "\n")
        for row in kept:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False)
            digest.update(line.encode("utf-8"))
            handle.write(line + "\n")
    payload_header["rows_sha256"] = digest.hexdigest()
    return payload_header


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the prompted-drafter evaluation set",
    )
    parser.add_argument(
        "--corpus", required=True, choices=("apigen", "swe_agent")
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-tokens", type=int, default=448)
    parser.add_argument("--token-cap", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--trace-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args(argv)

    header = build(args)
    for key in sorted(header):
        print("%-32s %s" % (key, header[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
