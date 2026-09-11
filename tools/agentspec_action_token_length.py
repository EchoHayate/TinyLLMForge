#!/usr/bin/env python3
"""How many tokens does a real agent action actually take to write?

The companion script inverts the Stage 1a-bis cost measurement into a
token budget: the number of tokens a drafter may emit before speculation
stops paying. That budget is only half of a decision. The other half is
the length of the thing that has to be emitted.

Put together the two answer a question that matters more than any head
architecture. If real actions fit inside the budget when written as
ordinary text, then a *prompted* drafter is admissible and no head needs
to be trained to test the line. If they do not fit, a compact output
space is forced, and only then is the training cost justified.

Three emission forms are measured, because they cost different amounts
and buy different things:

``verbatim``
    The action exactly as it appears in the trace. This is what a plain
    prompted drafter emits, with no constrained decoding and no custom
    format. It is the cheapest thing to build and the most expensive
    thing to run.

``minimal``
    Tool name plus argument values, stripped of JSON punctuation and
    keys. This is the floor for a grammar-constrained decoder that knows
    the tool schema and only has to produce the values.

``tool_only``
    The tool name alone. A drafter that emits nothing else can only
    speculate on argument-free actions, but it establishes what the
    cheapest possible arm can reach.

Nothing here is trained and nothing is inferred. This is tokenisation
arithmetic over the same corpora the baseline used.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)

APIGEN_ELIGIBLE_PREFIXES = (
    "get_",
    "find_",
    "search_",
    "list_",
    "check_",
    "lookup_",
)
APIGEN_ELIGIBLE_EXACT = ("think", "calculate")

SWE_INELIGIBLE_HEADS = (
    "submit",
    "curl",
    "wget",
    "ssh",
    "apt",
    "apt-get",
    "conda",
)

DEFAULT_BUDGETS = (8, 11, 16, 27, 42, 67, 102)
PERCENTILES = (50, 75, 90, 99)


def _scalar_values(arguments):
    values = []
    stack = [arguments]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif item is None:
            continue
        else:
            values.append(str(item))
    return values


def _apigen_eligible(tool_name):
    if tool_name in APIGEN_ELIGIBLE_EXACT:
        return True
    return any(
        tool_name.startswith(prefix)
        for prefix in APIGEN_ELIGIBLE_PREFIXES
    )


def scan_apigen(path, limit):
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if limit:
        records = records[:limit]
    for record in records:
        for turn in record.get("conversations", []):
            if turn.get("from") != "function_call":
                continue
            value = turn.get("value")
            try:
                call = json.loads(value)
                tool_name = str(call["name"])
                arguments = call.get("arguments", {})
            except Exception:
                continue
            verbatim = json.dumps(
                {"name": tool_name, "arguments": arguments},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            minimal = " ".join(
                [tool_name] + _scalar_values(arguments)
            )
            yield (
                tool_name,
                verbatim,
                minimal,
                _apigen_eligible(tool_name),
            )


def scan_swe_agent(path, limit):
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    emitted = 0
    for group in range(reader.metadata.num_row_groups):
        for row in reader.read_row_group(group).to_pylist():
            if limit and emitted >= limit:
                return
            emitted += 1
            for message in row.get("trajectory", []):
                if message.get("role") != "ai":
                    continue
                blocks = _FENCE.findall(message.get("text") or "")
                if not blocks:
                    continue
                command = " ".join(blocks[-1].split())
                head, _, tail = command.partition(" ")
                yield (
                    head,
                    command,
                    command,
                    head not in SWE_INELIGIBLE_HEADS,
                )


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return 0
    index = min(
        len(sorted_values) - 1,
        int(round((percentile / 100.0) * (len(sorted_values) - 1))),
    )
    return sorted_values[index]


def _summarise(lengths, budgets):
    ordered = sorted(lengths)
    summary = {
        "count": len(ordered),
        "mean": (sum(ordered) / len(ordered)) if ordered else 0.0,
    }
    for percentile in PERCENTILES:
        summary["p%d" % percentile] = _percentile(ordered, percentile)
    summary["fraction_within_budget"] = {
        str(budget): (
            sum(1 for value in ordered if value <= budget) / len(ordered)
            if ordered
            else 0.0
        )
        for budget in budgets
    }
    return summary


def build_payload(corpus, path, tokenizer_path, limit, budgets):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    scanner = (
        scan_apigen(path, limit)
        if corpus == "apigen"
        else scan_swe_agent(path, limit)
    )
    forms = ("verbatim", "minimal", "tool_only")
    lengths = {form: [] for form in forms}
    eligible_lengths = {form: [] for form in forms}
    per_tool = collections.defaultdict(list)
    totals = collections.Counter()
    cache = {}

    def encode_length(text):
        if text not in cache:
            cache[text] = len(tokenizer.encode(text).ids)
        return cache[text]

    for tool_name, verbatim, minimal, eligible in scanner:
        totals["steps"] += 1
        measured = {
            "verbatim": encode_length(verbatim),
            "minimal": encode_length(minimal),
            "tool_only": encode_length(tool_name),
        }
        for form in forms:
            lengths[form].append(measured[form])
        if eligible:
            totals["eligible"] += 1
            for form in forms:
                eligible_lengths[form].append(measured[form])
            per_tool[tool_name].append(measured["verbatim"])

    payload = {
        "worker": "agentspec_action_token_length",
        "corpus": corpus,
        "tokenizer": tokenizer_path,
        "budgets": list(budgets),
        "totals": dict(totals),
        "all_steps": {
            form: _summarise(lengths[form], budgets) for form in forms
        },
        "eligible_steps": {
            form: _summarise(eligible_lengths[form], budgets)
            for form in forms
        },
        "per_tool_verbatim_median": {
            tool: {
                "steps": len(values),
                "median_tokens": _percentile(sorted(values), 50),
            }
            for tool, values in sorted(
                per_tool.items(), key=lambda item: -len(item[1])
            )[:20]
        },
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def render(payload):
    budgets = payload["budgets"]
    lines = [
        "corpus         %s" % payload["corpus"],
        "steps scanned  %d" % payload["totals"].get("steps", 0),
        "eligible steps %d" % payload["totals"].get("eligible", 0),
        "",
        "action length in Qwen3 tokens, eligible steps only",
        "",
        "%-11s %7s %6s %6s %6s %6s"
        % ("form", "mean", "p50", "p75", "p90", "p99"),
    ]
    for form, summary in payload["eligible_steps"].items():
        lines.append(
            "%-11s %7.1f %6d %6d %6d %6d"
            % (
                form,
                summary["mean"],
                summary["p50"],
                summary["p75"],
                summary["p90"],
                summary["p99"],
            )
        )
    lines.extend(
        [
            "",
            "fraction of eligible actions that fit in a token budget",
            "",
            "%-11s %s"
            % ("form", " ".join("%7s" % ("<=%d" % b) for b in budgets)),
        ]
    )
    for form, summary in payload["eligible_steps"].items():
        cells = " ".join(
            "%7.4f" % summary["fraction_within_budget"][str(budget)]
            for budget in budgets
        )
        lines.append("%-11s %s" % (form, cells))
    lines.extend(
        [
            "",
            "%-30s %8s %8s" % ("tool", "steps", "p50 tok"),
        ]
    )
    for tool, row in payload["per_tool_verbatim_median"].items():
        lines.append(
            "%-30s %8d %8d"
            % (tool, row["steps"], row["median_tokens"])
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure action length against the drafter budget",
    )
    parser.add_argument(
        "--corpus", required=True, choices=("apigen", "swe_agent")
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
    )
    args = parser.parse_args(argv)

    budgets = tuple(
        int(value) for value in args.budgets.split(",") if value
    )
    payload = build_payload(
        args.corpus, args.input, args.tokenizer, args.limit, budgets
    )
    print(render(payload))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("")
        print("artifact %s" % args.output)
        print("payload sha256 %s" % payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
