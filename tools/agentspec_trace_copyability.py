#!/usr/bin/env python3
"""Stage 1b step 0b: can the next action's arguments be copied?

The baseline script measures a fixed-codebook drafter and finds it
capped well below the required match probability on real coding-agent
traces, because the action strings are repository-specific and mostly
unseen in training. That result only condemns the *fixed codebook*
representation. It says nothing about the obvious alternative: a head
that points at spans already present in the context it is reading.

This script measures the ceiling of that alternative directly, using
the raw corpora rather than the digest-only normalised file. For each
action it asks whether every scalar argument value already appears
verbatim in the text the drafter would have seen: the system prompt,
the user turns, the previous actions and their observations.

An action is ``copyable`` when a perfect pointer head could reconstruct
it by selecting the tool and copying each argument value out of the
context. That is an oracle, not a method: it assumes the head knows
which spans to copy. It is an upper bound, and the useful reading is
the gap between it and the fixed-codebook upper bound.

The residue matters as much as the number. Arguments that are free text
authored by the model, an ``edit`` body or a chain-of-thought string,
cannot be copied and cannot be enumerated, so they bound every
mechanism in this family.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)
_MIN_COPY_LENGTH = 2

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


def _scalar_values(arguments):
    """Flatten an argument object into the strings a head must emit."""

    values = []
    stack = [arguments]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, bool) or item is None:
            continue
        else:
            text = str(item)
            if len(text) >= _MIN_COPY_LENGTH:
                values.append(text)
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
        context_parts = [record.get("system") or ""]
        seen_actions = 0
        for turn in record.get("conversations", []):
            role = turn.get("from")
            value = turn.get("value")
            if role == "function_call":
                try:
                    call = json.loads(value)
                    tool_name = str(call["name"])
                    arguments = call.get("arguments", {})
                except Exception:
                    context_parts.append(str(value))
                    continue
                if seen_actions >= 1:
                    yield (
                        tool_name,
                        _scalar_values(arguments),
                        "\n".join(context_parts),
                        _apigen_eligible(tool_name),
                    )
                seen_actions += 1
            context_parts.append(str(value))


def scan_swe_agent(path, limit):
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    emitted = 0
    for group in range(reader.metadata.num_row_groups):
        for row in reader.read_row_group(group).to_pylist():
            if limit and emitted >= limit:
                return
            emitted += 1
            context_parts = []
            seen_actions = 0
            for message in row.get("trajectory", []):
                text = message.get("text") or ""
                if message.get("role") == "system":
                    context_parts.append(
                        message.get("system_prompt") or text
                    )
                    continue
                if message.get("role") != "ai":
                    context_parts.append(text)
                    continue
                blocks = _FENCE.findall(text)
                if blocks:
                    command = " ".join(blocks[-1].split())
                    head, _, tail = command.partition(" ")
                    if seen_actions >= 1:
                        yield (
                            head,
                            [tail] if len(tail) >= _MIN_COPY_LENGTH
                            else [],
                            "\n".join(context_parts),
                            head not in SWE_INELIGIBLE_HEADS,
                        )
                    seen_actions += 1
                context_parts.append(text)


def build_payload(corpus, path, limit):
    scanner = (
        scan_apigen(path, limit)
        if corpus == "apigen"
        else scan_swe_agent(path, limit)
    )
    totals = collections.Counter()
    per_tool = collections.defaultdict(collections.Counter)
    for tool_name, values, context, eligible in scanner:
        totals["steps"] += 1
        if not eligible:
            totals["ineligible"] += 1
            continue
        totals["eligible"] += 1
        if not values:
            totals["no_arguments"] += 1
            copyable = True
            missing = 0
        else:
            missing = sum(
                1 for value in values if value not in context
            )
            copyable = missing == 0
        if copyable:
            totals["copyable"] += 1
            per_tool[tool_name]["copyable"] += 1
        else:
            totals["not_copyable"] += 1
            totals["missing_values"] += missing
        per_tool[tool_name]["steps"] += 1
    eligible = totals["eligible"] or 1
    payload = {
        "worker": "agentspec_trace_copyability",
        "corpus": corpus,
        "totals": dict(totals),
        "copyable_fraction_of_eligible": totals["copyable"] / eligible,
        "no_argument_fraction_of_eligible": (
            totals["no_arguments"] / eligible
        ),
        "per_tool": {
            tool: {
                "steps": counter["steps"],
                "copyable": counter["copyable"],
                "rate": counter["copyable"] / counter["steps"],
            }
            for tool, counter in sorted(
                per_tool.items(),
                key=lambda item: -item[1]["steps"],
            )[:20]
        },
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def render(payload):
    lines = [
        "corpus                        %s" % payload["corpus"],
        "steps scanned                 %d"
        % payload["totals"].get("steps", 0),
        "eligible steps                %d"
        % payload["totals"].get("eligible", 0),
        "copyable from context         %.4f of eligible"
        % payload["copyable_fraction_of_eligible"],
        "of which argument-free        %.4f of eligible"
        % payload["no_argument_fraction_of_eligible"],
        "",
        "%-30s %8s %10s" % ("tool", "steps", "copyable"),
    ]
    for tool, row in payload["per_tool"].items():
        lines.append(
            "%-30s %8d %10.4f" % (tool, row["steps"], row["rate"])
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure argument copyability from context",
    )
    parser.add_argument(
        "--corpus", required=True, choices=("apigen", "swe_agent")
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    payload = build_payload(args.corpus, args.input, args.limit)
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
