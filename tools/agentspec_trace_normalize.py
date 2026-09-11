#!/usr/bin/env python3
"""Stage 1b step 0: normalise public agent traces into action sequences.

Stage 1a-bis established that the compressed-context code drafter is
cheap enough to be admissible, and pinned the only remaining question:
can a drafter predict the actor's *next action* often enough. The
required top-1 match probability is 0.478 to 0.491 depending on
context length.

Before training anything, the corpus itself has to be checked. This
script converts public agent traces into the action identity this
package already commits to in ``tinyvllm/agentspec/action.py``: a tool
name plus canonically serialised arguments, hashed into a digest. Two
actions match only when their digests are byte-identical, because that
is the condition under which a speculative observation may be reused.

Deliberately, no raw arguments are written out. The normalised record
keeps digests, argument key names, sizes and a short preview. That
keeps the artifact small, reviewable and free of corpus text, and it
makes the downstream baseline script dependency-free.

Corpora:

``apigen``
    ``Salesforce/APIGen-MT-5k`` (CC BY-NC 4.0). Multi-turn tau-bench
    style airline and retail agents. ``function_call`` turns carry an
    explicit ``{"name", "arguments"}`` object, so the action identity
    is exact rather than parsed.

``swe_agent``
    ``nebius/SWE-agent-trajectories`` (CC BY 4.0). Real SWE-agent runs
    against real repositories. The action is the fenced command block
    in each assistant turn, which is a shell-style command line rather
    than a typed call, so the first token is treated as the tool and
    the remainder as a single ``command_line`` argument.

Usage::

    python3 tools/agentspec_trace_normalize.py \
        --corpus apigen --input apigen-mt_5k.json --output apigen.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import types


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_action_module():
    """Import ``tinyvllm.agentspec.action`` without importing torch."""

    package_root = os.path.join(_REPO_ROOT, "tinyvllm")
    if "tinyvllm" not in sys.modules:
        parent = types.ModuleType("tinyvllm")
        parent.__path__ = [package_root]
        sys.modules["tinyvllm"] = parent
    if "tinyvllm.agentspec" not in sys.modules:
        child = types.ModuleType("tinyvllm.agentspec")
        child.__path__ = [os.path.join(package_root, "agentspec")]
        sys.modules["tinyvllm.agentspec"] = child
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return importlib.import_module("tinyvllm.agentspec.action")


ACTION = _load_action_module()

# Side-effect classification is declared per corpus and fails closed.
# An unrecognised tool is ``unknown``, which the Stage 0 router refuses
# to speculate on. Getting this wrong in the permissive direction would
# mean speculating on an action that mutates an external system, so the
# tables below are prefix rules that only ever *add* restrictions.
APIGEN_READ_ONLY_PREFIXES = (
    "get_",
    "find_",
    "search_",
    "list_",
    "check_",
    "lookup_",
)
APIGEN_READ_ONLY_EXACT = (
    "think",
    "calculate",
)
APIGEN_MUTATING_PREFIXES = (
    "book_",
    "cancel_",
    "modify_",
    "update_",
    "exchange_",
    "return_",
    "send_",
    "place_",
    "create_",
    "delete_",
    "transfer_",
)

# SWE-agent runs inside a per-instance container. ``sandboxable`` is
# used for commands that only touch that container's filesystem in a
# way the harness can roll back; ``irreversible`` is used for anything
# that ends the episode or leaves the sandbox.
SWE_READ_ONLY = (
    "open",
    "goto",
    "scroll_down",
    "scroll_up",
    "ls",
    "cat",
    "find_file",
    "search_dir",
    "search_file",
    "grep",
    "find",
    "head",
    "tail",
    "wc",
    "diff",
    "git",
    "pwd",
    "which",
)
SWE_SANDBOXABLE = (
    "edit",
    "create",
    "python",
    "python3",
    "pytest",
    "mypy",
    "touch",
    "mkdir",
    "cp",
    "mv",
    "rm",
    "cd",
    "echo",
    "sed",
    "chmod",
    "export",
    "pip",
)
SWE_IRREVERSIBLE = (
    "submit",
    "curl",
    "wget",
    "ssh",
    "apt",
    "apt-get",
    "conda",
)

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)
_PREVIEW_LIMIT = 120


def _classify_apigen(tool_name: str) -> str:
    if tool_name in APIGEN_READ_ONLY_EXACT:
        return "read_only"
    for prefix in APIGEN_READ_ONLY_PREFIXES:
        if tool_name.startswith(prefix):
            return "read_only"
    for prefix in APIGEN_MUTATING_PREFIXES:
        if tool_name.startswith(prefix):
            return "irreversible"
    return "unknown"


def _classify_swe(tool_name: str) -> str:
    if tool_name in SWE_READ_ONLY:
        return "read_only"
    if tool_name in SWE_SANDBOXABLE:
        return "sandboxable"
    if tool_name in SWE_IRREVERSIBLE:
        return "irreversible"
    return "unknown"


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _PREVIEW_LIMIT:
        return collapsed
    return collapsed[:_PREVIEW_LIMIT] + "..."


def _coerce_arguments(arguments):
    """Force any argument payload into a dict, and say so.

    A minority of traces carry a non-dict ``arguments`` value, usually
    a bare string produced by a model that ignored the schema. Dropping
    those steps would silently remove the hardest actions from the
    denominator, so they are kept under a reserved key and counted.
    """

    if isinstance(arguments, dict):
        return arguments, False
    return (
        {
            "__nondict_arguments__": json.dumps(
                arguments, sort_keys=True, ensure_ascii=False
            )
        },
        True,
    )


def _record(index, tool_name, arguments, side_effect_class):
    arguments, coerced = _coerce_arguments(arguments)
    canonical = ACTION.canonical_arguments_json(arguments)
    signature = ACTION.build_action_signature(
        tool_name=tool_name,
        arguments=arguments,
    )
    if isinstance(arguments, dict):
        arg_keys = sorted(str(key) for key in arguments)
    else:
        arg_keys = []
    return {
        "index": index,
        "tool": tool_name,
        "side_effect_class": side_effect_class,
        "action_digest": signature.digest,
        "args_digest": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "arg_keys": arg_keys,
        "args_bytes": len(canonical.encode("utf-8")),
        "args_coerced": coerced,
        "args_preview": _preview(canonical),
    }


def normalise_apigen(path, limit):
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if limit:
        records = records[:limit]
    for position, record in enumerate(records):
        actions = []
        unparsed = 0
        for turn in record.get("conversations", []):
            if turn.get("from") != "function_call":
                continue
            try:
                call = json.loads(turn["value"])
                tool_name = str(call["name"])
                arguments = call.get("arguments", {})
            except Exception:
                unparsed += 1
                continue
            actions.append(
                _record(
                    len(actions),
                    tool_name,
                    arguments,
                    _classify_apigen(tool_name),
                )
            )
        if not actions:
            continue
        yield {
            "corpus": "apigen",
            "trace_id": "apigen-%05d" % position,
            "unparsed_calls": unparsed,
            "actions": actions,
        }


def normalise_swe_agent(path, limit):
    import pyarrow.parquet as pq  # imported lazily; parquet only

    reader = pq.ParquetFile(path)
    emitted = 0
    for group in range(reader.metadata.num_row_groups):
        rows = reader.read_row_group(group).to_pylist()
        for row in rows:
            if limit and emitted >= limit:
                return
            actions = []
            unparsed = 0
            for message in row.get("trajectory", []):
                if message.get("role") != "ai":
                    continue
                blocks = _FENCE.findall(message.get("text") or "")
                if not blocks:
                    unparsed += 1
                    continue
                # SWE-agent emits at most one command per turn; when a
                # turn contains several fences the executed one is the
                # last, which is what the harness itself parses.
                command = " ".join(blocks[-1].split())
                if not command:
                    unparsed += 1
                    continue
                head, _, tail = command.partition(" ")
                actions.append(
                    _record(
                        len(actions),
                        head,
                        {"command_line": tail},
                        _classify_swe(head),
                    )
                )
            if not actions:
                continue
            emitted += 1
            yield {
                "corpus": "swe_agent",
                "trace_id": "swe-%s" % row.get("instance_id"),
                "model_name": row.get("model_name"),
                "resolved": bool(row.get("target")),
                "unparsed_calls": unparsed,
                "actions": actions,
            }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Normalise agent traces into action sequences",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=("apigen", "swe_agent"),
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.corpus == "apigen":
        traces = normalise_apigen(args.input, args.limit)
    else:
        traces = normalise_swe_agent(args.input, args.limit)

    trace_count = 0
    action_count = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
            trace_count += 1
            action_count += len(trace["actions"])
    print(
        "corpus %s  traces %d  actions %d  -> %s"
        % (args.corpus, trace_count, action_count, args.output)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
