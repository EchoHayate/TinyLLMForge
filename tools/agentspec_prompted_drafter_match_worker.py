#!/usr/bin/env python3
"""Stage 1b step 1: does an untrained prompted drafter guess the action?

Step 0b retired the training project and pre-registered a single
measurement in its place. A Qwen3-0.6B that has never been fine-tuned
for this task reads 512 tokens of compressed context, is allowed to
emit at most `k` tokens, and must produce the *exact* action the actor
was about to take. Exact means the canonical `tool + arguments` digest
from ``tinyvllm/agentspec/action.py`` is byte-identical, because that
is the only condition under which the Stage 0 router permits reusing a
speculative observation.

Three quantities come out, and conflating them would flatter the
result:

``coverage``
    Fraction of eligible steps where the drafter finished inside the
    token cap and produced something parseable. Overruns are not
    misses; the drafter abstains, the actor path is untouched, and the
    only cost is the draft itself.

``p_speculated``
    Match rate among the steps it actually spoke on. This is the
    quantity the cost model calls `p`, and on its own it is easy to
    make large by abstaining more.

``p_effective``
    Matches divided by *all* eligible steps. Treating abstentions as
    misses is conservative, because an abstention pays the draft tax
    but never pays rollback. The truth sits between the two, and both
    are reported rather than a chosen one.

Nothing is trained here and nothing is timed here. Cost was settled on
the serving path in Stage 1a-bis; this job only asks whether the
prediction is right.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import sys
import types


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_FENCE_LINE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*$")
EXAMPLE_LIMIT = 12


def _load_action_module():
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


SYSTEM_PROMPT = {
    ("swe_agent", "v1"): (
        "You are the action drafter for a software engineering agent. "
        "You are shown the tail of the agent's terminal session. "
        "Predict the single next shell command the agent will run. "
        "Answer with the command only, on one line. "
        "No explanation, no markdown, no code fences."
    ),
    ("apigen", "v1"): (
        "You are the action drafter for a tool-using assistant. "
        "You are shown the tail of the conversation. "
        "Predict the single next tool call. "
        'Answer with compact JSON only, of the form '
        '{"name": "tool_name", "arguments": {...}}. '
        "No explanation, no markdown, no code fences."
    ),
    # v1 fails in a specific, informative way: Qwen3-0.6B often answers
    # the literal string "None". It reads the task as a question it may
    # decline. v2 removes the option to decline.
    ("swe_agent", "v2"): (
        "Continue a software engineering agent's terminal session. "
        "You are given the tail of the session. Output the next shell "
        "command, exactly as the agent would type it at the prompt.\n"
        "Rules: output one line, the command only. Never output "
        "'None', an apology, a question, an explanation, markdown or "
        "code fences. If you are unsure, still output your single best "
        "guess at a command."
    ),
    ("apigen", "v2"): (
        "Continue a tool-using assistant's session. You are given the "
        "tail of the conversation. Output the next tool call as "
        'compact JSON: {"name": "tool_name", "arguments": {...}}.\n'
        "Rules: output the JSON object only. Never output 'None', an "
        "apology, a question, an explanation, markdown or code "
        "fences. If you are unsure, still output your single best "
        "guess at a tool call."
    ),
}

# v3 is v2 plus two hand-written examples. The examples are invented
# rather than sampled from either corpus, so nothing from the
# evaluation distribution leaks into the prompt.
FEW_SHOT = {
    "swe_agent": (
        "Examples of the expected output format:\n"
        "  open src/parser.py 120\n"
        "  python reproduce.py\n"
        "  search_dir \"def resolve\" src\n"
    ),
    "apigen": (
        "Examples of the expected output format:\n"
        '  {"name": "get_user_details", "arguments": {"user_id": '
        '"amy_lee_1234"}}\n'
        '  {"name": "search_flights", "arguments": {"origin": "SFO", '
        '"destination": "JFK"}}\n'
    ),
}

USER_SUFFIX = {
    "swe_agent": "\n\nNext command:",
    "apigen": "\n\nNext tool call:",
}
PROMPT_STYLES = ("v1", "v2", "v3")


def signature_of(corpus, text):
    """Parse an action string into the repository's action identity.

    Applied to the model's output and to the gold string by the same
    code path, so neither side gets a parsing advantage.
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if corpus == "swe_agent":
        lines = [
            line
            for line in cleaned.splitlines()
            if line.strip() and not _FENCE_LINE.match(line)
        ]
        if not lines:
            return None
        command = " ".join(lines[0].split())
        if not command:
            return None
        head, _, tail = command.partition(" ")
        try:
            return ACTION.build_action_signature(
                tool_name=head, arguments={"command_line": tail}
            )
        except ValueError:
            return None
    match = _JSON_OBJECT.search(cleaned)
    if not match:
        return None
    try:
        call = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    try:
        return ACTION.build_action_signature(
            tool_name=str(call["name"]), arguments=arguments
        )
    except ValueError:
        return None


def load_evalset(path, limit, offset=0):
    header = None
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            payload = json.loads(line)
            if index == 0:
                header = payload
                continue
            if index <= offset:
                continue
            rows.append(payload)
            if limit and len(rows) >= limit:
                break
    return header, rows


def build_prompt(tokenizer, corpus, row, variant, style, budget):
    """Assemble the drafter prompt, trimmed to the priced token budget.

    Stage 1a-bis priced a 512-token drafter prefill. Prompt scaffolding
    is not free, and few-shot examples are not free either, so the
    context tail is trimmed to whatever the instruction leaves behind.
    Otherwise a better prompt would silently buy itself a bigger
    prefill than the cost model was ever shown.
    """

    system = SYSTEM_PROMPT[(corpus, "v1" if style == "v1" else "v2")]
    if style == "v3":
        system = system + "\n" + FEW_SHOT[corpus]
    header = ""
    if variant == "tail_tools" and row.get("tools_seen"):
        header = "Tools used so far: %s\n\n" % row["tools_seen"]

    def render(context):
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": header + context + USER_SUFFIX[corpus],
            },
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    context = row["context_tail"]
    prompt = render(context)
    length = len(tokenizer(prompt)["input_ids"])
    if length <= budget:
        return prompt, length
    overflow = length - budget
    context_ids = tokenizer(context, add_special_tokens=False)[
        "input_ids"
    ]
    # Re-rendering can re-tokenise across the cut, so trim with a small
    # margin and verify rather than assume.
    keep = max(len(context_ids) - overflow - 4, 0)
    context = tokenizer.decode(
        context_ids[len(context_ids) - keep :], skip_special_tokens=True
    )
    prompt = render(context)
    return prompt, len(tokenizer(prompt)["input_ids"])


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    header, rows = load_evalset(args.evalset, args.limit, args.offset)
    corpus = header["corpus"]
    if (corpus, "v1") not in SYSTEM_PROMPT:
        raise SystemExit("unsupported corpus %s" % corpus)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"trust_remote_code": True}
    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to("cuda")
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        raise SystemExit("model did not load in bfloat16")

    totals = {
        "eligible": 0,
        "terminated_within_cap": 0,
        "parsed": 0,
        "digest_match": 0,
        "tool_match": 0,
        "gold_within_cap": 0,
        "digest_match_gold_within_cap": 0,
        "declined": 0,
    }
    examples = []
    prompt_lengths = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        built = [
            build_prompt(
                tokenizer,
                corpus,
                row,
                args.variant,
                args.prompt_style,
                args.prompt_budget,
            )
            for row in batch
        ]
        prompts = [item[0] for item in built]
        prompt_lengths.extend(item[1] for item in built)
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.token_cap,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = generated[:, prompt_length:]

        for row, token_ids in zip(batch, new_tokens.tolist()):
            totals["eligible"] += 1
            if row["gold_within_cap"]:
                totals["gold_within_cap"] += 1
            stop_positions = [
                index
                for index, value in enumerate(token_ids)
                if value
                in (tokenizer.eos_token_id, tokenizer.pad_token_id)
            ]
            terminated = bool(stop_positions)
            body = (
                token_ids[: stop_positions[0]]
                if terminated
                else token_ids
            )
            prediction = tokenizer.decode(
                body, skip_special_tokens=True
            )
            # A drafter that has not finished inside the cap abstains.
            # Newline termination counts, because the prompt asks for a
            # single line and the harness would cut there anyway.
            if not terminated and "\n" in prediction:
                prediction = prediction.split("\n", 1)[0]
                terminated = True
            if not terminated:
                continue
            totals["terminated_within_cap"] += 1
            if prediction.strip().lower() in ("none", "null", "n/a", ""):
                # A refusal is not a prediction. Counting it as a
                # parse failure would hide it inside coverage, so it
                # gets its own counter.
                totals["declined"] += 1
                continue
            predicted = signature_of(corpus, prediction)
            gold = signature_of(corpus, row["gold_text"])
            if predicted is None or gold is None:
                continue
            totals["parsed"] += 1
            if predicted.tool_name == gold.tool_name:
                totals["tool_match"] += 1
            if predicted.digest == gold.digest:
                totals["digest_match"] += 1
                if row["gold_within_cap"]:
                    totals["digest_match_gold_within_cap"] += 1
            elif len(examples) < EXAMPLE_LIMIT:
                examples.append(
                    {
                        "tool": row["tool_name"],
                        "gold": row["gold_text"][:120],
                        "prediction": prediction[:120],
                    }
                )

    eligible = max(totals["eligible"], 1)
    speculated = max(totals["parsed"], 1)
    payload = {
        "worker": "agentspec_prompted_drafter_match",
        "corpus": corpus,
        "variant": args.variant,
        "prompt_style": args.prompt_style,
        "prompt_budget": args.prompt_budget,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_tokens_mean": (
            sum(prompt_lengths) / len(prompt_lengths)
            if prompt_lengths
            else 0.0
        ),
        "offset": args.offset,
        "model": args.model,
        "token_cap": args.token_cap,
        "context_tokens": header["context_tokens"],
        "evalset_rows_sha256": header.get("rows_sha256"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "totals": totals,
        "coverage": totals["parsed"] / eligible,
        "p_speculated": totals["digest_match"] / speculated,
        "p_effective": totals["digest_match"] / eligible,
        "tool_accuracy_speculated": totals["tool_match"] / speculated,
        "gold_within_cap_fraction": (
            totals["gold_within_cap"] / eligible
        ),
        "p_among_gold_within_cap": (
            totals["digest_match_gold_within_cap"]
            / max(totals["gold_within_cap"], 1)
        ),
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload, examples


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure prompted drafter action match",
    )
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant", default="tail", choices=("tail", "tail_tools")
    )
    parser.add_argument(
        "--prompt-style", default="v2", choices=PROMPT_STYLES
    )
    parser.add_argument("--prompt-budget", type=int, default=512)
    parser.add_argument("--token-cap", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    payload, examples = run(args)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print("corpus            %s" % payload["corpus"])
    print("variant           %s" % payload["variant"])
    print("prompt style      %s" % payload["prompt_style"])
    print(
        "prompt tokens     mean %.1f max %d (budget %d)"
        % (
            payload["prompt_tokens_mean"],
            payload["prompt_tokens_max"],
            payload["prompt_budget"],
        )
    )
    print("token cap         %d" % payload["token_cap"])
    print("eligible steps    %d" % payload["totals"]["eligible"])
    print("declined          %d" % payload["totals"]["declined"])
    print("coverage          %.4f" % payload["coverage"])
    print("p_speculated      %.4f" % payload["p_speculated"])
    print("p_effective       %.4f" % payload["p_effective"])
    print(
        "tool accuracy     %.4f" % payload["tool_accuracy_speculated"]
    )
    print(
        "gold within cap   %.4f" % payload["gold_within_cap_fraction"]
    )
    print(
        "p | gold in cap   %.4f" % payload["p_among_gold_within_cap"]
    )
    print("payload sha256    %s" % payload["payload_sha256"])
    if examples:
        print("")
        print("sample misses")
        for example in examples:
            print("  tool %s" % example["tool"])
            print("    gold %s" % example["gold"])
            print("    pred %s" % example["prediction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
