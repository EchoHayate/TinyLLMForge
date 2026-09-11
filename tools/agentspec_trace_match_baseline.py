#!/usr/bin/env python3
"""Stage 1b step 0: training-free next-action match baselines.

Stage 1a-bis pinned the surviving question: the compressed-context code
drafter is profitable only if its top-1 action match probability clears
0.478 at 16384, 0.485 at 4096 and 0.491 at 1024 tokens of context.

This script answers a cheaper question first, and it is deliberately
the question that can kill or trivialise the line before any training
budget is spent:

1. How predictable is the next action *without any model at all*? A
   frequency table, a bigram and "repeat the last action" cost nothing.
   If they already clear the threshold, a learned drafter is not the
   contribution and the honest report is that agent loops are
   repetitive. If they are far below it, the learned drafter has real
   work to do and the threshold is the thing to design against.
2. What is the ceiling for a *fixed codebook* drafter? Stage 1a-bis
   priced a 4096-entry code head. If a fifth of evaluation actions are
   strings never seen in training, no head over that codebook can ever
   emit them, and the achievable match rate is capped below the
   requirement no matter how good the model is.
3. How many actions are even eligible to speculate on? The Stage 0
   router refuses to speculate on irreversible or unclassified tools.
   Eligibility multiplies the match rate, so a high match rate on a
   small eligible slice is worth less than it looks.

Match is exact by construction: two actions match when the canonical
``tool + arguments`` digest is byte-identical, which is the condition
under which a speculative observation may be reused. A tool-name-only
match rate is also reported, not as a result, but to quantify how much
of the difficulty lives in the arguments.

The script is dependency-free and reads only the normalised JSONL
written by ``tools/agentspec_trace_normalize.py``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json


ELIGIBLE_CLASSES = ("read_only", "sandboxable", "reversible")
CODEBOOK_SIZE = 4096
EVAL_FRACTION_DENOMINATOR = 5  # one trace in five is held out

# Pinned by Stage 1a-bis. Stage 1b does not get to re-derive these.
REQUIRED_MATCH_PROBABILITY = {
    1024: 0.491,
    4096: 0.485,
    16384: 0.478,
}


def load_traces(path):
    traces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def split_traces(traces):
    """Deterministic held-out split by trace id, never by step.

    Splitting by step would leak: the same trace would contribute both
    context and target across the split, and a bigram fitted on the
    same trajectory it is evaluated on is not a baseline, it is a
    lookup of the answer.
    """

    train, evaluation = [], []
    for trace in traces:
        digest = hashlib.sha1(
            trace["trace_id"].encode("utf-8")
        ).hexdigest()
        bucket = int(digest[:8], 16) % EVAL_FRACTION_DENOMINATOR
        (evaluation if bucket == 0 else train).append(trace)
    return train, evaluation


def _key(action, granularity):
    if granularity == "exact":
        return action["action_digest"]
    if granularity == "tool":
        return action["tool"]
    raise ValueError("unknown granularity %r" % granularity)


def fit_tables(train, granularity):
    unigram = collections.Counter()
    bigram = collections.defaultdict(collections.Counter)
    trigram = collections.defaultdict(collections.Counter)
    for trace in train:
        keys = [
            _key(action, granularity) for action in trace["actions"]
        ]
        for index, key in enumerate(keys):
            unigram[key] += 1
            if index >= 1:
                bigram[keys[index - 1]][key] += 1
            if index >= 2:
                trigram[(keys[index - 2], keys[index - 1])][key] += 1
    global_top = unigram.most_common(1)[0][0] if unigram else None
    bigram_top = {
        prior: counter.most_common(1)[0][0]
        for prior, counter in bigram.items()
    }
    trigram_top = {
        prior: counter.most_common(1)[0][0]
        for prior, counter in trigram.items()
    }
    codebook = {
        key for key, _count in unigram.most_common(CODEBOOK_SIZE)
    }
    return {
        "unigram": unigram,
        "global_top": global_top,
        "bigram_top": bigram_top,
        "trigram_top": trigram_top,
        "vocabulary": set(unigram),
        "codebook": codebook,
    }


def evaluate(evaluation, tables, granularity, class_filter):
    """Score every predictor over held-out steps with a prefix.

    Only steps at index >= 1 are scored. The first action of a trace
    has no preceding action, so "repeat the last action" is undefined
    there. Excluding it is the choice that favours the hypothesis, so
    the excluded count is reported alongside the rates.
    """

    if class_filter == "all_actions":
        allowed = None
    elif class_filter == "eligible_only":
        allowed = set(ELIGIBLE_CLASSES)
    elif class_filter == "read_only_only":
        allowed = {"read_only"}
    else:
        raise ValueError("unknown class filter %r" % class_filter)

    hits = collections.Counter()
    total = 0
    skipped_first = 0
    skipped_ineligible = 0
    for trace in evaluation:
        actions = trace["actions"]
        keys = [_key(action, granularity) for action in actions]
        seen = collections.Counter()
        for index, action in enumerate(actions):
            key = keys[index]
            if index == 0:
                seen[key] += 1
                skipped_first += 1
                continue
            prefix_mode = seen.most_common(1)[0][0]
            in_prefix = key in seen
            seen[key] += 1
            if allowed is not None and (
                action["side_effect_class"] not in allowed
            ):
                skipped_ineligible += 1
                continue
            total += 1
            previous = keys[index - 1]
            if previous == key:
                hits["repeat_last"] += 1
            if tables["global_top"] == key:
                hits["global_top1"] += 1
            if tables["bigram_top"].get(previous, tables["global_top"]) == key:
                hits["bigram_prev"] += 1
            trigram_key = (
                keys[index - 2] if index >= 2 else None,
                previous,
            )
            trigram_prediction = tables["trigram_top"].get(
                trigram_key,
                tables["bigram_top"].get(
                    previous, tables["global_top"]
                ),
            )
            if trigram_prediction == key:
                hits["trigram_prev2"] += 1
            if prefix_mode == key:
                hits["prefix_mode"] += 1
            if in_prefix:
                hits["oracle_in_own_prefix"] += 1
            if key in tables["vocabulary"]:
                hits["oracle_in_train_vocabulary"] += 1
            in_codebook = key in tables["codebook"]
            if in_codebook:
                hits["oracle_in_codebook_4096"] += 1
            # A drafter is not restricted to a fixed codebook: it can
            # also point at an action already present in the trajectory
            # it is reading. This union is the ceiling for a
            # codebook-plus-copy head, and it is the only ceiling that
            # matters once the fixed codebook alone falls short.
            if in_codebook or in_prefix:
                hits["oracle_codebook_or_prefix"] += 1
    return {
        "scored_steps": total,
        "skipped_first_action": skipped_first,
        "skipped_ineligible": skipped_ineligible,
        "rates": {
            name: (hits[name] / total if total else 0.0)
            for name in (
                "global_top1",
                "repeat_last",
                "prefix_mode",
                "bigram_prev",
                "trigram_prev2",
                "oracle_in_own_prefix",
                "oracle_in_train_vocabulary",
                "oracle_in_codebook_4096",
                "oracle_codebook_or_prefix",
            )
        },
    }


def corpus_shape(traces):
    lengths = sorted(len(trace["actions"]) for trace in traces)
    tools = collections.Counter()
    classes = collections.Counter()
    exact = collections.Counter()
    arg_bytes = []
    for trace in traces:
        for action in trace["actions"]:
            tools[action["tool"]] += 1
            classes[action["side_effect_class"]] += 1
            exact[action["action_digest"]] += 1
            arg_bytes.append(action["args_bytes"])
    total_actions = sum(lengths)
    eligible = sum(
        classes[name] for name in ELIGIBLE_CLASSES if name in classes
    )
    arg_bytes.sort()
    return {
        "traces": len(traces),
        "actions": total_actions,
        "median_actions_per_trace": (
            lengths[len(lengths) // 2] if lengths else 0
        ),
        "max_actions_per_trace": lengths[-1] if lengths else 0,
        "distinct_tools": len(tools),
        "distinct_exact_actions": len(exact),
        "exact_actions_seen_once": sum(
            1 for count in exact.values() if count == 1
        ),
        "median_argument_bytes": (
            arg_bytes[len(arg_bytes) // 2] if arg_bytes else 0
        ),
        "side_effect_classes": dict(classes),
        "eligible_fraction": (
            eligible / total_actions if total_actions else 0.0
        ),
        "top_tools": tools.most_common(10),
    }


def build_payload(path):
    traces = load_traces(path)
    train, evaluation = split_traces(traces)
    payload = {
        "worker": "agentspec_trace_match_baseline",
        "corpus": traces[0]["corpus"] if traces else "empty",
        "codebook_size": CODEBOOK_SIZE,
        "eligible_classes": list(ELIGIBLE_CLASSES),
        "required_match_probability": REQUIRED_MATCH_PROBABILITY,
        "shape": corpus_shape(traces),
        "split": {
            "train_traces": len(train),
            "eval_traces": len(evaluation),
        },
        "results": {},
    }
    for granularity in ("exact", "tool"):
        tables = fit_tables(train, granularity)
        for class_filter in (
            "all_actions",
            "eligible_only",
            "read_only_only",
        ):
            name = "%s|%s" % (granularity, class_filter)
            payload["results"][name] = evaluate(
                evaluation, tables, granularity, class_filter
            )
    serialised = json.dumps(payload, sort_keys=True)
    payload["payload_sha256"] = hashlib.sha256(
        serialised.encode("utf-8")
    ).hexdigest()
    return payload


def render(payload):
    shape = payload["shape"]
    lines = [
        "corpus                    %s" % payload["corpus"],
        "traces                    %d" % shape["traces"],
        "actions                   %d" % shape["actions"],
        "actions per trace         median %d, max %d"
        % (
            shape["median_actions_per_trace"],
            shape["max_actions_per_trace"],
        ),
        "distinct tools            %d" % shape["distinct_tools"],
        "distinct exact actions    %d (%d seen once)"
        % (
            shape["distinct_exact_actions"],
            shape["exact_actions_seen_once"],
        ),
        "median argument bytes     %d" % shape["median_argument_bytes"],
        "speculation eligible      %.4f of actions"
        % shape["eligible_fraction"],
        "side effect classes       %s"
        % json.dumps(shape["side_effect_classes"], sort_keys=True),
        "split                     %d train traces, %d eval traces"
        % (
            payload["split"]["train_traces"],
            payload["split"]["eval_traces"],
        ),
        "",
    ]
    predictors = (
        "global_top1",
        "repeat_last",
        "prefix_mode",
        "bigram_prev",
        "trigram_prev2",
        "oracle_in_own_prefix",
        "oracle_in_train_vocabulary",
        "oracle_in_codebook_4096",
        "oracle_codebook_or_prefix",
    )
    columns = (
        ("exact/all", "exact|all_actions"),
        ("exact/elig", "exact|eligible_only"),
        ("exact/ronly", "exact|read_only_only"),
        ("tool/elig", "tool|eligible_only"),
        ("tool/ronly", "tool|read_only_only"),
    )
    lines.append(
        "%-28s %11s %11s %11s %11s %11s"
        % (("predictor",) + tuple(name for name, _ in columns))
    )
    for predictor in predictors:
        lines.append(
            "%-28s %11.4f %11.4f %11.4f %11.4f %11.4f"
            % (
                (predictor,)
                + tuple(
                    payload["results"][key]["rates"][predictor]
                    for _name, key in columns
                )
            )
        )
    lines.append("")
    lines.append(
        "scored steps              all %d, eligible %d, read_only %d"
        % (
            payload["results"]["exact|all_actions"]["scored_steps"],
            payload["results"]["exact|eligible_only"]["scored_steps"],
            payload["results"]["exact|read_only_only"]["scored_steps"],
        )
    )
    best = max(
        payload["results"]["exact|eligible_only"]["rates"][name]
        for name in (
            "global_top1",
            "repeat_last",
            "prefix_mode",
            "bigram_prev",
            "trigram_prev2",
        )
    )
    ceiling = payload["results"]["exact|eligible_only"]["rates"][
        "oracle_in_codebook_4096"
    ]
    union_ceiling = payload["results"]["exact|eligible_only"]["rates"][
        "oracle_codebook_or_prefix"
    ]
    lines.append(
        "best training-free (exact, eligible)   %.4f" % best
    )
    lines.append(
        "codebook ceiling  (exact, eligible)    %.4f" % ceiling
    )
    lines.append(
        "codebook+copy ceiling (exact, eligible) %.4f" % union_ceiling
    )
    for context, required in sorted(
        payload["required_match_probability"].items(),
        key=lambda item: int(item[0]),
    ):
        lines.append(
            "required p @%-6s %.3f   training-free %s   "
            "codebook %s   codebook+copy %s"
            % (
                context,
                required,
                "PASS" if best >= required else "FAIL",
                "PASS" if ceiling >= required else "FAIL",
                "PASS" if union_ceiling >= required else "FAIL",
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Training-free next-action match baselines",
    )
    parser.add_argument("traces")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    payload = build_payload(args.traces)
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
