from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--failure-diagnostics", action="append", required=True)
    p.add_argument("--default-policy-dir", required=True)
    p.add_argument("--base-safe-policy-dir", default="")
    p.add_argument("--drop-heads", required=True, help="Comma-separated layer:kv_head list to drop on fragile tasks.")
    p.add_argument("--top-tasks", type=int, default=4)
    p.add_argument(
        "--per-doc-top-tasks",
        default="",
        help="Optional comma-separated doc=N mapping, e.g. first=4,second=3. Overrides --doc/--top-tasks.",
    )
    p.add_argument("--doc", default="", help="Optional doc filter, e.g. first or second.")
    p.add_argument("--output", required=True)
    p.add_argument("--note", default="")
    return p.parse_args()


def parse_heads(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_per_doc_top_tasks(text):
    mapping = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --per-doc-top-tasks item: {item}")
        doc, count_text = item.split("=", 1)
        doc = doc.strip()
        if not doc:
            raise ValueError(f"missing doc in --per-doc-top-tasks item: {item}")
        mapping[doc] = int(count_text)
    return mapping


def iter_failed_tasks(row):
    task_id = row.get("task_id", "").strip()
    if task_id:
        yield task_id
    fail_tasks = row.get("fail_tasks", "").strip()
    if fail_tasks:
        for item in fail_tasks.split(","):
            item = item.strip()
            if item:
                yield item


def row_is_failure(row):
    if row.get("task_id", "").strip():
        return True
    pass_gate = row.get("pass_gate", "").strip().lower()
    fail_tasks = row.get("fail_tasks", "").strip()
    if pass_gate in {"true", "1", "yes"}:
        return False
    return bool(fail_tasks)


def read_failure_counts(path, doc_filter):
    counts = Counter()
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if doc_filter and row.get("doc") != doc_filter:
                continue
            if not row_is_failure(row):
                continue
            for task_id in iter_failed_tasks(row):
                counts[task_id] += 1
    return counts


def read_failure_counts_many(paths, doc_filter):
    counts = Counter()
    for path in paths:
        counts.update(read_failure_counts(path, doc_filter))
    return counts


def build_policy(args):
    drop_heads = parse_heads(args.drop_heads)
    doc_top_tasks = parse_per_doc_top_tasks(args.per_doc_top_tasks)
    overrides = {}
    if doc_top_tasks:
        for doc, top_tasks in doc_top_tasks.items():
            counts = read_failure_counts_many(args.failure_diagnostics, doc)
            selected_tasks = [task_id for task_id, _ in counts.most_common(top_tasks)]
            for task_id in selected_tasks:
                overrides[task_id] = {
                    "drop_heads": drop_heads,
                    "reason": f"doc={doc} fragile_task_count={counts[task_id]}",
                }
    else:
        counts = read_failure_counts_many(args.failure_diagnostics, args.doc)
        selected_tasks = [task_id for task_id, _ in counts.most_common(args.top_tasks)]
        overrides = {
            task_id: {
                "drop_heads": drop_heads,
                "reason": f"fragile_task_count={counts[task_id]}",
            }
            for task_id in selected_tasks
        }
    return {
        "kind": "task_adaptive_light_doc_cache_policy",
        "default_policy_dir": args.default_policy_dir,
        "base_safe_policy_dir": args.base_safe_policy_dir,
        "failure_diagnostics": args.failure_diagnostics,
        "doc_filter": args.doc,
        "top_tasks": args.top_tasks,
        "doc_top_tasks": doc_top_tasks,
        "note": args.note,
        "overrides": overrides,
        "claim_boundary": (
            "Quality-only adaptive simulation; report average effective entry saving, "
            "not global runtime compression."
        ),
    }


def main():
    args = parse_args()
    payload = build_policy(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
