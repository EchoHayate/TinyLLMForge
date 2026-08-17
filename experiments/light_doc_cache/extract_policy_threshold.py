from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-rows", required=True)
    p.add_argument("--source-threshold", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-threshold", type=float, default=None)
    p.add_argument("--note", default="")
    return p.parse_args()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else v)
                for k, v in row.items()
            })


def as_int(row, key):
    return int(float(row[key]))


def as_float(row, key):
    return float(row[key])


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(row)
        for row in read_rows(args.policy_rows)
        if abs(float(row["threshold"]) - args.source_threshold) < 1e-9
    ]
    if not rows:
        raise ValueError(f"no rows found for source threshold {args.source_threshold}")
    target_threshold = args.source_threshold if args.target_threshold is None else args.target_threshold
    for row in rows:
        row["threshold"] = target_threshold

    compact_rows = [r for r in rows if r["action"] == "compact"]
    sampled_tokens = max(as_int(r, "selected_budget") for r in rows)
    original_entries = len(rows) * sampled_tokens
    compact_entries = sum(as_int(r, "selected_budget") for r in rows)
    qualities = [as_float(r, "quality") for r in compact_rows]
    summary = {
        "threshold": target_threshold,
        "compressed_heads": len(compact_rows),
        "total_heads": len(rows),
        "original_entries": original_entries,
        "compact_entries": compact_entries,
        "cache_entry_saving_fraction": 1.0 - compact_entries / original_entries,
        "compressed_quality_mean": sum(qualities) / len(qualities) if qualities else float("nan"),
        "selected_heads": ";".join(f"{as_int(r, 'layer')}:{as_int(r, 'kv_head')}" for r in compact_rows),
    }
    write_rows(out / "policy_rows.csv", rows)
    write_rows(out / "policy_summary.csv", [summary])
    (out / "policy.json").write_text(json.dumps({
        "source_policy_rows": str(args.policy_rows),
        "source_threshold": args.source_threshold,
        "target_threshold": target_threshold,
        "note": args.note,
        "summary": summary,
        "compact_heads": [
            {
                "layer": as_int(r, "layer"),
                "kv_head": as_int(r, "kv_head"),
                "budget": as_int(r, "selected_budget"),
                "quality": as_float(r, "quality"),
            }
            for r in compact_rows
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(
        "\n".join([
            "# Extracted Policy Threshold",
            "",
            f"Source rows: `{args.policy_rows}`",
            f"Source threshold: `{args.source_threshold}`",
            f"Target threshold: `{target_threshold}`",
            "",
            args.note,
            "",
            f"Compressed heads: {summary['compressed_heads']} / {summary['total_heads']}",
            f"Entry saving: {summary['cache_entry_saving_fraction']:.2%}",
            f"Selected heads: `{summary['selected_heads']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
