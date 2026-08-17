from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.77)
    p.add_argument("--compact-layer", default=None, help="Layer index to compact, or 'all'.")
    p.add_argument("--compact-layer-range", default=None, help="Half-open layer range, e.g. 24:28.")
    p.add_argument("--compact-heads", default=None, help="Comma-separated layer:kv_head list, e.g. 24:0,24:1.")
    p.add_argument("--budget-fraction", type=float, required=True)
    p.add_argument("--seq-len", type=int, default=1536)
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--note", default="layer_recovery_task_policy")
    return p.parse_args()


def compact_layers(args):
    if args.compact_heads:
        layers = {int(item.split(":", 1)[0]) for item in args.compact_heads.split(",") if item.strip()}
        return layers, "heads"
    if args.compact_layer_range:
        start_text, end_text = args.compact_layer_range.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        if not (0 <= start < end <= args.num_layers):
            raise ValueError("--compact-layer-range must be within [0, num-layers]")
        return set(range(start, end)), f"{start}:{end}"
    if args.compact_layer is None:
        raise ValueError("one of --compact-layer, --compact-layer-range, or --compact-heads is required")
    if args.compact_layer == "all":
        return set(range(args.num_layers)), "all"
    layer = int(args.compact_layer)
    if not (0 <= layer < args.num_layers):
        raise ValueError("--compact-layer must be 'all' or within [0, num-layers)")
    return {layer}, str(layer)


def compact_head_set(args):
    if not args.compact_heads:
        return None
    heads = set()
    for item in args.compact_heads.split(","):
        item = item.strip()
        if not item:
            continue
        layer_text, head_text = item.split(":", 1)
        layer = int(layer_text)
        kv_head = int(head_text)
        if not (0 <= layer < args.num_layers):
            raise ValueError(f"compact head layer out of range: {item}")
        if not (0 <= kv_head < args.num_kv_heads):
            raise ValueError(f"compact kv_head out of range: {item}")
        heads.add((layer, kv_head))
    if not heads:
        raise ValueError("--compact-heads must contain at least one item")
    return heads


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


def build_policy_rows(args):
    selected_budget = max(2, min(args.seq_len, int(round(args.seq_len * args.budget_fraction))))
    layers, _ = compact_layers(args)
    heads = compact_head_set(args)
    rows = []
    for layer in range(args.num_layers):
        for kv_head in range(args.num_kv_heads):
            compact = (layer, kv_head) in heads if heads is not None else layer in layers
            rows.append(
                {
                    "threshold": args.threshold,
                    "layer": layer,
                    "kv_head": kv_head,
                    "action": "compact" if compact else "full",
                    "selected_budget": selected_budget if compact else args.seq_len,
                    "budget_fraction": selected_budget / args.seq_len if compact else 1.0,
                    "quality": 0.0 if compact else 1.0,
                    "direct_val_r2": "",
                    "fitv_val_r2": "",
                    "reason": args.note if compact else "not_target_layer",
                }
            )
    return rows


def summarize(rows, args):
    compact_rows = [row for row in rows if row["action"] == "compact"]
    original_entries = args.num_layers * args.num_kv_heads * args.seq_len
    compact_entries = (
        len(compact_rows) * int(compact_rows[0]["selected_budget"])
        + (args.num_layers * args.num_kv_heads - len(compact_rows)) * args.seq_len
        if compact_rows
        else original_entries
    )
    return [
        {
            "threshold": args.threshold,
            "compact_layer": compact_layers(args)[1],
            "compressed_heads": len(compact_rows),
            "selected_budget": int(compact_rows[0]["selected_budget"]) if compact_rows else args.seq_len,
            "budget_fraction": float(compact_rows[0]["budget_fraction"]) if compact_rows else 1.0,
            "entry_saving_fraction": 1.0 - compact_entries / original_entries,
            "note": args.note,
        }
    ]


def main():
    args = parse_args()
    if not (0.0 < args.budget_fraction <= 1.0):
        raise ValueError("--budget-fraction must be in (0, 1]")
    layer_label = compact_layers(args)[1]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_policy_rows(args)
    summary = summarize(rows, args)
    write_rows(out / "policy_rows.csv", rows)
    write_rows(out / "policy_summary.csv", summary)
    (out / "policy.json").write_text(
        json.dumps(
            {
                "kind": "layer_recovery_task_policy",
                "settings": vars(args),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = [
        "# Recovery Task Policy",
        "",
        f"Kind: `layer_recovery_task_policy`",
        f"Compact layer: `{layer_label}`",
        f"Budget fraction: `{args.budget_fraction:.2%}`",
        f"Threshold: `{args.threshold}`",
        "",
        "| Threshold | Compact Layer | Compressed Heads | Selected Budget | Budget Frac | Entry Saving |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    row = summary[0]
    report.append(
        f"| {row['threshold']:.2f} | {row['compact_layer']} | {row['compressed_heads']} | "
        f"{row['selected_budget']} | {row['budget_fraction']:.2%} | {row['entry_saving_fraction']:.2%} |"
    )
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
