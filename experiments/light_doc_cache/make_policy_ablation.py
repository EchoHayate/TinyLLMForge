from __future__ import annotations

import argparse
import csv
import json
import math
import itertools
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-rows", required=True, help="Source policy_rows.csv with one threshold to ablate.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-threshold", type=float, required=True)
    p.add_argument("--variant-step", type=float, default=0.001)
    p.add_argument("--mode", default="leave_one_out", choices=["leave_one_out", "pair_drop", "prefix"])
    p.add_argument("--max-variants", type=int, default=0)
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


def source_rows(rows, threshold):
    selected = [r for r in rows if abs(float(r["threshold"]) - threshold) < 1e-9]
    if not selected:
        raise ValueError(f"no rows found for source threshold {threshold}")
    return selected


def compact_keys(rows):
    return [
        (as_int(r, "layer"), as_int(r, "kv_head"))
        for r in rows
        if r["action"] == "compact"
    ]


def variant_specs(keys, mode, max_variants):
    if mode == "leave_one_out":
        specs = [
            {
                "name": f"drop_l{layer}_h{kv_head}",
                "drop": {(layer, kv_head)},
            }
            for layer, kv_head in keys
        ]
    elif mode == "pair_drop":
        specs = [
            {
                "name": f"drop_l{left[0]}_h{left[1]}__l{right[0]}_h{right[1]}",
                "drop": {left, right},
            }
            for left, right in itertools.combinations(keys, 2)
        ]
    elif mode == "prefix":
        specs = [
            {
                "name": f"first_{count}",
                "keep": set(keys[:count]),
            }
            for count in range(1, len(keys) + 1)
        ]
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return specs[:max_variants] if max_variants > 0 else specs


def full_budget(rows):
    return max(as_int(row, "selected_budget") for row in rows)


def remap_rows(rows, threshold, spec):
    drop = spec.get("drop", set())
    keep = spec.get("keep")
    full_selected_budget = full_budget(rows)
    out = []
    for row in rows:
        new_row = dict(row)
        new_row["threshold"] = threshold
        key = (as_int(row, "layer"), as_int(row, "kv_head"))
        should_compact = row["action"] == "compact"
        if keep is not None:
            should_compact = key in keep and row["action"] == "compact"
        if key in drop:
            should_compact = False
        if row["action"] == "compact" and not should_compact:
            new_row["action"] = "full"
            new_row["selected_budget"] = full_selected_budget
            new_row["budget_fraction"] = 1.0
            new_row["quality"] = 1.0
            new_row["direct_val_r2"] = ""
            new_row["fitv_val_r2"] = ""
            new_row["reason"] = spec["name"]
        out.append(new_row)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = source_rows(read_rows(args.policy_rows), args.source_threshold)
    keys = compact_keys(rows)
    specs = variant_specs(keys, args.mode, args.max_variants)

    all_rows = []
    summary_rows = []
    variants = []
    for index, spec in enumerate(specs, start=1):
        threshold = args.source_threshold + args.variant_step * index
        variant_rows = remap_rows(rows, threshold, spec)
        all_rows.extend(variant_rows)
        compact = compact_keys(variant_rows)
        variants.append({
            "threshold": threshold,
            "name": spec["name"],
            "compressed_heads": len(compact),
            "compact_heads": [{"layer": layer, "kv_head": kv_head} for layer, kv_head in compact],
        })
        summary_rows.append({
            "threshold": threshold,
            "variant": spec["name"],
            "compressed_heads": len(compact),
            "dropped_heads": len(keys) - len(compact),
        })

    write_rows(out_dir / "policy_rows.csv", all_rows)
    write_rows(out_dir / "policy_summary.csv", summary_rows)
    (out_dir / "policy.json").write_text(json.dumps({
        "source_policy_rows": str(args.policy_rows),
        "source_threshold": args.source_threshold,
        "mode": args.mode,
        "variant_step": args.variant_step,
        "note": args.note,
        "source_compressed_heads": len(keys),
        "variants": variants,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Policy Ablation Variants",
        "",
        f"Source policy rows: `{args.policy_rows}`",
        f"Source threshold: `{args.source_threshold}`",
        f"Mode: `{args.mode}`",
        "",
        args.note,
        "",
        "| Threshold | Variant | Compressed Heads | Dropped Heads |",
        "|---:|---|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['threshold']:.3f} | {row['variant']} | {row['compressed_heads']} | {row['dropped_heads']} |"
        )
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
