from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--am-run", required=True, help="Directory produced by probe_am_compact_cache.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--regime", default="holdout", choices=["holdout", "in_sample"])
    p.add_argument("--selector", default="highest")
    p.add_argument("--metric", default="fitv_val_r2")
    p.add_argument("--quality-thresholds", default="0.25,0.35,0.50,0.80")
    p.add_argument("--min-layer", type=int, default=0)
    p.add_argument("--max-layer", type=int, default=-1)
    p.add_argument("--max-compact-heads", type=int, default=0, help="Optional global cap for compacted heads.")
    p.add_argument(
        "--max-compact-heads-per-layer",
        type=int,
        default=0,
        help="Optional per-layer cap for compacted heads.",
    )
    p.add_argument(
        "--min-saving-fraction",
        type=float,
        default=0.0,
        help="Only compact heads whose selected budget saves at least this fraction of sampled tokens.",
    )
    p.add_argument("--full-head-quality", type=float, default=1.0)
    p.add_argument("--note", default="")
    return p.parse_args()


def parse_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_summary(run_dir):
    path = Path(run_dir) / "summary.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def percentile(vals, q):
    vals = sorted(float(v) for v in vals)
    if not vals:
        return float("nan")
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


def as_int(row, key):
    return int(float(row[key]))


def as_float(row, key):
    return float(row[key])


def pick_min_budget(candidates, metric, threshold):
    passing = [r for r in candidates if as_float(r, metric) >= threshold]
    if not passing:
        return None
    passing.sort(key=lambda r: (as_int(r, "budget"), -as_float(r, metric)))
    return passing[0]


def rank_compact_candidates(candidates, sampled_tokens, metric):
    return sorted(
        candidates,
        key=lambda item: (
            -as_float(item["choice"], metric),
            as_int(item["choice"], "budget") / sampled_tokens,
            item["layer"],
            item["kv_head"],
        ),
    )


def select_constrained_candidates(candidate_by_head, sampled_tokens, metric, args):
    selected = {}
    per_layer_counts = {}
    max_heads = max(0, int(args.max_compact_heads))
    max_heads_per_layer = max(0, int(args.max_compact_heads_per_layer))
    for item in rank_compact_candidates(candidate_by_head.values(), sampled_tokens, metric):
        layer = item["layer"]
        key = (layer, item["kv_head"])
        if max_heads and len(selected) >= max_heads:
            continue
        if max_heads_per_layer and per_layer_counts.get(layer, 0) >= max_heads_per_layer:
            continue
        selected[key] = item["choice"]
        per_layer_counts[layer] = per_layer_counts.get(layer, 0) + 1
    return selected


def build_policy(rows, meta, args, threshold):
    sampled_tokens = int(meta["sampled_tokens"])
    num_layers = int(meta["num_layers"])
    num_kv_heads = int(meta["num_kv_heads"])
    max_layer = num_layers - 1 if args.max_layer < 0 else min(args.max_layer, num_layers - 1)
    min_layer = max(0, args.min_layer)

    filtered = []
    for row in rows:
        layer = as_int(row, "layer")
        if row["regime"] != args.regime or row["selector"] != args.selector:
            continue
        if layer < min_layer or layer > max_layer:
            continue
        filtered.append(row)

    by_head = {}
    for row in filtered:
        key = (as_int(row, "layer"), as_int(row, "kv_head"))
        by_head.setdefault(key, []).append(row)

    candidate_by_head = {}
    below_threshold_heads = set()
    min_saving_filtered_heads = set()
    for key, candidates in by_head.items():
        choice = pick_min_budget(candidates, args.metric, threshold)
        if choice is None:
            below_threshold_heads.add(key)
            continue
        budget = as_int(choice, "budget")
        saving_fraction = 1.0 - budget / sampled_tokens
        if saving_fraction + 1e-12 < args.min_saving_fraction:
            min_saving_filtered_heads.add(key)
            continue
        candidate_by_head[key] = {
            "layer": key[0],
            "kv_head": key[1],
            "choice": choice,
        }
    selected_choices = select_constrained_candidates(candidate_by_head, sampled_tokens, args.metric, args)

    policy_rows = []
    compact_entries = 0
    original_entries = num_layers * num_kv_heads * sampled_tokens
    compacted_qualities = []
    effective_qualities = []
    compressed_by_layer = {layer: 0 for layer in range(num_layers)}
    available_heads = set(by_head)

    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            key = (layer, kv_head)
            eligible = min_layer <= layer <= max_layer and key in by_head
            choice = selected_choices.get(key)
            if choice is None:
                compact_entries += sampled_tokens
                effective_qualities.append(args.full_head_quality)
                if not eligible:
                    reason = "outside_layer_filter"
                elif key in below_threshold_heads:
                    reason = "below_threshold"
                elif key in min_saving_filtered_heads:
                    reason = "below_min_saving"
                elif key in candidate_by_head:
                    reason = "constrained_out"
                else:
                    reason = "below_threshold"
                policy_rows.append(dict(
                    threshold=threshold,
                    layer=layer,
                    kv_head=kv_head,
                    action="full",
                    selected_budget=sampled_tokens,
                    budget_fraction=1.0,
                    quality=args.full_head_quality,
                    direct_val_r2="",
                    fitv_val_r2="",
                    reason=reason,
                ))
                continue

            budget = as_int(choice, "budget")
            quality = as_float(choice, args.metric)
            compact_entries += budget
            compacted_qualities.append(quality)
            effective_qualities.append(quality)
            compressed_by_layer[layer] += 1
            policy_rows.append(dict(
                threshold=threshold,
                layer=layer,
                kv_head=kv_head,
                action="compact",
                selected_budget=budget,
                budget_fraction=budget / sampled_tokens,
                quality=quality,
                direct_val_r2=as_float(choice, "direct_val_r2"),
                fitv_val_r2=as_float(choice, "fitv_val_r2"),
                reason="meets_threshold",
            ))

    compressed_heads = len(compacted_qualities)
    total_heads = num_layers * num_kv_heads
    selected_layers = [layer for layer, count in compressed_by_layer.items() if count > 0]
    summary = dict(
        threshold=threshold,
        regime=args.regime,
        selector=args.selector,
        metric=args.metric,
        min_layer=min_layer,
        max_layer=max_layer,
        sampled_tokens=sampled_tokens,
        total_heads=total_heads,
        candidate_heads=len(available_heads),
        compressed_heads=compressed_heads,
        compressed_head_fraction=compressed_heads / total_heads,
        original_entries=original_entries,
        compact_entries=compact_entries,
        cache_entry_fraction=compact_entries / original_entries,
        cache_entry_saving_fraction=1.0 - compact_entries / original_entries,
        compressed_quality_mean=sum(compacted_qualities) / len(compacted_qualities) if compacted_qualities else float("nan"),
        compressed_quality_min=min(compacted_qualities) if compacted_qualities else float("nan"),
        compressed_quality_p10=percentile(compacted_qualities, 0.10),
        compressed_quality_p50=percentile(compacted_qualities, 0.50),
        compressed_quality_p90=percentile(compacted_qualities, 0.90),
        effective_quality_mean=sum(effective_qualities) / len(effective_qualities),
        selected_layer_count=len(selected_layers),
        selected_layers=";".join(str(x) for x in selected_layers),
        compressed_heads_by_layer=";".join(f"{layer}:{count}" for layer, count in compressed_by_layer.items() if count > 0),
        max_compact_heads=max(0, int(args.max_compact_heads)),
        max_compact_heads_per_layer=max(0, int(args.max_compact_heads_per_layer)),
        min_saving_fraction=args.min_saving_fraction,
        eligible_compact_candidates=len(candidate_by_head),
    )
    return summary, policy_rows


def main():
    args = parse_args()
    am_run = Path(args.am_run)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(am_run / "am_head_rows.csv")
    summary_json = read_summary(am_run)
    meta = summary_json["metadata"]
    thresholds = parse_floats(args.quality_thresholds)

    all_summaries = []
    all_policy_rows = []
    policy_json = {
        "source_run": str(am_run),
        "source_decision": summary_json.get("decision"),
        "source_reason": summary_json.get("decision_reason"),
        "metadata": meta,
        "settings": vars(args),
        "policies": [],
    }
    for threshold in thresholds:
        summary, policy_rows = build_policy(rows, meta, args, threshold)
        all_summaries.append(summary)
        all_policy_rows.extend(policy_rows)
        policy_json["policies"].append({
            "summary": summary,
            "compact_heads": [
                {
                    "layer": r["layer"],
                    "kv_head": r["kv_head"],
                    "budget": r["selected_budget"],
                    "quality": r["quality"],
                }
                for r in policy_rows if r["action"] == "compact"
            ],
        })

    write_rows(out / "policy_summary.csv", all_summaries)
    write_rows(out / "policy_rows.csv", all_policy_rows)
    (out / "policy.json").write_text(json.dumps(policy_json, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Attention-Output Selective Compression Policy",
        "",
        f"Source run: `{am_run}`",
        f"Source decision: **{summary_json.get('decision')}**",
        "",
        summary_json.get("decision_reason", ""),
        "",
        f"Regime: `{args.regime}`; selector: `{args.selector}`; metric: `{args.metric}`; layer filter: `{args.min_layer}`..`{args.max_layer}`.",
    ]
    if args.note:
        report.extend(["", args.note])
    report.extend([
        "",
        "| R2 threshold | Compressed Heads | Head Fraction | Cache Entry Fraction | Cache Saving | Mean Compressed R2 | P10/P50/P90 R2 | Layers |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for s in all_summaries:
        report.append(
            f"| {s['threshold']:.2f} | {s['compressed_heads']} / {s['total_heads']} | "
            f"{s['compressed_head_fraction']:.2%} | {s['cache_entry_fraction']:.2%} | "
            f"{s['cache_entry_saving_fraction']:.2%} | {s['compressed_quality_mean']:.4f} | "
            f"{s['compressed_quality_p10']:.3f}/{s['compressed_quality_p50']:.3f}/{s['compressed_quality_p90']:.3f} | "
            f"{s['selected_layers']} |"
        )
    report.extend([
        "",
        "Interpretation:",
        "- `Cache Entry Fraction` treats each KV head-token entry as one unit. A compacted head uses the selected AM budget; an uncompressed head keeps all sampled tokens.",
        "- This is still an attention-output proxy, not an end-to-end generation quality measurement.",
        "- A policy is useful only if the holdout policy gives meaningful cache savings at a quality threshold high enough for downstream tasks.",
    ])
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out)
    for s in all_summaries:
        print(
            f"thr={s['threshold']:.2f} compressed={s['compressed_heads']}/{s['total_heads']} "
            f"cache_frac={s['cache_entry_fraction']:.2%} saving={s['cache_entry_saving_fraction']:.2%} "
            f"mean_r2={s['compressed_quality_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
