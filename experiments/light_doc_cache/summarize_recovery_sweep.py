from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Run directories or parent directories containing summary.json files.")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def as_float(payload, key):
    try:
        return float(payload.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def find_summary_files(inputs):
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.name == "summary.json":
            files.append(path)
        elif (path / "summary.json").is_file():
            files.append(path / "summary.json")
        elif path.is_dir():
            files.extend(sorted(path.glob("*/summary.json")))
    return sorted(set(files))


def collect_summaries(inputs):
    rows = []
    for summary_file in find_summary_files(inputs):
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
        settings = payload.get("settings", {}) if isinstance(payload.get("settings"), dict) else {}
        rows.append(
            {
                "run_name": summary_file.parent.name,
                "run_dir": str(summary_file.parent),
                "decision": payload.get("decision", ""),
                "heads": int(payload.get("heads", 0) or 0),
                "budgets": str(settings.get("budgets", "")),
                "start_layer": int(settings.get("start_layer", 0) or 0),
                "end_layer": int(settings.get("end_layer", 0) or 0),
                "max_heads": int(settings.get("max_heads", 0) or 0),
                "mean_budget_fraction": as_float(payload, "mean_budget_fraction"),
                "mean_direct_val_r2": as_float(payload, "mean_direct_val_r2"),
                "mean_fitv_val_r2": as_float(payload, "mean_fitv_val_r2"),
                "mean_recovery_val_r2": as_float(payload, "mean_recovery_val_r2"),
                "p50_recovery_val_r2": as_float(payload, "p50_recovery_val_r2"),
                "p90_recovery_val_r2": as_float(payload, "p90_recovery_val_r2"),
                "recovery_val_ge_accept": as_float(payload, "recovery_val_ge_accept"),
                "mean_recovery_gain_vs_direct": as_float(payload, "mean_recovery_gain_vs_direct"),
                "mean_recovery_gain_vs_fitv": as_float(payload, "mean_recovery_gain_vs_fitv"),
            }
        )
    rows.sort(key=lambda row: (row["mean_recovery_val_r2"], row["mean_recovery_gain_vs_fitv"]), reverse=True)
    return rows


def fmt(value, digits=4):
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.{digits}f}"
        return "nan"
    return str(value)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "run_name",
        "decision",
        "heads",
        "budgets",
        "start_layer",
        "end_layer",
        "max_heads",
        "mean_budget_fraction",
        "mean_direct_val_r2",
        "mean_fitv_val_r2",
        "mean_recovery_val_r2",
        "p50_recovery_val_r2",
        "p90_recovery_val_r2",
        "recovery_val_ge_accept",
        "mean_recovery_gain_vs_direct",
        "mean_recovery_gain_vs_fitv",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value, 6) if isinstance(value, float) else value for key, value in row.items()})


def write_sweep_outputs(output_dir, rows):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "recovery_sweep_summary.csv", rows)
    report = [
        "# Recovery Sweep Summary",
        "",
        "| Run | Decision | Start Layer | Budget | Heads | Budget Frac | Direct R2 | FitV R2 | Recovery R2 | Gain vs FitV | Coverage |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['run_name']} | {row['decision']} | {row['start_layer']} | {row['budgets']} | {row['heads']} | "
            f"{row['mean_budget_fraction']:.2%} | {row['mean_direct_val_r2']:.4f} | "
            f"{row['mean_fitv_val_r2']:.4f} | {row['mean_recovery_val_r2']:.4f} | "
            f"{row['mean_recovery_gain_vs_fitv']:.4f} | {row['recovery_val_ge_accept']:.2%} |"
        )
    if rows:
        best = rows[0]
        report.extend(
            [
                "",
                "## Best Run",
                "",
                f"- Run: `{best['run_name']}`",
                f"- Decision: `{best['decision']}`",
                f"- Mean recovery val R2: `{best['mean_recovery_val_r2']:.4f}`",
                f"- Mean gain vs FitV: `{best['mean_recovery_gain_vs_fitv']:.4f}`",
            ]
        )
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = collect_summaries(args.inputs)
    write_sweep_outputs(args.output_dir, rows)
    print(f"runs {len(rows)}")
    if rows:
        best = rows[0]
        print("best", best["run_name"], f"recovery_r2={best['mean_recovery_val_r2']:.4f}")
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
