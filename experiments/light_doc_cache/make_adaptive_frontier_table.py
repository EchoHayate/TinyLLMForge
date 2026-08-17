from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "Comma-separated key=value frontier entry. Required keys: "
            "name,kind,heads,first,second. Optional: saving,fallback_tasks,claim."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_entry(text):
    payload = {}
    for item in text.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid entry item: {item}")
        key, value = item.split("=", 1)
        payload[key.strip()] = value.strip()
    required = {"name", "kind", "heads", "first", "second"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing entry keys: {missing}")
    return payload


def read_summary(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("summary", data)
    if isinstance(rows, list):
        if not rows:
            raise ValueError(f"empty summary: {path}")
        return rows[0]
    if isinstance(rows, dict):
        return rows
    raise ValueError(f"unsupported summary format: {path}")


def pct(value):
    return f"{float(value) * 100:.2f}%"


def saving_for(row, entry):
    if "effective_entry_saving_fraction" in row:
        return pct(row["effective_entry_saving_fraction"])
    if entry.get("saving"):
        return pct(float(entry["saving"]) / 100 if float(entry["saving"]) > 1 else float(entry["saving"]))
    return ""


def gate(row):
    tasks = int(row["tasks"])
    ok = (
        float(row.get("baseline_accuracy", 1.0)) == 1.0
        and float(row.get("compact_accuracy", 0.0)) == 1.0
        and float(row.get("agreement", 0.0)) == 1.0
    )
    return f"{tasks}/{tasks}" if ok else f"{float(row.get('compact_accuracy', 0.0)) * tasks:.0f}/{tasks}"


def build_row(entry):
    first = read_summary(entry["first"])
    second = read_summary(entry["second"])
    return {
        "name": entry["name"],
        "kind": entry["kind"],
        "heads": entry["heads"],
        "first_gate": gate(first),
        "second_gate": gate(second),
        "first_saving": saving_for(first, entry),
        "second_saving": saving_for(second, entry),
        "first_delta": f"{float(first.get('mean_answer_score_delta', 0.0)):.4f}",
        "second_delta": f"{float(second.get('mean_answer_score_delta', 0.0)):.4f}",
        "first_bank_build_s": f"{float(first.get('mean_bank_build_s', 0.0)):.4f}",
        "second_bank_build_s": f"{float(second.get('mean_bank_build_s', 0.0)):.4f}",
        "fallback_tasks": entry.get("fallback_tasks", "0"),
        "claim": entry.get("claim", "quality_only"),
    }


def write_csv(path, rows):
    fields = [
        "name",
        "kind",
        "heads",
        "first_gate",
        "second_gate",
        "first_saving",
        "second_saving",
        "first_delta",
        "second_delta",
        "first_bank_build_s",
        "second_bank_build_s",
        "fallback_tasks",
        "claim",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Light Doc Cache Frontier Table",
        "",
        "Quality-only recovery-bank simulation results. These rows preserve the strict task gates in the listed offline smoke tests, but they are not runtime KV-cache compression measurements.",
        "",
        "| Frontier | Kind | Heads | First Doc Gate | Second Doc Gate | First Saving | Second Saving | First Delta | Second Delta | Fallback Tasks | Claim |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        claim = "quality-only" if row["claim"] == "quality_only" else row["claim"]
        lines.append(
            "| {name} | {kind} | {heads} | {first_gate} | {second_gate} | "
            "{first_saving} | {second_saving} | {first_delta} | {second_delta} | "
            "{fallback_tasks} | {claim} |".format(**{**row, "claim": claim})
        )
    lines.extend(
        [
            "",
            "Claim boundary: report these as offline task/document-adaptive recovery-bank quality results with average effective KV head-token entry saving. Do not describe them as 2x+ runtime doc-cache compression.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [build_row(parse_entry(text)) for text in args.entry]
    write_csv(output_dir / "frontier_table.csv", rows)
    write_markdown(output_dir / "frontier_table.md", rows)
    print(output_dir)


if __name__ == "__main__":
    main()
