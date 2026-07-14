"""Aggregate Light Doc Cache multi-target read-path artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

REQUIRED_CATEGORIES = {
    "short_factual",
    "long_document_qa",
    "source_code",
    "mathematical_reasoning",
    "structured_text",
    "repetitive_text",
    "cross_paragraph_dependency",
    "out_of_distribution",
}
LENGTH_BUCKETS = {"short", "medium", "long"}
TOKEN_BUCKET_RANGES = {
    "short": (16, 48),
    "medium": (49, 160),
    "long": (161, 384),
}
REQUIRED_MODES = (
    "repeat_last_target",
    "correlated_same_layer_target",
    "calibration_holdout",
)
ROW_FIELDS = (
    "target_id",
    "category",
    "length_bucket",
    "mode",
    "role",
    "status",
    "error",
    "prompt_tokens",
    "calibration_bank_sha256",
    "logical_byte_saving_fraction",
    "missing_tokens",
    "missing_mse",
    "missing_mae",
    "missing_max_abs",
    "max_abs_logit_diff",
    "mean_abs_logit_diff",
    "argmax_match",
    "original_argmax",
    "restored_argmax",
    "artifact",
)


def load_target_dataset(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = validate_target_dataset(payload)
    return {"version": int(payload["version"]), "targets": targets}


def validate_target_dataset(payload: dict[str, object]) -> list[dict[str, str]]:
    if int(payload.get("version", 0)) != 1:
        raise ValueError("target dataset version must be 1")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 8:
        raise ValueError("target dataset must contain exactly eight targets")
    targets: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    category_counts = {category: 0 for category in REQUIRED_CATEGORIES}
    bucket_counts = {bucket: 0 for bucket in LENGTH_BUCKETS}
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise ValueError(f"target {index} must be an object")
        target = {
            "id": str(raw.get("id", "")).strip(),
            "category": str(raw.get("category", "")).strip(),
            "length_bucket": str(raw.get("length_bucket", "")).strip(),
            "prompt": str(raw.get("prompt", "")).strip(),
        }
        if not target["id"] or not target["prompt"]:
            raise ValueError(f"target {index} requires non-empty id and prompt")
        if target["id"] in seen_ids:
            raise ValueError(f"duplicate target id: {target['id']}")
        if target["category"] not in REQUIRED_CATEGORIES:
            raise ValueError(f"unknown target category: {target['category']}")
        if target["length_bucket"] not in LENGTH_BUCKETS:
            raise ValueError(f"unknown length bucket: {target['length_bucket']}")
        seen_ids.add(target["id"])
        category_counts[target["category"]] += 1
        bucket_counts[target["length_bucket"]] += 1
        targets.append(target)
    if {category for category, count in category_counts.items() if count} != REQUIRED_CATEGORIES:
        raise ValueError("target dataset must cover every required category exactly once")
    if any(count < 2 for count in bucket_counts.values()):
        raise ValueError("target dataset requires at least two targets in each length bucket")
    return targets


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def hashlib_sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": nearest_rank_percentile(values, 0.90),
        "worst": max(values),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, dict[str, Any]] = {}
    for mode in REQUIRED_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        successful = [row for row in mode_rows if row["status"] == "success"]
        argmax_match_count = sum(bool(row["argmax_match"]) for row in successful)
        modes[mode] = {
            "attempted_targets": len(mode_rows),
            "completed_targets": len(successful),
            "failed_targets": len(mode_rows) - len(successful),
            "argmax_match_count": argmax_match_count,
            "argmax_match_rate": (
                argmax_match_count / len(successful) if successful else 0.0
            ),
            "mean_abs_logit_diff": _metric_summary(
                [float(row["mean_abs_logit_diff"]) for row in successful]
            ),
            "max_abs_logit_diff": _metric_summary(
                [float(row["max_abs_logit_diff"]) for row in successful]
            ),
            "missing_mse": _metric_summary(
                [float(row["missing_mse"]) for row in successful]
            ),
            "mean_logical_byte_saving_fraction": (
                statistics.fmean(
                    float(row["logical_byte_saving_fraction"]) for row in successful
                )
                if successful
                else 0.0
            ),
        }
    return {
        "claim_boundary": "default_off_multi_target_read_path_gate",
        "row_count": len(rows),
        "modes": modes,
        "gate": evaluate_gate(rows),
    }


def evaluate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful_by_key = {
        (str(row["target_id"]), str(row["mode"])): row
        for row in rows
        if row["status"] == "success"
    }
    target_ids = sorted(
        {
            str(row["target_id"])
            for row in rows
            if row["mode"]
            in {
                "correlated_same_layer_target",
                "calibration_holdout",
            }
        }
    )
    pairs = []
    for target_id in target_ids:
        correlated = successful_by_key.get(
            (target_id, "correlated_same_layer_target")
        )
        holdout = successful_by_key.get((target_id, "calibration_holdout"))
        if correlated is not None and holdout is not None:
            pairs.append((target_id, correlated, holdout))

    correlated_mean = (
        statistics.fmean(
            float(correlated["mean_abs_logit_diff"])
            for _, correlated, _ in pairs
        )
        if pairs
        else math.inf
    )
    holdout_mean = (
        statistics.fmean(
            float(holdout["mean_abs_logit_diff"]) for _, _, holdout in pairs
        )
        if pairs
        else math.inf
    )
    improvement = (
        (correlated_mean - holdout_mean) / correlated_mean
        if pairs and correlated_mean > 0.0
        else 0.0
    )
    relative_changes = [
        (
            float(holdout["mean_abs_logit_diff"])
            - float(correlated["mean_abs_logit_diff"])
        )
        / float(correlated["mean_abs_logit_diff"])
        for _, correlated, holdout in pairs
        if float(correlated["mean_abs_logit_diff"]) > 0.0
    ]
    holdout_wins = sum(
        float(holdout["mean_abs_logit_diff"])
        < float(correlated["mean_abs_logit_diff"])
        for _, correlated, holdout in pairs
    )
    correlated_argmax = sum(
        bool(correlated["argmax_match"]) for _, correlated, _ in pairs
    )
    holdout_argmax = sum(bool(holdout["argmax_match"]) for _, _, holdout in pairs)
    argmax_regressions = [
        target_id
        for target_id, correlated, holdout in pairs
        if bool(correlated["argmax_match"]) and not bool(holdout["argmax_match"])
    ]
    token_bucket_mismatches = sorted(
        {
            str(row["target_id"])
            for row in rows
            if row["status"] == "success"
            and row["length_bucket"] in TOKEN_BUCKET_RANGES
            and not (
                TOKEN_BUCKET_RANGES[str(row["length_bucket"])][0]
                <= int(row["prompt_tokens"])
                <= TOKEN_BUCKET_RANGES[str(row["length_bucket"])][1]
            )
        }
    )
    conditions = {
        "all eight paired targets completed": len(pairs) == 8,
        "holdout argmax rate not lower": holdout_argmax >= correlated_argmax,
        "holdout wins at least five targets": holdout_wins >= 5,
        "mean logit diff improves at least five percent": improvement >= 0.05,
        "worst relative regression no more than twenty five percent": (
            bool(relative_changes) and max(relative_changes) <= 0.25
        ),
        "no correlated argmax match regressed": not argmax_regressions,
        "actual prompt tokens match intended length buckets": (
            not token_bucket_mismatches
        ),
    }
    return {
        "decision": "GO" if all(conditions.values()) else "NO_GO",
        "conditions": conditions,
        "failed_conditions": [
            name for name, passed in conditions.items() if not passed
        ],
        "paired_targets": len(pairs),
        "holdout_win_count": holdout_wins,
        "holdout_win_rate": holdout_wins / len(pairs) if pairs else 0.0,
        "correlated_mean_abs_logit_diff": correlated_mean,
        "holdout_mean_abs_logit_diff": holdout_mean,
        "aggregate_relative_improvement": improvement,
        "worst_relative_regression": (
            max(relative_changes) if relative_changes else None
        ),
        "argmax_regressions": argmax_regressions,
        "token_bucket_mismatches": token_bucket_mismatches,
    }


def normalize_summary_row(
    *,
    target_id: str,
    category: str,
    length_bucket: str,
    mode: str,
    summary_path: str | Path,
) -> dict[str, Any]:
    path = Path(summary_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    error_metrics = payload["sidecar"]["error_metrics"]
    logit_compare = payload["logit_compare"]
    bank_path_text = (
        str(payload.get("recovery_bank_file") or "")
        if mode == "calibration_holdout"
        else ""
    )
    bank_path = Path(bank_path_text) if bank_path_text else None
    bank_sha256 = (
        hashlib_sha256_file(bank_path)
        if bank_path is not None and bank_path.is_file()
        else ""
    )
    return {
        "target_id": target_id,
        "category": category,
        "length_bucket": length_bucket,
        "mode": mode,
        "role": "trained" if mode == "calibration_holdout" else "baseline",
        "status": "success",
        "error": "",
        "prompt_tokens": int(payload["prompt_tokens"]),
        "calibration_bank_sha256": bank_sha256,
        "logical_byte_saving_fraction": float(
            payload["sidecar"]["logical_byte_saving_fraction"]
        ),
        "missing_tokens": int(error_metrics["num_missing_compact_tokens"]),
        "missing_mse": float(error_metrics["mse_missing_compact_tokens"]),
        "missing_mae": float(error_metrics["mae_missing_compact_tokens"]),
        "missing_max_abs": float(error_metrics["max_abs_missing_compact_tokens"]),
        "max_abs_logit_diff": float(logit_compare["max_abs_logit_diff"]),
        "mean_abs_logit_diff": float(logit_compare["mean_abs_logit_diff"]),
        "argmax_match": bool(logit_compare["argmax_match"]),
        "original_argmax": int(logit_compare["original_argmax"]),
        "restored_argmax": int(logit_compare["restored_argmax"]),
        "artifact": str(path.parent),
    }


def write_outputs(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        rows, key=lambda row: (str(row["target_id"]), str(row["mode"]))
    )
    with (output_dir / "multi_target_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in ROW_FIELDS}
            for row in ordered_rows
        )
    (output_dir / "multi_target_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = summary["gate"]
    lines = [
        "# Light Doc Cache Multi-Target Gate",
        "",
        "Boundary: default-off restored-sidecar next-token comparison; no physical KV allocation or attention hot-path change.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Paired targets: `{gate['paired_targets']}/8`",
        f"- Holdout wins: `{gate['holdout_win_count']}`",
        f"- Aggregate relative improvement: `{gate['aggregate_relative_improvement']:.2%}`",
        f"- Worst relative regression: `{gate['worst_relative_regression']}`",
        "",
        "## Conditions",
        "",
    ]
    for name, passed in gate["conditions"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend(
        [
            "",
            "## Per-Target Rows",
            "",
            "| Target | Mode | Status | Tokens | Mean Logit Diff | Argmax Match |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in ordered_rows:
        mean_diff = (
            f"{float(row['mean_abs_logit_diff']):.6g}"
            if row["status"] == "success"
            else "-"
        )
        lines.append(
            f"| `{row['target_id']}` | `{row['mode']}` | {row['status']} | "
            f"{row.get('prompt_tokens', '-')} | {mean_diff} | "
            f"{row.get('argmax_match', '-')} |"
        )
    (output_dir / "multi_target_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="target_id:category:length_bucket:mode:path-to-summary.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    rows = []
    for spec in args.summary:
        target_id, category, length_bucket, mode, path_text = spec.split(":", 4)
        rows.append(
            normalize_summary_row(
                target_id=target_id,
                category=category,
                length_bucket=length_bucket,
                mode=mode,
                summary_path=path_text,
            )
        )
    summary = aggregate_rows(rows)
    write_outputs(args.output_dir, rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
