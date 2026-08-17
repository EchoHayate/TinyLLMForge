"""Build a read-path recovery comparison table from smoke artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Matrix entry in mode:role:path-to-tinyllm_sidecar_read_path_summary.json form.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    rows = [_load_row(spec) for spec in args.artifact]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "read_path_recovery_matrix.csv", rows)
    _write_markdown(output_dir / "read_path_recovery_matrix.md", rows)
    return 0


def _load_row(spec: str) -> dict[str, Any]:
    mode, role, path_text = spec.split(":", 2)
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    error_metrics = payload["sidecar"]["error_metrics"]
    logit_compare = payload["logit_compare"]
    return {
        "mode": mode,
        "role": role,
        "prompt_tokens": int(payload["prompt_tokens"]),
        "logical_byte_saving_fraction": float(payload["sidecar"]["logical_byte_saving_fraction"]),
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "role",
        "prompt_tokens",
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    prompt_tokens = rows[0]["prompt_tokens"] if rows else 0
    logical_saving = rows[0]["logical_byte_saving_fraction"] if rows else 0.0
    missing_tokens = rows[0]["missing_tokens"] if rows else 0
    original_argmax = rows[0]["original_argmax"] if rows else 0
    non_oracle_matches = [
        row for row in rows if row["role"] != "upper_bound" and row["argmax_match"]
    ]
    lines = [
        "# TinyLLM Read-Path Recovery Matrix",
        "",
        "Boundary: default-off restored-sidecar read-path comparison; no attention hot-path or KV allocation lifetime change.",
        "",
        "| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match | Restored Argmax |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | {row['role']} | {row['missing_mse']:.6g} | "
            f"{row['missing_max_abs']:.6g} | {row['max_abs_logit_diff']:.6g} | "
            f"{row['mean_abs_logit_diff']:.6g} | {row['argmax_match']} | {row['restored_argmax']} |"
        )
    lines.extend(
        [
            "",
            "Common setup:",
            "",
            f"- Prompt tokens: `{prompt_tokens}`.",
            f"- Logical byte saving fraction: `{logical_saving:.2%}`.",
            f"- Missing compact tokens: `{missing_tokens}`.",
            f"- Original argmax: `{original_argmax}`.",
            "",
            "Interpretation:",
            "",
        ]
    )
    if not non_oracle_matches:
        lines.append("- No non-oracle mode preserves argmax on this prompt.")
    else:
        matched_modes = ", ".join(f"`{row['mode']}`" for row in non_oracle_matches)
        lines.append(f"- Non-oracle argmax-preserving modes: {matched_modes}.")
    oracle_rows = [row for row in rows if row["role"] == "upper_bound"]
    if oracle_rows and oracle_rows[0]["argmax_match"] and oracle_rows[0]["max_abs_logit_diff"] == 0.0:
        lines.append("- Oracle is exact, so layout, restore indexing, and temporary read-path pointer swap are correct.")
    best_mean = _best_non_oracle(rows, "mean_abs_logit_diff")
    best_mse = _best_non_oracle(rows, "missing_mse")
    if best_mean is not None:
        lines.append(f"- `{best_mean['mode']}` has the best non-oracle mean logit diff.")
    if best_mse is not None:
        lines.append(f"- `{best_mse['mode']}` has the best non-oracle missing-token MSE.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_non_oracle(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["role"] != "upper_bound"]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


if __name__ == "__main__":
    raise SystemExit(main())
