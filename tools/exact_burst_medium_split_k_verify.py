#!/usr/bin/env python3
"""Independent verifier for medium-context split-K gate artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tools.profile_exact_burst_medium_split_k import (
    CONTEXT_LENGTHS,
    POLICIES,
    REPETITIONS,
    SOURCE_FILES,
    expected_flash_attn_num_splits,
    read_float32_sidecar,
    sha256_file,
    validate_case_row,
    validate_correctness_rows,
)


GO_CLASSIFICATION = "GO_EXACT_BURST_MEDIUM_SPLIT_K"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_GRAPH_SELECTION = "NO_GO_GRAPH_SELECTION"
NO_GO_MEMORY = "NO_GO_MEMORY"
NO_GO_CAPTURE_COST = "NO_GO_CAPTURE_COST"

TARGET_CONTEXTS = (1537, 2049, 2561, 3073, 3585, 4090)
CONTROL_CONTEXTS = (1025, 6145)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _latency_regression(left: float, right: float) -> float:
    if left <= 0:
        raise ValueError("baseline latency must be positive")
    return float(right) / float(left) - 1.0


def _throughput_regression(left: float, right: float) -> float:
    if left <= 0:
        raise ValueError("baseline throughput must be positive")
    return (float(left) - float(right)) / float(left)


def _row_tpot(row: dict, percentile: float) -> float:
    samples = row["amortized_tpot_samples_ns"]
    if percentile == 0.5:
        return float(statistics.median(samples))
    return _nearest_rank(samples, percentile)


def _verify_manifest(run_dir: Path) -> dict:
    manifest = _read_json(run_dir / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest artifact inventory is invalid")
    for relative, expected in artifacts.items():
        path = run_dir / relative
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or len(expected) != 64
            or sha256_file(path) != expected
        ):
            artifact_kind = (
                "sidecar manifest"
                if relative.endswith(".f32")
                else "manifest"
            )
            raise ValueError(
                f"{artifact_kind} hash mismatch: {relative}"
            )
    source_hashes = manifest.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(SOURCE_FILES)
    ):
        raise ValueError("manifest source inventory mismatch")
    repo_root = Path(__file__).resolve().parents[1]
    for relative, expected in source_hashes.items():
        if expected == "0" * 64:
            continue
        path = repo_root / relative
        if path.is_file() and sha256_file(path) != expected:
            raise ValueError(
                f"manifest source hash mismatch: {relative}"
            )
    return manifest


def _pair_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    expected = {
        (repetition, context_length, policy)
        for repetition in range(REPETITIONS)
        for context_length in CONTEXT_LENGTHS
        for policy in POLICIES
    }
    identities = [
        (
            row["repetition"],
            row["context_length"],
            row["policy"],
        )
        for row in rows
    ]
    if (
        len(identities) != len(set(identities))
        or set(identities) != expected
    ):
        raise ValueError("raw performance inventory mismatch")
    by_identity = {
        identity: row
        for identity, row in zip(identities, rows)
    }
    return [
        (
            by_identity[(repetition, context_length, "auto")],
            by_identity[(repetition, context_length, "split12")],
        )
        for repetition in range(REPETITIONS)
        for context_length in CONTEXT_LENGTHS
    ]


def _reconstruct_performance(rows: list[dict]) -> dict:
    pairs = _pair_rows(rows)
    per_context = {}
    for context_length in CONTEXT_LENGTHS:
        selected = [
            pair
            for pair in pairs
            if pair[0]["context_length"] == context_length
        ]
        median_regressions = [
            _latency_regression(
                _row_tpot(left, 0.5),
                _row_tpot(right, 0.5),
            )
            for left, right in selected
        ]
        p95_regressions = [
            _latency_regression(
                _row_tpot(left, 0.95),
                _row_tpot(right, 0.95),
            )
            for left, right in selected
        ]
        per_context[str(context_length)] = {
            "sample_count": len(selected),
            "tpot_median_regression_fraction":
                statistics.median(median_regressions),
            "tpot_median_regression_p95_fraction":
                _nearest_rank(median_regressions, 0.95),
            "tpot_p95_regression_fraction":
                statistics.median(p95_regressions),
            "tpot_p95_regression_p95_fraction":
                _nearest_rank(p95_regressions, 0.95),
            "ttft_regression_fraction": statistics.median(
                _latency_regression(
                    float(left["ttft_ns"]),
                    float(right["ttft_ns"]),
                )
                for left, right in selected
            ),
            "e2e_regression_fraction": statistics.median(
                _latency_regression(
                    float(left["e2e_ns"]),
                    float(right["e2e_ns"]),
                )
                for left, right in selected
            ),
            "throughput_regression_fraction": statistics.median(
                _throughput_regression(
                    float(left["output_tokens_per_second"]),
                    float(right["output_tokens_per_second"]),
                )
                for left, right in selected
            ),
        }
    target_regressions = [
        _latency_regression(
            _row_tpot(left, 0.5),
            _row_tpot(right, 0.5),
        )
        for left, right in pairs
        if left["context_length"] in TARGET_CONTEXTS
    ]
    return {
        "target": {
            "contexts": list(TARGET_CONTEXTS),
            "sample_count": len(target_regressions),
            "tpot_median_improvement_fraction": -statistics.median(
                target_regressions
            ),
        },
        "controls": {"contexts": list(CONTROL_CONTEXTS)},
        "per_context": per_context,
    }


def _reconstruct_graph_selection(rows: list[dict]) -> dict:
    mismatches = []
    for row in rows:
        expected = expected_flash_attn_num_splits(
            policy=row["policy"],
            context_length=row["context_length"],
        )
        if row["replay_flash_attn_num_splits"] != expected:
            mismatches.append({
                "repetition": row["repetition"],
                "context_length": row["context_length"],
                "policy": row["policy"],
                "expected": expected,
                "actual": row["replay_flash_attn_num_splits"],
            })
    return {"all_exact": not mismatches, "mismatches": mismatches}


def _reconstruct_correctness(
    rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    by_identity = {
        (
            row["context_length"],
            row["policy"],
            row["sampling_point"],
        ): row
        for row in rows
    }
    pairs = []
    global_max = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_ids = True
    for context_length in CONTEXT_LENGTHS:
        for point in (
            "prefill-final",
            "decode-first",
            "decode-middle",
            "decode-final",
        ):
            left = by_identity[(context_length, "auto", point)]
            right = by_identity[(context_length, "split12", point)]
            left_values = read_float32_sidecar(
                run_dir,
                path=left["logits_path"],
                expected_element_count=left[
                    "logits_element_count"
                ],
                expected_byte_length=left["logits_byte_length"],
                expected_sha256=left["logits_sha256"],
            )
            right_values = read_float32_sidecar(
                run_dir,
                path=right["logits_path"],
                expected_element_count=right[
                    "logits_element_count"
                ],
                expected_byte_length=right[
                    "logits_byte_length"
                ],
                expected_sha256=right["logits_sha256"],
            )
            differences = [
                abs(float(a) - float(b))
                for a, b in zip(left_values, right_values)
            ]
            maximum = max(differences, default=0.0)
            mean = (
                sum(differences) / len(differences)
                if differences
                else 0.0
            )
            left_argmax = max(
                range(len(left_values)),
                key=left_values.__getitem__,
            )
            right_argmax = max(
                range(len(right_values)),
                key=right_values.__getitem__,
            )
            argmax_exact = left_argmax == right_argmax
            ids_exact = (
                left["output_token_ids"]
                == right["output_token_ids"]
            )
            global_max = max(global_max, maximum)
            total_abs += sum(differences)
            total_count += len(differences)
            all_argmax = all_argmax and argmax_exact
            all_ids = all_ids and ids_exact
            pairs.append({
                "context_length": context_length,
                "sampling_point": point,
                "max_abs": maximum,
                "mean_abs": mean,
                "argmax_exact": argmax_exact,
                "token_ids_exact": ids_exact,
            })
    global_mean = total_abs / total_count if total_count else 0.0
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "global_max_abs": global_max,
        "global_mean_abs": global_mean,
        "all_argmax_exact": all_argmax,
        "all_token_ids_exact": all_ids,
        "passed": (
            all_ids
            and all_argmax
            and global_max <= 0.25
            and global_mean <= 0.05
        ),
    }


def _reconstruct_cost(rows: list[dict]) -> dict:
    auto = [row for row in rows if row["policy"] == "auto"]
    candidate = [row for row in rows if row["policy"] == "split12"]

    def maximum(values: list[dict], field: str) -> int:
        return max(int(row[field]) for row in values)

    return {
        "added_retained_static_bytes": max(
            0,
            maximum(candidate, "capture_retained_static_bytes")
            - maximum(auto, "capture_retained_static_bytes"),
        ),
        "added_reserved_bytes": max(
            0,
            maximum(candidate, "capture_reserved_delta_bytes")
            - maximum(auto, "capture_reserved_delta_bytes"),
        ),
        "added_capture_duration_ns": max(
            0,
            maximum(candidate, "capture_duration_ns")
            - maximum(auto, "capture_duration_ns"),
        ),
        "added_scratch_blocks": max(
            0,
            maximum(candidate, "reserved_scratch_blocks")
            - maximum(auto, "reserved_scratch_blocks"),
        ),
        "candidate_peak_allocated_bytes": maximum(
            candidate,
            "cuda_peak_allocated_bytes",
        ),
        "auto_peak_allocated_bytes": maximum(
            auto,
            "cuda_peak_allocated_bytes",
        ),
        "candidate_peak_reserved_bytes": maximum(
            candidate,
            "cuda_peak_reserved_bytes",
        ),
        "auto_peak_reserved_bytes": maximum(
            auto,
            "cuda_peak_reserved_bytes",
        ),
    }


def _classify(
    performance: dict,
    correctness: dict,
    graph_selection: dict,
    cost: dict,
) -> str:
    if not correctness["passed"]:
        return NO_GO_CORRECTNESS
    if not graph_selection["all_exact"]:
        return NO_GO_GRAPH_SELECTION
    if (
        performance["target"]["tpot_median_improvement_fraction"]
        < 0.01
    ):
        return NO_GO_PERFORMANCE
    for context_length in TARGET_CONTEXTS:
        row = performance["per_context"][str(context_length)]
        if (
            row["tpot_median_regression_p95_fraction"] > 0.02
            or row["tpot_p95_regression_p95_fraction"] > 0.02
        ):
            return NO_GO_PERFORMANCE
    for context_length in CONTROL_CONTEXTS:
        row = performance["per_context"][str(context_length)]
        if (
            row["tpot_median_regression_p95_fraction"] > 0.01
            or row["tpot_p95_regression_p95_fraction"] > 0.01
        ):
            return NO_GO_PERFORMANCE
    if any(
        row[field] > 0.02
        for row in performance["per_context"].values()
        for field in (
            "ttft_regression_fraction",
            "e2e_regression_fraction",
            "throughput_regression_fraction",
        )
    ):
        return NO_GO_PERFORMANCE
    if (
        cost["added_scratch_blocks"] != 0
        or cost["added_retained_static_bytes"] > 8 * 1024**2
        or cost["added_reserved_bytes"] > 64 * 1024**2
    ):
        return NO_GO_MEMORY
    if cost["added_capture_duration_ns"] > 5_000_000_000:
        return NO_GO_CAPTURE_COST
    return GO_CLASSIFICATION


def verify_artifact_directory(path: Path) -> dict:
    run_dir = Path(path)
    _verify_manifest(run_dir)
    performance_rows = [
        validate_case_row(row)
        for row in _read_jsonl(
            run_dir / "performance_rows.jsonl"
        )
    ]
    correctness_rows = validate_correctness_rows(
        _read_jsonl(run_dir / "correctness_rows.jsonl"),
        run_dir=run_dir,
    )
    performance = _reconstruct_performance(performance_rows)
    correctness = _reconstruct_correctness(
        correctness_rows,
        run_dir=run_dir,
    )
    graph_selection = _reconstruct_graph_selection(
        performance_rows
    )
    cost = _reconstruct_cost(performance_rows)
    classification = _classify(
        performance,
        correctness,
        graph_selection,
        cost,
    )
    summary = _read_json(run_dir / "summary.json")
    comparison = _read_json(run_dir / "comparison.json")
    if summary.get("classification") != classification:
        raise ValueError(
            "reconstructed classification does not match summary"
        )
    expected_comparison = {
        "schema_version":
            "exact-burst-medium-split-k.comparison.v1",
        "performance": performance,
        "correctness": correctness,
        "graph_selection": graph_selection,
        "cost": cost,
    }
    if comparison != expected_comparison:
        raise ValueError(
            "reconstructed metrics do not match comparison"
        )
    return {
        "verified": True,
        "classification": classification,
        "manifest_verified": True,
        "raw_metrics_reconstructed": True,
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--receipt-path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipt = verify_artifact_directory(
        Path(args.artifact_dir)
    )
    payload = json.dumps(
        receipt,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if args.receipt_path:
        Path(args.receipt_path).write_text(
            payload,
            encoding="utf-8",
        )
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
