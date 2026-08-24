#!/usr/bin/env python3
"""Producer gate for medium-context split-K exact-burst evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tools.profile_exact_burst_medium_split_k import (
    CONTEXT_LENGTHS,
    GENERATED_TOKENS,
    POLICIES,
    REPETITIONS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    expected_flash_attn_num_splits,
    read_float32_sidecar,
    sha256_file,
    validate_case_row,
    validate_correctness_rows,
)


GO_EXACT_BURST_MEDIUM_SPLIT_K = (
    "GO_EXACT_BURST_MEDIUM_SPLIT_K"
)
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_GRAPH_SELECTION = "NO_GO_GRAPH_SELECTION"
NO_GO_LIFECYCLE = "NO_GO_LIFECYCLE"
NO_GO_MEMORY = "NO_GO_MEMORY"
NO_GO_CAPTURE_COST = "NO_GO_CAPTURE_COST"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"

TARGET_CONTEXTS = (1537, 2049, 2561, 3073, 3585, 4090)
CONTROL_CONTEXTS = (1025, 6145)
LOGIT_MAX_ABS_LIMIT = 0.25
LOGIT_MEAN_ABS_LIMIT = 0.05
TARGET_TPOT_IMPROVEMENT_MINIMUM = 0.01
TARGET_BUCKET_REGRESSION_LIMIT = 0.02
CONTROL_REGRESSION_LIMIT = 0.01
END_TO_END_REGRESSION_LIMIT = 0.02
ADDED_RETAINED_STATIC_BYTES_LIMIT = 8 * 1024**2
ADDED_RESERVED_BYTES_LIMIT = 64 * 1024**2
ADDED_CAPTURE_DURATION_NS_LIMIT = 5_000_000_000

COMPARISON_SCHEMA_VERSION = (
    "exact-burst-medium-split-k.comparison.v1"
)
GATE_SCHEMA_VERSION = "exact-burst-medium-split-k.gate.v1"
MANIFEST_SCHEMA_VERSION = (
    "exact-burst-medium-split-k.manifest.v1"
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _latency_regression(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline latency must be positive")
    return float(candidate) / float(baseline) - 1.0


def _throughput_regression(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        raise ValueError("baseline throughput must be positive")
    return (float(baseline) - float(candidate)) / float(baseline)


def _row_tpot(row: dict, percentile: float) -> float:
    values = row["amortized_tpot_samples_ns"]
    if percentile == 0.5:
        return float(statistics.median(values))
    return _nearest_rank(values, percentile)


def _expected_performance_identities() -> set[tuple[int, int, str]]:
    return {
        (repetition, context_length, policy)
        for repetition in range(REPETITIONS)
        for context_length in CONTEXT_LENGTHS
        for policy in POLICIES
    }


def _validate_inventory(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> None:
    identities = [
        (
            row.get("repetition"),
            row.get("context_length"),
            row.get("policy"),
        )
        for row in performance_rows
        if isinstance(row, dict)
    ]
    if (
        len(identities) != len(performance_rows)
        or len(identities) != len(set(identities))
        or set(identities) != _expected_performance_identities()
    ):
        raise ValueError("performance row inventory is incomplete")
    correctness_identities = [
        (
            row.get("context_length"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        for row in correctness_rows
        if isinstance(row, dict)
    ]
    expected_correctness = {
        (context_length, policy, point)
        for context_length in CONTEXT_LENGTHS
        for policy in POLICIES
        for point in SAMPLING_POINTS
    }
    if (
        len(correctness_identities) != len(correctness_rows)
        or len(correctness_identities)
        != len(set(correctness_identities))
        or set(correctness_identities) != expected_correctness
    ):
        raise ValueError("correctness row inventory is incomplete")


def _graph_selection(rows: list[dict]) -> dict:
    mismatches = []
    for row in rows:
        expected = expected_flash_attn_num_splits(
            policy=row["policy"],
            context_length=row["context_length"],
        )
        if row.get("replay_flash_attn_num_splits") != expected:
            mismatches.append({
                "repetition": row.get("repetition"),
                "context_length": row.get("context_length"),
                "policy": row.get("policy"),
                "expected": expected,
                "actual": row.get(
                    "replay_flash_attn_num_splits"
                ),
            })
    return {
        "all_exact": not mismatches,
        "mismatches": mismatches,
    }


def _paired(rows: list[dict]) -> list[tuple[dict, dict]]:
    by_identity = {
        (
            row["repetition"],
            row["context_length"],
            row["policy"],
        ): row
        for row in rows
    }
    return [
        (
            by_identity[(repetition, context_length, "auto")],
            by_identity[(repetition, context_length, "split12")],
        )
        for repetition in range(REPETITIONS)
        for context_length in CONTEXT_LENGTHS
    ]


def _performance_metrics(rows: list[dict]) -> dict:
    pairs = _paired(rows)
    per_context = {}
    for context_length in CONTEXT_LENGTHS:
        context_pairs = [
            pair
            for pair in pairs
            if pair[0]["context_length"] == context_length
        ]
        median_regressions = [
            _latency_regression(
                _row_tpot(left, 0.5),
                _row_tpot(right, 0.5),
            )
            for left, right in context_pairs
        ]
        p95_regressions = [
            _latency_regression(
                _row_tpot(left, 0.95),
                _row_tpot(right, 0.95),
            )
            for left, right in context_pairs
        ]
        per_context[str(context_length)] = {
            "sample_count": len(context_pairs),
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
                for left, right in context_pairs
            ),
            "e2e_regression_fraction": statistics.median(
                _latency_regression(
                    float(left["e2e_ns"]),
                    float(right["e2e_ns"]),
                )
                for left, right in context_pairs
            ),
            "throughput_regression_fraction": statistics.median(
                _throughput_regression(
                    float(left["output_tokens_per_second"]),
                    float(right["output_tokens_per_second"]),
                )
                for left, right in context_pairs
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


def _correctness_metrics(
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
    global_max = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_ids = True
    pairs = []
    for context_length in CONTEXT_LENGTHS:
        for point in SAMPLING_POINTS:
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
            ids_exact = (
                left["output_token_ids"]
                == right["output_token_ids"]
            )
            argmax_exact = left_argmax == right_argmax
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
            and global_max <= LOGIT_MAX_ABS_LIMIT
            and global_mean <= LOGIT_MEAN_ABS_LIMIT
        ),
    }


def _cost_metrics(rows: list[dict]) -> dict:
    auto_rows = [row for row in rows if row["policy"] == "auto"]
    candidate_rows = [
        row for row in rows if row["policy"] == "split12"
    ]

    def maximum(subset: list[dict], field: str) -> int:
        return max(int(row[field]) for row in subset)

    auto_retained = maximum(
        auto_rows,
        "capture_retained_static_bytes",
    )
    candidate_retained = maximum(
        candidate_rows,
        "capture_retained_static_bytes",
    )
    auto_reserved = maximum(
        auto_rows,
        "capture_reserved_delta_bytes",
    )
    candidate_reserved = maximum(
        candidate_rows,
        "capture_reserved_delta_bytes",
    )
    auto_duration = maximum(auto_rows, "capture_duration_ns")
    candidate_duration = maximum(
        candidate_rows,
        "capture_duration_ns",
    )
    auto_scratch = maximum(auto_rows, "reserved_scratch_blocks")
    candidate_scratch = maximum(
        candidate_rows,
        "reserved_scratch_blocks",
    )
    return {
        "added_retained_static_bytes": max(
            0,
            candidate_retained - auto_retained,
        ),
        "added_reserved_bytes": max(
            0,
            candidate_reserved - auto_reserved,
        ),
        "added_capture_duration_ns": max(
            0,
            candidate_duration - auto_duration,
        ),
        "added_scratch_blocks": max(
            0,
            candidate_scratch - auto_scratch,
        ),
        "candidate_peak_allocated_bytes": max(
            int(row["cuda_peak_allocated_bytes"])
            for row in candidate_rows
        ),
        "auto_peak_allocated_bytes": max(
            int(row["cuda_peak_allocated_bytes"])
            for row in auto_rows
        ),
        "candidate_peak_reserved_bytes": max(
            int(row["cuda_peak_reserved_bytes"])
            for row in candidate_rows
        ),
        "auto_peak_reserved_bytes": max(
            int(row["cuda_peak_reserved_bytes"])
            for row in auto_rows
        ),
    }


def _performance_passed(metrics: dict) -> bool:
    if (
        metrics["target"]["tpot_median_improvement_fraction"]
        < TARGET_TPOT_IMPROVEMENT_MINIMUM
    ):
        return False
    for context_length in TARGET_CONTEXTS:
        row = metrics["per_context"][str(context_length)]
        if (
            row["tpot_median_regression_p95_fraction"]
            > TARGET_BUCKET_REGRESSION_LIMIT
            or row["tpot_p95_regression_p95_fraction"]
            > TARGET_BUCKET_REGRESSION_LIMIT
        ):
            return False
    for context_length in CONTROL_CONTEXTS:
        row = metrics["per_context"][str(context_length)]
        if (
            row["tpot_median_regression_p95_fraction"]
            > CONTROL_REGRESSION_LIMIT
            or row["tpot_p95_regression_p95_fraction"]
            > CONTROL_REGRESSION_LIMIT
        ):
            return False
    return all(
        row[field] <= END_TO_END_REGRESSION_LIMIT
        for row in metrics["per_context"].values()
        for field in (
            "ttft_regression_fraction",
            "e2e_regression_fraction",
            "throughput_regression_fraction",
        )
    )


def _classification(
    *,
    correctness: dict,
    graph_selection: dict,
    performance: dict,
    cost: dict,
) -> str:
    if not correctness["passed"]:
        return NO_GO_CORRECTNESS
    if not graph_selection["all_exact"]:
        return NO_GO_GRAPH_SELECTION
    if not _performance_passed(performance):
        return NO_GO_PERFORMANCE
    if (
        cost["added_scratch_blocks"] != 0
        or cost["added_retained_static_bytes"]
        > ADDED_RETAINED_STATIC_BYTES_LIMIT
        or cost["added_reserved_bytes"]
        > ADDED_RESERVED_BYTES_LIMIT
    ):
        return NO_GO_MEMORY
    if (
        cost["added_capture_duration_ns"]
        > ADDED_CAPTURE_DURATION_NS_LIMIT
    ):
        return NO_GO_CAPTURE_COST
    return GO_EXACT_BURST_MEDIUM_SPLIT_K


def _failure_classification(error: Exception) -> str:
    message = str(error)
    if "selected split-K" in message or "graph identity" in message:
        return NO_GO_GRAPH_SELECTION
    if "capture cost mismatch: capture_retained" in message:
        return NO_GO_MEMORY
    if "capture cost mismatch: capture_reserved" in message:
        return NO_GO_MEMORY
    if "capture cost mismatch: reserved_scratch" in message:
        return NO_GO_MEMORY
    if "capture cost mismatch: capture_duration" in message:
        return NO_GO_CAPTURE_COST
    if (
        "lifecycle" in message
        or "pending" in message
        or "quarantine" in message
    ):
        return NO_GO_LIFECYCLE
    if "correctness" in message or "logits" in message:
        return NO_GO_CORRECTNESS
    return NO_GO_EVIDENCE_INCOMPLETE


def _validate_manifests(
    run_dir: Path,
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> tuple[dict, dict]:
    workload = _read_json(run_dir / "workload_manifest.json")
    source = _read_json(run_dir / "source_manifest.json")
    if workload.get("performance_row_count") != len(
        performance_rows
    ) or workload.get("correctness_row_count") != len(
        correctness_rows
    ):
        raise ValueError("manifest row inventory mismatch")
    run_tags = {
        row.get("run_tag")
        for row in performance_rows + correctness_rows
    }
    commits = {
        row.get("source_commit")
        for row in performance_rows + correctness_rows
    }
    if (
        run_tags != {workload.get("run_tag")}
        or commits != {workload.get("source_commit")}
        or source.get("run_tag") != workload.get("run_tag")
        or source.get("source_commit")
        != workload.get("source_commit")
    ):
        raise ValueError("source identity mismatch")
    source_hashes = source.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(SOURCE_FILES)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            for value in source_hashes.values()
        )
    ):
        raise ValueError("source manifest inventory mismatch")
    return workload, source


def classify(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    try:
        performance_rows = _read_jsonl(
            run_dir / "performance_rows.jsonl"
        )
        correctness_rows = _read_jsonl(
            run_dir / "correctness_rows.jsonl"
        )
        _validate_inventory(performance_rows, correctness_rows)
        workload, source = _validate_manifests(
            run_dir,
            performance_rows,
            correctness_rows,
        )
        graph_selection = _graph_selection(performance_rows)
        if not graph_selection["all_exact"]:
            return {
                "schema_version": GATE_SCHEMA_VERSION,
                "classification": NO_GO_GRAPH_SELECTION,
                "graph_selection": graph_selection,
            }
        validated_performance = [
            validate_case_row(row) for row in performance_rows
        ]
        validated_correctness = validate_correctness_rows(
            correctness_rows,
            run_dir=run_dir,
        )
        performance = _performance_metrics(validated_performance)
        correctness = _correctness_metrics(
            validated_correctness,
            run_dir=run_dir,
        )
        cost = _cost_metrics(validated_performance)
        classification = _classification(
            correctness=correctness,
            graph_selection=graph_selection,
            performance=performance,
            cost=cost,
        )
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "classification": classification,
            "run_tag": workload["run_tag"],
            "source_commit": workload["source_commit"],
            "performance": performance,
            "correctness": correctness,
            "graph_selection": graph_selection,
            "cost": cost,
            "row_counts": {
                "performance": len(validated_performance),
                "correctness": len(validated_correctness),
            },
            "source_file_count": len(
                source["source_sha256"]
            ),
        }
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "classification": _failure_classification(error),
            "error": f"{type(error).__name__}: {error}",
        }


def _report(result: dict) -> str:
    lines = [
        "# Exact Burst Medium-Context Split-K Gate",
        "",
        f"- Classification: `{result['classification']}`",
    ]
    if "performance" in result:
        improvement = result["performance"]["target"][
            "tpot_median_improvement_fraction"
        ]
        lines.extend([
            f"- Target median TPOT improvement: `{improvement:.6%}`",
            (
                "- Added retained static bytes: "
                f"`{result['cost']['added_retained_static_bytes']}`"
            ),
            (
                "- Added reserved bytes: "
                f"`{result['cost']['added_reserved_bytes']}`"
            ),
            (
                "- Added capture duration ns: "
                f"`{result['cost']['added_capture_duration_ns']}`"
            ),
            (
                "- Correctness max/mean abs: "
                f"`{result['correctness']['global_max_abs']}` / "
                f"`{result['correctness']['global_mean_abs']}`"
            ),
        ])
    if "error" in result:
        lines.append(f"- Error: `{result['error']}`")
    return "\n".join(lines) + "\n"


def produce_gate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    result = classify(run_dir)
    comparison = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "performance": result.get("performance"),
        "correctness": result.get("correctness"),
        "graph_selection": result.get("graph_selection"),
        "cost": result.get("cost"),
    }
    _write_json(run_dir / "comparison.json", comparison)
    _write_json(run_dir / "summary.json", result)
    (run_dir / "report.md").write_text(
        _report(result),
        encoding="utf-8",
    )
    artifact_names = tuple(
        sorted(
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.name != "manifest.json"
        )
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "classification": result["classification"],
        "artifacts": {
            name: sha256_file(run_dir / name)
            for name in artifact_names
        },
        "source_sha256": (
            _read_json(run_dir / "source_manifest.json").get(
                "source_sha256",
                {},
            )
            if (run_dir / "source_manifest.json").exists()
            else {}
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = produce_gate(Path(args.artifact_dir))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return (
        0
        if result["classification"]
        == GO_EXACT_BURST_MEDIUM_SPLIT_K
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
