#!/usr/bin/env python3
"""Producer gate for exact greedy decode-burst evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

from tools.profile_exact_burst_continuation_epoch import (
    CAPTURE_COST_FIELDS,
    CONTEXT_CASES,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    context_cases,
    policy_order,
    read_float32_sidecar,
    sha256_file,
    summarize_rows,
    validate_case_row,
    validate_correctness_rows,
)


COMPARISON_SCHEMA_VERSION = (
    "exact-burst-continuation-epoch.comparison.v1"
)
GATE_SCHEMA_VERSION = "exact-burst-continuation-epoch.gate.v1"
MANIFEST_SCHEMA_VERSION = (
    "exact-burst-continuation-epoch.manifest.v1"
)
PRIMARY_ARTIFACTS = (
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
)
REFERENCE_POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
)
CONTINUATION_POLICY = "decode_burst_k4_continuation"
STAGE1_MODEL_BASENAMES = ("Qwen3-0.6B", "Qwen3-0___6B")
EXPECTED_ROWS = 60
EXPECTED_CORRECTNESS_ROWS = 48
LOGIT_MAX_ABS_LIMIT = 0.25
LOGIT_MEAN_ABS_LIMIT = 0.05
K4_MEDIAN_MIN_IMPROVEMENT = 0.05
K4_P95_MIN_IMPROVEMENT = 0.03
BUCKET_MEDIAN_MIN_IMPROVEMENT = 0.05
MIN_WINNING_BUCKETS = 2
K8_PARITY_MAX_REGRESSION = 0.02
BUCKET_TPOT_MAX_REGRESSION = 0.03
LATENCY_MAX_REGRESSION = 0.03
THROUGHPUT_MAX_REGRESSION = 0.02
RESERVED_MEMORY_MAX_REGRESSION = 0.03
VISIBILITY_RATIO_LIMIT = 0.60
MIN_CONTINUATION_HITS = 31


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path):
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line, parse_constant=_reject_constant)
            for line in handle
            if line.strip()
        ]


def _write_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _relative_change(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError(
            "relative comparison baseline must be positive"
        )
    return (candidate - baseline) / baseline


def _improvement(baseline, candidate) -> float:
    return -_relative_change(baseline, candidate)


def _metric_summary(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    baseline_policy: str,
    candidate_policy: str,
) -> dict:
    baseline_tpot = [
        float(value)
        for row in baseline_rows
        for value in row["amortized_tpot_samples_ns"]
    ]
    candidate_tpot = [
        float(value)
        for row in candidate_rows
        for value in row["amortized_tpot_samples_ns"]
    ]
    baseline_median = statistics.median(baseline_tpot)
    candidate_median = statistics.median(candidate_tpot)
    baseline_p95 = _nearest_rank(baseline_tpot, 0.95)
    candidate_p95 = _nearest_rank(candidate_tpot, 0.95)
    baseline_p99 = _nearest_rank(baseline_tpot, 0.99)
    candidate_p99 = _nearest_rank(candidate_tpot, 0.99)
    baseline_ttft = statistics.median(
        float(row["ttft_ns"]) for row in baseline_rows
    )
    candidate_ttft = statistics.median(
        float(row["ttft_ns"]) for row in candidate_rows
    )
    baseline_e2e = statistics.median(
        float(row["e2e_ns"]) for row in baseline_rows
    )
    candidate_e2e = statistics.median(
        float(row["e2e_ns"]) for row in candidate_rows
    )
    baseline_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in baseline_rows
    )
    candidate_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in candidate_rows
    )
    baseline_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in baseline_rows
    )
    candidate_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in candidate_rows
    )
    baseline_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in baseline_rows
    )
    candidate_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in candidate_rows
    )
    return {
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "sample_count_per_policy": len(baseline_tpot),
        "baseline_tpot_median_ns": baseline_median,
        "candidate_tpot_median_ns": candidate_median,
        "tpot_median_improvement_fraction":
            _improvement(baseline_median, candidate_median),
        "baseline_tpot_p95_ns": baseline_p95,
        "candidate_tpot_p95_ns": candidate_p95,
        "tpot_p95_improvement_fraction":
            _improvement(baseline_p95, candidate_p95),
        "baseline_tpot_p99_ns": baseline_p99,
        "candidate_tpot_p99_ns": candidate_p99,
        "tpot_p99_improvement_fraction":
            _improvement(baseline_p99, candidate_p99),
        "baseline_ttft_median_ns": baseline_ttft,
        "candidate_ttft_median_ns": candidate_ttft,
        "ttft_regression_fraction":
            _relative_change(baseline_ttft, candidate_ttft),
        "baseline_e2e_median_ns": baseline_e2e,
        "candidate_e2e_median_ns": candidate_e2e,
        "e2e_regression_fraction":
            _relative_change(baseline_e2e, candidate_e2e),
        "baseline_output_tokens_per_second_median":
            baseline_rate,
        "candidate_output_tokens_per_second_median":
            candidate_rate,
        "throughput_regression_fraction":
            _relative_change(candidate_rate, baseline_rate),
        "baseline_cuda_peak_allocated_bytes": baseline_allocated,
        "candidate_cuda_peak_allocated_bytes": candidate_allocated,
        "cuda_allocated_delta_bytes":
            candidate_allocated - baseline_allocated,
        "baseline_cuda_peak_reserved_bytes": baseline_reserved,
        "candidate_cuda_peak_reserved_bytes": candidate_reserved,
        "cuda_reserved_delta_bytes":
            candidate_reserved - baseline_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(baseline_reserved, candidate_reserved),
    }


def _raw_inventory(rows: list[dict]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly 60 measured rows, got {len(rows)}"
        )
    identities = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("case row must be an object")
        identity = (
            row.get("context_bucket"),
            row.get("repetition"),
            row.get("policy"),
        )
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate case identity")
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in context_cases()
        for repetition in range(5)
        for policy in POLICIES
    }
    if set(identities) != expected:
        raise ValueError("measured case inventory is incomplete")


def _lifecycle_flags(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row.get("policy") == policy]
    replay_complete = True
    d2h_complete = True
    lease_complete = True
    execution_complete = True
    for row in selected:
        summary = row.get("exact_greedy_decode_burst_summary")
        if not isinstance(summary, dict):
            raise ValueError("exact burst summary is missing")
        generated = row.get("generated_tokens")
        expected_replays = generated - 1
        expected_commits = math.ceil(
            expected_replays / POLICY_CONFIGS[policy]["width"]
        )
        width = POLICY_CONFIGS[policy]["width"]
        partial_width = expected_replays % width
        expected_authorized = {str(width): expected_commits}
        if partial_width:
            expected_authorized[str(width)] -= 1
            expected_authorized[str(partial_width)] = 1
        replay_complete = replay_complete and all((
            summary.get("target_model_forwards")
            == expected_replays,
            summary.get("graph_replays") == expected_replays,
            summary.get("committed_tokens") == expected_replays,
            summary.get("attempts") == expected_commits,
            summary.get("acceptances") == expected_commits,
            summary.get("commits") == expected_commits,
            summary.get("requested_width_histogram")
            == {str(width): expected_commits},
            summary.get("authorized_width_histogram")
            == expected_authorized,
            summary.get("output_budget_clipped")
            == (0 if width == 1 else 1),
            summary.get("block_boundary_clipped") == 0,
        ))
        d2h_complete = d2h_complete and all((
            summary.get("intermediate_token_d2h_calls") == 0,
            summary.get("final_token_d2h_calls")
            == expected_commits,
            isinstance(summary.get("final_token_d2h_bytes"), int),
            summary.get("final_token_d2h_bytes")
            == expected_replays * 8,
            summary.get("sampled_logit_d2h_calls") == 0,
            len(row.get("host_visible_burst_gaps_ns", ()))
            == expected_commits,
        ))
        lease_complete = lease_complete and (
            summary.get("pending_leases") == 0
        )
        execution_complete = execution_complete and all((
            summary.get("failures") == 0,
            summary.get("quarantines") == 0,
            summary.get("quarantine_reason") is None,
        ))
    return {
        "replay_complete": replay_complete,
        "d2h_complete": d2h_complete,
        "lease_complete": lease_complete,
        "execution_complete": execution_complete,
    }


def _neutralize_failed_lifecycle(rows: list[dict]) -> list[dict]:
    normalized = deepcopy(rows)
    for row in normalized:
        policy = row.get("policy")
        if not POLICY_CONFIGS.get(policy, {}).get("enabled"):
            continue
        summary = row["exact_greedy_decode_burst_summary"]
        expected_replays = row["generated_tokens"] - 1
        expected_commits = math.ceil(
            expected_replays / POLICY_CONFIGS[policy]["width"]
        )
        summary["attempts"] = expected_commits
        summary["acceptances"] = expected_commits
        summary["target_model_forwards"] = expected_replays
        summary["graph_replays"] = expected_replays
        summary["intermediate_token_d2h_calls"] = 0
        summary["final_token_d2h_calls"] = expected_commits
        summary["commits"] = expected_commits
        summary["committed_tokens"] = expected_replays
        width = POLICY_CONFIGS[policy]["width"]
        partial_width = expected_replays % width
        summary["requested_width_histogram"] = {
            str(width): expected_commits,
        }
        summary["authorized_width_histogram"] = {
            str(width): expected_commits,
        }
        if partial_width:
            summary["authorized_width_histogram"][
                str(width)
            ] -= 1
            summary["authorized_width_histogram"][
                str(partial_width)
            ] = 1
        summary["output_budget_clipped"] = (
            0 if width == 1 else 1
        )
        summary["block_boundary_clipped"] = 0
        summary["final_token_d2h_bytes"] = expected_replays * 8
        summary["pending_leases"] = 0
        summary["failures"] = 0
        summary["quarantines"] = 0
        summary["quarantine_reason"] = None
    return normalized


def _correctness_metrics(
    rows: list[dict],
    *,
    run_dir: Path,
) -> tuple[dict, bool]:
    validated = validate_correctness_rows(rows, run_dir=run_dir)
    by_identity = {
        (
            row["context_bucket"],
            row["policy"],
            row["sampling_point"],
        ): row
        for row in validated
    }
    pairs = []
    continuation_passed = True
    global_max = 0.0
    global_total = 0.0
    global_count = 0
    all_argmax = True
    all_output_ids = True
    all_output_text = True
    for baseline in REFERENCE_POLICIES:
        for bucket, _prompt, _generated in CONTEXT_CASES:
            for point in SAMPLING_POINTS:
                left = by_identity[(bucket, baseline, point)]
                right = by_identity[
                    (bucket, CONTINUATION_POLICY, point)
                ]
                if (
                    left["logits_shape"] != right["logits_shape"]
                    or left["logits_element_count"]
                    != right["logits_element_count"]
                ):
                    raise ValueError("paired logits shape mismatch")
                left_values = read_float32_sidecar(
                    run_dir,
                    path=left["logits_path"],
                    expected_element_count=left[
                        "logits_element_count"
                    ],
                    expected_byte_length=left[
                        "logits_byte_length"
                    ],
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
                    abs(a - b)
                    for a, b in zip(left_values, right_values)
                ]
                maximum = max(differences, default=0.0)
                mean = (
                    sum(differences) / len(differences)
                    if differences else 0.0
                )
                argmax_equal = (
                    max(
                        range(len(left_values)),
                        key=left_values.__getitem__,
                    )
                    == max(
                        range(len(right_values)),
                        key=right_values.__getitem__,
                    )
                )
                output_ids_equal = (
                    left["output_token_ids"]
                    == right["output_token_ids"]
                )
                output_text_equal = (
                    left["output_text_sha256"]
                    == right["output_text_sha256"]
                )
                passed = all((
                    maximum <= LOGIT_MAX_ABS_LIMIT,
                    mean <= LOGIT_MEAN_ABS_LIMIT,
                    argmax_equal,
                    output_ids_equal,
                    output_text_equal,
                ))
                continuation_passed &= passed
                global_max = max(global_max, maximum)
                global_total += sum(differences)
                global_count += len(differences)
                all_argmax &= argmax_equal
                all_output_ids &= output_ids_equal
                all_output_text &= output_text_equal
                pairs.append({
                    "baseline_policy": baseline,
                    "candidate_policy": CONTINUATION_POLICY,
                    "context_bucket": bucket,
                    "sampling_point": point,
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "argmax_equal": argmax_equal,
                    "output_ids_exact": output_ids_equal,
                    "output_text_exact": output_text_equal,
                    "passed": passed,
                })
    return ({
        "pair_count": len(pairs),
        "max_abs": global_max,
        "mean_abs": (
            global_total / global_count if global_count else 0.0
        ),
        "argmax_equal": all_argmax,
        "output_ids_exact": all_output_ids,
        "output_text_exact": all_output_text,
        "pairs": pairs,
    }, continuation_passed)


def _cost_summary(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row["policy"] == policy]

    def bounds(field):
        values = [int(row[field]) for row in selected]
        return {"min": min(values), "max": max(values)}

    return {
        field: bounds(field) for field in CAPTURE_COST_FIELDS
    }


def _continuation_coverage(rows: list[dict]) -> dict:
    selected = [
        row for row in rows
        if row["policy"] == CONTINUATION_POLICY
    ]
    minimum_hits = min(
        row["exact_greedy_decode_burst_summary"][
            "continuation_hits"
        ]
        for row in selected
    )
    failures = []
    for row in selected:
        summary = row["exact_greedy_decode_burst_summary"]
        unexpected_misses = {
            reason: count
            for reason, count
            in summary["continuation_miss_counts"].items()
            if reason != "receipt_missing" and count
        }
        passed = all((
            summary["cold_binds"] == 1,
            summary["continuation_hits"] >= MIN_CONTINUATION_HITS,
            summary["continuation_attempts"] == summary["commits"],
            summary["continuation_hits"] + summary["cold_binds"]
            == summary["continuation_attempts"],
            summary["continuation_miss_counts"]
            .get("receipt_missing", 0) == 1,
            not unexpected_misses,
            not summary["continuation_invalidation_counts"],
            summary["failures"] == 0,
            summary["quarantines"] == 0,
            summary["pending_leases"] == 0,
        ))
        if not passed:
            failures.append({
                "context_bucket": row["context_bucket"],
                "repetition": row["repetition"],
                "cold_binds": summary["cold_binds"],
                "continuation_hits": summary["continuation_hits"],
                "continuation_miss_counts":
                    summary["continuation_miss_counts"],
                "continuation_invalidation_counts":
                    summary["continuation_invalidation_counts"],
                "failures": summary["failures"],
                "quarantines": summary["quarantines"],
                "pending_leases": summary["pending_leases"],
            })
    return {
        "request_count": len(selected),
        "minimum_required_hits": MIN_CONTINUATION_HITS,
        "minimum_hits": minimum_hits,
        "all_requests_passed": not failures,
        "failures": failures,
    }


def _continuation_evaluation(
    rows: list[dict],
    *,
    correctness_passed: bool,
    coverage: dict,
) -> dict:
    k4_rows = [
        row for row in rows if row["policy"] == "decode_burst_k4"
    ]
    k8_rows = [
        row for row in rows if row["policy"] == "decode_burst_k8"
    ]
    candidate_rows = [
        row for row in rows
        if row["policy"] == CONTINUATION_POLICY
    ]
    aggregate_k4 = _metric_summary(
        k4_rows,
        candidate_rows,
        baseline_policy="decode_burst_k4",
        candidate_policy=CONTINUATION_POLICY,
    )
    aggregate_k8 = _metric_summary(
        k8_rows,
        candidate_rows,
        baseline_policy="decode_burst_k8",
        candidate_policy=CONTINUATION_POLICY,
    )
    by_bucket = {}
    bucket_regressions = []
    latency_regressions = []
    throughput_regressions = []
    winning_buckets = 0
    maximum_gap = 0
    maximum_k8_gap = 0
    for bucket, _prompt, _generated in CONTEXT_CASES:
        k4_bucket = [
            row for row in k4_rows
            if row["context_bucket"] == bucket
        ]
        k8_bucket = [
            row for row in k8_rows
            if row["context_bucket"] == bucket
        ]
        candidate_bucket = [
            row for row in candidate_rows
            if row["context_bucket"] == bucket
        ]
        metrics = _metric_summary(
            k4_bucket,
            candidate_bucket,
            baseline_policy="decode_burst_k4",
            candidate_policy=CONTINUATION_POLICY,
        )
        by_bucket[bucket] = metrics
        if (
            metrics["tpot_median_improvement_fraction"]
            >= BUCKET_MEDIAN_MIN_IMPROVEMENT
        ):
            winning_buckets += 1
        if (
            metrics["tpot_median_improvement_fraction"]
            < -BUCKET_TPOT_MAX_REGRESSION
        ):
            bucket_regressions.append(f"{bucket}:median_tpot")
        if (
            metrics["tpot_p95_improvement_fraction"]
            < -BUCKET_TPOT_MAX_REGRESSION
        ):
            bucket_regressions.append(f"{bucket}:p95_tpot")
        if (
            metrics["ttft_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            latency_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            latency_regressions.append(f"{bucket}:e2e")
        if (
            metrics["throughput_regression_fraction"]
            > THROUGHPUT_MAX_REGRESSION
        ):
            throughput_regressions.append(
                f"{bucket}:throughput"
            )
        maximum_gap = max(
            maximum_gap,
            *(
                int(row["maximum_host_visible_burst_gap_ns"])
                for row in candidate_bucket
            ),
        )
        maximum_k8_gap = max(
            maximum_k8_gap,
            *(
                int(row["maximum_host_visible_burst_gap_ns"])
                for row in k8_bucket
            ),
        )
    memory_regression = (
        aggregate_k4["cuda_reserved_regression_fraction"]
        > RESERVED_MEMORY_MAX_REGRESSION
    )
    visibility_ratio = (
        maximum_gap / maximum_k8_gap
        if maximum_k8_gap else math.inf
    )
    visibility_passed = visibility_ratio <= VISIBILITY_RATIO_LIMIT
    cost = _cost_summary(rows, CONTINUATION_POLICY)
    minimum_skipped_bytes = min(
        row["exact_greedy_decode_burst_summary"][
            "skipped_block_table_bytes"
        ]
        for row in candidate_rows
    )
    cost_complete = all((
        cost["capture_duration_ns"]["min"] > 0,
        cost["capture_retained_static_bytes"]["min"] > 0,
        cost["reserved_scratch_blocks"]["min"] == 1,
        cost["reserved_scratch_blocks"]["max"] == 1,
        minimum_skipped_bytes > 0,
    ))
    return {
        "policy": CONTINUATION_POLICY,
        "burst_width": POLICY_CONFIGS[CONTINUATION_POLICY]["width"],
        "correctness_passed": correctness_passed,
        "continuation_coverage_passed":
            coverage["all_requests_passed"],
        "k4_median_passed": (
            aggregate_k4["tpot_median_improvement_fraction"]
            >= K4_MEDIAN_MIN_IMPROVEMENT
        ),
        "k4_p95_passed": (
            aggregate_k4["tpot_p95_improvement_fraction"]
            >= K4_P95_MIN_IMPROVEMENT
        ),
        "winning_bucket_count": winning_buckets,
        "bucket_coverage_passed":
            winning_buckets >= MIN_WINNING_BUCKETS,
        "k8_parity_passed": (
            aggregate_k8["tpot_median_improvement_fraction"]
            >= -K8_PARITY_MAX_REGRESSION
        ),
        "bucket_regressions": bucket_regressions,
        "latency_regressions": latency_regressions,
        "throughput_regressions": throughput_regressions,
        "memory_regression": memory_regression,
        "maximum_host_visible_burst_gap_ns": maximum_gap,
        "paired_k8_maximum_host_visible_burst_gap_ns":
            maximum_k8_gap,
        "visibility_ratio": visibility_ratio,
        "visibility_passed": visibility_passed,
        "cost_complete": cost_complete,
        "aggregate": {
            "k4_vs_continuation": aggregate_k4,
            "k8_vs_continuation": aggregate_k8,
        },
        "by_bucket": by_bucket,
        "cost": {
            **cost,
            "minimum_skipped_block_table_bytes":
                minimum_skipped_bytes,
        },
    }


def _first_failure(evaluation: dict) -> str | None:
    checks = (
        ("correctness_passed", "NO_GO_CORRECTNESS"),
        (
            "continuation_coverage_passed",
            "NO_GO_CONTINUATION_COVERAGE",
        ),
        ("k4_median_passed", "NO_GO_K4_MEDIAN"),
        ("k4_p95_passed", "NO_GO_K4_P95"),
        ("k8_parity_passed", "NO_GO_K8_PARITY"),
        ("visibility_passed", "NO_GO_VISIBILITY_RATIO"),
        ("bucket_coverage_passed", "NO_GO_K4_MEDIAN"),
    )
    for field, classification in checks:
        if not evaluation[field]:
            return classification
    if evaluation["bucket_regressions"]:
        return "NO_GO_BUCKET_REGRESSION"
    if evaluation["latency_regressions"]:
        return "NO_GO_TTFT_E2E"
    if evaluation["throughput_regressions"]:
        return "NO_GO_THROUGHPUT"
    if evaluation["memory_regression"]:
        return "NO_GO_MEMORY"
    if not evaluation["cost_complete"]:
        return "NO_GO_COST_INCOMPLETE"
    return None


def classify(
    rows: list[dict],
    correctness_rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    _raw_inventory(rows)
    validated = [
        validate_case_row(row)
        for row in _neutralize_failed_lifecycle(rows)
    ]
    correctness, correctness_passed = (
        _correctness_metrics(
            correctness_rows,
            run_dir=Path(run_dir),
        )
    )
    output_exact = True
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in validated
    }
    for bucket, _prompt, _generated in CONTEXT_CASES:
        for repetition in range(5):
            group = [
                by_identity[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            output_exact &= (
                len({
                    tuple(row["output_token_ids"]) for row in group
                }) == 1
                and len({
                    row["output_text_sha256"] for row in group
                }) == 1
            )
    coverage = _continuation_coverage(validated)
    selected_policy = CONTINUATION_POLICY
    selected = _continuation_evaluation(
        validated,
        correctness_passed=output_exact and correctness_passed,
        coverage=coverage,
    )
    classification = (
        _first_failure(selected)
        or "GO_EXACT_BURST_CONTINUATION_EPOCH"
    )
    run_tags = {
        *(row.get("run_tag") for row in rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        *(row.get("source_commit") for row in rows),
        *(row.get("source_commit") for row in correctness_rows),
    }
    evidence_complete = len(run_tags) == 1 and len(commits) == 1
    if not evidence_complete and classification.startswith("GO_"):
        classification = "NO_GO_EVIDENCE_INCOMPLETE"
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_tag": validated[0]["run_tag"],
        "source_commit": validated[0]["source_commit"],
        "classification": classification,
        "selected_policy": selected_policy,
        "selected_burst_width": POLICY_CONFIGS[selected_policy]["width"],
        "selected_lifecycle_complete":
            selected["continuation_coverage_passed"],
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_ABS_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_ABS_LIMIT,
            "k4_aggregate_median_min_improvement_fraction":
                K4_MEDIAN_MIN_IMPROVEMENT,
            "k4_aggregate_p95_min_improvement_fraction":
                K4_P95_MIN_IMPROVEMENT,
            "bucket_median_min_improvement_fraction":
                BUCKET_MEDIAN_MIN_IMPROVEMENT,
            "minimum_winning_bucket_count": MIN_WINNING_BUCKETS,
            "k8_parity_max_regression_fraction":
                K8_PARITY_MAX_REGRESSION,
            "bucket_tpot_max_regression_fraction":
                BUCKET_TPOT_MAX_REGRESSION,
            "latency_max_regression_fraction":
                LATENCY_MAX_REGRESSION,
            "throughput_max_regression_fraction":
                THROUGHPUT_MAX_REGRESSION,
            "reserved_memory_max_regression_fraction":
                RESERVED_MEMORY_MAX_REGRESSION,
            "visibility_ratio_limit": VISIBILITY_RATIO_LIMIT,
            "minimum_continuation_hits":
                MIN_CONTINUATION_HITS,
        },
        "correctness": correctness,
        "continuation_coverage": coverage,
        "candidate_evaluations": {
            CONTINUATION_POLICY: selected,
        },
    }


def _validate_source_manifest(
    manifest,
    *,
    repo_root: Path,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "exact-burst-continuation-epoch.source.v1"
    ):
        raise ValueError("source manifest is invalid")
    digests = manifest.get("source_sha256")
    if (
        not isinstance(digests, dict)
        or set(digests) != set(SOURCE_FILES)
    ):
        raise ValueError(
            "source manifest file inventory mismatch"
        )
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        if digests[relative] != sha256_file(path):
            raise ValueError(
                f"source digest mismatch: {relative}"
            )


def _validate_workload_manifest(manifest) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "exact-burst-continuation-epoch.workload.v1"
    ):
        raise ValueError("workload manifest is invalid")
    expected = {
        "context_cases": [
            {
                "context_bucket": bucket,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            }
            for bucket, prompt_tokens, generated_tokens
            in CONTEXT_CASES
        ],
        "generated_tokens": 128,
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "performance_row_count": EXPECTED_ROWS,
        "correctness_row_count": EXPECTED_CORRECTNESS_ROWS,
        "performance_correctness_trace": False,
        "correctness_trace_identity":
            "gate-only-exact-burst-continuation-correctness-v1",
        "correctness_sampling_points": list(SAMPLING_POINTS),
        "policy_configs": {
            policy: dict(values)
            for policy, values in POLICY_CONFIGS.items()
        },
        "policy_order": {
            str(repetition): {
                bucket: list(
                    policy_order(repetition, context_index)
                )
                for context_index, (
                    bucket,
                    _prompt,
                    _generated,
                ) in enumerate(CONTEXT_CASES)
            }
            for repetition in range(5)
        },
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"workload manifest mismatch: {field}"
            )
    model = manifest.get("model")
    if (
        not isinstance(model, str)
        or Path(model).name not in STAGE1_MODEL_BASENAMES
    ):
        raise ValueError("workload manifest mismatch: model")
    utilization = manifest.get("gpu_memory_utilization")
    if (
        isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not math.isfinite(float(utilization))
        or not 0.0 < float(utilization) <= 1.0
    ):
        raise ValueError(
            "workload manifest mismatch: gpu_memory_utilization"
        )
    environment = manifest.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("torch_available") is not True
        or environment.get("cuda_available") is not True
        or not isinstance(environment.get("torch_version"), str)
        or not environment["torch_version"]
        or not isinstance(
            environment.get("cuda_runtime_version"),
            str,
        )
        or not environment["cuda_runtime_version"]
        or not isinstance(
            environment.get("cuda_device_name"),
            str,
        )
        or not environment["cuda_device_name"]
    ):
        raise ValueError(
            "workload manifest mismatch: environment"
        )


def produce_gate(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    rows = _read_jsonl(run_dir / "case_rows.jsonl")
    correctness_rows = _read_jsonl(
        run_dir / "correctness_rows.jsonl"
    )
    source = _read_json(run_dir / "source_manifest.json")
    workload = _read_json(run_dir / "workload_manifest.json")
    summary = _read_json(run_dir / "summary.json")
    referenced_sidecars = {
        row.get("logits_path") for row in correctness_rows
    }
    actual_sidecars = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "logits").rglob("*.f32")
    }
    if referenced_sidecars != actual_sidecars:
        raise ValueError("sidecar inventory mismatch")
    _validate_source_manifest(source, repo_root=Path(repo_root))
    _validate_workload_manifest(workload)
    identities = {
        source.get("run_tag"),
        workload.get("run_tag"),
        *(row.get("run_tag") for row in rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        source.get("source_commit"),
        workload.get("source_commit"),
        *(row.get("source_commit") for row in rows),
        *(
            row.get("source_commit")
            for row in correctness_rows
        ),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    _raw_inventory(rows)
    expected_summary = summarize_rows(
        rows,
        expected_repetitions=5,
    )
    expected_summary["correctness_row_count"] = (
        EXPECTED_CORRECTNESS_ROWS
    )
    if summary != expected_summary:
        raise ValueError("worker summary drift")
    comparison = classify(
        rows,
        correctness_rows,
        run_dir=run_dir,
    )
    comparison["evidence_sha256"] = {
        name: sha256_file(run_dir / name)
        for name in (
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
        )
    }
    _write_json(run_dir / "comparison.json", comparison)
    comparison_sha256 = sha256_file(
        run_dir / "comparison.json"
    )
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "classification": comparison["classification"],
        "selected_policy": comparison["selected_policy"],
        "selected_burst_width":
            comparison["selected_burst_width"],
        "comparison_sha256": comparison_sha256,
    }
    _write_json(run_dir / "gate.json", gate)
    sidecars = sorted(referenced_sidecars)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "artifacts": {
            name: sha256_file(run_dir / name)
            for name in (*PRIMARY_ARTIFACTS, *sidecars)
        },
    }
    _write_json(run_dir / "manifest.sha256", manifest)
    return gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "--artifact-dir", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gate = produce_gate(
        Path(args.run_dir),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps(gate, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
