#!/usr/bin/env python3
"""Producer gate for exact-burst ragged coalescing evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tools.exact_burst_split_phase_gate import (
    _read_json,
    _read_jsonl,
    _regression,
    _throughput_regression,
    _write_json,
)
from tools.profile_exact_burst_ragged_coalescing import (
    CONTEXT_CASES,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    policy_order,
    read_float32_sidecar,
    sha256_file,
    summarize_rows,
    validate_case_row,
    validate_correctness_rows,
)


COMPARISON_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.comparison.v1"
)
GATE_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.gate.v1"
)
MANIFEST_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.manifest.v1"
)
PRIMARY_ARTIFACTS = (
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "source.patch",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
)
REFERENCE_POLICIES = (
    "decode_burst_k4",
    "decode_burst_k8_split_phase",
)
K4_POLICY = "decode_burst_k4"
SPLIT_POLICY = "decode_burst_k8_split_phase"
CANDIDATE_POLICY = "decode_burst_k8_split_phase_ragged"
STAGE1_MODEL_BASENAMES = ("Qwen3-0.6B", "Qwen3-0___6B")
EXPECTED_ROWS = 45
EXPECTED_CORRECTNESS_ROWS = 36

LOGIT_MAX_ABS_LIMIT = 0.25
LOGIT_MEAN_ABS_LIMIT = 0.05
TAIL_SEVEN_IMPROVEMENT_MINIMUM = 0.10
AGGREGATE_TPOT_REGRESSION_LIMIT = 0.01
THROUGHPUT_REGRESSION_LIMIT = 0.01
BUCKET_TPOT_REGRESSION_LIMIT = 0.02
BUCKET_E2E_REGRESSION_LIMIT = 0.02
BUCKET_TTFT_REGRESSION_LIMIT = 0.03
MEMORY_REGRESSION_LIMIT = 0.03
MEDIAN_GAP_REGRESSION_LIMIT = 0.03
MAXIMUM_GAP_REGRESSION_LIMIT = 0.05


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _paired_rows(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
) -> list[tuple[dict, dict]]:
    def identity(row: dict) -> tuple[str, int]:
        return row["context_bucket"], row["repetition"]

    baseline = {identity(row): row for row in baseline_rows}
    candidate = {identity(row): row for row in candidate_rows}
    if (
        len(baseline) != len(baseline_rows)
        or len(candidate) != len(candidate_rows)
        or set(baseline) != set(candidate)
    ):
        raise ValueError("paired request inventory mismatch")
    return [(baseline[key], candidate[key]) for key in sorted(baseline)]


def _row_tpot(row: dict, percentile: float) -> float:
    samples = row["amortized_tpot_samples_ns"]
    if percentile == 0.5:
        return float(statistics.median(samples))
    return _nearest_rank(samples, percentile)


def _metric_summary(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    baseline_policy: str,
) -> dict:
    pairs = _paired_rows(baseline_rows, candidate_rows)

    def paired_regression(field) -> float:
        return statistics.median(
            _regression(field(left), field(right))
            for left, right in pairs
        )

    def paired_throughput(field) -> float:
        return statistics.median(
            _throughput_regression(field(left), field(right))
            for left, right in pairs
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
        "candidate_policy": CANDIDATE_POLICY,
        "sample_count_per_policy": len(pairs),
        "tpot_median_regression_fraction": paired_regression(
            lambda row: _row_tpot(row, 0.5)
        ),
        "tpot_p95_regression_fraction": paired_regression(
            lambda row: _row_tpot(row, 0.95)
        ),
        "ttft_regression_fraction": paired_regression(
            lambda row: float(row["ttft_ns"])
        ),
        "e2e_regression_fraction": paired_regression(
            lambda row: float(row["e2e_ns"])
        ),
        "throughput_regression_fraction": paired_throughput(
            lambda row: float(row["output_tokens_per_second"])
        ),
        "cuda_allocated_regression_fraction": _regression(
            baseline_allocated,
            candidate_allocated,
        ),
        "cuda_reserved_regression_fraction": _regression(
            baseline_reserved,
            candidate_reserved,
        ),
        "baseline_cuda_peak_allocated_bytes":
            baseline_allocated,
        "candidate_cuda_peak_allocated_bytes":
            candidate_allocated,
        "baseline_cuda_peak_reserved_bytes":
            baseline_reserved,
        "candidate_cuda_peak_reserved_bytes":
            candidate_reserved,
    }


def _validate_performance_inventory(rows: list[dict]) -> list[dict]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly 45 measured rows, got {len(rows)}"
        )
    identities = [
        (
            row.get("context_bucket"),
            row.get("repetition"),
            row.get("policy"),
        )
        for row in rows
        if isinstance(row, dict)
    ]
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXT_CASES
        for repetition in range(5)
        for policy in POLICIES
    }
    if len(identities) != len(rows) or set(identities) != expected:
        raise ValueError("measured case inventory is incomplete")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate case identity")
    return [validate_case_row(row) for row in rows]


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
    global_max = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_ids = True
    all_text = True
    all_passed = True
    for baseline in REFERENCE_POLICIES:
        for bucket, _prompt, _generated in CONTEXT_CASES:
            for point in SAMPLING_POINTS:
                left = by_identity[(bucket, baseline, point)]
                right = by_identity[
                    (bucket, CANDIDATE_POLICY, point)
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
                    abs(left_value - right_value)
                    for left_value, right_value
                    in zip(left_values, right_values)
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
                ids_equal = (
                    left["output_token_ids"]
                    == right["output_token_ids"]
                )
                text_equal = (
                    left["output_text_sha256"]
                    == right["output_text_sha256"]
                )
                passed = all((
                    maximum <= LOGIT_MAX_ABS_LIMIT,
                    mean <= LOGIT_MEAN_ABS_LIMIT,
                    argmax_equal,
                    ids_equal,
                    text_equal,
                ))
                global_max = max(global_max, maximum)
                total_abs += sum(differences)
                total_count += len(differences)
                all_argmax &= argmax_equal
                all_ids &= ids_equal
                all_text &= text_equal
                all_passed &= passed
                pairs.append({
                    "baseline_policy": baseline,
                    "candidate_policy": CANDIDATE_POLICY,
                    "context_bucket": bucket,
                    "sampling_point": point,
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "argmax_equal": argmax_equal,
                    "output_ids_exact": ids_equal,
                    "output_text_exact": text_equal,
                    "passed": passed,
                })
    return ({
        "pair_count": len(pairs),
        "max_abs": global_max,
        "mean_abs": (
            total_abs / total_count if total_count else 0.0
        ),
        "argmax_equal": all_argmax,
        "output_ids_exact": all_ids,
        "output_text_exact": all_text,
        "pairs": pairs,
    }, all_passed)


def _outputs_are_exact(rows: list[dict]) -> bool:
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    exact = True
    for bucket, _prompt, _generated in CONTEXT_CASES:
        for repetition in range(5):
            group = [
                by_identity[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            exact &= (
                len({
                    tuple(row["output_token_ids"]) for row in group
                }) == 1
                and len({
                    row["output_text_sha256"] for row in group
                }) == 1
            )
    return exact


def _capture_cost(rows: list[dict]) -> tuple[int, int]:
    receipt_counts = []
    retained = []
    for row in rows:
        receipts = row[
            "exact_greedy_decode_burst_summary"
        ]["capture_receipts"]
        receipt_counts.append(len(receipts))
        retained.extend(
            int(receipt["retained_static_bytes"])
            for receipt in receipts
        )
    return max(receipt_counts, default=0), max(retained, default=0)


def _lifecycle_summary(rows: list[dict]) -> dict:
    selected = [
        row for row in rows
        if row["policy"] == CANDIDATE_POLICY
    ]
    totals = {
        "request_count": len(selected),
        "attempts": 0,
        "acceptances": 0,
        "commits": 0,
        "committed_tokens": 0,
        "target_model_forwards": 0,
        "graph_replays": 0,
        "final_token_d2h_calls": 0,
        "final_token_d2h_bytes": 0,
        "prefix_commits": 0,
        "suffix_commits": 0,
        "parent_leases": 0,
        "unexpected_scheduler_calls": 0,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
    }
    requested = {}
    authorized = {}
    fallbacks = {}
    split_failures = {}
    for row in selected:
        summary = row["exact_greedy_decode_burst_summary"]
        inventory = row["split_phase_inventory"]
        for field in (
            "attempts",
            "acceptances",
            "commits",
            "committed_tokens",
            "target_model_forwards",
            "graph_replays",
            "final_token_d2h_calls",
            "final_token_d2h_bytes",
            "prefix_commits",
            "suffix_commits",
            "failures",
            "quarantines",
            "pending_leases",
        ):
            totals[field] += summary[field]
        totals["parent_leases"] += inventory[
            "parent_lease_count"
        ]
        totals["unexpected_scheduler_calls"] += inventory[
            "unexpected_scheduler_calls"
        ]
        for target, source in (
            (requested, summary["requested_width_histogram"]),
            (authorized, summary["authorized_width_histogram"]),
            (fallbacks, summary["fallback_counts"]),
            (split_failures, summary["split_phase_failure_counts"]),
        ):
            for key, value in source.items():
                target[key] = target.get(key, 0) + value
    totals.update({
        "requested_width_histogram": requested,
        "authorized_width_histogram": authorized,
        "fallback_counts": fallbacks,
        "split_phase_failure_counts": split_failures,
    })
    totals["complete"] = all((
        totals["request_count"] == 15,
        totals["attempts"] == 255,
        totals["acceptances"] == 255,
        totals["commits"] == 255,
        totals["committed_tokens"] == 1_905,
        totals["target_model_forwards"] == 1_905,
        totals["graph_replays"] == 1_905,
        totals["final_token_d2h_calls"] == 30,
        totals["final_token_d2h_bytes"] == 840,
        totals["prefix_commits"] == 225,
        totals["suffix_commits"] == 225,
        totals["parent_leases"] == 225,
        totals["unexpected_scheduler_calls"] == 0,
        totals["failures"] == 0,
        totals["quarantines"] == 0,
        totals["pending_leases"] == 0,
        requested == {"3": 15, "4": 15, "8": 225},
        authorized == {"3": 15, "4": 15, "8": 225},
        fallbacks == {},
        split_failures == {},
    ))
    return totals


def _candidate_evaluation(
    rows: list[dict],
    *,
    correctness_passed: bool,
) -> dict:
    candidate = [
        row for row in rows
        if row["policy"] == CANDIDATE_POLICY
    ]
    split = [row for row in rows if row["policy"] == SPLIT_POLICY]
    k4 = [row for row in rows if row["policy"] == K4_POLICY]
    aggregate = _metric_summary(
        split,
        candidate,
        baseline_policy=SPLIT_POLICY,
    )
    pairs = _paired_rows(split, candidate)
    tail_improvement = statistics.median(
        (
            float(left["tail_seven_elapsed_ns"])
            - float(right["tail_seven_elapsed_ns"])
        )
        / float(left["tail_seven_elapsed_ns"])
        for left, right in pairs
    )
    by_bucket = {}
    bucket_regressions = []
    for bucket, _prompt, _generated in CONTEXT_CASES:
        metrics = _metric_summary(
            [
                row for row in split
                if row["context_bucket"] == bucket
            ],
            [
                row for row in candidate
                if row["context_bucket"] == bucket
            ],
            baseline_policy=SPLIT_POLICY,
        )
        by_bucket[bucket] = metrics
        if (
            metrics["tpot_median_regression_fraction"]
            > BUCKET_TPOT_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:median_tpot")
        if (
            metrics["tpot_p95_regression_fraction"]
            > BUCKET_TPOT_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:p95_tpot")
        if (
            metrics["ttft_regression_fraction"]
            > BUCKET_TTFT_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > BUCKET_E2E_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:e2e")
    candidate_median_gap = statistics.median(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in candidate
    )
    k4_median_gap = statistics.median(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in k4
    )
    candidate_maximum_gap = max(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in candidate
    )
    k4_maximum_gap = max(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in k4
    )
    median_gap_regression = _regression(
        k4_median_gap,
        candidate_median_gap,
    )
    maximum_gap_regression = _regression(
        k4_maximum_gap,
        candidate_maximum_gap,
    )
    split_capture_count, split_retained = _capture_cost(split)
    candidate_capture_count, candidate_retained = _capture_cost(
        candidate
    )
    lifecycle = _lifecycle_summary(rows)
    performance_passed = all((
        tail_improvement >= TAIL_SEVEN_IMPROVEMENT_MINIMUM,
        aggregate["tpot_median_regression_fraction"]
        <= AGGREGATE_TPOT_REGRESSION_LIMIT,
        aggregate["throughput_regression_fraction"]
        <= THROUGHPUT_REGRESSION_LIMIT,
        aggregate["cuda_allocated_regression_fraction"]
        <= MEMORY_REGRESSION_LIMIT,
        aggregate["cuda_reserved_regression_fraction"]
        <= MEMORY_REGRESSION_LIMIT,
        median_gap_regression <= MEDIAN_GAP_REGRESSION_LIMIT,
        maximum_gap_regression <= MAXIMUM_GAP_REGRESSION_LIMIT,
        candidate_capture_count <= split_capture_count,
        candidate_retained <= split_retained,
        not bucket_regressions,
    ))
    return {
        "policy": CANDIDATE_POLICY,
        "correctness_passed": correctness_passed,
        "performance_passed": performance_passed,
        "tail_seven_improvement_fraction": tail_improvement,
        "aggregate": {"split_vs_ragged": aggregate},
        "by_bucket": by_bucket,
        "bucket_regressions": bucket_regressions,
        "candidate_median_max_gap_ns": candidate_median_gap,
        "k4_median_max_gap_ns": k4_median_gap,
        "median_max_gap_regression_vs_k4":
            median_gap_regression,
        "candidate_maximum_gap_ns": candidate_maximum_gap,
        "k4_maximum_gap_ns": k4_maximum_gap,
        "maximum_gap_regression_vs_k4":
            maximum_gap_regression,
        "split_capture_count": split_capture_count,
        "candidate_capture_count": candidate_capture_count,
        "split_capture_retained_static_bytes": split_retained,
        "candidate_capture_retained_static_bytes":
            candidate_retained,
        "lifecycle": lifecycle,
    }


def classify(
    rows: list[dict],
    correctness_rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    validated = _validate_performance_inventory(rows)
    correctness, logits_passed = _correctness_metrics(
        correctness_rows,
        run_dir=Path(run_dir),
    )
    output_exact = _outputs_are_exact(validated)
    run_tags = {
        *(row.get("run_tag") for row in rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    source_commits = {
        *(row.get("source_commit") for row in rows),
        *(row.get("source_commit") for row in correctness_rows),
    }
    evidence_complete = (
        len(run_tags) == 1
        and len(source_commits) == 1
    )
    evaluation = _candidate_evaluation(
        validated,
        correctness_passed=output_exact and logits_passed,
    )
    if not evaluation["correctness_passed"]:
        classification = (
            "NO_GO_EXACT_BURST_RAGGED_COALESCING_CORRECTNESS"
        )
    elif not evidence_complete or not evaluation["lifecycle"]["complete"]:
        classification = (
            "INCOMPLETE_EXACT_BURST_RAGGED_COALESCING_EVIDENCE"
        )
    elif not evaluation["performance_passed"]:
        classification = (
            "NO_GO_EXACT_BURST_RAGGED_COALESCING_PERFORMANCE"
        )
    else:
        classification = "GO_EXACT_BURST_RAGGED_COALESCING"
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_tag": validated[0]["run_tag"],
        "source_commit": validated[0]["source_commit"],
        "classification": classification,
        "selected_policy": CANDIDATE_POLICY,
        "selected_burst_width": 8,
        "ragged_width_cap": 4,
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_ABS_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_ABS_LIMIT,
            "tail_seven_improvement_minimum":
                TAIL_SEVEN_IMPROVEMENT_MINIMUM,
            "aggregate_tpot_regression_limit":
                AGGREGATE_TPOT_REGRESSION_LIMIT,
            "throughput_regression_limit":
                THROUGHPUT_REGRESSION_LIMIT,
            "bucket_tpot_regression_limit":
                BUCKET_TPOT_REGRESSION_LIMIT,
            "bucket_e2e_regression_limit":
                BUCKET_E2E_REGRESSION_LIMIT,
            "bucket_ttft_regression_limit":
                BUCKET_TTFT_REGRESSION_LIMIT,
            "memory_regression_limit": MEMORY_REGRESSION_LIMIT,
            "median_gap_regression_limit":
                MEDIAN_GAP_REGRESSION_LIMIT,
            "maximum_gap_regression_limit":
                MAXIMUM_GAP_REGRESSION_LIMIT,
        },
        "correctness": correctness,
        "candidate_evaluation": evaluation,
    }


def _validate_source_manifest(
    manifest,
    *,
    repo_root: Path,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "exact-burst-ragged-coalescing.source.v1"
    ):
        raise ValueError("source manifest is invalid")
    digests = manifest.get("source_sha256")
    if (
        not isinstance(digests, dict)
        or set(digests) != set(SOURCE_FILES)
    ):
        raise ValueError("source manifest file inventory mismatch")
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        if digests[relative] != sha256_file(path):
            raise ValueError(f"source digest mismatch: {relative}")


def _validate_workload_manifest(manifest) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "exact-burst-ragged-coalescing.workload.v1"
    ):
        raise ValueError("workload manifest is invalid")
    expected = {
        "context_cases": [
            {
                "context_bucket": bucket,
                "prompt_tokens": prompt,
                "generated_tokens": generated,
            }
            for bucket, prompt, generated in CONTEXT_CASES
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
            "gate-only-exact-burst-ragged-"
            "coalescing-correctness-v1",
        "correctness_sampling_points": list(SAMPLING_POINTS),
        "policy_configs": {
            policy: json.loads(json.dumps(config))
            for policy, config in POLICY_CONFIGS.items()
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
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
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
            environment.get("cuda_runtime_version"), str
        )
        or not environment["cuda_runtime_version"]
        or not isinstance(
            environment.get("cuda_device_name"), str
        )
        or not environment["cuda_device_name"]
    ):
        raise ValueError(
            "workload manifest mismatch: environment"
        )


def _validate_clean_source_patch(run_dir: Path) -> None:
    path = run_dir / "source.patch"
    if not path.is_file():
        raise ValueError("source.patch is missing")
    if path.read_bytes():
        raise ValueError("dirty source patch is not allowed")


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
    _validate_clean_source_patch(run_dir)
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
        *(row.get("source_commit") for row in correctness_rows),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    _validate_performance_inventory(rows)
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
            "source.patch",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
        )
    }
    _write_json(run_dir / "comparison.json", comparison)
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "source_patch_sha256": sha256_file(
            run_dir / "source.patch"
        ),
        "classification": comparison["classification"],
        "selected_policy": comparison["selected_policy"],
        "selected_burst_width":
            comparison["selected_burst_width"],
        "ragged_width_cap": comparison["ragged_width_cap"],
        "comparison_sha256": sha256_file(
            run_dir / "comparison.json"
        ),
    }
    _write_json(run_dir / "gate.json", gate)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "source_patch_sha256": gate["source_patch_sha256"],
        "artifacts": {
            name: sha256_file(run_dir / name)
            for name in (
                *PRIMARY_ARTIFACTS,
                *sorted(referenced_sidecars),
            )
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
