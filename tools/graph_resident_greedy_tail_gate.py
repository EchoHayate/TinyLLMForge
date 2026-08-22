#!/usr/bin/env python3
"""Producer gate for graph-resident greedy-tail evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tools.profile_graph_resident_greedy_tail import (
    GRAPH_COST_FIELDS,
    POLICIES,
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
    "graph-resident-greedy-tail.comparison.v1"
)
GATE_SCHEMA_VERSION = "graph-resident-greedy-tail.gate.v1"
MANIFEST_SCHEMA_VERSION = (
    "graph-resident-greedy-tail.manifest.v1"
)
CLASSIFICATIONS = (
    "GO_GRAPH_RESIDENT_GREEDY_TAIL",
    "NO_GO_CORRECTNESS",
    "NO_GO_GRAPH_REPLAY_INCOMPLETE",
    "NO_GO_LEGACY_TPOT_MEDIAN",
    "NO_GO_LEGACY_TPOT_P95",
    "NO_GO_HOST_GREEDY_INCREMENTAL",
    "NO_GO_PROTECTED_REGRESSION",
    "NO_GO_COST_INCOMPLETE",
    "NO_GO_EVIDENCE_INCOMPLETE",
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
EXPECTED_ROWS = 45
EXPECTED_CORRECTNESS_ROWS = 27
LEGACY_MEDIAN_MIN_IMPROVEMENT = 0.05
LEGACY_AGGREGATE_P95_MIN_IMPROVEMENT = 0.05
HOST_AGGREGATE_MEDIAN_MIN_IMPROVEMENT = 0.02
HOST_BUCKET_TPOT_MAX_REGRESSION = 0.02
LEGACY_BUCKET_TPOT_MAX_REGRESSION = 0.03
LATENCY_MAX_REGRESSION = 0.03
THROUGHPUT_MAX_REGRESSION = 0.02
RESERVED_MEMORY_MAX_REGRESSION = 0.02
LOGIT_MAX_ABS_LIMIT = 0.25
LOGIT_MEAN_ABS_LIMIT = 0.05


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path):
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
        )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(
                line,
                parse_constant=_reject_constant,
            )
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
        for value in row["tpot_samples_ns"]
    ]
    candidate_tpot = [
        float(value)
        for row in candidate_rows
        for value in row["tpot_samples_ns"]
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
        "baseline_cuda_peak_allocated_bytes":
            baseline_allocated,
        "candidate_cuda_peak_allocated_bytes":
            candidate_allocated,
        "cuda_allocated_delta_bytes":
            candidate_allocated - baseline_allocated,
        "baseline_cuda_peak_reserved_bytes":
            baseline_reserved,
        "candidate_cuda_peak_reserved_bytes":
            candidate_reserved,
        "cuda_reserved_delta_bytes":
            candidate_reserved - baseline_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(
                baseline_reserved,
                candidate_reserved,
            ),
    }


def _validate_performance_inventory(
    rows: list[dict],
) -> list[dict]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly 45 measured rows, got {len(rows)}"
        )
    validated = [
        validate_case_row(
            row,
            require_complete_optimized_path=False,
        )
        for row in rows
    ]
    identities = [
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        )
        for row in validated
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate case identity")
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in context_cases()
        for repetition in range(5)
        for policy in POLICIES
    }
    if set(identities) != expected:
        raise ValueError(
            "measured case inventory is incomplete"
        )
    return validated


def _correctness_metrics(
    correctness_rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    rows = validate_correctness_rows(
        correctness_rows,
        run_dir=run_dir,
    )
    by_identity = {
        (
            row["context_bucket"],
            row["sampling_point"],
            row["policy"],
        ): row
        for row in rows
    }
    pairs = []
    global_max = 0.0
    worst_pair_mean = 0.0
    total_abs = 0.0
    total_elements = 0
    all_argmax_equal = True
    all_output_ids_exact = True
    all_output_text_exact = True
    policy_pairs = (
        ("legacy", "host_greedy"),
        ("legacy", "graph_greedy"),
        ("host_greedy", "graph_greedy"),
    )
    for bucket, _prompt, _generated in context_cases():
        for point in SAMPLING_POINTS:
            for baseline_policy, candidate_policy in policy_pairs:
                baseline = by_identity[
                    (bucket, point, baseline_policy)
                ]
                candidate = by_identity[
                    (bucket, point, candidate_policy)
                ]
                if (
                    baseline["logits_shape"]
                    != candidate["logits_shape"]
                    or baseline["logits_element_count"]
                    != candidate["logits_element_count"]
                ):
                    raise ValueError(
                        "paired logits shape mismatch"
                    )
                baseline_values = read_float32_sidecar(
                    run_dir,
                    path=baseline["logits_path"],
                    expected_element_count=baseline[
                        "logits_element_count"
                    ],
                    expected_byte_length=baseline[
                        "logits_byte_length"
                    ],
                    expected_sha256=baseline[
                        "logits_sha256"
                    ],
                )
                candidate_values = read_float32_sidecar(
                    run_dir,
                    path=candidate["logits_path"],
                    expected_element_count=candidate[
                        "logits_element_count"
                    ],
                    expected_byte_length=candidate[
                        "logits_byte_length"
                    ],
                    expected_sha256=candidate[
                        "logits_sha256"
                    ],
                )
                differences = [
                    abs(left - right)
                    for left, right in zip(
                        baseline_values,
                        candidate_values,
                    )
                ]
                maximum = max(differences)
                mean = sum(differences) / len(differences)
                baseline_argmax = max(
                    range(len(baseline_values)),
                    key=baseline_values.__getitem__,
                )
                candidate_argmax = max(
                    range(len(candidate_values)),
                    key=candidate_values.__getitem__,
                )
                argmax_equal = (
                    baseline_argmax == candidate_argmax
                )
                output_ids_exact = (
                    baseline["output_token_ids"]
                    == candidate["output_token_ids"]
                )
                output_text_exact = (
                    baseline["output_text_sha256"]
                    == candidate["output_text_sha256"]
                )
                global_max = max(global_max, maximum)
                worst_pair_mean = max(
                    worst_pair_mean,
                    mean,
                )
                total_abs += sum(differences)
                total_elements += len(differences)
                all_argmax_equal = (
                    all_argmax_equal and argmax_equal
                )
                all_output_ids_exact = (
                    all_output_ids_exact
                    and output_ids_exact
                )
                all_output_text_exact = (
                    all_output_text_exact
                    and output_text_exact
                )
                pairs.append({
                    "context_bucket": bucket,
                    "sampling_point": point,
                    "baseline_policy": baseline_policy,
                    "candidate_policy": candidate_policy,
                    "element_count": len(differences),
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "baseline_argmax": baseline_argmax,
                    "candidate_argmax": candidate_argmax,
                    "argmax_equal": argmax_equal,
                    "output_ids_exact": output_ids_exact,
                    "output_text_exact": output_text_exact,
                })
    return {
        "row_count": len(rows),
        "pair_count": len(pairs),
        "max_abs": global_max,
        "mean_abs": worst_pair_mean,
        "aggregate_mean_abs":
            total_abs / total_elements,
        "argmax_equal": all_argmax_equal,
        "output_ids_exact": all_output_ids_exact,
        "output_text_exact": all_output_text_exact,
        "pairs": pairs,
    }


def _graph_replay_complete(
    rows: list[dict],
) -> bool:
    for row in rows:
        greedy = row["greedy_fast_path_summary"]
        graph = row[
            "graph_resident_greedy_tail_summary"
        ]
        generated = row["generated_tokens"]
        if row["policy"] == "legacy":
            if (
                greedy["eligible_steps"] != 0
                or greedy["optimized_steps"] != 0
                or graph["replayed_steps"] != 0
                or graph["captured_graphs"] != 0
            ):
                return False
        elif row["policy"] == "host_greedy":
            if (
                greedy["eligible_steps"] != generated
                or greedy["optimized_steps"] != generated
                or graph["replayed_steps"] != 0
                or graph["captured_graphs"] != 0
            ):
                return False
        else:
            expected = generated - 1
            if (
                greedy["eligible_steps"] != 1
                or greedy["optimized_steps"] != 1
                or graph["eligible_steps"] != expected
                or graph["replayed_steps"] != expected
                or graph["final_token_d2h_calls"] != expected
                or graph[
                    "avoided_external_compute_logits_calls"
                ] != expected
                or graph[
                    "avoided_external_float32_conversions"
                ] != expected
                or graph[
                    "avoided_external_argmax_calls"
                ] != expected
                or graph["captured_graphs"] != 1
                or graph["fallback_counts"]
                or graph["quarantine_reason"] is not None
                or graph["capture_receipt"] is None
            ):
                return False
    return True


def _cost_summary(rows: list[dict]) -> dict:
    graph_rows = [
        row for row in rows
        if row["policy"] == "graph_greedy"
    ]

    def summary(field: str) -> dict:
        values = [
            int(row[field])
            for row in graph_rows
        ]
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    graph_summaries = [
        row["graph_resident_greedy_tail_summary"]
        for row in graph_rows
    ]
    return {
        "capture_duration_ns":
            summary("graph_capture_duration_ns"),
        "allocated_delta_bytes":
            summary("graph_allocated_delta_bytes"),
        "reserved_delta_bytes":
            summary("graph_reserved_delta_bytes"),
        "retained_static_bytes":
            summary("graph_retained_static_bytes"),
        "final_token_d2h_calls": sum(
            item["final_token_d2h_calls"]
            for item in graph_summaries
        ),
        "avoided_work": {
            field: sum(item[field] for item in graph_summaries)
            for field in (
                "avoided_external_compute_logits_calls",
                "avoided_external_float32_conversions",
                "avoided_external_argmax_calls",
            )
        },
    }


def classify(
    rows: list[dict],
    correctness_rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    validated = _validate_performance_inventory(rows)
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in validated
    }
    exact_outputs = True
    for bucket, _prompt, _generated in context_cases():
        for repetition in range(5):
            triple = [
                by_identity[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            exact_outputs = exact_outputs and (
                len({
                    tuple(row["output_token_ids"])
                    for row in triple
                }) == 1
                and len({
                    row["output_text_sha256"]
                    for row in triple
                }) == 1
            )
    correctness = _correctness_metrics(
        correctness_rows,
        run_dir=Path(run_dir),
    )
    correctness_passed = (
        exact_outputs
        and correctness["output_ids_exact"]
        and correctness["output_text_exact"]
        and correctness["max_abs"] <= LOGIT_MAX_ABS_LIMIT
        and correctness["mean_abs"] <= LOGIT_MEAN_ABS_LIMIT
        and correctness["argmax_equal"]
    )
    graph_replay_complete = _graph_replay_complete(validated)
    by_bucket = {}
    for bucket, _prompt, _generated in context_cases():
        selected = [
            row
            for row in validated
            if row["context_bucket"] == bucket
        ]
        graph_rows = [
            row for row in selected
            if row["policy"] == "graph_greedy"
        ]
        by_bucket[bucket] = {
            "legacy_vs_graph": _metric_summary(
                [
                    row for row in selected
                    if row["policy"] == "legacy"
                ],
                graph_rows,
                baseline_policy="legacy",
                candidate_policy="graph_greedy",
            ),
            "host_greedy_vs_graph": _metric_summary(
                [
                    row for row in selected
                    if row["policy"] == "host_greedy"
                ],
                graph_rows,
                baseline_policy="host_greedy",
                candidate_policy="graph_greedy",
            ),
        }
    graph_rows = [
        row for row in validated
        if row["policy"] == "graph_greedy"
    ]
    aggregate = {
        "legacy_vs_graph": _metric_summary(
            [
                row for row in validated
                if row["policy"] == "legacy"
            ],
            graph_rows,
            baseline_policy="legacy",
            candidate_policy="graph_greedy",
        ),
        "host_greedy_vs_graph": _metric_summary(
            [
                row for row in validated
                if row["policy"] == "host_greedy"
            ],
            graph_rows,
            baseline_policy="host_greedy",
            candidate_policy="graph_greedy",
        ),
    }
    legacy_winning_buckets = sum(
        metrics["legacy_vs_graph"][
            "tpot_median_improvement_fraction"
        ] >= LEGACY_MEDIAN_MIN_IMPROVEMENT
        for metrics in by_bucket.values()
    )
    host_incremental_regressions = []
    protected_regressions = []
    for bucket, comparisons in by_bucket.items():
        legacy = comparisons["legacy_vs_graph"]
        host = comparisons["host_greedy_vs_graph"]
        if (
            host["tpot_median_improvement_fraction"]
            < -HOST_BUCKET_TPOT_MAX_REGRESSION
        ):
            host_incremental_regressions.append(
                f"{bucket}:median_tpot"
            )
        if (
            host["tpot_p95_improvement_fraction"]
            < -HOST_BUCKET_TPOT_MAX_REGRESSION
        ):
            host_incremental_regressions.append(
                f"{bucket}:p95_tpot"
            )
        if (
            legacy["tpot_median_improvement_fraction"]
            < -LEGACY_BUCKET_TPOT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:median_tpot"
            )
        if (
            legacy["tpot_p95_improvement_fraction"]
            < -LEGACY_BUCKET_TPOT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:p95_tpot"
            )
        if (
            legacy["ttft_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            protected_regressions.append(f"{bucket}:ttft")
        if (
            legacy["e2e_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            protected_regressions.append(f"{bucket}:e2e")
        if (
            legacy["throughput_regression_fraction"]
            > THROUGHPUT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:throughput"
            )
    if (
        aggregate["legacy_vs_graph"][
            "cuda_reserved_regression_fraction"
        ] > RESERVED_MEMORY_MAX_REGRESSION
    ):
        protected_regressions.append(
            "aggregate:cuda_reserved"
        )
    cost = _cost_summary(validated)
    cost_complete = (
        cost["capture_duration_ns"]["min"] > 0
        and cost["retained_static_bytes"]["min"] > 0
        and cost["final_token_d2h_calls"]
        == 15 * 127
    )
    run_tags = {
        *(row.get("run_tag") for row in validated),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        *(row.get("source_commit") for row in validated),
        *(
            row.get("source_commit")
            for row in correctness_rows
        ),
    }
    evidence_complete = (
        len(run_tags) == 1
        and len(commits) == 1
    )
    if not correctness_passed:
        classification = "NO_GO_CORRECTNESS"
    elif not graph_replay_complete:
        classification = (
            "NO_GO_GRAPH_REPLAY_INCOMPLETE"
        )
    elif legacy_winning_buckets < 2:
        classification = "NO_GO_LEGACY_TPOT_MEDIAN"
    elif (
        aggregate["legacy_vs_graph"][
            "tpot_p95_improvement_fraction"
        ] < LEGACY_AGGREGATE_P95_MIN_IMPROVEMENT
    ):
        classification = "NO_GO_LEGACY_TPOT_P95"
    elif (
        aggregate["host_greedy_vs_graph"][
            "tpot_median_improvement_fraction"
        ] < HOST_AGGREGATE_MEDIAN_MIN_IMPROVEMENT
        or host_incremental_regressions
    ):
        classification = (
            "NO_GO_HOST_GREEDY_INCREMENTAL"
        )
    elif protected_regressions:
        classification = "NO_GO_PROTECTED_REGRESSION"
    elif not cost_complete:
        classification = "NO_GO_COST_INCOMPLETE"
    elif not evidence_complete:
        classification = "NO_GO_EVIDENCE_INCOMPLETE"
    else:
        classification = "GO_GRAPH_RESIDENT_GREEDY_TAIL"
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_tag": validated[0]["run_tag"],
        "source_commit": validated[0]["source_commit"],
        "classification": classification,
        "correctness_passed": correctness_passed,
        "graph_replay_complete": graph_replay_complete,
        "legacy_median_tpot_winning_bucket_count":
            legacy_winning_buckets,
        "host_incremental_regressions":
            host_incremental_regressions,
        "protected_regressions": protected_regressions,
        "cost_complete": cost_complete,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_ABS_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_ABS_LIMIT,
            "legacy_median_tpot_min_improvement_fraction":
                LEGACY_MEDIAN_MIN_IMPROVEMENT,
            "legacy_aggregate_p95_min_improvement_fraction":
                LEGACY_AGGREGATE_P95_MIN_IMPROVEMENT,
            "host_aggregate_median_min_improvement_fraction":
                HOST_AGGREGATE_MEDIAN_MIN_IMPROVEMENT,
            "host_bucket_tpot_max_regression_fraction":
                HOST_BUCKET_TPOT_MAX_REGRESSION,
            "legacy_bucket_tpot_max_regression_fraction":
                LEGACY_BUCKET_TPOT_MAX_REGRESSION,
            "latency_max_regression_fraction":
                LATENCY_MAX_REGRESSION,
            "throughput_max_regression_fraction":
                THROUGHPUT_MAX_REGRESSION,
            "reserved_memory_max_regression_fraction":
                RESERVED_MEMORY_MAX_REGRESSION,
        },
        "correctness": correctness,
        "by_bucket": by_bucket,
        "aggregate": aggregate,
        "cost": cost,
    }


def _validate_source_manifest(
    manifest,
    *,
    repo_root: Path,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "graph-resident-greedy-tail.source.v1"
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
            raise ValueError(
                f"source file is missing: {relative}"
            )
        if digests[relative] != sha256_file(path):
            raise ValueError(
                f"source digest mismatch: {relative}"
            )


def _validate_workload_manifest(manifest) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "graph-resident-greedy-tail.workload.v1"
    ):
        raise ValueError("workload manifest is invalid")
    expected_cases = [
        {
            "context_bucket": bucket,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        }
        for bucket, prompt_tokens, generated_tokens
        in context_cases()
    ]
    required_values = {
        "context_cases": expected_cases,
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "policy_flags": {
            "legacy": {
                "zero_temperature_greedy_fast_path": False,
                "graph_resident_greedy_tail": False,
            },
            "host_greedy": {
                "zero_temperature_greedy_fast_path": True,
                "graph_resident_greedy_tail": False,
            },
            "graph_greedy": {
                "zero_temperature_greedy_fast_path": True,
                "graph_resident_greedy_tail": True,
            },
        },
        "policy_order": {
            str(index): list(policy_order(index))
            for index in range(5)
        },
        "correctness_sampling_points":
            list(SAMPLING_POINTS),
    }
    for field, expected in required_values.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"workload manifest mismatch: {field}"
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
    _validate_source_manifest(
        source,
        repo_root=Path(repo_root),
    )
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
    _write_json(run_dir / "comparison.json", comparison)
    comparison_sha256 = sha256_file(
        run_dir / "comparison.json"
    )
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "classification": comparison["classification"],
        "comparison_sha256": comparison_sha256,
    }
    _write_json(run_dir / "gate.json", gate)
    sidecars = sorted({
        row["logits_path"]
        for row in correctness_rows
    })
    artifact_names = (*PRIMARY_ARTIFACTS, *sidecars)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "artifacts": {
            name: sha256_file(run_dir / name)
            for name in artifact_names
        },
    }
    _write_json(run_dir / "manifest.sha256", manifest)
    return gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
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
    print(
        json.dumps(
            gate,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
