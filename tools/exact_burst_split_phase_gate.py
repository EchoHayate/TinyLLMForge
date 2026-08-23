#!/usr/bin/env python3
"""Producer gate for split-phase K8 exact-burst evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tools.profile_exact_burst_split_phase import (
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
    "exact-burst-split-phase.comparison.v1"
)
GATE_SCHEMA_VERSION = "exact-burst-split-phase.gate.v1"
MANIFEST_SCHEMA_VERSION = "exact-burst-split-phase.manifest.v1"
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
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
)
CANDIDATE_POLICY = "decode_burst_k8_split_phase"
STAGE1_MODEL_BASENAMES = ("Qwen3-0.6B", "Qwen3-0___6B")
EXPECTED_ROWS = 60
EXPECTED_CORRECTNESS_ROWS = 48

LOGIT_MAX_ABS_LIMIT = 0.25
LOGIT_MEAN_ABS_LIMIT = 0.05
AGGREGATE_TPOT_REGRESSION_LIMIT = 0.02
THROUGHPUT_REGRESSION_LIMIT = 0.02
LATENCY_REGRESSION_LIMIT = 0.03
RESERVED_MEMORY_REGRESSION_LIMIT = 0.03
MAXIMUM_GAP_RATIO_LIMIT = 0.60
MEDIAN_GAP_REGRESSION_LIMIT = 0.03
BUCKET_TPOT_REGRESSION_LIMIT = 0.03


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path):
    if not path.is_file():
        raise ValueError(f"primary artifact is missing: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"primary artifact is missing: {path.name}")
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


def _regression(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError("relative comparison baseline must be positive")
    return (candidate - baseline) / baseline


def _throughput_regression(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError("throughput baseline must be positive")
    return (baseline - candidate) / baseline


def _metric_summary(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    baseline_policy: str,
) -> dict:
    def identity(row: dict) -> tuple[str, int]:
        return row["context_bucket"], row["repetition"]

    baseline_by_identity = {
        identity(row): row for row in baseline_rows
    }
    candidate_by_identity = {
        identity(row): row for row in candidate_rows
    }
    if (
        len(baseline_by_identity) != len(baseline_rows)
        or len(candidate_by_identity) != len(candidate_rows)
        or set(baseline_by_identity) != set(candidate_by_identity)
    ):
        raise ValueError("paired request inventory mismatch")
    pairs = [
        (
            baseline_by_identity[key],
            candidate_by_identity[key],
        )
        for key in sorted(baseline_by_identity)
    ]

    def row_tpot(row: dict, percentile: float) -> float:
        samples = row["amortized_tpot_samples_ns"]
        if percentile == 0.5:
            return float(statistics.median(samples))
        return _nearest_rank(samples, percentile)

    def paired_regression(field) -> float:
        return statistics.median(
            _regression(field(left), field(right))
            for left, right in pairs
        )

    def paired_throughput_regression(field) -> float:
        return statistics.median(
            _throughput_regression(field(left), field(right))
            for left, right in pairs
        )

    baseline_median = statistics.median(
        row_tpot(row, 0.5) for row in baseline_rows
    )
    candidate_median = statistics.median(
        row_tpot(row, 0.5) for row in candidate_rows
    )
    baseline_p95 = statistics.median(
        row_tpot(row, 0.95) for row in baseline_rows
    )
    candidate_p95 = statistics.median(
        row_tpot(row, 0.95) for row in candidate_rows
    )
    baseline_p99 = statistics.median(
        row_tpot(row, 0.99) for row in baseline_rows
    )
    candidate_p99 = statistics.median(
        row_tpot(row, 0.99) for row in candidate_rows
    )
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
    baseline_throughput = statistics.median(
        float(row["output_tokens_per_second"])
        for row in baseline_rows
    )
    candidate_throughput = statistics.median(
        float(row["output_tokens_per_second"])
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
        "baseline_tpot_median_ns": baseline_median,
        "candidate_tpot_median_ns": candidate_median,
        "tpot_median_regression_fraction":
            paired_regression(
                lambda row: row_tpot(row, 0.5)
            ),
        "baseline_tpot_p95_ns": baseline_p95,
        "candidate_tpot_p95_ns": candidate_p95,
        "tpot_p95_regression_fraction":
            paired_regression(
                lambda row: row_tpot(row, 0.95)
            ),
        "baseline_tpot_p99_ns": baseline_p99,
        "candidate_tpot_p99_ns": candidate_p99,
        "tpot_p99_regression_fraction":
            paired_regression(
                lambda row: row_tpot(row, 0.99)
            ),
        "baseline_ttft_median_ns": baseline_ttft,
        "candidate_ttft_median_ns": candidate_ttft,
        "ttft_regression_fraction":
            paired_regression(
                lambda row: float(row["ttft_ns"])
            ),
        "baseline_e2e_median_ns": baseline_e2e,
        "candidate_e2e_median_ns": candidate_e2e,
        "e2e_regression_fraction":
            paired_regression(
                lambda row: float(row["e2e_ns"])
            ),
        "baseline_output_tokens_per_second_median":
            baseline_throughput,
        "candidate_output_tokens_per_second_median":
            candidate_throughput,
        "throughput_regression_fraction":
            paired_throughput_regression(
                lambda row: float(
                    row["output_tokens_per_second"]
                )
            ),
        "baseline_cuda_peak_reserved_bytes": baseline_reserved,
        "candidate_cuda_peak_reserved_bytes": candidate_reserved,
        "cuda_reserved_delta_bytes":
            candidate_reserved - baseline_reserved,
        "cuda_reserved_regression_fraction":
            _regression(baseline_reserved, candidate_reserved),
    }


def _validate_performance_inventory(rows: list[dict]) -> list[dict]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly 60 measured rows, got {len(rows)}"
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
        for bucket, _prompt, _generated in context_cases()
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


def _lifecycle_summary(rows: list[dict]) -> dict:
    selected = [
        row for row in rows
        if row["policy"] == CANDIDATE_POLICY
    ]
    fields = {
        "request_count": len(selected),
        "parent_leases": 0,
        "prefix_commits": 0,
        "suffix_commits": 0,
        "prefix_tickets": 0,
        "suffix_tickets": 0,
        "graph_replays": 0,
        "prefix_d2h_calls": 0,
        "suffix_d2h_calls": 0,
        "prefix_phase_waits": 0,
        "suffix_phase_waits": 0,
        "suffix_drains": 0,
        "ordinary_tail_tokens": 0,
        "unexpected_scheduler_calls": 0,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
    }
    for row in selected:
        summary = row["exact_greedy_decode_burst_summary"]
        inventory = row["split_phase_inventory"]
        fields["parent_leases"] += inventory["parent_lease_count"]
        fields["prefix_commits"] += summary["prefix_commits"]
        fields["suffix_commits"] += summary["suffix_commits"]
        fields["prefix_tickets"] += summary[
            "prefix_publication_tickets"
        ]
        fields["suffix_tickets"] += summary[
            "suffix_publication_tickets"
        ]
        fields["graph_replays"] += summary["graph_replays"]
        fields["prefix_d2h_calls"] += summary[
            "prefix_token_d2h_calls"
        ]
        fields["suffix_d2h_calls"] += summary[
            "suffix_token_d2h_calls"
        ]
        fields["prefix_phase_waits"] += summary[
            "prefix_phase_waits"
        ]
        fields["suffix_phase_waits"] += summary[
            "suffix_phase_waits"
        ]
        fields["suffix_drains"] += summary["suffix_drains"]
        fields["ordinary_tail_tokens"] += (
            row["generated_tokens"]
            - 1
            - summary["committed_tokens"]
        )
        fields["unexpected_scheduler_calls"] += inventory[
            "unexpected_scheduler_calls"
        ]
        fields["failures"] += summary["failures"]
        fields["quarantines"] += summary["quarantines"]
        fields["pending_leases"] += summary["pending_leases"]
    fields["complete"] = all((
        fields["request_count"] == 15,
        fields["parent_leases"] == 225,
        fields["prefix_commits"] == 225,
        fields["suffix_commits"] == 225,
        fields["prefix_tickets"] == 225,
        fields["suffix_tickets"] == 225,
        fields["graph_replays"] == 1_800,
        fields["prefix_d2h_calls"] == 225,
        fields["suffix_d2h_calls"] == 225,
        fields["prefix_phase_waits"] == 225,
        fields["suffix_phase_waits"] == 225,
        fields["suffix_drains"] == 225,
        fields["ordinary_tail_tokens"] == 105,
        fields["unexpected_scheduler_calls"] == 0,
        fields["failures"] == 0,
        fields["quarantines"] == 0,
        fields["pending_leases"] == 0,
    ))
    return fields


def _candidate_evaluation(
    rows: list[dict],
    *,
    correctness_passed: bool,
) -> dict:
    candidate = [
        row for row in rows
        if row["policy"] == CANDIDATE_POLICY
    ]
    k8 = [row for row in rows if row["policy"] == "decode_burst_k8"]
    k4 = [row for row in rows if row["policy"] == "decode_burst_k4"]
    aggregate_k8 = _metric_summary(
        k8,
        candidate,
        baseline_policy="decode_burst_k8",
    )
    by_bucket = {}
    bucket_regressions = []
    latency_regressions = []
    for bucket, _prompt, _generated in CONTEXT_CASES:
        baseline_bucket = [
            row for row in k8 if row["context_bucket"] == bucket
        ]
        candidate_bucket = [
            row for row in candidate
            if row["context_bucket"] == bucket
        ]
        metrics = _metric_summary(
            baseline_bucket,
            candidate_bucket,
            baseline_policy="decode_burst_k8",
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
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:e2e")
    candidate_maximum_gap = max(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in candidate
    )
    k8_maximum_gap = max(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in k8
    )
    k4_median_gap = statistics.median(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in k4
    )
    candidate_median_gap = statistics.median(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in candidate
    )
    maximum_gap_ratio = (
        candidate_maximum_gap / k8_maximum_gap
        if k8_maximum_gap else math.inf
    )
    median_gap_regression = _regression(
        k4_median_gap,
        candidate_median_gap,
    )
    lifecycle = _lifecycle_summary(rows)
    performance_passed = all((
        aggregate_k8["tpot_median_regression_fraction"]
        <= AGGREGATE_TPOT_REGRESSION_LIMIT,
        aggregate_k8["throughput_regression_fraction"]
        <= THROUGHPUT_REGRESSION_LIMIT,
        aggregate_k8["cuda_reserved_regression_fraction"]
        <= RESERVED_MEMORY_REGRESSION_LIMIT,
        maximum_gap_ratio <= MAXIMUM_GAP_RATIO_LIMIT,
        median_gap_regression <= MEDIAN_GAP_REGRESSION_LIMIT,
        not bucket_regressions,
        not latency_regressions,
    ))
    return {
        "policy": CANDIDATE_POLICY,
        "correctness_passed": correctness_passed,
        "lifecycle": lifecycle,
        "performance_passed": performance_passed,
        "bucket_regressions": bucket_regressions,
        "latency_regressions": latency_regressions,
        "memory_regression": (
            aggregate_k8["cuda_reserved_regression_fraction"]
            > RESERVED_MEMORY_REGRESSION_LIMIT
        ),
        "candidate_maximum_host_visible_gap_ns":
            candidate_maximum_gap,
        "k8_maximum_host_visible_gap_ns": k8_maximum_gap,
        "maximum_gap_ratio_vs_k8": maximum_gap_ratio,
        "candidate_median_max_gap_ns": candidate_median_gap,
        "k4_median_max_gap_ns": k4_median_gap,
        "median_max_gap_regression_vs_k4":
            median_gap_regression,
        "aggregate": {"k8_vs_split": aggregate_k8},
        "by_bucket": by_bucket,
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
            "NO_GO_EXACT_BURST_SPLIT_PHASE_CORRECTNESS"
        )
    elif not evidence_complete or not evaluation["lifecycle"]["complete"]:
        classification = (
            "INCOMPLETE_EXACT_BURST_SPLIT_PHASE_EVIDENCE"
        )
    elif not evaluation["performance_passed"]:
        classification = (
            "NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE"
        )
    else:
        classification = "GO_EXACT_BURST_SPLIT_PHASE"
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_tag": validated[0]["run_tag"],
        "source_commit": validated[0]["source_commit"],
        "classification": classification,
        "selected_policy": CANDIDATE_POLICY,
        "selected_burst_width": 8,
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_ABS_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_ABS_LIMIT,
            "aggregate_tpot_regression_limit":
                AGGREGATE_TPOT_REGRESSION_LIMIT,
            "throughput_regression_limit":
                THROUGHPUT_REGRESSION_LIMIT,
            "latency_regression_limit":
                LATENCY_REGRESSION_LIMIT,
            "reserved_memory_regression_limit":
                RESERVED_MEMORY_REGRESSION_LIMIT,
            "maximum_gap_ratio_limit":
                MAXIMUM_GAP_RATIO_LIMIT,
            "median_gap_regression_limit":
                MEDIAN_GAP_REGRESSION_LIMIT,
            "bucket_tpot_regression_limit":
                BUCKET_TPOT_REGRESSION_LIMIT,
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
        != "exact-burst-split-phase.source.v1"
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
        != "exact-burst-split-phase.workload.v1"
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
            "gate-only-exact-burst-split-phase-correctness-v1",
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
    comparison_sha256 = sha256_file(
        run_dir / "comparison.json"
    )
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
        "comparison_sha256": comparison_sha256,
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
