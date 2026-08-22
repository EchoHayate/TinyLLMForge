#!/usr/bin/env python3
"""Producer gate for zero-temperature greedy fast-path evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.profile_zero_temperature_greedy_fast_path import (
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
    "zero-temperature-greedy-fast-path.comparison.v1"
)
GATE_SCHEMA_VERSION = (
    "zero-temperature-greedy-fast-path.gate.v1"
)
MANIFEST_SCHEMA_VERSION = (
    "zero-temperature-greedy-fast-path.manifest.v1"
)
CLASSIFICATIONS = (
    "GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH",
    "NO_GO_CORRECTNESS",
    "NO_GO_OPTIMIZED_PATH_INCOMPLETE",
    "NO_GO_TPOT_MEDIAN",
    "NO_GO_TPOT_P95",
    "NO_GO_PROTECTED_REGRESSION",
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
EXPECTED_ROWS = 30
EXPECTED_CORRECTNESS_ROWS = 18
MEDIAN_TPOT_MIN_IMPROVEMENT = 0.05
AGGREGATE_P95_MIN_IMPROVEMENT = 0.05
TPOT_MAX_REGRESSION = 0.03
LATENCY_MAX_REGRESSION = 0.03
THROUGHPUT_MAX_REGRESSION = 0.02
RESERVED_MEMORY_MAX_REGRESSION = 0.01
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
    off_rows: list[dict],
    on_rows: list[dict],
) -> dict:
    off_tpot = [
        float(value)
        for row in off_rows
        for value in row["tpot_samples_ns"]
    ]
    on_tpot = [
        float(value)
        for row in on_rows
        for value in row["tpot_samples_ns"]
    ]
    off_median = statistics.median(off_tpot)
    on_median = statistics.median(on_tpot)
    off_p95 = _nearest_rank(off_tpot, 0.95)
    on_p95 = _nearest_rank(on_tpot, 0.95)
    off_p99 = _nearest_rank(off_tpot, 0.99)
    on_p99 = _nearest_rank(on_tpot, 0.99)
    off_ttft = statistics.median(
        float(row["ttft_ns"]) for row in off_rows
    )
    on_ttft = statistics.median(
        float(row["ttft_ns"]) for row in on_rows
    )
    off_e2e = statistics.median(
        float(row["e2e_ns"]) for row in off_rows
    )
    on_e2e = statistics.median(
        float(row["e2e_ns"]) for row in on_rows
    )
    off_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in off_rows
    )
    on_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in on_rows
    )
    off_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in off_rows
    )
    on_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in on_rows
    )
    off_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in off_rows
    )
    on_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in on_rows
    )
    return {
        "sample_count_per_policy": len(off_tpot),
        "off_tpot_median_ns": off_median,
        "on_tpot_median_ns": on_median,
        "tpot_median_improvement_fraction":
            _improvement(off_median, on_median),
        "off_tpot_p95_ns": off_p95,
        "on_tpot_p95_ns": on_p95,
        "tpot_p95_improvement_fraction":
            _improvement(off_p95, on_p95),
        "off_tpot_p99_ns": off_p99,
        "on_tpot_p99_ns": on_p99,
        "tpot_p99_improvement_fraction":
            _improvement(off_p99, on_p99),
        "off_ttft_median_ns": off_ttft,
        "on_ttft_median_ns": on_ttft,
        "ttft_regression_fraction":
            _relative_change(off_ttft, on_ttft),
        "off_e2e_median_ns": off_e2e,
        "on_e2e_median_ns": on_e2e,
        "e2e_regression_fraction":
            _relative_change(off_e2e, on_e2e),
        "off_output_tokens_per_second_median": off_rate,
        "on_output_tokens_per_second_median": on_rate,
        "throughput_regression_fraction":
            _relative_change(on_rate, off_rate),
        "off_cuda_peak_allocated_bytes": off_allocated,
        "on_cuda_peak_allocated_bytes": on_allocated,
        "cuda_allocated_delta_bytes": on_allocated - off_allocated,
        "off_cuda_peak_reserved_bytes": off_reserved,
        "on_cuda_peak_reserved_bytes": on_reserved,
        "cuda_reserved_delta_bytes": on_reserved - off_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(off_reserved, on_reserved),
    }


def _validate_performance_inventory(rows: list[dict]) -> list[dict]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly 30 measured rows, got {len(rows)}"
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
        for policy in ("off", "on")
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
    pair_rows = []
    total_abs = 0.0
    total_elements = 0
    global_max = 0.0
    worst_pair_mean = 0.0
    all_argmax_equal = True
    all_output_ids_exact = True
    all_output_text_exact = True
    for bucket, _prompt, _generated in context_cases():
        for point in (
            "prefill-final",
            "decode-first",
            "decode-final",
        ):
            off = by_identity[(bucket, point, "off")]
            on = by_identity[(bucket, point, "on")]
            if (
                off["logits_shape"] != on["logits_shape"]
                or off["logits_element_count"]
                != on["logits_element_count"]
            ):
                raise ValueError(
                    "paired logits shape mismatch"
                )
            off_values = read_float32_sidecar(
                run_dir,
                path=off["logits_path"],
                expected_element_count=off[
                    "logits_element_count"
                ],
                expected_byte_length=off[
                    "logits_byte_length"
                ],
                expected_sha256=off["logits_sha256"],
            )
            on_values = read_float32_sidecar(
                run_dir,
                path=on["logits_path"],
                expected_element_count=on[
                    "logits_element_count"
                ],
                expected_byte_length=on[
                    "logits_byte_length"
                ],
                expected_sha256=on["logits_sha256"],
            )
            differences = [
                abs(left - right)
                for left, right in zip(off_values, on_values)
            ]
            max_abs = max(differences)
            mean_abs = sum(differences) / len(differences)
            off_argmax = max(
                range(len(off_values)),
                key=off_values.__getitem__,
            )
            on_argmax = max(
                range(len(on_values)),
                key=on_values.__getitem__,
            )
            argmax_equal = off_argmax == on_argmax
            output_ids_exact = (
                off["output_token_ids"] == on["output_token_ids"]
            )
            output_text_exact = (
                off["output_text_sha256"]
                == on["output_text_sha256"]
            )
            global_max = max(global_max, max_abs)
            worst_pair_mean = max(worst_pair_mean, mean_abs)
            total_abs += sum(differences)
            total_elements += len(differences)
            all_argmax_equal = (
                all_argmax_equal and argmax_equal
            )
            all_output_ids_exact = (
                all_output_ids_exact and output_ids_exact
            )
            all_output_text_exact = (
                all_output_text_exact and output_text_exact
            )
            pair_rows.append({
                "context_bucket": bucket,
                "sampling_point": point,
                "element_count": len(differences),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "off_argmax": off_argmax,
                "on_argmax": on_argmax,
                "argmax_equal": argmax_equal,
                "output_ids_exact": output_ids_exact,
                "output_text_exact": output_text_exact,
            })
    return {
        "row_count": len(rows),
        "pair_count": len(pair_rows),
        "max_abs": global_max,
        "mean_abs": worst_pair_mean,
        "aggregate_mean_abs": total_abs / total_elements,
        "argmax_equal": all_argmax_equal,
        "output_ids_exact": all_output_ids_exact,
        "output_text_exact": all_output_text_exact,
        "pairs": pair_rows,
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
    optimized_complete = True
    for bucket, _prompt, generated_tokens in context_cases():
        for repetition in range(5):
            off = by_identity[(bucket, repetition, "off")]
            on = by_identity[(bucket, repetition, "on")]
            exact_outputs = exact_outputs and (
                off["output_token_ids"]
                == on["output_token_ids"]
                and off["output_text_sha256"]
                == on["output_text_sha256"]
            )
            off_summary = off["greedy_fast_path_summary"]
            on_summary = on["greedy_fast_path_summary"]
            optimized_complete = optimized_complete and (
                off_summary["eligible_steps"] == 0
                and off_summary["optimized_steps"] == 0
                and on_summary["eligible_steps"] == generated_tokens
                and on_summary["optimized_steps"] == generated_tokens
                and not on_summary["fallback_counts"]
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
    by_bucket = {}
    for bucket, _prompt, _generated in context_cases():
        selected = [
            row
            for row in validated
            if row["context_bucket"] == bucket
        ]
        by_bucket[bucket] = _metric_summary(
            [
                row for row in selected
                if row["policy"] == "off"
            ],
            [
                row for row in selected
                if row["policy"] == "on"
            ],
        )
    aggregate = _metric_summary(
        [row for row in validated if row["policy"] == "off"],
        [row for row in validated if row["policy"] == "on"],
    )
    winning_buckets = sum(
        metric["tpot_median_improvement_fraction"]
        >= MEDIAN_TPOT_MIN_IMPROVEMENT
        for metric in by_bucket.values()
    )
    protected_regressions = []
    for bucket, metric in by_bucket.items():
        if (
            metric["tpot_median_improvement_fraction"]
            < -TPOT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:median_tpot"
            )
        if (
            metric["tpot_p95_improvement_fraction"]
            < -TPOT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:p95_tpot"
            )
        if (
            metric["ttft_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            protected_regressions.append(f"{bucket}:ttft")
        if (
            metric["e2e_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            protected_regressions.append(f"{bucket}:e2e")
        if (
            metric["throughput_regression_fraction"]
            > THROUGHPUT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:throughput"
            )
    if (
        aggregate["cuda_reserved_regression_fraction"]
        > RESERVED_MEMORY_MAX_REGRESSION
    ):
        protected_regressions.append(
            "aggregate:cuda_reserved"
        )
    if not correctness_passed:
        classification = "NO_GO_CORRECTNESS"
    elif not optimized_complete:
        classification = (
            "NO_GO_OPTIMIZED_PATH_INCOMPLETE"
        )
    elif winning_buckets < 2:
        classification = "NO_GO_TPOT_MEDIAN"
    elif (
        aggregate["tpot_p95_improvement_fraction"]
        < AGGREGATE_P95_MIN_IMPROVEMENT
    ):
        classification = "NO_GO_TPOT_P95"
    elif protected_regressions:
        classification = "NO_GO_PROTECTED_REGRESSION"
    else:
        classification = (
            "GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH"
        )
    on_summaries = [
        row["greedy_fast_path_summary"]
        for row in validated
        if row["policy"] == "on"
    ]
    avoided_work = {
        field: sum(summary[field] for summary in on_summaries)
        for field in (
            "avoided_temperature_h2d_bytes",
            "avoided_softmax_calls",
            "avoided_gumbel_rng_calls",
            "avoided_stochastic_divisions",
            "avoided_stochastic_argmax_calls",
            "avoided_where_calls",
        )
    }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_tag": validated[0]["run_tag"],
        "source_commit": validated[0]["source_commit"],
        "classification": classification,
        "correctness_passed": correctness_passed,
        "optimized_path_complete": optimized_complete,
        "median_tpot_winning_bucket_count": winning_buckets,
        "protected_regressions": protected_regressions,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_ABS_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_ABS_LIMIT,
            "median_tpot_min_improvement_fraction":
                MEDIAN_TPOT_MIN_IMPROVEMENT,
            "aggregate_p95_min_improvement_fraction":
                AGGREGATE_P95_MIN_IMPROVEMENT,
            "tpot_max_regression_fraction":
                TPOT_MAX_REGRESSION,
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
        "cost": {
            "persistent_cuda_memory_delta_bytes": 0,
            "host_counter_integer_fields": 8,
            "host_fallback_counter_mapping": True,
            "avoided_work": avoided_work,
            "cuda_peak_allocated_delta_bytes":
                aggregate["cuda_allocated_delta_bytes"],
            "cuda_peak_reserved_delta_bytes":
                aggregate["cuda_reserved_delta_bytes"],
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
        != "zero-temperature-greedy-fast-path.source.v1"
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
        != "zero-temperature-greedy-fast-path.workload.v1"
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
    if manifest.get("context_cases") != expected_cases:
        raise ValueError("workload context cases mismatch")
    required_values = {
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "policy_order": {
            str(index): list(policy_order(index))
            for index in range(5)
        },
        "correctness_sampling_points": [
            "prefill-final",
            "decode-first",
            "decode-final",
        ],
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
    expected_summary = summarize_rows(
        rows,
        require_complete_optimized_path=False,
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
        row["logits_path"] for row in correctness_rows
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
