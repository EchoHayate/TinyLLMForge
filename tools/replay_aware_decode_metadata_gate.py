#!/usr/bin/env python3
"""Producer gate for replay-aware decode metadata benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.profile_replay_aware_decode_metadata import (
    SOURCE_FILES,
    context_cases,
    nearest_rank_percentile,
    policy_order,
    sha256_file,
    summarize_rows,
    validate_case_row,
)


COMPARISON_SCHEMA_VERSION = (
    "replay-aware-decode-metadata.comparison.v1"
)
GATE_SCHEMA_VERSION = "replay-aware-decode-metadata.gate.v1"
MANIFEST_SCHEMA_VERSION = (
    "replay-aware-decode-metadata.manifest.v1"
)
CLASSIFICATIONS = (
    "GO_REPLAY_AWARE_METADATA",
    "NO_GO_CORRECTNESS",
    "NO_GO_OPTIMIZED_PATH_INCOMPLETE",
    "NO_GO_TPOT_MEDIAN",
    "NO_GO_TPOT_P95",
    "NO_GO_PROTECTED_REGRESSION",
    "NO_GO_EVIDENCE_INCOMPLETE",
)
PRIMARY_ARTIFACTS = (
    "case_rows.jsonl",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
)
MEDIAN_TPOT_MIN_IMPROVEMENT = 0.05
AGGREGATE_P95_MIN_IMPROVEMENT = 0.05
TPOT_MAX_REGRESSION = 0.03
LATENCY_MAX_REGRESSION = 0.03
THROUGHPUT_MAX_REGRESSION = 0.02
RESERVED_MEMORY_MAX_REGRESSION = 0.01
MAX_PINNED_CAPACITY_BYTES = 1_792
EXPECTED_ROWS = 3 * 5 * 2


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=lambda value: (
                    (_ for _ in ()).throw(
                        ValueError(
                            f"non-finite JSON value: {value}"
                        )
                    )
                ),
            )
    except FileNotFoundError as error:
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        ) from error


def _read_jsonl(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return [
                json.loads(
                    line,
                    parse_constant=lambda value: (
                        (_ for _ in ()).throw(
                            ValueError(
                                "non-finite JSON value: "
                                f"{value}"
                            )
                        )
                    ),
                )
                for line in handle
                if line.strip()
            ]
    except FileNotFoundError as error:
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        ) from error


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


def _relative_change(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError(
            "relative comparison baseline must be positive"
        )
    return (candidate - baseline) / baseline


def _improvement(
    baseline: float,
    candidate: float,
) -> float:
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
    off_tpot_median = statistics.median(off_tpot)
    on_tpot_median = statistics.median(on_tpot)
    off_tpot_p95 = nearest_rank_percentile(
        off_tpot,
        0.95,
    )
    on_tpot_p95 = nearest_rank_percentile(
        on_tpot,
        0.95,
    )
    off_tpot_p99 = nearest_rank_percentile(
        off_tpot,
        0.99,
    )
    on_tpot_p99 = nearest_rank_percentile(
        on_tpot,
        0.99,
    )
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
    off_throughput = statistics.median(
        float(row["output_tokens_per_second"])
        for row in off_rows
    )
    on_throughput = statistics.median(
        float(row["output_tokens_per_second"])
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
    off_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in off_rows
    )
    on_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in on_rows
    )
    return {
        "sample_count_per_policy": len(off_tpot),
        "off_tpot_median_ns": off_tpot_median,
        "on_tpot_median_ns": on_tpot_median,
        "tpot_median_improvement_fraction": _improvement(
            off_tpot_median,
            on_tpot_median,
        ),
        "off_tpot_p95_ns": off_tpot_p95,
        "on_tpot_p95_ns": on_tpot_p95,
        "tpot_p95_improvement_fraction": _improvement(
            off_tpot_p95,
            on_tpot_p95,
        ),
        "off_tpot_p99_ns": off_tpot_p99,
        "on_tpot_p99_ns": on_tpot_p99,
        "tpot_p99_improvement_fraction": _improvement(
            off_tpot_p99,
            on_tpot_p99,
        ),
        "off_ttft_median_ns": off_ttft,
        "on_ttft_median_ns": on_ttft,
        "ttft_regression_fraction": _relative_change(
            off_ttft,
            on_ttft,
        ),
        "off_e2e_median_ns": off_e2e,
        "on_e2e_median_ns": on_e2e,
        "e2e_regression_fraction": _relative_change(
            off_e2e,
            on_e2e,
        ),
        "off_output_tokens_per_second_median":
            off_throughput,
        "on_output_tokens_per_second_median":
            on_throughput,
        "throughput_regression_fraction": (
            _relative_change(
                on_throughput,
                off_throughput,
            )
        ),
        "off_cuda_peak_allocated_bytes": off_allocated,
        "on_cuda_peak_allocated_bytes": on_allocated,
        "off_cuda_peak_reserved_bytes": off_reserved,
        "on_cuda_peak_reserved_bytes": on_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(off_reserved, on_reserved),
    }


def _validate_inventory(rows: list[dict]) -> list[dict]:
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
    if len({row["run_tag"] for row in validated}) != 1:
        raise ValueError("run tag inventory mismatch")
    if len(
        {row["source_commit"] for row in validated}
    ) != 1:
        raise ValueError("source commit inventory mismatch")
    return validated


def classify(rows: list[dict]) -> dict:
    validated = _validate_inventory(rows)
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
            expected_steps = generated_tokens - 1
            landing = on["landing_summary"]
            optimized_complete = optimized_complete and (
                landing["eligible_steps"] == expected_steps
                and landing["optimized_steps"] == expected_steps
                and not landing["fallback_counts"]
            )
    by_bucket = {}
    for bucket, _prompt, _generated in context_cases():
        bucket_rows = [
            row
            for row in validated
            if row["context_bucket"] == bucket
        ]
        by_bucket[bucket] = _metric_summary(
            [
                row for row in bucket_rows
                if row["policy"] == "off"
            ],
            [
                row for row in bucket_rows
                if row["policy"] == "on"
            ],
        )
    aggregate = _metric_summary(
        [
            row for row in validated
            if row["policy"] == "off"
        ],
        [
            row for row in validated
            if row["policy"] == "on"
        ],
    )
    winning_buckets = sum(
        metrics[
            "tpot_median_improvement_fraction"
        ] >= MEDIAN_TPOT_MIN_IMPROVEMENT
        for metrics in by_bucket.values()
    )
    pinned_peak = max(
        row["landing_summary"][
            "peak_pinned_capacity_bytes"
        ]
        for row in validated
        if row["policy"] == "on"
    )
    protected_regressions = []
    for bucket, metrics in by_bucket.items():
        if (
            metrics[
                "tpot_median_improvement_fraction"
            ] < -TPOT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:median_tpot"
            )
        if (
            metrics["tpot_p95_improvement_fraction"]
            < -TPOT_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:p95_tpot"
            )
        if (
            metrics["ttft_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:ttft"
            )
        if (
            metrics["e2e_regression_fraction"]
            > LATENCY_MAX_REGRESSION
        ):
            protected_regressions.append(
                f"{bucket}:e2e"
            )
        if (
            metrics["throughput_regression_fraction"]
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
    if pinned_peak > MAX_PINNED_CAPACITY_BYTES:
        protected_regressions.append(
            "aggregate:pinned_capacity"
        )
    if not exact_outputs:
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
        classification = "GO_REPLAY_AWARE_METADATA"
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "run_tag": validated[0]["run_tag"],
        "source_commit": validated[0]["source_commit"],
        "classification": classification,
        "exact_outputs": exact_outputs,
        "optimized_path_complete": optimized_complete,
        "median_tpot_winning_bucket_count":
            winning_buckets,
        "pinned_peak_bytes": pinned_peak,
        "pinned_capacity_limit_bytes":
            MAX_PINNED_CAPACITY_BYTES,
        "protected_regressions": protected_regressions,
        "thresholds": {
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
        "by_bucket": by_bucket,
        "aggregate": aggregate,
    }


def _validate_source_manifest(
    manifest,
    *,
    repo_root: Path,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "replay-aware-decode-metadata.source.v1"
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
        actual = sha256_file(path)
        if digests[relative] != actual:
            raise ValueError(
                f"source digest mismatch: {relative}"
            )


def _validate_workload_manifest(manifest) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "replay-aware-decode-metadata.workload.v1"
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
        raise ValueError(
            "workload context cases mismatch"
        )
    expected_order = {
        str(index): list(policy_order(index))
        for index in range(5)
    }
    required_values = {
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "policy_order": expected_order,
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
    repo_root = Path(repo_root)
    rows = _read_jsonl(run_dir / "case_rows.jsonl")
    source = _read_json(
        run_dir / "source_manifest.json"
    )
    workload = _read_json(
        run_dir / "workload_manifest.json"
    )
    summary = _read_json(run_dir / "summary.json")
    _validate_source_manifest(
        source,
        repo_root=repo_root,
    )
    _validate_workload_manifest(workload)
    identities = {
        source.get("run_tag"),
        workload.get("run_tag"),
        *(row.get("run_tag") for row in rows),
    }
    commits = {
        source.get("source_commit"),
        workload.get("source_commit"),
        *(row.get("source_commit") for row in rows),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError(
            "source-bound identity mismatch"
        )
    expected_summary = summarize_rows(rows)
    if summary != expected_summary:
        raise ValueError("worker summary drift")
    comparison = classify(rows)
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "classification": comparison["classification"],
        "comparison_sha256": hashlib.sha256(
            json.dumps(
                comparison,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(
        run_dir / "comparison.json",
        comparison,
    )
    _write_json(run_dir / "gate.json", gate)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_tag": comparison["run_tag"],
        "source_commit": comparison["source_commit"],
        "artifacts": {
            name: sha256_file(run_dir / name)
            for name in PRIMARY_ARTIFACTS
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
