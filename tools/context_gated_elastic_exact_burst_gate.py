#!/usr/bin/env python3
"""Terminal paired gate for context-gated elastic exact burst."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics

from tools import profile_context_gated_elastic_exact_burst as profile


SCHEMA_VERSION = "context-gated-elastic-exact-burst.terminal.v1"
MANIFEST_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.terminal-manifest.v1"
)
TERMINAL_REPETITIONS = 5
PERFORMANCE_ROW_COUNT = 40
CORRECTNESS_ROW_COUNT = 32

MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT = 2.0
MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT = 1.0
MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT = 2.0
MAXIMUM_LATENCY_REGRESSION_PCT = 2.0
MAXIMUM_THROUGHPUT_REGRESSION_PCT = 1.0
MAXIMUM_MEMORY_REGRESSION_PCT = 3.0
MAXIMUM_K16_HOST_VISIBLE_GAP_NS = 40_000_000

GO_CONTEXT_GATED_ELASTIC_EXACT_BURST = (
    "GO_CONTEXT_GATED_ELASTIC_EXACT_BURST"
)
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_WIDTH_POLICY = "NO_GO_WIDTH_POLICY"
NO_GO_RUNTIME_INVARIANT = "NO_GO_RUNTIME_INVARIANT"
NO_GO_BURST_GAP = "NO_GO_BURST_GAP"
NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT = (
    "NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT"
)
NO_GO_PROTECTED_REGRESSION = "NO_GO_PROTECTED_REGRESSION"

SOURCE_FILES = tuple(dict.fromkeys((
    *profile.SOURCE_FILES,
    "tools/context_gated_elastic_exact_burst_ceiling.py",
    "tools/test_context_gated_elastic_exact_burst_ceiling.py",
    "tools/context_gated_elastic_exact_burst_gate.py",
    "tools/test_context_gated_elastic_exact_burst_gate.py",
    "tools/context_gated_elastic_exact_burst_verify.py",
    "tools/test_context_gated_elastic_exact_burst_verify.py",
    "tools/run_context_gated_elastic_exact_burst_remote.py",
    "tools/test_run_context_gated_elastic_exact_burst_remote.py",
)))
PRIMARY_ARTIFACTS = (
    "workload_manifest.json",
    "source_manifest.json",
    "source.patch",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "profile_summary.json",
    "terminal_source_manifest.json",
    "terminal_summary.json",
    "terminal_gate.json",
    "producer_receipt.json",
)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _validate_digest(
    value,
    field: str,
    lengths: tuple[int, ...] = (64,),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(_finite(value, "metric sample") for value in values)
    if not ordered:
        raise ValueError("metric sample inventory is empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _relative_change_pct(control: float, candidate: float) -> float:
    control = _finite(control, "control metric")
    candidate = _finite(candidate, "candidate metric")
    if control <= 0.0:
        if candidate == control:
            return 0.0
        raise ValueError("control metric must be positive")
    return (candidate - control) / control * 100.0


def _improvement_pct(control: float, candidate: float) -> float:
    return -_relative_change_pct(control, candidate)


def expected_performance_identities(
) -> tuple[tuple[int, int, str], ...]:
    return profile.performance_identities(
        repetitions=TERMINAL_REPETITIONS,
    )


def expected_correctness_identities(
) -> tuple[tuple[int, str, str], ...]:
    return profile.correctness_identities()


def _inventory_status(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> tuple[bool, str | None]:
    if not all(isinstance(row, dict) for row in performance_rows):
        return False, "performance row payload is not an object"
    if not all(isinstance(row, dict) for row in correctness_rows):
        return False, "correctness row payload is not an object"
    performance = [
        (
            row.get("repetition"),
            row.get("context_length"),
            row.get("policy"),
        )
        for row in performance_rows
    ]
    correctness = [
        (
            row.get("context_length"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        for row in correctness_rows
    ]
    if (
        len(performance) != len(set(performance))
        or len(correctness) != len(set(correctness))
    ):
        return False, "duplicate row identity"
    if set(performance) != set(expected_performance_identities()):
        return False, "performance row inventory is incomplete"
    if set(correctness) != set(expected_correctness_identities()):
        return False, "correctness row inventory is incomplete"
    return True, None


def _selected_k16(summary: dict) -> bool:
    return (
        summary.get("k16_acceptances", 0) > 0
        and summary.get("authorized_width_histogram", {}).get("16", 0) > 0
        and summary.get("per_width_commits", {}).get("16", 0) > 0
    )


def _width_policy_exact(rows: list[dict]) -> bool:
    for row in rows:
        summary = row.get("exact_greedy_decode_burst_summary")
        if not isinstance(summary, dict):
            return False
        selected = _selected_k16(summary)
        expected = (
            row.get("policy") == "context_gated_elastic_k16"
            and row.get("context_length") in (256, 2048)
        )
        if selected is not expected:
            return False
    return True


def _runtime_inventory_exact(rows: list[dict]) -> bool:
    for row in rows:
        summary = row.get("exact_greedy_decode_burst_summary")
        if not isinstance(summary, dict):
            return False
        if not (
            summary.get("target_model_forwards")
            == summary.get("graph_replays")
            == summary.get("committed_tokens")
            == profile.GENERATED_TOKENS - 1
        ):
            return False
        if summary.get("intermediate_token_d2h_calls") != 0:
            return False
        if summary.get("final_token_d2h_calls") != summary.get("commits"):
            return False
        if (
            summary.get("final_token_d2h_bytes")
            != summary.get("committed_tokens", -1) * 8
        ):
            return False
    return True


def _expected_elastic_fallbacks(row: dict) -> set[str]:
    if row.get("policy") != "context_gated_elastic_k16":
        return set()
    if row.get("context_length") in (256, 2048):
        return {"output_budget_below_16"}
    return {"context_above_2048"}


def _zero_unexpected_lifecycle_events(rows: list[dict]) -> bool:
    for row in rows:
        summary = row.get("exact_greedy_decode_burst_summary")
        if not isinstance(summary, dict):
            return False
        if (
            summary.get("failures") != 0
            or summary.get("quarantines") != 0
            or summary.get("pending_leases") != 0
            or summary.get("quarantine_reason") is not None
            or summary.get("lease_local_delta_journal_rollbacks") != 0
            or summary.get(
                "lease_local_delta_journal_one_phase_rollbacks"
            ) != 0
            or summary.get("fallback_counts") != {}
        ):
            return False
        if row.get("policy") == "fixed_k8" and (
            summary.get("k16_attempts") != 0
            or summary.get("k16_acceptances") != 0
            or summary.get("k8_fallbacks") != 0
            or summary.get("elastic_k16_fallback_counts") != {}
        ):
            return False
        elastic_fallbacks = summary.get(
            "elastic_k16_fallback_counts"
        )
        if (
            not isinstance(elastic_fallbacks, dict)
            or set(elastic_fallbacks)
            - _expected_elastic_fallbacks(row)
        ):
            return False
        for field in (
            "lease_local_delta_journal_fallback_counts",
            "lease_local_delta_journal_one_phase_fallback_counts",
        ):
            fallbacks = summary.get(field)
            if (
                not isinstance(fallbacks, dict)
                or set(fallbacks) - {"unsupported_burst_shape"}
            ):
                return False
    return True


def _metric_summary(
    control_rows: list[dict],
    candidate_rows: list[dict],
) -> dict:
    control_tpot = [
        _finite(sample, "control TPOT")
        for row in control_rows
        for sample in row["amortized_tpot_samples_ns"]
    ]
    candidate_tpot = [
        _finite(sample, "candidate TPOT")
        for row in candidate_rows
        for sample in row["amortized_tpot_samples_ns"]
    ]
    control_median = statistics.median(control_tpot)
    candidate_median = statistics.median(candidate_tpot)
    control_p95 = _nearest_rank(control_tpot, 0.95)
    candidate_p95 = _nearest_rank(candidate_tpot, 0.95)
    control_p99 = _nearest_rank(control_tpot, 0.99)
    candidate_p99 = _nearest_rank(candidate_tpot, 0.99)
    control_ttft = statistics.median(
        _finite(row["ttft_ns"], "control TTFT")
        for row in control_rows
    )
    candidate_ttft = statistics.median(
        _finite(row["ttft_ns"], "candidate TTFT")
        for row in candidate_rows
    )
    control_e2e = statistics.median(
        _finite(row["e2e_ns"], "control E2E")
        for row in control_rows
    )
    candidate_e2e = statistics.median(
        _finite(row["e2e_ns"], "candidate E2E")
        for row in candidate_rows
    )
    control_throughput = statistics.median(
        _finite(
            row["output_tokens_per_second"],
            "control throughput",
        )
        for row in control_rows
    )
    candidate_throughput = statistics.median(
        _finite(
            row["output_tokens_per_second"],
            "candidate throughput",
        )
        for row in candidate_rows
    )
    control_allocated = max(
        _finite(
            row["cuda_peak_allocated_bytes"],
            "control allocated memory",
        )
        for row in control_rows
    )
    candidate_allocated = max(
        _finite(
            row["cuda_peak_allocated_bytes"],
            "candidate allocated memory",
        )
        for row in candidate_rows
    )
    control_reserved = max(
        _finite(
            row["cuda_peak_reserved_bytes"],
            "control reserved memory",
        )
        for row in control_rows
    )
    candidate_reserved = max(
        _finite(
            row["cuda_peak_reserved_bytes"],
            "candidate reserved memory",
        )
        for row in candidate_rows
    )
    return {
        "sample_count_per_policy": len(control_tpot),
        "control_tpot_median_ns": control_median,
        "candidate_tpot_median_ns": candidate_median,
        "tpot_median_improvement_pct": _improvement_pct(
            control_median,
            candidate_median,
        ),
        "control_tpot_p95_ns": control_p95,
        "candidate_tpot_p95_ns": candidate_p95,
        "tpot_p95_improvement_pct": _improvement_pct(
            control_p95,
            candidate_p95,
        ),
        "control_tpot_p99_ns": control_p99,
        "candidate_tpot_p99_ns": candidate_p99,
        "tpot_p99_regression_pct": _relative_change_pct(
            control_p99,
            candidate_p99,
        ),
        "control_ttft_median_ns": control_ttft,
        "candidate_ttft_median_ns": candidate_ttft,
        "ttft_regression_pct": _relative_change_pct(
            control_ttft,
            candidate_ttft,
        ),
        "control_e2e_median_ns": control_e2e,
        "candidate_e2e_median_ns": candidate_e2e,
        "e2e_regression_pct": _relative_change_pct(
            control_e2e,
            candidate_e2e,
        ),
        "control_throughput_median": control_throughput,
        "candidate_throughput_median": candidate_throughput,
        "throughput_regression_pct": _relative_change_pct(
            candidate_throughput,
            control_throughput,
        ),
        "control_cuda_peak_allocated_bytes": control_allocated,
        "candidate_cuda_peak_allocated_bytes": candidate_allocated,
        "allocated_memory_regression_pct": _relative_change_pct(
            control_allocated,
            candidate_allocated,
        ),
        "control_cuda_peak_reserved_bytes": control_reserved,
        "candidate_cuda_peak_reserved_bytes": candidate_reserved,
        "reserved_memory_regression_pct": _relative_change_pct(
            control_reserved,
            candidate_reserved,
        ),
    }


def classify(metrics: dict) -> str:
    if metrics.get("evidence_complete") is not True:
        return NO_GO_EVIDENCE_INCOMPLETE
    if metrics.get("correctness_exact") is not True:
        return NO_GO_CORRECTNESS
    if metrics.get("width_policy_exact") is not True:
        return NO_GO_WIDTH_POLICY
    if (
        metrics.get("runtime_inventory_exact") is not True
        or metrics.get("zero_unexpected_lifecycle_events") is not True
    ):
        return NO_GO_RUNTIME_INVARIANT
    gap = metrics.get("maximum_selected_k16_host_visible_gap_ns")
    if (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not math.isfinite(float(gap))
        or float(gap) > MAXIMUM_K16_HOST_VISIBLE_GAP_NS
    ):
        return NO_GO_BURST_GAP
    eligible = metrics.get("eligible_aggregate")
    if not isinstance(eligible, dict):
        return NO_GO_EVIDENCE_INCOMPLETE
    if (
        _finite(
            eligible.get("tpot_median_improvement_pct"),
            "eligible median TPOT improvement",
        )
        < MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT
        or _finite(
            eligible.get("tpot_p95_improvement_pct"),
            "eligible P95 TPOT improvement",
        )
        < MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT
    ):
        return NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT
    by_context = metrics.get("by_context")
    if (
        not isinstance(by_context, dict)
        or set(by_context)
        != {str(value) for value in profile.CONTEXT_LENGTHS}
    ):
        return NO_GO_EVIDENCE_INCOMPLETE
    for context_metrics in by_context.values():
        if (
            context_metrics["tpot_median_improvement_pct"]
            < -MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT
            or context_metrics["tpot_p95_improvement_pct"]
            < -MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT
            or context_metrics["tpot_p99_regression_pct"]
            > MAXIMUM_LATENCY_REGRESSION_PCT
            or context_metrics["ttft_regression_pct"]
            > MAXIMUM_LATENCY_REGRESSION_PCT
            or context_metrics["e2e_regression_pct"]
            > MAXIMUM_LATENCY_REGRESSION_PCT
            or context_metrics["throughput_regression_pct"]
            > MAXIMUM_THROUGHPUT_REGRESSION_PCT
            or context_metrics["allocated_memory_regression_pct"]
            > MAXIMUM_MEMORY_REGRESSION_PCT
            or context_metrics["reserved_memory_regression_pct"]
            > MAXIMUM_MEMORY_REGRESSION_PCT
        ):
            return NO_GO_PROTECTED_REGRESSION
    return GO_CONTEXT_GATED_ELASTIC_EXACT_BURST


def summarize_evidence(
    performance_rows: list[dict],
    correctness_rows: list[dict],
    *,
    run_dir: Path,
) -> dict:
    complete, reason = _inventory_status(
        performance_rows,
        correctness_rows,
    )
    if not complete:
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "evidence_complete": False,
            "evidence_error": reason,
            "performance_row_count": len(performance_rows),
            "correctness_row_count": len(correctness_rows),
        }
        metrics["classification"] = classify(metrics)
        return metrics

    width_policy_exact = _width_policy_exact(performance_rows)
    runtime_inventory_exact = _runtime_inventory_exact(
        performance_rows
    )
    lifecycle_exact = _zero_unexpected_lifecycle_events(
        performance_rows
    )
    try:
        validated_performance = [
            profile.validate_case_row(row)
            for row in performance_rows
        ]
        profile_summary = profile.summarize_rows(
            validated_performance,
            expected_repetitions=TERMINAL_REPETITIONS,
        )
        validated_correctness = profile.validate_correctness_rows(
            correctness_rows,
            run_dir=run_dir,
        )
        correctness_exact = (
            profile_summary["all_outputs_exact"] is True
            and len(validated_correctness) == CORRECTNESS_ROW_COUNT
        )
        evidence_error = None
    except (KeyError, TypeError, ValueError) as error:
        routed_runtime_failure = (
            not width_policy_exact
            or not runtime_inventory_exact
            or not lifecycle_exact
        )
        if (
            not routed_runtime_failure
            and str(error) != "correctness policy mismatch"
        ):
            raise
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "evidence_complete": True,
            "evidence_error": str(error),
            "performance_row_count": len(performance_rows),
            "correctness_row_count": len(correctness_rows),
            "correctness_exact": routed_runtime_failure,
            "width_policy_exact": width_policy_exact,
            "runtime_inventory_exact": runtime_inventory_exact,
            "zero_unexpected_lifecycle_events": lifecycle_exact,
        }
        metrics["classification"] = classify(metrics)
        return metrics

    by_context_policy: dict[tuple[int, str], list[dict]] = {}
    for row in validated_performance:
        by_context_policy.setdefault(
            (row["context_length"], row["policy"]),
            [],
        ).append(row)
    by_context = {
        str(context): _metric_summary(
            by_context_policy[(context, "fixed_k8")],
            by_context_policy[
                (context, "context_gated_elastic_k16")
            ],
        )
        for context in profile.CONTEXT_LENGTHS
    }
    eligible_control = [
        row for row in validated_performance
        if row["policy"] == "fixed_k8"
        and row["context_length"] in (256, 2048)
    ]
    eligible_candidate = [
        row for row in validated_performance
        if row["policy"] == "context_gated_elastic_k16"
        and row["context_length"] in (256, 2048)
    ]
    all_control = [
        row for row in validated_performance
        if row["policy"] == "fixed_k8"
    ]
    all_candidate = [
        row for row in validated_performance
        if row["policy"] == "context_gated_elastic_k16"
    ]
    selected_k16_rows = [
        row for row in eligible_candidate
        if _selected_k16(row["exact_greedy_decode_burst_summary"])
    ]
    selected_gaps = [
        int(gap)
        for row in selected_k16_rows
        for gap in row["host_visible_burst_gaps_ns"]
    ]
    candidate_attempts = sum(
        row["exact_greedy_decode_burst_summary"]["attempts"]
        for row in all_candidate
    )
    candidate_k8_fallbacks = sum(
        row["exact_greedy_decode_burst_summary"]["k8_fallbacks"]
        for row in all_candidate
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validated_performance[0]["run_tag"],
        "source_commit": validated_performance[0]["source_commit"],
        "evidence_complete": True,
        "evidence_error": evidence_error,
        "performance_row_count": len(validated_performance),
        "correctness_row_count": len(validated_correctness),
        "correctness_exact": correctness_exact,
        "width_policy_exact": width_policy_exact,
        "runtime_inventory_exact": runtime_inventory_exact,
        "zero_unexpected_lifecycle_events": lifecycle_exact,
        "eligible_aggregate": _metric_summary(
            eligible_control,
            eligible_candidate,
        ),
        "overall": _metric_summary(all_control, all_candidate),
        "by_context": by_context,
        "maximum_selected_k16_host_visible_gap_ns": max(
            selected_gaps,
            default=0,
        ),
        "p95_selected_k16_host_visible_gap_ns": (
            _nearest_rank(selected_gaps, 0.95)
            if selected_gaps
            else 0
        ),
        "shared_capture_duration_ns_by_policy": {
            policy: max(
                row["shared_capture_duration_ns"]
                for row in validated_performance
                if row["policy"] == policy
            )
            for policy in profile.POLICIES
        },
        "elastic_incremental_capture_duration_ns": 0,
        "elastic_incremental_retained_static_bytes": max(
            row["elastic_incremental_retained_static_bytes"]
            for row in eligible_candidate
        ),
        "elastic_incremental_allocated_bytes": max(
            row["elastic_incremental_allocated_bytes"]
            for row in eligible_candidate
        ),
        "elastic_incremental_reserved_bytes": max(
            row["elastic_incremental_reserved_bytes"]
            for row in eligible_candidate
        ),
        "candidate_k8_fallback_count": candidate_k8_fallbacks,
        "candidate_attempt_count": candidate_attempts,
        "k8_fallback_rate": (
            candidate_k8_fallbacks / candidate_attempts
            if candidate_attempts
            else 0.0
        ),
        "k16_width_health_quarantine_count": sum(
            row["exact_greedy_decode_burst_summary"]["quarantines"]
            for row in all_candidate
        ),
        "lifecycle_totals": {
            field: sum(
                row["exact_greedy_decode_burst_summary"][field]
                for row in all_candidate
            )
            for field in (
                "attempts",
                "acceptances",
                "commits",
                "committed_tokens",
                "target_model_forwards",
                "graph_replays",
                "intermediate_token_d2h_calls",
                "final_token_d2h_calls",
                "final_token_d2h_bytes",
                "failures",
                "quarantines",
                "lease_local_delta_journal_rollbacks",
                "lease_local_delta_journal_one_phase_rollbacks",
            )
        },
        "thresholds": {
            "minimum_eligible_median_tpot_improvement_pct":
                MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT,
            "minimum_eligible_p95_tpot_improvement_pct":
                MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT,
            "maximum_per_context_tpot_regression_pct":
                MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT,
            "maximum_latency_regression_pct":
                MAXIMUM_LATENCY_REGRESSION_PCT,
            "maximum_throughput_regression_pct":
                MAXIMUM_THROUGHPUT_REGRESSION_PCT,
            "maximum_memory_regression_pct":
                MAXIMUM_MEMORY_REGRESSION_PCT,
            "maximum_k16_host_visible_gap_ns":
                MAXIMUM_K16_HOST_VISIBLE_GAP_NS,
        },
    }
    metrics["classification"] = classify(metrics)
    return metrics


def build_source_manifest(
    *,
    source_root: Path,
    run_tag: str,
    source_commit: str,
) -> dict:
    root = Path(source_root).resolve()
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run tag is invalid")
    _validate_digest(source_commit, "source commit", (40, 64))
    hashes = {}
    for relative in SOURCE_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"terminal source file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_sha256": hashes,
    }


def _verify_source_files(manifest: dict, source_root: Path) -> None:
    root = Path(source_root).resolve()
    hashes = manifest.get("source_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("source hashes are invalid")
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("source manifest path is invalid")
        if sha256_file(path) != expected:
            raise ValueError(f"source hash mismatch: {relative}")


def _safe_artifact_path(run_dir: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not isinstance(relative, str)
        or not relative
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError("artifact path is invalid")
    root = Path(run_dir).resolve()
    resolved = (root / relative).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"artifact is missing: {relative}")
    return resolved


def _manifest_authority(run_dir: Path) -> tuple[dict, dict]:
    workload = read_json(run_dir / "workload_manifest.json")
    source = read_json(run_dir / "source_manifest.json")
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version")
        != profile.WORKLOAD_SCHEMA_VERSION
        or not isinstance(workload.get("run_tag"), str)
        or not workload["run_tag"]
        or not isinstance(workload.get("model"), str)
        or not workload["model"]
        or workload.get("device") != "cuda:0"
        or workload.get("repetitions") != TERMINAL_REPETITIONS
        or workload.get("warmup_repetitions")
        != profile.WARMUP_REPETITIONS
        or workload.get("performance_row_count")
        != PERFORMANCE_ROW_COUNT
        or workload.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
        or workload.get("contexts") != list(profile.CONTEXT_LENGTHS)
        or workload.get("policies") != list(profile.POLICIES)
        or workload.get("generated_tokens")
        != profile.GENERATED_TOKENS
        or workload.get("temperature") != 0.0
        or workload.get("ignore_eos") is not True
        or workload.get("tensor_parallel_size") != 1
        or workload.get("max_num_seqs") != 1
        or workload.get("completion_only") is not True
        or (
            isinstance(workload.get("gpu_memory_utilization"), bool)
            or not isinstance(
                workload.get("gpu_memory_utilization"),
                (int, float),
            )
            or not 0.0
            < float(workload["gpu_memory_utilization"])
            <= 1.0
        )
        or not isinstance(workload.get("environment"), dict)
        or not workload["environment"]
    ):
        raise ValueError("workload manifest inventory mismatch")
    _validate_digest(
        workload.get("source_commit"),
        "workload source commit",
        (40, 64),
    )
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != profile.SOURCE_SCHEMA_VERSION
        or workload.get("run_tag") != source.get("run_tag")
        or workload.get("source_commit") != source.get("source_commit")
    ):
        raise ValueError("manifest source authority mismatch")
    hashes = source.get("source_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(profile.SOURCE_FILES)
        or any(
            _validate_digest(digest, "source hash") != digest
            for digest in hashes.values()
        )
    ):
        raise ValueError("source manifest inventory mismatch")
    patch = run_dir / "source.patch"
    if not patch.is_file() or patch.read_bytes() != b"":
        raise ValueError("source patch must be empty")
    return workload, source


def produce_artifacts(
    run_dir: Path,
    *,
    source_root: Path,
) -> dict:
    root = Path(run_dir)
    workload, source = _manifest_authority(root)
    _verify_source_files(source, source_root)
    performance = read_jsonl(root / "performance_rows.jsonl")
    correctness = read_jsonl(root / "correctness_rows.jsonl")
    worker_summary = read_json(root / "profile_summary.json")
    expected_worker_summary = profile.summarize_rows(
        performance,
        expected_repetitions=TERMINAL_REPETITIONS,
    )
    expected_worker_summary["correctness_row_count"] = len(correctness)
    if worker_summary != expected_worker_summary:
        raise ValueError("profile summary mismatch")
    terminal_source = build_source_manifest(
        source_root=source_root,
        run_tag=workload["run_tag"],
        source_commit=workload["source_commit"],
    )
    write_json(root / "terminal_source_manifest.json", terminal_source)
    _verify_source_files(terminal_source, source_root)
    summary = summarize_evidence(
        performance,
        correctness,
        run_dir=root,
    )
    write_json(root / "terminal_summary.json", summary)
    if summary["classification"] == NO_GO_EVIDENCE_INCOMPLETE:
        raise ValueError(
            "incomplete evidence cannot produce a terminal gate"
        )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": workload["run_tag"],
        "source_commit": workload["source_commit"],
        "classification": summary["classification"],
    }
    receipt = {
        **gate,
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
    }
    write_json(root / "terminal_gate.json", gate)
    write_json(root / "producer_receipt.json", receipt)
    sidecars = sorted({
        row["logits_path"]
        for row in correctness
    })
    artifact_paths = tuple(PRIMARY_ARTIFACTS) + tuple(sidecars)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_tag": workload["run_tag"],
        "source_commit": workload["source_commit"],
        "source_patch_sha256": hashlib.sha256(b"").hexdigest(),
        "artifact_sha256": {
            relative: sha256_file(
                _safe_artifact_path(root, relative)
            )
            for relative in artifact_paths
        },
    }
    write_json(root / "terminal_manifest.json", manifest)
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipt = produce_artifacts(
        args.run_dir,
        source_root=args.source_root,
    )
    if args.output is None:
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    else:
        write_json(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
