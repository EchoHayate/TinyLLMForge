#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


SCHEMA_VERSION = 1
EDGE_ALLOWANCE_NS = 400_000_000
MAX_SAMPLE_GAP_NS = 600_000_000
EXPECTED_MEASURED_RUNS = 8
POLICIES = ("target", "learned")
POLICY_ORDERS = ("target,learned", "learned,target")
TIMING_METRICS = (
    "e2e_s",
    "tpot_s",
    "executor_proposal_forward_ms",
)
CUMULATIVE_COUNTERS = (
    "cpu_user_ticks",
    "cpu_nice_ticks",
    "cpu_system_ticks",
    "cpu_idle_ticks",
    "cpu_iowait_ticks",
    "cpu_irq_ticks",
    "cpu_softirq_ticks",
    "cpu_steal_ticks",
    "context_switches_total",
    "processes_forked_total",
    "major_faults_total",
    "page_in_kib_total",
    "page_out_kib_total",
    "swap_in_kib_total",
    "swap_out_kib_total",
    "cpu_psi_some_total_us",
    "io_psi_some_total_us",
    "io_psi_full_total_us",
    "memory_psi_some_total_us",
    "memory_psi_full_total_us",
)
OPTIONAL_COUNTERS = ("cpu_psi_full_total_us",)
SOURCE_FILE_PATHS = (
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_b4_timing_diagnostic.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_host_sampler.py",
    "tools/autoregressive_draft_host_semantic_diagnostic.py",
    "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
)
LIMITATIONS = (
    "host correlation is not causal proof",
    "system-wide pressure does not identify a responsible process",
    "campaign does not establish stable long-context performance",
    "campaign does not establish Proposal-KV offload benefit",
    "campaign does not establish Phase-1 promotion",
)
THRESHOLDS = {
    "sample_interval_seconds": 0.2,
    "edge_allowance_seconds": 0.4,
    "maximum_sample_gap_seconds": 0.6,
    "position_effect_fraction": 0.10,
    "host_metric_worse_fraction": 0.10,
    "spearman_rho_minimum": 0.60,
    "minimum_worse_primary_metrics": 2,
    "minimum_correlated_primary_metrics": 2,
}
PRIMARY_HOST_METRICS = (
    "cpu_system_fraction",
    "cpu_iowait_fraction",
    "run_queue_mean",
    "context_switches_per_second",
    "major_faults_per_second",
    "io_psi_some_fraction",
    "memory_psi_some_fraction",
    "memory_dirty_kib_max",
    "memory_writeback_kib_max",
)


def _validate_sample(sample: object) -> dict:
    if not isinstance(sample, dict):
        raise ValueError("host sample is invalid")
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("host sample schema is invalid")
    normalized = copy.deepcopy(sample)
    for name, value in normalized.items():
        if name == "schema_version":
            continue
        if name in OPTIONAL_COUNTERS and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"host sample {name} is invalid")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"host sample {name} is invalid")
    return normalized


def parse_host_jsonl(text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("host JSONL is empty")
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"host JSONL line {line_number} is invalid"
            ) from error
        rows.append(_validate_sample(row))
    for previous, current in zip(rows, rows[1:]):
        if (
            current["sampled_at_unix_ns"]
            <= previous["sampled_at_unix_ns"]
            or current["sampled_at_monotonic_ns"]
            <= previous["sampled_at_monotonic_ns"]
        ):
            raise ValueError("host sample timestamp regressed")
    return rows


def _campaign_interval(run: dict) -> tuple[int, int]:
    interval = run.get("campaign_interval")
    if not isinstance(interval, dict):
        raise ValueError("campaign interval is missing")
    start = interval.get("started_at_unix_ns")
    finish = interval.get("finished_at_unix_ns")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(finish, bool)
        or not isinstance(finish, int)
        or start <= 0
        or finish <= start
    ):
        raise ValueError("campaign interval is invalid")
    return start, finish


def _validate_ordered_samples(rows: list[dict]) -> None:
    for previous, current in zip(rows, rows[1:]):
        if (
            current["sampled_at_unix_ns"]
            <= previous["sampled_at_unix_ns"]
            or current["sampled_at_monotonic_ns"]
            <= previous["sampled_at_monotonic_ns"]
        ):
            raise ValueError("host sample timestamp regressed")
        if (
            current["sampled_at_monotonic_ns"]
            - previous["sampled_at_monotonic_ns"]
            > MAX_SAMPLE_GAP_NS
        ):
            raise ValueError("host sample gap exceeds limit")
        for name in CUMULATIVE_COUNTERS + OPTIONAL_COUNTERS:
            before = previous[name]
            after = current[name]
            if before is None and after is None:
                continue
            if before is None or after is None or after < before:
                raise ValueError(f"host counter regressed: {name}")


def align_repeat_samples(
    worker: dict,
    samples: list[dict],
) -> list[dict]:
    validated = [_validate_sample(row) for row in samples]
    aligned = []
    for run in worker["measured_runs"]:
        start, finish = _campaign_interval(run)
        start_candidates = [
            row for row in validated
            if row["sampled_at_unix_ns"] <= start
        ]
        finish_candidates = [
            row for row in validated
            if row["sampled_at_unix_ns"] >= finish
        ]
        if not start_candidates:
            raise ValueError("host start boundary is missing")
        if not finish_candidates:
            raise ValueError("host finish boundary is missing")
        first = start_candidates[-1]
        last = finish_candidates[0]
        if start - first["sampled_at_unix_ns"] > EDGE_ALLOWANCE_NS:
            raise ValueError("host start boundary exceeds allowance")
        if last["sampled_at_unix_ns"] - finish > EDGE_ALLOWANCE_NS:
            raise ValueError("host finish boundary exceeds allowance")
        interval_rows = [
            row for row in validated
            if first["sampled_at_unix_ns"]
            <= row["sampled_at_unix_ns"]
            <= last["sampled_at_unix_ns"]
        ]
        if len(interval_rows) < 2:
            raise ValueError("host repeat has fewer than two samples")
        _validate_ordered_samples(interval_rows)
        aligned.append({
            "repeat": run["repeat"],
            "campaign_interval": copy.deepcopy(
                run["campaign_interval"]
            ),
            "host_sample_interval": {
                "started_at_unix_ns": first["sampled_at_unix_ns"],
                "finished_at_unix_ns": last["sampled_at_unix_ns"],
                "started_at_monotonic_ns": (
                    first["sampled_at_monotonic_ns"]
                ),
                "finished_at_monotonic_ns": (
                    last["sampled_at_monotonic_ns"]
                ),
            },
            "samples": interval_rows,
        })
    return aligned


def _delta(first: dict, last: dict, name: str) -> float:
    return float(last[name] - first[name])


def _mean(rows: list[dict], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def derive_repeat_metrics(alignment: dict) -> dict:
    rows = alignment["samples"]
    first = rows[0]
    last = rows[-1]
    elapsed_seconds = (
        alignment["host_sample_interval"][
            "finished_at_monotonic_ns"
        ]
        - alignment["host_sample_interval"][
            "started_at_monotonic_ns"
        ]
    ) / 1e9
    if elapsed_seconds <= 0.0:
        raise ValueError("host sample duration is invalid")
    cpu_names = (
        "cpu_user_ticks",
        "cpu_nice_ticks",
        "cpu_system_ticks",
        "cpu_idle_ticks",
        "cpu_iowait_ticks",
        "cpu_irq_ticks",
        "cpu_softirq_ticks",
        "cpu_steal_ticks",
    )
    cpu_deltas = {name: _delta(first, last, name) for name in cpu_names}
    cpu_total = sum(cpu_deltas.values())
    if cpu_total <= 0.0:
        raise ValueError("aggregate CPU delta is zero")
    busy = (
        cpu_deltas["cpu_user_ticks"]
        + cpu_deltas["cpu_nice_ticks"]
        + cpu_deltas["cpu_system_ticks"]
        + cpu_deltas["cpu_irq_ticks"]
        + cpu_deltas["cpu_softirq_ticks"]
        + cpu_deltas["cpu_steal_ticks"]
    )

    def rate(name: str) -> float:
        return _delta(first, last, name) / elapsed_seconds

    def psi_fraction(name: str) -> float | None:
        if first[name] is None and last[name] is None:
            return None
        return _delta(first, last, name) / (
            elapsed_seconds * 1_000_000
        )

    return {
        "cpu_busy_fraction": busy / cpu_total,
        "cpu_system_fraction": (
            cpu_deltas["cpu_system_ticks"] / cpu_total
        ),
        "cpu_iowait_fraction": (
            cpu_deltas["cpu_iowait_ticks"] / cpu_total
        ),
        "cpu_steal_fraction": (
            cpu_deltas["cpu_steal_ticks"] / cpu_total
        ),
        "run_queue_mean": _mean(rows, "procs_running"),
        "run_queue_max": max(row["procs_running"] for row in rows),
        "blocked_processes_mean": _mean(rows, "procs_blocked"),
        "blocked_processes_max": max(
            row["procs_blocked"] for row in rows
        ),
        "loadavg_1m_mean": _mean(rows, "loadavg_1m"),
        "context_switches_per_second": rate(
            "context_switches_total"
        ),
        "forks_per_second": rate("processes_forked_total"),
        "major_faults_per_second": rate("major_faults_total"),
        "page_in_kib_per_second": rate("page_in_kib_total"),
        "page_out_kib_per_second": rate("page_out_kib_total"),
        "swap_in_kib_per_second": rate("swap_in_kib_total"),
        "swap_out_kib_per_second": rate("swap_out_kib_total"),
        "memory_available_kib_min": min(
            row["memory_available_kib"] for row in rows
        ),
        "memory_dirty_kib_max": max(
            row["memory_dirty_kib"] for row in rows
        ),
        "memory_writeback_kib_max": max(
            row["memory_writeback_kib"] for row in rows
        ),
        "cpu_psi_some_fraction": psi_fraction(
            "cpu_psi_some_total_us"
        ),
        "cpu_psi_full_fraction": psi_fraction(
            "cpu_psi_full_total_us"
        ),
        "io_psi_some_fraction": psi_fraction(
            "io_psi_some_total_us"
        ),
        "io_psi_full_fraction": psi_fraction(
            "io_psi_full_total_us"
        ),
        "memory_psi_some_fraction": psi_fraction(
            "memory_psi_some_total_us"
        ),
        "memory_psi_full_fraction": psi_fraction(
            "memory_psi_full_total_us"
        ),
    }


def extract_repeat_timing(worker: dict) -> list[dict]:
    rows = []
    for run in worker["measured_runs"]:
        per_request = run["timing"]["per_request"]
        rows.append({
            "repeat": run["repeat"],
            "e2e_s": statistics.median(
                row["completion_latency_s"] for row in per_request
            ),
            "tpot_s": statistics.median(
                row["tpot_s"] for row in per_request
            ),
            "executor_proposal_forward_ms": float(
                run["runtime"]["draft_executor_timing"][
                    "max_rank_ms"
                ]["proposal_forward"]
            ),
        })
    return rows


def build_host_semantic_artifact(
    *,
    timing_artifact: dict,
    gpu_telemetry_artifact: dict,
    target_worker: dict,
    learned_worker: dict,
    target_samples: list[dict],
    learned_samples: list[dict],
    policy_order: str,
    prime_each_policy: bool,
    source_files: dict[str, str],
    input_files: dict[str, dict],
) -> dict:
    if policy_order not in POLICY_ORDERS:
        raise ValueError("policy order is invalid")
    if prime_each_policy is not True:
        raise ValueError("same-policy priming is required")
    if (
        timing_artifact.get("status") != "PASS"
        or timing_artifact.get("exact_parity") is not True
        or gpu_telemetry_artifact.get("status") != "PASS"
        or gpu_telemetry_artifact.get("exact_parity") is not True
    ):
        raise ValueError("upstream artifact is invalid")
    for policy, worker in (
        ("target", target_worker),
        ("learned", learned_worker),
    ):
        if worker.get("policy") != policy:
            raise ValueError("worker policy mismatch")
        if len(worker.get("measured_runs", [])) != EXPECTED_MEASURED_RUNS:
            raise ValueError("worker measured repeat count is invalid")
    for repeat in range(EXPECTED_MEASURED_RUNS):
        if (
            target_worker["measured_runs"][repeat]["outputs"]
            != learned_worker["measured_runs"][repeat]["outputs"]
        ):
            raise ValueError(
                f"host diagnostic exact parity failed at repeat {repeat}"
            )
    policies = {}
    for policy, worker, samples in (
        ("target", target_worker, target_samples),
        ("learned", learned_worker, learned_samples),
    ):
        alignments = align_repeat_samples(worker, samples)
        timing_rows = extract_repeat_timing(worker)
        policies[policy] = {
            "worker_sha256": input_files[f"{policy}_worker"]["sha256"],
            "host_jsonl_sha256": input_files[
                f"{policy}_host_jsonl"
            ]["sha256"],
            "sample_count": len(samples),
            "measured_runs": [
                {
                    "repeat": alignment["repeat"],
                    "campaign_interval": alignment[
                        "campaign_interval"
                    ],
                    "host_sample_interval": alignment[
                        "host_sample_interval"
                    ],
                    "sample_count": len(alignment["samples"]),
                    "duration_seconds": (
                        alignment["host_sample_interval"][
                            "finished_at_monotonic_ns"
                        ]
                        - alignment["host_sample_interval"][
                            "started_at_monotonic_ns"
                        ]
                    ) / 1e9,
                    "metrics": derive_repeat_metrics(alignment),
                    "timing": timing_rows[index],
                }
                for index, alignment in enumerate(alignments)
            ],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": "ALIGNED_CAMPAIGN",
        "classification_reasons": [],
        "exact_parity": True,
        "policy_order": policy_order,
        "prime_each_policy": True,
        "timing_artifact_sha256": input_files[
            "timing_artifact"
        ]["sha256"],
        "gpu_telemetry_artifact_sha256": input_files[
            "gpu_telemetry_artifact"
        ]["sha256"],
        "policies": policies,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "source_files": copy.deepcopy(source_files),
        "input_files": copy.deepcopy(input_files),
        "limitations": list(LIMITATIONS),
    }


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda row: row[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average
        cursor = end
    return ranks


def spearman_rho(
    left: list[float],
    right: list[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman vectors are invalid")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_sum = sum(
        (value - left_mean) ** 2 for value in left_ranks
    )
    right_sum = sum(
        (value - right_mean) ** 2 for value in right_ranks
    )
    if left_sum <= 0.0 or right_sum <= 0.0:
        return None
    return numerator / math.sqrt(left_sum * right_sum)


def classify_host_comparison(
    *,
    learned_e2e_relative_delta: float,
    worse_primary_metrics: set[str],
    e2e_correlated_metrics: set[str],
    proposal_correlated_metrics: set[str],
) -> tuple[str, list[str]]:
    reasons = []
    if learned_e2e_relative_delta < THRESHOLDS[
        "position_effect_fraction"
    ]:
        return (
            "HOST_ALIGNMENT_INCONCLUSIVE",
            ["learned E2E position effect is below 10%"],
        )
    correlated_union = (
        e2e_correlated_metrics | proposal_correlated_metrics
    )
    associated = (
        len(worse_primary_metrics)
        >= THRESHOLDS["minimum_worse_primary_metrics"]
        and len(correlated_union)
        >= THRESHOLDS["minimum_correlated_primary_metrics"]
        and bool(e2e_correlated_metrics)
        and bool(proposal_correlated_metrics)
    )
    if associated:
        return (
            "HOST_PRESSURE_ASSOCIATED",
            [
                "learned first-position E2E is at least 10% slower",
                "at least two primary host metrics are worse",
                "expected-direction correlation covers E2E and proposal-forward",
            ],
        )
    if len(worse_primary_metrics) < 2:
        reasons.append("fewer than two primary host metrics are worse")
    if len(correlated_union) < 2:
        reasons.append(
            "fewer than two primary host metrics meet the rho threshold"
        )
    if not e2e_correlated_metrics:
        reasons.append("no primary host metric correlates with learned E2E")
    if not proposal_correlated_metrics:
        reasons.append(
            "no primary host metric correlates with proposal-forward"
        )
    return "HOST_PRESSURE_NOT_SUPPORTED", reasons


def build_host_semantic_comparison(
    *,
    first_artifact: dict,
    second_artifact: dict,
    first_reference: dict,
    second_reference: dict,
) -> dict:
    campaigns = [first_artifact, second_artifact]
    if any(
        row.get("status") != "PASS"
        or row.get("classification") != "ALIGNED_CAMPAIGN"
        or row.get("exact_parity") is not True
        for row in campaigns
    ):
        raise ValueError("comparison campaign is invalid")
    orders = {row["policy_order"] for row in campaigns}
    if orders != set(POLICY_ORDERS):
        raise ValueError("comparison policy orders are invalid")
    if any(row.get("prime_each_policy") is not True for row in campaigns):
        raise ValueError("comparison requires primed campaigns")
    if campaigns[0]["source_files"] != campaigns[1]["source_files"]:
        raise ValueError("comparison source identity mismatch")
    for name in (
        "timing_artifact",
        "gpu_telemetry_artifact",
        "target_worker",
        "learned_worker",
        "target_host_jsonl",
        "learned_host_jsonl",
    ):
        if (
            campaigns[0]["input_files"][name]["sha256"]
            == campaigns[1]["input_files"][name]["sha256"]
        ):
            raise ValueError(
                f"comparison input is not distinct: {name}"
            )
    learned_first = next(
        row for row in campaigns
        if row["policy_order"] == "learned,target"
    )
    learned_second = next(
        row for row in campaigns
        if row["policy_order"] == "target,learned"
    )
    first_runs = learned_first["policies"]["learned"]["measured_runs"]
    second_runs = learned_second["policies"]["learned"]["measured_runs"]

    def median_timing(runs, name):
        return statistics.median(
            row["timing"][name] for row in runs
        )

    position = {}
    for name in TIMING_METRICS:
        first_value = median_timing(first_runs, name)
        second_value = median_timing(second_runs, name)
        position[name] = {
            "learned_first_median": first_value,
            "learned_second_median": second_value,
            "relative_delta": (
                (first_value - second_value) / second_value
            ),
        }

    metric_comparison = {}
    worse = set()
    correlations = {}
    combined_runs = first_runs + second_runs
    for metric in PRIMARY_HOST_METRICS:
        first_value = statistics.median(
            row["metrics"][metric] for row in first_runs
        )
        second_value = statistics.median(
            row["metrics"][metric] for row in second_runs
        )
        relative = (
            math.inf
            if second_value <= 1e-12 and first_value > 1e-12
            else (
                0.0
                if second_value <= 1e-12
                else (first_value - second_value) / second_value
            )
        )
        is_worse = (
            first_value > second_value
            and first_value - second_value > 1e-12
            and relative >= THRESHOLDS["host_metric_worse_fraction"]
        )
        if is_worse:
            worse.add(metric)
        metric_comparison[metric] = {
            "learned_first_median": first_value,
            "learned_second_median": second_value,
            "absolute_difference": first_value - second_value,
            "relative_increase": relative,
            "worse_in_learned_first": is_worse,
        }
        correlations[metric] = {}
        host_values = [row["metrics"][metric] for row in combined_runs]
        for timing_name in (
            "e2e_s",
            "executor_proposal_forward_ms",
        ):
            timing_values = [
                row["timing"][timing_name] for row in combined_runs
            ]
            rho = spearman_rho(host_values, timing_values)
            correlations[metric][timing_name] = {
                "sample_count": len(host_values),
                "host_rank_variance": len(set(host_values)) > 1,
                "timing_rank_variance": len(set(timing_values)) > 1,
                "rho": rho,
            }

    e2e_correlated = {
        metric for metric, rows in correlations.items()
        if rows["e2e_s"]["rho"] is not None
        and rows["e2e_s"]["rho"] >= THRESHOLDS["spearman_rho_minimum"]
    }
    proposal_correlated = {
        metric for metric, rows in correlations.items()
        if rows["executor_proposal_forward_ms"]["rho"] is not None
        and rows["executor_proposal_forward_ms"]["rho"]
        >= THRESHOLDS["spearman_rho_minimum"]
    }
    classification, reasons = classify_host_comparison(
        learned_e2e_relative_delta=position["e2e_s"]["relative_delta"],
        worse_primary_metrics=worse,
        e2e_correlated_metrics=e2e_correlated,
        proposal_correlated_metrics=proposal_correlated,
    )
    expected_roles = {
        "target,learned": "learned_second",
        "learned,target": "learned_first",
    }
    for artifact, reference in zip(
        campaigns,
        (first_reference, second_reference),
    ):
        pure = PurePosixPath(reference["path"])
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("comparison reference path is invalid")
        if reference["policy_order"] != artifact["policy_order"]:
            raise ValueError("comparison reference policy order is invalid")
        if (
            reference["role"]
            != expected_roles[reference["policy_order"]]
        ):
            raise ValueError("comparison reference role is invalid")
    references = {
        reference["role"]: copy.deepcopy(reference)
        for reference in (first_reference, second_reference)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": classification,
        "classification_reasons": reasons,
        "campaign_artifacts": references,
        "learned_position_effect": position,
        "primary_metric_comparison": metric_comparison,
        "correlations": correlations,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "source_identity": copy.deepcopy(
            campaigns[0]["source_files"]
        ),
        "limitations": list(LIMITATIONS),
    }


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object is required: {path}")
    return value


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(path: Path, *, base: Path, name: str) -> str:
    try:
        relative = path.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"{name} path must be below output directory") from error
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} path is invalid")
    return pure.as_posix()


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-artifact")
    parser.add_argument("--gpu-telemetry-artifact")
    parser.add_argument("--target-worker")
    parser.add_argument("--learned-worker")
    parser.add_argument("--target-host-jsonl")
    parser.add_argument("--learned-host-jsonl")
    parser.add_argument("--policy-order", choices=POLICY_ORDERS)
    parser.add_argument("--prime-each-policy", action="store_true")
    parser.add_argument("--repo-root")
    parser.add_argument("--campaign-artifact")
    parser.add_argument("--comparison-artifact")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _build_campaign_from_args(args) -> dict:
    required = {
        "timing artifact": args.timing_artifact,
        "GPU telemetry artifact": args.gpu_telemetry_artifact,
        "target worker": args.target_worker,
        "learned worker": args.learned_worker,
        "target host JSONL": args.target_host_jsonl,
        "learned host JSONL": args.learned_host_jsonl,
        "policy order": args.policy_order,
        "repo root": args.repo_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or args.prime_each_policy is not True:
        raise ValueError(
            "campaign mode requires all inputs and same-policy priming"
        )
    output_path = Path(args.out)
    output_directory = output_path.parent
    input_paths = {
        "timing_artifact": Path(args.timing_artifact),
        "gpu_telemetry_artifact": Path(args.gpu_telemetry_artifact),
        "target_worker": Path(args.target_worker),
        "learned_worker": Path(args.learned_worker),
        "target_host_jsonl": Path(args.target_host_jsonl),
        "learned_host_jsonl": Path(args.learned_host_jsonl),
    }
    input_files = {
        name: {
            "path": _relative_path(
                path,
                base=output_directory,
                name=name,
            ),
            "sha256": _sha256_path(path),
        }
        for name, path in input_paths.items()
    }
    repo_root = Path(args.repo_root)
    source_files = {
        relative: _sha256_path(repo_root / relative)
        for relative in SOURCE_FILE_PATHS
    }
    return build_host_semantic_artifact(
        timing_artifact=_load_json(input_paths["timing_artifact"]),
        gpu_telemetry_artifact=_load_json(
            input_paths["gpu_telemetry_artifact"]
        ),
        target_worker=_load_json(input_paths["target_worker"]),
        learned_worker=_load_json(input_paths["learned_worker"]),
        target_samples=parse_host_jsonl(
            input_paths["target_host_jsonl"].read_text(encoding="utf-8")
        ),
        learned_samples=parse_host_jsonl(
            input_paths["learned_host_jsonl"].read_text(encoding="utf-8")
        ),
        policy_order=args.policy_order,
        prime_each_policy=True,
        source_files=source_files,
        input_files=input_files,
    )


def _build_comparison_from_args(args) -> dict:
    if not args.campaign_artifact or not args.comparison_artifact:
        raise ValueError("comparison mode requires two campaign artifacts")
    output_path = Path(args.out)
    output_directory = output_path.parent
    first_path = Path(args.campaign_artifact)
    second_path = Path(args.comparison_artifact)
    first_artifact = _load_json(first_path)
    second_artifact = _load_json(second_path)

    def reference(path: Path, artifact: dict) -> dict:
        policy_order = artifact.get("policy_order")
        roles = {
            "target,learned": "learned_second",
            "learned,target": "learned_first",
        }
        if policy_order not in roles:
            raise ValueError("comparison reference policy order is invalid")
        return {
            "path": _relative_path(
                path,
                base=output_directory,
                name="comparison reference",
            ),
            "sha256": _sha256_path(path),
            "policy_order": policy_order,
            "role": roles[policy_order],
        }

    return build_host_semantic_comparison(
        first_artifact=first_artifact,
        second_artifact=second_artifact,
        first_reference=reference(first_path, first_artifact),
        second_reference=reference(second_path, second_artifact),
    )


def main(argv=None):
    args = parse_args(argv)
    comparison_requested = bool(
        args.campaign_artifact or args.comparison_artifact
    )
    campaign_requested = bool(
        args.timing_artifact
        or args.gpu_telemetry_artifact
        or args.target_worker
        or args.learned_worker
        or args.target_host_jsonl
        or args.learned_host_jsonl
        or args.policy_order
        or args.prime_each_policy
        or args.repo_root
    )
    if comparison_requested and campaign_requested:
        raise ValueError("diagnostic mode arguments cannot be mixed")
    if comparison_requested:
        artifact = _build_comparison_from_args(args)
    else:
        artifact = _build_campaign_from_args(args)
    _atomic_write_json(Path(args.out), artifact)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
