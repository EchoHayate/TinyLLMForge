"""Frozen contract for the TP4 collective-stable decode replay gate."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict


WORKLOADS = {
    "Q0": {
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 4,
    },
    "Q1": {
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 8,
    },
    "Q2": {
        "prompt_tokens": 2048,
        "output_tokens": 128,
        "concurrency": 4,
    },
}
RANKS = (0, 1, 2, 3)
ARMS = ("eager", "graph")
MEASURED_REPETITIONS = 5
CLASSIFICATIONS = (
    "GO_STAGE1_JUSTIFIED",
    "NO_GO_PERFORMANCE",
    "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "NO_GO_MECHANISM_NOT_EXERCISED",
    "INCOMPLETE",
)
THRESHOLDS = {
    "aggregate_output_throughput_ratio": 1.05,
    "aggregate_median_tpot_ratio": 0.95,
    "minimum_workload_output_throughput_ratio": 0.97,
    "maximum_workload_median_tpot_ratio": 1.03,
    "maximum_workload_p99_e2e_ratio": 1.03,
    "maximum_workload_ttft_ratio": 1.03,
    "minimum_replay_coverage": 0.80,
    "maximum_added_peak_allocated_bytes_per_rank": 512 * 1024 * 1024,
    "maximum_added_peak_reserved_bytes_per_rank": 512 * 1024 * 1024,
}

PERFORMANCE_FIELDS = (
    "output_tokens_per_second",
    "qps",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "median_e2e_ms",
    "p99_e2e_ms",
    "ttft_ms",
    "initialization_ms",
)
MEMORY_FIELDS = (
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)
CAPTURE_COST_FIELDS = (
    "capture_duration_ns",
    "static_bytes",
    "allocated_delta_bytes",
    "reserved_delta_bytes",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_case_matrix() -> tuple[dict, ...]:
    rows = []
    for workload, profile in WORKLOADS.items():
        for repetition in range(MEASURED_REPETITIONS):
            order = (
                ("eager", "graph")
                if repetition % 2 == 0
                else ("graph", "eager")
            )
            pair_id = f"{workload}__r{repetition}"
            for order_index, arm in enumerate(order):
                rows.append({
                    "case_id": f"{pair_id}__{arm}",
                    "pair_id": pair_id,
                    "workload": workload,
                    "repetition": repetition,
                    "arm": arm,
                    "order_index": order_index,
                    "profile": dict(profile),
                })
    return tuple(rows)


def _expected_cases() -> dict[str, dict]:
    return {row["case_id"]: row for row in build_case_matrix()}


def _expected_pairs() -> set[str]:
    return {
        f"{workload}__r{repetition}"
        for workload in WORKLOADS
        for repetition in range(MEASURED_REPETITIONS)
    }


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _check_row_ids(row_groups: tuple[list[dict], ...]) -> list[str]:
    seen = set()
    for rows in row_groups:
        for row in rows:
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id:
                return ["missing_row_id"]
            if row_id in seen:
                return ["duplicate_row_id"]
            seen.add(row_id)
    return []


def _validate_case_fields(row: dict) -> bool:
    expected = _expected_cases().get(row.get("case_id"))
    return bool(
        expected is not None
        and row.get("pair_id") == expected["pair_id"]
        and row.get("workload") == expected["workload"]
        and row.get("repetition") == expected["repetition"]
        and row.get("arm") == expected["arm"]
    )


def _group_by(rows: list[dict], fields: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    return grouped


def validate_rank_dispatch_rows(rows: list[dict]) -> dict:
    failures = []
    expected_cases = set(_expected_cases())
    observed_cases = set()
    groups = _group_by(rows, ("case_id", "phase", "step_index"))
    eligible = 0
    replayed = 0
    for (case_id, phase, step_index), group in groups.items():
        del phase, step_index
        if case_id not in expected_cases:
            failures.append("rank_dispatch_unknown_case")
            continue
        observed_cases.add(case_id)
        ordered = sorted(group, key=lambda row: row.get("rank", -1))
        if (
            len(ordered) != len(RANKS)
            or tuple(row.get("rank") for row in ordered) != RANKS
            or any(row.get("world_size") != len(RANKS) for row in ordered)
            or any(not _validate_case_fields(row) for row in ordered)
        ):
            failures.append("rank_dispatch_inventory_incomplete")
            continue
        agreement_fields = (
            "feature_enabled",
            "graph_eligible",
            "dispatch",
            "graph_identity_sha256",
            "cache_state",
            "capture_attempted",
            "fallback_reason",
        )
        reference = tuple(ordered[0].get(field) for field in agreement_fields)
        if any(
            tuple(row.get(field) for field in agreement_fields) != reference
            for row in ordered[1:]
        ):
            failures.append("rank_dispatch_disagreement")
            continue
        arm = ordered[0]["arm"]
        dispatch = ordered[0].get("dispatch")
        if arm == "eager":
            if (
                dispatch != "eager"
                or ordered[0].get("feature_enabled") is not False
                or any(row.get("graph_replay_count") != 0 for row in ordered)
            ):
                failures.append("baseline_graph_dispatch")
        elif ordered[0].get("graph_eligible") is True:
            eligible += len(ordered)
            if dispatch == "graph":
                if (
                    not _valid_sha256(
                        ordered[0].get("graph_identity_sha256")
                    )
                    or ordered[0].get("cache_state") != "ready"
                    or any(
                        not isinstance(row.get("graph_replay_count"), int)
                        or isinstance(row.get("graph_replay_count"), bool)
                        or row["graph_replay_count"] <= 0
                        for row in ordered
                    )
                ):
                    failures.append("invalid_graph_replay")
                else:
                    replayed += len(ordered)
    if observed_cases != expected_cases:
        failures.append("rank_dispatch_case_matrix_incomplete")
    return {
        "failures": sorted(set(failures)),
        "eligible_rank_steps": eligible,
        "replayed_rank_steps": replayed,
        "replay_coverage": (
            replayed / eligible if eligible else 0.0
        ),
    }


def validate_correctness_rows(rows: list[dict]) -> dict:
    failures = []
    expected_pairs = _expected_pairs()
    observed_pairs = set()
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id not in expected_pairs:
            failures.append("correctness_unknown_pair")
            continue
        observed_pairs.add(pair_id)
        if (
            row.get("workload") not in WORKLOADS
            or row.get("repetition")
            not in range(MEASURED_REPETITIONS)
        ):
            failures.append("correctness_identity_invalid")
        if (
            row.get("exact_match") is not True
            or row.get("eager_outputs") != row.get("graph_outputs")
        ):
            failures.append("exact_output_mismatch")
    if observed_pairs != expected_pairs or len(rows) != len(expected_pairs):
        failures.append("correctness_pair_matrix_incomplete")
    return {"failures": sorted(set(failures))}


def _validate_performance_rows(rows: list[dict]) -> list[str]:
    failures = []
    expected_cases = set(_expected_cases())
    observed_cases = set()
    for row in rows:
        if not _validate_case_fields(row):
            failures.append("performance_identity_invalid")
            continue
        observed_cases.add(row["case_id"])
        if any(
            not _finite_nonnegative(row.get(field))
            for field in PERFORMANCE_FIELDS
        ):
            failures.append("nonfinite_or_invalid_evidence")
    if observed_cases != expected_cases or len(rows) != len(expected_cases):
        failures.append("performance_case_matrix_incomplete")
    return failures


def _validate_collective_rows(rows: list[dict]) -> list[str]:
    failures = []
    expected_cases = set(_expected_cases())
    observed_cases = set()
    groups = _group_by(rows, ("case_id",))
    for (case_id,), group in groups.items():
        if case_id not in expected_cases:
            failures.append("collective_unknown_case")
            continue
        observed_cases.add(case_id)
        ordered = sorted(group, key=lambda row: row.get("rank", -1))
        if (
            len(ordered) != len(RANKS)
            or tuple(row.get("rank") for row in ordered) != RANKS
            or any(row.get("world_size") != 4 for row in ordered)
            or any(not _validate_case_fields(row) for row in ordered)
            or any(row.get("complete") is not True for row in ordered)
        ):
            failures.append("collective_rank_inventory_incomplete")
            continue
        reference = (
            ordered[0].get("collective_count"),
            ordered[0].get("collective_order_sha256"),
        )
        if (
            not isinstance(reference[0], int)
            or isinstance(reference[0], bool)
            or reference[0] <= 0
            or not _valid_sha256(reference[1])
            or any(
                (
                    row.get("collective_count"),
                    row.get("collective_order_sha256"),
                )
                != reference
                for row in ordered[1:]
            )
        ):
            failures.append("collective_order_disagreement")
    if observed_cases != expected_cases:
        failures.append("collective_case_matrix_incomplete")
    return failures


def _validate_lifecycle_rows(rows: list[dict]) -> list[str]:
    failures = []
    expected_cases = set(_expected_cases())
    observed_cases = set()
    groups = _group_by(rows, ("case_id",))
    for (case_id,), group in groups.items():
        if case_id not in expected_cases:
            failures.append("lifecycle_unknown_case")
            continue
        observed_cases.add(case_id)
        ordered = sorted(group, key=lambda row: row.get("rank", -1))
        if (
            len(ordered) != len(RANKS)
            or tuple(row.get("rank") for row in ordered) != RANKS
            or any(row.get("world_size") != 4 for row in ordered)
            or any(not _validate_case_fields(row) for row in ordered)
        ):
            failures.append("lifecycle_rank_inventory_incomplete")
            continue
        if any(
            row.get("complete") is not True
            or row.get("exit_code") != 0
            or row.get("process_group_destroyed") is not True
            or row.get("replay_exception") is not False
            for row in ordered
        ):
            failures.append("rank_lifecycle_failure")
    if observed_cases != expected_cases:
        failures.append("lifecycle_case_matrix_incomplete")
    return failures


def _validate_memory_rows(rows: list[dict]) -> list[str]:
    failures = []
    expected_cases = set(_expected_cases())
    observed_cases = set()
    groups = _group_by(rows, ("case_id",))
    for (case_id,), group in groups.items():
        if case_id not in expected_cases:
            failures.append("memory_unknown_case")
            continue
        observed_cases.add(case_id)
        ordered = sorted(group, key=lambda row: row.get("rank", -1))
        if (
            len(ordered) != len(RANKS)
            or tuple(row.get("rank") for row in ordered) != RANKS
            or any(not _validate_case_fields(row) for row in ordered)
            or any(
                not _finite_nonnegative(row.get(field))
                for row in ordered
                for field in MEMORY_FIELDS
            )
        ):
            failures.append("memory_rank_inventory_incomplete")
    if observed_cases != expected_cases:
        failures.append("memory_case_matrix_incomplete")
    return failures


def _validate_capture_cost_rows(rows: list[dict]) -> list[str]:
    failures = []
    expected_cases = {
        row["case_id"]
        for row in build_case_matrix()
        if row["arm"] == "graph"
    }
    observed_cases = set()
    groups = _group_by(rows, ("case_id",))
    for (case_id,), group in groups.items():
        expected = _expected_cases().get(case_id)
        if expected is None or expected["arm"] != "graph":
            failures.append("capture_cost_unknown_case")
            continue
        observed_cases.add(case_id)
        ordered = sorted(group, key=lambda row: row.get("rank", -1))
        if (
            len(ordered) != len(RANKS)
            or tuple(row.get("rank") for row in ordered) != RANKS
            or any(
                not _validate_case_fields(row)
                or row.get("arm") != "graph"
                or not _valid_sha256(
                    row.get("graph_identity_sha256")
                )
                or row.get("complete") is not True
                or any(
                    not _finite_nonnegative(row.get(field))
                    for field in CAPTURE_COST_FIELDS
                )
                for row in ordered
            )
        ):
            failures.append("capture_cost_rank_inventory_incomplete")
            continue
        if any(
            row["graph_identity_sha256"]
            != ordered[0]["graph_identity_sha256"]
            for row in ordered[1:]
        ):
            failures.append("capture_identity_disagreement")
    if observed_cases != expected_cases:
        failures.append("capture_cost_case_matrix_incomplete")
    return failures


def _median(values):
    return float(statistics.median(float(value) for value in values))


def _performance_summary(rows: list[dict]) -> tuple[dict, dict]:
    by_workload_arm = _group_by(rows, ("workload", "arm"))
    workloads = {}
    for workload in WORKLOADS:
        eager = by_workload_arm[(workload, "eager")]
        graph = by_workload_arm[(workload, "graph")]
        workloads[workload] = {
            "output_throughput_ratio": (
                _median(
                    row["output_tokens_per_second"] for row in graph
                )
                / _median(
                    row["output_tokens_per_second"] for row in eager
                )
            ),
            "median_tpot_ratio": (
                _median(row["median_tpot_ms"] for row in graph)
                / _median(row["median_tpot_ms"] for row in eager)
            ),
            "p99_e2e_ratio": (
                _median(row["p99_e2e_ms"] for row in graph)
                / _median(row["p99_e2e_ms"] for row in eager)
            ),
            "ttft_ratio": (
                _median(row["ttft_ms"] for row in graph)
                / _median(row["ttft_ms"] for row in eager)
            ),
        }
    eager_rows = [row for row in rows if row["arm"] == "eager"]
    graph_rows = [row for row in rows if row["arm"] == "graph"]
    aggregate = {
        "output_throughput_ratio": (
            sum(row["output_tokens_per_second"] for row in graph_rows)
            / sum(row["output_tokens_per_second"] for row in eager_rows)
        ),
        "median_tpot_ratio": (
            _median(row["median_tpot_ms"] for row in graph_rows)
            / _median(row["median_tpot_ms"] for row in eager_rows)
        ),
    }
    return workloads, aggregate


def _memory_summary(rows: list[dict]) -> tuple[int, int]:
    grouped = _group_by(rows, ("pair_id", "rank"))
    allocated = []
    reserved = []
    for group in grouped.values():
        by_arm = {row["arm"]: row for row in group}
        if set(by_arm) != set(ARMS):
            raise ValueError("memory pair is incomplete")
        allocated.append(
            int(
                by_arm["graph"]["peak_allocated_bytes"]
                - by_arm["eager"]["peak_allocated_bytes"]
            )
        )
        reserved.append(
            int(
                by_arm["graph"]["peak_reserved_bytes"]
                - by_arm["eager"]["peak_reserved_bytes"]
            )
        )
    return max(allocated), max(reserved)


def classify(
    *,
    performance_rows: list[dict],
    correctness_rows: list[dict],
    rank_dispatch_rows: list[dict],
    rank_collective_rows: list[dict],
    rank_lifecycle_rows: list[dict],
    memory_rows: list[dict],
    capture_cost_rows: list[dict],
) -> dict:
    row_groups = (
        performance_rows,
        correctness_rows,
        rank_dispatch_rows,
        rank_collective_rows,
        rank_lifecycle_rows,
        memory_rows,
        capture_cost_rows,
    )
    incomplete_failures = _check_row_ids(row_groups)
    incomplete_failures.extend(
        _validate_performance_rows(performance_rows)
    )
    incomplete_failures.extend(_validate_memory_rows(memory_rows))
    incomplete_failures.extend(
        _validate_capture_cost_rows(capture_cost_rows)
    )
    correctness = validate_correctness_rows(correctness_rows)
    dispatch = validate_rank_dispatch_rows(rank_dispatch_rows)
    collective_failures = _validate_collective_rows(
        rank_collective_rows
    )
    lifecycle_failures = _validate_lifecycle_rows(
        rank_lifecycle_rows
    )

    incomplete_names = {
        "missing_row_id",
        "duplicate_row_id",
        "nonfinite_or_invalid_evidence",
        "performance_identity_invalid",
        "performance_case_matrix_incomplete",
        "memory_unknown_case",
        "memory_rank_inventory_incomplete",
        "memory_case_matrix_incomplete",
        "capture_cost_unknown_case",
        "capture_cost_case_matrix_incomplete",
        "capture_cost_rank_inventory_incomplete",
        "correctness_unknown_pair",
        "correctness_identity_invalid",
        "correctness_pair_matrix_incomplete",
        "rank_dispatch_unknown_case",
        "rank_dispatch_inventory_incomplete",
        "rank_dispatch_case_matrix_incomplete",
        "collective_unknown_case",
        "collective_rank_inventory_incomplete",
        "collective_case_matrix_incomplete",
        "lifecycle_unknown_case",
        "lifecycle_rank_inventory_incomplete",
        "lifecycle_case_matrix_incomplete",
    }
    all_validation_failures = sorted(
        set(
            incomplete_failures
            + correctness["failures"]
            + dispatch["failures"]
            + collective_failures
            + lifecycle_failures
        )
    )
    incomplete = [
        failure
        for failure in all_validation_failures
        if failure in incomplete_names
    ]
    correctness_failures = [
        failure
        for failure in all_validation_failures
        if failure not in incomplete_names
    ]

    base_result = {
        "classification": "INCOMPLETE",
        "failed_gates": incomplete,
        "workloads": {},
        "aggregate": {},
        "replay_coverage": dispatch["replay_coverage"],
        "maximum_added_peak_allocated_bytes": None,
        "maximum_added_peak_reserved_bytes": None,
        "capture_duration_ns": None,
        "capture_amortization_tokens": None,
    }
    if incomplete:
        return base_result
    if correctness_failures:
        return base_result | {
            "classification": "NO_GO_CORRECTNESS_OR_LIFECYCLE",
            "failed_gates": correctness_failures,
        }

    try:
        workloads, aggregate = _performance_summary(performance_rows)
        maximum_allocated, maximum_reserved = _memory_summary(
            memory_rows
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return base_result | {
            "failed_gates": ["nonfinite_or_invalid_evidence"],
        }

    capture_duration_ns = sum(
        int(row["capture_duration_ns"]) for row in capture_cost_rows
    )
    saved_ms_per_token = max(
        0.0,
        _median(
            row["median_tpot_ms"]
            for row in performance_rows
            if row["arm"] == "eager"
        )
        - _median(
            row["median_tpot_ms"]
            for row in performance_rows
            if row["arm"] == "graph"
        ),
    )
    capture_amortization_tokens = (
        None
        if saved_ms_per_token <= 0.0
        else (capture_duration_ns / 1_000_000.0)
        / saved_ms_per_token
    )
    result = base_result | {
        "workloads": workloads,
        "aggregate": aggregate,
        "maximum_added_peak_allocated_bytes": maximum_allocated,
        "maximum_added_peak_reserved_bytes": maximum_reserved,
        "capture_duration_ns": capture_duration_ns,
        "capture_amortization_tokens": capture_amortization_tokens,
    }

    if (
        dispatch["replay_coverage"]
        < THRESHOLDS["minimum_replay_coverage"]
    ):
        return result | {
            "classification": "NO_GO_MECHANISM_NOT_EXERCISED",
            "failed_gates": ["replay_coverage"],
        }

    performance_failures = []
    if (
        aggregate["output_throughput_ratio"]
        < THRESHOLDS["aggregate_output_throughput_ratio"]
    ):
        performance_failures.append(
            "aggregate_output_throughput"
        )
    if (
        aggregate["median_tpot_ratio"]
        > THRESHOLDS["aggregate_median_tpot_ratio"]
    ):
        performance_failures.append("aggregate_median_tpot")
    for workload, summary in workloads.items():
        if (
            summary["output_throughput_ratio"]
            < THRESHOLDS[
                "minimum_workload_output_throughput_ratio"
            ]
        ):
            performance_failures.append(
                f"{workload}_output_throughput"
            )
        if (
            summary["median_tpot_ratio"]
            > THRESHOLDS["maximum_workload_median_tpot_ratio"]
        ):
            performance_failures.append(f"{workload}_median_tpot")
        if (
            summary["p99_e2e_ratio"]
            > THRESHOLDS["maximum_workload_p99_e2e_ratio"]
        ):
            performance_failures.append(f"{workload}_p99_e2e")
        if (
            summary["ttft_ratio"]
            > THRESHOLDS["maximum_workload_ttft_ratio"]
        ):
            performance_failures.append(f"{workload}_ttft")
    if (
        maximum_allocated
        > THRESHOLDS[
            "maximum_added_peak_allocated_bytes_per_rank"
        ]
    ):
        performance_failures.append("peak_allocated_memory")
    if (
        maximum_reserved
        > THRESHOLDS[
            "maximum_added_peak_reserved_bytes_per_rank"
        ]
    ):
        performance_failures.append("peak_reserved_memory")
    if performance_failures:
        return result | {
            "classification": "NO_GO_PERFORMANCE",
            "failed_gates": sorted(set(performance_failures)),
        }
    return result | {
        "classification": "GO_STAGE1_JUSTIFIED",
        "failed_gates": [],
    }
