from __future__ import annotations

import math
from statistics import median


WORLD_SIZE = 4
ACTIVE_TOKEN_GROUPS = (1, 4, 8)
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 15
DIAGNOSTIC_ITERATION_COUNT = 15
HIDDEN_SIZE = 5120
STATE_BYTES_PER_TOKEN_PER_LAYER = 271_360
LINEAR_LAYER_COUNT = 48
MAX_RESERVED_SLACK_BYTES = 64 * 1024 * 1024
MIN_OVERLAP_RATIO = 0.20
MIN_AGGREGATE_SPEEDUP = 0.05
MAX_SINGLE_TOKEN_MEDIAN_REGRESSION = 0.01
MAX_P99_REGRESSION = 0.03
MAX_HOST_SUBMISSION_REGRESSION = 0.03
MIN_DIRECTIONAL_PAIR_COUNT = 11
MAX_GPU_MEMORY_USED_MIB = 1024
MAX_GPU_UTILIZATION_PERCENT = 5
RUNTIME_CAPABILITY_FIELDS = frozenset({
    "rank",
    "device_index",
    "device_name",
    "device_uuid",
    "compute_capability",
    "hostname",
    "python_version",
    "driver_version",
    "cuda_version",
    "torch_version",
    "nccl_available",
    "nccl_version",
    "world_size",
    "hidden_size",
    "collective_dtype",
    "output_dtype",
    "state_dtype",
})


def validate_runtime_capabilities(capabilities, gpu_rank_rows):
    if (
        not isinstance(capabilities, dict)
        or not isinstance(capabilities.get("rank_rows"), list)
        or len(capabilities["rank_rows"]) != WORLD_SIZE
        or not isinstance(gpu_rank_rows, list)
        or len(gpu_rank_rows) != WORLD_SIZE
    ):
        raise ValueError("runtime capability identity is invalid")
    expected_by_rank = {
        row.get("rank"): row
        for row in gpu_rank_rows
        if isinstance(row, dict)
    }
    if (
        set(expected_by_rank) != set(range(WORLD_SIZE))
        or any(
            set(row) != {"rank", "device_index", "device_uuid"}
            or type(row["rank"]) is not int
            or type(row["device_index"]) is not int
            or row["device_index"] < 0
            or not isinstance(row["device_uuid"], str)
            or not row["device_uuid"].startswith("GPU-")
            for row in gpu_rank_rows
        )
        or len({row["device_index"] for row in gpu_rank_rows}) != WORLD_SIZE
        or len({row["device_uuid"] for row in gpu_rank_rows}) != WORLD_SIZE
    ):
        raise ValueError("runtime capability identity is invalid")
    shared_versions = set()
    seen_ranks = set()
    for row in capabilities["rank_rows"]:
        if not isinstance(row, dict) or set(row) != RUNTIME_CAPABILITY_FIELDS:
            raise ValueError("runtime capability schema is invalid")
        rank = row["rank"]
        expected = expected_by_rank.get(rank)
        compute_capability = row["compute_capability"]
        if (
            type(rank) is not int
            or rank not in range(WORLD_SIZE)
            or rank in seen_ranks
            or type(row["device_index"]) is not int
            or row["device_index"] != rank
            or not isinstance(row["device_name"], str)
            or not row["device_name"]
            or not isinstance(row["device_uuid"], str)
            or expected is None
            or row["device_uuid"] != expected.get("device_uuid")
            or not isinstance(compute_capability, list)
            or len(compute_capability) != 2
            or any(type(value) is not int or value < 0 for value in compute_capability)
            or not isinstance(row["hostname"], str)
            or not row["hostname"]
            or not isinstance(row["python_version"], str)
            or not row["python_version"]
            or not isinstance(row["driver_version"], str)
            or not row["driver_version"]
            or not isinstance(row["cuda_version"], str)
            or not row["cuda_version"]
            or row["cuda_version"] == "None"
            or not isinstance(row["torch_version"], str)
            or not row["torch_version"]
            or row["nccl_available"] is not True
            or not isinstance(row["nccl_version"], str)
            or not row["nccl_version"]
            or row["nccl_version"] == "None"
            or row["world_size"] != WORLD_SIZE
            or row["hidden_size"] != HIDDEN_SIZE
            or row["collective_dtype"] != "float32"
            or row["output_dtype"] != "bfloat16"
            or row["state_dtype"] != "bfloat16"
        ):
            raise ValueError("runtime capability identity is invalid")
        seen_ranks.add(rank)
        shared_versions.add((
            row["hostname"],
            row["python_version"],
            row["driver_version"],
            row["cuda_version"],
            row["torch_version"],
            row["nccl_version"],
        ))
    if seen_ranks != set(range(WORLD_SIZE)) or len(shared_versions) != 1:
        raise ValueError("runtime capability identity is invalid")
    return {
        "rank_rows": [
            dict(row)
            for row in sorted(
                capabilities["rank_rows"],
                key=lambda row: row["rank"],
            )
        ]
    }


def validate_strict_clean_admission(admission):
    rows = admission.get("rank_rows") if isinstance(admission, dict) else None
    if (
        admission.get("classification") != "STRICT_CLEAN"
        if isinstance(admission, dict)
        else True
    ) or not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        raise ValueError("strict-clean admission is invalid")
    seen_ranks = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "rank",
                "memory_mib",
                "utilization_percent",
                "compute_processes",
            }
            or type(row["rank"]) is not int
            or row["rank"] not in range(WORLD_SIZE)
            or row["rank"] in seen_ranks
            or type(row["memory_mib"]) is not int
            or not 0 <= row["memory_mib"] <= MAX_GPU_MEMORY_USED_MIB
            or type(row["utilization_percent"]) is not int
            or not 0
            <= row["utilization_percent"]
            <= MAX_GPU_UTILIZATION_PERCENT
            or row["compute_processes"] != []
        ):
            raise ValueError("strict-clean admission is invalid")
        seen_ranks.add(row["rank"])
    if seen_ranks != set(range(WORLD_SIZE)):
        raise ValueError("strict-clean admission is invalid")
    return {
        "classification": "STRICT_CLEAN",
        "rank_rows": [
            dict(row)
            for row in sorted(rows, key=lambda row: row["rank"])
        ],
    }


def interval_intersection_ns(left, right):
    if (
        not isinstance(left, (tuple, list))
        or not isinstance(right, (tuple, list))
        or len(left) != 2
        or len(right) != 2
    ):
        raise ValueError("interval must contain two endpoints")
    left_start, left_end = left
    right_start, right_end = right
    values = (left_start, left_end, right_start, right_end)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("interval endpoints are invalid")
    if left_end < left_start or right_end < right_start:
        raise ValueError("interval endpoints are invalid")
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _nearest_rank_percentile(values, percentile):
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_measurement_row(row):
    if not isinstance(row, dict):
        raise ValueError("measurement row must be an object")
    numeric = (
        "baseline_critical_ns",
        "candidate_critical_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "overlap_intersection_ns",
    )
    for name in numeric:
        if not _finite_nonnegative(row.get(name)):
            raise ValueError(f"{name} is invalid")
    for name in ("allreduce_interval_ns", "state_copy_interval_ns"):
        value = row.get(name)
        interval_intersection_ns(value, value)
    if row["overlap_intersection_ns"] != interval_intersection_ns(
        row["allreduce_interval_ns"],
        row["state_copy_interval_ns"],
    ):
        raise ValueError("overlap_intersection_ns is invalid")
    active_tokens = row.get("active_tokens")
    pair_index = row.get("pair_index")
    rank = row.get("rank")
    if active_tokens not in ACTIVE_TOKEN_GROUPS:
        raise ValueError("active_tokens is invalid")
    if type(pair_index) is not int or pair_index not in range(
        MEASURED_PAIR_COUNT
    ):
        raise ValueError("pair_index is invalid")
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError("rank is invalid")
    expected_order = (
        ["baseline", "candidate"]
        if pair_index % 2 == 0
        else ["candidate", "baseline"]
    )
    if row.get("arm_order") != expected_order:
        raise ValueError("arm_order is invalid")
    booleans = (
        "reduced_output_exact",
        "final_output_exact",
        "shadow_payload_exact",
        "active_state_preserved_before_publish",
        "published_state_exact",
        "abort_preserved_old_state",
        "commit_identity_match",
        "finite_output",
        "timed_out",
    )
    if any(type(row.get(name)) is not bool for name in booleans):
        raise ValueError("correctness or lifecycle flag is invalid")
    if (
        type(row.get("timed_path_allocation_count")) is not int
        or row["timed_path_allocation_count"] < 0
    ):
        raise ValueError("timed_path_allocation_count is invalid")
    return dict(row)


def validate_stage01_diagnostic_row(row):
    if not isinstance(row, dict):
        raise ValueError("diagnostic row must be an object")
    active_tokens = row.get("active_tokens")
    diagnostic_index = row.get("diagnostic_index")
    rank = row.get("rank")
    if active_tokens not in ACTIVE_TOKEN_GROUPS:
        raise ValueError("diagnostic active_tokens is invalid")
    if (
        type(diagnostic_index) is not int
        or diagnostic_index not in range(DIAGNOSTIC_ITERATION_COUNT)
    ):
        raise ValueError("diagnostic_index is invalid")
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError("diagnostic rank is invalid")
    booleans = (
        "baseline_reduced_exact",
        "baseline_final_exact",
        "completion_owned_reduced_exact",
        "completion_owned_final_exact",
        "event_only_reduced_exact",
        "event_only_final_exact",
    )
    if any(type(row.get(name)) is not bool for name in booleans):
        raise ValueError("diagnostic correctness flag is invalid")
    return dict(row)


def validate_stage01_measurement_row(row):
    if not isinstance(row, dict):
        raise ValueError("measurement row must be an object")
    numeric = (
        "baseline_critical_ns",
        "candidate_critical_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "overlap_intersection_ns",
    )
    for name in numeric:
        if not _finite_nonnegative(row.get(name)):
            raise ValueError(f"{name} is invalid")
    interval_names = (
        "collective_outstanding_window_ns",
        "side_effect_window_ns",
    )
    for name in interval_names:
        value = row.get(name)
        interval_intersection_ns(value, value)
    if row["overlap_intersection_ns"] != interval_intersection_ns(
        row["collective_outstanding_window_ns"],
        row["side_effect_window_ns"],
    ):
        raise ValueError("overlap_intersection_ns is invalid")
    active_tokens = row.get("active_tokens")
    pair_index = row.get("pair_index")
    rank = row.get("rank")
    if active_tokens not in ACTIVE_TOKEN_GROUPS:
        raise ValueError("active_tokens is invalid")
    if type(pair_index) is not int or pair_index not in range(
        MEASURED_PAIR_COUNT
    ):
        raise ValueError("pair_index is invalid")
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError("rank is invalid")
    expected_order = (
        ["baseline", "completion_owned"]
        if pair_index % 2 == 0
        else ["completion_owned", "baseline"]
    )
    if row.get("arm_order") != expected_order:
        raise ValueError("arm_order is invalid")
    booleans = (
        "expected_reduced_exact",
        "baseline_reduced_exact",
        "candidate_reduced_exact",
        "baseline_final_exact",
        "candidate_final_exact",
        "baseline_candidate_exact",
        "shadow_payload_exact",
        "active_state_preserved_before_publish",
        "published_state_exact",
        "abort_preserved_old_state",
        "commit_identity_match",
        "collective_wait_invoked",
        "collective_dependency_transferred",
        "side_effect_dependency_joined",
        "finite_output",
        "timed_out",
    )
    if any(type(row.get(name)) is not bool for name in booleans):
        raise ValueError("correctness or lifecycle flag is invalid")
    if (
        type(row.get("timed_path_allocation_count")) is not int
        or row["timed_path_allocation_count"] < 0
    ):
        raise ValueError("timed_path_allocation_count is invalid")
    return dict(row)


def _result(classification, summaries, row_count):
    return {
        "classification": classification,
        "stage1_authorized": (
            classification == "GO_LEASE_SEALED_OVERLAP_MICROGATE"
        ),
        "measurement_row_count": row_count,
        "shape_summaries": summaries,
    }


def _stage01_result(
    classification,
    summaries,
    row_count,
    diagnostic_row_count,
):
    return {
        "classification": classification,
        "stage1_authorized": (
            classification == "GO_COMPLETION_OWNED_OVERLAP_MICROGATE"
        ),
        "measurement_row_count": row_count,
        "diagnostic_row_count": diagnostic_row_count,
        "shape_summaries": summaries,
    }


def classify_stage01(
    rows,
    diagnostic_rows,
    memory,
    cleanup,
    *,
    resource_identity_valid=True,
):
    expected_rows = {
        (shape, pair, rank)
        for shape in ACTIVE_TOKEN_GROUPS
        for pair in range(MEASURED_PAIR_COUNT)
        for rank in range(WORLD_SIZE)
    }
    expected_diagnostics = {
        (shape, diagnostic_index, rank)
        for shape in ACTIVE_TOKEN_GROUPS
        for diagnostic_index in range(DIAGNOSTIC_ITERATION_COUNT)
        for rank in range(WORLD_SIZE)
    }
    validated = []
    seen = set()
    incomplete = False
    correctness_failed = False
    allocation_failed = False
    for raw in rows if isinstance(rows, (list, tuple)) else ():
        try:
            row = validate_stage01_measurement_row(raw)
        except ValueError:
            incomplete = True
            continue
        identity = (
            row["active_tokens"],
            row["pair_index"],
            row["rank"],
        )
        if identity in seen:
            incomplete = True
            continue
        seen.add(identity)
        validated.append(row)
        correctness_failed |= not all(
            row[name]
            for name in (
                "expected_reduced_exact",
                "baseline_reduced_exact",
                "candidate_reduced_exact",
                "baseline_final_exact",
                "candidate_final_exact",
                "baseline_candidate_exact",
                "shadow_payload_exact",
                "active_state_preserved_before_publish",
                "published_state_exact",
                "abort_preserved_old_state",
                "commit_identity_match",
                "collective_wait_invoked",
                "collective_dependency_transferred",
                "side_effect_dependency_joined",
                "finite_output",
            )
        )
        correctness_failed |= row["timed_out"]
        allocation_failed |= row["timed_path_allocation_count"] != 0
    incomplete |= seen != expected_rows

    validated_diagnostics = []
    seen_diagnostics = set()
    for raw in (
        diagnostic_rows
        if isinstance(diagnostic_rows, (list, tuple))
        else ()
    ):
        try:
            row = validate_stage01_diagnostic_row(raw)
        except ValueError:
            incomplete = True
            continue
        identity = (
            row["active_tokens"],
            row["diagnostic_index"],
            row["rank"],
        )
        if identity in seen_diagnostics:
            incomplete = True
            continue
        seen_diagnostics.add(identity)
        validated_diagnostics.append(row)
        correctness_failed |= not all(
            row[name]
            for name in (
                "baseline_reduced_exact",
                "baseline_final_exact",
                "completion_owned_reduced_exact",
                "completion_owned_final_exact",
            )
        )
    incomplete |= seen_diagnostics != expected_diagnostics

    if correctness_failed:
        return _stage01_result(
            "NO_GO_CORRECTNESS_OR_LIFECYCLE",
            [],
            len(validated),
            len(validated_diagnostics),
        )
    if resource_identity_valid is not True:
        return _stage01_result(
            "NO_GO_RESOURCE_IDENTITY",
            [],
            len(validated),
            len(validated_diagnostics),
        )

    memory_rows = memory.get("rank_rows") if isinstance(memory, dict) else None
    if not isinstance(memory_rows, list) or len(memory_rows) != WORLD_SIZE:
        incomplete = True
    else:
        seen_memory_ranks = set()
        for row in memory_rows:
            if not isinstance(row, dict):
                incomplete = True
                continue
            rank = row.get("rank")
            required = (
                row.get("maximum_reserved_delta_bytes"),
                row.get("maximum_theoretical_shadow_bytes"),
            )
            if (
                type(rank) is not int
                or rank not in range(WORLD_SIZE)
                or rank in seen_memory_ranks
                or any(not _finite_nonnegative(value) for value in required)
            ):
                incomplete = True
                continue
            seen_memory_ranks.add(rank)
            allocation_failed |= (
                required[0] > required[1] + MAX_RESERVED_SLACK_BYTES
            )
        incomplete |= seen_memory_ranks != set(range(WORLD_SIZE))
    if allocation_failed:
        return _stage01_result(
            "NO_GO_MEMORY_OR_ALLOCATION",
            [],
            len(validated),
            len(validated_diagnostics),
        )
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        incomplete = True
    if incomplete:
        return _stage01_result(
            "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT",
            [],
            len(validated),
            len(validated_diagnostics),
        )

    event_only_failure_reproduced = any(
        not row["event_only_reduced_exact"]
        or not row["event_only_final_exact"]
        for row in validated_diagnostics
    )
    if not event_only_failure_reproduced:
        return _stage01_result(
            "INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED",
            [],
            len(validated),
            len(validated_diagnostics),
        )

    summaries = []
    for shape in ACTIVE_TOKEN_GROUPS:
        pair_rows = []
        for pair in range(MEASURED_PAIR_COUNT):
            ranks = [
                row
                for row in validated
                if row["active_tokens"] == shape
                and row["pair_index"] == pair
            ]
            pair_rows.append({
                "baseline": max(
                    row["baseline_critical_ns"] for row in ranks
                ),
                "candidate": max(
                    row["candidate_critical_ns"] for row in ranks
                ),
                "baseline_host": max(
                    row["baseline_host_submission_ns"] for row in ranks
                ),
                "candidate_host": max(
                    row["candidate_host_submission_ns"] for row in ranks
                ),
                "overlap_ratio": min(
                    row["overlap_intersection_ns"]
                    / max(
                        1,
                        min(
                            row["collective_outstanding_window_ns"][1]
                            - row["collective_outstanding_window_ns"][0],
                            row["side_effect_window_ns"][1]
                            - row["side_effect_window_ns"][0],
                        ),
                    )
                    for row in ranks
                ),
            })
        baseline = [row["baseline"] for row in pair_rows]
        candidate = [row["candidate"] for row in pair_rows]
        baseline_median = median(baseline)
        candidate_median = median(candidate)
        baseline_p99 = _nearest_rank_percentile(baseline, 0.99)
        candidate_p99 = _nearest_rank_percentile(candidate, 0.99)
        baseline_host = median(row["baseline_host"] for row in pair_rows)
        candidate_host = median(row["candidate_host"] for row in pair_rows)
        summaries.append({
            "active_tokens": shape,
            "median_speedup_ratio": 1 - candidate_median / baseline_median,
            "p99_regression_ratio": candidate_p99 / baseline_p99 - 1,
            "host_submission_regression_ratio": (
                candidate_host / baseline_host - 1
            ),
            "median_realized_overlap_ratio": median(
                row["overlap_ratio"] for row in pair_rows
            ),
            "improving_pair_count": sum(
                row["candidate"] < row["baseline"] for row in pair_rows
            ),
        })

    by_shape = {row["active_tokens"]: row for row in summaries}
    if any(
        by_shape[shape]["median_realized_overlap_ratio"]
        < MIN_OVERLAP_RATIO
        for shape in (4, 8)
    ):
        return _stage01_result(
            "NO_GO_INSUFFICIENT_OVERLAP",
            summaries,
            len(validated),
            len(validated_diagnostics),
        )
    aggregate_speedup = 1 - math.sqrt(
        (1 - by_shape[4]["median_speedup_ratio"])
        * (1 - by_shape[8]["median_speedup_ratio"])
    )
    performance_failed = (
        aggregate_speedup < MIN_AGGREGATE_SPEEDUP
        or any(
            by_shape[shape]["median_speedup_ratio"] < 0
            for shape in (4, 8)
        )
        or by_shape[1]["median_speedup_ratio"]
        < -MAX_SINGLE_TOKEN_MEDIAN_REGRESSION
        or any(
            row["p99_regression_ratio"] > MAX_P99_REGRESSION
            or row["host_submission_regression_ratio"]
            > MAX_HOST_SUBMISSION_REGRESSION
            for row in summaries
        )
        or any(
            by_shape[shape]["improving_pair_count"]
            < MIN_DIRECTIONAL_PAIR_COUNT
            for shape in (4, 8)
        )
    )
    return _stage01_result(
        (
            "NO_GO_PERFORMANCE"
            if performance_failed
            else "GO_COMPLETION_OWNED_OVERLAP_MICROGATE"
        ),
        summaries,
        len(validated),
        len(validated_diagnostics),
    )


def classify_stage0(rows, memory, cleanup):
    expected = {
        (shape, pair, rank)
        for shape in ACTIVE_TOKEN_GROUPS
        for pair in range(MEASURED_PAIR_COUNT)
        for rank in range(WORLD_SIZE)
    }
    validated = []
    seen = set()
    incomplete = False
    correctness_failed = False
    allocation_failed = False
    for raw in rows if isinstance(rows, (list, tuple)) else ():
        try:
            row = validate_measurement_row(raw)
        except ValueError:
            incomplete = True
            continue
        identity = (
            row["active_tokens"],
            row["pair_index"],
            row["rank"],
        )
        if identity in seen:
            incomplete = True
            continue
        seen.add(identity)
        validated.append(row)
        correctness_failed |= not all(
            row[name]
            for name in (
                "reduced_output_exact",
                "final_output_exact",
                "shadow_payload_exact",
                "active_state_preserved_before_publish",
                "published_state_exact",
                "abort_preserved_old_state",
                "commit_identity_match",
                "finite_output",
            )
        )
        correctness_failed |= row["timed_out"]
        allocation_failed |= row["timed_path_allocation_count"] != 0
    incomplete |= seen != expected
    if correctness_failed:
        return _result(
            "NO_GO_CORRECTNESS_OR_LIFECYCLE",
            [],
            len(validated),
        )

    memory_rows = memory.get("rank_rows") if isinstance(memory, dict) else None
    if not isinstance(memory_rows, list) or len(memory_rows) != WORLD_SIZE:
        incomplete = True
    else:
        for row in memory_rows:
            required = (
                row.get("maximum_reserved_delta_bytes"),
                row.get("maximum_theoretical_shadow_bytes"),
            )
            if any(not _finite_nonnegative(value) for value in required):
                incomplete = True
                continue
            allocation_failed |= (
                required[0] > required[1] + MAX_RESERVED_SLACK_BYTES
            )
    if allocation_failed:
        return _result(
            "NO_GO_MEMORY_OR_ALLOCATION",
            [],
            len(validated),
        )
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        incomplete = True
    if incomplete:
        return _result(
            "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT",
            [],
            len(validated),
        )

    summaries = []
    for shape in ACTIVE_TOKEN_GROUPS:
        pair_rows = []
        for pair in range(MEASURED_PAIR_COUNT):
            ranks = [
                row
                for row in validated
                if row["active_tokens"] == shape
                and row["pair_index"] == pair
            ]
            pair_rows.append({
                "baseline": max(row["baseline_critical_ns"] for row in ranks),
                "candidate": max(row["candidate_critical_ns"] for row in ranks),
                "baseline_host": max(
                    row["baseline_host_submission_ns"] for row in ranks
                ),
                "candidate_host": max(
                    row["candidate_host_submission_ns"] for row in ranks
                ),
                "overlap_ratio": min(
                    row["overlap_intersection_ns"]
                    / max(
                        1,
                        min(
                            row["allreduce_interval_ns"][1]
                            - row["allreduce_interval_ns"][0],
                            row["state_copy_interval_ns"][1]
                            - row["state_copy_interval_ns"][0],
                        ),
                    )
                    for row in ranks
                ),
            })
        baseline = [row["baseline"] for row in pair_rows]
        candidate = [row["candidate"] for row in pair_rows]
        baseline_median = median(baseline)
        candidate_median = median(candidate)
        baseline_p99 = _nearest_rank_percentile(baseline, 0.99)
        candidate_p99 = _nearest_rank_percentile(candidate, 0.99)
        baseline_host = median(row["baseline_host"] for row in pair_rows)
        candidate_host = median(row["candidate_host"] for row in pair_rows)
        summaries.append({
            "active_tokens": shape,
            "median_speedup_ratio": 1 - candidate_median / baseline_median,
            "p99_regression_ratio": candidate_p99 / baseline_p99 - 1,
            "host_submission_regression_ratio": (
                candidate_host / baseline_host - 1
            ),
            "median_realized_overlap_ratio": median(
                row["overlap_ratio"] for row in pair_rows
            ),
            "improving_pair_count": sum(
                row["candidate"] < row["baseline"] for row in pair_rows
            ),
        })

    by_shape = {row["active_tokens"]: row for row in summaries}
    if any(
        by_shape[shape]["median_realized_overlap_ratio"]
        < MIN_OVERLAP_RATIO
        for shape in (4, 8)
    ):
        return _result(
            "NO_GO_INSUFFICIENT_OVERLAP",
            summaries,
            len(validated),
        )
    aggregate_speedup = 1 - math.sqrt(
        (1 - by_shape[4]["median_speedup_ratio"])
        * (1 - by_shape[8]["median_speedup_ratio"])
    )
    performance_failed = (
        aggregate_speedup < MIN_AGGREGATE_SPEEDUP
        or any(
            by_shape[shape]["median_speedup_ratio"] < 0
            for shape in (4, 8)
        )
        or by_shape[1]["median_speedup_ratio"]
        < -MAX_SINGLE_TOKEN_MEDIAN_REGRESSION
        or any(
            row["p99_regression_ratio"] > MAX_P99_REGRESSION
            or row["host_submission_regression_ratio"]
            > MAX_HOST_SUBMISSION_REGRESSION
            for row in summaries
        )
        or any(
            by_shape[shape]["improving_pair_count"]
            < MIN_DIRECTIONAL_PAIR_COUNT
            for shape in (4, 8)
        )
    )
    return _result(
        "NO_GO_PERFORMANCE"
        if performance_failed
        else "GO_LEASE_SEALED_OVERLAP_MICROGATE",
        summaries,
        len(validated),
    )
