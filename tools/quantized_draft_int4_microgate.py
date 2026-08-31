from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median


@dataclass(frozen=True)
class DraftLinearShape:
    shape_id: str
    input_features: int
    output_features: int
    execution_count: int
    group_size: int


@dataclass(frozen=True)
class QuantizedDraftInt4Policy:
    minimum_pairs_per_shape: int = 200
    maximum_candidate_to_bf16_median_ratio: float = 0.75
    maximum_candidate_to_bf16_p99_ratio: float = 0.95
    maximum_weight_bytes_ratio: float = 0.40
    maximum_absolute_error: float = 0.08
    maximum_relative_error: float = 0.08


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_shape_manifest(payload):
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("shapes"), list)
        or not payload["shapes"]
    ):
        raise ValueError("shape manifest is invalid")

    shapes = []
    shape_ids = set()
    for row in payload["shapes"]:
        if not isinstance(row, dict):
            raise ValueError("shape row must be an object")
        shape_id = row.get("shape_id")
        if (
            not isinstance(shape_id, str)
            or not shape_id
            or shape_id in shape_ids
        ):
            raise ValueError("shape_id must be unique and non-empty")
        input_features = _positive_integer(
            row.get("input_features"),
            "input_features",
        )
        output_features = _positive_integer(
            row.get("output_features"),
            "output_features",
        )
        execution_count = _positive_integer(
            row.get("execution_count"),
            "execution_count",
        )
        group_size = _positive_integer(
            row.get("group_size"),
            "group_size",
        )
        if input_features % group_size != 0:
            raise ValueError(
                "input_features must be divisible by group_size"
            )
        if input_features % 2 != 0:
            raise ValueError(
                "input_features must support packed INT4"
            )
        shape_ids.add(shape_id)
        shapes.append(DraftLinearShape(
            shape_id=shape_id,
            input_features=input_features,
            output_features=output_features,
            execution_count=execution_count,
            group_size=group_size,
        ))
    return tuple(shapes)


def _nearest_rank(values, percentile):
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _expected_weight_bytes(shapes):
    bf16 = sum(
        shape.output_features
        * shape.input_features
        * 2
        * shape.execution_count
        for shape in shapes
    )
    packed = sum(
        (
            shape.output_features * (shape.input_features // 2)
            + shape.output_features
            * (shape.input_features // shape.group_size)
            * 4
        )
        * shape.execution_count
        for shape in shapes
    )
    return bf16, packed


def _empty_result(classification, *, shape_summaries=()):
    return {
        "classification": classification,
        "shape_summaries": list(shape_summaries),
    }


def classify_int4_microgate(
    *,
    shapes,
    rows,
    memory,
    graph,
    cleanup,
):
    policy = QuantizedDraftInt4Policy()
    if (
        not isinstance(shapes, tuple)
        or not shapes
        or any(not isinstance(shape, DraftLinearShape) for shape in shapes)
        or len({shape.shape_id for shape in shapes}) != len(shapes)
    ):
        return _empty_result("INCONCLUSIVE_EVIDENCE")

    shape_by_id = {shape.shape_id: shape for shape in shapes}
    grouped = {shape.shape_id: {} for shape in shapes}
    incomplete = not isinstance(rows, (list, tuple)) or not rows
    correctness_failed = False

    required_timings = (
        "bf16_cuda_ns",
        "dequant_cuda_ns",
        "fused_int4_cuda_ns",
        "bf16_host_submission_ns",
        "dequant_host_submission_ns",
        "fused_int4_host_submission_ns",
    )
    if not incomplete:
        for row in rows:
            if not isinstance(row, dict):
                incomplete = True
                continue
            shape_id = row.get("shape_id")
            pair_index = row.get("pair_index")
            if (
                shape_id not in shape_by_id
                or isinstance(pair_index, bool)
                or not isinstance(pair_index, int)
                or pair_index < 0
                or pair_index in grouped[shape_id]
            ):
                incomplete = True
                continue
            if (
                row.get("arm_order")
                not in (
                    ["bf16", "dequant", "fused_int4"],
                    ["fused_int4", "dequant", "bf16"],
                )
                or any(
                    not _finite_nonnegative(row.get(name))
                    or row[name] <= 0
                    for name in required_timings
                )
                or not _finite_nonnegative(
                    row.get("maximum_absolute_error")
                )
                or not _finite_nonnegative(
                    row.get("maximum_relative_error")
                )
                or row.get("fallback_reason") is not None
                or row.get("full_dequant_allocation_observed")
                is not False
            ):
                incomplete = True
                continue
            if (
                row["maximum_absolute_error"]
                > policy.maximum_absolute_error
                or row["maximum_relative_error"]
                > policy.maximum_relative_error
            ):
                correctness_failed = True
            grouped[shape_id][pair_index] = dict(row)

    expected_bf16, minimum_packed = _expected_weight_bytes(shapes)
    memory_summary = None
    memory_failed = False
    if not isinstance(memory, dict):
        incomplete = True
    else:
        observed_bf16 = memory.get("observed_bf16_weight_bytes")
        observed_candidate = memory.get(
            "observed_candidate_weight_bytes"
        )
        observed_minimum = memory.get(
            "minimum_packed_weight_bytes"
        )
        allocated_delta = memory.get(
            "maximum_candidate_allocated_delta_bytes"
        )
        if (
            memory.get("classification") != "PASS"
            or isinstance(observed_bf16, bool)
            or not isinstance(observed_bf16, int)
            or observed_bf16 != expected_bf16
            or isinstance(observed_candidate, bool)
            or not isinstance(observed_candidate, int)
            or observed_candidate < minimum_packed
            or observed_minimum != minimum_packed
            or not _finite_nonnegative(allocated_delta)
            or memory.get("full_dequant_allocation_observed")
            is not False
        ):
            incomplete = True
        else:
            weight_ratio = observed_candidate / observed_bf16
            memory_summary = {
                "observed_bf16_weight_bytes": observed_bf16,
                "observed_candidate_weight_bytes": observed_candidate,
                "minimum_packed_weight_bytes": minimum_packed,
                "weight_bytes_ratio": weight_ratio,
                "maximum_candidate_allocated_delta_bytes": (
                    allocated_delta
                ),
            }
            memory_failed = (
                weight_ratio > policy.maximum_weight_bytes_ratio
            )

    graph_failed = False
    if not isinstance(graph, dict):
        incomplete = True
    else:
        graph_rows = graph.get("shapes")
        if not isinstance(graph_rows, list):
            incomplete = True
            graph_rows = []
        graph_by_id = {}
        for row in graph_rows:
            if not isinstance(row, dict):
                incomplete = True
                continue
            shape_id = row.get("shape_id")
            if shape_id not in shape_by_id or shape_id in graph_by_id:
                incomplete = True
                continue
            if (
                not _finite_nonnegative(
                    row.get("maximum_absolute_error")
                )
                or not _finite_nonnegative(
                    row.get("maximum_relative_error")
                )
            ):
                incomplete = True
                continue
            if (
                row["maximum_absolute_error"]
                > policy.maximum_absolute_error
                or row["maximum_relative_error"]
                > policy.maximum_relative_error
            ):
                correctness_failed = True
            if (
                row.get("capture_succeeded") is not True
                or isinstance(row.get("replay_count"), bool)
                or not isinstance(row.get("replay_count"), int)
                or row["replay_count"] < 2
                or row.get("static_pointers_stable") is not True
            ):
                graph_failed = True
            graph_by_id[shape_id] = row
        if set(graph_by_id) != set(shape_by_id):
            incomplete = True
        if graph.get("classification") != "PASS":
            graph_failed = True

    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        incomplete = True

    shape_summaries = []
    performance_failed = False
    for shape in shapes:
        pair_rows = grouped[shape.shape_id]
        required_pair_ids = set(range(policy.minimum_pairs_per_shape))
        if set(pair_rows) != required_pair_ids:
            incomplete = True
            continue
        ordered = [pair_rows[index] for index in sorted(pair_rows)]
        first_arm_counts = {
            "bf16": sum(
                row["arm_order"][0] == "bf16"
                for row in ordered
            ),
            "fused_int4": sum(
                row["arm_order"][0] == "fused_int4"
                for row in ordered
            ),
        }
        if abs(
            first_arm_counts["bf16"]
            - first_arm_counts["fused_int4"]
        ) > 1:
            incomplete = True
            continue
        bf16 = [row["bf16_cuda_ns"] for row in ordered]
        fused = [row["fused_int4_cuda_ns"] for row in ordered]
        dequant = [row["dequant_cuda_ns"] for row in ordered]
        bf16_median = median(bf16)
        fused_median = median(fused)
        dequant_median = median(dequant)
        bf16_p99 = _nearest_rank(bf16, 0.99)
        fused_p99 = _nearest_rank(fused, 0.99)
        median_ratio = fused_median / bf16_median
        p99_ratio = fused_p99 / bf16_p99
        if (
            median_ratio
            > policy.maximum_candidate_to_bf16_median_ratio
            or p99_ratio
            > policy.maximum_candidate_to_bf16_p99_ratio
        ):
            performance_failed = True
        shape_summaries.append({
            "shape_id": shape.shape_id,
            "execution_count": shape.execution_count,
            "pair_count": len(ordered),
            "bf16_median_cuda_ns": bf16_median,
            "dequant_median_cuda_ns": dequant_median,
            "fused_int4_median_cuda_ns": fused_median,
            "candidate_to_bf16_median_ratio": median_ratio,
            "bf16_p99_cuda_ns": bf16_p99,
            "fused_int4_p99_cuda_ns": fused_p99,
            "candidate_to_bf16_p99_ratio": p99_ratio,
        })

    if correctness_failed:
        return _empty_result(
            "NO_GO_CORRECTNESS",
            shape_summaries=shape_summaries,
        )
    if incomplete:
        return _empty_result(
            "INCONCLUSIVE_EVIDENCE",
            shape_summaries=shape_summaries,
        )

    total_executions = sum(
        summary["execution_count"] for summary in shape_summaries
    )
    weighted_summary = {
        "candidate_to_bf16_median_ratio": sum(
            summary["candidate_to_bf16_median_ratio"]
            * summary["execution_count"]
            for summary in shape_summaries
        ) / total_executions,
        "candidate_to_bf16_p99_ratio": sum(
            summary["candidate_to_bf16_p99_ratio"]
            * summary["execution_count"]
            for summary in shape_summaries
        ) / total_executions,
        "execution_count": total_executions,
    }
    result = {
        "classification": "GO_FUSED_INT4_DRAFT_KERNEL",
        "shape_summaries": shape_summaries,
        "weighted_summary": weighted_summary,
        "memory_summary": memory_summary,
    }
    if memory_failed:
        result["classification"] = "NO_GO_MEMORY"
    elif graph_failed:
        result["classification"] = "NO_GO_GRAPH"
    elif performance_failed:
        result["classification"] = "NO_GO_PERFORMANCE"
    return result
