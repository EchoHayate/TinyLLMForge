"""Pure contracts for source-bound P5 step-cost calibration."""

from __future__ import annotations

import hashlib
import json
import math


INT64_MAX = (1 << 63) - 1
CANONICAL_MAX_NUM_SEQS = 512
CANONICAL_MAX_PREFILL_TOKENS = 128


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build_required_shapes(
    *,
    max_num_seqs: int,
    max_prefill_tokens: int,
) -> list[dict]:
    if (
        not _is_int(max_num_seqs)
        or not _is_int(max_prefill_tokens)
        or max_num_seqs <= 0
        or max_prefill_tokens != CANONICAL_MAX_PREFILL_TOKENS
    ):
        raise ValueError("unsupported P5 calibration limits")
    decode_counts = sorted({
        count for count in (1, 8, 32, max_num_seqs)
        if count <= max_num_seqs
    })
    shapes = []
    for context_class in ("short", "medium", "long"):
        for decode_rows in decode_counts:
            shapes.append({
                "shape_id": f"decode-{context_class}-d{decode_rows}",
                "kind": "decode",
                "context_class": context_class,
                "decode_rows": decode_rows,
                "prefill_rows": 0,
                "prefill_tokens": 0,
                "warmup_iterations": 1,
                "measured_iterations": 7,
            })
    for prefill_tokens in (16, 32, 64, 128):
        prefill_rows = sorted({
            1,
            min(8, prefill_tokens // 16),
        })
        for decode_rows in decode_counts:
            for row_count in prefill_rows:
                shapes.append({
                    "shape_id": (
                        f"mixed-p{prefill_tokens}-r{row_count}"
                        f"-d{decode_rows}"
                    ),
                    "kind": "mixed",
                    "context_class": "mixed",
                    "decode_rows": decode_rows,
                    "prefill_rows": row_count,
                    "prefill_tokens": prefill_tokens,
                    "warmup_iterations": 1,
                    "measured_iterations": 7,
                })
    return shapes


def nearest_rank_p99_ns(values: list[int]) -> int:
    if len(values) < 7:
        raise ValueError("cost calibration requires at least seven samples")
    if any(not _is_int(value) or value <= 0 for value in values):
        raise ValueError("invalid calibration duration")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.99) - 1]


def inflate_duration_ns(measured_p99_ns: int) -> int:
    if not _is_int(measured_p99_ns) or measured_p99_ns <= 0:
        raise ValueError("invalid measured p99 duration")
    inflated = (measured_p99_ns * 5 + 3) // 4
    if inflated > INT64_MAX:
        raise OverflowError("calibration inflation overflows int64")
    return inflated


def _shape_contract(shape: dict) -> dict:
    fields = (
        "shape_id",
        "kind",
        "context_class",
        "decode_rows",
        "prefill_rows",
        "prefill_tokens",
        "warmup_iterations",
        "measured_iterations",
    )
    if not isinstance(shape, dict) or any(
        field not in shape for field in fields
    ):
        raise ValueError("invalid calibration shape")
    return {field: shape[field] for field in fields}


def recompute_cost_envelope(
    rows: list[dict],
    *,
    required_shapes: list[dict] | None = None,
) -> dict:
    if required_shapes is None:
        required_shapes = build_required_shapes(
            max_num_seqs=CANONICAL_MAX_NUM_SEQS,
            max_prefill_tokens=CANONICAL_MAX_PREFILL_TOKENS,
        )
    expected_by_id = {}
    for shape in required_shapes:
        contract = _shape_contract(shape)
        shape_id = contract["shape_id"]
        if shape_id in expected_by_id:
            raise ValueError("duplicate required calibration shape")
        expected_by_id[shape_id] = contract

    if not isinstance(rows, list):
        raise ValueError("calibration rows must be a list")
    grouped = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid calibration row")
        shape_id = row.get("shape_id")
        if shape_id not in expected_by_id:
            raise ValueError("unexpected calibration shape")
        if _shape_contract(row) != expected_by_id[shape_id]:
            raise ValueError("inconsistent calibration shape")
        iteration = row.get("iteration")
        if not _is_int(iteration) or iteration < 0:
            raise ValueError("invalid calibration iteration")
        shape_rows = grouped.setdefault(shape_id, {})
        if iteration in shape_rows:
            raise ValueError("duplicate calibration iteration")
        shape_rows[iteration] = row

    missing = sorted(set(expected_by_id) - set(grouped))
    if missing:
        raise ValueError(
            "missing required calibration shapes: " + ", ".join(missing)
        )

    shape_summaries = []
    for shape_id in sorted(expected_by_id):
        shape = expected_by_id[shape_id]
        shape_rows = grouped[shape_id]
        expected_iterations = shape["measured_iterations"]
        if set(shape_rows) != set(range(expected_iterations)):
            if len(shape_rows) < 7:
                raise ValueError(
                    "cost calibration requires at least seven samples"
                )
            raise ValueError("invalid calibration iteration set")
        durations = [
            shape_rows[index].get("duration_ns")
            for index in range(expected_iterations)
        ]
        measured_p99_ns = nearest_rank_p99_ns(durations)
        shape_summaries.append({
            **shape,
            "measured_duration_ns": durations,
            "measured_p99_ns": measured_p99_ns,
            "inflated_duration_ns": inflate_duration_ns(
                measured_p99_ns
            ),
        })

    decode_points = [
        row for row in shape_summaries if row["kind"] == "decode"
    ]
    mixed_points = [
        row for row in shape_summaries if row["kind"] == "mixed"
    ]
    if not decode_points or not mixed_points:
        raise ValueError("incomplete calibration shape classes")
    intercept = max(
        row["inflated_duration_ns"] for row in decode_points
    )
    slope = 1
    for point in mixed_points:
        tokens = point["prefill_tokens"]
        if not _is_int(tokens) or tokens <= 0:
            raise ValueError("infeasible calibration shape")
        excess = max(0, point["inflated_duration_ns"] - intercept)
        required = max(1, (excess + tokens - 1) // tokens)
        slope = max(slope, required)

    max_tokens = max(
        row["prefill_tokens"] for row in mixed_points
    )
    if (
        intercept > INT64_MAX
        or slope > (INT64_MAX - intercept) // max_tokens
    ):
        raise OverflowError("cost envelope overflows int64")
    for point in mixed_points:
        predicted = intercept + point["prefill_tokens"] * slope
        if predicted < point["inflated_duration_ns"]:
            raise ValueError("cost envelope does not dominate point")
    return {
        "shape_summaries": shape_summaries,
        "cost_intercept_ns": intercept,
        "cost_per_prefill_token_ns": slope,
    }


def build_cost_calibration_summary(
    *,
    source_tree_sha256: str,
    environment_sha256: str,
    engine_config_sha256: str,
    required_shapes: list[dict],
    raw_rows: list[dict],
) -> dict:
    identities = (
        source_tree_sha256,
        environment_sha256,
        engine_config_sha256,
    )
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in identities
    ):
        raise ValueError("invalid cost calibration identity")
    envelope = recompute_cost_envelope(
        raw_rows,
        required_shapes=required_shapes,
    )
    return {
        "status": "PASS",
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "engine_config_sha256": engine_config_sha256,
        "required_shape_sha256": canonical_json_sha256(
            required_shapes
        ),
        "raw_rows_sha256": canonical_json_sha256(raw_rows),
        "cost_intercept_ns": envelope["cost_intercept_ns"],
        "cost_per_prefill_token_ns": envelope[
            "cost_per_prefill_token_ns"
        ],
        "envelope_sha256": canonical_json_sha256(envelope),
    }
