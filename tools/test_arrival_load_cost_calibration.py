"""Dependency-light tests for the P5 step-cost calibration contract."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "arrival_load_cost_calibration.py"
GATE_PATH = REPO_ROOT / "tools" / "arrival_load_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_cost_calibration",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_cost_calibration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibration = _load_module()


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_gate_for_cost_calibration_test",
        GATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_shapes():
    return calibration.build_required_shapes(
        max_num_seqs=512,
        max_prefill_tokens=128,
    )


def _complete_rows():
    rows = []
    for shape in _required_shapes():
        if shape["kind"] == "decode":
            context_cost = {
                "short": 100_000,
                "medium": 200_000,
                "long": 300_000,
            }[shape["context_class"]]
            base_duration_ns = (
                1_000_000
                + context_cost
                + shape["decode_rows"] * 1_000
            )
        else:
            base_duration_ns = (
                1_300_000
                + shape["decode_rows"] * 1_000
                + shape["prefill_rows"] * 10_000
                + shape["prefill_tokens"] * 20_000
            )
        for iteration in range(shape["measured_iterations"]):
            rows.append({
                **shape,
                "iteration": iteration,
                "duration_ns": base_duration_ns + iteration,
            })
    return rows


def _expect_error(exception_type, message: str, callback):
    try:
        callback()
    except exception_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(
            f"expected {exception_type.__name__}: {message}"
        )


def test_required_shapes_cover_decode_rows_contexts_and_mixed_cross_product():
    shapes = _required_shapes()
    decode = [shape for shape in shapes if shape["kind"] == "decode"]
    mixed = [shape for shape in shapes if shape["kind"] == "mixed"]
    assert {row["decode_rows"] for row in decode} == {1, 8, 32, 512}
    assert {
        row["context_class"] for row in decode
    } == {"short", "medium", "long"}
    assert {
        row["prefill_tokens"] for row in mixed
    } == {16, 32, 64, 128}
    assert {row["decode_rows"] for row in mixed} == {1, 8, 32, 512}
    assert {
        (row["prefill_tokens"], row["prefill_rows"])
        for row in mixed
    } == {
        (16, 1),
        (32, 1),
        (32, 2),
        (64, 1),
        (64, 4),
        (128, 1),
        (128, 8),
    }
    assert len({row["shape_id"] for row in shapes}) == len(shapes)
    assert all(row["measured_iterations"] == 7 for row in shapes)
    assert all(row["warmup_iterations"] >= 1 for row in shapes)


def test_required_shapes_reject_unsupported_limits():
    for max_num_seqs, max_prefill_tokens in (
        (0, 128),
        (512, 64),
        (True, 128),
        (512, True),
    ):
        _expect_error(
            ValueError,
            "unsupported P5 calibration limits",
            lambda max_num_seqs=max_num_seqs,
            max_prefill_tokens=max_prefill_tokens: (
                calibration.build_required_shapes(
                    max_num_seqs=max_num_seqs,
                    max_prefill_tokens=max_prefill_tokens,
                )
            ),
        )


def test_nearest_rank_and_inflation_use_integer_contract():
    values = [11, 17, 13, 19, 15, 23, 21]
    assert calibration.nearest_rank_p99_ns(values) == 23
    assert calibration.inflate_duration_ns(23) == 29
    assert calibration.inflate_duration_ns(4) == 5


def test_integer_envelope_uses_observed_max_and_25_percent_ceiling():
    result = calibration.recompute_cost_envelope(_complete_rows())
    decode = [
        row for row in result["shape_summaries"]
        if row["kind"] == "decode"
    ]
    mixed = [
        row for row in result["shape_summaries"]
        if row["kind"] == "mixed"
    ]
    assert result["cost_intercept_ns"] == max(
        row["inflated_duration_ns"] for row in decode
    )
    assert all(
        result["cost_intercept_ns"]
        + row["prefill_tokens"]
        * result["cost_per_prefill_token_ns"]
        >= row["inflated_duration_ns"]
        for row in mixed
    )
    assert result["shape_summaries"][0]["measured_p99_ns"] == max(
        result["shape_summaries"][0]["measured_duration_ns"]
    )
    assert result["cost_per_prefill_token_ns"] > 0
    assert any(
        result["cost_intercept_ns"]
        + row["prefill_tokens"]
        * (result["cost_per_prefill_token_ns"] - 1)
        < row["inflated_duration_ns"]
        for row in mixed
    )


def test_calibration_rejects_missing_short_or_invalid_rows():
    complete = _complete_rows()
    first_shape_id = complete[0]["shape_id"]
    cases = []
    cases.append((
        [
            row for row in complete
            if row["shape_id"] != first_shape_id
        ],
        "missing required calibration shapes",
    ))
    cases.append((
        [
            row for row in complete
            if not (
                row["shape_id"] == first_shape_id
                and row["iteration"] == 6
            )
        ],
        "at least seven samples",
    ))
    non_positive = [dict(row) for row in complete]
    non_positive[0]["duration_ns"] = 0
    cases.append((non_positive, "invalid calibration duration"))
    non_integer = [dict(row) for row in complete]
    non_integer[0]["duration_ns"] = math.nan
    cases.append((non_integer, "invalid calibration duration"))
    duplicate = [dict(row) for row in complete]
    duplicate.append(dict(duplicate[0]))
    cases.append((duplicate, "duplicate calibration iteration"))
    inconsistent = [dict(row) for row in complete]
    inconsistent[1]["decode_rows"] += 1
    cases.append((inconsistent, "inconsistent calibration shape"))
    infeasible = [dict(row) for row in complete]
    mixed_index = next(
        index for index, row in enumerate(infeasible)
        if row["kind"] == "mixed"
    )
    infeasible[mixed_index]["prefill_tokens"] = 0
    cases.append((infeasible, "inconsistent calibration shape"))

    for rows, message in cases:
        _expect_error(
            ValueError,
            message,
            lambda rows=rows: calibration.recompute_cost_envelope(rows),
        )


def test_calibration_rejects_int64_overflow():
    _expect_error(
        OverflowError,
        "calibration inflation overflows int64",
        lambda: calibration.inflate_duration_ns(
            calibration.INT64_MAX
        ),
    )

    rows = _complete_rows()
    original_limit = calibration.INT64_MAX
    try:
        calibration.INT64_MAX = 1_000
        for row in rows:
            row["duration_ns"] = (
                768 if row["kind"] == "mixed" else 760
            )
        _expect_error(
            OverflowError,
            "cost envelope overflows int64",
            lambda: calibration.recompute_cost_envelope(rows),
        )
    finally:
        calibration.INT64_MAX = original_limit


def test_summary_binds_source_environment_engine_shapes_and_raw_rows():
    shapes = _required_shapes()
    rows = _complete_rows()
    envelope = calibration.recompute_cost_envelope(rows)
    summary = calibration.build_cost_calibration_summary(
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        engine_config_sha256="c" * 64,
        required_shapes=shapes,
        raw_rows=rows,
    )
    assert summary == {
        "status": "PASS",
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "engine_config_sha256": "c" * 64,
        "required_shape_sha256": calibration.canonical_json_sha256(
            shapes
        ),
        "raw_rows_sha256": calibration.canonical_json_sha256(rows),
        "cost_intercept_ns": envelope["cost_intercept_ns"],
        "cost_per_prefill_token_ns": envelope[
            "cost_per_prefill_token_ns"
        ],
        "envelope_sha256": calibration.canonical_json_sha256(
            envelope
        ),
    }


def test_gate_freezes_cost_calibration_artifact_and_source_contract():
    gate = _load_gate()
    assert gate.COST_CALIBRATION_ARTIFACT_FILES == (
        "cost_calibration_manifest.jsonl",
        "cost_calibration_rows.jsonl",
        "cost_calibration_summary.json",
    )
    assert {
        "tools/arrival_load_cost_calibration.py",
        "tools/test_arrival_load_cost_calibration.py",
    }.issubset(set(gate.OWNED_SOURCE_ROOTS))


def main():
    test_required_shapes_cover_decode_rows_contexts_and_mixed_cross_product()
    test_required_shapes_reject_unsupported_limits()
    test_nearest_rank_and_inflation_use_integer_contract()
    test_integer_envelope_uses_observed_max_and_25_percent_ceiling()
    test_calibration_rejects_missing_short_or_invalid_rows()
    test_calibration_rejects_int64_overflow()
    test_summary_binds_source_environment_engine_shapes_and_raw_rows()
    test_gate_freezes_cost_calibration_artifact_and_source_contract()
    print("arrival load cost calibration tests passed")


if __name__ == "__main__":
    main()
