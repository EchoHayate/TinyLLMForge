"""Dependency-light tests for the P5 step-cost calibration contract."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


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
        num_kvcache_blocks=448,
        block_size=256,
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


class _IncrementingClock:
    def __init__(self, step_ns: int = 100):
        self.value = 0
        self.step_ns = step_ns

    def __call__(self):
        self.value += self.step_ns
        return self.value


class _FakeShapeEngine:
    def __init__(self, shape: dict):
        self.shape = shape
        self.last_step_observation = None
        self.decode_request_count = 0
        self.prefill_request_lengths = []
        self.pending_prefill_lengths = []
        self.next_seq_id = 0
        self.decode_seq_ids = []

    def add_request(self, prompt, sampling_params):
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        if sampling_params.max_tokens == 1:
            prompt_length = len(prompt)
            self.prefill_request_lengths.append(prompt_length)
            self.pending_prefill_lengths.append((seq_id, prompt_length))
        else:
            self.decode_request_count += 1
            self.decode_seq_ids.append(seq_id)

    def step(self):
        scheduled = [{
            "seq_id": seq_id,
            "is_decode": True,
            "prefill_chunk_start": 0,
            "prefill_chunk_end": 0,
        } for seq_id in self.decode_seq_ids]
        batch_kind = None
        if self.pending_prefill_lengths:
            batch_kind = "mixed"
            scheduled.extend({
                "seq_id": seq_id,
                "is_decode": False,
                "prefill_chunk_start": 0,
                "prefill_chunk_end": prompt_length,
            } for seq_id, prompt_length in self.pending_prefill_lengths)
            self.pending_prefill_lengths.clear()
        self.last_step_observation = {
            "batch_kind": batch_kind,
            "scheduled": scheduled,
        }
        return [], -len(self.decode_seq_ids)


def _sampling_params_factory(*, max_tokens: int):
    return SimpleNamespace(max_tokens=max_tokens)


def test_execute_shape_uses_exact_engine_batch_and_excludes_warmup():
    for shape in (
        next(
            row for row in _required_shapes()
            if row["kind"] == "decode"
            and row["context_class"] == "medium"
            and row["decode_rows"] == 8
        ),
        next(
            row for row in _required_shapes()
            if row["kind"] == "mixed"
            and row["prefill_tokens"] == 64
            and row["prefill_rows"] == 4
            and row["decode_rows"] == 32
        ),
    ):
        engine = _FakeShapeEngine(shape)
        synchronizations = []
        rows = calibration.execute_calibration_shape(
            shape,
            engine=engine,
            sampling_params_factory=_sampling_params_factory,
            synchronize=lambda: synchronizations.append("sync"),
            clock_ns=_IncrementingClock(),
        )

        assert engine.decode_request_count == shape["decode_rows"]
        assert len(rows) == 7
        assert [row["iteration"] for row in rows] == list(range(7))
        assert all(row["duration_ns"] == 100 for row in rows)
        assert all(
            {
                key: row[key]
                for key in (
                    "shape_id",
                    "kind",
                    "context_class",
                    "decode_rows",
                    "prefill_rows",
                    "prefill_tokens",
                    "warmup_iterations",
                    "measured_iterations",
                )
            } == shape
            for row in rows
        )
        assert len(synchronizations) == 2 * (
            shape["warmup_iterations"]
            + shape["measured_iterations"]
        )
        if shape["kind"] == "mixed":
            measured_batches = (
                shape["warmup_iterations"]
                + shape["measured_iterations"]
            )
            assert len(engine.prefill_request_lengths) == (
                measured_batches * shape["prefill_rows"]
            )
            assert all(
                sum(engine.prefill_request_lengths[
                    offset:offset + shape["prefill_rows"]
                ]) == shape["prefill_tokens"]
                for offset in range(
                    0,
                    len(engine.prefill_request_lengths),
                    shape["prefill_rows"],
                )
            )
        else:
            assert engine.prefill_request_lengths == []


def test_execute_shape_rejects_non_positive_synchronous_duration():
    shape = next(
        row for row in _required_shapes()
        if row["kind"] == "decode"
        and row["context_class"] == "short"
        and row["decode_rows"] == 1
    )
    _expect_error(
        ValueError,
        "non-positive synchronous duration",
        lambda: calibration.execute_calibration_shape(
            shape,
            engine=_FakeShapeEngine(shape),
            sampling_params_factory=_sampling_params_factory,
            synchronize=lambda: None,
            clock_ns=lambda: 10,
        ),
    )


def test_shape_orchestrator_uses_one_fresh_launch_and_port_pair_per_shape():
    shapes = _required_shapes()[:4]
    next_port = iter((19001, 19002, 19003, 19004, 19005, 19006, 19007, 19008))
    launches = []

    def allocate_port_pair():
        return next(next_port), next(next_port)

    def launch_shape(*, shape, tinyvllm_dist_port, master_port):
        launches.append((
            shape["shape_id"],
            tinyvllm_dist_port,
            master_port,
        ))
        return [{
            **shape,
            "iteration": iteration,
            "duration_ns": 100 + iteration,
        } for iteration in range(shape["measured_iterations"])]

    rows = calibration.orchestrate_calibration_shapes(
        shapes,
        allocate_port_pair=allocate_port_pair,
        launch_shape=launch_shape,
    )
    assert [row[0] for row in launches] == [
        shape["shape_id"] for shape in shapes
    ]
    assert len({port for launch in launches for port in launch[1:]}) == 8
    assert len(rows) == 4 * 7


def test_required_shapes_cover_decode_rows_contexts_and_mixed_cross_product():
    shapes = _required_shapes()
    decode = [shape for shape in shapes if shape["kind"] == "decode"]
    mixed = [shape for shape in shapes if shape["kind"] == "mixed"]
    assert {
        row["context_class"]: max(
            candidate["decode_rows"]
            for candidate in decode
            if candidate["context_class"] == row["context_class"]
        )
        for row in decode
    } == {
        "short": 448,
        "medium": 149,
        "long": 64,
    }
    assert {
        row["context_class"] for row in decode
    } == {"short", "medium", "long"}
    assert {
        row["prefill_tokens"] for row in mixed
    } == {16, 32, 64, 128}
    assert all(
        row["decode_rows"] + row["prefill_rows"] <= 512
        for row in mixed
    )
    assert {
        row["prefill_rows"]: max(
            candidate["decode_rows"]
            for candidate in mixed
            if candidate["prefill_rows"] == row["prefill_rows"]
        )
        for row in mixed
    } == {
        1: 149,
        2: 148,
        4: 148,
        8: 146,
    }
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
    for (
        max_num_seqs,
        max_prefill_tokens,
        num_kvcache_blocks,
        block_size,
    ) in (
        (0, 128, 448, 256),
        (512, 64, 448, 256),
        (True, 128, 448, 256),
        (512, True, 448, 256),
        (512, 128, 0, 256),
        (512, 128, True, 256),
        (512, 128, 448, 0),
        (512, 128, 448, True),
    ):
        _expect_error(
            ValueError,
            "unsupported P5 calibration limits",
            lambda max_num_seqs=max_num_seqs,
            max_prefill_tokens=max_prefill_tokens: (
                calibration.build_required_shapes(
                    max_num_seqs=max_num_seqs,
                    max_prefill_tokens=max_prefill_tokens,
                    num_kvcache_blocks=num_kvcache_blocks,
                    block_size=block_size,
                )
            ),
        )


def test_required_shapes_reject_infeasible_required_decode_counts():
    _expect_error(
        ValueError,
        "infeasible required calibration shape",
        lambda: calibration.build_required_shapes(
            max_num_seqs=512,
            max_prefill_tokens=128,
            num_kvcache_blocks=31,
            block_size=256,
        ),
    )


def test_nearest_rank_and_inflation_use_integer_contract():
    values = [11, 17, 13, 19, 15, 23, 21]
    assert calibration.nearest_rank_p99_ns(values) == 23
    assert calibration.inflate_duration_ns(23) == 29
    assert calibration.inflate_duration_ns(4) == 5


def test_integer_envelope_uses_observed_max_and_25_percent_ceiling():
    result = calibration.recompute_cost_envelope(
        _complete_rows(),
        required_shapes=_required_shapes(),
    )
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
            lambda rows=rows: calibration.recompute_cost_envelope(
                rows,
                required_shapes=_required_shapes(),
            ),
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
            lambda: calibration.recompute_cost_envelope(
                rows,
                required_shapes=_required_shapes(),
            ),
        )
    finally:
        calibration.INT64_MAX = original_limit


def test_summary_binds_source_environment_engine_shapes_and_raw_rows():
    shapes = _required_shapes()
    rows = _complete_rows()
    envelope = calibration.recompute_cost_envelope(
        rows,
        required_shapes=shapes,
    )
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
        "cost_calibration_capacity.json",
        "cost_calibration_manifest.jsonl",
        "cost_calibration_rows.jsonl",
        "cost_calibration_summary.json",
    )
    assert {
        "tools/arrival_load_cost_calibration.py",
        "tools/test_arrival_load_cost_calibration.py",
    }.issubset(set(gate.OWNED_SOURCE_ROOTS))


def test_capacity_evidence_binds_base_config_and_resolved_capacity():
    base_config = {
        "max_num_seqs": 512,
        "max_num_prefill_tokens_per_step": 128,
    }
    evidence = calibration.build_capacity_evidence(
        base_engine_config=base_config,
        num_kvcache_blocks=448,
        block_size=256,
    )
    assert evidence == {
        "schema_version": 1,
        "base_engine_config_sha256":
            calibration.canonical_json_sha256(base_config),
        "num_kvcache_blocks": 448,
        "block_size": 256,
        "resolved_engine_config": {
            **base_config,
            "num_kvcache_blocks": 448,
        },
    }
    for num_kvcache_blocks, block_size in (
        (0, 256),
        (True, 256),
        (448, 0),
        (448, True),
    ):
        _expect_error(
            ValueError,
            "invalid calibration capacity",
            lambda num_kvcache_blocks=num_kvcache_blocks,
            block_size=block_size: calibration.build_capacity_evidence(
                base_engine_config=base_config,
                num_kvcache_blocks=num_kvcache_blocks,
                block_size=block_size,
            ),
        )


def main():
    test_execute_shape_uses_exact_engine_batch_and_excludes_warmup()
    test_execute_shape_rejects_non_positive_synchronous_duration()
    test_shape_orchestrator_uses_one_fresh_launch_and_port_pair_per_shape()
    test_required_shapes_cover_decode_rows_contexts_and_mixed_cross_product()
    test_required_shapes_reject_unsupported_limits()
    test_required_shapes_reject_infeasible_required_decode_counts()
    test_nearest_rank_and_inflation_use_integer_contract()
    test_integer_envelope_uses_observed_max_and_25_percent_ceiling()
    test_calibration_rejects_missing_short_or_invalid_rows()
    test_calibration_rejects_int64_overflow()
    test_summary_binds_source_environment_engine_shapes_and_raw_rows()
    test_gate_freezes_cost_calibration_artifact_and_source_contract()
    test_capacity_evidence_binds_base_config_and_resolved_capacity()
    print("arrival load cost calibration tests passed")


if __name__ == "__main__":
    main()
