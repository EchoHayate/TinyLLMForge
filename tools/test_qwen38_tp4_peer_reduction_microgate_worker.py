from __future__ import annotations

import pytest

from tools.qwen38_tp4_peer_reduction_microgate_worker import (
    ARTIFACT_NAMES,
    build_argument_parser,
    build_workload_schedule,
    validate_measurement_row,
)


def test_workload_schedule_freezes_shape_order_warmups_and_pairs():
    schedule = build_workload_schedule()

    assert [group["active_tokens"] for group in schedule] == [1, 4, 8]
    assert all(len(group["warmups"]) == 2 for group in schedule)
    assert all(len(group["measurements"]) == 200 for group in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "candidate",
    )
    assert schedule[0]["measurements"][1]["arm_order"] == (
        "candidate",
        "baseline",
    )
    assert schedule[-1]["measurements"][-1]["pair_index"] == 199


def test_workload_schedule_uses_immutable_distinct_seeds():
    schedule = build_workload_schedule()

    assert [group["seed"] for group in schedule] == [
        2026083001,
        2026083004,
        2026083008,
    ]
    assert len({group["seed"] for group in schedule}) == 3


def test_argument_parser_requires_remote_identity_fields():
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args([
        "--attempt",
        "attempt-r1",
        "--source-revision",
        "a" * 40,
        "--output-dir",
        "/approved/output",
        "--rank",
        "0",
        "--world-size",
        "4",
        "--dist-port",
        "29601",
    ])
    assert args.attempt == "attempt-r1"
    assert args.world_size == 4


def test_worker_declares_exact_compact_artifacts():
    assert ARTIFACT_NAMES == (
        "peer_access_matrix.json",
        "ipc_roundtrip.jsonl",
        "microgate_rows.jsonl",
        "memory_summary.json",
        "cleanup.json",
    )


def _valid_measurement_row():
    return {
        "active_tokens": 4,
        "pair_index": 17,
        "rank": 2,
        "arm_order": ["candidate", "baseline"],
        "baseline_cuda_ns": 100_000,
        "candidate_cuda_ns": 80_000,
        "baseline_host_submission_ns": 20_000,
        "candidate_host_submission_ns": 15_000,
        "cross_rank_max_abs_error": 0.0,
        "cross_rank_max_rel_error": 0.0,
        "baseline_max_abs_error": 0.0,
        "baseline_max_rel_error": 0.0,
        "timed_out": False,
        "device_status": 0,
    }


@pytest.mark.parametrize(
    "field",
    (
        "timed_out",
        "device_status",
        "baseline_cuda_ns",
        "candidate_cuda_ns",
        "cross_rank_max_abs_error",
        "baseline_max_abs_error",
    ),
)
def test_measurement_row_requires_timing_correctness_and_status(field):
    row = _valid_measurement_row()
    row.pop(field)

    with pytest.raises(ValueError, match=field):
        validate_measurement_row(row)


def test_measurement_row_rejects_nonfinite_and_wrong_rank_inventory():
    row = _valid_measurement_row()
    row["candidate_cuda_ns"] = float("nan")
    with pytest.raises(ValueError, match="candidate_cuda_ns"):
        validate_measurement_row(row)

    row = _valid_measurement_row()
    row["rank"] = 4
    with pytest.raises(ValueError, match="rank"):
        validate_measurement_row(row)
