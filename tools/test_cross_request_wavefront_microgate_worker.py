from __future__ import annotations

import inspect

import pytest

from tools.cross_request_wavefront_microgate_worker import (
    ARTIFACT_NAMES,
    WavefrontBuffers,
    _run_candidate,
    build_argument_parser,
    build_workload_schedule,
    validate_measurement_row,
)


class _FakeCuda:
    def __init__(self):
        self.streams = []
        self.events = []

    def Stream(self, *, device):
        stream = ("stream", device, len(self.streams))
        self.streams.append(stream)
        return stream

    def Event(self, *, enable_timing):
        event = ("event", enable_timing, len(self.events))
        self.events.append(event)
        return event


class _FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self):
        self.cuda = _FakeCuda()
        self.allocations = []

    def empty(self, shape, *, dtype, device):
        allocation = {
            "shape": shape,
            "dtype": dtype,
            "device": device,
        }
        self.allocations.append(allocation)
        return allocation


def test_wavefront_buffers_preallocate_streams_events_and_tensors():
    torch = _FakeTorch()

    buffers = WavefrontBuffers.create(torch, "cuda:0", active_tokens=8)

    assert len(buffers.compute_streams) == 2
    assert buffers.communication_stream == ("stream", "cuda:0", 2)
    assert buffers.baseline_started != buffers.origin
    assert buffers.baseline_completed != buffers.completed
    assert len(torch.cuda.events) == 16
    assert [row["shape"] for row in torch.allocations] == [
        (8, 5120),
        (8, 5120),
        (8, 5120),
        (8, 5120),
        (8, 5120),
        (8, 5120),
    ]
    assert [row["dtype"] for row in torch.allocations] == [
        "float32",
        "bfloat16",
        "bfloat16",
        "float32",
        "bfloat16",
        "bfloat16",
    ]


def test_schedule_freezes_two_shapes_warmups_pairs_and_abba_order():
    schedule = build_workload_schedule()

    assert [row["active_tokens"] for row in schedule] == [4, 8]
    assert all(len(row["warmups"]) == 2 for row in schedule)
    assert all(len(row["measurements"]) == 300 for row in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "candidate",
    )
    assert schedule[0]["measurements"][1]["arm_order"] == (
        "candidate",
        "baseline",
    )
    assert schedule[-1]["measurements"][-1]["pair_index"] == 299


def test_schedule_uses_stable_distinct_shape_seeds():
    schedule = build_workload_schedule()

    assert [row["seed"] for row in schedule] == [2026083104, 2026083108]
    assert len({row["seed"] for row in schedule}) == 2


def test_candidate_timed_function_contains_no_device_sync_or_allocation():
    source = inspect.getsource(_run_candidate)

    assert "torch.cuda.synchronize" not in source
    assert "empty(" not in source
    assert "zeros(" not in source
    assert "Stream(" not in source
    assert "Event(" not in source


def test_argument_parser_requires_source_and_distributed_identity():
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--attempt",
            "attempt-r1",
            "--source-revision",
            "a" * 40,
            "--source-tree-sha256",
            "b" * 64,
            "--output-dir",
            "/approved/output",
            "--rank",
            "0",
            "--world-size",
            "4",
            "--dist-port",
            "29601",
        ]
    )
    assert args.attempt == "attempt-r1"
    assert args.world_size == 4


def test_worker_declares_exact_compact_artifacts():
    assert ARTIFACT_NAMES == (
        "microgate_rows.jsonl",
        "memory_summary.json",
        "cleanup.json",
        "runtime_capabilities.json",
    )


def _valid_measurement_row():
    return {
        "attempt": "attempt-r1",
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "active_tokens": 4,
        "pair_index": 17,
        "rank": 2,
        "arm_order": ["candidate", "baseline"],
        "cohort_digest": "c" * 64,
        "collective_order_digest": "d" * 64,
        "baseline_cuda_ns": 100_000,
        "candidate_cuda_ns": 90_000,
        "baseline_host_submission_ns": 20_000,
        "candidate_host_submission_ns": 21_000,
        "candidate_communication_union_ns": 40_000,
        "candidate_realized_overlap_ns": 12_000,
        "cross_rank_max_abs_error": 0.0,
        "cross_rank_max_rel_error": 0.0,
        "baseline_max_abs_error": 0.0,
        "baseline_max_rel_error": 0.0,
        "nan_count": 0,
        "inf_count": 0,
        "timed_out": False,
    }


def test_measurement_row_requires_overlap_and_order_evidence():
    row = _valid_measurement_row()
    assert validate_measurement_row(row) == row
    for field in (
        "candidate_communication_union_ns",
        "candidate_realized_overlap_ns",
        "cohort_digest",
        "collective_order_digest",
    ):
        broken = dict(row)
        broken.pop(field)
        with pytest.raises(ValueError, match=field):
            validate_measurement_row(broken)


@pytest.mark.parametrize(
    "field",
    (
        "attempt",
        "source_revision",
        "source_tree_sha256",
        "baseline_cuda_ns",
        "candidate_cuda_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "cross_rank_max_abs_error",
        "baseline_max_abs_error",
        "nan_count",
        "inf_count",
        "timed_out",
    ),
)
def test_measurement_row_requires_identity_timing_and_correctness(field):
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
