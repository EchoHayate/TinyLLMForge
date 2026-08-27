from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen38_nsys_intervals.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen38_nsys_intervals_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


intervals = _load()
Interval = intervals.Interval
subtract_intervals = intervals.subtract_intervals
union_duration = intervals.union_duration


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        (((0, 10), (20, 30)), 20),
        (((0, 10), (10, 20)), 20),
        (((0, 30), (5, 10)), 30),
        (((0, 10), (5, 15)), 15),
        (((0, 10), (0, 10)), 10),
        (((0, 0), (5, 5)), 0),
    ),
)
def test_interval_union_never_double_counts(values, expected):
    assert union_duration(
        Interval(start, end) for start, end in values
    ) == expected


@pytest.mark.parametrize(
    ("base", "covered", "expected"),
    (
        (((0, 30),), ((5, 10), (20, 25)), ((0, 5), (10, 20), (25, 30))),
        (((0, 10), (20, 30)), ((5, 25),), ((0, 5), (25, 30))),
        (((0, 30),), ((0, 35),), ()),
        (((0, 30),), ((10, 10),), ((0, 30),)),
        ((), ((0, 10),), ()),
    ),
)
def test_interval_subtraction_returns_disjoint_uncovered_segments(
    base,
    covered,
    expected,
):
    result = subtract_intervals(
        tuple(Interval(start, end) for start, end in base),
        tuple(Interval(start, end) for start, end in covered),
    )

    assert result == tuple(
        Interval(start, end) for start, end in expected
    )


@pytest.mark.parametrize(
    ("start_ns", "end_ns"),
    ((-1, 0), (2, 1), (True, 1), (0, False), (1.0, 2)),
)
def test_interval_rejects_negative_or_non_monotonic_bounds(
    start_ns,
    end_ns,
):
    with pytest.raises((TypeError, ValueError)):
        Interval(start_ns, end_ns)


def _identity(rank):
    return {
        "attempt": "attempt-a",
        "workload": "Q1",
        "repetition": 2,
        "request_set_sha256": "a" * 64,
        "decode_ordinal": 3,
        "rank": rank,
        "layer_index": 0,
        "layer_role": "full_attention",
    }


def _structured_rows(ranks=(0, 1, 2, 3)):
    rows = []
    for rank in ranks:
        identity = _identity(rank)
        rows.extend((
            identity | {
                "operation_ordinal": 0,
                "operation_class": "gemm",
                "operation_name": "qkv_projection",
                "tensor_shape": [4, 5120],
                "tensor_dtype": "torch.bfloat16",
                "source_stream": f"cuda:{rank}:stream:7",
                "completion_stream": f"cuda:{rank}:stream:7",
            },
            identity | {
                "operation_ordinal": 1,
                "operation_class": "collective",
                "operation_name": "row_parallel_all_reduce",
                "collective_kind": "all_reduce",
                "collective_bytes": 4096,
                "tensor_shape": [4, 5120],
                "tensor_dtype": "torch.bfloat16",
                "source_stream": f"cuda:{rank}:stream:11",
                "completion_stream": f"cuda:{rank}:stream:11",
            },
            identity | {
                "operation_ordinal": 2,
                "operation_class": "attention",
                "operation_name": "flash_attention",
                "tensor_shape": [4, 5120],
                "tensor_dtype": "torch.bfloat16",
                "source_stream": f"cuda:{rank}:stream:7",
                "completion_stream": f"cuda:{rank}:stream:7",
            },
        ))
    return rows


def _profile_prefix(rank, *, include_rank=True):
    prefix = (
        "decode_internal/attempt=attempt-a/workload=Q1/"
        "repetition=2"
    )
    if include_rank:
        return f"{prefix}/rank={rank}"
    return prefix


def _create_trace(
    path,
    *,
    ranks=(0, 1, 2, 3),
    missing_nccl_rank=None,
    duplicate_operation=False,
    range_mismatch=False,
    cross_step_kernel=False,
    omit_runtime_table=False,
    unrelated_null_kernel=False,
    correlated_short_name_fallback=False,
    labels_include_rank=True,
):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE StringIds ("
        "id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE NVTX_EVENTS ("
        "start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL, "
        "text TEXT, globalTid INTEGER, textId INTEGER)"
    )
    if not omit_runtime_table:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
            "start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "globalTid INTEGER NOT NULL, correlationId INTEGER NOT NULL, "
            "nameId INTEGER)"
        )
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER NOT NULL, end INTEGER NOT NULL, "
        "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, "
        "streamId INTEGER NOT NULL, correlationId INTEGER, "
        "globalPid INTEGER, demangledName INTEGER, "
        "shortName INTEGER NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        (
            (1, "cutlass_gemm"),
            (2, "ncclKernel_AllReduce_RING_LL"),
            (3, "flash_attention"),
            (4, "cudaLaunchKernel"),
        ),
    )
    for rank in ranks:
        base = rank * 1_000
        global_pid = (100 + rank) << 24
        global_tid = global_pid | 7
        prefix = _profile_prefix(
            rank,
            include_rank=labels_include_rank,
        )
        nvtx_rows = [
            (base + 100, base + 500, 59, f"{prefix}/decode_steady",
             global_tid, None),
            (base + 110, base + 490, 59,
             f"{prefix}/layer/0/full_attention", global_tid, None),
            (base + 120, base + 130, 59,
             f"{prefix}/operation/0/gemm/qkv_projection",
             global_tid, None),
            (
                base + (50 if range_mismatch and rank == 0 else 210),
                base + 220,
                59,
                f"{prefix}/operation/1/collective/"
                "row_parallel_all_reduce",
                global_tid,
                None,
            ),
            (base + 230, base + 240, 59,
             f"{prefix}/operation/2/attention/flash_attention",
             global_tid, None),
        ]
        if duplicate_operation and rank == 0:
            nvtx_rows.append(nvtx_rows[2])
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
            nvtx_rows,
        )
        runtime_rows = [
            (base + 122, base + 128, global_tid, rank * 10 + 1, 4),
            (base + 232, base + 238, global_tid, rank * 10 + 3, 4),
        ]
        if missing_nccl_rank != rank:
            runtime_rows.append((
                base + 212,
                base + 218,
                global_tid,
                rank * 10 + 2,
                4,
            ))
        if not omit_runtime_table:
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                "VALUES (?, ?, ?, ?, ?)",
                runtime_rows,
            )
        attention_end = base + 350 + rank * 10
        if cross_step_kernel and rank == 0:
            attention_end = base + 550
        kernel_rows = [
            (base + 140, base + 200, rank, 1, 7, rank * 10 + 1,
             global_pid, 1, 1),
            (base + 280, attention_end, rank, 1, 7, rank * 10 + 3,
             global_pid, 3, 3),
        ]
        if missing_nccl_rank != rank:
            kernel_rows.append((
                base + 220,
                base + 300,
                rank,
                1,
                11,
                rank * 10 + 2,
                global_pid,
                (
                    None
                    if correlated_short_name_fallback and rank == 0
                    else 2
                ),
                2,
            ))
        else:
            kernel_rows.append((
                base + 225,
                base + 295,
                rank,
                1,
                11,
                999,
                global_pid,
                2,
                2,
            ))
        if unrelated_null_kernel and rank == 0:
            kernel_rows.append((
                1,
                2,
                rank,
                1,
                99,
                None,
                None,
                1,
                1,
            ))
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            kernel_rows,
        )
    connection.commit()
    connection.close()


def test_parse_nsys_sqlite_correlates_four_ranks_and_exact_unions(tmp_path):
    path = tmp_path / "trace.sqlite"
    _create_trace(path)

    result = intervals.parse_nsys_sqlite(path, _structured_rows())

    assert result["classification"] == "COMPLETE"
    assert result["coverage_errors"] == []
    assert len(result["rows"]) == 4
    for row in result["rows"]:
        rank = row["rank"]
        assert row | {
            "cpu_global_tids": [],
            "stream_ids": [],
        } == {
            **_identity(rank),
            "step_critical_interval_ns": 210 + rank * 10,
            "gemm_ns": 60,
            "collective_ns": 80,
            "compute_ns": 130 + rank * 10,
            "exposed_collective_ns": 60,
            "compute_collective_overlap_ns": 20,
            "gpu_idle_ns": 20,
            "collective_count": 1,
            "collective_bytes": 4096,
            "critical_path_ns": 190 + rank * 10,
            "cpu_global_tids": [],
            "stream_ids": [],
        }
        assert row["cpu_global_tids"] == [((100 + rank) << 24) | 7]
        assert row["stream_ids"] == [7, 11]
    assert result["critical_rows"] == [{
        "attempt": "attempt-a",
        "workload": "Q1",
        "repetition": 2,
        "request_set_sha256": "a" * 64,
        "decode_ordinal": 3,
        "critical_rank": 3,
        "step_critical_interval_ns": 240,
        "final_required_offset_ns": 280,
    }]
    assert result["step_rows"] == [
        {
            **{
                name: value
                for name, value in _identity(rank).items()
                if name not in {"layer_index", "layer_role"}
            },
            "step_critical_interval_ns": 210 + rank * 10,
            "final_required_offset_ns": 250 + rank * 10,
        }
        for rank in range(4)
    ]


def test_parse_nsys_sqlite_closes_connection(tmp_path, monkeypatch):
    path = tmp_path / "trace.sqlite"
    _create_trace(path)
    closed = []
    original_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            closed.append(True)
            return super().close()

    def tracking_connect(*args, **kwargs):
        return original_connect(
            *args,
            factory=TrackingConnection,
            **kwargs,
        )

    monkeypatch.setattr(intervals.sqlite3, "connect", tracking_connect)

    result = intervals.parse_nsys_sqlite(path, _structured_rows())

    assert result["classification"] == "COMPLETE"
    assert closed == [True]


def test_parse_nsys_sqlite_infers_rank_from_correlated_cuda_device(
    tmp_path,
):
    path = tmp_path / "trace.sqlite"
    _create_trace(path, labels_include_rank=False)

    result = intervals.parse_nsys_sqlite(path, _structured_rows())

    assert result["classification"] == "COMPLETE"
    assert {row["rank"] for row in result["rows"]} == {0, 1, 2, 3}


def test_missing_nccl_correlation_is_inconclusive_not_estimated(tmp_path):
    path = tmp_path / "missing-nccl.sqlite"
    _create_trace(path, missing_nccl_rank=2)

    result = intervals.parse_nsys_sqlite(path, _structured_rows())

    assert result["classification"] == "INCONCLUSIVE_TRACE_COVERAGE"
    assert result["rows"] == []
    assert result["critical_rows"] == []
    assert result["step_rows"] == []
    assert result["coverage_errors"] == [
        "rank 2 operation 1 has no correlated NCCL kernel"
    ]


def test_missing_collective_bytes_fails_closed(tmp_path):
    path = tmp_path / "missing-collective-bytes.sqlite"
    _create_trace(path)
    rows = _structured_rows()
    for row in rows:
        if row["operation_class"] == "collective":
            row.pop("collective_bytes")

    with pytest.raises(ValueError, match="collective_bytes"):
        intervals.parse_nsys_sqlite(path, rows)


def test_unrelated_null_kernel_identity_is_not_loaded(tmp_path):
    path = tmp_path / "unrelated-null.sqlite"
    _create_trace(path, unrelated_null_kernel=True)

    result = intervals.parse_nsys_sqlite(path, _structured_rows())

    assert result["classification"] == "COMPLETE"


def test_correlated_kernel_falls_back_to_short_name(tmp_path):
    path = tmp_path / "short-name-fallback.sqlite"
    _create_trace(path, correlated_short_name_fallback=True)

    result = intervals.parse_nsys_sqlite(path, _structured_rows())

    assert result["classification"] == "COMPLETE"


def test_missing_rank_fails_closed(tmp_path):
    path = tmp_path / "missing-rank.sqlite"
    _create_trace(path, ranks=(0, 1, 2))

    with pytest.raises(ValueError, match="exactly ranks 0, 1, 2, and 3"):
        intervals.parse_nsys_sqlite(
            path,
            _structured_rows(ranks=(0, 1, 2)),
        )


def test_rank_identity_drift_fails_aligned_step_validation(tmp_path):
    path = tmp_path / "rank-drift.sqlite"
    _create_trace(path)
    rows = _structured_rows()
    for row in rows:
        if row["rank"] == 3:
            row["request_set_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="aligned step must contain"):
        intervals.parse_nsys_sqlite(path, rows)


def test_duplicate_operation_mapping_fails_closed(tmp_path):
    path = tmp_path / "duplicate.sqlite"
    _create_trace(path, duplicate_operation=True)

    with pytest.raises(ValueError, match="duplicate NVTX operation"):
        intervals.parse_nsys_sqlite(path, _structured_rows())


def test_operation_range_mismatch_fails_closed(tmp_path):
    path = tmp_path / "range-mismatch.sqlite"
    _create_trace(path, range_mismatch=True)

    with pytest.raises(ValueError, match="outside its step or layer"):
        intervals.parse_nsys_sqlite(path, _structured_rows())


def test_cross_step_kernel_fails_closed(tmp_path):
    path = tmp_path / "cross-step.sqlite"
    _create_trace(path, cross_step_kernel=True)

    with pytest.raises(ValueError, match="crosses its step boundary"):
        intervals.parse_nsys_sqlite(path, _structured_rows())


def test_unsupported_nsys_schema_fails_closed(tmp_path):
    path = tmp_path / "unsupported.sqlite"
    _create_trace(path, omit_runtime_table=True)

    with pytest.raises(ValueError, match="unsupported Nsight SQLite schema"):
        intervals.parse_nsys_sqlite(path, _structured_rows())
