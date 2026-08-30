#!/usr/bin/env python3
"""Tests for dependency-light persistent-decode Nsight parsing."""

from __future__ import annotations

import sqlite3

import pytest

from tools.persistent_decode_kernel_trace import (
    assign_kernels_to_ranges,
    build_candidate_segments,
    classify_kernel,
    classify_kernel_rows,
    parse_trace_label,
    read_decode_trace,
    summarize_trace_coverage,
)


def _label(
    *,
    attempt: str = "attempt-a",
    workload: str = "exact",
    repetition: int = 0,
    context: int = 256,
    burst: int = 0,
    logical_tokens: int = 8,
) -> str:
    return (
        "persistent_decode_trace/"
        f"attempt={attempt}/"
        f"workload={workload}/"
        f"repetition={repetition}/"
        f"context={context}/"
        f"burst={burst}/"
        f"logical_tokens={logical_tokens}"
    )


def _create_trace(
    path,
    *,
    ranges=None,
    kernels=None,
    graph_traces=None,
    omit_table=None,
):
    ranges = ranges or [(_label(), 100, 300, 0x1000007)]
    kernels = kernels or [
        (120, 160, 7, 0x1000000, "void rms_norm_kernel"),
        (170, 260, 7, 0x1000000, "ampere_sgemm_128x64"),
    ]
    graph_traces = graph_traces or []
    connection = sqlite3.connect(path)
    if omit_table != "StringIds":
        connection.execute(
            "CREATE TABLE StringIds ("
            "id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )
    if omit_table != "NVTX_EVENTS":
        connection.execute(
            "CREATE TABLE NVTX_EVENTS ("
            "start INTEGER NOT NULL, end INTEGER, "
            "eventType INTEGER NOT NULL, text TEXT, "
            "globalTid INTEGER, textId INTEGER)"
        )
    if omit_table != "CUPTI_ACTIVITY_KIND_KERNEL":
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
            "start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, "
            "streamId INTEGER NOT NULL, correlationId INTEGER, "
            "globalPid INTEGER, demangledName INTEGER, "
            "shortName INTEGER)"
        )
    if omit_table != "CUPTI_ACTIVITY_KIND_GRAPH_TRACE":
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE ("
            "start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, "
            "greenContextId INTEGER, streamId INTEGER NOT NULL, "
            "correlationId INTEGER, globalPid INTEGER, "
            "graphId INTEGER NOT NULL, graphExecId INTEGER NOT NULL)"
        )
    if omit_table is None:
        names = sorted({row[4] for row in kernels})
        name_ids = {
            name: index for index, name in enumerate(names, start=1)
        }
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [(name_id, name) for name, name_id in name_ids.items()],
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
            [
                (start, end, 59, label, global_tid, None)
                for label, start, end, global_tid in ranges
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    start,
                    end,
                    0,
                    1,
                    stream_id,
                    index,
                    global_pid,
                    name_ids[name],
                    name_ids[name],
                )
                for index, (
                    start,
                    end,
                    stream_id,
                    global_pid,
                    name,
                ) in enumerate(kernels, start=1)
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    start,
                    end,
                    0,
                    1,
                    None,
                    stream_id,
                    index,
                    global_pid,
                    graph_id,
                    graph_exec_id,
                )
                for index, (
                    start,
                    end,
                    stream_id,
                    global_pid,
                    graph_id,
                    graph_exec_id,
                ) in enumerate(graph_traces, start=1)
            ],
        )
    connection.commit()
    connection.close()
    return path


def test_parse_trace_label_accepts_complete_identity():
    parsed = parse_trace_label(_label())

    assert parsed == (
        "transaction",
        {
            "attempt": "attempt-a",
            "workload": "exact",
            "repetition": 0,
            "context": 256,
            "burst": 0,
            "logical_tokens": 8,
        },
    )


@pytest.mark.parametrize(
    "label",
    [
        "unrelated/range",
        "persistent_decode_trace/attempt=a/workload=w",
        (
            "persistent_decode_trace/attempt=a/workload=w/"
            "repetition=-1/context=256/burst=0/logical_tokens=8"
        ),
        (
            "persistent_decode_trace/attempt=a/workload=w/"
            "repetition=0/context=0/burst=0/logical_tokens=8"
        ),
        (
            "persistent_decode_trace/attempt=a/workload=w/"
            "repetition=0/context=256/burst=0/logical_tokens=0"
        ),
    ],
)
def test_parse_trace_label_ignores_unrelated_and_rejects_bad_identity(label):
    if label == "unrelated/range":
        assert parse_trace_label(label) is None
    else:
        with pytest.raises(ValueError):
            parse_trace_label(label)


def test_read_decode_trace_maps_kernels_to_non_overlapping_ranges(tmp_path):
    trace = _create_trace(tmp_path / "trace.sqlite")

    parsed = read_decode_trace(trace)

    assert parsed["classification"] == "COMPLETE"
    assert [row["name"] for row in parsed["kernel_rows"]] == [
        "void rms_norm_kernel",
        "ampere_sgemm_128x64",
    ]
    assert {
        row["logical_tokens"]
        for row in parsed["kernel_rows"]
    } == {8}
    assert parsed["ranges"] == [{
        "attempt": "attempt-a",
        "workload": "exact",
        "repetition": 0,
        "context": 256,
        "burst": 0,
        "logical_tokens": 8,
        "start_ns": 100,
        "end_ns": 300,
        "global_tid": 0x1000007,
    }]


def test_graph_execution_is_a_barrier_between_candidate_segments(tmp_path):
    trace = _create_trace(
        tmp_path / "graph-barrier.sqlite",
        kernels=[
            (120, 130, 7, 0x1000000, "void rms_norm_kernel"),
            (170, 180, 7, 0x1000000, "void silu_and_mul_kernel"),
        ],
        graph_traces=[
            (140, 160, 7, 0x1000000, 13, 14),
        ],
    )

    parsed = read_decode_trace(trace)
    classified = classify_kernel_rows(parsed["kernel_rows"])
    segments = build_candidate_segments(classified)

    assert [
        (row["name"], row["role"])
        for row in classified
    ] == [
        ("void rms_norm_kernel", "NORMALIZATION"),
        ("cuda_graph_execution", "RUNTIME_OR_GRAPH"),
        ("void silu_and_mul_kernel", "ELEMENTWISE"),
    ]
    assert [row["kernel_count"] for row in segments] == [1, 1]
    assert [row["wall_union_ns"] for row in segments] == [10, 10]
    assert summarize_trace_coverage(classified) == {
        "kernel_launch_count": 3,
        "classified_kernel_launch_count": 3,
        "classified_launch_ratio": 1.0,
        "kernel_duration_ns": 40,
        "classified_kernel_duration_ns": 40,
        "classified_duration_ratio": 1.0,
        "role_histogram": {
            "ELEMENTWISE": 1,
            "NORMALIZATION": 1,
            "RUNTIME_OR_GRAPH": 1,
        },
    }


def test_trace_rejects_overlapping_transaction_ranges(tmp_path):
    trace = _create_trace(
        tmp_path / "overlap.sqlite",
        ranges=[
            (_label(burst=0), 100, 300, 0x1000007),
            (_label(burst=1), 250, 400, 0x1000007),
        ],
    )

    with pytest.raises(ValueError, match="ranges overlap"):
        read_decode_trace(trace)


def test_trace_rejects_kernel_crossing_transaction_boundary(tmp_path):
    trace = _create_trace(
        tmp_path / "crossing.sqlite",
        kernels=[
            (90, 120, 7, 0x1000000, "void rms_norm_kernel"),
        ],
    )

    with pytest.raises(ValueError, match="crosses transaction"):
        read_decode_trace(trace)


def test_trace_rejects_duplicate_identity(tmp_path):
    trace = _create_trace(
        tmp_path / "duplicate.sqlite",
        ranges=[
            (_label(), 100, 200, 0x1000007),
            (_label(), 300, 400, 0x1000007),
        ],
        kernels=[
            (120, 160, 7, 0x1000000, "void rms_norm_kernel"),
            (320, 360, 7, 0x1000000, "void rms_norm_kernel"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate identity"):
        read_decode_trace(trace)


def test_trace_ignores_unrelated_nvtx_and_loader_kernels(tmp_path):
    trace = _create_trace(
        tmp_path / "unrelated.sqlite",
        ranges=[
            ("unrelated/model_load", 0, 80, 0x1000007),
            (_label(), 100, 300, 0x1000007),
        ],
        kernels=[
            (10, 50, 7, 0x1000000, "loader_kernel"),
            (120, 160, 7, 0x1000000, "rms_norm_kernel"),
        ],
    )

    parsed = read_decode_trace(trace)

    assert {row["name"] for row in parsed["kernel_rows"]} == {
        "rms_norm_kernel",
    }


@pytest.mark.parametrize(
    "table",
    [
        "StringIds",
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    ],
)
def test_trace_requires_supported_nsys_schema(tmp_path, table):
    trace = _create_trace(
        tmp_path / f"missing-{table}.sqlite",
        omit_table=table,
    )

    with pytest.raises(ValueError, match=f"missing table {table}"):
        read_decode_trace(trace)


def test_assign_kernels_rejects_non_positive_intervals():
    ranges = [{
        "attempt": "a",
        "workload": "w",
        "repetition": 0,
        "context": 256,
        "burst": 0,
        "logical_tokens": 8,
        "start_ns": 100,
        "end_ns": 300,
        "global_tid": 0x1000007,
    }]

    with pytest.raises(ValueError, match="kernel interval"):
        assign_kernels_to_ranges(
            ranges,
            [{
                "start_ns": 150,
                "end_ns": 150,
                "stream_id": 7,
                "global_pid": 0x1000000,
                "name": "kernel",
            }],
        )


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("ampere_bf16_s16816gemm", "MATMUL"),
        ("flash_fwd_splitkv_kernel", "ATTENTION"),
        ("rms_norm_kernel", "NORMALIZATION"),
        ("silu_and_mul_kernel", "ELEMENTWISE"),
        ("reduce_kernel", "REDUCTION"),
        ("index_put_kernel", "INDEX_OR_STATE_UPDATE"),
        ("argmax_reduce_kernel", "TOKEN_SELECTION"),
        ("vectorized_memcpy", "COPY_OR_FILL"),
        ("cudaGraphLaunch", "RUNTIME_OR_GRAPH"),
        ("unrecognized_vendor_kernel", "UNKNOWN"),
    ],
)
def test_classify_kernel_uses_generic_roles(name, role):
    assert classify_kernel(name) == role


def _kernel_rows(*roles, timestamp_shift=0, streams=None):
    streams = streams or [7] * len(roles)
    names = {
        "MATMUL": "ampere_bf16_gemm",
        "ATTENTION": "flash_attention_kernel",
        "NORMALIZATION": "rms_norm_kernel",
        "ELEMENTWISE": "silu_and_mul_kernel",
        "REDUCTION": "reduce_kernel",
        "INDEX_OR_STATE_UPDATE": "index_put_kernel",
        "TOKEN_SELECTION": "argmax_kernel",
        "COPY_OR_FILL": "vectorized_memcpy",
        "RUNTIME_OR_GRAPH": "cudaGraphLaunch",
        "UNKNOWN": "mystery_kernel",
    }
    rows = []
    cursor = 100 + timestamp_shift
    for role, stream_id in zip(roles, streams):
        rows.append({
            "attempt": "attempt-a",
            "workload": "exact",
            "repetition": 0,
            "context": 256,
            "burst": 0,
            "logical_tokens": 8,
            "start_ns": cursor,
            "end_ns": cursor + 10,
            "duration_ns": 10,
            "stream_id": stream_id,
            "global_pid": 0x1000000,
            "name": names[role],
        })
        cursor += 15
    return rows


def test_candidate_segment_stops_at_excluded_roles():
    rows = classify_kernel_rows(_kernel_rows(
        "NORMALIZATION",
        "ELEMENTWISE",
        "MATMUL",
        "INDEX_OR_STATE_UPDATE",
        "ATTENTION",
        "TOKEN_SELECTION",
        "UNKNOWN",
    ))

    segments = build_candidate_segments(rows)

    assert [row["kernel_count"] for row in segments] == [2, 1, 1]
    assert [row["kernel_duration_sum_ns"] for row in segments] == [
        20,
        10,
        10,
    ]
    assert [row["internal_gap_sum_ns"] for row in segments] == [
        5,
        0,
        0,
    ]


def test_candidate_segment_does_not_cross_streams():
    rows = classify_kernel_rows(_kernel_rows(
        "NORMALIZATION",
        "ELEMENTWISE",
        streams=[7, 9],
    ))

    segments = build_candidate_segments(rows)

    assert len(segments) == 2
    assert [row["stream_id"] for row in segments] == [7, 9]


def test_candidate_segment_signature_ignores_absolute_timestamps():
    base = build_candidate_segments(classify_kernel_rows(_kernel_rows(
        "NORMALIZATION",
        "ELEMENTWISE",
    )))
    shifted = build_candidate_segments(classify_kernel_rows(_kernel_rows(
        "NORMALIZATION",
        "ELEMENTWISE",
        timestamp_shift=100_000,
    )))

    assert base[0]["normalized_kernel_signature_sha256"] == (
        shifted[0]["normalized_kernel_signature_sha256"]
    )


def test_kernel_rows_reject_overlapping_intervals_on_one_stream():
    rows = _kernel_rows("NORMALIZATION", "ELEMENTWISE")
    rows[1]["start_ns"] = rows[0]["end_ns"] - 1
    rows[1]["duration_ns"] = (
        rows[1]["end_ns"] - rows[1]["start_ns"]
    )

    with pytest.raises(ValueError, match="kernel intervals overlap"):
        build_candidate_segments(classify_kernel_rows(rows))


def test_trace_coverage_counts_unknown_launches_and_duration():
    rows = classify_kernel_rows(_kernel_rows(
        "NORMALIZATION",
        "UNKNOWN",
        "MATMUL",
    ))

    coverage = summarize_trace_coverage(rows)

    assert coverage == {
        "kernel_launch_count": 3,
        "classified_kernel_launch_count": 2,
        "classified_launch_ratio": pytest.approx(2 / 3),
        "kernel_duration_ns": 30,
        "classified_kernel_duration_ns": 20,
        "classified_duration_ratio": pytest.approx(2 / 3),
        "role_histogram": {
            "MATMUL": 1,
            "NORMALIZATION": 1,
            "UNKNOWN": 1,
        },
    }


def test_generic_trace_module_has_no_profile_specific_terms():
    source = (
        __import__(
            "tools.persistent_decode_kernel_trace",
            fromlist=["__file__"],
        )
        .__file__
    )
    text = open(source, encoding="utf-8").read().lower()

    for prohibited in ("qwen", "llama", "k8", "octet", "a100"):
        assert prohibited not in text
