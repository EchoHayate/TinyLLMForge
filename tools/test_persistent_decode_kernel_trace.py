#!/usr/bin/env python3
"""Tests for dependency-light persistent-decode Nsight parsing."""

from __future__ import annotations

import sqlite3

import pytest

from tools.persistent_decode_kernel_trace import (
    assign_kernels_to_ranges,
    parse_trace_label,
    read_decode_trace,
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
    omit_table=None,
):
    ranges = ranges or [(_label(), 100, 300, 0x1000007)]
    kernels = kernels or [
        (120, 160, 7, 0x1000000, "void rms_norm_kernel"),
        (170, 260, 7, 0x1000000, "ampere_sgemm_128x64"),
    ]
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
