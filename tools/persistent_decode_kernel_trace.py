#!/usr/bin/env python3
"""Dependency-light parsing for persistent-decode Nsight traces."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3


TRACE_PREFIX = "persistent_decode_trace"
TRACE_FIELDS = (
    "attempt",
    "workload",
    "repetition",
    "context",
    "burst",
    "logical_tokens",
)
INTEGER_FIELDS = {
    "repetition": 0,
    "context": 1,
    "burst": 0,
    "logical_tokens": 1,
}
REQUIRED_TABLES = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {
        "start",
        "end",
        "text",
        "textId",
        "globalTid",
    },
    "CUPTI_ACTIVITY_KIND_KERNEL": {
        "start",
        "end",
        "streamId",
        "globalPid",
        "demangledName",
        "shortName",
    },
}


@dataclass(frozen=True)
class KernelInterval:
    start_ns: int
    end_ns: int
    stream_id: int
    name: str
    global_pid: int | None = None


@dataclass(frozen=True)
class TraceRange:
    identity: dict
    start_ns: int
    end_ns: int
    global_tid: int


def _require_integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def parse_trace_label(text: str):
    if not isinstance(text, str):
        raise ValueError("trace label must be a string")
    parts = text.split("/")
    if not parts or parts[0] != TRACE_PREFIX:
        return None
    if len(parts) != len(TRACE_FIELDS) + 1:
        raise ValueError("persistent decode trace identity is incomplete")
    identity = {}
    for expected, component in zip(TRACE_FIELDS, parts[1:]):
        key, separator, value = component.partition("=")
        if not separator or key != expected or not value:
            raise ValueError("persistent decode trace identity is invalid")
        if key in INTEGER_FIELDS:
            try:
                parsed = int(value)
            except ValueError as error:
                raise ValueError(
                    f"persistent decode trace {key} must be an integer"
                ) from error
            identity[key] = _require_integer(
                parsed,
                f"persistent decode trace {key}",
                minimum=INTEGER_FIELDS[key],
            )
        else:
            identity[key] = value
    return "transaction", identity


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    for table, required_columns in REQUIRED_TABLES.items():
        if table not in tables:
            raise ValueError(
                f"unsupported Nsight SQLite schema: missing table {table}"
            )
        columns = {
            str(row[1])
            for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            )
        }
        missing = required_columns - columns
        if missing:
            raise ValueError(
                "unsupported Nsight SQLite schema: "
                f"{table} missing columns {sorted(missing)}"
            )


def _read_ranges(
    connection: sqlite3.Connection,
    strings: dict[int, str],
) -> list[dict]:
    ranges = []
    identities = set()
    query = (
        "SELECT start, end, text, textId, globalTid "
        "FROM NVTX_EVENTS WHERE end IS NOT NULL"
    )
    for row in connection.execute(query):
        text = row["text"]
        if text is None and row["textId"] is not None:
            text = strings.get(int(row["textId"]))
        if text is None:
            continue
        parsed = parse_trace_label(str(text))
        if parsed is None:
            continue
        _kind, identity = parsed
        start_ns = int(row["start"])
        end_ns = int(row["end"])
        if start_ns < 0 or end_ns <= start_ns:
            raise ValueError("trace range interval is invalid")
        global_tid = row["globalTid"]
        if global_tid is None:
            raise ValueError("trace range global thread is missing")
        identity_key = tuple(identity[field] for field in TRACE_FIELDS)
        if identity_key in identities:
            raise ValueError("duplicate identity in persistent decode trace")
        identities.add(identity_key)
        ranges.append({
            **identity,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "global_tid": int(global_tid),
        })
    ranges.sort(key=lambda item: (item["start_ns"], item["end_ns"]))
    for previous, current in zip(ranges, ranges[1:]):
        if previous["end_ns"] > current["start_ns"]:
            raise ValueError("persistent decode trace ranges overlap")
    if not ranges:
        raise ValueError("persistent decode trace has no transaction ranges")
    return ranges


def _read_kernels(
    connection: sqlite3.Connection,
    strings: dict[int, str],
) -> list[dict]:
    kernels = []
    query = (
        "SELECT start, end, streamId, globalPid, "
        "demangledName, shortName "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    )
    for row in connection.execute(query):
        demangled = row["demangledName"]
        short = row["shortName"]
        name = (
            strings.get(int(demangled))
            if demangled is not None
            else None
        )
        if name is None and short is not None:
            name = strings.get(int(short))
        kernels.append({
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "stream_id": int(row["streamId"]),
            "global_pid": (
                None
                if row["globalPid"] is None
                else int(row["globalPid"])
            ),
            "name": "" if name is None else str(name),
        })
    kernels.sort(
        key=lambda item: (
            item["start_ns"],
            item["end_ns"],
            item["stream_id"],
            item["name"],
        )
    )
    return kernels


def assign_kernels_to_ranges(
    ranges: list[dict],
    kernels: list[dict],
) -> list[dict]:
    assigned = []
    for kernel in kernels:
        start_ns = _require_integer(
            kernel.get("start_ns"),
            "kernel start",
            minimum=0,
        )
        end_ns = _require_integer(
            kernel.get("end_ns"),
            "kernel end",
            minimum=0,
        )
        if end_ns <= start_ns:
            raise ValueError("kernel interval must be positive")
        stream_id = _require_integer(
            kernel.get("stream_id"),
            "kernel stream",
            minimum=0,
        )
        overlaps = [
            trace_range
            for trace_range in ranges
            if (
                start_ns < trace_range["end_ns"]
                and end_ns > trace_range["start_ns"]
            )
        ]
        if not overlaps:
            continue
        if len(overlaps) != 1:
            raise ValueError("kernel overlaps multiple transaction ranges")
        trace_range = overlaps[0]
        if (
            start_ns < trace_range["start_ns"]
            or end_ns > trace_range["end_ns"]
        ):
            raise ValueError("kernel crosses transaction boundary")
        name = kernel.get("name")
        if not isinstance(name, str):
            raise ValueError("kernel name must be a string")
        assigned.append({
            field: trace_range[field] for field in TRACE_FIELDS
        } | {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "stream_id": stream_id,
            "global_pid": kernel.get("global_pid"),
            "name": name,
        })
    return assigned


def read_decode_trace(path: Path) -> dict:
    trace_path = Path(path)
    if not trace_path.is_file():
        raise ValueError("Nsight SQLite path must be an existing file")
    with closing(sqlite3.connect(trace_path)) as connection:
        connection.row_factory = sqlite3.Row
        _validate_schema(connection)
        strings = {
            int(row["id"]): str(row["value"])
            for row in connection.execute(
                "SELECT id, value FROM StringIds"
            )
        }
        ranges = _read_ranges(connection, strings)
        kernels = _read_kernels(connection, strings)
    return {
        "classification": "COMPLETE",
        "ranges": ranges,
        "kernel_rows": assign_kernels_to_ranges(ranges, kernels),
    }
