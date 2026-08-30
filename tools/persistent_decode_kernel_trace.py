#!/usr/bin/env python3
"""Dependency-light parsing for persistent-decode Nsight traces."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
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
KERNEL_ROLES = (
    "MATMUL",
    "ATTENTION",
    "NORMALIZATION",
    "ELEMENTWISE",
    "REDUCTION",
    "INDEX_OR_STATE_UPDATE",
    "TOKEN_SELECTION",
    "COPY_OR_FILL",
    "RUNTIME_OR_GRAPH",
    "UNKNOWN",
)
CANDIDATE_ROLES = frozenset({
    "NORMALIZATION",
    "ELEMENTWISE",
    "REDUCTION",
    "INDEX_OR_STATE_UPDATE",
    "TOKEN_SELECTION",
})
_ROLE_PATTERNS = (
    (
        "TOKEN_SELECTION",
        (
            "argmax",
            "topk",
            "sampling",
            "sample_token",
        ),
    ),
    (
        "ATTENTION",
        (
            "flash",
            "attention",
            "fmha",
            "paged_attn",
        ),
    ),
    (
        "MATMUL",
        (
            "gemm",
            "matmul",
            "cublas",
            "cutlass",
            "sgemm",
            "bgemm",
        ),
    ),
    (
        "NORMALIZATION",
        (
            "rms_norm",
            "rmsnorm",
            "layer_norm",
            "layernorm",
            "norm_kernel",
        ),
    ),
    (
        "COPY_OR_FILL",
        (
            "memcpy",
            "memset",
            "vectorized_copy",
            "vectorized_mem",
            "fill_kernel",
        ),
    ),
    (
        "INDEX_OR_STATE_UPDATE",
        (
            "index_put",
            "index_select",
            "scatter",
            "slot_mapping",
            "cache_store",
            "state_update",
        ),
    ),
    (
        "ELEMENTWISE",
        (
            "silu",
            "gelu",
            "elementwise",
            "pointwise",
            "add_kernel",
            "mul_kernel",
        ),
    ),
    (
        "REDUCTION",
        (
            "reduce",
            "softmax",
        ),
    ),
    (
        "RUNTIME_OR_GRAPH",
        (
            "cudagraph",
            "cuda_graph",
            "graphlaunch",
            "barrier",
        ),
    ),
)


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


def classify_kernel(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("kernel name must be a string")
    normalized = name.strip().lower().replace(" ", "_")
    for role, patterns in _ROLE_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return role
    return "UNKNOWN"


def classify_kernel_rows(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("kernel rows must be a list")
    classified = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("kernel row must be an object")
        start_ns = _require_integer(
            row.get("start_ns"),
            "kernel start",
            minimum=0,
        )
        end_ns = _require_integer(
            row.get("end_ns"),
            "kernel end",
            minimum=0,
        )
        if end_ns <= start_ns:
            raise ValueError("kernel interval must be positive")
        duration_ns = row.get("duration_ns", end_ns - start_ns)
        _require_integer(
            duration_ns,
            "kernel duration",
            minimum=1,
        )
        if duration_ns != end_ns - start_ns:
            raise ValueError("kernel duration does not match interval")
        stream_id = _require_integer(
            row.get("stream_id"),
            "kernel stream",
            minimum=0,
        )
        name = row.get("name")
        if not isinstance(name, str):
            raise ValueError("kernel name must be a string")
        identity = {}
        for field in TRACE_FIELDS:
            if field not in row:
                raise ValueError(
                    f"kernel row missing identity field {field}"
                )
            identity[field] = row[field]
        classified.append({
            **row,
            **identity,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": duration_ns,
            "stream_id": stream_id,
            "role": classify_kernel(name),
        })
    return classified


def _normalized_kernel_name(name: str) -> str:
    normalized = " ".join(name.strip().lower().split())
    normalized = re.sub(r"0x[0-9a-f]+", "0x#", normalized)
    return normalized


def _segment_from_rows(
    rows: list[dict],
    *,
    ordinal: int,
) -> dict:
    first = rows[0]
    last = rows[-1]
    duration_ns = sum(row["duration_ns"] for row in rows)
    wall_union_ns = last["end_ns"] - first["start_ns"]
    role_histogram = {}
    signature_rows = []
    for row in rows:
        role = row["role"]
        role_histogram[role] = role_histogram.get(role, 0) + 1
        signature_rows.append((
            role,
            _normalized_kernel_name(row["name"]),
        ))
    signature = hashlib.sha256(
        json.dumps(
            signature_rows,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **{
            field: first[field] for field in TRACE_FIELDS
        },
        "segment_id": ordinal,
        "stream_id": first["stream_id"],
        "first_kernel_start_ns": first["start_ns"],
        "last_kernel_end_ns": last["end_ns"],
        "kernel_count": len(rows),
        "kernel_duration_sum_ns": duration_ns,
        "internal_gap_sum_ns": wall_union_ns - duration_ns,
        "wall_union_ns": wall_union_ns,
        "role_histogram": dict(sorted(role_histogram.items())),
        "normalized_kernel_signature_sha256": signature,
    }


def build_candidate_segments(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("classified kernel rows must be a list")
    ordered = sorted(
        rows,
        key=lambda row: (
            tuple(row[field] for field in TRACE_FIELDS),
            row["start_ns"],
            row["end_ns"],
            row["stream_id"],
        ),
    )
    previous_by_stream = {}
    active_by_stream = {}
    segments = []
    segment_ordinal = 0
    active_identity = None

    def flush(stream_id):
        nonlocal segment_ordinal
        active = active_by_stream.pop(stream_id, None)
        if active:
            segments.append(
                _segment_from_rows(
                    active,
                    ordinal=segment_ordinal,
                )
            )
            segment_ordinal += 1

    for row in ordered:
        identity = tuple(row[field] for field in TRACE_FIELDS)
        if active_identity is None:
            active_identity = identity
        elif identity != active_identity:
            for stream in tuple(active_by_stream):
                flush(stream)
            previous_by_stream.clear()
            active_identity = identity
        stream_id = row["stream_id"]
        previous = previous_by_stream.get(stream_id)
        if previous is not None and row["start_ns"] < previous["end_ns"]:
            raise ValueError("kernel intervals overlap on one stream")
        previous_by_stream[stream_id] = row
        if row.get("role") not in KERNEL_ROLES:
            raise ValueError("kernel role is invalid")
        if row["role"] not in CANDIDATE_ROLES:
            flush(stream_id)
            continue
        active_by_stream.setdefault(stream_id, []).append(row)
    for stream in tuple(active_by_stream):
        flush(stream)
    return sorted(
        segments,
        key=lambda row: (
            tuple(row[field] for field in TRACE_FIELDS),
            row["first_kernel_start_ns"],
            row["stream_id"],
        ),
    )


def summarize_trace_coverage(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("classified kernel rows must be non-empty")
    duration_ns = 0
    classified_duration_ns = 0
    classified_launches = 0
    histogram = {}
    for row in rows:
        role = row.get("role")
        if role not in KERNEL_ROLES:
            raise ValueError("kernel role is invalid")
        row_duration = _require_integer(
            row.get("duration_ns"),
            "kernel duration",
            minimum=1,
        )
        duration_ns += row_duration
        histogram[role] = histogram.get(role, 0) + 1
        if role != "UNKNOWN":
            classified_launches += 1
            classified_duration_ns += row_duration
    launch_count = len(rows)
    return {
        "kernel_launch_count": launch_count,
        "classified_kernel_launch_count": classified_launches,
        "classified_launch_ratio": classified_launches / launch_count,
        "kernel_duration_ns": duration_ns,
        "classified_kernel_duration_ns": classified_duration_ns,
        "classified_duration_ratio": (
            classified_duration_ns / duration_ns
        ),
        "role_histogram": dict(sorted(histogram.items())),
    }
