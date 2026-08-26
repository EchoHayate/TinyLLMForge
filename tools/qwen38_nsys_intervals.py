from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable


@dataclass(frozen=True, order=True)
class Interval:
    start_ns: int
    end_ns: int

    def __post_init__(self):
        for name, value in (
            ("start_ns", self.start_ns),
            ("end_ns", self.end_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must not precede start_ns")


def _merged_intervals(
    intervals: Iterable[Interval],
) -> tuple[Interval, ...]:
    ordered = sorted(intervals)
    if any(not isinstance(interval, Interval) for interval in ordered):
        raise TypeError("intervals must contain Interval values")
    non_empty = [
        interval
        for interval in ordered
        if interval.end_ns > interval.start_ns
    ]
    if not non_empty:
        return ()
    merged = [non_empty[0]]
    for interval in non_empty[1:]:
        previous = merged[-1]
        if interval.start_ns <= previous.end_ns:
            merged[-1] = Interval(
                previous.start_ns,
                max(previous.end_ns, interval.end_ns),
            )
            continue
        merged.append(interval)
    return tuple(merged)


def union_duration(intervals: Iterable[Interval]) -> int:
    return sum(
        interval.end_ns - interval.start_ns
        for interval in _merged_intervals(intervals)
    )


def subtract_intervals(
    base: Iterable[Interval],
    covered: Iterable[Interval],
) -> tuple[Interval, ...]:
    base_union = _merged_intervals(base)
    covered_union = _merged_intervals(covered)
    result = []
    covered_index = 0
    for interval in base_union:
        cursor = interval.start_ns
        while (
            covered_index < len(covered_union)
            and covered_union[covered_index].end_ns <= cursor
        ):
            covered_index += 1
        index = covered_index
        while (
            index < len(covered_union)
            and covered_union[index].start_ns < interval.end_ns
        ):
            exclusion = covered_union[index]
            if exclusion.start_ns > cursor:
                result.append(Interval(
                    cursor,
                    min(exclusion.start_ns, interval.end_ns),
                ))
            cursor = max(cursor, exclusion.end_ns)
            if cursor >= interval.end_ns:
                break
            index += 1
        if cursor < interval.end_ns:
            result.append(Interval(cursor, interval.end_ns))
    return tuple(result)


def parse_nsys_sqlite(
    path: Path,
    structured_rows: list[dict],
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError("Nsight SQLite path must be an existing file")
    rows_by_key, step_identities = _validate_structured_rows(
        structured_rows
    )
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        _validate_schema(connection)
        strings = {
            int(row["id"]): str(row["value"])
            for row in connection.execute(
                "SELECT id, value FROM StringIds"
            )
        }
        nvtx = _read_nvtx(connection, strings)
        steps, layers, operations = _index_nvtx_ranges(
            nvtx,
            step_identities,
        )
        operation_correlations = _read_runtime_correlations(
            connection,
            operations,
        )
        kernels = _read_correlated_kernels(
            connection,
            strings,
            operation_correlations,
        )
    kernel_map = _kernel_map(kernels)
    coverage_errors = []
    correlated = {}
    for key, structured in rows_by_key.items():
        operation_range = operations.get(key)
        if operation_range is None:
            if structured["operation_class"] == "collective":
                coverage_errors.append(
                    f"rank {structured['rank']} operation "
                    f"{structured['operation_ordinal']} has no "
                    "correlated NCCL kernel"
                )
                continue
            raise ValueError("structured operation has no NVTX mapping")
        if (
            operation_range["operation_class"]
            != structured["operation_class"]
            or operation_range["operation_name"]
            != structured["operation_name"]
        ):
            raise ValueError(
                "NVTX operation payload does not match structured row"
            )
        step_key = key[:-1]
        step_range = steps[step_key]
        layer_key = step_key + (
            structured["layer_index"],
            structured["layer_role"],
        )
        layer_range = layers.get(layer_key)
        if (
            layer_range is None
            or not _contains(step_range["interval"], operation_range["interval"])
            or not _contains(layer_range["interval"], operation_range["interval"])
        ):
            raise ValueError(
                "NVTX operation is outside its step or layer"
            )
        operation_kernels = _correlate_operation_kernels(
            key,
            operation_range,
            operation_correlations,
            kernel_map,
        )
        for kernel in operation_kernels:
            if not _contains(step_range["interval"], kernel["interval"]):
                raise ValueError("correlated kernel crosses its step boundary")
        if structured["operation_class"] == "collective":
            operation_kernels = [
                kernel
                for kernel in operation_kernels
                if "nccl" in kernel["name"].lower()
            ]
            if not operation_kernels:
                coverage_errors.append(
                    f"rank {structured['rank']} operation "
                    f"{structured['operation_ordinal']} has no "
                    "correlated NCCL kernel"
                )
                continue
        elif not operation_kernels:
            raise ValueError("structured operation has no correlated kernel")
        correlated[key] = operation_kernels

    extra_operations = operations.keys() - rows_by_key.keys()
    if extra_operations:
        raise ValueError("NVTX operation has no structured mapping")
    if coverage_errors:
        return {
            "classification": "INCONCLUSIVE_TRACE_COVERAGE",
            "coverage_errors": sorted(set(coverage_errors)),
            "rows": [],
            "critical_rows": [],
        }

    result_rows, step_metrics = _build_metric_rows(
        rows_by_key,
        step_identities,
        steps,
        operations,
        correlated,
    )
    return {
        "classification": "COMPLETE",
        "coverage_errors": [],
        "rows": result_rows,
        "critical_rows": _build_critical_rows(
            result_rows,
            step_metrics,
        ),
    }


_REQUIRED_COLUMNS = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {
        "start",
        "end",
        "text",
        "globalTid",
        "textId",
    },
    "CUPTI_ACTIVITY_KIND_RUNTIME": {
        "start",
        "end",
        "globalTid",
        "correlationId",
    },
    "CUPTI_ACTIVITY_KIND_KERNEL": {
        "start",
        "end",
        "streamId",
        "correlationId",
        "globalPid",
        "demangledName",
        "shortName",
    },
}

_IDENTITY_FIELDS = (
    "attempt",
    "workload",
    "repetition",
    "request_set_sha256",
    "decode_ordinal",
    "rank",
)

_COMPUTE_CLASSES = frozenset({
    "gemm",
    "attention",
    "recurrent",
    "normalization",
    "other_compute",
})


def _validate_schema(connection):
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    for table, required in _REQUIRED_COLUMNS.items():
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
        missing = required - columns
        if missing:
            raise ValueError(
                "unsupported Nsight SQLite schema: "
                f"{table} missing columns {sorted(missing)}"
            )


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_structured_rows(structured_rows):
    if not isinstance(structured_rows, list) or not structured_rows:
        raise ValueError("structured_rows must be a non-empty list")
    required = set(_IDENTITY_FIELDS) | {
        "layer_index",
        "layer_role",
        "operation_ordinal",
        "operation_class",
        "operation_name",
    }
    rows_by_key = {}
    step_identities = {}
    ranks = set()
    ranks_by_aligned_step = {}
    for row in structured_rows:
        if not isinstance(row, dict):
            raise ValueError("structured rows must be dictionaries")
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"structured row missing fields {sorted(missing)}"
            )
        rank = _integer(row["rank"], "rank")
        repetition = _integer(row["repetition"], "repetition")
        decode_ordinal = _integer(
            row["decode_ordinal"],
            "decode_ordinal",
        )
        layer_index = _integer(row["layer_index"], "layer_index")
        operation_ordinal = _integer(
            row["operation_ordinal"],
            "operation_ordinal",
        )
        normalized = dict(row)
        normalized.update({
            "rank": rank,
            "repetition": repetition,
            "decode_ordinal": decode_ordinal,
            "layer_index": layer_index,
            "operation_ordinal": operation_ordinal,
        })
        if normalized["operation_class"] == "collective":
            if "collective_bytes" not in normalized:
                raise ValueError(
                    "collective row missing field collective_bytes"
                )
            normalized["collective_bytes"] = _integer(
                normalized["collective_bytes"],
                "collective_bytes",
            )
        step_key = tuple(normalized[name] for name in _IDENTITY_FIELDS)
        key = step_key + (operation_ordinal,)
        if key in rows_by_key:
            raise ValueError("duplicate structured operation identity")
        rows_by_key[key] = normalized
        step_identities[step_key] = {
            name: normalized[name] for name in _IDENTITY_FIELDS
        }
        ranks.add(rank)
        aligned_step_key = step_key[:-1]
        ranks_by_aligned_step.setdefault(aligned_step_key, set()).add(rank)
    if ranks != {0, 1, 2, 3}:
        raise ValueError(
            "structured rows must contain exactly ranks 0, 1, 2, and 3"
        )
    for aligned_step_ranks in ranks_by_aligned_step.values():
        if aligned_step_ranks != {0, 1, 2, 3}:
            raise ValueError(
                "aligned step must contain exactly ranks 0, 1, 2, and 3"
            )
    return rows_by_key, step_identities


def _read_nvtx(connection, strings):
    result = []
    for row in connection.execute(
        "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL"
    ):
        text = row["text"]
        if text is None and row["textId"] is not None:
            text = strings.get(int(row["textId"]))
        if not isinstance(text, str) or not text.startswith(
            "decode_internal/"
        ):
            continue
        result.append({
            "interval": Interval(int(row["start"]), int(row["end"])),
            "text": text,
            "global_tid": int(row["globalTid"]),
        })
    return result


def _read_runtime_correlations(connection, operations):
    by_thread = {}
    for key, operation in operations.items():
        by_thread.setdefault(operation["global_tid"], []).append((
            operation["interval"].start_ns,
            operation["interval"].end_ns,
            key,
        ))
    for thread_operations in by_thread.values():
        thread_operations.sort()
    correlations = {key: set() for key in operations}
    if not by_thread:
        return correlations
    tids = sorted(by_thread)
    placeholders = ", ".join("?" for _ in tids)
    minimum_start = min(
        operation["interval"].start_ns
        for operation in operations.values()
    )
    maximum_end = max(
        operation["interval"].end_ns
        for operation in operations.values()
    )
    query = (
        "SELECT start, end, globalTid, correlationId "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        f"WHERE globalTid IN ({placeholders}) "
        "AND correlationId IS NOT NULL "
        "AND start >= ? AND end <= ?"
    )
    for row in connection.execute(
        query,
        (*tids, minimum_start, maximum_end),
    ):
        global_tid = int(row["globalTid"])
        start = int(row["start"])
        end = int(row["end"])
        thread_operations = by_thread[global_tid]
        index = bisect_right(
            [item[0] for item in thread_operations],
            start,
        ) - 1
        if index < 0:
            continue
        operation_start, operation_end, key = thread_operations[index]
        if operation_start <= start and end <= operation_end:
            correlations[key].add(int(row["correlationId"]))
    return correlations


def _read_correlated_kernels(
    connection,
    strings,
    operation_correlations,
):
    correlation_ids = sorted({
        correlation_id
        for values in operation_correlations.values()
        for correlation_id in values
    })
    result = []
    for offset in range(0, len(correlation_ids), 500):
        chunk = correlation_ids[offset:offset + 500]
        placeholders = ", ".join("?" for _ in chunk)
        query = (
            "SELECT start, end, streamId, correlationId, globalPid, "
            "demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL "
            f"WHERE correlationId IN ({placeholders}) "
            "AND globalPid IS NOT NULL"
        )
        for row in connection.execute(query, chunk):
            demangled_name = row["demangledName"]
            name = (
                strings.get(int(demangled_name))
                if demangled_name is not None
                else None
            )
            short_name = row["shortName"]
            if name is None and short_name is not None:
                name = strings.get(int(short_name))
            result.append({
                "interval": Interval(
                    int(row["start"]),
                    int(row["end"]),
                ),
                "stream_id": int(row["streamId"]),
                "correlation_id": int(row["correlationId"]),
                "global_pid": int(row["globalPid"]),
                "name": "" if name is None else str(name),
            })
    return result


def _profile_identity(parts):
    identity = {}
    for component in parts:
        key, separator, value = component.partition("=")
        if not separator:
            continue
        if key in identity:
            raise ValueError(f"duplicate NVTX profile identity field {key}")
        identity[key] = value
    required = {"attempt", "workload", "repetition", "rank"}
    if required - identity.keys():
        raise ValueError("NVTX profile identity is incomplete")
    return (
        identity["attempt"],
        identity["workload"],
        _integer(int(identity["repetition"]), "repetition"),
        _integer(int(identity["rank"]), "rank"),
    )


def _parse_nvtx_label(text):
    parts = text.split("/")
    if not parts or parts[0] != "decode_internal":
        return None
    for index, component in enumerate(parts[1:], start=1):
        if component in {"decode_first", "decode_steady"}:
            if index != len(parts) - 1:
                raise ValueError("invalid NVTX step label")
            return ("step", _profile_identity(parts[1:index]), ())
        if component == "layer":
            if len(parts) != index + 3:
                raise ValueError("invalid NVTX layer label")
            return (
                "layer",
                _profile_identity(parts[1:index]),
                (
                    _integer(int(parts[index + 1]), "layer_index"),
                    parts[index + 2],
                ),
            )
        if component == "operation":
            if len(parts) < index + 4:
                raise ValueError("invalid NVTX operation label")
            return (
                "operation",
                _profile_identity(parts[1:index]),
                (
                    _integer(
                        int(parts[index + 1]),
                        "operation_ordinal",
                    ),
                    parts[index + 2],
                    "/".join(parts[index + 3:]),
                ),
            )
    return None


def _contains(outer, inner):
    return (
        outer.start_ns <= inner.start_ns
        and inner.end_ns <= outer.end_ns
    )


def _index_nvtx_ranges(nvtx, step_identities):
    raw_steps = {}
    raw_layers = []
    raw_operations = []
    for row in nvtx:
        parsed = _parse_nvtx_label(row["text"])
        if parsed is None:
            continue
        kind, prefix, suffix = parsed
        if kind == "step":
            raw_steps.setdefault(prefix, []).append(row)
        elif kind == "layer":
            raw_layers.append((prefix, suffix, row))
        else:
            raw_operations.append((prefix, suffix, row))

    structured_by_prefix = {}
    for step_key in step_identities:
        prefix = (
            step_key[0],
            step_key[1],
            step_key[2],
            step_key[5],
        )
        structured_by_prefix.setdefault(prefix, []).append(step_key)
    steps = {}
    for prefix, step_keys in structured_by_prefix.items():
        trace_steps = sorted(
            raw_steps.get(prefix, ()),
            key=lambda row: (
                row["interval"].start_ns,
                row["interval"].end_ns,
            ),
        )
        ordered_step_keys = sorted(
            set(step_keys),
            key=lambda key: key[4],
        )
        if len(trace_steps) != len(ordered_step_keys):
            raise ValueError("NVTX step inventory does not match structured rows")
        for step_key, trace_step in zip(ordered_step_keys, trace_steps):
            steps[step_key] = trace_step

    def containing_step(prefix, row):
        candidates = [
            (step_key, step)
            for step_key, step in steps.items()
            if (
                (
                    step_key[0],
                    step_key[1],
                    step_key[2],
                    step_key[5],
                )
                == prefix
                and step["global_tid"] == row["global_tid"]
                and _contains(step["interval"], row["interval"])
            )
        ]
        if len(candidates) != 1:
            raise ValueError("NVTX operation is outside its step or layer")
        return candidates[0][0]

    layers = {}
    for prefix, (layer_index, layer_role), row in raw_layers:
        step_key = containing_step(prefix, row)
        key = step_key + (layer_index, layer_role)
        if key in layers:
            raise ValueError("duplicate NVTX layer identity")
        layers[key] = row

    operations = {}
    for prefix, (
        operation_ordinal,
        operation_class,
        operation_name,
    ), row in raw_operations:
        step_key = containing_step(prefix, row)
        key = step_key + (operation_ordinal,)
        if key in operations:
            raise ValueError("duplicate NVTX operation identity")
        structured = step_identities.get(step_key)
        if structured is None:
            raise ValueError("NVTX operation has no structured step")
        row = dict(row)
        row.update({
            "operation_class": operation_class,
            "operation_name": operation_name,
        })
        operations[key] = row
    return steps, layers, operations


def _serialized_process(value):
    return int(value) >> 24


def _kernel_map(kernels):
    result = {}
    for kernel in kernels:
        key = (
            _serialized_process(kernel["global_pid"]),
            kernel["correlation_id"],
        )
        result.setdefault(key, []).append(kernel)
    return result


def _correlate_operation_kernels(
    operation_key,
    operation_range,
    operation_correlations,
    kernel_map,
):
    process = _serialized_process(operation_range["global_tid"])
    correlated = []
    seen = set()
    for correlation_id in operation_correlations[operation_key]:
        for kernel in kernel_map.get((process, correlation_id), ()):
            identity = (
                kernel["interval"],
                kernel["stream_id"],
                kernel["correlation_id"],
            )
            if identity in seen:
                continue
            seen.add(identity)
            correlated.append(kernel)
    return correlated


def _duration(intervals):
    return union_duration(intervals)


def _build_metric_rows(
    rows_by_key,
    step_identities,
    steps,
    operations,
    correlated,
):
    result = []
    step_metrics = {}
    for step_key, identity in sorted(step_identities.items()):
        step_operations = [
            (structured, correlated[key])
            for key, structured in rows_by_key.items()
            if key[:-1] == step_key
        ]
        all_kernels = [
            kernel
            for _, kernels in step_operations
            for kernel in kernels
        ]
        if not all_kernels:
            raise ValueError("step has no correlated GPU work")
        required = [kernel["interval"] for kernel in all_kernels]
        first_start = min(interval.start_ns for interval in required)
        final_end = max(interval.end_ns for interval in required)
        critical_interval = Interval(first_start, final_end)
        step_metrics[step_key] = {
            "critical_interval": critical_interval,
            "final_required_offset_ns": (
                final_end - steps[step_key]["interval"].start_ns
            ),
        }
        step_idle = _duration(
            subtract_intervals((critical_interval,), required)
        )
        layer_keys = sorted({
            (
                structured["layer_index"],
                structured["layer_role"],
            )
            for structured, _ in step_operations
        })
        for layer_index, layer_role in layer_keys:
            layer_operations = [
                (structured, kernels)
                for structured, kernels in step_operations
                if (
                    structured["layer_index"] == layer_index
                    and structured["layer_role"] == layer_role
                )
            ]
            gemm = [
                kernel["interval"]
                for structured, kernels in layer_operations
                if structured["operation_class"] == "gemm"
                for kernel in kernels
            ]
            collectives = [
                kernel["interval"]
                for structured, kernels in layer_operations
                if structured["operation_class"] == "collective"
                for kernel in kernels
            ]
            compute = [
                kernel["interval"]
                for structured, kernels in layer_operations
                if structured["operation_class"] in _COMPUTE_CLASSES
                for kernel in kernels
            ]
            layer_required = [
                kernel["interval"]
                for _, kernels in layer_operations
                for kernel in kernels
            ]
            exposed = subtract_intervals(collectives, compute)
            overlap_ns = (
                _duration(collectives) - _duration(exposed)
            )
            collective_rows = [
                structured
                for structured, _ in layer_operations
                if structured["operation_class"] == "collective"
            ]
            result.append(
                identity
                | {
                    "layer_index": layer_index,
                    "layer_role": layer_role,
                    "step_critical_interval_ns": (
                        critical_interval.end_ns
                        - critical_interval.start_ns
                    ),
                    "gemm_ns": _duration(gemm),
                    "collective_ns": _duration(collectives),
                    "compute_ns": _duration(compute),
                    "exposed_collective_ns": _duration(exposed),
                    "compute_collective_overlap_ns": overlap_ns,
                    "gpu_idle_ns": step_idle,
                    "collective_count": len(collective_rows),
                    "collective_bytes": sum(
                        structured["collective_bytes"]
                        for structured in collective_rows
                    ),
                    "critical_path_ns": _duration(layer_required),
                    "cpu_global_tids": sorted({
                        operations[
                            tuple(
                                structured[name]
                                for name in _IDENTITY_FIELDS
                            )
                            + (structured["operation_ordinal"],)
                        ]["global_tid"]
                        for structured, _ in layer_operations
                    }),
                    "stream_ids": sorted({
                        kernel["stream_id"]
                        for _, kernels in layer_operations
                        for kernel in kernels
                    }),
                }
            )
    return result, step_metrics


def _build_critical_rows(result_rows, step_metrics):
    grouped = {}
    for row in result_rows:
        alignment_key = tuple(
            row[name] for name in _IDENTITY_FIELDS[:-1]
        )
        step_key = tuple(row[name] for name in _IDENTITY_FIELDS)
        grouped.setdefault(alignment_key, []).append(
            (
                row,
                step_metrics[step_key]["final_required_offset_ns"],
            )
        )
    critical_rows = []
    for alignment_key, candidates in sorted(grouped.items()):
        row, final_required_offset = max(
            candidates,
            key=lambda item: (
                item[1],
                item[0]["rank"],
            ),
        )
        critical_rows.append({
            name: row[name] for name in _IDENTITY_FIELDS[:-1]
        } | {
            "critical_rank": row["rank"],
            "step_critical_interval_ns": (
                row["step_critical_interval_ns"]
            ),
            "final_required_offset_ns": final_required_offset,
        })
    return critical_rows
