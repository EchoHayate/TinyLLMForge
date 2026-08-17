#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


SCHEMA_VERSION = 1
PROC_PATHS = (
    "/proc/stat",
    "/proc/loadavg",
    "/proc/vmstat",
    "/proc/meminfo",
    "/proc/pressure/cpu",
    "/proc/pressure/io",
    "/proc/pressure/memory",
)
CPU_NAMES = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)


def _non_negative_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _finite_non_negative_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} is invalid")
    return parsed


def parse_proc_stat(text: str) -> dict[str, int]:
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if fields:
            if fields[0] in rows:
                raise ValueError("duplicate /proc/stat field")
            rows[fields[0]] = fields[1:]
    cpu = rows.get("cpu")
    if cpu is None or len(cpu) < len(CPU_NAMES):
        raise ValueError("aggregate cpu row is invalid")
    result = {
        f"cpu_{name}_ticks": _non_negative_int(
            cpu[index],
            name=f"cpu {name}",
        )
        for index, name in enumerate(CPU_NAMES)
    }
    for source, destination in (
        ("procs_running", "procs_running"),
        ("procs_blocked", "procs_blocked"),
        ("ctxt", "context_switches_total"),
        ("processes", "processes_forked_total"),
    ):
        values = rows.get(source)
        if values is None or len(values) != 1:
            raise ValueError(f"{source} is missing")
        result[destination] = _non_negative_int(
            values[0],
            name=source,
        )
    return result


def parse_loadavg(text: str) -> dict[str, float]:
    fields = text.split()
    if len(fields) < 3:
        raise ValueError("load average row is invalid")
    return {
        "loadavg_1m": _finite_non_negative_float(
            fields[0], name="load average 1m"
        ),
        "loadavg_5m": _finite_non_negative_float(
            fields[1], name="load average 5m"
        ),
        "loadavg_15m": _finite_non_negative_float(
            fields[2], name="load average 15m"
        ),
    }


def _key_value_rows(text: str, *, name: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        fields = line.replace(":", " ").split()
        if not fields:
            continue
        if fields[0] in rows:
            raise ValueError(f"duplicate {name} field")
        rows[fields[0]] = fields[1:]
    return rows


def parse_vmstat(text: str, *, page_size_kib: int) -> dict[str, int]:
    if page_size_kib <= 0:
        raise ValueError("page size must be positive")
    rows = _key_value_rows(text, name="vmstat")
    values = {}
    for key in ("pgmajfault", "pgpgin", "pgpgout", "pswpin", "pswpout"):
        fields = rows.get(key)
        if fields is None or len(fields) != 1:
            raise ValueError(f"{key} is missing")
        values[key] = _non_negative_int(fields[0], name=key)
    return {
        "major_faults_total": values["pgmajfault"],
        "page_in_kib_total": values["pgpgin"],
        "page_out_kib_total": values["pgpgout"],
        "swap_in_kib_total": values["pswpin"] * page_size_kib,
        "swap_out_kib_total": values["pswpout"] * page_size_kib,
    }


def parse_meminfo(text: str) -> dict[str, int]:
    rows = _key_value_rows(text, name="meminfo")
    result = {}
    for source, destination in (
        ("MemAvailable", "memory_available_kib"),
        ("Cached", "memory_cached_kib"),
        ("Dirty", "memory_dirty_kib"),
        ("Writeback", "memory_writeback_kib"),
    ):
        fields = rows.get(source)
        if fields is None or len(fields) != 2 or fields[1] != "kB":
            raise ValueError(f"{source} must use kB")
        result[destination] = _non_negative_int(
            fields[0],
            name=source,
        )
    return result


def parse_psi(text: str, *, resource: str) -> dict[str, int | None]:
    rows = _key_value_rows(text, name=f"{resource} PSI")
    result = {}
    for category in ("some", "full"):
        fields = rows.get(category)
        destination = f"{resource}_psi_{category}_total_us"
        if fields is None:
            if resource == "cpu" and category == "full":
                result[destination] = None
                continue
            raise ValueError(f"{resource} PSI {category} is missing")
        totals = [field for field in fields if field.startswith("total=")]
        if len(totals) != 1:
            raise ValueError(f"{resource} PSI total is invalid")
        result[destination] = _non_negative_int(
            totals[0].split("=", 1)[1],
            name=f"{resource} PSI total",
        )
    return result


def collect_sample(
    *,
    read_text,
    unix_ns,
    monotonic_ns,
    page_size_kib: int,
) -> dict:
    sampled_at_unix_ns = unix_ns()
    sampled_at_monotonic_ns = monotonic_ns()
    return {
        "schema_version": SCHEMA_VERSION,
        "sampled_at_unix_ns": sampled_at_unix_ns,
        "sampled_at_monotonic_ns": sampled_at_monotonic_ns,
        **parse_proc_stat(read_text("/proc/stat")),
        **parse_loadavg(read_text("/proc/loadavg")),
        **parse_vmstat(
            read_text("/proc/vmstat"),
            page_size_kib=page_size_kib,
        ),
        **parse_meminfo(read_text("/proc/meminfo")),
        **parse_psi(
            read_text("/proc/pressure/cpu"),
            resource="cpu",
        ),
        **parse_psi(
            read_text("/proc/pressure/io"),
            resource="io",
        ),
        **parse_psi(
            read_text("/proc/pressure/memory"),
            resource="memory",
        ),
    }


def validate_interval_seconds(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("interval seconds must be finite and positive")
    return float(value)


def emit_json_line(sample: dict, *, write) -> None:
    write(json.dumps(sample, sort_keys=True) + "\n")


def run_sampler(
    *,
    interval_seconds: float,
    collect,
    emit,
    stop_requested,
    monotonic_ns=time.monotonic_ns,
    sleep=time.sleep,
) -> int:
    interval_ns = int(validate_interval_seconds(interval_seconds) * 1e9)
    next_deadline_ns = monotonic_ns()
    while not stop_requested():
        emit(collect())
        next_deadline_ns += interval_ns
        remaining_ns = next_deadline_ns - monotonic_ns()
        if remaining_ns > 0:
            sleep(remaining_ns / 1e9)
    return 0


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.2,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    interval_seconds = validate_interval_seconds(
        args.interval_seconds
    )
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    page_size_kib = os.sysconf("SC_PAGE_SIZE") // 1024

    def collect():
        return collect_sample(
            read_text=_read_text,
            unix_ns=time.time_ns,
            monotonic_ns=time.monotonic_ns,
            page_size_kib=page_size_kib,
        )

    def emit(sample):
        emit_json_line(
            sample,
            write=lambda text: (
                sys.stdout.write(text),
                sys.stdout.flush(),
            ),
        )

    return run_sampler(
        interval_seconds=interval_seconds,
        collect=collect,
        emit=emit,
        stop_requested=lambda: stopping,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
