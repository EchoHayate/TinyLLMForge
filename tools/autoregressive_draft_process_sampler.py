from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence, TextIO


PROCESS_SAMPLE_SCHEMA_VERSION = 1
DEFAULT_INTERVAL_SECONDS = 0.01

_BINDING_FIELDS = (
    "rank",
    "gpu_uuid",
    "pid",
    "starttime_ticks",
)


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _parse_non_negative_int(text: str, *, field: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be a non-negative integer"
        ) from error
    return _non_negative_int(value, field=field)


def validate_bindings(bindings: object) -> tuple[dict, ...]:
    if (
        isinstance(bindings, (str, bytes, bytearray))
        or not isinstance(bindings, Sequence)
        or not bindings
    ):
        raise ValueError("bindings must be a non-empty sequence")

    validated = []
    seen_ranks = set()
    seen_gpu_uuids = set()
    seen_pids = set()
    for index, raw_binding in enumerate(bindings):
        if not isinstance(raw_binding, Mapping):
            raise ValueError(f"binding {index} must be a mapping")
        missing = [
            field
            for field in _BINDING_FIELDS
            if field not in raw_binding
        ]
        if missing:
            raise ValueError(
                f"binding {index} is missing {missing[0]}"
            )
        rank = _non_negative_int(
            raw_binding["rank"],
            field=f"binding {index} rank",
        )
        gpu_uuid = raw_binding["gpu_uuid"]
        if not isinstance(gpu_uuid, str) or not gpu_uuid:
            raise ValueError(
                f"binding {index} GPU UUID must be non-empty"
            )
        pid = _non_negative_int(
            raw_binding["pid"],
            field=f"binding {index} PID",
        )
        if pid == 0:
            raise ValueError(f"binding {index} PID must be positive")
        starttime_ticks = _non_negative_int(
            raw_binding["starttime_ticks"],
            field=f"binding {index} start time",
        )

        if rank in seen_ranks:
            raise ValueError(f"duplicate rank: {rank}")
        if gpu_uuid in seen_gpu_uuids:
            raise ValueError(f"duplicate GPU UUID: {gpu_uuid}")
        if pid in seen_pids:
            raise ValueError(f"duplicate PID: {pid}")
        seen_ranks.add(rank)
        seen_gpu_uuids.add(gpu_uuid)
        seen_pids.add(pid)
        validated.append({
            "rank": rank,
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "starttime_ticks": starttime_ticks,
        })
    return tuple(validated)


def parse_schedstat(text: str) -> tuple[int, int, int]:
    fields = text.split()
    if len(fields) != 3:
        raise ValueError("schedstat must contain exactly three counters")
    return tuple(
        _parse_non_negative_int(
            value,
            field=field,
        )
        for value, field in zip(
            fields,
            (
                "run time",
                "runqueue wait",
                "scheduler timeslices",
            ),
        )
    )


def parse_proc_stat(text: str) -> dict:
    opening = text.find("(")
    closing = text.rfind(")")
    if opening <= 0 or closing <= opening:
        raise ValueError("proc stat has an invalid command field")
    pid = _parse_non_negative_int(
        text[:opening].strip(),
        field="proc stat PID",
    )
    fields = text[closing + 1:].split()
    if len(fields) < 40:
        raise ValueError("proc stat is missing required fields")
    state = fields[0]
    if len(state) != 1:
        raise ValueError("proc stat state must be one character")
    return {
        "pid": pid,
        "state": state,
        "utime_ticks": _parse_non_negative_int(
            fields[11],
            field="proc stat utime",
        ),
        "stime_ticks": _parse_non_negative_int(
            fields[12],
            field="proc stat stime",
        ),
        "thread_count": _parse_non_negative_int(
            fields[17],
            field="proc stat thread count",
        ),
        "starttime_ticks": _parse_non_negative_int(
            fields[19],
            field="proc stat start time",
        ),
        "last_cpu": _parse_non_negative_int(
            fields[36],
            field="proc stat processor",
        ),
        "delayacct_blkio_ticks": _parse_non_negative_int(
            fields[39],
            field="proc stat block I/O delay",
        ),
    }


def parse_proc_status(text: str) -> tuple[int, int]:
    values = {}
    for line in text.splitlines():
        key, separator, raw_value = line.partition(":")
        if separator and key in {
            "voluntary_ctxt_switches",
            "nonvoluntary_ctxt_switches",
        }:
            values[key] = _parse_non_negative_int(
                raw_value.strip(),
                field=key,
            )
    missing = [
        key
        for key in (
            "voluntary_ctxt_switches",
            "nonvoluntary_ctxt_switches",
        )
        if key not in values
    ]
    if missing:
        raise ValueError(f"proc status is missing {missing[0]}")
    return (
        values["voluntary_ctxt_switches"],
        values["nonvoluntary_ctxt_switches"],
    )


def parse_process_sample(
    *,
    binding: Mapping[str, object],
    schedstat_text: str,
    stat_text: str,
    status_text: str,
    wchan_text: str,
    unix_ns: int,
    monotonic_ns: int,
) -> dict:
    validated_binding = validate_bindings([binding])[0]
    unix_ns = _non_negative_int(unix_ns, field="Unix timestamp")
    monotonic_ns = _non_negative_int(
        monotonic_ns,
        field="monotonic timestamp",
    )
    run_time_ns, runqueue_wait_ns, scheduler_timeslices = (
        parse_schedstat(schedstat_text)
    )
    stat = parse_proc_stat(stat_text)
    if stat["pid"] != validated_binding["pid"]:
        raise ValueError("proc stat PID does not match binding")
    if (
        stat["starttime_ticks"]
        != validated_binding["starttime_ticks"]
    ):
        raise ValueError("process start time changed")
    voluntary, involuntary = parse_proc_status(status_text)
    return {
        "schema_version": PROCESS_SAMPLE_SCHEMA_VERSION,
        "status": "sample",
        "unix_ns": unix_ns,
        "monotonic_ns": monotonic_ns,
        **validated_binding,
        "state": stat["state"],
        "last_cpu": stat["last_cpu"],
        "thread_count": stat["thread_count"],
        "wchan": wchan_text.strip(),
        "run_time_ns": run_time_ns,
        "runqueue_wait_ns": runqueue_wait_ns,
        "scheduler_timeslices": scheduler_timeslices,
        "utime_ticks": stat["utime_ticks"],
        "stime_ticks": stat["stime_ticks"],
        "delayacct_blkio_ticks": stat[
            "delayacct_blkio_ticks"
        ],
        "voluntary_context_switches": voluntary,
        "involuntary_context_switches": involuntary,
    }


def _terminal_row(
    binding: Mapping[str, object],
    *,
    unix_ns: int,
    monotonic_ns: int,
) -> dict:
    return {
        "schema_version": PROCESS_SAMPLE_SCHEMA_VERSION,
        "status": "exited",
        "unix_ns": unix_ns,
        "monotonic_ns": monotonic_ns,
        **binding,
    }


def collect_process_samples(
    bindings: object,
    *,
    proc_root: Path = Path("/proc"),
    unix_ns: int | None = None,
    monotonic_ns: int | None = None,
    read_text: Callable[[Path], str] | None = None,
) -> list[dict]:
    validated_bindings = validate_bindings(bindings)
    if unix_ns is None:
        unix_ns = time.time_ns()
    if monotonic_ns is None:
        monotonic_ns = time.monotonic_ns()
    unix_ns = _non_negative_int(unix_ns, field="Unix timestamp")
    monotonic_ns = _non_negative_int(
        monotonic_ns,
        field="monotonic timestamp",
    )
    if read_text is None:
        def read_text(path: Path) -> str:
            return path.read_text(encoding="utf-8")

    rows = []
    for binding in validated_bindings:
        process_root = proc_root / str(binding["pid"])
        try:
            stat_text = read_text(process_root / "stat")
            schedstat_text = read_text(process_root / "schedstat")
            status_text = read_text(process_root / "status")
            wchan_text = read_text(process_root / "wchan")
        except (FileNotFoundError, ProcessLookupError):
            rows.append(_terminal_row(
                binding,
                unix_ns=unix_ns,
                monotonic_ns=monotonic_ns,
            ))
            continue
        rows.append(parse_process_sample(
            binding=binding,
            schedstat_text=schedstat_text,
            stat_text=stat_text,
            status_text=status_text,
            wchan_text=wchan_text,
            unix_ns=unix_ns,
            monotonic_ns=monotonic_ns,
        ))
    return rows


def _canonical_json(row: Mapping[str, object]) -> str:
    return json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
    )


def run_sampler(
    bindings: object,
    *,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    output: TextIO = sys.stdout,
    stop_requested: Callable[[], bool] | None = None,
    proc_root: Path = Path("/proc"),
) -> int:
    validated_bindings = validate_bindings(bindings)
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or interval_seconds <= 0
    ):
        raise ValueError("interval seconds must be positive")
    if stop_requested is None:
        stop_requested = lambda: False

    active = validated_bindings
    while active and not stop_requested():
        started_ns = time.monotonic_ns()
        rows = collect_process_samples(
            active,
            proc_root=proc_root,
        )
        for row in rows:
            output.write(_canonical_json(row) + "\n")
        output.flush()
        exited_pids = {
            row["pid"]
            for row in rows
            if row["status"] == "exited"
        }
        active = tuple(
            binding
            for binding in active
            if binding["pid"] not in exited_pids
        )
        remaining_seconds = (
            interval_seconds
            - (time.monotonic_ns() - started_ns) / 1_000_000_000
        )
        if active and remaining_seconds > 0:
            time.sleep(remaining_seconds)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bindings-json",
        required=True,
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bindings = json.loads(args.bindings_json)
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return run_sampler(
        bindings,
        interval_seconds=args.interval_seconds,
        stop_requested=lambda: stopping,
    )


if __name__ == "__main__":
    raise SystemExit(main())
