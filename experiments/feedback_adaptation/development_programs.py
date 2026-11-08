import argparse
import hashlib
import itertools
import json
import os
import pathlib
import platform
import re
import resource
import time


def record_highs(values):
    highest = None
    count = 0
    for value in values:
        if highest is None or value > highest:
            count += 1
            highest = value
    return count


def record_ties(values):
    highest = None
    count = 0
    for value in values:
        if highest is None or value >= highest:
            count += 1
            highest = value
    return count


def adjacent_rises(values):
    return sum(
        index == 0 or value > values[index - 1]
        for index, value in enumerate(values)
    )


def half_open(values):
    start, end, point = values
    return int(start <= point < end)


def both_closed(values):
    start, end, point = values
    return int(start <= point <= end)


def both_open(values):
    start, end, point = values
    return int(start < point < end)


def binary_unsigned(text):
    if not text:
        return -1
    value = 0
    for char in text:
        if char not in "01":
            return -1
        value = value * 2 + int(char)
    return value


def binary_plus(text):
    if text.startswith("+"):
        text = text[1:]
    if not text:
        return -1
    value = 0
    for char in text:
        if char not in "01":
            return -1
        value = value * 2 + int(char)
    return value


def binary_population(text):
    if not text or any(char not in "01" for char in text):
        return -1
    return text.count("1")


def open_reads(text):
    opened = False
    count = 0
    for event in text:
        if event == "o":
            opened = True
        elif event == "c":
            opened = False
        elif opened:
            count += 1
    return count


def close_toggles(text):
    opened = False
    count = 0
    for event in text:
        if event == "o":
            opened = True
        elif event == "c":
            opened = not opened
        elif opened:
            count += 1
    return count


def open_toggles(text):
    opened = False
    count = 0
    for event in text:
        if event == "o":
            opened = not opened
        elif event == "c":
            opened = False
        elif opened:
            count += 1
    return count


PROGRAMS = {
    "dev-001": (record_highs, record_ties, adjacent_rises),
    "dev-002": (half_open, both_closed, both_open),
    "dev-003": (binary_unsigned, binary_plus, binary_population),
    "dev-004": (open_reads, close_toggles, open_toggles),
}
ROLES = ("correct", "defect", "other")
PROBES = {
    "dev-001": ([0, 0], [-1, 0, 1]),
    "dev-002": ([1, 3, 1], [1, 3, 2]),
    "dev-003": ("+1", "10"),
    "dev-004": ("or", "ocr"),
}


def domains():
    return {
        "dev-001": [
            list(values) for length in range(5)
            for values in itertools.product((-1, 0, 1), repeat=length)
        ],
        "dev-002": [
            [start, end, point] for start in range(4)
            for end in range(start, 4) for point in range(4)
        ],
        "dev-003": [
            "".join(chars) for length in range(5)
            for chars in itertools.product("01+", repeat=length)
        ],
        "dev-004": [
            "".join(events) for length in range(5)
            for events in itertools.product("ocr", repeat=length)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed development programs only; not a sandbox.")
    parser.add_argument("mode", choices=("audit", "probe"))
    parser.add_argument("task_id", nargs="?", choices=tuple(PROGRAMS))
    parser.add_argument("initial", nargs="?", choices=("correct", "defect"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{32}", args.run_id):
        parser.error("invalid run ID")
    if (args.mode == "audit" and (args.task_id or args.initial)) or (
        args.mode == "probe" and not (args.task_id and args.initial)
    ):
        parser.error("invalid fixed dispatch")
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started = time.perf_counter_ns()
    records = []
    for task_id, values in domains().items():
        if args.mode == "probe" and task_id != args.task_id:
            continue
        for role, function in zip(ROLES, PROGRAMS[task_id]):
            if args.mode == "probe" and role != args.initial:
                continue
            for index, value in enumerate(values):
                if args.mode == "probe" and value not in PROBES[task_id]:
                    continue
                before = time.perf_counter_ns()
                actual = function(value)
                elapsed = time.perf_counter_ns() - before
                records.append({
                    "task_id": task_id, "program": role, "case_id": f"case-{index:03d}",
                    "input": value, "value": actual, "elapsed_ns": elapsed,
                })
    observation = {
        "kind": "FIXED_DEVELOPMENT_OUTPUT", "run_id": args.run_id,
        "worker_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "hostname": platform.node(), "uid": os.getuid(), "python": platform.python_version(),
        "limits": {
            "cpu_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
            "core_bytes": resource.getrlimit(resource.RLIMIT_CORE)[0],
        },
        "records": records, "elapsed_ns": time.perf_counter_ns() - started,
    }
    print(json.dumps(observation, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
