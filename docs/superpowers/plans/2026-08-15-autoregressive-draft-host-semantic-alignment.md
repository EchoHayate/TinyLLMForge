# Autoregressive Draft Host Semantic Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Do not use
> subagents, create or switch branches/worktrees, stage, commit, push, stash,
> reset, or clean.

**Goal:** Add repeat-aligned `/proc` host telemetry, an independently verified
campaign artifact, and an independently verified r7/r8 cross-order comparison
that classifies whether the learned first-position slowdown is associated with
host pressure.

**Architecture:** A standalone standard-library sampler emits system-wide
JSONL every `200 ms` outside the measured worker. A separate diagnostic aligns
boundary samples to the worker's nanosecond repeat intervals, derives
repeat-local host metrics, and builds campaign-local and cross-campaign
artifacts. A separate verifier reloads raw inputs and recomputes every field;
the existing timing and GPU telemetry artifacts remain unchanged.

**Tech Stack:** Python 3.11 standard library, Bash, pytest, JSON/JSONL,
`/proc`, SSH, rsync, SHA-256.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Remote host is `sitian@10.232.195.203`.
- Remote Python is
  `/data00/home/sitian/miniconda3/envs/py311/bin/python`.
- Remote base is
  `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write new experiment artifacts under `/data00`.
- Preserve `MAX_PROPOSAL_TOKENS=4`.
- Preserve temperature zero, exact greedy parity, accepted-prefix semantics,
  and workload-derived Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Do not terminate unrelated GPU processes; preserve GPU-7 PID `703088`
  `python3`.
- Do not treat synthetic KV movement as real movement.
- Do not stage, commit, push, stash, reset, clean, or switch branches or
  worktrees.
- Prime workers remain outside all samplers and measured aggregation.
- Existing timing and GPU telemetry artifact schemas remain unchanged.

---

### Task 1: Standard-Library `/proc` Host Sampler

**Files:**
- Create: `tools/test_autoregressive_draft_host_sampler.py`
- Create: `tools/autoregressive_draft_host_sampler.py`

**Interfaces:**
- Produces:
  `parse_proc_stat(text: str) -> dict[str, int]`
- Produces:
  `parse_loadavg(text: str) -> dict[str, float | int]`
- Produces:
  `parse_vmstat(text: str, *, page_size_kib: int) -> dict[str, int]`
- Produces:
  `parse_meminfo(text: str) -> dict[str, int]`
- Produces:
  `parse_psi(text: str, *, resource: str) -> dict[str, int | None]`
- Produces:
  `collect_sample(*, read_text, unix_ns, monotonic_ns, page_size_kib) -> dict`
- Produces:
  `run_sampler(*, interval_seconds, collect, emit, stop_requested, monotonic_ns, sleep) -> int`
- CLI:
  `--interval-seconds 0.2`

- [ ] **Step 1: Write parser and complete-sample RED tests**

Create `tools/test_autoregressive_draft_host_sampler.py` with fixtures that
cover all required fields:

```python
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "autoregressive_draft_host_sampler.py"


def _load_module():
    assert MODULE_PATH.exists(), f"missing module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "autoregressive_draft_host_sampler_test_module",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROC_STAT = """\
cpu  100 2 30 400 5 6 7 8 0 0
ctxt 900
processes 77
procs_running 3
procs_blocked 1
"""

LOADAVG = "1.25 2.50 3.75 4/100 12345\n"

VMSTAT = """\
pgmajfault 11
pgpgin 120
pgpgout 80
pswpin 3
pswpout 5
"""

MEMINFO = """\
MemAvailable: 100000 kB
Cached: 20000 kB
Dirty: 300 kB
Writeback: 40 kB
"""

CPU_PSI = """\
some avg10=0.01 avg60=0.02 avg300=0.03 total=100
"""

IO_PSI = """\
some avg10=0.10 avg60=0.20 avg300=0.30 total=200
full avg10=0.04 avg60=0.05 avg300=0.06 total=50
"""


def test_collect_sample_has_exact_schema():
    module = _load_module()
    texts = {
        "/proc/stat": PROC_STAT,
        "/proc/loadavg": LOADAVG,
        "/proc/vmstat": VMSTAT,
        "/proc/meminfo": MEMINFO,
        "/proc/pressure/cpu": CPU_PSI,
        "/proc/pressure/io": IO_PSI,
        "/proc/pressure/memory": IO_PSI,
    }
    sample = module.collect_sample(
        read_text=texts.__getitem__,
        unix_ns=lambda: 1_000,
        monotonic_ns=lambda: 2_000,
        page_size_kib=4,
    )

    assert sample == {
        "schema_version": 1,
        "sampled_at_unix_ns": 1_000,
        "sampled_at_monotonic_ns": 2_000,
        "cpu_user_ticks": 100,
        "cpu_nice_ticks": 2,
        "cpu_system_ticks": 30,
        "cpu_idle_ticks": 400,
        "cpu_iowait_ticks": 5,
        "cpu_irq_ticks": 6,
        "cpu_softirq_ticks": 7,
        "cpu_steal_ticks": 8,
        "procs_running": 3,
        "procs_blocked": 1,
        "context_switches_total": 900,
        "processes_forked_total": 77,
        "loadavg_1m": 1.25,
        "loadavg_5m": 2.5,
        "loadavg_15m": 3.75,
        "major_faults_total": 11,
        "page_in_kib_total": 120,
        "page_out_kib_total": 80,
        "swap_in_kib_total": 12,
        "swap_out_kib_total": 20,
        "memory_available_kib": 100_000,
        "memory_cached_kib": 20_000,
        "memory_dirty_kib": 300,
        "memory_writeback_kib": 40,
        "cpu_psi_some_total_us": 100,
        "cpu_psi_full_total_us": None,
        "io_psi_some_total_us": 200,
        "io_psi_full_total_us": 50,
        "memory_psi_some_total_us": 200,
        "memory_psi_full_total_us": 50,
    }


@pytest.mark.parametrize(
    "parser,text,match",
    (
        ("parse_proc_stat", "cpu 1 2\n", "aggregate cpu"),
        ("parse_loadavg", "nan 1 2 3/4 5\n", "load average"),
        ("parse_vmstat", "pgmajfault -1\n", "non-negative"),
        ("parse_meminfo", "MemAvailable: 1 MB\n", "kB"),
        ("parse_psi", "some avg10=0 total=x\n", "PSI"),
    ),
)
def test_parsers_reject_invalid_inputs(parser, text, match):
    module = _load_module()
    function = getattr(module, parser)
    kwargs = {}
    if parser == "parse_vmstat":
        kwargs["page_size_kib"] = 4
    if parser == "parse_psi":
        kwargs["resource"] = "cpu"
    with pytest.raises(ValueError, match=match):
        function(text, **kwargs)
```

- [ ] **Step 2: Run the parser tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_sampler.py \
  -k 'collect_sample or parsers'
```

Expected: collection fails because
`tools/autoregressive_draft_host_sampler.py` does not exist.

- [ ] **Step 3: Implement strict parsers and `collect_sample`**

Create `tools/autoregressive_draft_host_sampler.py` with these constants and
functions:

```python
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
```

- [ ] **Step 4: Write sampler cadence, interval, and signal RED tests**

Append:

```python
def test_run_sampler_emits_immediately_and_uses_deadlines():
    module = _load_module()
    emitted = []
    sleeps = []
    monotonic_values = iter((0, 100_000_000, 300_000_000))
    samples = iter(({"sampled_at_unix_ns": 1}, {"sampled_at_unix_ns": 2}))

    status = module.run_sampler(
        interval_seconds=0.2,
        collect=lambda: next(samples),
        emit=emitted.append,
        stop_requested=lambda: len(emitted) == 2,
        monotonic_ns=lambda: next(monotonic_values),
        sleep=sleeps.append,
    )

    assert status == 0
    assert emitted == [
        {"sampled_at_unix_ns": 1},
        {"sampled_at_unix_ns": 2},
    ]
    assert sleeps == [0.1, 0.1]


@pytest.mark.parametrize("value", (0.0, -0.1, math.nan, math.inf))
def test_validate_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="interval"):
        _load_module().validate_interval_seconds(value)


def test_emit_json_line_is_atomic_per_sample():
    module = _load_module()
    writes = []
    module.emit_json_line(
        {"schema_version": 1, "sampled_at_unix_ns": 1},
        write=writes.append,
    )
    assert len(writes) == 1
    assert json.loads(writes[0])["schema_version"] == 1
    assert writes[0].endswith("\n")
```

- [ ] **Step 5: Run cadence tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_sampler.py \
  -k 'run_sampler or interval or atomic'
```

Expected: failure because sampler loop helpers are not implemented.

- [ ] **Step 6: Implement deadline-based loop and CLI**

Append:

```python
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
```

- [ ] **Step 7: Run Task 1 GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_sampler.py

python3 -m py_compile \
  tools/autoregressive_draft_host_sampler.py
```

Expected: all sampler tests pass and compilation succeeds.

---

### Task 2: JSONL Validation, Repeat Alignment, and Derived Metrics

**Files:**
- Create: `tools/test_autoregressive_draft_host_semantic_diagnostic.py`
- Create: `tools/autoregressive_draft_host_semantic_diagnostic.py`

**Interfaces:**
- Consumes sampler schema-v1 rows and measured worker JSON.
- Produces:
  `parse_host_jsonl(text: str) -> list[dict]`
- Produces:
  `align_repeat_samples(worker: dict, samples: list[dict]) -> list[dict]`
- Produces:
  `derive_repeat_metrics(alignment: dict) -> dict`
- Produces:
  `extract_repeat_timing(worker: dict) -> list[dict]`

- [ ] **Step 1: Write JSONL and boundary-alignment RED tests**

Create `tools/test_autoregressive_draft_host_semantic_diagnostic.py`:

```python
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "autoregressive_draft_host_semantic_diagnostic.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_host_semantic_diagnostic.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_b4_instability_telemetry_remote.sh"
)


def _load_module():
    assert MODULE_PATH.exists(), f"missing module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "autoregressive_draft_host_semantic_test_module",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(unix_ns, monotonic_ns, offset=0):
    return {
        "schema_version": 1,
        "sampled_at_unix_ns": unix_ns,
        "sampled_at_monotonic_ns": monotonic_ns,
        "cpu_user_ticks": 100 + offset,
        "cpu_nice_ticks": 2,
        "cpu_system_ticks": 20 + offset,
        "cpu_idle_ticks": 300 + offset,
        "cpu_iowait_ticks": 5 + offset,
        "cpu_irq_ticks": 1,
        "cpu_softirq_ticks": 2,
        "cpu_steal_ticks": 0,
        "procs_running": 2 + offset,
        "procs_blocked": 1,
        "context_switches_total": 1000 + 10 * offset,
        "processes_forked_total": 50 + offset,
        "loadavg_1m": 1.0 + offset,
        "loadavg_5m": 2.0,
        "loadavg_15m": 3.0,
        "major_faults_total": 10 + offset,
        "page_in_kib_total": 100 + 4 * offset,
        "page_out_kib_total": 200 + 8 * offset,
        "swap_in_kib_total": 0,
        "swap_out_kib_total": 0,
        "memory_available_kib": 10000 - offset,
        "memory_cached_kib": 5000,
        "memory_dirty_kib": 20 + offset,
        "memory_writeback_kib": 2 + offset,
        "cpu_psi_some_total_us": 100 + 10 * offset,
        "cpu_psi_full_total_us": None,
        "io_psi_some_total_us": 200 + 20 * offset,
        "io_psi_full_total_us": 20 + 2 * offset,
        "memory_psi_some_total_us": 300 + 30 * offset,
        "memory_psi_full_total_us": 30 + 3 * offset,
    }


def _worker():
    return {
        "policy": "learned",
        "batch_size": 4,
        "measured_runs": [{
            "repeat": 0,
            "campaign_interval": {
                "started_at_unix_ns": 1_000_000_000,
                "finished_at_unix_ns": 2_000_000_000,
            },
            "timing": {
                "per_request": [
                    {"completion_latency_s": value, "tpot_s": value / 10}
                    for value in (1.0, 2.0, 3.0, 4.0)
                ],
            },
            "runtime": {
                "draft_executor_timing": {
                    "max_rank_ms": {"proposal_forward": 55.0}
                }
            },
            "outputs": [[1], [2], [3], [4]],
        }],
    }


def test_align_repeat_uses_outer_boundary_samples():
    module = _load_module()
    samples = [
        _sample(800_000_000, 100, 0),
        _sample(1_100_000_000, 400, 1),
        _sample(1_700_000_000, 1_000, 2),
        _sample(2_200_000_000, 1_500, 3),
    ]
    aligned = module.align_repeat_samples(_worker(), samples)
    assert aligned[0]["samples"] == samples
    assert aligned[0]["host_sample_interval"] == {
        "started_at_unix_ns": 800_000_000,
        "finished_at_unix_ns": 2_200_000_000,
        "started_at_monotonic_ns": 100,
        "finished_at_monotonic_ns": 1_500,
    }


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda rows: rows.__setitem__(
                0, _sample(500_000_000, 100, 0)
            ),
            "start boundary",
        ),
        (
            lambda rows: rows.__setitem__(
                -1, _sample(2_500_000_000, 1_500, 3)
            ),
            "finish boundary",
        ),
        (
            lambda rows: rows[2].__setitem__(
                "sampled_at_monotonic_ns", 100
            ),
            "timestamp",
        ),
        (
            lambda rows: rows[2].__setitem__(
                "context_switches_total", 1
            ),
            "counter regressed",
        ),
    ),
)
def test_align_repeat_rejects_invalid_coverage(mutate, match):
    rows = [
        _sample(800_000_000, 100, 0),
        _sample(1_100_000_000, 300_000_000, 1),
        _sample(1_700_000_000, 900_000_000, 2),
        _sample(2_200_000_000, 1_400_000_000, 3),
    ]
    mutate(rows)
    with pytest.raises(ValueError, match=match):
        _load_module().align_repeat_samples(_worker(), rows)
```

- [ ] **Step 2: Run alignment tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k align_repeat
```

Expected: failure because the diagnostic module does not exist.

- [ ] **Step 3: Implement schema validation and boundary alignment**

Create the module with:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


SCHEMA_VERSION = 1
EDGE_ALLOWANCE_NS = 400_000_000
MAX_SAMPLE_GAP_NS = 600_000_000
EXPECTED_MEASURED_RUNS = 8
POLICIES = ("target", "learned")
POLICY_ORDERS = ("target,learned", "learned,target")
TIMING_METRICS = (
    "e2e_s",
    "tpot_s",
    "executor_proposal_forward_ms",
)
CUMULATIVE_COUNTERS = (
    "cpu_user_ticks",
    "cpu_nice_ticks",
    "cpu_system_ticks",
    "cpu_idle_ticks",
    "cpu_iowait_ticks",
    "cpu_irq_ticks",
    "cpu_softirq_ticks",
    "cpu_steal_ticks",
    "context_switches_total",
    "processes_forked_total",
    "major_faults_total",
    "page_in_kib_total",
    "page_out_kib_total",
    "swap_in_kib_total",
    "swap_out_kib_total",
    "cpu_psi_some_total_us",
    "io_psi_some_total_us",
    "io_psi_full_total_us",
    "memory_psi_some_total_us",
    "memory_psi_full_total_us",
)
OPTIONAL_COUNTERS = ("cpu_psi_full_total_us",)


def _validate_sample(sample: object) -> dict:
    if not isinstance(sample, dict):
        raise ValueError("host sample is invalid")
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("host sample schema is invalid")
    normalized = copy.deepcopy(sample)
    for name, value in normalized.items():
        if name == "schema_version":
            continue
        if name in OPTIONAL_COUNTERS and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"host sample {name} is invalid")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"host sample {name} is invalid")
    return normalized


def parse_host_jsonl(text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("host JSONL is empty")
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"host JSONL line {line_number} is invalid"
            ) from error
        rows.append(_validate_sample(row))
    for previous, current in zip(rows, rows[1:]):
        if (
            current["sampled_at_unix_ns"]
            <= previous["sampled_at_unix_ns"]
            or current["sampled_at_monotonic_ns"]
            <= previous["sampled_at_monotonic_ns"]
        ):
            raise ValueError("host sample timestamp regressed")
        if (
            current["sampled_at_monotonic_ns"]
            - previous["sampled_at_monotonic_ns"]
            > MAX_SAMPLE_GAP_NS
        ):
            raise ValueError("host sample gap exceeds limit")
    return rows


def _campaign_interval(run: dict) -> tuple[int, int]:
    interval = run.get("campaign_interval")
    if not isinstance(interval, dict):
        raise ValueError("campaign interval is missing")
    start = interval.get("started_at_unix_ns")
    finish = interval.get("finished_at_unix_ns")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(finish, bool)
        or not isinstance(finish, int)
        or start <= 0
        or finish <= start
    ):
        raise ValueError("campaign interval is invalid")
    return start, finish


def align_repeat_samples(
    worker: dict,
    samples: list[dict],
) -> list[dict]:
    validated = [_validate_sample(row) for row in samples]
    aligned = []
    for run in worker["measured_runs"]:
        start, finish = _campaign_interval(run)
        start_candidates = [
            row for row in validated
            if row["sampled_at_unix_ns"] <= start
        ]
        finish_candidates = [
            row for row in validated
            if row["sampled_at_unix_ns"] >= finish
        ]
        if not start_candidates:
            raise ValueError("host start boundary is missing")
        if not finish_candidates:
            raise ValueError("host finish boundary is missing")
        first = start_candidates[-1]
        last = finish_candidates[0]
        if start - first["sampled_at_unix_ns"] > EDGE_ALLOWANCE_NS:
            raise ValueError("host start boundary exceeds allowance")
        if last["sampled_at_unix_ns"] - finish > EDGE_ALLOWANCE_NS:
            raise ValueError("host finish boundary exceeds allowance")
        interval_rows = [
            row for row in validated
            if first["sampled_at_unix_ns"]
            <= row["sampled_at_unix_ns"]
            <= last["sampled_at_unix_ns"]
        ]
        if len(interval_rows) < 2:
            raise ValueError("host repeat has fewer than two samples")
        for previous, current in zip(
            interval_rows,
            interval_rows[1:],
        ):
            if (
                current["sampled_at_unix_ns"]
                <= previous["sampled_at_unix_ns"]
                or current["sampled_at_monotonic_ns"]
                <= previous["sampled_at_monotonic_ns"]
            ):
                raise ValueError("host sample timestamp regressed")
            if (
                current["sampled_at_monotonic_ns"]
                - previous["sampled_at_monotonic_ns"]
                > MAX_SAMPLE_GAP_NS
            ):
                raise ValueError("host sample gap exceeds limit")
        for name in CUMULATIVE_COUNTERS + OPTIONAL_COUNTERS:
            before = first[name]
            after = last[name]
            if before is None and after is None:
                continue
            if before is None or after is None or after < before:
                raise ValueError(f"host counter regressed: {name}")
        aligned.append({
            "repeat": run["repeat"],
            "campaign_interval": copy.deepcopy(
                run["campaign_interval"]
            ),
            "host_sample_interval": {
                "started_at_unix_ns": first["sampled_at_unix_ns"],
                "finished_at_unix_ns": last["sampled_at_unix_ns"],
                "started_at_monotonic_ns": (
                    first["sampled_at_monotonic_ns"]
                ),
                "finished_at_monotonic_ns": (
                    last["sampled_at_monotonic_ns"]
                ),
            },
            "samples": interval_rows,
        })
    return aligned
```

- [ ] **Step 4: Write hand-computed metric and timing RED tests**

Append:

```python
def test_derive_repeat_metrics_matches_hand_computation():
    module = _load_module()
    rows = [
        _sample(1_000_000_000, 1_000_000_000, 0),
        _sample(2_000_000_000, 2_000_000_000, 1),
    ]
    metrics = module.derive_repeat_metrics({
        "samples": rows,
        "host_sample_interval": {
            "started_at_monotonic_ns": 1_000_000_000,
            "finished_at_monotonic_ns": 2_000_000_000,
        },
    })

    assert metrics["context_switches_per_second"] == 10.0
    assert metrics["forks_per_second"] == 1.0
    assert metrics["major_faults_per_second"] == 1.0
    assert metrics["page_in_kib_per_second"] == 4.0
    assert metrics["page_out_kib_per_second"] == 8.0
    assert metrics["run_queue_mean"] == 2.5
    assert metrics["run_queue_max"] == 3
    assert metrics["memory_available_kib_min"] == 9_999
    assert metrics["memory_dirty_kib_max"] == 21
    assert metrics["io_psi_some_fraction"] == 20 / 1_000_000


def test_extract_repeat_timing_uses_batch_medians():
    row = _load_module().extract_repeat_timing(_worker())[0]
    assert row == {
        "repeat": 0,
        "e2e_s": 2.5,
        "tpot_s": 0.25,
        "executor_proposal_forward_ms": 55.0,
    }
```

- [ ] **Step 5: Run metric tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k 'derive_repeat_metrics or extract_repeat_timing'
```

Expected: failure because metric derivation is not implemented.

- [ ] **Step 6: Implement metric and timing derivation**

Append:

```python
def _delta(first: dict, last: dict, name: str) -> float:
    return float(last[name] - first[name])


def _mean(rows: list[dict], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def derive_repeat_metrics(alignment: dict) -> dict:
    rows = alignment["samples"]
    first = rows[0]
    last = rows[-1]
    elapsed_seconds = (
        alignment["host_sample_interval"][
            "finished_at_monotonic_ns"
        ]
        - alignment["host_sample_interval"][
            "started_at_monotonic_ns"
        ]
    ) / 1e9
    if elapsed_seconds <= 0.0:
        raise ValueError("host sample duration is invalid")
    cpu_names = (
        "cpu_user_ticks",
        "cpu_nice_ticks",
        "cpu_system_ticks",
        "cpu_idle_ticks",
        "cpu_iowait_ticks",
        "cpu_irq_ticks",
        "cpu_softirq_ticks",
        "cpu_steal_ticks",
    )
    cpu_deltas = {name: _delta(first, last, name) for name in cpu_names}
    cpu_total = sum(cpu_deltas.values())
    if cpu_total <= 0.0:
        raise ValueError("aggregate CPU delta is zero")
    busy = (
        cpu_deltas["cpu_user_ticks"]
        + cpu_deltas["cpu_nice_ticks"]
        + cpu_deltas["cpu_system_ticks"]
        + cpu_deltas["cpu_irq_ticks"]
        + cpu_deltas["cpu_softirq_ticks"]
        + cpu_deltas["cpu_steal_ticks"]
    )

    def rate(name: str) -> float:
        return _delta(first, last, name) / elapsed_seconds

    def psi_fraction(name: str) -> float | None:
        if first[name] is None and last[name] is None:
            return None
        return _delta(first, last, name) / (
            elapsed_seconds * 1_000_000
        )

    return {
        "cpu_busy_fraction": busy / cpu_total,
        "cpu_system_fraction": (
            cpu_deltas["cpu_system_ticks"] / cpu_total
        ),
        "cpu_iowait_fraction": (
            cpu_deltas["cpu_iowait_ticks"] / cpu_total
        ),
        "cpu_steal_fraction": (
            cpu_deltas["cpu_steal_ticks"] / cpu_total
        ),
        "run_queue_mean": _mean(rows, "procs_running"),
        "run_queue_max": max(row["procs_running"] for row in rows),
        "blocked_processes_mean": _mean(rows, "procs_blocked"),
        "blocked_processes_max": max(
            row["procs_blocked"] for row in rows
        ),
        "loadavg_1m_mean": _mean(rows, "loadavg_1m"),
        "context_switches_per_second": rate(
            "context_switches_total"
        ),
        "forks_per_second": rate("processes_forked_total"),
        "major_faults_per_second": rate("major_faults_total"),
        "page_in_kib_per_second": rate("page_in_kib_total"),
        "page_out_kib_per_second": rate("page_out_kib_total"),
        "swap_in_kib_per_second": rate("swap_in_kib_total"),
        "swap_out_kib_per_second": rate("swap_out_kib_total"),
        "memory_available_kib_min": min(
            row["memory_available_kib"] for row in rows
        ),
        "memory_dirty_kib_max": max(
            row["memory_dirty_kib"] for row in rows
        ),
        "memory_writeback_kib_max": max(
            row["memory_writeback_kib"] for row in rows
        ),
        "cpu_psi_some_fraction": psi_fraction(
            "cpu_psi_some_total_us"
        ),
        "cpu_psi_full_fraction": psi_fraction(
            "cpu_psi_full_total_us"
        ),
        "io_psi_some_fraction": psi_fraction(
            "io_psi_some_total_us"
        ),
        "io_psi_full_fraction": psi_fraction(
            "io_psi_full_total_us"
        ),
        "memory_psi_some_fraction": psi_fraction(
            "memory_psi_some_total_us"
        ),
        "memory_psi_full_fraction": psi_fraction(
            "memory_psi_full_total_us"
        ),
    }


def extract_repeat_timing(worker: dict) -> list[dict]:
    rows = []
    for run in worker["measured_runs"]:
        per_request = run["timing"]["per_request"]
        rows.append({
            "repeat": run["repeat"],
            "e2e_s": statistics.median(
                row["completion_latency_s"] for row in per_request
            ),
            "tpot_s": statistics.median(
                row["tpot_s"] for row in per_request
            ),
            "executor_proposal_forward_ms": float(
                run["runtime"]["draft_executor_timing"][
                    "max_rank_ms"
                ]["proposal_forward"]
            ),
        })
    return rows
```

- [ ] **Step 7: Run Task 2 GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k 'jsonl or align_repeat or derive_repeat_metrics or extract_repeat_timing'
```

Expected: focused parser, alignment, metric, and timing tests pass.

---

### Task 3: Campaign Artifact and Cross-Campaign Classification

**Files:**
- Modify: `tools/test_autoregressive_draft_host_semantic_diagnostic.py`
- Modify: `tools/autoregressive_draft_host_semantic_diagnostic.py`

**Interfaces:**
- Produces:
  `build_host_semantic_artifact(*, timing_artifact, gpu_telemetry_artifact, target_worker, learned_worker, target_samples, learned_samples, policy_order, prime_each_policy, source_files, input_files) -> dict`
- Produces:
  `average_ranks(values: list[float]) -> list[float]`
- Produces:
  `spearman_rho(left: list[float], right: list[float]) -> float | None`
- Produces:
  `build_host_semantic_comparison(*, first_artifact, second_artifact, first_reference, second_reference) -> dict`

- [ ] **Step 1: Write campaign artifact construction RED tests**

Add these concrete fixture builders:

```python
def _campaign_worker(policy, *, learned_scale=1.0):
    runs = []
    for repeat in range(8):
        start = 10_000_000_000 + repeat * 2_000_000_000
        scale = learned_scale if policy == "learned" else 1.0
        base = (2.0 + repeat * 0.01) * scale
        runs.append({
            "repeat": repeat,
            "campaign_interval": {
                "started_at_unix_ns": start,
                "finished_at_unix_ns": start + 1_000_000_000,
            },
            "timing": {
                "per_request": [
                    {
                        "completion_latency_s": base + offset,
                        "tpot_s": base / 10 + offset / 10,
                    }
                    for offset in (0.0, 0.1, 0.2, 0.3)
                ],
            },
            "runtime": {
                "draft_executor_timing": {
                    "max_rank_ms": {
                        "proposal_forward": (
                            50.0 + repeat
                        ) * scale
                    }
                }
            },
            "outputs": [
                [repeat, request] for request in range(4)
            ],
        })
    return {
        "policy": policy,
        "batch_size": 4,
        "tensor_parallel_size": 4,
        "measured_runs": runs,
    }


def _campaign_samples(*, pressure_scale=1.0):
    rows = []
    start = 9_800_000_000
    for index in range(78):
        row = _sample(
            start + index * 200_000_000,
            1_000_000_000 + index * 200_000_000,
            index,
        )
        row["procs_running"] = 2 + int(
            pressure_scale * (index % 4)
        )
        row["context_switches_total"] = (
            1_000 + int(pressure_scale * 10 * index)
        )
        row["major_faults_total"] = (
            10 + int(pressure_scale * index)
        )
        row["io_psi_some_total_us"] = (
            200 + int(pressure_scale * 20 * index)
        )
        row["memory_psi_some_total_us"] = (
            300 + int(pressure_scale * 30 * index)
        )
        rows.append(row)
    return rows


def _timing_artifact():
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "STABLE",
        "exact_parity": True,
    }


def _gpu_artifact():
    return {
        "schema_version": 1,
        "status": "PASS",
        "timing_classification": "STABLE",
        "telemetry_classification": "STABLE_BASELINE",
        "exact_parity": True,
    }


def _input_files():
    names = (
        "timing_artifact",
        "gpu_telemetry_artifact",
        "target_worker",
        "learned_worker",
        "target_host_jsonl",
        "learned_host_jsonl",
    )
    paths = (
        "result.json",
        "telemetry.json",
        "workers/target-b4.json",
        "workers/learned-b4.json",
        "host-semantic/target-host.jsonl",
        "host-semantic/learned-host.jsonl",
    )
    return {
        name: {
            "path": path,
            "sha256": f"{index + 1:064x}",
        }
        for index, (name, path) in enumerate(zip(names, paths))
    }


def _campaign_artifact_kwargs(
    *,
    policy_order="target,learned",
    learned_scale=1.0,
    learned_pressure_scale=1.0,
):
    return {
        "timing_artifact": _timing_artifact(),
        "gpu_telemetry_artifact": _gpu_artifact(),
        "target_worker": _campaign_worker("target"),
        "learned_worker": _campaign_worker(
            "learned",
            learned_scale=learned_scale,
        ),
        "target_samples": _campaign_samples(),
        "learned_samples": _campaign_samples(
            pressure_scale=learned_pressure_scale
        ),
        "policy_order": policy_order,
        "prime_each_policy": True,
        "source_files": {
            "tools/source.py": "a" * 64,
        },
        "input_files": _input_files(),
    }
```

Then add:

```python
def test_build_campaign_artifact_is_aligned_not_cross_classified():
    module = _load_module()
    artifact = module.build_host_semantic_artifact(
        **_campaign_artifact_kwargs()
    )

    assert artifact["status"] == "PASS"
    assert artifact["classification"] == "ALIGNED_CAMPAIGN"
    assert artifact["exact_parity"] is True
    assert len(
        artifact["policies"]["learned"]["measured_runs"]
    ) == 8


def test_build_campaign_artifact_is_deterministic():
    module = _load_module()
    kwargs = _campaign_artifact_kwargs()
    assert module.build_host_semantic_artifact(
        **copy.deepcopy(kwargs)
    ) == module.build_host_semantic_artifact(
        **copy.deepcopy(kwargs)
    )
```

- [ ] **Step 2: Run campaign artifact tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k campaign_artifact
```

Expected: failure because artifact builders do not exist.

- [ ] **Step 3: Implement source-bound campaign artifact**

Add:

```python
SOURCE_FILE_PATHS = (
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_b4_timing_diagnostic.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_host_sampler.py",
    "tools/autoregressive_draft_host_semantic_diagnostic.py",
    "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
)
LIMITATIONS = (
    "host correlation is not causal proof",
    "system-wide pressure does not identify a responsible process",
    "campaign does not establish stable long-context performance",
    "campaign does not establish Proposal-KV offload benefit",
    "campaign does not establish Phase-1 promotion",
)
THRESHOLDS = {
    "sample_interval_seconds": 0.2,
    "edge_allowance_seconds": 0.4,
    "maximum_sample_gap_seconds": 0.6,
    "position_effect_fraction": 0.10,
    "host_metric_worse_fraction": 0.10,
    "spearman_rho_minimum": 0.60,
    "minimum_worse_primary_metrics": 2,
    "minimum_correlated_primary_metrics": 2,
}


def build_host_semantic_artifact(
    *,
    timing_artifact: dict,
    gpu_telemetry_artifact: dict,
    target_worker: dict,
    learned_worker: dict,
    target_samples: list[dict],
    learned_samples: list[dict],
    policy_order: str,
    prime_each_policy: bool,
    source_files: dict[str, str],
    input_files: dict[str, dict],
) -> dict:
    if policy_order not in POLICY_ORDERS:
        raise ValueError("policy order is invalid")
    if prime_each_policy is not True:
        raise ValueError("same-policy priming is required")
    if (
        timing_artifact.get("status") != "PASS"
        or timing_artifact.get("exact_parity") is not True
        or gpu_telemetry_artifact.get("status") != "PASS"
        or gpu_telemetry_artifact.get("exact_parity") is not True
    ):
        raise ValueError("upstream artifact is invalid")
    for repeat in range(EXPECTED_MEASURED_RUNS):
        if (
            target_worker["measured_runs"][repeat]["outputs"]
            != learned_worker["measured_runs"][repeat]["outputs"]
        ):
            raise ValueError(
                f"host diagnostic exact parity failed at repeat {repeat}"
            )
    policies = {}
    for policy, worker, samples in (
        ("target", target_worker, target_samples),
        ("learned", learned_worker, learned_samples),
    ):
        if worker.get("policy") != policy:
            raise ValueError("worker policy mismatch")
        if len(worker.get("measured_runs", [])) != EXPECTED_MEASURED_RUNS:
            raise ValueError("worker measured repeat count is invalid")
        alignments = align_repeat_samples(worker, samples)
        timing_rows = extract_repeat_timing(worker)
        policies[policy] = {
            "worker_sha256": input_files[f"{policy}_worker"]["sha256"],
            "host_jsonl_sha256": input_files[
                f"{policy}_host_jsonl"
            ]["sha256"],
            "sample_count": len(samples),
            "measured_runs": [
                {
                    "repeat": alignment["repeat"],
                    "campaign_interval": alignment[
                        "campaign_interval"
                    ],
                    "host_sample_interval": alignment[
                        "host_sample_interval"
                    ],
                    "sample_count": len(alignment["samples"]),
                    "duration_seconds": (
                        alignment["host_sample_interval"][
                            "finished_at_monotonic_ns"
                        ]
                        - alignment["host_sample_interval"][
                            "started_at_monotonic_ns"
                        ]
                    ) / 1e9,
                    "metrics": derive_repeat_metrics(alignment),
                    "timing": timing_rows[index],
                }
                for index, alignment in enumerate(alignments)
            ],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": "ALIGNED_CAMPAIGN",
        "classification_reasons": [],
        "exact_parity": True,
        "policy_order": policy_order,
        "prime_each_policy": True,
        "timing_artifact_sha256": input_files[
            "timing_artifact"
        ]["sha256"],
        "gpu_telemetry_artifact_sha256": input_files[
            "gpu_telemetry_artifact"
        ]["sha256"],
        "policies": policies,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "source_files": copy.deepcopy(source_files),
        "input_files": copy.deepcopy(input_files),
        "limitations": list(LIMITATIONS),
    }
```

Filesystem-bound recomputation and tamper detection belong exclusively to the
independent verifier in Task 4; do not add a validator that trusts summaries
embedded in the artifact.

- [ ] **Step 4: Write rank, correlation, and classification RED tests**

Append:

```python
def test_average_ranks_handles_ties():
    assert _load_module().average_ranks([10.0, 20.0, 20.0, 40.0]) == [
        1.0, 2.5, 2.5, 4.0
    ]


def test_spearman_returns_none_for_zero_variance():
    assert _load_module().spearman_rho(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ) is None


@pytest.mark.parametrize(
    "position_effect,worse_count,e2e_corr,proposal_corr,expected",
    (
        (
            0.20,
            2,
            {"run_queue_mean", "io_psi_some_fraction"},
            {"run_queue_mean", "major_faults_per_second"},
            "HOST_PRESSURE_ASSOCIATED",
        ),
        (
            0.20,
            1,
            {"run_queue_mean"},
            {"run_queue_mean"},
            "HOST_PRESSURE_NOT_SUPPORTED",
        ),
        (
            0.05,
            3,
            {"run_queue_mean", "io_psi_some_fraction"},
            {"run_queue_mean", "io_psi_some_fraction"},
            "HOST_ALIGNMENT_INCONCLUSIVE",
        ),
    ),
)
def test_classify_host_comparison(
    position_effect,
    worse_count,
    e2e_corr,
    proposal_corr,
    expected,
):
    module = _load_module()
    classification, reasons = module.classify_host_comparison(
        learned_e2e_relative_delta=position_effect,
        worse_primary_metrics={
            f"metric_{index}" for index in range(worse_count)
        },
        e2e_correlated_metrics=e2e_corr,
        proposal_correlated_metrics=proposal_corr,
    )
    assert classification == expected
    assert isinstance(reasons, list)
```

- [ ] **Step 5: Run comparison tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k 'average_ranks or spearman or classify_host'
```

Expected: failure because comparison helpers do not exist.

- [ ] **Step 6: Implement deterministic ranks, Spearman, and classification**

Add:

```python
PRIMARY_HOST_METRICS = (
    "cpu_system_fraction",
    "cpu_iowait_fraction",
    "run_queue_mean",
    "context_switches_per_second",
    "major_faults_per_second",
    "io_psi_some_fraction",
    "memory_psi_some_fraction",
    "memory_dirty_kib_max",
    "memory_writeback_kib_max",
)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda row: row[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average
        cursor = end
    return ranks


def spearman_rho(
    left: list[float],
    right: list[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman vectors are invalid")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_sum = sum(
        (value - left_mean) ** 2 for value in left_ranks
    )
    right_sum = sum(
        (value - right_mean) ** 2 for value in right_ranks
    )
    if left_sum <= 0.0 or right_sum <= 0.0:
        return None
    return numerator / math.sqrt(left_sum * right_sum)


def classify_host_comparison(
    *,
    learned_e2e_relative_delta: float,
    worse_primary_metrics: set[str],
    e2e_correlated_metrics: set[str],
    proposal_correlated_metrics: set[str],
) -> tuple[str, list[str]]:
    reasons = []
    if learned_e2e_relative_delta < THRESHOLDS[
        "position_effect_fraction"
    ]:
        return (
            "HOST_ALIGNMENT_INCONCLUSIVE",
            ["learned E2E position effect is below 10%"],
        )
    correlated_union = (
        e2e_correlated_metrics | proposal_correlated_metrics
    )
    associated = (
        len(worse_primary_metrics)
        >= THRESHOLDS["minimum_worse_primary_metrics"]
        and len(correlated_union)
        >= THRESHOLDS["minimum_correlated_primary_metrics"]
        and bool(e2e_correlated_metrics)
        and bool(proposal_correlated_metrics)
    )
    if associated:
        return (
            "HOST_PRESSURE_ASSOCIATED",
            [
                "learned first-position E2E is at least 10% slower",
                "at least two primary host metrics are worse",
                "expected-direction correlation covers E2E and proposal-forward",
            ],
        )
    if len(worse_primary_metrics) < 2:
        reasons.append("fewer than two primary host metrics are worse")
    if len(correlated_union) < 2:
        reasons.append(
            "fewer than two primary host metrics meet the rho threshold"
        )
    if not e2e_correlated_metrics:
        reasons.append("no primary host metric correlates with learned E2E")
    if not proposal_correlated_metrics:
        reasons.append(
            "no primary host metric correlates with proposal-forward"
        )
    return "HOST_PRESSURE_NOT_SUPPORTED", reasons
```

- [ ] **Step 7: Implement cross-campaign builder and exact comparison schema**

Add a builder whose inputs are two validated campaign artifacts plus their
relative paths and hashes:

```python
def build_host_semantic_comparison(
    *,
    first_artifact: dict,
    second_artifact: dict,
    first_reference: dict,
    second_reference: dict,
) -> dict:
    campaigns = [first_artifact, second_artifact]
    if any(
        row.get("status") != "PASS"
        or row.get("classification") != "ALIGNED_CAMPAIGN"
        or row.get("exact_parity") is not True
        for row in campaigns
    ):
        raise ValueError("comparison campaign is invalid")
    orders = {row["policy_order"] for row in campaigns}
    if orders != set(POLICY_ORDERS):
        raise ValueError("comparison policy orders are invalid")
    if any(row.get("prime_each_policy") is not True for row in campaigns):
        raise ValueError("comparison requires primed campaigns")
    if campaigns[0]["source_files"] != campaigns[1]["source_files"]:
        raise ValueError("comparison source identity mismatch")
    for name in (
        "timing_artifact",
        "gpu_telemetry_artifact",
        "target_worker",
        "learned_worker",
        "target_host_jsonl",
        "learned_host_jsonl",
    ):
        if (
            campaigns[0]["input_files"][name]["sha256"]
            == campaigns[1]["input_files"][name]["sha256"]
        ):
            raise ValueError(
                f"comparison input is not distinct: {name}"
            )
    learned_first = next(
        row for row in campaigns
        if row["policy_order"] == "learned,target"
    )
    learned_second = next(
        row for row in campaigns
        if row["policy_order"] == "target,learned"
    )
    first_runs = learned_first["policies"]["learned"]["measured_runs"]
    second_runs = learned_second["policies"]["learned"]["measured_runs"]

    def median_timing(runs, name):
        return statistics.median(
            row["timing"][name] for row in runs
        )

    position = {}
    for name in TIMING_METRICS:
        first_value = median_timing(first_runs, name)
        second_value = median_timing(second_runs, name)
        position[name] = {
            "learned_first_median": first_value,
            "learned_second_median": second_value,
            "relative_delta": (
                (first_value - second_value) / second_value
            ),
        }

    metric_comparison = {}
    worse = set()
    correlations = {}
    combined_runs = first_runs + second_runs
    for metric in PRIMARY_HOST_METRICS:
        first_value = statistics.median(
            row["metrics"][metric] for row in first_runs
        )
        second_value = statistics.median(
            row["metrics"][metric] for row in second_runs
        )
        relative = (
            math.inf
            if second_value <= 1e-12 and first_value > 1e-12
            else (
                0.0
                if second_value <= 1e-12
                else (first_value - second_value) / second_value
            )
        )
        is_worse = (
            first_value > second_value
            and first_value - second_value > 1e-12
            and relative >= THRESHOLDS["host_metric_worse_fraction"]
        )
        if is_worse:
            worse.add(metric)
        metric_comparison[metric] = {
            "learned_first_median": first_value,
            "learned_second_median": second_value,
            "absolute_difference": first_value - second_value,
            "relative_increase": relative,
            "worse_in_learned_first": is_worse,
        }
        correlations[metric] = {}
        host_values = [row["metrics"][metric] for row in combined_runs]
        for timing_name in (
            "e2e_s",
            "executor_proposal_forward_ms",
        ):
            timing_values = [
                row["timing"][timing_name] for row in combined_runs
            ]
            rho = spearman_rho(host_values, timing_values)
            correlations[metric][timing_name] = {
                "sample_count": len(host_values),
                "host_rank_variance": len(set(host_values)) > 1,
                "timing_rank_variance": len(set(timing_values)) > 1,
                "rho": rho,
            }

    e2e_correlated = {
        metric for metric, rows in correlations.items()
        if rows["e2e_s"]["rho"] is not None
        and rows["e2e_s"]["rho"] >= THRESHOLDS["spearman_rho_minimum"]
    }
    proposal_correlated = {
        metric for metric, rows in correlations.items()
        if rows["executor_proposal_forward_ms"]["rho"] is not None
        and rows["executor_proposal_forward_ms"]["rho"]
        >= THRESHOLDS["spearman_rho_minimum"]
    }
    classification, reasons = classify_host_comparison(
        learned_e2e_relative_delta=position["e2e_s"]["relative_delta"],
        worse_primary_metrics=worse,
        e2e_correlated_metrics=e2e_correlated,
        proposal_correlated_metrics=proposal_correlated,
    )
    expected_roles = {
        "target,learned": "learned_second",
        "learned,target": "learned_first",
    }
    for reference in (first_reference, second_reference):
        pure = PurePosixPath(reference["path"])
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("comparison reference path is invalid")
        if (
            reference["role"]
            != expected_roles[reference["policy_order"]]
        ):
            raise ValueError("comparison reference role is invalid")
    references = {
        reference["role"]: copy.deepcopy(reference)
        for reference in (first_reference, second_reference)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": classification,
        "classification_reasons": reasons,
        "campaign_artifacts": references,
        "learned_position_effect": position,
        "primary_metric_comparison": metric_comparison,
        "correlations": correlations,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "source_identity": copy.deepcopy(
            campaigns[0]["source_files"]
        ),
        "limitations": list(LIMITATIONS),
    }
```

Before accepting references, validate relative paths with `PurePosixPath`,
reject absolute paths and any `..` part, and require roles
`learned_first`/`learned_second` to match the validated policy orders.

- [ ] **Step 8: Add campaign and comparison CLI modes**

Implement exact CLI modes:

```bash
# Campaign-local artifact
python tools/autoregressive_draft_host_semantic_diagnostic.py \
  --timing-artifact artifacts/result.json \
  --gpu-telemetry-artifact artifacts/telemetry.json \
  --target-worker artifacts/workers/target-b4.json \
  --learned-worker artifacts/workers/learned-b4.json \
  --target-host-jsonl artifacts/host-semantic/target-host.jsonl \
  --learned-host-jsonl artifacts/host-semantic/learned-host.jsonl \
  --policy-order target,learned \
  --prime-each-policy \
  --repo-root source \
  --out artifacts/host-semantic.json

# Cross-campaign comparison
python tools/autoregressive_draft_host_semantic_diagnostic.py \
  --campaign-artifact \
    experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815/host-semantic.json \
  --comparison-artifact \
    experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815/host-semantic.json \
  --out \
    experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json
```

Use a mutually exclusive mode check in `main`: comparison mode requires both
`--campaign-artifact` and `--comparison-artifact`; campaign mode rejects both.
Use atomic JSON writes with a temporary sibling file and `Path.replace`.

- [ ] **Step 9: Run Task 3 GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k 'campaign or comparison or rank or spearman or classify or main'

python3 -m py_compile \
  tools/autoregressive_draft_host_semantic_diagnostic.py
```

Expected: all campaign, comparison, CLI, and classification tests pass.

---

### Task 4: Independent Filesystem-Bound Verifier

**Files:**
- Create:
  `tools/verify_autoregressive_draft_host_semantic_diagnostic.py`
- Modify:
  `tools/test_autoregressive_draft_host_semantic_diagnostic.py`

**Interfaces:**
- Produces:
  `verify_host_semantic_artifact(artifact_path: Path, repo_root: Path) -> dict`
- Produces:
  `verify_host_semantic_comparison(artifact_path: Path, repo_root: Path) -> dict`
- CLI:
  `--artifact`, `--repo-root`, optional `--receipt`

- [ ] **Step 1: Write campaign verifier tamper RED tests**

Append these bundle helpers:

```python
import hashlib


SOURCE_FIXTURE_PATHS = (
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_b4_timing_diagnostic.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_host_sampler.py",
    "tools/autoregressive_draft_host_semantic_diagnostic.py",
    "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
)


def _sha256_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_campaign_bundle_at(
    bundle,
    repo_root,
    *,
    policy_order,
    campaign_tag,
    learned_scale,
    learned_pressure_scale,
):
    module = _load_module()
    for relative_path in SOURCE_FIXTURE_PATHS:
        source_path = repo_root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            f"SOURCE = {relative_path!r}\n",
            encoding="utf-8",
        )
    bundle.mkdir(parents=True, exist_ok=True)
    timing = {
        **_timing_artifact(),
        "campaign_tag": campaign_tag,
    }
    gpu = {
        **_gpu_artifact(),
        "campaign_tag": campaign_tag,
    }
    target_worker = {
        **_campaign_worker("target"),
        "campaign_tag": campaign_tag,
    }
    learned_worker = {
        **_campaign_worker(
            "learned",
            learned_scale=learned_scale,
        ),
        "campaign_tag": campaign_tag,
    }
    target_samples = _campaign_samples(
        pressure_scale=1.0 + 0.01 * learned_scale
    )
    learned_samples = _campaign_samples(
        pressure_scale=learned_pressure_scale
    )
    input_payloads = {
        "timing_artifact": ("result.json", timing),
        "gpu_telemetry_artifact": ("telemetry.json", gpu),
        "target_worker": (
            "workers/target-b4.json",
            target_worker,
        ),
        "learned_worker": (
            "workers/learned-b4.json",
            learned_worker,
        ),
    }
    for _, (relative_path, payload) in input_payloads.items():
        _write_json(bundle / relative_path, payload)
    host_payloads = {
        "target_host_jsonl": (
            "host-semantic/target-host.jsonl",
            target_samples,
        ),
        "learned_host_jsonl": (
            "host-semantic/learned-host.jsonl",
            learned_samples,
        ),
    }
    for _, (relative_path, rows) in host_payloads.items():
        path = bundle / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                json.dumps(row, sort_keys=True) for row in rows
            ) + "\n",
            encoding="utf-8",
        )
    input_files = {
        name: {
            "path": relative_path,
            "sha256": _sha256_path(bundle / relative_path),
        }
        for name, (relative_path, _) in {
            **input_payloads,
            **host_payloads,
        }.items()
    }
    source_files = {
        relative_path: _sha256_path(repo_root / relative_path)
        for relative_path in SOURCE_FIXTURE_PATHS
    }
    artifact = module.build_host_semantic_artifact(
        timing_artifact=timing,
        gpu_telemetry_artifact=gpu,
        target_worker=target_worker,
        learned_worker=learned_worker,
        target_samples=target_samples,
        learned_samples=learned_samples,
        policy_order=policy_order,
        prime_each_policy=True,
        source_files=source_files,
        input_files=input_files,
    )
    artifact_path = bundle / "host-semantic.json"
    _write_json(artifact_path, artifact)
    return artifact_path


def _write_campaign_bundle(tmp_path):
    repo_root = tmp_path / "repo"
    artifact_path = _write_campaign_bundle_at(
        tmp_path / "bundle",
        repo_root,
        policy_order="target,learned",
        campaign_tag="r7",
        learned_scale=1.0,
        learned_pressure_scale=1.0,
    )
    return artifact_path, repo_root


def _tamper_campaign_bundle(
    artifact_path,
    repo_root,
    target,
):
    artifact = json.loads(
        artifact_path.read_text(encoding="utf-8")
    )
    if target == "source":
        relative_path = next(iter(artifact["source_files"]))
        (repo_root / relative_path).write_text(
            "SOURCE = 'tampered'\n",
            encoding="utf-8",
        )
    elif target == "worker":
        row = artifact["input_files"]["target_worker"]
        (artifact_path.parent / row["path"]).write_text(
            "{}\n",
            encoding="utf-8",
        )
    elif target == "host":
        row = artifact["input_files"]["learned_host_jsonl"]
        with (artifact_path.parent / row["path"]).open(
            "a",
            encoding="utf-8",
        ) as output:
            output.write("{}\n")
    elif target == "summary":
        artifact["policies"]["learned"]["measured_runs"][0][
            "metrics"
        ]["run_queue_mean"] += 1.0
        _write_json(artifact_path, artifact)
    else:
        raise AssertionError(target)


def _write_comparison_bundle(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    r7_path = _write_campaign_bundle_at(
        tmp_path / "r7",
        repo_root,
        policy_order="target,learned",
        campaign_tag="r7",
        learned_scale=1.0,
        learned_pressure_scale=1.0,
    )
    r8_path = _write_campaign_bundle_at(
        tmp_path / "r8",
        repo_root,
        policy_order="learned,target",
        campaign_tag="r8",
        learned_scale=1.2,
        learned_pressure_scale=2.0,
    )
    r7 = json.loads(r7_path.read_text(encoding="utf-8"))
    r8 = json.loads(r8_path.read_text(encoding="utf-8"))
    comparison = module.build_host_semantic_comparison(
        first_artifact=r7,
        second_artifact=r8,
        first_reference={
            "role": "learned_second",
            "path": "r7/host-semantic.json",
            "sha256": _sha256_path(r7_path),
            "policy_order": "target,learned",
        },
        second_reference={
            "role": "learned_first",
            "path": "r8/host-semantic.json",
            "sha256": _sha256_path(r8_path),
            "policy_order": "learned,target",
        },
    )
    comparison_path = tmp_path / "comparison.json"
    _write_json(comparison_path, comparison)
    return comparison_path, repo_root
```

Then append:

```python
def _load_verifier():
    assert VERIFIER_PATH.exists(), f"missing verifier: {VERIFIER_PATH}"
    diagnostic = _load_module()
    sys.modules[
        "autoregressive_draft_host_semantic_diagnostic"
    ] = diagnostic
    spec = importlib.util.spec_from_file_location(
        "verify_autoregressive_draft_host_semantic_test_module",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_recomputes_campaign_from_raw_inputs(tmp_path):
    artifact_path, repo_root = _write_campaign_bundle(tmp_path)
    receipt = _load_verifier().verify_host_semantic_artifact(
        artifact_path,
        repo_root,
    )
    assert receipt == {
        "status": "PASS",
        "schema_version": 1,
        "classification": "ALIGNED_CAMPAIGN",
        "source_files_verified": 6,
        "input_files_verified": 6,
        "policy_repeat_coverage": {"target": 8, "learned": 8},
    }


@pytest.mark.parametrize(
    "target,match",
    (
        ("source", "source hash mismatch"),
        ("worker", "input hash mismatch"),
        ("host", "input hash mismatch"),
        ("summary", "recomputation mismatch"),
    ),
)
def test_verifier_rejects_campaign_tampering(tmp_path, target, match):
    artifact_path, repo_root = _write_campaign_bundle(tmp_path)
    _tamper_campaign_bundle(artifact_path, repo_root, target)
    with pytest.raises(ValueError, match=match):
        _load_verifier().verify_host_semantic_artifact(
            artifact_path,
            repo_root,
        )
```

- [ ] **Step 2: Run campaign verifier tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k verifier_recomputes_campaign
```

Expected: failure because the verifier does not exist.

- [ ] **Step 3: Implement campaign verification**

Create:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_host_semantic_diagnostic import (
    build_host_semantic_artifact,
    build_host_semantic_comparison,
    parse_host_jsonl,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} is invalid")
    return value


def _resolve_relative(root: Path, value: str, *, name: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} path is invalid")
    return root / Path(*pure.parts)


def verify_host_semantic_artifact(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = _read_json(
        artifact_path,
        name="host semantic artifact",
    )
    if artifact.get("classification") != "ALIGNED_CAMPAIGN":
        raise ValueError("campaign classification is invalid")
    source_files_verified = 0
    for relative_path, expected_hash in artifact["source_files"].items():
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(f"source file is missing: {relative_path}")
        if _sha256(source_path) != expected_hash:
            raise ValueError(f"source hash mismatch: {relative_path}")
        source_files_verified += 1
    resolved = {}
    input_files_verified = 0
    for name, row in artifact["input_files"].items():
        path = _resolve_relative(
            artifact_path.parent,
            row["path"],
            name=name,
        )
        if not path.is_file():
            raise ValueError(f"input file is missing: {name}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        resolved[name] = path
        input_files_verified += 1
    expected = build_host_semantic_artifact(
        timing_artifact=_read_json(
            resolved["timing_artifact"],
            name="timing artifact",
        ),
        gpu_telemetry_artifact=_read_json(
            resolved["gpu_telemetry_artifact"],
            name="GPU telemetry artifact",
        ),
        target_worker=_read_json(
            resolved["target_worker"],
            name="target worker",
        ),
        learned_worker=_read_json(
            resolved["learned_worker"],
            name="learned worker",
        ),
        target_samples=parse_host_jsonl(
            resolved["target_host_jsonl"].read_text(encoding="utf-8")
        ),
        learned_samples=parse_host_jsonl(
            resolved["learned_host_jsonl"].read_text(encoding="utf-8")
        ),
        policy_order=artifact["policy_order"],
        prime_each_policy=artifact["prime_each_policy"],
        source_files=artifact["source_files"],
        input_files=artifact["input_files"],
    )
    if artifact != expected:
        raise ValueError("host semantic artifact recomputation mismatch")
    return {
        "status": "PASS",
        "schema_version": 1,
        "classification": artifact["classification"],
        "source_files_verified": source_files_verified,
        "input_files_verified": input_files_verified,
        "policy_repeat_coverage": {
            policy: len(artifact["policies"][policy]["measured_runs"])
            for policy in ("target", "learned")
        },
    }
```

- [ ] **Step 4: Write comparison verifier RED tests**

Append:

```python
def test_verifier_recomputes_cross_campaign_comparison(tmp_path):
    comparison_path, repo_root = _write_comparison_bundle(tmp_path)
    receipt = _load_verifier().verify_host_semantic_comparison(
        comparison_path,
        repo_root,
    )
    assert receipt["status"] == "PASS"
    assert receipt["schema_version"] == 1
    assert receipt["classification"] in {
        "HOST_PRESSURE_ASSOCIATED",
        "HOST_PRESSURE_NOT_SUPPORTED",
        "HOST_ALIGNMENT_INCONCLUSIVE",
    }
    assert receipt["campaign_artifacts_verified"] == 2


def test_comparison_verifier_rejects_tampered_campaign(tmp_path):
    comparison_path, repo_root = _write_comparison_bundle(tmp_path)
    comparison = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )
    reference = comparison["campaign_artifacts"]["learned_first"]
    campaign_path = comparison_path.parent / reference["path"]
    campaign_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="campaign artifact hash mismatch"):
        _load_verifier().verify_host_semantic_comparison(
            comparison_path,
            repo_root,
        )
```

- [ ] **Step 5: Implement recursive comparison verification and CLI**

Add:

```python
def verify_host_semantic_comparison(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    artifact = _read_json(
        artifact_path,
        name="host semantic comparison",
    )
    loaded = {}
    references = {}
    for role in ("learned_first", "learned_second"):
        row = artifact["campaign_artifacts"][role]
        path = _resolve_relative(
            artifact_path.parent,
            row["path"],
            name=role,
        )
        if _sha256(path) != row["sha256"]:
            raise ValueError(
                f"campaign artifact hash mismatch: {role}"
            )
        verify_host_semantic_artifact(path, repo_root)
        loaded[role] = _read_json(path, name=f"{role} campaign")
        references[role] = row
    expected = build_host_semantic_comparison(
        first_artifact=loaded["learned_first"],
        second_artifact=loaded["learned_second"],
        first_reference=references["learned_first"],
        second_reference=references["learned_second"],
    )
    if artifact != expected:
        raise ValueError("host comparison recomputation mismatch")
    return {
        "status": "PASS",
        "schema_version": 1,
        "classification": artifact["classification"],
        "campaign_artifacts_verified": 2,
        "source_files_verified_per_campaign": len(
            artifact["source_identity"]
        ),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--receipt")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    artifact_path = Path(args.artifact)
    artifact = _read_json(artifact_path, name="artifact")
    if artifact.get("classification") == "ALIGNED_CAMPAIGN":
        receipt = verify_host_semantic_artifact(
            artifact_path,
            Path(args.repo_root),
        )
    else:
        receipt = verify_host_semantic_comparison(
            artifact_path,
            Path(args.repo_root),
        )
    if args.receipt:
        _write_json_atomic(Path(args.receipt), receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run Task 4 GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  -k verifier

python3 -m py_compile \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py
```

Expected: campaign and comparison verifier tests pass.

---

### Task 5: Remote Runner Integration Without Measured-Path Changes

**Files:**
- Modify:
  `tools/test_autoregressive_draft_instability_telemetry.py`
- Modify:
  `tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh`

**Interfaces:**
- Adds `host-semantic/target-host.jsonl` and
  `host-semantic/learned-host.jsonl`.
- Adds `host-semantic.json`.
- Adds remote/local host verifier receipts.
- Leaves existing timing/GPU artifact commands byte-for-byte semantically
  unchanged.

- [ ] **Step 1: Write runner source-contract RED test**

Append to `tools/test_autoregressive_draft_instability_telemetry.py`:

```python
def test_remote_runner_owns_host_semantic_alignment_contract():
    script = RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "tools/autoregressive_draft_host_sampler.py",
        "tools/autoregressive_draft_host_semantic_diagnostic.py",
        "tools/verify_autoregressive_draft_host_semantic_diagnostic.py",
        "tools/test_autoregressive_draft_host_sampler.py",
        "tools/test_autoregressive_draft_host_semantic_diagnostic.py",
        "'${REMOTE_ARTIFACTS}/host-semantic'",
        '--interval-seconds 0.2',
        '"${artifacts}/host-semantic/${policy}-host.jsonl"',
        '"${artifacts}/host-semantic/${policy}-host.stderr.log"',
        '--target-host-jsonl "${artifacts}/host-semantic/target-host.jsonl"',
        '--learned-host-jsonl "${artifacts}/host-semantic/learned-host.jsonl"',
        '--out "${artifacts}/host-semantic.json"',
        'verify.host.remote.json',
        'verify.host.local.json',
        'verify-host-remote-exit-code.txt',
    ):
        assert expected in script

    prime_index = script.index('prime_policy "${policy}"')
    sampler_index = script.index('start_samplers "${policy}"')
    measured_index = script.index(
        'tools/autoregressive_draft_performance_worker.py',
        sampler_index,
    )
    assert prime_index < sampler_index < measured_index
    assert "torch.cuda.synchronize" not in script
```

Also assert the pre-existing timing and GPU telemetry command fragments are
still present exactly once.

- [ ] **Step 2: Run runner contract and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k host_semantic_alignment_contract
```

Expected: failure because host-semantic files and verifier calls are absent.

- [ ] **Step 3: Package and preflight the new files**

Add to `SOURCE_PATHS`:

```bash
tools/autoregressive_draft_host_sampler.py
tools/autoregressive_draft_host_semantic_diagnostic.py
tools/verify_autoregressive_draft_host_semantic_diagnostic.py
tools/test_autoregressive_draft_host_sampler.py
tools/test_autoregressive_draft_host_semantic_diagnostic.py
```

Add remote directory creation:

```bash
'${REMOTE_ARTIFACTS}/host-semantic'
```

Add the new files to remote `py_compile` and both new test files to remote
pytest. Do not remove any existing preflight target.

- [ ] **Step 4: Start the sampler only inside measured `start_samplers`**

After the GPU sampler block and before `vmstat`, add:

```bash
  "${python_executable}" \
    tools/autoregressive_draft_host_sampler.py \
      --interval-seconds 0.2 \
    >"${artifacts}/host-semantic/${policy}-host.jsonl" \
    2>"${artifacts}/host-semantic/${policy}-host.stderr.log" &
  sampler_pids+=("$!")
```

Do not modify `prime_policy`. Keep `stop_samplers` as the only cleanup path.

- [ ] **Step 5: Assemble the campaign-local host artifact**

After the existing `telemetry.json` assembler succeeds, add:

```bash
"${python_executable}" \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
    --timing-artifact "${artifacts}/result.json" \
    --gpu-telemetry-artifact "${artifacts}/telemetry.json" \
    --target-worker "${artifacts}/workers/target-b4.json" \
    --learned-worker "${artifacts}/workers/learned-b4.json" \
    --target-host-jsonl \
      "${artifacts}/host-semantic/target-host.jsonl" \
    --learned-host-jsonl \
      "${artifacts}/host-semantic/learned-host.jsonl" \
    --policy-order "${policy_order_csv}" \
    --prime-each-policy \
    --repo-root "${repo_root}" \
    --out "${artifacts}/host-semantic.json" \
    >"${artifacts}/host-semantic-assemble.log" 2>&1
```

Require `prime_each_policy == 1` before assembly; otherwise fail with exit code
`2`, because this diagnostic schema is only valid for primed campaigns.

- [ ] **Step 6: Add remote and local independent verification**

Add remote state:

```bash
host_verify_status=1
```

After existing telemetry verification succeeds:

```bash
if [[ "${telemetry_verify_status}" -eq 0 ]]; then
  PYTHONPATH="${remote_package_root}:${remote_source}" \
  "${remote_python}" \
    tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
      --artifact "${remote_artifacts}/host-semantic.json" \
      --repo-root "${remote_source}" \
      --receipt "${remote_artifacts}/verify.host.remote.json" \
      >"${remote_artifacts}/verify.host.remote.log" 2>&1
  host_verify_status=$?
fi
printf '%s\n' "${host_verify_status}" \
  >"${remote_artifacts}/verify-host-remote-exit-code.txt"
```

Return `host_verify_status` after timing and telemetry status checks.

After download and the existing local telemetry verifier:

```bash
PYTHONPATH="${REPO_ROOT}" \
python3 \
  "${REPO_ROOT}/tools/verify_autoregressive_draft_host_semantic_diagnostic.py" \
    --artifact "${LOCAL_RUN}/host-semantic.json" \
    --repo-root "${REPO_ROOT}" \
    --receipt "${LOCAL_RUN}/verify.host.local.json"
```

Add host assembler and verifier logs to the failure-tail list. Leave final
manifest generation unchanged so it automatically covers the new files.

- [ ] **Step 7: Run focused runner GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k runner

bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

Expected: all runner source contracts pass and Bash syntax is valid.

---

### Task 6: Complete Local Regression and Self-Review Gate

**Files:**
- Verify all files from Tasks 1-5.
- Verify:
  `docs/superpowers/specs/2026-08-15-autoregressive-draft-host-semantic-alignment-design.md`
- Verify:
  `docs/superpowers/plans/2026-08-15-autoregressive-draft-host-semantic-alignment.md`

**Interfaces:**
- Proves no existing diagnostic contract regressed.
- Proves the new evidence chain is locally deterministic and independently
  recomputable.

- [ ] **Step 1: Run the complete focused pytest gate**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py
```

Expected: all tests pass.

- [ ] **Step 2: Run compilation and shell syntax**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/autoregressive_draft_executor.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py

bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

Expected: compilation and shell parsing succeed.

- [ ] **Step 3: Scan for measured-path synchronization and incomplete markers**

Run:

```bash
! rg -n "torch\\.cuda\\.synchronize" \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh

python3 - <<'PY'
from pathlib import Path

needles = ("T" + "BD", "TO" + "DO", "implement " + "later")
paths = (
    Path("tools/autoregressive_draft_host_sampler.py"),
    Path("tools/autoregressive_draft_host_semantic_diagnostic.py"),
    Path("tools/verify_autoregressive_draft_host_semantic_diagnostic.py"),
    Path("docs/superpowers/specs/2026-08-15-autoregressive-draft-host-semantic-alignment-design.md"),
    Path("docs/superpowers/plans/2026-08-15-autoregressive-draft-host-semantic-alignment.md"),
)
matches = [
    f"{path}:{needle}"
    for path in paths
    for needle in needles
    if needle in path.read_text(encoding="utf-8")
]
if matches:
    raise SystemExit("\n".join(matches))
PY
```

Expected: both commands return success because no matches exist.

- [ ] **Step 4: Run scoped whitespace validation**

Run:

```bash
git diff --check -- \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  docs/superpowers/specs/2026-08-15-autoregressive-draft-host-semantic-alignment-design.md \
  docs/superpowers/plans/2026-08-15-autoregressive-draft-host-semantic-alignment.md
```

Expected: no whitespace errors.

---

### Task 7: r7/r8 Remote Authority, Comparison, and Handoff

**Files:**
- Create:
  `experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815/`
- Create:
  `experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815/`
- Modify:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces two source-bound, independently verified campaign bundles.
- Produces one independently verified cross-order comparison.
- Produces the only allowed final host classification.

- [x] **Step 1: Run read-only remote resource preflight**

Run:

```bash
ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=20 \
  -o ControlMaster=no \
  -o ControlPath=/tmp/ssh-sitian-10.232.195.203 \
  -o GSSAPIAuthentication=yes \
  sitian@10.232.195.203 \
  'set -euo pipefail
   df -h /dev/shm
   test -d /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/target-qwen3-1.7b
   test -d /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/draft
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
     --format=csv,noheader
   ps -p 703088 -o pid=,user=,comm=,args='
```

Expected:

```text
/dev/shm has enough free space for two campaign bundles
both model directories exist
GPU 3,4,6 have no conflicting compute process
GPU 7 PID 703088 python3 remains present
```

Do not kill any process.

- [x] **Step 2: Run r7 target-then-learned**

Run:

```bash
bash tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  --remote-host sitian@10.232.195.203 \
  --ssh-control-path /tmp/ssh-sitian-10.232.195.203 \
  --remote-python /data00/home/sitian/miniconda3/envs/py311/bin/python \
  --remote-base /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815 \
  --target-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/target-qwen3-1.7b \
  --draft-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/draft \
  --gpu-indices 3,4,6,7 \
  --policy-order target,learned \
  --prime-each-policy \
  --run-tag \
    tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815 \
  --local-run \
    experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815
```

Expected: runner exits zero, remote/local timing, GPU telemetry, and host
verifier receipts all report `PASS`, and manifest validation passes.

- [x] **Step 3: Re-run preflight and run r8 learned-then-target**

Repeat Step 1, then run:

```bash
bash tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  --remote-host sitian@10.232.195.203 \
  --ssh-control-path /tmp/ssh-sitian-10.232.195.203 \
  --remote-python /data00/home/sitian/miniconda3/envs/py311/bin/python \
  --remote-base /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815 \
  --target-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/target-qwen3-1.7b \
  --draft-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/draft \
  --gpu-indices 3,4,6,7 \
  --policy-order learned,target \
  --prime-each-policy \
  --run-tag \
    tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815 \
  --local-run \
    experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815
```

Expected: same PASS conditions as r7.

- [x] **Step 4: Build the source-identical cross-order comparison**

Run:

```bash
python3 \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
    --campaign-artifact \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815/host-semantic.json \
    --comparison-artifact \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815/host-semantic.json \
    --out \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json

python3 \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
    --artifact \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json \
    --repo-root . \
    --receipt \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.verify.json
```

Expected: comparison verifier reports `PASS` and exactly one classification:

```text
HOST_PRESSURE_ASSOCIATED
HOST_PRESSURE_NOT_SUPPORTED
HOST_ALIGNMENT_INCONCLUSIVE
```

- [x] **Step 5: Build and verify a dedicated comparison manifest**

Keep both campaign manifests unchanged. Bind the comparison, its receipt, and
both referenced campaign artifacts in a separate manifest:

```bash
(
  cd experiments/autoregressive_draft
  shasum -a 256 \
    tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json \
    tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.verify.json \
    tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815/host-semantic.json \
    tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815/host-semantic.json \
    > tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.manifest.sha256
  shasum -a 256 -c \
    tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.manifest.sha256
)
```

Expected: every file verifies.

- [x] **Step 6: Write campaign README evidence**

Create one `README.md` in each bundle. Record:

```text
source hashes
policy order
prime-each-policy state
exact parity
sample coverage and maximum sample gap
remote/local verifier receipts
learned target E2E, TPOT, proposal-forward medians
primary host metric medians
Spearman results
final comparison classification
causal and promotion limitations
```

Do not describe association as causality. Do not claim Phase 1 is achieved.

- [x] **Step 7: Update audit and handoff**

Append the exact r7/r8 artifact paths, verifier commands/results, manifest
entry counts, source-identity evidence, learned position delta, host metric
counts, correlations, and final classification to:

```text
docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md
AGENT_HANDOFF_STATE.md
```

Set the next action by classification:

```text
HOST_PRESSURE_ASSOCIATED
  -> isolate responsible host resource/process class

HOST_PRESSURE_NOT_SUPPORTED
  -> run primed learned/learned process-boundary A/A

HOST_ALIGNMENT_INCONCLUSIVE
  -> repair evidence gap and rerun r7/r8
```

- [x] **Step 8: Run final authority verification**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py

python3 \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
    --artifact \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-target-learned-gpu3467-r7-20260815/host-semantic.json \
    --repo-root .

python3 \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
    --artifact \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-primed-learned-target-gpu3467-r8-20260815/host-semantic.json \
    --repo-root .

python3 \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
    --artifact \
      experiments/autoregressive_draft/tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json \
    --repo-root .

git diff --check -- \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  docs/superpowers/specs/2026-08-15-autoregressive-draft-host-semantic-alignment-design.md \
  docs/superpowers/plans/2026-08-15-autoregressive-draft-host-semantic-alignment.md \
  docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md \
  AGENT_HANDOFF_STATE.md
```

Expected:

```text
all focused tests pass
r7 campaign verifier PASS
r8 campaign verifier PASS
comparison verifier PASS
r7/r8 manifests PASS
scoped diff check PASS
PHASE_1 remains NOT_ACHIEVED
PROMOTION remains NOT_PROMOTABLE
```

Final result:

```text
focused tests:
  98 passed in 0.89s

r7 campaign verifier:
  PASS / ALIGNED_CAMPAIGN
  6 input files verified
  6 source files verified
  8 target repeats and 8 learned repeats covered

r8 campaign verifier:
  PASS / ALIGNED_CAMPAIGN
  6 input files verified
  6 source files verified
  8 target repeats and 8 learned repeats covered

comparison verifier:
  PASS / HOST_ALIGNMENT_INCONCLUSIVE
  2 campaign artifacts verified
  6 source files verified per campaign

manifests:
  r7:         PASS / 72 entries
  r8:         PASS / 54 entries
  comparison: PASS / 4 entries

runner bash syntax:
  PASS

Python compile:
  PASS

scoped git diff --check:
  PASS

PHASE_1:
  NOT_ACHIEVED

PROMOTION:
  NOT_PROMOTABLE
```
