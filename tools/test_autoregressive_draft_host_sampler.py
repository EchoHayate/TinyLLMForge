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
