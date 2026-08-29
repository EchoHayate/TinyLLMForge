from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence


RESOURCE_SCHEMA_VERSION = "cross-engine-k8.resources.v1"
NOT_EXPOSED = "NOT_EXPOSED"
_METRICS = (
    "gpu_memory_bytes",
    "gpu_utilization_percent",
    "rss_bytes",
    "cpu_time_ns",
)


def _metric(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return NOT_EXPOSED
    return value


class ResourceSampler:
    def __init__(
        self,
        *,
        pid: int,
        gpu_uuid: str,
        interval_s: float,
        max_samples: int,
        nvml_reader: Callable[[str, int], Mapping],
        process_reader: Callable[[int], Mapping],
        clock: Callable[[], int] = time.monotonic_ns,
    ):
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not isinstance(gpu_uuid, str) or not gpu_uuid:
            raise ValueError("gpu_uuid must be a non-empty string")
        if (
            isinstance(interval_s, bool)
            or not isinstance(interval_s, (int, float))
            or not math.isfinite(float(interval_s))
            or interval_s <= 0
        ):
            raise ValueError("interval_s must be positive")
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        self.pid = pid
        self.gpu_uuid = gpu_uuid
        self.interval_s = float(interval_s)
        self._nvml_reader = nvml_reader
        self._process_reader = process_reader
        self._clock = clock
        self._samples = deque(maxlen=max_samples)
        self._observed = 0
        self._peaks = {name: None for name in _METRICS}

    def sample(self) -> dict:
        gpu = dict(self._nvml_reader(self.gpu_uuid, self.pid))
        process = dict(self._process_reader(self.pid))
        row = {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "timestamp_ns": self._clock(),
            "pid": self.pid,
            "gpu_uuid": self.gpu_uuid,
            "gpu_memory_bytes": _metric(gpu.get("gpu_memory_bytes")),
            "gpu_utilization_percent": _metric(
                gpu.get("gpu_utilization_percent")
            ),
            "rss_bytes": _metric(process.get("rss_bytes")),
            "cpu_time_ns": _metric(process.get("cpu_time_ns")),
        }
        self._observed += 1
        self._samples.append(row)
        for name in _METRICS:
            value = row[name]
            if value == NOT_EXPOSED:
                continue
            current = self._peaks[name]
            if current is None or value > current:
                self._peaks[name] = value
        return row

    def finalize(self) -> dict:
        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "pid": self.pid,
            "gpu_uuid": self.gpu_uuid,
            "interval_s": self.interval_s,
            "samples_observed": self._observed,
            "samples_retained": len(self._samples),
            "samples": list(self._samples),
            **{
                f"peak_{name}": (
                    NOT_EXPOSED
                    if value is None
                    else value
                )
                for name, value in self._peaks.items()
            },
        }


def reduce_resource_samples(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("resource samples cannot be empty")
    identities = {
        (row.get("pid"), row.get("gpu_uuid"))
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("resource sample identity changed")
    pid, gpu_uuid = next(iter(identities))
    result = {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "pid": pid,
        "gpu_uuid": gpu_uuid,
        "samples_observed": len(rows),
    }
    for name in _METRICS:
        exposed = [
            row.get(name)
            for row in rows
            if row.get(name) != NOT_EXPOSED
            and isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
        ]
        result[f"peak_{name}"] = max(exposed) if exposed else NOT_EXPOSED
    return result


def read_process_metrics(pid: int) -> dict:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    rss_kib = None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            rss_kib = int(line.split()[1])
            break
    _, separator, suffix = stat.rpartition(")")
    fields = suffix.split()
    if not separator or len(fields) < 13:
        raise ValueError("process stat is invalid")
    ticks = int(fields[11]) + int(fields[12])
    ticks_per_second = int(subprocess.check_output(
        ["getconf", "CLK_TCK"],
        text=True,
    ).strip())
    return {
        "rss_bytes": (
            NOT_EXPOSED if rss_kib is None else rss_kib * 1024
        ),
        "cpu_time_ns": int(ticks * 1e9 / ticks_per_second),
    }


def read_nvidia_smi_metrics(gpu_uuid: str, pid: int) -> dict:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu.returncode != 0 or apps.returncode != 0:
        return {}
    utilization = NOT_EXPOSED
    for line in gpu.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 1)]
        if len(fields) == 2 and fields[0] == gpu_uuid:
            utilization = int(fields[1])
            break
    memory_mib = None
    for line in apps.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 2)]
        if (
            len(fields) == 3
            and int(fields[0]) == pid
            and fields[1] == gpu_uuid
        ):
            memory_mib = int(fields[2])
            break
    return {
        "gpu_memory_bytes": (
            NOT_EXPOSED
            if memory_mib is None
            else memory_mib * 1024**2
        ),
        "gpu_utilization_percent": utilization,
    }


class ProcessResourceSession:
    def __init__(
        self,
        *,
        pid: int,
        gpu_uuid: str,
        interval_s: float = 0.05,
        max_samples: int = 4_096,
    ):
        self._sampler = ResourceSampler(
            pid=pid,
            gpu_uuid=gpu_uuid,
            interval_s=interval_s,
            max_samples=max_samples,
            nvml_reader=read_nvidia_smi_metrics,
            process_reader=read_process_metrics,
        )
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sampler.sample()
            except (FileNotFoundError, ProcessLookupError) as error:
                self._error = error
                break
            self._stop.wait(self._sampler.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource session already started")
        self._thread = threading.Thread(
            target=self._run,
            name="cross-engine-k8-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict:
        if self._thread is None:
            raise RuntimeError("resource session was not started")
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("resource sampler did not stop")
        result = self._sampler.finalize()
        if self._error is not None and result["samples_observed"] == 0:
            result["sampling_error"] = type(self._error).__name__
        return result
