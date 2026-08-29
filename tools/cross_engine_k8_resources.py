from __future__ import annotations

from collections import deque
import math
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
