from __future__ import annotations

import pytest

from tools.cross_engine_k8_resources import (
    ResourceSampler,
    reduce_resource_samples,
)


class SequenceReader:
    def __init__(self, rows):
        self._rows = list(rows)
        self._index = 0

    def __call__(self, *_args):
        row = self._rows[min(self._index, len(self._rows) - 1)]
        self._index += 1
        return dict(row)


def test_sampler_records_external_gpu_and_process_metrics():
    sampler = ResourceSampler(
        pid=123,
        gpu_uuid="GPU-a",
        interval_s=0.05,
        max_samples=3,
        nvml_reader=lambda _uuid, _pid: {
            "gpu_memory_bytes": 4_000,
            "gpu_utilization_percent": 25,
        },
        process_reader=lambda _pid: {
            "rss_bytes": 8_000,
            "cpu_time_ns": 9_000,
        },
        clock=lambda: 1_000,
    )

    row = sampler.sample()

    assert row["timestamp_ns"] == 1_000
    assert row["pid"] == 123
    assert row["gpu_uuid"] == "GPU-a"
    assert row["gpu_memory_bytes"] == 4_000
    assert row["rss_bytes"] == 8_000


def test_sampler_keeps_bounded_rows_and_exact_peaks():
    gpu_reader = SequenceReader([
        {"gpu_memory_bytes": value, "gpu_utilization_percent": value // 100}
        for value in (1_000, 9_000, 4_000, 3_000)
    ])
    process_reader = SequenceReader([
        {"rss_bytes": value, "cpu_time_ns": value * 2}
        for value in (2_000, 5_000, 8_000, 6_000)
    ])
    ticks = iter((1, 2, 3, 4))
    sampler = ResourceSampler(
        pid=123,
        gpu_uuid="GPU-a",
        interval_s=0.05,
        max_samples=3,
        nvml_reader=gpu_reader,
        process_reader=process_reader,
        clock=lambda: next(ticks),
    )

    for _ in range(4):
        sampler.sample()
    final = sampler.finalize()

    assert final["samples_retained"] == 3
    assert final["samples_observed"] == 4
    assert final["peak_gpu_memory_bytes"] == 9_000
    assert final["peak_rss_bytes"] == 8_000
    assert [row["timestamp_ns"] for row in final["samples"]] == [2, 3, 4]


def test_sampler_marks_unavailable_metrics_not_exposed():
    sampler = ResourceSampler(
        pid=123,
        gpu_uuid="GPU-a",
        interval_s=0.05,
        max_samples=3,
        nvml_reader=lambda _uuid, _pid: {},
        process_reader=lambda _pid: {},
        clock=lambda: 1,
    )

    row = sampler.sample()
    final = sampler.finalize()

    assert row["gpu_memory_bytes"] == "NOT_EXPOSED"
    assert row["rss_bytes"] == "NOT_EXPOSED"
    assert final["peak_gpu_memory_bytes"] == "NOT_EXPOSED"


@pytest.mark.parametrize(
    ("interval_s", "max_samples"),
    ((0, 1), (-1, 1), (0.1, 0), (0.1, True)),
)
def test_sampler_rejects_invalid_bounds(interval_s, max_samples):
    with pytest.raises(ValueError):
        ResourceSampler(
            pid=123,
            gpu_uuid="GPU-a",
            interval_s=interval_s,
            max_samples=max_samples,
            nvml_reader=lambda *_args: {},
            process_reader=lambda *_args: {},
            clock=lambda: 1,
        )


def test_reduce_resource_samples_rejects_mixed_identity():
    rows = [
        {
            "pid": 1,
            "gpu_uuid": "GPU-a",
            "gpu_memory_bytes": 10,
            "rss_bytes": 20,
            "cpu_time_ns": 30,
            "gpu_utilization_percent": 40,
        },
        {
            "pid": 2,
            "gpu_uuid": "GPU-a",
            "gpu_memory_bytes": 11,
            "rss_bytes": 21,
            "cpu_time_ns": 31,
            "gpu_utilization_percent": 41,
        },
    ]

    with pytest.raises(ValueError, match="identity"):
        reduce_resource_samples(rows)
