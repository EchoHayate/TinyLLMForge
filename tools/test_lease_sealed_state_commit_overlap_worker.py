from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


from tools.lease_sealed_state_commit_overlap_worker import (
    OverlapBuffers,
    _runtime_capability_row,
    _run_candidate,
    _tensor_digest,
    build_argument_parser,
    build_workload_schedule,
)


class FakeCuda:
    def __init__(self):
        self.streams = []
        self.events = []

    def Stream(self, *, device):
        value = ("stream", device, len(self.streams))
        self.streams.append(value)
        return value

    def Event(self, *, enable_timing):
        value = ("event", enable_timing, len(self.events))
        self.events.append(value)
        return value


class FakeTorch:
    bfloat16 = "bfloat16"
    float32 = "float32"
    uint8 = "uint8"

    def __init__(self):
        self.cuda = FakeCuda()
        self.allocations = []

    def empty(self, shape, *, dtype, device):
        value = {"shape": shape, "dtype": dtype, "device": device}
        self.allocations.append(value)
        return value


class FakeNumpy:
    def tobytes(self):
        return b"raw-tensor-bytes"


class FakeTensor:
    def __init__(self):
        self.view_dtype = None

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, dtype):
        self.view_dtype = dtype
        return self

    def cpu(self):
        return self

    def numpy(self):
        return FakeNumpy()


def test_schedule_freezes_shapes_warmups_pairs_and_abba_order():
    schedule = build_workload_schedule()

    assert [row["active_tokens"] for row in schedule] == [1, 4, 8]
    assert all(len(row["warmups"]) == 2 for row in schedule)
    assert all(len(row["measurements"]) == 15 for row in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "candidate",
    )
    assert schedule[0]["measurements"][1]["arm_order"] == (
        "candidate",
        "baseline",
    )


def test_buffers_preallocate_streams_events_and_real_shapes():
    torch = FakeTorch()
    buffers = OverlapBuffers.create(torch, "cuda:0", active_tokens=8)

    assert len(torch.cuda.streams) == 2
    assert len(torch.cuda.events) >= 5
    assert buffers.local_result["shape"] == (8, 5120)
    assert buffers.side_effect_payload["shape"] == (8, 271360 // 2)
    assert [
        row["dtype"] for row in torch.allocations[:3]
    ] == ["float32", "float32", "float32"]
    assert [
        row["dtype"] for row in torch.allocations[3:]
    ] == ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"]


def test_candidate_timed_path_has_no_sync_item_allocation_or_construction():
    source = inspect.getsource(_run_candidate)

    for forbidden in (
        "torch.cuda.synchronize",
        ".item(",
        "torch.empty",
        "torch.zeros",
        "torch.cuda.Stream",
        "torch.cuda.Event",
        "LeaseSealedCollectiveSideEffect(",
    ):
        assert forbidden not in source


def test_worker_records_peak_reserved_memory_for_gate_evidence():
    source = inspect.getsource(
        __import__(
            "tools.lease_sealed_state_commit_overlap_worker",
            fromlist=["run_worker"],
        ).run_worker
    )

    assert "torch.cuda.max_memory_reserved(device)" in source
    assert '"peak_reserved_delta_bytes"' in source
    assert (
        'row["peak_reserved_delta_bytes"] for row in memory_rows'
        in source
    )


def test_tensor_digest_views_raw_bytes_with_torch_uint8():
    tensor = FakeTensor()

    digest = _tensor_digest(tensor, FakeTorch)

    assert tensor.view_dtype == "uint8"
    assert digest == hashlib.sha256(b"raw-tensor-bytes").hexdigest()


def test_cli_requires_attempt_source_rank_and_output_identity():
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    script = Path(__file__).with_name(
        "lease_sealed_state_commit_overlap_worker.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_capability_records_driver_cuda_nccl_and_device_identity(
    monkeypatch,
):
    properties = SimpleNamespace(
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-expected",
        major=8,
        minor=0,
    )
    cuda = SimpleNamespace(
        get_device_properties=lambda _device: properties,
        nccl=SimpleNamespace(version=lambda: (2, 21, 5)),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.8"),
        __version__="2.8.0",
    )
    dist = SimpleNamespace(is_nccl_available=lambda: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="550.54.15\n",
            stderr="",
        ),
    )

    row = _runtime_capability_row(2, "cuda:2", torch, dist)

    assert row["rank"] == 2
    assert row["device_uuid"] == "GPU-expected"
    assert row["driver_version"] == "550.54.15"
    assert row["cuda_version"] == "12.8"
    assert row["nccl_version"] == "(2, 21, 5)"
    assert row["nccl_available"] is True
    assert isinstance(row["hostname"], str) and row["hostname"]
    assert isinstance(row["python_version"], str) and row["python_version"]


def test_runtime_capability_uses_nvml_uuid_when_properties_omit_uuid(
    monkeypatch,
):
    properties = SimpleNamespace(
        name="NVIDIA A100 80GB PCIe",
        major=8,
        minor=0,
    )
    uuids = [f"GPU-physical-{index}" for index in range(8)]
    cuda = SimpleNamespace(
        get_device_properties=lambda _device: properties,
        _get_nvml_device_index=lambda _device: 6,
        _raw_device_uuid_nvml=lambda: uuids,
        nccl=SimpleNamespace(version=lambda: (2, 20, 5)),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.1"),
        __version__="2.4.1+cu121",
    )
    dist = SimpleNamespace(is_nccl_available=lambda: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="535.261.03\n",
            stderr="",
        ),
    )

    row = _runtime_capability_row(2, "cuda:2", torch, dist)

    assert row["device_uuid"] == "GPU-physical-6"
