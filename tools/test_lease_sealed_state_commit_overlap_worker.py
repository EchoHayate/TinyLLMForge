from __future__ import annotations

import hashlib
import inspect
import json
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
    _expected_reduction_value,
    _merge_rank_artifacts,
    _run_completion_owned,
    _run_event_only_diagnostic,
    _runtime_capability_row,
    _tensor_digest,
    build_argument_parser,
    build_workload_schedule,
    run_worker,
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
    assert all(len(row["diagnostics"]) == 15 for row in schedule)
    assert all(len(row["warmups"]) == 2 for row in schedule)
    assert all(len(row["measurements"]) == 15 for row in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "completion_owned",
    )
    assert schedule[0]["measurements"][1]["arm_order"] == (
        "completion_owned",
        "baseline",
    )


def test_worker_parser_requires_stage01_protocol():
    parser = build_argument_parser()
    common = [
        "--attempt",
        "attempt",
        "--source-revision",
        "a" * 40,
        "--source-tree-sha256",
        "b" * 64,
        "--output-dir",
        "/data00/home/sitian/output",
        "--rank",
        "0",
        "--world-size",
        "4",
        "--dist-port",
        "29741",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    parsed = parser.parse_args([
        "--protocol",
        "completion-owned-stage01",
        *common,
    ])
    assert parsed.protocol == "completion-owned-stage01"


def test_buffers_preallocate_streams_events_and_real_shapes():
    torch = FakeTorch()
    buffers = OverlapBuffers.create(torch, "cuda:0", active_tokens=8)

    assert len(torch.cuda.streams) == 2
    assert len(torch.cuda.events) >= 5
    assert buffers.collective_visible_event in torch.cuda.events
    assert not hasattr(buffers, "consumer_ready_event")
    assert buffers.local_result["shape"] == (8, 5120)
    assert buffers.expected_result["shape"] == (8, 5120)
    assert buffers.expected_output["shape"] == (8, 5120)
    assert buffers.diagnostic_result["shape"] == (8, 5120)
    assert buffers.event_only_snapshot["shape"] == (8, 5120)
    assert buffers.diagnostic_output["shape"] == (8, 5120)
    assert buffers.diagnostic_shadow["shape"] == (8, 271360 // 2)
    assert buffers.side_effect_payload["shape"] == (8, 271360 // 2)
    assert [
        row["dtype"] for row in torch.allocations[:6]
    ] == ["float32"] * 6
    assert [
        row["dtype"] for row in torch.allocations[6:]
    ] == ["bfloat16"] * 8


def test_completion_owned_timed_path_has_no_sync_item_or_allocation():
    source = inspect.getsource(_run_completion_owned)

    for forbidden in (
        "torch.cuda.synchronize",
        ".item(",
        ".synchronize(",
        "torch.empty",
        "torch.zeros",
        "torch.cuda.Stream",
        "torch.cuda.Event",
        "LeaseSealedCollectiveSideEffect(",
    ):
        assert forbidden not in source


def test_event_only_arm_is_diagnostic_only():
    diagnostic_source = inspect.getsource(_run_event_only_diagnostic)
    assert "collective_work.wait()" not in diagnostic_source
    formal_source = inspect.getsource(run_worker).split(
        'for pair in workload["measurements"]:', 1
    )[1]
    assert "_run_event_only_diagnostic(" not in formal_source


def test_expected_reduction_value_is_independent_and_exact():
    assert _expected_reduction_value(0) == 6.0
    assert _expected_reduction_value(1) == 10.0
    assert _expected_reduction_value(96) == 390.0
    assert _expected_reduction_value(97) == 6.0


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


def test_rank_artifact_merge_flattens_lifecycle_shape_rows(tmp_path):
    for rank in range(4):
        (tmp_path / f"diagnostic_rows.rank-{rank}.jsonl").write_text("")
        (tmp_path / f"measurement_rows.rank-{rank}.jsonl").write_text("")
        (tmp_path / f"memory.rank-{rank}.json").write_text(
            json.dumps({"rank": rank})
        )
        (tmp_path / f"lifecycle.rank-{rank}.json").write_text(
            json.dumps({
                "rank": rank,
                "classification": "PASS",
                "shape_rows": [
                    {
                        "rank": rank,
                        "active_tokens": active_tokens,
                        "commit_identity_match": True,
                    }
                    for active_tokens in (1, 4, 8)
                ],
            })
        )
        (tmp_path / f"cleanup.rank-{rank}.json").write_text(
            json.dumps({
                "rank": rank,
                "process_group_destroyed": True,
                "streams_released": True,
                "events_released": True,
                "timed_out": False,
            })
        )
        (tmp_path / f"capability.rank-{rank}.json").write_text(
            json.dumps({"rank": rank})
        )

    _merge_rank_artifacts(tmp_path)

    lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
    assert [
        (row["rank"], row["active_tokens"])
        for row in lifecycle["rank_rows"]
    ] == [
        (rank, active_tokens)
        for rank in range(4)
        for active_tokens in (1, 4, 8)
    ]
