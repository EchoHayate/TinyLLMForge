from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


from tools.lease_sealed_state_commit_overlap_worker import (
    OverlapBuffers,
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
