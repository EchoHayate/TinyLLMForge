from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tinyvllm"
    / "engine"
    / "tp4_peer_reduction.py"
)


class FakeDevice:
    def __init__(self, device_type, index):
        self.type = device_type
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, FakeDevice)
            and self.type == other.type
            and self.index == other.index
        )

    def __repr__(self):
        return f"{self.type}:{self.index}"


class FakeTensor:
    def __init__(self, shape, *, dtype, device):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.copy_calls = []
        self.value = 0

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise AssertionError(f"unexpected tensor index: {index!r}")
        layer_index, slot_index, token_slice = index
        assert isinstance(layer_index, int)
        assert isinstance(slot_index, int)
        assert isinstance(token_slice, slice)
        token_count = token_slice.stop
        return FakeTensor(
            (token_count, self.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )

    def copy_(self, other):
        self.copy_calls.append(other)
        return self

    def item(self):
        return self.value


class FakeTorch:
    float32 = "torch.float32"
    bfloat16 = "torch.bfloat16"
    uint64 = "torch.uint64"
    int32 = "torch.int32"

    def __init__(self):
        self.allocations = []
        self.cuda = SimpleNamespace(
            current_stream=lambda device=None: SimpleNamespace(
                cuda_stream=1234,
            ),
        )

    def empty(self, shape, *, dtype, device):
        tensor = FakeTensor(shape, dtype=dtype, device=device)
        self.allocations.append(tensor)
        return tensor

    def zeros(self, shape, *, dtype, device):
        tensor = FakeTensor(shape, dtype=dtype, device=device)
        self.allocations.append(tensor)
        return tensor

    def inference_mode(self):
        return nullcontext()


class FakeDistributed:
    def __init__(self):
        self.payloads = None

    def all_gather_object(self, output, payload):
        self.payloads = [
            {
                "rank": rank,
                "slot_handle": bytes([rank]) * 64,
                "flag_handle": bytes([rank + 4]) * 64,
            }
            for rank in range(4)
        ]
        output[:] = self.payloads


class FakeExtension:
    def __init__(self):
        self.calls = []
        self.reduce_status = 0

    def export_ipc_handle(self, tensor):
        self.calls.append(("export_ipc_handle", tensor.shape))
        return bytes([len(self.calls)]) * 64

    def open_mapping(
        self,
        slot_handle,
        flag_handle,
        slot_shape,
        flag_shape,
        peer_rank,
    ):
        self.calls.append(("open_mapping", peer_rank))
        return peer_rank

    def publish(
        self,
        local_slot,
        local_flags,
        layer_index,
        slot_index,
        generation,
        stream,
    ):
        self.calls.append(
            (
                "publish",
                layer_index,
                slot_index,
                generation,
                stream,
            )
        )

    def reduce_add_residual(
        self,
        peer_mappings,
        local_slot,
        local_flags,
        residual,
        output,
        status,
        rank,
        layer_index,
        slot_index,
        generation,
        active_tokens,
        hidden_size,
        timeout_clocks,
        stream,
    ):
        self.calls.append(
            (
                "reduce_add_residual",
                tuple(peer_mappings),
                status.shape,
                rank,
                layer_index,
                slot_index,
                generation,
                active_tokens,
                hidden_size,
                timeout_clocks,
                stream,
            )
        )
        status.value = self.reduce_status
        return 0

    def close_mapping(self, mapping):
        self.calls.append(("close_mapping", mapping))

    def release_owned(self):
        self.calls.append(("release_owned",))


def _load_runtime():
    fake_torch = FakeTorch()
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        spec = importlib.util.spec_from_file_location(
            "tp4_peer_reduction_under_test",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch
    return module, fake_torch


def _ready_group():
    module, fake_torch = _load_runtime()
    extension = FakeExtension()
    group = module.TP4PeerReductionGroup.create(
        rank=0,
        world_size=4,
        device=FakeDevice("cuda", 0),
        layer_count=64,
        max_active_tokens=8,
        hidden_size=5120,
        extension=extension,
        distributed=FakeDistributed(),
        torch_module=fake_torch,
    )
    return module, fake_torch, extension, group


def _fake_fp32(shape):
    return FakeTensor(
        shape,
        dtype="torch.float32",
        device=FakeDevice("cuda", 0),
    )


def _fake_bf16(shape):
    return FakeTensor(
        shape,
        dtype="torch.bfloat16",
        device=FakeDevice("cuda", 0),
    )


def test_create_rejects_non_tp4():
    module, fake_torch = _load_runtime()

    with pytest.raises(ValueError, match="world_size must be 4"):
        module.TP4PeerReductionGroup.create(
            rank=0,
            world_size=2,
            device=FakeDevice("cuda", 0),
            layer_count=16,
            max_active_tokens=8,
            hidden_size=5120,
            extension=FakeExtension(),
            distributed=FakeDistributed(),
            torch_module=fake_torch,
        )


def test_extension_flags_do_not_enable_or_misconfigure_fast_math():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "--use_fast_math" not in source


def test_create_allocates_fixed_ring_and_opens_each_peer_once():
    _, fake_torch, extension, group = _ready_group()

    assert [tensor.shape for tensor in fake_torch.allocations] == [
        (64, 2, 8, 5120),
        (64, 2),
        (64, 2, 8, 5120),
        (1,),
    ]
    assert group.state == "READY"
    assert [
        call for call in extension.calls if call[0] == "open_mapping"
    ] == [
        ("open_mapping", 1),
        ("open_mapping", 2),
        ("open_mapping", 3),
    ]


@pytest.mark.parametrize(
    ("generation", "partial", "residual", "message"),
    (
        (-1, _fake_fp32((1, 5120)), _fake_bf16((1, 5120)),
         "generation"),
        (0, _fake_fp32((1, 5120)), _fake_bf16((1, 5120)),
         "generation"),
        (1, _fake_fp32((1, 4096)), _fake_bf16((1, 5120)),
         "local_partial"),
        (1, _fake_bf16((1, 5120)), _fake_bf16((1, 5120)),
         "local_partial"),
        (1, _fake_fp32((1, 5120)), _fake_fp32((1, 5120)),
         "residual"),
        (1, _fake_fp32((9, 5120)), _fake_bf16((9, 5120)),
         "active token"),
    ),
)
def test_reduce_rejects_wrong_generation_shape_and_dtype(
    generation,
    partial,
    residual,
    message,
):
    _, _, _, group = _ready_group()

    with pytest.raises(ValueError, match=message):
        group.reduce_add_residual(
            layer_index=0,
            generation=generation,
            local_partial=partial,
            residual=residual,
        )


def test_reduce_uses_generation_slot_and_current_stream():
    _, _, extension, group = _ready_group()

    output = group.reduce_add_residual(
        layer_index=7,
        generation=5,
        local_partial=_fake_fp32((4, 5120)),
        residual=_fake_bf16((4, 5120)),
    )

    assert output.shape == (4, 5120)
    assert output.dtype == "torch.bfloat16"
    assert (
        "publish",
        7,
        1,
        5,
        1234,
    ) in extension.calls
    reduce_call = [
        call
        for call in extension.calls
        if call[0] == "reduce_add_residual"
    ][0]
    assert reduce_call[1:10] == (
        (1, 2, 3),
        (1,),
        0,
        7,
        1,
        5,
        4,
        5120,
        group.timeout_clocks,
    )


def test_timeout_poisons_group_and_future_work_is_rejected():
    _, _, extension, group = _ready_group()
    extension.reduce_status = 1

    group.reduce_add_residual(
        layer_index=0,
        generation=1,
        local_partial=_fake_fp32((1, 5120)),
        residual=_fake_bf16((1, 5120)),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        group.check_status()
    assert group.state == "POISONED"

    with pytest.raises(RuntimeError, match="POISONED"):
        group.reduce_add_residual(
            layer_index=0,
            generation=2,
            local_partial=_fake_fp32((1, 5120)),
            residual=_fake_bf16((1, 5120)),
        )


def test_close_releases_imports_then_owned_allocations_once():
    _, _, extension, group = _ready_group()
    extension.calls.clear()

    group.close()
    group.close()

    assert extension.calls == [
        ("close_mapping", 1),
        ("close_mapping", 2),
        ("close_mapping", 3),
        ("release_owned",),
    ]
    assert group.state == "CLOSED"
