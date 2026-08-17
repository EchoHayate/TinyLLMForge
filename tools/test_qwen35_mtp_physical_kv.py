from __future__ import annotations

from math import prod
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_MODULE_NAMES = (
    "tinyvllm",
    "tinyvllm.engine",
    "torch",
    "tinyvllm.engine.proposal_kv_allocator",
    "tinyvllm.engine.proposal_kv_residency",
    "tinyvllm.engine.qwen35_mtp_registration",
)
_MISSING = object()
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _MISSING)
    for name in _MODULE_NAMES
}
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package


class _FakeDType:

    def __init__(self, name, element_size):
        self.name = name
        self.element_size = element_size


class _FakeDevice:

    def __init__(self, value):
        self.type = str(value).split(":", 1)[0]

    def __str__(self):
        return self.type


class _FakeTensor:

    def __init__(
        self,
        shape,
        *,
        dtype,
        device,
        pin_memory=False,
        rows=None,
        selected_rows=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _FakeDevice(device)
        self._pinned = pin_memory
        row_width = prod(self.shape[1:]) if self.shape else 1
        self._rows = (
            [[0] * row_width for _ in range(self.shape[0])]
            if rows is None and self.shape
            else rows
        )
        self._selected_rows = (
            tuple(range(self.shape[0]))
            if selected_rows is None and self.shape
            else selected_rows
        )

    def numel(self):
        return prod(self.shape) if self.shape else 1

    def element_size(self):
        return self.dtype.element_size

    def is_pinned(self):
        return self._pinned

    def data_ptr(self):
        if not self._selected_rows:
            return id(self._rows)
        return id(self._rows[self._selected_rows[0]])

    def fill_(self, value):
        for row_id in self._selected_rows:
            self._rows[row_id][:] = [value] * len(
                self._rows[row_id]
            )
        return self

    def copy_(self, other, non_blocking=False):
        del non_blocking
        for target_row, source_row in zip(
            self._selected_rows,
            other._selected_rows,
        ):
            self._rows[target_row][:] = other._rows[source_row]
        return self

    def index_fill_(self, dimension, indices, value):
        assert dimension == 0
        for row_id in indices.values:
            self._rows[row_id][:] = [value] * len(self._rows[row_id])
        return self

    def row_values(self, row_id):
        return tuple(self._rows[row_id])

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        if isinstance(index, int):
            selected_rows = (self._selected_rows[index],)
        elif isinstance(index, (list, tuple)):
            selected_rows = tuple(
                self._selected_rows[row_id] for row_id in index
            )
        else:
            raise TypeError("unsupported fake tensor index")
        return _FakeTensor(
            (len(selected_rows),) + self.shape[1:],
            dtype=self.dtype,
            device=self.device,
            pin_memory=self._pinned,
            rows=self._rows,
            selected_rows=selected_rows,
        )


class _FakeIndexTensor:

    def __init__(self, values):
        self.values = tuple(values)


fake_torch = types.ModuleType("torch")
fake_torch.dtype = _FakeDType
fake_torch.float16 = _FakeDType("float16", 2)
fake_torch.bfloat16 = _FakeDType("bfloat16", 2)
fake_torch.float32 = _FakeDType("float32", 4)
fake_torch.long = _FakeDType("long", 8)
fake_torch.Tensor = _FakeTensor
fake_torch.device = _FakeDevice


def _zeros(*shape, dtype, device, pin_memory=False):
    return _FakeTensor(
        shape,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
    )


def _zeros_like(tensor):
    return _FakeTensor(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        pin_memory=tensor.is_pinned(),
    )


fake_torch.zeros = _zeros
fake_torch.zeros_like = _zeros_like
fake_torch.empty = lambda size: _FakeTensor(
    (size,),
    dtype=fake_torch.float16,
    device="cpu",
)
fake_torch.tensor = lambda values, **kwargs: _FakeIndexTensor(values)
sys.modules["torch"] = fake_torch


from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
)
from tinyvllm.engine.qwen35_mtp_registration import (
    Qwen35MTPProposalKVStorage,
    build_qwen35_mtp_proposal_kv_allocator,
)

for module_name, original in _ORIGINAL_MODULES.items():
    if original is _MISSING:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = original


def _storage(
    *,
    logical_capacity=8,
    gpu_capacity=4,
    dtype=fake_torch.float16,
    allocate_cpu_backing=True,
    allocate_pinned_cpu=False,
    device="cpu",
    block_size=1,
):
    return Qwen35MTPProposalKVStorage(
        logical_capacity=logical_capacity,
        gpu_capacity=gpu_capacity,
        num_kv_heads=2,
        head_dim=4,
        dtype=dtype,
        device=device,
        allocate_cpu_backing=allocate_cpu_backing,
        allocate_pinned_cpu=allocate_pinned_cpu,
        block_size=block_size,
    )


def test_storage_geometry_bytes_and_attention_binding_use_gpu_only():
    storage = _storage()
    backend = SimpleNamespace(
        k_cache=fake_torch.empty(0),
        v_cache=fake_torch.empty(0),
    )

    storage.bind_attention_backend(backend)

    assert storage.gpu_key_cache.shape == (4, 1, 2, 4)
    assert storage.gpu_value_cache.shape == (4, 1, 2, 4)
    assert storage.cpu_key_cache.shape == (8, 1, 2, 4)
    assert storage.cpu_value_cache.shape == (8, 1, 2, 4)
    assert storage.entry_nbytes() == 32
    assert backend.k_cache is storage.gpu_key_cache
    assert backend.v_cache is storage.gpu_value_cache


def test_direct_builder_allocates_no_cpu_backing_and_keeps_commit_in_place():
    allocator = build_qwen35_mtp_proposal_kv_allocator(
        offload_enabled=False,
        logical_entry_capacity=4,
        gpu_slot_capacity=4,
        cpu_backing_capacity=4,
        async_copy=False,
        batch_copy=True,
        num_kv_heads=2,
        head_dim=4,
        dtype=fake_torch.float16,
        device="cpu",
    )

    assert isinstance(allocator, DirectProposalKVAllocator)
    storage = allocator.physical_store
    assert storage.cpu_key_cache is None
    assert storage.cpu_value_cache is None
    identities = allocator.reserve_entries(1)
    lease = allocator.ensure_writable(identities)
    slot_id = lease.physical_slot_ids[0]
    storage.gpu_key_cache[slot_id].fill_(7)
    pointer = storage.slot_identity(slot_id)
    allocator.record_write_complete(lease)
    allocator.commit_entries(identities)

    assert allocator.ensure_readable(identities) == lease
    assert storage.slot_identity(slot_id) == pointer
    assert storage.gpu_key_cache.row_values(slot_id) == (7,) * 8


def test_offload_builder_constructs_fixed_cpu_backing_and_residency():
    allocator = build_qwen35_mtp_proposal_kv_allocator(
        offload_enabled=True,
        logical_entry_capacity=8,
        gpu_slot_capacity=4,
        cpu_backing_capacity=8,
        async_copy=False,
        batch_copy=True,
        num_kv_heads=2,
        head_dim=4,
        dtype=fake_torch.bfloat16,
        device="cpu",
        _copy_backend=SynchronousProposalKVCopyBackend(),
    )

    assert isinstance(allocator, ProposalKVResidencyManager)
    assert allocator.storage.cpu_key_cache.shape == (8, 1, 2, 4)
    assert allocator.storage.cpu_value_cache.shape == (8, 1, 2, 4)


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"dtype": fake_torch.float32}, "FP16 or BF16"),
        ({"block_size": 2}, "block size one"),
        (
            {
                "device": "cuda",
                "allocate_pinned_cpu": False,
            },
            "pinned",
        ),
    ),
)
def test_storage_rejects_unsupported_v1_geometry(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _storage(**kwargs)
