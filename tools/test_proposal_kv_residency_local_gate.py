from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from math import prod
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
)


TERMINAL_CLASSIFICATIONS = {
    "PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT": (
        "ESTABLISHED"
    ),
    "PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT": "ESTABLISHED",
    "QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION": (
        "ESTABLISHED"
    ),
    "REAL_PROPOSAL_KV_MOVEMENT": "NOT_ESTABLISHED",
    "PERFORMANCE_IMPROVEMENT": "NOT_ESTABLISHED",
    "PHASE_1": "NOT_ACHIEVED",
    "PROMOTION": "NOT_PROMOTABLE",
}


class _DirectStore:

    def __init__(self):
        self.capacity = 2
        self.free = [1, 0]

    def reserve_slots(self, count):
        result = tuple(self.free[:count])
        del self.free[:count]
        return result

    def release_slots(self, slot_ids):
        self.free.extend(slot_ids)


class _Row:

    def __init__(self):
        self.values = [0, 0]

    def fill_(self, value):
        self.values[:] = [value, value]
        return self

    def copy_(self, other):
        self.values[:] = other.values
        return self


class _Rows:

    def __init__(self, capacity):
        self.rows = [_Row() for _ in range(capacity)]

    def __getitem__(self, index):
        return self.rows[index]


class _ResidencyStorage:

    def __init__(self):
        self.logical_capacity = 3
        self.gpu_capacity = 1
        self.block_size = 1
        self.dtype = "float16"
        self.gpu_key_cache = _Rows(1)
        self.gpu_value_cache = _Rows(1)
        self.cpu_key_cache = _Rows(3)
        self.cpu_value_cache = _Rows(3)

    def entry_nbytes(self):
        return 16

    def copy_gpu_to_cpu(self, rows):
        for logical_id, slot_id in rows:
            self.cpu_key_cache[logical_id].copy_(
                self.gpu_key_cache[slot_id]
            )
            self.cpu_value_cache[logical_id].copy_(
                self.gpu_value_cache[slot_id]
            )

    def copy_cpu_to_gpu(self, rows):
        for logical_id, slot_id in rows:
            self.gpu_key_cache[slot_id].copy_(
                self.cpu_key_cache[logical_id]
            )
            self.gpu_value_cache[slot_id].copy_(
                self.cpu_value_cache[logical_id]
            )


def _residency_manager():
    return ProposalKVResidencyManager(
        storage=_ResidencyStorage(),
        copy_backend=SynchronousProposalKVCopyBackend(),
    )


def _materialize(manager, value):
    identities = manager.reserve_entries(1)
    lease = manager.ensure_writable(identities)
    manager.storage.gpu_key_cache[
        lease.physical_slot_ids[0]
    ].fill_(value)
    manager.record_write_complete(lease)
    return identities, lease


def test_logical_physical_decoupling_contract():
    allocator = DirectProposalKVAllocator(_DirectStore())
    identities = allocator.reserve_entries(2)
    lease = allocator.ensure_writable(identities)

    assert tuple(
        identity.logical_entry_id for identity in identities
    ) == (0, 1)
    assert lease.physical_slot_ids == (1, 0)


def test_residency_transaction_contract():
    manager = _residency_manager()
    first, _ = _materialize(manager, 7)
    manager.commit_entries(first)
    replacement = manager.reserve_entries(1)
    manager.ensure_writable(replacement)
    snapshot = manager.authority_snapshot()

    assert snapshot["d2h_entry_count"] == 1
    manager.retire_entries(replacement, writeback=False)
    restored = manager.ensure_readable(first)
    assert restored.physical_slot_ids == (0,)
    assert manager.authority_snapshot()["h2d_entry_count"] == 1


class _FakeDType:

    def __init__(self, element_size):
        self.element_size = element_size


class _FakeDevice:

    def __init__(self, value):
        self.type = str(value).split(":", 1)[0]

    def __str__(self):
        return self.type


class _FakeTensor:

    def __init__(self, shape, *, dtype, device, pin_memory=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _FakeDevice(device)
        self._pinned = pin_memory
        width = prod(self.shape[1:]) if self.shape else 1
        self.rows = [
            [0] * width for _ in range(self.shape[0])
        ]

    def element_size(self):
        return self.dtype.element_size

    def numel(self):
        return prod(self.shape)

    def index_fill_(self, dimension, indices, value):
        assert dimension == 0
        for row_id in indices:
            self.rows[row_id][:] = [value] * len(self.rows[row_id])


def _load_qwen_builder():
    fake_torch = types.ModuleType("torch")
    fake_torch.dtype = _FakeDType
    fake_torch.float16 = _FakeDType(2)
    fake_torch.bfloat16 = _FakeDType(2)
    fake_torch.long = _FakeDType(8)
    fake_torch.Tensor = _FakeTensor
    fake_torch.device = _FakeDevice
    fake_torch.zeros = lambda *shape, **kwargs: _FakeTensor(
        shape,
        dtype=kwargs["dtype"],
        device=kwargs["device"],
        pin_memory=kwargs.get("pin_memory", False),
    )
    fake_torch.zeros_like = lambda tensor: _FakeTensor(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        pin_memory=tensor._pinned,
    )
    fake_torch.tensor = lambda values, **kwargs: tuple(values)
    module_name = "_qwen35_mtp_registration_local_gate"
    module_path = (
        ROOT / "tinyvllm/engine/qwen35_mtp_registration.py"
    )
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_qwen35_mtp_proposal_kv_allocator
    finally:
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch


def test_qwen35_mtp_local_integration_contract():
    builder = _load_qwen_builder()
    allocator = builder(
        offload_enabled=False,
        logical_entry_capacity=2,
        gpu_slot_capacity=2,
        cpu_backing_capacity=2,
        async_copy=False,
        batch_copy=True,
        num_kv_heads=1,
        head_dim=2,
        dtype=builder.__globals__["torch"].float16,
        device="cpu",
    )
    cache = ProposalKVCache(allocator)
    transaction = cache.begin(7, 0, 1)
    lease = allocator.ensure_writable(
        transaction.staged_entry_identities
    )
    allocator.record_write_complete(lease)
    cache.mark_materialized(transaction, 1)
    ticket = cache.prepare_finalize(
        transaction.transaction_id,
        accepted_proposal_tokens=2,
    )
    cache.commit_finalize(ticket.ticket_id)

    assert cache.committed_entry_identities(
        7
    ) == transaction.staged_entry_identities
    assert cache.authority_snapshot()["entry_allocator"][
        "allocator_mode"
    ] == "direct"


def test_default_off_zero_movement_contract():
    allocator = DirectProposalKVAllocator(_DirectStore())
    identities = allocator.reserve_entries(1)
    lease = allocator.ensure_writable(identities)
    allocator.record_write_complete(lease)
    allocator.commit_entries(identities)
    snapshot = allocator.authority_snapshot()

    assert snapshot["h2d_entry_count"] == 0
    assert snapshot["d2h_entry_count"] == 0


def test_rejected_suffix_zero_d2h_contract():
    manager = _residency_manager()
    identities, _ = _materialize(manager, 9)
    manager.retire_entries(identities, writeback=False)
    snapshot = manager.authority_snapshot()

    assert snapshot["rejected_entry_count"] == 1
    assert snapshot["rejected_entry_d2h_count"] == 0
    assert snapshot["rejected_entry_d2h_bytes"] == 0
    assert snapshot["d2h_entry_count"] == 0


def test_local_gate_keeps_gpu_and_performance_unestablished():
    assert TERMINAL_CLASSIFICATIONS == {
        "PROPOSAL_KV_LOGICAL_PHYSICAL_DECOUPLING_CONTRACT": (
            "ESTABLISHED"
        ),
        "PROPOSAL_KV_RESIDENCY_TRANSACTION_CONTRACT": "ESTABLISHED",
        "QWEN35_MTP_PROPOSAL_KV_OFFLOAD_LOCAL_INTEGRATION": (
            "ESTABLISHED"
        ),
        "REAL_PROPOSAL_KV_MOVEMENT": "NOT_ESTABLISHED",
        "PERFORMANCE_IMPROVEMENT": "NOT_ESTABLISHED",
        "PHASE_1": "NOT_ACHIEVED",
        "PROMOTION": "NOT_PROMOTABLE",
    }
