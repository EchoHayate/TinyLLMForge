from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.qwen3_draft_proposal_kv import (
    Qwen3DraftPhysicalSlotStore,
    Qwen3DraftProposalKVStorage,
    build_qwen3_draft_proposal_kv_allocator,
)
from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
)


def _model(
    *,
    layer_count: int = 2,
    local_kv_heads: int = 2,
    head_dim: int = 4,
):
    layers = []
    for _ in range(layer_count):
        backend = SimpleNamespace(
            k_cache=torch.Tensor(),
            v_cache=torch.Tensor(),
            kv_quant_bits=None,
        )
        attention = SimpleNamespace(
            num_kv_heads=local_kv_heads,
            head_dim=head_dim,
            attn=backend,
        )
        layers.append(SimpleNamespace(self_attn=attention))
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers),
    )


def test_builder_default_direct_mode_allocates_no_cpu_backing():
    allocator = build_qwen3_draft_proposal_kv_allocator(
        _model(),
        offload_enabled=False,
        logical_entry_capacity=8,
        gpu_slot_capacity=8,
        cpu_backing_capacity=8,
        async_copy=True,
        batch_copy=True,
        dtype=torch.float32,
        device="cpu",
    )

    assert isinstance(allocator, DirectProposalKVAllocator)
    assert isinstance(
        allocator.physical_store,
        Qwen3DraftPhysicalSlotStore,
    )
    assert allocator.physical_store.cpu_key_cache is None
    assert allocator.physical_store.cpu_value_cache is None


def test_builder_offload_mode_uses_generic_residency_and_multilayer_storage():
    allocator = build_qwen3_draft_proposal_kv_allocator(
        _model(layer_count=3),
        offload_enabled=True,
        logical_entry_capacity=4,
        gpu_slot_capacity=2,
        cpu_backing_capacity=4,
        async_copy=False,
        batch_copy=True,
        dtype=torch.float32,
        device="cpu",
        _copy_backend=SynchronousProposalKVCopyBackend(),
    )

    assert isinstance(allocator, ProposalKVResidencyManager)
    assert isinstance(
        allocator.storage,
        Qwen3DraftProposalKVStorage,
    )
    assert allocator.storage.layer_count == 3
    assert allocator.storage.cpu_key_cache is not None
    assert allocator.storage.cpu_value_cache is not None
    assert allocator.batch_copy is True


def test_multilayer_storage_allocates_full_gpu_and_cpu_payload():
    model = _model()

    storage = Qwen3DraftProposalKVStorage(
        model,
        logical_capacity=4,
        gpu_capacity=2,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )

    assert storage.gpu_key_cache.shape == (2, 2, 1, 2, 4)
    assert storage.gpu_value_cache.shape == (2, 2, 1, 2, 4)
    assert storage.cpu_key_cache.shape == (2, 4, 1, 2, 4)
    assert storage.cpu_value_cache.shape == (2, 4, 1, 2, 4)
    assert storage.key_cache is storage.gpu_key_cache
    assert storage.value_cache is storage.gpu_value_cache
    assert storage.logical_capacity == 4
    assert storage.gpu_capacity == 2
    assert storage.capacity == 2


def test_entry_nbytes_counts_key_value_and_every_layer():
    storage = Qwen3DraftProposalKVStorage(
        _model(layer_count=3, local_kv_heads=2, head_dim=5),
        logical_capacity=4,
        gpu_capacity=2,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=False,
    )

    assert storage.entry_nbytes() == (
        2
        * 3
        * 1
        * 2
        * 5
        * torch.tensor([], dtype=torch.float32).element_size()
    )


def test_storage_binds_each_attention_backend_to_its_layer_slice():
    model = _model(layer_count=3)

    storage = Qwen3DraftProposalKVStorage(
        model,
        logical_capacity=4,
        gpu_capacity=2,
        dtype=torch.bfloat16,
        device="cpu",
        allocate_cpu_backing=False,
    )

    for layer_index, layer in enumerate(model.model.layers):
        backend = layer.self_attn.attn
        assert backend.k_cache.data_ptr() == (
            storage.gpu_key_cache[layer_index].data_ptr()
        )
        assert backend.v_cache.data_ptr() == (
            storage.gpu_value_cache[layer_index].data_ptr()
        )
        assert backend.kv_quant_bits == 0


def test_foreign_attention_cache_preflight_is_failure_atomic():
    model = _model(layer_count=2)
    first_backend = model.model.layers[0].self_attn.attn
    second_backend = model.model.layers[1].self_attn.attn
    first_key = first_backend.k_cache
    first_value = first_backend.v_cache
    second_backend.k_cache = torch.ones(1, 1, 2, 4)

    with pytest.raises(RuntimeError, match="already owns"):
        Qwen3DraftProposalKVStorage(
            model,
            logical_capacity=4,
            gpu_capacity=2,
            dtype=torch.float32,
            device="cpu",
            allocate_cpu_backing=False,
        )

    assert first_backend.k_cache is first_key
    assert first_backend.v_cache is first_value
    assert first_backend.kv_quant_bits is None


def test_copy_round_trip_moves_every_layer_and_key_value_row():
    storage = Qwen3DraftProposalKVStorage(
        _model(layer_count=3),
        logical_capacity=4,
        gpu_capacity=2,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )
    first_key = torch.arange(
        storage.gpu_key_cache[:, 0].numel(),
        dtype=torch.float32,
    ).reshape_as(storage.gpu_key_cache[:, 0])
    first_value = first_key + 100
    second_key = first_key + 200
    second_value = first_key + 300
    storage.gpu_key_cache[:, 0].copy_(first_key)
    storage.gpu_value_cache[:, 0].copy_(first_value)
    storage.gpu_key_cache[:, 1].copy_(second_key)
    storage.gpu_value_cache[:, 1].copy_(second_value)

    storage.copy_gpu_to_cpu(((3, 0), (1, 1)))

    assert torch.equal(storage.cpu_key_cache[:, 3], first_key)
    assert torch.equal(storage.cpu_value_cache[:, 3], first_value)
    assert torch.equal(storage.cpu_key_cache[:, 1], second_key)
    assert torch.equal(storage.cpu_value_cache[:, 1], second_value)

    storage.gpu_key_cache.zero_()
    storage.gpu_value_cache.zero_()
    storage.copy_cpu_to_gpu(((3, 1), (1, 0)))

    assert torch.equal(storage.gpu_key_cache[:, 1], first_key)
    assert torch.equal(storage.gpu_value_cache[:, 1], first_value)
    assert torch.equal(storage.gpu_key_cache[:, 0], second_key)
    assert torch.equal(storage.gpu_value_cache[:, 0], second_value)


@pytest.mark.parametrize(
    "rows",
    (
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (4, 1)),
        ((0, 0), (1, 2)),
        ((0, 0), ("1", 1)),
        ((0, 0), (1, True)),
        ((0, 0), (1,)),
    ),
)
def test_copy_row_validation_precedes_any_destination_mutation(rows):
    storage = Qwen3DraftProposalKVStorage(
        _model(),
        logical_capacity=4,
        gpu_capacity=2,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )
    storage.gpu_key_cache.fill_(7)
    storage.gpu_value_cache.fill_(9)
    before_key = storage.cpu_key_cache.clone()
    before_value = storage.cpu_value_cache.clone()

    with pytest.raises(ValueError, match="copy rows"):
        storage.copy_gpu_to_cpu(rows)

    assert torch.equal(storage.cpu_key_cache, before_key)
    assert torch.equal(storage.cpu_value_cache, before_value)


def test_copy_without_cpu_backing_fails_closed():
    storage = Qwen3DraftProposalKVStorage(
        _model(),
        logical_capacity=2,
        gpu_capacity=2,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=False,
    )

    with pytest.raises(RuntimeError, match="not allocated"):
        storage.copy_gpu_to_cpu(((0, 0),))
    with pytest.raises(RuntimeError, match="not allocated"):
        storage.copy_cpu_to_gpu(((0, 0),))


def test_generic_storage_authority_reports_multilayer_backing():
    storage = Qwen3DraftProposalKVStorage(
        _model(layer_count=3),
        logical_capacity=5,
        gpu_capacity=2,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )

    snapshot = storage.authority_snapshot()

    assert snapshot == {
        "logical_capacity": 5,
        "gpu_capacity": 2,
        "capacity": 2,
        "block_size": 1,
        "layer_count": 3,
        "local_kv_heads": 2,
        "head_dim": 4,
        "entry_nbytes": storage.entry_nbytes(),
        "dtype": "torch.float32",
        "device": "cpu",
        "cpu_backing_allocated": True,
    }
    assert not hasattr(storage, "reserve_slots")
    assert not hasattr(storage, "release_slots")


def test_direct_store_preserves_slot_ownership_and_legacy_snapshot():
    model = _model(layer_count=2)
    storage = Qwen3DraftPhysicalSlotStore(
        model,
        capacity=3,
        dtype=torch.float32,
        device="cpu",
    )

    assert isinstance(storage, Qwen3DraftProposalKVStorage)
    assert storage.logical_capacity == 3
    assert storage.gpu_capacity == 3
    assert storage.cpu_key_cache is None
    assert storage.cpu_value_cache is None
    assert storage.reserve_slots(2) == (0, 1)
    storage.gpu_key_cache[:, :2].fill_(3)
    storage.gpu_value_cache[:, :2].fill_(4)

    storage.release_slots((1,))

    assert torch.count_nonzero(storage.gpu_key_cache[:, 1]) == 0
    assert torch.count_nonzero(storage.gpu_value_cache[:, 1]) == 0
    assert torch.count_nonzero(storage.gpu_key_cache[:, 0]) > 0
    assert storage.reserve_slots(1) == (1,)
    assert storage.authority_snapshot() == {
        "capacity": 3,
        "block_size": 1,
        "layer_count": 2,
        "local_kv_heads": 2,
        "head_dim": 4,
        "dtype": "torch.float32",
        "device": "cpu",
        "allocated_slot_count": 2,
        "free_slot_count": 1,
    }


def test_direct_store_slot_identity_covers_every_layer():
    storage = Qwen3DraftPhysicalSlotStore(
        _model(layer_count=2),
        capacity=2,
        dtype=torch.float32,
        device="cpu",
    )

    identity = storage.slot_identity(1)

    assert identity == (
        (
            storage.gpu_key_cache[0, 1].data_ptr(),
            storage.gpu_value_cache[0, 1].data_ptr(),
        ),
        (
            storage.gpu_key_cache[1, 1].data_ptr(),
            storage.gpu_value_cache[1, 1].data_ptr(),
        ),
    )


def test_storage_round_trips_through_generic_residency_manager():
    storage = Qwen3DraftProposalKVStorage(
        _model(layer_count=3),
        logical_capacity=3,
        gpu_capacity=1,
        dtype=torch.float32,
        device="cpu",
        allocate_cpu_backing=True,
        allocate_pinned_cpu=False,
    )
    manager = ProposalKVResidencyManager(
        storage=storage,
        copy_backend=SynchronousProposalKVCopyBackend(),
    )
    first = manager.reserve_entries(1)
    first_lease = manager.ensure_writable(first)
    expected_key = torch.arange(
        storage.gpu_key_cache[:, 0].numel(),
        dtype=torch.float32,
    ).reshape_as(storage.gpu_key_cache[:, 0])
    expected_value = expected_key + 100
    storage.gpu_key_cache[:, 0].copy_(expected_key)
    storage.gpu_value_cache[:, 0].copy_(expected_value)
    manager.record_write_complete(first_lease)
    manager.commit_entries(first)

    second = manager.reserve_entries(1)
    second_lease = manager.ensure_writable(second)
    manager.record_write_complete(second_lease)
    manager.commit_entries(second)
    manager.retire_entries(second, writeback=False)
    restored = manager.ensure_readable(first)

    assert restored.physical_slot_ids == (0,)
    assert torch.equal(storage.gpu_key_cache[:, 0], expected_key)
    assert torch.equal(storage.gpu_value_cache[:, 0], expected_value)
    snapshot = manager.authority_snapshot()
    assert snapshot["d2h_entry_count"] == 1
    assert snapshot["h2d_entry_count"] == 1
    assert snapshot["d2h_bytes"] == storage.entry_nbytes()
    assert snapshot["h2d_bytes"] == storage.entry_nbytes()
