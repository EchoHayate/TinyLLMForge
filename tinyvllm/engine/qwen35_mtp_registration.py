from __future__ import annotations

import torch

from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.proposal_kv_residency import (
    ProposalKVResidencyManager,
    SynchronousProposalKVCopyBackend,
    TorchProposalKVCopyBackend,
)


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


class Qwen35MTPProposalKVStorage:
    block_size = 1

    def __init__(
        self,
        *,
        logical_capacity: int,
        gpu_capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
        allocate_cpu_backing: bool,
        allocate_pinned_cpu: bool = True,
        block_size: int = 1,
    ):
        self.logical_capacity = _positive_integer(
            logical_capacity,
            "logical_capacity",
        )
        self.gpu_capacity = _positive_integer(
            gpu_capacity,
            "gpu_capacity",
        )
        if self.gpu_capacity > self.logical_capacity:
            raise ValueError(
                "gpu_capacity must not exceed logical_capacity"
            )
        self.num_kv_heads = _positive_integer(
            num_kv_heads,
            "num_kv_heads",
        )
        self.head_dim = _positive_integer(head_dim, "head_dim")
        if block_size != 1:
            raise ValueError(
                "Qwen3.5 MTP proposal KV requires block size one"
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "Qwen3.5 MTP proposal KV supports only FP16 or BF16"
            )
        if not isinstance(allocate_cpu_backing, bool):
            raise ValueError("allocate_cpu_backing must be a bool")
        if not isinstance(allocate_pinned_cpu, bool):
            raise ValueError("allocate_pinned_cpu must be a bool")
        device = torch.device(device)
        if (
            allocate_cpu_backing
            and device.type == "cuda"
            and not allocate_pinned_cpu
        ):
            raise ValueError(
                "CUDA proposal KV offload requires pinned CPU backing"
            )
        self.dtype = dtype
        self.device = device
        self.allocate_cpu_backing = allocate_cpu_backing
        self.gpu_key_cache = torch.zeros(
            self.gpu_capacity,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=dtype,
            device=device,
        )
        self.gpu_value_cache = torch.zeros_like(
            self.gpu_key_cache
        )
        self.key_cache = self.gpu_key_cache
        self.value_cache = self.gpu_value_cache
        if allocate_cpu_backing:
            self.cpu_key_cache = torch.zeros(
                self.logical_capacity,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=dtype,
                device="cpu",
                pin_memory=allocate_pinned_cpu,
            )
            self.cpu_value_cache = torch.zeros(
                self.logical_capacity,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=dtype,
                device="cpu",
                pin_memory=allocate_pinned_cpu,
            )
        else:
            self.cpu_key_cache = None
            self.cpu_value_cache = None
        self.capacity = self.gpu_capacity
        self._free_slot_ids = list(range(self.gpu_capacity))
        self._allocated_slot_ids: set[int] = set()

    def bind_attention_backend(self, backend) -> None:
        if backend is None:
            raise ValueError("attention backend is required")
        for name, cache in (
            ("k_cache", self.gpu_key_cache),
            ("v_cache", self.gpu_value_cache),
        ):
            existing = getattr(backend, name, None)
            if (
                isinstance(existing, torch.Tensor)
                and existing.numel() > 0
                and existing is not cache
            ):
                raise RuntimeError(
                    "attention backend already owns a different KV cache"
                )
        backend.k_cache = self.gpu_key_cache
        backend.v_cache = self.gpu_value_cache
        backend.kv_quant_bits = 0

    def entry_nbytes(self) -> int:
        element_count = (
            self.block_size * self.num_kv_heads * self.head_dim
        )
        return (
            2
            * element_count
            * int(self.gpu_key_cache.element_size())
        )

    def _require_cpu_backing(self) -> None:
        if self.cpu_key_cache is None or self.cpu_value_cache is None:
            raise RuntimeError(
                "proposal KV CPU backing is not allocated"
            )

    def copy_gpu_to_cpu(
        self,
        rows: tuple[tuple[int, int], ...],
    ) -> None:
        self._require_cpu_backing()
        for logical_entry_id, physical_slot_id in rows:
            self.cpu_key_cache[logical_entry_id].copy_(
                self.gpu_key_cache[physical_slot_id],
                non_blocking=True,
            )
            self.cpu_value_cache[logical_entry_id].copy_(
                self.gpu_value_cache[physical_slot_id],
                non_blocking=True,
            )

    def copy_cpu_to_gpu(
        self,
        rows: tuple[tuple[int, int], ...],
    ) -> None:
        self._require_cpu_backing()
        for logical_entry_id, physical_slot_id in rows:
            self.gpu_key_cache[physical_slot_id].copy_(
                self.cpu_key_cache[logical_entry_id],
                non_blocking=True,
            )
            self.gpu_value_cache[physical_slot_id].copy_(
                self.cpu_value_cache[logical_entry_id],
                non_blocking=True,
            )

    def _slot_id(self, slot_id: int) -> int:
        if (
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            or slot_id >= self.gpu_capacity
        ):
            raise ValueError("slot_id is outside the physical store")
        return slot_id

    def slot_identity(self, slot_id: int) -> tuple[int, int]:
        slot_id = self._slot_id(slot_id)
        return (
            self.gpu_key_cache[slot_id].data_ptr(),
            self.gpu_value_cache[slot_id].data_ptr(),
        )

    def is_allocated(self, slot_id: int) -> bool:
        return self._slot_id(slot_id) in self._allocated_slot_ids

    def reserve_slots(self, count: int) -> tuple[int, ...]:
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                "slot reservation count must be nonnegative"
            )
        if count > len(self._free_slot_ids):
            raise RuntimeError(
                "Qwen3.5 MTP proposal KV slots are exhausted"
            )
        slot_ids = tuple(self._free_slot_ids[:count])
        del self._free_slot_ids[:count]
        self._allocated_slot_ids.update(slot_ids)
        return slot_ids

    def release_slots(self, slot_ids: tuple[int, ...]) -> None:
        if (
            not isinstance(slot_ids, tuple)
            or len(set(slot_ids)) != len(slot_ids)
            or not set(slot_ids).issubset(
                self._allocated_slot_ids
            )
        ):
            raise RuntimeError(
                "Qwen3.5 MTP proposal KV slot ownership is stale"
            )
        if slot_ids:
            indices = torch.tensor(
                slot_ids,
                dtype=torch.long,
                device=self.gpu_key_cache.device,
            )
            self.gpu_key_cache.index_fill_(0, indices, 0)
            self.gpu_value_cache.index_fill_(0, indices, 0)
        self._allocated_slot_ids.difference_update(slot_ids)
        self._free_slot_ids.extend(slot_ids)
        self._free_slot_ids.sort()


class Qwen35MTPPhysicalSlotStore(Qwen35MTPProposalKVStorage):

    def __init__(
        self,
        capacity: int,
        *,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        super().__init__(
            logical_capacity=capacity,
            gpu_capacity=capacity,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            allocate_cpu_backing=False,
        )


def build_qwen35_mtp_proposal_kv_allocator(
    *,
    offload_enabled: bool,
    logical_entry_capacity: int,
    gpu_slot_capacity: int,
    cpu_backing_capacity: int,
    async_copy: bool,
    batch_copy: bool,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str | torch.device,
    _copy_backend=None,
):
    if not isinstance(offload_enabled, bool):
        raise ValueError("offload_enabled must be a bool")
    if not isinstance(async_copy, bool):
        raise ValueError("async_copy must be a bool")
    if not isinstance(batch_copy, bool):
        raise ValueError("batch_copy must be a bool")
    if not offload_enabled:
        if logical_entry_capacity != gpu_slot_capacity:
            raise ValueError(
                "direct proposal KV requires logical and GPU "
                "capacities to match"
            )
        storage = Qwen35MTPProposalKVStorage(
            logical_capacity=logical_entry_capacity,
            gpu_capacity=gpu_slot_capacity,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            allocate_cpu_backing=False,
        )
        return DirectProposalKVAllocator(storage)
    if logical_entry_capacity != cpu_backing_capacity:
        raise ValueError(
            "logical and CPU backing capacities must match"
        )
    if logical_entry_capacity <= gpu_slot_capacity:
        raise ValueError(
            "offloaded proposal KV requires logical capacity "
            "greater than GPU capacity"
        )
    storage = Qwen35MTPProposalKVStorage(
        logical_capacity=logical_entry_capacity,
        gpu_capacity=gpu_slot_capacity,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        allocate_cpu_backing=True,
        allocate_pinned_cpu=(
            torch.device(device).type == "cuda"
        ),
    )
    copy_backend = _copy_backend
    if copy_backend is None:
        copy_backend = (
            TorchProposalKVCopyBackend()
            if async_copy
            else SynchronousProposalKVCopyBackend()
        )
    return ProposalKVResidencyManager(
        storage=storage,
        copy_backend=copy_backend,
        batch_copy=batch_copy,
    )
