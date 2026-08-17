from __future__ import annotations

import torch


class Qwen35MTPPhysicalSlotStore:

    def __init__(
        self,
        capacity: int,
        *,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer")
        for value, name in (
            (num_kv_heads, "num_kv_heads"),
            (head_dim, "head_dim"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be a torch dtype")
        device = torch.device(device)
        self.capacity = capacity
        self.block_size = 1
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.key_cache = torch.zeros(
            capacity,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        self._free_slot_ids = list(range(capacity))
        self._allocated_slot_ids: set[int] = set()

    def bind_attention_backend(self, backend) -> None:
        if backend is None:
            raise ValueError("attention backend is required")
        for name in ("k_cache", "v_cache"):
            existing = getattr(backend, name, None)
            if (
                isinstance(existing, torch.Tensor)
                and existing.numel() > 0
                and existing is not getattr(
                    self,
                    "key_cache" if name == "k_cache" else "value_cache",
                )
            ):
                raise RuntimeError(
                    "attention backend already owns a different KV cache"
                )
        backend.k_cache = self.key_cache
        backend.v_cache = self.value_cache
        backend.kv_quant_bits = 0

    def _slot_id(self, slot_id: int) -> int:
        if (
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            or slot_id >= self.capacity
        ):
            raise ValueError("slot_id is outside the physical store")
        return slot_id

    def slot_identity(self, slot_id: int) -> tuple[int, int]:
        slot_id = self._slot_id(slot_id)
        return (
            self.key_cache[slot_id].data_ptr(),
            self.value_cache[slot_id].data_ptr(),
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
                device=self.key_cache.device,
            )
            self.key_cache.index_fill_(0, indices, 0)
            self.value_cache.index_fill_(0, indices, 0)
        self._allocated_slot_ids.difference_update(slot_ids)
        self._free_slot_ids.extend(slot_ids)
        self._free_slot_ids.sort()
