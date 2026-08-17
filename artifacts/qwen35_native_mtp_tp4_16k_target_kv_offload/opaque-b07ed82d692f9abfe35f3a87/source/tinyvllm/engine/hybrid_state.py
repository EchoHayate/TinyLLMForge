from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import torch


HybridStateRole = Literal[
    "linear_convolution",
    "linear_recurrent",
]
_VALID_ROLES = {
    "linear_convolution",
    "linear_recurrent",
}
_VALID_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


@dataclass(frozen=True)
class HybridStateComponentSpec:
    layer_index: int
    role: HybridStateRole
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class HybridStateLayout:
    components: tuple[HybridStateComponentSpec, ...]

    def __post_init__(self):
        components = tuple(self.components)
        keys = set()
        for component in components:
            if (
                isinstance(component.layer_index, bool)
                or not isinstance(component.layer_index, int)
                or component.layer_index < 0
            ):
                raise ValueError("hybrid state layer_index must be non-negative")
            if component.role not in _VALID_ROLES:
                raise ValueError(
                    f"unsupported hybrid state role: {component.role}"
                )
            shape = tuple(component.shape)
            if not shape or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            ):
                raise ValueError("hybrid state shape must contain positive integers")
            if component.dtype not in _VALID_DTYPES:
                raise ValueError(
                    f"unsupported hybrid state dtype: {component.dtype}"
                )
            key = (component.layer_index, component.role)
            if key in keys:
                raise ValueError(f"duplicate hybrid state component: {key}")
            keys.add(key)
        canonical = tuple(sorted(
            components,
            key=lambda component: (component.layer_index, component.role),
        ))
        object.__setattr__(self, "components", canonical)

    @property
    def bytes_per_slot(self) -> int:
        return sum(
            _numel(component.shape) * component.dtype.itemsize
            for component in self.components
        )

    @property
    def bytes_by_role(self) -> dict[str, int]:
        result = {role: 0 for role in sorted(_VALID_ROLES)}
        for component in self.components:
            result[component.role] += (
                _numel(component.shape) * component.dtype.itemsize
            )
        return result

    @property
    def fingerprint(self) -> str:
        payload = [
            {
                "layer_index": component.layer_index,
                "role": component.role,
                "shape": list(component.shape),
                "dtype": str(component.dtype).removeprefix("torch."),
            }
            for component in self.components
        ]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HybridStateLease:
    slot_id: int
    generation: int
    request_id: int


class HybridStateSlotAllocator:
    def __init__(self, capacity: int):
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise ValueError("hybrid state capacity must be positive")
        self.capacity = capacity
        self._free_slots = deque(range(capacity))
        self._generations = [0] * capacity
        self._owners: dict[int, HybridStateLease] = {}
        self._request_leases: dict[int, HybridStateLease] = {}

    def can_allocate(self) -> bool:
        return bool(self._free_slots)

    def allocate(self, request_id: int) -> HybridStateLease:
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id < 0
        ):
            raise ValueError(
                "hybrid state request_id must be a non-negative integer"
            )
        if request_id in self._request_leases:
            raise RuntimeError(
                f"hybrid state request already allocated: {request_id}"
            )
        if not self._free_slots:
            raise RuntimeError("hybrid state slots exhausted")
        slot_id = self._free_slots.popleft()
        self._generations[slot_id] += 1
        lease = HybridStateLease(
            slot_id=slot_id,
            generation=self._generations[slot_id],
            request_id=request_id,
        )
        self._owners[slot_id] = lease
        self._request_leases[request_id] = lease
        return lease

    def release(self, lease: HybridStateLease) -> None:
        self.validate(lease)
        del self._owners[lease.slot_id]
        del self._request_leases[lease.request_id]
        self._free_slots.append(lease.slot_id)

    def validate(self, lease: HybridStateLease) -> HybridStateLease:
        current = self._current_lease(lease.slot_id)
        if current != lease:
            raise RuntimeError(
                "hybrid state lease mismatch: "
                f"expected={current}, received={lease}"
            )
        if self._request_leases.get(lease.request_id) != lease:
            raise RuntimeError(
                f"hybrid state request ownership mismatch: {lease}"
            )
        return lease

    def lease_for_request(self, request_id: int) -> HybridStateLease | None:
        return self._request_leases.get(request_id)

    def observation_snapshot(self) -> dict:
        return {
            "capacity": self.capacity,
            "free_slots": len(self._free_slots),
            "used_slots": len(self._owners),
            "owners": {
                str(slot_id): lease.request_id
                for slot_id, lease in sorted(self._owners.items())
            },
            "generations": {
                str(slot_id): generation
                for slot_id, generation in enumerate(self._generations)
                if generation > 0
            },
        }

    def _current_lease(self, slot_id: int) -> HybridStateLease | None:
        if (
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            or slot_id >= self.capacity
        ):
            raise RuntimeError(f"hybrid state slot out of range: {slot_id}")
        return self._owners.get(slot_id)


class HybridStateTensorPool:
    def __init__(
        self,
        layout: HybridStateLayout,
        capacity: int,
        device: torch.device | str,
    ):
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise ValueError("hybrid state capacity must be positive")
        self.layout = layout
        self.capacity = capacity
        self.device = torch.device(device)
        self._tensors = {
            (component.layer_index, component.role): torch.zeros(
                (capacity, *component.shape),
                dtype=component.dtype,
                device=self.device,
            )
            for component in layout.components
        }
        self._bindings: dict[int, tuple[int, int]] = {}

    @property
    def logical_bytes(self) -> int:
        return self.layout.bytes_per_slot * self.capacity

    @property
    def physical_storage_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes()
            for tensor in self._tensors.values()
        )

    def component_tensor(
        self,
        layer_index: int,
        role: HybridStateRole,
    ) -> torch.Tensor:
        try:
            return self._tensors[(layer_index, role)]
        except KeyError as error:
            raise KeyError(
                f"unknown hybrid state component: {(layer_index, role)}"
            ) from error

    def activate(self, lease: HybridStateLease) -> None:
        self._validate_slot_id(lease.slot_id)
        binding = (lease.request_id, lease.generation)
        current = self._bindings.get(lease.slot_id)
        if current is not None:
            if current != binding:
                raise RuntimeError(
                    "hybrid state tensor slot already bound: "
                    f"slot={lease.slot_id}, current={current}, requested={binding}"
                )
            return
        self._zero_slot(lease.slot_id)
        self._bindings[lease.slot_id] = binding

    def validate(self, lease: HybridStateLease) -> int:
        self._validate_slot_id(lease.slot_id)
        binding = self._bindings.get(lease.slot_id)
        expected = (lease.request_id, lease.generation)
        if binding != expected:
            raise RuntimeError(
                "hybrid state tensor lease mismatch: "
                f"slot={lease.slot_id}, current={binding}, expected={expected}"
            )
        return lease.slot_id

    def release(self, lease: HybridStateLease) -> None:
        slot_id = self.validate(lease)
        self._zero_slot(slot_id)
        del self._bindings[slot_id]

    def slot_ids(
        self,
        leases: list[HybridStateLease] | tuple[HybridStateLease, ...],
    ) -> torch.Tensor:
        return torch.tensor(
            [self.validate(lease) for lease in leases],
            dtype=torch.int32,
            device=self.device,
        )

    def _validate_slot_id(self, slot_id: int) -> None:
        if (
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            or slot_id >= self.capacity
        ):
            raise ValueError(f"hybrid state slot out of range: {slot_id}")

    def _zero_slot(self, slot_id: int) -> None:
        for tensor in self._tensors.values():
            tensor[slot_id].zero_()


class HybridStateRuntimeBridge:
    def __init__(self, pool: HybridStateTensorPool):
        self.pool = pool

    def prepare_batch(
        self,
        released_leases: tuple[HybridStateLease, ...],
        active_leases: tuple[HybridStateLease, ...],
    ) -> torch.Tensor:
        self.release(released_leases)
        for lease in active_leases:
            self.pool.activate(lease)
        return self.pool.slot_ids(active_leases)

    def release(
        self,
        released_leases: tuple[HybridStateLease, ...],
    ) -> None:
        for lease in released_leases:
            self.pool.release(lease)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result
