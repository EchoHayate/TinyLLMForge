from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tinyvllm.engine.proposal_kv_allocator import (
    ProposalKVEntryIdentity,
    ProposalKVResidencyLease,
)


class ProposalKVStorageAdapter(Protocol):
    logical_capacity: int
    gpu_capacity: int
    block_size: int
    dtype: object

    def entry_nbytes(self) -> int: ...

    def copy_gpu_to_cpu(
        self,
        rows: tuple[tuple[int, int], ...],
    ) -> None: ...

    def copy_cpu_to_gpu(
        self,
        rows: tuple[tuple[int, int], ...],
    ) -> None: ...


class ProposalKVCopyCompletion(Protocol):

    def query(self) -> bool: ...

    def wait_current_stream(self) -> None: ...


class ProposalKVCopyBackend(Protocol):

    def enqueue_h2d(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion: ...

    def enqueue_d2h(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion: ...

    def record_consumer_completion(
        self,
    ) -> ProposalKVCopyCompletion: ...


class _SynchronousCompletion:

    def query(self) -> bool:
        return True

    def wait_current_stream(self) -> None:
        return None


class SynchronousProposalKVCopyBackend:

    def enqueue_h2d(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion:
        storage.copy_cpu_to_gpu(rows)
        return _SynchronousCompletion()

    def enqueue_d2h(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion:
        storage.copy_gpu_to_cpu(rows)
        return _SynchronousCompletion()

    def record_consumer_completion(
        self,
    ) -> ProposalKVCopyCompletion:
        return _SynchronousCompletion()


class _TorchCompletion:

    def __init__(self, torch_module, event):
        self._torch = torch_module
        self._event = event

    def query(self) -> bool:
        return bool(self._event.query())

    def wait_current_stream(self) -> None:
        self._torch.cuda.current_stream().wait_event(self._event)


class TorchProposalKVCopyBackend:

    def __init__(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "proposal KV asynchronous copies require CUDA"
            )
        self._torch = torch
        self._copy_stream = torch.cuda.Stream()

    def _completion(self):
        event = self._torch.cuda.Event()
        event.record(self._copy_stream)
        return _TorchCompletion(self._torch, event)

    def enqueue_h2d(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion:
        with self._torch.cuda.stream(self._copy_stream):
            storage.copy_cpu_to_gpu(rows)
            completion = self._completion()
        completion.wait_current_stream()
        return completion

    def enqueue_d2h(
        self,
        storage: ProposalKVStorageAdapter,
        rows: tuple[tuple[int, int], ...],
    ) -> ProposalKVCopyCompletion:
        producer = self._torch.cuda.Event()
        producer.record(self._torch.cuda.current_stream())
        self._copy_stream.wait_event(producer)
        with self._torch.cuda.stream(self._copy_stream):
            storage.copy_gpu_to_cpu(rows)
            completion = self._completion()
        completion.wait_current_stream()
        return completion

    def record_consumer_completion(
        self,
    ) -> ProposalKVCopyCompletion:
        event = self._torch.cuda.Event()
        event.record(self._torch.cuda.current_stream())
        return _TorchCompletion(self._torch, event)


@dataclass
class _LogicalEntry:
    generation: int = 0
    logical_state: str = "free"
    residency_state: str = "none"
    physical_slot_id: int | None = None
    cpu_valid: bool = False
    dirty: bool = False
    last_access_ordinal: int = 0
    pending_completion: object | None = None


@dataclass
class _PhysicalOccupancy:
    generation: int = 0
    identity: ProposalKVEntryIdentity | None = None
    consumer_completion: object | None = None
    retiring: bool = False


class _ProposalKVBlockwiseAttentionAdapter:

    _STAT_NAMES = (
        "prefetch_plans",
        "prefetch_read_blocks",
        "decode_plan_builds",
        "decode_plan_cache_hits",
        "decode_plan_identity_invalidations",
        "decode_windows_with_spare_capacity",
        "decode_cross_layer_hint_blocks",
        "decode_cross_layer_hint_resident",
        "decode_cross_layer_hint_retained",
    )

    def __init__(self, manager: "ProposalKVResidencyManager"):
        self._manager = manager
        self.stats = {name: 0 for name in self._STAT_NAMES}

    @property
    def gpu_blocks(self) -> int:
        return self._manager.gpu_capacity

    @property
    def logical_to_slot(self) -> dict[int, int]:
        mapping = {}
        for slot_id, occupancy in enumerate(
            self._manager._occupancies
        ):
            identity = occupancy.identity
            if identity is None or occupancy.retiring:
                continue
            entry = self._manager._entry(identity)
            if entry.physical_slot_id != slot_id:
                raise RuntimeError(
                    "proposal KV occupancy mapping is stale"
                )
            mapping[identity.logical_entry_id] = slot_id
        return mapping

    @property
    def pending_wait_blocks(self) -> set[int]:
        return {
            logical_entry_id
            for logical_entry_id, entry in enumerate(
                self._manager._entries
            )
            if entry.residency_state == "h2d_pending"
        }

    def _identity(self, logical_entry_id: int) -> ProposalKVEntryIdentity:
        if (
            isinstance(logical_entry_id, bool)
            or not isinstance(logical_entry_id, int)
            or logical_entry_id < 0
            or logical_entry_id >= self._manager.logical_capacity
        ):
            raise RuntimeError(
                "proposal KV logical entry id is invalid"
            )
        entry = self._manager._entries[logical_entry_id]
        identity = ProposalKVEntryIdentity(
            logical_entry_id=logical_entry_id,
            generation=entry.generation,
        )
        self._manager._entry(identity)
        return identity

    def _identities(
        self,
        logical_entry_ids,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        ordered = []
        seen = set()
        for logical_entry_id in logical_entry_ids:
            logical_entry_id = int(logical_entry_id)
            if logical_entry_id < 0 or logical_entry_id in seen:
                continue
            ordered.append(self._identity(logical_entry_id))
            seen.add(logical_entry_id)
        return tuple(ordered)

    def _touch(self, slot_id: int) -> None:
        if (
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            or slot_id >= self.gpu_blocks
        ):
            raise RuntimeError("proposal KV physical slot is invalid")
        occupancy = self._manager._occupancies[slot_id]
        if occupancy.identity is None or occupancy.retiring:
            raise RuntimeError("proposal KV physical slot is not resident")
        entry = self._manager._entry(occupancy.identity)
        if entry.physical_slot_id != slot_id:
            raise RuntimeError("proposal KV occupancy mapping is stale")
        self._manager._touch(entry)

    def ensure_resident(
        self,
        logical_blocks,
        require_valid: bool,
        future_logical_blocks=None,
        protected_logical_blocks=None,
    ) -> dict[int, int]:
        if not isinstance(require_valid, bool):
            raise ValueError("require_valid must be a bool")
        identities = self._identities(logical_blocks)
        if not identities:
            return {}
        if not require_valid:
            raise RuntimeError(
                "proposal KV blockwise reads require valid entries"
            )
        future_identities = self._identities(
            future_logical_blocks or ()
        )
        protected_identities = self._identities(
            protected_logical_blocks or ()
        )
        protected_ids = {
            identity.logical_entry_id for identity in identities
        }
        protected_ids.update(
            identity.logical_entry_id
            for identity in protected_identities
        )
        if len(protected_ids) > self.gpu_blocks:
            raise RuntimeError(
                "proposal KV blockwise staging capacity exceeded: "
                f"required={len(protected_ids)}, "
                f"gpu_blocks={self.gpu_blocks}"
            )
        resident = []
        missing = []
        for identity in identities:
            entry = self._manager._entry(identity)
            if entry.physical_slot_id is None:
                missing.append(identity)
                continue
            if entry.logical_state not in (
                "reserved",
                "active_staged",
                "committed",
            ):
                raise RuntimeError(
                    "proposal KV entry state is not readable"
                )
            resident.append(entry)
        if missing:
            self._manager._ensure_readable(
                tuple(missing),
                protected_logical_entry_ids=protected_ids,
                future_logical_entry_ids={
                    identity.logical_entry_id
                    for identity in future_identities
                },
            )
        for entry in resident:
            self._touch(entry.physical_slot_id)
        mapping = self.logical_to_slot
        return {
            identity.logical_entry_id:
            mapping[identity.logical_entry_id]
            for identity in identities
        }

    def wait_for_blocks(
        self,
        logical_blocks,
        clear_pending: bool = False,
    ) -> None:
        if not isinstance(clear_pending, bool):
            raise ValueError("clear_pending must be a bool")
        for identity in self._identities(logical_blocks):
            entry = self._manager._entry(identity)
            completion = entry.pending_completion
            if completion is not None:
                completion.wait_current_stream()

    def mark_dirty(self, logical_blocks) -> None:
        for identity in self._identities(logical_blocks):
            entry = self._manager._entry(identity)
            if entry.physical_slot_id is None:
                raise RuntimeError(
                    "proposal KV write entry is not GPU resident"
                )
            entry.residency_state = "gpu_dirty"
            entry.dirty = True

    def record_h2d_slot_read_window(self, **_kwargs) -> None:
        return None


class ProposalKVResidencyManager:

    _COUNTER_NAMES = (
        "logical_entries_reserved",
        "logical_entries_committed",
        "logical_entries_retired",
        "lease_read_count",
        "lease_write_count",
        "lease_stale_rejections",
        "gpu_slot_rebindings",
        "h2d_operation_count",
        "h2d_entry_count",
        "h2d_bytes",
        "d2h_operation_count",
        "d2h_entry_count",
        "d2h_bytes",
        "dirty_writeback_entry_count",
        "clean_eviction_entry_count",
        "rejected_entry_count",
        "rejected_entry_d2h_count",
        "rejected_entry_d2h_bytes",
        "accepted_entry_copy_count",
        "accepted_entry_replay_count",
        "accepted_entry_rematerialization_count",
        "retirement_wait_count",
    )

    def __init__(
        self,
        *,
        storage: ProposalKVStorageAdapter,
        copy_backend: ProposalKVCopyBackend,
        batch_copy: bool = True,
    ):
        logical_capacity = int(storage.logical_capacity)
        gpu_capacity = int(storage.gpu_capacity)
        if logical_capacity <= gpu_capacity or gpu_capacity <= 0:
            raise ValueError(
                "logical capacity must exceed positive GPU capacity"
            )
        if int(storage.block_size) != 1:
            raise ValueError("proposal KV residency requires block size one")
        entry_nbytes = int(storage.entry_nbytes())
        if entry_nbytes <= 0:
            raise ValueError("entry_nbytes must be positive")
        if not isinstance(batch_copy, bool):
            raise ValueError("batch_copy must be a bool")
        self.storage = storage
        self.copy_backend = copy_backend
        self.batch_copy = batch_copy
        self.logical_capacity = logical_capacity
        self.gpu_capacity = gpu_capacity
        self._entry_nbytes = entry_nbytes
        self._entries = [
            _LogicalEntry() for _ in range(logical_capacity)
        ]
        self._occupancies = [
            _PhysicalOccupancy() for _ in range(gpu_capacity)
        ]
        self._free_logical_ids = list(range(logical_capacity))
        self._free_gpu_slot_ids = list(range(gpu_capacity))
        self._access_ordinal = 0
        self._counters = {
            name: 0 for name in self._COUNTER_NAMES
        }
        self.blockwise_attention_adapter = (
            _ProposalKVBlockwiseAttentionAdapter(self)
        )

    @staticmethod
    def _identity_tuple(
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        if not isinstance(identities, tuple):
            raise ValueError("identities must be a tuple")
        if len(set(identities)) != len(identities):
            raise ValueError("identities must not contain duplicates")
        if not all(
            isinstance(identity, ProposalKVEntryIdentity)
            for identity in identities
        ):
            raise ValueError(
                "identities must contain ProposalKVEntryIdentity values"
            )
        return identities

    def _entry(
        self,
        identity: ProposalKVEntryIdentity,
    ) -> _LogicalEntry:
        if identity.logical_entry_id >= self.logical_capacity:
            raise RuntimeError("proposal KV identity is stale")
        entry = self._entries[identity.logical_entry_id]
        if (
            entry.logical_state == "free"
            or entry.generation != identity.generation
        ):
            raise RuntimeError("proposal KV identity is stale")
        return entry

    def _touch(self, entry: _LogicalEntry) -> None:
        self._access_ordinal += 1
        entry.last_access_ordinal = self._access_ordinal

    @staticmethod
    def _completion_ready(completion) -> bool:
        return completion is None or bool(completion.query())

    def _bind(
        self,
        identity: ProposalKVEntryIdentity,
        physical_slot_id: int,
    ) -> None:
        entry = self._entry(identity)
        occupancy = self._occupancies[physical_slot_id]
        if occupancy.identity is not None:
            raise RuntimeError("proposal KV GPU slot is occupied")
        occupancy.generation += 1
        occupancy.identity = identity
        occupancy.consumer_completion = None
        occupancy.retiring = False
        entry.physical_slot_id = physical_slot_id
        self._counters["gpu_slot_rebindings"] += 1

    def _unbind(self, physical_slot_id: int) -> None:
        occupancy = self._occupancies[physical_slot_id]
        identity = occupancy.identity
        if identity is None:
            raise RuntimeError("proposal KV GPU slot is already free")
        entry = self._entries[identity.logical_entry_id]
        if (
            entry.generation == identity.generation
            and entry.physical_slot_id == physical_slot_id
        ):
            entry.physical_slot_id = None
        occupancy.identity = None
        occupancy.consumer_completion = None
        occupancy.retiring = False
        self._free_gpu_slot_ids.append(physical_slot_id)
        self._free_gpu_slot_ids.sort()

    def _victims(
        self,
        count: int,
        *,
        protected_logical_entry_ids: set[int] | None = None,
        future_logical_entry_ids: set[int] | None = None,
        allow_active_staged: bool = False,
    ) -> tuple[_LogicalEntry, ...]:
        protected_logical_entry_ids = (
            protected_logical_entry_ids or set()
        )
        future_logical_entry_ids = (
            future_logical_entry_ids or set()
        )
        candidates = []
        for logical_entry_id, entry in enumerate(self._entries):
            evictable_logical_states = (
                ("active_staged", "committed")
                if allow_active_staged
                else ("committed",)
            )
            if (
                logical_entry_id in protected_logical_entry_ids
                or
                entry.logical_state not in evictable_logical_states
                or entry.physical_slot_id is None
                or entry.residency_state
                not in ("gpu_dirty", "gpu_clean")
            ):
                continue
            occupancy = self._occupancies[entry.physical_slot_id]
            if (
                occupancy.retiring
                or not self._completion_ready(
                    occupancy.consumer_completion
                )
            ):
                continue
            candidates.append(
                (
                    logical_entry_id in future_logical_entry_ids,
                    entry.last_access_ordinal,
                    logical_entry_id,
                    entry,
                )
            )
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2])
        )
        if len(candidates) < count:
            raise RuntimeError(
                "no eligible committed proposal KV eviction victim"
            )
        return tuple(item[3] for item in candidates[:count])

    def _enqueue_d2h(
        self,
        entries: tuple[_LogicalEntry, ...],
    ) -> None:
        dirty_entries = tuple(entry for entry in entries if entry.dirty)
        if not dirty_entries:
            return
        groups = (
            (dirty_entries,)
            if self.batch_copy
            else tuple((entry,) for entry in dirty_entries)
        )
        for group in groups:
            rows = tuple(
                (
                    self._occupancies[
                        entry.physical_slot_id
                    ].identity.logical_entry_id,
                    entry.physical_slot_id,
                )
                for entry in group
            )
            completion = self.copy_backend.enqueue_d2h(
                self.storage,
                rows,
            )
            completion.wait_current_stream()
            self._counters["d2h_operation_count"] += 1
            self._counters["d2h_entry_count"] += len(group)
            self._counters["d2h_bytes"] += (
                len(group) * self._entry_nbytes
            )
            self._counters["dirty_writeback_entry_count"] += len(
                group
            )
            for entry in group:
                entry.pending_completion = completion
                entry.cpu_valid = True
                entry.dirty = False

    def _evict(
        self,
        count: int,
        *,
        protected_logical_entry_ids: set[int] | None = None,
        future_logical_entry_ids: set[int] | None = None,
        allow_active_staged: bool = False,
    ) -> None:
        victims = self._victims(
            count,
            protected_logical_entry_ids=(
                protected_logical_entry_ids
            ),
            future_logical_entry_ids=future_logical_entry_ids,
            allow_active_staged=allow_active_staged,
        )
        self._enqueue_d2h(victims)
        for entry in victims:
            if not entry.cpu_valid:
                raise RuntimeError(
                    "proposal KV victim has no authoritative CPU copy"
                )
            if entry.residency_state == "gpu_clean":
                self._counters["clean_eviction_entry_count"] += 1
            physical_slot_id = entry.physical_slot_id
            entry.residency_state = "cpu_clean"
            entry.pending_completion = None
            self._unbind(physical_slot_id)

    def _acquire_slots(
        self,
        count: int,
        *,
        protected_logical_entry_ids: set[int] | None = None,
        future_logical_entry_ids: set[int] | None = None,
        allow_active_staged: bool = False,
    ) -> tuple[int, ...]:
        missing = count - len(self._free_gpu_slot_ids)
        if missing > 0:
            self._evict(
                missing,
                protected_logical_entry_ids=(
                    protected_logical_entry_ids
                ),
                future_logical_entry_ids=future_logical_entry_ids,
                allow_active_staged=allow_active_staged,
            )
        result = tuple(self._free_gpu_slot_ids[:count])
        del self._free_gpu_slot_ids[:count]
        return result

    def _lease(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease:
        physical_slot_ids = []
        occupancy_generations = []
        for identity in identities:
            entry = self._entry(identity)
            if entry.physical_slot_id is None:
                raise RuntimeError("proposal KV entry is not GPU resident")
            occupancy = self._occupancies[entry.physical_slot_id]
            if occupancy.identity != identity:
                raise RuntimeError("proposal KV occupancy is stale")
            physical_slot_ids.append(entry.physical_slot_id)
            occupancy_generations.append(occupancy.generation)
        return ProposalKVResidencyLease(
            identities=identities,
            physical_slot_ids=tuple(physical_slot_ids),
            occupancy_generations=tuple(occupancy_generations),
        )

    def _validate_lease(
        self,
        lease: ProposalKVResidencyLease,
        *,
        allowed_states: tuple[str, ...],
    ) -> tuple[_LogicalEntry, ...]:
        if not isinstance(lease, ProposalKVResidencyLease):
            raise ValueError(
                "lease must be a ProposalKVResidencyLease"
            )
        try:
            entries = tuple(
                self._entry(identity) for identity in lease.identities
            )
            if len(entries) != len(lease.physical_slot_ids):
                raise RuntimeError("proposal KV lease is stale")
            for identity, entry, slot_id, occupancy_generation in zip(
                lease.identities,
                entries,
                lease.physical_slot_ids,
                lease.occupancy_generations,
            ):
                occupancy = self._occupancies[slot_id]
                if (
                    entry.logical_state not in allowed_states
                    or entry.physical_slot_id != slot_id
                    or occupancy.identity != identity
                    or occupancy.generation != occupancy_generation
                ):
                    raise RuntimeError("proposal KV lease is stale")
            return entries
        except (IndexError, RuntimeError):
            self._counters["lease_stale_rejections"] += 1
            raise RuntimeError("proposal KV lease is stale") from None

    def reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("count must be a nonnegative integer")
        self.poll_retirements()
        if count > len(self._free_logical_ids):
            raise RuntimeError("proposal KV logical entries are exhausted")
        logical_entry_ids = tuple(self._free_logical_ids[:count])
        del self._free_logical_ids[:count]
        identities = []
        for logical_entry_id in logical_entry_ids:
            entry = self._entries[logical_entry_id]
            entry.generation += 1
            entry.logical_state = "reserved"
            entry.residency_state = "none"
            entry.physical_slot_id = None
            entry.cpu_valid = False
            entry.dirty = False
            entry.last_access_ordinal = 0
            entry.pending_completion = None
            identities.append(
                ProposalKVEntryIdentity(
                    logical_entry_id=logical_entry_id,
                    generation=entry.generation,
                )
            )
        self._counters["logical_entries_reserved"] += count
        return tuple(identities)

    def ensure_writable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease:
        return self._ensure_writable(
            identities,
            allow_active_staged_eviction=False,
        )

    def ensure_blockwise_writable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease:
        return self._ensure_writable(
            identities,
            allow_active_staged_eviction=True,
        )

    def _ensure_writable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        allow_active_staged_eviction: bool,
    ) -> ProposalKVResidencyLease:
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        if any(
            entry.logical_state != "reserved"
            or entry.physical_slot_id is not None
            for entry in entries
        ):
            raise RuntimeError(
                "only unbound reserved proposal KV entries are writable"
            )
        physical_slot_ids = self._acquire_slots(
            len(entries),
            allow_active_staged=allow_active_staged_eviction,
        )
        for identity, entry, physical_slot_id in zip(
            identities,
            entries,
            physical_slot_ids,
        ):
            self._bind(identity, physical_slot_id)
            entry.residency_state = "gpu_dirty"
        self._counters["lease_write_count"] += len(entries)
        return self._lease(identities)

    def ensure_readable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease:
        return self._ensure_readable(identities)

    def _ensure_readable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        protected_logical_entry_ids: set[int] | None = None,
        future_logical_entry_ids: set[int] | None = None,
    ) -> ProposalKVResidencyLease:
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        if any(
            entry.logical_state
            not in ("active_staged", "committed")
            for entry in entries
        ):
            raise RuntimeError(
                "proposal KV entry state is not readable"
            )
        missing = tuple(
            (identity, entry)
            for identity, entry in zip(identities, entries)
            if entry.physical_slot_id is None
        )
        if missing:
            if any(
                entry.logical_state
                not in ("active_staged", "committed")
                or entry.residency_state != "cpu_clean"
                or not entry.cpu_valid
                for _, entry in missing
            ):
                raise RuntimeError(
                    "proposal KV entry has no readable residency"
                )
            protected_ids = {
                identity.logical_entry_id for identity in identities
            }
            protected_ids.update(
                protected_logical_entry_ids or set()
            )
            slots = self._acquire_slots(
                len(missing),
                protected_logical_entry_ids=protected_ids,
                future_logical_entry_ids=(
                    future_logical_entry_ids
                ),
                allow_active_staged=True,
            )
            for (identity, entry), slot_id in zip(missing, slots):
                self._bind(identity, slot_id)
                entry.residency_state = "h2d_pending"
            groups = (
                (missing,)
                if self.batch_copy
                else tuple((item,) for item in missing)
            )
            for group in groups:
                rows = tuple(
                    (
                        identity.logical_entry_id,
                        entry.physical_slot_id,
                    )
                    for identity, entry in group
                )
                completion = self.copy_backend.enqueue_h2d(
                    self.storage,
                    rows,
                )
                completion.wait_current_stream()
                self._counters["h2d_operation_count"] += 1
                self._counters["h2d_entry_count"] += len(group)
                self._counters["h2d_bytes"] += (
                    len(group) * self._entry_nbytes
                )
                for _, entry in group:
                    entry.residency_state = "gpu_clean"
                    entry.pending_completion = completion
        for entry in entries:
            self._touch(entry)
        self._counters["lease_read_count"] += len(entries)
        return self._lease(identities)

    def record_write_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None:
        entries = self._validate_lease(
            lease,
            allowed_states=("reserved",),
        )
        completion = self.copy_backend.record_consumer_completion()
        for entry, physical_slot_id in zip(
            entries,
            lease.physical_slot_ids,
        ):
            entry.logical_state = "active_staged"
            entry.residency_state = "gpu_dirty"
            entry.cpu_valid = False
            entry.dirty = True
            entry.pending_completion = completion
            self._occupancies[
                physical_slot_id
            ].consumer_completion = completion

    def record_read_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None:
        entries = self._validate_lease(
            lease,
            allowed_states=("active_staged", "committed"),
        )
        completion = self.copy_backend.record_consumer_completion()
        for entry, physical_slot_id in zip(
            entries,
            lease.physical_slot_ids,
        ):
            entry.pending_completion = completion
            self._occupancies[
                physical_slot_id
            ].consumer_completion = completion

    def commit_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> None:
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        if any(
            entry.logical_state != "active_staged"
            for entry in entries
        ):
            raise RuntimeError(
                "only active staged proposal KV entries can commit"
            )
        for entry in entries:
            entry.logical_state = "committed"
            self._touch(entry)
        self._counters["logical_entries_committed"] += len(entries)

    def _finish_retirement(
        self,
        logical_entry_id: int,
        entry: _LogicalEntry,
    ) -> None:
        physical_slot_id = entry.physical_slot_id
        if physical_slot_id is not None:
            self._unbind(physical_slot_id)
        entry.logical_state = "free"
        entry.residency_state = "none"
        entry.physical_slot_id = None
        entry.cpu_valid = False
        entry.dirty = False
        entry.last_access_ordinal = 0
        entry.pending_completion = None
        self._free_logical_ids.append(logical_entry_id)
        self._free_logical_ids.sort()

    def retire_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        writeback: bool,
    ) -> None:
        if not isinstance(writeback, bool):
            raise ValueError("writeback must be a bool")
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        for identity, entry in zip(identities, entries):
            previous_state = entry.logical_state
            if previous_state == "retiring":
                raise RuntimeError("proposal KV entry is already retiring")
            if previous_state in ("reserved", "active_staged"):
                self._counters["rejected_entry_count"] += 1
            if writeback and entry.dirty and entry.physical_slot_id is not None:
                self._enqueue_d2h((entry,))
            entry.logical_state = "retiring"
            self._counters["logical_entries_retired"] += 1
            physical_slot_id = entry.physical_slot_id
            completion = None
            if physical_slot_id is not None:
                occupancy = self._occupancies[physical_slot_id]
                occupancy.retiring = True
                completion = occupancy.consumer_completion
            if self._completion_ready(completion):
                self._finish_retirement(
                    identity.logical_entry_id,
                    entry,
                )
            else:
                self._counters["retirement_wait_count"] += 1

    def poll_retirements(self) -> None:
        for logical_entry_id, entry in enumerate(self._entries):
            if entry.logical_state != "retiring":
                continue
            physical_slot_id = entry.physical_slot_id
            completion = (
                None
                if physical_slot_id is None
                else self._occupancies[
                    physical_slot_id
                ].consumer_completion
            )
            if self._completion_ready(completion):
                self._finish_retirement(logical_entry_id, entry)

    def authority_snapshot(self) -> dict:
        snapshot = dict(self._counters)
        snapshot["allocator_mode"] = "residency"
        snapshot["logical_entry_capacity"] = self.logical_capacity
        snapshot["gpu_slot_capacity"] = self.gpu_capacity
        snapshot["free_logical_entry_count"] = len(
            self._free_logical_ids
        )
        snapshot["free_gpu_slot_count"] = len(
            self._free_gpu_slot_ids
        )
        for state in (
            "reserved",
            "active_staged",
            "committed",
            "retiring",
        ):
            snapshot[f"{state}_entry_count"] = sum(
                entry.logical_state == state
                for entry in self._entries
            )
        for state in (
            "gpu_dirty",
            "gpu_clean",
            "cpu_clean",
            "h2d_pending",
            "d2h_pending",
        ):
            snapshot[f"{state}_entry_count"] = sum(
                entry.residency_state == state
                for entry in self._entries
            )
        return snapshot
