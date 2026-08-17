from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


def _nonnegative_integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ProposalKVEntryIdentity:
    logical_entry_id: int
    generation: int

    def __post_init__(self) -> None:
        _nonnegative_integer(
            self.logical_entry_id,
            "logical_entry_id",
        )
        _positive_integer(self.generation, "generation")


@dataclass(frozen=True)
class ProposalKVResidencyLease:
    identities: tuple[ProposalKVEntryIdentity, ...]
    physical_slot_ids: tuple[int, ...]
    occupancy_generations: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identities, tuple):
            raise ValueError("identities must be a tuple")
        if not isinstance(self.physical_slot_ids, tuple):
            raise ValueError("physical_slot_ids must be a tuple")
        if not isinstance(self.occupancy_generations, tuple):
            raise ValueError("occupancy_generations must be a tuple")
        if not (
            len(self.identities)
            == len(self.physical_slot_ids)
            == len(self.occupancy_generations)
        ):
            raise ValueError("lease tuple lengths must match")
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("lease contains duplicate identities")
        if len(set(self.physical_slot_ids)) != len(
            self.physical_slot_ids
        ):
            raise ValueError("lease contains duplicate physical slots")
        for physical_slot_id in self.physical_slot_ids:
            _nonnegative_integer(
                physical_slot_id,
                "physical_slot_id",
            )
        for generation in self.occupancy_generations:
            _positive_integer(
                generation,
                "occupancy_generation",
            )


class ProposalKVEntryAllocator(Protocol):

    def reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]: ...

    def ensure_writable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease: ...

    def ensure_readable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease: ...

    def record_write_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None: ...

    def record_read_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None: ...

    def commit_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> None: ...

    def retire_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        writeback: bool,
    ) -> None: ...

    def authority_snapshot(self) -> dict: ...


@dataclass
class _DirectEntry:
    identity: ProposalKVEntryIdentity
    physical_slot_id: int
    occupancy_generation: int
    state: str = "reserved"


class DirectProposalKVAllocator:

    def __init__(self, physical_store):
        reserve_slots = getattr(physical_store, "reserve_slots", None)
        release_slots = getattr(physical_store, "release_slots", None)
        if not callable(reserve_slots) or not callable(release_slots):
            raise ValueError(
                "physical_store must expose reserve_slots and release_slots"
            )
        self.physical_store = physical_store
        self.logical_capacity = _positive_integer(
            int(physical_store.capacity),
            "physical_store.capacity",
        )
        self._free_logical_ids = list(range(self.logical_capacity))
        self._generations = [0] * self.logical_capacity
        self._entries: dict[int, _DirectEntry] = {}
        self._slot_occupancy_generations: dict[int, int] = {}

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
    ) -> _DirectEntry:
        if identity.logical_entry_id >= self.logical_capacity:
            raise RuntimeError("proposal KV identity is stale")
        entry = self._entries.get(identity.logical_entry_id)
        if entry is None or entry.identity != identity:
            raise RuntimeError("proposal KV identity is stale")
        return entry

    def _lease(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        allowed_states: tuple[str, ...],
    ) -> ProposalKVResidencyLease:
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        for entry in entries:
            if entry.state not in allowed_states:
                raise RuntimeError(
                    "proposal KV entry state is not valid for this lease"
                )
            current_generation = (
                self._slot_occupancy_generations.get(
                    entry.physical_slot_id,
                )
            )
            if current_generation != entry.occupancy_generation:
                raise RuntimeError(
                    "proposal KV physical occupancy is stale"
                )
        return ProposalKVResidencyLease(
            identities=identities,
            physical_slot_ids=tuple(
                entry.physical_slot_id for entry in entries
            ),
            occupancy_generations=tuple(
                entry.occupancy_generation for entry in entries
            ),
        )

    def _validate_lease(
        self,
        lease: ProposalKVResidencyLease,
        *,
        allowed_states: tuple[str, ...],
    ) -> tuple[_DirectEntry, ...]:
        if not isinstance(lease, ProposalKVResidencyLease):
            raise ValueError(
                "lease must be a ProposalKVResidencyLease"
            )
        current = self._lease(
            lease.identities,
            allowed_states=allowed_states,
        )
        if current != lease:
            raise RuntimeError("proposal KV lease is stale")
        return tuple(
            self._entry(identity) for identity in lease.identities
        )

    def reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        count = _nonnegative_integer(count, "count")
        if count > len(self._free_logical_ids):
            raise RuntimeError("proposal KV logical entries are exhausted")
        physical_slot_ids = self.physical_store.reserve_slots(count)
        if (
            not isinstance(physical_slot_ids, tuple)
            or len(physical_slot_ids) != count
            or len(set(physical_slot_ids)) != count
        ):
            raise RuntimeError(
                "reserve_slots returned invalid physical slots"
            )
        logical_entry_ids = tuple(self._free_logical_ids[:count])
        del self._free_logical_ids[:count]
        identities = []
        for logical_entry_id, physical_slot_id in zip(
            logical_entry_ids,
            physical_slot_ids,
        ):
            _nonnegative_integer(
                physical_slot_id,
                "physical_slot_id",
            )
            self._generations[logical_entry_id] += 1
            identity = ProposalKVEntryIdentity(
                logical_entry_id=logical_entry_id,
                generation=self._generations[logical_entry_id],
            )
            occupancy_generation = (
                self._slot_occupancy_generations.get(
                    physical_slot_id,
                    0,
                )
                + 1
            )
            self._slot_occupancy_generations[
                physical_slot_id
            ] = occupancy_generation
            self._entries[logical_entry_id] = _DirectEntry(
                identity=identity,
                physical_slot_id=physical_slot_id,
                occupancy_generation=occupancy_generation,
            )
            identities.append(identity)
        return tuple(identities)

    def ensure_writable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease:
        return self._lease(
            identities,
            allowed_states=("reserved",),
        )

    def ensure_readable(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> ProposalKVResidencyLease:
        return self._lease(
            identities,
            allowed_states=("active_staged", "committed"),
        )

    def record_write_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None:
        entries = self._validate_lease(
            lease,
            allowed_states=("reserved",),
        )
        for entry in entries:
            entry.state = "active_staged"

    def record_read_complete(
        self,
        lease: ProposalKVResidencyLease,
    ) -> None:
        self._validate_lease(
            lease,
            allowed_states=("active_staged", "committed"),
        )

    def commit_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> None:
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        if any(entry.state != "active_staged" for entry in entries):
            raise RuntimeError(
                "only active staged proposal KV entries can commit"
            )
        for entry in entries:
            entry.state = "committed"

    def retire_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
        *,
        writeback: bool,
    ) -> None:
        if not isinstance(writeback, bool):
            raise ValueError("writeback must be a bool")
        if writeback:
            raise RuntimeError(
                "direct proposal KV allocation does not support writeback"
            )
        identities = self._identity_tuple(identities)
        entries = tuple(self._entry(identity) for identity in identities)
        physical_slot_ids = tuple(
            entry.physical_slot_id for entry in entries
        )
        self.physical_store.release_slots(physical_slot_ids)
        for entry in entries:
            logical_entry_id = entry.identity.logical_entry_id
            del self._entries[logical_entry_id]
            self._free_logical_ids.append(logical_entry_id)
        self._free_logical_ids.sort()

    def authority_snapshot(self) -> dict:
        state_counts = {
            "reserved_entry_count": 0,
            "active_staged_entry_count": 0,
            "committed_entry_count": 0,
        }
        for entry in self._entries.values():
            state_counts[f"{entry.state}_entry_count"] += 1
        snapshot = {
            "allocator_mode": "direct",
            "logical_entry_capacity": self.logical_capacity,
            "owned_entry_count": len(self._entries),
            "free_entry_count": len(self._free_logical_ids),
            "gpu_resident_entry_count": len(self._entries),
            "cpu_resident_entry_count": 0,
            "h2d_operation_count": 0,
            "h2d_entry_count": 0,
            "h2d_bytes": 0,
            "d2h_operation_count": 0,
            "d2h_entry_count": 0,
            "d2h_bytes": 0,
            "accepted_entry_copy_count": 0,
            "accepted_entry_replay_count": 0,
            "accepted_entry_rematerialization_count": 0,
            "rejected_entry_count": 0,
            "rejected_entry_d2h_count": 0,
            "rejected_entry_d2h_bytes": 0,
        }
        snapshot.update(state_counts)
        physical_snapshot = getattr(
            self.physical_store,
            "authority_snapshot",
            None,
        )
        if callable(physical_snapshot):
            snapshot["physical_store"] = physical_snapshot()
        return snapshot
