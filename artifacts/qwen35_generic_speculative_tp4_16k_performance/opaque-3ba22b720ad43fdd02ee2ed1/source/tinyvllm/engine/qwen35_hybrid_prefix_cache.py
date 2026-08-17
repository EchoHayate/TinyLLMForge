from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from itertools import count
from typing import Optional

import torch

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)


_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


@dataclass(frozen=True)
class Qwen35HybridPrefixKey:
    token_hash: int
    token_count: int
    terminal_block_hash: int
    block_size: int
    model_fingerprint: str
    layout_fingerprint: str
    tensor_parallel_size: int
    dtype: torch.dtype


@dataclass(frozen=True)
class Qwen35HybridPrefixSnapshot:
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    convolution_states: tuple[torch.Tensor, ...]
    recurrent_states: tuple[torch.Tensor, ...]
    storage_bytes: int


@dataclass(frozen=True)
class Qwen35HybridPrefixPreparedPublication:
    publication_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    storage_bytes: int


@dataclass
class _PreparedPublicationState:
    handle: Qwen35HybridPrefixPreparedPublication
    convolution_states: tuple[torch.Tensor, ...]
    recurrent_states: tuple[torch.Tensor, ...]
    intern_candidates: tuple[tuple[torch.Tensor, "_TensorInternKey"], ...]
    unique_candidate_bytes: int
    phase: str = "prepared"
    snapshot: Optional[Qwen35HybridPrefixSnapshot] = None
    acquired: tuple[torch.Tensor, ...] = ()
    intern_counter_snapshot: Optional[dict[str, int]] = None
    reserved_new_bytes: int = 0
    previous_entry: Optional[Qwen35HybridPrefixSnapshot] = None
    previous_entry_index: Optional[int] = None
    counter_snapshot: Optional[dict[str, int]] = None
    evicted_entries: tuple[
        tuple[
            int,
            tuple[Qwen35HybridPrefixKey, tuple[int, ...]],
            Qwen35HybridPrefixSnapshot,
        ],
        ...,
    ] = ()


@dataclass(frozen=True)
class _TensorInternKey:
    dtype: torch.dtype
    shape: tuple[int, ...]
    device_type: str
    device_index: Optional[int]
    digest: str


@dataclass
class _InternedTensor:
    key: _TensorInternKey
    tensor: torch.Tensor
    refcount: int
    visible_refcount: int
    storage_bytes: int


def _tensor_digest(tensor):
    byte_view = tensor.detach().contiguous().view(torch.uint8)
    return sha256(byte_view.cpu().numpy().tobytes()).hexdigest()


def _tensor_bytes_equal(left, right):
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


class Qwen35HybridPrefixSnapshotCache:

    def __init__(
        self,
        state_transaction: Qwen35CrossLayerStateTransaction,
        *,
        max_entries: int,
        max_bytes: int,
    ):
        if not isinstance(
            state_transaction,
            Qwen35CrossLayerStateTransaction,
        ):
            raise ValueError(
                "state_transaction must be a "
                "Qwen35CrossLayerStateTransaction"
            )
        self.max_entries = self._positive_integer(
            max_entries,
            "max_entries",
        )
        self.max_bytes = self._positive_integer(
            max_bytes,
            "max_bytes",
        )
        self.state_transaction = state_transaction
        self._entries = OrderedDict()
        self._intern_table = {}
        self._intern_records = {}
        self._publication_ids = count()
        self._prepared_publication = None
        self._current_bytes = 0
        self._intern_total_bytes = 0
        self._current_logical_bytes = 0
        self._current_intern_references = 0
        self._peak_entries = 0
        self._peak_bytes = 0
        self._peak_logical_bytes = 0
        self._peak_prepared_bytes = 0
        self._counters = {
            "publishes": 0,
            "replacements": 0,
            "hits": 0,
            "misses": 0,
            "collision_misses": 0,
            "stale_block_misses": 0,
            "validation_failures": 0,
            "evictions": 0,
            "entry_limit_evictions": 0,
            "byte_limit_evictions": 0,
            "invalidations": 0,
            "failed_restores": 0,
            "oversize_rejections": 0,
            "clears": 0,
            "intern_hits": 0,
            "intern_misses": 0,
            "intern_collisions": 0,
            "publication_prepares": 0,
            "publication_precommits": 0,
            "publication_commits": 0,
            "publication_rollbacks": 0,
            "publication_prepare_conflicts": 0,
        }

    @staticmethod
    def _positive_integer(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _non_negative_integer(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @classmethod
    def _validate_key(cls, key):
        if not isinstance(key, Qwen35HybridPrefixKey):
            raise ValueError("key must be a Qwen35HybridPrefixKey")
        cls._non_negative_integer(key.token_hash, "token_hash")
        cls._positive_integer(key.token_count, "token_count")
        cls._non_negative_integer(
            key.terminal_block_hash,
            "terminal_block_hash",
        )
        cls._positive_integer(key.block_size, "block_size")
        cls._positive_integer(
            key.tensor_parallel_size,
            "tensor_parallel_size",
        )
        for value, name in (
            (key.model_fingerprint, "model_fingerprint"),
            (key.layout_fingerprint, "layout_fingerprint"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if key.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {key.dtype}")
        if key.token_count % key.block_size != 0:
            raise ValueError(
                "token_count must end at a full block boundary"
            )

    @classmethod
    def _validate_tokens(cls, key, token_ids):
        if not isinstance(token_ids, tuple):
            raise ValueError("token_ids must be a tuple")
        if len(token_ids) != key.token_count:
            raise ValueError(
                "token_ids length must match key token_count"
            )
        for token_id in token_ids:
            cls._non_negative_integer(token_id, "token_id")

    @classmethod
    def _validate_block_identities(
        cls,
        block_identities,
        *,
        key=None,
    ):
        if not isinstance(block_identities, tuple) or not block_identities:
            raise ValueError(
                "block_identities must be a non-empty tuple"
            )
        block_ids = []
        for identity in block_identities:
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise ValueError(
                    "each block identity must be a three-item tuple"
                )
            block_id, generation, block_hash = identity
            cls._non_negative_integer(block_id, "block_id")
            cls._non_negative_integer(
                generation,
                "block_generation",
            )
            cls._non_negative_integer(block_hash, "block_hash")
            block_ids.append(block_id)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError(
                "block identities must reference unique block ids"
            )
        if key is not None:
            expected_blocks = key.token_count // key.block_size
            if len(block_identities) != expected_blocks:
                raise ValueError(
                    "block identity count must match token boundary"
                )
            if block_identities[-1][2] != key.terminal_block_hash:
                raise ValueError(
                    "terminal block hash must match key"
                )

    @classmethod
    def _validate_identity(
        cls,
        key,
        token_ids,
        block_identities,
    ):
        cls._validate_key(key)
        cls._validate_tokens(key, token_ids)
        cls._validate_block_identities(
            block_identities,
            key=key,
        )

    @staticmethod
    def _entry_key(key, token_ids):
        return key, token_ids

    @staticmethod
    def _owned_clone(tensor):
        return tensor.detach().clone().contiguous()

    @staticmethod
    def _storage_bytes(tensors):
        return sum(
            tensor.untyped_storage().nbytes()
            for tensor in tensors
        )

    @staticmethod
    def _intern_key(tensor):
        return _TensorInternKey(
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            digest=_tensor_digest(tensor),
        )

    @classmethod
    def _prepare_candidate_interning(cls, tensors):
        buckets = {}
        storage_bytes = 0
        prepared = []
        for tensor in tensors:
            key = cls._intern_key(tensor)
            prepared.append((tensor, key))
            bucket = buckets.setdefault(key, [])
            if any(
                _tensor_bytes_equal(tensor, existing)
                for existing in bucket
            ):
                continue
            bucket.append(tensor)
            storage_bytes += tensor.untyped_storage().nbytes()
        return tuple(prepared), storage_bytes

    def _acquire_interned_tensor(self, candidate, key=None):
        if key is None:
            key = self._intern_key(candidate)
        bucket = self._intern_table.setdefault(key, [])
        for record in bucket:
            if _tensor_bytes_equal(candidate, record.tensor):
                record.refcount += 1
                self._current_intern_references += 1
                self._counters["intern_hits"] += 1
                return record.tensor
        if bucket:
            self._counters["intern_collisions"] += 1
        storage_bytes = candidate.untyped_storage().nbytes()
        record = _InternedTensor(
            key=key,
            tensor=candidate,
            refcount=1,
            visible_refcount=0,
            storage_bytes=storage_bytes,
        )
        bucket.append(record)
        self._intern_records[id(candidate)] = record
        self._intern_total_bytes += storage_bytes
        self._current_intern_references += 1
        self._counters["intern_misses"] += 1
        return candidate

    def _release_interned_tensor(self, tensor, *, visible=False):
        record = self._intern_records[id(tensor)]
        if visible:
            record.visible_refcount -= 1
            if record.visible_refcount == 0:
                self._current_bytes -= record.storage_bytes
        record.refcount -= 1
        self._current_intern_references -= 1
        if record.refcount:
            return
        bucket = self._intern_table[record.key]
        bucket.remove(record)
        if not bucket:
            del self._intern_table[record.key]
        del self._intern_records[id(tensor)]
        self._intern_total_bytes -= record.storage_bytes

    def _mark_snapshot_visible(self, snapshot):
        for tensor in (
            *snapshot.convolution_states,
            *snapshot.recurrent_states,
        ):
            record = self._intern_records[id(tensor)]
            if record.visible_refcount == 0:
                self._current_bytes += record.storage_bytes
            record.visible_refcount += 1

    def _release_snapshot(self, snapshot):
        for tensor in (
            *snapshot.convolution_states,
            *snapshot.recurrent_states,
        ):
            self._release_interned_tensor(tensor, visible=True)
        self._current_logical_bytes -= snapshot.storage_bytes

    def _remove_entry(self, entry_key):
        snapshot = self._entries.pop(entry_key)
        self._release_snapshot(snapshot)
        return snapshot

    def _detach_entry(self, entry_key):
        snapshot = self._entries.pop(entry_key)
        for tensor in (
            *snapshot.convolution_states,
            *snapshot.recurrent_states,
        ):
            record = self._intern_records[id(tensor)]
            record.visible_refcount -= 1
            if record.visible_refcount == 0:
                self._current_bytes -= record.storage_bytes
        self._current_logical_bytes -= snapshot.storage_bytes
        return snapshot

    def _attach_entry(self, entry_key, snapshot, index=None):
        items = list(self._entries.items())
        if index is None:
            index = len(items)
        items.insert(index, (entry_key, snapshot))
        self._entries = OrderedDict(items)
        self._mark_snapshot_visible(snapshot)
        self._current_logical_bytes += snapshot.storage_bytes

    def _release_detached_snapshot(self, snapshot):
        for tensor in (
            *snapshot.convolution_states,
            *snapshot.recurrent_states,
        ):
            self._release_interned_tensor(tensor)

    def _evict_oldest(self, reason):
        entry_key = next(iter(self._entries))
        self._remove_entry(entry_key)
        self._counters["evictions"] += 1
        self._counters[reason] += 1

    def _enforce_limits(self):
        while len(self._entries) > self.max_entries:
            self._evict_oldest("entry_limit_evictions")
        while self._current_bytes > self.max_bytes:
            self._evict_oldest("byte_limit_evictions")

    def _enforce_limits_reversible(self):
        evicted = []
        while len(self._entries) > self.max_entries:
            entry_key = next(iter(self._entries))
            snapshot = self._detach_entry(entry_key)
            evicted.append((0, entry_key, snapshot))
            self._counters["evictions"] += 1
            self._counters["entry_limit_evictions"] += 1
        while self._current_bytes > self.max_bytes:
            entry_key = next(iter(self._entries))
            snapshot = self._detach_entry(entry_key)
            evicted.append((0, entry_key, snapshot))
            self._counters["evictions"] += 1
            self._counters["byte_limit_evictions"] += 1
        return tuple(evicted)

    def prepare_publication(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        lease: HybridStateLease,
    ) -> Optional[Qwen35HybridPrefixPreparedPublication]:
        if self._prepared_publication is not None:
            self._counters["publication_prepare_conflicts"] += 1
            raise RuntimeError(
                "a hybrid prefix publication is already prepared"
            )
        try:
            self._validate_identity(
                key,
                token_ids,
                block_identities,
            )
            if not isinstance(lease, HybridStateLease):
                raise ValueError("lease must be a HybridStateLease")
            gathered = self.state_transaction.gather((lease,))
            convolution_states = tuple(
                self._owned_clone(convolution[0])
                for convolution, _ in gathered
            )
            recurrent_states = tuple(
                self._owned_clone(recurrent[0])
                for _, recurrent in gathered
            )
            for adapter, convolution, recurrent in zip(
                self.state_transaction.adapters,
                convolution_states,
                recurrent_states,
            ):
                slot_id = adapter.pool.validate(lease)
                adapter._validate_candidate(
                    convolution,
                    adapter.convolution[slot_id],
                    "convolution_state",
                )
                adapter._validate_candidate(
                    recurrent,
                    adapter.recurrent[slot_id],
                    "recurrent_state",
                )
        except (ValueError, RuntimeError):
            self._counters["validation_failures"] += 1
            raise

        storage_bytes = self._storage_bytes(
            (*convolution_states, *recurrent_states)
        )
        intern_candidates, unique_candidate_bytes = (
            self._prepare_candidate_interning(
            (*convolution_states, *recurrent_states)
            )
        )
        if unique_candidate_bytes > self.max_bytes:
            self._counters["oversize_rejections"] += 1
            return None
        handle = Qwen35HybridPrefixPreparedPublication(
            publication_id=next(self._publication_ids),
            key=key,
            token_ids=token_ids,
            block_identities=block_identities,
            storage_bytes=storage_bytes,
        )
        self._prepared_publication = _PreparedPublicationState(
            handle=handle,
            convolution_states=convolution_states,
            recurrent_states=recurrent_states,
            intern_candidates=intern_candidates,
            unique_candidate_bytes=unique_candidate_bytes,
        )
        self._peak_prepared_bytes = max(
            self._peak_prepared_bytes,
            storage_bytes,
        )
        self._counters["publication_prepares"] += 1
        return handle

    def _prepared_state(self, prepared):
        if not isinstance(
            prepared,
            Qwen35HybridPrefixPreparedPublication,
        ):
            raise ValueError(
                "prepared must be a "
                "Qwen35HybridPrefixPreparedPublication"
            )
        current = self._prepared_publication
        if current is None or current.handle is not prepared:
            raise RuntimeError(
                "prepared publication is not current for this cache"
            )
        return current

    def rollback_publication(self, prepared) -> None:
        state = self._prepared_state(prepared)
        if state.phase == "precommitted":
            for tensor in reversed(state.acquired):
                self._release_interned_tensor(tensor)
            self._counters.update(state.intern_counter_snapshot)
        elif state.phase == "finalized_unsealed":
            entry_key = self._entry_key(
                prepared.key,
                prepared.token_ids,
            )
            current = self._entries.get(entry_key)
            if current is not state.snapshot:
                raise RuntimeError(
                    "finalized publication entry changed before rollback"
                )
            self._remove_entry(entry_key)
            if state.previous_entry is not None:
                self._attach_entry(
                    entry_key,
                    state.previous_entry,
                    state.previous_entry_index,
                )
            for index, evicted_key, evicted_snapshot in reversed(
                state.evicted_entries
            ):
                self._attach_entry(
                    evicted_key,
                    evicted_snapshot,
                    index,
                )
            self._counters.update(state.counter_snapshot)
        self._prepared_publication = None
        self._counters["publication_rollbacks"] += 1

    def abort_current_publication(self) -> bool:
        state = self._prepared_publication
        if state is None:
            return False
        self.rollback_publication(state.handle)
        return True

    def precommit_publication(self, prepared) -> None:
        state = self._prepared_state(prepared)
        if state.phase == "precommitted":
            return
        if state.phase != "prepared":
            raise RuntimeError(
                f"prepared publication phase is invalid: {state.phase}"
            )
        intern_candidates = state.intern_candidates
        acquired = []
        intern_counter_snapshot = {
            name: self._counters[name]
            for name in (
                "intern_hits",
                "intern_misses",
                "intern_collisions",
            )
        }
        intern_bytes_before = self._intern_total_bytes
        try:
            interned_convolution_states = []
            convolution_count = len(state.convolution_states)
            for tensor, intern_key in intern_candidates[:convolution_count]:
                interned = self._acquire_interned_tensor(
                    tensor,
                    intern_key,
                )
                acquired.append(interned)
                interned_convolution_states.append(interned)
            interned_recurrent_states = []
            for tensor, intern_key in intern_candidates[convolution_count:]:
                interned = self._acquire_interned_tensor(
                    tensor,
                    intern_key,
                )
                acquired.append(interned)
                interned_recurrent_states.append(interned)
            snapshot = Qwen35HybridPrefixSnapshot(
                key=prepared.key,
                token_ids=prepared.token_ids,
                block_identities=prepared.block_identities,
                convolution_states=tuple(interned_convolution_states),
                recurrent_states=tuple(interned_recurrent_states),
                storage_bytes=prepared.storage_bytes,
            )
        except (ValueError, RuntimeError):
            for tensor in reversed(acquired):
                self._release_interned_tensor(tensor)
            self._counters.update(intern_counter_snapshot)
            raise
        state.phase = "precommitted"
        state.snapshot = snapshot
        state.acquired = tuple(acquired)
        state.intern_counter_snapshot = intern_counter_snapshot
        state.reserved_new_bytes = (
            self._intern_total_bytes - intern_bytes_before
        )
        self._counters["publication_precommits"] += 1

    def finalize_publication(self, prepared) -> bool:
        state = self._prepared_state(prepared)
        if state.phase != "precommitted" or state.snapshot is None:
            raise RuntimeError(
                "prepared publication must be precommitted before finalize"
            )
        snapshot = state.snapshot
        entry_key = self._entry_key(
            prepared.key,
            prepared.token_ids,
        )
        state.counter_snapshot = dict(self._counters)
        if entry_key in self._entries:
            state.previous_entry_index = tuple(
                self._entries
            ).index(entry_key)
            state.previous_entry = self._detach_entry(entry_key)
            self._counters["replacements"] += 1
        self._mark_snapshot_visible(snapshot)
        self._entries[entry_key] = snapshot
        self._current_logical_bytes += prepared.storage_bytes
        self._counters["publishes"] += 1
        self._counters["publication_commits"] += 1
        state.evicted_entries = self._enforce_limits_reversible()
        self._peak_entries = max(
            self._peak_entries,
            len(self._entries),
        )
        self._peak_bytes = max(
            self._peak_bytes,
            self._current_bytes,
        )
        self._peak_logical_bytes = max(
            self._peak_logical_bytes,
            self._current_logical_bytes,
        )
        state.phase = "finalized_unsealed"
        return entry_key in self._entries

    def seal_publication(self, prepared) -> None:
        state = self._prepared_state(prepared)
        if state.phase != "finalized_unsealed":
            raise RuntimeError(
                "prepared publication must be finalized before seal"
            )
        if state.previous_entry is not None:
            self._release_detached_snapshot(state.previous_entry)
        for _, _, snapshot in state.evicted_entries:
            self._release_detached_snapshot(snapshot)
        self._prepared_publication = None

    def commit_publication(self, prepared) -> bool:
        self.precommit_publication(prepared)
        retained = self.finalize_publication(prepared)
        self.seal_publication(prepared)
        return retained

    def publish(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        lease: HybridStateLease,
    ) -> bool:
        prepared = self.prepare_publication(
            key,
            token_ids,
            block_identities,
            lease,
        )
        if prepared is None:
            return False
        try:
            return self.commit_publication(prepared)
        except BaseException:
            if (
                self._prepared_publication is not None
                and self._prepared_publication.handle is prepared
            ):
                self.rollback_publication(prepared)
            raise

    def _lookup(self, key, token_ids):
        entry_key = self._entry_key(key, token_ids)
        snapshot = self._entries.get(entry_key)
        if snapshot is not None:
            return entry_key, snapshot
        if any(existing_key == key for existing_key, _ in self._entries):
            self._counters["collision_misses"] += 1
        self._counters["misses"] += 1
        return None, None

    def acquire(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        leases: tuple[HybridStateLease, ...],
    ) -> bool:
        try:
            self._validate_key(key)
            self._validate_tokens(key, token_ids)
            self._validate_block_identities(
                block_identities,
            )
        except ValueError:
            self._counters["validation_failures"] += 1
            raise
        entry_key, snapshot = self._lookup(key, token_ids)
        if snapshot is None:
            return False
        if snapshot.block_identities != block_identities:
            self._remove_entry(entry_key)
            self._counters["stale_block_misses"] += 1
            self._counters["misses"] += 1
            return False
        if not isinstance(leases, tuple) or not leases:
            self._counters["validation_failures"] += 1
            raise ValueError("leases must be a non-empty tuple")

        batch_size = len(leases)
        candidates = []
        try:
            slot_ids = tuple(
                adapter._validate_lease_batch(leases)
                for adapter in self.state_transaction.adapters
            )
            reference_slots = slot_ids[0]
            if any(value != reference_slots for value in slot_ids[1:]):
                raise RuntimeError(
                    "adapters resolved inconsistent slot ids"
                )
            for layer_index, adapter in enumerate(
                self.state_transaction.adapters
            ):
                convolution = snapshot.convolution_states[layer_index]
                recurrent = snapshot.recurrent_states[layer_index]
                for slot_id in reference_slots:
                    adapter._validate_candidate(
                        convolution,
                        adapter.convolution[slot_id],
                        "convolution_state",
                    )
                    adapter._validate_candidate(
                        recurrent,
                        adapter.recurrent[slot_id],
                        "recurrent_state",
                    )
                candidates.append((
                    convolution.unsqueeze(0).expand(
                        batch_size,
                        *convolution.shape,
                    ),
                    recurrent.unsqueeze(0).expand(
                        batch_size,
                        *recurrent.shape,
                    ),
                ))
            self.state_transaction.commit(
                leases,
                tuple(candidates),
            )
        except ValueError:
            self._counters["validation_failures"] += 1
            raise
        except RuntimeError:
            self._counters["failed_restores"] += 1
            raise

        self._entries.move_to_end(entry_key)
        self._counters["hits"] += 1
        return True

    def invalidate_blocks(
        self,
        block_identities: tuple[tuple[int, int, int], ...],
    ) -> int:
        self._validate_block_identities(block_identities)
        invalidated = set(block_identities)
        entry_keys = [
            entry_key
            for entry_key, snapshot in self._entries.items()
            if invalidated.intersection(snapshot.block_identities)
        ]
        for entry_key in entry_keys:
            self._remove_entry(entry_key)
        self._counters["invalidations"] += len(entry_keys)
        return len(entry_keys)

    def clear(self) -> int:
        cleared = len(self._entries)
        for entry_key in tuple(self._entries):
            self._remove_entry(entry_key)
        self._counters["clears"] += 1
        return cleared

    def observation_snapshot(self) -> dict[str, int]:
        return {
            "current_entries": len(self._entries),
            "current_bytes": self._current_bytes,
            "current_logical_bytes": self._current_logical_bytes,
            "deduplicated_bytes": (
                self._current_logical_bytes - self._current_bytes
            ),
            "current_interned_tensors": len(self._intern_records),
            "current_intern_references": self._current_intern_references,
            "current_prepared_publications": int(
                self._prepared_publication is not None
            ),
            "current_prepared_bytes": (
                self._prepared_publication.handle.storage_bytes
                if self._prepared_publication is not None
                else 0
            ),
            "current_precommitted_bytes": (
                self._prepared_publication.reserved_new_bytes
                if (
                    self._prepared_publication is not None
                    and self._prepared_publication.phase == "precommitted"
                )
                else 0
            ),
            "current_precommitted_references": (
                len(self._prepared_publication.acquired)
                if (
                    self._prepared_publication is not None
                    and self._prepared_publication.phase == "precommitted"
                )
                else 0
            ),
            "peak_entries": self._peak_entries,
            "peak_bytes": self._peak_bytes,
            "peak_logical_bytes": self._peak_logical_bytes,
            "peak_prepared_bytes": self._peak_prepared_bytes,
            **self._counters,
        }
