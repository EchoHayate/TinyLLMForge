from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import count
import json
import math
from typing import Optional

import torch

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
)
from tinyvllm.engine.qwen35_hybrid_prefix_representation import (
    QWEN35_HYBRID_PREFIX_INT8_VERSION,
    QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
)
from tinyvllm.engine.qwen35_recurrent_int8_codec import (
    QWEN35_RECURRENT_INT8_CODEC,
    Qwen35EncodedRecurrentInt8,
    decode_qwen35_recurrent_int8_per_row,
    encode_qwen35_recurrent_int8_per_row,
)
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)


EXPECTED_LINEAR_LAYER_COUNT = 18

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


class _DecodeWorkspaceTelemetryError(RuntimeError):
    pass


class _LayerInventoryError(ValueError):
    pass


@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Layer:
    layer_index: int
    convolution_state: torch.Tensor
    recurrent_values: torch.Tensor
    recurrent_scales: torch.Tensor
    source_shape: tuple[int, int, int]
    source_dtype: torch.dtype
    codec: str


@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Accounting:
    full_fidelity_logical_bytes: int
    encoded_physical_bytes: int
    codec_metadata_bytes: int
    temporary_encode_workspace_bytes: int
    temporary_decode_workspace_bytes: int


@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Snapshot:
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    layers: tuple[Qwen35HybridPrefixInt8Layer, ...]
    accounting: Qwen35HybridPrefixInt8Accounting


@dataclass(frozen=True)
class Qwen35HybridPrefixInt8PreparedPublication:
    publication_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    accounting: Qwen35HybridPrefixInt8Accounting


class Qwen35HybridPrefixInt8ReaderLease:

    __slots__ = (
        "_cache",
        "_snapshot",
        "_private_snapshot",
        "_released",
        "_transferred",
    )

    def __init__(self, cache, snapshot):
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_private_snapshot", None)
        object.__setattr__(self, "_released", False)
        object.__setattr__(self, "_transferred", False)

    def __setattr__(self, name, value):
        raise AttributeError(
            "Qwen35HybridPrefixInt8ReaderLease is immutable"
        )

    @property
    def snapshot(self) -> Qwen35HybridPrefixInt8Snapshot:
        if self._released:
            raise RuntimeError("reader lease has been released")
        private_snapshot = self._private_snapshot
        if private_snapshot is None:
            private_snapshot = self._cache._private_snapshot(
                self._snapshot
            )
            object.__setattr__(
                self,
                "_private_snapshot",
                private_snapshot,
            )
        return private_snapshot

    def take(self) -> Qwen35HybridPrefixInt8Snapshot:
        if self._transferred:
            raise RuntimeError(
                "reader lease snapshot ownership was already transferred"
            )
        if self._released:
            raise RuntimeError("reader lease has been released")
        snapshot = self.snapshot
        object.__setattr__(self, "_transferred", True)
        object.__setattr__(self, "_private_snapshot", None)
        self.release()
        return snapshot

    def release(self) -> None:
        if self._released:
            return
        object.__setattr__(self, "_released", True)
        cache = self._cache
        snapshot = self._snapshot
        object.__setattr__(self, "_cache", None)
        object.__setattr__(self, "_snapshot", None)
        object.__setattr__(self, "_private_snapshot", None)
        cache._release_reader(snapshot)

    def __enter__(self):
        if self._released:
            raise RuntimeError("reader lease has been released")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


@dataclass(frozen=True)
class _TensorInternKey:
    dtype: torch.dtype
    shape: tuple[int, ...]
    device_type: str
    device_index: Optional[int]
    codec: str
    digest: str


@dataclass
class _InternedTensor:
    key: _TensorInternKey
    tensor: torch.Tensor
    refcount: int
    visible_refcount: int
    storage_bytes: int


@dataclass
class _PreparedPublicationState:
    handle: Qwen35HybridPrefixInt8PreparedPublication
    private_layers: tuple[Qwen35HybridPrefixInt8Layer, ...]
    intern_candidates: tuple[
        tuple[torch.Tensor, _TensorInternKey],
        ...,
    ]
    additional_candidate_bytes: int
    phase: str = "prepared"
    snapshot: Optional[Qwen35HybridPrefixInt8Snapshot] = None
    acquired: tuple[torch.Tensor, ...] = ()
    intern_counter_snapshot: Optional[dict[str, int]] = None
    reserved_new_bytes: int = 0
    previous_entry: Optional[Qwen35HybridPrefixInt8Snapshot] = None
    previous_entry_index: Optional[int] = None
    previous_lru_order: tuple[
        tuple[Qwen35HybridPrefixKey, tuple[int, ...]],
        ...,
    ] = ()
    pre_publication_counters: Optional[dict[str, int]] = None
    pre_publication_peaks: Optional[dict[str, int]] = None
    publication_counter_deltas: dict[str, int] = field(
        default_factory=dict
    )
    counter_snapshot: Optional[dict[str, int]] = None
    evicted_entries: tuple[
        tuple[
            int,
            tuple[Qwen35HybridPrefixKey, tuple[int, ...]],
            Qwen35HybridPrefixInt8Snapshot,
        ],
        ...,
    ] = ()


def _tensor_digest(tensor):
    byte_view = tensor.detach().contiguous().view(torch.uint8)
    return sha256(byte_view.cpu().numpy().tobytes()).hexdigest()


def _tensor_bytes_equal(left, right):
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _cuda_memory_allocated(device):
    return torch.cuda.memory_allocated(device)


def _cuda_max_memory_allocated(device):
    return torch.cuda.max_memory_allocated(device)


def _cuda_reset_peak_memory_stats(device):
    torch.cuda.reset_peak_memory_stats(device)


def _cuda_synchronize(device):
    torch.cuda.synchronize(device)


def _cuda_memory_reserved(device):
    return torch.cuda.memory_reserved(device)


def _encode_recurrent_with_cuda_workspace_accounting(recurrent):
    if recurrent.device.type != "cuda":
        return encode_qwen35_recurrent_int8_per_row(recurrent), 0
    device = recurrent.device
    baseline_allocated = _cuda_memory_allocated(device)
    _cuda_reset_peak_memory_stats(device)
    encoded = encode_qwen35_recurrent_int8_per_row(recurrent)
    _cuda_synchronize(device)
    peak_allocated = _cuda_max_memory_allocated(device)
    persistent_output_bytes = (
        encoded.values.untyped_storage().nbytes()
        + encoded.scales.untyped_storage().nbytes()
    )
    temporary_workspace_bytes = max(
        0,
        peak_allocated
        - baseline_allocated
        - persistent_output_bytes,
    )
    return encoded, temporary_workspace_bytes


def _encoded_bundle_digest(layer):
    digest = sha256()
    for tensor in (
        layer.recurrent_values,
        layer.recurrent_scales,
    ):
        digest.update(
            tensor.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _canonical_json_bytes(value):
    return len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


def encoded_metadata_bytes(
    layers: tuple[Qwen35HybridPrefixInt8Layer, ...],
) -> int:
    return _canonical_json_bytes({
        "codec": QWEN35_RECURRENT_INT8_CODEC,
        "layers": [
            {
                "codec": layer.codec,
                "layer_index": layer.layer_index,
                "source_dtype": str(layer.source_dtype),
                "source_shape": list(layer.source_shape),
            }
            for layer in layers
        ],
        "representation": QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
        "version": QWEN35_HYBRID_PREFIX_INT8_VERSION,
    })


def _validate_layers(
    layers: tuple[Qwen35HybridPrefixInt8Layer, ...],
) -> None:
    if len(layers) != EXPECTED_LINEAR_LAYER_COUNT:
        raise _LayerInventoryError(
            "INT8 snapshot requires exactly 18 layers"
        )
    indices = tuple(layer.layer_index for layer in layers)
    if len(set(indices)) != EXPECTED_LINEAR_LAYER_COUNT:
        raise _LayerInventoryError(
            "INT8 snapshot layer identities are not unique"
        )
    if indices != tuple(sorted(indices)):
        raise _LayerInventoryError(
            "INT8 snapshot layers are not ordered"
        )
    for layer in layers:
        if layer.convolution_state.dtype != torch.bfloat16:
            raise ValueError("INT8 snapshot convolution must remain BF16")
        if layer.recurrent_values.dtype != torch.int8:
            raise ValueError("INT8 recurrent payload dtype mismatch")
        if layer.recurrent_scales.dtype != torch.float32:
            raise ValueError("INT8 recurrent scale dtype mismatch")
        if layer.codec != QWEN35_RECURRENT_INT8_CODEC:
            raise ValueError("INT8 recurrent codec identity mismatch")
        if layer.source_dtype != torch.float32:
            raise ValueError("INT8 recurrent source dtype mismatch")
        if (
            not isinstance(layer.source_shape, tuple)
            or len(layer.source_shape) != 3
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in layer.source_shape
            )
        ):
            raise ValueError("INT8 recurrent source shape mismatch")
        if tuple(layer.recurrent_values.shape) != layer.source_shape:
            raise ValueError("INT8 recurrent payload shape mismatch")
        if (
            tuple(layer.recurrent_scales.shape)
            != layer.source_shape[:-1]
        ):
            raise ValueError("INT8 recurrent scale shape mismatch")
        if (
            layer.convolution_state.device
            != layer.recurrent_values.device
            or layer.recurrent_values.device
            != layer.recurrent_scales.device
        ):
            raise ValueError("INT8 snapshot layer device mismatch")
        if not layer.convolution_state.is_contiguous():
            raise ValueError("INT8 snapshot convolution must be contiguous")
        if not layer.recurrent_values.is_contiguous():
            raise ValueError("INT8 recurrent payload must be contiguous")
        if not layer.recurrent_scales.is_contiguous():
            raise ValueError("INT8 recurrent scales must be contiguous")
        if not torch.isfinite(layer.recurrent_scales).all().item():
            raise ValueError("INT8 recurrent scales must be finite")
        if not torch.all(layer.recurrent_scales > 0).item():
            raise ValueError("INT8 recurrent scales must be positive")
        if torch.any(layer.recurrent_values == -128).item():
            raise ValueError(
                "INT8 recurrent payload contains forbidden -128"
            )


class Qwen35HybridPrefixInt8SnapshotCache:

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
        self._current_encoded_logical_bytes = 0
        self._current_full_fidelity_logical_bytes = 0
        self._current_codec_metadata_bytes = 0
        self._current_intern_references = 0
        self._reader_counts = {}
        self._deferred_snapshots = {}
        self._current_reader_leases = 0
        self._current_temporary_decode_workspace_bytes = 0
        self._current_temporary_decode_cuda_allocated_bytes = 0
        self._current_temporary_decode_cuda_reserved_bytes = 0
        self._decode_cuda_allocated_baseline = 0
        self._decode_cuda_reserved_baseline = 0
        self._peak_entries = 0
        self._peak_bytes = 0
        self._peak_encoded_logical_bytes = 0
        self._peak_full_fidelity_logical_bytes = 0
        self._peak_codec_metadata_bytes = 0
        self._peak_prepared_bytes = 0
        self._peak_reader_leases = 0
        self._peak_temporary_encode_workspace_bytes = 0
        self._peak_temporary_decode_workspace_bytes = 0
        self._peak_temporary_decode_cuda_allocated_bytes = 0
        self._peak_temporary_decode_cuda_reserved_bytes = 0
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
            "deferred_snapshot_releases": 0,
            "quarantines": 0,
            "decode_failures": 0,
            "commit_failures": 0,
            "rollback_failures": 0,
            "fallbacks": 0,
            "partial_restore_attempts": 0,
            "mixed_representation_rejections": 0,
            "missing_layer_rejections": 0,
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
        return tensor.detach().contiguous().clone()

    @staticmethod
    def _intern_key(tensor, codec):
        return _TensorInternKey(
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            codec=codec,
            digest=_tensor_digest(tensor),
        )

    def _prepare_candidate_interning(self, tensors):
        private_buckets = {}
        prepared = []
        additional_bytes = 0
        for tensor, codec in tensors:
            key = self._intern_key(tensor, codec)
            prepared.append((tensor, key))
            bucket = private_buckets.setdefault(key, [])
            if any(
                _tensor_bytes_equal(tensor, existing)
                for existing in bucket
            ):
                continue
            bucket.append(tensor)
            resident_bucket = self._intern_table.get(key, ())
            if any(
                _tensor_bytes_equal(tensor, record.tensor)
                for record in resident_bucket
            ):
                continue
            additional_bytes += tensor.untyped_storage().nbytes()
        return tuple(prepared), additional_bytes

    def _acquire_interned_tensor(self, candidate, key=None):
        if key is None:
            key = self._intern_key(
                candidate,
                QWEN35_RECURRENT_INT8_CODEC,
            )
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

    @staticmethod
    def _snapshot_tensors(snapshot):
        return tuple(
            tensor
            for layer in snapshot.layers
            for tensor in (
                layer.convolution_state,
                layer.recurrent_values,
                layer.recurrent_scales,
            )
        )

    @classmethod
    def _private_snapshot(cls, snapshot):
        return Qwen35HybridPrefixInt8Snapshot(
            key=snapshot.key,
            token_ids=snapshot.token_ids,
            block_identities=snapshot.block_identities,
            layers=tuple(
                Qwen35HybridPrefixInt8Layer(
                    layer_index=layer.layer_index,
                    convolution_state=cls._owned_clone(
                        layer.convolution_state
                    ),
                    recurrent_values=cls._owned_clone(
                        layer.recurrent_values
                    ),
                    recurrent_scales=cls._owned_clone(
                        layer.recurrent_scales
                    ),
                    source_shape=layer.source_shape,
                    source_dtype=layer.source_dtype,
                    codec=layer.codec,
                )
                for layer in snapshot.layers
            ),
            accounting=snapshot.accounting,
        )

    def _mark_snapshot_visible(self, snapshot):
        for tensor in self._snapshot_tensors(snapshot):
            record = self._intern_records[id(tensor)]
            if record.visible_refcount == 0:
                self._current_bytes += record.storage_bytes
            record.visible_refcount += 1

    def _add_snapshot_accounting(self, snapshot):
        accounting = snapshot.accounting
        self._current_encoded_logical_bytes += (
            accounting.encoded_physical_bytes
        )
        self._current_full_fidelity_logical_bytes += (
            accounting.full_fidelity_logical_bytes
        )
        self._current_codec_metadata_bytes += (
            accounting.codec_metadata_bytes
        )

    def _subtract_snapshot_accounting(self, snapshot):
        accounting = snapshot.accounting
        self._current_encoded_logical_bytes -= (
            accounting.encoded_physical_bytes
        )
        self._current_full_fidelity_logical_bytes -= (
            accounting.full_fidelity_logical_bytes
        )
        self._current_codec_metadata_bytes -= (
            accounting.codec_metadata_bytes
        )

    def _release_snapshot(self, snapshot):
        for tensor in self._snapshot_tensors(snapshot):
            self._release_interned_tensor(tensor, visible=True)
        self._subtract_snapshot_accounting(snapshot)

    def _dispose_detached_snapshot(self, snapshot):
        snapshot_id = id(snapshot)
        if self._reader_counts.get(snapshot_id, 0):
            if snapshot_id not in self._deferred_snapshots:
                self._deferred_snapshots[snapshot_id] = snapshot
                self._counters["deferred_snapshot_releases"] += 1
            return
        self._release_detached_snapshot(snapshot)

    def _remove_entry(self, entry_key):
        snapshot = self._detach_entry(entry_key)
        self._dispose_detached_snapshot(snapshot)
        return snapshot

    def _detach_entry(self, entry_key):
        snapshot = self._entries.pop(entry_key)
        for tensor in self._snapshot_tensors(snapshot):
            record = self._intern_records[id(tensor)]
            record.visible_refcount -= 1
            if record.visible_refcount == 0:
                self._current_bytes -= record.storage_bytes
        self._subtract_snapshot_accounting(snapshot)
        return snapshot

    def _attach_entry(self, entry_key, snapshot, index=None):
        items = list(self._entries.items())
        if index is None:
            index = len(items)
        items.insert(index, (entry_key, snapshot))
        self._entries = OrderedDict(items)
        self._mark_snapshot_visible(snapshot)
        self._add_snapshot_accounting(snapshot)

    def _release_detached_snapshot(self, snapshot):
        for tensor in self._snapshot_tensors(snapshot):
            self._release_interned_tensor(tensor)

    def _release_reader(self, snapshot):
        snapshot_id = id(snapshot)
        reader_count = self._reader_counts.get(snapshot_id)
        if reader_count is None or reader_count <= 0:
            raise RuntimeError("reader lease accounting is inconsistent")
        self._current_reader_leases -= 1
        if reader_count > 1:
            self._reader_counts[snapshot_id] = reader_count - 1
            return
        del self._reader_counts[snapshot_id]
        deferred = self._deferred_snapshots.pop(snapshot_id, None)
        if deferred is not None:
            self._release_detached_snapshot(deferred)

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
    ) -> Optional[Qwen35HybridPrefixInt8PreparedPublication]:
        if self._prepared_publication is not None:
            self._counters["publication_prepare_conflicts"] += 1
            raise RuntimeError(
                "a hybrid prefix INT8 publication is already prepared"
            )
        try:
            self._validate_identity(
                key,
                token_ids,
                block_identities,
            )
            if not isinstance(lease, HybridStateLease):
                raise ValueError("lease must be a HybridStateLease")
            adapters = self.state_transaction.adapters
            if len(adapters) != EXPECTED_LINEAR_LAYER_COUNT:
                raise _LayerInventoryError(
                    "INT8 snapshot requires exactly 18 layers"
                )
            layer_indices = tuple(
                adapter.layer_index for adapter in adapters
            )
            if len(set(layer_indices)) != EXPECTED_LINEAR_LAYER_COUNT:
                raise _LayerInventoryError(
                    "INT8 snapshot layer identities are not unique"
                )
            if layer_indices != tuple(sorted(layer_indices)):
                raise _LayerInventoryError(
                    "INT8 snapshot layers are not ordered"
                )

            gathered = self.state_transaction.gather((lease,))
            private_layers = []
            encode_workspace_peak_bytes = 0
            for adapter, (convolution_batch, recurrent_batch) in zip(
                adapters,
                gathered,
            ):
                slot_id = adapter.pool.validate(lease)
                convolution = self._owned_clone(convolution_batch[0])
                recurrent = recurrent_batch[0]
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
                if convolution.dtype != torch.bfloat16:
                    raise ValueError(
                        "INT8 snapshot convolution must remain BF16"
                    )
                if recurrent.dtype != torch.float32:
                    raise ValueError(
                        "INT8 recurrent source dtype mismatch"
                    )
                encoded, encode_workspace_bytes = (
                    _encode_recurrent_with_cuda_workspace_accounting(
                        recurrent
                    )
                )
                encode_workspace_peak_bytes = max(
                    encode_workspace_peak_bytes,
                    encode_workspace_bytes,
                )
                self._peak_temporary_encode_workspace_bytes = max(
                    self._peak_temporary_encode_workspace_bytes,
                    encode_workspace_bytes,
                )
                if type(encoded) is not Qwen35EncodedRecurrentInt8:
                    raise ValueError(
                        "encoded recurrent must use the approved type"
                    )
                private_layers.append(
                    Qwen35HybridPrefixInt8Layer(
                        layer_index=adapter.layer_index,
                        convolution_state=convolution,
                        recurrent_values=encoded.values,
                        recurrent_scales=encoded.scales,
                        source_shape=encoded.source_shape,
                        source_dtype=encoded.source_dtype,
                        codec=encoded.codec,
                    )
                )
            private_layers = tuple(private_layers)
            try:
                _validate_layers(private_layers)
            except ValueError as error:
                message = str(error)
                if (
                    "codec identity mismatch" in message
                    or "payload dtype mismatch" in message
                    or "scale dtype mismatch" in message
                ):
                    self._counters[
                        "mixed_representation_rejections"
                    ] += 1
                raise
        except (ValueError, RuntimeError) as error:
            if isinstance(error, _LayerInventoryError):
                self._counters["missing_layer_rejections"] += 1
            self._counters["validation_failures"] += 1
            raise

        convolution_bytes = sum(
            layer.convolution_state.numel()
            * layer.convolution_state.element_size()
            for layer in private_layers
        )
        recurrent_logical_bytes = sum(
            math.prod(layer.source_shape)
            * torch.tensor([], dtype=torch.float32).element_size()
            for layer in private_layers
        )
        recurrent_payload_bytes = sum(
            layer.recurrent_values.numel()
            * layer.recurrent_values.element_size()
            for layer in private_layers
        )
        scale_bytes = sum(
            layer.recurrent_scales.numel()
            * layer.recurrent_scales.element_size()
            for layer in private_layers
        )
        accounting = Qwen35HybridPrefixInt8Accounting(
            full_fidelity_logical_bytes=(
                convolution_bytes + recurrent_logical_bytes
            ),
            encoded_physical_bytes=(
                convolution_bytes
                + recurrent_payload_bytes
                + scale_bytes
            ),
            codec_metadata_bytes=encoded_metadata_bytes(private_layers),
            temporary_encode_workspace_bytes=(
                encode_workspace_peak_bytes
            ),
            temporary_decode_workspace_bytes=0,
        )
        self._peak_temporary_encode_workspace_bytes = max(
            self._peak_temporary_encode_workspace_bytes,
            accounting.temporary_encode_workspace_bytes,
        )
        intern_candidates = []
        for layer in private_layers:
            bundle_digest = _encoded_bundle_digest(layer)
            intern_candidates.extend((
                (
                    layer.convolution_state,
                    (
                        f"{layer.codec}:layer={layer.layer_index}:"
                        "component=convolution"
                    ),
                ),
                (
                    layer.recurrent_values,
                    (
                        f"{layer.codec}:layer={layer.layer_index}:"
                        "component=recurrent_values:"
                        f"bundle={bundle_digest}"
                    ),
                ),
                (
                    layer.recurrent_scales,
                    (
                        f"{layer.codec}:layer={layer.layer_index}:"
                        "component=recurrent_scales:"
                        f"bundle={bundle_digest}"
                    ),
                ),
            ))
        intern_candidates, additional_candidate_bytes = (
            self._prepare_candidate_interning(tuple(intern_candidates))
        )
        entry_key = self._entry_key(key, token_ids)
        replacement_exceeds_capacity = (
            entry_key in self._entries
            and (
                self._intern_total_bytes + additional_candidate_bytes
                > self.max_bytes
            )
        )
        if (
            additional_candidate_bytes > self.max_bytes
            or replacement_exceeds_capacity
        ):
            self._counters["oversize_rejections"] += 1
            return None
        handle = Qwen35HybridPrefixInt8PreparedPublication(
            publication_id=next(self._publication_ids),
            key=key,
            token_ids=token_ids,
            block_identities=block_identities,
            accounting=accounting,
        )
        state = _PreparedPublicationState(
            handle=handle,
            private_layers=private_layers,
            intern_candidates=intern_candidates,
            additional_candidate_bytes=additional_candidate_bytes,
            pre_publication_counters=dict(self._counters),
            pre_publication_peaks={
                "peak_entries": self._peak_entries,
                "peak_bytes": self._peak_bytes,
                "peak_encoded_logical_bytes": (
                    self._peak_encoded_logical_bytes
                ),
                "peak_full_fidelity_logical_bytes": (
                    self._peak_full_fidelity_logical_bytes
                ),
                "peak_codec_metadata_bytes": (
                    self._peak_codec_metadata_bytes
                ),
                "peak_prepared_bytes": self._peak_prepared_bytes,
            },
        )
        self._prepared_publication = state
        self._peak_prepared_bytes = max(
            self._peak_prepared_bytes,
            accounting.encoded_physical_bytes,
        )
        self._counters["publication_prepares"] += 1
        state.publication_counter_deltas["publication_prepares"] = 1
        return handle

    def _prepared_state(self, prepared):
        if not isinstance(
            prepared,
            Qwen35HybridPrefixInt8PreparedPublication,
        ):
            raise ValueError(
                "prepared must be a "
                "Qwen35HybridPrefixInt8PreparedPublication"
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
            if tuple(self._entries) != state.previous_lru_order:
                restored = dict(self._entries)
                self._entries = OrderedDict(
                    (entry_key, restored[entry_key])
                    for entry_key in state.previous_lru_order
                )
        for name, delta in state.publication_counter_deltas.items():
            self._counters[name] -= delta
        self._peak_entries = state.pre_publication_peaks["peak_entries"]
        self._peak_bytes = state.pre_publication_peaks["peak_bytes"]
        self._peak_encoded_logical_bytes = state.pre_publication_peaks[
            "peak_encoded_logical_bytes"
        ]
        self._peak_full_fidelity_logical_bytes = (
            state.pre_publication_peaks[
                "peak_full_fidelity_logical_bytes"
            ]
        )
        self._peak_codec_metadata_bytes = state.pre_publication_peaks[
            "peak_codec_metadata_bytes"
        ]
        self._peak_prepared_bytes = state.pre_publication_peaks[
            "peak_prepared_bytes"
        ]
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
            interned_tensors = []
            for tensor, intern_key in state.intern_candidates:
                interned = self._acquire_interned_tensor(
                    tensor,
                    intern_key,
                )
                acquired.append(interned)
                interned_tensors.append(interned)
            layers = tuple(
                Qwen35HybridPrefixInt8Layer(
                    layer_index=private.layer_index,
                    convolution_state=interned_tensors[index * 3],
                    recurrent_values=interned_tensors[index * 3 + 1],
                    recurrent_scales=interned_tensors[index * 3 + 2],
                    source_shape=private.source_shape,
                    source_dtype=private.source_dtype,
                    codec=private.codec,
                )
                for index, private in enumerate(state.private_layers)
            )
            _validate_layers(layers)
            snapshot = Qwen35HybridPrefixInt8Snapshot(
                key=prepared.key,
                token_ids=prepared.token_ids,
                block_identities=prepared.block_identities,
                layers=layers,
                accounting=prepared.accounting,
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
        for name in intern_counter_snapshot:
            delta = self._counters[name] - intern_counter_snapshot[name]
            if delta:
                state.publication_counter_deltas[name] = (
                    state.publication_counter_deltas.get(name, 0)
                    + delta
                )
        state.publication_counter_deltas["publication_precommits"] = 1

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
        state.previous_lru_order = tuple(self._entries)
        if entry_key in self._entries:
            state.previous_entry_index = tuple(
                self._entries
            ).index(entry_key)
            state.previous_entry = self._detach_entry(entry_key)
            self._counters["replacements"] += 1
        self._mark_snapshot_visible(snapshot)
        self._entries[entry_key] = snapshot
        self._add_snapshot_accounting(snapshot)
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
        self._peak_encoded_logical_bytes = max(
            self._peak_encoded_logical_bytes,
            self._current_encoded_logical_bytes,
        )
        self._peak_full_fidelity_logical_bytes = max(
            self._peak_full_fidelity_logical_bytes,
            self._current_full_fidelity_logical_bytes,
        )
        self._peak_codec_metadata_bytes = max(
            self._peak_codec_metadata_bytes,
            self._current_codec_metadata_bytes,
        )
        for name, before in state.counter_snapshot.items():
            delta = self._counters[name] - before
            if delta:
                state.publication_counter_deltas[name] = (
                    state.publication_counter_deltas.get(name, 0)
                    + delta
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
            self._dispose_detached_snapshot(state.previous_entry)
        for _, _, snapshot in state.evicted_entries:
            self._dispose_detached_snapshot(snapshot)
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

    def acquire_reader(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
    ) -> Optional[Qwen35HybridPrefixInt8ReaderLease]:
        try:
            self._validate_key(key)
            self._validate_tokens(key, token_ids)
            self._validate_block_identities(block_identities)
        except ValueError:
            self._counters["validation_failures"] += 1
            raise
        entry_key, snapshot = self._lookup(key, token_ids)
        if snapshot is None:
            return None
        if snapshot.block_identities != block_identities:
            self._remove_entry(entry_key)
            self._counters["stale_block_misses"] += 1
            self._counters["misses"] += 1
            return None
        snapshot_id = id(snapshot)
        self._reader_counts[snapshot_id] = (
            self._reader_counts.get(snapshot_id, 0) + 1
        )
        self._current_reader_leases += 1
        self._peak_reader_leases = max(
            self._peak_reader_leases,
            self._current_reader_leases,
        )
        return Qwen35HybridPrefixInt8ReaderLease(self, snapshot)

    @staticmethod
    def _encoded_from_layer(layer):
        payload_bytes = layer.recurrent_values.untyped_storage().nbytes()
        scale_bytes = layer.recurrent_scales.untyped_storage().nbytes()
        return Qwen35EncodedRecurrentInt8(
            codec=layer.codec,
            values=layer.recurrent_values,
            scales=layer.recurrent_scales,
            source_shape=layer.source_shape,
            source_dtype=layer.source_dtype,
            logical_bytes=math.prod(layer.source_shape) * 4,
            payload_bytes=payload_bytes,
            scale_bytes=scale_bytes,
            encoded_bytes=payload_bytes + scale_bytes,
        )

    def _begin_decode_workspace_accounting(self, device):
        self._current_temporary_decode_workspace_bytes = 0
        self._current_temporary_decode_cuda_allocated_bytes = 0
        self._current_temporary_decode_cuda_reserved_bytes = 0
        self._decode_cuda_allocated_baseline = 0
        self._decode_cuda_reserved_baseline = 0
        if device.type != "cuda":
            return
        try:
            _cuda_synchronize(device)
            self._decode_cuda_allocated_baseline = (
                _cuda_memory_allocated(device)
            )
            self._decode_cuda_reserved_baseline = (
                _cuda_memory_reserved(device)
            )
            _cuda_reset_peak_memory_stats(device)
        except BaseException:
            self._decode_cuda_allocated_baseline = 0
            self._decode_cuda_reserved_baseline = 0
            raise

    def _record_decode_workspace(self, candidates, device):
        logical_bytes = sum(
            tensor.numel() * tensor.element_size()
            for candidate in candidates
            for tensor in candidate
        )
        self._current_temporary_decode_workspace_bytes = logical_bytes
        self._peak_temporary_decode_workspace_bytes = max(
            self._peak_temporary_decode_workspace_bytes,
            logical_bytes,
        )
        if device.type != "cuda":
            return
        try:
            _cuda_synchronize(device)
            allocated = max(
                0,
                _cuda_memory_allocated(device)
                - self._decode_cuda_allocated_baseline,
            )
            peak_allocated = max(
                allocated,
                _cuda_max_memory_allocated(device)
                - self._decode_cuda_allocated_baseline,
            )
            reserved = max(
                0,
                _cuda_memory_reserved(device)
                - self._decode_cuda_reserved_baseline,
            )
        except RuntimeError as error:
            raise _DecodeWorkspaceTelemetryError(str(error)) from error
        self._current_temporary_decode_cuda_allocated_bytes = allocated
        self._current_temporary_decode_cuda_reserved_bytes = reserved
        self._peak_temporary_decode_cuda_allocated_bytes = max(
            self._peak_temporary_decode_cuda_allocated_bytes,
            peak_allocated,
        )
        self._peak_temporary_decode_cuda_reserved_bytes = max(
            self._peak_temporary_decode_cuda_reserved_bytes,
            reserved,
        )

    def _finish_decode_workspace_accounting(
        self,
        device,
        release_staging,
    ):
        measurements_complete = False
        try:
            release_staging()
            self._current_temporary_decode_workspace_bytes = 0
            if device.type == "cuda":
                _cuda_synchronize(device)
                allocated = max(
                    0,
                    _cuda_memory_allocated(device)
                    - self._decode_cuda_allocated_baseline,
                )
                reserved = max(
                    0,
                    _cuda_memory_reserved(device)
                    - self._decode_cuda_reserved_baseline,
                )
                self._current_temporary_decode_cuda_allocated_bytes = (
                    allocated
                )
                self._current_temporary_decode_cuda_reserved_bytes = (
                    reserved
                )
            else:
                self._current_temporary_decode_cuda_allocated_bytes = 0
                self._current_temporary_decode_cuda_reserved_bytes = 0
            measurements_complete = True
        finally:
            self._current_temporary_decode_workspace_bytes = 0
            if not measurements_complete:
                self._current_temporary_decode_cuda_allocated_bytes = 0
                self._current_temporary_decode_cuda_reserved_bytes = 0
            self._decode_cuda_allocated_baseline = 0
            self._decode_cuda_reserved_baseline = 0

    @staticmethod
    def _raise_primary_or_cleanup(primary, cleanup):
        if primary is not None:
            error, traceback = primary
            raise error.with_traceback(traceback)
        if cleanup is not None:
            error, traceback = cleanup
            raise error.with_traceback(traceback)

    def acquire(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        leases: tuple[HybridStateLease, ...],
    ) -> bool:
        if not isinstance(leases, tuple) or not leases:
            self._counters["validation_failures"] += 1
            raise ValueError("leases must be a non-empty tuple")
        reader = self.acquire_reader(key, token_ids, block_identities)
        if reader is None:
            return False

        entry_key = self._entry_key(key, token_ids)
        resident_snapshot = reader._snapshot
        candidates = []
        commit_started = False
        encoded = None
        recurrent = None
        convolution_batch = None
        recurrent_batch = None
        restored = None
        primary_error = None
        cleanup_error = None
        device = None
        accounting_started = False
        try:
            try:
                _validate_layers(resident_snapshot.layers)
                device = (
                    resident_snapshot.layers[0]
                    .recurrent_values.device
                )
                try:
                    self._begin_decode_workspace_accounting(device)
                except RuntimeError as error:
                    raise _DecodeWorkspaceTelemetryError(
                        str(error)
                    ) from error
                accounting_started = True
                slot_ids = tuple(
                    adapter._validate_lease_batch(leases)
                    for adapter in self.state_transaction.adapters
                )
                layer_indices = tuple(
                    adapter.layer_index
                    for adapter in self.state_transaction.adapters
                )
                if tuple(
                    layer.layer_index
                    for layer in resident_snapshot.layers
                ) != layer_indices:
                    raise _LayerInventoryError(
                        "INT8 snapshot layers do not match "
                        "transaction adapters"
                    )
                reference_slots = slot_ids[0]
                if any(
                    value != reference_slots for value in slot_ids[1:]
                ):
                    raise RuntimeError(
                        "adapters resolved inconsistent slot ids"
                    )
                batch_size = len(leases)
                for adapter, layer in zip(
                    self.state_transaction.adapters,
                    resident_snapshot.layers,
                ):
                    encoded = self._encoded_from_layer(layer)
                    recurrent = decode_qwen35_recurrent_int8_per_row(
                        encoded,
                        device=adapter.recurrent.device,
                    )
                    convolution_batch = (
                        layer.convolution_state.unsqueeze(0)
                        .expand(
                            batch_size,
                            *layer.convolution_state.shape,
                        )
                        .clone()
                        .contiguous()
                    )
                    recurrent_batch = (
                        recurrent.unsqueeze(0)
                        .expand(batch_size, *recurrent.shape)
                        .clone()
                        .contiguous()
                    )
                    for slot_id in reference_slots:
                        adapter._validate_candidate(
                            convolution_batch[0],
                            adapter.convolution[slot_id],
                            "convolution_state",
                        )
                        adapter._validate_candidate(
                            recurrent_batch[0],
                            adapter.recurrent[slot_id],
                            "recurrent_state",
                        )
                    candidates.append((
                        convolution_batch,
                        recurrent_batch,
                    ))
                    self._record_decode_workspace(candidates, device)
                commit_started = True
                self.state_transaction.commit(
                    leases,
                    tuple(candidates),
                )
                restored = True
            except (ValueError, RuntimeError) as error:
                if isinstance(error, _DecodeWorkspaceTelemetryError):
                    raise
                if isinstance(error, _LayerInventoryError):
                    self._counters["missing_layer_rejections"] += 1
                message = str(error)
                if (
                    "codec identity mismatch" in message
                    or "payload dtype mismatch" in message
                    or "scale dtype mismatch" in message
                ):
                    self._counters[
                        "mixed_representation_rejections"
                    ] += 1
                if commit_started:
                    self._counters["commit_failures"] += 1
                    self._counters["failed_restores"] += 1
                    if error.__context__ is not None:
                        self._counters["rollback_failures"] += 1
                    raise
                if candidates:
                    self._counters["partial_restore_attempts"] += 1
                current = self._entries.get(entry_key)
                if current is resident_snapshot:
                    self._remove_entry(entry_key)
                self._counters["quarantines"] += 1
                self._counters["decode_failures"] += 1
                self._counters["misses"] += 1
                restored = False
        except BaseException as error:
            primary_error = (error, error.__traceback__)
        finally:
            def release_staging():
                nonlocal encoded
                nonlocal recurrent
                nonlocal convolution_batch
                nonlocal recurrent_batch
                candidates.clear()
                encoded = None
                recurrent = None
                convolution_batch = None
                recurrent_batch = None
                reader.release()

            if accounting_started:
                try:
                    self._finish_decode_workspace_accounting(
                        device,
                        release_staging,
                    )
                except BaseException as error:
                    cleanup_error = (error, error.__traceback__)
            else:
                try:
                    release_staging()
                except BaseException as error:
                    cleanup_error = (error, error.__traceback__)

        self._raise_primary_or_cleanup(primary_error, cleanup_error)
        if not restored:
            return False
        if self._entries.get(entry_key) is resident_snapshot:
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

    def observation_snapshot(self) -> dict[str, object]:
        prepared = self._prepared_publication
        return {
            "representation": QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
            "representation_version": QWEN35_HYBRID_PREFIX_INT8_VERSION,
            "codec": QWEN35_RECURRENT_INT8_CODEC,
            "current_entries": len(self._entries),
            "current_bytes": self._current_bytes,
            "current_encoded_physical_bytes": self._current_bytes,
            "current_encoded_logical_bytes": (
                self._current_encoded_logical_bytes
            ),
            "current_full_fidelity_logical_bytes": (
                self._current_full_fidelity_logical_bytes
            ),
            "current_codec_metadata_bytes": (
                self._current_codec_metadata_bytes
            ),
            "deduplicated_bytes": (
                self._current_encoded_logical_bytes
                - self._current_bytes
            ),
            "current_interned_tensors": len(self._intern_records),
            "current_intern_references": (
                self._current_intern_references
            ),
            "current_reader_leases": self._current_reader_leases,
            "current_prepared_publications": int(prepared is not None),
            "current_prepared_bytes": (
                prepared.handle.accounting.encoded_physical_bytes
                if prepared is not None
                else 0
            ),
            "current_precommitted_bytes": (
                prepared.reserved_new_bytes
                if (
                    prepared is not None
                    and prepared.phase == "precommitted"
                )
                else 0
            ),
            "current_precommitted_references": (
                len(prepared.acquired)
                if (
                    prepared is not None
                    and prepared.phase == "precommitted"
                )
                else 0
            ),
            "current_temporary_encode_workspace_bytes": (
                prepared.handle.accounting
                .temporary_encode_workspace_bytes
                if prepared is not None
                else 0
            ),
            "current_temporary_decode_workspace_bytes": (
                self._current_temporary_decode_workspace_bytes
            ),
            "current_temporary_decode_cuda_allocated_bytes": (
                self._current_temporary_decode_cuda_allocated_bytes
            ),
            "current_temporary_decode_cuda_reserved_bytes": (
                self._current_temporary_decode_cuda_reserved_bytes
            ),
            "peak_entries": self._peak_entries,
            "peak_bytes": self._peak_bytes,
            "peak_encoded_logical_bytes": (
                self._peak_encoded_logical_bytes
            ),
            "peak_full_fidelity_logical_bytes": (
                self._peak_full_fidelity_logical_bytes
            ),
            "peak_codec_metadata_bytes": (
                self._peak_codec_metadata_bytes
            ),
            "peak_prepared_bytes": self._peak_prepared_bytes,
            "peak_reader_leases": self._peak_reader_leases,
            "peak_temporary_encode_workspace_bytes": (
                self._peak_temporary_encode_workspace_bytes
            ),
            "peak_temporary_decode_workspace_bytes": (
                self._peak_temporary_decode_workspace_bytes
            ),
            "peak_temporary_decode_cuda_allocated_bytes": (
                self._peak_temporary_decode_cuda_allocated_bytes
            ),
            "peak_temporary_decode_cuda_reserved_bytes": (
                self._peak_temporary_decode_cuda_reserved_bytes
            ),
            **self._counters,
        }
