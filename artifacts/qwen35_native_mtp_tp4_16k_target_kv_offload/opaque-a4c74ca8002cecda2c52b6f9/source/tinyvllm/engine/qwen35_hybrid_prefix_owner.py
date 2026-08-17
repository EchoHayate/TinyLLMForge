from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixSnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_int8_cache import (
    Qwen35HybridPrefixInt8SnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket import (
    Qwen35HybridPrefixPublicationParticipant,
)
from tinyvllm.engine.qwen35_hybrid_prefix_representation import (
    QWEN35_HYBRID_PREFIX_DEFAULT,
    QWEN35_HYBRID_PREFIX_EXACT,
    resolve_qwen35_hybrid_prefix_representation,
)
from tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket import (
    Qwen35HybridPrefixRestoreParticipant,
)
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)


@dataclass(frozen=True)
class Qwen35HybridPrefixRestoreOwner:
    pool: HybridStateTensorPool
    adapters: tuple[Qwen35LayerStateAdapter, ...]
    state_transaction: Qwen35CrossLayerStateTransaction
    snapshot_cache: (
        Qwen35HybridPrefixSnapshotCache
        | Qwen35HybridPrefixInt8SnapshotCache
    )
    participant: Qwen35HybridPrefixRestoreParticipant
    publication_participant: Qwen35HybridPrefixPublicationParticipant
    max_entries: int
    max_bytes: int
    representation: str
    representation_version: str
    codec: str | None


def _non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _linear_layer_indices(pool):
    roles_by_layer = {}
    for component in pool.layout.components:
        roles_by_layer.setdefault(
            component.layer_index,
            set(),
        ).add(component.role)
    if not roles_by_layer:
        raise ValueError(
            "restore owner requires at least one complete linear layer"
        )
    expected_roles = {
        "linear_convolution",
        "linear_recurrent",
    }
    if any(
        roles != expected_roles
        for roles in roles_by_layer.values()
    ):
        raise ValueError(
            "restore owner requires complete convolution/recurrent "
            "layer pairs"
        )
    return tuple(sorted(roles_by_layer))


def _build_participant(
    participant_type,
    participant_id,
    pool,
    snapshot_cache,
):
    if not isinstance(
        snapshot_cache,
        (
            Qwen35HybridPrefixSnapshotCache,
            Qwen35HybridPrefixInt8SnapshotCache,
        ),
    ):
        raise ValueError("unsupported hybrid prefix snapshot cache")
    return participant_type(
        participant_id,
        pool,
        snapshot_cache,
    )


def build_qwen35_hybrid_prefix_restore_owner(
    pool,
    *,
    participant_id,
    max_entries,
    max_bytes,
    representation=QWEN35_HYBRID_PREFIX_DEFAULT,
):
    if not isinstance(pool, HybridStateTensorPool):
        raise ValueError("pool must be a HybridStateTensorPool")
    participant_id = _non_negative_integer(
        participant_id,
        "participant_id",
    )
    max_entries = _positive_integer(max_entries, "max_entries")
    max_bytes = _positive_integer(max_bytes, "max_bytes")
    representation_identity = (
        resolve_qwen35_hybrid_prefix_representation(
            representation
        )
    )
    adapters = tuple(
        Qwen35LayerStateAdapter(pool, layer_index)
        for layer_index in _linear_layer_indices(pool)
    )
    state_transaction = Qwen35CrossLayerStateTransaction(
        adapters
    )
    cache_type = (
        Qwen35HybridPrefixSnapshotCache
        if representation_identity.name == QWEN35_HYBRID_PREFIX_EXACT
        else Qwen35HybridPrefixInt8SnapshotCache
    )
    snapshot_cache = cache_type(
        state_transaction,
        max_entries=max_entries,
        max_bytes=max_bytes,
    )
    participant = _build_participant(
        Qwen35HybridPrefixRestoreParticipant,
        participant_id,
        pool,
        snapshot_cache,
    )
    publication_participant = (
        _build_participant(
            Qwen35HybridPrefixPublicationParticipant,
            participant_id,
            pool,
            snapshot_cache,
        )
    )
    return Qwen35HybridPrefixRestoreOwner(
        pool=pool,
        adapters=adapters,
        state_transaction=state_transaction,
        snapshot_cache=snapshot_cache,
        participant=participant,
        publication_participant=publication_participant,
        max_entries=max_entries,
        max_bytes=max_bytes,
        representation=representation_identity.name,
        representation_version=representation_identity.version,
        codec=representation_identity.codec,
    )
