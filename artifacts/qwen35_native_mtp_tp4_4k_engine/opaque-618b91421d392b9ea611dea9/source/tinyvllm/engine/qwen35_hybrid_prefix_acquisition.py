from tinyvllm.engine.block_manager import (
    BlockManager,
    PrefixBlockReservation,
)
from tinyvllm.engine.hybrid_state import (
    HybridStateLease,
    HybridStateSlotAllocator,
    HybridStateTensorPool,
)
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
    Qwen35HybridPrefixSnapshotCache,
)
from tinyvllm.engine.sequence import Sequence


class Qwen35HybridPrefixAcquireCoordinator:

    def __init__(
        self,
        block_manager: BlockManager,
        state_allocator: HybridStateSlotAllocator,
        state_pool: HybridStateTensorPool,
        snapshot_cache: Qwen35HybridPrefixSnapshotCache,
    ):
        if not isinstance(block_manager, BlockManager):
            raise ValueError("block_manager must be a BlockManager")
        if not isinstance(
            state_allocator,
            HybridStateSlotAllocator,
        ):
            raise ValueError(
                "state_allocator must be a HybridStateSlotAllocator"
            )
        if not isinstance(state_pool, HybridStateTensorPool):
            raise ValueError(
                "state_pool must be a HybridStateTensorPool"
            )
        if not isinstance(
            snapshot_cache,
            Qwen35HybridPrefixSnapshotCache,
        ):
            raise ValueError(
                "snapshot_cache must be a "
                "Qwen35HybridPrefixSnapshotCache"
            )
        if snapshot_cache.state_transaction.pool is not state_pool:
            raise ValueError(
                "snapshot cache transaction must use the coordinator pool"
            )
        if state_allocator.capacity != state_pool.capacity:
            raise ValueError(
                "state allocator and pool capacities must match"
            )
        self.block_manager = block_manager
        self.state_allocator = state_allocator
        self.state_pool = state_pool
        self.snapshot_cache = snapshot_cache

    def _validate_request(
        self,
        sequences: tuple[Sequence, ...],
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
    ) -> None:
        if not isinstance(sequences, tuple) or not sequences:
            raise ValueError("sequences must be a non-empty tuple")
        if any(not isinstance(seq, Sequence) for seq in sequences):
            raise ValueError("sequences must contain Sequence values")
        if len({id(seq) for seq in sequences}) != len(sequences):
            raise ValueError("sequences must contain unique objects")
        if len({seq.seq_id for seq in sequences}) != len(sequences):
            raise ValueError("sequences must contain unique seq_id values")
        for seq in sequences:
            if (
                seq.block_table
                or seq.num_cached_tokens != 0
                or seq.num_computed_tokens != 0
            ):
                raise ValueError(
                    "destination sequence already owns KV metadata"
                )
            if (
                seq.hybrid_state_slot_id != -1
                or seq.hybrid_state_generation != 0
            ):
                raise ValueError(
                    "destination sequence already owns hybrid state"
                )
            if self.state_allocator.lease_for_request(seq.seq_id) is not None:
                raise ValueError(
                    "destination request already has a state lease"
                )
        self.snapshot_cache._validate_key(key)
        self.snapshot_cache._validate_tokens(key, token_ids)
        if key.block_size != self.block_manager.block_size:
            raise ValueError(
                "prefix key block size must match BlockManager"
            )
        for seq in sequences:
            if (
                len(seq.token_ids) < key.token_count
                or tuple(seq.token_ids[:key.token_count]) != token_ids
            ):
                raise ValueError(
                    "destination sequence must start with exact prefix tokens"
                )

    def _release_resources(
        self,
        reservation: PrefixBlockReservation,
        leases: list[HybridStateLease],
        activated_leases: list[HybridStateLease],
    ) -> None:
        for lease in reversed(activated_leases):
            self.state_pool.release(lease)
        for lease in reversed(leases):
            self.state_allocator.release(lease)
        self.block_manager.release_prefix_reservation(reservation)

    def acquire(
        self,
        sequences: tuple[Sequence, ...],
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
    ) -> bool:
        self._validate_request(sequences, key, token_ids)
        reservation = self.block_manager.reserve_exact_prefix(
            token_ids,
            owner_count=len(sequences),
        )
        if reservation is None:
            return False

        leases = []
        activated_leases = []
        try:
            for seq in sequences:
                lease = self.state_allocator.allocate(seq.seq_id)
                leases.append(lease)
                self.state_pool.activate(lease)
                activated_leases.append(lease)

            restored = self.snapshot_cache.acquire(
                key,
                token_ids,
                reservation.block_identities,
                tuple(leases),
            )
            if not restored:
                self._release_resources(
                    reservation,
                    leases,
                    activated_leases,
                )
                return False

            self.block_manager.attach_prefix_reservation(
                reservation,
                sequences,
            )
            for seq, lease in zip(sequences, leases):
                seq.hybrid_state_slot_id = lease.slot_id
                seq.hybrid_state_generation = lease.generation
            return True
        except BaseException:
            if reservation.state == "reserved":
                self._release_resources(
                    reservation,
                    leases,
                    activated_leases,
                )
            raise
