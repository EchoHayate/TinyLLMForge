from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Optional

from tinyvllm.engine.block_manager import (
    BlockManager,
    SequenceBlockReservation,
)
from tinyvllm.engine.hybrid_state import (
    HybridStateSlotAllocator,
)
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
    Qwen35HybridPrefixSnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket import (
    Qwen35HybridPrefixRestorePayload,
)
from tinyvllm.engine.sequence import Sequence


@dataclass
class Qwen35HybridPrefixEngineRestoreTicket:
    payload: Qwen35HybridPrefixRestorePayload
    sequence: Sequence
    reservation: SequenceBlockReservation
    prepare_results: tuple[dict, ...] = ()
    state: str = "reserved"


class Qwen35HybridPrefixEngineRestoreCoordinator:

    def __init__(
        self,
        engine,
        block_manager: BlockManager,
        state_allocator: HybridStateSlotAllocator,
        *,
        timeout_s: float,
    ):
        if engine is None:
            raise ValueError("engine must be provided")
        if not isinstance(block_manager, BlockManager):
            raise ValueError("block_manager must be a BlockManager")
        if not isinstance(
            state_allocator,
            HybridStateSlotAllocator,
        ):
            raise ValueError(
                "state_allocator must be a HybridStateSlotAllocator"
            )
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        world_size = getattr(
            getattr(engine, "model_runner", None),
            "world_size",
            None,
        )
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
        ):
            raise ValueError(
                "engine ModelRunner world_size must be positive"
            )
        scheduler = getattr(engine, "scheduler", None)
        if (
            scheduler is None
            or getattr(scheduler, "block_manager", None)
            is not block_manager
        ):
            raise ValueError(
                "block_manager must be the Engine Scheduler "
                "BlockManager"
            )
        if getattr(
            scheduler,
            "hybrid_state_allocator",
            None,
        ) is not state_allocator:
            raise ValueError(
                "state_allocator must be the Engine Scheduler "
                "state allocator"
            )
        self.engine = engine
        self.block_manager = block_manager
        self.state_allocator = state_allocator
        self.timeout_s = float(timeout_s)
        self.world_size = world_size
        self._ticket_ids = count()
        self._poisoned_error = None
        self.last_ticket: Optional[
            Qwen35HybridPrefixEngineRestoreTicket
        ] = None

    def _ensure_healthy(self):
        if self._poisoned_error is not None:
            raise RuntimeError(
                "Engine hybrid prefix restore coordinator is poisoned: "
                f"{self._poisoned_error}"
            )

    def _poison(self, error):
        reason = str(error)
        if self._poisoned_error is None:
            self._poisoned_error = reason
        poison_transport = getattr(
            self.engine,
            "_poison_model_runner_ack_collector",
            None,
        )
        if poison_transport is not None:
            poison_transport(reason)

    def _validate_request(self, sequence, key, token_ids):
        if not isinstance(sequence, Sequence):
            raise ValueError("sequence must be a Sequence")
        if (
            sequence.block_table
            or sequence.num_cached_tokens != 0
            or sequence.num_computed_tokens != 0
        ):
            raise ValueError(
                "destination sequence already owns KV metadata"
            )
        if (
            sequence.hybrid_state_slot_id != -1
            or sequence.hybrid_state_generation != 0
        ):
            raise ValueError(
                "destination sequence already owns hybrid state"
            )
        if self.state_allocator.lease_for_request(
            sequence.seq_id
        ) is not None:
            raise ValueError(
                "destination request already has a state lease"
            )
        Qwen35HybridPrefixSnapshotCache._validate_key(key)
        Qwen35HybridPrefixSnapshotCache._validate_tokens(
            key,
            token_ids,
        )
        if key.block_size != self.block_manager.block_size:
            raise ValueError(
                "prefix key block size must match BlockManager"
            )
        if key.tensor_parallel_size != self.world_size:
            raise ValueError(
                "prefix key tensor parallel size must match Engine"
            )
        if (
            len(sequence.token_ids) < key.token_count
            or tuple(sequence.token_ids[:key.token_count])
            != token_ids
        ):
            raise ValueError(
                "destination sequence must start with exact prefix tokens"
            )
        terminal_hash = -1
        for start in range(
            0,
            len(token_ids),
            self.block_manager.block_size,
        ):
            terminal_hash = self.block_manager.compute_hash(
                list(
                    token_ids[
                        start:start + self.block_manager.block_size
                    ]
                ),
                terminal_hash,
            )
        if terminal_hash != key.terminal_block_hash:
            raise ValueError(
                "terminal block hash must match exact token chain"
            )

    def _reserve(
        self,
        sequence,
        key,
        token_ids,
    ):
        reservation = self.block_manager.reserve_sequence_blocks(
            sequence,
            max_cached_tokens=key.token_count,
        )
        lease = None
        try:
            if (
                reservation.cached_tokens != key.token_count
                or not reservation.block_identities
                or reservation.block_identities[-1][2]
                != key.terminal_block_hash
            ):
                self.block_manager.release_sequence_reservation(
                    reservation
                )
                return None
            Qwen35HybridPrefixSnapshotCache._validate_block_identities(
                reservation.block_identities,
                key=key,
            )
            lease = self.state_allocator.allocate(sequence.seq_id)
            payload = Qwen35HybridPrefixRestorePayload(
                ticket_id=next(self._ticket_ids),
                request_id=sequence.seq_id,
                key=key,
                token_ids=token_ids,
                block_identities=reservation.block_identities,
                lease=lease,
            )
            return Qwen35HybridPrefixEngineRestoreTicket(
                payload=payload,
                sequence=sequence,
                reservation=reservation,
            )
        except BaseException:
            if lease is not None:
                self.state_allocator.release(lease)
            if reservation.state == "reserved":
                self.block_manager.release_sequence_reservation(
                    reservation
                )
            raise

    def _validate_prepare_results(self, ticket, results):
        results = tuple(results)
        if len(results) != self.world_size:
            raise RuntimeError(
                "all-rank prepare result count is incomplete"
            )
        for participant_id, row in enumerate(results):
            if (
                not isinstance(row, dict)
                or row.get("ticket_id")
                != ticket.payload.ticket_id
                or row.get("participant_id") != participant_id
                or row.get("operation") != "prepare"
                or row.get("status")
                not in {"prepared", "miss", "error"}
                or not isinstance(row.get("detail"), str)
            ):
                raise RuntimeError(
                    "all-rank prepare result is invalid"
                )
        return results

    def _validate_pristine_sequence(self, ticket):
        sequence = ticket.sequence
        if (
            sequence.block_table
            or sequence.num_cached_tokens != 0
            or sequence.num_computed_tokens != 0
        ):
            raise ValueError(
                "destination sequence already owns KV metadata"
            )
        if (
            sequence.hybrid_state_slot_id != -1
            or sequence.hybrid_state_generation != 0
        ):
            raise ValueError(
                "destination sequence already owns hybrid state"
            )

    def _validate_reservation(self, ticket):
        reservation = ticket.reservation
        sequence = ticket.sequence
        self.block_manager._validate_sequence_reservation_structure(
            reservation
        )
        if reservation.state != "reserved":
            raise RuntimeError(
                "sequence reservation is not attachable"
            )
        if len(reservation.block_ids) != sequence.num_blocks:
            raise ValueError(
                "reservation block count must match sequence"
            )
        if (
            reservation.cached_tokens
            > self.block_manager.max_reusable_tokens(sequence)
        ):
            raise ValueError(
                "reservation exceeds sampleable prefix cap"
            )
        prefix_hash = -1
        for block_index, identity in enumerate(
            reservation.block_identities
        ):
            block_id, generation, block_hash = identity
            if reservation.block_ids[block_index] != block_id:
                raise RuntimeError(
                    "reservation identity order is inconsistent"
                )
            block_tokens = sequence.block(block_index)
            prefix_hash = self.block_manager.compute_hash(
                block_tokens,
                prefix_hash,
            )
            block = self.block_manager.blocks[block_id]
            if (
                block.generation != generation
                or block.hash != block_hash
                or block_hash != prefix_hash
                or block.token_ids != block_tokens
            ):
                raise RuntimeError(
                    "sequence reservation prefix identity is stale"
                )
        for block_id in reservation.block_ids:
            block = self.block_manager.blocks[block_id]
            if (
                block_id not in self.block_manager.used_block_ids
                or block.ref_count <= 0
            ):
                raise RuntimeError(
                    "sequence reservation ownership is stale"
                )

    def _validate_precommit(self, ticket):
        if ticket.state != "prepared":
            raise RuntimeError(
                f"restore ticket is not validatable: {ticket.state}"
            )
        self.state_allocator.validate(ticket.payload.lease)
        self._validate_pristine_sequence(ticket)
        self._validate_reservation(ticket)

    def _release_private_resources(self, ticket):
        errors = []
        try:
            lease = self.state_allocator.lease_for_request(
                ticket.payload.request_id
            )
            if lease == ticket.payload.lease:
                self.state_allocator.release(ticket.payload.lease)
            elif lease is not None:
                raise RuntimeError(
                    "allocator request lease changed before cleanup"
                )
        except BaseException as error:
            errors.append(error)
        try:
            if ticket.reservation.state == "reserved":
                self.block_manager.release_sequence_reservation(
                    ticket.reservation
                )
        except BaseException as error:
            errors.append(error)
        if errors:
            raise RuntimeError(
                "Engine resource cleanup failed: "
                f"{errors[0]}"
            ) from errors[0]

    def _rollback_private(self, ticket):
        try:
            self.engine.rollback_model_runner_hybrid_prefix_restore(
                ticket.payload,
                timeout_s=self.timeout_s,
            )
        except BaseException as error:
            ticket.state = "rollback_failed"
            self._poison(error)
            raise RuntimeError(
                "all-rank hybrid prefix rollback failed: "
                f"{error}"
            ) from error
        try:
            self._release_private_resources(ticket)
        except BaseException as error:
            ticket.state = "rollback_failed"
            self._poison(error)
            raise
        ticket.state = "rolled_back"

    def acquire(
        self,
        sequence: Sequence,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
    ) -> bool:
        self._ensure_healthy()
        self._validate_request(sequence, key, token_ids)
        ticket = self._reserve(sequence, key, token_ids)
        if ticket is None:
            return False
        self.last_ticket = ticket

        try:
            prepare_results = (
                self.engine.prepare_model_runner_hybrid_prefix_restore(
                    ticket.payload,
                    timeout_s=self.timeout_s,
                )
            )
            ticket.prepare_results = self._validate_prepare_results(
                ticket,
                prepare_results,
            )
        except BaseException:
            self._rollback_private(ticket)
            raise

        if any(
            row["status"] != "prepared"
            for row in ticket.prepare_results
        ):
            self._rollback_private(ticket)
            return False
        ticket.state = "prepared"

        try:
            self._validate_precommit(ticket)
            self.engine.validate_model_runner_hybrid_prefix_restore(
                ticket.payload,
                timeout_s=self.timeout_s,
            )
        except BaseException:
            self._rollback_private(ticket)
            raise

        try:
            self.block_manager.attach_sequence_reservation(
                ticket.reservation,
                ticket.sequence,
            )
            ticket.state = "published"
            ticket.sequence.hybrid_state_slot_id = (
                ticket.payload.lease.slot_id
            )
            ticket.sequence.hybrid_state_generation = (
                ticket.payload.lease.generation
            )
        except BaseException as error:
            if (
                ticket.reservation.state == "attached"
                or bool(ticket.sequence.block_table)
            ):
                ticket.state = "commit_failed"
                self._poison(error)
                raise
            self._rollback_private(ticket)
            raise

        try:
            self.engine.commit_model_runner_hybrid_prefix_restore(
                ticket.payload,
                timeout_s=self.timeout_s,
            )
        except BaseException as error:
            ticket.state = "commit_failed"
            self._poison(error)
            raise
        ticket.state = "committed"
        return True
