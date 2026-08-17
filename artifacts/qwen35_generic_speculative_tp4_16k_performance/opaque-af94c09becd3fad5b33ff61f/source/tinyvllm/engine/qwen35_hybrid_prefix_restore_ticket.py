from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Optional

from tinyvllm.engine.block_manager import (
    BlockManager,
    SequenceBlockReservation,
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
from tinyvllm.engine.qwen35_hybrid_prefix_int8_cache import (
    Qwen35HybridPrefixInt8SnapshotCache,
)
from tinyvllm.engine.sequence import Sequence


_PREPARE_STATUSES = {
    "prepared",
    "miss",
    "error",
}


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


@dataclass(frozen=True)
class Qwen35HybridPrefixRestorePayload:
    ticket_id: int
    request_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    lease: HybridStateLease

    def __post_init__(self):
        _non_negative_integer(self.ticket_id, "ticket_id")
        _non_negative_integer(self.request_id, "request_id")
        Qwen35HybridPrefixSnapshotCache._validate_identity(
            self.key,
            self.token_ids,
            self.block_identities,
        )
        if not isinstance(self.lease, HybridStateLease):
            raise ValueError("lease must be a HybridStateLease")
        if self.lease.request_id != self.request_id:
            raise ValueError(
                "lease request_id must match payload request_id"
            )


@dataclass(frozen=True)
class Qwen35HybridPrefixPrepareAck:
    ticket_id: int
    participant_id: int
    status: str
    detail: str = ""

    def __post_init__(self):
        _non_negative_integer(self.ticket_id, "ticket_id")
        _non_negative_integer(
            self.participant_id,
            "participant_id",
        )
        if self.status not in _PREPARE_STATUSES:
            raise ValueError(
                f"unsupported prepare acknowledgement status: {self.status}"
            )
        if not isinstance(self.detail, str):
            raise ValueError("acknowledgement detail must be a string")


@dataclass
class Qwen35HybridPrefixRestoreTicket:
    payload: Qwen35HybridPrefixRestorePayload
    sequence: Sequence
    reservation: SequenceBlockReservation
    participant_ids: tuple[int, ...]
    acknowledgements: tuple[Qwen35HybridPrefixPrepareAck, ...] = ()
    state: str = "reserved"


class Qwen35HybridPrefixRestoreParticipant:

    def __init__(
        self,
        participant_id: int,
        pool: HybridStateTensorPool,
        snapshot_cache: (
            Qwen35HybridPrefixSnapshotCache
            | Qwen35HybridPrefixInt8SnapshotCache
        ),
    ):
        self.participant_id = _non_negative_integer(
            participant_id,
            "participant_id",
        )
        if not isinstance(pool, HybridStateTensorPool):
            raise ValueError("pool must be a HybridStateTensorPool")
        if not isinstance(
            snapshot_cache,
            (
                Qwen35HybridPrefixSnapshotCache,
                Qwen35HybridPrefixInt8SnapshotCache,
            ),
        ):
            raise ValueError(
                "snapshot_cache must be a supported "
                "Qwen35 hybrid prefix snapshot cache"
            )
        if snapshot_cache.state_transaction.pool is not pool:
            raise ValueError(
                "snapshot cache transaction must use participant pool"
            )
        self.pool = pool
        self.snapshot_cache = snapshot_cache
        self._prepared = {}
        self._terminal = {}
        self._terminal_payloads = {}

    def _ack(self, payload, status, detail=""):
        return Qwen35HybridPrefixPrepareAck(
            ticket_id=payload.ticket_id,
            participant_id=self.participant_id,
            status=status,
            detail=detail,
        )

    @staticmethod
    def _validate_payload(payload):
        if not isinstance(
            payload,
            Qwen35HybridPrefixRestorePayload,
        ):
            raise ValueError(
                "payload must be a Qwen35HybridPrefixRestorePayload"
            )

    def prepare(
        self,
        payload: Qwen35HybridPrefixRestorePayload,
    ) -> Qwen35HybridPrefixPrepareAck:
        self._validate_payload(payload)
        prepared = self._prepared.get(payload.ticket_id)
        if prepared is not None:
            if prepared == payload:
                return self._ack(payload, "prepared")
            return self._ack(
                payload,
                "error",
                "ticket id already prepared with different payload",
            )
        terminal = self._terminal.get(payload.ticket_id)
        if terminal is not None:
            return self._ack(
                payload,
                "error",
                f"ticket is terminal: {terminal}",
            )

        activated = False
        try:
            self.pool.activate(payload.lease)
            activated = True
            restored = self.snapshot_cache.acquire(
                payload.key,
                payload.token_ids,
                payload.block_identities,
                (payload.lease,),
            )
            if not restored:
                self.pool.release(payload.lease)
                self._terminal[payload.ticket_id] = "rolled_back"
                self._terminal_payloads[payload.ticket_id] = payload
                return self._ack(payload, "miss")
            self._prepared[payload.ticket_id] = payload
            return self._ack(payload, "prepared")
        except Exception as error:
            cleanup_error = None
            if activated:
                try:
                    self.pool.release(payload.lease)
                except Exception as release_error:
                    cleanup_error = release_error
            if cleanup_error is not None:
                self._prepared[payload.ticket_id] = payload
                return self._ack(
                    payload,
                    "error",
                    f"{error}; cleanup failed: {cleanup_error}",
                )
            self._terminal[payload.ticket_id] = "rolled_back"
            self._terminal_payloads[payload.ticket_id] = payload
            return self._ack(payload, "error", str(error))

    def validate_prepared(
        self,
        payload: Qwen35HybridPrefixRestorePayload,
    ) -> None:
        self._validate_payload(payload)
        current = self._prepared.get(payload.ticket_id)
        if current != payload:
            terminal = self._terminal.get(payload.ticket_id)
            if terminal is not None:
                raise RuntimeError(
                    f"participant ticket is terminal: {terminal}"
                )
            raise RuntimeError(
                "participant does not own the exact prepared payload"
            )
        self.pool.validate(payload.lease)

    def commit(
        self,
        payload: Qwen35HybridPrefixRestorePayload,
    ) -> None:
        self.validate_prepared(payload)
        del self._prepared[payload.ticket_id]
        self._terminal[payload.ticket_id] = "committed"
        self._terminal_payloads[payload.ticket_id] = payload

    def rollback(
        self,
        payload: Qwen35HybridPrefixRestorePayload,
    ) -> None:
        self._validate_payload(payload)
        terminal = self._terminal.get(payload.ticket_id)
        if terminal is not None:
            if (
                terminal == "rolled_back"
                and self._terminal_payloads.get(payload.ticket_id)
                == payload
            ):
                return
            raise RuntimeError(
                f"participant ticket is terminal: {terminal}"
            )
        current = self._prepared.get(payload.ticket_id)
        if current != payload:
            raise RuntimeError(
                "participant does not own the exact prepared payload"
            )
        self.pool.release(payload.lease)
        del self._prepared[payload.ticket_id]
        self._terminal[payload.ticket_id] = "rolled_back"
        self._terminal_payloads[payload.ticket_id] = payload


class Qwen35HybridPrefixRestoreCoordinator:

    def __init__(
        self,
        block_manager: BlockManager,
        state_allocator: HybridStateSlotAllocator,
        participants: tuple[
            Qwen35HybridPrefixRestoreParticipant,
            ...,
        ],
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
        if not isinstance(participants, tuple) or not participants:
            raise ValueError(
                "participants must be a non-empty tuple"
            )
        if any(
            not isinstance(
                participant,
                Qwen35HybridPrefixRestoreParticipant,
            )
            for participant in participants
        ):
            raise ValueError(
                "participants must contain restore participants"
            )
        participant_ids = tuple(
            participant.participant_id
            for participant in participants
        )
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("participant ids must be unique")
        if any(
            participant.pool.capacity != state_allocator.capacity
            for participant in participants
        ):
            raise ValueError(
                "participant and allocator capacities must match"
            )
        self.block_manager = block_manager
        self.state_allocator = state_allocator
        self.participants = tuple(sorted(
            participants,
            key=lambda participant: participant.participant_id,
        ))
        self.participant_ids = tuple(
            participant.participant_id
            for participant in self.participants
        )
        self._ticket_ids = count()
        self._poisoned_error = None

    def _ensure_healthy(self):
        if self._poisoned_error is not None:
            raise RuntimeError(
                "hybrid prefix restore coordinator is poisoned after "
                f"rollback failure: {self._poisoned_error}"
            )

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = str(error)

    def _validate_reserve_request(
        self,
        sequence,
        key,
        token_ids,
    ):
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
        if key.tensor_parallel_size != len(self.participants):
            raise ValueError(
                "prefix key tensor parallel size must match "
                "participant count"
            )
        if (
            len(sequence.token_ids) < key.token_count
            or tuple(
                sequence.token_ids[:key.token_count]
            ) != token_ids
        ):
            raise ValueError(
                "destination sequence must start with exact prefix tokens"
            )
        prefix_hash = -1
        for start in range(
            0,
            len(token_ids),
            self.block_manager.block_size,
        ):
            prefix_hash = self.block_manager.compute_hash(
                list(
                    token_ids[
                        start:start + self.block_manager.block_size
                    ]
                ),
                prefix_hash,
            )
        if prefix_hash != key.terminal_block_hash:
            raise ValueError(
                "terminal block hash must match exact token chain"
            )

    @staticmethod
    def _validate_ticket(ticket):
        if not isinstance(
            ticket,
            Qwen35HybridPrefixRestoreTicket,
        ):
            raise ValueError(
                "ticket must be a Qwen35HybridPrefixRestoreTicket"
            )

    def _release_engine_resources(self, ticket):
        lease = self.state_allocator.lease_for_request(
            ticket.payload.request_id
        )
        if lease == ticket.payload.lease:
            self.state_allocator.release(ticket.payload.lease)
        if ticket.reservation.state == "reserved":
            self.block_manager.release_sequence_reservation(
                ticket.reservation
            )

    def reserve(
        self,
        sequence: Sequence,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
    ) -> Optional[Qwen35HybridPrefixRestoreTicket]:
        self._ensure_healthy()
        self._validate_reserve_request(
            sequence,
            key,
            token_ids,
        )
        reservation = self.block_manager.reserve_sequence_blocks(
            sequence,
            max_cached_tokens=key.token_count,
        )
        lease = None
        try:
            if (
                reservation.cached_tokens != key.token_count
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
            return Qwen35HybridPrefixRestoreTicket(
                payload=payload,
                sequence=sequence,
                reservation=reservation,
                participant_ids=self.participant_ids,
            )
        except BaseException:
            if lease is not None:
                self.state_allocator.release(lease)
            if reservation.state == "reserved":
                self.block_manager.release_sequence_reservation(
                    reservation
                )
            raise

    def _rollback_prepared_participants(
        self,
        ticket,
        prepared_participants,
    ):
        errors = []
        for participant in reversed(tuple(prepared_participants)):
            try:
                participant.rollback(ticket.payload)
            except Exception as error:
                errors.append(error)
        return errors

    def prepare(
        self,
        ticket: Qwen35HybridPrefixRestoreTicket,
    ) -> tuple[Qwen35HybridPrefixPrepareAck, ...]:
        self._validate_ticket(ticket)
        if ticket.state != "reserved":
            raise RuntimeError(
                f"restore ticket is not preparable: {ticket.state}"
            )
        acknowledgements = []
        prepared_participants = []
        try:
            for participant in self.participants:
                acknowledgement = participant.prepare(
                    ticket.payload
                )
                acknowledgements.append(acknowledgement)
                if acknowledgement.status != "prepared":
                    rollback_errors = self._rollback_prepared_participants(
                        ticket,
                        prepared_participants,
                    )
                    self._release_engine_resources(ticket)
                    ticket.acknowledgements = tuple(
                        acknowledgements
                    )
                    if rollback_errors:
                        ticket.state = "rollback_failed"
                        first_error = rollback_errors[0]
                        self._poison(first_error)
                        raise RuntimeError(
                            "participant rollback failed during prepare: "
                            f"{first_error}"
                        ) from first_error
                    ticket.state = "rolled_back"
                    return ticket.acknowledgements
                prepared_participants.append(participant)
            ticket.acknowledgements = tuple(acknowledgements)
            ticket.state = "prepared"
            return ticket.acknowledgements
        except BaseException:
            if ticket.state == "rollback_failed":
                raise
            rollback_errors = self._rollback_prepared_participants(
                ticket,
                prepared_participants,
            )
            self._release_engine_resources(ticket)
            ticket.acknowledgements = tuple(acknowledgements)
            if rollback_errors:
                ticket.state = "rollback_failed"
                first_error = rollback_errors[0]
                self._poison(first_error)
                raise RuntimeError(
                    "participant rollback failed during prepare: "
                    f"{first_error}"
                ) from first_error
            ticket.state = "rolled_back"
            raise

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

    def _validate_reservation_precommit(self, ticket):
        reservation = ticket.reservation
        sequence = ticket.sequence
        self.block_manager._validate_sequence_reservation_structure(
            reservation
        )
        if reservation.state != "reserved":
            raise RuntimeError(
                "sequence reservation is not attachable: "
                f"{reservation.state}"
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
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise ValueError(
                    "reservation block identity is malformed"
                )
            block_id, generation, block_hash = identity
            if reservation.block_ids[block_index] != block_id:
                raise ValueError(
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
                    "sequence reservation block ownership is stale"
                )

    def _validate_commit(self, ticket):
        if ticket.state != "prepared":
            raise RuntimeError(
                f"restore ticket is not committable: {ticket.state}"
            )
        if ticket.participant_ids != self.participant_ids:
            raise RuntimeError(
                "restore ticket participant identity is stale"
            )
        if len(ticket.acknowledgements) != len(self.participants):
            raise RuntimeError(
                "restore ticket acknowledgement count is incomplete"
            )
        for participant, acknowledgement in zip(
            self.participants,
            ticket.acknowledgements,
        ):
            if (
                acknowledgement.ticket_id
                != ticket.payload.ticket_id
                or acknowledgement.participant_id
                != participant.participant_id
                or acknowledgement.status != "prepared"
            ):
                raise RuntimeError(
                    "restore ticket acknowledgement is invalid"
                )
        self.state_allocator.validate(ticket.payload.lease)
        self._validate_pristine_sequence(ticket)
        for participant in self.participants:
            participant.validate_prepared(ticket.payload)
        self._validate_reservation_precommit(ticket)

    def commit(
        self,
        ticket: Qwen35HybridPrefixRestoreTicket,
    ) -> None:
        self._validate_ticket(ticket)
        self._validate_commit(ticket)
        self.block_manager.attach_sequence_reservation(
            ticket.reservation,
            ticket.sequence,
        )
        ticket.sequence.hybrid_state_slot_id = (
            ticket.payload.lease.slot_id
        )
        ticket.sequence.hybrid_state_generation = (
            ticket.payload.lease.generation
        )
        for participant in self.participants:
            participant.commit(ticket.payload)
        ticket.state = "committed"

    def rollback(
        self,
        ticket: Qwen35HybridPrefixRestoreTicket,
    ) -> None:
        self._validate_ticket(ticket)
        if ticket.state not in {"reserved", "prepared"}:
            raise RuntimeError(
                f"restore ticket is not rollbackable: {ticket.state}"
            )
        rollback_errors = []
        if ticket.state == "prepared":
            rollback_errors = self._rollback_prepared_participants(
                ticket,
                self.participants,
            )
        self._release_engine_resources(ticket)
        if rollback_errors:
            ticket.state = "rollback_failed"
            first_error = rollback_errors[0]
            self._poison(first_error)
            raise RuntimeError(
                "participant rollback failed: "
                f"{first_error}"
            ) from first_error
        ticket.state = "rolled_back"
