from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProposalKVSequenceState:
    sequence_id: int
    sequence_epoch: int
    committed_slot_ids: tuple[int, ...] = ()
    active_transaction_id: str | None = None
    active_ticket_id: str | None = None


@dataclass
class ProposalKVTransaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    original_committed_length: int
    staged_slot_ids: tuple[int, ...]
    materialized_entry_count: int = 0
    state: str = "reserved"


@dataclass
class ProposalKVFinalizeTicket:
    ticket_id: str
    transaction_id: str
    commit_entry_count: int
    release_slot_ids: tuple[int, ...]
    state: str = "prepared"


def _nonnegative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


class ProposalKVCache:

    def __init__(self, physical_store):
        reserve_slots = getattr(physical_store, "reserve_slots", None)
        release_slots = getattr(physical_store, "release_slots", None)
        if not callable(reserve_slots) or not callable(release_slots):
            raise ValueError(
                "physical_store must expose reserve_slots and release_slots"
            )
        self._physical_store = physical_store
        self._sequence_states: dict[int, ProposalKVSequenceState] = {}
        self._transactions: dict[str, ProposalKVTransaction] = {}
        self._tickets: dict[str, ProposalKVFinalizeTicket] = {}
        self._owned_slot_ids: set[int] = set()
        self._next_transaction_id = 1
        self._next_ticket_id = 1

    @property
    def physical_store(self):
        return self._physical_store

    def sequence_state(
        self,
        sequence_id: int,
    ) -> ProposalKVSequenceState | None:
        _nonnegative_integer(sequence_id, "sequence_id")
        return self._sequence_states.get(sequence_id)

    def committed_slot_ids(
        self,
        sequence_id: int,
    ) -> tuple[int, ...]:
        state = self.sequence_state(sequence_id)
        return () if state is None else state.committed_slot_ids

    def committed_length(self, sequence_id: int) -> int:
        return len(self.committed_slot_ids(sequence_id))

    def authority_snapshot(self) -> dict:
        return {
            "active_sequence_count": len(self._sequence_states),
            "active_transaction_count": sum(
                transaction.state
                in ("reserved", "materialized", "prepared")
                for transaction in self._transactions.values()
            ),
            "prepared_ticket_count": sum(
                ticket.state == "prepared"
                for ticket in self._tickets.values()
            ),
            "owned_slot_count": len(self._owned_slot_ids),
            "transactions": [
                {
                    "transaction_id": transaction.transaction_id,
                    "sequence_id": transaction.sequence_id,
                    "sequence_epoch": transaction.sequence_epoch,
                    "original_committed_length": (
                        transaction.original_committed_length
                    ),
                    "staged_entry_count": len(
                        transaction.staged_slot_ids
                    ),
                    "materialized_entry_count": (
                        transaction.materialized_entry_count
                    ),
                    "state": transaction.state,
                }
                for transaction in self._transactions.values()
            ],
            "tickets": [
                {
                    "ticket_id": ticket.ticket_id,
                    "transaction_id": ticket.transaction_id,
                    "commit_entry_count": ticket.commit_entry_count,
                    "release_entry_count": len(
                        ticket.release_slot_ids
                    ),
                    "state": ticket.state,
                }
                for ticket in self._tickets.values()
            ],
        }

    def transaction(
        self,
        transaction_id: str,
    ) -> ProposalKVTransaction | None:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError(
                "transaction_id must be a non-empty string"
            )
        return self._transactions.get(transaction_id)

    def _reserve_slots(self, count: int) -> tuple[int, ...]:
        slot_ids = self._physical_store.reserve_slots(count)
        if not isinstance(slot_ids, tuple):
            raise RuntimeError("reserve_slots must return a tuple")
        if len(slot_ids) != count:
            raise RuntimeError(
                "reserve_slots returned an unexpected slot count"
            )
        if len(set(slot_ids)) != len(slot_ids):
            raise RuntimeError("reserve_slots returned duplicate slot IDs")
        if any(
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            for slot_id in slot_ids
        ):
            raise RuntimeError(
                "reserve_slots returned an invalid slot ID"
            )
        if self._owned_slot_ids.intersection(slot_ids):
            raise RuntimeError(
                "reserve_slots returned an already-owned slot ID"
            )
        self._owned_slot_ids.update(slot_ids)
        return slot_ids

    def _release_slots(self, slot_ids: tuple[int, ...]) -> None:
        if not slot_ids:
            return
        if not set(slot_ids).issubset(self._owned_slot_ids):
            raise RuntimeError("proposal KV slot ownership is stale")
        self._physical_store.release_slots(slot_ids)
        self._owned_slot_ids.difference_update(slot_ids)

    def begin(
        self,
        sequence_id: int,
        sequence_epoch: int,
        staged_entry_count: int,
    ) -> ProposalKVTransaction:
        sequence_id = _nonnegative_integer(sequence_id, "sequence_id")
        sequence_epoch = _nonnegative_integer(
            sequence_epoch,
            "sequence_epoch",
        )
        staged_entry_count = _nonnegative_integer(
            staged_entry_count,
            "staged_entry_count",
        )
        state = self._sequence_states.get(sequence_id)
        if state is None:
            state = ProposalKVSequenceState(
                sequence_id=sequence_id,
                sequence_epoch=sequence_epoch,
            )
            self._sequence_states[sequence_id] = state
        elif state.sequence_epoch != sequence_epoch:
            raise RuntimeError("sequence epoch is stale")
        if (
            state.active_transaction_id is not None
            or state.active_ticket_id is not None
        ):
            raise RuntimeError(
                "sequence already has an active transaction"
            )

        staged_slot_ids = self._reserve_slots(staged_entry_count)
        transaction_id = (
            f"proposal-kv-transaction-{self._next_transaction_id}"
        )
        self._next_transaction_id += 1
        transaction = ProposalKVTransaction(
            transaction_id=transaction_id,
            sequence_id=sequence_id,
            sequence_epoch=sequence_epoch,
            original_committed_length=len(state.committed_slot_ids),
            staged_slot_ids=staged_slot_ids,
        )
        self._transactions[transaction_id] = transaction
        state.active_transaction_id = transaction_id
        return transaction

    def _owned_transaction(
        self,
        transaction: ProposalKVTransaction,
    ) -> ProposalKVTransaction:
        if not isinstance(transaction, ProposalKVTransaction):
            raise ValueError(
                "transaction must be a ProposalKVTransaction"
            )
        owned = self._transactions.get(transaction.transaction_id)
        if owned is not transaction:
            raise ValueError(
                "transaction does not belong to this proposal KV cache"
            )
        return owned

    def mark_materialized(
        self,
        transaction: ProposalKVTransaction,
        materialized_entry_count: int,
    ) -> None:
        owned = self._owned_transaction(transaction)
        if owned.state != "reserved":
            raise RuntimeError(
                "transaction is not reserved for materialization"
            )
        materialized_entry_count = _nonnegative_integer(
            materialized_entry_count,
            "materialized_entry_count",
        )
        if materialized_entry_count > len(owned.staged_slot_ids):
            raise ValueError(
                "materialized entry count exceeds staged entry count"
            )
        owned.materialized_entry_count = materialized_entry_count
        owned.state = "materialized"

    def prepare_finalize(
        self,
        transaction_id: str,
        *,
        accepted_proposal_tokens: int,
    ) -> ProposalKVFinalizeTicket:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError(
                "transaction_id must be a non-empty string"
            )
        transaction = self._transactions.get(transaction_id)
        if transaction is None:
            raise ValueError("unknown transaction ID")
        if transaction.state == "prepared":
            raise RuntimeError("transaction is already prepared")
        if transaction.state != "materialized":
            raise RuntimeError(
                "transaction must be fully materialized before finalize"
            )
        if transaction.materialized_entry_count != len(
            transaction.staged_slot_ids
        ):
            raise RuntimeError(
                "all staged entries must be materialized before finalize"
            )
        accepted_proposal_tokens = _nonnegative_integer(
            accepted_proposal_tokens,
            "accepted_proposal_tokens",
        )
        max_accepted = len(transaction.staged_slot_ids) + 1
        if accepted_proposal_tokens > max_accepted:
            raise ValueError(
                "accepted proposal token count exceeds staged proposal"
            )
        state = self._sequence_states.get(transaction.sequence_id)
        if (
            state is None
            or state.sequence_epoch != transaction.sequence_epoch
            or state.active_transaction_id != transaction_id
            or state.active_ticket_id is not None
            or len(state.committed_slot_ids)
            != transaction.original_committed_length
        ):
            raise RuntimeError(
                "proposal KV transaction ownership is stale"
            )

        commit_entry_count = max(accepted_proposal_tokens - 1, 0)
        ticket_id = f"proposal-kv-ticket-{self._next_ticket_id}"
        self._next_ticket_id += 1
        ticket = ProposalKVFinalizeTicket(
            ticket_id=ticket_id,
            transaction_id=transaction_id,
            commit_entry_count=commit_entry_count,
            release_slot_ids=transaction.staged_slot_ids[
                commit_entry_count:
            ],
        )
        self._tickets[ticket_id] = ticket
        transaction.state = "prepared"
        state.active_ticket_id = ticket_id
        return ticket

    def _prepared_ticket(
        self,
        ticket_id: str,
    ) -> tuple[
        ProposalKVFinalizeTicket,
        ProposalKVTransaction,
        ProposalKVSequenceState,
    ]:
        if not isinstance(ticket_id, str) or not ticket_id:
            raise ValueError("ticket_id must be a non-empty string")
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise ValueError("unknown finalize ticket ID")
        if ticket.state == "committed":
            raise RuntimeError("finalize ticket is already committed")
        if ticket.state == "rolled_back":
            raise RuntimeError("finalize ticket was rolled back")
        if ticket.state != "prepared":
            raise RuntimeError("finalize ticket is not prepared")
        transaction = self._transactions[ticket.transaction_id]
        state = self._sequence_states.get(transaction.sequence_id)
        if (
            transaction.state != "prepared"
            or state is None
            or state.sequence_epoch != transaction.sequence_epoch
            or state.active_transaction_id
            != transaction.transaction_id
            or state.active_ticket_id != ticket_id
            or len(state.committed_slot_ids)
            != transaction.original_committed_length
        ):
            raise RuntimeError(
                "proposal KV finalize ownership is stale"
            )
        return ticket, transaction, state

    def commit_finalize(self, ticket_id: str) -> None:
        ticket, transaction, state = self._prepared_ticket(ticket_id)
        committed_slots = transaction.staged_slot_ids[
            :ticket.commit_entry_count
        ]
        self._release_slots(ticket.release_slot_ids)
        state.committed_slot_ids = (
            state.committed_slot_ids + committed_slots
        )
        state.active_transaction_id = None
        state.active_ticket_id = None
        transaction.state = "committed"
        ticket.state = "committed"

    def rollback_finalize(self, ticket_id: str) -> None:
        ticket = self._tickets.get(ticket_id)
        if ticket is not None and ticket.state == "committed":
            raise RuntimeError("finalize ticket was committed")
        if ticket is not None and ticket.state == "rolled_back":
            raise RuntimeError(
                "finalize ticket is already rolled back"
            )
        ticket, transaction, state = self._prepared_ticket(ticket_id)
        self._release_slots(transaction.staged_slot_ids)
        state.active_transaction_id = None
        state.active_ticket_id = None
        transaction.state = "rolled_back"
        ticket.state = "rolled_back"

    def abort(self, transaction_id: str) -> None:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError(
                "transaction_id must be a non-empty string"
            )
        transaction = self._transactions.get(transaction_id)
        if transaction is None:
            raise ValueError("unknown transaction ID")
        if transaction.state not in ("reserved", "materialized"):
            raise RuntimeError(
                "transaction cannot be aborted in its current state"
            )
        state = self._sequence_states.get(transaction.sequence_id)
        if (
            state is None
            or state.active_transaction_id != transaction_id
            or state.active_ticket_id is not None
        ):
            raise RuntimeError(
                "proposal KV transaction ownership is stale"
            )
        self._release_slots(transaction.staged_slot_ids)
        state.active_transaction_id = None
        transaction.state = "aborted"

    def release_sequence(
        self,
        sequence_id: int,
        *,
        sequence_epoch: int,
    ) -> None:
        sequence_id = _nonnegative_integer(sequence_id, "sequence_id")
        sequence_epoch = _nonnegative_integer(
            sequence_epoch,
            "sequence_epoch",
        )
        state = self._sequence_states.get(sequence_id)
        if state is None:
            return
        if state.sequence_epoch != sequence_epoch:
            raise RuntimeError("sequence epoch is stale")
        if state.active_transaction_id is not None:
            raise RuntimeError(
                "sequence has an active transaction"
            )
        if state.active_ticket_id is not None:
            raise RuntimeError("sequence has an active finalize ticket")
        self._release_slots(state.committed_slot_ids)
        del self._sequence_states[sequence_id]
