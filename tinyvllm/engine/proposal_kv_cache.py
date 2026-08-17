from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.proposal_kv_allocator import (
    ProposalKVEntryAllocator,
    ProposalKVEntryIdentity,
)


@dataclass
class ProposalKVSequenceState:
    sequence_id: int
    sequence_epoch: int
    committed_entry_identities: tuple[
        ProposalKVEntryIdentity, ...
    ] = ()
    active_transaction_id: str | None = None
    active_ticket_id: str | None = None


@dataclass
class ProposalKVTransaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    original_committed_length: int
    staged_entry_identities: tuple[
        ProposalKVEntryIdentity, ...
    ]
    materialized_entry_count: int = 0
    state: str = "reserved"


@dataclass
class ProposalKVFinalizeTicket:
    ticket_id: str
    transaction_id: str
    commit_entry_count: int
    retire_entry_identities: tuple[
        ProposalKVEntryIdentity, ...
    ]
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

    def __init__(self, entry_allocator: ProposalKVEntryAllocator):
        required_methods = (
            "reserve_entries",
            "commit_entries",
            "retire_entries",
            "authority_snapshot",
        )
        if any(
            not callable(getattr(entry_allocator, method_name, None))
            for method_name in required_methods
        ):
            raise ValueError(
                "entry_allocator does not satisfy the proposal KV "
                "allocator contract"
            )
        self._entry_allocator = entry_allocator
        self._sequence_states: dict[int, ProposalKVSequenceState] = {}
        self._transactions: dict[str, ProposalKVTransaction] = {}
        self._tickets: dict[str, ProposalKVFinalizeTicket] = {}
        self._owned_entry_identities: set[
            ProposalKVEntryIdentity
        ] = set()
        self._next_transaction_id = 1
        self._next_ticket_id = 1

    @property
    def entry_allocator(self) -> ProposalKVEntryAllocator:
        return self._entry_allocator

    def sequence_state(
        self,
        sequence_id: int,
    ) -> ProposalKVSequenceState | None:
        _nonnegative_integer(sequence_id, "sequence_id")
        return self._sequence_states.get(sequence_id)

    def committed_entry_identities(
        self,
        sequence_id: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        state = self.sequence_state(sequence_id)
        return (
            ()
            if state is None
            else state.committed_entry_identities
        )

    def committed_length(self, sequence_id: int) -> int:
        return len(self.committed_entry_identities(sequence_id))

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
            "owned_entry_count": len(
                self._owned_entry_identities
            ),
            "entry_allocator": (
                self._entry_allocator.authority_snapshot()
            ),
            "transactions": [
                {
                    "transaction_id": transaction.transaction_id,
                    "sequence_id": transaction.sequence_id,
                    "sequence_epoch": transaction.sequence_epoch,
                    "original_committed_length": (
                        transaction.original_committed_length
                    ),
                    "staged_entry_count": len(
                        transaction.staged_entry_identities
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
                        ticket.retire_entry_identities
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

    def _reserve_entries(
        self,
        count: int,
    ) -> tuple[ProposalKVEntryIdentity, ...]:
        identities = self._entry_allocator.reserve_entries(count)
        if not isinstance(identities, tuple):
            raise RuntimeError("reserve_entries must return a tuple")
        if len(identities) != count:
            raise RuntimeError(
                "reserve_entries returned an unexpected entry count"
            )
        if len(set(identities)) != len(identities):
            raise RuntimeError(
                "reserve_entries returned duplicate identities"
            )
        if any(
            not isinstance(identity, ProposalKVEntryIdentity)
            for identity in identities
        ):
            raise RuntimeError(
                "reserve_entries returned an invalid identity"
            )
        if self._owned_entry_identities.intersection(identities):
            raise RuntimeError(
                "reserve_entries returned an already-owned identity"
            )
        self._owned_entry_identities.update(identities)
        return identities

    def _retire_entries(
        self,
        identities: tuple[ProposalKVEntryIdentity, ...],
    ) -> None:
        if not identities:
            return
        if not set(identities).issubset(
            self._owned_entry_identities
        ):
            raise RuntimeError("proposal KV entry ownership is stale")
        self._entry_allocator.retire_entries(
            identities,
            writeback=False,
        )
        self._owned_entry_identities.difference_update(identities)

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

        staged_entry_identities = self._reserve_entries(
            staged_entry_count
        )
        transaction_id = (
            f"proposal-kv-transaction-{self._next_transaction_id}"
        )
        self._next_transaction_id += 1
        transaction = ProposalKVTransaction(
            transaction_id=transaction_id,
            sequence_id=sequence_id,
            sequence_epoch=sequence_epoch,
            original_committed_length=len(
                state.committed_entry_identities
            ),
            staged_entry_identities=staged_entry_identities,
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
        if materialized_entry_count > len(
            owned.staged_entry_identities
        ):
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
            transaction.staged_entry_identities
        ):
            raise RuntimeError(
                "all staged entries must be materialized before finalize"
            )
        accepted_proposal_tokens = _nonnegative_integer(
            accepted_proposal_tokens,
            "accepted_proposal_tokens",
        )
        max_accepted = (
            len(transaction.staged_entry_identities) + 1
        )
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
            or len(state.committed_entry_identities)
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
            retire_entry_identities=(
                transaction.staged_entry_identities[
                commit_entry_count:
                ]
            ),
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
            or len(state.committed_entry_identities)
            != transaction.original_committed_length
        ):
            raise RuntimeError(
                "proposal KV finalize ownership is stale"
            )
        return ticket, transaction, state

    def commit_finalize(self, ticket_id: str) -> None:
        ticket, transaction, state = self._prepared_ticket(ticket_id)
        committed_entries = transaction.staged_entry_identities[
            :ticket.commit_entry_count
        ]
        if committed_entries:
            self._entry_allocator.commit_entries(committed_entries)
        self._retire_entries(ticket.retire_entry_identities)
        state.committed_entry_identities = (
            state.committed_entry_identities + committed_entries
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
        self._retire_entries(
            transaction.staged_entry_identities
        )
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
        self._retire_entries(
            transaction.staged_entry_identities
        )
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
        self._retire_entries(
            state.committed_entry_identities
        )
        del self._sequence_states[sequence_id]
