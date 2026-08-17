from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.speculative_proposal_executor import (
    ProposalFinalizeRow,
)
from tinyvllm.speculative.adapter import DraftProposal


@dataclass(frozen=True)
class ProposalKVRegistration:
    sequence_id: int
    sequence_epoch: int
    proposal: DraftProposal


@dataclass(frozen=True)
class _ProposalKVBatchFinalize:
    underlying_ticket_ids: tuple[str, ...]
    transaction_ids: tuple[str, ...]


def _nonnegative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class ProposalKVLifecycleCoordinator:

    def __init__(
        self,
        proposal_kv_cache: ProposalKVCache,
        *,
        ticket_namespace: str,
    ):
        if not isinstance(proposal_kv_cache, ProposalKVCache):
            raise ValueError(
                "proposal_kv_cache must be a ProposalKVCache"
            )
        if (
            not isinstance(ticket_namespace, str)
            or not ticket_namespace
        ):
            raise ValueError(
                "ticket_namespace must be a non-empty string"
            )
        self.proposal_kv_cache = proposal_kv_cache
        self.ticket_namespace = ticket_namespace
        self._active_transactions: dict[str, tuple[int, int]] = {}
        self._batch_tickets: dict[
            str,
            _ProposalKVBatchFinalize,
        ] = {}
        self._next_batch_ticket_id = 1
        self._authority_rows: dict[str, dict] = {}
        self._release_rows: list[dict] = []

    @property
    def active_transaction_count(self) -> int:
        return len(self._active_transactions)

    @property
    def prepared_ticket_count(self) -> int:
        return len(self._batch_tickets)

    def _abort_unregistered_transactions(
        self,
        transaction_ids: tuple[str, ...],
    ) -> None:
        first_error = None
        for transaction_id in reversed(transaction_ids):
            if transaction_id in self._active_transactions:
                continue
            try:
                transaction = self.proposal_kv_cache.transaction(
                    transaction_id
                )
                if (
                    transaction is not None
                    and transaction.state
                    in ("reserved", "materialized")
                ):
                    self.proposal_kv_cache.abort(transaction_id)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def register_batch(
        self,
        rows: tuple[ProposalKVRegistration, ...],
    ) -> tuple[DraftProposal, ...]:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "proposal KV registration rows must be a non-empty tuple"
            )
        cleanup_transaction_ids = tuple(dict.fromkeys(
            row.proposal.proposal_transaction_id
            for row in rows
            if (
                isinstance(row, ProposalKVRegistration)
                and isinstance(row.proposal, DraftProposal)
                and row.proposal.token_ids
                and isinstance(
                    row.proposal.proposal_transaction_id,
                    str,
                )
                and row.proposal.proposal_transaction_id
            )
        ))
        registrations = {}
        try:
            sequence_ids = []
            transaction_ids = []
            for row in rows:
                if not isinstance(row, ProposalKVRegistration):
                    raise ValueError(
                        "proposal KV registration row must be "
                        "ProposalKVRegistration"
                    )
                sequence_id = _nonnegative_integer(
                    row.sequence_id,
                    "proposal registration sequence_id",
                )
                _nonnegative_integer(
                    row.sequence_epoch,
                    "proposal registration sequence_epoch",
                )
                if not isinstance(row.proposal, DraftProposal):
                    raise ValueError(
                        "proposal registration must contain DraftProposal"
                    )
                if row.proposal.sequence_id != sequence_id:
                    raise ValueError(
                        "proposal registration sequence does not "
                        "match proposal"
                    )
                sequence_ids.append(sequence_id)
                transaction_id = (
                    row.proposal.proposal_transaction_id
                )
                if row.proposal.token_ids:
                    if (
                        not isinstance(transaction_id, str)
                        or not transaction_id
                    ):
                        raise ValueError(
                            "non-empty proposal transaction ID is "
                            "required"
                        )
                    transaction_ids.append(transaction_id)
                elif transaction_id is not None:
                    raise ValueError(
                        "empty proposal must not own a transaction"
                    )
            if len(set(sequence_ids)) != len(sequence_ids):
                raise ValueError(
                    "proposal registration sequence IDs must be "
                    "unique"
                )
            if len(set(transaction_ids)) != len(transaction_ids):
                raise ValueError(
                    "proposal registration transaction IDs must be "
                    "unique"
                )
            for row in rows:
                proposal = row.proposal
                if not proposal.token_ids:
                    continue
                transaction_id = proposal.proposal_transaction_id
                if transaction_id in self._active_transactions:
                    raise ValueError(
                        "proposal transaction is already active"
                    )
                transaction = self.proposal_kv_cache.transaction(
                    transaction_id
                )
                if transaction is None:
                    raise ValueError(
                        "proposal transaction is not active"
                    )
                if transaction.state != "materialized":
                    raise RuntimeError(
                        "proposal transaction must be materialized"
                    )
                if transaction.sequence_id != row.sequence_id:
                    raise ValueError(
                        "proposal transaction sequence does not match"
                    )
                if transaction.sequence_epoch != row.sequence_epoch:
                    raise RuntimeError(
                        "proposal transaction epoch does not match"
                    )
                staged_entry_count = len(
                    transaction.staged_entry_identities
                )
                if len(proposal.token_ids) != staged_entry_count + 1:
                    raise ValueError(
                        "proposal token count does not match staged KV"
                    )
                registrations[transaction_id] = (
                    row.sequence_id,
                    row.sequence_epoch,
                )
        except BaseException:
            self._abort_unregistered_transactions(
                cleanup_transaction_ids
            )
            raise

        for row in rows:
            proposal = row.proposal
            if not proposal.token_ids:
                continue
            transaction_id = proposal.proposal_transaction_id
            self._authority_rows[transaction_id] = {
                "sequence_id": row.sequence_id,
                "sequence_epoch": row.sequence_epoch,
                "transaction_id": transaction_id,
                "token_ids": list(proposal.token_ids),
                "staged_entry_count": len(proposal.token_ids) - 1,
                "staged_entry_identities": [
                    {
                        "logical_entry_id": identity.logical_entry_id,
                        "generation": identity.generation,
                    }
                    for identity in transaction.staged_entry_identities
                ],
                "accepted_proposal_tokens": None,
                "rejected_proposal_tokens": None,
                "finalize_ticket_id": None,
                "state": "active",
            }
        self._active_transactions.update(registrations)
        return tuple(row.proposal for row in rows)

    def _validate_finalize_rows(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> tuple[str, ...]:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "proposal finalize rows must be a non-empty tuple"
            )
        sequence_ids = []
        transaction_ids = []
        for row in rows:
            if not isinstance(row, ProposalFinalizeRow):
                raise ValueError(
                    "proposal finalize row must be ProposalFinalizeRow"
                )
            sequence_id = _nonnegative_integer(
                row.sequence_id,
                "proposal finalize sequence_id",
            )
            _nonnegative_integer(
                row.accepted_proposal_tokens,
                "accepted_proposal_tokens",
            )
            transaction_id = row.proposal_transaction_id
            if (
                not isinstance(transaction_id, str)
                or not transaction_id
            ):
                raise ValueError(
                    "proposal finalize transaction ID must be non-empty"
                )
            owner = self._active_transactions.get(transaction_id)
            if owner is None:
                raise ValueError(
                    "proposal transaction is not active"
                )
            if owner[0] != sequence_id:
                raise ValueError(
                    "proposal transaction sequence does not match"
                )
            sequence_ids.append(sequence_id)
            transaction_ids.append(transaction_id)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "proposal finalize sequence IDs must be unique"
            )
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError(
                "proposal finalize transaction IDs must be unique"
            )
        return tuple(transaction_ids)

    def _clear_failed_prepare(
        self,
        transaction_ids: tuple[str, ...],
        prepared_ticket_ids: tuple[str, ...],
    ) -> None:
        prepared_count = len(prepared_ticket_ids)
        first_cleanup_error = None
        for ticket_id in reversed(prepared_ticket_ids):
            try:
                self.proposal_kv_cache.rollback_finalize(ticket_id)
            except BaseException as error:
                if first_cleanup_error is None:
                    first_cleanup_error = error
        for transaction_id in reversed(
            transaction_ids[prepared_count:]
        ):
            try:
                transaction = self.proposal_kv_cache.transaction(
                    transaction_id
                )
                if (
                    transaction is not None
                    and transaction.state
                    in ("reserved", "materialized")
                ):
                    self.proposal_kv_cache.abort(transaction_id)
            except BaseException as error:
                if first_cleanup_error is None:
                    first_cleanup_error = error
        for index, transaction_id in enumerate(transaction_ids):
            transaction = self.proposal_kv_cache.transaction(
                transaction_id
            )
            state = (
                "rolled_back"
                if index < prepared_count
                else "aborted"
            )
            if transaction is not None:
                state = transaction.state
            self._authority_rows[transaction_id]["state"] = state
            self._active_transactions.pop(transaction_id, None)
        if first_cleanup_error is not None:
            raise first_cleanup_error

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str:
        transaction_ids = self._validate_finalize_rows(rows)
        underlying_ticket_ids = []
        try:
            for row in rows:
                ticket = self.proposal_kv_cache.prepare_finalize(
                    row.proposal_transaction_id,
                    accepted_proposal_tokens=(
                        row.accepted_proposal_tokens
                    ),
                )
                underlying_ticket_ids.append(ticket.ticket_id)
        except BaseException as error:
            try:
                self._clear_failed_prepare(
                    transaction_ids,
                    tuple(underlying_ticket_ids),
                )
            except BaseException as cleanup_error:
                raise cleanup_error from error
            raise
        ticket_id = (
            f"{self.ticket_namespace}-finalize-"
            f"{self._next_batch_ticket_id}"
        )
        self._next_batch_ticket_id += 1
        self._batch_tickets[ticket_id] = (
            _ProposalKVBatchFinalize(
                underlying_ticket_ids=tuple(
                    underlying_ticket_ids
                ),
                transaction_ids=transaction_ids,
            )
        )
        for row in rows:
            authority_row = self._authority_rows[
                row.proposal_transaction_id
            ]
            authority_row["accepted_proposal_tokens"] = (
                row.accepted_proposal_tokens
            )
            authority_row["rejected_proposal_tokens"] = (
                len(authority_row["token_ids"])
                - row.accepted_proposal_tokens
            )
            authority_row["finalize_ticket_id"] = ticket_id
            authority_row["state"] = "prepared"
        return ticket_id

    def _take_batch_ticket(
        self,
        ticket_id: str,
    ) -> _ProposalKVBatchFinalize:
        if not isinstance(ticket_id, str) or not ticket_id:
            raise ValueError(
                "batch finalize ticket must be a non-empty string"
            )
        ticket = self._batch_tickets.pop(ticket_id, None)
        if ticket is None:
            raise ValueError("batch finalize ticket is not active")
        return ticket

    def commit_finalize_batch(self, ticket_id: str) -> None:
        ticket = self._take_batch_ticket(ticket_id)
        for underlying_ticket_id in ticket.underlying_ticket_ids:
            self.proposal_kv_cache.commit_finalize(
                underlying_ticket_id
            )
        for transaction_id in ticket.transaction_ids:
            self._authority_rows[transaction_id]["state"] = (
                "committed"
            )
            del self._active_transactions[transaction_id]

    def rollback_finalize_batch(self, ticket_id: str) -> None:
        ticket = self._take_batch_ticket(ticket_id)
        for underlying_ticket_id in reversed(
            ticket.underlying_ticket_ids
        ):
            self.proposal_kv_cache.rollback_finalize(
                underlying_ticket_id
            )
        for transaction_id in ticket.transaction_ids:
            self._authority_rows[transaction_id]["state"] = (
                "rolled_back"
            )
            del self._active_transactions[transaction_id]

    def assert_sequence_releasable(
        self,
        sequence_id: int,
        sequence_epoch: int,
    ) -> None:
        sequence_id = _nonnegative_integer(
            sequence_id,
            "sequence_id",
        )
        sequence_epoch = _nonnegative_integer(
            sequence_epoch,
            "sequence_epoch",
        )
        if any(
            owner[0] == sequence_id
            for owner in self._active_transactions.values()
        ):
            raise RuntimeError(
                "sequence has an active proposal transaction"
            )
        state = self.proposal_kv_cache.sequence_state(sequence_id)
        if (
            state is not None
            and state.sequence_epoch != sequence_epoch
        ):
            raise RuntimeError("sequence epoch is stale")

    def release_sequence(
        self,
        sequence_id: int,
        sequence_epoch: int,
    ) -> None:
        self.assert_sequence_releasable(
            sequence_id,
            sequence_epoch,
        )
        self.proposal_kv_cache.release_sequence(
            sequence_id,
            sequence_epoch=sequence_epoch,
        )
        self._release_rows.append({
            "sequence_id": sequence_id,
            "sequence_epoch": sequence_epoch,
        })

    def authority_snapshot(self) -> dict:
        return {
            "transactions": [
                dict(row)
                for row in self._authority_rows.values()
            ],
            "release_rows": [
                dict(row)
                for row in self._release_rows
            ],
            "active_transaction_count": (
                self.active_transaction_count
            ),
            "prepared_ticket_count": self.prepared_ticket_count,
            "proposal_kv_cache": (
                self.proposal_kv_cache.authority_snapshot()
            ),
        }
