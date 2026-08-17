from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalInput,
    ProposalFinalizeRow,
    TargetPrefillObservation,
)
from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
)
from tinyvllm.utils.context import temporary_context


@dataclass
class Qwen35MTPPendingPrefix:
    sequence_id: int
    sequence_epoch: int
    token_ids: tuple[int, ...]
    positions: torch.Tensor
    target_hidden: torch.Tensor
    is_final: bool


@dataclass(frozen=True)
class _BootstrappedSequence:
    sequence_id: int
    sequence_epoch: int
    prefix_token_count: int


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


class Qwen35MTPProposalExecutor:

    def __init__(
        self,
        *,
        module,
        proposal_kv_cache: ProposalKVCache,
        max_proposal_tokens: int,
        graph_runner=None,
    ):
        if not callable(getattr(module, "forward_step", None)):
            raise ValueError("module must expose callable forward_step")
        if not isinstance(proposal_kv_cache, ProposalKVCache):
            raise ValueError(
                "proposal_kv_cache must be a ProposalKVCache"
            )
        max_proposal_tokens = _positive_integer(
            max_proposal_tokens,
            "max_proposal_tokens",
        )
        self.module = module
        self.proposal_kv_cache = proposal_kv_cache
        if (
            graph_runner is not None
            and not callable(getattr(graph_runner, "run", None))
        ):
            raise ValueError(
                "graph_runner must expose callable run"
            )
        self.graph_runner = graph_runner
        self._capabilities = DraftCapabilities(
            source_type="native_model_runner",
            supports_batch=True,
            requires_target_hidden=True,
            requires_target_logits=False,
            max_proposal_tokens=max_proposal_tokens,
            execution_domain="model_runner",
            requires_proposal_lifecycle=True,
        )
        self._pending_prefixes: dict[
            int,
            Qwen35MTPPendingPrefix,
        ] = {}
        self._bootstrapped: dict[int, _BootstrappedSequence] = {}
        self._proposal_transactions: dict[str, tuple[int, int]] = {}
        self._batch_tickets: dict[str, tuple[str, ...]] = {}
        self._batch_ticket_transactions: dict[
            str,
            tuple[str, ...],
        ] = {}
        self._next_batch_ticket_id = 1

    @property
    def capabilities(self) -> DraftCapabilities:
        return self._capabilities

    def pending_prefix(
        self,
        sequence_id: int,
    ) -> Qwen35MTPPendingPrefix | None:
        _nonnegative_integer(sequence_id, "sequence_id")
        return self._pending_prefixes.get(sequence_id)

    @staticmethod
    def _forward_bootstrap(
        module,
        transaction,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
    ):
        token_count = int(input_ids.shape[0])
        slot_mapping = torch.tensor(
            transaction.staged_slot_ids,
            dtype=torch.int32,
            device=target_hidden.device,
        )
        offsets = torch.tensor(
            [0, token_count],
            dtype=torch.int32,
            device=target_hidden.device,
        )
        with temporary_context(
            mode="prefill",
            is_prefill=True,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            max_seqlen_q=token_count,
            max_seqlen_k=token_count,
            quest_top_k_blocks=-1,
            am_compact_blocks=0,
            kv_offload_manager=None,
            kv_offload_blockwise_decode=False,
            kv_offload_blockwise_prefill=False,
        ):
            return module.forward_step(
                input_ids,
                positions,
                target_hidden,
            )

    def _forward_proposal_step(
        self,
        transaction,
        *,
        step: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        current_hidden: torch.Tensor,
    ):
        committed_slots = self.proposal_kv_cache.committed_slot_ids(
            transaction.sequence_id
        )
        visible_slots = (
            committed_slots
            + transaction.staged_slot_ids[:step + 1]
        )
        slot_mapping = torch.tensor(
            [transaction.staged_slot_ids[step]],
            dtype=torch.int32,
            device=current_hidden.device,
        )
        block_tables = torch.tensor(
            [visible_slots],
            dtype=torch.int32,
            device=current_hidden.device,
        )
        context_lens = torch.tensor(
            [len(visible_slots)],
            dtype=torch.int32,
            device=current_hidden.device,
        )
        with temporary_context(
            mode="decode",
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=len(visible_slots),
            quest_top_k_blocks=-1,
            am_compact_blocks=0,
            kv_offload_manager=None,
            kv_offload_blockwise_decode=False,
            kv_offload_blockwise_prefill=False,
        ):
            return self.module.forward_step(
                input_ids,
                positions,
                current_hidden,
            )

    @staticmethod
    def _validate_observation(
        row: TargetPrefillObservation,
    ) -> None:
        if not isinstance(row, TargetPrefillObservation):
            raise ValueError(
                "row must be a TargetPrefillObservation"
            )
        _nonnegative_integer(row.sequence_id, "sequence_id")
        _nonnegative_integer(row.sequence_epoch, "sequence_epoch")
        if not isinstance(row.token_ids, tuple) or not row.token_ids:
            raise ValueError(
                "target prefill token_ids must be a non-empty tuple"
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token_id in row.token_ids
        ):
            raise ValueError(
                "target prefill token IDs must be integers"
            )
        if not isinstance(row.positions, torch.Tensor):
            raise ValueError("target prefill positions must be a tensor")
        if row.positions.ndim != 1:
            raise ValueError(
                "target prefill positions must be rank one"
            )
        if row.positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "target prefill positions must use an integer dtype"
            )
        if not isinstance(row.target_hidden, torch.Tensor):
            raise ValueError(
                "target prefill hidden must be a tensor"
            )
        if row.target_hidden.ndim != 2:
            raise ValueError(
                "target prefill hidden must be rank two"
            )
        if not row.target_hidden.is_floating_point():
            raise ValueError(
                "target prefill hidden must use a floating point dtype"
            )
        token_count = len(row.token_ids)
        if (
            row.positions.shape[0] != token_count
            or row.target_hidden.shape[0] != token_count
        ):
            raise ValueError(
                "target prefill token count must match positions "
                "and hidden"
            )
        if row.positions.device != row.target_hidden.device:
            raise ValueError(
                "target prefill positions and hidden device must match"
            )

    def observe_target_prefill(
        self,
        rows: tuple[TargetPrefillObservation, ...],
    ) -> None:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "target prefill rows must be a non-empty tuple"
            )
        for row in rows:
            self._validate_observation(row)
            if row.sequence_id in self._bootstrapped:
                raise RuntimeError(
                    "sequence already completed MTP bootstrap"
                )
            pending = self._pending_prefixes.get(row.sequence_id)
            if pending is None:
                expected_start = 0
                expected_positions = torch.arange(
                    expected_start,
                    expected_start + len(row.token_ids),
                    dtype=row.positions.dtype,
                    device=row.positions.device,
                )
                if not torch.equal(row.positions, expected_positions):
                    raise ValueError(
                        "target prefill positions must be contiguous"
                    )
                self._pending_prefixes[row.sequence_id] = (
                    Qwen35MTPPendingPrefix(
                        sequence_id=row.sequence_id,
                        sequence_epoch=row.sequence_epoch,
                        token_ids=row.token_ids,
                        positions=row.positions,
                        target_hidden=row.target_hidden,
                        is_final=row.is_final_chunk,
                    )
                )
                continue
            if pending.is_final:
                raise RuntimeError(
                    "target prefill final chunk was already observed"
                )
            if pending.sequence_epoch != row.sequence_epoch:
                raise RuntimeError("target prefill sequence epoch changed")
            if (
                pending.target_hidden.shape[1]
                != row.target_hidden.shape[1]
            ):
                raise ValueError("target prefill hidden width changed")
            if pending.target_hidden.dtype != row.target_hidden.dtype:
                raise ValueError("target prefill hidden dtype changed")
            if pending.target_hidden.device != row.target_hidden.device:
                raise ValueError("target prefill hidden device changed")
            expected_start = len(pending.token_ids)
            expected_positions = torch.arange(
                expected_start,
                expected_start + len(row.token_ids),
                dtype=row.positions.dtype,
                device=row.positions.device,
            )
            if not torch.equal(row.positions, expected_positions):
                raise ValueError(
                    "target prefill positions must be contiguous"
                )
            pending.token_ids = pending.token_ids + row.token_ids
            pending.positions = torch.cat((
                pending.positions,
                row.positions,
            ))
            pending.target_hidden = torch.cat((
                pending.target_hidden,
                row.target_hidden,
            ))
            pending.is_final = row.is_final_chunk

    def _bootstrap_sequence(
        self,
        input_row: ModelRunnerProposalInput,
    ) -> _BootstrappedSequence:
        existing = self._bootstrapped.get(input_row.sequence_id)
        if existing is not None:
            return existing
        pending = self._pending_prefixes.get(input_row.sequence_id)
        if pending is None:
            raise RuntimeError(
                "target prefill observation is required before bootstrap"
            )
        if not pending.is_final:
            raise RuntimeError(
                "target prefill final chunk is required before bootstrap"
            )
        shifted_token_ids = pending.token_ids[1:] + (
            input_row.first_target_token,
        )
        input_ids = torch.tensor(
            shifted_token_ids,
            dtype=torch.int64,
            device=pending.target_hidden.device,
        )
        transaction = self.proposal_kv_cache.begin(
            pending.sequence_id,
            pending.sequence_epoch,
            len(shifted_token_ids),
        )
        try:
            self._forward_bootstrap(
                self.module,
                transaction,
                input_ids,
                pending.positions,
                pending.target_hidden,
            )
            self.proposal_kv_cache.mark_materialized(
                transaction,
                len(shifted_token_ids),
            )
            ticket = self.proposal_kv_cache.prepare_finalize(
                transaction.transaction_id,
                accepted_proposal_tokens=(
                    len(shifted_token_ids) + 1
                ),
            )
            self.proposal_kv_cache.commit_finalize(ticket.ticket_id)
        except BaseException:
            if transaction.state in ("reserved", "materialized"):
                self.proposal_kv_cache.abort(
                    transaction.transaction_id
                )
            raise
        state = _BootstrappedSequence(
            sequence_id=pending.sequence_id,
            sequence_epoch=pending.sequence_epoch,
            prefix_token_count=len(pending.token_ids),
        )
        self._bootstrapped[input_row.sequence_id] = state
        del self._pending_prefixes[input_row.sequence_id]
        return state

    @staticmethod
    def _validate_proposal_input(
        input_row: ModelRunnerProposalInput,
    ) -> None:
        if not isinstance(input_row, ModelRunnerProposalInput):
            raise ValueError(
                "proposal input must be ModelRunnerProposalInput"
            )
        _nonnegative_integer(input_row.sequence_id, "sequence_id")
        _nonnegative_integer(
            input_row.remaining_output_tokens,
            "remaining_output_tokens",
        )
        _nonnegative_integer(
            input_row.max_proposal_tokens,
            "max_proposal_tokens",
        )
        if not isinstance(input_row.target_hidden, torch.Tensor):
            raise ValueError("target_hidden must be a tensor")
        if (
            input_row.target_hidden.ndim != 2
            or input_row.target_hidden.shape[0] != 1
        ):
            raise ValueError(
                "target_hidden must have exact shape [1, hidden_size]"
            )
        if not input_row.target_hidden.is_floating_point():
            raise ValueError(
                "target_hidden must use a floating point dtype"
            )

    def _run_proposal(
        self,
        input_row: ModelRunnerProposalInput,
        exact_q: int,
        bootstrap: _BootstrappedSequence,
    ) -> DraftProposal:
        staged_entry_count = exact_q - 1
        transaction = self.proposal_kv_cache.begin(
            input_row.sequence_id,
            bootstrap.sequence_epoch,
            staged_entry_count,
        )
        proposal_tokens = [input_row.first_target_token]
        current_token = input_row.first_target_token
        current_hidden = input_row.target_hidden
        try:
            for step in range(staged_entry_count):
                input_ids = torch.tensor(
                    [current_token],
                    dtype=torch.int64,
                    device=current_hidden.device,
                )
                positions = torch.tensor(
                    [
                        max(
                            len(input_row.token_ids) - 1,
                            0,
                        ) + step
                    ],
                    dtype=torch.int64,
                    device=current_hidden.device,
                )
                next_hidden, logits = self._forward_proposal_step(
                    transaction,
                    step=step,
                    input_ids=input_ids,
                    positions=positions,
                    current_hidden=current_hidden,
                )
                if (
                    not isinstance(next_hidden, torch.Tensor)
                    or next_hidden.shape != current_hidden.shape
                    or next_hidden.dtype != current_hidden.dtype
                    or next_hidden.device != current_hidden.device
                ):
                    raise ValueError(
                        "MTP hidden output must exactly match input hidden"
                    )
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 2
                    or logits.shape[0] != 1
                    or not logits.is_floating_point()
                    or logits.device != current_hidden.device
                ):
                    raise ValueError(
                        "MTP logits must be floating rank-two one-row tensor"
                    )
                current_token = int(
                    torch.argmax(logits[0], dim=-1).item()
                )
                proposal_tokens.append(current_token)
                current_hidden = next_hidden
            self.proposal_kv_cache.mark_materialized(
                transaction,
                staged_entry_count,
            )
        except BaseException:
            if transaction.state in ("reserved", "materialized"):
                self.proposal_kv_cache.abort(
                    transaction.transaction_id
                )
            raise
        return DraftProposal(
            sequence_id=input_row.sequence_id,
            token_ids=tuple(proposal_tokens),
            source_type=self.capabilities.source_type,
            metadata={
                "exact_q": exact_q,
                "staged_entry_count": staged_entry_count,
            },
            proposal_transaction_id=transaction.transaction_id,
        )

    def _abort_unregistered_group_proposals(
        self,
        proposals,
    ) -> None:
        transaction_ids = []
        for proposal in proposals:
            transaction_id = getattr(
                proposal,
                "proposal_transaction_id",
                None,
            )
            if (
                isinstance(transaction_id, str)
                and transaction_id
                and transaction_id
                not in self._proposal_transactions
                and transaction_id not in transaction_ids
            ):
                transaction_ids.append(transaction_id)
        first_error = None
        for transaction_id in reversed(transaction_ids):
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

    def _register_group_proposals(
        self,
        proposals,
        rows,
    ) -> tuple[DraftProposal, ...]:
        if (
            not isinstance(proposals, tuple)
            or not isinstance(rows, tuple)
            or len(proposals) != len(rows)
        ):
            raise ValueError(
                "proposal group must exactly match execution rows"
            )
        try:
            transaction_ids = tuple(
                proposal.proposal_transaction_id
                for proposal in proposals
            )
            if any(
                not isinstance(transaction_id, str)
                or not transaction_id
                for transaction_id in transaction_ids
            ):
                raise ValueError(
                    "proposal transaction is not active"
                )
            if len(set(transaction_ids)) != len(transaction_ids):
                raise ValueError(
                    "proposal transaction IDs must be unique"
                )
            registrations = {}
            for proposal, (
                input_row,
                bootstrap,
            ), transaction_id in zip(
                proposals,
                rows,
                transaction_ids,
            ):
                if not isinstance(proposal, DraftProposal):
                    raise ValueError(
                        "proposal group must contain DraftProposal"
                    )
                if proposal.sequence_id != input_row.sequence_id:
                    raise ValueError(
                        "proposal sequence does not match input row"
                    )
                if transaction_id in self._proposal_transactions:
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
                if (
                    transaction.sequence_id
                    != input_row.sequence_id
                    or transaction.sequence_epoch
                    != bootstrap.sequence_epoch
                ):
                    raise ValueError(
                        "proposal transaction sequence or epoch "
                        "does not match"
                    )
                registrations[transaction_id] = (
                    input_row.sequence_id,
                    bootstrap.sequence_epoch,
                )
        except BaseException:
            self._abort_unregistered_group_proposals(proposals)
            raise
        self._proposal_transactions.update(registrations)
        return proposals

    def _run_exact_q_eager(
        self,
        exact_q: int,
        rows: tuple[
            tuple[
                ModelRunnerProposalInput,
                _BootstrappedSequence,
            ],
            ...,
        ],
    ) -> tuple[DraftProposal, ...]:
        return tuple(
            self._run_proposal(input_row, exact_q, bootstrap)
            for input_row, bootstrap in rows
        )

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]:
        if not isinstance(inputs, tuple) or not inputs:
            raise ValueError(
                "proposal inputs must be a non-empty tuple"
            )
        proposals: list[DraftProposal | None] = [
            None for _ in inputs
        ]
        sequence_ids = set()
        grouped_rows: dict[
            int,
            list[
                tuple[
                    int,
                    ModelRunnerProposalInput,
                    _BootstrappedSequence,
                ]
            ],
        ] = {}
        for input_index, input_row in enumerate(inputs):
            self._validate_proposal_input(input_row)
            if input_row.sequence_id in sequence_ids:
                raise ValueError(
                    "proposal input sequence IDs must be unique"
                )
            sequence_ids.add(input_row.sequence_id)
            exact_q = min(
                input_row.remaining_output_tokens,
                input_row.max_proposal_tokens,
                self.capabilities.max_proposal_tokens,
            )
            if exact_q == 0:
                proposals[input_index] = DraftProposal(
                    sequence_id=input_row.sequence_id,
                    token_ids=(),
                    source_type=self.capabilities.source_type,
                )
                continue
            bootstrap = self._bootstrap_sequence(input_row)
            grouped_rows.setdefault(exact_q, []).append(
                (input_index, input_row, bootstrap)
            )
        for exact_q, indexed_rows in grouped_rows.items():
            eager_rows = tuple(
                (input_row, bootstrap)
                for _, input_row, bootstrap in indexed_rows
            )
            if exact_q == 1 or self.graph_runner is None:
                group_proposals = self._run_exact_q_eager(
                    exact_q,
                    eager_rows,
                )
            else:
                group_proposals = self.graph_runner.run(
                    exact_q=exact_q,
                    rows=eager_rows,
                    eager=self._run_exact_q_eager,
                )
            if len(group_proposals) != len(indexed_rows):
                raise ValueError(
                    "exact-Q execution result count must match rows"
                )
            group_proposals = self._register_group_proposals(
                group_proposals,
                eager_rows,
            )
            for (
                input_index,
                _,
                _,
            ), proposal in zip(indexed_rows, group_proposals):
                proposals[input_index] = proposal
        if any(proposal is None for proposal in proposals):
            raise RuntimeError("proposal execution left an empty result")
        return tuple(proposals)

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "proposal finalize rows must be a non-empty tuple"
            )
        transaction_ids = []
        for row in rows:
            if not isinstance(row, ProposalFinalizeRow):
                raise ValueError(
                    "finalize row must be ProposalFinalizeRow"
                )
            owner = self._proposal_transactions.get(
                row.proposal_transaction_id
            )
            if owner is None:
                raise ValueError(
                    "proposal transaction is not active"
                )
            if owner[0] != row.sequence_id:
                raise ValueError(
                    "proposal transaction sequence does not match"
                )
            transaction_ids.append(row.proposal_transaction_id)
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError(
                "proposal transaction IDs must be unique"
            )

        underlying_tickets = []
        try:
            for row in rows:
                ticket = self.proposal_kv_cache.prepare_finalize(
                    row.proposal_transaction_id,
                    accepted_proposal_tokens=(
                        row.accepted_proposal_tokens
                    ),
                )
                underlying_tickets.append(ticket.ticket_id)
        except BaseException:
            for ticket_id in reversed(underlying_tickets):
                self.proposal_kv_cache.rollback_finalize(ticket_id)
            raise
        batch_ticket_id = (
            f"qwen35-mtp-finalize-{self._next_batch_ticket_id}"
        )
        self._next_batch_ticket_id += 1
        self._batch_tickets[batch_ticket_id] = tuple(
            underlying_tickets
        )
        self._batch_ticket_transactions[batch_ticket_id] = tuple(
            transaction_ids
        )
        return batch_ticket_id

    def _take_batch_ticket(
        self,
        batch_ticket_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if (
            not isinstance(batch_ticket_id, str)
            or not batch_ticket_id
        ):
            raise ValueError(
                "batch finalize ticket must be a non-empty string"
            )
        tickets = self._batch_tickets.pop(batch_ticket_id, None)
        transactions = self._batch_ticket_transactions.pop(
            batch_ticket_id,
            None,
        )
        if tickets is None or transactions is None:
            raise ValueError("batch finalize ticket is not active")
        return tickets, transactions

    def commit_finalize_batch(self, ticket_id: str) -> None:
        tickets, transactions = self._take_batch_ticket(ticket_id)
        for underlying_ticket in tickets:
            self.proposal_kv_cache.commit_finalize(
                underlying_ticket
            )
        for transaction_id in transactions:
            del self._proposal_transactions[transaction_id]

    def rollback_finalize_batch(self, ticket_id: str) -> None:
        tickets, transactions = self._take_batch_ticket(ticket_id)
        for underlying_ticket in reversed(tickets):
            self.proposal_kv_cache.rollback_finalize(
                underlying_ticket
            )
        for transaction_id in transactions:
            del self._proposal_transactions[transaction_id]

    def release_sequence(
        self,
        sequence_id: int,
        *,
        sequence_epoch: int,
    ) -> None:
        _nonnegative_integer(sequence_id, "sequence_id")
        sequence_epoch = _nonnegative_integer(
            sequence_epoch,
            "sequence_epoch",
        )
        if any(
            owner[0] == sequence_id
            for owner in self._proposal_transactions.values()
        ):
            raise RuntimeError(
                "sequence has an active proposal transaction"
            )
        state = self._bootstrapped.get(sequence_id)
        if (
            state is not None
            and state.sequence_epoch != sequence_epoch
        ):
            raise RuntimeError("sequence epoch is stale")
        pending = self._pending_prefixes.get(sequence_id)
        if (
            pending is not None
            and pending.sequence_epoch != sequence_epoch
        ):
            raise RuntimeError("sequence epoch is stale")
        self.proposal_kv_cache.release_sequence(
            sequence_id,
            sequence_epoch=sequence_epoch,
        )
        self._bootstrapped.pop(sequence_id, None)
        self._pending_prefixes.pop(sequence_id, None)
