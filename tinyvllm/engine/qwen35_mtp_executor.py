from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.proposal_kv_lifecycle import (
    ProposalKVLifecycleCoordinator,
    ProposalKVRegistration,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalInput,
    ProposalFinalizeRow,
    TargetPrefillObservation,
    proposal_input_context_token_count,
)
from tinyvllm.engine.tensor_parallel_greedy import (
    select_tensor_parallel_greedy_tokens,
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
        tensor_parallel_rank: int = 0,
        tensor_parallel_size: int = 1,
        token_broadcast=None,
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
        tensor_parallel_size = _positive_integer(
            tensor_parallel_size,
            "tensor_parallel_size",
        )
        if (
            isinstance(tensor_parallel_rank, bool)
            or not isinstance(tensor_parallel_rank, int)
            or tensor_parallel_rank < 0
            or tensor_parallel_rank >= tensor_parallel_size
        ):
            raise ValueError(
                "tensor_parallel_rank must be in "
                "[0, tensor_parallel_size)"
            )
        if token_broadcast is not None and not callable(token_broadcast):
            raise ValueError("token_broadcast must be callable")
        self.module = module
        self.proposal_kv_cache = proposal_kv_cache
        if (
            graph_runner is not None
            and not callable(getattr(graph_runner, "run", None))
        ):
            raise ValueError(
                "graph_runner must expose callable run"
            )
        if tensor_parallel_size > 1 and graph_runner is not None:
            raise RuntimeError("Qwen3.5 MTP CUDA graphs require TP1")
        self.graph_runner = graph_runner
        self.tensor_parallel_rank = tensor_parallel_rank
        self.tensor_parallel_size = tensor_parallel_size
        self.token_broadcast = token_broadcast
        self._capabilities = DraftCapabilities(
            source_type="native_model_runner",
            supports_batch=True,
            requires_target_hidden=True,
            requires_target_logits=False,
            max_proposal_tokens=max_proposal_tokens,
            execution_domain="model_runner",
            requires_proposal_lifecycle=True,
            requires_full_token_history=False,
        )
        self._pending_prefixes: dict[
            int,
            Qwen35MTPPendingPrefix,
        ] = {}
        self._bootstrapped: dict[int, _BootstrappedSequence] = {}
        self.proposal_kv_lifecycle = (
            ProposalKVLifecycleCoordinator(
                proposal_kv_cache,
                ticket_namespace="qwen35-mtp",
            )
        )
        self._proposal_exact_q: dict[str, int] = {}
        self._selected_token_rows: list[dict] = []

    @property
    def capabilities(self) -> DraftCapabilities:
        return self._capabilities

    def pending_prefix(
        self,
        sequence_id: int,
    ) -> Qwen35MTPPendingPrefix | None:
        _nonnegative_integer(sequence_id, "sequence_id")
        return self._pending_prefixes.get(sequence_id)

    def _forward_bootstrap(
        self,
        module,
        transaction,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
    ):
        token_count = int(input_ids.shape[0])
        allocator = self.proposal_kv_cache.entry_allocator
        writable_lease = allocator.ensure_writable(
            transaction.staged_entry_identities
        )
        slot_mapping = torch.tensor(
            writable_lease.physical_slot_ids,
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
            try:
                output = module.forward_hidden(
                    input_ids,
                    positions,
                    target_hidden,
                )
            except BaseException:
                allocator.record_write_complete(writable_lease)
                raise
        allocator.record_write_complete(writable_lease)
        return output

    @torch.inference_mode()
    def _forward_proposal_step(
        self,
        transaction,
        *,
        step: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        current_hidden: torch.Tensor,
    ):
        allocator = self.proposal_kv_cache.entry_allocator
        committed_entries = (
            self.proposal_kv_cache.committed_entry_identities(
            transaction.sequence_id
            )
        )
        read_prefix = (
            committed_entries
            + transaction.staged_entry_identities[:step]
        )
        read_lease = allocator.ensure_readable(read_prefix)
        write_lease = allocator.ensure_writable((
            transaction.staged_entry_identities[step],
        ))
        visible_physical_slots = (
            read_lease.physical_slot_ids
            + write_lease.physical_slot_ids
        )
        slot_mapping = torch.tensor(
            write_lease.physical_slot_ids,
            dtype=torch.int32,
            device=current_hidden.device,
        )
        block_tables = torch.tensor(
            [visible_physical_slots],
            dtype=torch.int32,
            device=current_hidden.device,
        )
        context_lens = torch.tensor(
            [len(visible_physical_slots)],
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
            max_seqlen_k=len(visible_physical_slots),
            quest_top_k_blocks=-1,
            am_compact_blocks=0,
            kv_offload_manager=None,
            kv_offload_blockwise_decode=False,
            kv_offload_blockwise_prefill=False,
        ):
            try:
                output = self.module.forward_step(
                    input_ids,
                    positions,
                    current_hidden,
                )
            except BaseException:
                allocator.record_read_complete(read_lease)
                allocator.record_write_complete(write_lease)
                raise
        allocator.record_read_complete(read_lease)
        allocator.record_write_complete(write_lease)
        return output

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
        context_token_count = proposal_input_context_token_count(
            input_row
        )
        if context_token_count <= 0:
            raise ValueError(
                "context_token_count must be positive"
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
                            proposal_input_context_token_count(
                                input_row
                            ) - 1,
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
                token_ids = select_tensor_parallel_greedy_tokens(
                    logits,
                    rank=self.tensor_parallel_rank,
                    world_size=self.tensor_parallel_size,
                    batch_size=1,
                    device=current_hidden.device,
                    broadcast=self.token_broadcast,
                )
                current_token = int(token_ids[0].item())
                self._selected_token_rows.append({
                    "sequence_id": input_row.sequence_id,
                    "transaction_id": transaction.transaction_id,
                    "step": step,
                    "token_id": current_token,
                })
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
        exact_q_by_transaction = {}
        registrations = []
        for proposal, (input_row, bootstrap) in zip(
            proposals,
            rows,
        ):
            if not isinstance(proposal, DraftProposal):
                raise ValueError(
                    "proposal group must contain DraftProposal"
                )
            transaction_id = proposal.proposal_transaction_id
            exact_q_by_transaction[transaction_id] = int(
                proposal.metadata["exact_q"]
            )
            registrations.append(
                ProposalKVRegistration(
                    sequence_id=input_row.sequence_id,
                    sequence_epoch=bootstrap.sequence_epoch,
                    proposal=proposal,
                )
            )
        registered = self.proposal_kv_lifecycle.register_batch(
            tuple(registrations)
        )
        self._proposal_exact_q.update(exact_q_by_transaction)
        return registered

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
        return self.proposal_kv_lifecycle.prepare_finalize_batch(
            rows
        )

    def commit_finalize_batch(self, ticket_id: str) -> None:
        self.proposal_kv_lifecycle.commit_finalize_batch(ticket_id)

    def rollback_finalize_batch(self, ticket_id: str) -> None:
        self.proposal_kv_lifecycle.rollback_finalize_batch(ticket_id)

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
        self.proposal_kv_lifecycle.assert_sequence_releasable(
            sequence_id,
            sequence_epoch,
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
        self.proposal_kv_lifecycle.release_sequence(
            sequence_id,
            sequence_epoch,
        )
        self._bootstrapped.pop(sequence_id, None)
        self._pending_prefixes.pop(sequence_id, None)

    def tp4_authority_snapshot(self) -> dict:
        lifecycle_snapshot = (
            self.proposal_kv_lifecycle.authority_snapshot()
        )
        cache_snapshot = lifecycle_snapshot["proposal_kv_cache"]
        allocator_snapshot = cache_snapshot["entry_allocator"]
        return {
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "tensor_parallel_size": self.tensor_parallel_size,
            "proposal_transactions": [
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "staged_entry_identities"
                    },
                    "exact_q": self._proposal_exact_q[
                        row["transaction_id"]
                    ],
                }
                for row in lifecycle_snapshot["transactions"]
            ],
            "selected_tokens": [
                dict(row)
                for row in self._selected_token_rows
            ],
            "release_rows": [
                dict(row)
                for row in lifecycle_snapshot["release_rows"]
            ],
            "active_transactions": lifecycle_snapshot[
                "active_transaction_count"
            ],
            "prepared_tickets": lifecycle_snapshot[
                "prepared_ticket_count"
            ],
            "pending_sequences": len(self._pending_prefixes),
            "bootstrapped_sequences": len(self._bootstrapped),
            "allocated_physical_slots": allocator_snapshot.get(
                "gpu_resident_entry_count",
                cache_snapshot["owned_entry_count"],
            ),
            "proposal_kv_cache": cache_snapshot,
        }
