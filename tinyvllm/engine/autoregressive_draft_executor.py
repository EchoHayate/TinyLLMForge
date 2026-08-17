from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import time
import torch

from tinyvllm.engine.proposal_kv_cache import (
    ProposalKVCache,
    ProposalKVTransaction,
)
from tinyvllm.engine.proposal_kv_lifecycle import (
    ProposalKVLifecycleCoordinator,
    ProposalKVRegistration,
)
from tinyvllm.engine.autoregressive_draft_tp import (
    AutoregressiveDraftTensorParallelCoordinator,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalInput,
    ProposalFinalizeRow,
    TargetPrefillObservation,
    assert_tensor_free,
    proposal_input_context_token_count,
)
from tinyvllm.engine.tensor_parallel_greedy import (
    select_tensor_parallel_greedy_tokens,
)
from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
)


@dataclass(frozen=True)
class AutoregressiveDraftPrefillRow:
    transaction: ProposalKVTransaction
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    physical_slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class AutoregressiveDraftDecodeRow:
    transaction: ProposalKVTransaction
    step: int
    input_token_id: int
    position: int
    writable_physical_slot_id: int
    visible_physical_slot_ids: tuple[int, ...]
    visible_logical_entry_ids: tuple[int, ...] = ()
    blockwise_offload: bool = False


@dataclass(frozen=True)
class AutoregressiveDraftGroupExecution:
    transactions: tuple[ProposalKVTransaction, ...]
    token_rows: tuple[tuple[int, ...], ...]
    execution_mode: str


@dataclass(frozen=True)
class AutoregressiveDraftPendingPrompt:
    sequence_id: int
    sequence_epoch: int
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    is_final: bool


@dataclass(frozen=True)
class _BootstrappedSequence:
    sequence_id: int
    sequence_epoch: int
    prompt_token_count: int


class AutoregressiveDraftBackend(Protocol):
    device: object
    backend_identity: str
    model_fingerprint: str
    tokenizer_fingerprint: str

    def prefill_batch(
        self,
        rows: tuple[AutoregressiveDraftPrefillRow, ...],
    ) -> None:
        pass

    def decode_step_batch(
        self,
        rows: tuple[AutoregressiveDraftDecodeRow, ...],
    ) -> tuple[object, ...] | None:
        pass


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


def _identity(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


class AutoregressiveDraftProposalExecutor:

    _LOGICAL_AUTHORITY_EVIDENCE_LIMIT = 128

    def __init__(
        self,
        *,
        backend: AutoregressiveDraftBackend,
        proposal_kv_cache: ProposalKVCache,
        max_proposal_tokens: int,
        tensor_parallel_rank: int = 0,
        tensor_parallel_size: int = 1,
        tensor_parallel_coordinator: (
            AutoregressiveDraftTensorParallelCoordinator | None
        ) = None,
        graph_runner=None,
        clock=None,
    ):
        if not callable(getattr(backend, "prefill_batch", None)):
            raise ValueError(
                "backend must expose callable prefill_batch"
            )
        if not callable(
            getattr(backend, "decode_step_batch", None)
        ):
            raise ValueError(
                "backend must expose callable decode_step_batch"
            )
        _identity(
            getattr(backend, "backend_identity", None),
            "backend identity",
        )
        _identity(
            getattr(backend, "model_fingerprint", None),
            "model fingerprint",
        )
        _identity(
            getattr(backend, "tokenizer_fingerprint", None),
            "tokenizer fingerprint",
        )
        if not isinstance(proposal_kv_cache, ProposalKVCache):
            raise ValueError(
                "proposal_kv_cache must be a ProposalKVCache"
            )
        max_proposal_tokens = _positive_integer(
            max_proposal_tokens,
            "max_proposal_tokens",
        )
        if max_proposal_tokens > 4:
            raise ValueError(
                "max_proposal_tokens must not exceed four"
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
        if tensor_parallel_size not in (1, 4):
            raise RuntimeError(
                "autoregressive draft executor supports TP1 or TP4"
            )
        self.backend = backend
        self.proposal_kv_cache = proposal_kv_cache
        self.max_proposal_tokens = max_proposal_tokens
        self.tensor_parallel_rank = tensor_parallel_rank
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_coordinator = (
            AutoregressiveDraftTensorParallelCoordinator(
                rank=tensor_parallel_rank,
                world_size=tensor_parallel_size,
                device=backend.device,
            )
            if tensor_parallel_coordinator is None
            else tensor_parallel_coordinator
        )
        if graph_runner is not None:
            for method_name in (
                "run",
                "summary",
                "bind_convergence",
            ):
                if not callable(
                    getattr(
                        graph_runner,
                        method_name,
                        None,
                    )
                ):
                    raise ValueError(
                        "graph runner must expose callable "
                        f"{method_name}"
                    )
        self.graph_runner = graph_runner
        self._clock = time.perf_counter if clock is None else clock
        if not callable(self._clock):
            raise ValueError("clock must be callable")
        for method_name in (
            "assert_logical_authority",
            "converge_stage",
        ):
            if not callable(
                getattr(
                    self.tensor_parallel_coordinator,
                    method_name,
                    None,
                )
            ):
                raise ValueError(
                    "tensor parallel coordinator must expose "
                    f"callable {method_name}"
                )
        if self.graph_runner is not None:
            self.graph_runner.bind_convergence(
                self._converge_stage
            )
        self._capabilities = DraftCapabilities(
            source_type="independent_draft_model",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=max_proposal_tokens,
            execution_domain="model_runner",
            requires_proposal_lifecycle=True,
            requires_full_token_history=False,
        )
        self.proposal_kv_lifecycle = (
            ProposalKVLifecycleCoordinator(
                proposal_kv_cache,
                ticket_namespace="autoregressive-draft",
            )
        )
        self._pending_prompts: dict[
            int,
            AutoregressiveDraftPendingPrompt,
        ] = {}
        self._bootstrapped = {}
        self._proposal_exact_q_by_transaction = {}
        self._proposal_token_ids_by_transaction = {}
        self._prepared_finalize_rows_by_ticket = {}
        self._logical_authority_rows = []
        self._logical_authority_digests = []
        self._selected_token_rows = []
        self._bootstrap_rows = []
        self._timing_ms = {
            "prompt_bootstrap": 0.0,
            "proposal_forward": 0.0,
            "proposal_finalize": 0.0,
        }
        self._proposal_forward_detail_ms = {
            "setup": 0.0,
            "backend_submit": 0.0,
            "selection_collective": 0.0,
            "decode_authority": 0.0,
            "token_readback": 0.0,
            "materialize_register": 0.0,
        }

    def _record_timing(
        self,
        name: str,
        started_at: float,
    ) -> None:
        self._timing_ms[name] += (
            self._clock() - started_at
        ) * 1000.0

    def _record_proposal_forward_detail(
        self,
        name: str,
        started_at: float,
    ) -> None:
        self._proposal_forward_detail_ms[name] += (
            self._clock() - started_at
        ) * 1000.0

    @property
    def capabilities(self) -> DraftCapabilities:
        return self._capabilities

    def _record_logical_authority(
        self,
        *,
        stage: str,
        rows: object,
        digest: str,
    ) -> str:
        self._logical_authority_rows.append({
            "stage": stage,
            "rows": rows,
        })
        self._logical_authority_digests.append(digest)
        if (
            len(self._logical_authority_rows)
            > self._LOGICAL_AUTHORITY_EVIDENCE_LIMIT
        ):
            self._logical_authority_rows = (
                self._logical_authority_rows[
                    -self._LOGICAL_AUTHORITY_EVIDENCE_LIMIT:
                ]
            )
            self._logical_authority_digests = (
                self._logical_authority_digests[
                    -self._LOGICAL_AUTHORITY_EVIDENCE_LIMIT:
                ]
            )
        return digest

    def _assert_logical_authority(
        self,
        *,
        stage: str,
        rows: object,
    ) -> str:
        digest = (
            self.tensor_parallel_coordinator
            .assert_logical_authority(
                stage=stage,
                rows=rows,
            )
        )
        return self._record_logical_authority(
            stage=stage,
            rows=rows,
            digest=digest,
        )

    def _converge_stage(
        self,
        *,
        stage: str,
        rows: object,
        local_error: BaseException | None,
    ) -> str:
        digest = self.tensor_parallel_coordinator.converge_stage(
            stage=stage,
            rows=rows,
            local_error=local_error,
        )
        return self._record_logical_authority(
            stage=stage,
            rows=rows,
            digest=digest,
        )

    def pending_prompt(
        self,
        sequence_id: int,
    ) -> AutoregressiveDraftPendingPrompt | None:
        sequence_id = _nonnegative_integer(
            sequence_id,
            "sequence_id",
        )
        return self._pending_prompts.get(sequence_id)

    @staticmethod
    def _validate_observation(
        row: TargetPrefillObservation,
    ) -> tuple[int, int, tuple[int, ...], tuple[int, ...], bool]:
        if not isinstance(row, TargetPrefillObservation):
            raise ValueError(
                "row must be a TargetPrefillObservation"
            )
        sequence_id = _nonnegative_integer(
            row.sequence_id,
            "sequence_id",
        )
        sequence_epoch = _nonnegative_integer(
            row.sequence_epoch,
            "sequence_epoch",
        )
        if not isinstance(row.token_ids, tuple) or not row.token_ids:
            raise ValueError(
                "target prefill token IDs must be a non-empty tuple"
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
            raise ValueError(
                "target prefill positions must be a tensor"
            )
        if row.positions.ndim != 1:
            raise ValueError(
                "target prefill positions must be rank one"
            )
        if row.positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "target prefill positions must use an integer dtype"
            )
        if int(row.positions.shape[0]) != len(row.token_ids):
            raise ValueError(
                "target prefill position row count mismatch"
            )
        hidden_shape = getattr(row.target_hidden, "shape", None)
        if (
            hidden_shape is None
            or len(hidden_shape) == 0
            or int(hidden_shape[0]) != len(row.token_ids)
        ):
            raise ValueError(
                "target hidden row count must match token row count"
            )
        if not isinstance(row.is_final_chunk, bool):
            raise ValueError(
                "target prefill final chunk flag must be bool"
            )
        positions = tuple(
            int(position)
            for position in row.positions.detach().cpu().tolist()
        )
        return (
            sequence_id,
            sequence_epoch,
            row.token_ids,
            positions,
            row.is_final_chunk,
        )

    def observe_target_prefill(
        self,
        rows: tuple[TargetPrefillObservation, ...],
    ) -> None:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "target prefill observations must be a non-empty tuple"
            )
        normalized = tuple(
            self._validate_observation(row)
            for row in rows
        )
        sequence_ids = tuple(row[0] for row in normalized)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "target prefill sequence IDs must be unique"
            )

        updates = {}
        for (
            sequence_id,
            sequence_epoch,
            token_ids,
            positions,
            is_final,
        ) in normalized:
            previous = self._pending_prompts.get(sequence_id)
            if previous is None:
                expected_start = 0
                prior_tokens = ()
                prior_positions = ()
            else:
                if previous.sequence_epoch != sequence_epoch:
                    raise RuntimeError(
                        "target prefill sequence epoch is stale"
                    )
                if previous.is_final:
                    raise RuntimeError(
                        "target prefill sequence is already final"
                    )
                expected_start = len(previous.token_ids)
                prior_tokens = previous.token_ids
                prior_positions = previous.positions
            expected_positions = tuple(
                range(
                    expected_start,
                    expected_start + len(token_ids),
                )
            )
            if expected_start == 0 and positions[0] != 0:
                raise ValueError(
                    "target prefill must start at position zero"
                )
            if positions != expected_positions:
                raise ValueError(
                    "target prefill positions must be contiguous"
                )
            updates[sequence_id] = (
                AutoregressiveDraftPendingPrompt(
                    sequence_id=sequence_id,
                    sequence_epoch=sequence_epoch,
                    token_ids=prior_tokens + token_ids,
                    positions=prior_positions + positions,
                    is_final=is_final,
                )
            )
        self._pending_prompts.update(updates)

    @staticmethod
    def _validate_proposal_input(
        input_row: ModelRunnerProposalInput,
    ) -> int:
        if not isinstance(input_row, ModelRunnerProposalInput):
            raise ValueError(
                "proposal input must be ModelRunnerProposalInput"
            )
        _nonnegative_integer(
            input_row.sequence_id,
            "sequence_id",
        )
        _nonnegative_integer(
            input_row.remaining_output_tokens,
            "remaining_output_tokens",
        )
        _nonnegative_integer(
            input_row.max_proposal_tokens,
            "max_proposal_tokens",
        )
        _nonnegative_integer(
            input_row.first_target_token,
            "first_target_token",
        )
        context_token_count = proposal_input_context_token_count(
            input_row
        )
        if context_token_count <= 0:
            raise ValueError(
                "context_token_count must be positive"
            )
        return context_token_count

    def _proposal_exact_q(
        self,
        input_row: ModelRunnerProposalInput,
    ) -> int:
        return min(
            input_row.remaining_output_tokens,
            input_row.max_proposal_tokens,
            self.max_proposal_tokens,
        )

    def _blockwise_residency_allocator(self):
        allocator = self.proposal_kv_cache.entry_allocator
        if (
            getattr(
                allocator,
                "blockwise_attention_adapter",
                None,
            )
            is None
            or not callable(
                getattr(
                    allocator,
                    "ensure_blockwise_writable",
                    None,
                )
            )
        ):
            return None
        return allocator

    def _bootstrap_sequences_blockwise(
        self,
        pending_rows: tuple[
            AutoregressiveDraftPendingPrompt, ...
        ],
        allocator,
    ) -> None:
        def converge_phase(
            stage: str,
            rows: object,
            operation,
        ) -> None:
            local_error = None
            try:
                operation()
            except BaseException as error:
                local_error = error
            try:
                self._converge_stage(
                    stage=stage,
                    rows=rows,
                    local_error=local_error,
                )
            except BaseException:
                if (
                    local_error is not None
                    and self.tensor_parallel_size == 1
                ):
                    raise local_error
                raise
            if local_error is not None:
                raise local_error

        def cleanup_round(
            transactions,
            prepared_ticket_ids,
        ) -> None:
            prepared_transactions = transactions[
                :len(prepared_ticket_ids)
            ]
            for transaction, ticket_id in reversed(tuple(zip(
                prepared_transactions,
                prepared_ticket_ids,
            ))):
                if transaction.state == "prepared":
                    self.proposal_kv_cache.rollback_finalize(
                        ticket_id
                    )
            for transaction in reversed(transactions):
                if transaction.state in (
                    "reserved",
                    "materialized",
                ):
                    self.proposal_kv_cache.abort(
                        transaction.transaction_id
                    )

        max_prompt_tokens = max(
            len(pending.token_ids) for pending in pending_rows
        )
        try:
            for token_index in range(max_prompt_tokens):
                active_rows = tuple(
                    pending
                    for pending in pending_rows
                    if token_index < len(pending.token_ids)
                )
                transactions = []
                writable_leases = []
                decode_rows = []
                prepared_ticket_ids = []
                phase_rows = tuple({
                    "sequence_id": pending.sequence_id,
                    "sequence_epoch": pending.sequence_epoch,
                    "prompt_token_index": token_index,
                    "prompt_token_count": len(
                        pending.token_ids
                    ),
                } for pending in active_rows)

                def begin_round() -> None:
                    for pending in active_rows:
                        transaction = (
                            self.proposal_kv_cache.begin(
                                pending.sequence_id,
                                pending.sequence_epoch,
                                1,
                            )
                        )
                        transactions.append(transaction)
                        write_identity = (
                            transaction
                            .staged_entry_identities[0]
                        )
                        writable_lease = (
                            allocator.ensure_blockwise_writable((
                                write_identity,
                            ))
                        )
                        writable_leases.append(
                            writable_lease
                        )
                        visible_identities = (
                            self.proposal_kv_cache
                            .committed_entry_identities(
                                pending.sequence_id
                            )
                            + (write_identity,)
                        )
                        decode_rows.append(
                            AutoregressiveDraftDecodeRow(
                                transaction=transaction,
                                step=0,
                                input_token_id=(
                                    pending.token_ids[
                                        token_index
                                    ]
                                ),
                                position=(
                                    pending.positions[
                                        token_index
                                    ]
                                ),
                                writable_physical_slot_id=(
                                    writable_lease
                                    .physical_slot_ids[0]
                                ),
                                visible_physical_slot_ids=(
                                    writable_lease
                                    .physical_slot_ids
                                ),
                                visible_logical_entry_ids=tuple(
                                    identity.logical_entry_id
                                    for identity
                                    in visible_identities
                                ),
                                blockwise_offload=True,
                            )
                        )

                def decode_round() -> None:
                    try:
                        self.backend.decode_step_batch(
                            tuple(decode_rows)
                        )
                    except BaseException:
                        for lease in writable_leases:
                            allocator.record_write_complete(
                                lease
                            )
                        raise
                    for lease in writable_leases:
                        allocator.record_write_complete(lease)

                def materialize_round() -> None:
                    for transaction in transactions:
                        self.proposal_kv_cache.mark_materialized(
                            transaction,
                            1,
                        )

                def prepare_round() -> None:
                    for transaction in transactions:
                        ticket = (
                            self.proposal_kv_cache
                            .prepare_finalize(
                                transaction.transaction_id,
                                accepted_proposal_tokens=2,
                            )
                        )
                        prepared_ticket_ids.append(
                            ticket.ticket_id
                        )

                try:
                    converge_phase(
                        "bootstrap_begin",
                        phase_rows,
                        begin_round,
                    )
                    converge_phase(
                        "bootstrap_prefill",
                        phase_rows,
                        decode_round,
                    )
                    converge_phase(
                        "bootstrap_materialize",
                        phase_rows,
                        materialize_round,
                    )
                    converge_phase(
                        "bootstrap_prepare",
                        phase_rows,
                        prepare_round,
                    )
                    converge_phase(
                        "bootstrap_prepared",
                        tuple({
                            **row,
                            "logical_state": "prepared",
                        } for row in phase_rows),
                        lambda: None,
                    )
                except BaseException:
                    cleanup_round(
                        transactions,
                        prepared_ticket_ids,
                    )
                    raise

                commit_error = None
                try:
                    for ticket_id in prepared_ticket_ids:
                        self.proposal_kv_cache.commit_finalize(
                            ticket_id
                        )
                except BaseException as error:
                    commit_error = error
                try:
                    self._converge_stage(
                        stage="bootstrap_commit",
                        rows=phase_rows,
                        local_error=commit_error,
                    )
                    if commit_error is not None:
                        raise commit_error
                except BaseException as error:
                    cleanup_round(
                        transactions,
                        prepared_ticket_ids,
                    )
                    raise RuntimeError(
                        "autoregressive draft runtime poisoned "
                        "after bootstrap commit"
                    ) from error
        except BaseException:
            for pending in reversed(pending_rows):
                state = self.proposal_kv_cache.sequence_state(
                    pending.sequence_id
                )
                if (
                    state is not None
                    and state.active_transaction_id is None
                    and state.active_ticket_id is None
                ):
                    self.proposal_kv_cache.release_sequence(
                        pending.sequence_id,
                        sequence_epoch=pending.sequence_epoch,
                    )
            raise

        committed_rows = tuple({
            "sequence_id": pending.sequence_id,
            "sequence_epoch": pending.sequence_epoch,
            "prompt_token_count": len(pending.token_ids),
            "committed_logical_length": (
                self.proposal_kv_cache.committed_length(
                    pending.sequence_id
                )
            ),
            "logical_state": "committed",
        } for pending in pending_rows)
        self._assert_logical_authority(
            stage="bootstrap_committed",
            rows=committed_rows,
        )
        for pending in pending_rows:
            if self.proposal_kv_cache.committed_length(
                pending.sequence_id
            ) != len(pending.token_ids):
                raise RuntimeError(
                    "incremental bootstrap committed length "
                    "mismatch"
                )
            self._bootstrapped[pending.sequence_id] = (
                _BootstrappedSequence(
                    sequence_id=pending.sequence_id,
                    sequence_epoch=pending.sequence_epoch,
                    prompt_token_count=len(pending.token_ids),
                )
            )
            self._bootstrap_rows.append({
                "sequence_id": pending.sequence_id,
                "sequence_epoch": pending.sequence_epoch,
                "prompt_token_count": len(pending.token_ids),
                "bootstrap_commit_encoding": (
                    len(pending.token_ids) + 1
                ),
            })

    def _bootstrap_sequences(
        self,
        sequence_ids: tuple[int, ...],
    ) -> None:
        pending_rows = []
        new_rows = []
        transactions = []
        for sequence_id in sequence_ids:
            existing = self._bootstrapped.get(sequence_id)
            pending = self._pending_prompts.get(sequence_id)
            if existing is not None:
                if (
                    pending is None
                    or existing.sequence_epoch
                    != pending.sequence_epoch
                ):
                    raise RuntimeError(
                        "bootstrapped sequence epoch is stale"
                    )
                continue
            if pending is None:
                raise RuntimeError(
                    "target prefill observation is required "
                    "before bootstrap"
                )
            if not pending.is_final:
                raise RuntimeError(
                    "target prefill final chunk is required "
                    "before bootstrap"
                )
            state = self.proposal_kv_cache.sequence_state(
                sequence_id
            )
            if state is not None and (
                state.committed_entry_identities
                or state.active_transaction_id is not None
                or state.active_ticket_id is not None
            ):
                raise RuntimeError(
                    "bootstrap retry requires zero owned prompt state"
                )
            pending_rows.append(pending)

        if not pending_rows:
            return

        bootstrap_preflight_rows = tuple({
            "sequence_id": pending.sequence_id,
            "sequence_epoch": pending.sequence_epoch,
            "prompt_token_ids": pending.token_ids,
            "prompt_positions": pending.positions,
            "final_chunk_seen": pending.is_final,
        } for pending in pending_rows)
        self._assert_logical_authority(
            stage="bootstrap_preflight",
            rows=bootstrap_preflight_rows,
        )
        blockwise_allocator = (
            self._blockwise_residency_allocator()
        )
        if blockwise_allocator is not None:
            self._bootstrap_sequences_blockwise(
                tuple(pending_rows),
                blockwise_allocator,
            )
            return

        prepared_ticket_ids = []
        writable_leases = []

        def converge_phase(
            stage: str,
            rows: object,
            operation,
        ) -> None:
            local_error = None
            try:
                operation()
            except BaseException as error:
                local_error = error
            try:
                self._converge_stage(
                    stage=stage,
                    rows=rows,
                    local_error=local_error,
                )
            except BaseException:
                if (
                    local_error is not None
                    and self.tensor_parallel_size == 1
                ):
                    raise local_error
                raise
            if local_error is not None:
                raise local_error

        bootstrap_phase_rows = tuple({
            "sequence_id": pending.sequence_id,
            "sequence_epoch": pending.sequence_epoch,
            "prompt_token_count": len(pending.token_ids),
        } for pending in pending_rows)

        def begin_transactions() -> None:
            for pending in pending_rows:
                transaction = self.proposal_kv_cache.begin(
                    pending.sequence_id,
                    pending.sequence_epoch,
                    len(pending.token_ids),
                )
                writable_lease = (
                    self.proposal_kv_cache.entry_allocator
                    .ensure_writable(
                        transaction.staged_entry_identities
                    )
                )
                transactions.append(transaction)
                writable_leases.append(writable_lease)
                new_rows.append(
                    AutoregressiveDraftPrefillRow(
                        transaction=transaction,
                        token_ids=pending.token_ids,
                        positions=pending.positions,
                        physical_slot_ids=(
                            writable_lease.physical_slot_ids
                        ),
                    )
                )

        def prefill_transactions() -> None:
            allocator = self.proposal_kv_cache.entry_allocator
            try:
                self.backend.prefill_batch(tuple(new_rows))
            except BaseException:
                for lease in writable_leases:
                    allocator.record_write_complete(lease)
                raise
            for lease in writable_leases:
                allocator.record_write_complete(lease)

        def materialize_transactions() -> None:
            for transaction in transactions:
                self.proposal_kv_cache.mark_materialized(
                    transaction,
                    len(transaction.staged_entry_identities),
                )

        def prepare_transactions() -> None:
            for transaction in transactions:
                ticket = self.proposal_kv_cache.prepare_finalize(
                    transaction.transaction_id,
                    accepted_proposal_tokens=(
                        len(
                            transaction.staged_entry_identities
                        )
                        + 1
                    ),
                )
                prepared_ticket_ids.append(ticket.ticket_id)

        try:
            converge_phase(
                "bootstrap_begin",
                bootstrap_phase_rows,
                begin_transactions,
            )
            converge_phase(
                "bootstrap_prefill",
                bootstrap_phase_rows,
                prefill_transactions,
            )
            converge_phase(
                "bootstrap_materialize",
                bootstrap_phase_rows,
                materialize_transactions,
            )
            converge_phase(
                "bootstrap_prepare",
                bootstrap_phase_rows,
                prepare_transactions,
            )
            prepared_rows = tuple({
                "sequence_id": transaction.sequence_id,
                "sequence_epoch": transaction.sequence_epoch,
                "prompt_token_count": len(
                    transaction.staged_entry_identities
                ),
                "logical_state": "prepared",
            } for transaction in transactions)
            converge_phase(
                "bootstrap_prepared",
                prepared_rows,
                lambda: None,
            )
        except BaseException:
            prepared_count = len(prepared_ticket_ids)
            for ticket_id in reversed(prepared_ticket_ids):
                self.proposal_kv_cache.rollback_finalize(ticket_id)
            for transaction in reversed(
                transactions[prepared_count:]
            ):
                if transaction.state in (
                    "reserved",
                    "materialized",
                ):
                    self.proposal_kv_cache.abort(
                        transaction.transaction_id
                    )
            raise

        commit_error = None
        try:
            for ticket_id in prepared_ticket_ids:
                self.proposal_kv_cache.commit_finalize(ticket_id)
        except BaseException as error:
            commit_error = error
        try:
            self._converge_stage(
                stage="bootstrap_commit",
                rows=tuple({
                    "sequence_id": transaction.sequence_id,
                    "sequence_epoch": transaction.sequence_epoch,
                    "prompt_token_count": len(
                        transaction.staged_entry_identities
                    ),
                } for transaction in transactions),
                local_error=commit_error,
            )
            if commit_error is not None:
                raise commit_error
            committed_rows = tuple({
                "sequence_id": transaction.sequence_id,
                "sequence_epoch": transaction.sequence_epoch,
                "prompt_token_count": len(
                    transaction.staged_entry_identities
                ),
                "committed_logical_length": (
                    self.proposal_kv_cache.committed_length(
                        transaction.sequence_id
                    )
                ),
                "logical_state": "committed",
            } for transaction in transactions)
            self._assert_logical_authority(
                stage="bootstrap_committed",
                rows=committed_rows,
            )
        except BaseException as error:
            raise RuntimeError(
                "autoregressive draft runtime poisoned "
                "after bootstrap commit"
            ) from error

        for transaction in transactions:
            pending = self._pending_prompts[
                transaction.sequence_id
            ]
            self._bootstrapped[transaction.sequence_id] = (
                _BootstrappedSequence(
                    sequence_id=transaction.sequence_id,
                    sequence_epoch=transaction.sequence_epoch,
                    prompt_token_count=len(pending.token_ids),
                )
            )
            self._bootstrap_rows.append({
                "sequence_id": transaction.sequence_id,
                "sequence_epoch": transaction.sequence_epoch,
                "prompt_token_count": len(pending.token_ids),
                "bootstrap_commit_encoding": (
                    len(pending.token_ids) + 1
                ),
            })

    def _validate_logit_rows(
        self,
        rows: object,
        *,
        expected_count: int,
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(rows, tuple):
            raise ValueError(
                "decode logits must be returned as a tuple"
            )
        if len(rows) != expected_count:
            raise ValueError(
                "decode logit row count must match batch"
            )
        normalized = []
        vocab_size = None
        device = torch.device(self.backend.device)
        for row in rows:
            if not isinstance(row, torch.Tensor):
                raise ValueError(
                    "decode logit row must be a tensor"
                )
            if row.ndim != 1:
                raise ValueError(
                    "decode logit row must be rank one"
                )
            if not row.is_floating_point():
                raise ValueError(
                    "decode logits must use a floating dtype"
                )
            if row.device != device:
                raise ValueError(
                    "decode logits must use the backend device"
                )
            if row.numel() <= 1:
                raise ValueError(
                    "decode vocabulary width must exceed one"
                )
            if vocab_size is None:
                vocab_size = int(row.numel())
            elif int(row.numel()) != vocab_size:
                raise ValueError(
                    "decode logit vocabulary widths must match"
                )
            if not bool(torch.isfinite(row).all().item()):
                raise ValueError(
                    "decode logits must contain finite values"
                )
            normalized.append(row)
        return tuple(normalized)

    def _run_exact_q_group_eager(
        self,
        exact_q: int,
        indexed_rows: tuple[
            tuple[int, ModelRunnerProposalInput, int],
            ...,
        ],
    ) -> AutoregressiveDraftGroupExecution:
        transactions = []
        proposal_tokens = []
        blockwise_allocator = (
            self._blockwise_residency_allocator()
        )
        try:
            started_at = self._clock()
            try:
                for _, input_row, _ in indexed_rows:
                    bootstrap = self._bootstrapped[
                        input_row.sequence_id
                    ]
                    transaction = self.proposal_kv_cache.begin(
                        input_row.sequence_id,
                        bootstrap.sequence_epoch,
                        exact_q - 1,
                    )
                    transactions.append(transaction)
                    proposal_tokens.append([
                        input_row.first_target_token
                    ])
            finally:
                self._record_proposal_forward_detail(
                    "setup",
                    started_at,
                )

            for step in range(exact_q - 1):
                started_at = self._clock()
                allocator = self.proposal_kv_cache.entry_allocator
                read_leases = []
                write_leases = []
                decode_rows = []
                for transaction, tokens, (
                    _,
                    _,
                    context_token_count,
                ) in zip(
                    transactions,
                    proposal_tokens,
                    indexed_rows,
                ):
                    read_identities = (
                        self.proposal_kv_cache
                        .committed_entry_identities(
                            transaction.sequence_id
                        )
                        + transaction.staged_entry_identities[:step]
                    )
                    write_identity = (
                        transaction.staged_entry_identities[step]
                    )
                    if blockwise_allocator is None:
                        read_lease = allocator.ensure_readable(
                            read_identities
                        )
                        write_lease = allocator.ensure_writable((
                            write_identity,
                        ))
                        visible_physical_slot_ids = (
                            read_lease.physical_slot_ids
                            + write_lease.physical_slot_ids
                        )
                        visible_logical_entry_ids = ()
                    else:
                        read_lease = None
                        write_lease = (
                            blockwise_allocator
                            .ensure_blockwise_writable((
                                write_identity,
                            ))
                        )
                        visible_physical_slot_ids = (
                            write_lease.physical_slot_ids
                        )
                        visible_logical_entry_ids = tuple(
                            identity.logical_entry_id
                            for identity in (
                                read_identities
                                + (write_identity,)
                            )
                        )
                    read_leases.append(read_lease)
                    write_leases.append(write_lease)
                    decode_rows.append(
                        AutoregressiveDraftDecodeRow(
                            transaction=transaction,
                            step=step,
                            input_token_id=tokens[-1],
                            position=context_token_count + step,
                            writable_physical_slot_id=(
                                write_lease.physical_slot_ids[0]
                            ),
                            visible_physical_slot_ids=(
                                visible_physical_slot_ids
                            ),
                            visible_logical_entry_ids=(
                                visible_logical_entry_ids
                            ),
                            blockwise_offload=(
                                blockwise_allocator is not None
                            ),
                        )
                    )
                decode_rows = tuple(decode_rows)
                self._record_proposal_forward_detail(
                    "setup",
                    started_at,
                )

                def run_decode_step():
                    decode_started_at = self._clock()
                    try:
                        return self.backend.decode_step_batch(
                            decode_rows
                        )
                    except BaseException:
                        for read_lease, write_lease in zip(
                            read_leases,
                            write_leases,
                        ):
                            if read_lease is not None:
                                allocator.record_read_complete(
                                    read_lease
                                )
                            allocator.record_write_complete(
                                write_lease
                            )
                        raise
                    finally:
                        self._record_proposal_forward_detail(
                            "backend_submit",
                            decode_started_at,
                        )

                def complete_decode_step() -> None:
                    for read_lease, write_lease in zip(
                        read_leases,
                        write_leases,
                    ):
                        if read_lease is not None:
                            allocator.record_read_complete(
                                read_lease
                            )
                        allocator.record_write_complete(write_lease)

                if self.tensor_parallel_size == 1:
                    root_logits_or_none = run_decode_step()
                    complete_decode_step()
                    selection_started_at = self._clock()
                    logit_rows = self._validate_logit_rows(
                        root_logits_or_none,
                        expected_count=len(decode_rows),
                    )
                    logits = torch.stack(logit_rows, dim=0)
                    selected = (
                        select_tensor_parallel_greedy_tokens(
                            logits,
                            rank=self.tensor_parallel_rank,
                            world_size=self.tensor_parallel_size,
                            batch_size=len(decode_rows),
                            device=torch.device(
                                self.backend.device
                            ),
                        )
                    )
                    self._record_proposal_forward_detail(
                        "selection_collective",
                        selection_started_at,
                    )
                else:
                    local_error = None
                    selected = None
                    try:
                        root_logits_or_none = run_decode_step()
                        complete_decode_step()
                        selection_started_at = self._clock()
                        if self.tensor_parallel_rank == 0:
                            logit_rows = (
                                self._validate_logit_rows(
                                    root_logits_or_none,
                                    expected_count=(
                                        len(decode_rows)
                                    ),
                                )
                            )
                            logits = torch.stack(
                                logit_rows,
                                dim=0,
                            )
                        else:
                            if root_logits_or_none is not None:
                                raise ValueError(
                                    "non-root logits must be None"
                                )
                            logits = None
                        selected = (
                            select_tensor_parallel_greedy_tokens(
                                logits,
                                rank=self.tensor_parallel_rank,
                                world_size=(
                                    self.tensor_parallel_size
                                ),
                                batch_size=len(decode_rows),
                                device=torch.device(
                                    self.backend.device
                                ),
                            )
                        )
                        self._record_proposal_forward_detail(
                            "selection_collective",
                            selection_started_at,
                        )
                    except BaseException as error:
                        local_error = error
                    authority_started_at = self._clock()
                    self._converge_stage(
                        stage=f"proposal_decode_step_{step}",
                        rows={
                            "sequence_ids": tuple(
                                transaction.sequence_id
                                for transaction in transactions
                            ),
                            "step": step,
                            "exact_q": exact_q,
                        },
                        local_error=local_error,
                    )
                    self._record_proposal_forward_detail(
                        "decode_authority",
                        authority_started_at,
                    )
                    if local_error is not None:
                        raise local_error
                    if selected is None:
                        raise RuntimeError(
                            "proposal token selection is unavailable"
                        )
                readback_started_at = self._clock()
                selected_token_ids = selected.tolist()
                for (
                    transaction,
                    tokens,
                    token_id,
                ) in zip(
                    transactions,
                    proposal_tokens,
                    selected_token_ids,
                ):
                    normalized_token = int(token_id)
                    tokens.append(normalized_token)
                    self._selected_token_rows.append({
                        "sequence_id": (
                            transaction.sequence_id
                        ),
                        "transaction_id": (
                            transaction.transaction_id
                        ),
                        "step": step,
                        "token_id": normalized_token,
                    })
                self._record_proposal_forward_detail(
                    "token_readback",
                    readback_started_at,
                )

        except BaseException:
            for transaction in reversed(transactions):
                if transaction.state in (
                    "reserved",
                    "materialized",
                ):
                    self.proposal_kv_cache.abort(
                        transaction.transaction_id
                    )
            raise
        return AutoregressiveDraftGroupExecution(
            transactions=tuple(transactions),
            token_rows=tuple(
                tuple(tokens) for tokens in proposal_tokens
            ),
            execution_mode="eager",
        )

    def _register_exact_q_group(
        self,
        exact_q: int,
        indexed_rows: tuple[
            tuple[int, ModelRunnerProposalInput, int],
            ...,
        ],
        execution: AutoregressiveDraftGroupExecution,
    ) -> tuple[DraftProposal, ...]:
        if not isinstance(
            execution,
            AutoregressiveDraftGroupExecution,
        ):
            raise ValueError(
                "exact-Q execution must return "
                "AutoregressiveDraftGroupExecution"
            )
        if (
            len(execution.transactions) != len(indexed_rows)
            or len(execution.token_rows) != len(indexed_rows)
        ):
            raise ValueError(
                "exact-Q execution result count must match rows"
            )
        if (
            not isinstance(execution.execution_mode, str)
            or not execution.execution_mode
        ):
            raise ValueError(
                "exact-Q execution mode must be non-empty"
            )
        materialize_started_at = self._clock()
        proposals = []
        materialized_rows = []
        try:
            for transaction, tokens, indexed_row in zip(
                execution.transactions,
                execution.token_rows,
                indexed_rows,
            ):
                input_index, input_row, _ = indexed_row
                if (
                    not isinstance(transaction, ProposalKVTransaction)
                    or self.proposal_kv_cache.transaction(
                        transaction.transaction_id
                    )
                    is not transaction
                    or transaction.sequence_id
                    != input_row.sequence_id
                ):
                    raise ValueError(
                        "exact-Q transaction ownership is invalid"
                    )
                if (
                    not isinstance(tokens, tuple)
                    or len(tokens) != exact_q
                    or any(
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or token_id < 0
                        for token_id in tokens
                    )
                    or tokens[0]
                    != input_row.first_target_token
                ):
                    raise ValueError(
                        "exact-Q proposal tokens are invalid"
                    )
                self.proposal_kv_cache.mark_materialized(
                    transaction,
                    exact_q - 1,
                )
                metadata = {
                    "exact_q": exact_q,
                    "staged_entry_count": exact_q - 1,
                }
                if execution.execution_mode != "eager":
                    metadata["execution_mode"] = (
                        execution.execution_mode
                    )
                proposals.append(
                    DraftProposal(
                        sequence_id=transaction.sequence_id,
                        token_ids=tokens,
                        source_type=(
                            self.capabilities.source_type
                        ),
                        metadata=metadata,
                        proposal_transaction_id=(
                            transaction.transaction_id
                        ),
                    )
                )
                materialized_rows.append({
                    "batch_index": input_index,
                    "sequence_id": transaction.sequence_id,
                    "sequence_epoch": transaction.sequence_epoch,
                    "exact_q": exact_q,
                    "proposal_token_ids": tokens,
                    "staged_entry_count": exact_q - 1,
                    "logical_state": "materialized",
                })
            self._assert_logical_authority(
                stage="proposal_materialized",
                rows=tuple(materialized_rows),
            )
            registrations = tuple(
                ProposalKVRegistration(
                    sequence_id=proposal.sequence_id,
                    sequence_epoch=(
                        self._bootstrapped[
                            proposal.sequence_id
                        ].sequence_epoch
                    ),
                    proposal=proposal,
                )
                for proposal in proposals
            )
            registered = (
                self.proposal_kv_lifecycle.register_batch(
                    registrations
                )
            )
        except BaseException:
            for transaction in reversed(
                execution.transactions
            ):
                if transaction.state in (
                    "reserved",
                    "materialized",
                ):
                    self.proposal_kv_cache.abort(
                        transaction.transaction_id
                    )
            raise
        for proposal in registered:
            transaction_id = proposal.proposal_transaction_id
            self._proposal_exact_q_by_transaction[
                transaction_id
            ] = exact_q
            self._proposal_token_ids_by_transaction[
                transaction_id
            ] = proposal.token_ids
        self._record_proposal_forward_detail(
            "materialize_register",
            materialize_started_at,
        )
        return registered

    def _run_exact_q_group(
        self,
        exact_q: int,
        indexed_rows: tuple[
            tuple[int, ModelRunnerProposalInput, int],
            ...,
        ],
    ) -> tuple[DraftProposal, ...]:
        if exact_q == 1 or self.graph_runner is None:
            execution = self._run_exact_q_group_eager(
                exact_q,
                indexed_rows,
            )
        else:
            execution = self.graph_runner.run(
                exact_q=exact_q,
                rows=indexed_rows,
                eager=self._run_exact_q_group_eager,
            )
        return self._register_exact_q_group(
            exact_q,
            indexed_rows,
            execution,
        )

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]:
        if not isinstance(inputs, tuple) or not inputs:
            raise ValueError(
                "proposal inputs must be a non-empty tuple"
            )
        normalized = []
        sequence_ids = []
        for input_index, input_row in enumerate(inputs):
            context_token_count = self._validate_proposal_input(
                input_row
            )
            if input_row.sequence_id in sequence_ids:
                raise ValueError(
                    "proposal input sequence IDs must be unique"
                )
            sequence_ids.append(input_row.sequence_id)
            normalized.append((
                input_index,
                input_row,
                context_token_count,
                self._proposal_exact_q(input_row),
            ))

        proposal_preflight_rows = tuple({
            "batch_index": input_index,
            "sequence_id": input_row.sequence_id,
            "sequence_epoch": (
                None
                if (
                    pending := self._pending_prompts.get(
                        input_row.sequence_id
                    )
                ) is None
                else pending.sequence_epoch
            ),
            "context_token_count": context_token_count,
            "exact_q": exact_q,
            "first_target_token": input_row.first_target_token,
        } for (
            input_index,
            input_row,
            context_token_count,
            exact_q,
        ) in normalized)
        self._assert_logical_authority(
            stage="proposal_preflight",
            rows=proposal_preflight_rows,
        )

        nonempty_sequence_ids = tuple(
            input_row.sequence_id
            for _, input_row, _, exact_q in normalized
            if exact_q > 0
        )
        started_at = self._clock()
        try:
            self._bootstrap_sequences(nonempty_sequence_ids)
        finally:
            self._record_timing(
                "prompt_bootstrap",
                started_at,
            )

        proposals: list[DraftProposal | None] = [
            None for _ in inputs
        ]
        grouped_rows: dict[
            int,
            list[tuple[int, ModelRunnerProposalInput, int]],
        ] = {}
        for (
            input_index,
            input_row,
            context_token_count,
            exact_q,
        ) in normalized:
            if exact_q == 0:
                proposals[input_index] = DraftProposal(
                    sequence_id=input_row.sequence_id,
                    token_ids=(),
                    source_type=self.capabilities.source_type,
                )
                continue
            grouped_rows.setdefault(exact_q, []).append((
                input_index,
                input_row,
                context_token_count,
            ))

        for exact_q, rows in grouped_rows.items():
            started_at = self._clock()
            try:
                group_proposals = self._run_exact_q_group(
                    exact_q,
                    tuple(rows),
                )
            finally:
                self._record_timing(
                    "proposal_forward",
                    started_at,
                )
            for (
                input_index,
                _,
                _,
            ), proposal in zip(rows, group_proposals):
                proposals[input_index] = proposal

        if any(proposal is None for proposal in proposals):
            raise RuntimeError(
                "proposal execution left an empty result"
            )
        return tuple(proposals)

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str:
        started_at = self._clock()
        try:
            return self._prepare_finalize_batch(rows)
        finally:
            self._record_timing(
                "proposal_finalize",
                started_at,
            )

    def _prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str:
        transaction_ids = (
            self.proposal_kv_lifecycle._validate_finalize_rows(
                rows
            )
        )
        logical_rows = []
        for batch_index, (row, transaction_id) in enumerate(
            zip(rows, transaction_ids)
        ):
            transaction = self.proposal_kv_cache.transaction(
                transaction_id
            )
            if transaction is None:
                raise RuntimeError(
                    "proposal transaction is unavailable"
                )
            exact_q = self._proposal_exact_q_by_transaction.get(
                transaction_id
            )
            proposal_token_ids = (
                self._proposal_token_ids_by_transaction.get(
                    transaction_id
                )
            )
            if exact_q is None or proposal_token_ids is None:
                raise RuntimeError(
                    "proposal finalize authority is unavailable"
                )
            logical_rows.append({
                "batch_index": batch_index,
                "sequence_id": row.sequence_id,
                "sequence_epoch": transaction.sequence_epoch,
                "exact_q": exact_q,
                "proposal_token_ids": proposal_token_ids,
                "accepted_proposal_tokens": (
                    row.accepted_proposal_tokens
                ),
                "committed_proposal_entries": max(
                    row.accepted_proposal_tokens - 1,
                    0,
                ),
            })
        logical_rows = tuple(logical_rows)
        self._assert_logical_authority(
            stage="finalize_preflight",
            rows=logical_rows,
        )

        local_error = None
        ticket_id = None
        try:
            ticket_id = (
                self.proposal_kv_lifecycle.prepare_finalize_batch(
                    rows
                )
            )
        except BaseException as error:
            local_error = error
        try:
            self._converge_stage(
                stage="finalize_prepare",
                rows=logical_rows,
                local_error=local_error,
            )
            if local_error is not None:
                raise local_error
            if ticket_id is None:
                raise RuntimeError(
                    "finalize prepare ticket is unavailable"
                )
            prepared_rows = tuple({
                **logical_row,
                "logical_state": "prepared",
            } for logical_row in logical_rows)
            self._assert_logical_authority(
                stage="finalize_prepared",
                rows=prepared_rows,
            )
        except BaseException:
            if ticket_id is not None:
                self.proposal_kv_lifecycle.rollback_finalize_batch(
                    ticket_id
                )
            if (
                local_error is not None
                and self.tensor_parallel_size == 1
            ):
                raise local_error
            raise

        self._prepared_finalize_rows_by_ticket[ticket_id] = (
            logical_rows
        )
        return ticket_id

    def commit_finalize_batch(self, ticket_id: str) -> None:
        started_at = self._clock()
        try:
            self._commit_finalize_batch(ticket_id)
        finally:
            self._record_timing(
                "proposal_finalize",
                started_at,
            )

    def _commit_finalize_batch(self, ticket_id: str) -> None:
        logical_rows = self._prepared_finalize_rows_by_ticket.get(
            ticket_id
        )
        if logical_rows is None:
            raise ValueError("batch finalize ticket is not active")
        local_error = None
        try:
            self.proposal_kv_lifecycle.commit_finalize_batch(
                ticket_id
            )
        except BaseException as error:
            local_error = error
        try:
            self._converge_stage(
                stage="finalize_commit",
                rows=logical_rows,
                local_error=local_error,
            )
            if local_error is not None:
                raise local_error
            committed_rows = tuple({
                **logical_row,
                "logical_state": "committed",
            } for logical_row in logical_rows)
            self._assert_logical_authority(
                stage="finalize_committed",
                rows=committed_rows,
            )
        except BaseException as error:
            raise RuntimeError(
                "autoregressive draft runtime poisoned "
                "after finalize commit"
            ) from error
        self._prepared_finalize_rows_by_ticket.pop(
            ticket_id,
            None,
        )

    def rollback_finalize_batch(self, ticket_id: str) -> None:
        started_at = self._clock()
        try:
            self._rollback_finalize_batch(ticket_id)
        finally:
            self._record_timing(
                "proposal_finalize",
                started_at,
            )

    def _rollback_finalize_batch(self, ticket_id: str) -> None:
        logical_rows = self._prepared_finalize_rows_by_ticket.get(
            ticket_id
        )
        if logical_rows is None:
            raise ValueError("batch finalize ticket is not active")
        local_error = None
        try:
            self.proposal_kv_lifecycle.rollback_finalize_batch(
                ticket_id
            )
        except BaseException as error:
            local_error = error
        try:
            self._converge_stage(
                stage="finalize_rollback",
                rows=logical_rows,
                local_error=local_error,
            )
            if local_error is not None:
                raise local_error
            rolled_back_rows = tuple({
                **logical_row,
                "logical_state": "rolled_back",
            } for logical_row in logical_rows)
            self._assert_logical_authority(
                stage="finalize_rolled_back",
                rows=rolled_back_rows,
            )
        except BaseException as error:
            raise RuntimeError(
                "autoregressive draft runtime poisoned "
                "after finalize rollback"
            ) from error
        self._prepared_finalize_rows_by_ticket.pop(
            ticket_id,
            None,
        )

    def release_sequence(
        self,
        sequence_id: int,
        *,
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
        self.proposal_kv_lifecycle.assert_sequence_releasable(
            sequence_id,
            sequence_epoch,
        )
        pending = self._pending_prompts.get(sequence_id)
        if (
            pending is not None
            and pending.sequence_epoch != sequence_epoch
        ):
            raise RuntimeError("sequence epoch is stale")
        bootstrap = self._bootstrapped.get(sequence_id)
        if (
            bootstrap is not None
            and bootstrap.sequence_epoch != sequence_epoch
        ):
            raise RuntimeError("sequence epoch is stale")

        release_preflight_rows = ({
            "sequence_id": sequence_id,
            "sequence_epoch": sequence_epoch,
        },)
        self._assert_logical_authority(
            stage="release_preflight",
            rows=release_preflight_rows,
        )
        local_error = None
        try:
            self.proposal_kv_lifecycle.release_sequence(
                sequence_id,
                sequence_epoch,
            )
        except BaseException as error:
            local_error = error
        try:
            self._converge_stage(
                stage="release_local",
                rows=release_preflight_rows,
                local_error=local_error,
            )
            if local_error is not None:
                raise local_error
            if self.proposal_kv_cache.sequence_state(
                sequence_id
            ) is not None:
                raise RuntimeError(
                    "released sequence state remains active"
                )
            release_complete_rows = ({
                "sequence_id": sequence_id,
                "sequence_epoch": sequence_epoch,
                "active_transaction_count": 0,
                "active_ticket_count": 0,
                "committed_logical_entries": 0,
                "live_local_slot_count": 0,
            },)
            self._assert_logical_authority(
                stage="release_complete",
                rows=release_complete_rows,
            )
        except BaseException as error:
            raise RuntimeError(
                "autoregressive draft runtime poisoned "
                "after release"
            ) from error

        self._pending_prompts.pop(sequence_id, None)
        self._bootstrapped.pop(sequence_id, None)
        self._bootstrap_rows = [
            row
            for row in self._bootstrap_rows
            if row["sequence_id"] != sequence_id
        ]
        self._selected_token_rows = [
            row
            for row in self._selected_token_rows
            if row["sequence_id"] != sequence_id
        ]
        transaction_ids = tuple(
            transaction_id
            for transaction_id
            in self._proposal_exact_q_by_transaction
            if (
                self.proposal_kv_cache.transaction(
                    transaction_id
                )
                is not None
                and self.proposal_kv_cache.transaction(
                    transaction_id
                ).sequence_id == sequence_id
            )
        )
        for transaction_id in transaction_ids:
            self._proposal_exact_q_by_transaction.pop(
                transaction_id,
                None,
            )
            self._proposal_token_ids_by_transaction.pop(
                transaction_id,
                None,
            )

    def authority_snapshot(self) -> dict:
        backend_snapshot = getattr(
            self.backend,
            "authority_snapshot",
            None,
        )
        snapshot = {
            "source_type": "independent_draft_model",
            "backend_identity": self.backend.backend_identity,
            "model_fingerprint": self.backend.model_fingerprint,
            "tokenizer_fingerprint": (
                self.backend.tokenizer_fingerprint
            ),
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "tensor_parallel_size": self.tensor_parallel_size,
            "rank": self.tensor_parallel_rank,
            "world_size": self.tensor_parallel_size,
            "logical_authority_rows": tuple(
                self._logical_authority_rows
            ),
            "logical_authority_digest_count": len(
                self._logical_authority_digests
            ),
            "last_logical_authority_sha256": (
                None
                if not self._logical_authority_digests
                else self._logical_authority_digests[-1]
            ),
            "bootstrap_rows": tuple(
                dict(row) for row in self._bootstrap_rows
            ),
            "selected_token_rows": tuple(
                dict(row) for row in self._selected_token_rows
            ),
            "proposal_exact_q": tuple(
                {
                    "transaction_id": transaction_id,
                    "exact_q": exact_q,
                }
                for transaction_id, exact_q
                in self._proposal_exact_q_by_transaction.items()
            ),
            "timing_ms": dict(self._timing_ms),
            "proposal_forward_detail_ms": dict(
                self._proposal_forward_detail_ms
            ),
            "proposal_kv_lifecycle": (
                self.proposal_kv_lifecycle.authority_snapshot()
            ),
            "backend": (
                None
                if not callable(backend_snapshot)
                else backend_snapshot()
            ),
            "cuda_graph": (
                None
                if self.graph_runner is None
                else self.graph_runner.summary()
            ),
        }
        assert_tensor_free(
            snapshot,
            name="autoregressive draft authority snapshot",
        )
        return snapshot
