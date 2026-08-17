from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import torch

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.engine.qwen35_speculative_trace import (
    Qwen35SpeculativeTraceRecorder,
)
from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateSelectionRow,
)


CandidatePair = tuple[torch.Tensor, torch.Tensor]
CandidateInventory = tuple[CandidatePair, ...]


@dataclass
class _Qwen35SpeculativeStateBatch:
    transaction_id: str
    sequence_ids: tuple[int, ...]
    leases: tuple[HybridStateLease, ...]
    original_candidates: CandidateInventory
    checkpoints: dict[
        int,
        dict[int, CandidateInventory],
    ] = field(default_factory=dict)
    selected: dict[int, CandidateInventory] = field(
        default_factory=dict
    )
    applied_originals: CandidateInventory | None = None
    phase: str = "prepared"


class Qwen35SpeculativeStateOwner:

    def __init__(
        self,
        state_transaction: Qwen35CrossLayerStateTransaction,
    ):
        if not isinstance(
            state_transaction,
            Qwen35CrossLayerStateTransaction,
        ):
            raise ValueError(
                "state_transaction must be a "
                "Qwen35CrossLayerStateTransaction"
            )
        self.state_transaction = state_transaction
        self._transaction_ids = count(1)
        self._active: _Qwen35SpeculativeStateBatch | None = None
        self._terminal: dict[str, tuple[str, tuple[int, ...]]] = {}
        self._trace = Qwen35SpeculativeTraceRecorder()

    @property
    def active(self) -> bool:
        return self._active is not None

    def enable_trace_recording(self, enabled: bool) -> dict:
        return self._trace.enable(enabled)

    def drain_trace_rows(self) -> tuple[dict, ...]:
        return self._trace.drain()

    @staticmethod
    def _sequence_ids(sequences) -> tuple[int, ...]:
        if not isinstance(sequences, tuple) or not sequences:
            raise ValueError("sequences must be a non-empty tuple")
        sequence_ids = tuple(
            getattr(sequence, "seq_id", None)
            for sequence in sequences
        )
        if any(
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
            for sequence_id in sequence_ids
        ):
            raise ValueError(
                "sequences must expose non-negative integer seq_id values"
            )
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("sequence IDs must be unique")
        return sequence_ids

    @staticmethod
    def _validate_leases(
        leases,
        *,
        expected_count: int,
    ) -> tuple[HybridStateLease, ...]:
        if (
            not isinstance(leases, tuple)
            or len(leases) != expected_count
        ):
            raise ValueError(
                "leases must match the sequence inventory"
            )
        if any(type(lease) is not HybridStateLease for lease in leases):
            raise ValueError(
                "leases must contain only HybridStateLease values"
            )
        return leases

    @staticmethod
    def _clone_candidates(
        candidates: CandidateInventory,
    ) -> CandidateInventory:
        return tuple(
            (
                convolution.clone(),
                recurrent.clone(),
            )
            for convolution, recurrent in candidates
        )

    @staticmethod
    def _receipt(
        batch: _Qwen35SpeculativeStateBatch,
        operation: str,
        status: str,
        **extra,
    ) -> dict:
        return {
            "operation": operation,
            "status": status,
            "transaction_id": batch.transaction_id,
            "sequence_ids": list(batch.sequence_ids),
            **extra,
        }

    def _validate_handle(
        self,
        handle,
        *,
        allowed_phases: tuple[str, ...] | None = None,
    ) -> _Qwen35SpeculativeStateBatch:
        transaction_id = (
            handle.get("transaction_id")
            if isinstance(handle, dict)
            else None
        )
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("side-state transaction handle is invalid")
        if self._active is None:
            terminal = self._terminal.get(transaction_id)
            if terminal is not None:
                raise RuntimeError(
                    f"{terminal[0]} side-state transaction is inactive"
                )
            raise ValueError("side-state transaction is not active")
        batch = self._active
        if transaction_id != batch.transaction_id:
            raise ValueError("side-state transaction ID mismatch")
        if handle.get("sequence_ids") != list(batch.sequence_ids):
            raise ValueError(
                "side-state transaction sequence inventory mismatch"
            )
        if (
            allowed_phases is not None
            and batch.phase not in allowed_phases
        ):
            raise RuntimeError(
                "side-state transaction phase mismatch: "
                f"{batch.phase}"
            )
        return batch

    def _validate_candidate_inventory(
        self,
        leases: tuple[HybridStateLease, ...],
        candidates,
    ) -> CandidateInventory:
        slot_ids = tuple(
            adapter._validate_lease_batch(leases)
            for adapter in self.state_transaction.adapters
        )
        reference_slot_ids = slot_ids[0]
        if any(
            value != reference_slot_ids
            for value in slot_ids[1:]
        ):
            raise RuntimeError(
                "adapters resolved inconsistent slot ids"
            )
        self.state_transaction._validate_candidates(
            self.state_transaction.adapters,
            reference_slot_ids,
            candidates,
        )
        return candidates

    @staticmethod
    def _split_candidates(
        candidates: CandidateInventory,
        sequence_ids: tuple[int, ...],
    ) -> dict[int, CandidateInventory]:
        return {
            sequence_id: tuple(
                (
                    convolution[batch_index].clone(),
                    recurrent[batch_index].clone(),
                )
                for convolution, recurrent in candidates
            )
            for batch_index, sequence_id in enumerate(sequence_ids)
        }

    @staticmethod
    def _assemble_candidates(
        sequence_ids: tuple[int, ...],
        per_sequence: dict[int, CandidateInventory],
    ) -> CandidateInventory:
        layer_count = len(per_sequence[sequence_ids[0]])
        return tuple(
            (
                torch.stack(tuple(
                    per_sequence[sequence_id][layer_index][0]
                    for sequence_id in sequence_ids
                )),
                torch.stack(tuple(
                    per_sequence[sequence_id][layer_index][1]
                    for sequence_id in sequence_ids
                )),
            )
            for layer_index in range(layer_count)
        )

    def _validate_sequence_subset(
        self,
        sequence_ids,
    ) -> tuple[
        _Qwen35SpeculativeStateBatch,
        tuple[int, ...],
        tuple[HybridStateLease, ...],
    ]:
        batch = self._active
        if batch is None:
            raise RuntimeError(
                "no active side-state transaction"
            )
        if not isinstance(sequence_ids, tuple) or not sequence_ids:
            raise ValueError(
                "sequence_ids must be a non-empty tuple"
            )
        positions = []
        for sequence_id in sequence_ids:
            if sequence_id not in batch.sequence_ids:
                raise ValueError(
                    "sequence inventory does not match active transaction"
                )
            positions.append(batch.sequence_ids.index(sequence_id))
        if positions != sorted(positions) or len(set(positions)) != len(
            positions
        ):
            raise ValueError(
                "sequence inventory order does not match active transaction"
            )
        leases = tuple(batch.leases[position] for position in positions)
        return batch, sequence_ids, leases

    def _clear_batch(
        self,
        batch: _Qwen35SpeculativeStateBatch,
        terminal_phase: str,
    ) -> None:
        batch.original_candidates = ()
        batch.checkpoints.clear()
        batch.selected.clear()
        batch.applied_originals = None
        batch.phase = terminal_phase
        self._terminal[batch.transaction_id] = (
            terminal_phase,
            batch.sequence_ids,
        )
        self._active = None

    def prepare(
        self,
        sequences,
        leases,
    ) -> dict:
        if self._active is not None:
            raise RuntimeError(
                "a speculative side-state batch is already active"
            )
        sequence_ids = self._sequence_ids(sequences)
        leases = self._validate_leases(
            leases,
            expected_count=len(sequence_ids),
        )
        original_candidates = self.state_transaction.gather(leases)
        transaction_id = (
            f"qwen35-side-state-{next(self._transaction_ids)}"
        )
        batch = _Qwen35SpeculativeStateBatch(
            transaction_id=transaction_id,
            sequence_ids=sequence_ids,
            leases=leases,
            original_candidates=original_candidates,
            checkpoints={
                sequence_id: {}
                for sequence_id in sequence_ids
            },
        )
        self._active = batch
        return self._receipt(
            batch,
            "prepare",
            "prepared",
        )

    def record_first_target(self, prepared_step) -> dict:
        batch = self._active
        if batch is None:
            raise RuntimeError(
                "no active side-state transaction"
            )
        if batch.phase != "prepared":
            raise RuntimeError(
                "first-target state requires a prepared transaction"
            )
        if getattr(prepared_step, "leases", None) != batch.leases:
            raise ValueError(
                "prepared step lease identity mismatch"
            )
        candidates = self._validate_candidate_inventory(
            batch.leases,
            getattr(prepared_step, "final_candidates", None),
        )
        per_sequence = self._split_candidates(
            candidates,
            batch.sequence_ids,
        )
        for sequence_id in batch.sequence_ids:
            if batch.checkpoints[sequence_id]:
                raise RuntimeError(
                    "first-target checkpoint is already recorded"
                )
            batch.checkpoints[sequence_id][1] = (
                per_sequence[sequence_id]
            )
            self._trace.record_checkpoint(
                sequence_id=sequence_id,
                event="first_target_checkpoint",
                checkpoint_index=1,
                candidates=batch.checkpoints[
                    sequence_id
                ][1],
            )
        return self._receipt(
            batch,
            "record_first_target",
            "recorded",
            checkpoint_indices=[1],
        )

    def initial_tail_candidates(
        self,
        sequence_ids,
    ) -> CandidateInventory:
        batch, sequence_ids, _ = self._validate_sequence_subset(
            sequence_ids
        )
        missing = tuple(
            sequence_id
            for sequence_id in sequence_ids
            if 1 not in batch.checkpoints[sequence_id]
        )
        if missing:
            raise RuntimeError(
                "first-target checkpoint is missing"
            )
        return self._assemble_candidates(
            sequence_ids,
            {
                sequence_id: batch.checkpoints[sequence_id][1]
                for sequence_id in sequence_ids
            },
        )

    def record_tail(
        self,
        prepared_step,
        sequence_ids,
    ) -> dict:
        batch, sequence_ids, expected_leases = (
            self._validate_sequence_subset(sequence_ids)
        )
        if batch.phase != "prepared":
            raise RuntimeError(
                "tail state requires a prepared transaction"
            )
        if getattr(prepared_step, "leases", None) != expected_leases:
            raise ValueError(
                "prepared step lease identity mismatch"
            )
        prefix_candidates = getattr(
            prepared_step,
            "prefix_candidates",
            None,
        )
        token_counts = getattr(prepared_step, "token_counts", None)
        if (
            not isinstance(prefix_candidates, tuple)
            or len(prefix_candidates) != len(sequence_ids)
        ):
            raise ValueError(
                "tail prefix candidate inventory mismatch"
            )
        if (
            not isinstance(token_counts, tuple)
            or len(token_counts) != len(sequence_ids)
        ):
            raise ValueError(
                "tail token count inventory mismatch"
            )
        checkpoint_indices = set()
        for sequence_index, sequence_id in enumerate(sequence_ids):
            sequence_prefixes = prefix_candidates[sequence_index]
            if (
                not isinstance(sequence_prefixes, tuple)
                or len(sequence_prefixes)
                != token_counts[sequence_index]
            ):
                raise ValueError(
                    "tail prefix count must match token count"
                )
            for prefix_index, candidate in enumerate(
                sequence_prefixes,
                start=2,
            ):
                if prefix_index in batch.checkpoints[sequence_id]:
                    raise RuntimeError(
                        "tail checkpoint is already recorded"
                    )
                if (
                    not isinstance(candidate, tuple)
                    or len(candidate)
                    != len(self.state_transaction.adapters)
                ):
                    raise ValueError(
                        "tail checkpoint layer inventory mismatch"
                    )
                validated = []
                for adapter, pair in zip(
                    self.state_transaction.adapters,
                    candidate,
                ):
                    if not isinstance(pair, tuple) or len(pair) != 2:
                        raise ValueError(
                            "tail checkpoint must contain state pairs"
                        )
                    convolution, recurrent = pair
                    adapter._validate_candidate(
                        convolution,
                        adapter.convolution[
                            expected_leases[sequence_index].slot_id
                        ],
                        "convolution_state",
                    )
                    adapter._validate_candidate(
                        recurrent,
                        adapter.recurrent[
                            expected_leases[sequence_index].slot_id
                        ],
                        "recurrent_state",
                    )
                    validated.append((
                        convolution.clone(),
                        recurrent.clone(),
                    ))
                batch.checkpoints[sequence_id][prefix_index] = tuple(
                    validated
                )
                self._trace.record_checkpoint(
                    sequence_id=sequence_id,
                    event="tail_checkpoint",
                    checkpoint_index=prefix_index,
                    candidates=batch.checkpoints[
                        sequence_id
                    ][prefix_index],
                )
                checkpoint_indices.add(prefix_index)
        return self._receipt(
            batch,
            "record_tail",
            "recorded",
            checkpoint_indices=sorted(checkpoint_indices),
        )

    def select(
        self,
        handle,
        selection_rows,
    ) -> dict:
        batch = self._validate_handle(
            handle,
            allowed_phases=("prepared",),
        )
        if (
            not isinstance(selection_rows, tuple)
            or len(selection_rows) != len(batch.sequence_ids)
        ):
            raise ValueError(
                "selection rows must match sequence inventory"
            )
        selected = {}
        receipt_rows = []
        for expected_sequence_id, row in zip(
            batch.sequence_ids,
            selection_rows,
        ):
            if type(row) is not SpeculativeSideStateSelectionRow:
                raise ValueError(
                    "selection rows must use the side-state row contract"
                )
            if row.sequence_id != expected_sequence_id:
                raise ValueError(
                    "selection row sequence inventory mismatch"
                )
            checkpoint_index = row.committed_input_count
            if (
                isinstance(checkpoint_index, bool)
                or not isinstance(checkpoint_index, int)
                or checkpoint_index <= 0
            ):
                raise ValueError(
                    "committed input count must be positive"
                )
            checkpoint = batch.checkpoints[
                expected_sequence_id
            ].get(checkpoint_index)
            if checkpoint is None:
                raise ValueError(
                    "selected side-state checkpoint is missing"
                )
            self._trace.record_selection(
                sequence_id=expected_sequence_id,
                committed_input_count=checkpoint_index,
                candidates=checkpoint,
            )
            selected[expected_sequence_id] = checkpoint
            receipt_rows.append({
                "sequence_id": expected_sequence_id,
                "committed_input_count": checkpoint_index,
                "checkpoint_index": checkpoint_index,
            })
        batch.selected = selected
        batch.phase = "selected"
        return self._receipt(
            batch,
            "select",
            "selected",
            rows=receipt_rows,
        )

    def apply(self, handle) -> dict:
        batch = self._validate_handle(
            handle,
            allowed_phases=("selected",),
        )
        current = self.state_transaction.gather(batch.leases)
        for current_pair, original_pair in zip(
            current,
            batch.original_candidates,
        ):
            if (
                not torch.equal(current_pair[0], original_pair[0])
                or not torch.equal(current_pair[1], original_pair[1])
            ):
                raise RuntimeError(
                    "live side state changed after transaction prepare"
                )
        batch.applied_originals = current
        selected = self._assemble_candidates(
            batch.sequence_ids,
            batch.selected,
        )
        self.state_transaction.commit(batch.leases, selected)
        batch.phase = "applied"
        return self._receipt(
            batch,
            "apply",
            "applied",
        )

    def seal(self, handle) -> dict:
        batch = self._validate_handle(
            handle,
            allowed_phases=("applied",),
        )
        receipt = self._receipt(
            batch,
            "seal",
            "sealed",
        )
        self._clear_batch(batch, "sealed")
        return receipt

    def rollback(self, handle) -> dict:
        transaction_id = (
            handle.get("transaction_id")
            if isinstance(handle, dict)
            else None
        )
        terminal = self._terminal.get(transaction_id)
        if terminal is not None:
            terminal_phase, sequence_ids = terminal
            if terminal_phase == "sealed":
                raise RuntimeError(
                    "sealed side-state transaction cannot be rolled back"
                )
            return {
                "operation": "rollback",
                "status": "rolled_back",
                "transaction_id": transaction_id,
                "sequence_ids": list(sequence_ids),
            }
        batch = self._validate_handle(
            handle,
            allowed_phases=(
                "prepared",
                "selected",
                "applied",
            ),
        )
        if batch.phase == "applied":
            self.state_transaction.commit(
                batch.leases,
                batch.applied_originals,
            )
        receipt = self._receipt(
            batch,
            "rollback",
            "rolled_back",
        )
        self._clear_batch(batch, "rolled_back")
        return receipt
