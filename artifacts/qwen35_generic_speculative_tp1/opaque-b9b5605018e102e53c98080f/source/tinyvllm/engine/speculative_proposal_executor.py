from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Protocol

from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
    validate_draft_capabilities,
)


@dataclass(frozen=True)
class ModelRunnerProposalInput:
    sequence_id: int
    token_ids: tuple[int, ...]
    remaining_output_tokens: int
    max_proposal_tokens: int
    first_target_token: int
    target_hidden: object | None = None
    target_logits: object | None = None


@dataclass(frozen=True)
class TargetPrefillObservation:
    sequence_id: int
    sequence_epoch: int
    token_ids: tuple[int, ...]
    positions: object
    target_hidden: object
    is_final_chunk: bool


@dataclass(frozen=True)
class ProposalFinalizeRow:
    sequence_id: int
    proposal_transaction_id: str
    accepted_proposal_tokens: int


class ProposalExecutor(Protocol):
    @property
    def capabilities(self) -> DraftCapabilities:
        ...

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]:
        ...

    def observe_target_prefill(
        self,
        rows: tuple[TargetPrefillObservation, ...],
    ) -> None:
        ...

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str:
        ...

    def commit_finalize_batch(self, ticket_id: str) -> None:
        ...

    def rollback_finalize_batch(self, ticket_id: str) -> None:
        ...


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_token_ids(
    token_ids: object,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(token_ids, tuple):
        raise ValueError(f"{name} token_ids must be a tuple")
    for token_id in token_ids:
        _validate_integer(token_id, f"{name} token")
    return token_ids


def _validate_inputs(
    inputs: object,
    capabilities: DraftCapabilities,
) -> tuple[int, ...]:
    if not isinstance(inputs, tuple) or not inputs:
        raise ValueError(
            "ModelRunner proposal inputs must be a non-empty tuple"
        )
    sequence_ids = []
    for input_row in inputs:
        if not isinstance(input_row, ModelRunnerProposalInput):
            raise ValueError(
                "ModelRunner proposal input must be "
                "ModelRunnerProposalInput"
            )
        sequence_id = _validate_integer(
            input_row.sequence_id,
            "proposal input sequence ID",
        )
        if sequence_id < 0:
            raise ValueError(
                "proposal input sequence ID must be non-negative"
            )
        _validate_token_ids(
            input_row.token_ids,
            "proposal input",
        )
        _validate_integer(
            input_row.remaining_output_tokens,
            "proposal input remaining output tokens",
        )
        if input_row.remaining_output_tokens < 0:
            raise ValueError(
                "proposal input remaining output tokens must be "
                "non-negative"
            )
        _validate_integer(
            input_row.max_proposal_tokens,
            "proposal input token limit",
        )
        if input_row.max_proposal_tokens < 0:
            raise ValueError(
                "proposal input token limit must be non-negative"
            )
        _validate_integer(
            input_row.first_target_token,
            "proposal input first target token",
        )
        if (
            capabilities.requires_target_hidden
            and input_row.target_hidden is None
        ):
            raise ValueError(
                "proposal executor requires target hidden payload"
            )
        if (
            capabilities.requires_target_logits
            and input_row.target_logits is None
        ):
            raise ValueError(
                "proposal executor requires target logits payload"
            )
        sequence_ids.append(sequence_id)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "proposal input sequence IDs must be unique"
        )
    return tuple(sequence_ids)


def _validate_timing(timing_ms: object) -> None:
    if timing_ms is None:
        return
    if not isinstance(timing_ms, dict):
        raise ValueError(
            "proposal timing_ms must be a dictionary"
        )
    for name, value in timing_ms.items():
        if not isinstance(name, str):
            raise ValueError(
                "proposal timing names must be strings"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "proposal timing values must be finite and non-negative"
            )


def _validate_proposals(
    proposals: object,
    inputs: tuple[ModelRunnerProposalInput, ...],
    capabilities: DraftCapabilities,
) -> tuple[DraftProposal, ...]:
    if not isinstance(proposals, tuple):
        raise ValueError(
            "proposal executor must return a tuple"
        )
    input_by_id = {
        input_row.sequence_id: input_row
        for input_row in inputs
    }
    proposals_by_id = {}
    transaction_ids = set()
    for proposal in proposals:
        if not isinstance(proposal, DraftProposal):
            raise ValueError(
                "proposal executor rows must be DraftProposal"
            )
        sequence_id = _validate_integer(
            proposal.sequence_id,
            "proposal sequence ID",
        )
        if sequence_id in proposals_by_id:
            raise ValueError(
                "proposal sequence IDs must be unique"
            )
        if proposal.source_type != capabilities.source_type:
            raise ValueError(
                "proposal source_type must match capabilities"
            )
        token_ids = _validate_token_ids(
            proposal.token_ids,
            "proposal",
        )
        transaction_id = proposal.proposal_transaction_id
        if (
            capabilities.requires_proposal_lifecycle
            and token_ids
        ):
            if (
                not isinstance(transaction_id, str)
                or not transaction_id
            ):
                raise ValueError(
                    "lifecycle proposal requires a non-empty "
                    "transaction ID"
                )
            if transaction_id in transaction_ids:
                raise ValueError(
                    "proposal transaction IDs must be unique"
                )
            transaction_ids.add(transaction_id)
        if len(token_ids) > capabilities.max_proposal_tokens:
            raise ValueError(
                "proposal length exceeds capability limit"
            )
        input_row = input_by_id.get(sequence_id)
        if (
            input_row is not None
            and len(token_ids)
            > input_row.max_proposal_tokens
        ):
            raise ValueError(
                "proposal length exceeds input limit"
            )
        _validate_timing(proposal.timing_ms)
        proposals_by_id[sequence_id] = proposal
    if set(proposals_by_id) != set(input_by_id):
        raise ValueError(
            "proposal sequence IDs must exactly match inputs"
        )
    return tuple(
        proposals_by_id[input_row.sequence_id]
        for input_row in inputs
    )


def _validate_prefill_observations(
    rows: object,
) -> tuple[TargetPrefillObservation, ...]:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError(
            "target prefill observations must be a non-empty tuple"
        )
    sequence_ids = []
    for row in rows:
        if not isinstance(row, TargetPrefillObservation):
            raise ValueError(
                "target prefill row must be TargetPrefillObservation"
            )
        sequence_id = _validate_integer(
            row.sequence_id,
            "target prefill sequence ID",
        )
        if sequence_id < 0:
            raise ValueError(
                "target prefill sequence ID must be non-negative"
            )
        sequence_epoch = _validate_integer(
            row.sequence_epoch,
            "target prefill sequence epoch",
        )
        if sequence_epoch < 0:
            raise ValueError(
                "target prefill sequence epoch must be non-negative"
            )
        _validate_token_ids(
            row.token_ids,
            "target prefill",
        )
        if row.positions is None:
            raise ValueError(
                "target prefill positions must be provided"
            )
        if row.target_hidden is None:
            raise ValueError(
                "target prefill hidden payload must be provided"
            )
        if not isinstance(row.is_final_chunk, bool):
            raise ValueError(
                "target prefill final-chunk flag must be bool"
            )
        sequence_ids.append(sequence_id)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "target prefill sequence IDs must be unique"
        )
    return rows


def _validate_finalize_rows(
    rows: object,
) -> tuple[ProposalFinalizeRow, ...]:
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
        sequence_id = _validate_integer(
            row.sequence_id,
            "proposal finalize sequence ID",
        )
        if sequence_id < 0:
            raise ValueError(
                "proposal finalize sequence ID must be non-negative"
            )
        transaction_id = row.proposal_transaction_id
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
        ):
            raise ValueError(
                "proposal finalize transaction ID must be non-empty"
            )
        accepted_tokens = _validate_integer(
            row.accepted_proposal_tokens,
            "proposal finalize accepted token count",
        )
        if accepted_tokens < 0:
            raise ValueError(
                "proposal finalize accepted token count must be "
                "non-negative"
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
    return rows


def _validate_ticket_id(ticket_id: object) -> str:
    if not isinstance(ticket_id, str) or not ticket_id:
        raise ValueError(
            "proposal finalize ticket must be a non-empty string"
        )
    return ticket_id


def _is_tensor(value: object) -> bool:
    value_type = type(value)
    return (
        value_type.__name__ == "Tensor"
        and value_type.__module__.startswith("torch")
    )


def assert_tensor_free(
    value: object,
    *,
    name: str,
) -> None:
    pending = [value]
    visited = set()
    while pending:
        current = pending.pop()
        if _is_tensor(current):
            raise ValueError(f"{name} must not contain a tensor")
        if isinstance(
            current,
            (str, bytes, int, float, bool, type(None)),
        ):
            continue
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if is_dataclass(current) and not isinstance(
            current,
            type,
        ):
            pending.extend(
                getattr(current, field.name)
                for field in fields(current)
            )
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (tuple, list, set, frozenset)):
            pending.extend(current)


class ModelRunnerProposalExecutorRegistry:
    def __init__(self):
        self._entries: dict[
            str,
            tuple[ProposalExecutor, DraftCapabilities],
        ] = {}

    def capabilities_for(
        self,
        executor_id: str,
    ) -> DraftCapabilities:
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError(
                "proposal executor ID must be a non-empty string"
            )
        entry = self._entries.get(executor_id)
        if entry is None:
            raise ValueError(
                "proposal executor is not registered"
            )
        return entry[1]

    def lifecycle_executor_ids(self) -> tuple[str, ...]:
        return tuple(
            executor_id
            for executor_id, (_, capabilities)
            in self._entries.items()
            if capabilities.requires_proposal_lifecycle
        )

    def _resolve_entry(
        self,
        executor_id: str,
        capabilities: DraftCapabilities,
        *,
        require_lifecycle: bool,
    ) -> tuple[ProposalExecutor, DraftCapabilities]:
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError(
                "proposal executor ID must be a non-empty string"
            )
        registered_capabilities = self.capabilities_for(
            executor_id
        )
        entry = self._entries[executor_id]
        executor, registered = entry
        if registered != registered_capabilities:
            raise RuntimeError(
                "proposal executor registry entry is inconsistent"
            )
        requested = validate_draft_capabilities(
            capabilities,
            expected_execution_domain="model_runner",
        )
        if requested != registered:
            raise ValueError(
                "proposal executor capabilities mismatch"
            )
        if require_lifecycle and not (
            registered.requires_proposal_lifecycle
        ):
            raise ValueError(
                "proposal executor lifecycle is not enabled"
            )
        return executor, requested

    def register(
        self,
        executor_id: str,
        executor: ProposalExecutor,
        capabilities: DraftCapabilities,
    ) -> None:
        normalized = validate_draft_capabilities(
            capabilities,
            expected_execution_domain="model_runner",
        )
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError(
                "proposal executor ID must be a non-empty string"
            )
        if executor_id in self._entries:
            raise ValueError(
                "proposal executor ID is already registered"
            )
        if getattr(executor, "capabilities", None) != normalized:
            raise ValueError(
                "proposal executor capabilities must exactly match"
            )
        if not callable(getattr(executor, "propose_batch", None)):
            raise ValueError(
                "proposal executor must expose callable propose_batch"
            )
        if normalized.requires_proposal_lifecycle:
            for method_name in (
                "observe_target_prefill",
                "prepare_finalize_batch",
                "commit_finalize_batch",
                "rollback_finalize_batch",
            ):
                if not callable(getattr(executor, method_name, None)):
                    raise ValueError(
                        "lifecycle proposal executor must expose "
                        f"callable {method_name}"
                    )
        self._entries[executor_id] = (
            executor,
            normalized,
        )

    def execute_batch(
        self,
        executor_id: str,
        inputs: tuple[ModelRunnerProposalInput, ...],
        capabilities: DraftCapabilities,
    ) -> tuple[DraftProposal, ...]:
        executor, requested = self._resolve_entry(
            executor_id,
            capabilities,
            require_lifecycle=False,
        )
        _validate_inputs(inputs, requested)
        proposals = executor.propose_batch(inputs)
        normalized = _validate_proposals(
            proposals,
            inputs,
            requested,
        )
        assert_tensor_free(
            normalized,
            name="proposal result",
        )
        return normalized

    def observe_target_prefill(
        self,
        executor_id: str,
        rows: tuple[TargetPrefillObservation, ...],
        capabilities: DraftCapabilities,
    ) -> None:
        executor, _ = self._resolve_entry(
            executor_id,
            capabilities,
            require_lifecycle=True,
        )
        normalized = _validate_prefill_observations(rows)
        executor.observe_target_prefill(normalized)

    def prepare_finalize_batch(
        self,
        executor_id: str,
        rows: tuple[ProposalFinalizeRow, ...],
        capabilities: DraftCapabilities,
    ) -> str:
        executor, _ = self._resolve_entry(
            executor_id,
            capabilities,
            require_lifecycle=True,
        )
        normalized = _validate_finalize_rows(rows)
        ticket_id = executor.prepare_finalize_batch(normalized)
        return _validate_ticket_id(ticket_id)

    def commit_finalize_batch(
        self,
        executor_id: str,
        ticket_id: str,
        capabilities: DraftCapabilities,
    ) -> None:
        executor, _ = self._resolve_entry(
            executor_id,
            capabilities,
            require_lifecycle=True,
        )
        executor.commit_finalize_batch(
            _validate_ticket_id(ticket_id)
        )

    def rollback_finalize_batch(
        self,
        executor_id: str,
        ticket_id: str,
        capabilities: DraftCapabilities,
    ) -> None:
        executor, _ = self._resolve_entry(
            executor_id,
            capabilities,
            require_lifecycle=True,
        )
        executor.rollback_finalize_batch(
            _validate_ticket_id(ticket_id)
        )
