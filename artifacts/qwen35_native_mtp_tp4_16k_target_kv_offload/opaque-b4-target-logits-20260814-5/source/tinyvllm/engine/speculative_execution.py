from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.speculative_selection import (
    SpeculativeSelectionRecord,
    validate_speculative_selection_record,
)
from tinyvllm.speculative.adapter import DraftProposal
from tinyvllm.speculative.batch_runtime import (
    NativeSpeculativeBatchResult,
    NativeSpeculativeSequenceResult,
    PreparedNativeSpeculativeBatch,
    TailBatchItem,
)
from tinyvllm.engine.speculative_residency import (
    SpeculativeResidencyPrecommitRow,
    SpeculativeResidencyPrepareRow,
)


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_non_negative_integer(
    value: object,
    name: str,
) -> int:
    normalized = _validate_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


@dataclass(frozen=True)
class EngineSpeculativePartition:
    schedule_generation: int
    scheduled_sequence_ids: tuple[int, ...]
    selected_sequence_ids: tuple[int, ...]
    suppressed_sequence_ids: tuple[int, ...]
    selected_sequences: tuple[object, ...]
    suppressed_sequences: tuple[object, ...]


@dataclass(frozen=True)
class EngineSpeculativeCommitRow:
    sequence_id: int
    output_tokens: tuple[int, ...]
    accepted_draft_tokens: tuple[int, ...]
    fallback_target_token: int | None
    finished_by_eos: bool
    finished_by_output_budget: bool


def build_speculative_residency_prepare_rows(
    items: tuple[TailBatchItem, ...],
) -> tuple[SpeculativeResidencyPrepareRow, ...]:
    if not isinstance(items, tuple) or not items:
        raise ValueError(
            "residency tail items must be a non-empty tuple"
        )
    if any(
        not isinstance(item, TailBatchItem)
        for item in items
    ):
        raise ValueError(
            "residency tail items must contain TailBatchItem"
        )
    return tuple(
        SpeculativeResidencyPrepareRow(
            sequence_id=item.sequence_id,
            original_block_identities=(
                item.original_block_identities
            ),
            reserved_block_identities=(
                item.reserved_block_identities
            ),
            proxy_block_table=item.proxy_block_table,
            logical_slots=item.plan.logical_slots,
        )
        for item in items
    )


def build_speculative_residency_precommit_rows(
    plans: tuple[object, ...],
) -> tuple[SpeculativeResidencyPrecommitRow, ...]:
    if not isinstance(plans, tuple):
        raise ValueError(
            "speculative KV commit plans must be a tuple"
        )
    rows = []
    sequence_ids = []
    for plan in plans:
        sequence_id = _validate_non_negative_integer(
            getattr(plan, "sequence_id", None),
            "speculative KV commit sequence id",
        )
        transaction = getattr(plan, "transaction", None)
        reserved_block_ids = getattr(
            transaction,
            "reserved_block_ids",
            None,
        )
        reserved_generations = getattr(
            transaction,
            "reserved_block_generations",
            None,
        )
        if (
            not isinstance(reserved_block_ids, tuple)
            or not isinstance(reserved_generations, tuple)
            or len(reserved_block_ids)
            != len(reserved_generations)
        ):
            raise ValueError(
                "reserved block identity snapshot is invalid"
            )
        identities = tuple(
            (
                _validate_non_negative_integer(
                    block_id,
                    "reserved block id",
                ),
                _validate_non_negative_integer(
                    generation,
                    "reserved block generation",
                ),
            )
            for block_id, generation in zip(
                reserved_block_ids,
                reserved_generations,
            )
        )
        identity_by_block = {
            block_id: (block_id, generation)
            for block_id, generation in identities
        }
        if len(identity_by_block) != len(identities):
            raise ValueError(
                "reserved block ids must be unique"
            )
        committed_block_ids = getattr(
            plan,
            "committed_block_ids",
            None,
        )
        unused_block_ids = getattr(
            plan,
            "unused_block_ids",
            None,
        )
        if (
            not isinstance(committed_block_ids, tuple)
            or not isinstance(unused_block_ids, tuple)
            or committed_block_ids + unused_block_ids
            != reserved_block_ids
        ):
            raise ValueError(
                "commit plan must exactly partition reserved blocks"
            )
        materialized_end = _validate_non_negative_integer(
            getattr(plan, "materialized_end", None),
            "accepted materialized end",
        )
        sequence_ids.append(sequence_id)
        rows.append(
            SpeculativeResidencyPrecommitRow(
                sequence_id=sequence_id,
                committed_block_identities=tuple(
                    identity_by_block[block_id]
                    for block_id in committed_block_ids
                ),
                rejected_block_identities=tuple(
                    identity_by_block[block_id]
                    for block_id in unused_block_ids
                ),
                accepted_materialized_end=materialized_end,
            )
        )
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "speculative KV commit sequence ids must be unique"
        )
    return tuple(rows)


def build_engine_prepared_speculative_commit_rows(
    prepared: PreparedNativeSpeculativeBatch,
    seqs: tuple[object, ...],
    *,
    eos_token: int,
) -> tuple[EngineSpeculativeCommitRow, ...]:
    if not isinstance(
        prepared,
        PreparedNativeSpeculativeBatch,
    ):
        raise ValueError(
            "prepared must be PreparedNativeSpeculativeBatch"
        )
    if prepared.state != "prepared":
        raise RuntimeError(
            "prepared speculative batch is not active: "
            f"{prepared.state}"
        )
    if not isinstance(prepared.sequences, tuple):
        raise ValueError(
            "prepared sequences must be a tuple"
        )
    result_rows = []
    for row in prepared.sequences:
        if getattr(row.sequence, "seq_id", None) != row.sequence_id:
            raise ValueError(
                "prepared row sequence identity mismatch"
            )
        result_rows.append(
            NativeSpeculativeSequenceResult(
                sequence_id=row.sequence_id,
                first_target_token=row.first_target_token,
                proposal=row.proposal,
                plan=row.plan,
                target_tokens=row.target_tokens,
                greedy_accepted_count=(
                    row.greedy_accepted_count
                ),
                accepted_tokens=row.accepted_tokens,
                eos_truncated=row.eos_truncated,
                output_budget_truncated=(
                    row.output_budget_truncated
                ),
                reserved_blocks=row.reserved_blocks,
                proxy_block_table=row.proxy_block_table,
                committed_blocks=(),
                released_blocks=(),
                first_target_metadata=(
                    row.first_target_metadata
                ),
                tail_metadata=row.tail_metadata,
                tail_auxiliary=row.tail_auxiliary,
            )
        )
    return build_engine_speculative_commit_rows(
        NativeSpeculativeBatchResult(
            sequences=tuple(result_rows),
            first_target_callback_count=(
                prepared.first_target_callback_count
            ),
            tail_callback_count=prepared.tail_callback_count,
            timing_ms=dict(prepared.timing_ms),
        ),
        seqs,
        eos_token=eos_token,
    )


def build_engine_speculative_partition(
    record: SpeculativeSelectionRecord,
    seqs: tuple[object, ...],
    *,
    expected_schedule_generation: int,
) -> EngineSpeculativePartition:
    selected_sequences = validate_speculative_selection_record(
        record,
        seqs,
        expected_schedule_generation=expected_schedule_generation,
    )
    selected_sequence_ids = tuple(
        _validate_integer(
            getattr(seq, "seq_id", None),
            "selected sequence id",
        )
        for seq in selected_sequences
    )
    selected_id_set = set(selected_sequence_ids)
    suppressed_sequences = tuple(
        seq
        for seq in seqs
        if getattr(seq, "seq_id", None) not in selected_id_set
    )
    suppressed_sequence_ids = tuple(
        _validate_integer(
            getattr(seq, "seq_id", None),
            "suppressed sequence id",
        )
        for seq in suppressed_sequences
    )
    scheduled_id_set = set(record.scheduled_sequence_ids)
    suppressed_id_set = set(suppressed_sequence_ids)
    if selected_id_set & suppressed_id_set:
        raise ValueError(
            "speculative selection partition must be disjoint"
        )
    if (
        selected_id_set | suppressed_id_set
        != scheduled_id_set
    ):
        raise ValueError(
            "speculative selection partition must exactly cover schedule"
        )
    return EngineSpeculativePartition(
        schedule_generation=record.schedule_generation,
        scheduled_sequence_ids=record.scheduled_sequence_ids,
        selected_sequence_ids=selected_sequence_ids,
        suppressed_sequence_ids=suppressed_sequence_ids,
        selected_sequences=selected_sequences,
        suppressed_sequences=suppressed_sequences,
    )


def _validate_token_tuple(
    value: object,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    for token_id in value:
        _validate_integer(token_id, f"{name} token")
    return value


def _validate_runtime_row(
    row: NativeSpeculativeSequenceResult,
) -> tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
]:
    if not isinstance(row, NativeSpeculativeSequenceResult):
        raise ValueError(
            "runtime row must be NativeSpeculativeSequenceResult"
        )
    sequence_id = _validate_integer(
        row.sequence_id,
        "runtime sequence id",
    )
    if not isinstance(row.proposal, DraftProposal):
        raise ValueError("runtime proposal must be DraftProposal")
    if row.proposal.sequence_id != sequence_id:
        raise ValueError(
            "runtime proposal sequence ID mismatch"
        )
    proposal_tokens = _validate_token_tuple(
        row.proposal.token_ids,
        "proposal tokens",
    )
    target_tokens = _validate_token_tuple(
        row.target_tokens,
        "target tokens",
    )
    accepted_tokens = _validate_token_tuple(
        row.accepted_tokens,
        "accepted tokens",
    )
    first_target_token = _validate_integer(
        row.first_target_token,
        "first target token",
    )
    greedy_count = _validate_non_negative_integer(
        row.greedy_accepted_count,
        "greedy accepted count",
    )
    if greedy_count > len(proposal_tokens):
        raise ValueError(
            "greedy accepted count exceeds proposal length"
        )
    if len(accepted_tokens) > greedy_count:
        raise ValueError(
            "accepted token count exceeds greedy accepted count"
        )
    if not isinstance(row.eos_truncated, bool):
        raise ValueError("eos_truncated must be a bool")
    if not isinstance(row.output_budget_truncated, bool):
        raise ValueError(
            "output_budget_truncated must be a bool"
        )
    if (
        len(accepted_tokens) < greedy_count
        and not (
            row.eos_truncated
            or row.output_budget_truncated
        )
    ):
        raise ValueError(
            "accepted token count requires an explicit truncation"
        )
    if accepted_tokens != proposal_tokens[:len(accepted_tokens)]:
        raise ValueError(
            "accepted tokens must match proposal prefix"
        )
    if greedy_count and (
        proposal_tokens[:greedy_count]
        != target_tokens[:greedy_count]
    ):
        raise ValueError(
            "greedy accepted prefix must match target tokens"
        )
    if not target_tokens:
        raise ValueError("target tokens must not be empty")
    if target_tokens[0] != first_target_token:
        raise ValueError(
            "first target token must match target token prefix"
        )
    if greedy_count < len(proposal_tokens) and (
        len(target_tokens) <= greedy_count
    ):
        raise ValueError(
            "target tokens missing partial-acceptance fallback"
        )
    return (
        sequence_id,
        proposal_tokens,
        target_tokens,
        accepted_tokens,
        greedy_count,
    )


def build_engine_speculative_commit_rows(
    result: NativeSpeculativeBatchResult,
    seqs: tuple[object, ...],
    *,
    eos_token: int,
) -> tuple[EngineSpeculativeCommitRow, ...]:
    if not isinstance(result, NativeSpeculativeBatchResult):
        raise ValueError(
            "result must be NativeSpeculativeBatchResult"
        )
    if not isinstance(seqs, tuple):
        raise ValueError("sequences must be a tuple")
    normalized_eos_token = _validate_integer(
        eos_token,
        "eos token",
    )
    sequence_ids = tuple(
        _validate_integer(
            getattr(seq, "seq_id", None),
            "sequence id",
        )
        for seq in seqs
    )
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence IDs must be unique")
    if not isinstance(result.sequences, tuple):
        raise ValueError("runtime sequences must be a tuple")
    result_sequence_ids = tuple(
        _validate_integer(
            getattr(row, "sequence_id", None),
            "runtime sequence id",
        )
        for row in result.sequences
    )
    if len(set(result_sequence_ids)) != len(result_sequence_ids):
        raise ValueError("runtime sequence IDs must be unique")
    if result_sequence_ids != sequence_ids:
        raise ValueError(
            "runtime result sequence order must exactly match sequences"
        )

    commit_rows = []
    for row, seq in zip(result.sequences, seqs):
        (
            sequence_id,
            proposal_tokens,
            target_tokens,
            accepted_tokens,
            greedy_count,
        ) = _validate_runtime_row(row)
        completion_tokens = _validate_non_negative_integer(
            getattr(seq, "num_completion_tokens", None),
            "completion token count",
        )
        max_tokens = _validate_non_negative_integer(
            getattr(seq, "max_tokens", None),
            "max tokens",
        )
        ignore_eos = getattr(seq, "ignore_eos", None)
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be a bool")
        remaining_output_tokens = max(
            0,
            max_tokens - completion_tokens,
        )
        if len(accepted_tokens) > remaining_output_tokens:
            raise ValueError(
                "accepted tokens exceed remaining output budget"
            )

        output_tokens = list(accepted_tokens)
        accepted_eos = (
            not ignore_eos
            and bool(accepted_tokens)
            and accepted_tokens[-1] == normalized_eos_token
        )

        fallback_target_token = None
        has_fallback = (
            not proposal_tokens
            or greedy_count < len(proposal_tokens)
        )
        if (
            has_fallback
            and not accepted_eos
            and len(output_tokens) < remaining_output_tokens
        ):
            fallback_target_token = (
                row.first_target_token
                if not proposal_tokens
                else target_tokens[greedy_count]
            )
            output_tokens.append(fallback_target_token)

        output_token_tuple = tuple(output_tokens)
        finished_by_eos = (
            not ignore_eos
            and bool(output_token_tuple)
            and output_token_tuple[-1] == normalized_eos_token
        )
        finished_by_output_budget = (
            len(output_token_tuple)
            >= remaining_output_tokens
        )
        commit_rows.append(
            EngineSpeculativeCommitRow(
                sequence_id=sequence_id,
                output_tokens=output_token_tuple,
                accepted_draft_tokens=accepted_tokens,
                fallback_target_token=fallback_target_token,
                finished_by_eos=finished_by_eos,
                finished_by_output_budget=(
                    finished_by_output_budget
                ),
            )
        )
    return tuple(commit_rows)
