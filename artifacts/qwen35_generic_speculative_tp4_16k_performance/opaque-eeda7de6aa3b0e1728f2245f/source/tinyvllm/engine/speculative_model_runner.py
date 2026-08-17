from __future__ import annotations

from dataclasses import dataclass
import math

from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftContext,
    DraftProposal,
    validate_draft_adapter_batch,
    validate_draft_capabilities,
)
from tinyvllm.speculative.batch_runtime import (
    FirstTargetProposalResult,
    FirstTargetResult,
    PreparedProposalFinalizeRow,
    TailBatchItem,
    TailBatchResult,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ProposalFinalizeRow,
    assert_tensor_free,
)
from tinyvllm.speculative.verifier import (
    SpecVerifyBatchResultRow,
)
from tinyvllm.engine.speculative_residency import (
    KVBlockIdentityRow,
)
from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateCallbacks,
)


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_model_runner(model_runner) -> None:
    if not callable(getattr(model_runner, "call", None)):
        raise ValueError(
            "model_runner must expose callable call"
        )


def build_model_runner_side_state_callbacks(
    model_runner,
    *,
    dispatch=None,
):
    _validate_model_runner(model_runner)
    available = getattr(
        model_runner,
        "speculative_side_state_available",
        None,
    )
    if not callable(available) or not available():
        return None
    if dispatch is None:
        dispatch = model_runner.call
    if not callable(dispatch):
        raise ValueError("dispatch must be callable")

    def prepare(seqs):
        assert_tensor_free(
            seqs,
            name="speculative side-state sequences",
        )
        return dispatch(
            "prepare_speculative_side_state_batch",
            seqs,
        )

    def select(_handle, rows):
        assert_tensor_free(
            rows,
            name="speculative side-state selection rows",
        )
        if not isinstance(rows, tuple):
            raise ValueError(
                "speculative side-state selection rows must be a tuple"
            )
        return dispatch(
            "select_speculative_side_state_batch",
            rows,
        )

    return SpeculativeSideStateCallbacks(
        prepare=prepare,
        select=select,
        apply=lambda _handle: dispatch(
            "apply_speculative_side_state_batch",
        ),
        seal=lambda _handle: dispatch(
            "seal_speculative_side_state_batch",
        ),
        rollback=lambda _handle: dispatch(
            "rollback_speculative_side_state_batch",
        ),
    )


def _validate_sequence_ids(
    values: tuple[object, ...],
    *,
    attribute: str,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    sequence_ids = tuple(
        _validate_integer(
            getattr(value, attribute, None),
            f"{name} sequence ID",
        )
        for value in values
    )
    if any(sequence_id < 0 for sequence_id in sequence_ids):
        raise ValueError(
            f"{name} sequence IDs must be non-negative"
        )
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            f"{name} sequence IDs must be unique"
        )
    return sequence_ids


@dataclass(frozen=True)
class FixedQTailBatch:
    query_len: int
    items: tuple[TailBatchItem, ...]


def build_fixed_q_tail_batches(
    items: tuple[TailBatchItem, ...],
) -> tuple[FixedQTailBatch, ...]:
    sequence_ids = _validate_sequence_ids(
        items,
        attribute="sequence_id",
        name="tail items",
    )
    groups_by_query_len: dict[
        int,
        list[TailBatchItem],
    ] = {}
    query_len_order = []
    for sequence_id, item in zip(sequence_ids, items):
        if not isinstance(item, TailBatchItem):
            raise ValueError(
                "tail items must contain TailBatchItem"
            )
        query_len = getattr(item.plan, "query_len", None)
        normalized_query_len = _validate_integer(
            query_len,
            "tail item query length",
        )
        if normalized_query_len <= 0:
            raise ValueError(
                "tail item query length must be > 0"
            )
        if item.sequence_id != sequence_id:
            raise ValueError(
                "tail item sequence ID mismatch"
            )
        if normalized_query_len not in groups_by_query_len:
            groups_by_query_len[
                normalized_query_len
            ] = []
            query_len_order.append(normalized_query_len)
        groups_by_query_len[
            normalized_query_len
        ].append(item)
    return tuple(
        FixedQTailBatch(
            query_len=query_len,
            items=tuple(groups_by_query_len[query_len]),
        )
        for query_len in query_len_order
    )


def _validate_capabilities(
    capabilities: DraftCapabilities,
) -> None:
    validate_draft_capabilities(capabilities)


def _validate_lifecycle_descriptor(descriptor):
    capabilities = validate_draft_capabilities(
        getattr(descriptor, "capabilities", None),
        expected_execution_domain="model_runner",
    )
    if not capabilities.requires_proposal_lifecycle:
        raise ValueError(
            "proposal executor lifecycle is not enabled"
        )
    executor_id = getattr(descriptor, "executor_id", None)
    if not isinstance(executor_id, str) or not executor_id:
        raise ValueError(
            "proposal executor ID must be non-empty"
        )
    return executor_id


def _normalize_proposal_finalize_rows(
    rows: object,
) -> tuple[ProposalFinalizeRow, ...]:
    assert_tensor_free(
        rows,
        name="proposal finalize rows",
    )
    if not isinstance(rows, tuple) or not rows:
        raise ValueError(
            "proposal finalize rows must be a non-empty tuple"
        )
    normalized = []
    sequence_ids = []
    transaction_ids = []
    for row in rows:
        if not isinstance(row, PreparedProposalFinalizeRow):
            raise ValueError(
                "proposal finalize rows must contain "
                "PreparedProposalFinalizeRow"
            )
        sequence_id = _validate_integer(
            row.sequence_id,
            "proposal finalize sequence ID",
        )
        if sequence_id < 0:
            raise ValueError(
                "proposal finalize sequence IDs must be non-negative"
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
        normalized.append(
            ProposalFinalizeRow(
                sequence_id=sequence_id,
                proposal_transaction_id=transaction_id,
                accepted_proposal_tokens=accepted_tokens,
            )
        )
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "proposal finalize sequence IDs must be unique"
        )
    if len(set(transaction_ids)) != len(transaction_ids):
        raise ValueError(
            "proposal finalize transaction IDs must be unique"
        )
    return tuple(normalized)


def _validate_finalize_ticket(ticket_id: object) -> str:
    assert_tensor_free(
        ticket_id,
        name="proposal finalize ticket",
    )
    if not isinstance(ticket_id, str) or not ticket_id:
        raise ValueError(
            "proposal finalize ticket must be a non-empty string"
        )
    return ticket_id


def prepare_model_runner_proposal_finalize_batch(
    model_runner,
    descriptor,
    rows: tuple[PreparedProposalFinalizeRow, ...],
) -> str:
    _validate_model_runner(model_runner)
    executor_id = _validate_lifecycle_descriptor(descriptor)
    normalized = _normalize_proposal_finalize_rows(rows)
    ticket_id = model_runner.call(
        "prepare_speculative_proposal_finalize_batch",
        executor_id,
        normalized,
    )
    return _validate_finalize_ticket(ticket_id)


def _complete_model_runner_proposal_finalize_batch(
    model_runner,
    descriptor,
    ticket_id: str,
    *,
    operation: str,
) -> None:
    _validate_model_runner(model_runner)
    executor_id = _validate_lifecycle_descriptor(descriptor)
    normalized_ticket = _validate_finalize_ticket(ticket_id)
    acknowledgement = model_runner.call(
        operation,
        executor_id,
        normalized_ticket,
    )
    assert_tensor_free(
        acknowledgement,
        name="proposal finalize acknowledgement",
    )
    if acknowledgement is not None:
        raise ValueError(
            "proposal finalize acknowledgement must be None"
        )


def commit_model_runner_proposal_finalize_batch(
    model_runner,
    descriptor,
    ticket_id: str,
) -> None:
    _complete_model_runner_proposal_finalize_batch(
        model_runner,
        descriptor,
        ticket_id,
        operation=(
            "commit_speculative_proposal_finalize_batch"
        ),
    )


def rollback_model_runner_proposal_finalize_batch(
    model_runner,
    descriptor,
    ticket_id: str,
) -> None:
    _complete_model_runner_proposal_finalize_batch(
        model_runner,
        descriptor,
        ticket_id,
        operation=(
            "rollback_speculative_proposal_finalize_batch"
        ),
    )


def run_model_runner_first_targets(
    model_runner,
    seqs: tuple[object, ...],
    capabilities: DraftCapabilities,
    kv_block_identity_rows: tuple[
        KVBlockIdentityRow,
        ...,
    ] = (),
) -> tuple[FirstTargetResult, ...]:
    _validate_model_runner(model_runner)
    sequence_ids = _validate_sequence_ids(
        seqs,
        attribute="seq_id",
        name="first-target sequences",
    )
    _validate_capabilities(capabilities)
    results = model_runner.call(
        "run_spec_first_target_batch",
        seqs,
        capabilities.requires_target_hidden,
        capabilities.requires_target_logits,
        kv_block_identity_rows,
    )
    if not isinstance(results, tuple):
        raise ValueError(
            "first-target ModelRunner result must be a tuple"
        )
    rows = {}
    for result in results:
        if not isinstance(result, FirstTargetResult):
            raise ValueError(
                "first-target ModelRunner rows must be "
                "FirstTargetResult"
            )
        sequence_id = _validate_integer(
            result.sequence_id,
            "first-target result sequence ID",
        )
        if sequence_id in rows:
            raise ValueError(
                "first-target result sequence IDs must be unique"
            )
        rows[sequence_id] = result
    if set(rows) != set(sequence_ids):
        raise ValueError(
            "first-target result sequence IDs must exactly match "
            "input sequences"
        )
    return tuple(rows[sequence_id] for sequence_id in sequence_ids)


def _validate_proposal_timing(
    timing_ms: object,
) -> None:
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


def _validate_first_target_proposal_results(
    results: object,
    sequence_ids: tuple[int, ...],
    capabilities: DraftCapabilities,
) -> tuple[FirstTargetProposalResult, ...]:
    assert_tensor_free(
        results,
        name="fused proposal result",
    )
    if not isinstance(results, tuple):
        raise ValueError(
            "fused proposal result must be a tuple"
        )
    rows = {}
    for result in results:
        if not isinstance(
            result,
            FirstTargetProposalResult,
        ):
            raise ValueError(
                "fused proposal rows must be "
                "FirstTargetProposalResult"
            )
        sequence_id = _validate_integer(
            result.sequence_id,
            "fused proposal sequence ID",
        )
        if sequence_id in rows:
            raise ValueError(
                "fused proposal sequence IDs must be unique"
            )
        _validate_integer(
            result.target_token,
            "fused first target token",
        )
        proposal = result.proposal
        if not isinstance(proposal, DraftProposal):
            raise ValueError(
                "fused proposal row must contain DraftProposal"
            )
        if proposal.sequence_id != sequence_id:
            raise ValueError(
                "proposal sequence ID must match fused row"
            )
        if proposal.source_type != capabilities.source_type:
            raise ValueError(
                "proposal source_type must match capabilities"
            )
        if not isinstance(proposal.token_ids, tuple):
            raise ValueError(
                "proposal token_ids must be a tuple"
            )
        for token_id in proposal.token_ids:
            _validate_integer(
                token_id,
                "proposal token",
            )
        if (
            len(proposal.token_ids)
            > capabilities.max_proposal_tokens
        ):
            raise ValueError(
                "proposal length exceeds capability limit"
            )
        _validate_proposal_timing(proposal.timing_ms)
        rows[sequence_id] = result
    if set(rows) != set(sequence_ids):
        raise ValueError(
            "fused proposal sequence IDs must exactly match input"
        )
    return tuple(
        rows[sequence_id]
        for sequence_id in sequence_ids
    )


def run_host_first_targets_and_proposals(
    model_runner,
    seqs: tuple[object, ...],
    draft_adapter,
    kv_block_identity_rows: tuple[
        KVBlockIdentityRow,
        ...,
    ] = (),
) -> tuple[FirstTargetProposalResult, ...]:
    capabilities = validate_draft_capabilities(
        getattr(draft_adapter, "capabilities", None),
        expected_execution_domain="host",
    )
    sequence_ids = _validate_sequence_ids(
        seqs,
        attribute="seq_id",
        name="host proposal sequences",
    )
    first_targets = run_model_runner_first_targets(
        model_runner,
        seqs,
        capabilities,
        kv_block_identity_rows,
    )
    contexts = tuple(
        DraftContext(
            sequence_id=sequence_id,
            token_ids=tuple(
                int(token_id)
                for token_id in seq.token_ids
            ),
            remaining_output_tokens=max(
                0,
                int(seq.max_tokens)
                - int(seq.num_completion_tokens),
            ),
            max_proposal_tokens=(
                capabilities.max_proposal_tokens
            ),
            first_target_token=first_target.target_token,
            target_hidden=first_target.target_hidden,
            target_logits=first_target.target_logits,
        )
        for sequence_id, seq, first_target
        in zip(sequence_ids, seqs, first_targets)
    )
    proposals = validate_draft_adapter_batch(
        draft_adapter,
        contexts,
    )
    return tuple(
        FirstTargetProposalResult(
            sequence_id=sequence_id,
            target_token=first_target.target_token,
            proposal=proposal,
            first_target_metadata=first_target.metadata,
            proposal_metadata=proposal.metadata,
        )
        for sequence_id, first_target, proposal
        in zip(sequence_ids, first_targets, proposals)
    )


def run_model_runner_first_targets_and_proposals(
    model_runner,
    seqs: tuple[object, ...],
    descriptor,
    kv_block_identity_rows: tuple[
        KVBlockIdentityRow,
        ...,
    ] = (),
) -> tuple[FirstTargetProposalResult, ...]:
    _validate_model_runner(model_runner)
    sequence_ids = _validate_sequence_ids(
        seqs,
        attribute="seq_id",
        name="fused proposal sequences",
    )
    capabilities = validate_draft_capabilities(
        getattr(descriptor, "capabilities", None),
        expected_execution_domain="model_runner",
    )
    executor_id = getattr(descriptor, "executor_id", None)
    if not isinstance(executor_id, str) or not executor_id:
        raise ValueError(
            "proposal executor ID must be non-empty"
        )
    results = model_runner.call(
        "run_spec_first_target_and_proposal_batch",
        seqs,
        descriptor,
        kv_block_identity_rows,
    )
    return _validate_first_target_proposal_results(
        results,
        sequence_ids,
        capabilities,
    )


def build_model_runner_proposal_provider(
    model_runner,
    runtime,
    kv_block_identity_rows_for,
):
    _validate_model_runner(model_runner)
    if not callable(kv_block_identity_rows_for):
        raise ValueError(
            "KV block identity builder must be callable"
        )
    capabilities = validate_draft_capabilities(
        getattr(runtime, "capabilities", None)
    )
    if capabilities.execution_domain == "host":
        draft_adapter = getattr(
            runtime,
            "draft_adapter",
            None,
        )
        if draft_adapter is None:
            raise ValueError(
                "host proposal runtime requires draft adapter"
            )

        def run_host(seqs):
            return run_host_first_targets_and_proposals(
                model_runner,
                seqs,
                draft_adapter,
                kv_block_identity_rows_for(seqs),
            )

        return run_host
    if capabilities.execution_domain == "model_runner":
        descriptor = getattr(
            runtime,
            "model_runner_executor",
            None,
        )
        if descriptor is None:
            raise ValueError(
                "ModelRunner proposal runtime requires descriptor"
            )

        def run_model_runner(seqs):
            return (
                run_model_runner_first_targets_and_proposals(
                    model_runner,
                    seqs,
                    descriptor,
                    kv_block_identity_rows_for(seqs),
                )
            )

        return run_model_runner
    raise ValueError(
        "unsupported proposal execution domain"
    )


def _validate_tail_group_results(
    results: object,
    group: FixedQTailBatch,
) -> dict[int, SpecVerifyBatchResultRow]:
    if not isinstance(results, tuple):
        raise ValueError(
            "tail ModelRunner result must be a tuple"
        )
    expected_ids = tuple(
        item.sequence_id for item in group.items
    )
    rows = {}
    for result in results:
        if not isinstance(
            result,
            SpecVerifyBatchResultRow,
        ):
            raise ValueError(
                "tail ModelRunner rows must be batch result row"
            )
        sequence_id = _validate_integer(
            result.sequence_id,
            "tail result sequence ID",
        )
        if sequence_id in rows:
            raise ValueError(
                "tail result sequence IDs must be unique"
            )
        if len(result.target_tokens) != group.query_len:
            raise ValueError(
                "tail result target count must match query length"
            )
        rows[sequence_id] = result
    if set(rows) != set(expected_ids):
        raise ValueError(
            "tail result sequence IDs must exactly match group"
        )
    return rows


def run_model_runner_tail_batch(
    model_runner,
    items: tuple[TailBatchItem, ...],
    residency_ticket_id: int | None = None,
) -> tuple[TailBatchResult, ...]:
    _validate_model_runner(model_runner)
    groups = build_fixed_q_tail_batches(items)
    result_by_sequence_id = {}
    group_count = len(groups)
    for group_index, group in enumerate(groups):
        call_args = (
            (group.items,)
            if residency_ticket_id is None
            else (group.items, residency_ticket_id)
        )
        rows = _validate_tail_group_results(
            model_runner.call(
                "run_spec_verify_batch",
                *call_args,
            ),
            group,
        )
        for item in group.items:
            row = rows[item.sequence_id]
            result_by_sequence_id[
                item.sequence_id
            ] = TailBatchResult(
                sequence_id=item.sequence_id,
                target_tokens=row.target_tokens,
                metadata={
                    "query_len": group.query_len,
                    "fixed_q_group_index": group_index,
                    "fixed_q_group_count": group_count,
                },
            )
    return tuple(
        result_by_sequence_id[item.sequence_id]
        for item in items
    )
