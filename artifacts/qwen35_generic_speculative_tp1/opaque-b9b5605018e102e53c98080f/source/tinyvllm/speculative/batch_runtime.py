from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, TYPE_CHECKING

from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateCallbacks,
    SpeculativeSideStateSelectionRow,
    build_speculative_side_state_selection_rows,
)
from tinyvllm.speculative.adapter import (
    DraftAdapter,
    DraftContext,
    DraftProposal,
    validate_draft_adapter_batch,
)
from tinyvllm.speculative.verifier import (
    SpecVerifyPlan,
    build_spec_verify_plan,
)

if TYPE_CHECKING:
    from tinyvllm.engine.block_manager import (
        SpeculativeKVTransactionAuthorization,
    )


@dataclass(frozen=True)
class FirstTargetResult:
    sequence_id: int
    target_token: int
    target_hidden: object | None = None
    target_logits: object | None = None
    metadata: object | None = None


@dataclass(frozen=True)
class FirstTargetProposalResult:
    sequence_id: int
    target_token: int
    proposal: DraftProposal
    first_target_metadata: object | None = None
    proposal_metadata: object | None = None


@dataclass(frozen=True)
class TailBatchItem:
    sequence_id: int
    plan: SpecVerifyPlan
    proxy_block_table: tuple[int, ...]
    original_block_identities: tuple[tuple[int, int], ...] = ()
    reserved_block_identities: tuple[tuple[int, int], ...] = ()
    transaction_authorization: (
        SpeculativeKVTransactionAuthorization | None
    ) = None


@dataclass(frozen=True)
class TailBatchResult:
    sequence_id: int
    target_tokens: tuple[int, ...]
    metadata: object | None = None
    auxiliary: object | None = None


@dataclass(frozen=True)
class NativeSpeculativeSequenceResult:
    sequence_id: int
    first_target_token: int
    proposal: DraftProposal
    plan: SpecVerifyPlan | None
    target_tokens: tuple[int, ...]
    greedy_accepted_count: int
    accepted_tokens: tuple[int, ...]
    eos_truncated: bool
    output_budget_truncated: bool
    reserved_blocks: tuple[int, ...]
    proxy_block_table: tuple[int, ...]
    committed_blocks: tuple[int, ...]
    released_blocks: tuple[int, ...]
    first_target_metadata: object | None = None
    tail_metadata: object | None = None
    tail_auxiliary: object | None = None


@dataclass(frozen=True)
class NativeSpeculativeBatchResult:
    sequences: tuple[NativeSpeculativeSequenceResult, ...]
    first_target_callback_count: int
    tail_callback_count: int
    timing_ms: dict[str, float]


@dataclass(frozen=True)
class PreparedNativeSpeculativeSequence:
    sequence_id: int
    sequence: object
    first_target_token: int
    proposal: DraftProposal
    plan: SpecVerifyPlan | None
    target_tokens: tuple[int, ...]
    greedy_accepted_count: int
    accepted_tokens: tuple[int, ...]
    eos_truncated: bool
    output_budget_truncated: bool
    transaction: object | None
    reserved_blocks: tuple[int, ...]
    proxy_block_table: tuple[int, ...]
    first_target_metadata: object | None = None
    tail_metadata: object | None = None
    tail_auxiliary: object | None = None


@dataclass(frozen=True)
class PreparedProposalFinalizeRow:
    sequence_id: int
    proposal_transaction_id: str
    accepted_proposal_tokens: int


@dataclass
class PreparedNativeSpeculativeBatch:
    sequences: tuple[PreparedNativeSpeculativeSequence, ...]
    first_target_callback_count: int
    tail_callback_count: int
    timing_ms: dict[str, float]
    state: str = "prepared"
    side_state_callbacks: SpeculativeSideStateCallbacks | None = None
    side_state_handle: object | None = None
    side_state_selection: tuple[
        SpeculativeSideStateSelectionRow,
        ...,
    ] = ()
    side_state_state: str = "disabled"


class NativeSpeculativeBatchError(RuntimeError):
    def __init__(
        self,
        phase: str,
        cause: Exception,
        *,
        committed_sequence_ids: tuple[int, ...] = (),
        rolled_back_sequence_ids: tuple[int, ...] = (),
        rollback_errors: dict[object, Exception] | None = None,
    ):
        self.phase = phase
        self.cause = cause
        self.committed_sequence_ids = committed_sequence_ids
        self.rolled_back_sequence_ids = (
            rolled_back_sequence_ids
        )
        self.rollback_errors = (
            {}
            if rollback_errors is None
            else dict(rollback_errors)
        )
        rollback_suffix = ""
        if self.rollback_errors:
            details = ", ".join(
                f"{sequence_id}: {error}"
                for sequence_id, error
                in self.rollback_errors.items()
            )
            rollback_suffix = (
                f"; rollback failures: {details}"
            )
        super().__init__(
            f"native speculative batch {phase} failed: "
            f"{cause}{rollback_suffix}"
        )


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_sequences(
    seqs: tuple[object, ...],
) -> tuple[int, ...]:
    if not isinstance(seqs, tuple) or not seqs:
        raise ValueError(
            "speculative batch seqs must be a non-empty tuple"
        )
    sequence_ids = []
    for seq in seqs:
        sequence_id = _validate_integer(
            getattr(seq, "seq_id", None),
            "sequence id",
        )
        if not hasattr(seq, "token_ids"):
            raise ValueError(
                "sequence must expose token_ids"
            )
        if not hasattr(seq, "block_table"):
            raise ValueError(
                "sequence must expose block_table"
            )
        if not hasattr(seq, "block_size"):
            raise ValueError(
                "sequence must expose block_size"
            )
        sequence_ids.append(sequence_id)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "speculative batch sequence IDs must be unique"
        )
    return tuple(sequence_ids)


def _validate_first_target_results(
    results: object,
    sequence_ids: tuple[int, ...],
) -> dict[int, FirstTargetResult]:
    if not isinstance(results, tuple):
        raise ValueError(
            "first-target callback must return a tuple"
        )
    rows = {}
    for result in results:
        if not isinstance(result, FirstTargetResult):
            raise ValueError(
                "first-target row must be FirstTargetResult"
            )
        sequence_id = _validate_integer(
            result.sequence_id,
            "first-target sequence id",
        )
        _validate_integer(
            result.target_token,
            "first-target token",
        )
        if sequence_id in rows:
            raise ValueError(
                "first-target sequence IDs must be unique"
            )
        rows[sequence_id] = result
    if set(rows) != set(sequence_ids):
        raise ValueError(
            "first-target sequence IDs must exactly match batch"
        )
    return rows


def _validate_first_target_proposal_results(
    results: object,
    sequence_ids: tuple[int, ...],
) -> dict[int, FirstTargetProposalResult]:
    if not isinstance(results, tuple):
        raise ValueError(
            "first-target proposal callback must return a tuple"
        )
    rows = {}
    for result in results:
        if not isinstance(
            result,
            FirstTargetProposalResult,
        ):
            raise ValueError(
                "first-target proposal row must be "
                "FirstTargetProposalResult"
            )
        sequence_id = _validate_integer(
            result.sequence_id,
            "first-target proposal sequence id",
        )
        _validate_integer(
            result.target_token,
            "first-target proposal token",
        )
        if sequence_id in rows:
            raise ValueError(
                "first-target proposal sequence IDs must be unique"
            )
        proposal = result.proposal
        if not isinstance(proposal, DraftProposal):
            raise ValueError(
                "first-target proposal row must contain DraftProposal"
            )
        if proposal.sequence_id != sequence_id:
            raise ValueError(
                "proposal sequence ID must match provider row"
            )
        if not isinstance(proposal.token_ids, tuple):
            raise ValueError(
                "provider proposal token_ids must be a tuple"
            )
        for token_id in proposal.token_ids:
            _validate_integer(
                token_id,
                "provider proposal token",
            )
        rows[sequence_id] = result
    if set(rows) != set(sequence_ids):
        raise ValueError(
            "first-target proposal sequence IDs must exactly match batch"
        )
    return rows


def _validate_tail_results(
    results: object,
    items: tuple[TailBatchItem, ...],
) -> dict[int, TailBatchResult]:
    if not isinstance(results, tuple):
        raise ValueError(
            "tail callback must return a tuple"
        )
    expected = {
        item.sequence_id: item
        for item in items
    }
    rows = {}
    for result in results:
        if not isinstance(result, TailBatchResult):
            raise ValueError(
                "tail row must be TailBatchResult"
            )
        sequence_id = _validate_integer(
            result.sequence_id,
            "tail sequence id",
        )
        if sequence_id in rows:
            raise ValueError(
                "tail sequence IDs must be unique"
            )
        if not isinstance(result.target_tokens, tuple):
            raise ValueError(
                "tail target_tokens must be a tuple"
            )
        for token_id in result.target_tokens:
            _validate_integer(token_id, "tail target token")
        item = expected.get(sequence_id)
        if item is not None and (
            len(result.target_tokens)
            != item.plan.query_len
        ):
            raise ValueError(
                "tail target count does not match verifier query length"
            )
        rows[sequence_id] = result
    if set(rows) != set(expected):
        raise ValueError(
            "tail sequence IDs must exactly match tail batch"
        )
    return rows


def _count_accepted_prefix(
    draft_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
) -> int:
    accepted = 0
    for draft_token, target_token in zip(
        draft_tokens,
        target_tokens,
    ):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def _rollback_active(
    block_manager,
    active_transactions: dict[int, tuple[object, object]],
) -> tuple[tuple[int, ...], dict[int, Exception]]:
    rolled_back = []
    rollback_errors = {}
    for sequence_id, (
        seq,
        transaction,
    ) in tuple(active_transactions.items()):
        try:
            block_manager.rollback_speculative_kv_transaction(
                transaction,
                seq,
            )
        except Exception as error:
            rollback_errors[sequence_id] = error
        else:
            rolled_back.append(sequence_id)
    return tuple(rolled_back), rollback_errors


def prepare_native_speculative_batch(
    *,
    block_manager,
    seqs: tuple[object, ...],
    eos_token: int,
    run_tail_batch: Callable[
        [tuple[TailBatchItem, ...]],
        tuple[TailBatchResult, ...],
    ],
    draft_adapter: DraftAdapter | None = None,
    run_first_targets: Callable[
        [tuple[object, ...]],
        tuple[FirstTargetResult, ...],
    ] | None = None,
    run_first_targets_and_proposals: Callable[
        [tuple[object, ...]],
        tuple[FirstTargetProposalResult, ...],
    ] | None = None,
    side_state_callbacks: (
        SpeculativeSideStateCallbacks | None
    ) = None,
) -> PreparedNativeSpeculativeBatch:
    sequence_ids = _validate_sequences(seqs)
    _validate_integer(eos_token, "eos_token")
    has_provider = callable(
        run_first_targets_and_proposals
    )
    has_legacy = (
        draft_adapter is not None
        or run_first_targets is not None
    )
    if has_provider and has_legacy:
        raise ValueError(
            "configure either first-target proposal provider "
            "or legacy adapter callbacks"
        )
    if not has_provider and (
        draft_adapter is None
        or not callable(run_first_targets)
    ):
        raise ValueError(
            "legacy runtime requires draft_adapter and callable "
            "run_first_targets"
        )
    if not callable(run_tail_batch):
        raise ValueError(
            "run_tail_batch must be callable"
        )
    if (
        side_state_callbacks is not None
        and not isinstance(
            side_state_callbacks,
            SpeculativeSideStateCallbacks,
        )
    ):
        raise ValueError(
            "side_state_callbacks must be "
            "SpeculativeSideStateCallbacks"
        )

    phase = "first_target_batch"
    active_transactions = {}
    committed_sequence_ids = []
    side_state_handle = None
    side_state_active = False
    timing_ms = {
        "first_target_batch_ms": 0.0,
        "draft_proposal_ms": 0.0,
        "reserve_blocks_ms": 0.0,
        "tail_batch_ms": 0.0,
        "kv_materialize_ms": 0.0,
        "accept_sample_ms": 0.0,
        "commit_metadata_ms": 0.0,
    }
    try:
        if side_state_callbacks is not None:
            phase = "side_state_prepare"
            side_state_handle = side_state_callbacks.prepare(
                seqs
            )
            side_state_active = True
        if has_provider:
            phase = "first_target_proposal_batch"
            started_at = time.perf_counter()
            provider_rows = (
                _validate_first_target_proposal_results(
                    run_first_targets_and_proposals(seqs),
                    sequence_ids,
                )
            )
            timing_ms["first_target_batch_ms"] = (
                time.perf_counter() - started_at
            ) * 1000.0
            first_targets = {
                sequence_id: FirstTargetResult(
                    sequence_id=sequence_id,
                    target_token=provider_rows[
                        sequence_id
                    ].target_token,
                    metadata=provider_rows[
                        sequence_id
                    ].first_target_metadata,
                )
                for sequence_id in sequence_ids
            }
            proposals = tuple(
                provider_rows[sequence_id].proposal
                for sequence_id in sequence_ids
            )
        else:
            started_at = time.perf_counter()
            first_targets = _validate_first_target_results(
                run_first_targets(seqs),
                sequence_ids,
            )
            timing_ms["first_target_batch_ms"] = (
                time.perf_counter() - started_at
            ) * 1000.0

            phase = "draft_proposal"
            started_at = time.perf_counter()
            capability_limit = getattr(
                getattr(
                    draft_adapter,
                    "capabilities",
                    None,
                ),
                "max_proposal_tokens",
                0,
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
                    max_proposal_tokens=max(
                        0,
                        int(capability_limit),
                    ),
                    first_target_token=first_targets[
                        sequence_id
                    ].target_token,
                    target_hidden=first_targets[
                        sequence_id
                    ].target_hidden,
                    target_logits=first_targets[
                        sequence_id
                    ].target_logits,
                )
                for sequence_id, seq
                in zip(sequence_ids, seqs)
            )
            proposals = validate_draft_adapter_batch(
                draft_adapter,
                contexts,
            )
            timing_ms["draft_proposal_ms"] = (
                time.perf_counter() - started_at
            ) * 1000.0

        phase = "reserve"
        started_at = time.perf_counter()
        records = {}
        for sequence_id, seq, proposal in zip(
            sequence_ids,
            seqs,
            proposals,
        ):
            if not proposal.token_ids:
                records[sequence_id] = {
                    "seq": seq,
                    "proposal": proposal,
                    "plan": None,
                    "reserved_blocks": (),
                    "proxy_block_table": (),
                    "tail": None,
                }
                continue
            plan = build_spec_verify_plan(
                len(seq),
                list(proposal.token_ids),
                seq.block_size,
            )
            transaction = (
                block_manager
                .begin_speculative_kv_transaction(
                    seq,
                    proposed_token_count=len(
                        proposal.token_ids
                    ),
                )
            )
            reserved_blocks = tuple(
                transaction.reserved_block_ids
            )
            proxy_block_table = (
                tuple(seq.block_table)
                + reserved_blocks
            )
            active_transactions[sequence_id] = (
                seq,
                transaction,
            )
            transaction_authorization = (
                block_manager.authorize_speculative_kv_write(
                    transaction,
                    seq,
                )
            )
            records[sequence_id] = {
                "seq": seq,
                "proposal": proposal,
                "plan": plan,
                "reserved_blocks": reserved_blocks,
                "proxy_block_table": proxy_block_table,
                "transaction_authorization": (
                    transaction_authorization
                ),
                "tail": None,
            }
        timing_ms["reserve_blocks_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0

        tail_items = tuple(
            TailBatchItem(
                sequence_id=sequence_id,
                plan=records[sequence_id]["plan"],
                proxy_block_table=records[
                    sequence_id
                ]["proxy_block_table"],
                original_block_identities=tuple(zip(
                    active_transactions[
                        sequence_id
                    ][1].original_block_table,
                    active_transactions[
                        sequence_id
                    ][1].original_block_generations,
                )),
                reserved_block_identities=tuple(zip(
                    active_transactions[
                        sequence_id
                    ][1].reserved_block_ids,
                    active_transactions[
                        sequence_id
                    ][1].reserved_block_generations,
                )),
                transaction_authorization=records[
                    sequence_id
                ]["transaction_authorization"],
            )
            for sequence_id in sequence_ids
            if (
                records[sequence_id]["plan"]
                is not None
                and records[
                    sequence_id
                ]["plan"].query_len
                > 0
            )
        )
        tail_callback_count = 0
        tail_results = {}
        if tail_items:
            phase = "tail_batch"
            started_at = time.perf_counter()
            tail_results = _validate_tail_results(
                run_tail_batch(tail_items),
                tail_items,
            )
            tail_callback_count = 1
            timing_ms["tail_batch_ms"] = (
                time.perf_counter() - started_at
            ) * 1000.0
        for sequence_id, tail in tail_results.items():
            records[sequence_id]["tail"] = tail

        phase = "kv_materialize"
        started_at = time.perf_counter()
        for sequence_id in sequence_ids:
            record = records[sequence_id]
            plan = record["plan"]
            if plan is None:
                continue
            transaction = active_transactions[
                sequence_id
            ][1]
            block_manager.mark_speculative_kv_materialized(
                transaction,
                plan.query_len,
            )
        timing_ms["kv_materialize_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0

        phase = "acceptance"
        started_at = time.perf_counter()
        accepted_by_id = {}
        target_tokens_by_id = {}
        greedy_counts = {}
        eos_truncated_by_id = {}
        budget_truncated_by_id = {}
        for sequence_id in sequence_ids:
            record = records[sequence_id]
            proposal = record["proposal"]
            first_target = first_targets[sequence_id]
            tail = record["tail"]
            tail_tokens = (
                ()
                if tail is None
                else tail.target_tokens
            )
            target_tokens = (
                (first_target.target_token,)
                + tail_tokens
            )
            target_tokens_by_id[
                sequence_id
            ] = target_tokens
            if not proposal.token_ids:
                greedy_count = 0
                accepted_tokens = []
            else:
                greedy_count = _count_accepted_prefix(
                    proposal.token_ids,
                    target_tokens,
                )
                accepted_tokens = list(
                    proposal.token_ids[:greedy_count]
                )
            eos_truncated = False
            seq = record["seq"]
            if (
                not seq.ignore_eos
                and eos_token in accepted_tokens
            ):
                eos_index = accepted_tokens.index(
                    eos_token
                )
                eos_truncated = (
                    eos_index + 1
                    < len(accepted_tokens)
                )
                accepted_tokens = accepted_tokens[
                    :eos_index + 1
                ]
            remaining_budget = max(
                0,
                int(seq.max_tokens)
                - int(seq.num_completion_tokens),
            )
            budget_truncated = (
                remaining_budget
                < len(accepted_tokens)
            )
            accepted_tokens = accepted_tokens[
                :remaining_budget
            ]
            accepted_by_id[sequence_id] = tuple(
                accepted_tokens
            )
            greedy_counts[sequence_id] = greedy_count
            eos_truncated_by_id[
                sequence_id
            ] = eos_truncated
            budget_truncated_by_id[
                sequence_id
            ] = budget_truncated
        timing_ms["accept_sample_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0

        phase = "prepare_result"
        prepared_rows = {}
        for sequence_id in sequence_ids:
            record = records[sequence_id]
            plan = record["plan"]
            tail = record["tail"]
            prepared_rows[sequence_id] = (
                PreparedNativeSpeculativeSequence(
                    sequence_id=sequence_id,
                    sequence=record["seq"],
                    first_target_token=first_targets[
                        sequence_id
                    ].target_token,
                    proposal=record["proposal"],
                    plan=plan,
                    target_tokens=target_tokens_by_id[
                        sequence_id
                    ],
                    greedy_accepted_count=greedy_counts[
                        sequence_id
                    ],
                    accepted_tokens=accepted_by_id[
                        sequence_id
                    ],
                    eos_truncated=eos_truncated_by_id[
                        sequence_id
                    ],
                    output_budget_truncated=(
                        budget_truncated_by_id[
                            sequence_id
                        ]
                    ),
                    transaction=(
                        None
                        if plan is None
                        else active_transactions[
                            sequence_id
                        ][1]
                    ),
                    reserved_blocks=record[
                        "reserved_blocks"
                    ],
                    proxy_block_table=record[
                        "proxy_block_table"
                    ],
                    first_target_metadata=first_targets[
                        sequence_id
                    ].metadata,
                    tail_metadata=(
                        None
                        if tail is None
                        else tail.metadata
                    ),
                    tail_auxiliary=(
                        None
                        if tail is None
                        else tail.auxiliary
                    ),
                )
            )
        side_state_selection = ()
        side_state_state = "disabled"
        if side_state_callbacks is not None:
            phase = "side_state_select"
            side_state_selection = (
                build_speculative_side_state_selection_rows(
                    tuple(
                        prepared_rows[sequence_id]
                        for sequence_id in sequence_ids
                    )
                )
            )
            side_state_callbacks.select(
                side_state_handle,
                side_state_selection,
            )
            side_state_state = "selected"
        return PreparedNativeSpeculativeBatch(
            sequences=tuple(
                prepared_rows[sequence_id]
                for sequence_id in sequence_ids
            ),
            first_target_callback_count=1,
            tail_callback_count=tail_callback_count,
            timing_ms=timing_ms,
            side_state_callbacks=side_state_callbacks,
            side_state_handle=side_state_handle,
            side_state_selection=side_state_selection,
            side_state_state=side_state_state,
        )
    except Exception as cause:
        rolled_back, rollback_errors = _rollback_active(
            block_manager,
            active_transactions,
        )
        if side_state_active:
            try:
                side_state_callbacks.rollback(
                    side_state_handle
                )
            except Exception as error:
                rollback_errors["side_state"] = error
        raise NativeSpeculativeBatchError(
            phase,
            cause,
            committed_sequence_ids=tuple(
                committed_sequence_ids
            ),
            rolled_back_sequence_ids=rolled_back,
            rollback_errors=rollback_errors,
        ) from cause


def _require_active_prepared_container(
    prepared: PreparedNativeSpeculativeBatch,
) -> None:
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


def _require_prepared_batch(
    prepared: PreparedNativeSpeculativeBatch,
) -> None:
    _require_active_prepared_container(prepared)
    if not isinstance(prepared.sequences, tuple):
        raise ValueError(
            "prepared sequences must be a tuple"
        )
    sequence_ids = []
    for row in prepared.sequences:
        if not isinstance(
            row,
            PreparedNativeSpeculativeSequence,
        ):
            raise ValueError(
                "prepared rows must be "
                "PreparedNativeSpeculativeSequence"
            )
        sequence_id = _validate_integer(
            row.sequence_id,
            "prepared sequence id",
        )
        if getattr(row.sequence, "seq_id", None) != sequence_id:
            raise ValueError(
                "prepared row sequence identity mismatch"
            )
        if row.transaction is None:
            if row.plan is not None:
                raise ValueError(
                    "prepared row plan requires a transaction"
                )
        else:
            if row.plan is None:
                raise ValueError(
                    "prepared transaction requires a plan"
                )
            if getattr(
                row.transaction,
                "sequence_id",
                None,
            ) != sequence_id:
                raise ValueError(
                    "prepared transaction sequence ID mismatch"
                )
            if getattr(
                row.transaction,
                "state",
                None,
            ) != "materialized":
                raise RuntimeError(
                    "prepared transaction is not materialized"
                )
        if row.accepted_tokens != row.proposal.token_ids[
            :len(row.accepted_tokens)
        ]:
            raise ValueError(
                "prepared accepted tokens must match proposal prefix"
            )
        sequence_ids.append(sequence_id)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "prepared sequence IDs must be unique"
        )


def build_prepared_proposal_finalize_rows(
    prepared: PreparedNativeSpeculativeBatch,
) -> tuple[PreparedProposalFinalizeRow, ...]:
    _require_prepared_batch(prepared)
    rows = []
    transaction_ids = set()
    for prepared_row in prepared.sequences:
        transaction_id = (
            prepared_row.proposal.proposal_transaction_id
        )
        if transaction_id is None:
            continue
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError(
                "proposal transaction ID must be a non-empty string"
            )
        if transaction_id in transaction_ids:
            raise ValueError(
                "proposal transaction IDs must be unique"
            )
        accepted_count = len(prepared_row.accepted_tokens)
        if accepted_count > len(
            prepared_row.proposal.token_ids
        ):
            raise ValueError(
                "accepted proposal token count exceeds proposal length"
            )
        transaction_ids.add(transaction_id)
        rows.append(
            PreparedProposalFinalizeRow(
                sequence_id=prepared_row.sequence_id,
                proposal_transaction_id=transaction_id,
                accepted_proposal_tokens=accepted_count,
            )
        )
    return tuple(rows)


def apply_prepared_speculative_side_state(
    prepared: PreparedNativeSpeculativeBatch,
):
    _require_active_prepared_container(prepared)
    if prepared.side_state_callbacks is None:
        return None
    if prepared.side_state_state != "selected":
        raise RuntimeError(
            "side state must be selected before apply"
        )
    try:
        receipt = prepared.side_state_callbacks.apply(
            prepared.side_state_handle
        )
    except Exception:
        prepared.side_state_state = "apply_failed"
        raise
    prepared.side_state_state = "applied"
    return receipt


def seal_prepared_speculative_side_state(
    prepared: PreparedNativeSpeculativeBatch,
):
    _require_active_prepared_container(prepared)
    if prepared.side_state_callbacks is None:
        return None
    if prepared.side_state_state != "applied":
        raise RuntimeError(
            "side state must be applied before seal"
        )
    try:
        receipt = prepared.side_state_callbacks.seal(
            prepared.side_state_handle
        )
    except Exception:
        prepared.side_state_state = "seal_failed"
        raise
    prepared.side_state_state = "sealed"
    return receipt


def rollback_prepared_speculative_side_state(
    prepared: PreparedNativeSpeculativeBatch,
):
    _require_active_prepared_container(prepared)
    if prepared.side_state_callbacks is None:
        return None
    if prepared.side_state_state == "rolled_back":
        return None
    if prepared.side_state_state == "sealed":
        raise RuntimeError(
            "sealed side state cannot be rolled back"
        )
    if prepared.side_state_state not in {
        "selected",
        "applied",
        "apply_failed",
        "seal_failed",
    }:
        raise RuntimeError(
            "side state cannot be rolled back from "
            f"{prepared.side_state_state}"
        )
    try:
        receipt = prepared.side_state_callbacks.rollback(
            prepared.side_state_handle
        )
    except Exception:
        prepared.side_state_state = "rollback_failed"
        raise
    prepared.side_state_state = "rolled_back"
    return receipt


def rollback_prepared_native_speculative_batch(
    *,
    block_manager,
    prepared: PreparedNativeSpeculativeBatch,
) -> tuple[int, ...]:
    _require_prepared_batch(prepared)
    rolled_back = []
    rollback_errors = {}
    for row in prepared.sequences:
        if row.transaction is None:
            continue
        try:
            block_manager.rollback_speculative_kv_transaction(
                row.transaction,
                row.sequence,
            )
        except Exception as error:
            rollback_errors[row.sequence_id] = error
        else:
            rolled_back.append(row.sequence_id)
    try:
        rollback_prepared_speculative_side_state(prepared)
    except Exception as error:
        rollback_errors["side_state"] = error
    if rollback_errors:
        prepared.state = "rollback_failed"
        raise NativeSpeculativeBatchError(
            "rollback",
            next(iter(rollback_errors.values())),
            rolled_back_sequence_ids=tuple(rolled_back),
            rollback_errors=rollback_errors,
        )
    prepared.state = "rolled_back"
    return tuple(rolled_back)


def commit_prepared_native_speculative_batch(
    *,
    block_manager,
    prepared: PreparedNativeSpeculativeBatch,
) -> NativeSpeculativeBatchResult:
    _require_prepared_batch(prepared)
    commit_plans = tuple(
        block_manager.prepare_speculative_kv_commit(
            row.transaction,
            row.sequence,
            row.accepted_tokens,
        )
        for row in prepared.sequences
        if row.transaction is not None
    )
    commit_plan_by_id = {
        plan.sequence_id: plan
        for plan in commit_plans
    }
    timing_ms = dict(prepared.timing_ms)
    started_at = time.perf_counter()
    side_state_applied = False
    try:
        apply_prepared_speculative_side_state(prepared)
        side_state_applied = (
            prepared.side_state_state == "applied"
        )
        if commit_plans:
            block_manager.commit_speculative_kv_commit_batch(
                commit_plans
            )
    except Exception as cause:
        active_transactions = {
            row.sequence_id: (
                row.sequence,
                row.transaction,
            )
            for row in prepared.sequences
            if (
                row.transaction is not None
                and getattr(
                    row.transaction,
                    "state",
                    None,
                ) == "materialized"
            )
        }
        rolled_back, rollback_errors = _rollback_active(
            block_manager,
            active_transactions,
        )
        if (
            prepared.side_state_callbacks is not None
            and prepared.side_state_state != "rolled_back"
        ):
            try:
                rollback_prepared_speculative_side_state(
                    prepared
                )
            except Exception as error:
                rollback_errors["side_state"] = error
        prepared.state = "commit_failed"
        raise NativeSpeculativeBatchError(
            "metadata_commit",
            cause,
            rolled_back_sequence_ids=rolled_back,
            rollback_errors=rollback_errors,
        ) from cause
    timing_ms["commit_metadata_ms"] = (
        time.perf_counter() - started_at
    ) * 1000.0
    if side_state_applied:
        try:
            seal_prepared_speculative_side_state(prepared)
        except Exception as cause:
            prepared.state = "seal_failed"
            raise NativeSpeculativeBatchError(
                "side_state_seal",
                cause,
            ) from cause
    result_rows = {}
    for row in prepared.sequences:
        plan = commit_plan_by_id.get(row.sequence_id)
        result_rows[row.sequence_id] = (
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
                committed_blocks=(
                    ()
                    if plan is None
                    else tuple(plan.committed_block_ids)
                ),
                released_blocks=(
                    ()
                    if plan is None
                    else tuple(plan.unused_block_ids)
                ),
                first_target_metadata=(
                    row.first_target_metadata
                ),
                tail_metadata=row.tail_metadata,
                tail_auxiliary=row.tail_auxiliary,
            )
        )
    prepared.state = "committed"
    return NativeSpeculativeBatchResult(
        sequences=tuple(
            result_rows[row.sequence_id]
            for row in prepared.sequences
        ),
        first_target_callback_count=(
            prepared.first_target_callback_count
        ),
        tail_callback_count=prepared.tail_callback_count,
        timing_ms=timing_ms,
    )


def execute_native_speculative_batch(
    *,
    block_manager,
    seqs: tuple[object, ...],
    eos_token: int,
    run_tail_batch: Callable[
        [tuple[TailBatchItem, ...]],
        tuple[TailBatchResult, ...],
    ],
    draft_adapter: DraftAdapter | None = None,
    run_first_targets: Callable[
        [tuple[object, ...]],
        tuple[FirstTargetResult, ...],
    ] | None = None,
    run_first_targets_and_proposals: Callable[
        [tuple[object, ...]],
        tuple[FirstTargetProposalResult, ...],
    ] | None = None,
    side_state_callbacks: (
        SpeculativeSideStateCallbacks | None
    ) = None,
) -> NativeSpeculativeBatchResult:
    prepared = prepare_native_speculative_batch(
        block_manager=block_manager,
        seqs=seqs,
        eos_token=eos_token,
        run_tail_batch=run_tail_batch,
        draft_adapter=draft_adapter,
        run_first_targets=run_first_targets,
        run_first_targets_and_proposals=(
            run_first_targets_and_proposals
        ),
        side_state_callbacks=side_state_callbacks,
    )
    result = commit_prepared_native_speculative_batch(
        block_manager=block_manager,
        prepared=prepared,
    )
    sequence_by_id = {
        row.sequence_id: row.sequence
        for row in prepared.sequences
    }
    try:
        for row in result.sequences:
            seq = sequence_by_id[row.sequence_id]
            for token_id in row.accepted_tokens:
                seq.append_token(token_id)
    except Exception as cause:
        prepared.state = "compatibility_append_failed"
        raise NativeSpeculativeBatchError(
            "legacy_token_append",
            cause,
            committed_sequence_ids=tuple(
                row.sequence_id
                for row in result.sequences
                if row.plan is not None
            ),
        ) from cause
    return result
