from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from tinyvllm.speculative.verifier import (
    SpecVerifyPlan,
    build_spec_verify_plan,
)


@dataclass(frozen=True)
class NativeTailResult:
    target_tokens: tuple[int, ...]
    metadata: object | None = None
    auxiliary: object | None = None


@dataclass(frozen=True)
class NativeSpeculativeStepResult:
    plan: SpecVerifyPlan
    target_tokens: tuple[int, ...]
    greedy_accepted_count: int
    accepted_tokens: tuple[int, ...]
    eos_truncated: bool
    output_budget_truncated: bool
    reserved_blocks: tuple[int, ...]
    proxy_block_table: tuple[int, ...]
    committed_blocks: tuple[int, ...]
    released_blocks: tuple[int, ...]
    tail_metadata: object | None = None
    tail_auxiliary: object | None = None
    timing_ms: dict[str, float] | None = None


class NativeSpeculativeStepError(RuntimeError):
    def __init__(
        self,
        phase: str,
        cause: Exception,
        rollback_error: Exception | None = None,
    ):
        self.phase = phase
        self.cause = cause
        self.rollback_error = rollback_error
        suffix = (
            ""
            if rollback_error is None
            else f"; rollback failed: {rollback_error}"
        )
        super().__init__(
            f"native speculative step {phase} failed: {cause}{suffix}"
        )


def _validate_token(token_id: object, name: str) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise ValueError(f"{name} must be an integer token id")
    return token_id


def _validate_inputs(
    draft_tokens: list[int],
    eos_token: int,
    callbacks: tuple[tuple[str, object], ...],
) -> None:
    if not isinstance(draft_tokens, list) or not draft_tokens:
        raise ValueError("draft_tokens must be a non-empty list")
    for token_id in draft_tokens:
        _validate_token(token_id, "draft token")
    _validate_token(eos_token, "eos_token")
    for name, callback in callbacks:
        if not callable(callback):
            raise ValueError(f"{name} must be callable")


def _validate_tail_result(
    result: NativeTailResult,
    expected_count: int,
) -> tuple[int, ...]:
    if not isinstance(result, NativeTailResult):
        raise ValueError(
            "run_tail must return a NativeTailResult"
        )
    if len(result.target_tokens) != expected_count:
        raise ValueError(
            "tail target count does not match verifier query length"
        )
    for token_id in result.target_tokens:
        _validate_token(token_id, "tail target")
    return result.target_tokens


def _count_accepted_prefix(
    draft_tokens: list[int],
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


def execute_native_speculative_step(
    *,
    block_manager,
    seq,
    draft_tokens: list[int],
    eos_token: int,
    run_first_target: Callable[[], int],
    prepare_tail: Callable[[SpecVerifyPlan, tuple[int, ...]], object],
    run_tail: Callable[[object], NativeTailResult],
) -> NativeSpeculativeStepResult:
    _validate_inputs(
        draft_tokens,
        eos_token,
        (
            ("run_first_target", run_first_target),
            ("prepare_tail", prepare_tail),
            ("run_tail", run_tail),
        ),
    )
    plan = build_spec_verify_plan(
        len(seq),
        draft_tokens,
        seq.block_size,
    )
    phase = "reserve"
    transaction = None
    timing_ms = {
        "reserve_blocks_ms": 0.0,
        "decode_first_target_ms": 0.0,
        "verify_prepare_ms": 0.0,
        "target_forward_ms": 0.0,
        "kv_materialize_ms": 0.0,
        "accept_sample_ms": 0.0,
        "commit_metadata_ms": 0.0,
    }
    try:
        started_at = time.perf_counter()
        transaction = (
            block_manager.begin_speculative_kv_transaction(
                seq,
                proposed_token_count=len(draft_tokens),
            )
        )
        timing_ms["reserve_blocks_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0
        reserved_blocks = tuple(
            transaction.reserved_block_ids
        )
        proxy_block_table = (
            tuple(seq.block_table) + reserved_blocks
        )

        phase = "first_target_decode"
        started_at = time.perf_counter()
        first_target = _validate_token(
            run_first_target(),
            "first target",
        )
        timing_ms["decode_first_target_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0

        tail_result = NativeTailResult(target_tokens=())
        if plan.query_len:
            phase = "verify_prepare"
            started_at = time.perf_counter()
            prepared_tail = prepare_tail(
                plan,
                proxy_block_table,
            )
            timing_ms["verify_prepare_ms"] = (
                time.perf_counter() - started_at
            ) * 1000.0
            phase = "tail_forward"
            started_at = time.perf_counter()
            tail_result = run_tail(prepared_tail)
            tail_targets = _validate_tail_result(
                tail_result,
                plan.query_len,
            )
            timing_ms["target_forward_ms"] = (
                time.perf_counter() - started_at
            ) * 1000.0
        else:
            tail_targets = ()

        phase = "kv_materialize"
        started_at = time.perf_counter()
        block_manager.mark_speculative_kv_materialized(
            transaction,
            plan.query_len,
        )
        timing_ms["kv_materialize_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0

        phase = "acceptance"
        started_at = time.perf_counter()
        target_tokens = (first_target,) + tail_targets
        greedy_accepted_count = _count_accepted_prefix(
            draft_tokens,
            target_tokens,
        )
        accepted_tokens = list(
            draft_tokens[:greedy_accepted_count]
        )
        eos_truncated = False
        if (
            not seq.ignore_eos
            and eos_token in accepted_tokens
        ):
            eos_index = accepted_tokens.index(eos_token)
            eos_truncated = (
                eos_index + 1 < len(accepted_tokens)
            )
            accepted_tokens = accepted_tokens[:eos_index + 1]
        remaining_budget = max(
            0,
            seq.max_tokens - seq.num_completion_tokens,
        )
        output_budget_truncated = (
            remaining_budget < len(accepted_tokens)
        )
        accepted_tokens = accepted_tokens[:remaining_budget]
        timing_ms["accept_sample_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0

        block_table_before = tuple(seq.block_table)
        phase = "metadata_commit"
        started_at = time.perf_counter()
        block_manager.commit_speculative_kv_transaction(
            transaction,
            seq,
            accepted_tokens,
        )
        timing_ms["commit_metadata_ms"] = (
            time.perf_counter() - started_at
        ) * 1000.0
        transaction = None
        committed_blocks = tuple(
            seq.block_table[len(block_table_before):]
        )
        released_blocks = tuple(
            block_id
            for block_id in reserved_blocks
            if block_id not in committed_blocks
        )
        return NativeSpeculativeStepResult(
            plan=plan,
            target_tokens=target_tokens,
            greedy_accepted_count=greedy_accepted_count,
            accepted_tokens=tuple(accepted_tokens),
            eos_truncated=eos_truncated,
            output_budget_truncated=output_budget_truncated,
            reserved_blocks=reserved_blocks,
            proxy_block_table=proxy_block_table,
            committed_blocks=committed_blocks,
            released_blocks=released_blocks,
            tail_metadata=tail_result.metadata,
            tail_auxiliary=tail_result.auxiliary,
            timing_ms=timing_ms,
        )
    except Exception as cause:
        rollback_error = None
        if transaction is not None:
            try:
                block_manager.rollback_speculative_kv_transaction(
                    transaction,
                    seq,
                )
            except Exception as error:
                rollback_error = error
        raise NativeSpeculativeStepError(
            phase,
            cause,
            rollback_error,
        ) from cause
