from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_cohort_burst.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exact_greedy_cohort_burst_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

CohortWriteAuthority = module.CohortWriteAuthority
ExactGreedyCohortBurstFallback = module.ExactGreedyCohortBurstFallback
ExactGreedyCohortBurstResult = module.ExactGreedyCohortBurstResult
ExactGreedyCohortBurstRowResult = module.ExactGreedyCohortBurstRowResult
ExactGreedyCohortBurstTransaction = (
    module.ExactGreedyCohortBurstTransaction
)
build_exact_greedy_cohort_burst_lease = (
    module.build_exact_greedy_cohort_burst_lease
)
validate_exact_greedy_cohort_burst_result = (
    module.validate_exact_greedy_cohort_burst_result
)


def _authority(
    sequence_id: int,
    *,
    slot_base: int | None = None,
    width: int = 4,
) -> CohortWriteAuthority:
    slot_base = (
        sequence_id * 256 if slot_base is None else slot_base
    )
    block_id = slot_base // 256
    return CohortWriteAuthority(
        sequence_id=sequence_id,
        sequence_generation=3,
        block_table_identity=((block_id, 5),),
        writable_block_identities=((block_id, 5),),
        first_write_position=256,
        last_write_position=256 + width - 1,
        first_physical_slot=slot_base,
        last_physical_slot=slot_base + width - 1,
        initial_completion_count=2,
        initial_sequence_length=257,
        remaining_output_tokens=8,
    )


def _lease(
    *,
    rows: tuple[CohortWriteAuthority, ...] | None = None,
    width: int = 4,
):
    return build_exact_greedy_cohort_burst_lease(
        schedule_generation=11,
        graph_generation=7,
        graph_identity_sha256="a" * 64,
        requested_width=width,
        authorized_width=width,
        decision_now_ns=100,
        cost_table_sha256="b" * 64,
        predicted_duration_ns=20,
        global_slack_ns=30,
        rows=rows or (_authority(7, width=width), _authority(9, width=width)),
    )


def _row_result(
    authority: CohortWriteAuthority,
    tokens: tuple[int, ...],
    *,
    sampled_logits: tuple[tuple[float, ...], ...] = (),
) -> ExactGreedyCohortBurstRowResult:
    return ExactGreedyCohortBurstRowResult(
        sequence_id=authority.sequence_id,
        sequence_generation=authority.sequence_generation,
        tokens=tokens,
        final_position=authority.first_write_position + len(tokens),
        final_context_length=authority.initial_sequence_length + len(tokens),
        final_physical_slot=authority.last_physical_slot + 1,
        sampled_logits=sampled_logits,
    )


def _result(
    lease,
    *,
    tokens: tuple[tuple[int, ...], ...] | None = None,
    rows: tuple[ExactGreedyCohortBurstRowResult, ...] | None = None,
    correctness_trace: bool = False,
) -> ExactGreedyCohortBurstResult:
    tokens = tokens or tuple(
        tuple(range(10 + offset, 10 + offset + lease.authorized_width))
        for offset, _ in enumerate(lease.rows)
    )
    if rows is None:
        rows = tuple(
            _row_result(
                authority,
                row_tokens,
                sampled_logits=(
                    tuple(
                        tuple(
                            1.0 if index == token else 0.0
                            for index in range(max(row_tokens) + 1)
                        )
                        for token in row_tokens
                    )
                    if correctness_trace
                    else ()
                ),
            )
            for authority, row_tokens in zip(lease.rows, tokens)
        )
    return ExactGreedyCohortBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        graph_identity_sha256=lease.graph_identity_sha256,
        graph_generation=lease.graph_generation,
        replay_count=lease.authorized_width,
        rows=rows,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=1 if correctness_trace else 0,
    )


def test_cohort_identity_binds_order_and_every_write_authority() -> None:
    lease = _lease()
    reversed_result = _result(
        lease,
        rows=tuple(
            reversed(
                tuple(
                    _row_result(authority, (1, 2, 3, 4))
                    for authority in lease.rows
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="ordered sequence IDs"):
        validate_exact_greedy_cohort_burst_result(
            lease,
            reversed_result,
            eos_token_id=2,
        )

    changed = replace(
        lease.rows[0],
        block_table_identity=((7, 6),),
        writable_block_identities=((7, 6),),
    )
    rebuilt = _lease(rows=(changed, lease.rows[1]))
    assert rebuilt.identity_sha256 != lease.identity_sha256

    forged = replace(lease, rows=(changed, lease.rows[1]))
    with pytest.raises(ValueError, match="lease identity"):
        validate_exact_greedy_cohort_burst_result(
            forged,
            _result(forged),
            eos_token_id=99,
        )


def test_eos_prefix_is_committed_and_suffix_is_counted_as_waste() -> None:
    lease = _lease()
    result = _result(
        lease,
        tokens=((11, 2, 91, 92), (21, 22, 23, 24)),
    )
    validated = validate_exact_greedy_cohort_burst_result(
        lease,
        result,
        eos_token_id=2,
    )
    assert validated.commit_tokens == (
        (11, 2),
        (21, 22, 23, 24),
    )
    assert validated.wasted_post_eos_tokens == 2
    assert validated.wasted_post_eos_forwards == 2


def test_lease_rejects_overlapping_physical_authority() -> None:
    rows = (
        _authority(7, slot_base=1024),
        _authority(9, slot_base=1026),
    )
    with pytest.raises(ValueError, match="overlap"):
        _lease(rows=rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("lease_identity_sha256", "c" * 64, "lease identity"),
        ("graph_identity_sha256", "c" * 64, "graph identity"),
        ("graph_generation", 8, "graph generation"),
        ("replay_count", 3, "replay count"),
        ("token_d2h_calls", 2, "one token D2H"),
    ),
)
def test_result_rejects_stale_or_incomplete_cohort_fields(
    field: str,
    value,
    message: str,
) -> None:
    lease = _lease()
    result = replace(_result(lease), **{field: value})
    with pytest.raises(ValueError, match=message):
        validate_exact_greedy_cohort_burst_result(
            lease,
            result,
            eos_token_id=2,
        )


def test_correctness_trace_requires_finite_argmax_equal_logits() -> None:
    lease = _lease(width=2)
    good = _result(
        lease,
        tokens=((1, 0), (0, 1)),
        correctness_trace=True,
    )
    validate_exact_greedy_cohort_burst_result(
        lease,
        good,
        eos_token_id=99,
        correctness_trace=True,
    )
    bad_row = replace(
        good.rows[0],
        sampled_logits=((0.0, 1.0), (0.0, 1.0)),
    )
    bad = replace(good, rows=(bad_row, good.rows[1]))
    with pytest.raises(ValueError, match="argmax"):
        validate_exact_greedy_cohort_burst_result(
            lease,
            bad,
            eos_token_id=99,
            correctness_trace=True,
        )


def test_fallback_requires_zero_replays() -> None:
    with pytest.raises(ValueError, match="cannot follow"):
        ExactGreedyCohortBurstFallback(
            fallback_reason="graph_unavailable",
            replay_count=1,
        )


def test_transaction_allows_only_declared_state_transitions() -> None:
    lease = _lease()
    result = _result(lease)
    transaction = ExactGreedyCohortBurstTransaction(lease)
    assert transaction.pending is True
    transaction.dispatch()
    assert transaction.pending is True
    publication = transaction.validate(result, eos_token_id=99)
    transaction.commit()
    assert transaction.state == "committed"
    assert transaction.pending is False
    assert transaction.publication is publication
    with pytest.raises(RuntimeError, match="committed"):
        transaction.commit()

    cancelled = ExactGreedyCohortBurstTransaction(lease)
    cancelled.cancel(
        ExactGreedyCohortBurstFallback("graph_unavailable")
    )
    assert cancelled.state == "cancelled"
    assert cancelled.pending is False
    with pytest.raises(RuntimeError, match="cancelled"):
        cancelled.dispatch()


def test_post_replay_failure_quarantines_before_terminal_failure() -> None:
    transaction = ExactGreedyCohortBurstTransaction(_lease())
    transaction.dispatch()
    transaction.quarantine_and_fail(
        "graph_replay_failure",
        completed_replays=1,
    )
    assert transaction.state == "failed"
    assert transaction.quarantined is True
    assert transaction.pending is False
    assert transaction.completed_replays == 1
    with pytest.raises(RuntimeError, match="failed"):
        transaction.cancel(
            ExactGreedyCohortBurstFallback("retry")
        )
