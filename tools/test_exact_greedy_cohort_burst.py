from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
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
ExactGreedyCohortBurstGraph = module.ExactGreedyCohortBurstGraph
ExactGreedyCohortBurstTerminalError = (
    module.ExactGreedyCohortBurstTerminalError
)
ExactGreedyCohortBurstResult = module.ExactGreedyCohortBurstResult
ExactGreedyCohortBurstRowResult = module.ExactGreedyCohortBurstRowResult
ExactGreedyCohortBurstTransaction = (
    module.ExactGreedyCohortBurstTransaction
)
build_exact_greedy_cohort_burst_execution_telemetry = (
    module.build_exact_greedy_cohort_burst_execution_telemetry
)
build_terminal_exact_greedy_cohort_burst_execution_telemetry = (
    module.build_terminal_exact_greedy_cohort_burst_execution_telemetry
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


def test_execution_telemetry_closes_identity_work_and_inventory() -> None:
    lease = _lease()
    result = _result(
        lease,
        tokens=((11, 2, 91, 92), (21, 22, 23, 24)),
    )
    publication = validate_exact_greedy_cohort_burst_result(
        lease,
        result,
        eos_token_id=2,
    )

    row = build_exact_greedy_cohort_burst_execution_telemetry(
        lease=lease,
        result=result,
        publication=publication,
        actual_duration_ns=25,
        host_visible_publication_gap_ns=4,
        token_d2h_bytes=64,
        quarantine_reason=None,
        fallback_reason=None,
        failure_reason=None,
        rollback_reason=None,
        pending_lease_count=0,
        pending_transaction_count=0,
    )

    assert row.lease_identity_sha256 == lease.identity_sha256
    assert row.result_identity_sha256
    assert row.graph_identity_sha256 == lease.graph_identity_sha256
    assert row.requested_width == 4
    assert row.authorized_width == 4
    assert row.completed_replay_count == 4
    assert row.token_d2h_calls == 1
    assert row.token_d2h_bytes == 64
    assert row.generated_token_counts == ((7, 4), (9, 4))
    assert row.committed_token_counts == ((7, 2), (9, 4))
    assert row.eos_discarded_token_counts == ((7, 2), (9, 0))
    assert row.post_eos_wasted_tokens == 2
    assert row.post_eos_wasted_forwards == 2
    assert row.pending_inventory == (
        ("leases", 0),
        ("transactions", 0),
    )
    with pytest.raises(FrozenInstanceError):
        row.actual_duration_ns = 26


def test_terminal_execution_telemetry_closes_failed_inventory() -> None:
    lease = _lease()
    row = build_terminal_exact_greedy_cohort_burst_execution_telemetry(
        lease=lease,
        completed_replay_count=2,
        actual_duration_ns=25,
        host_visible_publication_gap_ns=0,
        fallback_reason=None,
        failure_reason="graph_replay_failure",
        rollback_reason=None,
        quarantine_reason="graph_replay_failure",
        pending_lease_count=0,
        pending_transaction_count=0,
    )

    assert row.result_identity_sha256 is None
    assert row.completed_replay_count == 2
    assert row.failure_reason == "graph_replay_failure"
    assert row.quarantined is True
    assert row.pending_inventory == (
        ("leases", 0),
        ("transactions", 0),
    )


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


def _graph_tensors(batch_size: int, block_table_width: int):
    return {
        "input_tokens": SimpleNamespace(shape=(batch_size,)),
        "positions": SimpleNamespace(shape=(batch_size,)),
        "context_lengths": SimpleNamespace(shape=(batch_size,)),
        "slot_mappings": SimpleNamespace(shape=(batch_size,)),
        "block_tables": SimpleNamespace(
            shape=(batch_size, block_table_width)
        ),
        "active_row_masks": SimpleNamespace(shape=(batch_size,)),
        "result_bundle": SimpleNamespace(
            shape=(batch_size, 8, 2)
        ),
        "token_history": SimpleNamespace(shape=(batch_size, 8)),
        "history_indices": SimpleNamespace(shape=(batch_size,)),
        "eos_observations": SimpleNamespace(shape=(batch_size, 8)),
    }


def _captured_graph(
    *,
    batch_size: int = 2,
    replay=None,
    history=None,
    eos=None,
):
    replay = replay or (lambda: None)
    history = history or (
        lambda: (
            (11, 12, 13, 14),
            (21, 22, 23, 24),
        )
    )
    eos = eos or (
        lambda: tuple(
            tuple(False for _ in row)
            for row in history()
        )
    )
    return ExactGreedyCohortBurstGraph.capture(
        tensors=_graph_tensors(batch_size, 8),
        graph_generation=7,
        batch_size=batch_size,
        block_table_width=8,
        dtype="torch.bfloat16",
        device_identity="GPU-test",
        tensor_parallel_size=1,
        correctness_trace=False,
        scratch_block_ids=tuple(range(100, 100 + batch_size)),
        capture_live_kv_mutations=(),
        bind_rows=lambda lease, rows: None,
        graph_replay=replay,
        read_result_bundle=lambda: (history(), eos()),
    )


def test_cohort_capture_uses_private_scratch_and_row_indexed_tensors() -> None:
    graph = _captured_graph(batch_size=4)
    assert graph.tensors["input_tokens"].shape == (4,)
    assert graph.tensors["context_lengths"].shape == (4,)
    assert graph.tensors["token_history"].shape == (4, 8)
    assert graph.receipt.capture_live_kv_mutations == ()
    assert graph.receipt.scratch_block_ids == (100, 101, 102, 103)


def test_cohort_capture_requires_packed_result_bundle_shape() -> None:
    tensors = _graph_tensors(2, 8)
    del tensors["result_bundle"]
    with pytest.raises(
        ValueError,
        match="missing cohort graph tensor: result_bundle",
    ):
        ExactGreedyCohortBurstGraph.capture(
            tensors=tensors,
            graph_generation=7,
            batch_size=2,
            block_table_width=8,
            dtype="torch.bfloat16",
            device_identity="GPU-test",
            tensor_parallel_size=1,
            correctness_trace=False,
            scratch_block_ids=(100, 101),
            capture_live_kv_mutations=(),
            bind_rows=lambda lease, rows: None,
            graph_replay=lambda: None,
            read_result_bundle=lambda: ((), ()),
        )


def test_cohort_replay_runs_k_steps_then_one_token_history_d2h() -> None:
    calls = {"replay": 0, "result_bundle": 0}

    def replay():
        calls["replay"] += 1

    def result_bundle():
        calls["result_bundle"] += 1
        return (
            (
                (11, 12, 13, 14),
                (21, 22, 23, 24),
            ),
            (
                (False, False, False, False),
                (False, False, False, False),
            ),
        )

    graph = ExactGreedyCohortBurstGraph.capture(
        tensors=_graph_tensors(2, 8),
        graph_generation=7,
        batch_size=2,
        block_table_width=8,
        dtype="torch.bfloat16",
        device_identity="GPU-test",
        tensor_parallel_size=1,
        correctness_trace=False,
        scratch_block_ids=(100, 101),
        capture_live_kv_mutations=(),
        bind_rows=lambda lease, rows: None,
        graph_replay=replay,
        read_result_bundle=result_bundle,
    )
    lease = _lease()
    lease = replace(
        lease,
        graph_identity_sha256=graph.receipt.graph_identity_sha256,
    )
    lease = build_exact_greedy_cohort_burst_lease(
        schedule_generation=lease.schedule_generation,
        graph_generation=lease.graph_generation,
        graph_identity_sha256=lease.graph_identity_sha256,
        requested_width=lease.requested_width,
        authorized_width=lease.authorized_width,
        decision_now_ns=lease.decision_now_ns,
        cost_table_sha256=lease.cost_table_sha256,
        predicted_duration_ns=lease.predicted_duration_ns,
        global_slack_ns=lease.global_slack_ns,
        rows=lease.rows,
    )
    result = graph.replay(lease, row_bindings=({}, {}))
    assert calls == {"replay": 4, "result_bundle": 1}
    assert result.token_d2h_calls == 1
    assert tuple(len(row.tokens) for row in result.rows) == (4, 4)


def test_correctness_logits_are_clipped_to_authorized_width() -> None:
    graph = ExactGreedyCohortBurstGraph.capture(
        tensors=_graph_tensors(2, 8),
        graph_generation=7,
        batch_size=2,
        block_table_width=8,
        dtype="torch.bfloat16",
        device_identity="GPU-test",
        tensor_parallel_size=1,
        correctness_trace=True,
        scratch_block_ids=(100, 101),
        capture_live_kv_mutations=(),
        bind_rows=lambda lease, rows: None,
        graph_replay=lambda: None,
        read_result_bundle=lambda: (
            (
                (0, 1, 2, 0, -1, -1, -1, -1),
                (1, 2, 0, 1, -1, -1, -1, -1),
            ),
            (
                (False,) * 8,
                (False,) * 8,
            ),
        ),
        read_sampled_logits=lambda: tuple(
            tuple(
                tuple(
                    1.0 if token == index else 0.0
                    for index in range(3)
                )
                for token in row
            )
            for row in (
                (0, 1, 2, 0, 0, 0, 0, 0),
                (1, 2, 0, 1, 0, 0, 0, 0),
            )
        ),
    )
    lease = _lease()
    lease = build_exact_greedy_cohort_burst_lease(
        schedule_generation=lease.schedule_generation,
        graph_generation=lease.graph_generation,
        graph_identity_sha256=graph.receipt.graph_identity_sha256,
        requested_width=lease.requested_width,
        authorized_width=lease.authorized_width,
        decision_now_ns=lease.decision_now_ns,
        cost_table_sha256=lease.cost_table_sha256,
        predicted_duration_ns=lease.predicted_duration_ns,
        global_slack_ns=lease.global_slack_ns,
        rows=lease.rows,
    )

    result = graph.replay(lease, row_bindings=({}, {}))

    assert tuple(
        len(row.sampled_logits) for row in result.rows
    ) == (4, 4)
    validate_exact_greedy_cohort_burst_result(
        lease,
        result,
        eos_token_id=99,
        correctness_trace=True,
    )


def test_post_launch_graph_failure_is_terminal_and_quarantines() -> None:
    calls = {"count": 0}

    def fail_second_replay():
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")

    graph = _captured_graph(replay=fail_second_replay)
    lease = _lease()
    lease = build_exact_greedy_cohort_burst_lease(
        schedule_generation=lease.schedule_generation,
        graph_generation=lease.graph_generation,
        graph_identity_sha256=graph.receipt.graph_identity_sha256,
        requested_width=lease.requested_width,
        authorized_width=lease.authorized_width,
        decision_now_ns=lease.decision_now_ns,
        cost_table_sha256=lease.cost_table_sha256,
        predicted_duration_ns=lease.predicted_duration_ns,
        global_slack_ns=lease.global_slack_ns,
        rows=lease.rows,
    )
    with pytest.raises(
        ExactGreedyCohortBurstTerminalError,
        match="graph replay failed",
    ) as error:
        graph.replay(lease, row_bindings=({}, {}))
    assert error.value.completed_replays == 1
    assert graph.capability()["quarantined"] is True


def test_model_runner_declares_cohort_runtime_entrypoints() -> None:
    source = (
        REPO_ROOT / "tinyvllm" / "engine" / "model_runner.py"
    ).read_text(encoding="utf-8")
    for method in (
        "exact_greedy_cohort_burst_capability",
        "capture_exact_greedy_cohort_burst_graph",
        "run_exact_greedy_cohort_burst",
        "quarantine_exact_greedy_cohort_burst_graph",
    ):
        assert f"def {method}(" in source
