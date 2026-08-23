#!/usr/bin/env python3
"""Dependency-light tests for split-phase exact burst contracts."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_decode_burst_split_phase.py"
)
assert MODULE_PATH.is_file(), (
    "split-phase exact burst contract module is missing"
)
SPEC = importlib.util.spec_from_file_location(
    "exact_greedy_decode_burst_split_phase_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ExactBurstPhaseTransfer = module.ExactBurstPhaseTransfer
ExactBurstPublicationTicket = module.ExactBurstPublicationTicket
ExactBurstSplitPhaseTransaction = (
    module.ExactBurstSplitPhaseTransaction
)
ExactGreedyDecodeBurstSplitResult = (
    module.ExactGreedyDecodeBurstSplitResult
)
build_exact_burst_publication_tickets = (
    module.build_exact_burst_publication_tickets
)
validate_exact_burst_split_result = (
    module.validate_exact_burst_split_result
)


class _Completion:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


class _Mailbox:
    def __init__(self, values):
        self.values = tuple(values)
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        return list(self.values)


def _assert_raises(
    error_type,
    message: str,
    call,
) -> None:
    try:
        call()
    except error_type as error:
        assert str(error) == message
    else:
        raise AssertionError(
            f"expected {error_type.__name__}: {message}"
        )


def _tickets():
    return build_exact_burst_publication_tickets(
        parent_lease_identity_sha256="a" * 64,
        first_write_position=259,
        first_physical_slot=11 * 256 + 3,
        parent_token_count=8,
        prefix_token_count=4,
    )


def _transfer(ticket, generation, values):
    return ExactBurstPhaseTransfer(
        ticket=ticket,
        mailbox_generation=generation,
        token_count=4,
        byte_count=32,
        completion=_Completion(),
        mailbox=_Mailbox(values),
    )


def _result():
    prefix, suffix = _tickets()
    return ExactGreedyDecodeBurstSplitResult(
        parent_lease_identity_sha256="a" * 64,
        graph_identity_sha256="b" * 64,
        replay_count=8,
        prefix=_transfer(prefix, 1, (10, 11, 12, 13)),
        suffix=_transfer(suffix, 1, (14, 15, 16, 17)),
    )


def _ticket_with(ticket, **changes):
    values = {
        "parent_lease_identity_sha256": (
            ticket.parent_lease_identity_sha256
        ),
        "phase": ticket.phase,
        "phase_start_ordinal": ticket.phase_start_ordinal,
        "phase_token_count": ticket.phase_token_count,
        "first_write_position": ticket.first_write_position,
        "last_write_position": ticket.last_write_position,
        "first_physical_slot": ticket.first_physical_slot,
        "last_physical_slot": ticket.last_physical_slot,
    }
    values.update(changes)
    return ExactBurstPublicationTicket(
        **values,
        identity_sha256=module._ticket_identity(**values),
    )


def test_publication_tickets_are_contiguous_exhaustive_and_stable():
    prefix, suffix = _tickets()
    repeated_prefix, repeated_suffix = _tickets()

    assert prefix.phase == "prefix"
    assert prefix.phase_start_ordinal == 0
    assert prefix.phase_token_count == 4
    assert prefix.first_write_position == 259
    assert prefix.last_write_position == 262
    assert prefix.first_physical_slot == 11 * 256 + 3
    assert prefix.last_physical_slot == 11 * 256 + 6
    assert suffix.phase == "suffix"
    assert suffix.phase_start_ordinal == 4
    assert suffix.phase_token_count == 4
    assert suffix.first_write_position == 263
    assert suffix.last_write_position == 266
    assert suffix.first_physical_slot == 11 * 256 + 7
    assert suffix.last_physical_slot == 11 * 256 + 10
    assert prefix.identity_sha256 == repeated_prefix.identity_sha256
    assert suffix.identity_sha256 == repeated_suffix.identity_sha256
    assert prefix.identity_sha256 != suffix.identity_sha256


def test_ticket_builder_rejects_non_k8_and_non_k4_shapes():
    for name, kwargs, message in (
        (
            "parent",
            {"parent_token_count": 7},
            "parent_token_count must equal 8",
        ),
        (
            "prefix",
            {"prefix_token_count": 3},
            "prefix_token_count must equal 4",
        ),
        (
            "digest",
            {"parent_lease_identity_sha256": "bad"},
            "parent_lease_identity_sha256 must be a SHA-256 digest",
        ),
    ):
        values = {
            "parent_lease_identity_sha256": "a" * 64,
            "first_write_position": 259,
            "first_physical_slot": 11 * 256 + 3,
            "parent_token_count": 8,
            "prefix_token_count": 4,
        }
        values.update(kwargs)
        _assert_raises(
            ValueError,
            message,
            lambda values=values: (
                build_exact_burst_publication_tickets(**values)
            ),
        )
        assert name


def test_split_result_inventory_is_exact():
    result = _result()
    assert validate_exact_burst_split_result(
        result,
        expected_parent_lease_identity_sha256="a" * 64,
        expected_graph_identity_sha256="b" * 64,
    ) is result

    invalid_cases = (
        (
            replace(result, replay_count=7),
            "split result replay_count must equal 8",
        ),
        (
            replace(
                result,
                parent_lease_identity_sha256="c" * 64,
            ),
            "split result parent lease identity mismatch",
        ),
        (
            replace(
                result,
                prefix=replace(result.prefix, byte_count=24),
            ),
            "prefix transfer byte_count mismatch",
        ),
        (
            replace(
                result,
                suffix=replace(
                    result.suffix,
                    ticket=_ticket_with(
                        result.suffix.ticket,
                        phase_start_ordinal=3,
                    ),
                ),
            ),
            "split publication ranges are not contiguous",
        ),
        (
            replace(
                result,
                suffix=replace(
                    result.suffix,
                    ticket=replace(
                        result.suffix.ticket,
                        phase="prefix",
                    ),
                ),
            ),
            "suffix ticket phase mismatch",
        ),
    )
    for invalid, message in invalid_cases:
        _assert_raises(
            ValueError,
            message,
            lambda invalid=invalid: (
                validate_exact_burst_split_result(
                    invalid,
                    expected_parent_lease_identity_sha256=(
                        "a" * 64
                    ),
                    expected_graph_identity_sha256="b" * 64,
                )
            ),
        )


def test_phase_transfer_waits_once_and_returns_exact_tokens():
    result = _result()
    assert result.prefix.wait_tokens() == (10, 11, 12, 13)
    assert result.prefix.wait_tokens() == (10, 11, 12, 13)
    assert result.prefix.completion.waits == 1
    assert result.prefix.mailbox.tolist_calls == 1


def test_transaction_requires_monotonic_ordered_transitions():
    result = _result()
    transaction = ExactBurstSplitPhaseTransaction.create(
        parent_lease_identity_sha256="a" * 64,
        result=result,
    )
    assert transaction.state == "enqueued"
    transaction.mark_prefix_ready()
    transaction.mark_prefix_committed()
    transaction.mark_suffix_ready()
    transaction.mark_suffix_committed()
    assert transaction.state == "suffix_committed"

    skipped = ExactBurstSplitPhaseTransaction.create(
        parent_lease_identity_sha256="a" * 64,
        result=_result(),
    )
    _assert_raises(
        ValueError,
        "split transaction state must be prefix_ready, got enqueued",
        skipped.mark_prefix_committed,
    )
    skipped.mark_prefix_ready()
    _assert_raises(
        ValueError,
        "split transaction state must be enqueued, got prefix_ready",
        skipped.mark_prefix_ready,
    )


def test_transaction_failure_phase_is_explicit():
    pre_prefix = ExactBurstSplitPhaseTransaction.create(
        parent_lease_identity_sha256="a" * 64,
        result=_result(),
    )
    pre_prefix.mark_pre_prefix_failed("prefix_wait_failure")
    assert pre_prefix.state == "pre_prefix_failed"
    assert pre_prefix.failure_reason == "prefix_wait_failure"

    post_prefix = ExactBurstSplitPhaseTransaction.create(
        parent_lease_identity_sha256="a" * 64,
        result=_result(),
    )
    post_prefix.mark_prefix_ready()
    post_prefix.mark_prefix_committed()
    post_prefix.mark_post_prefix_failed("suffix_commit_failure")
    assert post_prefix.state == "post_prefix_failed"
    assert post_prefix.failure_reason == "suffix_commit_failure"


def main() -> None:
    test_publication_tickets_are_contiguous_exhaustive_and_stable()
    test_ticket_builder_rejects_non_k8_and_non_k4_shapes()
    test_split_result_inventory_is_exact()
    test_phase_transfer_waits_once_and_returns_exact_tokens()
    test_transaction_requires_monotonic_ordered_transitions()
    test_transaction_failure_phase_is_explicit()
    print("exact greedy decode burst split-phase tests passed")


if __name__ == "__main__":
    main()
