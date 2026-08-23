"""Dependency-light contracts for split-phase exact greedy bursts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Optional


def _require_digest(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _require_non_negative_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_positive_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_reason(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _ticket_identity(
    *,
    parent_lease_identity_sha256: str,
    phase: str,
    phase_start_ordinal: int,
    phase_token_count: int,
    first_write_position: int,
    last_write_position: int,
    first_physical_slot: int,
    last_physical_slot: int,
) -> str:
    payload = {
        "first_physical_slot": first_physical_slot,
        "first_write_position": first_write_position,
        "last_physical_slot": last_physical_slot,
        "last_write_position": last_write_position,
        "parent_lease_identity_sha256": (
            parent_lease_identity_sha256
        ),
        "phase": phase,
        "phase_start_ordinal": phase_start_ordinal,
        "phase_token_count": phase_token_count,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExactBurstPublicationTicket:
    parent_lease_identity_sha256: str
    phase: str
    phase_start_ordinal: int
    phase_token_count: int
    first_write_position: int
    last_write_position: int
    first_physical_slot: int
    last_physical_slot: int
    identity_sha256: str

    def __post_init__(self) -> None:
        _require_digest(
            self.parent_lease_identity_sha256,
            "parent_lease_identity_sha256",
        )
        if self.phase not in ("prefix", "suffix"):
            raise ValueError(
                "publication ticket phase must be prefix or suffix"
            )
        _require_non_negative_int(
            self.phase_start_ordinal,
            "phase_start_ordinal",
        )
        _require_positive_int(
            self.phase_token_count,
            "phase_token_count",
        )
        for name in (
            "first_write_position",
            "last_write_position",
            "first_physical_slot",
            "last_physical_slot",
        ):
            _require_non_negative_int(getattr(self, name), name)
        _require_digest(
            self.identity_sha256,
            "identity_sha256",
        )


def _build_ticket(
    *,
    parent_lease_identity_sha256: str,
    phase: str,
    phase_start_ordinal: int,
    phase_token_count: int,
    first_write_position: int,
    first_physical_slot: int,
) -> ExactBurstPublicationTicket:
    last_write_position = (
        first_write_position + phase_token_count - 1
    )
    last_physical_slot = (
        first_physical_slot + phase_token_count - 1
    )
    identity_sha256 = _ticket_identity(
        parent_lease_identity_sha256=(
            parent_lease_identity_sha256
        ),
        phase=phase,
        phase_start_ordinal=phase_start_ordinal,
        phase_token_count=phase_token_count,
        first_write_position=first_write_position,
        last_write_position=last_write_position,
        first_physical_slot=first_physical_slot,
        last_physical_slot=last_physical_slot,
    )
    return ExactBurstPublicationTicket(
        parent_lease_identity_sha256=(
            parent_lease_identity_sha256
        ),
        phase=phase,
        phase_start_ordinal=phase_start_ordinal,
        phase_token_count=phase_token_count,
        first_write_position=first_write_position,
        last_write_position=last_write_position,
        first_physical_slot=first_physical_slot,
        last_physical_slot=last_physical_slot,
        identity_sha256=identity_sha256,
    )


def build_exact_burst_publication_tickets(
    *,
    parent_lease_identity_sha256: str,
    first_write_position: int,
    first_physical_slot: int,
    parent_token_count: int,
    prefix_token_count: int,
) -> tuple[
    ExactBurstPublicationTicket,
    ExactBurstPublicationTicket,
]:
    _require_digest(
        parent_lease_identity_sha256,
        "parent_lease_identity_sha256",
    )
    _require_non_negative_int(
        first_write_position,
        "first_write_position",
    )
    _require_non_negative_int(
        first_physical_slot,
        "first_physical_slot",
    )
    if parent_token_count != 8:
        raise ValueError("parent_token_count must equal 8")
    if prefix_token_count != 4:
        raise ValueError("prefix_token_count must equal 4")
    prefix = _build_ticket(
        parent_lease_identity_sha256=(
            parent_lease_identity_sha256
        ),
        phase="prefix",
        phase_start_ordinal=0,
        phase_token_count=prefix_token_count,
        first_write_position=first_write_position,
        first_physical_slot=first_physical_slot,
    )
    suffix = _build_ticket(
        parent_lease_identity_sha256=(
            parent_lease_identity_sha256
        ),
        phase="suffix",
        phase_start_ordinal=prefix_token_count,
        phase_token_count=parent_token_count - prefix_token_count,
        first_write_position=(
            first_write_position + prefix_token_count
        ),
        first_physical_slot=(
            first_physical_slot + prefix_token_count
        ),
    )
    return prefix, suffix


@dataclass
class ExactBurstPhaseTransfer:
    ticket: ExactBurstPublicationTicket
    mailbox_generation: int
    token_count: int
    byte_count: int
    completion: object
    mailbox: object
    _tokens: Optional[tuple[int, ...]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.ticket,
            ExactBurstPublicationTicket,
        ):
            raise ValueError(
                "phase transfer ticket has an invalid type"
            )
        _require_non_negative_int(
            self.mailbox_generation,
            "mailbox_generation",
        )
        _require_positive_int(self.token_count, "token_count")
        _require_non_negative_int(self.byte_count, "byte_count")
        if not callable(
            getattr(self.completion, "synchronize", None)
        ):
            raise ValueError(
                "phase transfer completion must synchronize"
            )
        if not callable(getattr(self.mailbox, "tolist", None)):
            raise ValueError(
                "phase transfer mailbox must expose tolist"
            )

    def wait_tokens(self) -> tuple[int, ...]:
        if self._tokens is None:
            self.completion.synchronize()
            values = self.mailbox.tolist()
            if (
                not isinstance(values, (list, tuple))
                or len(values) != self.token_count
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in values
                )
            ):
                raise RuntimeError(
                    f"{self.ticket.phase} mailbox token inventory "
                    "is invalid"
                )
            self._tokens = tuple(values)
        return self._tokens


@dataclass(frozen=True)
class ExactGreedyDecodeBurstSplitResult:
    parent_lease_identity_sha256: str
    graph_identity_sha256: str
    replay_count: int
    prefix: ExactBurstPhaseTransfer
    suffix: ExactBurstPhaseTransfer

    def __post_init__(self) -> None:
        _require_digest(
            self.parent_lease_identity_sha256,
            "parent_lease_identity_sha256",
        )
        _require_digest(
            self.graph_identity_sha256,
            "graph_identity_sha256",
        )
        _require_positive_int(self.replay_count, "replay_count")
        for value, name in (
            (self.prefix, "prefix"),
            (self.suffix, "suffix"),
        ):
            if not isinstance(value, ExactBurstPhaseTransfer):
                raise ValueError(
                    f"split result {name} transfer has an invalid type"
                )


class ExactBurstSplitPhaseMailboxBackend:
    """Own two pinned mailboxes for one in-flight split transaction."""

    def __init__(
        self,
        *,
        copy_stream,
        prefix_mailbox,
        suffix_mailbox,
        event_factory,
        current_stream,
        stream_context,
        synchronize=None,
    ):
        if not callable(event_factory):
            raise ValueError("event_factory must be callable")
        if not callable(current_stream):
            raise ValueError("current_stream must be callable")
        if not callable(stream_context):
            raise ValueError("stream_context must be callable")
        if synchronize is not None and not callable(synchronize):
            raise ValueError("synchronize must be callable or None")
        if not callable(getattr(copy_stream, "wait_event", None)):
            raise ValueError("copy_stream must wait on events")
        for mailbox, name in (
            (prefix_mailbox, "prefix_mailbox"),
            (suffix_mailbox, "suffix_mailbox"),
        ):
            if not callable(getattr(mailbox, "copy_", None)):
                raise ValueError(f"{name} must expose copy_")
            if not callable(getattr(mailbox, "tolist", None)):
                raise ValueError(f"{name} must expose tolist")
        self.copy_stream = copy_stream
        self.prefix_mailbox = prefix_mailbox
        self.suffix_mailbox = suffix_mailbox
        self._event_factory = event_factory
        self._current_stream = current_stream
        self._stream_context = stream_context
        self._synchronize = synchronize
        self._generation = 0
        self._active_generation: Optional[int] = None
        self._enqueued_phases: set[str] = set()

    @property
    def active_generation(self) -> Optional[int]:
        return self._active_generation

    def begin_transaction(self) -> int:
        if self._active_generation is not None:
            raise RuntimeError(
                "split-phase mailboxes are already owned"
            )
        self._generation += 1
        self._active_generation = self._generation
        self._enqueued_phases.clear()
        return self._generation

    def enqueue_phase(
        self,
        *,
        ticket: ExactBurstPublicationTicket,
        token_slice,
        mailbox_generation: int,
    ) -> ExactBurstPhaseTransfer:
        if not isinstance(ticket, ExactBurstPublicationTicket):
            raise ValueError(
                "phase ticket has an invalid type"
            )
        if mailbox_generation != self._active_generation:
            raise ValueError(
                "mailbox generation does not own the active "
                "transaction"
            )
        if ticket.phase in self._enqueued_phases:
            raise ValueError(
                f"{ticket.phase} transfer was already enqueued"
            )
        mailbox = (
            self.prefix_mailbox
            if ticket.phase == "prefix"
            else self.suffix_mailbox
        )
        producer = self._event_factory(
            f"{ticket.phase}_compute_done"
        )
        producer.record(self._current_stream())
        self.copy_stream.wait_event(producer)
        completion = self._event_factory(
            f"{ticket.phase}_copy_done"
        )
        with self._stream_context(self.copy_stream):
            mailbox.copy_(token_slice, non_blocking=True)
            completion.record(self.copy_stream)
        self._enqueued_phases.add(ticket.phase)
        return ExactBurstPhaseTransfer(
            ticket=ticket,
            mailbox_generation=mailbox_generation,
            token_count=ticket.phase_token_count,
            byte_count=ticket.phase_token_count * 8,
            completion=completion,
            mailbox=mailbox,
        )

    @staticmethod
    def build_tickets(
        *,
        parent_lease_identity_sha256: str,
        first_write_position: int,
        first_physical_slot: int,
        parent_token_count: int,
        prefix_token_count: int,
    ) -> tuple[
        ExactBurstPublicationTicket,
        ExactBurstPublicationTicket,
    ]:
        return build_exact_burst_publication_tickets(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            first_write_position=first_write_position,
            first_physical_slot=first_physical_slot,
            parent_token_count=parent_token_count,
            prefix_token_count=prefix_token_count,
        )

    @staticmethod
    def build_result(
        *,
        parent_lease_identity_sha256: str,
        graph_identity_sha256: str,
        replay_count: int,
        prefix: ExactBurstPhaseTransfer,
        suffix: ExactBurstPhaseTransfer,
    ) -> ExactGreedyDecodeBurstSplitResult:
        result = ExactGreedyDecodeBurstSplitResult(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            graph_identity_sha256=graph_identity_sha256,
            replay_count=replay_count,
            prefix=prefix,
            suffix=suffix,
        )
        return validate_exact_burst_split_result(
            result,
            expected_parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            expected_graph_identity_sha256=(
                graph_identity_sha256
            ),
        )

    def abort_transaction(self, mailbox_generation: int) -> None:
        if mailbox_generation != self._active_generation:
            raise ValueError(
                "mailbox generation does not own the active "
                "transaction"
            )
        if self._synchronize is not None:
            self._synchronize()
        self.release_transaction(mailbox_generation)

    def release_transaction(self, mailbox_generation: int) -> None:
        if mailbox_generation != self._active_generation:
            raise ValueError(
                "mailbox generation does not own the active "
                "transaction"
            )
        self._active_generation = None
        self._enqueued_phases.clear()


def _validate_ticket_identity(
    ticket: ExactBurstPublicationTicket,
) -> None:
    expected = _ticket_identity(
        parent_lease_identity_sha256=(
            ticket.parent_lease_identity_sha256
        ),
        phase=ticket.phase,
        phase_start_ordinal=ticket.phase_start_ordinal,
        phase_token_count=ticket.phase_token_count,
        first_write_position=ticket.first_write_position,
        last_write_position=ticket.last_write_position,
        first_physical_slot=ticket.first_physical_slot,
        last_physical_slot=ticket.last_physical_slot,
    )
    if ticket.identity_sha256 != expected:
        raise ValueError(
            f"{ticket.phase} ticket identity mismatch"
        )


def validate_exact_burst_split_result(
    result,
    *,
    expected_parent_lease_identity_sha256: str,
    expected_graph_identity_sha256: str,
) -> ExactGreedyDecodeBurstSplitResult:
    if not isinstance(
        result,
        ExactGreedyDecodeBurstSplitResult,
    ):
        raise ValueError("split result has an invalid type")
    _require_digest(
        expected_parent_lease_identity_sha256,
        "expected_parent_lease_identity_sha256",
    )
    _require_digest(
        expected_graph_identity_sha256,
        "expected_graph_identity_sha256",
    )
    if (
        result.parent_lease_identity_sha256
        != expected_parent_lease_identity_sha256
    ):
        raise ValueError(
            "split result parent lease identity mismatch"
        )
    if (
        result.graph_identity_sha256
        != expected_graph_identity_sha256
    ):
        raise ValueError("split result graph identity mismatch")
    if result.replay_count != 8:
        raise ValueError(
            "split result replay_count must equal 8"
        )
    prefix = result.prefix
    suffix = result.suffix
    if prefix.ticket.phase != "prefix":
        raise ValueError("prefix ticket phase mismatch")
    if suffix.ticket.phase != "suffix":
        raise ValueError("suffix ticket phase mismatch")
    for transfer, name in (
        (prefix, "prefix"),
        (suffix, "suffix"),
    ):
        ticket = transfer.ticket
        if (
            ticket.parent_lease_identity_sha256
            != result.parent_lease_identity_sha256
        ):
            raise ValueError(
                f"{name} ticket parent lease identity mismatch"
            )
        _validate_ticket_identity(ticket)
        if transfer.token_count != ticket.phase_token_count:
            raise ValueError(
                f"{name} transfer token_count mismatch"
            )
        if transfer.byte_count != transfer.token_count * 8:
            raise ValueError(
                f"{name} transfer byte_count mismatch"
            )
    if (
        prefix.ticket.phase_start_ordinal != 0
        or suffix.ticket.phase_start_ordinal
        != (
            prefix.ticket.phase_start_ordinal
            + prefix.ticket.phase_token_count
        )
        or suffix.ticket.first_write_position
        != prefix.ticket.last_write_position + 1
        or suffix.ticket.first_physical_slot
        != prefix.ticket.last_physical_slot + 1
    ):
        raise ValueError(
            "split publication ranges are not contiguous"
        )
    if (
        prefix.ticket.phase_token_count
        + suffix.ticket.phase_token_count
        != result.replay_count
    ):
        raise ValueError(
            "split publication ranges are not exhaustive"
        )
    return result


@dataclass
class ExactBurstSplitPhaseTransaction:
    parent_lease_identity_sha256: str
    result: ExactGreedyDecodeBurstSplitResult
    state: str = "enqueued"
    failure_reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        parent_lease_identity_sha256: str,
        result: ExactGreedyDecodeBurstSplitResult,
    ) -> "ExactBurstSplitPhaseTransaction":
        validate_exact_burst_split_result(
            result,
            expected_parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            expected_graph_identity_sha256=(
                result.graph_identity_sha256
            ),
        )
        return cls(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            result=result,
        )

    def _transition(self, expected: str, target: str) -> None:
        if self.state != expected:
            raise ValueError(
                "split transaction state must be "
                f"{expected}, got {self.state}"
            )
        self.state = target

    def mark_prefix_ready(self) -> None:
        self._transition("enqueued", "prefix_ready")

    def mark_prefix_committed(self) -> None:
        self._transition(
            "prefix_ready",
            "prefix_committed",
        )

    def mark_suffix_ready(self) -> None:
        self._transition(
            "prefix_committed",
            "suffix_ready",
        )

    def mark_suffix_committed(self) -> None:
        self._transition(
            "suffix_ready",
            "suffix_committed",
        )

    def mark_pre_prefix_failed(self, reason: str) -> None:
        if self.state not in ("enqueued", "prefix_ready"):
            raise ValueError(
                "pre-prefix failure requires enqueued or "
                "prefix_ready state"
            )
        self.failure_reason = _require_reason(
            reason,
            "failure reason",
        )
        self.state = "pre_prefix_failed"

    def mark_post_prefix_failed(self, reason: str) -> None:
        if self.state not in (
            "prefix_committed",
            "suffix_ready",
        ):
            raise ValueError(
                "post-prefix failure requires prefix_committed "
                "or suffix_ready state"
            )
        self.failure_reason = _require_reason(
            reason,
            "failure reason",
        )
        self.state = "post_prefix_failed"
