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
