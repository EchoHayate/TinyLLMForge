"""Dependency-light contracts for phase-stitched exact graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Optional


PARENT_TOKEN_COUNT = 8
AUTHORIZED_DECODE_REPLAY_COUNT = 7
FIRST_TOKEN_ORDINAL = 0
SUFFIX_START_ORDINAL = 1


def _require_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_non_negative_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def _require_reason(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_block_table_identity(value) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(
            "block_table_identity must be a non-empty tuple"
        )
    normalized = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(
                "block_table_identity rows must be pairs"
            )
        block_id, generation = row
        normalized.append((
            _require_non_negative_int(block_id, "block_id"),
            _require_non_negative_int(
                generation,
                "block_generation",
            ),
        ))
    return tuple(normalized)


def _canonical_identity(values: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PhaseStitchLease:
    sequence_id: int
    sequence_generation: int
    schedule_generation: int
    prefill_graph_identity_sha256: str
    prefill_graph_generation: int
    decode_graph_identity_sha256: str
    decode_graph_generation: int
    prompt_token_count: int
    final_prefill_first_position: int
    final_prefill_last_position: int
    initial_completion_count: int
    remaining_output_tokens: int
    decode_first_write_position: int
    decode_last_write_position: int
    decode_first_physical_slot: int
    decode_last_physical_slot: int
    block_table_identity: tuple[tuple[int, int], ...]
    completion_only: bool
    source_identity_sha256: str
    identity_sha256: str
    parent_token_count: int = PARENT_TOKEN_COUNT
    authorized_decode_replay_count: int = (
        AUTHORIZED_DECODE_REPLAY_COUNT
    )
    first_token_ordinal: int = FIRST_TOKEN_ORDINAL
    suffix_start_ordinal: int = SUFFIX_START_ORDINAL


@dataclass(frozen=True)
class PhaseStitchDecision:
    optimized: bool
    fallback_reason: Optional[str]


def decide_phase_stitch_admission(
    *,
    enabled: bool,
    prefill_graph_available: bool,
    decode_graph_available: bool,
    prompt_token_count: int,
    prompt_token_allowlist: tuple[int, ...],
    sequence_count: int,
    waiting_count: int,
    prefilling_count: int,
    do_sample: bool,
    temperatures: tuple[float, ...],
    ignore_eos: tuple[bool, ...],
    completion_only: bool,
    remaining_output_tokens: int,
    decode_kv_capacity_tokens: int,
    tensor_parallel_size: int,
    rank: int,
    incompatible_modes: tuple[str, ...],
    pending_lease: bool,
    quarantined: bool,
) -> PhaseStitchDecision:
    for value, name in (
        (enabled, "enabled"),
        (prefill_graph_available, "prefill_graph_available"),
        (decode_graph_available, "decode_graph_available"),
        (do_sample, "do_sample"),
        (completion_only, "completion_only"),
        (pending_lease, "pending_lease"),
        (quarantined, "quarantined"),
    ):
        _require_bool(value, name)
    for value, name in (
        (prompt_token_count, "prompt_token_count"),
        (sequence_count, "sequence_count"),
        (waiting_count, "waiting_count"),
        (prefilling_count, "prefilling_count"),
        (remaining_output_tokens, "remaining_output_tokens"),
        (decode_kv_capacity_tokens, "decode_kv_capacity_tokens"),
        (tensor_parallel_size, "tensor_parallel_size"),
        (rank, "rank"),
    ):
        _require_non_negative_int(value, name)
    if (
        not isinstance(prompt_token_allowlist, tuple)
        or not prompt_token_allowlist
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in prompt_token_allowlist
        )
    ):
        raise ValueError(
            "prompt_token_allowlist must contain positive integers"
        )
    if (
        not isinstance(temperatures, tuple)
        or not temperatures
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in temperatures
        )
    ):
        raise ValueError(
            "temperatures must match the sequence inventory"
        )
    if (
        not isinstance(ignore_eos, tuple)
        or not ignore_eos
        or any(not isinstance(value, bool) for value in ignore_eos)
    ):
        raise ValueError(
            "ignore_eos must match the sequence inventory"
        )
    if (
        not isinstance(incompatible_modes, tuple)
        or any(
            not isinstance(value, str) or not value
            for value in incompatible_modes
        )
    ):
        raise ValueError(
            "incompatible_modes must contain non-empty strings"
        )

    reason = None
    if not enabled:
        reason = "disabled"
    elif not prefill_graph_available:
        reason = "prefill_graph_unavailable"
    elif not decode_graph_available:
        reason = "decode_graph_unavailable"
    elif prompt_token_count not in prompt_token_allowlist:
        reason = "prompt_shape_not_allowlisted"
    elif sequence_count != 1:
        reason = "sequence_count_unsupported"
    elif len(temperatures) != 1 or len(ignore_eos) != 1:
        reason = "sequence_metadata_mismatch"
    elif waiting_count:
        reason = "waiting_request_present"
    elif prefilling_count:
        reason = "prefilling_request_present"
    elif not do_sample:
        reason = "sampling_unsupported"
    elif any(float(value) != 0.0 for value in temperatures):
        reason = "temperature_nonzero"
    elif not all(ignore_eos):
        reason = "ignore_eos_required"
    elif not completion_only:
        reason = "completion_only_required"
    elif remaining_output_tokens < PARENT_TOKEN_COUNT:
        reason = "output_budget_insufficient"
    elif (
        decode_kv_capacity_tokens
        < AUTHORIZED_DECODE_REPLAY_COUNT
    ):
        reason = "decode_kv_capacity_insufficient"
    elif tensor_parallel_size != 1:
        reason = "tensor_parallel_unsupported"
    elif rank != 0:
        reason = "non_root_rank"
    elif incompatible_modes:
        reason = f"incompatible_mode:{incompatible_modes[0]}"
    elif pending_lease:
        reason = "lease_pending"
    elif quarantined:
        reason = "identity_quarantined"
    return PhaseStitchDecision(
        optimized=reason is None,
        fallback_reason=reason,
    )


def build_phase_stitch_lease(
    *,
    sequence_id: int,
    sequence_generation: int,
    schedule_generation: int,
    prefill_graph_identity_sha256: str,
    prefill_graph_generation: int,
    decode_graph_identity_sha256: str,
    decode_graph_generation: int,
    prompt_token_count: int,
    final_prefill_first_position: int,
    final_prefill_last_position: int,
    initial_completion_count: int,
    remaining_output_tokens: int,
    decode_first_write_position: int,
    decode_last_write_position: int,
    decode_first_physical_slot: int,
    decode_last_physical_slot: int,
    block_table_identity: tuple[tuple[int, int], ...],
    completion_only: bool,
    source_identity_sha256: str,
) -> PhaseStitchLease:
    values = {
        "sequence_id": _require_non_negative_int(
            sequence_id,
            "sequence_id",
        ),
        "sequence_generation": _require_non_negative_int(
            sequence_generation,
            "sequence_generation",
        ),
        "schedule_generation": _require_non_negative_int(
            schedule_generation,
            "schedule_generation",
        ),
        "prefill_graph_identity_sha256": _require_digest(
            prefill_graph_identity_sha256,
            "prefill_graph_identity_sha256",
        ),
        "prefill_graph_generation": _require_non_negative_int(
            prefill_graph_generation,
            "prefill_graph_generation",
        ),
        "decode_graph_identity_sha256": _require_digest(
            decode_graph_identity_sha256,
            "decode_graph_identity_sha256",
        ),
        "decode_graph_generation": _require_non_negative_int(
            decode_graph_generation,
            "decode_graph_generation",
        ),
        "prompt_token_count": _require_positive_int(
            prompt_token_count,
            "prompt_token_count",
        ),
        "final_prefill_first_position": _require_non_negative_int(
            final_prefill_first_position,
            "final_prefill_first_position",
        ),
        "final_prefill_last_position": _require_non_negative_int(
            final_prefill_last_position,
            "final_prefill_last_position",
        ),
        "initial_completion_count": _require_non_negative_int(
            initial_completion_count,
            "initial_completion_count",
        ),
        "remaining_output_tokens": _require_non_negative_int(
            remaining_output_tokens,
            "remaining_output_tokens",
        ),
        "decode_first_write_position": _require_non_negative_int(
            decode_first_write_position,
            "decode_first_write_position",
        ),
        "decode_last_write_position": _require_non_negative_int(
            decode_last_write_position,
            "decode_last_write_position",
        ),
        "decode_first_physical_slot": _require_non_negative_int(
            decode_first_physical_slot,
            "decode_first_physical_slot",
        ),
        "decode_last_physical_slot": _require_non_negative_int(
            decode_last_physical_slot,
            "decode_last_physical_slot",
        ),
        "block_table_identity": _validate_block_table_identity(
            block_table_identity
        ),
        "completion_only": _require_bool(
            completion_only,
            "completion_only",
        ),
        "source_identity_sha256": _require_digest(
            source_identity_sha256,
            "source_identity_sha256",
        ),
    }
    if (
        values["final_prefill_last_position"]
        - values["final_prefill_first_position"]
        + 1
        != values["prompt_token_count"]
    ):
        raise ValueError("prefill position interval is inconsistent")
    if values["decode_first_write_position"] != (
        values["final_prefill_last_position"] + 1
    ):
        raise ValueError(
            "decode write interval must follow final prefill"
        )
    if (
        values["decode_last_write_position"]
        - values["decode_first_write_position"]
        + 1
        != AUTHORIZED_DECODE_REPLAY_COUNT
    ):
        raise ValueError("decode write interval is inconsistent")
    if (
        values["decode_last_physical_slot"]
        - values["decode_first_physical_slot"]
        + 1
        != AUTHORIZED_DECODE_REPLAY_COUNT
    ):
        raise ValueError("decode physical interval is inconsistent")
    if values["remaining_output_tokens"] < PARENT_TOKEN_COUNT:
        raise ValueError(
            "remaining_output_tokens must authorize eight tokens"
        )
    if not values["completion_only"]:
        raise ValueError("phase stitch lease must be completion-only")
    identity_payload = {
        **values,
        "block_table_identity": [
            list(row) for row in values["block_table_identity"]
        ],
        "parent_token_count": PARENT_TOKEN_COUNT,
        "authorized_decode_replay_count": (
            AUTHORIZED_DECODE_REPLAY_COUNT
        ),
        "first_token_ordinal": FIRST_TOKEN_ORDINAL,
        "suffix_start_ordinal": SUFFIX_START_ORDINAL,
    }
    return PhaseStitchLease(
        **values,
        identity_sha256=_canonical_identity(identity_payload),
    )


@dataclass
class PhaseStitchPrefixResult:
    parent_lease_identity_sha256: str
    token: Optional[int]
    token_ordinal: int
    replay_count: int
    d2h_calls: int
    d2h_bytes: int
    completion: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    mailbox: object = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_digest(
            self.parent_lease_identity_sha256,
            "parent_lease_identity_sha256",
        )
        if self.token is not None:
            _require_non_negative_int(self.token, "token")
        elif (
            not callable(
                getattr(self.completion, "synchronize", None)
            )
            or not callable(getattr(self.mailbox, "tolist", None))
        ):
            raise ValueError(
                "unresolved prefix result requires completion "
                "and mailbox"
            )
        for name in (
            "token_ordinal",
            "replay_count",
            "d2h_calls",
            "d2h_bytes",
        ):
            _require_non_negative_int(getattr(self, name), name)

    @property
    def resolved(self) -> bool:
        return self.token is not None

    def wait_token(self) -> int:
        if self.token is None:
            self.completion.synchronize()
            values = self.mailbox.tolist()
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 1
            ):
                raise RuntimeError(
                    "first-token mailbox inventory is invalid"
                )
            token = values[0]
            _require_non_negative_int(token, "token")
            self.token = token
        return self.token


@dataclass
class PhaseStitchSuffixResult:
    parent_lease_identity_sha256: str
    tokens: Optional[tuple[int, ...]]
    first_token_ordinal: int
    replay_count: int
    d2h_calls: int
    d2h_bytes: int
    completion: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    mailbox: object = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_digest(
            self.parent_lease_identity_sha256,
            "parent_lease_identity_sha256",
        )
        if self.tokens is not None:
            self._validate_tokens(self.tokens)
        elif (
            not callable(
                getattr(self.completion, "synchronize", None)
            )
            or not callable(getattr(self.mailbox, "tolist", None))
        ):
            raise ValueError(
                "unresolved suffix result requires completion "
                "and mailbox"
            )
        for name in (
            "first_token_ordinal",
            "replay_count",
            "d2h_calls",
            "d2h_bytes",
        ):
            _require_non_negative_int(getattr(self, name), name)

    @staticmethod
    def _validate_tokens(tokens) -> tuple[int, ...]:
        if (
            not isinstance(tokens, tuple)
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in tokens
            )
        ):
            raise ValueError(
                "suffix tokens must be a tuple of non-negative integers"
            )
        return tokens

    @property
    def resolved(self) -> bool:
        return self.tokens is not None

    def wait_tokens(self) -> tuple[int, ...]:
        if self.tokens is None:
            self.completion.synchronize()
            values = self.mailbox.tolist()
            if not isinstance(values, (list, tuple)):
                raise RuntimeError(
                    "suffix mailbox inventory is invalid"
                )
            tokens = tuple(values)
            try:
                self._validate_tokens(tokens)
            except ValueError as error:
                raise RuntimeError(
                    "suffix mailbox inventory is invalid"
                ) from error
            self.tokens = tokens
        return self.tokens


def validate_phase_stitch_prefix(
    lease: PhaseStitchLease,
    result: PhaseStitchPrefixResult,
) -> PhaseStitchPrefixResult:
    if not isinstance(lease, PhaseStitchLease):
        raise ValueError("phase stitch lease has an invalid type")
    if not isinstance(result, PhaseStitchPrefixResult):
        raise ValueError("phase stitch prefix has an invalid type")
    if result.parent_lease_identity_sha256 != lease.identity_sha256:
        raise ValueError("prefix parent lease identity mismatch")
    result.wait_token()
    if result.token_ordinal != lease.first_token_ordinal:
        raise ValueError("prefix token ordinal mismatch")
    if result.replay_count != 0:
        raise ValueError("prefix cannot report a decode replay")
    if result.d2h_calls != 1 or result.d2h_bytes != 8:
        raise ValueError("prefix D2H accounting mismatch")
    return result


def validate_phase_stitch_suffix(
    lease: PhaseStitchLease,
    result: PhaseStitchSuffixResult,
) -> PhaseStitchSuffixResult:
    if not isinstance(lease, PhaseStitchLease):
        raise ValueError("phase stitch lease has an invalid type")
    if not isinstance(result, PhaseStitchSuffixResult):
        raise ValueError("phase stitch suffix has an invalid type")
    if result.parent_lease_identity_sha256 != lease.identity_sha256:
        raise ValueError("suffix parent lease identity mismatch")
    tokens = result.wait_tokens()
    if result.first_token_ordinal != lease.suffix_start_ordinal:
        raise ValueError("suffix first token ordinal mismatch")
    if len(tokens) != lease.authorized_decode_replay_count:
        raise ValueError("suffix token count mismatch")
    if result.replay_count != lease.authorized_decode_replay_count:
        raise ValueError("suffix replay count mismatch")
    if result.d2h_calls != 1 or result.d2h_bytes != 56:
        raise ValueError("suffix D2H accounting mismatch")
    return result


class PhaseStitchMailboxBackend:
    """Own independent pinned mailboxes for one 1/7 transaction."""

    def __init__(
        self,
        *,
        copy_stream,
        first_token_mailbox,
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
            (first_token_mailbox, "first_token_mailbox"),
            (suffix_mailbox, "suffix_mailbox"),
        ):
            if not callable(getattr(mailbox, "copy_", None)):
                raise ValueError(f"{name} must expose copy_")
            if not callable(getattr(mailbox, "tolist", None)):
                raise ValueError(f"{name} must expose tolist")
        self.copy_stream = copy_stream
        self.first_token_mailbox = first_token_mailbox
        self.suffix_mailbox = suffix_mailbox
        self._event_factory = event_factory
        self._current_stream = current_stream
        self._stream_context = stream_context
        self._synchronize = synchronize
        self._generation = 0
        self._active_generation: Optional[int] = None
        self._active_parent_identity_sha256: Optional[str] = None
        self._enqueued_phases: set[str] = set()
        self._owned_results: dict[
            str,
            PhaseStitchPrefixResult | PhaseStitchSuffixResult,
        ] = {}

    @property
    def active_generation(self) -> Optional[int]:
        return self._active_generation

    def begin_transaction(
        self,
        parent_lease_identity_sha256: str,
    ) -> int:
        _require_digest(
            parent_lease_identity_sha256,
            "parent_lease_identity_sha256",
        )
        if self._active_generation is not None:
            raise RuntimeError(
                "phase stitch mailboxes are already owned"
            )
        self._generation += 1
        self._active_generation = self._generation
        self._active_parent_identity_sha256 = (
            parent_lease_identity_sha256
        )
        self._enqueued_phases.clear()
        self._owned_results.clear()
        return self._generation

    def _validate_enqueue(
        self,
        *,
        phase: str,
        parent_lease_identity_sha256: str,
        mailbox_generation: int,
    ) -> None:
        _require_digest(
            parent_lease_identity_sha256,
            "parent_lease_identity_sha256",
        )
        if mailbox_generation != self._active_generation:
            raise ValueError(
                "mailbox generation does not own the active "
                "transaction"
            )
        if (
            parent_lease_identity_sha256
            != self._active_parent_identity_sha256
        ):
            raise ValueError(
                "parent lease identity does not own the active "
                "transaction"
            )
        if phase in self._enqueued_phases:
            raise ValueError(
                f"{phase} transfer was already enqueued"
            )

    def _enqueue_copy(
        self,
        *,
        phase: str,
        token_slice,
        mailbox,
    ):
        producer = self._event_factory(
            f"{phase}_compute_done"
        )
        producer.record(self._current_stream())
        self.copy_stream.wait_event(producer)
        completion = self._event_factory(
            f"{phase}_copy_done"
        )
        with self._stream_context(self.copy_stream):
            mailbox.copy_(token_slice, non_blocking=True)
            completion.record(self.copy_stream)
        self._enqueued_phases.add(phase)
        return completion

    def enqueue_first_token(
        self,
        *,
        parent_lease_identity_sha256: str,
        token_slice,
        mailbox_generation: int,
    ) -> PhaseStitchPrefixResult:
        phase = "first_token"
        self._validate_enqueue(
            phase=phase,
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            mailbox_generation=mailbox_generation,
        )
        completion = self._enqueue_copy(
            phase=phase,
            token_slice=token_slice,
            mailbox=self.first_token_mailbox,
        )
        result = PhaseStitchPrefixResult(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            token=None,
            token_ordinal=FIRST_TOKEN_ORDINAL,
            replay_count=0,
            d2h_calls=1,
            d2h_bytes=8,
            completion=completion,
            mailbox=self.first_token_mailbox,
        )
        self._owned_results[phase] = result
        return result

    def enqueue_suffix(
        self,
        *,
        parent_lease_identity_sha256: str,
        token_slice,
        mailbox_generation: int,
    ) -> PhaseStitchSuffixResult:
        phase = "suffix"
        self._validate_enqueue(
            phase=phase,
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            mailbox_generation=mailbox_generation,
        )
        completion = self._enqueue_copy(
            phase=phase,
            token_slice=token_slice,
            mailbox=self.suffix_mailbox,
        )
        result = PhaseStitchSuffixResult(
            parent_lease_identity_sha256=(
                parent_lease_identity_sha256
            ),
            tokens=None,
            first_token_ordinal=SUFFIX_START_ORDINAL,
            replay_count=AUTHORIZED_DECODE_REPLAY_COUNT,
            d2h_calls=1,
            d2h_bytes=AUTHORIZED_DECODE_REPLAY_COUNT * 8,
            completion=completion,
            mailbox=self.suffix_mailbox,
        )
        self._owned_results[phase] = result
        return result

    def _validate_owner(self, mailbox_generation: int) -> None:
        if mailbox_generation != self._active_generation:
            raise ValueError(
                "mailbox generation does not own the active "
                "transaction"
            )

    def _release_unchecked(self) -> None:
        self._active_generation = None
        self._active_parent_identity_sha256 = None
        self._enqueued_phases.clear()
        self._owned_results.clear()

    def release_transaction(
        self,
        mailbox_generation: int,
    ) -> None:
        self._validate_owner(mailbox_generation)
        if set(self._owned_results) != {
            "first_token",
            "suffix",
        } or not all(
            result.resolved
            for result in self._owned_results.values()
        ):
            raise RuntimeError(
                "phase stitch mailboxes have unfinished transfers"
            )
        self._release_unchecked()

    def abort_transaction(
        self,
        mailbox_generation: int,
    ) -> None:
        self._validate_owner(mailbox_generation)
        if self._synchronize is not None:
            self._synchronize()
        else:
            for result in self._owned_results.values():
                result.completion.synchronize()
        self._release_unchecked()


class PhaseStitchTransaction:
    """Host-side lifecycle for one authoritative stitched parent lease."""

    def __init__(self, lease: PhaseStitchLease):
        if not isinstance(lease, PhaseStitchLease):
            raise ValueError("phase stitch lease has an invalid type")
        self.lease = lease
        self.state = "created"
        self.last_authoritative_phase = "created"
        self.completed_decode_replays = 0
        self.failure_reason: Optional[str] = None
        self.partial_visibility = False

    def _transition(
        self,
        *,
        expected: str,
        target: str,
        action: str,
    ) -> None:
        if self.state != expected:
            raise ValueError(
                f"{action} requires state {expected}, got {self.state}"
            )
        self.state = target
        self.last_authoritative_phase = target

    def mark_replay_started(self) -> None:
        self._transition(
            expected="created",
            target="replay_started",
            action="replay start",
        )

    def mark_prefix_ready(self) -> None:
        self._transition(
            expected="replay_started",
            target="prefix_ready",
            action="prefix ready",
        )

    def mark_prefix_committed(self) -> None:
        self._transition(
            expected="prefix_ready",
            target="prefix_committed",
            action="prefix commit",
        )

    def mark_suffix_ready(self, *, replay_count: int) -> None:
        if replay_count != self.lease.authorized_decode_replay_count:
            raise ValueError("suffix replay count mismatch")
        self._transition(
            expected="prefix_committed",
            target="suffix_ready",
            action="suffix ready",
        )
        self.completed_decode_replays = replay_count

    def mark_suffix_committed(self) -> None:
        self._transition(
            expected="suffix_ready",
            target="suffix_committed",
            action="suffix commit",
        )

    def close(self) -> None:
        self._transition(
            expected="suffix_committed",
            target="closed",
            action="close",
        )
        self.last_authoritative_phase = "suffix_committed"

    def cancel(self, reason: str) -> None:
        _require_reason(reason, "cancellation reason")
        if self.state != "created":
            raise ValueError(
                "cannot cancel a phase stitch after replay starts"
            )
        self.state = "cancelled"
        self.last_authoritative_phase = "cancelled"
        self.failure_reason = reason

    def fail(self, reason: str) -> None:
        _require_reason(reason, "failure reason")
        if self.state in ("replay_started", "prefix_ready"):
            self.state = "failed_before_prefix"
            self.partial_visibility = False
        elif self.state in ("prefix_committed", "suffix_ready"):
            self.state = "failed_after_prefix"
            self.partial_visibility = True
        else:
            raise ValueError(
                f"cannot fail phase stitch from state {self.state}"
            )
        self.last_authoritative_phase = self.state
        self.failure_reason = reason
