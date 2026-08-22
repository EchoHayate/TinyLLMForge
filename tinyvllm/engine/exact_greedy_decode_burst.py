"""Generic contracts for exact multi-step greedy decode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Optional


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


def _validate_integer_tuple(
    value,
    name: str,
    *,
    non_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    if non_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        for item in value
    ):
        raise ValueError(
            f"{name} must contain non-negative integers"
        )
    return value


def _validate_block_identity(
    value,
) -> tuple[tuple[int, int], ...]:
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
        _require_non_negative_int(block_id, "block id")
        _require_non_negative_int(
            generation,
            "block generation",
        )
        normalized.append((block_id, generation))
    return tuple(normalized)


@dataclass(frozen=True)
class ExactGreedyDecodeBurstDecision:
    optimized: bool
    authorized_token_count: int
    first_write_position: Optional[int]
    last_write_position: Optional[int]
    fallback_reason: Optional[str]
    output_budget_clipped: bool = False
    block_boundary_clipped: bool = False


def build_exact_greedy_decode_burst_decision(
    *,
    enabled: bool,
    configured_width: int,
    remaining_output_tokens: int,
    initial_sequence_length: int,
    block_size: int,
    sequence_count: int,
    waiting_count: int,
    prefilling_count: int,
    is_prefill: bool,
    do_sample: bool,
    batch_kind: Optional[str],
    temperatures: tuple[object, ...],
    ignore_eos: tuple[object, ...],
    completion_only: bool,
    tensor_parallel_size: int,
    rank: int,
    graph_available: bool,
    incompatible_modes: tuple[str, ...],
    pending_lease: bool,
    quarantined: bool,
) -> ExactGreedyDecodeBurstDecision:
    _require_bool(enabled, "enabled")
    if (
        isinstance(configured_width, bool)
        or not isinstance(configured_width, int)
        or configured_width < 2
        or configured_width > 8
    ):
        raise ValueError(
            "configured_width must be an integer in [2, 8]"
        )
    _require_non_negative_int(
        remaining_output_tokens,
        "remaining_output_tokens",
    )
    _require_positive_int(
        initial_sequence_length,
        "initial_sequence_length",
    )
    _require_positive_int(block_size, "block_size")
    _require_non_negative_int(
        sequence_count,
        "sequence_count",
    )
    _require_non_negative_int(waiting_count, "waiting_count")
    _require_non_negative_int(
        prefilling_count,
        "prefilling_count",
    )
    _require_bool(is_prefill, "is_prefill")
    _require_bool(do_sample, "do_sample")
    if batch_kind is not None and not isinstance(batch_kind, str):
        raise ValueError(
            "batch_kind must be a string or None"
        )
    if not isinstance(temperatures, tuple):
        raise ValueError("temperatures must be a tuple")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in temperatures
    ):
        raise ValueError(
            "temperatures must contain finite numbers"
        )
    if not isinstance(ignore_eos, tuple):
        raise ValueError("ignore_eos must be a tuple")
    if any(not isinstance(value, bool) for value in ignore_eos):
        raise ValueError(
            "ignore_eos must contain bool values"
        )
    _require_bool(completion_only, "completion_only")
    _require_positive_int(
        tensor_parallel_size,
        "tensor_parallel_size",
    )
    _require_non_negative_int(rank, "rank")
    _require_bool(graph_available, "graph_available")
    if not isinstance(incompatible_modes, tuple):
        raise ValueError("incompatible_modes must be a tuple")
    if any(
        not isinstance(value, str) or not value
        for value in incompatible_modes
    ):
        raise ValueError(
            "incompatible_modes must contain non-empty strings"
        )
    _require_bool(pending_lease, "pending_lease")
    _require_bool(quarantined, "quarantined")

    first_write_position = initial_sequence_length - 1
    writable_positions = (
        block_size
        - (first_write_position % block_size)
    )
    authorized = min(
        configured_width,
        remaining_output_tokens,
        writable_positions,
    )
    last_write_position = (
        first_write_position + max(0, authorized - 1)
    )
    output_budget_clipped = (
        remaining_output_tokens < configured_width
        and remaining_output_tokens <= writable_positions
    )
    block_boundary_clipped = (
        writable_positions < configured_width
        and writable_positions <= remaining_output_tokens
    )

    reason = None
    if not enabled:
        reason = "disabled"
    elif sequence_count != 1:
        reason = "sequence_count_unsupported"
    elif waiting_count:
        reason = "waiting_present"
    elif prefilling_count:
        reason = "prefilling_present"
    elif is_prefill:
        reason = "prefill_unsupported"
    elif not do_sample:
        reason = "sampling_disabled"
    elif batch_kind is not None:
        reason = "mixed_batch_unsupported"
    elif len(temperatures) != 1:
        reason = "temperature_count_mismatch"
    elif float(temperatures[0]) != 0.0:
        reason = "nonzero_temperature"
    elif len(ignore_eos) != 1:
        reason = "ignore_eos_count_mismatch"
    elif not ignore_eos[0]:
        reason = "eos_sensitive"
    elif not completion_only:
        reason = "visibility_unsupported"
    elif tensor_parallel_size != 1:
        reason = "tensor_parallel_unsupported"
    elif rank != 0:
        reason = "non_root_rank"
    elif not graph_available:
        reason = "graph_unavailable"
    elif incompatible_modes:
        reason = f"incompatible_mode:{incompatible_modes[0]}"
    elif pending_lease:
        reason = "lease_pending"
    elif quarantined:
        reason = "quarantined"
    elif remaining_output_tokens < 2:
        reason = "insufficient_output_budget"
    elif authorized < 2:
        reason = "authorized_width_below_two"

    return ExactGreedyDecodeBurstDecision(
        optimized=reason is None,
        authorized_token_count=authorized,
        first_write_position=first_write_position,
        last_write_position=last_write_position,
        fallback_reason=reason,
        output_budget_clipped=output_budget_clipped,
        block_boundary_clipped=block_boundary_clipped,
    )


@dataclass(frozen=True)
class ExactGreedyDecodeBurstLease:
    sequence_id: int
    schedule_generation: int
    graph_generation: int
    requested_token_count: int
    authorized_token_count: int
    initial_completion_count: int
    initial_sequence_length: int
    block_table_identity: tuple[tuple[int, int], ...]
    write_block_id: int
    write_block_generation: int
    first_write_position: int
    last_write_position: int
    first_physical_slot: int
    last_physical_slot: int
    remaining_output_tokens: int
    completion_only: bool
    identity_sha256: str


def _lease_payload(**values) -> dict:
    return {
        "sequence_id": values["sequence_id"],
        "schedule_generation": values["schedule_generation"],
        "graph_generation": values["graph_generation"],
        "requested_token_count": values[
            "requested_token_count"
        ],
        "authorized_token_count": values[
            "authorized_token_count"
        ],
        "initial_completion_count": values[
            "initial_completion_count"
        ],
        "initial_sequence_length": values[
            "initial_sequence_length"
        ],
        "block_table_identity": [
            list(row) for row in values["block_table_identity"]
        ],
        "write_block_id": values["write_block_id"],
        "write_block_generation": values[
            "write_block_generation"
        ],
        "first_write_position": values[
            "first_write_position"
        ],
        "last_write_position": values["last_write_position"],
        "first_physical_slot": values["first_physical_slot"],
        "last_physical_slot": values["last_physical_slot"],
        "remaining_output_tokens": values[
            "remaining_output_tokens"
        ],
        "completion_only": values["completion_only"],
    }


def build_exact_greedy_decode_burst_lease(
    *,
    sequence_id: int,
    schedule_generation: int,
    graph_generation: int,
    requested_token_count: int,
    authorized_token_count: int,
    initial_completion_count: int,
    initial_sequence_length: int,
    block_table_identity: tuple[tuple[int, int], ...],
    write_block_id: int,
    write_block_generation: int,
    first_write_position: int,
    last_write_position: int,
    first_physical_slot: int,
    last_physical_slot: int,
    remaining_output_tokens: int,
    completion_only: bool,
) -> ExactGreedyDecodeBurstLease:
    for name, value in (
        ("sequence_id", sequence_id),
        ("initial_completion_count", initial_completion_count),
        ("write_block_id", write_block_id),
        ("write_block_generation", write_block_generation),
        ("first_write_position", first_write_position),
        ("last_write_position", last_write_position),
        ("first_physical_slot", first_physical_slot),
        ("last_physical_slot", last_physical_slot),
        ("remaining_output_tokens", remaining_output_tokens),
    ):
        _require_non_negative_int(value, name)
    for name, value in (
        ("schedule_generation", schedule_generation),
        ("graph_generation", graph_generation),
        ("requested_token_count", requested_token_count),
        ("authorized_token_count", authorized_token_count),
        ("initial_sequence_length", initial_sequence_length),
    ):
        _require_positive_int(value, name)
    if authorized_token_count > requested_token_count:
        raise ValueError(
            "authorized token count exceeds requested count"
        )
    if authorized_token_count > remaining_output_tokens:
        raise ValueError(
            "authorized token count exceeds output budget"
        )
    if (
        last_write_position
        != first_write_position + authorized_token_count - 1
    ):
        raise ValueError("lease write positions are inconsistent")
    if (
        last_physical_slot
        != first_physical_slot + authorized_token_count - 1
    ):
        raise ValueError("lease physical slots are inconsistent")
    if first_write_position != initial_sequence_length - 1:
        raise ValueError(
            "lease first write position is inconsistent"
        )
    _require_bool(completion_only, "completion_only")
    if not completion_only:
        raise ValueError("burst lease must be completion-only")
    identities = _validate_block_identity(
        block_table_identity
    )
    if identities[-1] != (
        write_block_id,
        write_block_generation,
    ):
        raise ValueError(
            "write block identity does not match block table"
        )
    values = {
        "sequence_id": sequence_id,
        "schedule_generation": schedule_generation,
        "graph_generation": graph_generation,
        "requested_token_count": requested_token_count,
        "authorized_token_count": authorized_token_count,
        "initial_completion_count": initial_completion_count,
        "initial_sequence_length": initial_sequence_length,
        "block_table_identity": identities,
        "write_block_id": write_block_id,
        "write_block_generation": write_block_generation,
        "first_write_position": first_write_position,
        "last_write_position": last_write_position,
        "first_physical_slot": first_physical_slot,
        "last_physical_slot": last_physical_slot,
        "remaining_output_tokens": remaining_output_tokens,
        "completion_only": completion_only,
    }
    encoded = json.dumps(
        _lease_payload(**values),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ExactGreedyDecodeBurstLease(
        **values,
        identity_sha256=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True)
class ExactGreedyDecodeBurstResult:
    lease_identity_sha256: str
    tokens: tuple[int, ...]
    replay_count: int
    final_input_token: int
    final_position: int
    final_context_length: int
    final_physical_slot: int
    graph_identity_sha256: str
    token_d2h_calls: int
    sampled_logit_d2h_calls: int
    sampled_logits: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = ()


def validate_exact_greedy_decode_burst_result(
    lease: ExactGreedyDecodeBurstLease,
    result: ExactGreedyDecodeBurstResult,
    *,
    correctness_trace: bool = False,
) -> ExactGreedyDecodeBurstResult:
    if not isinstance(lease, ExactGreedyDecodeBurstLease):
        raise ValueError("burst lease has an invalid type")
    if not isinstance(result, ExactGreedyDecodeBurstResult):
        raise ValueError("burst result has an invalid type")
    _require_bool(correctness_trace, "correctness_trace")
    _require_digest(
        result.lease_identity_sha256,
        "lease_identity_sha256",
    )
    if result.lease_identity_sha256 != lease.identity_sha256:
        raise ValueError("burst result lease identity mismatch")
    tokens = _validate_integer_tuple(
        result.tokens,
        "tokens",
        non_empty=True,
    )
    _require_positive_int(result.replay_count, "replay_count")
    if result.replay_count != lease.authorized_token_count:
        raise ValueError(
            "burst result replay count does not match lease"
        )
    if len(tokens) != result.replay_count:
        raise ValueError(
            "burst result token count does not match replay count"
        )
    for name, value in (
        ("final_input_token", result.final_input_token),
        ("final_position", result.final_position),
        ("final_context_length", result.final_context_length),
        ("final_physical_slot", result.final_physical_slot),
        ("token_d2h_calls", result.token_d2h_calls),
        (
            "sampled_logit_d2h_calls",
            result.sampled_logit_d2h_calls,
        ),
    ):
        _require_non_negative_int(value, name)
    if result.final_input_token != tokens[-1]:
        raise ValueError("final input token does not match output")
    if result.final_position != (
        lease.first_write_position + result.replay_count
    ):
        raise ValueError("final position advance mismatch")
    if result.final_context_length != (
        lease.initial_sequence_length + result.replay_count
    ):
        raise ValueError("final context-length advance mismatch")
    if result.final_physical_slot != (
        lease.last_physical_slot + 1
    ):
        raise ValueError("final physical-slot advance mismatch")
    _require_digest(
        result.graph_identity_sha256,
        "graph_identity_sha256",
    )
    if result.token_d2h_calls != 1:
        raise ValueError(
            "burst result must contain one final token D2H"
        )
    if not isinstance(result.sampled_logits, tuple):
        raise ValueError("sampled_logits must be a tuple")
    previous_ordinal = -1
    for row in result.sampled_logits:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(
                "sampled logit rows must be ordinal/value pairs"
            )
        ordinal, values = row
        _require_non_negative_int(
            ordinal,
            "sampled logit ordinal",
        )
        if ordinal <= previous_ordinal:
            raise ValueError(
                "sampled logit ordinals must be strictly increasing"
            )
        if ordinal >= result.replay_count:
            raise ValueError(
                "sampled logit ordinal exceeds replay count"
            )
        if (
            not isinstance(values, tuple)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError(
                "sampled logits must contain finite values"
            )
        previous_ordinal = ordinal
    if not correctness_trace:
        if result.sampled_logits:
            raise ValueError(
                "production burst cannot return sampled logits"
            )
        if result.sampled_logit_d2h_calls != 0:
            raise ValueError(
                "production burst cannot transfer sampled logits"
            )
    else:
        expected_calls = int(bool(result.sampled_logits))
        if result.sampled_logit_d2h_calls != expected_calls:
            raise ValueError(
                "sampled logit D2H count mismatch"
            )
    return result


@dataclass(frozen=True)
class ExactGreedyDecodeBurstCaptureReceipt:
    graph_identity_sha256: str
    graph_generation: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    retained_static_bytes: int
    scratch_block_count: int
    correctness_trace: bool

    def __post_init__(self) -> None:
        _require_digest(
            self.graph_identity_sha256,
            "graph_identity_sha256",
        )
        _require_positive_int(
            self.graph_generation,
            "graph_generation",
        )
        for name in (
            "capture_duration_ns",
            "allocated_delta_bytes",
            "reserved_delta_bytes",
            "retained_static_bytes",
            "scratch_block_count",
        ):
            _require_non_negative_int(getattr(self, name), name)
        _require_bool(
            self.correctness_trace,
            "correctness_trace",
        )


@dataclass
class ExactGreedyDecodeBurstStats:
    attempts: int = 0
    acceptances: int = 0
    target_model_forwards: int = 0
    graph_replays: int = 0
    intermediate_token_d2h_calls: int = 0
    final_token_d2h_calls: int = 0
    final_token_d2h_bytes: int = 0
    sampled_logit_d2h_calls: int = 0
    output_budget_clipped: int = 0
    block_boundary_clipped: int = 0
    commits: int = 0
    committed_tokens: int = 0
    failures: int = 0
    quarantines: int = 0
    pending_leases: int = 0
    maximum_host_visible_gap_ns: int = 0
    requested_width_histogram: dict[int, int] = field(
        default_factory=dict
    )
    authorized_width_histogram: dict[int, int] = field(
        default_factory=dict
    )
    fallback_counts: dict[str, int] = field(default_factory=dict)
    quarantine_reason: Optional[str] = None
    capture_receipts: list[
        ExactGreedyDecodeBurstCaptureReceipt
    ] = field(default_factory=list)

    def record_attempt(self) -> None:
        self.attempts += 1

    def record_acceptance(
        self,
        *,
        requested_token_count: int,
        authorized_token_count: int,
        output_budget_clipped: bool,
        block_boundary_clipped: bool,
    ) -> None:
        _require_positive_int(
            requested_token_count,
            "requested_token_count",
        )
        _require_positive_int(
            authorized_token_count,
            "authorized_token_count",
        )
        _require_bool(
            output_budget_clipped,
            "output_budget_clipped",
        )
        _require_bool(
            block_boundary_clipped,
            "block_boundary_clipped",
        )
        self.acceptances += 1
        self.pending_leases += 1
        self.requested_width_histogram[requested_token_count] = (
            self.requested_width_histogram.get(
                requested_token_count,
                0,
            )
            + 1
        )
        self.authorized_width_histogram[authorized_token_count] = (
            self.authorized_width_histogram.get(
                authorized_token_count,
                0,
            )
            + 1
        )
        self.output_budget_clipped += int(output_budget_clipped)
        self.block_boundary_clipped += int(
            block_boundary_clipped
        )

    def record_fallback(self, reason: str) -> None:
        reason = _require_reason(reason, "fallback reason")
        self.fallback_counts[reason] = (
            self.fallback_counts.get(reason, 0) + 1
        )

    def record_capture(
        self,
        receipt: ExactGreedyDecodeBurstCaptureReceipt,
    ) -> None:
        if not isinstance(
            receipt,
            ExactGreedyDecodeBurstCaptureReceipt,
        ):
            raise ValueError("capture receipt has an invalid type")
        self.capture_receipts.append(receipt)

    def record_replays(self, count: int) -> None:
        _require_positive_int(count, "replay count")
        self.graph_replays += count
        self.target_model_forwards += count

    def record_final_token_d2h(
        self,
        *,
        token_count: int,
        byte_count: int,
    ) -> None:
        _require_positive_int(token_count, "token_count")
        _require_non_negative_int(byte_count, "byte_count")
        self.final_token_d2h_calls += 1
        self.final_token_d2h_bytes += byte_count

    def record_sampled_logit_d2h(self) -> None:
        self.sampled_logit_d2h_calls += 1

    def record_commit(
        self,
        *,
        token_count: int,
        host_visible_gap_ns: int,
    ) -> None:
        _require_positive_int(token_count, "token_count")
        _require_non_negative_int(
            host_visible_gap_ns,
            "host_visible_gap_ns",
        )
        if self.pending_leases <= 0:
            raise ValueError("no pending burst lease to commit")
        self.pending_leases -= 1
        self.commits += 1
        self.committed_tokens += token_count
        self.maximum_host_visible_gap_ns = max(
            self.maximum_host_visible_gap_ns,
            host_visible_gap_ns,
        )

    def record_failure(self, *, terminal: bool) -> None:
        _require_bool(terminal, "terminal")
        self.failures += 1
        if terminal and self.pending_leases:
            self.pending_leases -= 1

    def cancel_pending(self, reason: str) -> None:
        if self.pending_leases <= 0:
            raise ValueError("no pending burst lease to cancel")
        self.pending_leases -= 1
        self.record_fallback(reason)

    def quarantine(self, reason: str) -> None:
        reason = _require_reason(reason, "quarantine reason")
        if self.quarantine_reason is None:
            self.quarantine_reason = reason
            self.quarantines += 1

    def summary(self) -> dict[str, object]:
        payload = {
            "attempts": self.attempts,
            "acceptances": self.acceptances,
            "target_model_forwards": self.target_model_forwards,
            "graph_replays": self.graph_replays,
            "intermediate_token_d2h_calls": (
                self.intermediate_token_d2h_calls
            ),
            "final_token_d2h_calls": self.final_token_d2h_calls,
            "final_token_d2h_bytes": self.final_token_d2h_bytes,
            "sampled_logit_d2h_calls": (
                self.sampled_logit_d2h_calls
            ),
            "output_budget_clipped": self.output_budget_clipped,
            "block_boundary_clipped": (
                self.block_boundary_clipped
            ),
            "commits": self.commits,
            "committed_tokens": self.committed_tokens,
            "failures": self.failures,
            "quarantines": self.quarantines,
            "pending_leases": self.pending_leases,
            "maximum_host_visible_gap_ns": (
                self.maximum_host_visible_gap_ns
            ),
            "requested_width_histogram": {
                str(key): value
                for key, value in sorted(
                    self.requested_width_histogram.items()
                )
            },
            "authorized_width_histogram": {
                str(key): value
                for key, value in sorted(
                    self.authorized_width_histogram.items()
                )
            },
            "fallback_counts": dict(
                sorted(self.fallback_counts.items())
            ),
            "quarantine_reason": self.quarantine_reason,
            "capture_receipts": [
                asdict(receipt)
                for receipt in self.capture_receipts
            ],
        }
        for name, value in payload.items():
            if isinstance(value, int):
                _require_non_negative_int(value, name)
        return payload
