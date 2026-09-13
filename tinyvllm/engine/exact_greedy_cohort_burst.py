"""Atomic contracts for exact-greedy multi-request decode bursts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real


SUPPORTED_WIDTHS = (2, 4, 8)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to "
            f"{minimum}"
        )
    return value


def _require_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_reason(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_block_identities(
    values: object,
    name: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    normalized = []
    seen = set()
    for row in values:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(
                f"{name} rows must be block/generation pairs"
            )
        block_id = _require_int(row[0], f"{name} block ID")
        generation = _require_int(
            row[1],
            f"{name} generation",
        )
        if block_id in seen:
            raise ValueError(f"{name} contains duplicate block IDs")
        seen.add(block_id)
        normalized.append((block_id, generation))
    return tuple(normalized)


@dataclass(frozen=True)
class CohortWriteAuthority:
    sequence_id: int
    sequence_generation: int
    block_table_identity: tuple[tuple[int, int], ...]
    writable_block_identities: tuple[tuple[int, int], ...]
    first_write_position: int
    last_write_position: int
    first_physical_slot: int
    last_physical_slot: int
    initial_completion_count: int
    initial_sequence_length: int
    remaining_output_tokens: int

    def validate(self, authorized_width: int) -> None:
        for name in (
            "sequence_id",
            "sequence_generation",
            "first_write_position",
            "last_write_position",
            "first_physical_slot",
            "last_physical_slot",
            "initial_completion_count",
            "remaining_output_tokens",
        ):
            _require_int(getattr(self, name), name)
        _require_int(
            self.initial_sequence_length,
            "initial_sequence_length",
            minimum=1,
        )
        table = _validate_block_identities(
            self.block_table_identity,
            "block_table_identity",
        )
        writable = _validate_block_identities(
            self.writable_block_identities,
            "writable_block_identities",
        )
        if not set(writable).issubset(set(table)):
            raise ValueError(
                "writable block identities are outside block table"
            )
        if self.remaining_output_tokens < authorized_width:
            raise ValueError(
                "authorized width exceeds remaining output budget"
            )
        if self.first_write_position != (
            self.initial_sequence_length - 1
        ):
            raise ValueError(
                "first write position is inconsistent"
            )
        if self.last_write_position != (
            self.first_write_position + authorized_width - 1
        ):
            raise ValueError(
                "logical write range does not match authorized width"
            )
        if self.last_physical_slot != (
            self.first_physical_slot + authorized_width - 1
        ):
            raise ValueError(
                "physical write range does not match authorized width"
            )


@dataclass(frozen=True)
class ExactGreedyCohortBurstLease:
    schedule_generation: int
    graph_generation: int
    graph_identity_sha256: str
    ordered_sequence_ids: tuple[int, ...]
    requested_width: int
    authorized_width: int
    decision_now_ns: int
    cost_table_sha256: str
    predicted_duration_ns: int
    global_slack_ns: int
    rows: tuple[CohortWriteAuthority, ...]
    identity_sha256: str


def _lease_payload(
    *,
    schedule_generation: int,
    graph_generation: int,
    graph_identity_sha256: str,
    requested_width: int,
    authorized_width: int,
    decision_now_ns: int,
    cost_table_sha256: str,
    predicted_duration_ns: int,
    global_slack_ns: int,
    rows: tuple[CohortWriteAuthority, ...],
) -> dict[str, object]:
    return {
        "schema_version": "exact-greedy-cohort-burst.lease.v1",
        "schedule_generation": schedule_generation,
        "graph_generation": graph_generation,
        "graph_identity_sha256": graph_identity_sha256,
        "ordered_sequence_ids": [
            row.sequence_id for row in rows
        ],
        "requested_width": requested_width,
        "authorized_width": authorized_width,
        "decision_now_ns": decision_now_ns,
        "cost_table_sha256": cost_table_sha256,
        "predicted_duration_ns": predicted_duration_ns,
        "global_slack_ns": global_slack_ns,
        "rows": [asdict(row) for row in rows],
    }


def build_exact_greedy_cohort_burst_lease(
    *,
    schedule_generation: int,
    graph_generation: int,
    graph_identity_sha256: str,
    requested_width: int,
    authorized_width: int,
    decision_now_ns: int,
    cost_table_sha256: str,
    predicted_duration_ns: int,
    global_slack_ns: int,
    rows: tuple[CohortWriteAuthority, ...],
) -> ExactGreedyCohortBurstLease:
    _require_int(
        schedule_generation,
        "schedule_generation",
        minimum=1,
    )
    _require_int(
        graph_generation,
        "graph_generation",
        minimum=1,
    )
    _require_digest(
        graph_identity_sha256,
        "graph_identity_sha256",
    )
    _require_digest(cost_table_sha256, "cost_table_sha256")
    for name, width in (
        ("requested_width", requested_width),
        ("authorized_width", authorized_width),
    ):
        _require_int(width, name, minimum=1)
        if width not in SUPPORTED_WIDTHS:
            raise ValueError(f"{name} is unsupported")
    if authorized_width > requested_width:
        raise ValueError(
            "authorized width exceeds requested width"
        )
    _require_int(decision_now_ns, "decision_now_ns")
    _require_int(
        predicted_duration_ns,
        "predicted_duration_ns",
        minimum=1,
    )
    _require_int(global_slack_ns, "global_slack_ns", minimum=1)
    if predicted_duration_ns > global_slack_ns:
        raise ValueError(
            "predicted duration exceeds global slack"
        )
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("cohort rows must be a non-empty tuple")
    if any(not isinstance(row, CohortWriteAuthority) for row in rows):
        raise ValueError("cohort row has an invalid type")
    for row in rows:
        row.validate(authorized_width)
    sequence_ids = tuple(row.sequence_id for row in rows)
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("cohort contains duplicate sequence IDs")
    ordered_ranges = sorted(
        (
            row.first_physical_slot,
            row.last_physical_slot,
            row.sequence_id,
        )
        for row in rows
    )
    for prior, current in zip(ordered_ranges, ordered_ranges[1:]):
        if current[0] <= prior[1]:
            raise ValueError(
                "cohort physical write authorities overlap"
            )
    payload = _lease_payload(
        schedule_generation=schedule_generation,
        graph_generation=graph_generation,
        graph_identity_sha256=graph_identity_sha256,
        requested_width=requested_width,
        authorized_width=authorized_width,
        decision_now_ns=decision_now_ns,
        cost_table_sha256=cost_table_sha256,
        predicted_duration_ns=predicted_duration_ns,
        global_slack_ns=global_slack_ns,
        rows=rows,
    )
    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return ExactGreedyCohortBurstLease(
        schedule_generation=schedule_generation,
        graph_generation=graph_generation,
        graph_identity_sha256=graph_identity_sha256,
        ordered_sequence_ids=sequence_ids,
        requested_width=requested_width,
        authorized_width=authorized_width,
        decision_now_ns=decision_now_ns,
        cost_table_sha256=cost_table_sha256,
        predicted_duration_ns=predicted_duration_ns,
        global_slack_ns=global_slack_ns,
        rows=rows,
        identity_sha256=identity_sha256,
    )


def _validate_lease_identity(
    lease: ExactGreedyCohortBurstLease,
) -> None:
    rebuilt = build_exact_greedy_cohort_burst_lease(
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
    if rebuilt.identity_sha256 != lease.identity_sha256:
        raise ValueError("cohort lease identity mismatch")
    if rebuilt.ordered_sequence_ids != lease.ordered_sequence_ids:
        raise ValueError(
            "cohort lease ordered sequence IDs mismatch"
        )


@dataclass(frozen=True)
class ExactGreedyCohortBurstRowResult:
    sequence_id: int
    sequence_generation: int
    tokens: tuple[int, ...]
    final_position: int
    final_context_length: int
    final_physical_slot: int
    sampled_logits: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class ExactGreedyCohortBurstResult:
    lease_identity_sha256: str
    graph_identity_sha256: str
    graph_generation: int
    replay_count: int
    rows: tuple[ExactGreedyCohortBurstRowResult, ...]
    token_d2h_calls: int
    sampled_logit_d2h_calls: int


@dataclass(frozen=True)
class ValidatedExactGreedyCohortBurstPublication:
    ordered_sequence_ids: tuple[int, ...]
    commit_tokens: tuple[tuple[int, ...], ...]
    wasted_post_eos_tokens: int
    wasted_post_eos_forwards: int


@dataclass(frozen=True)
class ExactGreedyCohortBurstExecutionTelemetry:
    lease_identity_sha256: str
    result_identity_sha256: str | None
    graph_identity_sha256: str
    requested_width: int
    authorized_width: int
    completed_replay_count: int
    predicted_duration_ns: int
    actual_duration_ns: int
    host_visible_publication_gap_ns: int
    token_d2h_calls: int
    token_d2h_bytes: int
    sampled_logit_d2h_calls: int
    generated_token_counts: tuple[tuple[int, int], ...]
    committed_token_counts: tuple[tuple[int, int], ...]
    eos_discarded_token_counts: tuple[tuple[int, int], ...]
    post_eos_wasted_tokens: int
    post_eos_wasted_forwards: int
    post_eos_wasted_forward_fraction: float
    fallback_reason: str | None
    failure_reason: str | None
    rollback_reason: str | None
    quarantined: bool
    quarantine_reason: str | None
    pending_inventory: tuple[tuple[str, int], ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "exact-greedy-cohort-burst.execution.v1"
            ),
            "lease_identity_sha256": self.lease_identity_sha256,
            "result_identity_sha256": self.result_identity_sha256,
            "graph_identity_sha256": self.graph_identity_sha256,
            "requested_width": self.requested_width,
            "authorized_width": self.authorized_width,
            "completed_replay_count": self.completed_replay_count,
            "predicted_duration_ns": self.predicted_duration_ns,
            "actual_duration_ns": self.actual_duration_ns,
            "host_visible_publication_gap_ns": (
                self.host_visible_publication_gap_ns
            ),
            "token_d2h_calls": self.token_d2h_calls,
            "token_d2h_bytes": self.token_d2h_bytes,
            "sampled_logit_d2h_calls": (
                self.sampled_logit_d2h_calls
            ),
            "generated_token_counts": dict(
                self.generated_token_counts
            ),
            "committed_token_counts": dict(
                self.committed_token_counts
            ),
            "eos_discarded_token_counts": dict(
                self.eos_discarded_token_counts
            ),
            "post_eos_wasted_tokens": self.post_eos_wasted_tokens,
            "post_eos_wasted_forwards": (
                self.post_eos_wasted_forwards
            ),
            "post_eos_wasted_forward_fraction": (
                self.post_eos_wasted_forward_fraction
            ),
            "fallback_reason": self.fallback_reason,
            "failure_reason": self.failure_reason,
            "rollback_reason": self.rollback_reason,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "pending_inventory": dict(self.pending_inventory),
        }


def build_exact_greedy_cohort_burst_execution_telemetry(
    *,
    lease: ExactGreedyCohortBurstLease,
    result: ExactGreedyCohortBurstResult,
    publication: ValidatedExactGreedyCohortBurstPublication,
    actual_duration_ns: int,
    host_visible_publication_gap_ns: int,
    token_d2h_bytes: int,
    quarantine_reason: str | None,
    fallback_reason: str | None,
    failure_reason: str | None,
    rollback_reason: str | None,
    pending_lease_count: int,
    pending_transaction_count: int,
) -> ExactGreedyCohortBurstExecutionTelemetry:
    if not isinstance(lease, ExactGreedyCohortBurstLease):
        raise ValueError("cohort lease has an invalid type")
    _validate_lease_identity(lease)
    if not isinstance(result, ExactGreedyCohortBurstResult):
        raise ValueError("cohort result has an invalid type")
    if not isinstance(
        publication,
        ValidatedExactGreedyCohortBurstPublication,
    ):
        raise ValueError("cohort publication has an invalid type")
    for name, value in (
        ("actual_duration_ns", actual_duration_ns),
        (
            "host_visible_publication_gap_ns",
            host_visible_publication_gap_ns,
        ),
        ("token_d2h_bytes", token_d2h_bytes),
        ("pending_lease_count", pending_lease_count),
        ("pending_transaction_count", pending_transaction_count),
    ):
        _require_int(value, name)
    for name, reason in (
        ("quarantine_reason", quarantine_reason),
        ("fallback_reason", fallback_reason),
        ("failure_reason", failure_reason),
        ("rollback_reason", rollback_reason),
    ):
        if reason is not None:
            _require_reason(reason, name)
    generated = tuple(
        (row.sequence_id, len(row.tokens))
        for row in result.rows
    )
    committed = tuple(
        (sequence_id, len(tokens))
        for sequence_id, tokens in zip(
            publication.ordered_sequence_ids,
            publication.commit_tokens,
        )
    )
    committed_by_sequence = dict(committed)
    discarded = tuple(
        (
            sequence_id,
            count - committed_by_sequence[sequence_id],
        )
        for sequence_id, count in generated
    )
    result_payload = {
        "schema_version": (
            "exact-greedy-cohort-burst.result-identity.v1"
        ),
        "lease_identity_sha256": result.lease_identity_sha256,
        "graph_identity_sha256": result.graph_identity_sha256,
        "graph_generation": result.graph_generation,
        "replay_count": result.replay_count,
        "rows": [asdict(row) for row in result.rows],
        "token_d2h_calls": result.token_d2h_calls,
        "sampled_logit_d2h_calls": (
            result.sampled_logit_d2h_calls
        ),
    }
    result_identity = hashlib.sha256(
        _canonical_json_bytes(result_payload)
    ).hexdigest()
    total_forward_slots = (
        result.replay_count * len(result.rows)
    )
    waste_fraction = (
        publication.wasted_post_eos_forwards
        / total_forward_slots
        if total_forward_slots
        else 0.0
    )
    return ExactGreedyCohortBurstExecutionTelemetry(
        lease_identity_sha256=lease.identity_sha256,
        result_identity_sha256=result_identity,
        graph_identity_sha256=result.graph_identity_sha256,
        requested_width=lease.requested_width,
        authorized_width=lease.authorized_width,
        completed_replay_count=result.replay_count,
        predicted_duration_ns=lease.predicted_duration_ns,
        actual_duration_ns=actual_duration_ns,
        host_visible_publication_gap_ns=(
            host_visible_publication_gap_ns
        ),
        token_d2h_calls=result.token_d2h_calls,
        token_d2h_bytes=token_d2h_bytes,
        sampled_logit_d2h_calls=(
            result.sampled_logit_d2h_calls
        ),
        generated_token_counts=generated,
        committed_token_counts=committed,
        eos_discarded_token_counts=discarded,
        post_eos_wasted_tokens=(
            publication.wasted_post_eos_tokens
        ),
        post_eos_wasted_forwards=(
            publication.wasted_post_eos_forwards
        ),
        post_eos_wasted_forward_fraction=waste_fraction,
        fallback_reason=fallback_reason,
        failure_reason=failure_reason,
        rollback_reason=rollback_reason,
        quarantined=quarantine_reason is not None,
        quarantine_reason=quarantine_reason,
        pending_inventory=(
            ("leases", pending_lease_count),
            ("transactions", pending_transaction_count),
        ),
    )


def build_terminal_exact_greedy_cohort_burst_execution_telemetry(
    *,
    lease: ExactGreedyCohortBurstLease,
    completed_replay_count: int,
    actual_duration_ns: int,
    host_visible_publication_gap_ns: int,
    fallback_reason: str | None,
    failure_reason: str | None,
    rollback_reason: str | None,
    quarantine_reason: str | None,
    pending_lease_count: int,
    pending_transaction_count: int,
) -> ExactGreedyCohortBurstExecutionTelemetry:
    if not isinstance(lease, ExactGreedyCohortBurstLease):
        raise ValueError("cohort lease has an invalid type")
    _validate_lease_identity(lease)
    for name, value in (
        ("completed_replay_count", completed_replay_count),
        ("actual_duration_ns", actual_duration_ns),
        (
            "host_visible_publication_gap_ns",
            host_visible_publication_gap_ns,
        ),
        ("pending_lease_count", pending_lease_count),
        ("pending_transaction_count", pending_transaction_count),
    ):
        _require_int(value, name)
    if completed_replay_count > lease.authorized_width:
        raise ValueError(
            "completed replay count exceeds authorization"
        )
    reasons = (
        ("fallback_reason", fallback_reason),
        ("failure_reason", failure_reason),
        ("rollback_reason", rollback_reason),
        ("quarantine_reason", quarantine_reason),
    )
    for name, reason in reasons:
        if reason is not None:
            _require_reason(reason, name)
    if not any(reason is not None for _name, reason in reasons):
        raise ValueError("terminal telemetry requires a terminal reason")
    return ExactGreedyCohortBurstExecutionTelemetry(
        lease_identity_sha256=lease.identity_sha256,
        result_identity_sha256=None,
        graph_identity_sha256=lease.graph_identity_sha256,
        requested_width=lease.requested_width,
        authorized_width=lease.authorized_width,
        completed_replay_count=completed_replay_count,
        predicted_duration_ns=lease.predicted_duration_ns,
        actual_duration_ns=actual_duration_ns,
        host_visible_publication_gap_ns=(
            host_visible_publication_gap_ns
        ),
        token_d2h_calls=0,
        token_d2h_bytes=0,
        sampled_logit_d2h_calls=0,
        generated_token_counts=(),
        committed_token_counts=(),
        eos_discarded_token_counts=(),
        post_eos_wasted_tokens=0,
        post_eos_wasted_forwards=0,
        post_eos_wasted_forward_fraction=0.0,
        fallback_reason=fallback_reason,
        failure_reason=failure_reason,
        rollback_reason=rollback_reason,
        quarantined=quarantine_reason is not None,
        quarantine_reason=quarantine_reason,
        pending_inventory=(
            ("leases", pending_lease_count),
            ("transactions", pending_transaction_count),
        ),
    )


@dataclass(frozen=True)
class ExactGreedyCohortBurstFallback:
    fallback_reason: str
    replay_count: int = 0

    def __post_init__(self) -> None:
        _require_reason(self.fallback_reason, "fallback_reason")
        _require_int(self.replay_count, "replay_count")
        if self.replay_count:
            raise ValueError(
                "cohort fallback cannot follow a graph replay"
            )


@dataclass(frozen=True)
class ExactGreedyCohortBurstGraphReceipt:
    graph_identity_sha256: str
    graph_generation: int
    batch_size: int
    block_table_width: int
    dtype: str
    device_identity: str
    tensor_parallel_size: int
    correctness_trace: bool
    scratch_block_ids: tuple[int, ...]
    capture_live_kv_mutations: tuple[object, ...]


class ExactGreedyCohortBurstTerminalError(RuntimeError):
    def __init__(self, reason: str, completed_replays: int):
        super().__init__(reason)
        self.reason = reason
        self.completed_replays = completed_replays


class ExactGreedyCohortBurstGraph:
    _REQUIRED_SHAPES = {
        "input_tokens": lambda batch, width: (batch,),
        "positions": lambda batch, width: (batch,),
        "context_lengths": lambda batch, width: (batch,),
        "slot_mappings": lambda batch, width: (batch,),
        "block_tables": lambda batch, width: (batch, width),
        "active_row_masks": lambda batch, width: (batch,),
        "result_bundle": lambda batch, width: (batch, 8, 2),
        "token_history": lambda batch, width: (batch, 8),
        "history_indices": lambda batch, width: (batch,),
        "eos_observations": lambda batch, width: (batch, 8),
    }

    def __init__(
        self,
        *,
        tensors: dict[str, object],
        receipt: ExactGreedyCohortBurstGraphReceipt,
        bind_rows,
        graph_replay,
        read_result_bundle,
        read_sampled_logits,
    ):
        self.tensors = tensors
        self.receipt = receipt
        self._bind_rows = bind_rows
        self._graph_replay = graph_replay
        self._read_result_bundle = read_result_bundle
        self._read_sampled_logits = read_sampled_logits
        self.quarantine_reason = None

    @staticmethod
    def _shape(value: object, name: str) -> tuple[int, ...]:
        try:
            shape = tuple(int(size) for size in value.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must expose an integer shape"
            ) from error
        return shape

    @classmethod
    def capture(
        cls,
        *,
        tensors: dict[str, object],
        graph_generation: int,
        batch_size: int,
        block_table_width: int,
        dtype: str,
        device_identity: str,
        tensor_parallel_size: int,
        correctness_trace: bool,
        scratch_block_ids: tuple[int, ...],
        capture_live_kv_mutations: tuple[object, ...],
        bind_rows,
        graph_replay,
        read_result_bundle,
        read_sampled_logits=None,
    ) -> "ExactGreedyCohortBurstGraph":
        _require_int(
            graph_generation,
            "graph_generation",
            minimum=1,
        )
        _require_int(batch_size, "batch_size", minimum=1)
        _require_int(
            block_table_width,
            "block_table_width",
            minimum=1,
        )
        _require_int(
            tensor_parallel_size,
            "tensor_parallel_size",
            minimum=1,
        )
        if tensor_parallel_size != 1:
            raise ValueError("cohort graph currently requires TP1")
        _require_reason(dtype, "dtype")
        _require_reason(device_identity, "device_identity")
        if not isinstance(correctness_trace, bool):
            raise ValueError("correctness_trace must be a bool")
        if not isinstance(tensors, dict):
            raise ValueError("cohort graph tensors must be a dict")
        shape_payload = {}
        for name, expected_shape in cls._REQUIRED_SHAPES.items():
            if name not in tensors:
                raise ValueError(
                    f"missing cohort graph tensor: {name}"
                )
            shape = cls._shape(tensors[name], name)
            expected = expected_shape(
                batch_size,
                block_table_width,
            )
            if shape != expected:
                raise ValueError(
                    f"{name} shape mismatch: {shape} != {expected}"
                )
            shape_payload[name] = list(shape)
        if not isinstance(scratch_block_ids, tuple):
            raise ValueError("scratch_block_ids must be a tuple")
        for block_id in scratch_block_ids:
            _require_int(block_id, "scratch block ID")
        if (
            len(scratch_block_ids) != batch_size
            or len(set(scratch_block_ids)) != batch_size
        ):
            raise ValueError(
                "cohort graph requires one private scratch block per row"
            )
        if capture_live_kv_mutations != ():
            raise RuntimeError(
                "cohort graph capture mutated live KV"
            )
        for callback, name in (
            (bind_rows, "bind_rows"),
            (graph_replay, "graph_replay"),
            (read_result_bundle, "read_result_bundle"),
        ):
            if not callable(callback):
                raise ValueError(f"{name} must be callable")
        if correctness_trace and not callable(read_sampled_logits):
            raise ValueError(
                "correctness graph requires sampled-logit reader"
            )
        identity_payload = {
            "schema_version": (
                "exact-greedy-cohort-burst.graph.v1"
            ),
            "graph_generation": graph_generation,
            "batch_size": batch_size,
            "block_table_width": block_table_width,
            "dtype": dtype,
            "device_identity": device_identity,
            "tensor_parallel_size": tensor_parallel_size,
            "correctness_trace": correctness_trace,
            "scratch_block_ids": list(scratch_block_ids),
            "tensor_shapes": shape_payload,
        }
        identity_sha256 = hashlib.sha256(
            _canonical_json_bytes(identity_payload)
        ).hexdigest()
        receipt = ExactGreedyCohortBurstGraphReceipt(
            graph_identity_sha256=identity_sha256,
            graph_generation=graph_generation,
            batch_size=batch_size,
            block_table_width=block_table_width,
            dtype=dtype,
            device_identity=device_identity,
            tensor_parallel_size=tensor_parallel_size,
            correctness_trace=correctness_trace,
            scratch_block_ids=scratch_block_ids,
            capture_live_kv_mutations=(),
        )
        return cls(
            tensors=tensors,
            receipt=receipt,
            bind_rows=bind_rows,
            graph_replay=graph_replay,
            read_result_bundle=read_result_bundle,
            read_sampled_logits=read_sampled_logits,
        )

    def capability(self) -> dict[str, object]:
        return {
            "available": self.quarantine_reason is None,
            "quarantined": self.quarantine_reason is not None,
            "quarantine_reason": self.quarantine_reason,
            "graph_identity_sha256": (
                self.receipt.graph_identity_sha256
            ),
            "graph_generation": self.receipt.graph_generation,
            "batch_size": self.receipt.batch_size,
            "block_table_width": self.receipt.block_table_width,
            "correctness_trace": self.receipt.correctness_trace,
        }

    def quarantine(self, reason: str) -> None:
        reason = _require_reason(reason, "quarantine reason")
        if self.quarantine_reason is None:
            self.quarantine_reason = reason

    def _terminal(
        self,
        reason: str,
        completed_replays: int,
    ) -> None:
        self.quarantine(reason)
        raise ExactGreedyCohortBurstTerminalError(
            reason,
            completed_replays,
        )

    def replay(
        self,
        lease: ExactGreedyCohortBurstLease,
        *,
        row_bindings: tuple[object, ...],
    ) -> (
        ExactGreedyCohortBurstResult
        | ExactGreedyCohortBurstFallback
    ):
        if self.quarantine_reason is not None:
            return ExactGreedyCohortBurstFallback(
                "graph_quarantined"
            )
        try:
            _validate_lease_identity(lease)
        except (TypeError, ValueError):
            return ExactGreedyCohortBurstFallback(
                "lease_identity_invalid"
            )
        if (
            lease.graph_identity_sha256
            != self.receipt.graph_identity_sha256
        ):
            return ExactGreedyCohortBurstFallback(
                "graph_identity_drift"
            )
        if lease.graph_generation != self.receipt.graph_generation:
            return ExactGreedyCohortBurstFallback(
                "graph_generation_drift"
            )
        if len(lease.rows) != self.receipt.batch_size:
            return ExactGreedyCohortBurstFallback(
                "batch_size_drift"
            )
        if (
            not isinstance(row_bindings, tuple)
            or len(row_bindings) != len(lease.rows)
        ):
            return ExactGreedyCohortBurstFallback(
                "row_binding_count_mismatch"
            )
        try:
            self._bind_rows(lease, row_bindings)
        except Exception:
            return ExactGreedyCohortBurstFallback(
                "row_bind_failure"
            )

        completed_replays = 0
        for _ in range(lease.authorized_width):
            try:
                self._graph_replay()
            except Exception as error:
                self._terminal(
                    "graph replay failed: "
                    f"{type(error).__name__}",
                    completed_replays,
                )
            completed_replays += 1
        try:
            raw_history, raw_eos_observations = (
                self._read_result_bundle()
            )
            history = tuple(
                tuple(int(token) for token in row)
                for row in raw_history
            )
            eos_observations = tuple(
                tuple(bool(value) for value in row)
                for row in raw_eos_observations
            )
            sampled_logits = (
                tuple(
                    tuple(
                        tuple(float(value) for value in values)
                        for values in row
                    )
                    for row in self._read_sampled_logits()
                )
                if self.receipt.correctness_trace
                else tuple(() for _ in lease.rows)
            )
        except Exception as error:
            self._terminal(
                "cohort result D2H failed: "
                f"{type(error).__name__}",
                completed_replays,
            )
        if (
            len(history) != len(lease.rows)
            or len(eos_observations) != len(lease.rows)
            or any(
                len(row) < lease.authorized_width
                for row in history
            )
            or any(
                len(row) < lease.authorized_width
                for row in eos_observations
            )
            or len(sampled_logits) != len(lease.rows)
        ):
            self._terminal(
                "cohort result shape mismatch",
                completed_replays,
            )
        result_rows = tuple(
            ExactGreedyCohortBurstRowResult(
                sequence_id=authority.sequence_id,
                sequence_generation=(
                    authority.sequence_generation
                ),
                tokens=history[index][
                    :lease.authorized_width
                ],
                final_position=(
                    authority.first_write_position
                    + lease.authorized_width
                ),
                final_context_length=(
                    authority.initial_sequence_length
                    + lease.authorized_width
                ),
                final_physical_slot=(
                    authority.last_physical_slot + 1
                ),
                sampled_logits=sampled_logits[index][
                    :lease.authorized_width
                ],
            )
            for index, authority in enumerate(lease.rows)
        )
        return ExactGreedyCohortBurstResult(
            lease_identity_sha256=lease.identity_sha256,
            graph_identity_sha256=(
                self.receipt.graph_identity_sha256
            ),
            graph_generation=self.receipt.graph_generation,
            replay_count=completed_replays,
            rows=result_rows,
            token_d2h_calls=1,
            sampled_logit_d2h_calls=int(
                self.receipt.correctness_trace
            ),
        )


def _validate_tokens(
    tokens: object,
    replay_count: int,
) -> tuple[int, ...]:
    if not isinstance(tokens, tuple) or len(tokens) != replay_count:
        raise ValueError(
            "row token count does not match replay count"
        )
    for token in tokens:
        _require_int(token, "token")
    return tokens


def _validate_correctness_trace(
    row: ExactGreedyCohortBurstRowResult,
    *,
    correctness_trace: bool,
) -> None:
    logits = row.sampled_logits
    if not isinstance(logits, tuple):
        raise ValueError("sampled logits must be a tuple")
    if not correctness_trace:
        if logits:
            raise ValueError(
                "production cohort result cannot contain logits"
            )
        return
    if len(logits) != len(row.tokens):
        raise ValueError(
            "sampled logit count does not match token count"
        )
    for token, values in zip(row.tokens, logits):
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
        argmax_token = max(
            range(len(values)),
            key=lambda index: values[index],
        )
        if argmax_token != token:
            raise ValueError(
                "sampled logits argmax does not match token"
            )


def validate_exact_greedy_cohort_burst_result(
    lease: ExactGreedyCohortBurstLease,
    result: ExactGreedyCohortBurstResult,
    *,
    eos_token_id: int,
    correctness_trace: bool = False,
) -> ValidatedExactGreedyCohortBurstPublication:
    if not isinstance(lease, ExactGreedyCohortBurstLease):
        raise ValueError("cohort lease has an invalid type")
    _validate_lease_identity(lease)
    if not isinstance(result, ExactGreedyCohortBurstResult):
        raise ValueError("cohort result has an invalid type")
    if not isinstance(correctness_trace, bool):
        raise ValueError("correctness_trace must be a bool")
    _require_int(eos_token_id, "eos_token_id")
    _require_digest(
        result.lease_identity_sha256,
        "lease_identity_sha256",
    )
    if result.lease_identity_sha256 != lease.identity_sha256:
        raise ValueError("cohort result lease identity mismatch")
    _require_digest(
        result.graph_identity_sha256,
        "graph_identity_sha256",
    )
    if result.graph_identity_sha256 != lease.graph_identity_sha256:
        raise ValueError("cohort result graph identity mismatch")
    _require_int(
        result.graph_generation,
        "graph_generation",
        minimum=1,
    )
    if result.graph_generation != lease.graph_generation:
        raise ValueError("cohort result graph generation mismatch")
    _require_int(result.replay_count, "replay_count", minimum=1)
    if result.replay_count != lease.authorized_width:
        raise ValueError(
            "cohort result replay count does not match lease"
        )
    if (
        not isinstance(result.rows, tuple)
        or len(result.rows) != len(lease.rows)
    ):
        raise ValueError("cohort result row count mismatch")
    result_sequence_ids = tuple(
        row.sequence_id for row in result.rows
    )
    if result_sequence_ids != lease.ordered_sequence_ids:
        raise ValueError(
            "cohort result ordered sequence IDs mismatch"
        )
    _require_int(result.token_d2h_calls, "token_d2h_calls")
    if result.token_d2h_calls != 1:
        raise ValueError(
            "cohort result must contain one token D2H"
        )
    _require_int(
        result.sampled_logit_d2h_calls,
        "sampled_logit_d2h_calls",
    )
    expected_logit_calls = int(correctness_trace)
    if result.sampled_logit_d2h_calls != expected_logit_calls:
        raise ValueError("sampled logit D2H count mismatch")

    commit_tokens = []
    wasted_post_eos_tokens = 0
    for authority, row in zip(lease.rows, result.rows):
        if not isinstance(row, ExactGreedyCohortBurstRowResult):
            raise ValueError("cohort result row has an invalid type")
        if row.sequence_generation != authority.sequence_generation:
            raise ValueError("row sequence generation mismatch")
        tokens = _validate_tokens(row.tokens, result.replay_count)
        for name, value in (
            ("final_position", row.final_position),
            ("final_context_length", row.final_context_length),
            ("final_physical_slot", row.final_physical_slot),
        ):
            _require_int(value, name)
        if row.final_position != (
            authority.first_write_position + result.replay_count
        ):
            raise ValueError("row final position mismatch")
        if row.final_context_length != (
            authority.initial_sequence_length
            + result.replay_count
        ):
            raise ValueError("row final context length mismatch")
        if row.final_physical_slot != (
            authority.last_physical_slot + 1
        ):
            raise ValueError("row final physical slot mismatch")
        _validate_correctness_trace(
            row,
            correctness_trace=correctness_trace,
        )
        try:
            eos_index = tokens.index(eos_token_id)
        except ValueError:
            committed = tokens
        else:
            committed = tokens[:eos_index + 1]
            wasted_post_eos_tokens += (
                len(tokens) - len(committed)
            )
        commit_tokens.append(committed)
    return ValidatedExactGreedyCohortBurstPublication(
        ordered_sequence_ids=lease.ordered_sequence_ids,
        commit_tokens=tuple(commit_tokens),
        wasted_post_eos_tokens=wasted_post_eos_tokens,
        wasted_post_eos_forwards=wasted_post_eos_tokens,
    )


class ExactGreedyCohortBurstTransaction:
    def __init__(self, lease: ExactGreedyCohortBurstLease):
        if not isinstance(lease, ExactGreedyCohortBurstLease):
            raise ValueError("cohort lease has an invalid type")
        _validate_lease_identity(lease)
        self.lease = lease
        self.state = "reserved"
        self.result = None
        self.publication = None
        self.fallback = None
        self.failure_reason = None
        self.completed_replays = 0
        self.quarantined = False

    @property
    def pending(self) -> bool:
        return self.state in (
            "reserved",
            "dispatched",
            "validated",
        )

    def _require_state(self, expected: str) -> None:
        if self.state != expected:
            raise RuntimeError(
                f"cohort transaction is {self.state}, "
                f"expected {expected}"
            )

    def dispatch(self) -> None:
        self._require_state("reserved")
        self.state = "dispatched"

    def validate(
        self,
        result: ExactGreedyCohortBurstResult,
        *,
        eos_token_id: int,
        correctness_trace: bool = False,
    ) -> ValidatedExactGreedyCohortBurstPublication:
        self._require_state("dispatched")
        publication = validate_exact_greedy_cohort_burst_result(
            self.lease,
            result,
            eos_token_id=eos_token_id,
            correctness_trace=correctness_trace,
        )
        self.result = result
        self.publication = publication
        self.completed_replays = result.replay_count
        self.state = "validated"
        return publication

    def commit(self) -> None:
        self._require_state("validated")
        self.state = "committed"

    def cancel(
        self,
        fallback: ExactGreedyCohortBurstFallback,
    ) -> None:
        self._require_state("reserved")
        if not isinstance(
            fallback,
            ExactGreedyCohortBurstFallback,
        ):
            raise ValueError("cohort fallback has an invalid type")
        self.fallback = fallback
        self.state = "cancelled"

    def quarantine_and_fail(
        self,
        reason: str,
        *,
        completed_replays: int,
    ) -> None:
        self._require_state("dispatched")
        _require_reason(reason, "failure reason")
        _require_int(
            completed_replays,
            "completed_replays",
            minimum=1,
        )
        if completed_replays > self.lease.authorized_width:
            raise ValueError(
                "completed replays exceed authorized width"
            )
        self.quarantined = True
        self.state = "quarantined"
        self.failure_reason = reason
        self.completed_replays = completed_replays
        self.state = "failed"
