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
