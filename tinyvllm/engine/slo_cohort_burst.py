"""Pure SLO policy for exact-greedy multi-request decode bursts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Mapping


COST_TABLE_SCHEMA_VERSION = "slo-cohort-burst.cost-table.v1"
SUPPORTED_WIDTHS = (1, 2, 4, 8)
PROTECTED_CATEGORIES = (
    "cohort",
    "omitted_decode",
    "waiting",
    "incomplete_prefill",
)


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


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_source_identity(
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "source_commit",
        "source_patch_sha256",
        "model",
        "checkpoint_sha256",
        "gpu_uuid",
        "gpu_name",
        "tensor_parallel_size",
        "dtype",
        "config_sha256",
    }
    if set(source_identity) != required:
        raise ValueError("cost table source identity is incomplete")
    normalized = dict(source_identity)
    for field, lengths in (
        ("source_commit", (40, 64)),
        ("source_patch_sha256", (64,)),
        ("checkpoint_sha256", (64,)),
        ("config_sha256", (64,)),
    ):
        digest = _require_non_empty_string(
            normalized[field],
            field,
        )
        if len(digest) not in lengths or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError(
                f"cost table source identity {field} is invalid"
            )
    for field in ("model", "gpu_uuid", "gpu_name", "dtype"):
        _require_non_empty_string(normalized[field], field)
    if _require_int(
        normalized["tensor_parallel_size"],
        "tensor_parallel_size",
        minimum=1,
    ) != 1:
        raise ValueError("cost table source identity requires TP1")
    return normalized


@dataclass(frozen=True)
class RequestSLOState:
    sequence_id: int
    arrival_ns: int
    first_token_visible_ns: int | None
    last_token_visible_ns: int | None
    service_class: str

    def validate(self) -> None:
        _require_int(self.sequence_id, "sequence_id")
        _require_int(self.arrival_ns, "arrival_ns")
        _require_non_empty_string(
            self.service_class,
            "service_class",
        )
        first = self.first_token_visible_ns
        last = self.last_token_visible_ns
        if first is None:
            if last is not None:
                raise ValueError(
                    "last token timestamp requires a first token"
                )
            return
        _require_int(first, "first_token_visible_ns")
        if first < self.arrival_ns:
            raise ValueError(
                "first token timestamp precedes arrival"
            )
        if last is None:
            raise ValueError(
                "first token timestamp requires a last token"
            )
        _require_int(last, "last_token_visible_ns")
        if last < first:
            raise ValueError(
                "last token timestamp precedes first token"
            )


@dataclass(frozen=True)
class ProtectedRequestSnapshot:
    sequence_id: int
    category: str
    context_bucket: int
    remaining_output_tokens: int
    writable_tokens: int
    slo_state: RequestSLOState | None

    def validate_structure(self) -> None:
        _require_int(self.sequence_id, "sequence_id")
        if self.category not in PROTECTED_CATEGORIES:
            raise ValueError("protected request category is unsupported")
        _require_int(
            self.context_bucket,
            "context_bucket",
            minimum=1,
        )
        _require_int(
            self.remaining_output_tokens,
            "remaining_output_tokens",
        )
        _require_int(self.writable_tokens, "writable_tokens")


@dataclass(frozen=True)
class SLOCohortCostTable:
    _payload_json: str
    _predicted_cost_entries: tuple[
        tuple[int, int, int, int],
        ...,
    ]
    valid: bool = True
    invalid_reason: str | None = None

    @classmethod
    def invalid(cls, reason: str) -> "SLOCohortCostTable":
        return cls(
            _payload_json="{}",
            _predicted_cost_entries=(),
            valid=False,
            invalid_reason=_require_non_empty_string(
                reason,
                "invalid reason",
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
    ) -> "SLOCohortCostTable":
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8")
            )
        except (OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cost table could not be loaded: {error}"
            ) from error
        return cls.from_payload(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "SLOCohortCostTable":
        if not isinstance(payload, Mapping):
            raise ValueError("cost table must be a mapping")
        required = {
            "schema_version",
            "source_identity",
            "entries",
            "table_sha256",
        }
        if set(payload) != required:
            raise ValueError("cost table fields are incomplete")
        if payload["schema_version"] != COST_TABLE_SCHEMA_VERSION:
            raise ValueError("cost table schema is invalid")
        if not isinstance(payload["source_identity"], Mapping):
            raise ValueError("cost table source identity is invalid")
        source_identity = _validate_source_identity(
            payload["source_identity"]
        )
        entries = payload["entries"]
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError("cost table entries are invalid")
        table_sha256 = _require_non_empty_string(
            payload["table_sha256"],
            "table_sha256",
        )
        if len(table_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in table_sha256
        ):
            raise ValueError(
                "table_sha256 must be a lowercase SHA-256 digest"
            )
        normalized_entries = {}
        for name, entry in entries.items():
            if not isinstance(name, str) or not isinstance(entry, Mapping):
                raise ValueError("cost table entry is invalid")
            expected_fields = {
                "batch_size",
                "context_bucket",
                "burst_width",
                "sample_count",
                "raw_sample_sha256",
                "p50_ns",
                "p95_ns",
                "p99_ns",
            }
            if set(entry) != expected_fields:
                raise ValueError("cost table entry fields are incomplete")
            batch_size = _require_int(
                entry["batch_size"],
                "batch_size",
                minimum=1,
            )
            context_bucket = _require_int(
                entry["context_bucket"],
                "context_bucket",
                minimum=1,
            )
            burst_width = _require_int(
                entry["burst_width"],
                "burst_width",
                minimum=1,
            )
            if burst_width not in SUPPORTED_WIDTHS:
                raise ValueError("cost table burst width is unsupported")
            canonical_name = (
                f"b{batch_size}-c{context_bucket}-k{burst_width}"
            )
            if name != canonical_name:
                raise ValueError("cost table entry name is not canonical")
            sample_count = _require_int(
                entry["sample_count"],
                "sample_count",
                minimum=1,
            )
            raw_digest = _require_non_empty_string(
                entry["raw_sample_sha256"],
                "raw_sample_sha256",
            )
            if len(raw_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in raw_digest
            ):
                raise ValueError(
                    "raw_sample_sha256 must be a SHA-256 digest"
                )
            p50_ns = _require_int(
                entry["p50_ns"],
                "p50_ns",
                minimum=1,
            )
            p95_ns = _require_int(
                entry["p95_ns"],
                "p95_ns",
                minimum=1,
            )
            p99_ns = _require_int(
                entry["p99_ns"],
                "p99_ns",
                minimum=1,
            )
            if not p50_ns <= p95_ns <= p99_ns:
                raise ValueError(
                    "cost table percentiles are not monotonic"
                )
            normalized_entries[name] = {
                "batch_size": batch_size,
                "context_bucket": context_bucket,
                "burst_width": burst_width,
                "sample_count": sample_count,
                "raw_sample_sha256": raw_digest,
                "p50_ns": p50_ns,
                "p95_ns": p95_ns,
                "p99_ns": p99_ns,
            }
        normalized_payload = {
            "schema_version": COST_TABLE_SCHEMA_VERSION,
            "source_identity": source_identity,
            "entries": normalized_entries,
        }
        actual_sha256 = hashlib.sha256(
            _canonical_json_bytes(normalized_payload)
        ).hexdigest()
        if actual_sha256 != table_sha256:
            raise ValueError("cost table hash does not match payload")
        normalized_payload["table_sha256"] = table_sha256
        return cls(
            _payload_json=_canonical_json_bytes(
                normalized_payload
            ).decode("utf-8"),
            _predicted_cost_entries=tuple(
                (
                    entry["batch_size"],
                    entry["context_bucket"],
                    entry["burst_width"],
                    entry["p99_ns"],
                )
                for entry in normalized_entries.values()
            ),
        )

    @property
    def table_sha256(self) -> str:
        if not self.valid:
            return ""
        return str(self.to_payload()["table_sha256"])

    def to_payload(self) -> dict[str, object]:
        return json.loads(self._payload_json)

    def predicted_cost_ns(
        self,
        batch_size: int,
        context_bucket: int,
        burst_width: int,
    ) -> int | None:
        if not self.valid:
            return None
        for (
            entry_batch_size,
            entry_context_bucket,
            entry_burst_width,
            predicted_cost_ns,
        ) in self._predicted_cost_entries:
            if (
                entry_batch_size == batch_size
                and entry_context_bucket == context_bucket
                and entry_burst_width == burst_width
            ):
                return predicted_cost_ns
        return None


@dataclass(frozen=True)
class SLOCohortBurstObservation:
    enabled: bool
    decision_now_ns: int
    target_itl_ns: int
    target_ttft_ns: int
    reserve_ns: int
    configured_widths: tuple[int, ...]
    cohort: tuple[ProtectedRequestSnapshot, ...]
    omitted_runnable_decode: tuple[ProtectedRequestSnapshot, ...]
    waiting: tuple[ProtectedRequestSnapshot, ...]
    incomplete_prefill: tuple[ProtectedRequestSnapshot, ...]
    clock_valid: bool
    all_greedy: bool
    mixed_mode_unsupported: bool
    graph_available: bool
    graph_quarantined: bool
    pending_lease: bool
    cohort_shape_supported: bool


@dataclass(frozen=True)
class SLOCohortBurstDecision:
    selected_width: int
    reason: str
    global_slack_ns: int
    predicted_cost_ns_by_width: tuple[tuple[int, int], ...]
    protected_sequence_ids: tuple[int, ...]


def _fallback(
    reason: str,
    *,
    global_slack_ns: int = 0,
    predicted_cost_ns_by_width: tuple[tuple[int, int], ...] = (),
    protected_sequence_ids: tuple[int, ...] = (),
) -> SLOCohortBurstDecision:
    return SLOCohortBurstDecision(
        selected_width=1,
        reason=reason,
        global_slack_ns=global_slack_ns,
        predicted_cost_ns_by_width=predicted_cost_ns_by_width,
        protected_sequence_ids=protected_sequence_ids,
    )


def _protected_requests(
    observation: SLOCohortBurstObservation,
) -> tuple[ProtectedRequestSnapshot, ...]:
    return (
        observation.cohort
        + observation.omitted_runnable_decode
        + observation.waiting
        + observation.incomplete_prefill
    )


def _timestamps_are_valid(
    requests: tuple[ProtectedRequestSnapshot, ...],
    decision_now_ns: int,
) -> bool:
    for request in requests:
        state = request.slo_state
        if state is None:
            continue
        try:
            state.validate()
        except ValueError:
            return False
        timestamps = (
            state.arrival_ns,
            state.first_token_visible_ns,
            state.last_token_visible_ns,
        )
        if any(
            timestamp is not None and timestamp > decision_now_ns
            for timestamp in timestamps
        ):
            return False
    return True


def _request_slack_ns(
    request: ProtectedRequestSnapshot,
    observation: SLOCohortBurstObservation,
) -> int:
    state = request.slo_state
    if state is None:
        raise ValueError("request SLO state is missing")
    if state.first_token_visible_ns is None:
        return (
            observation.target_ttft_ns
            - (observation.decision_now_ns - state.arrival_ns)
            - observation.reserve_ns
        )
    return (
        observation.target_itl_ns
        - (
            observation.decision_now_ns
            - state.last_token_visible_ns
        )
        - observation.reserve_ns
    )


def _validate_observation(
    observation: SLOCohortBurstObservation,
) -> None:
    _require_bool(observation.enabled, "enabled")
    _require_int(observation.decision_now_ns, "decision_now_ns")
    _require_int(observation.target_itl_ns, "target_itl_ns")
    _require_int(observation.target_ttft_ns, "target_ttft_ns")
    _require_int(observation.reserve_ns, "reserve_ns")
    for name in (
        "clock_valid",
        "all_greedy",
        "mixed_mode_unsupported",
        "graph_available",
        "graph_quarantined",
        "pending_lease",
        "cohort_shape_supported",
    ):
        _require_bool(getattr(observation, name), name)
    widths = observation.configured_widths
    if (
        not isinstance(widths, tuple)
        or not widths
        or widths[0] != 1
        or any(
            isinstance(width, bool)
            or width not in SUPPORTED_WIDTHS
            for width in widths
        )
        or tuple(sorted(set(widths))) != widths
    ):
        raise ValueError("configured_widths are invalid")
    groups = (
        observation.cohort,
        observation.omitted_runnable_decode,
        observation.waiting,
        observation.incomplete_prefill,
    )
    if any(not isinstance(group, tuple) for group in groups):
        raise ValueError("protected request groups must be tuples")
    for request in _protected_requests(observation):
        if not isinstance(request, ProtectedRequestSnapshot):
            raise ValueError("protected request snapshot is invalid")
        request.validate_structure()


def select_slo_cohort_burst_width(
    observation: SLOCohortBurstObservation,
    cost_table: SLOCohortCostTable,
) -> SLOCohortBurstDecision:
    """Choose the largest safe width without mutating scheduler state."""

    if not isinstance(observation, SLOCohortBurstObservation):
        raise ValueError("observation has an invalid type")
    if not isinstance(cost_table, SLOCohortCostTable):
        raise ValueError("cost_table has an invalid type")
    _validate_observation(observation)
    requests = _protected_requests(observation)
    protected_sequence_ids = tuple(
        request.sequence_id for request in requests
    )

    if not observation.enabled:
        return _fallback(
            "disabled",
            protected_sequence_ids=protected_sequence_ids,
        )
    if (
        not observation.clock_valid
        or not _timestamps_are_valid(
            requests,
            observation.decision_now_ns,
        )
    ):
        return _fallback(
            "clock_invalid",
            protected_sequence_ids=protected_sequence_ids,
        )
    if any(
        request.slo_state is None
        or request.slo_state.sequence_id != request.sequence_id
        for request in requests
    ):
        return _fallback(
            "missing_slo_state",
            protected_sequence_ids=protected_sequence_ids,
        )

    candidate_widths = tuple(
        width
        for width in (8, 4, 2)
        if width in observation.configured_widths
    )
    predicted_costs = []
    if not cost_table.valid or not observation.cohort:
        return _fallback(
            "cost_table_invalid",
            protected_sequence_ids=protected_sequence_ids,
        )
    batch_size = len(observation.cohort)
    for width in candidate_widths:
        context_costs = tuple(
            cost_table.predicted_cost_ns(
                batch_size,
                request.context_bucket,
                width,
            )
            for request in observation.cohort
        )
        if not context_costs or any(
            cost is None for cost in context_costs
        ):
            return _fallback(
                "cost_table_invalid",
                protected_sequence_ids=protected_sequence_ids,
            )
        predicted_costs.append((width, max(context_costs)))
    predicted_cost_ns_by_width = tuple(predicted_costs)

    if not observation.all_greedy:
        return _fallback(
            "non_greedy_request",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    if observation.mixed_mode_unsupported:
        return _fallback(
            "mixed_mode_unsupported",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    if not observation.graph_available:
        return _fallback(
            "graph_unavailable",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    if observation.graph_quarantined:
        return _fallback(
            "graph_quarantined",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    if observation.pending_lease:
        return _fallback(
            "pending_lease",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    sequence_ids = tuple(
        request.sequence_id for request in observation.cohort
    )
    if (
        not observation.cohort_shape_supported
        or len(sequence_ids) != len(set(sequence_ids))
    ):
        return _fallback(
            "cohort_shape_unsupported",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    if min(
        request.remaining_output_tokens
        for request in observation.cohort
    ) < 2:
        return _fallback(
            "insufficient_output_budget",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )
    if min(
        request.writable_tokens for request in observation.cohort
    ) < 2:
        return _fallback(
            "kv_block_boundary",
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )

    slacks = tuple(
        _request_slack_ns(request, observation)
        for request in requests
    )
    global_slack_ns = min(slacks)
    if global_slack_ns <= 0:
        return _fallback(
            "no_slo_slack",
            global_slack_ns=global_slack_ns,
            predicted_cost_ns_by_width=predicted_cost_ns_by_width,
            protected_sequence_ids=protected_sequence_ids,
        )

    maximum_authorized_width = min(
        min(
            request.remaining_output_tokens
            for request in observation.cohort
        ),
        min(
            request.writable_tokens
            for request in observation.cohort
        ),
    )
    for width, predicted_cost_ns in predicted_cost_ns_by_width:
        if (
            width <= maximum_authorized_width
            and predicted_cost_ns <= global_slack_ns
        ):
            return SLOCohortBurstDecision(
                selected_width=width,
                reason="selected",
                global_slack_ns=global_slack_ns,
                predicted_cost_ns_by_width=(
                    predicted_cost_ns_by_width
                ),
                protected_sequence_ids=protected_sequence_ids,
            )
    return _fallback(
        "predicted_cost_exceeds_slack",
        global_slack_ns=global_slack_ns,
        predicted_cost_ns_by_width=predicted_cost_ns_by_width,
        protected_sequence_ids=protected_sequence_ids,
    )
