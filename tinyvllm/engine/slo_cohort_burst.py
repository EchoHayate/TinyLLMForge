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
    prefill_start_ns: int | None = None
    prefill_complete_ns: int | None = None
    host_visible_token_timestamps_ns: tuple[int, ...] = ()
    output_token_ids: tuple[int, ...] = ()

    def validate(self) -> None:
        _require_int(self.sequence_id, "sequence_id")
        _require_int(self.arrival_ns, "arrival_ns")
        _require_non_empty_string(
            self.service_class,
            "service_class",
        )
        first = self.first_token_visible_ns
        last = self.last_token_visible_ns
        timestamps = self.host_visible_token_timestamps_ns
        if not isinstance(timestamps, tuple):
            raise ValueError(
                "host-visible token timestamps must be a tuple"
            )
        prior = None
        for timestamp in timestamps:
            _require_int(
                timestamp,
                "host_visible_token_timestamp_ns",
            )
            if timestamp < self.arrival_ns:
                raise ValueError(
                    "host-visible token timestamp precedes arrival"
                )
            if prior is not None and timestamp < prior:
                raise ValueError(
                    "host-visible token timestamps regressed"
                )
            prior = timestamp
        if not isinstance(self.output_token_ids, tuple) or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in self.output_token_ids
        ):
            raise ValueError("output token IDs must be non-negative integers")
        if self.output_token_ids and (
            len(self.output_token_ids) != len(timestamps)
        ):
            raise ValueError(
                "output token IDs must match visible timestamps"
            )
        for name, timestamp in (
            ("prefill_start_ns", self.prefill_start_ns),
            ("prefill_complete_ns", self.prefill_complete_ns),
        ):
            if timestamp is not None:
                _require_int(timestamp, name)
                if timestamp < self.arrival_ns:
                    raise ValueError(
                        f"{name} precedes request arrival"
                    )
        if (
            self.prefill_start_ns is not None
            and self.prefill_complete_ns is not None
            and self.prefill_complete_ns < self.prefill_start_ns
        ):
            raise ValueError(
                "prefill completion precedes prefill start"
            )
        if first is None:
            if last is not None or timestamps:
                raise ValueError(
                    "token timeline requires a first token"
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
        if (
            self.prefill_complete_ns is not None
            and first < self.prefill_complete_ns
        ):
            raise ValueError(
                "first token precedes prefill completion"
            )
        if timestamps and (
            timestamps[0] != first or timestamps[-1] != last
        ):
            raise ValueError(
                "host-visible token timeline endpoints mismatch"
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
        candidates = []
        for (
            entry_batch_size,
            entry_context_bucket,
            entry_burst_width,
            predicted_cost_ns,
        ) in self._predicted_cost_entries:
            if (
                entry_batch_size == batch_size
                and entry_burst_width == burst_width
                and entry_context_bucket >= context_bucket
            ):
                candidates.append((
                    entry_context_bucket,
                    predicted_cost_ns,
                ))
        if not candidates:
            return None
        return min(candidates)[1]


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


@dataclass(frozen=True)
class SLOCohortProtectedRequestTelemetry:
    sequence_id: int
    category: str
    service_class: str | None
    age_ns: int | None
    slack_ns: int | None


@dataclass(frozen=True)
class SLOCohortBurstDecisionTelemetry:
    decision_now_ns: int
    schedule_generation: int
    batch_size: int
    ordered_cohort_sequence_ids: tuple[int, ...]
    queue_depths: tuple[tuple[str, int], ...]
    context_buckets: tuple[tuple[int, int, int, int], ...]
    protected_requests: tuple[
        SLOCohortProtectedRequestTelemetry,
        ...,
    ]
    global_slack_ns: int
    predicted_cost_ns_by_width: tuple[tuple[int, int], ...]
    structural_eligibility_by_width: tuple[tuple[int, bool], ...]
    selected_width: int
    reason: str
    cost_table_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "slo-cohort-burst.decision.v1",
            "decision_now_ns": self.decision_now_ns,
            "schedule_generation": self.schedule_generation,
            "batch_size": self.batch_size,
            "ordered_cohort_sequence_ids": list(
                self.ordered_cohort_sequence_ids
            ),
            "queue_depths": dict(self.queue_depths),
            "context_buckets": [
                {
                    "sequence_id": sequence_id,
                    "context_bucket": context_bucket,
                    "remaining_output_tokens": remaining_output_tokens,
                    "writable_tokens": writable_tokens,
                }
                for (
                    sequence_id,
                    context_bucket,
                    remaining_output_tokens,
                    writable_tokens,
                )
                in self.context_buckets
            ],
            "protected_requests": [
                {
                    "sequence_id": request.sequence_id,
                    "category": request.category,
                    "service_class": request.service_class,
                    "age_ns": request.age_ns,
                    "slack_ns": request.slack_ns,
                }
                for request in self.protected_requests
            ],
            "global_slack_ns": self.global_slack_ns,
            "predicted_cost_ns_by_width": dict(
                self.predicted_cost_ns_by_width
            ),
            "structural_eligibility_by_width": dict(
                self.structural_eligibility_by_width
            ),
            "selected_width": self.selected_width,
            "reason": self.reason,
            "cost_table_sha256": self.cost_table_sha256,
        }


@dataclass(frozen=True)
class SLOCohortRequestTelemetry:
    request_id: str
    sequence_id: int
    service_class: str
    arrival_ns: int
    prefill_start_ns: int
    prefill_complete_ns: int
    first_token_visible_ns: int
    host_visible_token_timestamps_ns: tuple[int, ...]
    completion_ns: int
    output_token_ids: tuple[int, ...]
    output_text_sha256: str
    terminal_reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "slo-cohort-burst.request.v1",
            "request_id": self.request_id,
            "sequence_id": self.sequence_id,
            "service_class": self.service_class,
            "arrival_ns": self.arrival_ns,
            "prefill_start_ns": self.prefill_start_ns,
            "prefill_complete_ns": self.prefill_complete_ns,
            "first_token_visible_ns": self.first_token_visible_ns,
            "token_visible_ns": list(
                self.host_visible_token_timestamps_ns
            ),
            "completion_ns": self.completion_ns,
            "output_token_ids": list(self.output_token_ids),
            "output_text_sha256": self.output_text_sha256,
            "terminal_reason": self.terminal_reason,
        }


def _request_age_ns(
    request: ProtectedRequestSnapshot,
    observation: SLOCohortBurstObservation,
) -> int | None:
    state = request.slo_state
    if state is None:
        return None
    anchor = (
        state.last_token_visible_ns
        if state.first_token_visible_ns is not None
        else state.arrival_ns
    )
    if anchor is None or anchor > observation.decision_now_ns:
        return None
    return observation.decision_now_ns - anchor


def build_slo_cohort_decision_telemetry(
    *,
    schedule_generation: int,
    observation: SLOCohortBurstObservation,
    decision: SLOCohortBurstDecision,
    cost_table_sha256: str,
) -> SLOCohortBurstDecisionTelemetry:
    _require_int(
        schedule_generation,
        "schedule_generation",
        minimum=1,
    )
    _validate_observation(observation)
    if not isinstance(decision, SLOCohortBurstDecision):
        raise ValueError("cohort decision has an invalid type")
    digest_valid = (
        len(cost_table_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in cost_table_sha256
        )
    )
    if not digest_valid and not (
        cost_table_sha256 == ""
        and decision.reason == "cost_table_invalid"
    ):
        raise ValueError("cost table SHA-256 is invalid")
    requests = _protected_requests(observation)
    protected_rows = []
    for request in requests:
        state = request.slo_state
        age_ns = _request_age_ns(request, observation)
        slack_ns = None
        if state is not None and age_ns is not None:
            slack_ns = _request_slack_ns(request, observation)
        protected_rows.append(
            SLOCohortProtectedRequestTelemetry(
                sequence_id=request.sequence_id,
                category=request.category,
                service_class=(
                    state.service_class if state is not None else None
                ),
                age_ns=age_ns,
                slack_ns=slack_ns,
            )
        )
    structural_base = (
        observation.enabled
        and observation.clock_valid
        and observation.all_greedy
        and not observation.mixed_mode_unsupported
        and observation.graph_available
        and not observation.graph_quarantined
        and not observation.pending_lease
        and observation.cohort_shape_supported
        and bool(observation.cohort)
    )
    structural_eligibility = tuple(
        (
            width,
            structural_base
            and width in observation.configured_widths
            and all(
                request.remaining_output_tokens >= width
                and request.writable_tokens >= width
                for request in observation.cohort
            ),
        )
        for width in (8, 4, 2)
    )
    return SLOCohortBurstDecisionTelemetry(
        decision_now_ns=observation.decision_now_ns,
        schedule_generation=schedule_generation,
        batch_size=len(observation.cohort),
        ordered_cohort_sequence_ids=tuple(
            request.sequence_id for request in observation.cohort
        ),
        queue_depths=(
            ("waiting", len(observation.waiting)),
            ("prefilling", len(observation.incomplete_prefill)),
            ("running", (
                len(observation.cohort)
                + len(observation.omitted_runnable_decode)
            )),
        ),
        context_buckets=tuple(
            (
                request.sequence_id,
                request.context_bucket,
                request.remaining_output_tokens,
                request.writable_tokens,
            )
            for request in observation.cohort
        ),
        protected_requests=tuple(protected_rows),
        global_slack_ns=decision.global_slack_ns,
        predicted_cost_ns_by_width=(
            decision.predicted_cost_ns_by_width
        ),
        structural_eligibility_by_width=structural_eligibility,
        selected_width=decision.selected_width,
        reason=decision.reason,
        cost_table_sha256=cost_table_sha256,
    )


def build_slo_cohort_request_telemetry(
    *,
    request_id: str,
    sequence_id: int,
    service_class: str,
    arrival_ns: int,
    prefill_start_ns: int,
    prefill_complete_ns: int,
    host_visible_token_timestamps_ns: tuple[int, ...],
    completion_ns: int,
    output_token_ids: tuple[int, ...],
    output_text_sha256: str,
    terminal_reason: str,
) -> SLOCohortRequestTelemetry:
    _require_non_empty_string(request_id, "request_id")
    _require_int(sequence_id, "sequence_id")
    _require_non_empty_string(service_class, "service_class")
    for name, timestamp in (
        ("arrival_ns", arrival_ns),
        ("prefill_start_ns", prefill_start_ns),
        ("prefill_complete_ns", prefill_complete_ns),
        ("completion_ns", completion_ns),
    ):
        _require_int(timestamp, name)
    if not isinstance(host_visible_token_timestamps_ns, tuple) or not (
        host_visible_token_timestamps_ns
    ):
        raise ValueError(
            "host-visible token timestamps must be a non-empty tuple"
        )
    if not isinstance(output_token_ids, tuple) or (
        len(output_token_ids)
        != len(host_visible_token_timestamps_ns)
    ):
        raise ValueError(
            "output token IDs must match host-visible timestamps"
        )
    state = RequestSLOState(
        sequence_id=sequence_id,
        arrival_ns=arrival_ns,
        first_token_visible_ns=(
            host_visible_token_timestamps_ns[0]
        ),
        last_token_visible_ns=(
            host_visible_token_timestamps_ns[-1]
        ),
        service_class=service_class,
        prefill_start_ns=prefill_start_ns,
        prefill_complete_ns=prefill_complete_ns,
        host_visible_token_timestamps_ns=(
            host_visible_token_timestamps_ns
        ),
        output_token_ids=output_token_ids,
    )
    state.validate()
    if completion_ns != host_visible_token_timestamps_ns[-1]:
        raise ValueError(
            "request completion must equal final token visibility"
        )
    _require_non_empty_string(terminal_reason, "terminal_reason")
    if (
        len(output_text_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in output_text_sha256
        )
    ):
        raise ValueError("output text SHA-256 is invalid")
    return SLOCohortRequestTelemetry(
        request_id=request_id,
        sequence_id=sequence_id,
        service_class=service_class,
        arrival_ns=arrival_ns,
        prefill_start_ns=prefill_start_ns,
        prefill_complete_ns=prefill_complete_ns,
        first_token_visible_ns=host_visible_token_timestamps_ns[0],
        host_visible_token_timestamps_ns=(
            host_visible_token_timestamps_ns
        ),
        completion_ns=completion_ns,
        output_token_ids=output_token_ids,
        output_text_sha256=output_text_sha256,
        terminal_reason=terminal_reason,
    )


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
