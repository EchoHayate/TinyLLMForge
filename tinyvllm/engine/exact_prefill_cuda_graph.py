from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import asdict, dataclass


PREFILL_GRAPH_FALLBACK_REASONS = (
    "feature_disabled",
    "not_prefill",
    "tensor_parallel_unsupported",
    "world_size_unsupported",
    "sequence_count_unsupported",
    "length_mismatch",
    "prefix_block_table_present",
    "token_count_not_allowlisted",
    "input_embeddings_requested",
    "hidden_state_return_requested",
    "cpu_offload_active",
    "kv_offload_active",
    "kv_quantization_active",
    "compact_attention_active",
    "model_forward_unsupported",
    "entry_missing",
    "entry_quarantined",
    "capture_failed",
    "replay_failed",
    "identity_drift",
)


class ExactPrefillGraphReplayError(RuntimeError):
    def __init__(self, identity_sha256: str, cause: BaseException):
        super().__init__(
            "exact prefill CUDA Graph replay failed for "
            f"{identity_sha256}: {cause}"
        )
        self.identity_sha256 = identity_sha256
        self.cause = cause


def _require_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _require_positive_int(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_canonical_allowlist(value: object) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in value
        )
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError(
            "token_allowlist must be a canonical positive tuple"
        )


@dataclass(frozen=True)
class ExactPrefillGraphIdentity:
    token_count: int
    active_batch_size: int
    world_size: int
    model_forward_kind: str
    attention_backend: str
    attention_backend_version: str
    input_dtype: str
    hidden_dtype: str
    num_layers: int
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    device_compute_capability: tuple[int, int]

    def __post_init__(self) -> None:
        for name in (
            "token_count",
            "active_batch_size",
            "world_size",
            "num_layers",
            "hidden_size",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "page_block_size",
        ):
            _require_positive_int(getattr(self, name), name)
        for name in (
            "model_forward_kind",
            "attention_backend",
            "attention_backend_version",
            "input_dtype",
            "hidden_dtype",
        ):
            _require_nonempty_string(getattr(self, name), name)
        capability = self.device_compute_capability
        if (
            not isinstance(capability, tuple)
            or len(capability) != 2
        ):
            raise ValueError(
                "device_compute_capability must be a two-item tuple"
            )
        _require_positive_int(
            capability[0],
            "device_compute_capability major",
        )
        _require_nonnegative_int(
            capability[1],
            "device_compute_capability minor",
        )

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExactPrefillGraphEligibility:
    eligible: bool
    fallback_reason: str | None


def check_exact_prefill_graph_eligibility(
    *,
    enabled: bool,
    is_prefill: bool,
    tensor_parallel_size: int,
    world_size: int,
    sequence_count: int,
    input_token_count: int,
    query_len: int,
    key_len: int,
    has_prefix_block_table: bool,
    token_allowlist: tuple[int, ...],
    input_embeddings_requested: bool,
    return_hidden_states: bool,
    cpu_offload: bool,
    kv_offload: bool,
    kv_quant_bits: int,
    compact_attention: bool,
    model_forward_kind: str,
) -> ExactPrefillGraphEligibility:
    _require_bool(enabled, "enabled")
    _require_bool(is_prefill, "is_prefill")
    _require_bool(
        has_prefix_block_table,
        "has_prefix_block_table",
    )
    _require_bool(
        input_embeddings_requested,
        "input_embeddings_requested",
    )
    _require_bool(return_hidden_states, "return_hidden_states")
    _require_bool(cpu_offload, "cpu_offload")
    _require_bool(kv_offload, "kv_offload")
    _require_bool(compact_attention, "compact_attention")
    for name, value in (
        ("tensor_parallel_size", tensor_parallel_size),
        ("world_size", world_size),
        ("sequence_count", sequence_count),
        ("input_token_count", input_token_count),
        ("query_len", query_len),
        ("key_len", key_len),
    ):
        _require_positive_int(value, name)
    _require_nonnegative_int(kv_quant_bits, "kv_quant_bits")
    _require_canonical_allowlist(token_allowlist)
    _require_nonempty_string(model_forward_kind, "model_forward_kind")

    checks = (
        (not enabled, "feature_disabled"),
        (not is_prefill, "not_prefill"),
        (
            tensor_parallel_size != 1,
            "tensor_parallel_unsupported",
        ),
        (world_size != 1, "world_size_unsupported"),
        (sequence_count != 1, "sequence_count_unsupported"),
        (
            query_len != input_token_count
            or key_len != input_token_count,
            "length_mismatch",
        ),
        (
            has_prefix_block_table,
            "prefix_block_table_present",
        ),
        (
            input_token_count not in token_allowlist,
            "token_count_not_allowlisted",
        ),
        (
            input_embeddings_requested,
            "input_embeddings_requested",
        ),
        (
            return_hidden_states,
            "hidden_state_return_requested",
        ),
        (cpu_offload, "cpu_offload_active"),
        (kv_offload, "kv_offload_active"),
        (kv_quant_bits != 0, "kv_quantization_active"),
        (compact_attention, "compact_attention_active"),
        (
            model_forward_kind != "forward",
            "model_forward_unsupported",
        ),
    )
    for rejected, reason in checks:
        if rejected:
            return ExactPrefillGraphEligibility(False, reason)
    return ExactPrefillGraphEligibility(True, None)


@dataclass(frozen=True)
class ExactPrefillCudaGraphCacheConfig:
    enabled: bool
    token_allowlist: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "enabled")
        _require_canonical_allowlist(self.token_allowlist)


@dataclass
class ExactPrefillCudaGraphEntry:
    identity: ExactPrefillGraphIdentity
    identity_sha256: str
    graph: object
    tensors: dict[str, object]
    static_bytes: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    replay_count: int = 0
    last_replay_step: int | None = None
    state: str = "ready"
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExactPrefillGraphIdentity):
            raise ValueError(
                "identity must be ExactPrefillGraphIdentity"
            )
        _require_nonempty_string(
            self.identity_sha256,
            "identity_sha256",
        )
        if not isinstance(self.tensors, dict):
            raise ValueError("tensors must be a dict")
        for name in (
            "static_bytes",
            "capture_duration_ns",
            "allocated_delta_bytes",
            "reserved_delta_bytes",
            "replay_count",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if self.last_replay_step is not None:
            _require_nonnegative_int(
                self.last_replay_step,
                "last_replay_step",
            )


class ExactPrefillCudaGraphCache:
    def __init__(
        self,
        config: ExactPrefillCudaGraphCacheConfig,
    ):
        if not isinstance(config, ExactPrefillCudaGraphCacheConfig):
            raise ValueError(
                "config must be ExactPrefillCudaGraphCacheConfig"
            )
        self.config = config
        self.ready_entries: dict[
            str,
            ExactPrefillCudaGraphEntry,
        ] = {}
        self.capturing: set[str] = set()
        self.quarantined: dict[str, str] = {}
        self.capture_errors_by_token: dict[int, str] = {}
        self.static_bytes = 0
        self.allocated_delta_bytes = 0
        self.reserved_delta_bytes = 0
        self.total_capture_ns = 0
        self.counters = collections.Counter()

    def begin_capture(
        self,
        identity: ExactPrefillGraphIdentity,
    ) -> bool:
        identity_sha256 = identity.sha256
        if (
            not self.config.enabled
            or identity.token_count not in self.config.token_allowlist
            or identity_sha256 in self.ready_entries
            or identity_sha256 in self.capturing
            or identity_sha256 in self.quarantined
        ):
            return False
        self.capturing.add(identity_sha256)
        self.counters["capture_attempts"] += 1
        return True

    def commit_capture(
        self,
        entry: ExactPrefillCudaGraphEntry,
    ) -> None:
        identity_sha256 = entry.identity.sha256
        if entry.identity_sha256 != identity_sha256:
            raise ValueError("entry identity SHA does not match identity")
        if identity_sha256 not in self.capturing:
            raise ValueError("identity is not awaiting capture commit")
        if entry.state != "ready" or entry.rejection_reason is not None:
            raise ValueError("captured entry must be ready")
        self.capturing.remove(identity_sha256)
        self.ready_entries[identity_sha256] = entry
        self.static_bytes += entry.static_bytes
        self.allocated_delta_bytes += entry.allocated_delta_bytes
        self.reserved_delta_bytes += entry.reserved_delta_bytes
        self.total_capture_ns += entry.capture_duration_ns
        self.counters["capture_successes"] += 1

    def ready_entry(
        self,
        identity: ExactPrefillGraphIdentity,
    ) -> ExactPrefillCudaGraphEntry | None:
        identity_sha256 = identity.sha256
        entry = self.ready_entries.get(identity_sha256)
        if entry is None:
            self.counters["misses"] += 1
            return None
        if (
            entry.identity != identity
            or entry.identity_sha256 != identity_sha256
            or entry.state != "ready"
        ):
            self.quarantine(identity, "identity_drift")
            self.counters["misses"] += 1
            return None
        self.counters["hits"] += 1
        return entry

    def record_replay(
        self,
        entry: ExactPrefillCudaGraphEntry,
        *,
        step: int,
    ) -> None:
        _require_nonnegative_int(step, "step")
        if (
            entry.state != "ready"
            or self.ready_entries.get(entry.identity_sha256) is not entry
        ):
            raise ValueError("entry is not ready for replay")
        entry.replay_count += 1
        entry.last_replay_step = step
        self.counters["replays"] += 1

    def record_fallback(self, reason: str) -> None:
        if reason not in PREFILL_GRAPH_FALLBACK_REASONS:
            raise ValueError(
                f"unsupported prefill graph fallback reason: {reason}"
            )
        self.counters["fallbacks"] += 1
        self.counters[f"fallback_{reason}"] += 1

    def record_capture_error(
        self,
        token_count: int,
        reason: str,
    ) -> None:
        _require_positive_int(token_count, "token_count")
        _require_nonempty_string(reason, "capture error")
        if token_count in self.capture_errors_by_token:
            return
        self.capture_errors_by_token[token_count] = reason
        self.counters["capture_failures"] += 1
        self.counters["fallback_capture_failed"] += 1

    def quarantine(
        self,
        identity: ExactPrefillGraphIdentity,
        reason: str,
    ) -> None:
        if reason not in PREFILL_GRAPH_FALLBACK_REASONS:
            raise ValueError(
                f"unsupported prefill graph quarantine reason: {reason}"
            )
        identity_sha256 = identity.sha256
        existing = self.quarantined.get(identity_sha256)
        if existing is not None:
            if existing != reason:
                raise ValueError(
                    "quarantined identity reason cannot change"
                )
            return
        entry = self.ready_entries.pop(identity_sha256, None)
        if entry is not None:
            entry.state = "quarantined"
            entry.rejection_reason = reason
        self.capturing.discard(identity_sha256)
        self.quarantined[identity_sha256] = reason
        self.counters["quarantines"] += 1
        if reason == "capture_failed":
            self.counters["capture_failures"] += 1
        if reason == "replay_failed":
            self.counters["replay_failures"] += 1
        self.counters[f"fallback_{reason}"] += 1

    def summary(self) -> dict[str, object]:
        counters = {
            key: self.counters[key]
            for key in (
                "hits",
                "misses",
                "capture_attempts",
                "capture_successes",
                "capture_failures",
                "replays",
                "replay_failures",
                "quarantines",
                "fallbacks",
            )
        }
        counters.update(
            {
                key: self.counters[key]
                for key in sorted(self.counters)
                if key not in counters
            }
        )
        return {
            "ready_entries": tuple(sorted(self.ready_entries)),
            "capturing": tuple(sorted(self.capturing)),
            "quarantined": {
                identity_sha256: self.quarantined[identity_sha256]
                for identity_sha256 in sorted(self.quarantined)
            },
            "capture_errors_by_token": {
                str(token_count): self.capture_errors_by_token[token_count]
                for token_count in sorted(self.capture_errors_by_token)
            },
            "static_bytes": self.static_bytes,
            "allocated_delta_bytes": self.allocated_delta_bytes,
            "reserved_delta_bytes": self.reserved_delta_bytes,
            "total_capture_ns": self.total_capture_ns,
            **counters,
        }
