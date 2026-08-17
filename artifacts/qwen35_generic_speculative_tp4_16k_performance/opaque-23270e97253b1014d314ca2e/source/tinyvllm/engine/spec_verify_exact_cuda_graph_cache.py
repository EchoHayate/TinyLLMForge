from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass


SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS = 16

SPEC_VERIFY_GRAPH_FALLBACK_REASONS = (
    "feature_disabled",
    "enforce_eager",
    "unsupported_mode",
    "tp_not_one",
    "kv_offload_enabled",
    "blockwise_enabled",
    "batch_not_allowlisted",
    "query_len_not_allowlisted",
    "non_greedy",
    "input_embeds_active",
    "hidden_state_return_active",
    "non_transactional_state",
    "transaction_unauthorized",
    "identity_invalid",
    "cold_identity",
    "entry_limit",
    "static_byte_budget",
    "reserved_byte_budget",
    "single_capture_budget",
    "total_capture_budget",
    "scratch_unavailable",
    "capture_failed",
    "identity_drift",
    "shape_drift",
    "cache_state_drift",
)

SPEC_VERIFY_GRAPH_QUARANTINE_REASONS = (
    "capture_failed",
    "capture_rollback_failed",
    "post_capture_budget",
    "identity_drift",
    "shape_drift",
    "replay_failed",
)


class SpecVerifyGraphReplayError(RuntimeError):
    def __init__(
        self,
        identity_sha256: str,
        cause: BaseException,
    ) -> None:
        self.identity_sha256 = identity_sha256
        self.cause = cause
        super().__init__(
            "spec-verify CUDA Graph replay failed: "
            f"{identity_sha256}"
        )


def _require_positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_canonical_positive_tuple(
    value: object,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
        for item in value
    ):
        raise ValueError(
            f"{name} must contain positive non-boolean integers"
        )
    if tuple(sorted(set(value))) != value:
        raise ValueError(f"{name} must be canonical")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def required_spec_verify_capture_scratch_blocks(
    *,
    batch_allowlist: tuple[int, ...],
    query_len_allowlist: tuple[int, ...],
    block_size: int,
) -> int:
    batch_allowlist = _require_canonical_positive_tuple(
        batch_allowlist,
        "batch_allowlist",
        allow_empty=False,
    )
    query_len_allowlist = _require_canonical_positive_tuple(
        query_len_allowlist,
        "query_len_allowlist",
        allow_empty=True,
    )
    block_size = _require_positive_integer(
        block_size,
        "block_size",
    )
    if not query_len_allowlist:
        return 0
    blocks_per_row = (
        block_size
        - 1
        + max(query_len_allowlist)
        + block_size
        - 1
    ) // block_size
    return max(batch_allowlist) * blocks_per_row


@dataclass
class SpecVerifyCaptureScratchLease:
    lease_id: int
    block_ids: tuple[int, ...]
    block_generations: tuple[int, ...]
    row_block_counts: tuple[int, ...]
    state: str = "active"


class SpecVerifyCaptureScratchPool:
    def __init__(
        self,
        *,
        block_ids: tuple[int, ...],
        block_size: int,
    ):
        if not isinstance(block_ids, tuple) or not block_ids:
            raise ValueError("block_ids must be a non-empty tuple")
        for block_id in block_ids:
            _require_nonnegative_integer(block_id, "block_id")
        if tuple(sorted(set(block_ids))) != block_ids:
            raise ValueError("block_ids must be sorted and unique")
        self.block_size = _require_positive_integer(
            block_size,
            "block_size",
        )
        self.block_ids = block_ids
        self.block_generations = {
            block_id: 0
            for block_id in block_ids
        }
        self.free_block_ids = list(block_ids)
        self.active_leases: dict[
            int,
            SpecVerifyCaptureScratchLease,
        ] = {}
        self._next_lease_id = 1

    def acquire(
        self,
        *,
        active_batch_size: int,
        query_len: int,
        row_offsets: tuple[int, ...],
    ) -> SpecVerifyCaptureScratchLease:
        active_batch_size = _require_positive_integer(
            active_batch_size,
            "active_batch_size",
        )
        query_len = _require_positive_integer(
            query_len,
            "query_len",
        )
        if (
            not isinstance(row_offsets, tuple)
            or len(row_offsets) != active_batch_size
        ):
            raise ValueError(
                "row_offsets must match active_batch_size"
            )
        for row_offset in row_offsets:
            _require_nonnegative_integer(
                row_offset,
                "row_offset",
            )
            if row_offset >= self.block_size:
                raise ValueError(
                    "row_offset must be smaller than block_size"
                )
        row_block_counts = tuple(
            (
                row_offset
                + query_len
                + self.block_size
                - 1
            )
            // self.block_size
            for row_offset in row_offsets
        )
        required_blocks = sum(row_block_counts)
        if required_blocks > len(self.free_block_ids):
            raise RuntimeError("scratch_unavailable")
        leased_ids = tuple(
            self.free_block_ids[:required_blocks]
        )
        del self.free_block_ids[:required_blocks]
        lease = SpecVerifyCaptureScratchLease(
            lease_id=self._next_lease_id,
            block_ids=leased_ids,
            block_generations=tuple(
                self.block_generations[block_id]
                for block_id in leased_ids
            ),
            row_block_counts=row_block_counts,
        )
        self._next_lease_id += 1
        self.active_leases[lease.lease_id] = lease
        return lease

    def rollback(
        self,
        lease: SpecVerifyCaptureScratchLease,
    ) -> None:
        if not isinstance(lease, SpecVerifyCaptureScratchLease):
            raise ValueError(
                "lease must be SpecVerifyCaptureScratchLease"
            )
        if lease.state != "active":
            raise RuntimeError("scratch lease is already rolled back")
        active = self.active_leases.get(lease.lease_id)
        if active is not lease:
            raise ValueError("unknown scratch lease")
        self.active_leases.pop(lease.lease_id)
        for block_id in lease.block_ids:
            self.block_generations[block_id] += 1
        self.free_block_ids.extend(lease.block_ids)
        self.free_block_ids.sort()
        lease.state = "rolled_back"


@dataclass(frozen=True)
class SpecVerifyGraphIdentity:
    active_batch_size: int
    query_len: int
    total_query_tokens: int
    page_table_width: int
    flash_attn_num_splits: int
    attention_backend: str
    attention_backend_version: str
    input_dtype: str
    output_dtype: str
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    device_compute_capability: tuple[int, int]

    def __post_init__(self) -> None:
        _require_positive_integer(
            self.active_batch_size,
            "active_batch_size",
        )
        _require_positive_integer(self.query_len, "query_len")
        _require_positive_integer(
            self.total_query_tokens,
            "total_query_tokens",
        )
        if (
            self.total_query_tokens
            != self.active_batch_size * self.query_len
        ):
            raise ValueError(
                "total_query_tokens must equal "
                "active_batch_size * query_len"
            )
        _require_positive_integer(
            self.page_table_width,
            "page_table_width",
        )
        _require_positive_integer(
            self.flash_attn_num_splits,
            "flash_attn_num_splits",
        )
        if (
            self.flash_attn_num_splits
            != SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS
        ):
            raise ValueError(
                "flash_attn_num_splits must match the fixed "
                "spec-verify split count"
            )
        _require_nonempty_string(
            self.attention_backend,
            "attention_backend",
        )
        _require_nonempty_string(
            self.attention_backend_version,
            "attention_backend_version",
        )
        _require_nonempty_string(self.input_dtype, "input_dtype")
        _require_nonempty_string(self.output_dtype, "output_dtype")
        _require_positive_integer(
            self.num_query_heads,
            "num_query_heads",
        )
        _require_positive_integer(self.num_kv_heads, "num_kv_heads")
        _require_positive_integer(self.head_dim, "head_dim")
        _require_positive_integer(
            self.page_block_size,
            "page_block_size",
        )
        capability = self.device_compute_capability
        if (
            not isinstance(capability, tuple)
            or len(capability) != 2
        ):
            raise ValueError(
                "device_compute_capability must be a two-item tuple"
            )
        major, minor = capability
        _require_positive_integer(
            major,
            "device_compute_capability major",
        )
        _require_nonnegative_integer(
            minor,
            "device_compute_capability minor",
        )

    @property
    def sha256(self) -> str:
        payload = (
            self.active_batch_size,
            self.query_len,
            self.total_query_tokens,
            self.page_table_width,
            self.flash_attn_num_splits,
            self.attention_backend,
            self.attention_backend_version,
            self.input_dtype,
            self.output_dtype,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_block_size,
            self.device_compute_capability,
        )
        return hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class SpecVerifyExactCudaGraphCacheConfig:
    enabled: bool
    batch_allowlist: tuple[int, ...]
    query_len_allowlist: tuple[int, ...]
    min_observations: int
    max_entries: int
    max_static_bytes: int
    max_reserved_bytes: int
    max_total_capture_ns: int
    max_single_capture_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        _require_canonical_positive_tuple(
            self.batch_allowlist,
            "batch_allowlist",
            allow_empty=False,
        )
        _require_canonical_positive_tuple(
            self.query_len_allowlist,
            "query_len_allowlist",
            allow_empty=True,
        )
        for name in (
            "min_observations",
            "max_entries",
            "max_static_bytes",
            "max_reserved_bytes",
            "max_total_capture_ns",
            "max_single_capture_ns",
        ):
            _require_positive_integer(getattr(self, name), name)


@dataclass
class SpecVerifyExactCudaGraphEntry:
    identity: SpecVerifyGraphIdentity
    identity_sha256: str
    graph: object
    tensors: dict[str, object]
    static_bytes: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    replay_count: int = 0
    last_replay_step: int | None = None
    last_use_step: int = 0
    in_flight_replays: int = 0
    state: str = "ready"
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SpecVerifyGraphIdentity):
            raise ValueError(
                "identity must be SpecVerifyGraphIdentity"
            )
        if (
            not isinstance(self.identity_sha256, str)
            or not self.identity_sha256
        ):
            raise ValueError(
                "identity_sha256 must be a non-empty string"
            )
        if not isinstance(self.tensors, dict):
            raise ValueError("tensors must be a dict")
        for name in (
            "static_bytes",
            "capture_duration_ns",
            "allocated_delta_bytes",
            "reserved_delta_bytes",
            "replay_count",
            "last_use_step",
            "in_flight_replays",
        ):
            _require_nonnegative_integer(getattr(self, name), name)
        if self.last_replay_step is not None:
            _require_nonnegative_integer(
                self.last_replay_step,
                "last_replay_step",
            )
        if self.state not in ("ready", "quarantined", "evicted"):
            raise ValueError("unsupported graph entry state")


@dataclass(frozen=True)
class SpecVerifyGraphAdmissionDecision:
    should_capture: bool
    cache_state: str
    decision: str
    fallback_reason: str | None
    observation_count: int


class SpecVerifyExactCudaGraphCache:
    def __init__(
        self,
        config: SpecVerifyExactCudaGraphCacheConfig,
    ):
        if not isinstance(
            config,
            SpecVerifyExactCudaGraphCacheConfig,
        ):
            raise ValueError(
                "config must be SpecVerifyExactCudaGraphCacheConfig"
            )
        self.config = config
        self.observation_counts: dict[str, int] = {}
        self.ready_entries: dict[
            str,
            SpecVerifyExactCudaGraphEntry,
        ] = {}
        self.quarantined: dict[str, str] = {}
        self.capturing: dict[str, int] = {}
        self.static_bytes = 0
        self.reserved_delta_bytes = 0
        self.total_capture_ns = 0
        self.counters = collections.Counter()

    def _fallback_decision(
        self,
        *,
        cache_state: str,
        decision: str,
        fallback_reason: str,
        observation_count: int,
    ) -> SpecVerifyGraphAdmissionDecision:
        if fallback_reason not in SPEC_VERIFY_GRAPH_FALLBACK_REASONS:
            raise ValueError(
                "unsupported spec-verify graph fallback reason"
            )
        self.counters["fallbacks"] += 1
        self.counters[f"fallback_{fallback_reason}"] += 1
        return SpecVerifyGraphAdmissionDecision(
            should_capture=False,
            cache_state=cache_state,
            decision=decision,
            fallback_reason=fallback_reason,
            observation_count=observation_count,
        )

    def _capacity_satisfied(
        self,
        *,
        estimated_static_bytes: int,
    ) -> bool:
        capturing_static_bytes = sum(self.capturing.values())
        return (
            len(self.ready_entries) + len(self.capturing)
            < self.config.max_entries
            and (
                self.static_bytes
                + capturing_static_bytes
                + estimated_static_bytes
                <= self.config.max_static_bytes
            )
        )

    def _evict_lru_ready_entry(self) -> bool:
        candidates = tuple(
            entry
            for entry in self.ready_entries.values()
            if (
                entry.state == "ready"
                and entry.in_flight_replays == 0
            )
        )
        if not candidates:
            return False
        victim = min(
            candidates,
            key=lambda entry: (
                entry.last_use_step,
                entry.identity_sha256,
            ),
        )
        self.ready_entries.pop(victim.identity_sha256)
        victim.state = "evicted"
        self.static_bytes -= victim.static_bytes
        self.counters["evictions"] += 1
        return True

    def _prepare_capture_capacity(
        self,
        *,
        estimated_static_bytes: int,
    ) -> str | None:
        if estimated_static_bytes > self.config.max_static_bytes:
            return "static_byte_budget"
        if (
            self.reserved_delta_bytes
            >= self.config.max_reserved_bytes
        ):
            return "reserved_byte_budget"
        if (
            self.total_capture_ns
            >= self.config.max_total_capture_ns
        ):
            return "total_capture_budget"
        while not self._capacity_satisfied(
            estimated_static_bytes=estimated_static_bytes,
        ):
            if not self._evict_lru_ready_entry():
                break
        if (
            len(self.ready_entries) + len(self.capturing)
            >= self.config.max_entries
        ):
            return "entry_limit"
        if (
            self.static_bytes
            + sum(self.capturing.values())
            + estimated_static_bytes
            > self.config.max_static_bytes
        ):
            return "static_byte_budget"
        return None

    def observe_success(
        self,
        identity: SpecVerifyGraphIdentity,
        *,
        estimated_static_bytes: int,
        step_id: int,
    ) -> SpecVerifyGraphAdmissionDecision:
        if not isinstance(identity, SpecVerifyGraphIdentity):
            raise ValueError(
                "identity must be SpecVerifyGraphIdentity"
            )
        _require_nonnegative_integer(
            estimated_static_bytes,
            "estimated_static_bytes",
        )
        _require_nonnegative_integer(step_id, "step_id")
        identity_sha256 = identity.sha256
        observation_count = self.observation_counts.get(
            identity_sha256,
            0,
        )
        if not self.config.enabled:
            return self._fallback_decision(
                cache_state="absent",
                decision="feature_disabled",
                fallback_reason="feature_disabled",
                observation_count=observation_count,
            )
        terminal_reason = self.quarantined.get(identity_sha256)
        if terminal_reason is not None:
            self.counters["fallbacks"] += 1
            self.counters[f"fallback_{terminal_reason}"] += 1
            return SpecVerifyGraphAdmissionDecision(
                should_capture=False,
                cache_state="quarantined",
                decision="quarantined",
                fallback_reason=terminal_reason,
                observation_count=observation_count,
            )
        if identity_sha256 in self.ready_entries:
            return SpecVerifyGraphAdmissionDecision(
                should_capture=False,
                cache_state="ready",
                decision="hit",
                fallback_reason=None,
                observation_count=observation_count,
            )
        if identity_sha256 in self.capturing:
            return self._fallback_decision(
                cache_state="capturing",
                decision="cold",
                fallback_reason="cold_identity",
                observation_count=observation_count,
            )

        observation_count += 1
        self.observation_counts[identity_sha256] = observation_count
        if observation_count < self.config.min_observations:
            return self._fallback_decision(
                cache_state="observing",
                decision="cold",
                fallback_reason="cold_identity",
                observation_count=observation_count,
            )

        reason = self._prepare_capture_capacity(
            estimated_static_bytes=estimated_static_bytes,
        )
        if reason is not None:
            return self._fallback_decision(
                cache_state="budget_exhausted",
                decision="incompatible",
                fallback_reason=reason,
                observation_count=observation_count,
            )

        self.capturing[identity_sha256] = estimated_static_bytes
        self.counters["capture_attempts"] += 1
        return SpecVerifyGraphAdmissionDecision(
            should_capture=True,
            cache_state="capturing",
            decision="capture",
            fallback_reason=None,
            observation_count=observation_count,
        )

    def ready_entry(
        self,
        identity: SpecVerifyGraphIdentity,
    ) -> SpecVerifyExactCudaGraphEntry | None:
        if not isinstance(identity, SpecVerifyGraphIdentity):
            raise ValueError(
                "identity must be SpecVerifyGraphIdentity"
            )
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

    def _post_capture_rejection_reason(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
    ) -> str | None:
        if (
            entry.capture_duration_ns
            > self.config.max_single_capture_ns
        ):
            return "post_capture_budget"
        if self.total_capture_ns > self.config.max_total_capture_ns:
            return "post_capture_budget"
        if (
            self.static_bytes + entry.static_bytes
            > self.config.max_static_bytes
        ):
            return "post_capture_budget"
        if (
            self.reserved_delta_bytes + entry.reserved_delta_bytes
            > self.config.max_reserved_bytes
        ):
            return "post_capture_budget"
        return None

    def commit_capture(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
    ) -> None:
        if not isinstance(entry, SpecVerifyExactCudaGraphEntry):
            raise ValueError(
                "entry must be SpecVerifyExactCudaGraphEntry"
            )
        identity = entry.identity
        identity_sha256 = identity.sha256
        if entry.identity_sha256 != identity_sha256:
            raise ValueError(
                "entry identity SHA does not match identity"
            )
        if identity_sha256 not in self.capturing:
            raise ValueError(
                "identity is not awaiting capture commit"
            )
        if (
            identity_sha256 in self.ready_entries
            or identity_sha256 in self.quarantined
        ):
            raise ValueError("identity is already terminal")
        self.capturing.pop(identity_sha256)
        self.total_capture_ns += entry.capture_duration_ns
        reason = self._post_capture_rejection_reason(entry)
        if reason is not None:
            entry.state = "quarantined"
            entry.terminal_reason = reason
            self.quarantined[identity_sha256] = reason
            self.reserved_delta_bytes += entry.reserved_delta_bytes
            self.counters["quarantines"] += 1
            self.counters["capture_failures"] += 1
            return
        self.ready_entries[identity_sha256] = entry
        self.static_bytes += entry.static_bytes
        self.reserved_delta_bytes += entry.reserved_delta_bytes
        self.counters["captures"] += 1

    def quarantine(
        self,
        identity: SpecVerifyGraphIdentity,
        reason: str,
        *,
        retained_reserved_bytes: int = 0,
    ) -> None:
        if not isinstance(identity, SpecVerifyGraphIdentity):
            raise ValueError(
                "identity must be SpecVerifyGraphIdentity"
            )
        if reason not in SPEC_VERIFY_GRAPH_QUARANTINE_REASONS:
            raise ValueError(
                "unsupported spec-verify graph quarantine reason"
            )
        _require_nonnegative_integer(
            retained_reserved_bytes,
            "retained_reserved_bytes",
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
            self.static_bytes -= entry.static_bytes
            entry.state = "quarantined"
            entry.terminal_reason = reason
        self.capturing.pop(identity_sha256, None)
        self.quarantined[identity_sha256] = reason
        self.reserved_delta_bytes += retained_reserved_bytes
        self.counters["quarantines"] += 1

    def begin_replay(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
        *,
        step_id: int,
    ) -> None:
        if not isinstance(entry, SpecVerifyExactCudaGraphEntry):
            raise ValueError(
                "entry must be SpecVerifyExactCudaGraphEntry"
            )
        _require_nonnegative_integer(step_id, "step_id")
        if (
            entry.state != "ready"
            or self.ready_entries.get(entry.identity_sha256) is not entry
        ):
            raise RuntimeError("graph entry is not replayable")
        entry.in_flight_replays += 1
        entry.last_use_step = max(entry.last_use_step, step_id)

    def finish_replay(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
        *,
        step_id: int,
        succeeded: bool,
    ) -> None:
        if not isinstance(entry, SpecVerifyExactCudaGraphEntry):
            raise ValueError(
                "entry must be SpecVerifyExactCudaGraphEntry"
            )
        _require_nonnegative_integer(step_id, "step_id")
        if not isinstance(succeeded, bool):
            raise ValueError("succeeded must be a bool")
        if entry.in_flight_replays <= 0:
            raise RuntimeError("graph entry has no in-flight replay")
        entry.in_flight_replays -= 1
        entry.last_use_step = max(entry.last_use_step, step_id)
        if succeeded:
            entry.replay_count += 1
            entry.last_replay_step = step_id

    def summary(self) -> dict[str, object]:
        counter_names = (
            "hits",
            "misses",
            "capture_attempts",
            "captures",
            "capture_failures",
            "fallbacks",
            "quarantines",
            "evictions",
        )
        return {
            "ready_entries": tuple(sorted(self.ready_entries)),
            "quarantined": tuple(sorted(self.quarantined.items())),
            "capturing": tuple(sorted(self.capturing)),
            "observation_counts": dict(
                sorted(self.observation_counts.items())
            ),
            "static_bytes": self.static_bytes,
            "reserved_delta_bytes": self.reserved_delta_bytes,
            "total_capture_ns": self.total_capture_ns,
            **{
                name: int(self.counters[name])
                for name in counter_names
            },
        }
