from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyvllm.engine.flash_attn_split_policy import (
        FlashAttentionGraphIdentity,
    )


FALLBACK_REASONS = (
    "feature_disabled",
    "enforce_eager",
    "unsupported_mode",
    "incompatible_feature",
    "batch_not_allowlisted",
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
    "replay_disabled",
)


@dataclass(frozen=True)
class ExactCudaGraphCacheConfig:
    enabled: bool
    batch_allowlist: tuple
    min_observations: int
    max_entries: int
    max_static_bytes: int
    max_reserved_bytes: int
    max_total_capture_ns: int
    max_single_capture_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not self.batch_allowlist or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 1
            for value in self.batch_allowlist
        ):
            raise ValueError("batch_allowlist must contain batches above one")
        if tuple(sorted(set(self.batch_allowlist))) != self.batch_allowlist:
            raise ValueError("batch_allowlist must be canonical")
        for field in (
            "min_observations",
            "max_entries",
            "max_static_bytes",
            "max_reserved_bytes",
            "max_total_capture_ns",
            "max_single_capture_ns",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{field} must be a positive integer")


@dataclass
class ExactCudaGraphEntry:
    identity: FlashAttentionGraphIdentity
    identity_sha256: str
    graph: object | None
    tensors: dict[str, object]
    static_bytes: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    replay_count: int = 0
    last_replay_step: int | None = None
    state: str = "ready"
    rejection_reason: str | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    should_capture: bool
    cache_state: str
    fallback_reason: str
    observation_count: int


class ExactCudaGraphCache:
    def __init__(self, config: ExactCudaGraphCacheConfig):
        self.config = config
        self.observation_counts: dict[str, int] = {}
        self.ready_entries: dict[str, ExactCudaGraphEntry] = {}
        self.rejected: dict[str, str] = {}
        self.capturing: set[str] = set()
        self.static_bytes = 0
        self.reserved_delta_bytes = 0
        self.total_capture_ns = 0
        self.counters = collections.Counter()

    def observe_success(
        self,
        identity: FlashAttentionGraphIdentity,
        *,
        estimated_static_bytes: int,
    ) -> AdmissionDecision:
        identity_sha256 = identity.sha256
        observation_count = self.observation_counts.get(
            identity_sha256,
            0,
        )
        if not self.config.enabled:
            return AdmissionDecision(
                False,
                "absent",
                "feature_disabled",
                observation_count,
            )
        if identity_sha256 in self.rejected:
            return AdmissionDecision(
                False,
                "rejected",
                self.rejected[identity_sha256],
                observation_count,
            )
        if identity_sha256 in self.ready_entries:
            return AdmissionDecision(
                False,
                "ready",
                "replay_disabled",
                observation_count,
            )
        if identity_sha256 in self.capturing:
            return AdmissionDecision(
                False,
                "observing",
                "cold_identity",
                observation_count,
            )

        observation_count += 1
        self.observation_counts[identity_sha256] = observation_count
        if observation_count < self.config.min_observations:
            return AdmissionDecision(
                False,
                "observing",
                "cold_identity",
                observation_count,
            )

        reason = self._pre_capture_rejection_reason(
            identity,
            estimated_static_bytes=estimated_static_bytes,
        )
        if reason is not None:
            self.reject(identity, reason)
            return AdmissionDecision(
                False,
                "budget_exhausted",
                reason,
                observation_count,
            )

        self.capturing.add(identity_sha256)
        self.counters["capture_attempts"] += 1
        return AdmissionDecision(
            True,
            "observing",
            "cold_identity",
            observation_count,
        )

    def ready_entry(
        self,
        identity: FlashAttentionGraphIdentity,
    ) -> ExactCudaGraphEntry | None:
        identity_sha256 = identity.sha256
        entry = self.ready_entries.get(identity_sha256)
        if entry is None:
            self.counters["misses"] += 1
            return None
        if (
            entry.identity_sha256 != identity_sha256
            or entry.identity != identity
            or entry.state != "ready"
        ):
            self.disable_entry(identity_sha256, "identity_drift")
            self.counters["misses"] += 1
            return None
        self.counters["hits"] += 1
        return entry

    def commit_capture(self, entry: ExactCudaGraphEntry) -> None:
        identity_sha256 = entry.identity.sha256
        if entry.identity_sha256 != identity_sha256:
            raise ValueError("entry identity SHA does not match identity")
        if identity_sha256 not in self.capturing:
            raise ValueError("identity is not awaiting capture commit")
        if (
            identity_sha256 in self.ready_entries
            or identity_sha256 in self.rejected
        ):
            raise ValueError("identity is already terminal")
        self.capturing.remove(identity_sha256)

        self.total_capture_ns += max(0, int(entry.capture_duration_ns))
        retained_reserved_bytes = max(
            0,
            int(entry.reserved_delta_bytes),
        )
        reason = self._post_capture_rejection_reason(entry)
        if reason is not None:
            entry.state = "rejected"
            entry.rejection_reason = reason
            self.rejected[identity_sha256] = reason
            self.reserved_delta_bytes += retained_reserved_bytes
            self.counters["capture_failures"] += 1
            self.counters[f"fallback_{reason}"] += 1
            return

        self.ready_entries[identity_sha256] = entry
        self.static_bytes += int(entry.static_bytes)
        self.reserved_delta_bytes += retained_reserved_bytes
        self.counters["capture_successes"] += 1

    def reject(
        self,
        identity: FlashAttentionGraphIdentity,
        reason: str,
        *,
        retained_reserved_bytes: int = 0,
    ) -> None:
        self._validate_fallback_reason(reason)
        identity_sha256 = identity.sha256
        if identity_sha256 in self.ready_entries:
            raise ValueError("ready identity cannot be rejected")
        existing = self.rejected.get(identity_sha256)
        if existing is not None and existing != reason:
            raise ValueError("rejected identity reason cannot change")
        self.capturing.discard(identity_sha256)
        if existing is None:
            self.rejected[identity_sha256] = reason
            self.reserved_delta_bytes += max(
                0,
                int(retained_reserved_bytes),
            )
            self.counters[f"fallback_{reason}"] += 1

    def disable_entry(
        self,
        identity_sha256: str,
        reason: str,
    ) -> None:
        self._validate_fallback_reason(reason)
        entry = self.ready_entries.pop(identity_sha256, None)
        if entry is None:
            existing = self.rejected.get(identity_sha256)
            if existing is not None and existing != reason:
                raise ValueError("rejected identity reason cannot change")
        else:
            entry.state = "rejected"
            entry.rejection_reason = reason
        self.rejected[identity_sha256] = reason
        self.capturing.discard(identity_sha256)
        self.counters[f"fallback_{reason}"] += 1

    def summary(self) -> dict:
        counters = {
            key: self.counters[key]
            for key in (
                "hits",
                "misses",
                "capture_attempts",
                "capture_successes",
                "capture_failures",
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
            "ready_entries": sorted(self.ready_entries),
            "rejected": {
                identity_sha256: self.rejected[identity_sha256]
                for identity_sha256 in sorted(self.rejected)
            },
            "capturing": sorted(self.capturing),
            "observation_counts": {
                identity_sha256: self.observation_counts[identity_sha256]
                for identity_sha256 in sorted(self.observation_counts)
            },
            "static_bytes": self.static_bytes,
            "reserved_delta_bytes": self.reserved_delta_bytes,
            "total_capture_ns": self.total_capture_ns,
            **counters,
        }

    def _pre_capture_rejection_reason(
        self,
        identity: FlashAttentionGraphIdentity,
        *,
        estimated_static_bytes: int,
    ) -> str | None:
        if identity.active_batch_size not in self.config.batch_allowlist:
            return "batch_not_allowlisted"
        if len(self.ready_entries) >= self.config.max_entries:
            return "entry_limit"
        if (
            self.static_bytes + int(estimated_static_bytes)
            > self.config.max_static_bytes
        ):
            return "static_byte_budget"
        if (
            self.reserved_delta_bytes
            >= self.config.max_reserved_bytes
        ):
            return "reserved_byte_budget"
        if self.total_capture_ns >= self.config.max_total_capture_ns:
            return "total_capture_budget"
        return None

    def _post_capture_rejection_reason(
        self,
        entry: ExactCudaGraphEntry,
    ) -> str | None:
        if entry.state != "ready" or entry.rejection_reason is not None:
            return entry.rejection_reason or "capture_failed"
        if len(self.ready_entries) >= self.config.max_entries:
            return "entry_limit"
        if (
            self.static_bytes + int(entry.static_bytes)
            > self.config.max_static_bytes
        ):
            return "static_byte_budget"
        if (
            self.reserved_delta_bytes
            + max(0, int(entry.reserved_delta_bytes))
            > self.config.max_reserved_bytes
        ):
            return "reserved_byte_budget"
        if (
            int(entry.capture_duration_ns)
            > self.config.max_single_capture_ns
        ):
            return "single_capture_budget"
        if self.total_capture_ns > self.config.max_total_capture_ns:
            return "total_capture_budget"
        return None

    @staticmethod
    def _validate_fallback_reason(reason: str) -> None:
        if reason not in FALLBACK_REASONS:
            raise ValueError(f"unknown fallback reason: {reason}")
