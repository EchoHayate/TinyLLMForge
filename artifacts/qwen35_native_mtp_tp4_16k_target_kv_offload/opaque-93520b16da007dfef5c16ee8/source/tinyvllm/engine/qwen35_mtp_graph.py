from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
import json


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Qwen35MTPGraphIdentity:
    exact_q: int
    exact_batch_size: int
    device_index: int
    compute_dtype: str
    hidden_size: int
    mtp_layer_count: int
    block_table_width: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.exact_q, bool)
            or not isinstance(self.exact_q, int)
            or self.exact_q < 2
        ):
            raise ValueError("exact_q must be an integer at least two")
        _positive_integer(
            self.exact_batch_size,
            "exact_batch_size",
        )
        if (
            isinstance(self.device_index, bool)
            or not isinstance(self.device_index, int)
            or self.device_index < 0
        ):
            raise ValueError(
                "device_index must be a nonnegative integer"
            )
        if (
            not isinstance(self.compute_dtype, str)
            or not self.compute_dtype
        ):
            raise ValueError(
                "compute_dtype must be a non-empty string"
            )
        _positive_integer(self.hidden_size, "hidden_size")
        _positive_integer(
            self.mtp_layer_count,
            "mtp_layer_count",
        )
        _positive_integer(
            self.block_table_width,
            "block_table_width",
        )

    @property
    def sha256(self) -> str:
        payload = {
            "block_table_width": self.block_table_width,
            "compute_dtype": self.compute_dtype,
            "device_index": self.device_index,
            "exact_batch_size": self.exact_batch_size,
            "exact_q": self.exact_q,
            "hidden_size": self.hidden_size,
            "mtp_layer_count": self.mtp_layer_count,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class Qwen35MTPGraphEntry:
    identity: Qwen35MTPGraphIdentity
    graph: object
    static_bytes: int
    capture_duration_ns: int
    reserved_delta_bytes: int


class Qwen35MTPGraphReplayError(RuntimeError):

    def __init__(
        self,
        identity: Qwen35MTPGraphIdentity,
        cause: BaseException,
    ):
        super().__init__(
            "Qwen3.5 MTP CUDA graph replay failed for "
            f"{identity.sha256}"
        )
        self.identity = identity
        self.cause = cause


class Qwen35MTPGraphPreReplayError(RuntimeError):
    pass


class Qwen35MTPExactGraphRunner:

    def __init__(
        self,
        *,
        enabled: bool,
        q_allowlist: tuple[int, ...],
        batch_allowlist: tuple[int, ...],
        min_observations: int,
        max_entries: int,
        max_static_bytes: int,
        max_reserved_bytes: int,
        max_total_capture_ns: int,
        max_single_capture_ns: int,
        device_index: int,
        compute_dtype: str,
        hidden_size: int,
        mtp_layer_count: int,
        block_table_width: int,
        capture_backend,
        scratch_owner,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        self.enabled = enabled
        self.q_allowlist = self._validate_allowlist(
            q_allowlist,
            name="q_allowlist",
            minimum=2,
        )
        self.batch_allowlist = self._validate_allowlist(
            batch_allowlist,
            name="batch_allowlist",
            minimum=1,
        )
        self.min_observations = _positive_integer(
            min_observations,
            "min_observations",
        )
        self.max_entries = _positive_integer(
            max_entries,
            "max_entries",
        )
        self.max_static_bytes = _positive_integer(
            max_static_bytes,
            "max_static_bytes",
        )
        self.max_reserved_bytes = _positive_integer(
            max_reserved_bytes,
            "max_reserved_bytes",
        )
        self.max_total_capture_ns = _positive_integer(
            max_total_capture_ns,
            "max_total_capture_ns",
        )
        self.max_single_capture_ns = _positive_integer(
            max_single_capture_ns,
            "max_single_capture_ns",
        )
        identity_probe = Qwen35MTPGraphIdentity(
            exact_q=self.q_allowlist[0],
            exact_batch_size=self.batch_allowlist[0],
            device_index=device_index,
            compute_dtype=compute_dtype,
            hidden_size=hidden_size,
            mtp_layer_count=mtp_layer_count,
            block_table_width=block_table_width,
        )
        self.device_index = identity_probe.device_index
        self.compute_dtype = identity_probe.compute_dtype
        self.hidden_size = identity_probe.hidden_size
        self.mtp_layer_count = identity_probe.mtp_layer_count
        self.block_table_width = identity_probe.block_table_width
        for name, owner, methods in (
            (
                "capture_backend",
                capture_backend,
                ("estimate_static_bytes", "capture", "replay"),
            ),
            (
                "scratch_owner",
                scratch_owner,
                ("acquire", "rollback"),
            ),
        ):
            for method in methods:
                if not callable(getattr(owner, method, None)):
                    raise ValueError(
                        f"{name} must expose callable {method}"
                    )
        self.capture_backend = capture_backend
        self.scratch_owner = scratch_owner
        self.observation_counts: dict[str, int] = {}
        self.ready_entries: dict[str, Qwen35MTPGraphEntry] = {}
        self.quarantined: dict[str, str] = {}
        self.static_bytes = 0
        self.reserved_bytes = 0
        self.total_capture_ns = 0
        self.counters = collections.Counter()

    @staticmethod
    def _validate_allowlist(
        values,
        *,
        name: str,
        minimum: int,
    ) -> tuple[int, ...]:
        if (
            not isinstance(values, tuple)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                for value in values
            )
            or tuple(sorted(set(values))) != values
        ):
            raise ValueError(f"{name} must be a canonical tuple")
        return values

    def _identity(
        self,
        exact_q: int,
        exact_batch_size: int,
    ) -> Qwen35MTPGraphIdentity:
        return Qwen35MTPGraphIdentity(
            exact_q=exact_q,
            exact_batch_size=exact_batch_size,
            device_index=self.device_index,
            compute_dtype=self.compute_dtype,
            hidden_size=self.hidden_size,
            mtp_layer_count=self.mtp_layer_count,
            block_table_width=self.block_table_width,
        )

    def _quarantine(
        self,
        identity: Qwen35MTPGraphIdentity,
        reason: str,
    ) -> None:
        identity_sha256 = identity.sha256
        existing = self.quarantined.get(identity_sha256)
        if existing is not None and existing != reason:
            raise ValueError(
                "quarantined identity reason cannot change"
            )
        if existing is None:
            self.quarantined[identity_sha256] = reason
            self.ready_entries.pop(identity_sha256, None)
            self.counters["quarantines"] += 1
            self.counters[f"fallback_{reason}"] += 1

    def quarantine_reason(
        self,
        identity: Qwen35MTPGraphIdentity,
    ) -> str | None:
        return self.quarantined.get(identity.sha256)

    def ready_identity_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self.ready_entries))

    def _pre_capture_reason(
        self,
        estimated_static_bytes: int,
    ) -> str | None:
        if len(self.ready_entries) >= self.max_entries:
            return "entry_limit"
        if (
            self.static_bytes + max(0, int(estimated_static_bytes))
            > self.max_static_bytes
        ):
            return "static_byte_budget"
        if self.reserved_bytes >= self.max_reserved_bytes:
            return "reserved_byte_budget"
        if self.total_capture_ns >= self.max_total_capture_ns:
            return "total_capture_budget"
        return None

    def _post_capture_reason(
        self,
        entry: Qwen35MTPGraphEntry,
    ) -> str | None:
        if not isinstance(entry, Qwen35MTPGraphEntry):
            return "capture_failed"
        if entry.identity.sha256 in self.quarantined:
            return self.quarantined[entry.identity.sha256]
        if entry.static_bytes < 0 or entry.reserved_delta_bytes < 0:
            return "capture_failed"
        if entry.capture_duration_ns < 0:
            return "capture_failed"
        if entry.capture_duration_ns > self.max_single_capture_ns:
            return "single_capture_budget"
        if (
            self.total_capture_ns + entry.capture_duration_ns
            > self.max_total_capture_ns
        ):
            return "total_capture_budget"
        if (
            self.static_bytes + entry.static_bytes
            > self.max_static_bytes
        ):
            return "static_byte_budget"
        if (
            self.reserved_bytes + entry.reserved_delta_bytes
            > self.max_reserved_bytes
        ):
            return "reserved_byte_budget"
        if len(self.ready_entries) >= self.max_entries:
            return "entry_limit"
        return None

    def run(self, *, exact_q: int, rows: tuple, eager):
        if not isinstance(rows, tuple) or not rows:
            raise ValueError("rows must be a non-empty tuple")
        if not callable(eager):
            raise ValueError("eager must be callable")
        identity = self._identity(exact_q, len(rows))
        identity_sha256 = identity.sha256
        entry = self.ready_entries.get(identity_sha256)
        if entry is not None:
            try:
                result = self.capture_backend.replay(entry, rows)
            except Qwen35MTPGraphPreReplayError:
                self.counters["fallback_pre_replay"] += 1
                return eager(exact_q, rows)
            except BaseException as error:
                self._quarantine(identity, "replay_failed")
                raise Qwen35MTPGraphReplayError(
                    identity,
                    error,
                ) from error
            self.counters["replays"] += 1
            return result

        result = eager(exact_q, rows)
        if (
            not self.enabled
            or exact_q not in self.q_allowlist
            or len(rows) not in self.batch_allowlist
            or identity_sha256 in self.quarantined
        ):
            return result

        observation_count = (
            self.observation_counts.get(identity_sha256, 0) + 1
        )
        self.observation_counts[identity_sha256] = observation_count
        if observation_count < self.min_observations:
            return result
        estimated_static_bytes = (
            self.capture_backend.estimate_static_bytes(
                identity,
                rows,
            )
        )
        reason = self._pre_capture_reason(estimated_static_bytes)
        if reason is not None:
            self._quarantine(identity, reason)
            return result

        self.counters["capture_attempts"] += 1
        try:
            scratch_lease = self.scratch_owner.acquire(
                identity,
                rows,
            )
        except BaseException:
            self._quarantine(identity, "scratch_unavailable")
            return result
        try:
            entry = self.capture_backend.capture(
                identity,
                scratch_lease.rows,
                eager,
                scratch_lease,
            )
        except BaseException:
            try:
                self.scratch_owner.rollback(scratch_lease)
            except BaseException:
                self._quarantine(
                    identity,
                    "capture_rollback_failed",
                )
                raise
            self._quarantine(identity, "capture_failed")
            return result
        try:
            self.scratch_owner.rollback(scratch_lease)
        except BaseException:
            self._quarantine(
                identity,
                "capture_rollback_failed",
            )
            raise
        if entry.identity != identity:
            self._quarantine(identity, "identity_drift")
            return result
        reason = self._post_capture_reason(entry)
        if reason is not None:
            self._quarantine(identity, reason)
            return result
        self.ready_entries[identity_sha256] = entry
        self.static_bytes += entry.static_bytes
        self.reserved_bytes += entry.reserved_delta_bytes
        self.total_capture_ns += entry.capture_duration_ns
        self.counters["captures"] += 1
        return result

    def summary(self) -> dict:
        return {
            "ready_entries": self.ready_identity_sha256s(),
            "quarantined": {
                identity_sha256: self.quarantined[identity_sha256]
                for identity_sha256 in sorted(self.quarantined)
            },
            "observation_counts": {
                identity_sha256: self.observation_counts[
                    identity_sha256
                ]
                for identity_sha256 in sorted(
                    self.observation_counts
                )
            },
            "static_bytes": self.static_bytes,
            "reserved_bytes": self.reserved_bytes,
            "total_capture_ns": self.total_capture_ns,
            **{
                name: self.counters[name]
                for name in (
                    "capture_attempts",
                    "captures",
                    "replays",
                    "quarantines",
                    "fallback_pre_replay",
                )
            },
        }
