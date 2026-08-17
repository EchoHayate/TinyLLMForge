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


def _nonnegative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _identity_string(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class AutoregressiveDraftGraphIdentity:
    exact_q: int
    exact_batch_size: int
    tensor_parallel_size: int
    tensor_parallel_rank: int
    device_index: int
    compute_dtype: str
    backend_identity: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    local_query_heads: int
    local_kv_heads: int
    kv_block_table_width: int
    proposal_kv_capacity: int
    blockwise_offload: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.exact_q, bool)
            or not isinstance(self.exact_q, int)
            or self.exact_q < 2
        ):
            raise ValueError(
                "exact_q must be an integer at least two"
            )
        _positive_integer(
            self.exact_batch_size,
            "exact_batch_size",
        )
        world_size = _positive_integer(
            self.tensor_parallel_size,
            "tensor_parallel_size",
        )
        rank = _nonnegative_integer(
            self.tensor_parallel_rank,
            "tensor_parallel_rank",
        )
        if rank >= world_size:
            raise ValueError(
                "tensor_parallel_rank must be in "
                "[0, tensor_parallel_size)"
            )
        _nonnegative_integer(
            self.device_index,
            "device_index",
        )
        _identity_string(
            self.compute_dtype,
            "compute_dtype",
        )
        _identity_string(
            self.backend_identity,
            "backend_identity",
        )
        _identity_string(
            self.model_fingerprint,
            "model_fingerprint",
        )
        _identity_string(
            self.tokenizer_fingerprint,
            "tokenizer_fingerprint",
        )
        _positive_integer(
            self.local_query_heads,
            "local_query_heads",
        )
        _positive_integer(
            self.local_kv_heads,
            "local_kv_heads",
        )
        _positive_integer(
            self.kv_block_table_width,
            "kv_block_table_width",
        )
        _positive_integer(
            self.proposal_kv_capacity,
            "proposal_kv_capacity",
        )
        if not isinstance(self.blockwise_offload, bool):
            raise ValueError(
                "blockwise_offload must be a bool"
            )

    @property
    def sha256(self) -> str:
        payload = {
            "backend_identity": self.backend_identity,
            "blockwise_offload": self.blockwise_offload,
            "compute_dtype": self.compute_dtype,
            "device_index": self.device_index,
            "exact_batch_size": self.exact_batch_size,
            "exact_q": self.exact_q,
            "kv_block_table_width": (
                self.kv_block_table_width
            ),
            "local_kv_heads": self.local_kv_heads,
            "local_query_heads": self.local_query_heads,
            "model_fingerprint": self.model_fingerprint,
            "proposal_kv_capacity": (
                self.proposal_kv_capacity
            ),
            "tensor_parallel_rank": (
                self.tensor_parallel_rank
            ),
            "tensor_parallel_size": (
                self.tensor_parallel_size
            ),
            "tokenizer_fingerprint": (
                self.tokenizer_fingerprint
            ),
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()


@dataclass
class AutoregressiveDraftGraphEntry:
    identity: AutoregressiveDraftGraphIdentity
    graph: object
    static_bytes: int
    capture_duration_ns: int
    reserved_delta_bytes: int


class AutoregressiveDraftGraphPreReplayError(RuntimeError):
    pass


class AutoregressiveDraftGraphReplayError(RuntimeError):

    def __init__(
        self,
        identity: AutoregressiveDraftGraphIdentity,
        cause: BaseException,
    ):
        super().__init__(
            "autoregressive draft CUDA graph replay failed for "
            f"{identity.sha256}"
        )
        self.identity = identity
        self.cause = cause


class AutoregressiveDraftExactGraphRunner:

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
        tensor_parallel_size: int,
        tensor_parallel_rank: int,
        device_index: int,
        compute_dtype: str,
        backend_identity: str,
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        local_query_heads: int,
        local_kv_heads: int,
        kv_block_table_width: int,
        proposal_kv_capacity: int,
        blockwise_offload: bool,
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
        identity_probe = AutoregressiveDraftGraphIdentity(
            exact_q=self.q_allowlist[0],
            exact_batch_size=self.batch_allowlist[0],
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            device_index=device_index,
            compute_dtype=compute_dtype,
            backend_identity=backend_identity,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            local_query_heads=local_query_heads,
            local_kv_heads=local_kv_heads,
            kv_block_table_width=kv_block_table_width,
            proposal_kv_capacity=proposal_kv_capacity,
            blockwise_offload=blockwise_offload,
        )
        self.tensor_parallel_size = (
            identity_probe.tensor_parallel_size
        )
        self.tensor_parallel_rank = (
            identity_probe.tensor_parallel_rank
        )
        self.device_index = identity_probe.device_index
        self.compute_dtype = identity_probe.compute_dtype
        self.backend_identity = (
            identity_probe.backend_identity
        )
        self.model_fingerprint = (
            identity_probe.model_fingerprint
        )
        self.tokenizer_fingerprint = (
            identity_probe.tokenizer_fingerprint
        )
        self.local_query_heads = (
            identity_probe.local_query_heads
        )
        self.local_kv_heads = identity_probe.local_kv_heads
        self.kv_block_table_width = (
            identity_probe.kv_block_table_width
        )
        self.proposal_kv_capacity = (
            identity_probe.proposal_kv_capacity
        )
        self.blockwise_offload = (
            identity_probe.blockwise_offload
        )
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
        self.ready_entries: dict[
            str,
            AutoregressiveDraftGraphEntry,
        ] = {}
        self.quarantined: dict[str, str] = {}
        self.static_bytes = 0
        self.reserved_bytes = 0
        self.total_capture_ns = 0
        self.counters = collections.Counter()
        self._converge = None

    def bind_convergence(self, convergence) -> None:
        if not callable(convergence):
            raise ValueError("convergence must be callable")
        if self._converge is not None and (
            self._converge != convergence
        ):
            raise RuntimeError(
                "graph convergence callback is already bound"
            )
        self._converge = convergence

    @staticmethod
    def _sequence_ids(rows) -> tuple[int, ...]:
        sequence_ids = []
        for row in rows:
            candidate = (
                row[1]
                if isinstance(row, tuple) and len(row) == 3
                else row
            )
            sequence_ids.append(candidate.sequence_id)
        return tuple(sequence_ids)

    @staticmethod
    def _transaction_ids(prepared) -> tuple[str, ...]:
        if prepared is None:
            return ()
        return tuple(
            transaction.transaction_id
            for transaction in prepared.transactions
        )

    def _convergence_rows(
        self,
        *,
        identity,
        rows,
        prepared,
        result=None,
    ) -> dict:
        authority_rows = {
            "exact_q": identity.exact_q,
            "sequence_ids": self._sequence_ids(rows),
            "transaction_ids": self._transaction_ids(
                prepared
            ),
        }
        if result is not None:
            authority_rows["token_rows"] = tuple(
                getattr(result, "token_rows", result)
            )
        return authority_rows

    def _run_prepared_replay(
        self,
        *,
        identity,
        entry,
        rows,
        eager,
    ):
        prepared = None
        preflight_error = None
        try:
            prepared = self.capture_backend.prepare_replay(
                entry,
                rows,
            )
        except BaseException as error:
            preflight_error = error
        convergence_error = None
        if self._converge is not None:
            try:
                self._converge(
                    stage="graph_pre_replay",
                    rows=self._convergence_rows(
                        identity=identity,
                        rows=rows,
                        prepared=prepared,
                    ),
                    local_error=preflight_error,
                )
            except BaseException as error:
                convergence_error = error
        if (
            preflight_error is not None
            or convergence_error is not None
        ):
            if prepared is not None:
                self.capture_backend.abort_prepared(prepared)
            self.counters["fallback_pre_replay"] += 1
            return eager(identity.exact_q, rows)

        result = None
        replay_error = None
        try:
            result = self.capture_backend.replay_prepared(
                entry,
                rows,
                prepared,
            )
        except BaseException as error:
            replay_error = error
        completion_error = None
        if self._converge is not None:
            try:
                self._converge(
                    stage="graph_replay_complete",
                    rows=self._convergence_rows(
                        identity=identity,
                        rows=rows,
                        prepared=prepared,
                        result=result,
                    ),
                    local_error=replay_error,
                )
            except BaseException as error:
                completion_error = error
        if (
            replay_error is not None
            or completion_error is not None
        ):
            if result is not None:
                self.capture_backend.abort_replay_result(
                    result
                )
            error = (
                replay_error
                if replay_error is not None
                else completion_error
            )
            self._quarantine(identity, "replay_failed")
            raise AutoregressiveDraftGraphReplayError(
                identity,
                error,
            ) from error
        self.counters["replays"] += 1
        return result

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
            raise ValueError(
                f"{name} must be a canonical tuple"
            )
        return values

    def _identity(
        self,
        exact_q: int,
        exact_batch_size: int,
    ) -> AutoregressiveDraftGraphIdentity:
        return AutoregressiveDraftGraphIdentity(
            exact_q=exact_q,
            exact_batch_size=exact_batch_size,
            tensor_parallel_size=self.tensor_parallel_size,
            tensor_parallel_rank=self.tensor_parallel_rank,
            device_index=self.device_index,
            compute_dtype=self.compute_dtype,
            backend_identity=self.backend_identity,
            model_fingerprint=self.model_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            local_query_heads=self.local_query_heads,
            local_kv_heads=self.local_kv_heads,
            kv_block_table_width=self.kv_block_table_width,
            proposal_kv_capacity=self.proposal_kv_capacity,
            blockwise_offload=self.blockwise_offload,
        )

    def _quarantine(
        self,
        identity: AutoregressiveDraftGraphIdentity,
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

    def _pre_capture_reason(
        self,
        estimated_static_bytes: int,
    ) -> str | None:
        if len(self.ready_entries) >= self.max_entries:
            return "entry_limit"
        if (
            self.static_bytes + max(
                0,
                int(estimated_static_bytes),
            )
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
        entry: AutoregressiveDraftGraphEntry,
    ) -> str | None:
        if not isinstance(
            entry,
            AutoregressiveDraftGraphEntry,
        ):
            return "capture_failed"
        if entry.identity.sha256 in self.quarantined:
            return self.quarantined[entry.identity.sha256]
        if (
            entry.static_bytes < 0
            or entry.reserved_delta_bytes < 0
            or entry.capture_duration_ns < 0
        ):
            return "capture_failed"
        if (
            entry.capture_duration_ns
            > self.max_single_capture_ns
        ):
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
            prepared_methods = (
                "prepare_replay",
                "replay_prepared",
                "abort_prepared",
                "abort_replay_result",
            )
            if all(
                callable(
                    getattr(
                        self.capture_backend,
                        method_name,
                        None,
                    )
                )
                for method_name in prepared_methods
            ):
                return self._run_prepared_replay(
                    identity=identity,
                    entry=entry,
                    rows=rows,
                    eager=eager,
                )
            try:
                result = self.capture_backend.replay(
                    entry,
                    rows,
                )
            except AutoregressiveDraftGraphPreReplayError:
                self.counters["fallback_pre_replay"] += 1
                return eager(exact_q, rows)
            except BaseException as error:
                self._quarantine(identity, "replay_failed")
                raise AutoregressiveDraftGraphReplayError(
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
            self.observation_counts.get(
                identity_sha256,
                0,
            )
            + 1
        )
        self.observation_counts[
            identity_sha256
        ] = observation_count
        if observation_count < self.min_observations:
            return result
        estimated_static_bytes = (
            self.capture_backend.estimate_static_bytes(
                identity,
                rows,
            )
        )
        reason = self._pre_capture_reason(
            estimated_static_bytes
        )
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
            self._quarantine(
                identity,
                "scratch_unavailable",
            )
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
            "ready_entries": tuple(
                sorted(self.ready_entries)
            ),
            "quarantined": {
                identity_sha256: self.quarantined[
                    identity_sha256
                ]
                for identity_sha256 in sorted(
                    self.quarantined
                )
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
