from __future__ import annotations

import json
import os
from pathlib import Path
import time


SCHEMA_VERSION = 2
RECEIPT_ROOT_ENVIRONMENT = (
    "TINYVLLM_EXACT_GRAPH_CAPTURE_RECEIPT_ROOT"
)
CAPTURE_PHASES = (
    "entered_capture",
    "hot_path_eager_prerequisite",
    "capture_begin",
    "capture_body_completed",
    "capture_end_synchronize_completed",
    "scratch_restore_completed",
)
REPLAY_PHASES = (
    "entered_replay",
    "lease_manifest_validated",
    "static_inputs_copied",
    "context_set",
    "graph_replay_returned",
    "logits_compute_returned",
    "context_reset_completed",
)


def _validate_sha256(
    name: str,
    value: str | None,
    *,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _validate_identity_fields(
    *,
    execution_protocol: str,
    program_key_sha256: str,
    invocation_identity_sha256: str,
    lease_manifest_sha256: str | None,
    ordered_slot_ids: tuple[int, ...] | list[int],
    cross_lease_replay: bool,
) -> tuple[str | None, tuple[int, ...]]:
    if execution_protocol not in {
        "forward_v1",
        "lease_transaction_v1",
        "lease_pool_index_v1",
    }:
        raise ValueError("execution_protocol is invalid")
    _validate_sha256("program_key_sha256", program_key_sha256)
    _validate_sha256(
        "invocation_identity_sha256",
        invocation_identity_sha256,
    )
    manifest_digest = _validate_sha256(
        "lease_manifest_sha256",
        lease_manifest_sha256,
        optional=execution_protocol != "lease_pool_index_v1",
    )
    if (
        not isinstance(ordered_slot_ids, (tuple, list))
        or any(
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 0
            for slot_id in ordered_slot_ids
        )
    ):
        raise ValueError(
            "ordered_slot_ids must contain non-negative integers"
        )
    slots = tuple(ordered_slot_ids)
    if execution_protocol == "lease_pool_index_v1":
        if not slots or len(set(slots)) != len(slots):
            raise ValueError(
                "lease_pool_index_v1 requires unique ordered_slot_ids"
            )
    elif manifest_digest is not None or slots:
        raise ValueError(
            "lease manifest evidence requires lease_pool_index_v1"
        )
    if not isinstance(cross_lease_replay, bool):
        raise ValueError("cross_lease_replay must be boolean")
    if cross_lease_replay and execution_protocol != "lease_pool_index_v1":
        raise ValueError(
            "cross_lease_replay requires lease_pool_index_v1"
        )
    return manifest_digest, slots


class ExactCudaGraphCaptureReceipt:
    def __init__(
        self,
        *,
        root: Path | None,
        rank: int,
        world_size: int,
        execution_protocol: str,
        program_key_sha256: str,
        invocation_identity_sha256: str,
        lease_manifest_sha256: str | None,
        ordered_slot_ids: tuple[int, ...] | list[int],
        cross_lease_replay: bool,
    ) -> None:
        manifest_digest, slots = _validate_identity_fields(
            execution_protocol=execution_protocol,
            program_key_sha256=program_key_sha256,
            invocation_identity_sha256=invocation_identity_sha256,
            lease_manifest_sha256=lease_manifest_sha256,
            ordered_slot_ids=ordered_slot_ids,
            cross_lease_replay=cross_lease_replay,
        )
        if cross_lease_replay:
            raise ValueError(
                "capture receipt cannot claim cross-lease replay"
            )
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.execution_protocol = execution_protocol
        self.program_key_sha256 = program_key_sha256
        self.invocation_identity_sha256 = invocation_identity_sha256
        self.lease_manifest_sha256 = manifest_digest
        self.ordered_slot_ids = slots
        self.cross_lease_replay = cross_lease_replay
        self.completed_phases: list[dict[str, int | str]] = []
        self._last_phase_index = -1

    @classmethod
    def from_environment(
        cls,
        *,
        rank: int,
        world_size: int,
        execution_protocol: str,
        program_key_sha256: str,
        invocation_identity_sha256: str,
        lease_manifest_sha256: str | None,
        ordered_slot_ids: tuple[int, ...] | list[int],
        cross_lease_replay: bool,
    ) -> ExactCudaGraphCaptureReceipt:
        root_value = os.environ.get(RECEIPT_ROOT_ENVIRONMENT)
        root = None if not root_value else Path(root_value)
        if root is not None and not root.is_absolute():
            raise ValueError(
                f"{RECEIPT_ROOT_ENVIRONMENT} must be an absolute path"
            )
        return cls(
            root=root,
            rank=rank,
            world_size=world_size,
            execution_protocol=execution_protocol,
            program_key_sha256=program_key_sha256,
            invocation_identity_sha256=invocation_identity_sha256,
            lease_manifest_sha256=lease_manifest_sha256,
            ordered_slot_ids=ordered_slot_ids,
            cross_lease_replay=cross_lease_replay,
        )

    def record(self, phase: str) -> None:
        if self.root is None:
            return
        try:
            phase_index = CAPTURE_PHASES.index(phase)
        except ValueError as error:
            raise ValueError(
                "unknown exact CUDA Graph capture receipt phase"
            ) from error
        if phase_index <= self._last_phase_index:
            raise ValueError(
                "exact CUDA Graph capture receipt phase order violation"
            )
        self._last_phase_index = phase_index
        self.completed_phases.append(
            {
                "phase": phase,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
            }
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "identity_sha256": self.invocation_identity_sha256,
            "execution_protocol": self.execution_protocol,
            "program_key_sha256": self.program_key_sha256,
            "invocation_identity_sha256": (
                self.invocation_identity_sha256
            ),
            "lease_manifest_sha256": self.lease_manifest_sha256,
            "ordered_slot_ids": list(self.ordered_slot_ids),
            "cross_lease_replay": self.cross_lease_replay,
            "lease_manifest_validated": True,
            "completed_phases": list(self.completed_phases),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / (
            f".rank-{self.rank}.{os.getpid()}.json.tmp"
        )
        final = self.root / f"rank-{self.rank}.json"
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        temporary.replace(final)


class ExactCudaGraphReplayReceipt:
    def __init__(
        self,
        *,
        root: Path | None,
        rank: int,
        world_size: int,
        execution_protocol: str,
        program_key_sha256: str,
        invocation_identity_sha256: str,
        lease_manifest_sha256: str | None,
        ordered_slot_ids: tuple[int, ...] | list[int],
        cross_lease_replay: bool,
        replay_ordinal: int,
    ) -> None:
        manifest_digest, slots = _validate_identity_fields(
            execution_protocol=execution_protocol,
            program_key_sha256=program_key_sha256,
            invocation_identity_sha256=invocation_identity_sha256,
            lease_manifest_sha256=lease_manifest_sha256,
            ordered_slot_ids=ordered_slot_ids,
            cross_lease_replay=cross_lease_replay,
        )
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.execution_protocol = execution_protocol
        self.program_key_sha256 = program_key_sha256
        self.invocation_identity_sha256 = invocation_identity_sha256
        self.lease_manifest_sha256 = manifest_digest
        self.ordered_slot_ids = slots
        self.cross_lease_replay = cross_lease_replay
        self.replay_ordinal = replay_ordinal
        self.completed_phases: list[dict[str, int | str]] = []
        self._last_phase_index = -1

    @classmethod
    def from_environment(
        cls,
        *,
        rank: int,
        world_size: int,
        execution_protocol: str,
        program_key_sha256: str,
        invocation_identity_sha256: str,
        lease_manifest_sha256: str | None,
        ordered_slot_ids: tuple[int, ...] | list[int],
        cross_lease_replay: bool,
        replay_ordinal: int,
    ) -> ExactCudaGraphReplayReceipt:
        root_value = os.environ.get(RECEIPT_ROOT_ENVIRONMENT)
        root = None if not root_value else Path(root_value)
        if root is not None and not root.is_absolute():
            raise ValueError(
                f"{RECEIPT_ROOT_ENVIRONMENT} must be an absolute path"
            )
        return cls(
            root=root,
            rank=rank,
            world_size=world_size,
            execution_protocol=execution_protocol,
            program_key_sha256=program_key_sha256,
            invocation_identity_sha256=invocation_identity_sha256,
            lease_manifest_sha256=lease_manifest_sha256,
            ordered_slot_ids=ordered_slot_ids,
            cross_lease_replay=cross_lease_replay,
            replay_ordinal=replay_ordinal,
        )

    def record(self, phase: str) -> None:
        if self.root is None:
            return
        try:
            phase_index = REPLAY_PHASES.index(phase)
        except ValueError as error:
            raise ValueError(
                "unknown exact CUDA Graph replay receipt phase"
            ) from error
        if phase_index <= self._last_phase_index:
            raise ValueError(
                "exact CUDA Graph replay receipt phase order violation"
            )
        self._last_phase_index = phase_index
        self.completed_phases.append(
            {
                "phase": phase,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
            }
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "identity_sha256": self.invocation_identity_sha256,
            "execution_protocol": self.execution_protocol,
            "program_key_sha256": self.program_key_sha256,
            "invocation_identity_sha256": (
                self.invocation_identity_sha256
            ),
            "lease_manifest_sha256": self.lease_manifest_sha256,
            "ordered_slot_ids": list(self.ordered_slot_ids),
            "cross_lease_replay": self.cross_lease_replay,
            "lease_manifest_validated": any(
                row["phase"] == "lease_manifest_validated"
                for row in self.completed_phases
            ),
            "replay_ordinal": self.replay_ordinal,
            "completed_phases": list(self.completed_phases),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / (
            f".rank-{self.rank}-replay.{os.getpid()}.json.tmp"
        )
        final = self.root / f"rank-{self.rank}-replay.json"
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        temporary.replace(final)
