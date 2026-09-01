from __future__ import annotations

import json
import os
from pathlib import Path
import time


SCHEMA_VERSION = 1
RECEIPT_ROOT_ENVIRONMENT = (
    "TINYVLLM_EXACT_GRAPH_CAPTURE_RECEIPT_ROOT"
)
CAPTURE_PHASES = (
    "entered_capture",
    "warmup_forward_completed",
    "warmup_synchronize_completed",
    "capture_begin",
    "capture_body_completed",
    "capture_end_synchronize_completed",
    "scratch_restore_completed",
)


class ExactCudaGraphCaptureReceipt:
    def __init__(
        self,
        *,
        root: Path | None,
        rank: int,
        world_size: int,
        identity_sha256: str,
    ) -> None:
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.identity_sha256 = identity_sha256
        self.completed_phases: list[dict[str, int | str]] = []
        self._last_phase_index = -1

    @classmethod
    def from_environment(
        cls,
        *,
        rank: int,
        world_size: int,
        identity_sha256: str,
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
            identity_sha256=identity_sha256,
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
            "identity_sha256": self.identity_sha256,
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
