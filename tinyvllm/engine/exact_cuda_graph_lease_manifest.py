from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

from tinyvllm.engine.hybrid_state import HybridStateLease


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ExactCudaGraphLeaseManifestRow:
    batch_index: int
    slot_id: int
    generation: int
    request_id: int


@dataclass(frozen=True)
class ExactCudaGraphLeaseManifest:
    rows: tuple[ExactCudaGraphLeaseManifestRow, ...]

    @property
    def sha256(self) -> str:
        payload = [asdict(row) for row in self.rows]
        return hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(row.slot_id for row in self.rows)


def build_exact_cuda_graph_lease_manifest(
    *,
    leases: tuple[HybridStateLease, ...],
    expected_request_ids: tuple[int, ...],
    validate_lease: Callable[[HybridStateLease], HybridStateLease],
) -> ExactCudaGraphLeaseManifest:
    if not isinstance(leases, tuple) or not leases:
        raise ValueError(
            "exact CUDA Graph lease manifest must be non-empty"
        )
    if not isinstance(expected_request_ids, tuple):
        raise ValueError("expected_request_ids must be a tuple")
    if len(leases) != len(expected_request_ids):
        raise ValueError("lease and request row counts must match")
    if not callable(validate_lease):
        raise ValueError("validate_lease must be callable")

    rows = []
    seen_slots = set()
    for batch_index, (lease, request_id) in enumerate(
        zip(leases, expected_request_ids)
    ):
        if type(lease) is not HybridStateLease:
            raise ValueError(
                "lease manifest requires HybridStateLease values"
            )
        validated = validate_lease(lease)
        if validated != lease:
            raise RuntimeError("lease validator changed identity")
        if lease.request_id != request_id:
            raise RuntimeError(
                "lease manifest request order mismatch"
            )
        if lease.slot_id in seen_slots:
            raise RuntimeError(
                "lease manifest requires distinct slots"
            )
        seen_slots.add(lease.slot_id)
        rows.append(ExactCudaGraphLeaseManifestRow(
            batch_index=batch_index,
            slot_id=lease.slot_id,
            generation=lease.generation,
            request_id=lease.request_id,
        ))
    return ExactCudaGraphLeaseManifest(tuple(rows))
