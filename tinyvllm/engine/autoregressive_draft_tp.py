from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class AutoregressiveDraftRankRegistrationStatus:
    rank: int
    world_size: int
    success: bool
    stage: str
    error_type: str | None
    message: str | None
    target_checkpoint_sha256: str | None
    draft_checkpoint_sha256: str | None
    target_tokenizer_sha256: str | None
    draft_tokenizer_sha256: str | None
    backend_identity: str | None
    executor_id: str | None
    capabilities_sha256: str | None


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rank(value, world_size: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= world_size
    ):
        raise ValueError("rank must be in [0, world_size)")
    return value


def _stage(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("stage must be a non-empty string")
    return value


def _canonical_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("logical floats must be finite")
        return value
    if isinstance(value, torch.Tensor):
        raise TypeError("logical authority rows must not contain tensors")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(
                "logical authority dictionaries require string keys"
            )
        return {
            key: _canonical_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("logical authority sets are not supported")
    raise TypeError(
        "logical authority contains unsupported "
        f"value type {type(value).__name__}"
    )


def _logical_digest(*, stage: str, rows) -> str:
    payload = {
        "stage": _stage(stage),
        "rows": _canonical_value(rows),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _failure_digest(*, stage: str, error: BaseException) -> str:
    payload = {
        "stage": _stage(stage),
        "error_type": type(error).__name__,
        "message": str(error),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AutoregressiveDraftTensorParallelCoordinator:

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: object,
        gather_registration_status=None,
        gather_digest=None,
    ):
        world_size = _positive_integer(world_size, "world_size")
        if world_size not in (1, 4):
            raise RuntimeError(
                "autoregressive draft tensor parallelism "
                "supports TP1 or TP4"
            )
        self.rank = _rank(rank, world_size)
        self.world_size = world_size
        self.device = torch.device(device)
        self._gather_registration_status = (
            self._default_gather_registration_status
            if gather_registration_status is None
            else gather_registration_status
        )
        self._gather_digest = (
            self._default_gather_digest
            if gather_digest is None
            else gather_digest
        )
        if not callable(self._gather_registration_status):
            raise ValueError(
                "gather_registration_status must be callable"
            )
        if not callable(self._gather_digest):
            raise ValueError("gather_digest must be callable")

    def _default_gather_registration_status(
        self,
        status: AutoregressiveDraftRankRegistrationStatus,
    ) -> tuple[AutoregressiveDraftRankRegistrationStatus, ...]:
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, status)
        return tuple(gathered)

    def _default_gather_digest(
        self,
        payload: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        gathered = [
            torch.empty_like(payload)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, payload)
        return tuple(gathered)

    def collect_registration_status(
        self,
        status: AutoregressiveDraftRankRegistrationStatus,
    ) -> tuple[AutoregressiveDraftRankRegistrationStatus, ...]:
        if (
            not isinstance(
                status,
                AutoregressiveDraftRankRegistrationStatus,
            )
            or status.rank != self.rank
            or status.world_size != self.world_size
        ):
            raise ValueError(
                "local registration status must match coordinator "
                "rank and world_size"
            )
        if self.world_size == 1:
            return (status,)
        gathered = self._gather_registration_status(status)
        if not isinstance(gathered, (tuple, list)):
            raise RuntimeError(
                "registration gather must return a sequence"
            )
        if len(gathered) != self.world_size:
            raise RuntimeError(
                "registration gather must return exactly world_size "
                "statuses"
            )
        if any(
            not isinstance(
                row,
                AutoregressiveDraftRankRegistrationStatus,
            )
            for row in gathered
        ):
            raise RuntimeError(
                "registration gather returned an invalid "
                "registration status"
            )
        statuses = tuple(gathered)
        if any(
            row.world_size != self.world_size
            for row in statuses
        ):
            raise RuntimeError(
                "registration status world_size mismatch"
            )
        if tuple(sorted(row.rank for row in statuses)) != tuple(
            range(self.world_size)
        ):
            raise RuntimeError(
                "registration statuses must contain ranks "
                "0..world_size-1 exactly once"
            )
        return tuple(sorted(statuses, key=lambda row: row.rank))

    def _payload(
        self,
        *,
        stage: str,
        rows,
        local_error: BaseException | None,
    ) -> tuple[torch.Tensor, str]:
        digest = (
            _logical_digest(stage=stage, rows=rows)
            if local_error is None
            else _failure_digest(stage=stage, error=local_error)
        )
        payload = torch.empty(
            33,
            dtype=torch.uint8,
            device=self.device,
        )
        payload[0] = 1 if local_error is None else 0
        payload[1:] = torch.tensor(
            tuple(bytes.fromhex(digest)),
            dtype=torch.uint8,
            device=self.device,
        )
        return payload.contiguous(), digest

    def _validated_digest_rows(
        self,
        gathered,
        *,
        stage: str,
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(gathered, (tuple, list)):
            raise RuntimeError(
                f"{stage} digest gather must return a sequence"
            )
        if len(gathered) != self.world_size:
            raise RuntimeError(
                f"{stage} digest gather must return exactly "
                "world_size rows"
            )
        rows = tuple(gathered)
        for row in rows:
            if not isinstance(row, torch.Tensor):
                raise RuntimeError(
                    f"{stage} digest gather returned a non-tensor row"
                )
            if row.dtype != torch.uint8 or row.shape != (33,):
                raise RuntimeError(
                    f"{stage} digest rows must use torch.uint8[33]"
                )
            if row.device != self.device:
                raise RuntimeError(
                    f"{stage} digest row device mismatch"
                )
            if not row.is_contiguous():
                raise RuntimeError(
                    f"{stage} digest rows must be contiguous"
                )
            if int(row[0].item()) not in (0, 1):
                raise RuntimeError(
                    f"{stage} digest success bit must be zero or one"
                )
        return rows

    def assert_logical_authority(
        self,
        *,
        stage: str,
        rows: object,
    ) -> str:
        return self.converge_stage(
            stage=stage,
            rows=rows,
            local_error=None,
        )

    def converge_stage(
        self,
        *,
        stage: str,
        rows: object,
        local_error: BaseException | None,
    ) -> str:
        stage = _stage(stage)
        payload, digest = self._payload(
            stage=stage,
            rows=rows,
            local_error=local_error,
        )
        gathered = (
            (payload,)
            if self.world_size == 1
            else self._validated_digest_rows(
                self._gather_digest(payload),
                stage=stage,
            )
        )
        failed_ranks = tuple(
            rank
            for rank, row in enumerate(gathered)
            if int(row[0].item()) == 0
        )
        if failed_ranks:
            ranks = ", ".join(str(rank) for rank in failed_ranks)
            common_error = RuntimeError(
                f"{stage} failed on rank {ranks}"
            )
            if local_error is not None:
                raise common_error from local_error
            raise common_error
        reference_digest = gathered[0][1:]
        if any(
            not torch.equal(row[1:], reference_digest)
            for row in gathered[1:]
        ):
            raise RuntimeError(
                f"{stage} logical authority mismatch across ranks"
            )
        return digest
