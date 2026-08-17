from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


TRACE_SCHEMA = (
    "qwen35.native-mtp-tp4-32k-paired-verify-trace.v1"
)


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def compact_topk_logits(
    logits: torch.Tensor,
    *,
    top_k: int = 5,
) -> tuple[dict, ...]:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("trace logits must be a rank-two tensor")
    top_k = _positive_integer(top_k, "trace top_k")
    if top_k < 5:
        raise ValueError("trace top_k must be at least five")
    if top_k > logits.shape[1]:
        raise ValueError("trace top_k exceeds vocabulary size")
    compact = []
    for values in logits.detach().float().cpu().tolist():
        ranked = sorted(
            enumerate(values),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        top_tokens = tuple(
            int(token_id) for token_id, _ in ranked
        )
        top_logits = tuple(
            float(value) for _, value in ranked
        )
        compact.append({
            "top_tokens": top_tokens,
            "top_logits": top_logits,
            "top1_margin": (
                float(top_logits[0] - top_logits[1])
                if len(top_logits) > 1
                else None
            ),
            "argmax_token": top_tokens[0],
        })
    return tuple(compact)


def logical_block_coverage(
    context_length: int,
    block_size: int,
) -> tuple[tuple[int, int, int], ...]:
    context_length = _positive_integer(
        context_length,
        "trace context_length",
    )
    block_size = _positive_integer(
        block_size,
        "trace block_size",
    )
    return tuple(
        (
            block_ordinal,
            block_ordinal * block_size,
            min(
                context_length,
                (block_ordinal + 1) * block_size,
            ),
        )
        for block_ordinal in range(
            (context_length + block_size - 1) // block_size
        )
    )


@dataclass(frozen=True)
class TargetForwardTraceContext:
    policy: str
    batch_size: int
    engine_step: int

    def __post_init__(self):
        if self.policy not in ("baseline", "native_mtp"):
            raise ValueError("trace policy is invalid")
        _positive_integer(self.batch_size, "trace batch_size")
        _non_negative_integer(
            self.engine_step,
            "trace engine_step",
        )


@dataclass(frozen=True)
class TargetForwardTraceRow:
    schema: str
    policy: str
    batch_size: int
    engine_step: int
    target_forward_ordinal: int
    stage: str
    execution_mode: str
    sequence_id: int
    query_offset: int
    query_len: int
    row_index: int
    prediction_index: int
    input_token_id: int
    position: int
    context_length: int
    logical_block_identities: tuple[tuple[int, int], ...]
    logical_block_coverage: tuple[tuple[int, int, int], ...]
    top_tokens: tuple[int, ...]
    top_logits: tuple[float, ...]
    top1_margin: float | None
    argmax_token: int

    def as_dict(self) -> dict:
        return asdict(self)


class SpecVerifyTraceRecorder:
    def __init__(self, *, rank: int, block_size: int):
        self.rank = _non_negative_integer(rank, "trace rank")
        self.block_size = _positive_integer(
            block_size,
            "trace block_size",
        )
        self._enabled = False
        self._context = None
        self._rows = []
        self._target_forward_ordinal = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, enabled: bool) -> dict:
        if not isinstance(enabled, bool):
            raise ValueError("trace enabled must be a boolean")
        self._enabled = enabled
        self._rows.clear()
        self._context = None
        self._target_forward_ordinal = 0
        return {"rank": self.rank, "enabled": enabled}

    def set_context(
        self,
        context: TargetForwardTraceContext,
    ) -> None:
        if not self._enabled:
            return
        if type(context) is not TargetForwardTraceContext:
            raise ValueError("trace context type mismatch")
        self._context = context

    def record_rows(
        self,
        *,
        stage: str,
        execution_mode: str,
        sequence_ids: tuple[int, ...],
        query_offset: int,
        query_len: int,
        input_tokens: tuple[int, ...],
        positions: tuple[int, ...],
        prediction_indices: tuple[int, ...],
        logical_block_identities: tuple[
            tuple[tuple[int, int], ...],
            ...,
        ],
        logits: torch.Tensor,
    ) -> None:
        if not self._enabled or self.rank != 0:
            return
        if self._context is None:
            raise RuntimeError("trace context is missing")
        if stage not in (
            "ordinary_decode",
            "first_target",
            "verify_tail",
        ):
            raise ValueError("trace stage is invalid")
        if not isinstance(execution_mode, str) or not execution_mode:
            raise ValueError("trace execution mode is invalid")
        query_offset = _non_negative_integer(
            query_offset,
            "trace query_offset",
        )
        query_len = _positive_integer(
            query_len,
            "trace query_len",
        )
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("trace logits must be a rank-two tensor")
        row_count = len(input_tokens)
        if (
            row_count != len(positions)
            or row_count != len(prediction_indices)
            or row_count != logits.shape[0]
            or len(sequence_ids)
            != len(logical_block_identities)
            or row_count != len(sequence_ids) * query_len
        ):
            raise ValueError("trace row inventory mismatch")
        compact_rows = compact_topk_logits(logits, top_k=5)
        forward_ordinal = self._target_forward_ordinal
        self._target_forward_ordinal += 1
        for flat_index, compact in enumerate(compact_rows):
            sequence_index = flat_index // query_len
            position = _non_negative_integer(
                int(positions[flat_index]),
                "trace position",
            )
            identities = logical_block_identities[
                sequence_index
            ]
            coverage = logical_block_coverage(
                position + 1,
                self.block_size,
            )
            if len(identities) < len(coverage):
                raise ValueError(
                    "trace block identity coverage is incomplete"
                )
            if any(
                not isinstance(identity, tuple)
                or len(identity) != 2
                or isinstance(identity[0], bool)
                or not isinstance(identity[0], int)
                or identity[0] < 0
                or isinstance(identity[1], bool)
                or not isinstance(identity[1], int)
                or identity[1] < 0
                for identity in identities
            ):
                raise ValueError(
                    "trace block identity is invalid"
                )
            self._rows.append(TargetForwardTraceRow(
                schema=TRACE_SCHEMA,
                policy=self._context.policy,
                batch_size=self._context.batch_size,
                engine_step=self._context.engine_step,
                target_forward_ordinal=forward_ordinal,
                stage=stage,
                execution_mode=execution_mode,
                sequence_id=_non_negative_integer(
                    int(sequence_ids[sequence_index]),
                    "trace sequence_id",
                ),
                query_offset=(
                    query_offset
                    + sequence_index * query_len
                ),
                query_len=query_len,
                row_index=flat_index,
                prediction_index=_non_negative_integer(
                    int(prediction_indices[flat_index]),
                    "trace prediction_index",
                ),
                input_token_id=int(input_tokens[flat_index]),
                position=position,
                context_length=position + 1,
                logical_block_identities=identities,
                logical_block_coverage=coverage,
                **compact,
            ))

    def drain(self) -> tuple[dict, ...]:
        rows = tuple(row.as_dict() for row in self._rows)
        self._rows.clear()
        return rows
