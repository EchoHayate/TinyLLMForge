from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Sequence


QUEST_ACTIVATION_EVENT_CAPACITY = 4096


@dataclass(frozen=True)
class QuestActivationDecision:
    requested_top_k: int
    resolved_top_k: int
    min_seq_len: int
    min_saved_blocks: int
    saved_blocks: int | None
    batch_size: int
    sequence_lengths: tuple[int, ...]
    sequence_block_counts: tuple[int, ...]
    reason: str


class QuestActivationTelemetry:
    def __init__(self) -> None:
        self._observation_id = 0
        self._latest = None
        self._reason_counts = {}
        self._resolved_top_k_counts = {}
        self._saved_blocks_min = None
        self._saved_blocks_max = None
        self._events = deque(
            maxlen=QUEST_ACTIVATION_EVENT_CAPACITY,
        )

    def publish(
        self,
        decision: QuestActivationDecision,
    ) -> dict:
        self._observation_id += 1
        event = {
            "observation_id": self._observation_id,
            **asdict(decision),
        }
        self._latest = event
        self._events.append(dict(event))
        self._reason_counts[decision.reason] = (
            self._reason_counts.get(decision.reason, 0) + 1
        )
        resolved_key = str(decision.resolved_top_k)
        self._resolved_top_k_counts[resolved_key] = (
            self._resolved_top_k_counts.get(resolved_key, 0) + 1
        )
        if decision.saved_blocks is not None:
            if self._saved_blocks_min is None:
                self._saved_blocks_min = decision.saved_blocks
                self._saved_blocks_max = decision.saved_blocks
            else:
                self._saved_blocks_min = min(
                    self._saved_blocks_min,
                    decision.saved_blocks,
                )
                self._saved_blocks_max = max(
                    self._saved_blocks_max,
                    decision.saved_blocks,
                )
        return dict(event)

    def observation(self) -> dict | None:
        return (
            None
            if self._latest is None
            else dict(self._latest)
        )

    def summary(self) -> dict:
        return {
            "steps": self._observation_id,
            "reason_counts": dict(
                sorted(self._reason_counts.items())
            ),
            "resolved_top_k_counts": dict(
                sorted(self._resolved_top_k_counts.items())
            ),
            "saved_blocks_min": self._saved_blocks_min,
            "saved_blocks_max": self._saved_blocks_max,
            "last_observation_id": self._observation_id,
            "events": [
                dict(event) for event in self._events
            ],
            "events_dropped": max(
                0,
                self._observation_id - len(self._events),
            ),
        }


def _decision(
    *,
    requested_top_k: int,
    resolved_top_k: int,
    min_seq_len: int,
    min_saved_blocks: int,
    saved_blocks: int | None,
    batch_size: int,
    sequence_lengths: tuple[int, ...],
    sequence_block_counts: tuple[int, ...],
    reason: str,
) -> QuestActivationDecision:
    return QuestActivationDecision(
        requested_top_k=int(requested_top_k),
        resolved_top_k=int(resolved_top_k),
        min_seq_len=int(min_seq_len),
        min_saved_blocks=int(min_saved_blocks),
        saved_blocks=(
            None if saved_blocks is None else int(saved_blocks)
        ),
        batch_size=int(batch_size),
        sequence_lengths=tuple(sequence_lengths),
        sequence_block_counts=tuple(sequence_block_counts),
        reason=str(reason),
    )


def resolve_quest_activation(
    *,
    requested_top_k: int,
    min_seq_len: int,
    min_saved_blocks: int,
    block_size: int,
    sequence_lengths: Sequence[int],
    sequence_block_counts: Sequence[int],
    incompatible_feature: bool,
) -> QuestActivationDecision:
    lengths = tuple(int(value) for value in sequence_lengths)
    block_counts = tuple(
        int(value) for value in sequence_block_counts
    )
    if len(lengths) != len(block_counts):
        raise ValueError(
            "sequence lengths and block counts must have equal size"
        )

    common = {
        "requested_top_k": requested_top_k,
        "resolved_top_k": -1,
        "min_seq_len": min_seq_len,
        "min_saved_blocks": min_saved_blocks,
        "saved_blocks": None,
        "batch_size": len(lengths),
        "sequence_lengths": lengths,
        "sequence_block_counts": block_counts,
    }
    if requested_top_k <= 0 or not lengths:
        return _decision(**common, reason="disabled")
    if incompatible_feature:
        return _decision(
            **common,
            reason="incompatible_feature",
        )
    if min(lengths) < min_seq_len:
        return _decision(**common, reason="below_min_seq_len")
    if min(block_counts) <= requested_top_k:
        return _decision(**common, reason="insufficient_blocks")
    if requested_top_k * block_size >= max(lengths) * 0.8:
        return _decision(
            **common,
            reason="insufficient_pruning",
        )

    saved_blocks = sum(
        max(0, blocks - requested_top_k)
        for blocks in block_counts
    )
    if min_saved_blocks > 0 and saved_blocks < min_saved_blocks:
        return _decision(
            **{
                **common,
                "saved_blocks": saved_blocks,
            },
            reason="below_saved_blocks",
        )
    return _decision(
        **{
            **common,
            "resolved_top_k": requested_top_k,
            "saved_blocks": saved_blocks,
        },
        reason="active",
    )
