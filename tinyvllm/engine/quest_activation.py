from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QuestActivationDecision:
    requested_top_k: int
    resolved_top_k: int
    min_seq_len: int
    min_saved_blocks: int
    saved_blocks: int | None
    batch_size: int
    reason: str


def _decision(
    *,
    requested_top_k: int,
    resolved_top_k: int,
    min_seq_len: int,
    min_saved_blocks: int,
    saved_blocks: int | None,
    batch_size: int,
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
