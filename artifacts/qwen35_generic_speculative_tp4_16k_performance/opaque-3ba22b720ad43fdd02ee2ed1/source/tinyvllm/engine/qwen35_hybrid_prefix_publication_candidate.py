from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm.engine.block_manager import BlockManager
from tinyvllm.engine.hybrid_state import (
    HybridStateLease,
    HybridStateSlotAllocator,
)
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
    Qwen35HybridPrefixSnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket import (
    Qwen35HybridPrefixPublicationPayload,
)
from tinyvllm.engine.sequence import Sequence


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Qwen35HybridPrefixPublicationCandidate:
    request_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    lease: HybridStateLease

    def __post_init__(self):
        _non_negative_integer(self.request_id, "request_id")
        Qwen35HybridPrefixSnapshotCache._validate_identity(
            self.key,
            self.token_ids,
            self.block_identities,
        )
        if not isinstance(self.lease, HybridStateLease):
            raise ValueError("lease must be a HybridStateLease")
        if self.lease.request_id != self.request_id:
            raise ValueError(
                "lease request_id must match candidate request_id"
            )

    def publication_payloads(
        self,
        *,
        ticket_id,
        world_size,
    ):
        ticket_id = _non_negative_integer(ticket_id, "ticket_id")
        world_size = _positive_integer(world_size, "world_size")
        if world_size != self.key.tensor_parallel_size:
            raise ValueError(
                "world_size must match candidate tensor parallel size"
            )
        return tuple(
            Qwen35HybridPrefixPublicationPayload(
                ticket_id=ticket_id,
                participant_id=participant_id,
                request_id=self.request_id,
                key=self.key,
                token_ids=self.token_ids,
                block_identities=self.block_identities,
                lease=self.lease,
            )
            for participant_id in range(world_size)
        )

    def validate_source(
        self,
        sequence,
        block_manager,
        state_allocator,
    ):
        try:
            current = (
                capture_qwen35_hybrid_prefix_publication_candidate(
                    sequence,
                    block_manager,
                    state_allocator,
                    model_fingerprint=self.key.model_fingerprint,
                    layout_fingerprint=self.key.layout_fingerprint,
                    tensor_parallel_size=(
                        self.key.tensor_parallel_size
                    ),
                    dtype=self.key.dtype,
                )
            )
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(
                "publication candidate source identity changed: "
                f"{error}"
            ) from error
        if current != self:
            raise RuntimeError(
                "publication candidate source identity changed"
            )
        return self


def capture_qwen35_hybrid_prefix_publication_candidate(
    sequence,
    block_manager,
    state_allocator,
    *,
    model_fingerprint,
    layout_fingerprint,
    tensor_parallel_size,
    dtype,
):
    if not isinstance(sequence, Sequence):
        raise ValueError("sequence must be a Sequence")
    if not isinstance(block_manager, BlockManager):
        raise ValueError("block_manager must be a BlockManager")
    if not isinstance(
        state_allocator,
        HybridStateSlotAllocator,
    ):
        raise ValueError(
            "state_allocator must be a HybridStateSlotAllocator"
        )
    total_prompt_token_count = _positive_integer(
        sequence.num_prompt_tokens,
        "sequence num_prompt_tokens",
    )
    prompt_token_count = (
        total_prompt_token_count
        // block_manager.block_size
        * block_manager.block_size
    )
    if prompt_token_count <= 0:
        raise ValueError(
            "prompt must contain one complete block"
        )
    if sequence.num_computed_tokens < prompt_token_count:
        raise ValueError(
            "prompt tokens are not fully computed"
        )
    prompt_block_count = (
        prompt_token_count // block_manager.block_size
    )
    if len(sequence.block_table) < prompt_block_count:
        raise ValueError(
            "sequence block table does not cover the prompt"
        )
    token_ids = tuple(
        sequence.token_ids[:prompt_token_count]
    )
    if len(token_ids) != prompt_token_count:
        raise ValueError(
            "sequence token storage does not cover the prompt"
        )
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(
            "prompt token ids must contain non-negative integers"
        )
    block_identities = []
    prefix_hash = -1
    seen_block_ids = set()
    for block_index in range(prompt_block_count):
        block_id = sequence.block_table[block_index]
        if (
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or block_id < 0
            or block_id >= len(block_manager.blocks)
            or block_id in seen_block_ids
        ):
            raise ValueError(
                "prompt block identity is invalid"
            )
        seen_block_ids.add(block_id)
        block = block_manager.blocks[block_id]
        block_tokens = list(token_ids[
            block_index * block_manager.block_size:
            (block_index + 1) * block_manager.block_size
        ])
        prefix_hash = block_manager.compute_hash(
            block_tokens,
            prefix_hash,
        )
        if block.hash < 0:
            raise ValueError(
                "prompt block hash is not published"
            )
        if block.hash != prefix_hash:
            raise ValueError(
                "prompt block hash chain is stale"
            )
        if block.token_ids != block_tokens:
            raise ValueError(
                "prompt block token identity is stale"
            )
        if (
            block_id not in block_manager.used_block_ids
            or block.ref_count <= 0
            or block.generation <= 0
        ):
            raise ValueError(
                "prompt block generation ownership is stale"
            )
        block_identities.append((
            block_id,
            block.generation,
            block.hash,
        ))
    lease = HybridStateLease(
        slot_id=sequence.hybrid_state_slot_id,
        generation=sequence.hybrid_state_generation,
        request_id=sequence.seq_id,
    )
    try:
        state_allocator.validate(lease)
    except (ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"hybrid state lease is not live: {error}"
        ) from error
    key = Qwen35HybridPrefixKey(
        token_hash=prefix_hash,
        token_count=prompt_token_count,
        terminal_block_hash=prefix_hash,
        block_size=block_manager.block_size,
        model_fingerprint=model_fingerprint,
        layout_fingerprint=layout_fingerprint,
        tensor_parallel_size=_positive_integer(
            tensor_parallel_size,
            "tensor_parallel_size",
        ),
        dtype=dtype,
    )
    Qwen35HybridPrefixSnapshotCache._validate_key(key)
    return Qwen35HybridPrefixPublicationCandidate(
        request_id=sequence.seq_id,
        key=key,
        token_ids=token_ids,
        block_identities=tuple(block_identities),
        lease=lease,
    )
