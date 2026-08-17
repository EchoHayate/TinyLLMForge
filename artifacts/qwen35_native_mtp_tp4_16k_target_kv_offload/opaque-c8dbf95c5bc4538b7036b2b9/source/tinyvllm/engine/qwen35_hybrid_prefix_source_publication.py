from __future__ import annotations

from itertools import count

import torch

from tinyvllm.engine.block_manager import BlockManager
from tinyvllm.engine.hybrid_state import HybridStateSlotAllocator
from tinyvllm.engine.qwen35_hybrid_prefix_publication_candidate import (
    capture_qwen35_hybrid_prefix_publication_candidate,
)


class Qwen35HybridPrefixSourcePublisher:

    def __init__(
        self,
        engine,
        *,
        model_fingerprint,
        layout_fingerprint,
        dtype,
    ):
        if engine is None:
            raise ValueError("engine must be provided")
        if (
            not isinstance(model_fingerprint, str)
            or not model_fingerprint
        ):
            raise ValueError(
                "model_fingerprint must be a non-empty string"
            )
        if (
            not isinstance(layout_fingerprint, str)
            or not layout_fingerprint
        ):
            raise ValueError(
                "layout_fingerprint must be a non-empty string"
            )
        if dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }:
            raise ValueError("dtype is unsupported")
        world_size = getattr(
            getattr(engine, "model_runner", None),
            "world_size",
            None,
        )
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
        ):
            raise ValueError(
                "engine ModelRunner world_size must be positive"
            )
        self.engine = engine
        self.model_fingerprint = model_fingerprint
        self.layout_fingerprint = layout_fingerprint
        self.dtype = dtype
        self.world_size = world_size
        self._ticket_ids = count()
        self._active = False

    def _owners(self):
        scheduler = getattr(self.engine, "scheduler", None)
        block_manager = getattr(
            scheduler,
            "block_manager",
            None,
        )
        state_allocator = getattr(
            scheduler,
            "hybrid_state_allocator",
            None,
        )
        if not isinstance(block_manager, BlockManager):
            raise RuntimeError(
                "Engine Scheduler BlockManager is not installed"
            )
        if not isinstance(
            state_allocator,
            HybridStateSlotAllocator,
        ):
            raise RuntimeError(
                "Engine Scheduler hybrid state allocator "
                "is not installed"
            )
        coordinator = getattr(
            self.engine,
            "qwen35_hybrid_prefix_engine_publication_coordinator",
            None,
        )
        if coordinator is None:
            raise RuntimeError(
                "Engine hybrid prefix publication coordinator "
                "is not installed"
            )
        return block_manager, state_allocator

    def publish(self, sequence):
        if self._active:
            raise RuntimeError(
                "source-bound publication is already active"
            )
        self._active = True
        try:
            block_manager, state_allocator = self._owners()
            ticket_id = next(self._ticket_ids)
            candidate = (
                capture_qwen35_hybrid_prefix_publication_candidate(
                    sequence,
                    block_manager,
                    state_allocator,
                    model_fingerprint=self.model_fingerprint,
                    layout_fingerprint=self.layout_fingerprint,
                    tensor_parallel_size=self.world_size,
                    dtype=self.dtype,
                )
            )
            candidate.validate_source(
                sequence,
                block_manager,
                state_allocator,
            )
            payloads = candidate.publication_payloads(
                ticket_id=ticket_id,
                world_size=self.world_size,
            )
            return self.engine.publish_qwen35_hybrid_prefix(
                payloads
            )
        finally:
            self._active = False
