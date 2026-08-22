"""Replay-aware decode metadata planning and staging primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecodeMetadataPlan:
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    context_lens: tuple[int, ...]
    block_table_rows: tuple[tuple[int, ...], ...]
    active_batch_size: int
    readable_page_table_width: int


@dataclass
class DecodeMetadataLandingStats:
    eligible_steps: int = 0
    optimized_steps: int = 0
    allocation_count: int = 0
    growth_count: int = 0
    staged_h2d_bytes: int = 0
    avoided_temporary_cuda_tensors: int = 0
    avoided_blanket_zero_bytes: int = 0
    peak_pinned_capacity_bytes: int = 0
    fallback_counts: dict[str, int] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DecodeMetadataLandingResult:
    optimized: bool
    fallback_reason: str | None
    input_ids: object | None = None
    positions: object | None = None
    slot_mapping: object | None = None
    context_lens: object | None = None
    block_tables: object | None = None


def build_decode_metadata_plan(
    seqs,
    block_size: int,
) -> DecodeMetadataPlan:
    if not seqs:
        raise ValueError(
            "decode metadata requires at least one sequence"
        )
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if any(not seq.block_table for seq in seqs):
        raise ValueError(
            "decode metadata requires allocated block tables"
        )

    page_table_width = max(
        len(seq.block_table) for seq in seqs
    )
    block_table_rows = tuple(
        tuple(int(block_id) for block_id in seq.block_table)
        + (-1,) * (
            page_table_width - len(seq.block_table)
        )
        for seq in seqs
    )
    return DecodeMetadataPlan(
        input_ids=tuple(
            int(seq.last_token) for seq in seqs
        ),
        positions=tuple(len(seq) - 1 for seq in seqs),
        slot_mapping=tuple(
            int(seq.block_table[-1]) * block_size
            + int(seq.last_block_num_tokens)
            - 1
            for seq in seqs
        ),
        context_lens=tuple(len(seq) for seq in seqs),
        block_table_rows=block_table_rows,
        active_batch_size=len(seqs),
        readable_page_table_width=page_table_width,
    )


class ReplayAwareDecodeMetadataArena:
    _FIELD_DTYPES = {
        "input_ids": "int64",
        "positions": "int64",
        "slot_mapping": "int32",
        "context_lens": "int32",
        "block_tables": "int32",
    }

    def __init__(self, torch_module):
        self._torch = torch_module
        self._host_buffers: dict[str, object] = {}
        self._stats = DecodeMetadataLandingStats()

    def _fallback(
        self,
        reason: str,
    ) -> DecodeMetadataLandingResult:
        self._stats.fallback_counts[reason] = (
            self._stats.fallback_counts.get(reason, 0) + 1
        )
        return DecodeMetadataLandingResult(
            optimized=False,
            fallback_reason=reason,
        )

    def record_fallback(self, reason: str) -> None:
        self._fallback(reason)

    @staticmethod
    def _shape(tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in tensor.size())

    def _validate_destination(
        self,
        plan: DecodeMetadataPlan,
        graph_vars,
        graph_batch_size: int,
    ) -> str | None:
        if plan.active_batch_size != 1:
            return "active_batch_size_unsupported"
        if graph_batch_size != plan.active_batch_size:
            return "graph_batch_size_mismatch"
        required = {
            "input_ids",
            "positions",
            "slot_mapping",
            "context_lens",
            "block_tables",
            "outputs",
        }
        if not isinstance(graph_vars, dict) or (
            required - set(graph_vars)
        ):
            return "graph_buffers_missing"
        try:
            vector_shapes = {
                name: self._shape(graph_vars[name])
                for name in (
                    "input_ids",
                    "positions",
                    "slot_mapping",
                    "context_lens",
                )
            }
            block_shape = self._shape(
                graph_vars["block_tables"]
            )
        except (AttributeError, TypeError, ValueError):
            return "graph_capacity_mismatch"
        if any(
            len(shape) != 1
            or shape[0] < plan.active_batch_size
            for shape in vector_shapes.values()
        ):
            return "graph_capacity_mismatch"
        if (
            len(block_shape) != 2
            or block_shape[0] < plan.active_batch_size
            or block_shape[1]
            < plan.readable_page_table_width
        ):
            return "graph_capacity_mismatch"
        return None

    def _current_pinned_capacity_bytes(self) -> int:
        return sum(
            int(buffer.numel()) * int(buffer.element_size())
            for buffer in self._host_buffers.values()
        )

    def _stage_flat(
        self,
        name: str,
        values: tuple[int, ...],
    ):
        dtype = getattr(
            self._torch,
            self._FIELD_DTYPES[name],
        )
        required = len(values)
        buffer = self._host_buffers.get(name)
        if buffer is None or int(buffer.numel()) < required:
            previous_capacity = (
                0 if buffer is None else int(buffer.numel())
            )
            capacity = max(
                required,
                max(64, previous_capacity * 2),
            )
            buffer = self._torch.empty(
                capacity,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._host_buffers[name] = buffer
            self._stats.allocation_count += 1
            self._stats.growth_count += 1
            current = self._current_pinned_capacity_bytes()
            self._stats.peak_pinned_capacity_bytes = max(
                self._stats.peak_pinned_capacity_bytes,
                current,
            )
        for index, value in enumerate(values):
            buffer[index] = value
        self._stats.staged_h2d_bytes += (
            required * int(buffer.element_size())
        )
        return buffer[:required]

    def land(
        self,
        plan: DecodeMetadataPlan,
        graph_vars,
        *,
        graph_batch_size: int,
    ) -> DecodeMetadataLandingResult:
        self._stats.eligible_steps += 1
        fallback_reason = self._validate_destination(
            plan,
            graph_vars,
            graph_batch_size,
        )
        if fallback_reason is not None:
            return self._fallback(fallback_reason)

        input_ids_host = self._stage_flat(
            "input_ids",
            plan.input_ids,
        )
        positions_host = self._stage_flat(
            "positions",
            plan.positions,
        )
        slot_mapping_host = self._stage_flat(
            "slot_mapping",
            plan.slot_mapping,
        )
        context_lens_host = self._stage_flat(
            "context_lens",
            plan.context_lens,
        )
        flattened_blocks = tuple(
            block_id
            for row in plan.block_table_rows
            for block_id in row
        )
        block_tables_host = self._stage_flat(
            "block_tables",
            flattened_blocks,
        ).view(
            plan.active_batch_size,
            plan.readable_page_table_width,
        )

        active = slice(0, plan.active_batch_size)
        readable = slice(
            0,
            plan.readable_page_table_width,
        )
        destinations = {
            "input_ids": graph_vars["input_ids"][active],
            "positions": graph_vars["positions"][active],
            "slot_mapping": graph_vars[
                "slot_mapping"
            ][active],
            "context_lens": graph_vars[
                "context_lens"
            ][active],
            "block_tables": graph_vars[
                "block_tables"
            ][active, readable],
        }
        sources = {
            "input_ids": input_ids_host,
            "positions": positions_host,
            "slot_mapping": slot_mapping_host,
            "context_lens": context_lens_host,
            "block_tables": block_tables_host,
        }
        for name, destination in destinations.items():
            destination.copy_(
                sources[name],
                non_blocking=True,
            )

        self._stats.optimized_steps += 1
        self._stats.avoided_temporary_cuda_tensors += 5
        self._stats.avoided_blanket_zero_bytes += sum(
            int(graph_vars[name].numel())
            * int(graph_vars[name].element_size())
            for name in destinations
        )
        return DecodeMetadataLandingResult(
            optimized=True,
            fallback_reason=None,
            input_ids=destinations["input_ids"],
            positions=destinations["positions"],
            slot_mapping=destinations["slot_mapping"],
            context_lens=destinations["context_lens"],
            block_tables=destinations["block_tables"],
        )

    def summary(self) -> dict:
        return {
            "eligible_steps": self._stats.eligible_steps,
            "optimized_steps": self._stats.optimized_steps,
            "fallback_counts": dict(
                sorted(self._stats.fallback_counts.items())
            ),
            "allocation_count": self._stats.allocation_count,
            "growth_count": self._stats.growth_count,
            "staged_h2d_bytes": self._stats.staged_h2d_bytes,
            "avoided_temporary_cuda_tensors": (
                self._stats.avoided_temporary_cuda_tensors
            ),
            "avoided_blanket_zero_bytes": (
                self._stats.avoided_blanket_zero_bytes
            ),
            "current_pinned_capacity_bytes": (
                self._current_pinned_capacity_bytes()
            ),
            "peak_pinned_capacity_bytes": (
                self._stats.peak_pinned_capacity_bytes
            ),
        }
