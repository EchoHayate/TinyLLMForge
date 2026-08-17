from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Qwen3DraftGraphScratchRow:
    indexed_row: object
    source_committed_physical_slot_ids: tuple[int, ...]
    source_read_lease: object
    transaction: object


@dataclass
class Qwen3DraftGraphScratchLease:
    identity: object
    rows: tuple[Qwen3DraftGraphScratchRow, ...]
    scratch_cache: object
    rolled_back: bool = False


class Qwen3DraftGraphScratchOwner:

    def __init__(self, *, live_cache, scratch_cache):
        if scratch_cache is live_cache:
            raise ValueError(
                "scratch_cache must be distinct from live_cache"
            )
        for name, owner, methods in (
            (
                "live_cache",
                live_cache,
                ("committed_entry_identities",),
            ),
            (
                "live_cache.entry_allocator",
                getattr(live_cache, "entry_allocator", None),
                (
                    "ensure_readable",
                    "record_read_complete",
                ),
            ),
            (
                "scratch_cache",
                scratch_cache,
                ("begin", "abort", "release_sequence"),
            ),
        ):
            for method in methods:
                if not callable(getattr(owner, method, None)):
                    raise ValueError(
                        f"{name} must expose callable {method}"
                    )
        self.live_cache = live_cache
        self.scratch_cache = scratch_cache
        self._next_synthetic_sequence_id = 1_000_000_000

    def _new_synthetic_sequence_id(
        self,
        source_sequence_ids: set[int],
    ) -> int:
        while (
            self._next_synthetic_sequence_id
            in source_sequence_ids
        ):
            self._next_synthetic_sequence_id += 1
        sequence_id = self._next_synthetic_sequence_id
        self._next_synthetic_sequence_id += 1
        return sequence_id

    def _abort_rows(
        self,
        rows: tuple[Qwen3DraftGraphScratchRow, ...],
    ) -> None:
        first_error = None
        allocator = self.live_cache.entry_allocator
        for row in reversed(rows):
            transaction = row.transaction
            for cleanup in (
                lambda: (
                    self.scratch_cache.abort(
                        transaction.transaction_id
                    )
                    if transaction.state
                    in ("reserved", "materialized")
                    else None
                ),
                lambda: self.scratch_cache.release_sequence(
                    transaction.sequence_id,
                    sequence_epoch=transaction.sequence_epoch,
                ),
                lambda: allocator.record_read_complete(
                    row.source_read_lease
                ),
            ):
                try:
                    cleanup()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error

    def acquire(self, identity, rows):
        if not isinstance(rows, tuple) or not rows:
            raise ValueError("rows must be a non-empty tuple")
        if len(rows) != identity.exact_batch_size:
            raise ValueError(
                "scratch row count must match exact batch size"
            )
        if identity.blockwise_offload:
            raise ValueError(
                "scratch capture does not support offload"
            )
        staged_entry_count = identity.exact_q - 1
        source_sequence_ids = {
            indexed_row[1].sequence_id
            for indexed_row in rows
        }
        scratch_rows = []
        allocator = self.live_cache.entry_allocator
        try:
            for indexed_row in rows:
                if (
                    not isinstance(indexed_row, tuple)
                    or len(indexed_row) != 3
                ):
                    raise ValueError(
                        "scratch indexed row is invalid"
                    )
                _, input_row, context_token_count = indexed_row
                committed = tuple(
                    self.live_cache.committed_entry_identities(
                        input_row.sequence_id
                    )
                )
                if len(committed) != context_token_count:
                    raise ValueError(
                        "scratch committed length must match "
                        "context token count"
                    )
                read_lease = allocator.ensure_readable(
                    committed
                )
                try:
                    transaction = self.scratch_cache.begin(
                        self._new_synthetic_sequence_id(
                            source_sequence_ids
                        ),
                        0,
                        staged_entry_count,
                    )
                except BaseException:
                    allocator.record_read_complete(read_lease)
                    raise
                scratch_rows.append(
                    Qwen3DraftGraphScratchRow(
                        indexed_row=indexed_row,
                        source_committed_physical_slot_ids=tuple(
                            read_lease.physical_slot_ids
                        ),
                        source_read_lease=read_lease,
                        transaction=transaction,
                    )
                )
        except BaseException:
            self._abort_rows(tuple(scratch_rows))
            raise
        return Qwen3DraftGraphScratchLease(
            identity=identity,
            rows=tuple(scratch_rows),
            scratch_cache=self.scratch_cache,
        )

    def rollback(self, lease):
        if not isinstance(
            lease,
            Qwen3DraftGraphScratchLease,
        ):
            raise ValueError(
                "lease must be a Qwen3DraftGraphScratchLease"
            )
        if lease.rolled_back:
            raise RuntimeError(
                "scratch lease is already rolled back"
            )
        self._abort_rows(lease.rows)
        lease.rolled_back = True
