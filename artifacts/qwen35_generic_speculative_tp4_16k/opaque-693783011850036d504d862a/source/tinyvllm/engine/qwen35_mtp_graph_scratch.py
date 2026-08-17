from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Qwen35MTPGraphScratchRow:
    input_row: object
    bootstrap: object
    source_committed_slot_ids: tuple[int, ...]
    transaction: object


@dataclass
class Qwen35MTPGraphScratchLease:
    identity: object
    rows: tuple[Qwen35MTPGraphScratchRow, ...]
    rolled_back: bool = False


class Qwen35MTPGraphScratchOwner:

    def __init__(self, *, live_cache, scratch_cache):
        if scratch_cache is live_cache:
            raise ValueError(
                "scratch_cache must be distinct from live_cache"
            )
        for name, owner, methods in (
            (
                "live_cache",
                live_cache,
                ("committed_slot_ids",),
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
        rows: tuple[Qwen35MTPGraphScratchRow, ...],
    ) -> None:
        first_error = None
        for row in reversed(rows):
            transaction = row.transaction
            try:
                if transaction.state in ("reserved", "materialized"):
                    self.scratch_cache.abort(
                        transaction.transaction_id
                    )
                self.scratch_cache.release_sequence(
                    transaction.sequence_id,
                    sequence_epoch=transaction.sequence_epoch,
                )
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
        staged_entry_count = identity.exact_q - 1
        source_sequence_ids = {
            input_row.sequence_id for input_row, _ in rows
        }
        scratch_rows = []
        try:
            for input_row, bootstrap in rows:
                synthetic_sequence_id = (
                    self._new_synthetic_sequence_id(
                        source_sequence_ids
                    )
                )
                transaction = self.scratch_cache.begin(
                    synthetic_sequence_id,
                    0,
                    staged_entry_count,
                )
                scratch_rows.append(
                    Qwen35MTPGraphScratchRow(
                        input_row=input_row,
                        bootstrap=bootstrap,
                        source_committed_slot_ids=tuple(
                            self.live_cache.committed_slot_ids(
                                input_row.sequence_id
                            )
                        ),
                        transaction=transaction,
                    )
                )
        except BaseException:
            self._abort_rows(tuple(scratch_rows))
            raise
        return Qwen35MTPGraphScratchLease(
            identity=identity,
            rows=tuple(scratch_rows),
        )

    def rollback(self, lease):
        if not isinstance(lease, Qwen35MTPGraphScratchLease):
            raise ValueError(
                "lease must be a Qwen35MTPGraphScratchLease"
            )
        if lease.rolled_back:
            raise RuntimeError("scratch lease is already rolled back")
        self._abort_rows(lease.rows)
        lease.rolled_back = True

