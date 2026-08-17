from __future__ import annotations

from dataclasses import dataclass


BlockIdentity = tuple[int, int]


def _non_negative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_block_identities(
    identities: object,
    name: str,
) -> tuple[BlockIdentity, ...]:
    if not isinstance(identities, tuple):
        raise ValueError(f"{name} must be a tuple")
    normalized = []
    seen = set()
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise ValueError(
                f"{name} entries must be block/generation tuples"
            )
        block_id = _non_negative_integer(
            identity[0],
            f"{name} block id",
        )
        generation = _non_negative_integer(
            identity[1],
            f"{name} generation",
        )
        if block_id in seen:
            raise ValueError(f"{name} block ids must be unique")
        seen.add(block_id)
        normalized.append((block_id, generation))
    return tuple(normalized)


@dataclass(frozen=True)
class KVBlockIdentityRow:
    sequence_id: int
    block_identities: tuple[BlockIdentity, ...]

    def __post_init__(self) -> None:
        _non_negative_integer(self.sequence_id, "sequence_id")
        object.__setattr__(
            self,
            "block_identities",
            _validate_block_identities(
                self.block_identities,
                "block_identities",
            ),
        )


def build_kv_block_identity_rows(
    block_manager,
    seqs: tuple[object, ...],
) -> tuple[KVBlockIdentityRow, ...]:
    if not isinstance(seqs, tuple):
        raise ValueError("seqs must be a tuple")
    rows = []
    sequence_ids = []
    for seq in seqs:
        sequence_id = _non_negative_integer(
            getattr(seq, "seq_id", None),
            "sequence id",
        )
        block_table = getattr(seq, "block_table", None)
        if not isinstance(block_table, list):
            raise ValueError(
                "sequence block_table must be a list"
            )
        sequence_ids.append(sequence_id)
        rows.append(
            KVBlockIdentityRow(
                sequence_id=sequence_id,
                block_identities=block_manager.block_identities(
                    tuple(block_table)
                ),
            )
        )
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence ids must be unique")
    return tuple(rows)


@dataclass(frozen=True)
class SpeculativeResidencyPrepareRow:
    sequence_id: int
    original_block_identities: tuple[BlockIdentity, ...]
    reserved_block_identities: tuple[BlockIdentity, ...]
    proxy_block_table: tuple[int, ...]
    logical_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        _non_negative_integer(self.sequence_id, "sequence_id")
        original = _validate_block_identities(
            self.original_block_identities,
            "original_block_identities",
        )
        reserved = _validate_block_identities(
            self.reserved_block_identities,
            "reserved_block_identities",
        )
        if set(block for block, _ in original).intersection(
            block for block, _ in reserved
        ):
            raise ValueError(
                "original and reserved block identities overlap"
            )
        if not isinstance(self.proxy_block_table, tuple):
            raise ValueError("proxy_block_table must be a tuple")
        if not isinstance(self.logical_slots, tuple):
            raise ValueError("logical_slots must be a tuple")
        proxy = tuple(
            _non_negative_integer(block, "proxy block id")
            for block in self.proxy_block_table
        )
        slots = tuple(
            _non_negative_integer(slot, "logical slot")
            for slot in self.logical_slots
        )
        if not slots:
            raise ValueError("logical_slots must be non-empty")
        if slots != tuple(
            range(slots[0], slots[0] + len(slots))
        ):
            raise ValueError("logical_slots must be consecutive")
        expected_proxy = tuple(
            block for block, _ in original + reserved
        )
        if proxy != expected_proxy:
            raise ValueError(
                "proxy block table must exactly match "
                "original and reserved identities"
            )
        object.__setattr__(
            self,
            "original_block_identities",
            original,
        )
        object.__setattr__(
            self,
            "reserved_block_identities",
            reserved,
        )
        object.__setattr__(self, "proxy_block_table", proxy)
        object.__setattr__(self, "logical_slots", slots)


@dataclass(frozen=True)
class SpeculativeResidencyPrecommitRow:
    sequence_id: int
    committed_block_identities: tuple[BlockIdentity, ...]
    rejected_block_identities: tuple[BlockIdentity, ...]
    accepted_materialized_end: int

    def __post_init__(self) -> None:
        _non_negative_integer(self.sequence_id, "sequence_id")
        committed = _validate_block_identities(
            self.committed_block_identities,
            "committed_block_identities",
        )
        rejected = _validate_block_identities(
            self.rejected_block_identities,
            "rejected_block_identities",
        )
        if set(block for block, _ in committed).intersection(
            block for block, _ in rejected
        ):
            raise ValueError(
                "committed and rejected identities overlap"
            )
        _non_negative_integer(
            self.accepted_materialized_end,
            "accepted_materialized_end",
        )
        object.__setattr__(
            self,
            "committed_block_identities",
            committed,
        )
        object.__setattr__(
            self,
            "rejected_block_identities",
            rejected,
        )


@dataclass(frozen=True)
class SpeculativeResidencyResult:
    ticket_id: int
    participant_id: int
    operation: str
    status: str
    sequence_ids: tuple[int, ...]
    committed_block_identities: tuple[BlockIdentity, ...] = ()
    rejected_block_identities: tuple[BlockIdentity, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class _PreparedResidencyRow:
    payload: SpeculativeResidencyPrepareRow
    materialized_block_identities: tuple[BlockIdentity, ...]


@dataclass
class _SpeculativeResidencyTicket:
    ticket_id: int
    rows: tuple[_PreparedResidencyRow, ...]
    state: str = "prepared"
    materialized_sequence_ids: tuple[int, ...] = ()
    precommit_rows: tuple[SpeculativeResidencyPrecommitRow, ...] = ()


class SpeculativeResidencyParticipant:
    def __init__(
        self,
        *,
        participant_id: int,
        manager,
        block_size: int,
    ):
        self.participant_id = _non_negative_integer(
            participant_id,
            "participant_id",
        )
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise ValueError("block_size must be positive")
        self.manager = manager
        self.block_size = block_size
        self._tickets: dict[int, _SpeculativeResidencyTicket] = {}

    @staticmethod
    def _sequence_ids(rows) -> tuple[int, ...]:
        sequence_ids = tuple(row.sequence_id for row in rows)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("sequence ids must be unique")
        return sequence_ids

    def _result(
        self,
        ticket: _SpeculativeResidencyTicket,
        operation: str,
        status: str,
        *,
        committed: tuple[BlockIdentity, ...] = (),
        rejected: tuple[BlockIdentity, ...] = (),
    ) -> SpeculativeResidencyResult:
        return SpeculativeResidencyResult(
            ticket_id=ticket.ticket_id,
            participant_id=self.participant_id,
            operation=operation,
            status=status,
            sequence_ids=self._sequence_ids(
                tuple(row.payload for row in ticket.rows)
            ),
            committed_block_identities=committed,
            rejected_block_identities=rejected,
        )

    def _ticket(
        self,
        ticket_id: int,
    ) -> _SpeculativeResidencyTicket:
        normalized = _non_negative_integer(
            ticket_id,
            "ticket_id",
        )
        ticket = self._tickets.get(normalized)
        if ticket is None:
            raise RuntimeError(
                f"unknown speculative residency ticket: {normalized}"
            )
        return ticket

    def prepare_batch(
        self,
        ticket_id: int,
        rows: tuple[SpeculativeResidencyPrepareRow, ...],
        *,
        stage_all_original_blocks: bool = True,
    ) -> SpeculativeResidencyResult:
        ticket_id = _non_negative_integer(ticket_id, "ticket_id")
        if not isinstance(stage_all_original_blocks, bool):
            raise ValueError(
                "stage_all_original_blocks must be a boolean"
            )
        if ticket_id in self._tickets:
            raise RuntimeError(
                "speculative residency ticket already exists"
            )
        if not isinstance(rows, tuple) or not rows:
            raise ValueError("prepare rows must be a non-empty tuple")
        if any(
            not isinstance(row, SpeculativeResidencyPrepareRow)
            for row in rows
        ):
            raise ValueError(
                "prepare rows must contain prepare payloads"
            )
        self._sequence_ids(rows)
        reserved_identities = tuple(
            identity
            for row in rows
            for identity in row.reserved_block_identities
        )
        if len({
            block_id for block_id, _ in reserved_identities
        }) != len(reserved_identities):
            raise ValueError(
                "reserved block identities must be batch-disjoint"
            )
        prepared_rows = []
        try:
            for row in rows:
                identity_by_block = dict(
                    row.original_block_identities
                    + row.reserved_block_identities
                )
                materialized_ids = []
                seen = set()
                for slot in row.logical_slots:
                    block_id = row.proxy_block_table[
                        slot // self.block_size
                    ]
                    if block_id not in seen:
                        materialized_ids.append(
                            (block_id, identity_by_block[block_id])
                        )
                        seen.add(block_id)
                for identity in row.original_block_identities:
                    self.manager.bind_logical_block_identity(
                        *identity
                    )
                for identity in row.reserved_block_identities:
                    self.manager.bind_logical_block_identity(
                        *identity
                    )
                original_blocks = [
                    block_id
                    for block_id, _ in row.original_block_identities
                ]
                original_block_set = set(original_blocks)
                materialized_original = [
                    block_id
                    for block_id, _ in materialized_ids
                    if block_id in original_block_set
                ]
                reserved_materialized = [
                    block_id
                    for block_id, _ in materialized_ids
                    if block_id in {
                        item[0]
                        for item in row.reserved_block_identities
                    }
                ]
                original_read_blocks = (
                    original_blocks
                    if stage_all_original_blocks
                    else materialized_original
                )
                protected = set(
                    original_read_blocks
                    + reserved_materialized
                )
                if original_read_blocks:
                    self.manager.ensure_resident(
                        original_read_blocks,
                        require_valid=True,
                        protected_logical_blocks=protected,
                        wait=True,
                    )
                if reserved_materialized:
                    self.manager.ensure_resident(
                        reserved_materialized,
                        require_valid=False,
                        protected_logical_blocks=protected,
                        wait=True,
                    )
                prepared_rows.append(
                    _PreparedResidencyRow(
                        payload=row,
                        materialized_block_identities=tuple(
                            materialized_ids
                        ),
                    )
                )
        except BaseException:
            resident_reserved = tuple(
                identity
                for identity in reserved_identities
                if identity[0] in self.manager.logical_to_slot
            )
            if resident_reserved:
                self.manager.discard_resident_blocks(
                    resident_reserved,
                    allow_dirty=False,
                )
            raise
        ticket = _SpeculativeResidencyTicket(
            ticket_id=ticket_id,
            rows=tuple(prepared_rows),
        )
        self._tickets[ticket_id] = ticket
        self.manager.stats[
            "speculative_residency_prepares"
        ] += 1
        return self._result(ticket, "prepare", "prepared")

    def is_prepared_for(
        self,
        ticket_id: int,
        sequence_ids: tuple[int, ...],
    ) -> bool:
        ticket = self._tickets.get(ticket_id)
        if ticket is None or ticket.state != "prepared":
            return False
        if (
            not isinstance(sequence_ids, tuple)
            or not sequence_ids
            or len(set(sequence_ids)) != len(sequence_ids)
        ):
            return False
        expected = self._sequence_ids(
            tuple(row.payload for row in ticket.rows)
        )
        requested = set(sequence_ids)
        return (
            requested.issubset(expected)
            and requested.isdisjoint(
                ticket.materialized_sequence_ids
            )
        )

    def mark_materialized(
        self,
        ticket_id: int,
        sequence_ids: tuple[int, ...],
    ) -> None:
        ticket = self._ticket(ticket_id)
        if ticket.state != "prepared":
            raise RuntimeError(
                "speculative residency ticket is not prepared"
            )
        expected = self._sequence_ids(
            tuple(row.payload for row in ticket.rows)
        )
        if not self.is_prepared_for(ticket_id, sequence_ids):
            raise ValueError(
                "materialized sequence ids do not match ticket"
            )
        materialized = (
            set(ticket.materialized_sequence_ids)
            | set(sequence_ids)
        )
        ticket.materialized_sequence_ids = tuple(
            sequence_id
            for sequence_id in expected
            if sequence_id in materialized
        )

    def precommit_batch(
        self,
        ticket_id: int,
        rows: tuple[SpeculativeResidencyPrecommitRow, ...],
    ) -> SpeculativeResidencyResult:
        ticket = self._ticket(ticket_id)
        if ticket.state != "prepared":
            raise RuntimeError(
                "speculative residency ticket is not precommittable"
            )
        if not isinstance(rows, tuple):
            raise ValueError("precommit rows must be a tuple")
        expected_ids = self._sequence_ids(
            tuple(row.payload for row in ticket.rows)
        )
        if (
            ticket.materialized_sequence_ids != expected_ids
            or self._sequence_ids(rows) != expected_ids
        ):
            raise RuntimeError(
                "speculative residency materialization is incomplete"
            )
        by_sequence = {
            row.payload.sequence_id: row
            for row in ticket.rows
        }
        committed = []
        rejected = []
        for row in rows:
            prepared = by_sequence[row.sequence_id]
            reserved = prepared.payload.reserved_block_identities
            partition = (
                row.committed_block_identities
                + row.rejected_block_identities
            )
            if (
                set(partition) != set(reserved)
                or len(partition) != len(reserved)
            ):
                raise RuntimeError(
                    "reserved residency partition is invalid"
                )
            materialized = set(
                prepared.materialized_block_identities
            )
            if not set(
                row.committed_block_identities
            ).issubset(materialized):
                raise RuntimeError(
                    "committed residency block was not materialized"
                )
            last_slot_end = (
                prepared.payload.logical_slots[-1] + 1
            )
            if row.accepted_materialized_end > last_slot_end:
                raise RuntimeError(
                    "accepted materialized end exceeds verifier writes"
                )
            committed.extend(row.committed_block_identities)
            rejected.extend(row.rejected_block_identities)
        ticket.precommit_rows = rows
        ticket.state = "precommitted"
        self.manager.stats[
            "speculative_residency_precommits"
        ] += 1
        return self._result(
            ticket,
            "precommit",
            "precommitted",
            committed=tuple(committed),
            rejected=tuple(rejected),
        )

    def rollback_batch(
        self,
        ticket_id: int,
    ) -> SpeculativeResidencyResult:
        ticket = self._ticket(ticket_id)
        if ticket.state not in ("prepared", "precommitted"):
            raise RuntimeError(
                "speculative residency ticket is not rollbackable"
            )
        reserved = tuple(
            identity
            for row in ticket.rows
            for identity in row.payload.reserved_block_identities
            if identity[0] in self.manager.logical_to_slot
        )
        if reserved:
            self.manager.discard_resident_blocks(
                reserved,
                allow_dirty=False,
            )
        ticket.state = "rolled_back"
        self.manager.stats[
            "speculative_residency_rollbacks"
        ] += 1
        self.manager.stats[
            "speculative_residency_rejected_blocks"
        ] += len(reserved)
        return self._result(
            ticket,
            "rollback",
            "rolled_back",
            rejected=reserved,
        )

    def seal_batch(
        self,
        ticket_id: int,
    ) -> SpeculativeResidencyResult:
        ticket = self._ticket(ticket_id)
        if ticket.state != "precommitted":
            raise RuntimeError(
                "speculative residency ticket is not sealable"
            )
        prepared_by_id = {
            row.payload.sequence_id: row
            for row in ticket.rows
        }
        committed = []
        rejected = []
        dirty_blocks = []
        seen_dirty = set()
        for row in ticket.precommit_rows:
            prepared = prepared_by_id[row.sequence_id]
            for block_id, _ in (
                prepared.materialized_block_identities
            ):
                if (
                    any(
                        slot < row.accepted_materialized_end
                        and prepared.payload.proxy_block_table[
                            slot // self.block_size
                        ] == block_id
                        for slot in prepared.payload.logical_slots
                    )
                    and block_id not in seen_dirty
                ):
                    dirty_blocks.append(block_id)
                    seen_dirty.add(block_id)
            committed.extend(row.committed_block_identities)
            rejected.extend(row.rejected_block_identities)
        if dirty_blocks:
            self.manager.mark_dirty(dirty_blocks)
        if rejected:
            self.manager.discard_resident_blocks(
                tuple(rejected),
                allow_dirty=False,
            )
        if dirty_blocks and not self.manager.writeback_on_evict:
            self.manager.writeback_dirty(dirty_blocks)
        ticket.state = "sealed"
        self.manager.stats[
            "speculative_residency_seals"
        ] += 1
        self.manager.stats[
            "speculative_residency_committed_blocks"
        ] += len(committed)
        self.manager.stats[
            "speculative_residency_rejected_blocks"
        ] += len(rejected)
        return self._result(
            ticket,
            "seal",
            "sealed",
            committed=tuple(committed),
            rejected=tuple(rejected),
        )
