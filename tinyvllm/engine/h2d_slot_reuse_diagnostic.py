from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


H2D_SLOT_REUSE_SCHEMA = (
    "qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1"
)
VALID_H2D_SLOT_REUSE_MODES = ("off", "observe", "control")
VALID_ATTENTION_STAGES = ("decode", "spec_verify", "prefill")
VALID_TIMING_STATUSES = (
    "UNSAFE_OVERLAP_OBSERVED",
    "ORDERING_AMBIGUOUS",
    "READ_COMPLETED_BEFORE_H2D",
    "NO_PRIOR_OCCUPANCY",
    "NO_PRIOR_READ",
)


@dataclass(frozen=True)
class SlotOccupancy:
    physical_slot: int
    occupancy_generation: int
    logical_block: int
    bound_generation: int


@dataclass(frozen=True)
class SlotReadAssociation:
    rank: int
    engine_step: int
    attention_stage: str
    layer_index: int
    window_ordinal: int
    current_stream_id: int
    physical_slot: int
    occupancy_generation: int
    logical_block: int
    bound_generation: int
    read_event_ordinal: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class H2DSlotOverwrite:
    rank: int
    engine_step: int
    attention_stage: str
    layer_index: int
    window_ordinal: int
    copy_batch_ordinal: int
    copy_span_ordinal: int
    physical_slot: int
    old_occupancy_generation: int | None
    old_logical_block: int | None
    old_bound_generation: int | None
    new_occupancy_generation: int
    new_logical_block: int
    new_bound_generation: int
    read_event_ordinals: tuple[int, ...]
    h2d_start_event_ordinal: int
    h2d_done_event_ordinal: int
    control_wait_event_ordinals: tuple[int, ...]
    control_wait_count: int
    timing_status: str
    read_done_after_h2d_start_ms: float | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class H2DSlotReuseDrain:
    schema: str
    rank: int
    mode: str
    stream_inventory: tuple[int, ...]
    read_rows: tuple[SlotReadAssociation, ...]
    overwrite_rows: tuple[H2DSlotOverwrite, ...]

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "rank": self.rank,
            "mode": self.mode,
            "stream_inventory": list(self.stream_inventory),
            "read_rows": [row.as_dict() for row in self.read_rows],
            "overwrite_rows": [
                row.as_dict() for row in self.overwrite_rows
            ],
        }


@dataclass(frozen=True)
class _SlotTransition:
    old_occupancy: SlotOccupancy | None
    new_occupancy: SlotOccupancy
    predecessor_events: tuple[tuple[int, object], ...]


@dataclass(frozen=True)
class _PendingOverwrite:
    context: tuple[int, str, int, int]
    copy_batch_ordinal: int
    copy_span_ordinal: int
    transition: _SlotTransition
    read_events: tuple[tuple[int, object], ...]
    h2d_start_event_ordinal: int
    h2d_start_event: object
    control_wait_event_ordinals: tuple[int, ...]
    h2d_done_event_ordinal: int | None = None
    h2d_done_event: object | None = None


@dataclass(frozen=True)
class _H2DSpanHandle:
    pending_indices: tuple[int, ...]


@dataclass(frozen=True)
class _DiagnosticStateSnapshot:
    mode: str
    slot_generations: tuple[int, ...]
    active_occupancies: tuple[SlotOccupancy | None, ...]
    read_events_by_occupancy: tuple[
        tuple[
            SlotOccupancy,
            tuple[tuple[int, tuple[int, object]], ...],
        ],
        ...,
    ]
    read_rows: tuple[SlotReadAssociation, ...]
    pending_overwrites: tuple[_PendingOverwrite, ...]
    pending_transitions: tuple[
        tuple[SlotOccupancy, _SlotTransition],
        ...,
    ]
    event_ordinals: tuple[tuple[int, object], ...]
    next_event_ordinal: int
    stream_inventory: tuple[int, ...]
    context: tuple[int, str, int, int] | None


def classify_read_h2d_ordering(
    delta_ms: float,
    epsilon_ms: float,
) -> str:
    delta_ms = float(delta_ms)
    epsilon_ms = float(epsilon_ms)
    if epsilon_ms <= 0:
        raise ValueError("timing epsilon must be positive")
    if delta_ms > epsilon_ms:
        return "UNSAFE_OVERLAP_OBSERVED"
    if delta_ms < -epsilon_ms:
        return "READ_COMPLETED_BEFORE_H2D"
    return "ORDERING_AMBIGUOUS"


class H2DSlotReuseDiagnostic:
    def __init__(
        self,
        *,
        rank: int,
        slot_count: int,
        event_factory: Callable[[], object],
        stream_id: Callable[[object], int],
        max_read_rows: int = 1_000_000,
        max_overwrite_rows: int = 1_000_000,
    ):
        if int(rank) < 0:
            raise ValueError("rank must be nonnegative")
        if int(slot_count) <= 0:
            raise ValueError("slot_count must be positive")
        if not callable(event_factory):
            raise ValueError("event_factory must be callable")
        if not callable(stream_id):
            raise ValueError("stream_id must be callable")
        if int(max_read_rows) <= 0:
            raise ValueError("max_read_rows must be positive")
        if int(max_overwrite_rows) <= 0:
            raise ValueError("max_overwrite_rows must be positive")
        self.rank = int(rank)
        self.slot_count = int(slot_count)
        self._event_factory = event_factory
        self._stream_id = stream_id
        self._max_read_rows = int(max_read_rows)
        self._max_overwrite_rows = int(max_overwrite_rows)
        self._mode = "off"
        self._reset_state()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    @property
    def retained_event_count(self) -> int:
        return len(self._event_ordinals)

    @property
    def read_row_count(self) -> int:
        return len(self._read_rows)

    @property
    def overwrite_row_count(self) -> int:
        return len(self._pending_overwrites)

    @property
    def active_occupancies(self) -> tuple[SlotOccupancy, ...]:
        return tuple(
            occupancy
            for occupancy in self._active_occupancies
            if occupancy is not None
        )

    def _reset_state(self) -> None:
        self._slot_generations = [0] * self.slot_count
        self._active_occupancies: list[SlotOccupancy | None] = (
            [None] * self.slot_count
        )
        self._read_events_by_occupancy: dict[
            SlotOccupancy,
            dict[int, tuple[int, object]],
        ] = {}
        self._read_rows: list[SlotReadAssociation] = []
        self._pending_overwrites: list[_PendingOverwrite] = []
        self._pending_transitions: dict[
            SlotOccupancy,
            _SlotTransition,
        ] = {}
        self._event_ordinals: dict[int, object] = {}
        self._next_event_ordinal = 1
        self._stream_inventory: set[int] = set()
        self._context: tuple[int, str, int, int] | None = None

    def configure(self, mode: str) -> dict:
        if mode not in VALID_H2D_SLOT_REUSE_MODES:
            raise ValueError("mode must be one of off, observe, control")
        if mode == "off":
            self._mode = "off"
            self._reset_state()
            return {"rank": self.rank, "mode": "off"}
        if self.enabled and (
            mode != self._mode
            or self._read_rows
            or self._pending_overwrites
            or self._event_ordinals
        ):
            raise RuntimeError("diagnostic has undrained enabled state")
        self._reset_state()
        self._mode = mode
        return {"rank": self.rank, "mode": mode}

    def set_context(
        self,
        *,
        engine_step: int,
        attention_stage: str,
        layer_index: int,
        window_ordinal: int,
    ) -> dict:
        if not self.enabled:
            raise RuntimeError("diagnostic is not enabled")
        self._context = self._validate_context(
            engine_step=engine_step,
            attention_stage=attention_stage,
            layer_index=layer_index,
            window_ordinal=window_ordinal,
        )
        return {
            "rank": self.rank,
            "mode": self._mode,
            "engine_step": self._context[0],
            "attention_stage": self._context[1],
            "layer_index": self._context[2],
            "window_ordinal": self._context[3],
        }

    def assign_slot(
        self,
        *,
        physical_slot: int,
        logical_block: int,
        bound_generation: int,
    ) -> SlotOccupancy:
        self._require_enabled()
        physical_slot = self._validate_slot(physical_slot)
        logical_block = self._nonnegative_int(
            logical_block,
            "logical_block",
        )
        bound_generation = self._nonnegative_int(
            bound_generation,
            "bound_generation",
        )
        old_occupancy = self._active_occupancies[physical_slot]
        predecessor_events = ()
        if old_occupancy is not None:
            predecessor_events = tuple(
                sorted(
                    self._read_events_by_occupancy.get(
                        old_occupancy,
                        {},
                    ).values(),
                    key=lambda item: item[0],
                )
            )
        self._slot_generations[physical_slot] += 1
        occupancy = SlotOccupancy(
            physical_slot=physical_slot,
            occupancy_generation=self._slot_generations[physical_slot],
            logical_block=logical_block,
            bound_generation=bound_generation,
        )
        self._active_occupancies[physical_slot] = occupancy
        self._read_events_by_occupancy.setdefault(occupancy, {})
        self._pending_transitions[occupancy] = _SlotTransition(
            old_occupancy=old_occupancy,
            new_occupancy=occupancy,
            predecessor_events=predecessor_events,
        )
        return occupancy

    def release_slot(
        self,
        *,
        physical_slot: int,
        logical_block: int,
    ) -> None:
        self._require_enabled()
        physical_slot = self._validate_slot(physical_slot)
        logical_block = self._nonnegative_int(
            logical_block,
            "logical_block",
        )
        occupancy = self._active_occupancies[physical_slot]
        if (
            occupancy is None
            or occupancy.logical_block != logical_block
        ):
            raise RuntimeError(
                "slot release does not match active occupancy"
            )
        self._active_occupancies[physical_slot] = None
        self._pending_transitions.pop(occupancy, None)

    def assert_mapping(self, expected) -> None:
        self._require_enabled()
        expected = tuple(expected)
        if len(expected) != self.slot_count:
            raise RuntimeError(
                "diagnostic mapping length does not match slot count"
            )
        for physical_slot, item in enumerate(expected):
            occupancy = self._active_occupancies[physical_slot]
            if item is None:
                if occupancy is not None:
                    raise RuntimeError(
                        "diagnostic mapping disagrees with production "
                        f"mapping at slot {physical_slot}"
                    )
                continue
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(
                    "expected mapping entries must be "
                    "(slot, logical_block, bound_generation)"
                )
            slot, logical_block, bound_generation = item
            identity = (
                self._validate_slot(slot),
                self._nonnegative_int(
                    logical_block,
                    "logical_block",
                ),
                self._nonnegative_int(
                    bound_generation,
                    "bound_generation",
                ),
            )
            actual = (
                None
                if occupancy is None
                else (
                    occupancy.physical_slot,
                    occupancy.logical_block,
                    occupancy.bound_generation,
                )
            )
            if identity != actual:
                raise RuntimeError(
                    "diagnostic mapping disagrees with production "
                    f"mapping at slot {physical_slot}"
                )

    def snapshot_state(self) -> _DiagnosticStateSnapshot:
        return _DiagnosticStateSnapshot(
            mode=self._mode,
            slot_generations=tuple(self._slot_generations),
            active_occupancies=tuple(self._active_occupancies),
            read_events_by_occupancy=tuple(
                (
                    occupancy,
                    tuple(sorted(per_stream.items())),
                )
                for occupancy, per_stream in (
                    self._read_events_by_occupancy.items()
                )
            ),
            read_rows=tuple(self._read_rows),
            pending_overwrites=tuple(self._pending_overwrites),
            pending_transitions=tuple(
                self._pending_transitions.items()
            ),
            event_ordinals=tuple(
                sorted(self._event_ordinals.items())
            ),
            next_event_ordinal=self._next_event_ordinal,
            stream_inventory=tuple(
                sorted(self._stream_inventory)
            ),
            context=self._context,
        )

    def restore_state(
        self,
        snapshot: _DiagnosticStateSnapshot,
    ) -> None:
        if not isinstance(snapshot, _DiagnosticStateSnapshot):
            raise TypeError(
                "snapshot must be a diagnostic state snapshot"
            )
        self._mode = snapshot.mode
        self._slot_generations = list(snapshot.slot_generations)
        self._active_occupancies = list(
            snapshot.active_occupancies
        )
        self._read_events_by_occupancy = {
            occupancy: dict(per_stream)
            for occupancy, per_stream in (
                snapshot.read_events_by_occupancy
            )
        }
        self._read_rows = list(snapshot.read_rows)
        self._pending_overwrites = list(
            snapshot.pending_overwrites
        )
        self._pending_transitions = dict(
            snapshot.pending_transitions
        )
        self._event_ordinals = dict(snapshot.event_ordinals)
        self._next_event_ordinal = snapshot.next_event_ordinal
        self._stream_inventory = set(
            snapshot.stream_inventory
        )
        self._context = snapshot.context

    def predecessor_event_ordinals(
        self,
        occupancy: SlotOccupancy,
    ) -> tuple[int, ...]:
        if not isinstance(occupancy, SlotOccupancy):
            raise TypeError("occupancy must be SlotOccupancy")
        events = self._read_events_by_occupancy.get(occupancy)
        if events is None:
            raise RuntimeError("stale occupancy is not known")
        return tuple(
            ordinal
            for ordinal, _ in sorted(
                events.values(),
                key=lambda item: item[0],
            )
        )

    def record_read_window(
        self,
        *,
        engine_step,
        attention_stage,
        layer_index,
        window_ordinal,
        logical_blocks,
        physical_slots,
        current_stream,
        bound_generations=None,
    ) -> None:
        if not self.enabled:
            return
        context = self._validate_context(
            engine_step=engine_step,
            attention_stage=attention_stage,
            layer_index=layer_index,
            window_ordinal=window_ordinal,
        )
        identities = self._validated_active_identities(
            logical_blocks=logical_blocks,
            physical_slots=physical_slots,
            bound_generations=bound_generations,
        )
        if len(self._read_rows) + len(identities) > self._max_read_rows:
            raise RuntimeError("diagnostic read row capacity exceeded")
        event = self._event_factory()
        event.record(current_stream)
        ordinal = self._register_event(event)
        stream = int(self._stream_id(current_stream))
        self._stream_inventory.add(stream)
        for occupancy in identities:
            per_stream = self._read_events_by_occupancy.setdefault(
                occupancy,
                {},
            )
            per_stream[stream] = (ordinal, event)
            self._read_rows.append(
                SlotReadAssociation(
                    rank=self.rank,
                    engine_step=context[0],
                    attention_stage=context[1],
                    layer_index=context[2],
                    window_ordinal=context[3],
                    current_stream_id=stream,
                    physical_slot=occupancy.physical_slot,
                    occupancy_generation=(
                        occupancy.occupancy_generation
                    ),
                    logical_block=occupancy.logical_block,
                    bound_generation=occupancy.bound_generation,
                    read_event_ordinal=ordinal,
                )
            )

    def begin_h2d_span(
        self,
        *,
        copy_batch_ordinal: int,
        copy_span_ordinal: int,
        pairs,
        copy_stream,
    ) -> _H2DSpanHandle:
        self._require_enabled()
        if self._context is None:
            raise RuntimeError("diagnostic context is not set")
        copy_batch_ordinal = self._nonnegative_int(
            copy_batch_ordinal,
            "copy_batch_ordinal",
        )
        copy_span_ordinal = self._nonnegative_int(
            copy_span_ordinal,
            "copy_span_ordinal",
        )
        transitions = []
        for logical_block, physical_slot in tuple(pairs):
            physical_slot = self._validate_slot(physical_slot)
            logical_block = self._nonnegative_int(
                logical_block,
                "logical_block",
            )
            occupancy = self._active_occupancies[physical_slot]
            if occupancy is None or occupancy.logical_block != logical_block:
                raise RuntimeError(
                    "H2D span does not match active occupancy"
                )
            transition = self._pending_transitions.get(occupancy)
            if transition is None:
                raise RuntimeError(
                    "H2D span lacks an occupancy transition"
                )
            transitions.append(transition)
        if (
            len(self._pending_overwrites) + len(transitions)
            > self._max_overwrite_rows
        ):
            raise RuntimeError(
                "diagnostic overwrite row capacity exceeded"
            )
        unique_events: dict[int, object] = {}
        for transition in transitions:
            for ordinal, event in transition.predecessor_events:
                unique_events[ordinal] = event
        wait_ordinals = tuple(sorted(unique_events))
        if self._mode == "control":
            for ordinal in wait_ordinals:
                copy_stream.wait_event(unique_events[ordinal])
        start_event = self._event_factory()
        start_event.record(copy_stream)
        start_ordinal = self._register_event(start_event)
        pending_indices = []
        for transition in transitions:
            index = len(self._pending_overwrites)
            self._pending_overwrites.append(
                _PendingOverwrite(
                    context=self._context,
                    copy_batch_ordinal=copy_batch_ordinal,
                    copy_span_ordinal=copy_span_ordinal,
                    transition=transition,
                    read_events=transition.predecessor_events,
                    h2d_start_event_ordinal=start_ordinal,
                    h2d_start_event=start_event,
                    control_wait_event_ordinals=(
                        wait_ordinals
                        if self._mode == "control"
                        else ()
                    ),
                )
            )
            pending_indices.append(index)
            del self._pending_transitions[transition.new_occupancy]
        return _H2DSpanHandle(tuple(pending_indices))

    def end_h2d_span(
        self,
        handle: _H2DSpanHandle,
        *,
        copy_stream,
    ) -> None:
        self._require_enabled()
        if not isinstance(handle, _H2DSpanHandle):
            raise TypeError("handle must be an H2D span handle")
        done_event = self._event_factory()
        done_event.record(copy_stream)
        done_ordinal = self._register_event(done_event)
        for index in handle.pending_indices:
            pending = self._pending_overwrites[index]
            if pending.h2d_done_event is not None:
                raise RuntimeError("H2D span already ended")
            self._pending_overwrites[index] = _PendingOverwrite(
                context=pending.context,
                copy_batch_ordinal=pending.copy_batch_ordinal,
                copy_span_ordinal=pending.copy_span_ordinal,
                transition=pending.transition,
                read_events=pending.read_events,
                h2d_start_event_ordinal=(
                    pending.h2d_start_event_ordinal
                ),
                h2d_start_event=pending.h2d_start_event,
                control_wait_event_ordinals=(
                    pending.control_wait_event_ordinals
                ),
                h2d_done_event_ordinal=done_ordinal,
                h2d_done_event=done_event,
            )

    def drain(
        self,
        *,
        synchronize,
        timing_epsilon_ms: float,
    ) -> H2DSlotReuseDrain:
        if not self.enabled:
            raise RuntimeError("diagnostic is not enabled")
        if not callable(synchronize):
            raise ValueError("synchronize must be callable")
        if float(timing_epsilon_ms) <= 0:
            raise ValueError("timing epsilon must be positive")
        synchronize()
        overwrite_rows = tuple(
            self._resolve_pending_overwrite(
                pending,
                timing_epsilon_ms=float(timing_epsilon_ms),
            )
            for pending in self._pending_overwrites
        )
        result = H2DSlotReuseDrain(
            schema=H2D_SLOT_REUSE_SCHEMA,
            rank=self.rank,
            mode=self._mode,
            stream_inventory=tuple(sorted(self._stream_inventory)),
            read_rows=tuple(self._read_rows),
            overwrite_rows=overwrite_rows,
        )
        self._read_rows.clear()
        self._pending_overwrites.clear()
        self._event_ordinals.clear()
        self._read_events_by_occupancy.clear()
        self._pending_transitions.clear()
        self._stream_inventory.clear()
        return result

    def _resolve_pending_overwrite(
        self,
        pending: _PendingOverwrite,
        *,
        timing_epsilon_ms: float,
    ) -> H2DSlotOverwrite:
        if (
            pending.h2d_done_event_ordinal is None
            or pending.h2d_done_event is None
        ):
            raise RuntimeError("H2D timing lifecycle is incomplete")
        transition = pending.transition
        old = transition.old_occupancy
        read_event_ordinals = tuple(
            ordinal for ordinal, _ in pending.read_events
        )
        if old is None:
            status = "NO_PRIOR_OCCUPANCY"
            delta_ms = None
        elif not pending.read_events:
            status = "NO_PRIOR_READ"
            delta_ms = None
        else:
            deltas = []
            for ordinal, read_event in pending.read_events:
                try:
                    delta = float(
                        pending.h2d_start_event.elapsed_time(
                            read_event
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        "diagnostic event timing is unqueryable "
                        f"for read event {ordinal}"
                    ) from error
                deltas.append(delta)
            delta_ms = max(deltas)
            status = classify_read_h2d_ordering(
                delta_ms,
                timing_epsilon_ms,
            )
        return H2DSlotOverwrite(
            rank=self.rank,
            engine_step=pending.context[0],
            attention_stage=pending.context[1],
            layer_index=pending.context[2],
            window_ordinal=pending.context[3],
            copy_batch_ordinal=pending.copy_batch_ordinal,
            copy_span_ordinal=pending.copy_span_ordinal,
            physical_slot=transition.new_occupancy.physical_slot,
            old_occupancy_generation=(
                None if old is None else old.occupancy_generation
            ),
            old_logical_block=(
                None if old is None else old.logical_block
            ),
            old_bound_generation=(
                None if old is None else old.bound_generation
            ),
            new_occupancy_generation=(
                transition.new_occupancy.occupancy_generation
            ),
            new_logical_block=transition.new_occupancy.logical_block,
            new_bound_generation=(
                transition.new_occupancy.bound_generation
            ),
            read_event_ordinals=read_event_ordinals,
            h2d_start_event_ordinal=(
                pending.h2d_start_event_ordinal
            ),
            h2d_done_event_ordinal=pending.h2d_done_event_ordinal,
            control_wait_event_ordinals=(
                pending.control_wait_event_ordinals
            ),
            control_wait_count=len(
                pending.control_wait_event_ordinals
            ),
            timing_status=status,
            read_done_after_h2d_start_ms=delta_ms,
        )

    def _register_event(self, event: object) -> int:
        supplied = getattr(event, "ordinal", None)
        ordinal = (
            self._next_event_ordinal
            if supplied is None
            else int(supplied)
        )
        if ordinal <= 0:
            raise RuntimeError("event ordinal must be positive")
        if ordinal in self._event_ordinals:
            raise RuntimeError(
                f"duplicate event ordinal: {ordinal}"
            )
        self._event_ordinals[ordinal] = event
        self._next_event_ordinal = max(
            self._next_event_ordinal,
            ordinal + 1,
        )
        return ordinal

    def _validated_active_identities(
        self,
        *,
        logical_blocks,
        physical_slots,
        bound_generations,
    ) -> tuple[SlotOccupancy, ...]:
        logical_blocks = tuple(logical_blocks)
        physical_slots = tuple(physical_slots)
        if len(logical_blocks) != len(physical_slots):
            raise ValueError(
                "logical_blocks and physical_slots must have equal length"
            )
        if not logical_blocks:
            raise ValueError("read window must contain at least one slot")
        if bound_generations is not None:
            bound_generations = tuple(bound_generations)
            if len(bound_generations) != len(logical_blocks):
                raise ValueError(
                    "bound_generations must match logical_blocks"
                )
        identities = []
        for index, (logical_block, physical_slot) in enumerate(
            zip(logical_blocks, physical_slots)
        ):
            physical_slot = self._validate_slot(physical_slot)
            logical_block = self._nonnegative_int(
                logical_block,
                "logical_block",
            )
            occupancy = self._active_occupancies[physical_slot]
            expected_bound_generation = (
                None
                if bound_generations is None
                else self._nonnegative_int(
                    bound_generations[index],
                    "bound_generation",
                )
            )
            if (
                occupancy is None
                or occupancy.logical_block != logical_block
                or (
                    expected_bound_generation is not None
                    and occupancy.bound_generation
                    != expected_bound_generation
                )
            ):
                raise RuntimeError(
                    "read window does not match active occupancy"
                )
            identities.append(occupancy)
        return tuple(identities)

    def _validate_context(
        self,
        *,
        engine_step,
        attention_stage,
        layer_index,
        window_ordinal,
    ) -> tuple[int, str, int, int]:
        if attention_stage not in VALID_ATTENTION_STAGES:
            raise ValueError(
                "attention_stage must be decode, spec_verify, or prefill"
            )
        return (
            self._nonnegative_int(engine_step, "engine_step"),
            attention_stage,
            self._nonnegative_int(layer_index, "layer_index"),
            self._nonnegative_int(
                window_ordinal,
                "window_ordinal",
            ),
        )

    def _validate_slot(self, physical_slot: int) -> int:
        physical_slot = self._nonnegative_int(
            physical_slot,
            "physical_slot",
        )
        if physical_slot >= self.slot_count:
            raise IndexError(
                f"physical_slot {physical_slot} is out of range"
            )
        return physical_slot

    @staticmethod
    def _nonnegative_int(value, name: str) -> int:
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return value

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("diagnostic is not enabled")
